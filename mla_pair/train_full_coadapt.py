"""Frozen 1M-token full-coadapt A/B2 viability pilot (one pass, no LR sweep)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .headwise_compiler import HeadwiseKVPairCompiler
from .qualify_token_mla import evaluate, load_validation
from .qwen3_split import convert_qwen3_to_split, convert_split_to_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A_TOKEN", "B2_HEADWISE"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=1024)
    parser.add_argument("--rope-dim-per-head", type=int, default=96)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--compiler-lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.steps < 1 or args.sequence_length < 2 or args.eval_sequences < 1
            or args.eval_every < 1 or args.backbone_lr <= 0 or args.compiler_lr <= 0):
        raise ValueError("invalid frozen training/evaluation configuration")
    data = torch.load(args.tokens, map_location="cpu", weights_only=True)
    train = data["train"].long()
    validation = load_validation(args.tokens)
    if train.numel() < args.steps * args.sequence_length:
        raise ValueError("not enough training tokens for a single non-repeating pass")
    if validation.numel() < args.eval_sequences * args.sequence_length:
        raise ValueError("not enough validation tokens")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=str(args.cache_dir), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    convert_qwen3_to_split(model, args.rope_dim_per_head)
    if args.variant == "B2_HEADWISE":
        convert_split_to_pair(model, compiler_kind="headwise_kv")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    compiler_params = [p for module in model.modules()
                       if isinstance(module, HeadwiseKVPairCompiler)
                       for p in module.parameters()]
    compiler_ids = {id(p) for p in compiler_params}
    backbone_params = [p for p in model.parameters() if id(p) not in compiler_ids]
    if not backbone_params or (args.variant == "B2_HEADWISE") != bool(compiler_params):
        raise ValueError("incorrect trainable parameter groups")
    if not all(p.requires_grad for p in backbone_params + compiler_params):
        raise ValueError("full co-adaptation requires all parameters trainable")
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.backbone_lr},
         {"params": compiler_params, "lr": args.compiler_lr}],
        weight_decay=0.01, fused=True)
    curve = []

    def record(step: int) -> None:
        nll, chunks = evaluate(
            model, validation, args.sequence_length, args.eval_sequences, "cuda")
        event = {"step": step, "train_tokens": step * args.sequence_length,
                 "validation_nll": nll, "per_sequence_nll": chunks}
        curve.append(event)
        print(json.dumps({k: event[k] for k in
                          ("step", "train_tokens", "validation_nll")}), flush=True)

    record(0)
    train_trace = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, args.steps + 1):
        model.train()
        ids = train[(step - 1) * args.sequence_length:step * args.sequence_length]
        ids = ids.unsqueeze(0).cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite train loss at step {step}")
        loss.backward()
        backbone_norm = float(torch.nn.utils.clip_grad_norm_(backbone_params, 1.0))
        compiler_norm = (float(torch.nn.utils.clip_grad_norm_(compiler_params, 1.0))
                         if compiler_params else 0.0)
        optimizer.step()
        if step == 1 or step % 256 == 0 or step == args.steps:
            event = {"step": step, "train_nll": float(loss.detach()),
                     "backbone_grad_norm": backbone_norm,
                     "compiler_grad_norm": compiler_norm}
            train_trace.append(event)
            print(json.dumps(event), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            record(step)
    summary = {
        "status": "SMOKE_ONLY" if args.smoke else "FULL_COADAPT_PILOT_VARIANT_COMPLETE",
        "variant": args.variant, "seed": args.seed, "steps": args.steps,
        "train_tokens": args.steps * args.sequence_length,
        "sequence_length": args.sequence_length,
        "eval_sequences": args.eval_sequences, "eval_every": args.eval_every,
        "rope_dim_per_head": args.rope_dim_per_head,
        "content_dim": model.config.kv_lora_rank,
        "rope_dim": model.config.qk_rope_head_dim,
        "backbone_lr": args.backbone_lr,
        "compiler_lr": args.compiler_lr,
        "backbone_parameters": sum(p.numel() for p in backbone_params),
        "extra_compiler_parameters": sum(p.numel() for p in compiler_params),
        "curve": curve, "train_trace": train_trace,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started,
        "claim_boundary": "single-seed 1M-token viability screen; not speed or final architecture result",
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "variant": args.variant,
                      "final_nll": curve[-1]["validation_nll"],
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
