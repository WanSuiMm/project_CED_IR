"""Matched short A/B adaptation from the same qualified split-Qwen3 recipe.

This is a screening pilot, not the 10–20M-token architecture endpoint.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from .qualify_token_mla import evaluate, load_validation
from .qwen3_split import convert_qwen3_to_split, convert_split_to_pair
from .reference import PairCompiler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A_TOKEN", "B_PAIR"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=32)
    parser.add_argument("--rope-dim-per-head", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--compiler-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.sequence_length < 2 or args.eval_sequences < 1:
        raise ValueError("invalid training/evaluation budget")
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
    if args.variant == "B_PAIR":
        convert_split_to_pair(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    # PairCompiler's deterministic average initialization consumes RNG through
    # nn.Linear construction. Reset so A/B LoRA matrices still start identically.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    if args.variant == "B_PAIR":
        for module in model.modules():
            if isinstance(module, PairCompiler):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    compiler_params = [parameter for module in model.modules()
                       if isinstance(module, PairCompiler)
                       for parameter in module.parameters()]
    compiler_ids = {id(parameter) for parameter in compiler_params}
    lora_params = [parameter for parameter in model.parameters()
                   if parameter.requires_grad and id(parameter) not in compiler_ids]
    trainable = lora_params + compiler_params
    trainable_count = sum(parameter.numel() for parameter in trainable)
    extra_compiler_count = sum(parameter.numel() for module in model.modules()
                               if isinstance(module, PairCompiler)
                               for parameter in module.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.learning_rate},
         {"params": compiler_params, "lr": args.compiler_learning_rate}],
        weight_decay=0.01)
    initial_nll, initial_chunks = evaluate(
        model, validation, args.sequence_length, args.eval_sequences, "cuda")
    trace = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        model.train()
        ids = train[step * args.sequence_length:(step + 1) * args.sequence_length]
        ids = ids.unsqueeze(0).cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite train loss at step {step + 1}")
        loss.backward()
        lora_grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, 1.0))
        compiler_grad_norm = (float(torch.nn.utils.clip_grad_norm_(compiler_params, 1.0))
                              if compiler_params else 0.0)
        optimizer.step()
        if step == 0 or (step + 1) % 64 == 0 or step + 1 == args.steps:
            trace.append({"step": step + 1, "train_nll": float(loss.detach()),
                          "lora_grad_norm": lora_grad_norm,
                          "compiler_grad_norm": compiler_grad_norm})
            print(json.dumps(trace[-1]), flush=True)
    final_nll, final_chunks = evaluate(
        model, validation, args.sequence_length, args.eval_sequences, "cuda")
    summary = {
        "status": "SMOKE_ONLY" if args.smoke else "PAIRED_PILOT_VARIANT_COMPLETE",
        "variant": args.variant,
        "seed": args.seed,
        "steps": args.steps,
        "train_tokens": args.steps * args.sequence_length,
        "sequence_length": args.sequence_length,
        "eval_sequences": args.eval_sequences,
        "rope_dim_per_head": args.rope_dim_per_head,
        "content_dim": model.config.kv_lora_rank,
        "rope_dim": model.config.qk_rope_head_dim,
        "lora_rank": 8,
        "learning_rate": args.learning_rate,
        "compiler_learning_rate": args.compiler_learning_rate,
        "trainable_parameters": trainable_count,
        "extra_compiler_parameters": extra_compiler_count,
        "initial_validation_nll": initial_nll,
        "final_validation_nll": final_nll,
        "initial_per_sequence_nll": initial_chunks,
        "final_per_sequence_nll": final_chunks,
        "trace": trace,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.time() - started,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.smoke:
        model.save_pretrained(args.output / "lora_adapter")
        if args.variant == "B_PAIR":
            base = model.get_base_model()
            compiler_state = {
                str(index): layer.self_attn.compiler.state_dict()
                for index, layer in enumerate(base.model.layers)
            }
            torch.save(compiler_state, args.output / "compilers.pt")
    print(json.dumps({"status": summary["status"],
                      "variant": args.variant,
                      "initial_nll": initial_nll,
                      "final_nll": final_nll,
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
