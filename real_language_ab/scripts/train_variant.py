#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interface_lm import InterfaceLM  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("A_TOKEN", "B_PAIR_WIDE"), required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--producer-layers", type=int, default=4)
    p.add_argument("--reader-layers", type=int, default=4)
    p.add_argument("--local-window", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-sequences", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--reference-summary", type=Path, default=None)
    return p.parse_args()


def chunk(stream: torch.Tensor, index: int, length: int, device: str) -> torch.Tensor:
    maximum = len(stream) - length
    start = (index * length) % maximum
    return stream[start : start + length].to(device).unsqueeze(0)


@torch.inference_mode()
def evaluate(model, validation, length: int, count: int) -> float:
    model.eval()
    losses = []
    for index in range(count):
        input_ids = chunk(validation, index, length, "cuda")
        losses.append(float(model(input_ids, labels=input_ids).loss))
    return float(np.mean(losses))


@torch.inference_mode()
def remote_metrics(model, validation, length: int, count: int = 8) -> dict[str, float]:
    model.eval()
    correct_losses = []
    mismatch_losses = []
    ablated_losses = []
    prefix = length // 2
    suffix_start = length - 64
    for index in range(count):
        correct = chunk(validation, index, length, "cuda")
        other = chunk(validation, index + count + 3, length, "cuda")
        mismatch = correct.clone()
        mismatch[:, :prefix] = other[:, :prefix]
        labels = torch.full_like(correct, -100)
        labels[:, suffix_start:] = correct[:, suffix_start:]
        correct_losses.append(float(model(correct, labels=labels).loss))
        mismatch_losses.append(float(model(mismatch, labels=labels).loss))
        ablated_losses.append(
            float(model(correct, labels=labels, disable_interface=True).loss)
        )
    correct = float(np.mean(correct_losses))
    mismatch = float(np.mean(mismatch_losses))
    ablated = float(np.mean(ablated_losses))
    return {
        "correct_suffix_nll": correct,
        "mismatched_prefix_suffix_nll": mismatch,
        "remote_context_gain": mismatch - correct,
        "ablated_suffix_nll": ablated,
        "interface_ablation_delta": ablated - correct,
    }


@torch.inference_mode()
def generate(model, tokenizer, validation, tokens: int = 32) -> dict:
    model.eval()
    input_ids = validation[:64].to("cuda").unsqueeze(0)
    prompt_length = input_ids.shape[1]
    for _ in range(tokens):
        logits = model(input_ids).logits[:, -1]
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    generated = input_ids[0, prompt_length:].tolist()
    return {
        "token_ids": generated,
        "distinct_token_ratio": len(set(generated)) / max(len(generated), 1),
        "text": tokenizer.decode(generated, skip_special_tokens=True),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_length % 2:
        raise ValueError("training sequence length must be even")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    train_stream = data["train"]
    validation = data["validation"]
    cache = str(args.cache_dir.resolve()) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=cache, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = InterfaceLM.from_qwen(
        base,
        variant=args.variant,
        producer_layers=args.producer_layers,
        reader_layers=args.reader_layers,
        local_window=args.local_window,
    ).to(device="cuda", dtype=torch.bfloat16)
    del base
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    trace = [{"step": 0, "validation_nll": evaluate(
        model, validation, args.sequence_length, args.eval_sequences
    )}]
    model.train()
    for step in range(1, args.steps + 1):
        input_ids = chunk(train_stream, step - 1, args.sequence_length, "cuda")
        output = model(input_ids, labels=input_ids)
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            trace.append({
                "step": step,
                "validation_nll": evaluate(
                    model, validation, args.sequence_length, args.eval_sequences
                ),
                "train_nll": float(output.loss.detach()),
                "grad_norm": grad_norm,
            })
            model.train()

    remote = remote_metrics(model, validation, args.sequence_length)
    generation = generate(model, tokenizer, validation)
    initial_nll = trace[0]["validation_nll"]
    final_nll = trace[-1]["validation_nll"]
    a_qualified = (
        initial_nll - final_nll >= 0.20
        and generation["distinct_token_ratio"] >= 0.10
        and remote["remote_context_gain"] > 0.0
        and remote["interface_ablation_delta"] >= 0.01
    )
    if args.variant == "A_TOKEN":
        verdict = "PASS_A_SUBSTRATE" if a_qualified else "INVALID_REAL_LANGUAGE_SUBSTRATE"
    else:
        verdict = "B_COMPLETE_UNCOMPARED"
        if args.reference_summary:
            reference = json.loads(args.reference_summary.read_text(encoding="utf-8"))
            a_metrics = reference["metrics"]
            remote_floor = max(a_metrics["remote"]["remote_context_gain"], 1e-12)
            noninferior = (
                final_nll - a_metrics["final_validation_nll"] <= 0.10
                and remote["remote_context_gain"] / remote_floor >= 0.80
            )
            verdict = "PASS_PAIR_WIDE_PILOT" if noninferior else "FAIL_PAIR_WIDE_PILOT"

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    payload = {
        "protocol": "REAL_LANGUAGE_INTERFACE_AB_v0.1",
        "variant": args.variant,
        "verdict": verdict,
        "arguments": {
            **vars(args),
            "cache_dir": "<external-model-cache>" if args.cache_dir else None,
            "data": "<local-token-cache>",
            "output": str(args.output),
            "reference_summary": str(args.reference_summary) if args.reference_summary else None,
        },
        "metrics": {
            "initial_validation_nll": initial_nll,
            "final_validation_nll": final_nll,
            "nll_improvement": initial_nll - final_nll,
            "trace": trace,
            "remote": remote,
            "generation": generation,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "persistent_scalars_at_sequence_length": model.persistent_scalars(
                args.sequence_length
            ),
            "persistent_records_at_sequence_length": model.persistent_records(
                args.sequence_length
            ),
            "persistent_bytes_bf16": 2 * model.persistent_scalars(
                args.sequence_length
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "duration_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
