#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def metrics(full: torch.Tensor, cached: torch.Tensor) -> dict[str, float]:
    delta = (full.float() - cached.float()).abs()
    full_logp = torch.log_softmax(full.float(), dim=-1)
    cached_logp = torch.log_softmax(cached.float(), dim=-1)
    kl = (full_logp.exp() * (full_logp - cached_logp)).sum(dim=-1)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative": float((full.float() - cached.float()).norm() / full.float().norm()),
        "mean_kl": float(kl.mean()),
        "top1_agreement": float((full.argmax(-1) == cached.argmax(-1)).float().mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate native Qwen BF16 cache numerics")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--length", type=int, default=32)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=cache_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    text = (
        "A causal language model should preserve exact temporal boundaries while "
        "learning a shared long-range interface. Numbers 1739 and 8421 are included. "
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * (args.length // len(ids) + 1))[: args.length]
    input_ids = torch.tensor(ids, device="cuda").unsqueeze(0)
    with torch.inference_mode():
        full = model(input_ids=input_ids, use_cache=False).logits
        prefix = model(input_ids=input_ids[:, :-1], use_cache=True)
        continuation = model(
            input_ids=input_ids[:, -1:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
        deployment_full = torch.cat([full[:, -2:-1], full[:, -1:]], dim=1)
        deployment_cached = torch.cat(
            [prefix.logits[:, -1:], continuation.logits], dim=1
        )
        pieces = []
        past = None
        for index in range(args.length):
            out = model(
                input_ids=input_ids[:, index : index + 1],
                past_key_values=past,
                use_cache=True,
            )
            pieces.append(out.logits)
            past = out.past_key_values
        cached = torch.cat(pieces, dim=1)
    payload = {
        "model": args.model,
        "dtype": "bfloat16",
        "length": args.length,
        "deployment_metrics": metrics(deployment_full, deployment_cached),
        "all_step_stress_metrics": metrics(full, cached),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
