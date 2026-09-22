#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ced_homotopy import HomotopyCED  # noqa: E402


def metric(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative": float(
            (reference.float() - candidate.float()).norm()
            / reference.float().norm().clamp_min(1e-12)
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--length", type=int, default=96)
    cfg = p.parse_args()
    cache_dir = str(cfg.cache_dir.resolve()) if cfg.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cache_dir)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model, cache_dir=cache_dir, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model = HomotopyCED.from_qwen(base).cuda().eval()
    text = "Layerwise parity reveals where a cached causal path begins to diverge from a parallel full sequence. "
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = (tokens * (cfg.length // len(tokens) + 1))[: cfg.length]
    input_ids = torch.tensor(tokens, device="cuda").unsqueeze(0)
    _, full = model.endpoint_full_trace(input_ids)
    _, endpoint_cache = model.endpoint_prefill(input_ids[:, :-1])
    _, _, cached = model.endpoint_step_trace(input_ids[:, -1:], endpoint_cache)
    payload = {
        "embedding": metric(full["embedding"], cached["embedding"]),
        "lower": [metric(a, b) for a, b in zip(full["lower"], cached["lower"], strict=True)],
        "memory_k": metric(full["memory_k"], cached["memory_k"]),
        "memory_v": metric(full["memory_v"], cached["memory_v"]),
        "upper": [metric(a, b) for a, b in zip(full["upper"], cached["upper"], strict=True)],
        "final_norm": metric(full["final_norm"], cached["final_norm"]),
        "logits": metric(full["logits"], cached["logits"]),
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["logits"], sort_keys=True))


if __name__ == "__main__":
    main()
