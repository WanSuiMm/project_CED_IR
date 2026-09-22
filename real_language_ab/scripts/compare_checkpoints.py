#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interface_lm import InterfaceLM  # noqa: E402


def chunk(stream: torch.Tensor, index: int, length: int) -> torch.Tensor:
    start = (index * length) % (len(stream) - length)
    return stream[start : start + length].cuda().unsqueeze(0)


def interval(values: np.ndarray, seed: int = 20260922) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(10000, n))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def load_model(args, variant: str, checkpoint: Path):
    cache = str(args.cache_dir.resolve()) if args.cache_dir else None
    base = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=cache, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model = InterfaceLM.from_qwen(
        base,
        variant=variant,
        producer_layers=4,
        reader_layers=4,
        local_window=16,
    ).to(device="cuda", dtype=torch.bfloat16)
    del base
    model.load_state_dict(
        torch.load(checkpoint, map_location="cuda", weights_only=True), strict=True
    )
    return model.eval()


@torch.inference_mode()
def evaluate(model, validation, length: int, count: int):
    nll = []
    correct = []
    mismatch = []
    ablated = []
    suffix_start = length - 64
    prefix = length // 2
    for index in range(count):
        ids = chunk(validation, index, length)
        nll.append(float(model(ids, labels=ids).loss))
        other = chunk(validation, index + count + 3, length)
        wrong = ids.clone()
        wrong[:, :prefix] = other[:, :prefix]
        labels = torch.full_like(ids, -100)
        labels[:, suffix_start:] = ids[:, suffix_start:]
        correct.append(float(model(ids, labels=labels).loss))
        mismatch.append(float(model(wrong, labels=labels).loss))
        ablated.append(float(model(ids, labels=labels, disable_interface=True).loss))
    return {
        "nll": np.asarray(nll),
        "remote_gain": np.asarray(mismatch) - np.asarray(correct),
        "ablation_delta": np.asarray(ablated) - np.asarray(correct),
    }


def summarize(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "ci95": interval(values), "n": len(values)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--a-checkpoint", type=Path, required=True)
    p.add_argument("--b-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--count", type=int, default=64)
    args = p.parse_args()
    data = torch.load(args.data, map_location="cpu", weights_only=True)
    validation = data["validation"]
    a_model = load_model(args, "A_TOKEN", args.a_checkpoint)
    a = evaluate(a_model, validation, args.sequence_length, args.count)
    del a_model
    torch.cuda.empty_cache()
    b_model = load_model(args, "B_PAIR_WIDE", args.b_checkpoint)
    b = evaluate(b_model, validation, args.sequence_length, args.count)
    payload = {
        "protocol": "REAL_LANGUAGE_INTERFACE_AB_PAIRED_EVAL_v0.1",
        "formal_training_verdict": "FAIL_PAIR_WIDE_PILOT",
        "a": {key: summarize(value) for key, value in a.items()},
        "b": {key: summarize(value) for key, value in b.items()},
        "paired_b_minus_a": {
            key: summarize(b[key] - a[key]) for key in a
        },
        "note": "Post-training paired precision audit; does not change frozen training verdict.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["paired_b_minus_a"], sort_keys=True))


if __name__ == "__main__":
    main()
