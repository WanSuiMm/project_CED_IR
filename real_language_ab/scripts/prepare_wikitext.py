#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def collect(split, tokenizer, cap: int) -> torch.Tensor:
    tokens: list[int] = []
    eos = tokenizer.eos_token_id
    for row in split:
        text = row["text"].strip()
        if not text:
            continue
        tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        tokens.append(eos)
        if len(tokens) >= cap:
            break
    if len(tokens) < cap:
        raise RuntimeError(f"dataset supplied only {len(tokens)} tokens, need {cap}")
    return torch.tensor(tokens[:cap], dtype=torch.long)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-tokens", type=int, default=100000)
    p.add_argument("--validation-tokens", type=int, default=20000)
    args = p.parse_args()
    cache = str(args.cache_dir.resolve()) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache)
    train = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", cache_dir=cache
    )
    validation = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="validation", cache_dir=cache
    )
    payload = {
        "train": collect(train, tokenizer, args.train_tokens),
        "validation": collect(validation, tokenizer, args.validation_tokens),
        "model": args.model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"train_tokens": len(payload["train"]), "validation_tokens": len(payload["validation"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
