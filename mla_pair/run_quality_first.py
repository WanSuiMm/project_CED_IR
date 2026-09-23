"""Bounded Qwen3 split-baseline pilot; no adaptation or B training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from mla_pair.qualify_token_mla import cache_probe, evaluate, load_validation
from mla_pair.qwen3_split import convert_qwen3_to_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rope-dim-per-head", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--validation-offset", type=int, default=8192,
                        help="Offset into the frozen validation stream; formal default avoids smoke tokens")
    parser.add_argument("--smoke", action="store_true",
                        help="Engineering check only; never emit a qualification verdict")
    args = parser.parse_args()
    if args.sequence_length < 2 or args.sequences < 1 or args.validation_offset < 0:
        raise ValueError("invalid evaluation span")
    tokens = load_validation(args.tokens)[args.validation_offset:]
    if tokens.numel() < args.sequence_length * args.sequences:
        raise ValueError("not enough non-overlapping validation tokens")
    if not torch.cuda.is_available():
        raise ValueError("this model pilot requires a CUDA device")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=str(args.cache_dir), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    native_nll, native_per_sequence = evaluate(
        model, tokens, args.sequence_length, args.sequences, "cuda")
    native_probe = cache_probe(
        model, tokens, length=16, device="cuda",
        content_dim=model.config.num_key_value_heads * model.config.head_dim,
        rope_dim=model.config.num_key_value_heads * model.config.head_dim)
    convert_qwen3_to_split(model, args.rope_dim_per_head)
    split_nll, split_per_sequence = evaluate(
        model, tokens, args.sequence_length, args.sequences, "cuda")
    probe = cache_probe(
        model, tokens, length=16, device="cuda",
        content_dim=model.config.kv_lora_rank,
        rope_dim=model.config.qk_rope_head_dim)
    delta = split_nll - native_nll
    parity_excess = (probe["parity"]["full_to_cached_kl"]
                     - native_probe["parity"]["full_to_cached_kl"])
    qualifies = (delta <= 0.10
                 and parity_excess <= 0.001
                 and probe["persistent_cache"]["matches_compact_mla_layout"])
    result = {
        "status": "SMOKE_ONLY" if args.smoke else (
            "A_QUALIFIED" if qualifies else "A_NOT_QUALIFIED"),
        "source_model": args.model,
        "conversion": "qwen3_partial_rope_exact_width_split_no_low_rank",
        "rope_dim_per_head": args.rope_dim_per_head,
        "content_dim": model.config.kv_lora_rank,
        "rope_dim": model.config.qk_rope_head_dim,
        "sequence_length": args.sequence_length,
        "sequences": args.sequences,
        "validation_offset": args.validation_offset,
        "native_nll": native_nll,
        "split_nll": split_nll,
        "delta_nll": delta,
        "native_per_sequence_nll": native_per_sequence,
        "split_per_sequence_nll": split_per_sequence,
        "cache_probe": probe,
        "native_cache_probe": native_probe,
        "full_cache_kl_excess_vs_native": parity_excess,
        "frozen_first_pass_gates": {
            "max_delta_nll": 0.10,
            "max_full_to_cached_kl_excess_vs_native": 0.001,
            "require_compact_split_cache": True,
        },
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "native_nll", "split_nll", "delta_nll",
                       "content_dim", "rope_dim")}, indent=2))


if __name__ == "__main__":
    main()
