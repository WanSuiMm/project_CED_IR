"""One short Qwen3 pair-cache parity smoke; not an A/B result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from .qualify_token_mla import load_validation
from .qwen3_cache import PairDynamicCache
from .qwen3_split import convert_qwen3_to_split, convert_split_to_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=16)
    args = parser.parse_args()
    if args.length < 2 or args.length % 2:
        raise ValueError("length must be even and >=2")
    tokens = load_validation(args.tokens)[:args.length].unsqueeze(0).cuda()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=str(args.cache_dir), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    convert_qwen3_to_split(model, 96)
    convert_split_to_pair(model)
    with torch.inference_mode():
        full = model(input_ids=tokens, use_cache=False).logits[:, -1].float()
        cache = PairDynamicCache(config=model.config)
        prefill = model(input_ids=tokens[:, :-1], past_key_values=cache,
                        use_cache=True)
        step = model(input_ids=tokens[:, -1:], past_key_values=cache,
                     use_cache=True).logits[:, -1].float()
    p = full.log_softmax(-1)
    q = step.log_softmax(-1)
    result = {
        "status": "SMOKE_ONLY",
        "token_length": args.length,
        "full_to_cached_kl": float((p.exp() * (p - q)).sum()),
        "max_abs_logit_difference": float((full - step).abs().max()),
        "cache_token_length": cache.get_seq_length(),
        "actual_persistent_scalars": cache.persistent_scalars(),
        "actual_persistent_bytes": cache.persistent_bytes(),
        "expected_persistent_scalars": model.config.num_hidden_layers * (
            (args.length // 2) * model.config.kv_lora_rank
            + args.length * model.config.qk_rope_head_dim),
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
