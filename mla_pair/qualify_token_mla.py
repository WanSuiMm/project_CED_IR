"""Evaluate a converted token-MLA checkpoint against its native reference.

This is a read-only gate. It does not perform TransMLA conversion or training.
The two checkpoints must use the same tokenizer and token IDs.
First-pass acceptance: delta NLL <= 0.10 nats/token, candidate full-vs-cache
KL no more than 0.001 above native, and observed cache growth == rank + RoPE
width per layer. The parity gate is native-relative because BF16 native Qwen3
itself can exceed 0.001 absolute KL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True, help="Original pretrained model ID/path")
    parser.add_argument("--candidate", required=True, help="Converted token-MLA model path")
    parser.add_argument("--tokens", type=Path, required=True,
                        help="torch file with a 1D validation tensor or {'validation': tensor}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--sequences", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-probe-length", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Only for a locally inspected converted model implementation")
    return parser.parse_args()


def load_validation(path: Path) -> torch.Tensor:
    data = torch.load(path, map_location="cpu", weights_only=True)
    tokens = data["validation"] if isinstance(data, dict) else data
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
        raise ValueError("validation must be a 1D tensor of token IDs")
    return tokens.long()


@torch.inference_mode()
def evaluate(model, tokens: torch.Tensor, sequence_length: int,
             sequences: int, device: str) -> tuple[float, list[float]]:
    model.eval()
    losses = []
    counts = []
    for index in range(sequences):
        start = index * sequence_length
        inputs = tokens[start:start + sequence_length].unsqueeze(0).to(device)
        logits = model(input_ids=inputs, use_cache=False).logits[:, :-1].float()
        targets = inputs[:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="mean")
        losses.append(float(per_token))
        counts.append(sequence_length - 1)
    return sum(loss * count for loss, count in zip(losses, counts)) / sum(counts), losses


def cache_tensors(cache: Any, path: str = "cache") -> dict[str, torch.Tensor]:
    """Collect only live tensor fields from known HF/custom cache containers.

    Unknown layouts fail closed rather than inventing a KV-cache size from config.
    """
    if isinstance(cache, torch.Tensor):
        return {path: cache}
    if isinstance(cache, (tuple, list)):
        return {name: tensor for index, item in enumerate(cache)
                for name, tensor in cache_tensors(item, f"{path}.{index}").items()}
    if isinstance(cache, dict):
        return {name: tensor for key, item in cache.items()
                for name, tensor in cache_tensors(item, f"{path}.{key}").items()}
    for fields in (("key_cache", "value_cache"), ("layers",),
                   ("keys", "values"), ("kv_cache", "pe_cache")):
        if any(hasattr(cache, field) for field in fields):
            return {name: tensor for field in fields if hasattr(cache, field)
                    for name, tensor in cache_tensors(
                        getattr(cache, field), f"{path}.{field}").items()}
    if cache is None:
        return {}
    raise TypeError(f"unrecognized cache container at {path}: {type(cache).__name__}")


def summarize_cache(before: dict[str, torch.Tensor],
                    after: dict[str, torch.Tensor], layers: int,
                    content_dim: int, rope_dim: int) -> dict:
    if not before or before.keys() != after.keys():
        raise ValueError("cache tensors missing or fields changed on append")
    details = []
    total_scalars = 0
    total_bytes = 0
    for name in sorted(before):
        old, new = before[name], after[name]
        old_shape, new_shape = tuple(old.shape), tuple(new.shape)
        if len(old_shape) != len(new_shape) or old.dtype != new.dtype:
            raise ValueError(f"cache field type changed: {name}")
        grown_axes = [axis for axis, (a, b) in enumerate(zip(old_shape, new_shape))
                      if b == a + 1]
        same_axes = all(a == b or axis in grown_axes
                        for axis, (a, b) in enumerate(zip(old_shape, new_shape)))
        if len(grown_axes) != 1 or not same_axes:
            raise ValueError(f"cache field has no single-token append: {name}")
        scalar_delta = new.numel() - old.numel()
        total_scalars += scalar_delta
        total_bytes += scalar_delta * new.element_size()
        details.append({"field": name, "prefill_shape": old_shape,
                        "after_one_token_shape": new_shape,
                        "dtype": str(new.dtype), "scalars_per_token": scalar_delta})
    expected = layers * (content_dim + rope_dim)
    return {"tensor_fields": details, "observed_scalars_per_token": total_scalars,
            "observed_bytes_per_token": total_bytes,
            "observed_scalars_per_token_per_layer": total_scalars / layers,
            "expected_scalars_per_token": expected,
            "matches_compact_mla_layout": total_scalars == expected}


@torch.inference_mode()
def cache_probe(model, tokens: torch.Tensor, length: int, device: str,
                content_dim: int, rope_dim: int) -> dict:
    if tokens.numel() < length + 1:
        raise ValueError("validation stream too short for cache probe")
    prefix = tokens[:length].unsqueeze(0).to(device)
    next_token = tokens[length:length + 1].unsqueeze(0).to(device)
    full = model(input_ids=torch.cat((prefix, next_token), dim=1),
                 use_cache=False).logits[:, -1].float()
    prefill = model(input_ids=prefix, use_cache=True)
    cache = prefill.past_key_values
    before = {name: tensor.detach().clone()
              for name, tensor in cache_tensors(cache).items()}
    continued = model(input_ids=next_token, past_key_values=cache,
                      attention_mask=torch.ones((1, length + 1),
                                                device=device, dtype=torch.long),
                      use_cache=True)
    after = cache_tensors(continued.past_key_values)
    delta = (full - continued.logits[:, -1].float()).abs()
    full_logp = full.log_softmax(-1)
    cached_logp = continued.logits[:, -1].float().log_softmax(-1)
    parity = {"max_abs_logit_difference": float(delta.max()),
              "mean_abs_logit_difference": float(delta.mean()),
              "full_to_cached_kl": float((full_logp.exp() *
                                          (full_logp - cached_logp)).sum())}
    layers = int(model.config.num_hidden_layers)
    return {"parity": parity,
            "persistent_cache": summarize_cache(before, after, layers,
                                                  content_dim, rope_dim)}


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2 or args.sequences < 1 or args.cache_probe_length < 2:
        raise ValueError("sequence-length must be >=2 and sequences >=1")
    tokens = load_validation(args.tokens)
    if tokens.numel() < args.sequence_length * args.sequences:
        raise ValueError("not enough validation tokens for non-overlapping sequences")
    native_tokenizer = AutoTokenizer.from_pretrained(
        args.native, trust_remote_code=args.trust_remote_code)
    candidate_tokenizer = AutoTokenizer.from_pretrained(
        args.candidate, trust_remote_code=args.trust_remote_code)
    if native_tokenizer.get_vocab() != candidate_tokenizer.get_vocab():
        raise ValueError("native and candidate tokenizers differ")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    results = {}
    for label, path in (("native", args.native), ("candidate", args.candidate)):
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code).to(args.device)
        if label == "candidate":
            config = model.config.to_dict()
            if not (isinstance(config.get("kv_lora_rank"), int)
                    and isinstance(config.get("qk_rope_head_dim"), int)):
                raise ValueError(
                    "candidate config lacks kv_lora_rank/qk_rope_head_dim; "
                    "refusing to qualify an unverified MLA architecture")
            results["candidate_mla_dimensions"] = {
                "kv_lora_rank": config["kv_lora_rank"],
                "qk_rope_head_dim": config["qk_rope_head_dim"],
            }
            results["candidate_cache_probe"] = cache_probe(
                model, tokens, args.cache_probe_length, args.device,
                config["kv_lora_rank"], config["qk_rope_head_dim"])
        else:
            config = model.config
            native_key_width = int(config.num_key_value_heads * config.head_dim)
            results["native_cache_probe"] = cache_probe(
                model, tokens, args.cache_probe_length, args.device,
                native_key_width, native_key_width)
        mean, per_sequence = evaluate(model, tokens, args.sequence_length,
                                      args.sequences, args.device)
        results[label] = {"checkpoint": path, "nll": mean,
                          "per_sequence_nll": per_sequence}
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    results["delta_nll_candidate_minus_native"] = (
        results["candidate"]["nll"] - results["native"]["nll"])
    results["sequence_length"] = args.sequence_length
    results["sequences"] = args.sequences
    cache_result = results["candidate_cache_probe"]
    parity_excess = (cache_result["parity"]["full_to_cached_kl"]
                     - results["native_cache_probe"]["parity"]["full_to_cached_kl"])
    results["full_cache_kl_excess_vs_native"] = parity_excess
    passes = (results["delta_nll_candidate_minus_native"] <= 0.10
              and parity_excess <= 0.001
              and cache_result["persistent_cache"]["matches_compact_mla_layout"])
    results["frozen_first_pass_gates"] = {
        "max_delta_nll": 0.10, "max_full_to_cached_kl_excess_vs_native": 0.001,
        "require_compact_mla_cache": True}
    results["qualification"] = "A_QUALIFIED" if passes else "A_NOT_QUALIFIED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"native_nll": results["native"]["nll"],
                      "candidate_nll": results["candidate"]["nll"],
                      "delta_nll": results["delta_nll_candidate_minus_native"],
                      "qualification": results["qualification"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
