"""Evaluate a converted token-MLA checkpoint against its native reference.

This is a read-only gate. It does not perform TransMLA conversion or training.
The two checkpoints must use the same tokenizer and token IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2 or args.sequences < 1:
        raise ValueError("sequence-length must be >=2 and sequences >=1")
    tokens = load_validation(args.tokens)
    if tokens.numel() < args.sequence_length * args.sequences:
        raise ValueError("not enough validation tokens for non-overlapping sequences")
    native_tokenizer = AutoTokenizer.from_pretrained(args.native)
    candidate_tokenizer = AutoTokenizer.from_pretrained(args.candidate)
    if native_tokenizer.get_vocab() != candidate_tokenizer.get_vocab():
        raise ValueError("native and candidate tokenizers differ")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    results = {}
    for label, path in (("native", args.native), ("candidate", args.candidate)):
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(args.device)
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
    results["qualification"] = "MEASURED_NOT_YET_JUDGED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"native_nll": results["native"]["nll"],
                      "candidate_nll": results["candidate"]["nll"],
                      "delta_nll": results["delta_nll_candidate_minus_native"],
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
