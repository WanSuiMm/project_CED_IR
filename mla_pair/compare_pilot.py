"""Summarize matched route-2 pilot NLL; no formal architecture verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    if a["variant"] != "A_TOKEN" or b["variant"] != "B_PAIR":
        raise ValueError("wrong A/B summaries")
    for field in ("seed", "steps", "train_tokens", "sequence_length",
                  "eval_sequences", "rope_dim_per_head", "lora_rank"):
        if a[field] != b[field]:
            raise ValueError(f"A/B mismatch in {field}")
    differences = np.array(b["final_per_sequence_nll"], dtype=np.float64) - np.array(
        a["final_per_sequence_nll"], dtype=np.float64)
    if len(differences) != a["eval_sequences"]:
        raise ValueError("paired evaluation length mismatch")
    rng = np.random.default_rng(a["seed"])
    resamples = rng.integers(0, len(differences), size=(10000, len(differences)))
    ci_low, ci_high = np.quantile(differences[resamples].mean(axis=1), [0.025, 0.975])
    content_dim, rope_dim = a["content_dim"], a["rope_dim"]
    ratio = (content_dim / 2 + rope_dim) / (content_dim + rope_dim)
    report = {
        "status": "SMALL_PAIRED_PILOT_ONLY",
        "variant_A": str(args.a), "variant_B": str(args.b),
        "train_tokens_per_variant": a["train_tokens"],
        "eval_sequences": len(differences),
        "a_final_nll": a["final_validation_nll"],
        "b_final_nll": b["final_validation_nll"],
        "paired_b_minus_a_nll": float(differences.mean()),
        "paired_bootstrap_95pct_ci": [float(ci_low), float(ci_high)],
        "screen_margin_nll": 0.10,
        "screen_signal_only": "within_margin" if differences.mean() <= 0.10
        else "outside_margin",
        "a_cache_scalars_per_token_per_layer": content_dim + rope_dim,
        "b_cache_scalars_per_two_tokens_per_layer": content_dim + 2 * rope_dim,
        "b_to_a_cache_ratio_even_length": ratio,
        "extra_compiler_parameters_B": b["extra_compiler_parameters"],
        "claim_boundary": "NLL screening and actual reference-cache storage only; no fused kernel or latency result",
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("status", "paired_b_minus_a_nll", "paired_bootstrap_95pct_ci",
                       "screen_signal_only", "b_to_a_cache_ratio_even_length")}, indent=2))


if __name__ == "__main__":
    main()
