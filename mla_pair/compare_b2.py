"""Frozen paired 1M-token A/B2 NLL screen and gap trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(a: dict, b: dict) -> dict:
    if a["variant"] != "A_TOKEN" or b["variant"] != "B2_HEADWISE":
        raise ValueError("wrong A/B2 variants")
    for field in ("seed", "steps", "train_tokens", "sequence_length",
                  "eval_sequences", "eval_every", "rope_dim_per_head",
                  "backbone_lr", "compiler_lr"):
        if a[field] != b[field]:
            raise ValueError(f"A/B2 mismatch in {field}")
    if [item["step"] for item in a["curve"]] != [item["step"] for item in b["curve"]]:
        raise ValueError("A/B2 validation checkpoints differ")
    gap_curve = []
    for a_item, b_item in zip(a["curve"], b["curve"]):
        differences = np.asarray(b_item["per_sequence_nll"], dtype=np.float64) - np.asarray(
            a_item["per_sequence_nll"], dtype=np.float64)
        if len(differences) != a["eval_sequences"]:
            raise ValueError("paired validation length mismatch")
        gap_curve.append({"step": a_item["step"],
                          "train_tokens": a_item["train_tokens"],
                          "a_nll": a_item["validation_nll"],
                          "b2_nll": b_item["validation_nll"],
                          "b2_minus_a_nll": float(differences.mean())})
    final_differences = np.asarray(b["curve"][-1]["per_sequence_nll"], dtype=np.float64) - np.asarray(
        a["curve"][-1]["per_sequence_nll"], dtype=np.float64)
    rng = np.random.default_rng(a["seed"])
    resamples = rng.integers(0, len(final_differences),
                             size=(10000, len(final_differences)))
    ci = np.quantile(final_differences[resamples].mean(axis=1), [0.025, 0.975])
    final_gap = gap_curve[-1]["b2_minus_a_nll"]
    # For the four-checkpoint formal run: compare the last 25% of training.
    last_quarter_reduction = (gap_curve[-2]["b2_minus_a_nll"] - final_gap
                              if len(gap_curve) >= 5 else None)
    if final_gap <= 0.10:
        screen = "WITHIN_MARGIN"
    elif last_quarter_reduction is None:
        screen = "SMOKE_ONLY"
    elif last_quarter_reduction > 0.05:
        screen = "OUTSIDE_MARGIN_STILL_IMPROVING"
    else:
        screen = "OUTSIDE_MARGIN_NO_CLEAR_LATE_IMPROVEMENT"
    content, rope = a["content_dim"], a["rope_dim"]
    return {
        "status": "FULL_COADAPT_B2_SCREEN_ONLY",
        "train_tokens_per_arm": a["train_tokens"],
        "eval_sequences": a["eval_sequences"],
        "gap_curve": gap_curve,
        "final_b2_minus_a_nll": final_gap,
        "final_paired_bootstrap_95pct_ci": [float(ci[0]), float(ci[1])],
        "screen_margin_nll": 0.10,
        "late_gap_reduction_threshold_nll": 0.05,
        "last_quarter_gap_reduction_nll": last_quarter_reduction,
        "screen_signal_only": screen,
        "b2_to_a_reference_cache_ratio_even_length":
            (content / 2 + rope) / (content + rope),
        "extra_compiler_parameters_B2": b["extra_compiler_parameters"],
        "claim_boundary": "single-seed viability NLL screen and reference-cache storage; no latency or architecture proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(json.loads(args.a.read_text(encoding="utf-8")),
                     json.loads(args.b.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
