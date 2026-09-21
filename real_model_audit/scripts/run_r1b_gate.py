#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_r01_gate as base  # noqa: E402
from aoc.oracle import block_targets, entropy_effective_rank, fit_oracle, oracle_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run final high-mass R1b gate")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--output", type=Path, default=ROOT / "runs/r1b_high_mass_v01")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 16, 27])
    p.add_argument("--examples-per-domain", type=int, default=4)
    p.add_argument("--candidate-start", type=int, default=256)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--num-blocks", type=int, default=8)
    p.add_argument("--select-start", type=int, default=1728)
    p.add_argument("--fit-start", type=int, default=1856)
    p.add_argument("--test-start", type=int, default=1984)
    p.add_argument("--span-len", type=int, default=64)
    p.add_argument("--recent-exact", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def weighted_unit(metric: torch.Tensor, mass: torch.Tensor) -> float:
    weight = mass.mean(dim=(-1, -2)).clamp_min(1e-12)
    return float(((metric.square() * weight).sum() / weight.sum()).sqrt())


def metrics_for(
    q: torch.Tensor,
    result: Any,
    keys: torch.Tensor,
    values: torch.Tensor,
    mass: torch.Tensor,
    scale: float,
) -> tuple[float, float]:
    true_z, true_mu = block_targets(q, keys, values, scale)
    z, mu, _ = oracle_metrics(
        q, result.keys, result.values, result.log_mass,
        true_z, true_mu, mass, scale,
    )
    return weighted_unit(z, mass), weighted_unit(mu, mass)


def choose_region(
    q_select: torch.Tensor,
    full_keys: torch.Tensor,
    candidate_starts: list[int],
    block_size: int,
    num_blocks: int,
    scale: float,
    query_pos: torch.Tensor,
) -> tuple[int, float]:
    scored = []
    for start in candidate_starts:
        _, region_mass = base.block_attention_mass(
            q_select, full_keys, start, block_size, num_blocks, scale, query_pos
        )
        scored.append(float(region_mass.mean()))
    best = int(np.argmax(scored))
    return candidate_starts[best], scored[best]


def aggregate(records: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    def med(name: str) -> float:
        return float(np.median([r[name] for r in records]))

    qualification = {
        "manual_exact_max": max(r["manual_exact_rel"] for r in records),
        "exact8_fit_z_max": max(r["exact8_fit_z"] for r in records),
        "exact8_fit_mu_max": max(r["exact8_fit_mu"] for r in records),
        "exact8_test_z_max": max(r["exact8_test_z"] for r in records),
        "exact8_test_mu_max": max(r["exact8_test_mu"] for r in records),
    }
    recovery_fraction = float(np.mean([
        r["recovery8_test_z"] <= 0.05 and r["recovery8_test_mu"] <= 0.05
        for r in records
    ]))
    c4_fraction = float(np.mean([
        r["c4_test_z"] <= 0.10 and r["c4_test_mu"] <= 0.10
        for r in records
    ]))
    stable_mass_fraction = float(np.mean([r["test_mass"] >= 0.02 for r in records]))
    summary = {
        "n_units": len(records),
        "qualification": qualification,
        "median_raw_effective_rank": med("raw_effective_rank"),
        "median_centered_effective_rank": med("centered_effective_rank"),
        "median_select_mass": med("select_mass"),
        "median_fit_mass": med("fit_mass"),
        "median_test_mass": med("test_mass"),
        "fraction_test_mass_ge_0_02": stable_mass_fraction,
        "median_recovery8_fit_z": med("recovery8_fit_z"),
        "median_recovery8_fit_mu": med("recovery8_fit_mu"),
        "median_recovery8_test_z": med("recovery8_test_z"),
        "median_recovery8_test_mu": med("recovery8_test_mu"),
        "fraction_recovery8_test_both_le_0_05": recovery_fraction,
        "median_c4_fit_z": med("c4_fit_z"),
        "median_c4_fit_mu": med("c4_fit_mu"),
        "median_c4_test_z": med("c4_test_z"),
        "median_c4_test_mu": med("c4_test_mu"),
        "fraction_c4_test_both_le_0_10": c4_fraction,
        "median_c4_projected_rel": med("c4_projected_rel"),
        "median_c4_logit_kl": med("c4_logit_kl"),
        "median_c4_delta_nll": med("c4_delta_nll"),
    }
    invalid = (
        qualification["manual_exact_max"] > 1e-2
        or any(qualification[k] > 1e-4 for k in qualification if k != "manual_exact_max")
    )
    recovery_ok = (
        summary["median_recovery8_fit_z"] <= 0.02
        and summary["median_recovery8_fit_mu"] <= 0.02
        and summary["median_recovery8_test_z"] <= 0.02
        and summary["median_recovery8_test_mu"] <= 0.02
        and recovery_fraction >= 0.75
    )
    c4_ok = (
        summary["median_c4_test_z"] <= 0.10
        and summary["median_c4_test_mu"] <= 0.10
        and c4_fraction >= 0.75
    )
    if invalid:
        verdict = "INVALID_IMPLEMENTATION"
    elif not recovery_ok:
        verdict = "ORACLE_OPTIMIZER_UNQUALIFIED"
    elif summary["median_select_mass"] < 0.02:
        verdict = "NO_HIGH_MASS_CANDIDATE"
    elif summary["median_test_mass"] < 0.02 or stable_mass_fraction < 0.50:
        verdict = "NO_STABLE_HIGH_MASS_TARGET"
    elif c4_ok:
        verdict = "PASS_R1B_ORACLE_8_TO_4"
    else:
        verdict = "STOP_LOCAL_OPERATOR_COMPRESSION"
    summary["optimizer_recovery_qualified"] = recovery_ok
    summary["c4_operator_pass"] = c4_ok
    summary["verdict"] = verdict
    summary["layers"] = layers
    return summary


def write_results(output: Path, agg: dict[str, Any]) -> None:
    q = agg["qualification"]
    lines = [
        "# R1b final high-mass audit",
        "",
        f"**Formal verdict: `{agg['verdict']}`**",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Units | {agg['n_units']} |",
        f"| Select mass median | {agg['median_select_mass']:.4f} |",
        f"| Fit mass median | {agg['median_fit_mass']:.4f} |",
        f"| Test mass median | {agg['median_test_mass']:.4f} |",
        f"| Test mass >=2% fraction | {agg['fraction_test_mass_ge_0_02']:.3f} |",
        f"| 8->8 recovery fit log-Z / mu | {agg['median_recovery8_fit_z']:.4f} / {agg['median_recovery8_fit_mu']:.4f} |",
        f"| 8->8 recovery test log-Z / mu | {agg['median_recovery8_test_z']:.4f} / {agg['median_recovery8_test_mu']:.4f} |",
        f"| 8->8 recovery test unit fraction | {agg['fraction_recovery8_test_both_le_0_05']:.3f} |",
        f"| 8->4 fit log-Z / mu | {agg['median_c4_fit_z']:.4f} / {agg['median_c4_fit_mu']:.4f} |",
        f"| 8->4 test log-Z / mu | {agg['median_c4_test_z']:.4f} / {agg['median_c4_test_mu']:.4f} |",
        f"| 8->4 test unit fraction | {agg['fraction_c4_test_both_le_0_10']:.3f} |",
        f"| 8->4 projected output error | {agg['median_c4_projected_rel']:.4f} |",
        f"| 8->4 logit KL / delta NLL | {agg['median_c4_logit_kl']:.6f} / {agg['median_c4_delta_nll']:.6f} |",
        f"| Raw / centered effective rank | {agg['median_raw_effective_rank']:.3f} / {agg['median_centered_effective_rank']:.3f} |",
        "",
        "## Qualification",
        "",
        f"- manual exact maximum: `{q['manual_exact_max']:.3e}`",
        f"- exact-start 8->8 fit maximum: Z `{q['exact8_fit_z_max']:.3e}`, mu `{q['exact8_fit_mu_max']:.3e}`",
        f"- exact-start 8->8 test maximum: Z `{q['exact8_test_z_max']:.3e}`, mu `{q['exact8_test_mu_max']:.3e}`",
        f"- non-identity optimizer recovery qualified: `{agg['optimizer_recovery_qualified']}`",
        "",
        "R2 compiler training and R3 adaptation remain frozen unless the formal verdict is `PASS_R1B_ORACLE_8_TO_4`.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.seq_len = 384
        args.layers = [0]
        args.examples_per_domain = 1
        args.candidate_start = 64
        args.select_start = 256
        args.fit_start = 304
        args.test_start = 352
        args.span_len = 16
        args.recent_exact = 64
        args.num_blocks = 2
        args.steps = 3
        args.restarts = 1
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    docs, sources = base.fetch_sources(output / "source_cache", args.examples_per_domain)
    examples = base.make_examples(tokenizer, docs, args.examples_per_domain, args.seq_len, args.seed)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=cache_dir, dtype=dtype, attn_implementation="eager"
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    region_len = args.block_size * args.num_blocks
    select_end = args.select_start + args.span_len
    fit_end = args.fit_start + args.span_len
    test_end = args.test_start + args.span_len
    if test_end > args.seq_len:
        raise ValueError("test span exceeds sequence length")
    candidate_last = args.select_start - args.recent_exact - region_len
    candidates = list(range(args.candidate_start, candidate_last + 1, region_len))
    if not candidates:
        raise ValueError("no eligible candidate regions")
    positions = torch.arange(args.seq_len, device=args.device).unsqueeze(0)
    select_pos = torch.arange(args.select_start, select_end, device=args.device)
    fit_pos = torch.arange(args.fit_start, fit_end, device=args.device)
    test_pos = torch.arange(args.test_start, test_end, device=args.device)
    records: list[dict[str, Any]] = []

    for ex_idx, example in enumerate(examples):
        input_ids = torch.tensor(example["ids"], device=args.device).unsqueeze(0)
        captured, handles = base.capture_hooks(model, args.layers, args.test_start, test_end)
        with torch.inference_mode():
            baseline = model(
                input_ids=input_ids, position_ids=positions,
                output_hidden_states=True, use_cache=False,
            )
        for handle in handles:
            handle.remove()
        base_logits = baseline.logits[:, args.test_start : test_end - 1].float()
        targets = input_ids[:, args.test_start + 1 : test_end]

        for layer_idx in args.layers:
            with torch.inference_mode():
                q, k, v, attn = base.qkv_for_layer(
                    model, baseline.hidden_states[layer_idx], layer_idx, positions
                )
            h_kv = k.shape[0]
            groups = q.shape[0] // h_kv
            q = q.reshape(h_kv, groups, args.seq_len, attn.head_dim)
            q_select = q[:, :, args.select_start:select_end]
            q_fit = q[:, :, args.fit_start:fit_end]
            q_test = q[:, :, args.test_start:test_end]
            region_start, select_mass = choose_region(
                q_select, k, candidates, args.block_size, args.num_blocks,
                attn.scaling, select_pos,
            )
            keys = torch.stack([
                k[:, region_start + b * args.block_size : region_start + (b + 1) * args.block_size]
                for b in range(args.num_blocks)
            ])
            values = torch.stack([
                v[:, region_start + b * args.block_size : region_start + (b + 1) * args.block_size]
                for b in range(args.num_blocks)
            ])
            fit_mass_blocks, fit_mass_region = base.block_attention_mass(
                q_fit, k, region_start, args.block_size, args.num_blocks,
                attn.scaling, fit_pos,
            )
            test_mass_blocks, test_mass_region = base.block_attention_mass(
                q_test, k, region_start, args.block_size, args.num_blocks,
                attn.scaling, test_pos,
            )
            score = torch.einsum("hgtd,bhrd->bhgtr", q_test.float(), keys.float()) * attn.scaling
            raw_rank = float(entropy_effective_rank(score.flatten(2, 3)).median())
            centered = score - score.mean(dim=-1, keepdim=True)
            centered_rank = float(entropy_effective_rank(centered.flatten(2, 3)).median())

            exact8 = fit_oracle(
                q_fit, keys, values, fit_mass_blocks, c=args.block_size,
                scale=attn.scaling, restarts=1, steps=50 if not args.smoke else 2,
                lr=0.01, seed=args.seed + ex_idx * 101 + layer_idx,
                init_mode="partition",
            )
            exact8_fit_z, exact8_fit_mu = metrics_for(
                q_fit, exact8, keys, values, fit_mass_blocks, attn.scaling
            )
            exact8_test_z, exact8_test_mu = metrics_for(
                q_test, exact8, keys, values, test_mass_blocks, attn.scaling
            )
            recovery8 = fit_oracle(
                q_fit, keys, values, fit_mass_blocks, c=args.block_size,
                scale=attn.scaling, restarts=args.restarts, steps=args.steps,
                lr=args.lr, seed=args.seed + ex_idx * 1009 + layer_idx * 17,
                init_mode="pair_split", noise_scale=0.02,
            )
            recovery8_fit_z, recovery8_fit_mu = metrics_for(
                q_fit, recovery8, keys, values, fit_mass_blocks, attn.scaling
            )
            recovery8_test_z, recovery8_test_mu = metrics_for(
                q_test, recovery8, keys, values, test_mass_blocks, attn.scaling
            )
            c4 = fit_oracle(
                q_fit, keys, values, fit_mass_blocks, c=4,
                scale=attn.scaling, restarts=args.restarts, steps=args.steps,
                lr=args.lr, seed=args.seed + ex_idx * 2017 + layer_idx * 31,
                init_mode="partition", noise_scale=0.02,
            )
            c4_fit_z, c4_fit_mu = metrics_for(
                q_fit, c4, keys, values, fit_mass_blocks, attn.scaling
            )
            c4_test_z, c4_test_mu = metrics_for(
                q_test, c4, keys, values, test_mass_blocks, attn.scaling
            )

            exact_heads = base.compressed_head_output(
                q_test, k, v, keys, values,
                torch.zeros(keys.shape[:-1], device=args.device),
                region_start, region_len, attn.scaling, test_pos,
            )
            exact_projected = base.project_heads(exact_heads, attn.o_proj, dtype)
            model_exact_heads = base.model_exact_head_output(q_test, k, v, attn.scaling, test_pos)
            model_exact_projected = base.project_heads(model_exact_heads, attn.o_proj, dtype)
            manual_exact_rel = base.rel_error(model_exact_projected, captured[layer_idx])
            c4_heads = base.compressed_head_output(
                q_test, k, v, c4.keys, c4.values, c4.log_mass,
                region_start, region_len, attn.scaling, test_pos,
            )
            c4_projected = base.project_heads(c4_heads, attn.o_proj, dtype)
            mod_logits = base.intervention_forward(
                model, input_ids, layer_idx, args.test_start, test_end, c4_projected
            )
            c4_kl, c4_dnll = base.evaluate_logits(base_logits, mod_logits, targets)
            row = {
                "domain": example["domain"], "source": example["source"],
                "example_index": ex_idx, "layer": layer_idx,
                "region_start": region_start, "select_mass": select_mass,
                "fit_mass": float(fit_mass_region.mean()),
                "test_mass": float(test_mass_region.mean()),
                "raw_effective_rank": raw_rank,
                "centered_effective_rank": centered_rank,
                "manual_exact_rel": manual_exact_rel,
                "exact8_fit_z": exact8_fit_z, "exact8_fit_mu": exact8_fit_mu,
                "exact8_test_z": exact8_test_z, "exact8_test_mu": exact8_test_mu,
                "recovery8_fit_z": recovery8_fit_z, "recovery8_fit_mu": recovery8_fit_mu,
                "recovery8_test_z": recovery8_test_z, "recovery8_test_mu": recovery8_test_mu,
                "c4_fit_z": c4_fit_z, "c4_fit_mu": c4_fit_mu,
                "c4_test_z": c4_test_z, "c4_test_mu": c4_test_mu,
                "c4_projected_rel": base.rel_error(c4_projected, exact_projected),
                "c4_logit_kl": c4_kl, "c4_delta_nll": c4_dnll,
            }
            records.append(row)
            print(json.dumps({"progress": row}, sort_keys=True), flush=True)
        del baseline
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    agg = aggregate(records, args.layers)
    payload = {
        "protocol": "R1B_HIGH_MASS_v0.1",
        "model": args.model,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "layout": {
            "candidate_starts": candidates,
            "select_span": [args.select_start, select_end],
            "fit_span": [args.fit_start, fit_end],
            "test_span": [args.test_start, test_end],
        },
        "sources": sources,
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        },
        "duration_seconds": time.time() - started,
        "aggregate": agg,
        "records": records,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_results(output, agg)
    print(json.dumps({"formal_verdict": agg["verdict"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
