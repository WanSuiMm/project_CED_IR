#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ced_migration import TokenAlignedCED  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Qualify token-aligned Qwen-to-CED implementation")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=ROOT / "runs/g0_qwen06b_v01")
    p.add_argument("--device", default="cuda")
    p.add_argument("--encoder-layers", type=int, default=14)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--parity-length", type=int, default=32)
    p.add_argument("--remote-length", type=int, default=896)
    p.add_argument("--train-length", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))


def sample_ids(tokenizer, length: int, device: str) -> torch.Tensor:
    text = (
        "A causal language model should preserve exact temporal boundaries while "
        "learning a shared long-range interface. This paragraph mixes prose, numbers "
        "1739 and 8421, and code-like names such as update_cache and memory_key. "
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = (ids * (length // len(ids) + 1))[:length]
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.parity_length = 12
        args.remote_length = args.encoder_layers * (args.window - 1) + 3
        args.train_length = 16
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cache_dir = str(args.cache_dir.resolve()) if args.cache_dir else None
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=cache_dir,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = TokenAlignedCED.from_qwen(
        base,
        encoder_layers=args.encoder_layers,
        encoder_window=args.window,
        local_query_chunk=args.chunk,
    ).to(args.device)
    del base
    model.eval()

    parity_ids = sample_ids(tokenizer, args.parity_length, args.device)
    with torch.inference_mode():
        batched = model(parity_ids).logits
        prefill_logits, deployment_cache = model.prefill(parity_ids[:, :-1])
        continued_logits, deployment_cache = model.step(
            parity_ids[:, -1:], deployment_cache
        )
        deployment_full = torch.cat(
            [batched[:, -2:-1], batched[:, -1:]], dim=1
        )
        deployment_cached = torch.cat([prefill_logits, continued_logits], dim=1)
        cache = model.new_cache()
        pieces = []
        for position in range(parity_ids.shape[1]):
            logits, cache = model.step(parity_ids[:, position : position + 1], cache)
            pieces.append(logits)
        cached = torch.cat(pieces, dim=1)
    parity_delta = (deployment_full - deployment_cached).abs()
    batched_logp = torch.log_softmax(deployment_full, dim=-1)
    cached_logp = torch.log_softmax(deployment_cached, dim=-1)
    cached_kl = float(
        (batched_logp.exp() * (batched_logp - cached_logp)).sum(dim=-1).mean()
    )
    cached_top1 = float(
        (deployment_full.argmax(dim=-1) == deployment_cached.argmax(dim=-1))
        .float()
        .mean()
    )
    stress_delta = (batched - cached).abs()
    stress_full_logp = torch.log_softmax(batched, dim=-1)
    stress_cached_logp = torch.log_softmax(cached, dim=-1)
    stress_kl = float(
        (stress_full_logp.exp() * (stress_full_logp - stress_cached_logp))
        .sum(dim=-1)
        .mean()
    )

    causal_ids = sample_ids(tokenizer, max(24, args.train_length), args.device)
    edited = causal_ids.clone()
    edited[:, -1] = (edited[:, -1] + 17) % model.config.vocab_size
    with torch.inference_mode():
        causal_a = model(causal_ids).logits[:, :-1]
        causal_b = model(edited).logits[:, :-1]
    future_edit_max = float((causal_a - causal_b).abs().max())

    remote_ids = sample_ids(tokenizer, args.remote_length, args.device)
    remote_edit = remote_ids.clone()
    remote_edit[:, 0] = (remote_edit[:, 0] + 29) % model.config.vocab_size
    with torch.inference_mode():
        remote_full = model(remote_ids).logits[:, -1]
        remote_full_edit = model(remote_edit).logits[:, -1]
        remote_local = model(remote_ids, disable_global_memory=True).logits[:, -1]
        remote_local_edit = model(remote_edit, disable_global_memory=True).logits[:, -1]
    remote_path = {
        "enabled_remote_edit_rel": rel(remote_full_edit, remote_full),
        "disabled_remote_edit_max": float((remote_local_edit - remote_local).abs().max()),
        "global_ablation_rel": rel(remote_local, remote_full),
    }

    accounting = model.cache_accounting(deployment_cache)
    expected_local_max = (
        len(model.encoder)
        * 2
        * parity_ids.shape[0]
        * model.config.num_key_value_heads
        * min(args.parity_length, args.window)
        * model.config.head_dim
    )
    expected_global = (
        2
        * parity_ids.shape[0]
        * model.config.num_key_value_heads
        * args.parity_length
        * model.config.head_dim
    )
    accounting["expected_local_kv_max"] = expected_local_max
    accounting["expected_global_kv"] = expected_global

    model.train()
    model.enable_gradient_checkpointing(True)
    train_ids = sample_ids(tokenizer, args.train_length, args.device)
    # FP32 is used only for structural parity. SGD avoids allocating two full
    # Adam state tensors on a shared GPU; the deployed BF16 qualification uses
    # AdamW and therefore still exercises the formal optimizer path.
    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=1e-6)
        if args.dtype == "bf16"
        else torch.optim.SGD(model.parameters(), lr=1e-6)
    )
    before = next(model.parameters()).detach().float().clone()
    train_out = model(train_ids, labels=train_ids)
    optimizer.zero_grad(set_to_none=True)
    train_out.loss.backward()
    grad_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    parameter_moved = bool((next(model.parameters()).detach().float() - before).abs().max() > 0)
    del optimizer
    model.enable_gradient_checkpointing(False)
    model.eval()

    checkpoint_path = output / "transient_reload.pt"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    with torch.inference_mode():
        reload_reference = model(parity_ids[:, :8]).logits
    first_parameter = next(model.parameters())
    with torch.no_grad():
        first_parameter.add_(0.125)
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    with torch.inference_mode():
        reload_actual = model(parity_ids[:, :8]).logits
    reload_max = float((reload_actual - reload_reference).abs().max())
    checkpoint_path.unlink()

    metrics = {
        "future_edit_max": future_edit_max,
        "cached_max_abs": float(parity_delta.max()),
        "cached_mean_abs": float(parity_delta.mean()),
        "cached_relative": rel(deployment_cached, deployment_full),
        "cached_mean_kl": cached_kl,
        "cached_top1_agreement": cached_top1,
        "all_step_stress_max_abs": float(stress_delta.max()),
        "all_step_stress_mean_abs": float(stress_delta.mean()),
        "all_step_stress_mean_kl": stress_kl,
        "remote_path": remote_path,
        "cache_accounting": accounting,
        "train_loss": float(train_out.loss.detach()),
        "grad_finite": grad_finite,
        "grad_norm": grad_norm,
        "parameter_moved": parameter_moved,
        "reload_max_abs": reload_max,
        "checkpoint_sha256_before_delete": checkpoint_sha256,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    parity_ok = (
        metrics["cached_max_abs"] <= 1e-3
        and metrics["cached_mean_abs"] <= 1e-4
    ) if args.dtype == "float32" else (
        metrics["cached_mean_kl"] <= 1e-3
        and metrics["cached_top1_agreement"] >= 0.99
    )
    passed = (
        metrics["future_edit_max"] == 0.0
        and parity_ok
        and remote_path["disabled_remote_edit_max"] <= 2e-3
        and remote_path["global_ablation_rel"] > 0.0
        and accounting["local_kv_elements"] <= expected_local_max
        and accounting["global_kv_elements"] == expected_global
        and accounting["retained_encoder_state_elements"] == 0
        and accounting["full_history_layer_kv_copies"] == 0
        and np.isfinite(metrics["train_loss"])
        and grad_finite
        and parameter_moved
        and reload_max == 0.0
    )
    verdict = "PASS_G0_CED_IMPLEMENTATION" if passed else "INVALID_CED_IMPLEMENTATION"
    payload = {
        "protocol": "QWEN_CED_MIGRATION_G0_v0.1",
        "verdict": verdict,
        "model": args.model,
        "arguments": {
            key: (
                "<external-model-cache>"
                if key == "cache_dir" and value is not None
                else str(value) if isinstance(value, Path) else value
            )
            for key, value in vars(args).items()
        },
        "metrics": metrics,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        },
        "duration_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (output / "RESULTS.md").write_text(
        "\n".join(
            [
                "# Qwen-to-CED G0 qualification",
                "",
                f"**Verdict: `{verdict}`**",
                "",
                f"- future-edit maximum: `{future_edit_max:.3e}`",
                f"- cached parity max / mean: `{metrics['cached_max_abs']:.3e}` / `{metrics['cached_mean_abs']:.3e}`",
                f"- cached parity relative / KL / top-1: `{metrics['cached_relative']:.3e}` / `{cached_kl:.3e}` / `{cached_top1:.3f}`",
                f"- local-only remote-edit maximum: `{remote_path['disabled_remote_edit_max']:.3e}`",
                f"- global-memory ablation relative change: `{remote_path['global_ablation_rel']:.3e}`",
                f"- train loss / grad norm: `{metrics['train_loss']:.6f}` / `{grad_norm:.6f}`",
                f"- save-reload maximum: `{reload_max:.3e}`",
                "",
                "A formal C/A migration run is authorized only by `PASS_G0_CED_IMPLEMENTATION`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
