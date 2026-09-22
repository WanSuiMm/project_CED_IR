#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ced_homotopy import HomotopyCED  # noqa: E402


TEXT = (
    "A continuous architecture migration should preserve the original causal "
    "computation before moving information into one shared global interface. "
    "The sequence contains prose, identifiers, numbers 1739 and 8421, and code. "
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dtype", choices=("float32", "bf16"), default="float32")
    p.add_argument("--length", type=int, default=96)
    p.add_argument("--remote-length", type=int, default=896)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--skip-remote", action="store_true")
    return p.parse_args()


def ids(tokenizer, length: int, device: str) -> torch.Tensor:
    tokens = tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    tokens = (tokens * (length // len(tokens) + 1))[:length]
    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)


def comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    delta = (ref - cand).abs()
    ref_logp = torch.log_softmax(ref, dim=-1)
    cand_logp = torch.log_softmax(cand, dim=-1)
    kl = (ref_logp.exp() * (ref_logp - cand_logp)).sum(dim=-1)
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "relative": float((ref - cand).norm() / ref.norm().clamp_min(1e-12)),
        "mean_kl": float(kl.mean()),
        "top1_agreement": float((ref.argmax(-1) == cand.argmax(-1)).float().mean()),
    }


def main() -> None:
    cfg = args()
    if cfg.length <= cfg.window:
        raise ValueError("H1 length must exceed the local window")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    started = time.time()
    dtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    cache = str(cfg.cache_dir.resolve()) if cfg.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cache)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model, cache_dir=cache, dtype=dtype, attn_implementation="sdpa"
    ).cuda().eval()
    model = HomotopyCED.from_qwen(base, encoder_window=cfg.window).cuda().eval()
    sample = ids(tokenizer, cfg.length, "cuda")

    with torch.inference_mode():
        native_full = model.base_model(sample, use_cache=False).logits.float()
        model.set_mix(0.0, 0.0)
        custom_start = model(sample, force_custom=True).logits
        prefix = model.base_model(sample[:, :-1], use_cache=True)
        continuation = model.base_model(
            sample[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
        )
        native_deploy_full = torch.cat([native_full[:, -2:-1], native_full[:, -1:]], 1)
        native_deploy_cache = torch.cat([prefix.logits[:, -1:], continuation.logits], 1)

        model.set_mix(1.0, 1.0)
        endpoint_full = model(sample, force_custom=True).logits
        endpoint_prefix, endpoint_cache = model.endpoint_prefill(sample[:, :-1])
        endpoint_next, endpoint_cache = model.endpoint_step(sample[:, -1:], endpoint_cache)
        endpoint_deploy_full = torch.cat(
            [endpoint_full[:, -2:-1], endpoint_full[:, -1:]], 1
        )
        endpoint_deploy_cache = torch.cat([endpoint_prefix, endpoint_next], 1)

        state_buffer = io.BytesIO()
        torch.save(
            {
                "memory_norm": model.memory_norm.state_dict(),
                "memory_k_proj": model.memory_k_proj.state_dict(),
                "memory_k_norm": model.memory_k_norm.state_dict(),
                "memory_v_proj": model.memory_v_proj.state_dict(),
                "alpha": model.alpha.detach().cpu(),
                "beta": model.beta.detach().cpu(),
            },
            state_buffer,
        )
        with torch.no_grad():
            model.memory_k_proj.weight.add_(0.125)
        state_buffer.seek(0)
        restored = torch.load(state_buffer, map_location="cuda", weights_only=True)
        model.memory_norm.load_state_dict(restored["memory_norm"])
        model.memory_k_proj.load_state_dict(restored["memory_k_proj"])
        model.memory_k_norm.load_state_dict(restored["memory_k_norm"])
        model.memory_v_proj.load_state_dict(restored["memory_v_proj"])
        model.set_mix(float(restored["alpha"]), float(restored["beta"]))
        reloaded = model(sample, force_custom=True).logits
        reload_max_abs = float((reloaded - endpoint_full).abs().max())

        causal = sample[:, :24]
        edited = causal.clone()
        edited[:, -1] = (edited[:, -1] + 17) % model.config.vocab_size
        causal_a = model(causal, force_custom=True).logits[:, :-1]
        causal_b = model(edited, force_custom=True).logits[:, :-1]
        future_edit_max = float((causal_a - causal_b).abs().max())

        remote = None
        if not cfg.skip_remote:
            remote_ids = ids(tokenizer, cfg.remote_length, "cuda")
            remote_edit = remote_ids.clone()
            remote_edit[:, 0] = (remote_edit[:, 0] + 29) % model.config.vocab_size
            enabled = model(remote_ids, force_custom=True).logits[:, -1]
            enabled_edit = model(remote_edit, force_custom=True).logits[:, -1]
            disabled = model(
                remote_ids, force_custom=True, disable_global_memory=True
            ).logits[:, -1]
            disabled_edit = model(
                remote_edit, force_custom=True, disable_global_memory=True
            ).logits[:, -1]
            remote = {
                "enabled_edit_relative": float(
                    (enabled - enabled_edit).float().norm()
                    / enabled.float().norm().clamp_min(1e-12)
                ),
                "disabled_edit_max": float((disabled - disabled_edit).abs().max()),
                "ablation_relative": float(
                    (enabled - disabled).float().norm()
                    / enabled.float().norm().clamp_min(1e-12)
                ),
            }

    accounting = model.endpoint_cache_accounting(endpoint_cache)
    model.set_backbone_trainable(False)
    model.train()
    short = sample[:, :16]
    optimizer = torch.optim.AdamW(list(model.bridge_parameters()), lr=1e-6)
    before = model.memory_k_proj.weight.detach().float().clone()
    train = model(short, labels=short, force_custom=True)
    optimizer.zero_grad(set_to_none=True)
    train.loss.backward()
    grad_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for p in model.bridge_parameters()
    )
    optimizer.step()
    parameter_moved = bool(
        (model.memory_k_proj.weight.detach().float() - before).abs().max() > 0
    )

    metrics = {
        "h0_custom_start_vs_native": comparison(native_full, custom_start),
        "native_full_vs_cache": comparison(native_deploy_full, native_deploy_cache),
        "h1_endpoint_full_vs_cache": comparison(
            endpoint_deploy_full, endpoint_deploy_cache
        ),
        "future_edit_max": future_edit_max,
        "remote_path": remote,
        "cache_accounting": accounting,
        "reload_max_abs": reload_max_abs,
        "train_loss": float(train.loss.detach()),
        "bridge_grad_finite": grad_finite,
        "bridge_parameter_moved": parameter_moved,
    }
    h0 = metrics["h0_custom_start_vs_native"]
    h1 = metrics["h1_endpoint_full_vs_cache"]
    h0_pass = (
        h0["max_abs"] <= (1e-3 if cfg.dtype == "float32" else 1.0)
        and h0["mean_kl"] <= (1e-6 if cfg.dtype == "float32" else 5e-3)
    )
    h1_pass = (
        h1["max_abs"] <= (1e-3 if cfg.dtype == "float32" else 1.0)
        and h1["mean_kl"] <= (1e-6 if cfg.dtype == "float32" else 5e-3)
        and h1["top1_agreement"] >= 0.99
        and future_edit_max == 0.0
        and accounting["upper_self_kv_elements"] == 0
        and accounting["retained_encoder_state_elements"] == 0
        and reload_max_abs == 0.0
        and (
            cfg.skip_remote
            or (
                remote is not None
                and remote["disabled_edit_max"] <= 2e-3
                and remote["enabled_edit_relative"] > 0.0
                and remote["ablation_relative"] > 0.0
            )
        )
        and grad_finite
        and parameter_moved
    )
    verdict = (
        "PASS_H0_H1" if h0_pass and h1_pass else
        "FAIL_H0_MORPHISM" if not h0_pass else "FAIL_H1_ENDPOINT"
    )
    payload = {
        "protocol": "QWEN_CED_HOMOTOPY_v0.2",
        "verdict": verdict,
        "model": cfg.model,
        "arguments": {
            **vars(cfg),
            "cache_dir": "<external-model-cache>" if cfg.cache_dir else None,
            "output": str(cfg.output),
        },
        "metrics": metrics,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "duration_seconds": time.time() - started,
    }
    cfg.output.mkdir(parents=True, exist_ok=True)
    (cfg.output / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "output": str(cfg.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
