#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


TRAIN_TEXTS = [
    "A shared causal memory should preserve long-range evidence while the decoder changes its interface.",
    "def update_cache(keys, values, position): return keys.append(position), values.append(position)",
    "Scientific migration separates implementation correctness from language-model recovery and systems speed.",
    "The identifiers atlas_1739 and cedar_8421 occur before a delayed query about the earlier record.",
]
TEST_TEXTS = [
    "A held-out paragraph asks whether one global representation can support several downstream transformations.",
    "class MemoryReader: pass  # evaluation text is disjoint from bridge fitting strings",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--length", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe-alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260922)
    return p.parse_args()


def make_ids(tokenizer, text: str, length: int) -> torch.Tensor:
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokens = (tokens * (length // len(tokens) + 1))[:length]
    return torch.tensor(tokens, dtype=torch.long, device="cuda").unsqueeze(0)


def evaluate_bridge(model, batches):
    values = []
    layers = []
    model.eval()
    for batch in batches:
        with torch.enable_grad():
            out = model.bridge_objective(batch)
        values.append(float(out.bridge_loss.detach()))
        layers.append([float(x.detach()) for x in out.bridge_by_layer])
    return float(np.mean(values)), np.mean(np.asarray(layers), axis=0).tolist()


def mean_kl(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    p = torch.log_softmax(reference.float(), -1)
    q = torch.log_softmax(candidate.float(), -1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "mean_kl": mean_kl(reference, candidate),
        "top1_agreement": float(
            (reference.argmax(-1) == candidate.argmax(-1)).float().mean()
        ),
    }


def main() -> None:
    cfg = args()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    started = time.time()
    cache = str(cfg.cache_dir.resolve()) if cfg.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, cache_dir=cache)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model, cache_dir=cache, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    model = HomotopyCED.from_qwen(base).cuda()
    model.set_backbone_trainable(False)
    train = [make_ids(tokenizer, text, cfg.length) for text in TRAIN_TEXTS]
    test = [make_ids(tokenizer, text, cfg.length) for text in TEST_TEXTS]
    initial_train, _ = evaluate_bridge(model, train)
    initial_test, initial_layers = evaluate_bridge(model, test)
    optimizer = torch.optim.AdamW(list(model.bridge_parameters()), lr=cfg.lr)
    trace = []
    model.train()
    for step in range(cfg.steps):
        batch = train[step % len(train)]
        out = model.bridge_objective(batch)
        optimizer.zero_grad(set_to_none=True)
        out.bridge_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.bridge_parameters()), 1.0)
        optimizer.step()
        if step in {0, cfg.steps // 4, cfg.steps // 2, cfg.steps - 1}:
            trace.append({"step": step + 1, "train_bridge": float(out.bridge_loss.detach())})
    final_train, _ = evaluate_bridge(model, train)
    final_test, final_layers = evaluate_bridge(model, test)

    probe = test[0]
    model.eval()
    with torch.inference_mode():
        native = model.base_model(probe, use_cache=False).logits.float()
        alpha_sweep = {}
        for alpha in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
            model.set_mix(alpha, 0.0)
            hybrid = model(probe, force_custom=True).logits
            alpha_sweep[str(alpha)] = {
                "lm_kl": mean_kl(native, hybrid),
                "top1_agreement": float(
                    (native.argmax(-1) == hybrid.argmax(-1)).float().mean()
                ),
            }
        model.set_mix(1.0, 1.0)
        endpoint_full = model(probe, force_custom=True).logits
        prefix_logits, endpoint_cache = model.endpoint_prefill(probe[:, :-1])
        next_logits, _ = model.endpoint_step(probe[:, -1:], endpoint_cache)
        endpoint_reference = torch.cat(
            [endpoint_full[:, -2:-1], endpoint_full[:, -1:]], dim=1
        )
        endpoint_cached = torch.cat([prefix_logits, next_logits], dim=1)
    probe_kl = alpha_sweep[str(cfg.probe_alpha)]["lm_kl"]
    heldout_ratio = final_test / max(initial_test, 1e-12)
    signal = heldout_ratio <= 0.70 and np.isfinite(probe_kl) and probe_kl <= 0.05
    qualified = signal and final_test <= 0.50
    verdict = (
        "PASS_H2_BRIDGE_QUALIFICATION" if qualified else
        "BRIDGE_SIGNAL_PRESENT_NOT_QUALIFIED" if signal else
        "FAIL_H2_BRIDGE_PILOT"
    )
    payload = {
        "protocol": "QWEN_CED_HOMOTOPY_H2_v0.2",
        "verdict": verdict,
        "model": cfg.model,
        "arguments": {
            **vars(cfg),
            "cache_dir": "<external-model-cache>" if cfg.cache_dir else None,
            "output": str(cfg.output),
        },
        "metrics": {
            "initial_train_bridge": initial_train,
            "final_train_bridge": final_train,
            "initial_test_bridge": initial_test,
            "final_test_bridge": final_test,
            "heldout_ratio": heldout_ratio,
            "probe_alpha": cfg.probe_alpha,
            "probe_lm_kl": probe_kl,
            "alpha_sweep": alpha_sweep,
            "trained_endpoint_full_vs_cache": parity(
                endpoint_reference, endpoint_cached
            ),
            "initial_test_by_upper_layer": initial_layers,
            "final_test_by_upper_layer": final_layers,
            "trace": trace,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "duration_seconds": time.time() - started,
    }
    cfg.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "memory_norm": model.memory_norm.state_dict(),
            "memory_k_proj": model.memory_k_proj.state_dict(),
            "memory_k_norm": model.memory_k_norm.state_dict(),
            "memory_v_proj": model.memory_v_proj.state_dict(),
        },
        cfg.output / "bridge_only_checkpoint.pt",
    )
    (cfg.output / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verdict": verdict, "output": str(cfg.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
