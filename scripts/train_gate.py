from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F

from ced_ir import CEDIRModel, make_batch
from train_g1 import (append_jsonl, atomic_json, autocast_context, build_configs,
                      environment, evaluate, file_sha256)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "g1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-dim", type=int, default=16)
    parser.add_argument("--max-updates", type=int, default=128)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--test-at-end", action="store_true")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def gate_parameters(model: CEDIRModel):
    return [(name, parameter) for name, parameter in model.named_parameters()
            if "gate_query" in name or "gate_slot_proj" in name]


def save_checkpoint(path: Path, model: CEDIRModel, optimizer: torch.optim.Optimizer,
                    gate_update: int, elapsed_seconds: float, raw: dict,
                    base_checkpoint: Path, gate_dim: int) -> None:
    temp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "gate_update": gate_update, "elapsed_device_seconds": elapsed_seconds,
                "raw_config": raw, "base_checkpoint": str(base_checkpoint.resolve()),
                "gate_dim": gate_dim, "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temp)
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    if args.max_updates < 1 or args.max_updates > 512:
        raise ValueError("gate experiment is capped at 512 updates")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    base_cfg, data_cfg = build_configs(raw, "B_PACK2_WIDE2")
    gated_cfg = replace(base_cfg, slot_gate_dim=args.gate_dim)
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("variant") != "B_PACK2_WIDE2" or int(checkpoint.get("update", -1)) != 2048:
        raise ValueError("gate must start from the terminal B-2048 checkpoint")

    torch.manual_seed(raw["seed"] + 3001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(raw["seed"] + 3001)
    base = CEDIRModel(base_cfg).to(device)
    base.load_state_dict(checkpoint["model"])
    base.eval()
    model = CEDIRModel(gated_cfg).to(device)
    incompat = model.load_state_dict(checkpoint["model"], strict=False)
    expected_missing = {f"decoder.{layer}.global_read.{suffix}"
                        for layer in range(gated_cfg.decoder_layers)
                        for suffix in ("gate_slot_proj", "gate_query.weight")}
    if set(incompat.missing_keys) != expected_missing or incompat.unexpected_keys:
        raise RuntimeError({"missing": incompat.missing_keys, "unexpected": incompat.unexpected_keys})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_gate = gate_parameters(model)
    for _, parameter in named_gate:
        parameter.requires_grad_(True)
    if not named_gate:
        raise RuntimeError("no gate parameters")

    args.output.mkdir(parents=True, exist_ok=True)
    probe = make_batch(data_cfg, "validation", range(2), device)
    with torch.inference_mode(), autocast_context(device):
        base_logits = base(probe["input_ids"])
        gate_logits = model(probe["input_ids"])
    equivalence = {
        "max_abs_logit_difference": float((base_logits.float() - gate_logits.float()).abs().max()),
        "mean_abs_logit_difference": float((base_logits.float() - gate_logits.float()).abs().mean()),
    }
    if equivalence["max_abs_logit_difference"] > 0.02:
        raise RuntimeError(f"update-zero gate is not equivalent: {equivalence}")
    del base, base_logits, gate_logits, probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW([p for _, p in named_gate], lr=raw["learning_rate"],
                                  betas=tuple(raw["betas"]), eps=raw["eps"],
                                  weight_decay=raw["weight_decay"])
    start_update = 0
    elapsed = 0.0
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        if resume["gate_dim"] != args.gate_dim or resume["raw_config"] != raw:
            raise ValueError("gate resume provenance mismatch")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_update = int(resume["gate_update"])
        elapsed = float(resume["elapsed_device_seconds"])
        torch.set_rng_state(resume["torch_rng"].cpu())
        if resume["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in resume["cuda_rng"]])

    manifest = {
        "case_id": raw["case_id"], "variant": "B_LEARNED_SLOT",
        "base_checkpoint": str(args.base_checkpoint.resolve()), "base_update": 2048,
        "gate_dim": args.gate_dim, "gate_parameter_count": sum(p.numel() for _, p in named_gate),
        "trainable_names": [name for name, _ in named_gate],
        "total_parameter_count": model.parameter_count(), "update_zero_equivalence": equivalence,
        "config_sha256": file_sha256(args.config), "environment": environment(device),
        "started_unix": time.time(), "requested_gate_updates": args.max_updates,
        "data_cursor_offset_examples": 2048 * raw["accumulation"] * raw["microbatch"],
    }
    atomic_json(args.output / "run_manifest.json", manifest)
    events = args.output / "metrics.jsonl"
    microbatch, accumulation = raw["microbatch"], raw["accumulation"]
    data_offset = manifest["data_cursor_offset_examples"]
    measured = []

    for update in range(start_update, args.max_updates):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for micro in range(accumulation):
            first = data_offset + (update * accumulation + micro) * microbatch
            batch = make_batch(data_cfg, "train", range(first, first + microbatch), device)
            with autocast_context(device):
                logits = model(batch["input_ids"])
                loss = F.cross_entropy(logits[batch["loss_mask"]].float(),
                                       batch["labels"][batch["loss_mask"]])
            (loss / accumulation).backward()
            loss_value += float(loss.detach()) / accumulation
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [p for _, p in named_gate], raw["gradient_clip_norm"]))
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        if update >= start_update + 10:
            measured.append(seconds)
        append_jsonl(events, {"event": "gate_train", "gate_update": update + 1,
                              "loss": loss_value, "grad_norm": grad_norm,
                              "step_seconds": seconds})
        if not math.isfinite(loss_value) or not math.isfinite(grad_norm):
            raise RuntimeError("nonfinite gate training")
        if (update + 1) % 64 == 0 or update + 1 == args.max_updates:
            validation = evaluate(model, data_cfg, "validation",
                                  args.val_examples or raw["validation_examples"],
                                  microbatch, device)
            append_jsonl(events, {"event": "validation", "gate_update": update + 1,
                                  **validation})
            save_checkpoint(args.output / f"checkpoint_gate_{update + 1:04d}.pt",
                            model, optimizer, update + 1, elapsed, raw,
                            args.base_checkpoint, args.gate_dim)

    validation = evaluate(model, data_cfg, "validation",
                          args.val_examples or raw["validation_examples"], microbatch, device)
    ablated = evaluate(model, data_cfg, "validation",
                       args.val_examples or raw["validation_examples"], microbatch, device,
                       ablate_global=True)
    summary = {
        "case_id": raw["case_id"], "variant": "B_LEARNED_SLOT",
        "base_update": 2048, "gate_updates": args.max_updates,
        "update_zero_equivalence": equivalence, "gate_dim": args.gate_dim,
        "gate_parameter_count": sum(p.numel() for _, p in named_gate),
        "gate_parameter_norms": {name: float(p.detach().float().norm()) for name, p in named_gate},
        "validation": validation, "validation_global_ablated": ablated,
        "global_ablation_drop": validation["query_exact_match"] - ablated["query_exact_match"],
        "elapsed_device_hours": elapsed / 3600.0,
        "mean_step_seconds": sum(measured) / len(measured) if measured else None,
    }
    if device.type == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if args.test_at_end:
        summary["test"] = evaluate(model, data_cfg, "test", raw["test_examples"],
                                   microbatch, device,
                                   save_path=args.output / "test_samples.jsonl")
        summary["test_global_ablated"] = evaluate(
            model, data_cfg, "test", raw["test_examples"], microbatch, device,
            ablate_global=True, save_path=args.output / "test_samples_global_ablated.jsonl")
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
