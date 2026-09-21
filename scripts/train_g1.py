from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F

from ced_ir import CEDConfig, CEDIRModel, SyntheticConfig, make_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["A_TOKEN", "B_PACK2_WIDE2", "C_PACK2_NARROW1"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "g1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates", type=int, default=512)
    parser.add_argument("--val-examples", type=int)
    parser.add_argument("--test-at-end", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-gpu-hours", type=float)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment(device: torch.device) -> dict:
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update({
            "gpu_name": props.name,
            "gpu_total_memory": props.total_memory,
            "gpu_compute_capability": [props.major, props.minor],
            "gpu_uuid": (str(getattr(props, "uuid", "")) or None),
        })
    return result


def build_configs(raw: dict, variant: str) -> tuple[CEDConfig, SyntheticConfig]:
    v = raw["variants"][variant]
    model = CEDConfig(
        vocab_size=raw["vocab_size"], d_model=raw["d_model"], heads=raw["heads"],
        encoder_layers=raw["encoder_layers"], decoder_layers=raw["decoder_layers"],
        local_window=raw["local_window"], mlp_intermediate=raw["mlp_intermediate"],
        block_size=v["block_size"], width_multiplier=v["width_multiplier"],
        init_seed=raw["seed"],
    )
    d = raw["data"]
    data = SyntheticConfig(
        sequence_length=raw["sequence_length"], record_count=d["record_count"],
        query_count=d["query_count"], value_digits=d["value_digits"],
        data_seed=raw["data_seed"], min_source_query_gap=d["min_source_query_gap"],
        key_start=d["key_start"], key_stop=d["key_stop"],
        digit_start=d["digit_start"], digit_stop=d["digit_stop"],
        noise_start=d["noise_start"], noise_stop=d["noise_stop"],
    )
    return model, data


def optimizer_for(model: torch.nn.Module, raw: dict) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "norm" in name or "embedding" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": raw["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=raw["learning_rate"], betas=tuple(raw["betas"]), eps=raw["eps"],
    )


def lr_at(update: int, raw: dict) -> float:
    if update < raw["warmup_updates"]:
        return raw["learning_rate"] * (update + 1) / raw["warmup_updates"]
    progress = min(1.0, (update - raw["warmup_updates"]) /
                   (raw["schedule_updates"] - raw["warmup_updates"]))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return raw["min_learning_rate"] + cosine * (raw["learning_rate"] - raw["min_learning_rate"])


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def evaluate(model: CEDIRModel, data_cfg: SyntheticConfig, split: str,
             count: int, batch_size: int, device: torch.device,
             ablate_global: bool = False, save_path: Path | None = None) -> dict:
    model.eval()
    loss_sum = 0.0
    target_count = 0
    query_correct = 0
    query_count = 0
    sample_rows = []
    for start in range(0, count, batch_size):
        ids = list(range(start, min(start + batch_size, count)))
        batch = make_batch(data_cfg, split, ids, device)
        with autocast_context(device):
            logits = model(batch["input_ids"], ablate_global=ablate_global)
        mask = batch["loss_mask"]
        selected_logits = logits[mask].float()
        selected_labels = batch["labels"][mask]
        losses = F.cross_entropy(selected_logits, selected_labels, reduction="none")
        loss_sum += float(losses.sum())
        target_count += losses.numel()
        predictions = logits.argmax(dim=-1)
        for row, example_id in enumerate(ids):
            groups = batch["query_groups"][row]
            correctness = (predictions[row, groups] == batch["labels"][row, groups]).all(dim=-1)
            correct = int(correctness.sum().item())
            query_correct += correct
            query_count += correctness.numel()
            sample_rows.append({"example_id": example_id, "correct_queries": correct,
                                "total_queries": int(correctness.numel())})
    result = {
        "split": split,
        "examples": count,
        "nll": loss_sum / target_count,
        "query_exact_match": query_correct / query_count,
        "query_correct": query_correct,
        "query_count": query_count,
        "global_ablation": ablate_global,
    }
    if save_path is not None:
        with save_path.open("w", encoding="utf-8") as handle:
            for row in sample_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return result


def save_checkpoint(path: Path, model: CEDIRModel, optimizer: torch.optim.Optimizer,
                    update: int, elapsed_device_seconds: float, raw: dict, variant: str) -> None:
    temp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
        "elapsed_device_seconds": elapsed_device_seconds, "raw_config": raw, "variant": variant,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }, temp)
    os.replace(temp, path)


def main() -> None:
    args = parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_updates < 1 or args.max_updates > raw["max_updates"]:
        raise ValueError("max-updates outside frozen protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    model_cfg, data_cfg = build_configs(raw, args.variant)
    random.seed(raw["seed"])
    np.random.seed(raw["seed"])
    torch.manual_seed(raw["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(raw["seed"])
    model = CEDIRModel(model_cfg).to(device)
    optimizer = optimizer_for(model, raw)
    start_update = 0
    elapsed_device_seconds = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["variant"] != args.variant or checkpoint["raw_config"] != raw:
            raise ValueError("resume provenance mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_update = int(checkpoint["update"])
        elapsed_device_seconds = float(checkpoint["elapsed_device_seconds"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])

    manifest = {
        "case_id": raw["case_id"], "variant": args.variant,
        "model_config": asdict(model_cfg), "data_config": asdict(data_cfg),
        "parameter_count": model.parameter_count(), "config_sha256": file_sha256(args.config),
        "source_sha256": file_sha256(ROOT / "src" / "ced_ir" / "model.py"),
        "started_unix": time.time(), "environment": environment(device),
        "start_update": start_update, "requested_terminal_update": args.max_updates,
    }
    atomic_json(args.output / "run_manifest.json", manifest)
    atomic_json(args.output / "environment.json", manifest["environment"])
    events = args.output / "metrics.jsonl"
    max_gpu_hours = args.max_gpu_hours if args.max_gpu_hours is not None else raw["max_gpu_hours"]
    microbatch = raw["microbatch"]
    accumulation = raw["accumulation"]
    measured_step_times: list[float] = []

    for update in range(start_update, args.max_updates):
        wall_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loss = 0.0
        for micro in range(accumulation):
            first_id = (update * accumulation + micro) * microbatch
            ids = range(first_id, first_id + microbatch)
            batch = make_batch(data_cfg, "train", ids, device)
            with autocast_context(device):
                logits = model(batch["input_ids"])
                loss = F.cross_entropy(logits[batch["loss_mask"]].float(),
                                       batch["labels"][batch["loss_mask"]])
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            train_loss += float(loss.detach()) / accumulation
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), raw["gradient_clip_norm"]))
        lr = lr_at(update, raw)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - wall_start
        elapsed_device_seconds += step_seconds
        if update >= start_update + 10:
            measured_step_times.append(step_seconds)
        record = {"event": "train", "update": update + 1, "loss": train_loss,
                  "grad_norm": grad_norm, "lr": lr, "step_seconds": step_seconds,
                  "elapsed_device_hours": elapsed_device_seconds / 3600.0}
        if not math.isfinite(train_loss) or not math.isfinite(grad_norm):
            append_jsonl(events, {**record, "fatal": "nonfinite"})
            save_checkpoint(args.output / "checkpoint_invalid.pt", model, optimizer, update + 1,
                            elapsed_device_seconds, raw, args.variant)
            raise RuntimeError("nonfinite training state")
        append_jsonl(events, record)

        should_validate = (update + 1) % raw["val_every_updates"] == 0 or update + 1 == args.max_updates
        if should_validate:
            val_count = args.val_examples or raw["validation_examples"]
            validation = evaluate(model, data_cfg, "validation", val_count,
                                  microbatch, device)
            validation.update({"event": "validation", "update": update + 1})
            append_jsonl(events, validation)
            save_checkpoint(args.output / f"checkpoint_{update + 1:04d}.pt", model, optimizer,
                            update + 1, elapsed_device_seconds, raw, args.variant)
        if elapsed_device_seconds / 3600.0 >= max_gpu_hours:
            append_jsonl(events, {"event": "budget_stop", "update": update + 1,
                                  "elapsed_device_hours": elapsed_device_seconds / 3600.0})
            break

    final_update = update + 1
    final_val = evaluate(model, data_cfg, "validation",
                         args.val_examples or raw["validation_examples"], microbatch, device)
    ablated_val = evaluate(model, data_cfg, "validation",
                           args.val_examples or raw["validation_examples"], microbatch, device,
                           ablate_global=True)
    summary = {
        "case_id": raw["case_id"], "variant": args.variant, "final_update": final_update,
        "parameter_count": model.parameter_count(), "validation": final_val,
        "validation_global_ablated": ablated_val,
        "global_ablation_drop": final_val["query_exact_match"] - ablated_val["query_exact_match"],
        "elapsed_device_hours": elapsed_device_seconds / 3600.0,
        "mean_measured_step_seconds": (sum(measured_step_times) / len(measured_step_times)
                                       if measured_step_times else None),
        "measured_input_tokens_per_second": (
            raw["effective_tokens_per_update"] * len(measured_step_times) / sum(measured_step_times)
            if measured_step_times else None),
        "budget_limited": elapsed_device_seconds / 3600.0 >= max_gpu_hours,
    }
    if device.type == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if args.test_at_end:
        summary["test"] = evaluate(model, data_cfg, "test", raw["test_examples"],
                                   microbatch, device, save_path=args.output / "test_samples.jsonl")
        summary["test_global_ablated"] = evaluate(
            model, data_cfg, "test", raw["test_examples"], microbatch, device,
            ablate_global=True, save_path=args.output / "test_samples_global_ablated.jsonl")
    atomic_json(args.output / "summary.json", summary)
    append_jsonl(ROOT / "runs" / "budget_ledger.jsonl", {
        "case_id": raw["case_id"], "variant": args.variant, "stage": "G1",
        "run_directory": str(args.output.resolve()), "device_hours": elapsed_device_seconds / 3600.0,
        "tokens": final_update * raw["effective_tokens_per_update"], "end_unix": time.time(),
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
