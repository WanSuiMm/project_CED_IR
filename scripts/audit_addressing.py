from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F

from ced_ir import CEDIRModel, make_batch
from train_g1 import atomic_json, autocast_context, build_configs, file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-training addressing audit for CED-IR G1 B")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "g1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


class RoutingAccumulator:
    def __init__(self, layers: int, heads: int):
        self.layers = layers
        self.heads = heads
        self.current_batch = None
        self.by_layer = [self._empty() for _ in range(layers)]

    def _empty(self):
        return [{"count": 0, "top1": 0, "top2": 0, "top4": 0,
                 "reciprocal_rank": 0.0, "digit_mass": 0.0,
                 "record_mass": 0.0, "entropy": 0.0,
                 "slot_count": 0, "slot_correct": 0,
                 "slot_probability": 0.0}
                for _ in range(self.heads)]

    def hook(self, layer: int):
        def inspect(module, args):
            x, memory, positions = args[:3]
            batch = self.current_batch
            target_positions = batch["query_groups"].reshape(batch["input_ids"].shape[0], -1)
            if not bool((target_positions == target_positions[:1]).all()):
                raise RuntimeError("audit expects aligned supervised positions within a batch")
            audit_positions = target_positions[0]
            coarse, gate = module.routing_diagnostics(
                x[:, audit_positions], memory, positions[audit_positions])
            source_positions = batch["source_digit_positions"].reshape_as(target_positions)
            record_spans = batch["source_record_spans"][:, :, None, :].expand(
                -1, -1, batch["query_groups"].shape[-1], -1).reshape(
                    batch["input_ids"].shape[0], -1, 2)
            selected = coarse.permute(0, 2, 1, 3)
            digit_blocks = torch.div(source_positions, 2, rounding_mode="floor")
            digit_mass = selected.gather(-1, digit_blocks[:, :, None, None].expand(
                -1, -1, self.heads, 1)).squeeze(-1)
            ranks = 1 + (selected > digit_mass[..., None]).sum(dim=-1)
            entropy = -(selected.float() * selected.float().clamp_min(1e-30).log()).sum(dim=-1)
            block_ids = torch.arange(selected.shape[-1], device=selected.device)
            record_first = torch.div(record_spans[..., 0], 2, rounding_mode="floor")
            record_last = torch.div(record_spans[..., 1], 2, rounding_mode="floor")
            in_record = ((block_ids[None, None, None, :] >= record_first[:, :, None, None]) &
                         (block_ids[None, None, None, :] <= record_last[:, :, None, None]))
            record_mass = (selected * in_record).sum(dim=-1)
            for head in range(self.heads):
                stats = self.by_layer[layer][head]
                head_ranks = ranks[:, :, head]
                stats["count"] += head_ranks.numel()
                stats["top1"] += int((head_ranks <= 1).sum())
                stats["top2"] += int((head_ranks <= 2).sum())
                stats["top4"] += int((head_ranks <= 4).sum())
                stats["reciprocal_rank"] += float((1.0 / head_ranks.float()).sum())
                stats["digit_mass"] += float(digit_mass[:, :, head].sum())
                stats["record_mass"] += float(record_mass[:, :, head].sum())
                stats["entropy"] += float(entropy[:, :, head].sum())
            if gate is not None:
                chosen_gate = gate.permute(0, 2, 1, 3, 4)
                chosen_gate = chosen_gate.gather(
                    3, digit_blocks[:, :, None, None, None].expand(-1, -1, self.heads, 1, 2)
                ).squeeze(3)
                physical_slot = (source_positions % 2)[:, :, None]
                slot_correct = chosen_gate.argmax(dim=-1) == physical_slot
                slot_probability = chosen_gate.gather(-1, physical_slot[..., None].expand(
                    -1, -1, self.heads, 1)).squeeze(-1)
                for head in range(self.heads):
                    stats = self.by_layer[layer][head]
                    stats["slot_count"] += slot_correct[:, :, head].numel()
                    stats["slot_correct"] += int(slot_correct[:, :, head].sum())
                    stats["slot_probability"] += float(slot_probability[:, :, head].sum())
        return inspect

    @staticmethod
    def _finalize(stats):
        count = stats["count"]
        result = {
            "observations": count,
            "correct_digit_block_top1": stats["top1"] / count,
            "correct_digit_block_top2": stats["top2"] / count,
            "correct_digit_block_top4": stats["top4"] / count,
            "correct_digit_block_mrr": stats["reciprocal_rank"] / count,
            "mean_correct_digit_block_mass": stats["digit_mass"] / count,
            "mean_source_record_mass": stats["record_mass"] / count,
            "mean_attention_entropy": stats["entropy"] / count,
        }
        if stats["slot_count"]:
            result["physical_slot_accuracy_at_digit_block"] = (
                stats["slot_correct"] / stats["slot_count"])
            result["mean_physical_slot_probability_at_digit_block"] = (
                stats["slot_probability"] / stats["slot_count"])
        return result

    def result(self):
        layers = []
        pooled = {key: 0 for key in self._empty()[0]}
        for layer, heads in enumerate(self.by_layer):
            finalized_heads = []
            for head, stats in enumerate(heads):
                finalized_heads.append({"head": head, **self._finalize(stats)})
                for key, value in stats.items():
                    pooled[key] += value
            layers.append({"layer": layer, "heads": finalized_heads})
        return {"pooled_layer_head": self._finalize(pooled), "layers": layers}


class PhaseAccumulator:
    def __init__(self):
        self.stats = [{"target_count": 0, "nll_sum": 0.0,
                       "query_count": 0, "query_correct": 0} for _ in range(4)]

    def update(self, logits, batch):
        losses = F.cross_entropy(logits.transpose(1, 2).float(), batch["labels"], reduction="none")
        predictions = logits.argmax(dim=-1)
        for phase in range(4):
            rows = batch["phase"] == phase
            if not bool(rows.any()):
                continue
            mask = batch["loss_mask"][rows]
            stats = self.stats[phase]
            stats["target_count"] += int(mask.sum())
            stats["nll_sum"] += float(losses[rows][mask].sum())
            positions = batch["query_groups"][rows]
            row_ids = torch.arange(int(rows.sum()), device=logits.device)[:, None, None]
            predicted = predictions[rows][row_ids, positions]
            expected = batch["labels"][rows][row_ids, positions]
            exact = (predicted == expected).all(dim=-1)
            stats["query_count"] += exact.numel()
            stats["query_correct"] += int(exact.sum())

    def result(self):
        return [{"phase": phase, "examples_implied": stats["query_count"] // 8,
                 "nll": stats["nll_sum"] / stats["target_count"],
                 "four_digit_exact": stats["query_correct"] / stats["query_count"]}
                for phase, stats in enumerate(self.stats)]


def compiler_geometry(model: CEDIRModel):
    weight = model.compiler.proj.weight.detach().float().cpu()
    d = model.cfg.d_model
    energies = [[float(weight[i*d:(i+1)*d, j*d:(j+1)*d].square().sum())
                 for j in range(2)] for i in range(2)]
    total = sum(sum(row) for row in energies)
    singular = torch.linalg.svdvals(weight)
    identity = torch.eye(weight.shape[0])
    return {
        "block_frobenius_energy": energies,
        "cross_mixing_ratio": (energies[0][1] + energies[1][0]) / total,
        "relative_distance_from_identity": float((weight - identity).norm() / identity.norm()),
        "spectral_condition_number": float(singular.max() / singular.min()),
        "singular_value_min": float(singular.min()),
        "singular_value_median": float(singular.median()),
        "singular_value_max": float(singular.max()),
    }


def load_models(args, raw, device):
    base_cfg, data_cfg = build_configs(raw, "B_PACK2_WIDE2")
    base_checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    if base_checkpoint.get("variant") != "B_PACK2_WIDE2" or base_checkpoint.get("update") != 2048:
        raise ValueError("base checkpoint must be terminal B update 2048")
    base = CEDIRModel(base_cfg).to(device)
    base.load_state_dict(base_checkpoint["model"])
    gate_checkpoint = torch.load(args.gate_checkpoint, map_location=device, weights_only=False)
    gate_dim = int(gate_checkpoint["gate_dim"])
    if int(gate_checkpoint["gate_update"]) != 512:
        raise ValueError("gate checkpoint must be terminal gate update 512")
    gated = CEDIRModel(replace(base_cfg, slot_gate_dim=gate_dim)).to(device)
    gated.load_state_dict(gate_checkpoint["model"])
    return base.eval(), gated.eval(), data_cfg


def audit_model(name, model, data_cfg, args, device):
    routing = RoutingAccumulator(len(model.decoder), model.cfg.heads)
    phase = PhaseAccumulator()
    handles = [block.global_read.register_forward_pre_hook(routing.hook(layer))
               for layer, block in enumerate(model.decoder)]
    try:
        for first in range(0, args.examples, args.batch_size):
            ids = range(first, min(first + args.batch_size, args.examples))
            batch = make_batch(data_cfg, "test", ids, device)
            routing.current_batch = batch
            with torch.inference_mode(), autocast_context(device):
                logits = model(batch["input_ids"])
            phase.update(logits, batch)
    finally:
        for handle in handles:
            handle.remove()
    return {"model": name, "routing": routing.result(), "phase": phase.result()}


def write_markdown(path: Path, report: dict):
    lines = ["# Addressing audit", "",
             "Zero-training diagnostics on the frozen 4,096-example G1 test split.", "",
             "## Layer-mean routing", "",
             "| model | layer | digit top-1 | digit top-4 | digit mass | record mass | slot accuracy | slot probability |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for result in report["models"]:
        for layer in result["routing"]["layers"]:
            heads = layer["heads"]
            mean = lambda key: sum(head[key] for head in heads) / len(heads)
            has_slot = "physical_slot_accuracy_at_digit_block" in heads[0]
            slot = mean("physical_slot_accuracy_at_digit_block") if has_slot else None
            slot_p = mean("mean_physical_slot_probability_at_digit_block") if has_slot else None
            lines.append(f"| {result['model']} | {layer['layer']} | "
                         f"{mean('correct_digit_block_top1'):.4f} | "
                         f"{mean('correct_digit_block_top4'):.4f} | "
                         f"{mean('mean_correct_digit_block_mass'):.4f} | "
                         f"{mean('mean_source_record_mass'):.4f} | "
                         f"{'n/a' if slot is None else f'{slot:.4f}'} | "
                         f"{'n/a' if slot_p is None else f'{slot_p:.4f}'} |")
    lines += ["", "## Phase-stratified task performance", "",
              "| model | phase | NLL | four-digit exact |", "|---|---:|---:|---:|"]
    for result in report["models"]:
        for phase in result["phase"]:
            lines.append(f"| {result['model']} | {phase['phase']} | {phase['nll']:.6f} | "
                         f"{phase['four_digit_exact']:.6f} |")
    geometry = report["compiler_geometry"]
    lines += ["", "## Compiler geometry", "",
              f"- cross-mixing ratio: `{geometry['cross_mixing_ratio']:.6f}`",
              f"- relative distance from identity: `{geometry['relative_distance_from_identity']:.6f}`",
              f"- spectral condition number: `{geometry['spectral_condition_number']:.6f}`",
              "", "Per-layer and per-head values are retained in `audit.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.examples < 1 or args.batch_size < 1:
        raise ValueError("examples and batch-size must be positive")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    base, gated, data_cfg = load_models(args, raw, device)
    geometry = compiler_geometry(base)
    report = {
        "case_id": raw["case_id"],
        "audit": "ZERO_TRAINING_ADDRESSING_AUDIT_V01",
        "split": "test",
        "examples": args.examples,
        "config_sha256": file_sha256(args.config),
        "compiler_geometry": geometry,
        "models": [audit_model("B_PACK2_WIDE2", base, data_cfg, args, device),
                   audit_model("B_LEARNED_SLOT", gated, data_cfg, args, device)],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "audit.json", report)
    write_markdown(args.output / "AUDIT.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
