from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.nn.functional as F

from ced_ir import CEDIRModel, SlotProbeSuite, make_batch
from train_g1 import atomic_json, autocast_context, build_configs, file_sha256


METHODS = SlotProbeSuite.methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised oracle-block physical-slot probe")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "g1.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-offset", type=int, default=100_000)
    parser.add_argument("--gate-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-examples", type=int, default=2048)
    parser.add_argument("--test-examples", type=int, default=4096)
    return parser.parse_args()


class FeatureCapture:
    def __init__(self, layers: int, heads: int, head_width: int):
        self.layers = layers
        self.heads = heads
        self.head_width = head_width
        self.batch = None
        self.features = {}

    def hook(self, layer: int):
        def capture(_module, args):
            x, memory, _positions = args[:3]
            batch = self.batch
            query_positions = batch["query_groups"].reshape(x.shape[0], -1)
            if not bool((query_positions == query_positions[:1]).all()):
                raise RuntimeError("probe expects aligned supervised positions within a batch")
            query = x[:, query_positions[0]].detach().float()
            source_positions = batch["source_digit_positions"].reshape(x.shape[0], -1)
            blocks = torch.div(source_positions, 2, rounding_mode="floor")
            slot_values = memory.values.view(
                x.shape[0], self.heads, -1, 2, self.head_width
            ).permute(0, 2, 1, 3, 4)
            row = torch.arange(x.shape[0], device=x.device)[:, None]
            slots = slot_values[row, blocks].detach().float()
            self.features[layer] = (query, slots)
        return capture


def extract(model, capture, batch, device):
    capture.batch = batch
    capture.features = {}
    with torch.inference_mode(), autocast_context(device):
        model(batch["input_ids"])
    if len(capture.features) != len(model.decoder):
        raise RuntimeError("not all decoder-layer features were captured")
    # Tensors created under inference_mode cannot be saved by probe autograd.
    # Clone outside that context to obtain ordinary immutable feature tensors.
    features = {layer: (query.clone(), slots.clone())
                for layer, (query, slots) in capture.features.items()}
    target = (batch["source_digit_positions"].reshape(batch["input_ids"].shape[0], -1) % 2)
    return features, target


def synthetic_control(device: torch.device) -> dict[str, float]:
    torch.manual_seed(991)
    batch, count, d_model, heads = 192, 2, 32, 4
    a = d_model // heads
    labels = torch.randint(0, 2, (batch, count), device=device)
    sign = labels.float().mul(2).sub(1)
    query = torch.randn(batch, count, d_model, device=device) * 0.03
    query[..., 0] = sign
    query[..., 1] = 1.0
    slots = torch.randn(batch, count, heads, 2, a, device=device) * 0.03
    target_index = labels[:, :, None, None, None].expand(-1, -1, heads, 1, a)
    other_index = (1 - labels)[:, :, None, None, None].expand(-1, -1, heads, 1, a)
    slots.scatter_(3, target_index, 1.0)
    slots.scatter_(3, other_index, -1.0)
    suite = SlotProbeSuite(d_model, heads, gate_dim=8).to(device)
    optimizer = torch.optim.Adam(suite.parameters(), lr=0.05)
    target = labels[:, :, None].expand(-1, -1, heads)
    for _ in range(100):
        outputs = suite(query, slots)
        loss = sum(F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
                   for logits in outputs.values()) / len(outputs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = {name: float((logits.argmax(dim=-1) == target).float().mean())
              for name, logits in suite(query, slots).items()}
    if min(result.values()) < 0.98:
        raise RuntimeError(f"synthetic probe control failed: {result}")
    return result


def empty_stats(layers: int, heads: int):
    return [{method: {"correct": [0] * heads, "count": [0] * heads,
                      "nll": [0.0] * heads,
                      "phase_correct": [[0] * heads for _ in range(4)],
                      "phase_count": [[0] * heads for _ in range(4)]}
             for method in METHODS} for _ in range(layers)]


@torch.inference_mode()
def evaluate(model, capture, suites, data_cfg, split, count, batch_size, device):
    stats = empty_stats(len(suites), model.cfg.heads)
    for first in range(0, count, batch_size):
        batch = make_batch(data_cfg, split, range(first, min(first + batch_size, count)), device)
        features, target = extract(model, capture, batch, device)
        expanded = target[:, :, None].expand(-1, -1, model.cfg.heads)
        for layer, suite in enumerate(suites):
            query, slots = features[layer]
            for method, logits in suite(query, slots).items():
                losses = F.cross_entropy(logits.reshape(-1, 2), expanded.reshape(-1),
                                         reduction="none").view_as(expanded)
                correct = logits.argmax(dim=-1) == expanded
                item = stats[layer][method]
                for head in range(model.cfg.heads):
                    item["correct"][head] += int(correct[:, :, head].sum())
                    item["count"][head] += correct[:, :, head].numel()
                    item["nll"][head] += float(losses[:, :, head].sum())
                for phase in range(4):
                    rows = batch["phase"] == phase
                    if not bool(rows.any()):
                        continue
                    for head in range(model.cfg.heads):
                        item["phase_correct"][phase][head] += int(correct[rows, :, head].sum())
                        item["phase_count"][phase][head] += correct[rows, :, head].numel()
    result = []
    for layer in range(len(suites)):
        methods = {}
        for method in METHODS:
            item = stats[layer][method]
            head_accuracy = [item["correct"][h] / item["count"][h]
                             for h in range(model.cfg.heads)]
            head_nll = [item["nll"][h] / item["count"][h]
                        for h in range(model.cfg.heads)]
            phase_accuracy = []
            for phase in range(4):
                correct = sum(item["phase_correct"][phase])
                observations = sum(item["phase_count"][phase])
                phase_accuracy.append({"phase": phase, "accuracy": correct / observations,
                                       "observations": observations})
            methods[method] = {
                "head_accuracy": head_accuracy,
                "head_nll": head_nll,
                "head_macro_accuracy": sum(head_accuracy) / len(head_accuracy),
                "head_macro_nll": sum(head_nll) / len(head_nll),
                "phase_accuracy": phase_accuracy,
            }
        result.append({"layer": layer, "methods": methods})
    return {"split": split, "examples": count, "layers": result}


def decide(test: dict) -> tuple[str, list[str]]:
    notes = []
    gate_clean = False
    broad_only = False
    all_near = True
    for layer in test["layers"]:
        methods = layer["methods"]
        q = methods["q_only"]["head_macro_accuracy"]
        b = methods["bilinear"]["head_macro_accuracy"]
        joint = methods["qz_linear"]["head_macro_accuracy"]
        if q >= 0.80:
            notes.append(f"layer {layer['layer']} q-only control is high ({q:.4f})")
        gate_clean |= b >= 0.90 and b - q >= 0.10
        broad_only |= joint >= 0.90 and joint - q >= 0.10
        all_near &= b <= 0.60 and joint <= 0.60
    if gate_clean:
        return "GATE_FAMILY_CLEANLY_DECODABLE", notes
    if broad_only:
        return "BROAD_LINEAR_DECODABLE_ONLY", notes
    if all_near:
        return "TESTED_FAMILIES_NEAR_CHANCE", notes
    return "INTERMEDIATE_OR_CONTROL_CONFOUNDED", notes


def write_markdown(path: Path, report: dict) -> None:
    lines = ["# Supervised oracle-block physical-slot probe", "",
             f"Formal verdict: `{report['verdict']}`", "",
             "The B backbone was frozen. The probe was trained with direct physical-slot labels; this is a decodability result, not an LM repair.", "",
             "## Test accuracy", "",
             "| layer | method | head macro | head 0 | head 1 | head 2 | head 3 |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for layer in report["test"]["layers"]:
        for method in METHODS:
            result = layer["methods"][method]
            heads = result["head_accuracy"]
            lines.append(f"| {layer['layer']} | {method} | {result['head_macro_accuracy']:.4f} | "
                         + " | ".join(f"{value:.4f}" for value in heads) + " |")
    lines += ["", "## Qualification", "",
              "Synthetic learnability control: " + ", ".join(
                  f"`{key}={value:.3f}`" for key, value in report["synthetic_control"].items()),
              "", "Full validation and phase-stratified test metrics are in `summary.json`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.updates != 256:
        raise ValueError("formal protocol fixes updates at 256")
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    control = synthetic_control(device)
    torch.manual_seed(raw["seed"] + 5001)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(raw["seed"] + 5001)
    model_cfg, data_cfg = build_configs(raw, "B_PACK2_WIDE2")
    checkpoint = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("variant") != "B_PACK2_WIDE2" or int(checkpoint.get("update", -1)) != 2048:
        raise ValueError("probe requires terminal B update-2048 checkpoint")
    model = CEDIRModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suites = torch.nn.ModuleList([
        SlotProbeSuite(model_cfg.d_model, model_cfg.heads, args.gate_dim)
        for _ in range(model_cfg.decoder_layers)]).to(device)
    optimizer = torch.optim.AdamW(suites.parameters(), lr=args.learning_rate, weight_decay=0.0)
    capture = FeatureCapture(model_cfg.decoder_layers, model_cfg.heads,
                             model_cfg.d_model // model_cfg.heads)
    handles = [block.global_read.register_forward_pre_hook(capture.hook(layer))
               for layer, block in enumerate(model.decoder)]
    train_events = []
    try:
        for update in range(args.updates):
            first = args.train_offset + update * args.batch_size
            batch = make_batch(data_cfg, "train", range(first, first + args.batch_size), device)
            features, target = extract(model, capture, batch, device)
            expanded = target[:, :, None].expand(-1, -1, model_cfg.heads)
            losses = {}
            total = torch.zeros((), device=device)
            for layer, suite in enumerate(suites):
                query, slots = features[layer]
                for method, logits in suite(query, slots).items():
                    loss = F.cross_entropy(logits.reshape(-1, 2), expanded.reshape(-1))
                    losses[f"layer{layer}_{method}"] = float(loss.detach())
                    total = total + loss
            total = total / (len(suites) * len(METHODS))
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(suites.parameters(), 1.0))
            optimizer.step()
            if (update + 1) % 64 == 0:
                train_events.append({"update": update + 1, "mean_loss": float(total.detach()),
                                     "grad_norm": grad_norm, **losses})
        validation = evaluate(model, capture, suites, data_cfg, "validation",
                              args.validation_examples, args.batch_size, device)
        test = evaluate(model, capture, suites, data_cfg, "test",
                        args.test_examples, args.batch_size, device)
    finally:
        for handle in handles:
            handle.remove()
    verdict, notes = decide(test)
    report = {
        "case_id": raw["case_id"],
        "probe": "SUPERVISED_ORACLE_BLOCK_PHYSICAL_SLOT_V01",
        "protocol": "slot_probe_protocol.md",
        "config_sha256": file_sha256(args.config),
        "base_checkpoint_update": 2048,
        "backbone_frozen": True,
        "train": {"updates": args.updates, "batch_size": args.batch_size,
                  "example_offset": args.train_offset, "learning_rate": args.learning_rate,
                  "gate_dim": args.gate_dim, "events": train_events},
        "synthetic_control": control,
        "validation": validation,
        "test": test,
        "verdict": verdict,
        "notes": notes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "summary.json", report)
    write_markdown(args.output / "PROBE.md", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
