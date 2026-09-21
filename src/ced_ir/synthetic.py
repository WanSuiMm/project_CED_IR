from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np
import torch


PAD, BOS, EOS, REC, VAL, SEP, ASK, ANS = range(8)


@dataclass(frozen=True)
class SyntheticConfig:
    sequence_length: int = 512
    record_count: int = 16
    query_count: int = 8
    value_digits: int = 4
    data_seed: int = 20260921
    min_source_query_gap: int = 192
    key_start: int = 32
    key_stop: int = 544
    digit_start: int = 544
    digit_stop: int = 560
    noise_start: int = 560
    noise_stop: int = 624


def _seed(cfg: SyntheticConfig, split: str, example_id: int) -> int:
    payload = f"associative_recall_v01|{cfg.data_seed}|{split}|{example_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def make_example(cfg: SyntheticConfig, split: str, example_id: int) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(_seed(cfg, split, example_id))
    keys = rng.choice(np.arange(cfg.key_start, cfg.key_stop), size=cfg.record_count, replace=False)
    values = rng.integers(cfg.digit_start, cfg.digit_stop,
                          size=(cfg.record_count, cfg.value_digits), dtype=np.int64)
    query_indices = rng.choice(cfg.record_count, size=cfg.query_count, replace=False)
    rng.shuffle(query_indices)

    tokens: list[int] = [BOS]
    phase = int(rng.integers(0, 4))
    tokens.extend(rng.integers(cfg.noise_start, cfg.noise_stop, size=phase).tolist())
    key_positions: dict[int, int] = {}
    for i in range(cfg.record_count):
        tokens.append(REC)
        key_positions[i] = len(tokens)
        tokens.extend([int(keys[i]), VAL, *values[i].tolist(), SEP])

    query_len = cfg.query_count * (4 + cfg.value_digits)
    total_len = cfg.sequence_length + 1
    filler_len = total_len - len(tokens) - query_len - 1
    if filler_len < 0:
        raise ValueError("sequence is too short for configured records and queries")
    tokens.extend(rng.integers(cfg.noise_start, cfg.noise_stop, size=filler_len).tolist())

    target_positions: list[int] = []
    query_groups: list[list[int]] = []
    for record_idx in query_indices.tolist():
        query_key_position = len(tokens) + 1
        if query_key_position - key_positions[record_idx] <= cfg.min_source_query_gap:
            raise AssertionError("source-query gap invariant violated")
        tokens.extend([ASK, int(keys[record_idx]), ANS])
        group: list[int] = []
        for digit in values[record_idx].tolist():
            # The current input position predicts the digit appended next.
            target_positions.append(len(tokens) - 1)
            group.append(len(tokens) - 1)
            tokens.append(int(digit))
        tokens.append(SEP)
        query_groups.append(group)
    tokens.append(EOS)

    if len(tokens) != total_len:
        raise AssertionError((len(tokens), total_len))
    input_ids = np.asarray(tokens[:-1], dtype=np.int64)
    labels = np.asarray(tokens[1:], dtype=np.int64)
    loss_mask = np.zeros(cfg.sequence_length, dtype=np.bool_)
    loss_mask[np.asarray(target_positions)] = True
    if int(loss_mask.sum()) != cfg.query_count * cfg.value_digits:
        raise AssertionError("wrong supervised target count")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "query_groups": np.asarray(query_groups, dtype=np.int64),
        "example_id": example_id,
    }


def make_batch(cfg: SyntheticConfig, split: str, example_ids: Iterable[int],
               device: torch.device | str) -> dict[str, torch.Tensor]:
    examples = [make_example(cfg, split, int(i)) for i in example_ids]
    result: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "labels", "loss_mask", "query_groups"):
        array = np.stack([e[key] for e in examples])
        result[key] = torch.from_numpy(array).to(device)
    result["example_id"] = torch.tensor([int(e["example_id"]) for e in examples], device=device)
    return result
