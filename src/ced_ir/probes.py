from __future__ import annotations

import math

import torch
from torch import nn


def permute_record_pairs(query: torch.Tensor, slots: torch.Tensor,
                         target: torch.Tensor, record_count: int,
                         digit_count: int, condition: str):
    """Apply a fixed-point-free record rotation while preserving digit ordinal."""
    if query.shape[1] != record_count * digit_count or target.shape[1] != query.shape[1]:
        raise ValueError("record/digit shape does not match flattened query count")
    batch = query.shape[0]
    query_view = query.view(batch, record_count, digit_count, *query.shape[2:])
    slots_view = slots.view(batch, record_count, digit_count, *slots.shape[2:])
    target_view = target.view(batch, record_count, digit_count)
    if not bool((target_view == target_view[:, :1]).all()):
        raise ValueError("physical-slot target is not record-invariant within ordinal")
    if condition == "matched":
        pass
    elif condition == "query_cyclic":
        query_view = torch.roll(query_view, shifts=1, dims=1)
    elif condition == "block_cyclic":
        slots_view = torch.roll(slots_view, shifts=1, dims=1)
    elif condition == "paired_cyclic":
        query_view = torch.roll(query_view, shifts=1, dims=1)
        slots_view = torch.roll(slots_view, shifts=1, dims=1)
    else:
        raise ValueError(f"unknown permutation condition: {condition}")
    return (query_view.reshape_as(query), slots_view.reshape_as(slots),
            target_view.reshape_as(target))


class SlotProbeSuite(nn.Module):
    """Head-wise controls for decoding a known block's physical source slot."""

    methods = ("q_only", "z_only", "qz_linear", "bilinear")

    def __init__(self, d_model: int, heads: int, gate_dim: int = 16):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.head_width = d_model // heads
        self.gate_dim = gate_dim
        self.q_only = nn.Linear(d_model, heads * 2)
        self.z_only_weight = nn.Parameter(torch.empty(heads, 2, 2 * self.head_width))
        self.z_only_bias = nn.Parameter(torch.zeros(heads, 2))
        self.qz_weight = nn.Parameter(
            torch.empty(heads, 2, d_model + 2 * self.head_width))
        self.qz_bias = nn.Parameter(torch.zeros(heads, 2))
        self.bilinear_query = nn.Linear(d_model, heads * gate_dim, bias=False)
        self.bilinear_slot = nn.Parameter(
            torch.empty(heads, 2, self.head_width, gate_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.q_only.weight, std=0.02)
        nn.init.zeros_(self.q_only.bias)
        nn.init.normal_(self.z_only_weight, std=0.02)
        nn.init.normal_(self.qz_weight, std=0.02)
        nn.init.normal_(self.bilinear_query.weight, std=0.02)
        nn.init.normal_(self.bilinear_slot, std=0.02)

    def forward(self, query: torch.Tensor, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return logits shaped [batch, query_digit, head, physical_slot].

        query is [B,Q,d]. slots is [B,Q,H,2,a] from the known correct block.
        """
        if query.ndim != 3 or slots.ndim != 5:
            raise ValueError("invalid probe feature rank")
        if slots.shape[2:] != (self.heads, 2, self.head_width):
            raise ValueError("invalid slot feature shape")
        batch, count, _ = query.shape
        q_only = self.q_only(query).view(batch, count, self.heads, 2)
        z_flat = slots.reshape(batch, count, self.heads, 2 * self.head_width)
        z_only = torch.einsum("bqhw,how->bqho", z_flat, self.z_only_weight)
        z_only = z_only + self.z_only_bias[None, None]
        q_expanded = query[:, :, None, :].expand(-1, -1, self.heads, -1)
        qz = torch.cat((q_expanded, z_flat), dim=-1)
        qz_linear = torch.einsum("bqhw,how->bqho", qz, self.qz_weight)
        qz_linear = qz_linear + self.qz_bias[None, None]
        gate_q = self.bilinear_query(query).view(
            batch, count, self.heads, self.gate_dim)
        gate_k = torch.einsum("bqhsa,hsac->bqhsc", slots, self.bilinear_slot)
        bilinear = torch.einsum("bqhc,bqhsc->bqhs", gate_q, gate_k)
        bilinear = bilinear / math.sqrt(self.gate_dim)
        return {"q_only": q_only, "z_only": z_only,
                "qz_linear": qz_linear, "bilinear": bilinear}
