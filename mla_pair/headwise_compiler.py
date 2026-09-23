"""Per-layer, per-KV-head pair compiler with separate key and value maps."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class HeadwiseKVPairCompiler(nn.Module):
    """Compile two [K_content; V] states without mixing heads or K with V.

    Key initialization assumes approximately independent adjacent keys and
    preserves their RMS with 1/sqrt(2) mixing. Value initialization averages.
    """

    def __init__(self, num_kv_heads: int, key_width: int, value_width: int):
        super().__init__()
        if min(num_kv_heads, key_width, value_width) < 1:
            raise ValueError("head count and K/V widths must be positive")
        self.num_kv_heads = num_kv_heads
        self.key_width = key_width
        self.value_width = value_width
        self.content_dim = num_kv_heads * (key_width + value_width)
        self.key_weight = nn.Parameter(torch.zeros(
            num_kv_heads, key_width, 2 * key_width))
        self.key_bias = nn.Parameter(torch.zeros(num_kv_heads, key_width))
        self.value_weight = nn.Parameter(torch.zeros(
            num_kv_heads, value_width, 2 * value_width))
        self.value_bias = nn.Parameter(torch.zeros(num_kv_heads, value_width))
        with torch.no_grad():
            key_eye = torch.eye(key_width) / math.sqrt(2)
            value_eye = torch.eye(value_width) / 2
            self.key_weight[:, :, :key_width].copy_(key_eye)
            self.key_weight[:, :, key_width:].copy_(key_eye)
            self.value_weight[:, :, :value_width].copy_(value_eye)
            self.value_weight[:, :, value_width:].copy_(value_eye)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.shape[-1] != self.content_dim:
            raise ValueError("pair latent shapes do not match headwise compiler")
        prefix = first.shape[:-1]
        width = self.key_width + self.value_width
        first = first.reshape(*prefix, self.num_kv_heads, width)
        second = second.reshape(*prefix, self.num_kv_heads, width)
        fk, fv = first.split((self.key_width, self.value_width), -1)
        sk, sv = second.split((self.key_width, self.value_width), -1)
        key = torch.cat((fk, sk), -1).to(self.key_weight.dtype)
        value = torch.cat((fv, sv), -1).to(self.value_weight.dtype)
        compiled_key = torch.einsum("...hi,hji->...hj", key, self.key_weight) + self.key_bias
        compiled_value = (torch.einsum("...hi,hji->...hj", value, self.value_weight)
                          + self.value_bias)
        return torch.cat((compiled_key, compiled_value), -1).reshape(
            *prefix, self.content_dim).to(first.dtype)
