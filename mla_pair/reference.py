"""Correctness-first MLA operators; inputs are already projected/rotated.

The content latent and the RoPE-bearing key remain distinct. This module does
not convert Qwen to MLA, implement a fused kernel, or claim decode speedups.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class PairCompiler(nn.Module):
    """Compile two token content latents into one latent of the same width."""

    def __init__(self, content_dim: int):
        super().__init__()
        if content_dim < 1:
            raise ValueError("content_dim must be positive")
        self.linear = nn.Linear(2 * content_dim, content_dim)
        # A neutral, non-exact warm start; the compiler must be learned.
        with torch.no_grad():
            self.linear.weight.zero_()
            eye = torch.eye(content_dim)
            self.linear.weight[:, :content_dim].copy_(eye / 2)
            self.linear.weight[:, content_dim:].copy_(eye / 2)
            self.linear.bias.zero_()

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        if first.shape != second.shape or first.shape[-1] * 2 != self.linear.in_features:
            raise ValueError("pair latent shapes do not match compiler width")
        return self.linear(torch.cat((first, second), dim=-1))


def _check(q_content: Tensor, q_rope: Tensor, content: Tensor, rope: Tensor,
           value_weight: Tensor) -> None:
    if q_content.ndim != 4 or q_rope.ndim != 4:
        raise ValueError("queries must be [batch, heads, time, width]")
    if content.ndim != 3 or rope.ndim != 3:
        raise ValueError("cache inputs must be [batch, time, width]")
    batch, heads, time, content_dim = q_content.shape
    if time < 1 or content.shape != (batch, time, content_dim):
        raise ValueError("content must match query batch, time, and content width")
    if q_rope.shape[:3] != (batch, heads, time):
        raise ValueError("RoPE query shape mismatch")
    if rope.shape != (batch, time, q_rope.shape[-1]):
        raise ValueError("RoPE key shape mismatch")
    if value_weight.ndim != 3 or value_weight.shape[:2] != (heads, content_dim):
        raise ValueError("value_weight must be [heads, content_dim, value_dim]")


def _values(content: Tensor, value_weight: Tensor) -> Tensor:
    return torch.einsum("bnd,hdv->bhnv", content, value_weight)


def token_attention(q_content: Tensor, q_rope: Tensor, content: Tensor,
                    rope: Tensor, value_weight: Tensor, scale: float) -> Tensor:
    """Token-level MLA reference, with causal self-attention."""
    _check(q_content, q_rope, content, rope, value_weight)
    if scale <= 0:
        raise ValueError("scale must be positive")
    scores = (torch.einsum("bhtd,bnd->bhtn", q_content, content)
              + torch.einsum("bhtr,bnr->bhtn", q_rope, rope)) * scale
    time = content.shape[1]
    positions = torch.arange(time, device=content.device)
    scores = scores.masked_fill(positions[None, :] > positions[:, None], -torch.inf)
    return torch.einsum("bhtn,bhnv->bhtv", scores.softmax(-1),
                        _values(content, value_weight))


def pair_attention(q_content: Tensor, q_rope: Tensor, content: Tensor,
                   rope: Tensor, value_weight: Tensor, compiler: PairCompiler,
                   scale: float) -> Tensor:
    """Causal full-sequence reference for pair-content/token-RoPE MLA.

    A completed pair contributes one content dot product, two RoPE dot
    products, and one value. At an even query position its own token remains
    an unpaired raw latent. No query can see a pair before its second token.
    """
    _check(q_content, q_rope, content, rope, value_weight)
    if scale <= 0:
        raise ValueError("scale must be positive")
    batch, heads, time, _ = q_content.shape
    pairs = time // 2
    even_count = (time + 1) // 2
    pair_c = compiler(content[:, 0:2 * pairs:2], content[:, 1:2 * pairs:2])
    even_c = content[:, ::2]

    if pairs:
        content_scores = torch.einsum("bhtd,bpd->bhtp", q_content, pair_c) * scale
        rope_even = torch.einsum("bhtr,bpr->bhtp", q_rope,
                                 rope[:, 0:2 * pairs:2]) * scale
        rope_odd = torch.einsum("bhtr,bpr->bhtp", q_rope,
                                rope[:, 1:2 * pairs:2]) * scale
        pair_scores = content_scores + torch.logaddexp(rope_even, rope_odd)
        query_pos = torch.arange(time, device=content.device)
        pair_end = 2 * torch.arange(pairs, device=content.device) + 1
        pair_scores = pair_scores.masked_fill(
            pair_end[None, :] > query_pos[:, None], -torch.inf)
    else:
        pair_scores = q_content.new_empty(batch, heads, time, 0)

    raw_scores = (torch.einsum("bhtd,bed->bhte", q_content, even_c)
                  + torch.einsum("bhtr,ber->bhte", q_rope, rope[:, ::2])) * scale
    query_pos = torch.arange(time, device=content.device)
    raw_pos = 2 * torch.arange(even_count, device=content.device)
    raw_scores = raw_scores.masked_fill(
        raw_pos[None, :] != query_pos[:, None], -torch.inf)
    weights = torch.cat((pair_scores, raw_scores), dim=-1).softmax(-1)
    values = torch.cat((_values(pair_c, value_weight),
                        _values(even_c, value_weight)), dim=2)
    return torch.einsum("bhtn,bhnv->bhtv", weights, values)


@dataclass(frozen=True)
class PairCache:
    pair_content: Tensor | None = None  # [batch, completed_pairs, content_dim]
    pair_rope: Tensor | None = None     # [batch, 2 * completed_pairs, rope_dim]
    pending_content: Tensor | None = None  # [batch, content_dim]
    pending_rope: Tensor | None = None     # [batch, rope_dim]

    @property
    def length(self) -> int:
        completed = 0 if self.pair_content is None else self.pair_content.shape[1]
        return 2 * completed + int(self.pending_content is not None)


def append_token(cache: PairCache, content: Tensor, rope: Tensor,
                 compiler: PairCompiler) -> PairCache:
    """Append a token; compact only after both pair members have arrived."""
    if content.ndim != 2 or rope.ndim != 2 or content.shape[0] != rope.shape[0]:
        raise ValueError("new token inputs must be [batch, width]")
    if (cache.pending_content is None) != (cache.pending_rope is None):
        raise ValueError("incomplete pending cache")
    if cache.pending_content is None:
        if cache.pair_content is not None and (
            cache.pair_content.shape[0] != content.shape[0]
            or cache.pair_content.shape[-1] != content.shape[-1]
            or cache.pair_rope is None
            or cache.pair_rope.shape[-1] != rope.shape[-1]
        ):
            raise ValueError("token/cache width mismatch")
        return PairCache(cache.pair_content, cache.pair_rope, content, rope)
    if cache.pending_content.shape != content.shape or cache.pending_rope.shape != rope.shape:
        raise ValueError("pending/new token shape mismatch")
    compiled = compiler(cache.pending_content, content).unsqueeze(1)
    pair_rope = torch.stack((cache.pending_rope, rope), dim=1)
    return PairCache(
        compiled if cache.pair_content is None else torch.cat((cache.pair_content, compiled), 1),
        pair_rope if cache.pair_rope is None else torch.cat((cache.pair_rope, pair_rope), 1),
    )


def cache_scalars(cache: PairCache) -> int:
    """Persistent scalars per batch element, excluding model parameters."""
    tensors = (cache.pair_content, cache.pair_rope,
               cache.pending_content, cache.pending_rope)
    return sum(t.numel() // t.shape[0] for t in tensors if t is not None)


def pair_attention_step(q_content: Tensor, q_rope: Tensor, cache: PairCache,
                        value_weight: Tensor, scale: float) -> Tensor:
    """One-query reference over completed pairs and the optional pending token."""
    if q_content.ndim != 3 or q_rope.ndim != 3 or q_content.shape[:2] != q_rope.shape[:2]:
        raise ValueError("step queries must be [batch, heads, width]")
    if cache.length == 0 or scale <= 0:
        raise ValueError("cache must be nonempty and scale positive")
    if value_weight.shape[:2] != (q_content.shape[1], q_content.shape[-1]):
        raise ValueError("value projection shape mismatch")
    scores, values = [], []
    if cache.pair_content is not None:
        count = cache.pair_content.shape[1]
        pair_c = torch.einsum("bhd,bpd->bhp", q_content, cache.pair_content) * scale
        rope_scores = torch.einsum("bhr,bnr->bhn", q_rope, cache.pair_rope) * scale
        pair_scores = pair_c + torch.logaddexp(
            rope_scores[..., 0:2 * count:2], rope_scores[..., 1:2 * count:2])
        scores.append(pair_scores)
        values.append(_values(cache.pair_content, value_weight))
    if cache.pending_content is not None:
        raw = ((q_content * cache.pending_content[:, None]).sum(-1)
               + (q_rope * cache.pending_rope[:, None]).sum(-1)) * scale
        scores.append(raw.unsqueeze(-1))
        values.append(_values(cache.pending_content[:, None], value_weight))
    probs = torch.cat(scores, -1).softmax(-1)
    return torch.einsum("bhn,bhnv->bhv", probs, torch.cat(values, 2))
