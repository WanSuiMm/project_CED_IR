"""Quality-first Qwen3 content/RoPE split with a real latent runtime cache.

This preserves Qwen3's q_norm/k_norm and all K/V widths. It changes only
which key coordinates receive RoPE. It is a high-fidelity *split* baseline,
not a low-rank TransMLA conversion and not yet the pair-content B model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .reference import PairCompiler
from .headwise_compiler import HeadwiseKVPairCompiler


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class SplitQwen3Attention(nn.Module):
    def __init__(self, source: nn.Module, rope_dim_per_head: int):
        super().__init__()
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.head_dim = source.head_dim
        self.num_query_heads = source.config.num_attention_heads
        self.num_kv_heads = source.config.num_key_value_heads
        if self.head_dim % 2 or rope_dim_per_head < 2 or (
            rope_dim_per_head > self.head_dim or rope_dim_per_head % 2
        ):
            raise ValueError("RoPE width must be positive, even, and <= head_dim")
        self.rope_dim_per_head = rope_dim_per_head
        half = self.head_dim // 2
        kept = rope_dim_per_head // 2
        self.register_buffer("rope_indices", torch.cat((
            torch.arange(kept), torch.arange(half, half + kept))),
            persistent=False)
        self.register_buffer("content_key_indices", torch.cat((
            torch.arange(kept, half), torch.arange(half + kept, self.head_dim))),
            persistent=False)
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.o_proj = source.o_proj
        self.q_norm = source.q_norm
        self.k_norm = source.k_norm
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = source.attention_dropout

    @property
    def content_dim(self) -> int:
        return self.num_kv_heads * (2 * self.head_dim - self.rope_dim_per_head)

    @property
    def rope_dim(self) -> int:
        return self.num_kv_heads * self.rope_dim_per_head

    def _split(self, hidden_states: Tensor,
               position_embeddings: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, length, _ = hidden_states.shape
        q = self.q_norm(self.q_proj(hidden_states).view(
            batch, length, self.num_query_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(
            batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q_c = q.index_select(-1, self.content_key_indices)
        k_c = k.index_select(-1, self.content_key_indices)
        q_r = q.index_select(-1, self.rope_indices)
        k_r = k.index_select(-1, self.rope_indices)
        cos, sin = position_embeddings
        cos = cos.index_select(-1, self.rope_indices).unsqueeze(1)
        sin = sin.index_select(-1, self.rope_indices).unsqueeze(1)
        q_r = q_r * cos + _rotate_half(q_r) * sin
        k_r = k_r * cos + _rotate_half(k_r) * sin
        # [batch, token, KV heads * width]; these are the only persisted states.
        content = torch.cat((k_c, v), dim=-1).transpose(1, 2).reshape(
            batch, length, self.content_dim)
        rope = k_r.transpose(1, 2).reshape(batch, length, self.rope_dim)
        return q_c, q_r, content, rope

    def forward(self, hidden_states: Tensor,
                position_embeddings: tuple[Tensor, Tensor],
                attention_mask: Tensor | None,
                past_key_values=None, **kwargs) -> tuple[Tensor, None]:
        batch, length, _ = hidden_states.shape
        q_c, q_r, content, rope = self._split(hidden_states, position_embeddings)
        if past_key_values is not None:
            content, rope = past_key_values.update(
                content.unsqueeze(1), rope.unsqueeze(1), self.layer_idx)
            content, rope = content.squeeze(1), rope.squeeze(1)
        all_length = content.shape[1]
        k_width = self.head_dim - self.rope_dim_per_head
        content = content.view(batch, all_length, self.num_kv_heads,
                               k_width + self.head_dim).transpose(1, 2)
        k_c, v = content.split((k_width, self.head_dim), dim=-1)
        k_r = rope.view(batch, all_length, self.num_kv_heads,
                        self.rope_dim_per_head).transpose(1, 2)
        repeat = self.num_query_heads // self.num_kv_heads
        k = torch.cat((k_c, k_r), dim=-1).repeat_interleave(repeat, dim=1)
        q = torch.cat((q_c, q_r), dim=-1)
        v = v.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=attention_mask is None and all_length == length,
            scale=self.scaling)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output), None


def convert_qwen3_to_split(model: nn.Module,
                           rope_dim_per_head: int) -> nn.Module:
    """In-place, parameter-preserving attention replacement.

    Does not save/reload a checkpoint and does not perform low-rank conversion.
    """
    if getattr(model.config, "model_type", None) != "qwen3":
        raise ValueError("expected an unmodified Qwen3 model")
    for layer in model.model.layers:
        source = layer.self_attn
        layer.self_attn = SplitQwen3Attention(source, rope_dim_per_head).to(
            device=source.q_proj.weight.device)
    first = model.model.layers[0].self_attn
    model.config.kv_lora_rank = first.content_dim
    model.config.qk_rope_head_dim = first.rope_dim
    return model


class PairQwen3Attention(SplitQwen3Attention):
    """Pair-compiled content with separate per-token RoPE keys (training path)."""

    def __init__(self, source: SplitQwen3Attention, compiler_kind: str = "dense"):
        super().__init__(source, source.rope_dim_per_head)
        if compiler_kind == "dense":
            compiler = PairCompiler(self.content_dim)
        elif compiler_kind == "headwise_kv":
            compiler = HeadwiseKVPairCompiler(
                self.num_kv_heads, self.head_dim - self.rope_dim_per_head,
                self.head_dim)
        else:
            raise ValueError(f"unknown compiler kind: {compiler_kind}")
        self.compiler = compiler.to(device=source.q_proj.weight.device)

    def forward(self, hidden_states: Tensor,
                position_embeddings: tuple[Tensor, Tensor],
                attention_mask: Tensor | None,
                past_key_values=None, **kwargs) -> tuple[Tensor, None]:
        if past_key_values is not None:
            from .qwen3_cache import PairDynamicCache
            if not isinstance(past_key_values, PairDynamicCache):
                raise TypeError("pair attention requires PairDynamicCache")
            return self._cached_forward(hidden_states, position_embeddings,
                                        past_key_values), None
        batch, length, _ = hidden_states.shape
        q_c, q_r, content, rope = self._split(hidden_states, position_embeddings)
        pair_count = length // 2
        even_count = (length + 1) // 2
        pair_content = self.compiler(
            content[:, 0:2 * pair_count:2], content[:, 1:2 * pair_count:2])
        raw_content = content[:, ::2]
        key_width = self.head_dim - self.rope_dim_per_head
        repeat = self.num_query_heads // self.num_kv_heads

        def unpack(latents: Tensor) -> tuple[Tensor, Tensor]:
            shaped = latents.view(batch, latents.shape[1], self.num_kv_heads,
                                  key_width + self.head_dim).transpose(1, 2)
            key, value = shaped.split((key_width, self.head_dim), dim=-1)
            return key.repeat_interleave(repeat, 1), value.repeat_interleave(repeat, 1)

        pair_key, pair_value = unpack(pair_content)
        raw_key, raw_value = unpack(raw_content)
        rope_key = rope.view(batch, length, self.num_kv_heads,
                             self.rope_dim_per_head).transpose(1, 2)
        rope_key = rope_key.repeat_interleave(repeat, 1)
        positions = torch.arange(length, device=hidden_states.device)

        if pair_count:
            pair_c_scores = torch.einsum("bhtd,bhpd->bhtp", q_c, pair_key) * self.scaling
            even_r_scores = torch.einsum("bhtr,bhpr->bhtp", q_r,
                                         rope_key[:, :, 0:2 * pair_count:2]) * self.scaling
            odd_r_scores = torch.einsum("bhtr,bhpr->bhtp", q_r,
                                        rope_key[:, :, 1:2 * pair_count:2]) * self.scaling
            pair_scores = pair_c_scores + torch.logaddexp(even_r_scores, odd_r_scores)
            pair_end = 2 * torch.arange(pair_count, device=hidden_states.device) + 1
            pair_scores = pair_scores.masked_fill(
                pair_end[None, :] > positions[:, None], -torch.inf)
        else:
            pair_scores = q_c.new_empty(batch, self.num_query_heads, length, 0)

        raw_scores = (torch.einsum("bhtd,bhed->bhte", q_c, raw_key)
                      + torch.einsum("bhtr,bher->bhte", q_r, rope_key[:, :, ::2]))
        raw_scores = raw_scores * self.scaling
        raw_pos = 2 * torch.arange(even_count, device=hidden_states.device)
        raw_scores = raw_scores.masked_fill(
            raw_pos[None, :] != positions[:, None], -torch.inf)
        weights = torch.cat((pair_scores, raw_scores), dim=-1).softmax(-1)
        values = torch.cat((pair_value, raw_value), dim=2)
        output = torch.einsum("bhtn,bhnd->bhtd", weights, values)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output), None

    def _cached_forward(self, hidden_states: Tensor,
                        position_embeddings: tuple[Tensor, Tensor], cache) -> Tensor:
        batch, length, _ = hidden_states.shape
        q_c, q_r, content, rope = self._split(hidden_states, position_embeddings)
        key_width = self.head_dim - self.rope_dim_per_head
        repeat = self.num_query_heads // self.num_kv_heads
        outputs = []
        for offset in range(length):
            state = cache.append_pair(self.layer_idx, content[:, offset],
                                      rope[:, offset], self.compiler)
            qc, qr = q_c[:, :, offset], q_r[:, :, offset]
            scores, values = [], []
            if state.pair_content is not None:
                count = state.pair_content.shape[1]
                shaped = state.pair_content.view(
                    batch, count, self.num_kv_heads,
                    key_width + self.head_dim).transpose(1, 2)
                pair_key, pair_value = shaped.split((key_width, self.head_dim), -1)
                pair_key = pair_key.repeat_interleave(repeat, 1)
                pair_value = pair_value.repeat_interleave(repeat, 1)
                pair_rope = state.pair_rope.view(
                    batch, 2 * count, self.num_kv_heads,
                    self.rope_dim_per_head).transpose(1, 2)
                pair_rope = pair_rope.repeat_interleave(repeat, 1)
                content_scores = torch.einsum("bhd,bhpd->bhp", qc, pair_key) * self.scaling
                rope_scores = torch.einsum("bhr,bhnr->bhn", qr, pair_rope) * self.scaling
                scores.append(content_scores + torch.logaddexp(
                    rope_scores[..., ::2], rope_scores[..., 1::2]))
                values.append(pair_value)
            if state.pending_content is not None:
                pending = state.pending_content.view(
                    batch, self.num_kv_heads, key_width + self.head_dim)
                raw_key, raw_value = pending.split((key_width, self.head_dim), -1)
                raw_key = raw_key.repeat_interleave(repeat, 1)
                raw_value = raw_value.repeat_interleave(repeat, 1)
                raw_rope = state.pending_rope.view(
                    batch, self.num_kv_heads, self.rope_dim_per_head)
                raw_rope = raw_rope.repeat_interleave(repeat, 1)
                raw_score = ((qc * raw_key).sum(-1)
                             + (qr * raw_rope).sum(-1)) * self.scaling
                scores.append(raw_score.unsqueeze(-1))
                values.append(raw_value.unsqueeze(2))
            weights = torch.cat(scores, dim=-1).softmax(-1)
            value = torch.cat(values, dim=2)
            outputs.append(torch.einsum("bhn,bhnd->bhd", weights, value))
        output = torch.stack(outputs, dim=2).transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


def convert_split_to_pair(model: nn.Module, compiler_kind: str = "dense") -> nn.Module:
    if not all(isinstance(layer.self_attn, SplitQwen3Attention)
               for layer in model.model.layers):
        raise ValueError("pair conversion requires a split-Qwen3 A model")
    for layer in model.model.layers:
        source = layer.self_attn
        layer.self_attn = PairQwen3Attention(source, compiler_kind=compiler_kind).to(
            device=source.q_proj.weight.device)
    return model
