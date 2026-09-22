from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - exercised on the GPU environment
    raise ImportError("TokenAlignedCED requires transformers with Qwen3 support (>=4.51)") from exc


@dataclass
class CEDOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    encoder_final: torch.Tensor | None = None


@dataclass
class CEDCache:
    local_keys: list[torch.Tensor | None]
    local_values: list[torch.Tensor | None]
    memory_keys: torch.Tensor | None = None
    memory_values: torch.Tensor | None = None
    length: int = 0


def _heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch, length, width = x.shape
    if width % head_dim:
        raise ValueError(f"projection width {width} is not divisible by head_dim {head_dim}")
    return x.view(batch, length, width // head_dim, head_dim).transpose(1, 2)


def _repeat_kv(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    if query_heads % x.shape[1]:
        raise ValueError("query heads must be a multiple of KV heads")
    return x.repeat_interleave(query_heads // x.shape[1], dim=1)


def _attention_scale(attn: nn.Module) -> float:
    return float(getattr(attn, "scaling", attn.head_dim**-0.5))


def _layer_mlp(layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    return residual + layer.mlp(hidden)


class CrossDecoderLayer(nn.Module):
    """Qwen upper block with global cross-attention in place of self-attention."""

    def __init__(
        self,
        source_layer: nn.Module,
        initial_cross_scale: float = 0.01,
    ):
        super().__init__()
        source_attn = source_layer.self_attn
        self.input_layernorm = source_layer.input_layernorm
        self.post_attention_layernorm = source_layer.post_attention_layernorm
        self.q_proj = source_attn.q_proj
        self.q_norm = source_attn.q_norm
        self.o_proj = source_attn.o_proj
        self.mlp = source_layer.mlp
        self.head_dim = int(source_attn.head_dim)
        self.scaling = _attention_scale(source_attn)
        if not 0.0 < initial_cross_scale < 1.0:
            raise ValueError("initial_cross_scale must lie strictly between zero and one")
        self.cross_gate_logit = nn.Parameter(
            torch.tensor(math.log(initial_cross_scale / (1.0 - initial_cross_scale)))
        )

    def forward(
        self,
        hidden: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        disable_global_memory: bool = False,
    ) -> torch.Tensor:
        residual = hidden
        normed = self.input_layernorm(hidden)
        q = _heads(self.q_proj(normed), self.head_dim)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        if disable_global_memory:
            attended = torch.zeros_like(q)
        else:
            k = _repeat_kv(memory_k, q.shape[1])
            v = _repeat_kv(memory_v, q.shape[1])
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=0.0, scale=self.scaling
            )
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + torch.sigmoid(self.cross_gate_logit) * self.o_proj(flat)
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)

    def step(
        self,
        hidden: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        disable_global_memory: bool = False,
    ) -> torch.Tensor:
        residual = hidden
        normed = self.input_layernorm(hidden)
        q = _heads(self.q_proj(normed), self.head_dim)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        if disable_global_memory:
            attended = torch.zeros_like(q)
        else:
            k = _repeat_kv(memory_k, q.shape[1])
            v = _repeat_kv(memory_v, q.shape[1])
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, dropout_p=0.0, scale=self.scaling
            )
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
        hidden = residual + torch.sigmoid(self.cross_gate_logit) * self.o_proj(flat)
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        return residual + self.mlp(hidden)


class TokenAlignedCED(nn.Module):
    """Causal token-aligned shared-memory CED initialized from Qwen3."""

    def __init__(
        self,
        base_model: nn.Module,
        encoder_layers: int = 14,
        encoder_window: int = 64,
        local_query_chunk: int = 256,
        initial_cross_scale: float = 0.01,
    ):
        super().__init__()
        backbone = base_model.model
        total_layers = len(backbone.layers)
        if not 0 < encoder_layers < total_layers:
            raise ValueError("encoder_layers must split the Qwen stack")
        if encoder_window < 1 or local_query_chunk < 1:
            raise ValueError("window and chunk must be positive")

        self.config = base_model.config
        self.encoder_window = int(encoder_window)
        self.local_query_chunk = int(local_query_chunk)
        self.embed_tokens = backbone.embed_tokens
        self.rotary_emb = backbone.rotary_emb
        self.encoder = nn.ModuleList(list(backbone.layers[:encoder_layers]))
        upper = list(backbone.layers[encoder_layers:])
        source = upper[0]
        self.memory_norm = copy.deepcopy(source.input_layernorm)
        self.memory_k_proj = source.self_attn.k_proj
        self.memory_k_norm = source.self_attn.k_norm
        self.memory_v_proj = source.self_attn.v_proj
        self.memory_head_dim = int(source.self_attn.head_dim)
        self.cross_decoder = nn.ModuleList(
            [
                CrossDecoderLayer(layer, initial_cross_scale)
                for layer in upper
            ]
        )
        self.final_norm = backbone.norm
        self.lm_head = base_model.lm_head
        self.gradient_checkpointing = False

    @classmethod
    def from_qwen(cls, base_model: nn.Module, **kwargs: Any) -> "TokenAlignedCED":
        return cls(base_model, **kwargs)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def new_cache(self) -> CEDCache:
        return CEDCache(
            local_keys=[None] * len(self.encoder),
            local_values=[None] * len(self.encoder),
        )

    def _position_embeddings(
        self, hidden: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(hidden, position_ids)

    def _local_attention(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_kv: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = layer.self_attn
        residual = hidden
        normed = layer.input_layernorm(hidden)
        q = _heads(attn.q_proj(normed), attn.head_dim)
        k = _heads(attn.k_proj(normed), attn.head_dim)
        v = _heads(attn.v_proj(normed), attn.head_dim)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache_k, cache_v = k, v
        k = _repeat_kv(k, q.shape[1])
        v = _repeat_kv(v, q.shape[1])

        outputs = []
        length = hidden.shape[1]
        for q_start in range(0, length, self.local_query_chunk):
            q_end = min(length, q_start + self.local_query_chunk)
            k_start = max(0, q_start - self.encoder_window + 1)
            k_end = q_end
            query_pos = torch.arange(q_start, q_end, device=hidden.device)
            key_pos = torch.arange(k_start, k_end, device=hidden.device)
            allowed = (
                (key_pos[None, :] <= query_pos[:, None])
                & (key_pos[None, :] >= query_pos[:, None] - self.encoder_window + 1)
            )
            outputs.append(
                F.scaled_dot_product_attention(
                    q[:, :, q_start:q_end],
                    k[:, :, k_start:k_end],
                    v[:, :, k_start:k_end],
                    attn_mask=allowed[None, None],
                    dropout_p=0.0,
                    scale=_attention_scale(attn),
                )
            )
        attended = torch.cat(outputs, dim=2)
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], length, -1)
        hidden = residual + attn.o_proj(flat)
        hidden = _layer_mlp(layer, hidden)
        if return_kv:
            return hidden, cache_k, cache_v
        return hidden

    def _memory(
        self,
        encoder_final: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.memory_norm(encoder_final)
        k = _heads(self.memory_k_proj(normed), self.memory_head_dim)
        v = _heads(self.memory_v_proj(normed), self.memory_head_dim)
        k = self.memory_k_norm(k)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        return k, v

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        disable_global_memory: bool = False,
        return_encoder_final: bool = False,
    ) -> CEDOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._position_embeddings(hidden, positions)

        for layer in self.encoder:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda x, module=layer: self._local_attention(module, x, cos, sin),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = self._local_attention(layer, hidden, cos, sin)
        encoder_final = hidden
        memory_k, memory_v = self._memory(encoder_final, cos, sin)

        for layer in self.cross_decoder:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(
                    lambda x, module=layer: module(
                        x, memory_k, memory_v, cos, sin, disable_global_memory
                    ),
                    hidden,
                    use_reentrant=False,
                )
            else:
                hidden = layer(
                    hidden, memory_k, memory_v, cos, sin, disable_global_memory
                )
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
        return CEDOutput(
            logits=logits,
            loss=loss,
            encoder_final=encoder_final if return_encoder_final else None,
        )

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        disable_global_memory: bool = False,
    ) -> tuple[torch.Tensor, CEDCache]:
        """Build inference state in parallel and return the final-position logits."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("prefill expects a non-empty [batch, sequence] tensor")
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._position_embeddings(hidden, positions)
        cache = self.new_cache()
        for index, layer in enumerate(self.encoder):
            hidden, k, v = self._local_attention(
                layer, hidden, cos, sin, return_kv=True
            )
            cache.local_keys[index] = k[:, :, -self.encoder_window :]
            cache.local_values[index] = v[:, :, -self.encoder_window :]
        memory_k, memory_v = self._memory(hidden, cos, sin)
        cache.memory_keys = memory_k
        cache.memory_values = memory_v
        cache.length = input_ids.shape[1]

        current = hidden[:, -1:]
        cos_last, sin_last = cos[:, -1:], sin[:, -1:]
        for layer in self.cross_decoder:
            current = layer.step(
                current,
                memory_k,
                memory_v,
                cos_last,
                sin_last,
                disable_global_memory,
            )
        logits = self.lm_head(self.final_norm(current)).float()
        return logits, cache

    @torch.inference_mode()
    def step(
        self,
        input_ids: torch.Tensor,
        cache: CEDCache,
        disable_global_memory: bool = False,
    ) -> tuple[torch.Tensor, CEDCache]:
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("step expects exactly one token per batch")
        if len(cache.local_keys) != len(self.encoder):
            raise ValueError("cache does not match encoder depth")
        hidden = self.embed_tokens(input_ids)
        position_ids = torch.full(
            (input_ids.shape[0], 1), cache.length, device=input_ids.device, dtype=torch.long
        )
        cos, sin = self._position_embeddings(hidden, position_ids)

        for index, layer in enumerate(self.encoder):
            attn = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q = _heads(attn.q_proj(normed), attn.head_dim)
            k_new = _heads(attn.k_proj(normed), attn.head_dim)
            v_new = _heads(attn.v_proj(normed), attn.head_dim)
            q = attn.q_norm(q)
            k_new = attn.k_norm(k_new)
            q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)
            old_k, old_v = cache.local_keys[index], cache.local_values[index]
            k = k_new if old_k is None else torch.cat([old_k, k_new], dim=2)
            v = v_new if old_v is None else torch.cat([old_v, v_new], dim=2)
            k = k[:, :, -self.encoder_window :]
            v = v[:, :, -self.encoder_window :]
            cache.local_keys[index], cache.local_values[index] = k, v
            attended = F.scaled_dot_product_attention(
                q,
                _repeat_kv(k, q.shape[1]),
                _repeat_kv(v, q.shape[1]),
                is_causal=False,
                dropout_p=0.0,
                scale=_attention_scale(attn),
            )
            flat = attended.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
            hidden = residual + attn.o_proj(flat)
            hidden = _layer_mlp(layer, hidden)

        encoder_final = hidden
        memory_k_new, memory_v_new = self._memory(encoder_final, cos, sin)
        cache.memory_keys = (
            memory_k_new
            if cache.memory_keys is None
            else torch.cat([cache.memory_keys, memory_k_new], dim=2)
        )
        cache.memory_values = (
            memory_v_new
            if cache.memory_values is None
            else torch.cat([cache.memory_values, memory_v_new], dim=2)
        )
        for layer in self.cross_decoder:
            hidden = layer.step(
                hidden,
                cache.memory_keys,
                cache.memory_values,
                cos,
                sin,
                disable_global_memory,
            )
        cache.length += 1
        logits = self.lm_head(self.final_norm(hidden)).float()
        return logits, cache

    def cache_accounting(self, cache: CEDCache) -> dict[str, int]:
        local = sum(
            0 if k is None else k.numel() for k in cache.local_keys
        ) + sum(0 if v is None else v.numel() for v in cache.local_values)
        global_memory = (
            (0 if cache.memory_keys is None else cache.memory_keys.numel())
            + (0 if cache.memory_values is None else cache.memory_values.numel())
        )
        return {
            "local_kv_elements": local,
            "global_kv_elements": global_memory,
            "retained_encoder_state_elements": 0,
            "full_history_layer_kv_copies": 0,
            "sequence_length": cache.length,
        }
