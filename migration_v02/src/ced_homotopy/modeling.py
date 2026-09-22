from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - exercised on the GPU environment
    raise ImportError("HomotopyCED requires transformers with Qwen3 support") from exc


@dataclass
class HomotopyOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    bridge_loss: torch.Tensor | None = None
    bridge_by_layer: list[torch.Tensor] | None = None


@dataclass
class EndpointCache:
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


def _scale(attn: nn.Module) -> float:
    return float(getattr(attn, "scaling", attn.head_dim**-0.5))


def _mix(a: torch.Tensor, b: torch.Tensor, weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return a
    if weight >= 1.0:
        return b
    return torch.lerp(a, b, weight)


class HomotopyCED(nn.Module):
    """Qwen-to-CED continuous migration with an exact native endpoint at (0, 0)."""

    def __init__(
        self,
        base_model: nn.Module,
        encoder_layers: int = 14,
        encoder_window: int = 64,
        local_query_chunk: int = 256,
    ):
        super().__init__()
        total_layers = len(base_model.model.layers)
        if not 0 < encoder_layers < total_layers:
            raise ValueError("encoder_layers must split the Qwen stack")
        if encoder_window < 1 or local_query_chunk < 1:
            raise ValueError("window and chunk must be positive")
        self.base_model = base_model
        self.config = base_model.config
        self.encoder_layers = int(encoder_layers)
        self.encoder_window = int(encoder_window)
        self.local_query_chunk = int(local_query_chunk)

        source = base_model.model.layers[encoder_layers]
        self.memory_norm = copy.deepcopy(source.input_layernorm)
        self.memory_k_proj = copy.deepcopy(source.self_attn.k_proj)
        self.memory_k_norm = copy.deepcopy(source.self_attn.k_norm)
        self.memory_v_proj = copy.deepcopy(source.self_attn.v_proj)
        self.memory_head_dim = int(source.self_attn.head_dim)
        self.register_buffer("alpha", torch.tensor(0.0), persistent=True)
        self.register_buffer("beta", torch.tensor(0.0), persistent=True)

    @classmethod
    def from_qwen(cls, base_model: nn.Module, **kwargs: Any) -> "HomotopyCED":
        return cls(base_model, **kwargs)

    @property
    def lower(self) -> list[nn.Module]:
        return list(self.base_model.model.layers[: self.encoder_layers])

    @property
    def upper(self) -> list[nn.Module]:
        return list(self.base_model.model.layers[self.encoder_layers :])

    def set_mix(self, alpha: float, beta: float) -> None:
        if not 0.0 <= alpha <= 1.0 or not 0.0 <= beta <= 1.0:
            raise ValueError("alpha and beta must lie in [0, 1]")
        self.alpha.fill_(float(alpha))
        self.beta.fill_(float(beta))

    def bridge_parameters(self):
        for module in (
            self.memory_norm,
            self.memory_k_proj,
            self.memory_k_norm,
            self.memory_v_proj,
        ):
            yield from module.parameters()

    def set_backbone_trainable(self, enabled: bool) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(enabled)
        for parameter in self.bridge_parameters():
            parameter.requires_grad_(True)

    def _positions(self, hidden: torch.Tensor, position_ids: torch.Tensor):
        return self.base_model.model.rotary_emb(hidden, position_ids)

    def _project_qkv(
        self, attn: nn.Module, normed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = _heads(attn.q_proj(normed), attn.head_dim)
        k = _heads(attn.k_proj(normed), attn.head_dim)
        v = _heads(attn.v_proj(normed), attn.head_dim)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def _local_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
    ) -> torch.Tensor:
        outputs = []
        length = q.shape[2]
        for q_start in range(0, length, self.local_query_chunk):
            q_end = min(length, q_start + self.local_query_chunk)
            k_start = max(0, q_start - self.encoder_window + 1)
            query_pos = torch.arange(q_start, q_end, device=q.device)
            key_pos = torch.arange(k_start, q_end, device=q.device)
            allowed = (
                (key_pos[None, :] <= query_pos[:, None])
                & (key_pos[None, :] >= query_pos[:, None] - self.encoder_window + 1)
            )
            outputs.append(
                F.scaled_dot_product_attention(
                    q[:, :, q_start:q_end],
                    _repeat_kv(k[:, :, k_start:q_end], q.shape[1]),
                    _repeat_kv(v[:, :, k_start:q_end], q.shape[1]),
                    attn_mask=allowed[None, None],
                    dropout_p=0.0,
                    scale=scale,
                )
            )
        return torch.cat(outputs, dim=2)

    def _lower_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        beta: float,
        return_kv: bool = False,
    ):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        q, k, v = self._project_qkv(layer.self_attn, normed, cos, sin)
        full = None
        local = None
        if beta < 1.0:
            full = F.scaled_dot_product_attention(
                q, _repeat_kv(k, q.shape[1]), _repeat_kv(v, q.shape[1]),
                is_causal=True, dropout_p=0.0, scale=_scale(layer.self_attn),
            )
        if beta > 0.0:
            local = self._local_attention(q, k, v, _scale(layer.self_attn))
        attended = local if full is None else full if local is None else _mix(full, local, beta)
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + layer.self_attn.o_proj(flat)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        return (hidden, k, v) if return_kv else hidden

    def _memory(self, encoder_final: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        normed = self.memory_norm(encoder_final)
        k = _heads(self.memory_k_proj(normed), self.memory_head_dim)
        v = _heads(self.memory_v_proj(normed), self.memory_head_dim)
        k = self.memory_k_norm(k)
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        return k, v

    def _cross_heads(
        self,
        layer: nn.Module,
        normed: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        attn = layer.self_attn
        q = _heads(attn.q_proj(normed), attn.head_dim)
        q = attn.q_norm(q)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return F.scaled_dot_product_attention(
            q,
            _repeat_kv(memory_k, q.shape[1]),
            _repeat_kv(memory_v, q.shape[1]),
            is_causal=causal,
            dropout_p=0.0,
            scale=_scale(attn),
        )

    def _upper_layer(
        self,
        layer: nn.Module,
        hidden: torch.Tensor,
        memory_k: torch.Tensor,
        memory_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        alpha: float,
        disable_global_memory: bool = False,
    ) -> torch.Tensor:
        residual = hidden
        normed = layer.input_layernorm(hidden)
        self_heads = None
        cross_heads = None
        if alpha < 1.0:
            q, k, v = self._project_qkv(layer.self_attn, normed, cos, sin)
            self_heads = F.scaled_dot_product_attention(
                q, _repeat_kv(k, q.shape[1]), _repeat_kv(v, q.shape[1]),
                is_causal=True, dropout_p=0.0, scale=_scale(layer.self_attn),
            )
        if alpha > 0.0:
            cross_heads = (
                torch.zeros(
                    normed.shape[0],
                    layer.self_attn.config.num_attention_heads,
                    normed.shape[1],
                    layer.self_attn.head_dim,
                    dtype=normed.dtype,
                    device=normed.device,
                )
                if disable_global_memory
                else self._cross_heads(
                    layer, normed, memory_k, memory_v, cos, sin, causal=True
                )
            )
        attended = cross_heads if self_heads is None else self_heads if cross_heads is None else _mix(self_heads, cross_heads, alpha)
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + layer.self_attn.o_proj(flat)
        residual = hidden
        return residual + layer.mlp(layer.post_attention_layernorm(hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        force_custom: bool = False,
        disable_global_memory: bool = False,
    ) -> HomotopyOutput:
        alpha = float(self.alpha)
        beta = float(self.beta)
        if not force_custom and alpha == 0.0 and beta == 0.0:
            native = self.base_model(input_ids=input_ids, labels=labels, use_cache=False)
            return HomotopyOutput(logits=native.logits.float(), loss=native.loss)
        hidden = self.base_model.model.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._positions(hidden, positions)
        for layer in self.lower:
            hidden = self._lower_layer(layer, hidden, cos, sin, beta)
        memory_k, memory_v = self._memory(hidden, cos, sin)
        for layer in self.upper:
            hidden = self._upper_layer(
                layer, hidden, memory_k, memory_v, cos, sin, alpha,
                disable_global_memory=disable_global_memory,
            )
        logits = self.base_model.lm_head(self.base_model.model.norm(hidden)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
            )
        return HomotopyOutput(logits=logits, loss=loss)

    def bridge_objective(self, input_ids: torch.Tensor) -> HomotopyOutput:
        """Match native upper attention outputs while the backbone is frozen."""
        captured_inputs: list[torch.Tensor] = []
        captured_outputs: list[torch.Tensor] = []

        def hook(_module, _args, kwargs, output):
            captured_inputs.append(kwargs["hidden_states"].detach())
            captured_outputs.append(output[0].detach())

        handles = [
            layer.self_attn.register_forward_hook(hook, with_kwargs=True)
            for layer in self.upper
        ]
        try:
            with torch.no_grad():
                native = self.base_model(
                    input_ids=input_ids, use_cache=False, output_hidden_states=True
                )
        finally:
            for handle in handles:
                handle.remove()
        encoder_final = native.hidden_states[self.encoder_layers].detach()
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._positions(encoder_final, positions)
        memory_k, memory_v = self._memory(encoder_final, cos, sin)
        losses = []
        for layer, normed, teacher in zip(
            self.upper, captured_inputs, captured_outputs, strict=True
        ):
            heads = self._cross_heads(
                layer, normed, memory_k, memory_v, cos, sin, causal=True
            )
            flat = heads.transpose(1, 2).reshape(normed.shape[0], normed.shape[1], -1)
            prediction = layer.self_attn.o_proj(flat)
            numerator = (prediction.float() - teacher.float()).square().mean()
            denominator = teacher.float().square().mean().clamp_min(1e-8)
            losses.append(numerator / denominator)
        bridge = torch.stack(losses).mean()
        return HomotopyOutput(
            logits=native.logits.float(), bridge_loss=bridge, bridge_by_layer=losses
        )

    def new_endpoint_cache(self) -> EndpointCache:
        return EndpointCache(
            local_keys=[None] * self.encoder_layers,
            local_values=[None] * self.encoder_layers,
        )

    @torch.inference_mode()
    def endpoint_prefill(self, input_ids: torch.Tensor):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("endpoint_prefill expects non-empty [batch, sequence]")
        hidden = self.base_model.model.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._positions(hidden, positions)
        cache = self.new_endpoint_cache()
        for index, layer in enumerate(self.lower):
            hidden, k, v = self._lower_layer(layer, hidden, cos, sin, 1.0, return_kv=True)
            cache.local_keys[index] = k[:, :, -self.encoder_window :]
            cache.local_values[index] = v[:, :, -self.encoder_window :]
        memory_k, memory_v = self._memory(hidden, cos, sin)
        cache.memory_keys, cache.memory_values = memory_k, memory_v
        cache.length = input_ids.shape[1]
        current = hidden[:, -1:]
        for layer in self.upper:
            residual = current
            normed = layer.input_layernorm(current)
            heads = self._cross_heads(
                layer, normed, memory_k, memory_v, cos[:, -1:], sin[:, -1:], causal=False
            )
            flat = heads.transpose(1, 2).reshape(current.shape[0], 1, -1)
            current = residual + layer.self_attn.o_proj(flat)
            residual = current
            current = residual + layer.mlp(layer.post_attention_layernorm(current))
        logits = self.base_model.lm_head(self.base_model.model.norm(current)).float()
        return logits, cache

    @torch.inference_mode()
    def endpoint_step(self, input_ids: torch.Tensor, cache: EndpointCache):
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("endpoint_step expects one token per batch")
        hidden = self.base_model.model.embed_tokens(input_ids)
        positions = torch.full(
            (input_ids.shape[0], 1), cache.length, device=input_ids.device, dtype=torch.long
        )
        cos, sin = self._positions(hidden, positions)
        for index, layer in enumerate(self.lower):
            attn = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q, k_new, v_new = self._project_qkv(attn, normed, cos, sin)
            old_k, old_v = cache.local_keys[index], cache.local_values[index]
            k = k_new if old_k is None else torch.cat([old_k, k_new], dim=2)
            v = v_new if old_v is None else torch.cat([old_v, v_new], dim=2)
            k = k[:, :, -self.encoder_window :]
            v = v[:, :, -self.encoder_window :]
            cache.local_keys[index], cache.local_values[index] = k, v
            heads = F.scaled_dot_product_attention(
                q, _repeat_kv(k, q.shape[1]), _repeat_kv(v, q.shape[1]),
                is_causal=False, dropout_p=0.0, scale=_scale(attn),
            )
            flat = heads.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
            hidden = residual + attn.o_proj(flat)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        memory_k_new, memory_v_new = self._memory(hidden, cos, sin)
        cache.memory_keys = torch.cat([cache.memory_keys, memory_k_new], dim=2)
        cache.memory_values = torch.cat([cache.memory_values, memory_v_new], dim=2)
        for layer in self.upper:
            residual = hidden
            normed = layer.input_layernorm(hidden)
            heads = self._cross_heads(
                layer, normed, cache.memory_keys, cache.memory_values, cos, sin, causal=False
            )
            flat = heads.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
            hidden = residual + layer.self_attn.o_proj(flat)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        cache.length += 1
        logits = self.base_model.lm_head(self.base_model.model.norm(hidden)).float()
        return logits, cache

    def endpoint_cache_accounting(self, cache: EndpointCache) -> dict[str, int]:
        local = sum(0 if x is None else x.numel() for x in cache.local_keys)
        local += sum(0 if x is None else x.numel() for x in cache.local_values)
        global_memory = 0 if cache.memory_keys is None else cache.memory_keys.numel()
        global_memory += 0 if cache.memory_values is None else cache.memory_values.numel()
        return {
            "local_kv_elements": local,
            "global_kv_elements": global_memory,
            "upper_self_kv_elements": 0,
            "retained_encoder_state_elements": 0,
            "sequence_length": cache.length,
        }

    @torch.inference_mode()
    def endpoint_full_trace(self, input_ids: torch.Tensor):
        hidden = self.base_model.model.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self._positions(hidden, positions)
        trace = {"embedding": hidden[:, -1:].float(), "lower": [], "upper": []}
        for layer in self.lower:
            hidden = self._lower_layer(layer, hidden, cos, sin, 1.0)
            trace["lower"].append(hidden[:, -1:].float())
        memory_k, memory_v = self._memory(hidden, cos, sin)
        trace["memory_k"] = memory_k[:, :, -1:].float()
        trace["memory_v"] = memory_v[:, :, -1:].float()
        for layer in self.upper:
            hidden = self._upper_layer(
                layer, hidden, memory_k, memory_v, cos, sin, 1.0
            )
            trace["upper"].append(hidden[:, -1:].float())
        normed = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(normed).float()
        trace["final_norm"] = normed[:, -1:].float()
        trace["logits"] = logits[:, -1:].float()
        return logits, trace

    @torch.inference_mode()
    def endpoint_step_trace(self, input_ids: torch.Tensor, cache: EndpointCache):
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("endpoint_step_trace expects one token per batch")
        hidden = self.base_model.model.embed_tokens(input_ids)
        positions = torch.full(
            (input_ids.shape[0], 1), cache.length, device=input_ids.device, dtype=torch.long
        )
        cos, sin = self._positions(hidden, positions)
        trace = {"embedding": hidden.float(), "lower": [], "upper": []}
        for index, layer in enumerate(self.lower):
            attn = layer.self_attn
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q, k_new, v_new = self._project_qkv(attn, normed, cos, sin)
            old_k, old_v = cache.local_keys[index], cache.local_values[index]
            k = k_new if old_k is None else torch.cat([old_k, k_new], dim=2)
            v = v_new if old_v is None else torch.cat([old_v, v_new], dim=2)
            k = k[:, :, -self.encoder_window :]
            v = v[:, :, -self.encoder_window :]
            cache.local_keys[index], cache.local_values[index] = k, v
            heads = F.scaled_dot_product_attention(
                q, _repeat_kv(k, q.shape[1]), _repeat_kv(v, q.shape[1]),
                is_causal=False, dropout_p=0.0, scale=_scale(attn),
            )
            flat = heads.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
            hidden = residual + attn.o_proj(flat)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            trace["lower"].append(hidden.float())
        memory_k_new, memory_v_new = self._memory(hidden, cos, sin)
        trace["memory_k"] = memory_k_new.float()
        trace["memory_v"] = memory_v_new.float()
        cache.memory_keys = torch.cat([cache.memory_keys, memory_k_new], dim=2)
        cache.memory_values = torch.cat([cache.memory_values, memory_v_new], dim=2)
        for layer in self.upper:
            residual = hidden
            normed = layer.input_layernorm(hidden)
            heads = self._cross_heads(
                layer, normed, cache.memory_keys, cache.memory_values, cos, sin, causal=False
            )
            flat = heads.transpose(1, 2).reshape(hidden.shape[0], 1, -1)
            hidden = residual + layer.self_attn.o_proj(flat)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            trace["upper"].append(hidden.float())
        cache.length += 1
        normed = self.base_model.model.norm(hidden)
        logits = self.base_model.lm_head(normed).float()
        trace["final_norm"] = normed.float()
        trace["logits"] = logits.float()
        return logits, cache, trace
