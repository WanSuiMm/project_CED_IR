from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError as exc:  # pragma: no cover - real GPU environment
    raise ImportError("InterfaceLM requires transformers with Qwen3 support") from exc


@dataclass
class InterfaceLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


def _heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    batch, length, width = x.shape
    return x.view(batch, length, width // head_dim, head_dim).transpose(1, 2)


def _repeat_kv(x: torch.Tensor, query_heads: int) -> torch.Tensor:
    return x.repeat_interleave(query_heads // x.shape[1], dim=1)


def _scale(attn: nn.Module) -> float:
    return float(getattr(attn, "scaling", attn.head_dim**-0.5))


class InterfaceReaderLayer(nn.Module):
    def __init__(self, source: nn.Module, hidden_size: int, value_head_multiplier: int = 2):
        super().__init__()
        if value_head_multiplier not in (1, 2):
            raise ValueError("value_head_multiplier must be 1 or 2")
        attn = source.self_attn
        self.input_layernorm = copy.deepcopy(source.input_layernorm)
        self.post_attention_layernorm = copy.deepcopy(source.post_attention_layernorm)
        self.q_proj = copy.deepcopy(attn.q_proj)
        self.q_norm = copy.deepcopy(attn.q_norm)
        self.k_norm = copy.deepcopy(attn.k_norm)
        self.mlp = copy.deepcopy(source.mlp)
        self.head_dim = int(attn.head_dim)
        self.query_heads = int(source.self_attn.config.num_attention_heads)
        self.kv_heads = int(source.self_attn.config.num_key_value_heads)
        self.scaling = _scale(attn)

        record_width = 2 * hidden_size
        key_width = self.kv_heads * self.head_dim
        value_head_dim = value_head_multiplier * self.head_dim
        value_width = self.kv_heads * value_head_dim
        self.value_head_dim = value_head_dim
        self.k_proj = nn.Linear(record_width, key_width, bias=attn.k_proj.bias is not None)
        self.v_proj = nn.Linear(record_width, value_width, bias=attn.v_proj.bias is not None)
        self.o_proj = nn.Linear(
            self.query_heads * value_head_dim,
            hidden_size,
            bias=attn.o_proj.bias is not None,
        )
        with torch.no_grad():
            self.k_proj.weight.zero_()
            self.k_proj.weight[:, :hidden_size].copy_(attn.k_proj.weight)
            if self.k_proj.bias is not None:
                self.k_proj.bias.copy_(attn.k_proj.bias)
            self.v_proj.weight.zero_()
            new_v = self.v_proj.weight.view(self.kv_heads, value_head_dim, record_width)
            old_v = attn.v_proj.weight.view(self.kv_heads, self.head_dim, hidden_size)
            new_v[:, : self.head_dim, :hidden_size].copy_(old_v)
            if self.v_proj.bias is not None:
                self.v_proj.bias.zero_()
                self.v_proj.bias.view(self.kv_heads, value_head_dim)[:, : self.head_dim].copy_(
                    attn.v_proj.bias.view(self.kv_heads, self.head_dim)
                )
            self.o_proj.weight.zero_()
            new_o = self.o_proj.weight.view(hidden_size, self.query_heads, value_head_dim)
            old_o = attn.o_proj.weight.view(hidden_size, self.query_heads, self.head_dim)
            new_o[:, :, : self.head_dim].copy_(old_o)
            if self.o_proj.bias is not None:
                self.o_proj.bias.copy_(attn.o_proj.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        records: torch.Tensor,
        allowed: torch.Tensor,
        disable_interface: bool = False,
    ) -> torch.Tensor:
        residual = hidden
        normed = self.input_layernorm(hidden)
        q = _heads(self.q_proj(normed), self.head_dim)
        q = self.q_norm(q)
        if disable_interface:
            attended = torch.zeros(
                hidden.shape[0], self.query_heads, hidden.shape[1], self.value_head_dim,
                dtype=hidden.dtype, device=hidden.device,
            )
        else:
            k = _heads(self.k_proj(records), self.head_dim)
            k = self.k_norm(k)
            v = _heads(self.v_proj(records), self.value_head_dim)
            attended = F.scaled_dot_product_attention(
                q,
                _repeat_kv(k, self.query_heads),
                _repeat_kv(v, self.query_heads),
                attn_mask=allowed[None, None],
                dropout_p=0.0,
                scale=self.scaling,
            )
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + self.o_proj(flat)
        residual = hidden
        return residual + self.mlp(self.post_attention_layernorm(hidden))


class InterfaceLM(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        variant: str,
        producer_layers: int = 4,
        reader_layers: int = 4,
        local_window: int = 16,
        value_head_multiplier: int = 2,
    ):
        super().__init__()
        if variant not in {"A_TOKEN", "B_PAIR_WIDE"}:
            raise ValueError("unknown variant")
        if producer_layers + reader_layers > len(base_model.model.layers):
            raise ValueError("requested more layers than checkpoint provides")
        self.variant = variant
        self.value_head_multiplier = value_head_multiplier
        self.config = base_model.config
        self.local_window = int(local_window)
        self.embed_tokens = copy.deepcopy(base_model.model.embed_tokens)
        self.rotary_emb = copy.deepcopy(base_model.model.rotary_emb)
        self.producer = nn.ModuleList(
            copy.deepcopy(list(base_model.model.layers[:producer_layers]))
        )
        hidden_size = int(base_model.config.hidden_size)
        self.reader = nn.ModuleList(
            InterfaceReaderLayer(layer, hidden_size, value_head_multiplier)
            for layer in base_model.model.layers[
                producer_layers : producer_layers + reader_layers
            ]
        )
        self.final_norm = copy.deepcopy(base_model.model.norm)
        self.lm_head = nn.Linear(hidden_size, base_model.config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.null_record = nn.Parameter(torch.zeros(1, 1, 2 * hidden_size))

    @classmethod
    def from_qwen(cls, base_model: nn.Module, **kwargs):
        return cls(base_model, **kwargs)

    def _producer_layer(self, layer, hidden, cos, sin):
        attn = layer.self_attn
        residual = hidden
        normed = layer.input_layernorm(hidden)
        q = _heads(attn.q_proj(normed), attn.head_dim)
        k = _heads(attn.k_proj(normed), attn.head_dim)
        v = _heads(attn.v_proj(normed), attn.head_dim)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        length = hidden.shape[1]
        pos = torch.arange(length, device=hidden.device)
        allowed = (
            (pos[None, :] <= pos[:, None])
            & (pos[None, :] >= pos[:, None] - self.local_window + 1)
        )
        attended = F.scaled_dot_product_attention(
            q,
            _repeat_kv(k, q.shape[1]),
            _repeat_kv(v, q.shape[1]),
            attn_mask=allowed[None, None],
            dropout_p=0.0,
            scale=_scale(attn),
        )
        flat = attended.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[1], -1)
        hidden = residual + attn.o_proj(flat)
        residual = hidden
        return residual + layer.mlp(layer.post_attention_layernorm(hidden))

    def _records(self, producer_hidden: torch.Tensor):
        batch, length, width = producer_hidden.shape
        if self.variant == "A_TOKEN":
            records = torch.cat([producer_hidden, torch.zeros_like(producer_hidden)], dim=-1)
            groups = torch.arange(length, device=producer_hidden.device) // 2
            persistent_scalars = length * width
        else:
            paired_length = 2 * (length // 2)
            records = producer_hidden[:, :paired_length].reshape(
                batch, paired_length // 2, 2 * width
            )
            groups = torch.arange(length // 2, device=producer_hidden.device)
            persistent_scalars = paired_length * width
        null = self.null_record.to(dtype=producer_hidden.dtype).expand(batch, -1, -1)
        records = torch.cat([null, records], dim=1)
        groups = torch.cat([torch.tensor([-1], device=groups.device), groups])
        query_groups = torch.arange(length, device=groups.device) // 2
        allowed = groups[None, :] < query_groups[:, None]
        allowed[:, 0] = True
        return records, allowed, persistent_scalars

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        disable_interface: bool = False,
    ) -> InterfaceLMOutput:
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, positions)
        for layer in self.producer:
            hidden = self._producer_layer(layer, hidden, cos, sin)
        records, allowed, _ = self._records(hidden)
        for layer in self.reader:
            hidden = layer(hidden, records, allowed, disable_interface)
        logits = self.lm_head(self.final_norm(hidden)).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return InterfaceLMOutput(logits=logits, loss=loss)

    def persistent_scalars(self, sequence_length: int) -> int:
        return sequence_length * int(self.config.hidden_size)

    def persistent_records(self, sequence_length: int) -> int:
        return sequence_length if self.variant == "A_TOKEN" else sequence_length // 2
