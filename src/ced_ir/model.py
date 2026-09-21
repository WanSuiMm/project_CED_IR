from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


def _linear(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Use identical per-record GEMMs in the strict FP64 reference path."""
    if x.dtype == torch.float64 and x.ndim == 3 and x.shape[1] > 1:
        return torch.cat([F.linear(x[:, i:i + 1], layer.weight, layer.bias)
                          for i in range(x.shape[1])], dim=1)
    return F.linear(x, layer.weight, layer.bias)


def _tied_head(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float64 and x.ndim == 3 and x.shape[1] > 1:
        return torch.cat([F.linear(x[:, i:i + 1], weight)
                          for i in range(x.shape[1])], dim=1)
    return F.linear(x, weight)


@dataclass(frozen=True)
class CEDConfig:
    vocab_size: int = 1024
    d_model: int = 256
    heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    local_window: int = 32
    mlp_intermediate: int = 688
    block_size: int = 1
    width_multiplier: int = 1
    init_seed: int = 17
    rope_base: float = 10000.0
    slot_gate_dim: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if (self.d_model // self.heads) % 2:
            raise ValueError("head width must be even for RoPE")
        if self.block_size < 1 or self.width_multiplier < 1:
            raise ValueError("invalid IR shape")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight.to(x.dtype)


def _rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    # x: [batch, heads, length, head_width]
    half = x.shape[-1] // 2
    inv = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    angles = positions.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = angles.cos()[None, None], angles.sin()[None, None]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(width, 2 * hidden, bias=False)
        self.out_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = _linear(self.in_proj, x).chunk(2, dim=-1)
        return _linear(self.out_proj, F.silu(gate) * value)


class LocalAttention(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.width = cfg.d_model
        self.heads = cfg.heads
        self.head_width = cfg.d_model // cfg.heads
        self.window = cfg.local_window
        self.rope_base = cfg.rope_base
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, _ = x.shape
        if positions is None:
            positions = torch.arange(length, device=x.device)
        qkv = _linear(self.qkv, x).view(batch, length, 3, self.heads, self.head_width)
        q, k, v = qkv.unbind(dim=2)
        q = _rope(q.transpose(1, 2), positions, self.rope_base)
        k = _rope(k.transpose(1, 2), positions, self.rope_base)
        v = v.transpose(1, 2)
        if x.dtype == torch.float64:
            # A deliberately simple reference path. Per-query evaluation makes
            # full and cached decoding use the same FP64 reduction order.
            rows = []
            for index, position in enumerate(positions.tolist()):
                begin = max(0, index - self.window + 1)
                q_row = q[:, :, index:index + 1]
                k_row = k[:, :, begin:index + 1]
                v_row = v[:, :, begin:index + 1]
                scores = torch.matmul(q_row, k_row.transpose(-1, -2)) / math.sqrt(self.head_width)
                rows.append(torch.matmul(torch.softmax(scores, dim=-1), v_row))
            out = torch.cat(rows, dim=2)
        else:
            row = positions[:, None]
            col = positions[None, :]
            allowed = (col <= row) & (col >= row - self.window + 1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                                 dropout_p=0.0, is_causal=False)
        return _linear(self.out, out.transpose(1, 2).contiguous().view(batch, length, self.width))

    def cache_kv(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        qkv = _linear(self.qkv, x).view(batch, length, 3, self.heads, self.head_width)
        _, k, v = qkv.unbind(dim=2)
        k = _rope(k.transpose(1, 2), positions, self.rope_base)
        return k, v.transpose(1, 2)

    def step(self, x: torch.Tensor, position: int,
             cache: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch = x.shape[0]
        qkv = _linear(self.qkv, x).view(batch, 1, 3, self.heads, self.head_width)
        q, k, v = qkv.unbind(dim=2)
        pos = torch.tensor([position], device=x.device)
        q = _rope(q.transpose(1, 2), pos, self.rope_base)
        k = _rope(k.transpose(1, 2), pos, self.rope_base)
        v = v.transpose(1, 2)
        if cache is not None:
            k_all = torch.cat((cache[0], k), dim=2)
            v_all = torch.cat((cache[1], v), dim=2)
        else:
            k_all, v_all = k, v
        k_all = k_all[:, :, -self.window:]
        v_all = v_all[:, :, -self.window:]
        if x.dtype == torch.float64:
            scores = torch.matmul(q, k_all.transpose(-1, -2)) / math.sqrt(self.head_width)
            out = torch.matmul(torch.softmax(scores, dim=-1), v_all)
        else:
            out = F.scaled_dot_product_attention(q, k_all, v_all, dropout_p=0.0, is_causal=False)
        result = _linear(self.out, out.transpose(1, 2).contiguous().view(batch, 1, self.width))
        keep = max(0, self.window - 1)
        new_cache = (k_all[:, :, -keep:].detach() if keep else k_all[:, :, :0].detach(),
                     v_all[:, :, -keep:].detach() if keep else v_all[:, :, :0].detach())
        return result, new_cache


class EncoderBlock(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = LocalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_intermediate)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), positions)
        return x + self.mlp(self.norm2(x))


class IRMemory(NamedTuple):
    keys: torch.Tensor       # [B,H,M,A]
    values: torch.Tensor     # [B,H,M,s*A]
    block_ends: torch.Tensor # [M]
    packed: torch.Tensor     # [B,M,s*d]


class IRCompiler(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.block_size * cfg.d_model,
                              cfg.width_multiplier * cfg.d_model, bias=False)
        self.key = nn.Linear(cfg.width_multiplier * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, h: torch.Tensor) -> IRMemory:
        batch, length, width = h.shape
        blocks = length // self.cfg.block_size
        source = h[:, :blocks * self.cfg.block_size].reshape(
            batch, blocks, self.cfg.block_size * width)
        z = _linear(self.proj, source)
        k = _linear(self.key, z).view(batch, blocks, self.cfg.heads, width // self.cfg.heads)
        k = k.transpose(1, 2)
        s = self.cfg.width_multiplier
        a = width // self.cfg.heads
        # Source-slot-major layout, then head and channel.
        v = z.view(batch, blocks, s, self.cfg.heads, a).permute(0, 3, 1, 2, 4)
        v = v.contiguous().view(batch, self.cfg.heads, blocks, s * a)
        ends = torch.arange(blocks, device=h.device) * self.cfg.block_size + self.cfg.block_size - 1
        return IRMemory(k, v, ends, z)


class GlobalRead(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.width_multiplier * cfg.d_model, cfg.d_model, bias=False)
        self.gate_dim = cfg.slot_gate_dim
        if self.gate_dim:
            if cfg.width_multiplier != 2:
                raise ValueError("slot gate v0.1 is defined only for two payload slots")
            self.gate_query = nn.Linear(cfg.d_model, cfg.heads * self.gate_dim, bias=False)
            head_width = cfg.d_model // cfg.heads
            self.gate_slot_proj = nn.Parameter(
                torch.zeros(cfg.heads, 2, head_width, self.gate_dim))

    def routing_diagnostics(self, x: torch.Tensor, memory: IRMemory,
                            positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return coarse block weights and optional within-block gate weights.

        This mirrors the regular float inference path and is intentionally
        separate from the returned activation so an audit cannot affect model
        behavior.
        """
        batch, length, width = x.shape
        a = width // self.cfg.heads
        q = _linear(self.query, x).view(batch, length, self.cfg.heads, a).transpose(1, 2)
        q = _rope(q, positions, self.cfg.rope_base)
        k = _rope(memory.keys, memory.block_ends, self.cfg.rope_base)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a)
        allowed = memory.block_ends[None, :] <= positions[:, None]
        scores = scores.float().masked_fill(~allowed[None, None], -torch.inf)
        valid = allowed.any(dim=-1)
        scores = torch.where(valid[None, None, :, None], scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1).to(x.dtype)
        weights = weights * valid[None, None, :, None].to(weights.dtype)
        if not self.gate_dim:
            return weights, None
        slot_values = memory.values.view(batch, self.cfg.heads, -1, 2, a)
        gate_q = _linear(self.gate_query, x).view(
            batch, length, self.cfg.heads, self.gate_dim).permute(0, 2, 1, 3)
        gate_k = torch.einsum("bhmsa,hsac->bhmsc", slot_values,
                              self.gate_slot_proj.to(slot_values.dtype))
        gate_logits = torch.einsum("bhtc,bhmsc->bhtms", gate_q, gate_k)
        gate = torch.softmax(gate_logits.float() / math.sqrt(self.gate_dim), dim=-1).to(x.dtype)
        return weights, gate

    def forward(self, x: torch.Tensor, memory: IRMemory, positions: torch.Tensor,
                ablate: bool = False) -> torch.Tensor:
        if ablate or memory.block_ends.numel() == 0:
            return torch.zeros_like(x)
        batch, length, width = x.shape
        a = width // self.cfg.heads
        q = _linear(self.query, x).view(batch, length, self.cfg.heads, a).transpose(1, 2)
        q = _rope(q, positions, self.cfg.rope_base)
        k = _rope(memory.keys, memory.block_ends, self.cfg.rope_base)
        if self.gate_dim:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a)
            allowed = memory.block_ends[None, :] <= positions[:, None]
            scores = scores.float().masked_fill(~allowed[None, None], -torch.inf)
            valid = allowed.any(dim=-1)
            scores = torch.where(valid[None, None, :, None], scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, dim=-1).to(x.dtype)
            weights = weights * valid[None, None, :, None].to(weights.dtype)
            slot_values = memory.values.view(batch, self.cfg.heads, -1, 2, a)
            gate_q = _linear(self.gate_query, x).view(
                batch, length, self.cfg.heads, self.gate_dim).permute(0, 2, 1, 3)
            gate_k = torch.einsum("bhmsa,hsac->bhmsc", slot_values,
                                  self.gate_slot_proj.to(slot_values.dtype))
            gate_logits = torch.einsum("bhtc,bhmsc->bhtms", gate_q, gate_k)
            gate_logits = gate_logits.float() / math.sqrt(self.gate_dim)
            gate = torch.softmax(gate_logits, dim=-1).to(x.dtype)
            # Factor 2 makes the zero-initialized uniform gate reproduce the
            # original concatenated wide-value reader at update zero.
            out_slots = []
            for slot in range(2):
                effective = weights * (2.0 * gate[..., slot])
                out_slots.append(torch.matmul(effective, slot_values[..., slot, :]))
            out = torch.cat(out_slots, dim=-1)
        elif x.dtype == torch.float64:
            rows = []
            for index, position in enumerate(positions.tolist()):
                visible = int((memory.block_ends <= position).sum().item())
                if visible == 0:
                    rows.append(torch.zeros(batch, self.cfg.heads, 1,
                                            self.cfg.width_multiplier * a,
                                            device=x.device, dtype=x.dtype))
                    continue
                scores = torch.matmul(q[:, :, index:index + 1],
                                      k[:, :, :visible].transpose(-1, -2)) / math.sqrt(a)
                rows.append(torch.matmul(torch.softmax(scores, dim=-1),
                                         memory.values[:, :, :visible]))
            out = torch.cat(rows, dim=2)
        else:
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(a)
            allowed = memory.block_ends[None, :] <= positions[:, None]
            scores = scores.float().masked_fill(~allowed[None, None], -torch.inf)
            valid = allowed.any(dim=-1)
            scores = torch.where(valid[None, None, :, None], scores, torch.zeros_like(scores))
            weights = torch.softmax(scores, dim=-1).to(x.dtype)
            weights = weights * valid[None, None, :, None].to(weights.dtype)
            out = torch.matmul(weights, memory.values)
        out = out.transpose(1, 2).contiguous().view(
            batch, length, self.cfg.width_multiplier * width)
        return _linear(self.out, out)


class DecoderBlock(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.local = LocalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.global_read = GlobalRead(cfg)
        self.norm3 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_intermediate)

    def forward(self, x: torch.Tensor, memory: IRMemory, positions: torch.Tensor,
                ablate_global: bool) -> torch.Tensor:
        x = x + self.local(self.norm1(x), positions)
        x = x + self.global_read(self.norm2(x), memory, positions, ablate_global)
        return x + self.mlp(self.norm3(x))


class CEDIRModel(nn.Module):
    def __init__(self, cfg: CEDConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.encoder_layers)])
        self.encoder_norm = RMSNorm(cfg.d_model)
        self.compiler = IRCompiler(cfg)
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.decoder_layers)])
        self.output_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Name-derived generators make all same-name/same-shape parameters equal
        # across A/B despite variant-specific matrices changing constructor RNG use.
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("weight") and "norm" in name:
                    parameter.fill_(1.0)
                    continue
                digest = hashlib.sha256(f"{self.cfg.init_seed}|{name}".encode()).digest()
                generator = torch.Generator(device=parameter.device)
                generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
            self.embedding.weight.normal_(mean=0.0, std=0.02,
                                          generator=torch.Generator().manual_seed(self.cfg.init_seed))
            self.lm_head.weight = self.embedding.weight
            self.compiler.proj.weight.zero_()
            rows, cols = self.compiler.proj.weight.shape
            if self.cfg.block_size == self.cfg.width_multiplier:
                self.compiler.proj.weight[:min(rows, cols), :min(rows, cols)].copy_(
                    torch.eye(min(rows, cols), device=self.compiler.proj.weight.device))
            elif self.cfg.block_size == 2 and self.cfg.width_multiplier == 1:
                d = self.cfg.d_model
                eye = torch.eye(d, device=self.compiler.proj.weight.device) / math.sqrt(2.0)
                self.compiler.proj.weight[:, :d].copy_(eye)
                self.compiler.proj.weight[:, d:].copy_(eye)
            for block in self.decoder:
                reader = block.global_read
                if reader.gate_dim:
                    digest = hashlib.sha256(
                        f"{self.cfg.init_seed}|slot_gate_query".encode()).digest()
                    generator = torch.Generator(device=reader.gate_query.weight.device)
                    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                    reader.gate_query.weight.normal_(0.0, 0.02, generator=generator)
                    reader.gate_slot_proj.zero_()

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.embedding(input_ids)
        for block in self.encoder:
            x = block(x, positions)
        return self.encoder_norm(x)

    def forward(self, input_ids: torch.Tensor, ablate_global: bool = False,
                return_memory: bool = False):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        h = self.encode(input_ids)
        memory = self.compiler(h)
        x = h
        for block in self.decoder:
            x = block(x, memory, positions, ablate_global)
        logits = _tied_head(self.output_norm(x), self.embedding.weight)
        if return_memory:
            return logits, memory
        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _empty_memory(self, batch: int, device: torch.device, dtype: torch.dtype) -> IRMemory:
        a = self.cfg.d_model // self.cfg.heads
        return IRMemory(
            torch.empty(batch, self.cfg.heads, 0, a, device=device, dtype=dtype),
            torch.empty(batch, self.cfg.heads, 0, self.cfg.width_multiplier * a,
                        device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(batch, 0, self.cfg.width_multiplier * self.cfg.d_model,
                        device=device, dtype=dtype),
        )

    def init_stream_state(self, batch: int, device: torch.device,
                          dtype: torch.dtype) -> "StreamState":
        return StreamState(position=0,
                           encoder_local=[None] * len(self.encoder),
                           decoder_local=[None] * len(self.decoder),
                           memory=self._empty_memory(batch, device, dtype),
                           tail=torch.empty(batch, 0, self.cfg.d_model,
                                            device=device, dtype=dtype))

    def _commit_tail(self, state: "StreamState") -> None:
        if state.tail.shape[1] < self.cfg.block_size:
            return
        source = state.tail[:, :self.cfg.block_size].reshape(
            state.tail.shape[0], 1, self.cfg.block_size * self.cfg.d_model)
        z = _linear(self.compiler.proj, source)
        a = self.cfg.d_model // self.cfg.heads
        k = _linear(self.compiler.key, z).view(state.tail.shape[0], 1, self.cfg.heads, a).transpose(1, 2)
        v = z.view(state.tail.shape[0], 1, self.cfg.width_multiplier,
                   self.cfg.heads, a).permute(0, 3, 1, 2, 4).contiguous()
        v = v.view(state.tail.shape[0], self.cfg.heads, 1, self.cfg.width_multiplier * a)
        end = torch.tensor([state.position - 1], device=state.tail.device, dtype=torch.long)
        state.memory = IRMemory(torch.cat((state.memory.keys, k), dim=2),
                                torch.cat((state.memory.values, v), dim=2),
                                torch.cat((state.memory.block_ends, end)),
                                torch.cat((state.memory.packed, z), dim=1))
        state.tail = state.tail[:, self.cfg.block_size:]

    def stream_step(self, token_ids: torch.Tensor, state: "StreamState") -> torch.Tensor:
        if token_ids.ndim != 1:
            raise ValueError("stream_step expects [batch] token IDs")
        position = state.position
        x = self.embedding(token_ids)[:, None, :]
        for layer_index, block in enumerate(self.encoder):
            local, cache = block.attn.step(block.norm1(x), position,
                                           state.encoder_local[layer_index])
            state.encoder_local[layer_index] = cache
            x = x + local
            x = x + block.mlp(block.norm2(x))
        h = self.encoder_norm(x)
        state.tail = torch.cat((state.tail, h), dim=1)
        state.position += 1
        self._commit_tail(state)
        x = h
        positions = torch.tensor([position], device=x.device)
        for layer_index, block in enumerate(self.decoder):
            local, cache = block.local.step(block.norm1(x), position,
                                            state.decoder_local[layer_index])
            state.decoder_local[layer_index] = cache
            x = x + local
            x = x + block.global_read(block.norm2(x), state.memory, positions)
            x = x + block.mlp(block.norm3(x))
        return _tied_head(self.output_norm(x[:, 0]), self.embedding.weight)

    def prefill_exact_replay(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, "StreamState"]:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        x = self.embedding(input_ids)
        encoder_caches = []
        for block in self.encoder:
            normalized = block.norm1(x)
            k, v = block.attn.cache_kv(normalized, positions)
            keep = max(0, self.cfg.local_window - 1)
            encoder_caches.append((k[:, :, -keep:].detach(), v[:, :, -keep:].detach()))
            x = x + block.attn(normalized, positions)
            x = x + block.mlp(block.norm2(x))
        h = self.encoder_norm(x)
        memory = self.compiler(h)
        replay = self.cfg.decoder_layers * (self.cfg.local_window - 1) + 1
        start = max(0, length - replay)
        replay_positions = positions[start:]
        x = h[:, start:]
        decoder_caches = []
        for block in self.decoder:
            normalized = block.norm1(x)
            k, v = block.local.cache_kv(normalized, replay_positions)
            keep = max(0, self.cfg.local_window - 1)
            decoder_caches.append((k[:, :, -keep:].detach(), v[:, :, -keep:].detach()))
            x = x + block.local(normalized, replay_positions)
            x = x + block.global_read(block.norm2(x), memory, replay_positions)
            x = x + block.mlp(block.norm3(x))
        logits = _tied_head(self.output_norm(x), self.embedding.weight)
        tail_count = length % self.cfg.block_size
        tail = h[:, -tail_count:].detach() if tail_count else h[:, :0].detach()
        state = StreamState(position=length, encoder_local=encoder_caches,
                            decoder_local=decoder_caches, memory=memory,
                            tail=tail)
        return logits, state


@dataclass
class StreamState:
    position: int
    encoder_local: list[tuple[torch.Tensor, torch.Tensor] | None]
    decoder_local: list[tuple[torch.Tensor, torch.Tensor] | None]
    memory: IRMemory
    tail: torch.Tensor
