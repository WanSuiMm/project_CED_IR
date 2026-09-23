from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from mla_pair.qwen3_split import PairQwen3Attention, SplitQwen3Attention, _rotate_half
from mla_pair.qwen3_cache import PairDynamicCache


class FakeSource(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
        self.layer_idx = 0
        self.head_dim = 8
        self.attention_dropout = 0.0
        self.q_proj = nn.Linear(16, 32, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        self.v_proj = nn.Linear(16, 16, bias=False)
        self.o_proj = nn.Linear(32, 16, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


class FakeCache:
    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, keys, values, layer_idx):
        assert layer_idx == 0
        self.keys = keys if self.keys is None else torch.cat((self.keys, keys), dim=2)
        self.values = values if self.values is None else torch.cat((self.values, values), dim=2)
        return self.keys, self.values


class Qwen3SplitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(39)
        self.source = FakeSource().double()
        self.hidden = torch.randn(2, 5, 16, dtype=torch.float64)
        angles = torch.randn(1, 5, 4, dtype=torch.float64)
        self.cos = torch.cat((angles.cos(), angles.cos()), -1)
        self.sin = torch.cat((angles.sin(), angles.sin()), -1)

    def test_full_rope_width_exactly_matches_native_attention(self):
        split = SplitQwen3Attention(self.source, rope_dim_per_head=8).double()
        got, _ = split(self.hidden, (self.cos, self.sin), None)
        batch, length, _ = self.hidden.shape
        q = self.source.q_proj(self.hidden).view(batch, length, 4, 8).transpose(1, 2)
        k = self.source.k_proj(self.hidden).view(batch, length, 2, 8).transpose(1, 2)
        v = self.source.v_proj(self.hidden).view(batch, length, 2, 8).transpose(1, 2)
        q = q * self.cos[:, None] + _rotate_half(q) * self.sin[:, None]
        k = k * self.cos[:, None] + _rotate_half(k) * self.sin[:, None]
        k, v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
        expected = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=8**-0.5)
        expected = self.source.o_proj(expected.transpose(1, 2).reshape(batch, length, -1))
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)

    def test_split_cache_is_compact_and_matches_full_continuation(self):
        split = SplitQwen3Attention(self.source, rope_dim_per_head=4).double()
        full, _ = split(self.hidden, (self.cos, self.sin), None)
        cache = FakeCache()
        split(self.hidden[:, :4], (self.cos[:, :4], self.sin[:, :4]), None,
              past_key_values=cache)
        step, _ = split(self.hidden[:, 4:], (self.cos[:, 4:], self.sin[:, 4:]),
                        None, past_key_values=cache)
        torch.testing.assert_close(step, full[:, 4:], rtol=1e-12, atol=1e-12)
        self.assertEqual(tuple(cache.keys.shape), (2, 1, 5, split.content_dim))
        self.assertEqual(tuple(cache.values.shape), (2, 1, 5, split.rope_dim))
        self.assertEqual(split.content_dim + split.rope_dim, 2 * 2 * 8)

    def test_pair_attention_matches_explicit_anchor_oracle(self):
        split = SplitQwen3Attention(self.source, rope_dim_per_head=4).double()
        pair = PairQwen3Attention(split).double()
        output, _ = pair(self.hidden, (self.cos, self.sin), None)
        q_c, q_r, content, rope = pair._split(self.hidden, (self.cos, self.sin))
        batch, length, _ = self.hidden.shape
        pair_c = pair.compiler(content[:, 0:4:2], content[:, 1:4:2])
        expected = []
        key_width = pair.head_dim - pair.rope_dim_per_head
        for t in range(length):
            completed = (t + 1) // 2
            latent = pair_c[:, :completed].repeat_interleave(2, 1)
            if t % 2 == 0:
                latent = torch.cat((latent, content[:, t:t + 1]), 1)
            shaped = latent.view(batch, t + 1, pair.num_kv_heads,
                                 key_width + pair.head_dim).transpose(1, 2)
            key, value = shaped.split((key_width, pair.head_dim), -1)
            key = key.repeat_interleave(2, 1)
            value = value.repeat_interleave(2, 1)
            pos_key = rope[:, :t + 1].view(batch, t + 1, pair.num_kv_heads,
                                            pair.rope_dim_per_head).transpose(1, 2)
            pos_key = pos_key.repeat_interleave(2, 1)
            scores = (torch.einsum("bhd,bhnd->bhn", q_c[:, :, t], key)
                      + torch.einsum("bhr,bhnr->bhn", q_r[:, :, t], pos_key))
            value_out = torch.einsum("bhn,bhnd->bhd",
                                     (scores * pair.scaling).softmax(-1), value)
            expected.append(value_out)
        expected = torch.stack(expected, 2).transpose(1, 2).reshape(batch, length, -1)
        expected = pair.o_proj(expected)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        output.square().mean().backward()
        self.assertGreater(float(pair.compiler.linear.weight.grad.abs().sum()), 0)

    def test_pair_cache_matches_full_sequence_and_real_storage(self):
        split = SplitQwen3Attention(self.source, rope_dim_per_head=4).double()
        pair = PairQwen3Attention(split).double()
        full, _ = pair(self.hidden, (self.cos, self.sin), None)
        cache = PairDynamicCache()
        prefix, _ = pair(self.hidden[:, :4],
                         (self.cos[:, :4], self.sin[:, :4]), None,
                         past_key_values=cache)
        step, _ = pair(self.hidden[:, 4:],
                       (self.cos[:, 4:], self.sin[:, 4:]), None,
                       past_key_values=cache)
        torch.testing.assert_close(torch.cat((prefix, step), 1), full,
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(cache.get_seq_length(), 5)
        self.assertEqual(cache.persistent_scalars(),
                         self.hidden.shape[0] * (3 * pair.content_dim
                                                 + 5 * pair.rope_dim))


if __name__ == "__main__":
    unittest.main()
