from __future__ import annotations

import math
import unittest

import torch

from mla_pair.headwise_compiler import HeadwiseKVPairCompiler
from mla_pair.qwen3_cache import PairDynamicCache
from mla_pair.qwen3_split import PairQwen3Attention, SplitQwen3Attention
from mla_pair.test_qwen3_split import FakeSource


class HeadwiseCompilerTest(unittest.TestCase):
    def test_parameter_count_and_initialization(self):
        compiler = HeadwiseKVPairCompiler(8, 32, 128)
        self.assertEqual(sum(p.numel() for p in compiler.parameters()) * 28,
                         7_834_624)
        first = torch.randn(2, 3, 8, 160)
        second = torch.randn_like(first)
        got = compiler(first.flatten(-2), second.flatten(-2)).view(2, 3, 8, 160)
        expected_key = (first[..., :32] + second[..., :32]) / math.sqrt(2)
        expected_value = (first[..., 32:] + second[..., 32:]) / 2
        torch.testing.assert_close(got[..., :32], expected_key)
        torch.testing.assert_close(got[..., 32:], expected_value)

    def test_no_cross_head_or_key_value_mixing(self):
        compiler = HeadwiseKVPairCompiler(2, 2, 3).double()
        first = torch.zeros(1, 10, dtype=torch.float64)
        second = first.clone()
        baseline = compiler(first, second)
        changed = first.clone()
        changed[0, 2:5] = 1  # value coordinates in KV head 0
        got = compiler(changed, second) - baseline
        torch.testing.assert_close(got[..., :2], torch.zeros_like(got[..., :2]))
        torch.testing.assert_close(got[..., 5:], torch.zeros_like(got[..., 5:]))
        self.assertGreater(float(got[..., 2:5].detach().abs().sum()), 0)

    def test_full_cache_parity_and_gradients(self):
        torch.manual_seed(39)
        source = FakeSource().double()
        pair = PairQwen3Attention(
            SplitQwen3Attention(source, rope_dim_per_head=4).double(),
            compiler_kind="headwise_kv").double()
        hidden = torch.randn(2, 5, 16, dtype=torch.float64)
        angles = torch.randn(1, 5, 4, dtype=torch.float64)
        cos = torch.cat((angles.cos(), angles.cos()), -1)
        sin = torch.cat((angles.sin(), angles.sin()), -1)
        full, _ = pair(hidden, (cos, sin), None)
        cache = PairDynamicCache()
        prefix, _ = pair(hidden[:, :4], (cos[:, :4], sin[:, :4]), None,
                         past_key_values=cache)
        step, _ = pair(hidden[:, 4:], (cos[:, 4:], sin[:, 4:]), None,
                       past_key_values=cache)
        torch.testing.assert_close(torch.cat((prefix, step), 1), full,
                                   rtol=1e-12, atol=1e-12)
        self.assertEqual(cache.persistent_scalars(),
                         2 * (3 * pair.content_dim + 5 * pair.rope_dim))
        full.square().mean().backward()
        self.assertGreater(float(pair.compiler.key_weight.grad.abs().sum()), 0)
        self.assertGreater(float(pair.compiler.value_weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
