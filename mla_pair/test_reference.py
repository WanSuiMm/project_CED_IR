from __future__ import annotations

import unittest

import torch

from mla_pair.reference import (
    PairCache,
    PairCompiler,
    append_token,
    cache_scalars,
    pair_attention,
    pair_attention_step,
    token_attention,
)


class DualResolutionReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch, self.heads, self.length = 2, 3, 7
        self.dc, self.dr, self.dv = 5, 2, 4
        self.content = torch.randn(self.batch, self.length, self.dc, dtype=torch.float64)
        self.rope = torch.randn(self.batch, self.length, self.dr, dtype=torch.float64)
        self.qc = torch.randn(self.batch, self.heads, self.length, self.dc, dtype=torch.float64)
        self.qr = torch.randn(self.batch, self.heads, self.length, self.dr, dtype=torch.float64)
        self.vw = torch.randn(self.heads, self.dc, self.dv, dtype=torch.float64)
        self.compiler = PairCompiler(self.dc).double()
        self.scale = 0.3

    def _pair(self, content=None, rope=None):
        return pair_attention(self.qc, self.qr,
                              self.content if content is None else content,
                              self.rope if rope is None else rope,
                              self.vw, self.compiler, self.scale)

    def test_lse_equals_explicit_two_anchor_softmax(self):
        """A direct token softmax with shared pair K/V is the algebraic oracle."""
        outputs = self._pair()
        for t in range(self.length):
            completed = (t + 1) // 2
            pair_c = self.compiler(
                self.content[:, 0:2 * completed:2],
                self.content[:, 1:2 * completed:2],
            )
            expanded_c = pair_c.repeat_interleave(2, dim=1)
            if t % 2 == 0:
                expanded_c = torch.cat((expanded_c, self.content[:, t:t + 1]), 1)
            keys = self.rope[:, :t + 1]
            score = (torch.einsum("bhd,bnd->bhn", self.qc[:, :, t], expanded_c)
                     + torch.einsum("bhr,bnr->bhn", self.qr[:, :, t], keys)) * self.scale
            values = torch.einsum("bnd,hdv->bhnv", expanded_c, self.vw)
            expected = torch.einsum("bhn,bhnv->bhv", score.softmax(-1), values)[:, :, None]
            torch.testing.assert_close(outputs[:, :, t:t + 1], expected,
                                       rtol=1e-12, atol=1e-12)

    def test_future_content_and_rope_do_not_change_past(self):
        original = self._pair()
        changed_c = self.content.clone()
        changed_r = self.rope.clone()
        changed_c[:, 5:] += 100
        changed_r[:, 5:] -= 70
        altered = self._pair(changed_c, changed_r)
        torch.testing.assert_close(original[:, :, :5], altered[:, :, :5],
                                   rtol=0, atol=0)

    def test_incremental_matches_full_sequence_and_cache_accounting(self):
        full = self._pair()
        cache = PairCache()
        for t in range(self.length):
            cache = append_token(cache, self.content[:, t], self.rope[:, t], self.compiler)
            step = pair_attention_step(self.qc[:, :, t], self.qr[:, :, t],
                                       cache, self.vw, self.scale)
            torch.testing.assert_close(step, full[:, :, t], rtol=1e-12, atol=1e-12)
            self.assertEqual(cache.length, t + 1)
            self.assertEqual(cache_scalars(cache),
                             ((t + 1) // 2) * self.dc + (t + 1) * self.dr
                             + ((t + 1) % 2) * self.dc)
        # For 2k completed tokens: k * dc + 2k * dr, versus 2k * (dc + dr).
        even = PairCache()
        for t in range(6):
            even = append_token(even, self.content[:, t], self.rope[:, t], self.compiler)
        self.assertEqual(cache_scalars(even), 3 * self.dc + 6 * self.dr)

    def test_compiler_receives_gradients(self):
        self._pair().square().mean().backward()
        self.assertGreater(float(self.compiler.linear.weight.grad.abs().sum()), 0)

    def test_token_reference_is_causal(self):
        original = token_attention(self.qc, self.qr, self.content,
                                   self.rope, self.vw, self.scale)
        changed = self.content.clone()
        changed[:, 5:] += 100
        altered = token_attention(self.qc, self.qr, changed,
                                  self.rope, self.vw, self.scale)
        torch.testing.assert_close(original[:, :, :5], altered[:, :, :5],
                                   rtol=0, atol=0)

    def test_bad_shapes_fail(self):
        with self.assertRaises(ValueError):
            self._pair(rope=self.rope[:, :-1])
        with self.assertRaises(ValueError):
            append_token(PairCache(), self.content[:, 0], self.rope[:, 0, None],
                         self.compiler)


if __name__ == "__main__":
    unittest.main()
