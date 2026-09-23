from __future__ import annotations

import unittest

from mla_pair.compare_b2 import compare


class CompareB2Test(unittest.TestCase):
    def _summary(self, variant, gaps):
        return {
            "variant": variant, "seed": 9, "steps": 4, "train_tokens": 8,
            "sequence_length": 2, "eval_sequences": 2, "eval_every": 1,
            "rope_dim_per_head": 2, "backbone_lr": 2e-5,
            "compiler_lr": 2e-4, "content_dim": 4, "rope_dim": 2,
            "extra_compiler_parameters": 8 if variant == "B2_HEADWISE" else 0,
            "curve": [
                {"step": step, "train_tokens": 2 * step,
                 "validation_nll": 2 + gap,
                 "per_sequence_nll": [2 + gap, 2 + gap]}
                for step, gap in enumerate(gaps)
            ],
        }

    def test_gap_and_late_trend(self):
        a = self._summary("A_TOKEN", [0] * 5)
        b = self._summary("B2_HEADWISE", [2, 1, .6, .3, .2])
        result = compare(a, b)
        self.assertAlmostEqual(result["final_b2_minus_a_nll"], .2)
        self.assertEqual(result["screen_signal_only"],
                         "OUTSIDE_MARGIN_STILL_IMPROVING")
        self.assertAlmostEqual(result["b2_to_a_reference_cache_ratio_even_length"],
                               2 / 3)

    def test_mismatch_fails(self):
        a = self._summary("A_TOKEN", [0] * 5)
        b = self._summary("B2_HEADWISE", [0] * 5)
        b["seed"] = 10
        with self.assertRaisesRegex(ValueError, "seed"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
