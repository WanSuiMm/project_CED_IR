from __future__ import annotations

import unittest

import torch

from mla_pair.qualify_token_mla import cache_tensors, summarize_cache


class CacheQualificationTest(unittest.TestCase):
    def test_compact_mla_cache_counts_actual_tensors(self):
        before = cache_tensors((
            (torch.empty(1, 8, 4), torch.empty(1, 8, 2)),
            (torch.empty(1, 8, 4), torch.empty(1, 8, 2)),
        ))
        after = cache_tensors((
            (torch.empty(1, 9, 4), torch.empty(1, 9, 2)),
            (torch.empty(1, 9, 4), torch.empty(1, 9, 2)),
        ))
        result = summarize_cache(before, after, layers=2, content_dim=4,
                                 rope_dim=2)
        self.assertTrue(result["matches_compact_mla_layout"])
        self.assertEqual(result["observed_scalars_per_token_per_layer"], 6)
        self.assertEqual(result["observed_bytes_per_token"], 48)

    def test_materialized_gqa_cache_is_rejected(self):
        before = cache_tensors(((torch.empty(1, 8, 8),
                                 torch.empty(1, 8, 8)),))
        after = cache_tensors(((torch.empty(1, 9, 8),
                                torch.empty(1, 9, 8)),))
        result = summarize_cache(before, after, layers=1, content_dim=4,
                                 rope_dim=2)
        self.assertFalse(result["matches_compact_mla_layout"])
        self.assertEqual(result["observed_scalars_per_token"], 16)

    def test_unknown_and_non_appending_cache_fail_closed(self):
        with self.assertRaises(TypeError):
            cache_tensors(object())
        before = {"key": torch.empty(1, 8, 4)}
        after = {"key": torch.empty(1, 8, 4)}
        with self.assertRaises(ValueError):
            summarize_cache(before, after, layers=1, content_dim=3,
                            rope_dim=1)


if __name__ == "__main__":
    unittest.main()
