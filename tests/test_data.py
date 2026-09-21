import unittest

import numpy as np

from ced_ir.synthetic import SyntheticConfig, make_example


class SyntheticDataTests(unittest.TestCase):
    def test_deterministic_and_split_separated(self):
        cfg = SyntheticConfig()
        a = make_example(cfg, "train", 9)
        b = make_example(cfg, "train", 9)
        c = make_example(cfg, "validation", 9)
        np.testing.assert_array_equal(a["input_ids"], b["input_ids"])
        self.assertFalse(np.array_equal(a["input_ids"], c["input_ids"]))

    def test_shapes_and_supervision(self):
        cfg = SyntheticConfig()
        ex = make_example(cfg, "test", 3)
        self.assertEqual(ex["input_ids"].shape, (512,))
        self.assertEqual(ex["labels"].shape, (512,))
        self.assertEqual(int(ex["loss_mask"].sum()), 32)
        self.assertEqual(ex["query_groups"].shape, (8, 4))
        self.assertEqual(ex["source_digit_positions"].shape, (8, 4))
        self.assertEqual(ex["source_record_spans"].shape, (8, 2))
        self.assertIn(ex["phase"], range(4))
        np.testing.assert_array_equal(ex["labels"][ex["query_groups"]],
                                     ex["labels"][ex["query_groups"]])
        source_tokens = ex["input_ids"][ex["source_digit_positions"]]
        np.testing.assert_array_equal(source_tokens, ex["labels"][ex["query_groups"]])

    def test_all_examples_respect_contract(self):
        cfg = SyntheticConfig()
        for example_id in range(100):
            ex = make_example(cfg, "train", example_id)
            self.assertTrue(np.all(ex["input_ids"] >= 0))
            self.assertTrue(np.all(ex["input_ids"] < 1024))


if __name__ == "__main__":
    unittest.main()
