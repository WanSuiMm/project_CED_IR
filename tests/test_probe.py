import unittest

import torch
import torch.nn.functional as F

from ced_ir.probes import SlotProbeSuite, permute_record_pairs


class SlotProbeTests(unittest.TestCase):
    def test_record_permutation_is_deranged_and_ordinal_preserving(self):
        records, digits = 4, 3
        query = torch.arange(records * digits).view(1, records * digits, 1).float()
        slots = (100 + torch.arange(records * digits)).view(
            1, records * digits, 1, 1, 1).float()
        target = (torch.arange(digits) % 2).view(1, 1, digits).expand(
            1, records, digits).reshape(1, -1)
        q_cycle, z_same, y = permute_record_pairs(
            query, slots, target, records, digits, "query_cyclic")
        self.assertTrue(torch.equal(z_same, slots))
        self.assertTrue(torch.equal(y, target))
        self.assertTrue(torch.equal(q_cycle.view(1, records, digits, 1)[:, 0],
                                    query.view(1, records, digits, 1)[:, -1]))
        self.assertFalse(bool((q_cycle == query).any()))
        q_pair, z_pair, _ = permute_record_pairs(
            query, slots, target, records, digits, "paired_cyclic")
        q_pair = q_pair.view(1, records * digits, 1, 1, 1)
        torch.testing.assert_close(z_pair - q_pair, torch.full_like(z_pair, 100.0))

    def test_cloned_inference_features_support_probe_autograd(self):
        with torch.inference_mode():
            inference_query = torch.randn(2, 3, 16)
            inference_slots = torch.randn(2, 3, 2, 2, 8)
        query = inference_query.clone()
        slots = inference_slots.clone()
        suite = SlotProbeSuite(16, 2, gate_dim=4)
        loss = sum(logits.square().mean() for logits in suite(query, slots).values())
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in suite.parameters()))

    def test_shapes_gradients_and_learnable_control(self):
        torch.manual_seed(3)
        batch, count, d_model, heads = 128, 3, 16, 2
        head_width = d_model // heads
        labels = torch.randint(0, 2, (batch, count))
        sign = labels.float().mul(2).sub(1)
        query = torch.randn(batch, count, d_model) * 0.03
        query[..., 0] = sign
        query[..., 1] = 1.0
        slots = torch.randn(batch, count, heads, 2, head_width) * 0.03
        target_index = labels[:, :, None, None, None].expand(-1, -1, heads, 1, 1)
        other_index = (1 - labels)[:, :, None, None, None].expand(-1, -1, heads, 1, 1)
        slots.scatter_(3, target_index.expand(-1, -1, -1, -1, head_width), 1.0)
        slots.scatter_(3, other_index.expand(-1, -1, -1, -1, head_width), -1.0)
        suite = SlotProbeSuite(d_model, heads, gate_dim=4)
        optimizer = torch.optim.Adam(suite.parameters(), lr=0.05)
        target = labels[:, :, None].expand(-1, -1, heads)
        for _ in range(80):
            outputs = suite(query, slots)
            loss = sum(F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
                       for logits in outputs.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        outputs = suite(query, slots)
        for name, logits in outputs.items():
            accuracy = float((logits.argmax(dim=-1) == target).float().mean())
            self.assertGreater(accuracy, 0.98, name)


if __name__ == "__main__":
    unittest.main()
