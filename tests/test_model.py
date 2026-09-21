import unittest
from dataclasses import replace

import torch

from ced_ir.model import CEDConfig, CEDIRModel


def tiny(block_size=1, width_multiplier=1):
    return CEDConfig(vocab_size=128, d_model=32, heads=4, encoder_layers=2,
                     decoder_layers=2, local_window=4, mlp_intermediate=48,
                     block_size=block_size, width_multiplier=width_multiplier)


class ModelCorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_shapes_and_wide_storage(self):
        for r, s in ((1, 1), (2, 2), (4, 4)):
            model = CEDIRModel(tiny(r, s)).double()
            ids = torch.randint(0, 128, (2, 9))
            logits, memory = model(ids, return_memory=True)
            self.assertEqual(logits.shape, (2, 9, 128))
            self.assertEqual(memory.packed.shape, (2, 9 // r, s * 32))
            self.assertEqual(memory.values.shape[-1], s * 8)

    def test_future_edits_do_not_change_past_logits(self):
        for r, s in ((1, 1), (2, 2), (4, 4)):
            model = CEDIRModel(tiny(r, s)).double().eval()
            ids = torch.randint(0, 128, (1, 17))
            changed = ids.clone()
            changed[:, 10:] = torch.randint(0, 128, changed[:, 10:].shape)
            with torch.no_grad():
                first = model(ids)
                second = model(changed)
            torch.testing.assert_close(first[:, :10], second[:, :10], atol=1e-9, rtol=1e-9)

    def test_future_embedding_gradient_is_zero(self):
        model = CEDIRModel(tiny(2, 2)).double().eval()
        ids = torch.randint(0, 128, (1, 9))
        captured = {}
        def hook(_module, _args, output):
            output.retain_grad()
            captured["embedding"] = output
        handle = model.embedding.register_forward_hook(hook)
        logits = model(ids)
        logits[:, 4].sum().backward()
        handle.remove()
        grad = captured["embedding"].grad
        self.assertEqual(float(grad[:, 5:].abs().max()), 0.0)

    def test_global_ablation_is_effective_path_cut(self):
        model = CEDIRModel(tiny(2, 2)).double().eval()
        ids = torch.randint(0, 128, (1, 9))
        with torch.no_grad():
            regular = model(ids)
            ablated = model(ids, ablate_global=True)
        self.assertGreater(float((regular - ablated).abs().max()), 1e-10)

    def test_common_parameters_match_across_variants(self):
        a = CEDIRModel(tiny(1, 1))
        b = CEDIRModel(tiny(2, 2))
        b_params = dict(b.named_parameters())
        compared = 0
        for name, parameter in a.named_parameters():
            other = b_params.get(name)
            if other is not None and other.shape == parameter.shape:
                torch.testing.assert_close(parameter, other)
                compared += 1
        self.assertGreater(compared, 10)

    def test_prefix_recompute_matches_full(self):
        model = CEDIRModel(tiny(2, 2)).double().eval()
        ids = torch.randint(0, 128, (1, 9))
        with torch.no_grad():
            full = model(ids)
            for end in range(1, ids.shape[1] + 1):
                prefix = model(ids[:, :end])
                torch.testing.assert_close(full[:, end - 1], prefix[:, -1], atol=1e-9, rtol=1e-9)

    def test_streaming_cache_matches_full_every_prefix(self):
        for r, s in ((1, 1), (2, 2), (4, 4)):
            model = CEDIRModel(tiny(r, s)).double().eval()
            ids = torch.randint(0, 128, (2, 17))
            state = model.init_stream_state(ids.shape[0], ids.device, model.embedding.weight.dtype)
            streamed = []
            with torch.no_grad():
                for position in range(ids.shape[1]):
                    streamed.append(model.stream_step(ids[:, position], state))
                streamed_logits = torch.stack(streamed, dim=1)
                full = model(ids)
            torch.testing.assert_close(streamed_logits, full, atol=1e-9, rtol=1e-9)
            self.assertLess(state.tail.shape[1], r)

    def test_exact_replay_prefill_and_continuation(self):
        for r, s in ((1, 1), (2, 2), (4, 4)):
            model = CEDIRModel(tiny(r, s)).double().eval()
            prefix = torch.randint(0, 128, (2, 17))
            continuation = torch.randint(0, 128, (2,))
            with torch.no_grad():
                replay_logits, state = model.prefill_exact_replay(prefix)
                full_prefix = model(prefix)
                next_logits = model.stream_step(continuation, state)
                full_next = model(torch.cat((prefix, continuation[:, None]), dim=1))[:, -1]
            torch.testing.assert_close(replay_logits[:, -1], full_prefix[:, -1], atol=1e-9, rtol=1e-9)
            torch.testing.assert_close(next_logits, full_next, atol=1e-9, rtol=1e-9)

    def test_uniform_slot_gate_reproduces_b_and_has_gradient(self):
        base_cfg = tiny(2, 2)
        gate_cfg = replace(base_cfg, slot_gate_dim=8)
        base = CEDIRModel(base_cfg).float().eval()
        gated = CEDIRModel(gate_cfg).float().eval()
        incompat = gated.load_state_dict(base.state_dict(), strict=False)
        self.assertEqual(len(incompat.missing_keys), 4)
        self.assertFalse(incompat.unexpected_keys)
        ids = torch.randint(0, 128, (2, 17))
        with torch.no_grad():
            expected = base(ids)
            actual = gated(ids)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        gated.train()
        objective = gated(ids)[:, -1].square().mean()
        objective.backward()
        slot_gradients = [p.grad for name, p in gated.named_parameters()
                          if "gate_slot_proj" in name]
        self.assertTrue(all(gradient is not None for gradient in slot_gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in slot_gradients), 0.0)

    def test_routing_diagnostics_are_normalized_and_non_mutating(self):
        cfg = replace(tiny(2, 2), slot_gate_dim=8)
        model = CEDIRModel(cfg).float().eval()
        ids = torch.randint(0, 128, (2, 17))
        captured = {}

        def inspect(_module, args):
            x, memory, positions = args[:3]
            coarse, gate = _module.routing_diagnostics(x, memory, positions)
            captured["coarse"] = coarse
            captured["gate"] = gate

        handle = model.decoder[0].global_read.register_forward_pre_hook(inspect)
        with torch.no_grad():
            expected = model(ids)
        handle.remove()
        with torch.no_grad():
            actual = model(ids)
        torch.testing.assert_close(actual, expected)
        coarse_sum = captured["coarse"].sum(dim=-1)
        expected_sum = torch.ones_like(coarse_sum)
        expected_sum[:, :, 0] = 0.0  # no pack-2 block has ended at position zero
        torch.testing.assert_close(coarse_sum, expected_sum)
        torch.testing.assert_close(captured["gate"].sum(dim=-1),
                                   torch.ones_like(captured["gate"].sum(dim=-1)))


if __name__ == "__main__":
    unittest.main()
