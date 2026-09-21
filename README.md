# CED IR Length-Width Probe

This repository contains a bounded causal-language-model probe of whether a
token-level long-range interface of shape `N x d` can be replaced by a packed
interface of shape `N/2 x 2d` without retaining a hidden token-level global
cache.

The project separates three questions:

1. Is packing information-preserving?
2. Can the decoder address and use the packed information?
3. Does the resulting shape improve measured system costs?

The current evidence answers the second question negatively for the tested
reader. The token-level baseline reached 99.71% four-digit exact retrieval,
whereas the packed-wide reader reached 7.95%. Its NLL was close to `ln(2)`, but
an identity-preserving low-rank two-way slot gate trained on the frozen packed
checkpoint did not recover performance (8.08%).

This result does not establish that all packed representations fail. It rejects
the tested one-key-per-block reader and the tested frozen-backbone low-rank slot
gate. See `runs/g1_b_learned_slot_c16_u0512_20260921/VERDICT.md` for the formal
claim boundary.

## Layout

- `protocol.md`: frozen G1 protocol and execution amendment
- `configs/g1.json`: experiment configuration
- `src/ced_ir/`: model and deterministic synthetic data generator
- `scripts/`: G1 and slot-gate training entry points
- `tests/`: causality, cache, replay, gradient, shape and gate-equivalence tests
- `runs/`: compact result summaries and per-sample evidence; checkpoints and
  machine-specific receipts are intentionally excluded

## Tests

```bash
export PYTHONPATH=src
python -m unittest discover -s tests -p 'test_*.py' -v
```

The implementation uses PyTorch. Formal GPU runs used BF16 autocast with FP32
parameters and optimizer state. See `protocol.md` for the complete frozen
conditions and stopping rules.
