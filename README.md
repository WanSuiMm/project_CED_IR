# CED IR Length-Width Probe

This repository is a self-contained, bounded causal-language-model probe of
whether a token-level long-range interface of shape `N x d` can be replaced by
a packed interface of shape `N/2 x 2d` without retaining a hidden token-level
global cache.

## Start here

For a fast technical review—especially through a GitHub-connected language
model—read files in this order:

1. [`GPT_HANDOFF.md`](GPT_HANDOFF.md): latest incremental review range and
   changed evidence
2. [`GPT_CONTEXT.md`](GPT_CONTEXT.md): compact task context, claim boundary and
   file-routing instructions
3. [`RESULTS.md`](RESULTS.md): canonical aggregate results and verdicts
4. [`ARCHITECTURE.md`](ARCHITECTURE.md): equations, invariants and code map
5. [`protocol.md`](protocol.md) and [`configs/g1.json`](configs/g1.json): frozen
   experimental contract
6. [`src/ced_ir/model.py`](src/ced_ir/model.py) and
   [`src/ced_ir/synthetic.py`](src/ced_ir/synthetic.py): implementation
7. [`tests/`](tests): executable correctness claims

The large `test_samples*.jsonl` files are raw evidence. They should not be the
first files used to understand the project.

The project separates three questions:

1. Is packing information-preserving?
2. Can the decoder address and use the packed information?
3. Does the resulting shape improve measured system costs?

The current evidence answers the second question negatively for the tested
reader. The token-level baseline reached 99.71% four-digit exact retrieval,
whereas the packed-wide reader reached 7.95%. Its NLL was close to `ln(2)`, but
an identity-preserving low-rank two-way slot gate trained on the frozen packed
checkpoint did not recover performance (8.08%). Exact numbers and formal status
fields are centralized in [`RESULTS.md`](RESULTS.md).

A subsequent directly supervised oracle-block probe changes the mechanism
interpretation: the rank-16 bilinear gate family decodes the physical source
slot at essentially 100% held-out accuracy when given the correct block. Thus
the slot relation exists in frozen B, but the original coarse-attention/LM-loss/
frozen-reader path does not learn to use it.

This result does not establish that all packed representations fail. It rejects
the tested one-key-per-block reader and the tested frozen-backbone low-rank slot
gate. See `runs/g1_b_learned_slot_c16_u0512_20260921/VERDICT.md` for the formal
claim boundary.

## Layout

- `GPT_HANDOFF.md`: latest incremental-review packet and commit range
- `GPT_CONTEXT.md`: compact entry point for connected LLMs and reviewers
- `RESULTS.md`: canonical aggregate evidence and claim boundary
- `ARCHITECTURE.md`: model equations, invariants and source-code map
- `protocol.md`: frozen G1 protocol and execution amendment
- `configs/g1.json`: experiment configuration
- `src/ced_ir/`: model and deterministic synthetic data generator
- `scripts/`: G1 and slot-gate training entry points
- `scripts/audit_addressing.py`: zero-training routing, slot, compiler, and
  phase audit
- `scripts/probe_physical_slot.py`: directly supervised frozen-backbone slot
  decodability probe
- `tests/`: causality, cache, replay, gradient, shape and gate-equivalence tests
- `runs/`: compact result summaries and per-sample evidence; checkpoints and
  machine-specific receipts are intentionally excluded

## Tests

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m unittest discover -s tests -p 'test_*.py' -v
```

The implementation uses PyTorch. Formal GPU runs used BF16 autocast with FP32
parameters and optimizer state. See `protocol.md` for the complete frozen
conditions and stopping rules.

The completed addressing audit is summarized in
[`runs/address_audit_v01/AUDIT.md`](runs/address_audit_v01/AUDIT.md). Its full
per-layer and per-head measurements are in `audit.json` in the same directory.
The supervised follow-up is in
[`runs/slot_probe_v01/PROBE.md`](runs/slot_probe_v01/PROBE.md).
