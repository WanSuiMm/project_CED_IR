# CED IR Research Project

**Current scope:** route 2 asks whether MLA content can be stored at half the
token rate while preserving one RoPE anchor per token. The new
[`mla_pair/README.md`](mla_pair/README.md) now contains the reference operator,
a high-fidelity Qwen3 split A that passed its small NLL/cache gate, and the
matched B pilot code and result. In the 98,304-token/arm pilot, B missed the
predeclared NLL screen despite its 31.25% reference-cache reduction; no speedup
or broad route-2 capacity failure is claimed. A bounded headwise/full-coadapt
B2 follow-up is frozen in [`mla_pair/PROTOCOL_B2.md`](mla_pair/PROTOCOL_B2.md).
Earlier synthetic,
oracle, CED migration, and direct standard-KV A/B branches are frozen history.

The latest completed standard-KV direct pilot found B slightly better on
held-out NLL but no robust remote-context gain or decode speed measurement.
It is not evidence for the new MLA route. See
[`real_language_ab/RESULTS_STANDARD_KV_v0.2.md`](real_language_ab/RESULTS_STANDARD_KV_v0.2.md).

This repository contains one research project on whether token-level address
records can be replaced by smaller persistent interfaces without losing the
computation a decoder needs. It contains the original synthetic length-width
wind tunnel and the subsequent frozen-Qwen real-model attention-operator audit.

## Start here

For a fast technical review—especially through a GitHub-connected language
model—read files in this order:

1. [`GPT_HANDOFF.md`](GPT_HANDOFF.md): latest incremental review range and
   changed evidence
2. [`mla_pair/README.md`](mla_pair/README.md): route-2 code, qualified A,
   and the negative small B pilot boundary
3. [`real_language_ab/RESULTS_STANDARD_KV_v0.2.md`](real_language_ab/RESULTS_STANDARD_KV_v0.2.md):
   latest completed A/B evidence (different architecture)
4. [`GPT_CONTEXT.md`](GPT_CONTEXT.md): compact historical context, claim boundary and
   file-routing instructions
5. [`RESULTS.md`](RESULTS.md): canonical aggregate results and verdicts across
   all three evidence stages
6. [`ARCHITECTURE.md`](ARCHITECTURE.md): equations, invariants and code map
7. [`protocol.md`](protocol.md) and [`configs/g1.json`](configs/g1.json): frozen
   experimental contract
8. [`src/ced_ir/model.py`](src/ced_ir/model.py) and
   [`src/ced_ir/synthetic.py`](src/ced_ir/synthetic.py): implementation
9. [`real_model_audit/README.md`](real_model_audit/README.md): Qwen3 R0/R1
   oracle audit, code, and evidence
10. [`tests/`](tests) and [`real_model_audit/tests/`](real_model_audit/tests):
   executable correctness claims

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

The real-model branch audits the same underlying hypothesis directly in frozen
Qwen3-0.6B attention. The final R1b follow-up selected old regions using a
disjoint calibration span and added a non-identity `8->8` optimizer-recovery
control. Exact reconstruction qualified, but learned `8->8` recovery failed on
held-out queries, and median test attention mass was only 0.391%. The formal
verdict is `ORACLE_OPTIMIZER_UNQUALIFIED`; the local operator-compression branch
is stopped. See [`real_model_audit/RESULTS.md`](real_model_audit/RESULTS.md).

The latest stage replaced the invalid 1%-gate warm start with a
function-preserving Qwen-to-CED homotopy. Native Qwen is recovered at the start,
but one shared K/V bridge remains poor on held-out layers/examples and the
fully migrated rollout collapses (`5.8122` LM KL, 3.1% top-1 agreement). The
larger migration and shorter/wider comparison were not run. See
[`migration_v02/RESULTS.md`](migration_v02/RESULTS.md).

A subsequent directly supervised oracle-block probe changes the mechanism
interpretation: the rank-16 bilinear gate family decodes the physical source
slot at essentially 100% held-out accuracy when given the correct block. Thus
the slot label is decodable. A matched record-permutation control then showed
that deliberately mismatched queries and blocks retain 100% accuracy: fixed
record formatting supplies a sufficient parity shortcut. The probe does not
establish that B learned a query-specific source-address relation.

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
  decodability probe and matched record-permutation control
- `tests/`: causality, cache, replay, gradient, shape and gate-equivalence tests
- `real_model_audit/`: frozen-Qwen attention-operator oracle protocols,
  implementation, tests, preserved v0.1 evidence, and terminal R1b audit
- `migration/`: Qwen-to-CED G0 implementation, negative qualification evidence,
  and stopped G1 plan
- `migration_v02/`: function-preserving homotopy implementation, matched gates,
  bridge pilot, and terminal stop decision
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
The shortcut control is in
[`runs/slot_permutation_v01/PROBE.md`](runs/slot_permutation_v01/PROBE.md).
