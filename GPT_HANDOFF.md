# GPT incremental-review handoff

Use this file when the repository has already been reviewed at the base commit.
It is an incremental handoff, not a replacement for `GPT_CONTEXT.md`.

## Review range

- Base commit: `930fb237c85f003228b854da62196385cd732253`
- Evidence head: `6e2e2ecaf7c1201cfd2b5550068e5f3362edbe12`
- Update: supervised oracle-block physical-slot probe
- Scientific status: complete

The commit that adds this handoff may be newer than the evidence head. That
metadata-only commit does not change the experiment or its conclusions.

## Read in this order

1. `runs/slot_probe_v01/PROBE.md` -- concise new result
2. the `Supervised oracle-block physical-slot probe` section of `RESULTS.md`
3. `slot_probe_protocol.md` -- frozen controls and decision thresholds
4. `scripts/probe_physical_slot.py` and `src/ced_ir/probes.py` -- implementation
5. `runs/slot_probe_v01/summary.json` -- validation, per-head, and per-phase data

Do not reread the old training pipeline or raw per-example JSONL unless this
incremental review finds a concrete dependency that requires it.

## What changed

- Added q-only, z-only, joint-linear, and rank-16 bilinear slot probes.
- Froze terminal B and supplied the known correct packed block for each query
  digit, then trained probes directly on physical-slot cross-entropy.
- Used fresh deterministic training examples and the independent frozen
  validation/test splits. Test was evaluated once at the terminal update.
- Updated mechanism wording to separate address decodability from address
  usability by the original coarse-attention/LM-loss/frozen-reader path.

## What did not change

- B, the failed LM-trained gate, and every earlier checkpoint remain unchanged.
- G1 task metrics, addressing-audit metrics, data semantics, and frozen test
  examples remain unchanged.
- This is a representation probe, not an LM repair, architecture result,
  systems result, or multi-seed training claim.

## New evidence

Four-head macro held-out physical-slot accuracy:

| layer | q-only | z-only | q+z linear | rank-16 bilinear |
|---:|---:|---:|---:|---:|
| 0 | 0.4990 | 0.6264 | 0.9901 | 1.0000 |
| 1 | 0.8063 | 0.6264 | 0.9916 | 1.0000 |

- Validation and test agree to within 0.001.
- Bilinear accuracy is essentially 100% in every head and phase.
- Every probe family reaches 1.0 on the separable implementation control.
- Formal verdict: `GATE_FAMILY_CLEANLY_DECODABLE`.

## Current interpretation

Physical left/right identity is cleanly present in the frozen query--correct-
block geometry. Layer 0 is the decisive control: q-only is at chance, while
adding the oracle block makes both joint-linear and bilinear decoding nearly
perfect. Therefore the failed LM-trained gate cannot be interpreted as evidence
that B lacks the address relation.

The remaining task failure lies between decodability and use: coarse block
weight dilution, indirect LM credit assignment, and/or the frozen output reader
can prevent a decodable relation from improving LM predictions. The result does
not show which of those mechanisms dominates and does not itself repair B.

## Questions for the reviewer

1. Are feature capture, oracle-block selection, split separation, controls, and
   probe losses implemented correctly?
2. Can the near-perfect bilinear result be explained by leakage not covered by
   the layer-0 q-only and phase-stratified controls?
3. Does the evidence justify `GATE_FAMILY_CLEANLY_DECODABLE` while preserving
   the stated non-causal/non-repair claim boundary?
4. What single smallest intervention would distinguish coarse dilution from
   frozen-reader incompatibility without training a new architecture?
