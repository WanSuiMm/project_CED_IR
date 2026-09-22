# Context for GPT and technical reviewers

## One-sentence question

Can token-level long-range address records be replaced by a smaller persistent
interface while preserving the computation needed by synthetic and real
pretrained decoders?

## What this project is—and is not

This is one research project with three evidence stages: a synthetic
architecture wind tunnel, a frozen-Qwen3 attention-operator audit under
`real_model_audit/`, and a stopped Qwen-to-CED implementation qualification
under `migration/`. None is a systems-speedup claim or a completed
post-training migration. The latest stage failed G0 before formal training.

## Variants

- `A_TOKEN`: one global record per source token, shape `N x d`.
- `B_PACK2_WIDE2`: concatenate two adjacent causal encoder states into one
  persistent record, shape `N/2 x 2d`. One coarse key addresses the whole
  record; the value remains wide until the decoder-layer output projection.
- `B_LEARNED_SLOT`: start from the completed B checkpoint and add a 16-dimensional
  query-conditioned two-way gate. The gate is uniform at initialization and
  reproduces B exactly. Only its 49,152 new parameters are trained.

The conditional `C_PACK2_NARROW1` diagnostic was not run because B already
failed the primary representation test.

## Canonical result

- A: 99.710% four-digit exact match; test NLL 0.00373.
- B: 7.953% exact match; test NLL 0.69721.
- B plus frozen-backbone slot gate: 8.081% exact match; test NLL 0.69398.
- Global-IR ablation reduces test exact match to zero in every trained variant.

The baseline is qualified and B is degraded far outside the pre-registered
two-percentage-point tolerance. The low-rank gate does not repair B.

## Correct interpretation

Packing is algebraically information-preserving when `r=s=2`, but this does
not imply that the tested reader can address the information. B's NLL near
`ln(2)` is a mechanism clue, not proof that the model merely lacks one explicit
address bit. The failed gate weakens that simple explanation.

The evidence rejects:

1. the tested one-key-per-block packed-wide reader as a general replacement for
   token-level exact retrieval in this setting; and
2. the claim that a small bilinear two-way gate trained alone on the frozen B
   representation is sufficient to repair it.

The evidence does not reject every packed representation, every hierarchical
reader, jointly trained address mechanisms, or task-dependent non-uniform
resolution.

## Formal statuses

- `correctness_status = PASS`
- `baseline_status = QUALIFIED`
- `representation_status = DEGRADED_IN_THIS_SETTING`
- `width_compensation_status = UNTESTED`
- `systems_status = NOT_MEASURED`
- gate diagnostic: `FROZEN_B_LOW_RANK_SLOT_GATE_FAIL`
- zero-training follow-up: `ADDRESSING_AUDIT_COMPLETE`
- supervised oracle-block probe: `GATE_FAMILY_CLEANLY_DECODABLE`
- address-relation status: `STRUCTURAL_SHORTCUT_DOMINANT`
- real-model R0/R1 v0.1 status: `INCONCLUSIVE_LOW_MASS`
- real-model final R1b status: `ORACLE_OPTIMIZER_UNQUALIFIED`
- real-model compiler/adaptation status: `FROZEN_NOT_AUTHORIZED`
- Qwen-to-CED G0 status: `INVALID_CED_IMPLEMENTATION`
- Qwen-to-CED G1 training status: `NOT_RUN`

## Code routing

- Model and attention logic: `src/ced_ir/model.py`
  - `CEDIRModel`: complete model
  - `IRCompiler`: causal block packing and shared K/V construction
  - `GlobalRead`: coarse attention and optional low-rank slot gate
  - `stream_step` / `prefill_exact_replay`: cache and replay paths
- Deterministic task generator: `src/ced_ir/synthetic.py`
- A/B training and evaluation: `scripts/train_g1.py`
- Frozen-backbone gate training: `scripts/train_gate.py`
- Executable invariants: `tests/test_model.py` and `tests/test_data.py`
- Aggregate evidence: `RESULTS.md` and each run's `summary.json`
- Mechanistic evidence: `runs/address_audit_v01/AUDIT.md` and `audit.json`
- Supervised decodability evidence: `runs/slot_probe_v01/PROBE.md` and
  `summary.json`
- Matched permutation control: `runs/slot_permutation_v01/PROBE.md` and
  `summary.json`
- Real-model final interpretation: `real_model_audit/RESULTS.md`
- Final frozen protocol: `real_model_audit/R1B_PROTOCOL.md`
- Final runner and oracle: `real_model_audit/scripts/run_r1b_gate.py` and
  `real_model_audit/src/aoc/oracle.py`
- Final aggregate: `real_model_audit/runs/r1b_high_mass_v01/summary.json`
- Preserved v0.1 protocol/run: `real_model_audit/PROTOCOL.md` and
  `real_model_audit/runs/r01_qwen3_06b_v03/summary.json`
- Migration stop decision: `migration/RESULTS.md`
- Migration implementation and runner: `migration/src/ced_migration/modeling.py`
  and `migration/scripts/qualify_g0.py`
- Migration formal G0 evidence:
  `migration/runs/g0_qwen06b_bf16_gated_v01/summary.json`
- Raw per-example evidence: `test_samples*.jsonl`

## Recommended review behavior

Read `RESULTS.md` before inspecting raw JSONL. Keep mathematical cache/FLOP
ratios separate from measured training memory and throughput. Do not infer
single-seed training uncertainty from the per-example test set, and do not turn
the observed `ln(2)` signature into a stronger causal claim than the gate
experiment supports.

The zero-training audit narrows the remaining mechanism. The learned compiler
has only 1.315% cross-half Frobenius energy, so wholesale compiler entanglement
is not the main explanation. Decoder layer 0 places the correct digit block in
its top four about 67.4% of the time, while layer 1 does so only about 4.0% of
the time. On the known correct block, the learned gate assigns the physical
slot mean probability 0.503 and has 51.9% pooled accuracy. Phase-conditioned
exact match remains approximately 8% in every phase. Therefore the evidence
does not support the claim that a clean left/right switch was already usable by
the LM-trained gate and frozen reader; addressing quality also differs sharply
by decoder layer.

Direct supervision separates decodability from usability.
On the independent test split, the rank-16 bilinear family reaches essentially
100% physical-slot accuracy in both layers. In layer 0, q-only remains at 49.9%,
while q+z linear reaches 99.0%, so the result is not explained by a query-only
packing rule. Physical slot is cleanly decodable from the frozen query--correct-
block features, but that alone does not identify a query-specific relation.

The matched record-permutation control exploits the fact that physical slot is
deterministically `(phase + digit ordinal) mod 2`. For layer 0, digit 0,
bilinear accuracy remains 1.0 for matched pairs, query-cyclic mismatches,
block-cyclic mismatches, and pair-preserving cyclic controls. All other
ordinals also remain above 0.999 when query identity and source-record identity
are deliberately mismatched. Formatting parity is sufficient to saturate the
probe. `GATE_FAMILY_CLEANLY_DECODABLE` is therefore a label-decoding metric,
not evidence that B learned semantic query-to-record addressing.

The qualified Qwen3 audit supplies a separate real-model check of the same
research hypothesis. Its exact-attention qualification error is at most 0.303%.
For `8->4`, median held-out log-mass RMSE is 0.534 and conditional-value relative
error is 0.278; zero of 36 sequence-layer units meets both 0.10 limits. Yet the
fixed replaced region carries only 0.568% median attention mass, so tiny global
KL and delta NLL are not positive compression evidence. Low `QK^T` effective
rank (median 1.339) likewise does not imply held-out operator preservation.

The terminal R1b follow-up selected candidate regions on `Q_select`, then used
disjoint `Q_fit` and `Q_test` spans. Exact-start `8->8` remained exact, but a
non-identity `8->8` start reached median test errors 0.569 (log-Z) and 0.206
(conditional value), with 0/36 units meeting both 0.05 limits. Thus the learned
oracle is not qualified to support an `8->4` capacity conclusion. Separately,
the selected region's median mass fell from 1.089% on selection to 0.391% on
test, and only 3/36 units retained 2% mass. The branch stops without R2/R3.
