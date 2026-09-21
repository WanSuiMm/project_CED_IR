# GPT incremental-review handoff

This is an incremental review packet for the **same CED IR project**. The
final R1b frozen-Qwen audit is inside `real_model_audit/`; it is not a separate
project. The synthetic branch and all earlier evidence remain unchanged.

## Review range

- Base commit: `15d61a7413ef1fba646763e877a1b34fcb33fd86`
- Evidence head: `b1fbab27a83502d32905e61641141a32261fbffb`
- Update: final high-mass R1b attention-operator audit
- Formal verdict: `ORACLE_OPTIMIZER_UNQUALIFIED`

The later commit that updates this handoff is metadata only. Review the fixed
range above; do not reread the unchanged synthetic training pipeline or the
preserved v0.1 Qwen evidence.

## Minimal reading order

1. `real_model_audit/RESULTS.md` — terminal interpretation and claim boundary.
2. `real_model_audit/R1B_PROTOCOL.md` — frozen design and ordered decision rule.
3. `real_model_audit/runs/r1b_high_mass_v01/RESULTS.md` — generated aggregate.
4. `real_model_audit/scripts/run_r1b_gate.py` — selection, fit/test split,
   controls, intervention, and aggregation.
5. `real_model_audit/src/aoc/oracle.py` — non-identity recovery initialization
   and optimizer.
6. `real_model_audit/runs/r1b_high_mass_v01/summary.json` only for per-unit
   verification; do not open it first.

## What changed

- Added three disjoint query spans: `Q_select`, `Q_fit`, and `Q_test`.
- Selected the highest-mass eligible old 64-token region using only
  `Q_select`; the selected region was then frozen for fit and test.
- Added separate fit and test operator metrics.
- Added an exact-start `8 -> 8` preservation control.
- Added a learned non-identity `8 -> 8` recovery control initialized from
  duplicated adjacent-pair means. Four noisy restarts break the duplicated
  symmetry; one deterministic restart is retained.
- Retested `8 -> 4` only; no compiler, adapter, or Qwen parameter was trained.

## New decision-relevant evidence

| Metric | Result | Frozen requirement |
|---|---:|---:|
| Manual-exact maximum mismatch | 0.3031% | <= 1% |
| Exact-start `8 -> 8` fit/test error | 0 | <= 1e-4 |
| Recovery `8 -> 8` fit log-Z / mu median | 0.0199 / 0.0429 | both <= 0.02 |
| Recovery `8 -> 8` test log-Z / mu median | 0.5688 / 0.2062 | both <= 0.02 |
| Recovery units meeting both 0.05 test limits | 0 / 36 | >= 27 / 36 |
| Selected-region mass on `Q_select` | 1.0885% median | >= 2% |
| Selected-region mass on `Q_test` | 0.3914% median | >= 2% |
| Units retaining at least 2% test mass | 3 / 36 | >= 18 / 36 |
| `8 -> 4` test log-Z / mu median | 0.6152 / 0.3006 | descriptive after recovery failure |

The exact algebra and intervention implementation qualify. The learned oracle
does not recover a known-to-exist eight-record solution on held-out queries,
so it is not qualified to support an `8 -> 4` capacity conclusion. The
calibration-selected region also fails to remain high-mass, but optimizer
qualification precedes mass in the frozen decision order.

## What did not change

- All synthetic checkpoints, G1 results, addressing audit, supervised slot
  probe, and matched record-permutation control are unchanged.
- The v0.1 Qwen run remains preserved with verdict `INCONCLUSIVE_LOW_MASS`.
- No amortized compiler, Qwen adaptation, continued pretraining, kernel, or
  system-speed measurement was performed.
- The result is not a universal impossibility theorem for learned interfaces.

## Reviewer questions

1. Is candidate selection isolated to `Q_select`, with no `Q_fit` or `Q_test`
   leakage into region choice?
2. Is the pair-mean `8 -> 8` initialization genuinely non-identity, and do the
   noisy restarts adequately break its duplicated-record symmetry?
3. Are fit/test log-Z and conditional-value errors weighted and aggregated at
   the preregistered sequence-layer independent-unit level?
4. Does `ORACLE_OPTIMIZER_UNQUALIFIED` correctly precede both mass diagnoses
   and the `8 -> 4` capacity decision?
5. Is the terminal claim properly scoped to stopping this local `8 -> c`
   formulation rather than asserting a universal impossibility result?
