# Frozen-Qwen attention-operator audit

## Final R1b verdict

`ORACLE_OPTIMIZER_UNQUALIFIED`

R1b was the final small audit permitted after v0.1. It selected the highest-mass
eligible old 64-token region using `Q_select`, fitted on a later disjoint
`Q_fit`, and evaluated once on a still later `Q_test`. The model, sources,
layers, and query layout were frozen in `R1B_PROTOCOL.md`.

The implementation controls passed. Across 36 sequence-layer units, maximum
manual-exact mismatch was `0.3031%`, and the exact-start `8 -> 8` replacement
had zero measured fit and test operator error. The non-identity `8 -> 8`
recovery control did not qualify:

| Metric | Fit median | Test median | Frozen requirement |
|---|---:|---:|---:|
| log-Z RMSE | 0.0199 | 0.5688 | <= 0.02 |
| conditional-value relative error | 0.0429 | 0.2062 | <= 0.02 |

Zero of 36 units had both recovery test errors at or below `0.05`, versus the
required 75%. The optimizer could fit much of the calibration geometry but did
not recover the known-to-exist eight-record operator on new queries. Therefore
the learned-oracle results cannot distinguish insufficient `8 -> 4` capacity
from optimizer/parameterization failure.

## High-mass selection result

The attempted loophole also did not produce a stable high-mass endpoint:

| Query split | Median selected-region attention mass |
|---|---:|
| Selection | 1.0885% |
| Fit | 0.5061% |
| Test | 0.3914% |

Only `3 / 36` units retained at least 2% test mass. Formally, the earlier
optimizer-recovery failure takes precedence in the preregistered decision
order, so `NO_HIGH_MASS_CANDIDATE` and `NO_STABLE_HIGH_MASS_TARGET` are
descriptive diagnostics rather than the formal verdict.

For completeness, `8 -> 4` had median fit errors `0.0265 / 0.1574` and test
errors `0.6152 / 0.3006` for log-Z / conditional value; zero units met both
test limits. Those numbers are not promoted to a capacity verdict because the
stronger `8 -> 8` recovery control failed first.

## Scientific decision

The local contiguous attention-operator compression branch stops here. R1b
closed neither prerequisite for R2: it found no stable high-mass target, and
the learned oracle did not recover a known exact solution out of sample. No
compiler training, Qwen adaptation, continued pretraining, kernel work, or
systems-speed claim is authorized.

This is not a universal impossibility result for learned memory interfaces. It
is a stop decision for this local `8 -> c` oracle formulation and optimizer
under the tested frozen-Qwen setting.

## Evidence

- Frozen final protocol: `R1B_PROTOCOL.md`
- Final generated aggregate: `runs/r1b_high_mass_v01/RESULTS.md`
- Sanitized per-unit evidence: `runs/r1b_high_mass_v01/summary.json`
- Final runner: `scripts/run_r1b_gate.py`
- Oracle implementation and recovery initialization: `src/aoc/oracle.py`

The earlier v0.1 result remains preserved under
`runs/r01_qwen3_06b_v03/`. Its formal verdict was `INCONCLUSIVE_LOW_MASS`;
R1b is the terminal follow-up rather than a rewrite of that evidence.
