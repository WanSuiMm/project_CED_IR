# R1b final high-mass audit protocol

## Purpose

R1b is the final existence gate for local contiguous `8 -> c` attention-
operator compression. It closes the two limitations of R0/R1 v0.1: the fixed
old region had low attention mass, and optimizer fit quality was not recorded
or independently qualified.

R1b does not train Qwen, a compiler, or an adapter. R2/R3 remain frozen.

## Frozen design

- Model: frozen `Qwen/Qwen3-0.6B-Base`.
- Data: the same four prose, four Python-code, and four deterministic structured
  examples as v0.1; sequence length 2048.
- Layers: zero-based 4, 16, and 27.
- Candidate regions: 64-token, non-overlapping, token-64-aligned regions in the
  old history, each containing eight adjacent 8-token blocks.
- Query spans, all contiguous and disjoint:
  - `Q_select = [1728, 1792)`;
  - `Q_fit = [1856, 1920)`;
  - `Q_test = [1984, 2048)`.
- Only `Q_select` may choose the candidate region. The region maximizing mean
  true causal attention mass is then frozen for fit and test.
- Oracle conditions:
  - `8 -> 4`, five restarts, 500 steps;
  - `8 -> 8` exact-start preservation control;
  - `8 -> 8` non-identity pair-mean-split recovery, five restarts, 500 steps.
- GQA, post-norm/post-RoPE Q/K, log-mass bias, and block-mass-weighted operator
  loss are unchanged from v0.1.

Selection mass, fit mass, and test mass are all recorded. Fit and test each
report log-Z RMSE and conditional-value relative error. Raw and row-centered
score effective rank are descriptive only.

## Independent unit

One source sequence at one layer. Heads, blocks, queries, and optimizer
restarts are repeated measurements.

## Ordered decision rule

1. `INVALID_IMPLEMENTATION` if manual exact attention exceeds 1% or the
   exact-start `8 -> 8` control exceeds `1e-4` on any fit/test operator metric.
2. `ORACLE_OPTIMIZER_UNQUALIFIED` unless non-identity `8 -> 8` recovery has:
   median fit and test log-Z/conditional-value errors <= 0.02, and at least 75%
   of units have both test errors <= 0.05.
3. `NO_HIGH_MASS_CANDIDATE` if median selected-region mass on `Q_select` is
   below 2%.
4. `NO_STABLE_HIGH_MASS_TARGET` if median mass on `Q_test` is below 2% or fewer
   than 50% of units retain at least 2% test mass.
5. `PASS_R1B_ORACLE_8_TO_4` only if `8 -> 4` has median test log-Z and
   conditional-value errors <= 0.10 and at least 75% of units meet both limits.
6. Otherwise: `STOP_LOCAL_OPERATOR_COMPRESSION`.

Only `PASS_R1B_ORACLE_8_TO_4` authorizes discussion of R2. Every other terminal
status keeps compiler training and adaptation frozen.
