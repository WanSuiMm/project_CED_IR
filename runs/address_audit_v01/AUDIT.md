# Addressing audit

Zero-training diagnostics on the frozen 4,096-example G1 test split.

## Layer-mean routing

| model | layer | digit top-1 | digit top-4 | digit mass | record mass | slot accuracy | slot probability |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_PACK2_WIDE2 | 0 | 0.2449 | 0.6742 | 0.2531 | 0.2744 | n/a | n/a |
| B_PACK2_WIDE2 | 1 | 0.0112 | 0.0399 | 0.0083 | 0.0202 | n/a | n/a |
| B_LEARNED_SLOT | 0 | 0.2449 | 0.6742 | 0.2531 | 0.2744 | 0.5346 | 0.5029 |
| B_LEARNED_SLOT | 1 | 0.0112 | 0.0397 | 0.0083 | 0.0201 | 0.5040 | 0.5023 |

## Phase-stratified task performance

| model | phase | NLL | four-digit exact |
|---|---:|---:|---:|
| B_PACK2_WIDE2 | 0 | 0.690670 | 0.077659 |
| B_PACK2_WIDE2 | 1 | 0.702680 | 0.083374 |
| B_PACK2_WIDE2 | 2 | 0.688053 | 0.079490 |
| B_PACK2_WIDE2 | 3 | 0.707015 | 0.077635 |
| B_LEARNED_SLOT | 0 | 0.686322 | 0.081759 |
| B_LEARNED_SLOT | 1 | 0.699820 | 0.080796 |
| B_LEARNED_SLOT | 2 | 0.685599 | 0.079490 |
| B_LEARNED_SLOT | 3 | 0.703747 | 0.081197 |

## Compiler geometry

- cross-mixing ratio: `0.013150`
- relative distance from identity: `0.165524`
- spectral condition number: `2.591676`

Per-layer and per-head values are retained in `audit.json`.
