# Qwen-to-CED homotopy migration protocol v0.2

## Purpose

This protocol replaces the invalid v0.1 warm start. The old endpoint deleted
all upper self-attention at update zero and multiplied the replacement path by
1%; it was not a small perturbation of pretrained Qwen. Its evidence remains
frozen under `migration/`.

Version 0.2 asks whether Qwen can be moved continuously to the same final
token-aligned CED endpoint. It does not test a shorter/wider interface.

## Architecture path

For each upper layer, mix attention head outputs before the original output
projection:

```text
a = (1 - alpha) * a_self + alpha * a_shared_cross
```

The original query and output projections are retained. The shared cross path
uses one separately trainable memory norm and one K/V projection pair computed
from the lower-stack output and read by all upper layers.

For each lower layer, use the same Q/K/V/O parameters with two masks:

```text
a = (1 - beta) * a_full_causal + beta * a_local_window
```

`alpha` and `beta` are externally scheduled buffers, not optimizer parameters.
At `(0, 0)` the computation is native Qwen. At `(1, 1)` the deployable state is
14 local windows plus one shared global K/V memory; upper per-layer K/V and the
lower full-history path are disabled.

## Ordered gates

### H0: morphism qualification

At `alpha=beta=0`, compare the custom homotopy path against native Qwen using
the same checkpoint, inputs, dtype, length, and SDPA backend. Compare both full
forward and cached decoding. Any structural mismatch blocks training.

### H1: endpoint qualification

At `alpha=beta=1`, check causality, full versus prefill-plus-cached continuation,
cache accounting, remote-memory dependence, backward, and save/reload. At least
one parity sequence must exceed the 64-token local window. BF16 tolerances must
be interpreted against a same-length native-Qwen numerical calibration.

### H2: bridge learnability

Freeze the Qwen backbone. Train only the shared memory norm and K/V projections
to match detached native upper self-attention residual outputs on fresh text.
Fit and held-out documents must be disjoint. Report every upper layer, the
aggregate normalized MSE, and the LM/logit disturbance at a small nonzero
`alpha`. Failure to improve held-out bridge error or an immediate LM cliff
stops the migration.

For the 512-step saturation pilot, bridge signal requires held-out error ratio
at most `0.70` and LM KL at `alpha=0.05` at most `0.05`. Qualification further
requires final held-out normalized MSE at most `0.50`. These thresholds were
fixed after the 64-step exploratory run and before the canonical 512-step run.

Only H0, H1, and H2 together authorize a larger migration pilot. They do not
authorize the 20M-token run automatically.

## Later migration order

If all gates pass, migrate one axis at a time:

1. keep `beta=0` and move `alpha: 0 -> 1`;
2. stabilize at `(alpha, beta)=(1, 0)`;
3. keep `alpha=1` and move `beta: 0 -> 1`;
4. stabilize the `(1, 1)` endpoint.

The original upper K/V cache and full lower attention are temporary migration
machinery. They are not counted as endpoint deployment state.
