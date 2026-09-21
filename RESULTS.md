# Canonical results

All primary numbers below come from the frozen 4,096-example test split at the
pre-registered terminal checkpoint. Exact match is computed per four-digit
query. Each example contains eight queries.

| Variant | Trainable parameters | Terminal training | Test NLL | Four-digit exact | Exact with global IR ablated |
|---|---:|---:|---:|---:|---:|
| `A_TOKEN` | 3,820,544 | 2,048 full-model updates | 0.003729 | 0.997101 | 0.000000 |
| `B_PACK2_WIDE2` | 4,213,760 | 2,048 full-model updates | 0.697206 | 0.079529 | 0.000000 |
| `B_LEARNED_SLOT` | 49,152 new gate parameters; base frozen | 512 gate-only updates after B | 0.693977 | 0.080811 | 0.000000 |

## Primary A/B decision

The A baseline passes both qualification gates:

- exact match is at least 0.95; and
- removing global IR lowers exact match by much more than 0.20 absolute.

B is 0.917572 absolute below A, far outside the 0.02 non-inferiority margin.
This is not an inconclusive near-threshold result.

Formal decision:

```text
correctness_status     = PASS
baseline_status        = QUALIFIED
representation_status  = DEGRADED_IN_THIS_SETTING
width_compensation     = UNTESTED
systems_status         = NOT_MEASURED
```

## Slot-gate decision

The low-rank gate satisfies its qualification controls:

- update-zero maximum absolute logit difference from B: 0.0;
- update-zero mean absolute logit difference from B: 0.0;
- all original B parameters frozen;
- gate parameters moved away from initialization;
- terminal budget of 512 gate updates completed.

Its exact-match change from B is only +0.001282 absolute and does not constitute
a repair. Formal gate verdict:

```text
FROZEN_B_LOW_RANK_SLOT_GATE_FAIL
```

## Mechanistic reading

The packed reader uses global IR: ablating it destroys performance. However,
using global IR is not the same as correctly localizing and selecting the
source information. B and the gate variant both remain close to `ln(2)` NLL.
This pattern is consistent with unresolved ambiguity, but the gate failure
means the stronger story—"the correct block is found and exactly one readable
slot bit is missing"—has not been established.

Remaining explanations include inadequate coarse-block localization, source
slot identity not being linearly exposed by the completed B representation, or
the need for joint representation/reader adaptation.

## Resource observations are not a systems verdict

During training on the same physical RTX 5090, measured input throughput was
approximately 610k tokens/s for A and 641k tokens/s for B; peak reserved memory
was approximately 1.15 GB for A and 0.95 GB for B. These measurements are
training-run diagnostics under a shared-machine environment. They are not the
specified full decode/cache benchmark and must not be reported as a deployment
speedup.

## Evidence files

- A aggregates: `runs/g1_a_formal_u2048_20260921/summary.json`
- B aggregates: `runs/g1_b_formal_u2048_20260921/summary.json`
- Gate aggregates: `runs/g1_b_learned_slot_c16_u0512_20260921/summary.json`
- Gate verdict: `runs/g1_b_learned_slot_c16_u0512_20260921/VERDICT.md`
- Per-example evidence: the corresponding `test_samples*.jsonl` files

The paired test examples describe sampling uncertainty only. This project used
one training seed, so these files do not measure training-seed uncertainty.
