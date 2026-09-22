# Canonical results

## Latest stage: Qwen-to-CED migration G0

The post-training architecture-migration attempt is archived under
`migration/`. FP32 structural cache parity passed, but the formal BF16
prefill-plus-one-continuation endpoint produced mean KL `0.02442`, above the
frozen `0.001` limit. Its verdict is `INVALID_CED_IMPLEMENTATION`. No 20M-token
C/A training or shorter/wider comparison was run. This is an implementation
qualification failure, not an impossibility result for CED migration; see
`migration/RESULTS.md` for the exact claim boundary.

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

Remaining explanations at this stage included inadequate coarse-block
localization, source-slot identity not being exposed to the LM objective, or
the need for joint representation/reader adaptation. The supervised probe below
separates direct decodability from those task-level explanations.

## Zero-training addressing audit

A read-only audit of the terminal B and gate checkpoints was run on all 4,096
frozen test examples. It uses the generator's exact queried source positions;
no parameter was updated. Because encoder states are contextual, both the exact
digit block and the full source-record block region are reported.

| Model/layer | Digit-block top-1 | Digit-block top-4 | Mean digit-block mass | Mean source-record mass | Physical-slot accuracy | Mean physical-slot probability |
|---|---:|---:|---:|---:|---:|---:|
| B, decoder 0 | 0.2449 | 0.6742 | 0.2531 | 0.2744 | n/a | n/a |
| B, decoder 1 | 0.0112 | 0.0399 | 0.0083 | 0.0202 | n/a | n/a |
| Gate, decoder 0 | 0.2449 | 0.6742 | 0.2531 | 0.2744 | 0.5346 | 0.5029 |
| Gate, decoder 1 | 0.0112 | 0.0397 | 0.0083 | 0.0201 | 0.5040 | 0.5023 |

Compiler geometry does not support the strong entanglement explanation:

- cross-half Frobenius-energy ratio: 0.013150;
- relative Frobenius distance from identity: 0.165524; and
- spectral condition number: 2.591676.

Phase-stratified four-digit exact match stays in a narrow range: B is
0.0776--0.0834 and the gate model is 0.0795--0.0818. Phases that pair adjacent
digits have modestly better NLL, but the physical packing phase does not create
the large exact-match split predicted by a pure left/right ambiguity story.

These measurements reject the explanation that one clean slot switch is already
usable by the tested LM-trained post-coarse gate and frozen reader. They do not
establish that physical slot is undecodable under direct supervision. Coarse
localization is not absent everywhere: decoder layer 0 often identifies the
relevant region, whereas decoder layer 1 is highly diffuse in physical source
coordinates.

## Supervised oracle-block physical-slot probe

The terminal B backbone was frozen and the known correct packed block was
supplied for every supervised query digit. Four probe families were trained on
fresh deterministic train examples and evaluated once on the independent test
split. The protocol and thresholds were frozen in `slot_probe_protocol.md`.

| Layer | q-only | z-only | q+z linear | Rank-16 bilinear |
|---:|---:|---:|---:|---:|
| 0 | 0.4990 | 0.6264 | 0.9901 | 1.0000 |
| 1 | 0.8063 | 0.6264 | 0.9916 | 1.0000 |

All numbers are four-head macro physical-slot accuracy over 4,096 held-out test
examples. Validation and test agree to within 0.001. Every probe family reached
1.0 on the separable implementation control. The pre-registered verdict is:

```text
GATE_FAMILY_CLEANLY_DECODABLE
```

Layer 0 is the decisive leakage control: q-only is at chance, while adding the
known correct block raises joint-linear accuracy to 99.0% and the exact
gate-family bilinear probe to essentially 100% in every phase and head. Thus
physical left/right label is cleanly decodable from the frozen query--block
features. This probe is not an LM repair and does not show that the decoded
label is a query-specific address relation or can be consumed by the reader.

The stronger query--block address interpretation is superseded by the matched
permutation control below. The pre-registered probe verdict remains a statement
about label decodability.

## Matched record-permutation shortcut control

The physical label has a deterministic formatting shortcut. For record index
`i`, packing phase `p`, and digit ordinal `j`, the source position is

```text
4 + p + 8*i + j
```

and the slot is `(p+j) mod 2`, independent of record identity, query key, and
digit value. The trained probe was reproduced and evaluated under fixed-point-
free cyclic record permutations that preserve phase, ordinal, target, and
marginal features while breaking query--record matching.

Primary layer-0, digit-0, rank-16 bilinear accuracy:

| Matched | Query cyclic | Block cyclic | Pair-preserving cyclic |
|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Digit 0 is the cleanest condition because its query is at `ANS` and contains no
previous answer digit. Every other mismatched ordinal also remains above 0.999.
The frozen control verdict is:

```text
STRUCTURAL_SHORTCUT_DOMINANT
```

Formatting parity is sufficient for near-perfect physical-slot decoding. The
probe neither establishes nor rules out an additional pair-specific signal,
but saturated accuracy cannot be used as evidence that B learned semantic
query-to-record addressing. The proposed oracle-routing intervention is not
motivated by this probe because its pair-specific precondition failed.

## Real-model attention-operator audit

The same project next tested the address-record hypothesis in frozen
`Qwen/Qwen3-0.6B-Base`. For each sequence and audited layer, eight adjacent
8-token blocks in an old 64-token region were replaced by independently fitted
latent KV records. Oracle parameters were fitted on one contiguous future-query
span and evaluated on a later disjoint span. Qwen parameters were never updated.

The corrected exact-attention implementation qualified with at most 0.303%
relative mismatch, below the preregistered 1% limit. Held-out aggregates are:

| Latent records per 8-token block | log-Z RMSE | conditional-value relative error | units meeting both 0.10 limits |
|---:|---:|---:|---:|
| 1 | 0.899 | 0.518 | 0 / 36 |
| 2 | 0.775 | 0.397 | 0 / 36 |
| 4 | 0.534 | 0.278 | 0 / 36 |

The median score-matrix effective rank was only 1.339, so descriptive low rank
did not imply preservation of block mass and conditional value output. The
replaced region carried only 0.568% median true attention mass, below the frozen
2% causal-relevance threshold. Consequently, small global KL and delta NLL are
not evidence of successful compression. The formal verdict is:

```text
INCONCLUSIVE_LOW_MASS
```

This result does not authorize the amortized compiler or adaptation stages. It
is also not a universal impossibility result. The exact claim boundary and any
permitted follow-up are in `real_model_audit/RESULTS.md`.

### Final R1b follow-up

The single permitted follow-up selected the highest-mass eligible old region
on `Q_select`, fitted on disjoint `Q_fit`, and tested on disjoint `Q_test`.
Exact-start `8->8` remained exact, but non-identity `8->8` recovery had median
test log-Z / conditional-value errors `0.5688 / 0.2062`; `0 / 36` units met
both 0.05 recovery limits. The formal verdict is:

```text
ORACLE_OPTIMIZER_UNQUALIFIED
```

The selection loophole also did not yield a stable high-mass target: median
mass was `1.0885%` on selection and `0.3914%` on test, with only `3 / 36` units
retaining at least 2%. Because optimizer recovery precedes mass in the frozen
decision order, these mass results are descriptive rather than the formal
verdict. The local `8 -> c` operator-compression branch is stopped, and R2/R3
remain frozen.

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
- Addressing audit: `runs/address_audit_v01/AUDIT.md` and `audit.json`
- Supervised slot probe: `runs/slot_probe_v01/PROBE.md` and `summary.json`
- Matched permutation control: `runs/slot_permutation_v01/PROBE.md` and
  `summary.json`
- Real-model oracle audit: `real_model_audit/RESULTS.md`,
  `real_model_audit/runs/r1b_high_mass_v01/RESULTS.md`, and sanitized
  `summary.json`; v0.1 is preserved under `runs/r01_qwen3_06b_v03/`
- Per-example evidence: the corresponding `test_samples*.jsonl` files

The paired test examples describe sampling uncertainty only. This project used
one training seed, so these files do not measure training-seed uncertainty.
