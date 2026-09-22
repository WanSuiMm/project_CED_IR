# Homotopy migration v0.2 results

## Decision

```text
h0_morphism_status       = PASS
h1_fp32_structure_status = PASS
h1_bf16_endpoint_status  = FAIL_NUMERICAL_CONDITIONING
h2_bridge_status         = BRIDGE_SIGNAL_PRESENT_NOT_QUALIFIED
larger_migration_status  = NOT_AUTHORIZED
```

The function-preserving formulation fixes the conceptual defect in v0.1, but
the tested single-memory bridge does not qualify for a larger migration run.
No 20M-token job was launched.

## H0/H1 qualification

The matched length is 96, which crosses the lower encoder's 64-token local
window.

| Check | FP32 | BF16 |
|---|---:|---:|
| `(alpha,beta)=(0,0)` custom path vs native, max abs | `4.98e-05` | `0` |
| `(alpha,beta)=(0,0)` custom path vs native, mean KL | `2.20e-09` | `0` |
| Native Qwen full vs cached, mean KL | `1.07e-08` | `2.82e-06` |
| CED endpoint full vs cached, mean KL | `-2.51e-07` | `0.02598` |
| CED endpoint top-1 agreement | `1.0` | `1.0` |

Thus the homotopy start is a valid morphism and the endpoint equations are
structurally consistent in FP32. The BF16 endpoint is much less numerically
stable than the same-length native Qwen baseline.

The layerwise trace shows no discrete mask or cache break. Relative discrepancy
starts at `0.0016` after lower layer 1, reaches `0.0125` after lower layer 14,
then grows through the out-of-distribution upper stack to `0.2577` at upper
layer 14. This is conditioning/amplification at the untrained endpoint, not
evidence of an off-by-one causal bug.

## H2 bridge pilot

The Qwen backbone was frozen. Only the shared memory norm and K/V projections
were trained for 512 updates on four fit strings and evaluated on two disjoint
held-out strings.

| Metric | Initial | Final |
|---|---:|---:|
| Fit normalized attention-output MSE | `1.4559` | `0.3632` |
| Held-out normalized attention-output MSE | `1.5780` | `0.9218` |
| Held-out / initial ratio | — | `0.5841` |

The branch is learnable, but it generalizes poorly and remains far above the
predeclared `0.50` qualification limit. The first upper layer is exact at
initialization because the memory K/V are copied from that layer; training one
shared memory to serve later layers worsens its held-out error to `0.4249`.
Most later layers remain near normalized MSE 1.

Rollout quality degrades smoothly and then collapses as the old self path is
removed:

| `alpha` (`beta=0`) | LM KL vs native | Token top-1 agreement |
|---:|---:|---:|
| 0.01 | `0.00058` | `1.000` |
| 0.05 | `0.00342` | `0.984` |
| 0.10 | `0.00791` | `0.984` |
| 0.25 | `0.0400` | `0.969` |
| 0.50 | `0.2292` | `0.906` |
| 1.00 | `5.8122` | `0.031` |

Bridge training improves the final endpoint's BF16 cache parity to mean KL
`0.00216`, but its language-model computation at `alpha=1` is not remotely
preserved. A larger homotopy run would therefore be testing recovery from a
severe capability cliff rather than a qualified continuous migration.

## Claim boundary

This rejects the tested bridge: one shared K/V projection pair trained to
imitate all 14 native upper attention outputs while the Qwen backbone is
frozen. It does not reject every homotopy, layer-conditioned shared memory,
additional adapters, or full-model joint training. Those are materially new
parameterizations and require a new protocol.

## Evidence

- FP32 H0/H1: `runs/h0_h1_fp32_l96_v02/summary.json`
- BF16 H0/H1: `runs/h0_h1_bf16_l96_v02/summary.json`
- BF16 layerwise trace: `runs/layerwise_bf16_l96_v01/summary.json`
- Canonical H2 pilot: `runs/h2_bridge_s512_l64_v02/summary.json`
