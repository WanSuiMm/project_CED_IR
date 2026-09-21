# R0/R1 result

**Formal verdict: `INCONCLUSIVE_LOW_MASS`**

The model was frozen. The independent unit is a source sequence at one layer.

| c | units | eff. rank | region mass | log-Z RMSE | mu rel. | unit pass | head rel. | o_proj rel. | KL | delta NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 1.3394 | 0.0057 | 0.8989 | 0.5181 | 0.000 | 0.0144 | 0.0154 | 0.000683 | -0.000460 |
| 2 | 36 | 1.3394 | 0.0057 | 0.7754 | 0.3970 | 0.000 | 0.0113 | 0.0132 | 0.000629 | -0.000540 |
| 4 | 36 | 1.3394 | 0.0057 | 0.5338 | 0.2783 | 0.000 | 0.0083 | 0.0098 | 0.000649 | 0.001095 |

## Qualification

- layer 4: manual exact `2.925e-03`, identity 8->8 `0.000e+00`
- layer 16: manual exact `3.031e-03`, identity 8->8 `0.000e+00`
- layer 27: manual exact `2.929e-03`, identity 8->8 `0.000e+00`

## Claim boundary

A pass establishes only a query-distribution-conditioned oracle existence result. It does not establish an amortizable compiler, a discrete codec, or a wall-clock gain.
