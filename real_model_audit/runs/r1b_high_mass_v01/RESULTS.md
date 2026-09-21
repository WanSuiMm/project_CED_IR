# R1b final high-mass audit

**Formal verdict: `ORACLE_OPTIMIZER_UNQUALIFIED`**

| Metric | Value |
|---|---:|
| Units | 36 |
| Select mass median | 0.0109 |
| Fit mass median | 0.0051 |
| Test mass median | 0.0039 |
| Test mass >=2% fraction | 0.083 |
| 8->8 recovery fit log-Z / mu | 0.0199 / 0.0429 |
| 8->8 recovery test log-Z / mu | 0.5688 / 0.2062 |
| 8->8 recovery test unit fraction | 0.000 |
| 8->4 fit log-Z / mu | 0.0265 / 0.1574 |
| 8->4 test log-Z / mu | 0.6152 / 0.3006 |
| 8->4 test unit fraction | 0.000 |
| 8->4 projected output error | 0.0077 |
| 8->4 logit KL / delta NLL | 0.000615 / -0.000510 |
| Raw / centered effective rank | 1.351 / 3.522 |

## Qualification

- manual exact maximum: `3.031e-03`
- exact-start 8->8 fit maximum: Z `0.000e+00`, mu `0.000e+00`
- exact-start 8->8 test maximum: Z `0.000e+00`, mu `0.000e+00`
- non-identity optimizer recovery qualified: `False`

R2 compiler training and R3 adaptation remain frozen unless the formal verdict is `PASS_R1B_ORACLE_8_TO_4`.
