# R0/R1 v0.1 result

## Verdict

`INCONCLUSIVE_LOW_MASS`

The corrected implementation qualified: maximum manual-exact mismatch was
`0.303%` across the three audited layers, below the preregistered `1%` limit.
The identity `8->8` replacement was exact.

The fixed 64-token old region received only `0.568%` median attention mass,
below the preregistered `2%` causal-relevance threshold. Therefore the very
small downstream KL and delta NLL are not evidence that compression succeeded.

## Operator result

| latent records | held-out log-Z RMSE | held-out mu relative error | units meeting both 0.10 limits |
|---:|---:|---:|---:|
| 1 | 0.899 | 0.518 | 0 / 36 |
| 2 | 0.775 | 0.397 | 0 / 36 |
| 4 | 0.534 | 0.278 | 0 / 36 |

Although the median score-matrix effective rank was only `1.339`, low spectral
rank did not translate into held-out preservation of block mass and conditional
value output. This is evidence against inferring functional compressibility from
`QK^T` spectrum alone.

## Claim boundary and next gate

This run is not a positive existence result and does not authorize an amortized
compiler or continued pretraining. It also is not a universal impossibility
result: the region selection produced low-mass blocks, and the run did not log a
learned `8->8` optimizer-recovery control or separate fit-versus-held-out oracle
metrics.

If the question is reopened, the only justified follow-up is a small matched
audit that selects high-mass old regions using a calibration query span, tests
on a later disjoint span, records fit and held-out errors, and requires an
optimized `8->8` oracle to recover the exact operator. R2/R3 remain frozen.
