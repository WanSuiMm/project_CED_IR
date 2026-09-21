# Supervised oracle-block physical-slot probe

Formal verdict: `GATE_FAMILY_CLEANLY_DECODABLE`

The B backbone was frozen. The probe was trained with direct physical-slot labels; this is a decodability result, not an LM repair.

## Test accuracy

| layer | method | head macro | head 0 | head 1 | head 2 | head 3 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | q_only | 0.4990 | 0.4990 | 0.4987 | 0.4989 | 0.4993 |
| 0 | z_only | 0.6264 | 0.6264 | 0.6264 | 0.6264 | 0.6264 |
| 0 | qz_linear | 0.9901 | 0.9874 | 0.9877 | 0.9899 | 0.9956 |
| 0 | bilinear | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 1 | q_only | 0.8063 | 0.8119 | 0.8043 | 0.8023 | 0.8069 |
| 1 | z_only | 0.6264 | 0.6264 | 0.6264 | 0.6264 | 0.6264 |
| 1 | qz_linear | 0.9916 | 0.9889 | 0.9919 | 0.9910 | 0.9947 |
| 1 | bilinear | 1.0000 | 0.9999 | 1.0000 | 1.0000 | 1.0000 |

## Matched record-permutation control

Control verdict: `STRUCTURAL_SHORTCUT_DOMINANT`

Primary layer-0 bilinear digit-0 accuracy:

| matched | query cyclic | block cyclic | paired cyclic |
|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Qualification

Synthetic learnability control: `q_only=1.000`, `z_only=1.000`, `qz_linear=1.000`, `bilinear=1.000`

Full validation and phase-stratified test metrics are in `summary.json`.
