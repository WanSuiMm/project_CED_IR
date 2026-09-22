# Qwen-to-CED migration qualification

## Decision

```text
g0_status          = INVALID_CED_IMPLEMENTATION
g1_training_status = NOT_RUN
short_wide_status  = NOT_AUTHORIZED
```

The tested Qwen3-0.6B warm start passes the FP32 structural smoke but fails the
frozen BF16 deployment-parity gate. The project stops here: no 20M-token
continued-pretraining comparison was launched.

## Evidence

| Check | Result | Requirement | Status |
|---|---:|---:|---|
| FP32 prefill + one cached continuation, max abs | `9.44e-05` | `<= 1e-3` | pass (smoke) |
| FP32 prefill + one cached continuation, mean abs | `9.75e-06` | `<= 1e-4` | pass (smoke) |
| BF16 prefill + one cached continuation, mean KL | `0.02442` | `<= 0.001` | **fail** |
| BF16 prefill + one cached continuation, top-1 agreement | `1.000` | `>= 0.99` | pass |
| Future-token edit effect on earlier logits | `0` | `0` | pass |
| Local-only response to a beyond-window edit | `0` | near `0` | pass |
| BF16 global-memory ablation relative change | `0.1355` | `> 0` | pass |
| Backward / optimizer / save-reload | finite / moved / exact | all required | pass |

The native Qwen BF16 length-12 calibration gives deployment mean KL
`0.0009323` and top-1 agreement `1.0`. It is a numerical reference, not a
matched-length baseline for the length-32 CED result.

## Interpretation

The zero cache-accounting, causal, gradient, and save/reload controls do not
explain away the failure. FP32 smoke parity indicates that the prefill and
incremental equations are structurally consistent at short length. However,
the actual BF16 route is far outside the preregistered KL threshold, so this
implementation is not qualified for a scientific C/A migration comparison.

This result does **not** show that causal CED migration is impossible. It shows
that the tested direct warm start—14 local-window Qwen layers, one shared
token-aligned K/V memory, 14 upper cross-decoder layers, and 1% trainable
cross-attention residual gates—cannot be treated as a valid BF16 starting point
under this protocol. Further architecture rescue work would require a new
protocol and is not part of this archived stage.

## Canonical files

- Formal BF16 result: `runs/g0_qwen06b_bf16_gated_v01/summary.json`
- Generated BF16 report: `runs/g0_qwen06b_bf16_gated_v01/RESULTS.md`
- FP32 structural smoke: `runs/g0_smoke_fp32_v02/summary.json`
- Native-Qwen BF16 calibration: `runs/qwen_bf16_cache_l12_v02.json`
- Tested implementation: `src/ced_migration/modeling.py`
- Qualification runner: `scripts/qualify_g0.py`
