# GPT incremental-review handoff

Use this file for incremental review of the **same CED IR project**. The new
Qwen audit is a real-model evidence stage inside this repository, not a separate
project and not a reinterpretation of the frozen synthetic runs.

## Review range

- Base commit: `c1c4ef297fe76e36d3af19443fd38721c2e49196`
- Evidence head: `511771f74b9b75401f72d566f4f967d57b889204`
- Update: frozen-Qwen attention-operator R0/R1 audit
- Scientific status: qualified run complete; formal verdict
  `INCONCLUSIVE_LOW_MASS`

The later commit that updates this handoff is metadata only. Review the evidence
range above; do not reread the unchanged synthetic training pipeline.

## Read in this order

1. `real_model_audit/RESULTS.md` — concise interpretation and limitations.
2. `real_model_audit/PROTOCOL.md` — frozen design, qualification, and decision
   order.
3. `real_model_audit/runs/r01_qwen3_06b_v03/RESULTS.md` — generated aggregate.
4. `real_model_audit/scripts/run_r01_gate.py` — Qwen extraction, query split,
   mass calculation, intervention, and aggregation.
5. `real_model_audit/src/aoc/oracle.py` — latent-KV optimizer and metrics.
6. `real_model_audit/runs/r01_qwen3_06b_v03/summary.json` only for per-unit
   verification; do not open it first.

## What changed

- Added a real-pretrained-model stage to the existing CED IR evidence chain.
- Froze `Qwen/Qwen3-0.6B-Base` and audited layers 4, 16, and 27 on four prose,
  four Python-code, and four structured examples.
- Replaced eight adjacent 8-token blocks with `c in {1,2,4}` post-RoPE latent
  KV records, fitted on 64 contiguous future queries and evaluated on a later
  disjoint 64-query span.
- Evaluated block log-mass, conditional value output, global attention output,
  downstream logit KL, and delta NLL after a real single-layer intervention.
- Corrected the exact-attention qualification path to match Qwen eager
  attention's BF16 matmul and BF16-restored softmax weights. The corrected
  maximum mismatch is 0.303%, below the frozen 1% limit.
- Added a sanitized aggregate; machine-specific paths, credentials, caches,
  logs, and launch receipts are excluded.

## New decision-relevant evidence

| latent records | held-out log-Z RMSE | held-out mu relative error | units meeting both 0.10 limits |
|---:|---:|---:|---:|
| 1 | 0.899 | 0.518 | 0 / 36 |
| 2 | 0.775 | 0.397 | 0 / 36 |
| 4 | 0.534 | 0.278 | 0 / 36 |

- Median `QK^T` entropy effective rank: 1.339.
- Median true attention mass of the replaced 64-token region: 0.568%.
- Pre-registered causal-relevance minimum: 2%.
- `8->4` median projected attention-output error: 0.983%.
- `8->4` median logit KL: 0.000649; median delta NLL: +0.001095.
- Formal verdict: `INCONCLUSIVE_LOW_MASS`.

Small global KL and delta NLL are not positive compression evidence because the
model rarely attended to the selected region. Conversely, failure of all 36
units to meet the operator thresholds means low score-matrix effective rank did
not imply held-out operator preservation for the tested oracle.

## What did not change

- All original G1 checkpoints, synthetic results, addressing audit, supervised
  slot probe, and record-permutation control remain unchanged.
- `STRUCTURAL_SHORTCUT_DOMINANT` remains the terminal interpretation of the
  physical-slot probe.
- No amortized compiler, Qwen parameter update, continued pretraining, kernel,
  or system-speed measurement was performed.
- The result is not a universal impossibility theorem.

## Reviewer questions

1. Does the implementation truly separate the contiguous fit and held-out
   future-query spans without teacher-query leakage into the intervention?
2. Does the GQA mapping jointly fit the two query heads served by each KV head,
   and are Q/K evaluated after the same normalization and RoPE used by Qwen?
3. Is block attention mass computed against the complete causal denominator,
   and does the low-mass condition correctly take precedence over downstream
   KL/NLL in the formal verdict?
4. Does the direct `8->8` identity control qualify the operator algebra, while
   the missing learned `8->8` recovery and fit/test metric split remain valid
   limitations of the optimizer audit?
5. Is the only decision-relevant follow-up a calibration-selected high-mass
   old-region audit with an optimized `8->8` recovery control, leaving R2/R3
   frozen unless that gate passes?
