# GPT incremental-review handoff

This is an incremental review packet for the **same CED IR project**. It adds
one stopped Qwen-to-CED migration stage under `migration/`; the synthetic and
frozen-attention-oracle branches are unchanged.

## Review range

- Base commit: `bf4080695c98044bacbdb2e8e94e9db1240c5aa2`
- Evidence head: `12322be874542afc5382cb7514d52327d27daf3c`
- Update: real-checkpoint G0 implementation qualification
- Formal verdict: `INVALID_CED_IMPLEMENTATION`

The later commit updating this handoff is metadata only. Review the fixed range
above; do not reread unchanged synthetic runs or `real_model_audit/`.

## Minimal reading order

1. `migration/RESULTS.md` — decision, exact metrics, and claim boundary.
2. `migration/PROTOCOL.md` — frozen architecture and ordered gate.
3. `migration/src/ced_migration/modeling.py` — actual tested CED, prefill,
   incremental step, and cache accounting.
4. `migration/scripts/qualify_g0.py` — test construction and verdict logic.
5. `migration/runs/g0_qwen06b_bf16_gated_v01/summary.json` — formal evidence.
6. `migration/runs/g0_smoke_fp32_v02/summary.json` and
   `migration/runs/qwen_bf16_cache_l12_v02.json` — structural smoke and native
   Qwen numerical calibration.

Do not start with older JSONL evidence. No large checkpoint or token corpus is
needed for this review.

## What changed

- Added a Qwen3-0.6B warm-start CED: 14 causal local-window layers, one shared
  token-aligned global K/V memory, and 14 upper cross-decoder layers with no
  upper time-axis self-attention.
- Added parallel prefill and incremental decode paths with explicit cache
  accounting.
- Added a trainable 1% scalar gate to each new cross-attention residual. The
  archived code does not contain the later, untested MLP-gating idea.
- Added real-weight causal, cache, remote-path, backward, optimizer, reload,
  and BF16 numerical qualification.
- Stopped before data preparation or 20M-token C/A training.

## New decision-relevant evidence

| Metric | Result | Frozen requirement |
|---|---:|---:|
| FP32 cached max / mean abs, length 12 | `9.44e-05 / 9.75e-06` | `<= 1e-3 / 1e-4` |
| BF16 deployment mean KL, length 32 | `0.02442` | `<= 0.001` |
| BF16 deployment top-1 agreement | `1.000` | `>= 0.99` |
| Future-edit maximum | `0` | `0` |
| Local-only beyond-window edit maximum | `0` | near `0` |
| Global-memory ablation relative change | `0.1355` | `> 0` |
| Native-Qwen BF16 deployment KL, length 12 | `0.0009323` | calibration only |

FP32 smoke suggests structurally consistent prefill/incremental equations, but
the actual BF16 endpoint fails by a wide margin. The native-Qwen calibration
has a different length and must not be treated as a matched quantitative
baseline.

## What did not happen

- No 20M-token Qwen-CPT versus token-CED comparison was run.
- No shorter/wider CED variant was implemented or trained.
- No claim is made that CED migration is impossible.
- No earlier evidence, checkpoint, or verdict was changed.

## Reviewer questions

1. Does `prefill()` plus `step()` implement the same causal computation as
   `forward()`, including rotary positions, local-window boundaries, and shared
   memory growth?
2. Is the BF16 discrepancy plausibly an implementation/numerical artifact, or
   does the architecture create an inherently ill-conditioned warm start under
   this parameterization?
3. Are the cache-accounting assertions sufficient to exclude hidden original
   full-history per-layer K/V state?
4. Is the 1% cross-residual gate applied in a way that preserves the intended
   gradient path without silently reintroducing Qwen upper self-attention?
5. Given the failed preregistered BF16 gate, is stopping before G1 the correct
   scientific decision, and what single code-level defect—if any—would justify
   reopening G0?
