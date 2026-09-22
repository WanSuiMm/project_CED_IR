# GPT incremental-review handoff

This is an incremental review packet for the **same CED IR project**. It adds
the function-preserving homotopy follow-up under `migration_v02/`. All synthetic
and frozen-attention-oracle evidence is unchanged.

## Review range

- Base commit: `83c78e9173d8f325a406a6dca418e8e8895b960a`
- Evidence head: `6738b1cd99f531725c1697c603e6802742aef2d4`
- Update: Qwen-to-CED homotopy implementation and three qualification gates
- Terminal verdict: `BRIDGE_SIGNAL_PRESENT_NOT_QUALIFIED`

The later commit updating this handoff is metadata only. Review the fixed range
above; do not reread old synthetic runs or `real_model_audit/`.

## Minimal reading order

1. `migration_v02/RESULTS.md` — terminal decision and exact claim boundary.
2. `migration_v02/PROTOCOL.md` — two-knob morphism and frozen gate order.
3. `migration_v02/src/ced_homotopy/modeling.py` — native/self/cross mixing,
   bridge objective, and endpoint cache.
4. `migration_v02/scripts/qualify_h0_h1.py` — matched native and endpoint gates.
5. `migration_v02/scripts/qualify_h2_bridge.py` — bridge fit/test split, alpha
   sweep, and verdict.
6. Canonical JSON summaries under `migration_v02/runs/`; read the 512-step H2
   summary first among raw evidence.

## What changed

- Removed the trainable 1% replacement gate from the new formulation.
- Added external, non-trainable `alpha` and `beta` schedules.
- At `(alpha,beta)=(0,0)`, the custom path reproduces native Qwen.
- At `(1,1)`, upper self K/V and lower full-history attention are absent from
  deployment state.
- Added a frozen-backbone teacher bridge for one shared memory K/V pair.
- Corrected the v0.1 write-up: its FP32 smoke was ungated and therefore not
  matched evidence for the later gated BF16 run.

## Decision-relevant evidence

| Check | Result |
|---|---:|
| FP32 start morphism max abs / KL, L96 | `4.98e-05 / 2.20e-09` |
| BF16 start morphism max abs / KL, L96 | `0 / 0` |
| Native Qwen BF16 full/cache KL, L96 | `2.82e-06` |
| Untrained CED endpoint BF16 full/cache KL, L96 | `0.02598` |
| 512-step bridge fit error | `1.4559 -> 0.3632` |
| 512-step bridge held-out error | `1.5780 -> 0.9218` |
| Rollout at `alpha=0.50`, KL / top-1 | `0.2292 / 0.906` |
| Rollout at `alpha=1.00`, KL / top-1 | `5.8122 / 0.031` |
| Trained endpoint BF16 full/cache KL | `0.00216` |

The exact homotopy start succeeds and bridge gradients carry useful signal.
However, one shared K/V pair does not generalize across the 14 upper layers
under this frozen-backbone bridge. The fully migrated rollout collapses, so no
larger migration was launched.

## What did not happen

- No 20M-token migration run.
- No full-model alpha schedule.
- No lower-window beta schedule beyond endpoint structural qualification.
- No shorter/wider interface experiment.
- No claim that every homotopy or shared-memory architecture is impossible.

## Reviewer questions

1. Does `(alpha,beta)=(0,0)` genuinely reproduce native Qwen rather than bypass
   the custom path in the reported H0 comparison?
2. Are self and shared-cross head outputs mixed in the correct location before
   each original `o_proj`, with Q/O shared and memory K/V independently copied?
3. Does the endpoint cache contain exactly 14 local windows plus one shared
   global K/V, with no hidden upper self-attention cache?
4. Is the bridge teacher extracted from each native upper attention output
   without allowing gradients into the Qwen backbone or fit/test leakage?
5. Is `BRIDGE_SIGNAL_PRESENT_NOT_QUALIFIED` the correct interpretation of the
   held-out error plateau and alpha=1 rollout collapse?
6. Is there one concrete implementation error that invalidates the negative
   bridge result, or would any rescue require a materially new parameterization
   such as layer-conditioned memory/adapters?
