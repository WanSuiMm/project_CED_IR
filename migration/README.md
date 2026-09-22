# Qwen-to-CED architecture migration

This directory is a completed negative qualification stage inside the same
`project_CED_IR` repository. It does not revise or extend the stopped
local-oracle branch.

The scope was deliberately limited to qualifying a token-aligned causal CED
initialized from Qwen3-0.6B. G0 failed on the deployed BF16 inference route, so
the 20M-token C/A migration and the shorter/wider branch were not run. Read
`RESULTS.md` first and `PROTOCOL.md` for the frozen decision rule.

## Entry points

- `RESULTS.md`: canonical G0 result and claim boundary.
- `src/ced_migration/modeling.py`: tested token-aligned CED and incremental cache.
- `scripts/qualify_g0.py`: real-checkpoint causal/cache/training qualification.
- `scripts/diagnose_bf16_cache.py`: native-Qwen BF16 cache calibration.
- `configs/g1_qwen06b.json`: planned first-budget configuration, never executed.
- `runs/`: compact generated evidence; start with the BF16 gated result.

Large token shards, checkpoints, transient logs, and machine launch receipts
remain local and are excluded from GitHub.
