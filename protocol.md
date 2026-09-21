# Frozen G1 Protocol

- Case: `CED_IR_LENGTH_WIDTH_PROBE_V01`
- Frozen: 2026-09-21 Asia/Shanghai, before formal data generation or training
- Scope: synthetic long-range associative retrieval only (G1)
- Variants: `A_TOKEN` (`r=1,s=1`) and `B_PACK2_WIDE2` (`r=2,s=2`)
- Conditional diagnostic: `C_PACK2_NARROW1` (`r=2,s=1`), not launched by the
  A/B commands
- Common conditions: seed 17, data seed 20260921, identical example order,
  target positions, effective input tokens per update and update count
- Model: d=256, 4 heads, 2 local encoder layers, 2 decoder layers, window 32,
  SwiGLU width 688, vocabulary 1024, tied embedding/head, no dropout
- Data: length 512 inputs, 16 unique key/value records, 8 unique queries,
  independent four-hex-digit values, minimum source-query gap 192
- Loss: next-token cross entropy only on the 32 answer digit targets
- Optimizer: AdamW, lr 3e-4, betas (0.9,0.95), eps 1e-8, weight decay 0.1,
  clip 1.0, BF16 autocast with FP32 parameters and optimizer state
- Batch: microbatch 16, accumulation 2, 16,384 input tokens/update
- Schedule: 100 warmup updates, cosine decay scheduled against 2048 updates
- Evaluation: frozen validation 2048 examples every 128 updates; frozen test
  4096 examples only at the selected terminal checkpoint
- First decision point: update 512. Continue to at most update 2048 only when
  the first decision is not conclusive and the implementation remains valid.
- Per-run cap: two GPU-hours; total workspace budget remains governed by the
  source protocol.
- Baseline qualification: A query-level four-digit exact match >= 0.95 and
  evaluation-only global ablation lowers it by at least 0.20 absolute.
- B non-inferiority tolerance: at most 0.02 absolute exact-match decrease,
  assessed with an example-clustered paired bootstrap in the final report.
- Stop immediately for causality/cache/gradient failure, NaN, data leakage, or
  a device-hour limit. Never change B-only hyperparameters after launch.

The source of truth for semantics and claim boundaries is this file together
with `ARCHITECTURE.md`, `RESULTS.md`, and the frozen JSON configuration.


## Pre-launch execution amendment (2026-09-21 16:18+08:00)

Remote preflight found GPU 0 occupied by an unrelated full-load training job
(about 22 GiB allocated, about 98% utilization) and GPU 1 holding an idle GLM
service (about 19 GiB allocated, 0% utilization at inspection). The G1 smoke
peak was about 1.1 GiB reserved. To avoid cross-device hardware differences and
not disturb GPU 0, A and B will be profiled and, only if contention remains
absent, trained sequentially on physical GPU 1. Existing processes will not be
stopped. Any observed contention invalidates timing comparisons and pauses the
run. Capability comparisons remain matched by examples, targets, updates and
device; concurrent A/B execution is not a scientific requirement.
