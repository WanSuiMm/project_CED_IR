# B2: final route-2 viability pilot (frozen before launch)

Question: with one content/value state per token pair and one RoPE anchor per
token, can a lightweight compiler and full Qwen3 co-adaptation approach the
token-content A baseline? This is not a decode-speed or systems endpoint.

## Arms and intervention

- Both arms begin from the same Qwen3-0.6B-Base checkpoint and the qualified
  96-dimensional-per-KV-head split (`d_C=1280`, `d_R=768`). A retains one
  content state per token. B2 retains one content state per completed pair,
  with both per-token RoPE anchors unchanged.
- B2 replaces the old cross-head `2560→1280` dense compiler with per-layer,
  per-KV-head **separate** `64→32` content-key and `256→128` value maps.
  They have 7,834,624 total parameters. Adjacent keys initialize as their
  sum divided by `sqrt(2)` (independence-based RMS approximation); values
  initialize as their average. No fitted calibration or learning-rate sweep.
- Both arms update the full pretrained transformer, not LoRA. All noncompiler
  parameters use AdamW `2e-5`; B2 compiler parameters use `2e-4`, both with
  weight decay `0.01` and separate gradient-norm clipping at `1.0`. BF16
  model, FP32 compiler, Qwen3 split and gradient checkpointing are shared.
- Fixed seed `20260923`. Same contiguous, nonrepeating first 1,048,576
  WikiText-103 Qwen3 tokens per arm: `4096 × 256`, batch one, one pass.
  Same 32 held-out 256-token validation chunks, evaluated at steps
  `0, 1024, 2048, 3072, 4096`.

## Decision and stopping

- Primary: final paired `B2−A` validation NLL, with a fixed `+0.10` margin.
  Use the 32 validation chunks as paired units; report a 10,000-resample
  paired bootstrap interval, not a per-token pseudo-replication.
- Also report the gap at each fixed checkpoint. If the final gap exceeds
  `+0.10`, mark `OUTSIDE_MARGIN_STILL_IMPROVING` only when the final quarter
  closes the gap by more than `0.05` nats; otherwise mark
  `OUTSIDE_MARGIN_NO_CLEAR_LATE_IMPROVEMENT`. This trend flag is descriptive,
  not an authorization to extend training.
- Stop on nonfinite loss, missing data, incompatible checkpoint, failed
  full/cache parity or invalid actual-cache accounting. For BF16 parity,
  require 16-token final-logit full/cache KL `<=0.005` and exact persistent
  scalar count; this engineering tolerance was set after the qualification
  smoke (`0.003929`), before the 1M-token training launch, and is not a
  scientific quality gate. Preserve failures.
  Do not sweep LR, compiler width, RoPE split or seeds to rescue a miss.
- A successful quality screen permits a separate kernel/latency proposal; it
  does not demonstrate speedup. A failed stable screen stops this one-state-
  per-pair implementation; it does not automatically initiate route 1.

## Provenance and output

Code: `headwise_compiler.py`, `qwen3_split.py`, `train_full_coadapt.py`,
`run_b2_full_coadapt.sh`, `compare_b2.py`; correctness controls in
`test_headwise_compiler.py`. Run into a new directory under `mla_pair/runs/`
locally and a distinct server run directory. The generated per-arm and paired
`summary.json` files are raw evidence, not committed because they include
machine-specific paths. Copy only sanitized aggregates into this repository.

The prior 98k-token result remains
`NAIVE_FULL_PAIR_COMPILER_SHORT_ADAPTATION_FAIL`: credible for that exact
91.8M-parameter compiler and training recipe, not a route-2 capacity verdict.
