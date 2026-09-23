# GPT incremental-review handoff

- Review base: `7b01be9`
- Code/evidence head: `9c32316`
- This handoff commit changes no experiment code. Review only that range.

Read [`mla_pair/PROTOCOL_B2.md`](mla_pair/PROTOCOL_B2.md), then
[`mla_pair/README.md`](mla_pair/README.md). The prior `+0.91571` NLL result is
retained but narrowly classified as
`NAIVE_FULL_PAIR_COMPILER_SHORT_ADAPTATION_FAIL`: a 91.8M-parameter dense
compiler with 98k-token short adaptation is not a route-2 capacity verdict.
The 10–20M-token endpoint remains on hold.

New implementation: `headwise_compiler.py` has separate per-KV-head K and V
pair maps (7,834,624 parameters total). `qwen3_split.py` selects it without
changing the prior dense implementation; `qwen3_cache.py` supports the same
pair lifecycle. `train_full_coadapt.py`, `run_b2_full_coadapt.sh`, and
`compare_b2.py` freeze a one-pass 1,048,576-token/arm full-parameter A/B2
pilot and its paired gap trajectory. The old CED, migration, oracle, and
standard-KV evidence is unchanged.

Local and server correctness tests pass. A 16-token real-Qwen3 B2 full/cache
smoke gave BF16 KL `0.003929` and exact cache count; both full-parameter arms
passed one-step training smokes. The 1M-token/arm run is **dispatched, not
complete**; no paired B2 NLL or speed result is reported yet. Machine-specific
launch receipts and raw summaries stay outside GitHub.

Reviewer questions:

1. Does headwise compilation avoid cross-head and K/V mixing while retaining
   the old pair-cache causal semantics and two RoPE anchors?
2. Are full-model parameter groups, fixed data order, evaluation checkpoints,
   and the `+0.10` NLL criterion matched and frozen before the long run?
3. Is the old negative result kept at its actual scope, without treating the
   new dispatch as evidence of B2 success?
