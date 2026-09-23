# GPT incremental-review handoff

- Review base: `904ecb1`
- Code/evidence head: `cf3e102`
- This handoff commit changes no experiment code. Review only that range.

Read `mla_pair/README.md` first. The new route-2 implementation is
`mla_pair/qwen3_split.py` (Qwen3 A and pair B), with persistent B state in
`mla_pair/qwen3_cache.py`; focused tests are `test_qwen3_split.py` and
`test_reference.py`. `run_quality_first.py` runs A qualification;
`train_variant.py`, `run_paired_pilot.sh`, and `compare_pilot.py` define the
lightweight matched A/B screen. Older CED, migration, oracle, and standard-KV
code/evidence are unchanged.

A is a **quality-first split**, not low-rank TransMLA: it retains the original
Qwen3 K/V scalar width, split into 1280 content and 768 token-RoPE elements.
On 32 held-out 256-token chunks A−native NLL was `−0.00107`, and actual A
cache growth matched 2048 BF16 elements/token/layer. B's 16-token cache smoke
matched its expected 31.25% reduction relative to A, but is not a quality or
speed result. A 98,304-token/arm A/B screen has been dispatched; its paired
NLL is **not yet reported**. No fused kernel or wall-clock speedup is claimed.

Reviewer questions:

1. Does the split-Qwen A preserve q_norm/k_norm and isolate only the RoPE
   dimension split, including cached continuation?
2. Does B's full/cached path compile only content while keeping both RoPE
   anchors and correct odd/even causal behavior?
3. Are A/B training initialization, token order, shared LoRA setup, and
   optimizer settings matched, with B's extra compiler parameters disclosed?
