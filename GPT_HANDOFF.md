# GPT incremental-review handoff

- Review base: `9ae5eae`
- Evidence/code head: `fad41ae`
- This handoff is metadata only; review the range above, not the whole repo.

New in this range: `mla_pair/` provides a correctness-first reference for
pair-content/token-RoPE MLA, six operator tests, and a read-only converted-A
NLL gate. Start with `mla_pair/README.md`, then `mla_pair/reference.py` and
`mla_pair/test_reference.py`; read `mla_pair/qualify_token_mla.py` only for
baseline qualification. `README.md` and `PROJECT.md` only update routing and
status. Older CED, migration, oracle, and standard-KV A/B code/evidence are
unchanged.

No new model result exists. Qwen3 conversion, A qualification, paired A/B
training, actual cache allocation and kernel/latency measurements are unrun.
Do not read the previous standard-KV NLL result as a route-2 MLA result.

Reviewer questions:

1. Does `logaddexp` exactly reproduce a token-level softmax **given shared
   pair content/value**, including causal odd/even boundaries?
2. Does the incremental cache agree with full-sequence attention, and do
   persistent elements count as `N/2*d_C + N*d_R` for completed pairs?
3. Does the baseline gate avoid treating an unchanged GQA checkpoint as MLA
   or promoting an unjudged NLL measurement to a qualified baseline?
