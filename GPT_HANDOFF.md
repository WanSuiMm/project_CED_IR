# GPT incremental-review handoff

Review this cost-model update in the same `project_CED_IR` repository.

- Review base: `bebdcdd4a949dd943367a41cac4dba0ca7f62efc`
- Evidence head: `62e4f57e6ee0518e8af3f7cf7d8dc4b1a4161b3b`
- The later handoff commit is metadata only.

Read `real_language_ab/COST_MODEL.md` first. Consult
`real_language_ab/src/interface_lm/modeling.py` for the actual record and
projection shapes, and `real_language_ab/RESULTS.md` for the unchanged pilot
outcome. Older project branches and raw runs need no reread.

The new arithmetic predicts that, at a fixed projected K/V width and 256-token
prompt, B has about half A's four-layer projected KV footprint (3.023 versus
6.023 MiB) and cross-attention QK+AV work per query (6.341 versus 12.632 M
FLOPs). These are hypothetical cached-decode component costs. The current code
recomputes the full growing sequence and does not implement that cache.
The pilot's NLL signal and unqualified remote-context gate are unchanged.
No latency, bandwidth, longer-context quality, or production benefit is claimed.

Reviewer questions:

1. Are the head widths, record counts, and four-layer byte/FLOP formulas right?
2. Does the document keep materialized zero padding distinct from logical
   payload, and hypothetical caching distinct from the current implementation?
3. Does the small N=256 opportunity support the restrained systems verdict?
