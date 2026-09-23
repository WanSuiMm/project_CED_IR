# GPT incremental-review handoff

Review this standard-KV A/B experiment in the same `project_CED_IR` repository.

- Review base: `d73db86ae05c9f7a0ee089f54c9e9708b3099e57`
- Evidence head: `9c9e91528cc95120c29f77cf0f1704a17e259359`
- The later handoff commit is metadata only.

Read `real_language_ab/RESULTS_STANDARD_KV_v0.2.md`, then
`real_language_ab/PROTOCOL_STANDARD_KV_v0.2.md`. For implementation review,
inspect `real_language_ab/src/interface_lm/modeling.py` and the two runner
changes under `real_language_ab/scripts/`. The three small JSON summaries under
`real_language_ab/runs/` are the canonical metrics. The v0.1 pilot and older
project branches are unchanged.

The only architectural change from the previous wide-V pilot is setting
`value_head_multiplier=1` for **both** A and B. Within v0.2, A/B differ only
in record construction. On 64 paired validation chunks, B−A NLL is `-0.0794`
with bootstrap 95% CI `[-0.0980,-0.0628]`, passing the fixed `+0.10` margin.
Both models depend on their interfaces. Neither has established positive
remote-prefix gain; no decode cache or latency measurement was run.

Reviewer questions:

1. Does multiplier 1 restore standard V and O widths while preserving matched
   A/B parameter shapes, masks, training order and initialization?
2. Is the paired NLL analysis faithful to the 64 non-overlapping chunks and
   its chunk-independence limitation?
3. Does the result support only a short-context LM non-inferiority claim?
