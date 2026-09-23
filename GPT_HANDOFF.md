# GPT incremental-review handoff

- Review base: `816685b`
- Code/evidence head: `2dac206`
- This handoff commit changes no experiment code. Review only that range.

Read the new B2 result in [`mla_pair/README.md`](mla_pair/README.md) first,
then the status changes in `PROJECT.md` and root `README.md`. The protocol
(`mla_pair/PROTOCOL_B2.md`), implementation, tests, earlier negative 98k
pilot, and all older project branches are unchanged in this review range.

Both full-parameter A and headwise B2 completed the fixed 4096-step,
1,048,576-token/arm run on the same token stream and 32 held-out validation
chunks. Final NLL: A `2.45599`, B2 `2.91093`; paired B2−A `+0.45494`, 95% CI
`[+0.41569, +0.49546]`, outside the frozen `+0.10` margin. Every validation
chunk favored A. The gap fell from `+3.80154` initially to `+0.42145` at
786,432 tokens, then rose to `+0.45494` at the endpoint, so the frozen late
trend flag is `OUTSIDE_MARGIN_NO_CLEAR_LATE_IMPROVEMENT`.

This stops the **specific** headwise-linear, one-content/value-state-per-pair
route-2 implementation at a single-seed 1M-token viability screen. It is not
a universal capacity theorem, a kernel benchmark, or a decode-latency result.
The reference cache is 31.25% smaller, but no speedup is claimed. Do not
automatically escalate to 10–20M tokens or start route 1. Generated raw
per-sequence summaries and machine-specific paths remain outside GitHub.

Reviewer questions:

1. Does the completed run match the fixed protocol, and is the paired gap
   computed from the same 32 validation chunks at each checkpoint?
2. Is the late-trend language appropriately limited to the last quarter,
   given earlier large recovery from a severe initialization cliff?
3. Are the storage and scientific conclusions kept separate from unmeasured
   decode latency and universal architecture claims?
