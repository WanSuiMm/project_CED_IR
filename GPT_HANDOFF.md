# GPT incremental-review handoff

- Review base: `0dfc23b`
- Code/evidence head: `a87bf53`
- This handoff commit changes no experiment code. Review only that range.

Read the B pilot result in `mla_pair/README.md` first, then inspect only the
three changed documentation files (`mla_pair/README.md`, `README.md`,
`PROJECT.md`). The route-2 code, A qualification, and older CED, migration,
oracle, and standard-KV evidence are unchanged in this range.

A remains qualified: on 32 held-out 256-token chunks A−native NLL was
`−0.00107`, and actual A cache growth matched 2048 BF16 elements/token/layer.
The completed 98,304-token/arm pilot gave A NLL `2.43202` and B NLL `3.34773`:
paired B−A `+0.91571`, bootstrap 95% CI `[+0.83633, +1.00060]`, outside the
preset `+0.10` screen margin. B's persistent reference cache is 31.25%
smaller, but no fused kernel or wall-clock speedup is claimed. The larger
training endpoint is on hold. This is a negative small-budget adaptation
screen, not a proof that all pair-content architectures fail.

Reviewer questions:

1. Is the `+0.10` screen criterion applied without changing it after seeing
   the pilot, and is the negative outcome stated at its proper evidence level?
2. Are the initial NLL mismatch and B's extra 91,786,240 compiler parameters
   explicit enough to avoid mistaking this for a compute-matched comparison?
3. Does the 31.25% claim refer only to reference-cache storage, not latency?
