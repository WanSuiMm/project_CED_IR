# Real-language A/B pilot results

## Decision

```text
language_model_signal       = PAIR_WIDE_NONINFERIOR
global_interface_use        = PRESENT_IN_BOTH
remote_context_gate         = UNQUALIFIED
overall_status              = FUNCTIONAL_SIGNAL_REMOTE_SUBSTRATE_INVALID
further_rescue_experiments  = NOT_AUTHORIZED
```

The direct A/B experiment has finally run. `B_PAIR_WIDE` matches and slightly
outperforms `A_TOKEN` on held-out language-model NLL at the same 256-update
budget, with identical parameter count and the same number of persistent
scalars. Both models rely strongly on the global interface. However, the
predeclared remote-context qualification is not robust: the larger paired
evaluation does not establish positive remote gain for A. Therefore this pilot
is encouraging functional evidence, not a qualified long-range result.

## Matched training

| Variant | Parameters | Records at N=256 | BF16 persistent bytes | Initial NLL | Final NLL | Interface-ablation delta |
|---|---:|---:|---:|---:|---:|---:|
| `A_TOKEN` | 306,598,912 | 256 | 524,288 | 17.7524 | 6.6625 | +4.5389 |
| `B_PAIR_WIDE` | 306,598,912 | 128 | 524,288 | 17.6271 | 6.6115 | +4.1119 |

Both used the same WikiText-103 token stream, seed, optimizer, update budget,
evaluation documents, and reader parameterization. The only variant switch is
record construction.

The frozen 8-sequence endpoint emitted `PASS_A_SUBSTRATE` for A and
`FAIL_PAIR_WIDE_PILOT` for B because A's measured remote gain was `+0.00079`
while B's was `-0.00551`. Those formal generated verdicts are preserved.

## Paired 64-sequence precision audit

| Metric | A mean (95% CI) | B mean (95% CI) | Paired B−A (95% CI) |
|---|---:|---:|---:|
| Held-out NLL | 6.4144 [6.2390, 6.5875] | 6.3673 [6.1918, 6.5390] | **−0.0471 [−0.0596, −0.0351]** |
| Remote-context gain | −0.00354 [−0.01095, 0.00342] | 0.01000 [−0.00127, 0.02416] | +0.01354 [0.00156, 0.02888] |
| Interface-ablation delta | 4.4662 [4.3164, 4.6188] | 4.2067 [4.0719, 4.3436] | −0.2596 [−0.3304, −0.1926] |

The NLL result supports the narrow functional possibility that two token states
can be carried by one twice-wide record when the reader co-adapts. The remote
metric cannot support the intended long-range claim because A itself has no
reliably positive gain. The large positive ablation deltas show that both models
use their interfaces, but do not prove use of the deliberately replaced remote
prefix.

Generation from both 256-step pilots remains repetitive. This is a small-budget
substrate experiment, not a competitive language model.

## Claim boundary

Supported in this pilot:

- Pair-wide records do not harm ordinary held-out NLL under matched co-training.
- Halving record count while doubling record width preserves persistent scalar
  count and retains strong global-interface dependence.

Not supported:

- retained long-range/remote-context computation;
- memory or bandwidth savings (bytes are equal);
- production-quality generation;
- a universal claim that token alignment is unnecessary.

Per `SCOPE.md`, the failed robust A remote-context qualification stops this
substrate. No probe, migration, longer rescue schedule, or new reader is added.

## Evidence

- A: `runs/a_token_s256_l256_v01/summary.json`
- B: `runs/b_pair_wide_s256_l256_v01/summary.json`
- paired audit: `runs/paired_eval_n64_v01.json`
