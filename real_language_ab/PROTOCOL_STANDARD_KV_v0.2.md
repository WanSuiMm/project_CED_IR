# Standard-KV direct A/B v0.2

## Question

At equal projected K/V width, can a `N/2 × 2d` pair interface match a
`N × d` token interface on real-language next-token NLL? This tests whether
the v0.1 B result depended on doubling the reader-facing V width.

## Frozen intervention and controls

Reuse the v0.1 Qwen3-0.6B initialization, WikiText-103 token cache, 256-token
sequences, 4 local producer layers (window 16), 4 cross-reader layers, BF16,
seed `20260922`, identical token order, AdamW settings, and 256 updates for
each variant. A remains `[h_i;0]`; B remains `[h_2j;h_2j+1]`. Both retain the
same query mask, null record, and `2d` projection input. The sole architectural
change relative to v0.1 is `value_head_multiplier: 2 → 1`, making both variants'
projected K and V each 8 heads × 128 dimensions. Q/O/MLP parameter shapes
are matched between v0.2 A and B. v0.1 stays intact with the default value 2.

## Decision and evidence

Run one A and one B from the same base initialization. A must lower validation
NLL by at least `0.20` and have interface-ablation delta at least `0.01`;
otherwise stop as `INVALID_STANDARD_KV_SUBSTRATE`. For the primary B/A
comparison, evaluate both checkpoints on the same 64 validation sequences and
bootstrap the paired `B−A` NLL mean over independent sequences (10,000 draws,
seed `20260922`). Declare `STANDARD_KV_NONINFERIOR` only if the upper 95% CI is
at most `+0.10` NLL. Declare `STANDARD_KV_INFERIOR` if the lower 95% CI exceeds
`+0.10`; otherwise `INCONCLUSIVE`. The 32-sequence training-time comparison
is a provisional point estimate. Report remote-prefix and interface-ablation
metrics descriptively; remote qualification is not this experiment's gate.

No kernel, incremental cache, wall-clock decode benefit, or long-context
functional claim follows from this pilot. Store the new checkpoints and logs
in a separate v0.2 run directory; keep only compact summaries in GitHub.
