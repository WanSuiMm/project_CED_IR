# GPT incremental-review handoff

Review only the new direct real-language A/B experiment in the same
`project_CED_IR` repository. Older synthetic, oracle, and migration branches are
unchanged and are not required reading.

## Fixed review range

- Base: `dceeaa6059ed894196837c03270b2d52c2b145b7`
- Evidence head: `19d0a640b091e79d5e72bfb3982becec6d5e4b92`
- Later handoff commit: metadata only

## Minimal reading order

1. `SCOPE.md` — the single direct A/B question and stop rule.
2. `real_language_ab/RESULTS.md` — canonical result and claim boundary.
3. `real_language_ab/PROTOCOL.md` — matched design and preregistered gates.
4. `real_language_ab/src/interface_lm/modeling.py` — the only architectural
   difference between A and B.
5. The three small JSON files under `real_language_ab/runs/` — raw aggregates.

## A versus B

| | A: token-aligned | B: pair-wide |
|---|---:|---:|
| persistent records at N=256 | 256 | 128 |
| persistent scalars / BF16 bytes | 262,144 / 524,288 | 262,144 / 524,288 |
| paired held-out NLL, n=64 | 6.4144 | 6.3673 |
| interface-ablation delta | +4.4662 | +4.2067 |
| remote-context gain, 95% CI | -0.0035 [-0.0109, 0.0034] | +0.0100 [-0.0013, 0.0242] |

The narrow positive result is that B is non-inferior to A for ordinary language
modeling while using half as many records, and both models depend strongly on
their interfaces. The primary remote-context substrate is **not qualified**:
neither variant has an individually positive 95% confidence interval. This is a
functional pilot, not a memory-saving, production-generation, or universal
architecture claim. No rescue experiment is authorized by this update.

## Reviewer questions

1. Is record construction the only A/B architectural difference?
2. Are causal availability and reader masks correct for both interfaces?
3. Are parameter shapes, initialization, token order, optimizer, and updates
   matched?
4. Do the reported numbers support exactly the narrow interpretation above?
