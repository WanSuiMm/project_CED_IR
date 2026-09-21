# B_LEARNED_SLOT verdict

## Question

Can a low-rank, query-conditioned two-way slot gate repair the completed
`B_PACK2_WIDE2` checkpoint while all original model parameters remain frozen?

## Qualification

- Base checkpoint: B at update 2048.
- Gate dimension: 16.
- New trainable parameters: 49,152 across two decoder layers.
- Coarse keys, compiler, backbone, original query projections and output
  projections: frozen.
- Gate is initialized uniformly. Update-zero maximum and mean absolute logit
  differences from B were both exactly zero.
- Training used fresh deterministic examples after the B training cursor.
- First check: 128 gate updates. Terminal check: 512 gate updates.

## Result

| Metric | B base | B + learned slot gate |
|---|---:|---:|
| Frozen test four-digit exact match | 0.0795288 | 0.0808105 |
| Frozen test NLL | 0.6972062 | 0.6939767 |
| Test exact with global IR ablated | 0.0 | 0.0 |

Validation exact match at 512 gate updates was 0.0809326. The gate parameters
moved away from initialization, so this is not a missing-gradient result.

## Verdict

`FROZEN_B_LOW_RANK_SLOT_GATE_FAIL`

The proposed minimal gate does not repair B. The result rejects the narrow claim
that a small bilinear two-way address head, trained alone on top of the frozen B
representation, is sufficient. It does not by itself distinguish among three
remaining explanations: coarse block localization is inadequate; the completed
B representation does not expose source-slot identity in a linearly usable form;
or the reader needs joint adaptation rather than a frozen-backbone gate.

The persistent NLL near `ln(2)` remains a mechanism clue, but it is no longer
enough to claim that simply supplying one learned address bit solves the task.
No gate-width, learning-rate, architecture or natural-language expansion is
authorized by this diagnostic.
