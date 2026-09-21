# Matched record-permutation control protocol

Status: frozen before the formal run.

## Question

Does the near-perfect supervised physical-slot probe depend on a matched
query--source-record relation, or can it solve the label using only digit
ordinal and packing/layout structure?

## Frozen setup

- Reproduce the frozen-backbone probe exactly as specified in
  `slot_probe_protocol.md` with the same seed, train IDs, 256 updates, probe
  families, and terminal B checkpoint.
- Evaluate one terminal probe on all 4,096 frozen test examples.
- Within each example and digit ordinal, records are rotated by one position.
  The rotation is a derangement: no query remains paired with its own record.
- Phase, digit ordinal, physical-slot target, and marginal query/block
  distributions remain unchanged.

## Conditions

- `matched`: original query and oracle block.
- `query_cyclic`: rotate queries across the eight queried records; keep blocks.
- `block_cyclic`: rotate blocks; keep queries.
- `paired_cyclic`: rotate queries and their matching blocks together. This is a
  positive control for the permutation implementation.

Report every layer, probe family, head, and digit ordinal. Digit ordinal 0 in
decoder layer 0 with the rank-16 bilinear probe is the primary statistic: its
query is at `ANS` and contains no previous answer digit.

## Frozen decision rule

Let `M`, `Q`, `Z`, and `P` be layer-0 bilinear head-macro accuracy for digit
ordinal 0 under matched, query-cyclic, block-cyclic, and paired-cyclic.

- If `M < 0.90` or `P < 0.90`: `PERMUTATION_CONTROL_INVALID`.
- If `Q <= 0.60` and `Z <= 0.60`: `PAIR_SPECIFIC_ADDRESS_SUPPORTED`.
- If `Q >= 0.90` and `Z >= 0.90`: `STRUCTURAL_SHORTCUT_DOMINANT`.
- Otherwise: `MIXED_PAIR_AND_STRUCTURAL_SIGNAL`.

The original `GATE_FAMILY_CLEANLY_DECODABLE` metric verdict remains recorded.
This control determines whether it can be interpreted as pair-specific address
evidence. Thresholds may not be changed after evaluation.
