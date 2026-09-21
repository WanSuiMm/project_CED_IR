# Supervised oracle-block physical-slot probe protocol

Status: frozen before the formal run.

## Question

Given the frozen terminal `B_PACK2_WIDE2` representation and the known correct
packed block for each supervised query digit, is the original physical slot
`source_position mod 2` cleanly decodable?

This is a representation diagnostic. It does not modify B, repair the language
model, or establish that a decoded slot can be used by the frozen reader.

## Frozen setup

- Backbone: terminal B update-2048 checkpoint; every backbone parameter frozen.
- Probe-train split: deterministic train examples beginning at ID 100,000.
- Probe validation/test: the existing independent validation/test splits,
  examples 0--2,047 and 0--4,095 respectively.
- Features: pre-global-read decoder state and the two head-local value halves
  from the known correct packed block.
- Target: physical source slot `source_digit_position mod 2`.
- Training: 256 updates, batch 16, AdamW, learning rate 0.001, no weight decay.
- Selection: no checkpoint selection; evaluate the terminal probe once on test.
- Reporting: every decoder layer and head, plus head-macro results and all four
  packing phases.

## Probe families and controls

- `q_only`: tests whether slot is already predictable without the oracle block.
- `z_only`: linear classifier on both value halves of the oracle block.
- `qz_linear`: linear classifier on query plus both oracle-block halves.
- `bilinear`: the same head-local rank-16 query/slot interaction family used by
  the failed LM-trained gate, now trained directly with physical-slot CE.

The implementation must first pass a separable synthetic learnability control.

## Decision rules

- `GATE_FAMILY_CLEANLY_DECODABLE`: at least one layer has bilinear head-macro
  test accuracy >= 0.90 and exceeds q-only by >= 0.10.
- `BROAD_LINEAR_DECODABLE_ONLY`: the bilinear rule fails, but at least one layer
  has `qz_linear` head-macro accuracy >= 0.90 and exceeds q-only by >= 0.10.
- `TESTED_FAMILIES_NEAR_CHANCE`: every layer's `bilinear` and `qz_linear`
  head-macro accuracy is <= 0.60.
- Otherwise: `INTERMEDIATE_OR_CONTROL_CONFOUNDED`.

A high q-only score is reported as a control warning and prevents attributing
the same accuracy to oracle-block slot information. No threshold may be changed
after seeing validation or test results.
