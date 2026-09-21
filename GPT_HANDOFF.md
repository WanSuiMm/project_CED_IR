# GPT incremental-review handoff

Use this file when the repository has already been reviewed at the base commit.
It is an incremental handoff, not a replacement for `GPT_CONTEXT.md`.

## Review range

- Base commit: `6e47fa6`
- Evidence head: `7ab92a61db57fcc392bc606e00e7662445243d96`
- Update: zero-training addressing audit
- Scientific status: complete

The commit that adds this handoff may be newer than the evidence head. That
metadata-only commit does not change the experiment or its conclusions.

## Read in this order

1. `runs/address_audit_v01/AUDIT.md` -- concise new results
2. the `Zero-training addressing audit` section of `RESULTS.md` -- claim boundary
3. the final paragraphs of `GPT_CONTEXT.md` -- updated interpretation
4. `scripts/audit_addressing.py` -- metric implementation, if verification is needed
5. `runs/address_audit_v01/audit.json` -- full per-layer and per-head evidence

Do not begin with the large per-example JSONL files. Do not reread unchanged
training code unless a diagnostic result creates a specific reason to do so.

## What changed

- The deterministic generator now exposes queried source-digit positions,
  source-record spans, and the existing packing phase as audit metadata.
- `GlobalRead.routing_diagnostics` exposes coarse block and optional slot-gate
  probabilities without changing the returned model activation.
- `scripts/audit_addressing.py` measures coarse localization, source-record
  mass, physical-slot prediction, compiler mixing, and phase-conditioned task
  performance.
- The audit was run on all 4,096 frozen test examples for terminal B and gate
  checkpoints. No parameter was updated.

## What did not change

- Token sequences, labels, loss masks, sampling seeds, and the frozen G1
  protocol did not change.
- The trained A, B, and gate checkpoints did not change.
- Previously reported A/B/gate aggregate metrics and formal verdicts did not
  change.
- This update makes no wall-clock, deployment, or multi-seed claim.

## New evidence

- Compiler cross-half Frobenius-energy ratio: `0.013150`.
- Decoder layer 0 correct digit-block top-4: `0.6742`.
- Decoder layer 1 correct digit-block top-4: `0.0399`.
- Learned gate pooled physical-slot accuracy: `0.5193`.
- Learned gate mean correct physical-slot probability: approximately `0.503`.
- Four-digit exact match remains approximately 8% in every packing phase.

## Current interpretation

The evidence does not support the simple mechanism that B merely lost one
clean, linearly recoverable left/right source-slot bit. Wholesale compiler
entanglement is also not the main explanation. Coarse localization is useful in
decoder layer 0 but highly diffuse in layer 1, while the learned gate remains
near 50/50 even when evaluated at the known physical source block.

The remaining realization gap spans the address function, contextual payload
representation, and compatibility with the frozen reader. This does not prove
that every packed or hierarchical interface must fail.

## Questions for the reviewer

1. Are the source-position, block-rank, mass, slot, and phase metrics computed
   correctly in `scripts/audit_addressing.py`?
2. Does the evidence justify rejecting the clean missing-slot-bit explanation?
3. Is any remaining implementation issue large enough to invalidate that
   mechanism conclusion?
4. If one more zero- or low-training intervention is warranted, what is the
   smallest experiment that separates addressing failure from frozen-reader
   incompatibility?
