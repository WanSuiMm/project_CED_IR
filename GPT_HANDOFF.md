# GPT incremental-review handoff

Use this file when the repository has already been reviewed at the base commit.
It is an incremental handoff, not a replacement for `GPT_CONTEXT.md`.

## Review range

- Base commit: `0da62c5d26e964e8816fac48ebe18bcaf069765e`
- Evidence head: `38896e4d76922d30429f544b12d84b1cc4540fbc`
- Update: matched record-permutation shortcut control
- Scientific status: complete

The commit that adds this handoff may be newer than the evidence head. That
metadata-only commit does not change the experiment or its conclusions.

## Read in this order

1. `runs/slot_permutation_v01/PROBE.md` -- concise control result
2. the `Matched record-permutation shortcut control` section of `RESULTS.md`
3. `slot_permutation_protocol.md` -- frozen transformations and thresholds
4. `src/ced_ir/probes.py` (`permute_record_pairs`) and the permutation evaluator
   in `scripts/probe_physical_slot.py`
5. `runs/slot_permutation_v01/summary.json` -- every ordinal, head, layer, and
   probe family

Do not reread the earlier training pipeline or raw JSONL unless this review
finds a concrete dependency requiring it.

## What changed

- Added fixed-point-free cyclic query, block, and pair-preserving record
  permutations while holding phase, digit ordinal, target, and feature marginals
  fixed.
- Reproduced the supervised slot probe and evaluated the terminal probes under
  all four conditions on the frozen 4,096-example test split.
- Made layer 0, digit ordinal 0, rank-16 bilinear accuracy the primary statistic
  because that query contains no previous answer digit.
- Narrowed documentation that previously treated slot-label decoding as
  evidence for a query-specific address relation.

## What did not change

- Terminal B, all earlier checkpoints, the original G1 result, and the zero-
  training addressing audit remain unchanged.
- The original metric verdict `GATE_FAMILY_CLEANLY_DECODABLE` remains true.
- No new architecture or language-model repair was trained.

## New evidence

Primary layer-0 bilinear digit-0 physical-slot accuracy:

| matched | query cyclic | block cyclic | paired cyclic |
|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 |

- Query and block cyclic conditions deliberately break query--record identity.
- Every other mismatched digit ordinal also remains above 0.999.
- Pair-preserving cyclic accuracy remains 1.0, qualifying the transformation.
- Frozen control verdict: `STRUCTURAL_SHORTCUT_DOMINANT`.

## Current interpretation

The physical label is `(phase + digit ordinal) mod 2`; it is independent of
record identity, query key, and digit value. Near-perfect probe accuracy survives
complete query/block mismatching, including digit 0 with no answer-prefix cue.
Formatting parity is therefore sufficient to saturate the supervised probe.

This result means the probe cannot be used as evidence that B learned semantic
query-to-record addressing. It does not prove that no pair-specific signal
exists; saturation makes such a signal unidentifiable with this label. The
earlier decodability verdict is retained only as a metric statement.

## Questions for the reviewer

1. Is `a=(phase+digit_ordinal) mod 2` the correct structural relation for every
   record, and does the implementation preserve its target under permutation?
2. Are the cyclic transformations truly fixed-point-free and do they isolate
   query--record matching without changing relevant marginals?
3. Does the evidence justify `STRUCTURAL_SHORTCUT_DOMINANT` while stopping short
   of claiming that no pair-specific signal exists?
4. Is any further experiment in this physical-slot probe family decision-
   relevant, or should the mechanism branch be closed here?
