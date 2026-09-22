# Real-language interface A/B pilot v0.1

## Question

With producer and reader co-training, can one `2d`-wide persistent record
replace two token-aligned `d`-wide records?

## Fixed substrate

- Initialization: `Qwen/Qwen3-0.6B-Base`.
- Producer: copied embedding plus the first 4 Qwen layers, using causal local
  attention with window 16.
- Reader: 4 cross-attention/MLP blocks initialized from Qwen layers 4--7.
- Sequence length: 256.
- All parameters trainable; no native-Qwen preservation objective.
- Real-language stream: WikiText-103 raw, with fixed train/validation token
  caps and deterministic packing.

The producer's maximum four-layer local receptive field is 61 tokens. Remote
metrics replace the first 128 tokens and score only the final 64 tokens, so the
changed prefix cannot reach the scored suffix through the local producer path.

## Only intervention

Both variants expose a common `2d` reader record type and have identical
parameter shapes.

- `A_TOKEN`: each producer state is stored as `[h_i; 0]`, retaining only `d`
  persistent scalars per token.
- `B_PAIR_WIDE`: adjacent states are stored as `[h_2j; h_2j+1]`, retaining
  `2d` persistent scalars per pair.

Queries in pair `j` may read only completed earlier pairs. A learned null record
handles the first pair. Reader Q/O/MLP and record K/V projection shapes are the
same in A and B.

## Matched training

- 256 optimizer updates, one 256-token sequence per update.
- AdamW, learning rate `3e-5`, weight decay `0.1`, gradient clip `1.0`.
- Seed `20260922` and the same deterministic sequence order.
- BF16 parameters and computation.
- Evaluation on 32 fixed validation sequences at updates 0, 64, 128, 192,
  and 256.

## Ordered decision

A qualifies only if all hold:

1. held-out NLL decreases by at least `0.20`;
2. generated 32-token continuation has distinct-token ratio at least `0.10`;
3. remote-context gain is positive;
4. global-interface ablation increases suffix NLL by at least `0.01`.

Failure is `INVALID_REAL_LANGUAGE_SUBSTRATE`; B is not run.

If A passes, B receives the identical budget. This pilot declares functional
non-inferiority only if B final NLL is within `0.10` of A and retains at least
80% of A's positive remote-context gain. Persistent bytes are reported; no
kernel or wall-clock speedup is claimed.
