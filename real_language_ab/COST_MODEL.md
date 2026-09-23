# A/B address-width cost model

The table below describes the v0.1 **wide-V** architecture. The later
standard-KV comparison and its narrower cost figures are in
`RESULTS_STANDARD_KV_v0.2.md`.

This is arithmetic for the v0.1 pilot architecture, **not a latency or
memory benchmark**. It answers whether halving addressable records can have a
systems payoff before another functional experiment is considered.

## Inputs and accounting boundary

- Prompt length `N=256`; A has `M_A=N+1=257` records and B has
  `M_B=N/2+1=129`, including one null record. Per-query rows model a query
  immediately after the entire prompt is available, rather than the average
  query inside the causal training sequence.
- [Qwen3-0.6B-Base config](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json):
  hidden width `d=1024`, 16 query heads, 8 KV heads, head dimension 128.
- The [pilot reader](src/interface_lm/modeling.py) has 4 independent cross-attention
  layers. Each layer projects a `2d=2048` record to 1024 K scalars and 2048 V
  scalars. Each query has 16 K heads of width 128 and 16 V heads of width 256.
- BF16 is 2 bytes/scalar. FLOPs count multiply and add as two operations.
  Counts exclude the producer, Q/O/MLP, normalization, vocabulary head,
  allocation and kernel overhead unless stated otherwise.

| Quantity at `N=256` | A: 257 records | B: 129 records | B/A |
|---|---:|---:|---:|
| Logical information payload, excluding null | 0.500 MiB | 0.500 MiB | 1.000 |
| Materialized `records` tensor, including null | 1.004 MiB | 0.504 MiB | 0.502 |
| Hypothetical projected K+V cache, one reader layer | 1.506 MiB | 0.756 MiB | 0.502 |
| Hypothetical projected K+V cache, four layers | 6.023 MiB | 3.023 MiB | 0.502 |
| QK FLOPs per query, four layers | 4.211 M | 2.114 M | 0.502 |
| AV FLOPs per query, four layers | 8.421 M | 4.227 M | 0.502 |
| QK+AV FLOPs per query, four layers | 12.632 M | 6.341 M | 0.502 |
| Logical attention logits per query, four layers | 16,448 | 8,256 | 0.502 |

The `+1` null record makes the ratio `129/257`, slightly above one half.
Projected cache bytes are `4 × M × (1024+2048) × 2`. Attention FLOPs are
`4 × 2 × M × 16 × (128+256)`. Logits are `4 × 16 × M`; fused attention need not
materialize this whole vector. Actual `repeat_interleave` also expands each
layer's K/V heads from 8 to 16 for the current full-forward call; that is a
temporary tensor, not a persistent cache.

## Repeated reads and projection

With a **fixed, already compiled** prompt and `T` future queries, four-layer
attention work is `T × 12.632 M` FLOPs for A and `T × 6.341 M` for B. For
`T=1024`, that is 12.935 G versus 6.493 G FLOPs, a saving of 6.442 G. This
counts only cross-attention QK and AV. If all four layers' projected K/V had to
be fetched from off-chip memory once per query, the corresponding *potential*
traffic would be `T × 6.023 MiB` versus `T × 3.023 MiB`: 3 MiB/query saved.
On-chip reuse and fused kernels can make actual off-chip traffic lower, so this
is not a bandwidth measurement or an attainable time saving. As a scale check,
3 MiB divided by an *assumed* 1 TB/s effective bandwidth is 3.15 microseconds
per query; this does not predict kernel or end-to-end latency.

In the **v0.1 implementation**, each reader layer uses a dense `2048 →
1024/2048` K/V projection. Projecting all records once costs 12.935 G FLOPs
for A versus 6.493 G for B across four layers at `N=256`. The apparent
projection saving comes from A physically constructing `[h; 0]` and still
running a dense `2d` projection. An optimized A could compute its K/V from
the first `d` columns only, costing about 6.442 G FLOPs; then A and B have
approximately equal one-time projection work. The two implementations would
no longer use the same dense projection shape, so this is a counterfactual
systems comparison, not the measured pilot.

The cost of **the whole decoder** is a separate question. At `N=256`, the
six-million-FLOP A/B difference in attention per query is small beside the
pilot's tied vocabulary projection alone (`2 × 1024 × 151936 = 311.165 M`
FLOPs/query), before counting four reader MLPs and the producer. If the
cross-attention read occupies fraction `f` of actual decode time and B halves
that time, the best total speedup from that change alone is
`1 / (1 - 0.498f)`: 1.05× at `f=0.10`, 1.18× at `f=0.30`. The value of `f`
has not been measured. Larger contexts improve the arithmetic opportunity:
at `N=2048`, the model predicts 48.023 versus 24.023 MiB of four-layer
projected K/V and 100.712 versus 50.381 M attention FLOPs/query. Functional
quality was tested only at `N=256`; the larger-context row is extrapolation.

## Decision

The address dimension has a real **component-level** exchange rate: equal
logical payload can yield about half the projected K/V, logits and QK+AV work
when output K/V widths remain fixed. The current pilot demonstrates no
end-to-end speedup: it has no incremental generation or persistent projected
KV cache, and its `generate()` reruns the whole model on each growing prefix.
Generated tokens would also enlarge the interface, so the fixed-prompt `T`
formula is an isolated repeated-read scenario, not the implemented generation
path. Actual speedup, bandwidth saving and quality at useful context lengths
remain unmeasured. The cost arithmetic alone does not reopen the failed
remote-context gate or justify a new functional run.
