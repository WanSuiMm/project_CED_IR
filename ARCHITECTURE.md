# Architecture and code map

## Data flow

For input tokens `x_0, ..., x_{N-1}`, a causal local encoder produces states

```text
tokens -> embedding -> local causal encoder -> H in R^(N x d)
```

No encoder layer has a global token cache. With block size `r` and persistent
width multiplier `s`, the compiler forms

```text
Z = C_{r,s}(H) in R^(floor(N/r) x s*d).
```

`A_TOKEN` uses `r=1,s=1`. `B_PACK2_WIDE2` uses `r=2,s=2` and an identity-initialized
square compiler, so two adjacent encoder states are concatenated rather than
dimensionally compressed.

The shared global memory is

```text
K = Z W_K in R^(M x d)
V = Z     in R^(M x s*d).
```

Each decoder layer has its own query and output projections but shares the same
persistent K/V. Every decoder layer applies local causal attention, global IR
read, and a SwiGLU MLP.

## Strict causal visibility

Block `j` ends at original source position

```text
e_j = (j+1)r - 1.
```

Query position `t` may attend only blocks with `e_j <= t`. An incomplete block
is held in a tail buffer and is not committed to global memory. The local path
continues to cover recent tokens.

The tests check future-token edits, future embedding gradients, empty global
memory, streaming/offline cache equivalence, and exact replay continuation.

## Tested packed-wide reader

For each attention head, one coarse scalar weight is assigned to each packed
block:

```text
beta_tj = softmax_j(q_t^T k_j).
```

The head value has width `s*a`, where `a=d/h`. For B, both source-slot payloads
share the same coarse block weight. The concatenated wide output is projected
back to the `d`-wide decoder stream only after aggregation.

This reader is intentionally concrete and GPU-regular. It is not claimed to be
an exact regrouping of token-level attention: arbitrary query-dependent
within-block selection generally requires additional address structure.

## Frozen-backbone slot gate

The diagnostic gate keeps the original coarse weights and splits each B value
into two head-local payloads. It computes low-rank slot logits

```text
ell_tja = <U q_t, V_a z_j^(a)> / sqrt(c),  c=16
g_tj = softmax_a(ell_tj).
```

The effective wide value preserves its two-slot layout:

```text
z_tj_tilde = [2 g_tj0 z_j^(0); 2 g_tj1 z_j^(1)].
```

`V_a` is initialized to zero, making `g=(1/2,1/2)` and reproducing B exactly.
Only `gate_query` and `gate_slot_proj` are trainable in the diagnostic. The
completed B representation, coarse keys and original decoder remain frozen.

## Cache-size and compute boundary

The persistent global K+V element ratio relative to A is

```text
(1+s)/(2r).
```

For `r=s=2`, this is 3/4: the theoretical global cache reduction is 25%, not
50%. The value aggregation is wider and the output projection is larger. For
the tested reader, the idealized global-read plus output-projection FLOP saving
is positive only when `N > 2d`. These formulas are not wall-clock claims.

## Source-code map

- `RMSNorm`, `SwiGLU`, `LocalAttention`: core local blocks
- `IRCompiler`: block packing, shared key construction and value layout
- `GlobalRead`: strict block mask, wide-value aggregation and optional gate
- `CEDIRModel.forward`: full training path
- `CEDIRModel.stream_step`: token-by-token cache path
- `CEDIRModel.prefill_exact_replay`: bounded replay prefill path
- `synthetic.make_example`: deterministic leakage-resistant retrieval task
- `train_g1.evaluate`: frozen evaluation and global-IR intervention
- `train_gate.py`: qualified frozen-backbone gate diagnostic
