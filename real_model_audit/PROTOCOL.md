# R0/R1 protocol v0.1 (preregistered)

## Question

For a frozen pretrained Qwen3-0.6B decoder, does a completed block of `r=8`
post-RoPE KV records admit a query-conditioned oracle representation using
`c in {1,2,4}` latent records that generalizes to a disjoint, later contiguous
span of real future queries?

The object being approximated for block `B` is

```text
log Z_B(q) = log sum_i exp(q^T k_i)
mu_B(q)    = sum_i softmax(q^T K_B)_i v_i
```

`mu_B=N_B/Z_B` is used instead of raw `N_B`, preventing block mass from
dominating value-shape error. The per-query fitting loss is weighted by the
block's true attention mass in the complete causal context.

## Frozen design

- Model: `Qwen/Qwen3-0.6B-Base`; no parameter updates.
- Domains: prose, Python code, deterministic structured/exact records.
- Independent examples: 4 per domain, each from a distinct source or seed.
- Sequence length: 2048 tokens.
- Layers: zero-based decoder layers 4, 16, 27.
- GQA: each KV head is fitted jointly against the two query heads it serves.
- Completed region: eight adjacent 8-token blocks (`64` original records).
- Recency control: at least 128 exact tokens separate the compressed region
  from the first fitting query.
- Fit queries: a contiguous 64-token span.
- Test queries: the immediately following, disjoint contiguous 64-token span.
- Oracle sizes: `8->1`, `8->2`, `8->4`; three deterministic restarts.
- Qualification control: exact `8->8` identity reconstruction.

The latent records are optimized directly in post-RoPE key space and include a
scalar log-mass bias per record. This is an unrestricted local oracle, not yet a
machine-realizable codec.

## Metrics

R0 reports the entropy effective rank of the real future-query score matrix
`Q_future K_B^T`.

R1 reports on held-out queries:

1. attention-mass-weighted `log Z_B` RMSE;
2. attention-mass-weighted relative `mu_B` error;
3. relative global head-output error after replacing all 64 selected records;
4. relative post-`o_proj` attention-output error;
5. recomputed downstream next-token KL and delta NLL after a real single-layer
   hook intervention. Teacher queries are not reused for downstream evaluation.

The independent statistical unit is one source sequence at one layer. Heads,
blocks, and query positions are repeated measurements, not independent samples.

## Qualification and decision

The implementation is qualified only if both hold for every audited layer:

- manual exact attention versus the model attention output: relative error <= 1e-2;
- `8->8` identity replacement: relative post-projection error <= 1e-4.

The primary R1 gate is evaluated for `8->4`. It is a **PASS** only if:

- median held-out weighted `log Z` RMSE <= 0.10;
- median held-out weighted relative `mu` error <= 0.10;
- at least 75% of sequence-layer units satisfy both limits; and
- median true attention mass of the replaced 64-token region is >= 0.02.

If the mass condition fails, the result is `INCONCLUSIVE_LOW_MASS`, never a
pass. If qualification fails, it is `INVALID_IMPLEMENTATION`. Otherwise failure
of `8->4` is `STOP_ORACLE_8_TO_4_FAIL`: do not build a compiler. `8->2` is
reported as a stronger opportunity signal but is not required for the gate.

Global attention error, KL, and delta NLL are causal impact checks, not a way to
override a failed operator gate. R2/R3 are frozen until R1 passes.
