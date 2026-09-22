# Qwen-to-CED migration G0/G1 protocol

## Question

Can a pretrained causal language model be migrated to a token-aligned shared
global interface without losing ordinary language modeling quality or genuine
use of remote context?

This stage does **not** test the shorter/wider interface. It qualifies the
architecture migration required before that comparison is scientifically
interpretable. Earlier synthetic, probe, and frozen-oracle evidence is retained
unchanged.

## Models

- Initialization: `Qwen/Qwen3-0.6B-Base`.
- `C_QWEN_CPT`: unmodified Qwen, continued on the same token stream.
- `A_TOKEN_CED`: layers 0--13 form a causal sliding-window self-decoder with
  window 64. Its final state produces one shared, token-aligned global K/V
  memory. Layers 14--27 retain their Q/O projections, norms, and MLPs but use
  causal cross-attention to that shared memory. There is no upper-layer
  time-axis self-attention and no full-history encoder cache.
- Shared K/V and its input norm are initialized from original layer 14. This is
  a warm start, not a function-preserving conversion.
- Each newly introduced cross-attention residual has a trainable scalar gate
  initialized to 1%. This keeps the incompatible shared-memory path a small
  perturbation at update zero while preserving gradient flow. It is not an old
  Qwen KV bypass; no upper-layer self-attention path exists.
- All parameters are trainable. No LoRA or frozen compiler is substituted.

For a sequence of length `N`, persistent autoregressive state must contain only
14 local encoder windows plus one `N`-token global K/V memory. Batch training
may retain activations for backpropagation; those are not inference state.

## G0 implementation gate

Before training, `A_TOKEN_CED` must pass on real Qwen weights:

1. future-token edits do not change earlier logits;
2. batched full-sequence logits match the actual inference route--parallel
   prefill followed by cached continuation--in FP32 (`max_abs <= 1e-3`,
   `mean_abs <= 1e-4`). The deployed BF16 route separately requires mean
   next-token KL <= `1e-3` and top-1 agreement >= 99%. Replaying the entire
   prompt one token at a time is retained as a numerical stress diagnostic,
   not the deployment parity endpoint;
3. changing a token more than 882 positions back can affect the output only
   through shared global memory; ablating that memory changes long-context
   logits;
4. cache accounting reports no original per-layer full-history K/V and no
   retained full encoder-state copy;
5. finite forward, backward, optimizer step, save, and reload all succeed.

Failure is `INVALID_CED_IMPLEMENTATION`; no scientific run is allowed.

## Data and optimization

- Training length: 4096.
- Mixture: 75% natural text and 25% source code by target tokens.
- Documents are deterministically assigned to train/validation/test before
  packing. EOS separates documents. C and A consume the identical packed token
  order, target-token count, optimizer schedule, and evaluation examples.
- First formal budget: 20M target tokens for C and 20M for A.
- Objective: next-token causal LM loss only. Any later use of output
  distillation requires a new protocol version and matched use in both models.
- Primary independent unit for evaluation is a source document, not a token.

The exact dataset revisions, hashes, optimizer settings, realized token count,
parameter count, FLOPs proxy, GPU, duration, and peak memory are recorded by
the run manifest.

## Evaluation

Report natural-text and code NLL separately. For remote-context dependence,
keep the most recent 1024 tokens fixed and replace only the older prefix with a
same-domain, approximately length-matched prefix selected without test labels.
For model `X`, define

```text
G_X = NLL_X(mismatched old context) - NLL_X(correct old context).
```

Report paired document-level intervals for `L_A-L_C`, `G_C`, `G_A`, and the
retention ratio `G_A/G_C`. Evidence-location, numeric-reference, and code-use
tasks are secondary endpoint families and must be frozen before test access.

## Ordered decision

1. `INVALID_CED_IMPLEMENTATION` if G0 fails.
2. `MIGRATION_INCOMPLETE` if A fails its validation-frozen non-inferiority
   margin against C on either text or code, or does not retain the
   validation-frozen fraction of positive remote-context gain.
3. `PASS_TOKEN_CED_MIGRATION` only if both language modeling and remote-context
   conditions pass on the untouched test documents.

Only `PASS_TOKEN_CED_MIGRATION` authorizes a new frozen protocol comparing
`N x d` against `N/2 x 2d`. No result here is a systems-speedup claim.
