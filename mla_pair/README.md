# Route 2: pair-content / token-RoPE MLA

This is a new route within `project_CED_IR`, separate from the earlier CED and
standard-KV A/B pilots. It asks whether content cardinality can become
`N_C = ceil(N/2)` while RoPE-anchor cardinality stays `N_P = N`.
`reference.py` is the independent algebraic oracle; `qwen3_split.py` is the
real-Qwen3 A/B implementation.

`PairCompiler` maps two `d_C` latents to one `d_C` latent. For a completed pair,
the content score is computed once, the two RoPE scores are combined by
`logaddexp`, and the pair contributes one value. A current unpaired even token
is kept raw until its successor arrives. The full-sequence and incremental
implementations agree. This is an exact softmax reduction **only after** the
two anchors share a compiled content latent and value; it does not preserve
two independent content addresses.

For even `N`, persistent per-layer elements are `N/2*d_C + N*d_R`, versus
`N*(d_C+d_R)` in A. The implemented quality-first split uses 96 RoPE dimensions
per KV head: `d_C=1280`, `d_R=768`. A stores 2048 scalars/token/layer (the
same width as native Qwen3 GQA); B stores 2816 scalars per two completed
tokens/layer, a 31.25% reduction. B adds 91,786,240 compiler parameters over
28 layers. This is **not** the 512+64 low-rank TransMLA setting, and there is
no fused kernel or measured decode latency.

## Local correctness checks

From the standalone repository root:

```powershell
$env:PYTHONPATH = "."
python -m unittest mla_pair.test_reference mla_pair.test_qualification mla_pair.test_qwen3_split -v
```

The model runners need a Qwen3-capable Transformers installation and PEFT;
see [`requirements.txt`](requirements.txt).

## Qualified A, and what it does not prove

The public [TransMLA](https://github.com/MuLabPKU/TransArch/tree/main/TransMLA_NeurIPS_2025)
converter does not list Qwen3 in its supported DeepSeek-style model types, and
its standard Transformers attention caches expanded K/V rather than the latent.
Therefore `qwen3_split.py` preserves native Qwen3 Q/K/V projections and
q_norm/k_norm, moves 96/128 key dimensions per KV head to the RoPE-bearing
branch, and stores the remaining K dimensions plus V as content. There is no
low-rank width compression. This avoids adding a second aggressive compression
question to the A/B comparison. `qwen3_cache.py` implements the B pair cache;
the A cache is a standard HF DynamicCache whose runtime tensors hold split
content and RoPE latents.

On the existing WikiText-103 token stream, the 5090 A gate used 32 nonoverlapping
256-token validation chunks beginning at offset 8192, separate from the first
smoke tokens. Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd` had NLL `2.62164`; the
split A had `2.62057`, delta `−0.00107` nats/token. Full/cache KL was
`0.000505` native and `0.000411` A. Runtime A cache grew by 2048 BF16
scalars/token/layer. A passes the first-pass quality/parity/cache gate. Raw
per-chunk values are retained in the local run directory, not in GitHub.

The original generic `qualify_token_mla.py` remains available for a separately
converted checkpoint; the actual Qwen3 split run uses `run_quality_first.py`:

```powershell
python -m mla_pair.run_quality_first `
  --cache-dir MODEL_CACHE --tokens FROZEN_TOKEN_STREAM.pt `
  --output NEW_RUN_DIR/summary.json
```

The NLL margin was `+0.10`. The first absolute BF16 parity threshold
`KL<=0.001` proved unsuitable because **native** Qwen3 itself exceeded it in
the 16-token smoke (`0.001158`). Before the independent 32×256 A run, the
engineering parity rule was changed to extra KL relative to native `<=0.001`;
cache growth still had to equal `d_C+d_R`. This numerical rule change is
recorded explicitly, not treated as a scientific success discovered post hoc.

## B pilot result: small adaptation screen failed

`smoke_pair_runtime.py` verified a 16-token B cache with exactly 630,784
persistent scalars across all 28 layers and BF16 bytes of 1,261,568, matching
the pair-content formula. Its full/cache KL was `0.001235`; this is an
engineering smoke, not B quality evidence. One-step A/B training smokes pass.
A matched pilot of 384 updates and 98,304 distinct training tokens per arm
completed on the 5090. `train_variant.py` uses the same split-Qwen A
initialization, token order, LoRA configuration and optimizer settings in both
arms; only B adds the FP32 pair compiler. The compiler's lower learning rate
is `1e-6` in both run configurations (A has no compiler).

| Variant | Initial validation NLL | Final validation NLL | Trainable parameters |
| --- | ---: | ---: | ---: |
| A, token content | 2.77078 | 2.43202 | 5,046,272 |
| B, paired content | 5.33054 | 3.34773 | 96,832,512 |

The paired B−A final NLL was `+0.91571` nats/token on 32 held-out 256-token
sequences (paired bootstrap 95% CI `[+0.83633, +1.00060]`). This is outside
the preset `+0.10` screen margin. B improved substantially from its poor
initialization, but did not catch A at this budget. Its additional 91,786,240
parameters are the pair compilers. The measured persistent reference-cache
ratio was `0.6875` for even sequence lengths; this is not a GPU latency result.
The 10–20M-token endpoint is on hold rather than silently launched. The result
rejects this **small-budget adaptation screen**, not every possible trained
pair-content architecture. Raw per-sequence summaries and checkpoints remain
in the local/remote run directory, excluded from GitHub because the generated
files contain machine-specific paths.
