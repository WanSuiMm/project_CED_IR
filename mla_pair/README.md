# Route 2: pair-content / token-RoPE MLA

This is a **reference implementation**, separate from the earlier CED and
standard-KV A/B pilots. It tests the operator with content cardinality
`N_C = ceil(N/2)` and RoPE-anchor cardinality `N_P = N`. Inputs to
`reference.py` are already projected content queries, RoPE-rotated queries and
keys, and token-level MLA content latents. No pretrained-model result is
claimed here.

`PairCompiler` maps two `d_C` latents to one `d_C` latent. For a completed pair,
the content score is computed once, the two RoPE scores are combined by
`logaddexp`, and the pair contributes one value. A current unpaired even token
is kept raw until its successor arrives. The full-sequence and incremental
implementations agree. This is an exact softmax reduction **only after** the
two anchors share a compiled content latent and value; it does not preserve
two independent content addresses.

For even `N`, persistent per-layer elements are `N/2*d_C + N*d_R`, versus
`N*(d_C+d_R)` for token MLA. With `d_C=512`, `d_R=64`, this is 640 versus 1152
elements per two tokens (44.4% fewer). This is theoretical element counting,
not a measured GPU allocation, kernel time, or end-to-end latency.

## Local correctness checks

From the standalone repository root:

```powershell
$env:PYTHONPATH = "."
python -m unittest mla_pair.test_reference -v
```

The reference operator needs PyTorch. The model qualification script also
needs a Qwen3-capable Transformers installation; see
[`requirements.txt`](requirements.txt). A converted checkpoint may require
additional packages from its conversion implementation.

## First 5090 gate: token-MLA baseline

Qwen3-0.6B-Base is the intended source model, but the public TransMLA examples
do not establish a ready Qwen3 conversion. Obtain and validate a converted
checkpoint separately; do **not** silently substitute a Qwen2.5 checkpoint or
train pair B against a broken token-MLA A. The read-only script checks that
native and converted checkpoints share a tokenizer and reports NLL on identical,
non-overlapping token chunks:

```powershell
python -m mla_pair.qualify_token_mla `
  --native Qwen/Qwen3-0.6B-Base `
  --candidate PATH_TO_CONVERTED_TOKEN_MLA `
  --tokens PATH_TO_FROZEN_TOKEN_STREAM.pt `
  --sequence-length 256 --sequences 32 `
  --output PATH_TO_NEW_RUN_DIR/token_mla_qualification.json
```

The script requires the candidate config to expose `kv_lora_rank` and
`qk_rope_head_dim`; an unchanged GQA model cannot masquerade as token MLA.
The script intentionally records `MEASURED_NOT_YET_JUDGED`: a numerical
acceptance margin must be frozen before treating a converted baseline as
qualified. It does not perform conversion, adaptation, A/B training, or
systems benchmarking. Those are the next implementation stages, contingent on
a usable A checkpoint. Keep both A and B on the same converted initialization,
data, sequence budget, and optimizer. No 10–20M-token run is authorized by this
reference implementation.
