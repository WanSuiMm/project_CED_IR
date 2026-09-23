# Standard-KV A/B v0.2 result

```text
primary_nll_gate        = STANDARD_KV_NONINFERIOR
interface_use           = PRESENT_IN_BOTH
remote_context_gate     = UNQUALIFIED
systems_measurement     = NOT_RUN
```

The predeclared 64-sequence paired NLL gate passes. B keeps half as many
addressable records with the **same projected K and V widths** as A, and has
slightly lower held-out NLL. Thus the v0.1 B NLL signal did not require its
doubled reader-facing V width in this 256-step pilot. This result is limited to
ordinary language modeling at sequence length 256.

| Metric | A token, standard KV | B pair, standard KV |
|---|---:|---:|
| Parameters | 289,821,696 | 289,821,696 |
| Records at N=256, excluding null | 256 | 128 |
| Logical persistent BF16 bytes, excluding null | 524,288 | 524,288 |
| Training-time final NLL, 32 sequences | 6.6792 | 6.6093 |
| Paired-eval NLL, 64 sequences | 6.4422 | 6.3628 |
| Interface ablation delta, 64 sequences | +4.3674 | +4.3029 |
| Remote-context gain, 64 sequences | -0.0015 [-0.0090, 0.0058] | +0.0089 [-0.0024, 0.0228] |

Paired `B−A` NLL is **-0.0794**, bootstrap 95% CI
**[-0.0980, -0.0628]**. Its upper bound is below the preregistered `+0.10`
non-inferiority margin. These are 64 non-overlapping validation chunks from a
single token stream; the bootstrap treats chunks as independent and does not
measure between-document or between-seed variation. A improved from 17.7497
to 6.6792 NLL, and its interface ablation exceeds the `+0.01` substrate
threshold. Both variants
were trained for 256 updates on the same WikiText-103 token order, seed,
optimizer, reader count and masks. The architectural intervention from v0.1
was `value_head_multiplier=1` in both A and B; within v0.2, only record
construction differs. The 32-sequence point verdicts are retained in the
training summaries; the 64-sequence paired gate is the primary decision.

At `N=256`, a *hypothetical* four-layer projected KV cache with standard
widths is 4.016 MiB for A versus 2.016 MiB for B; QK+AV is 8.421 versus
4.227 M FLOPs per future query. O projection has the same width in both.
These are component counts, not measured cache allocation or decode speed.
The current generation code reruns the growing full sequence. Generation
remains repetitive, and neither variant has an individually positive 95%
interval for the remote-prefix test. This pilot does not establish precise
long-range retrieval, useful long-context quality, or end-to-end speedup.

Evidence: `runs/a_token_standard_kv_s256_l256_v02/summary.json`,
`runs/b_pair_standard_kv_s256_l256_v02/summary.json`, and
`runs/paired_eval_n64_standard_kv_v02.json`. The run used the fixed source
commit `8ca2459294a9db949f5ae2e9738c972cfa6a968b` on one RTX 5090;
checkpoints and logs remain with the server run and are not in GitHub.
