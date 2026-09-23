# Real-language A/B

Read `RESULTS_STANDARD_KV_v0.2.md` and `PROTOCOL_STANDARD_KV_v0.2.md` for the
current direct A/B comparison. The original wide-V pilot remains under
`PROTOCOL.md` and `RESULTS.md`.

- `src/interface_lm/modeling.py`: matched A/B model.
- `scripts/prepare_wikitext.py`: deterministic capped token stream.
- `scripts/train_variant.py`: training, A gate, remote-context evaluation, and
  interface ablation.
- `runs/`: compact summaries; checkpoints and token caches stay off GitHub.
- `COST_MODEL.md`: address-width arithmetic and its implementation limits.
