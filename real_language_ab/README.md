# Real-language A/B

This directory is the only active experiment. Read `PROTOCOL.md` first.

- `src/interface_lm/modeling.py`: matched A/B model.
- `scripts/prepare_wikitext.py`: deterministic capped token stream.
- `scripts/train_variant.py`: training, A gate, remote-context evaluation, and
  interface ablation.
- `runs/`: compact summaries; checkpoints and token caches stay off GitHub.
