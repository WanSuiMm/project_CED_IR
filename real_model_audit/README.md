# Real-model attention-operator audit

This directory is the real-pretrained-model branch of the **same CED IR
project**. It follows the synthetic length-width probe and asks whether the
attention operator supplied by eight real Qwen3 KV records can be represented
by fewer latent records on disjoint future queries.

It is not a separate research project. Its role in the shared CED IR evidence
chain is:

```text
synthetic packed-interface failure
-> real pretrained attention-operator existence audit
-> only if the oracle passes: amortized compiler and adaptation
```

## Read in this order

1. [`RESULTS.md`](RESULTS.md) — interpretation and claim boundary.
2. [`PROTOCOL.md`](PROTOCOL.md) — frozen R0/R1 design and stop conditions.
3. [`runs/r01_qwen3_06b_v03/RESULTS.md`](runs/r01_qwen3_06b_v03/RESULTS.md) — generated aggregate.
4. [`runs/r01_qwen3_06b_v03/summary.json`](runs/r01_qwen3_06b_v03/summary.json) — sanitized machine-readable evidence.
5. [`scripts/run_r01_gate.py`](scripts/run_r01_gate.py) and
   [`src/aoc/oracle.py`](src/aoc/oracle.py) — implementation.

The formal v0.1 verdict is `INCONCLUSIVE_LOW_MASS`. The tested `8->4` oracle
met neither operator-error threshold in any of 36 sequence-layer units, but the
fixed old region received only 0.568% median attention mass. R2 compiler
training and R3 adaptation remain frozen.

## Tests

From this directory:

```bash
python -m pytest tests -q
```
