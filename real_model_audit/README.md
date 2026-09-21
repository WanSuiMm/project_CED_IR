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
2. [`R1B_PROTOCOL.md`](R1B_PROTOCOL.md) — frozen final audit and decision order.
3. [`runs/r1b_high_mass_v01/RESULTS.md`](runs/r1b_high_mass_v01/RESULTS.md) — generated final aggregate.
4. [`runs/r1b_high_mass_v01/summary.json`](runs/r1b_high_mass_v01/summary.json) — sanitized per-unit evidence.
5. [`scripts/run_r1b_gate.py`](scripts/run_r1b_gate.py) and
   [`src/aoc/oracle.py`](src/aoc/oracle.py) — implementation.
6. [`PROTOCOL.md`](PROTOCOL.md) and `runs/r01_qwen3_06b_v03/` — preserved v0.1 protocol and evidence.

The terminal R1b verdict is `ORACLE_OPTIMIZER_UNQUALIFIED`. Exact reconstruction
qualified, but the non-identity `8->8` optimizer control failed on held-out
queries. The calibration-selected region also retained only 0.391% median test
attention mass. The local operator-compression branch is stopped; R2 compiler
training and R3 adaptation remain frozen.

## Tests

From this directory:

```bash
python -m pytest tests -q
```
