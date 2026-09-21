# CED IR Research Project

Status: G1 A/B, the frozen-backbone low-rank slot-gate diagnostic, the
zero-training addressing audit, and the supervised oracle-block slot probe are
complete. The matched record-permutation shortcut control and the frozen-Qwen
R0/R1 attention-operator audit are also complete.

Case ID: `CED_IR_LENGTH_WIDTH_PROBE_V01`

This project asks whether token-level long-range address records can be replaced
by smaller persistent interfaces while preserving downstream computation. The
synthetic branch tests a causal CED-style packed interface; the real-model
branch tests oracle attention-operator compression in frozen Qwen3-0.6B. These
are two stages of one project, not separate projects.

The frozen protocol is in `protocol.md`. The repository is self-contained;
`GPT_CONTEXT.md`, `RESULTS.md`, and `ARCHITECTURE.md` provide the concise
research context that was originally spread across the planning materials.

## G1 commands

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/train_g1.py --variant A_TOKEN --output runs/smoke_a --max-updates 2 --device cuda
python scripts/train_g1.py --variant B_PACK2_WIDE2 --output runs/smoke_b --max-updates 2 --device cuda
```

Formal runs use the frozen `configs/g1.json`, one model per RTX 5090, and a
maximum of two device-hours per run. No result is claimed until the frozen test
set and the global-branch ablation have been evaluated.

The A/B runs reached the frozen update-2048 endpoint and evaluated the test set
once. Machine-specific launch receipts and checkpoints are retained locally but
excluded from the repository.

The gate diagnostic verdict is in
`runs/g1_b_learned_slot_c16_u0512_20260921/VERDICT.md`. The identity-preserving
gate did not materially improve B, so the claim that a small linearly exposed
one-bit sub-address alone repairs the frozen B representation is rejected.

The follow-up audit is in `runs/address_audit_v01/AUDIT.md`. It directly
measures coarse localization, physical-slot prediction, compiler mixing, and
phase-stratified performance without updating any parameters. The compiler
largely preserves the two input halves, but the gate remains near 50/50 on the
known physical slot. Coarse localization is moderately concentrated in decoder
layer 0 and diffuse in layer 1, so the failure is not explained by one uniform
"missing slot bit" mechanism.

The directly supervised follow-up is in `runs/slot_probe_v01/PROBE.md`. With
the correct packed block supplied, the same rank-16 bilinear family decodes the
physical slot at essentially 100% held-out accuracy. The frozen representation
therefore makes the slot label decodable. However, the matched permutation
control leaves accuracy at 100% after queries and blocks are deliberately
mismatched across records. Formatting parity is sufficient to saturate the
probe, which does not establish a query-specific address relation.

The real-model branch is in `real_model_audit/`. Its qualified v0.1 run has
formal verdict `INCONCLUSIVE_LOW_MASS`: `8->4` missed both held-out operator
thresholds in all 36 units, while the selected old region carried only 0.568%
median attention mass. Compiler training and model adaptation remain frozen.
