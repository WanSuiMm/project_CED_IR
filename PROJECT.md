# CED IR Research Project

Status: G1 A/B, the frozen-backbone low-rank slot-gate diagnostic, the
zero-training addressing audit, and the supervised oracle-block slot probe are
complete. The matched record-permutation shortcut control and the frozen-Qwen
R0/R1 plus final R1b attention-operator audits are also complete. A subsequent
Qwen-to-CED migration attempt stopped at G0 with `INVALID_CED_IMPLEMENTATION`;
its function-preserving v0.2 replacement found bridge signal but failed the
shared-memory qualification. No 20M-token migration training was run.

The latest direct real-language standard-KV A/B pilot passes its paired NLL
non-inferiority gate at 256 tokens; remote-context use and decode speed remain
unqualified. See `real_language_ab/RESULTS_STANDARD_KV_v0.2.md`.

Case ID: `CED_IR_LENGTH_WIDTH_PROBE_V01`

This project asks whether token-level long-range address records can be replaced
by smaller persistent interfaces while preserving downstream computation. The
synthetic branch tests a causal CED-style packed interface; the real-model
branch tests oracle attention-operator compression in frozen Qwen3-0.6B; the
latest branch tests a direct Qwen-to-CED warm start. These are three stages of
one project, not separate projects.

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

The real-model branch is in `real_model_audit/`. Its final R1b audit passed the
exact implementation controls but received `ORACLE_OPTIMIZER_UNQUALIFIED`:
non-identity `8->8` recovery failed on independent test queries, while even the
calibration-selected region retained only 0.391% median test attention mass.
This branch is stopped; compiler training and model adaptation remain frozen.

The later architecture-migration stage is in `migration/`. Its causal and
cache-accounting controls pass and its FP32 structural smoke is consistent, but
the formal BF16 prefill-plus-continuation parity KL is `0.02442`, above the
frozen `0.001` limit. The migration stage is archived at G0; G1 and the
shorter/wider comparison remain unrun.

`migration_v02/` replaces the invalid warm start with exact Qwen/CED homotopy
controls. The start morphism and FP32 endpoint structure pass. However, the
shared-memory bridge overfits the tiny fit set, leaves `0.9218` normalized
held-out error, and collapses at full removal of upper self-attention. The
formal v0.2 status is `BRIDGE_SIGNAL_PRESENT_NOT_QUALIFIED`; larger migration
remains unauthorized.
