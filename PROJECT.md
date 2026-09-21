# CED IR Length-Width Probe

Status: G1 A/B, the frozen-backbone low-rank slot-gate diagnostic, the
zero-training addressing audit, and the supervised oracle-block slot probe are
complete.

Case ID: `CED_IR_LENGTH_WIDTH_PROBE_V01`

This project is a bounded architecture probe. It asks whether a causal
CED-style decoder can learn to use a shared long-range interface whose shape is
changed from `N x d` to `N/2 x 2d`, without retaining a hidden token-level
global cache. It is not a post-training migration experiment.

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
therefore contains the slot relation; the earlier LM-trained gate failed to
turn that relation into task recovery.
