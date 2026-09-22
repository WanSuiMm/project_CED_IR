# Homotopy migration v0.2

This is a new protocol, not a reinterpretation of the stopped `migration/`
run. Start with `RESULTS.md`, then read `PROTOCOL.md`.

- `src/ced_homotopy/modeling.py`: two-knob Qwen/CED homotopy, bridge objective,
  and endpoint cache implementation.
- `scripts/qualify_h0_h1.py`: native morphism and final-endpoint controls.
- `scripts/qualify_h2_bridge.py`: frozen-backbone bridge pilot.
- `runs/`: sanitized qualification summaries only; checkpoints and model caches
  remain excluded.

No 20M-token job is authorized until all three gates pass.

Current result: H0 and FP32 endpoint structure pass, but BF16 endpoint
conditioning fails and the 512-step shared-memory bridge remains unqualified.
The larger migration is stopped.
