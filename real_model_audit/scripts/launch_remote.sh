#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:?project directory required}"
GPU_INDEX="${2:?GPU index required}"
RUN_NAME="${3:-r01_qwen3_06b_v01}"
HF_CACHE="${4:?external Hugging Face cache directory required}"
RUN_DIR="$PROJECT_DIR/runs/$RUN_NAME"

mkdir -p "$HF_CACHE" "$RUN_DIR"
export HF_HOME="$HF_CACHE"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

exec conda run --no-capture-output -n latent-hub \
  python "$PROJECT_DIR/scripts/run_r01_gate.py" \
  --cache-dir "$HF_CACHE" \
  --output "$RUN_DIR"
