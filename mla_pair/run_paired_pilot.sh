#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo 'usage: run_paired_pilot.sh PYTHON MODEL_CACHE TOKENS OUTPUT_ROOT GPU_INDEX PROJECT_ROOT' >&2
  exit 2
fi
python_bin=$1
model_cache=$2
tokens=$3
output_root=$4
gpu_index=$5
project_root=$6
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="$project_root"

"$python_bin" -m mla_pair.train_variant --variant A_TOKEN \
  --cache-dir "$model_cache" --tokens "$tokens" \
  --output "$output_root/a_token" --steps 384 --sequence-length 256 \
  --eval-sequences 32 --rope-dim-per-head 96 --seed 20260923

"$python_bin" -m mla_pair.train_variant --variant B_PAIR \
  --cache-dir "$model_cache" --tokens "$tokens" \
  --output "$output_root/b_pair" --steps 384 --sequence-length 256 \
  --eval-sequences 32 --rope-dim-per-head 96 --seed 20260923

"$python_bin" -m mla_pair.compare_pilot \
  --a "$output_root/a_token/summary.json" \
  --b "$output_root/b_pair/summary.json" \
  --output "$output_root/paired_summary.json"
