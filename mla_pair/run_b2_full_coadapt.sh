#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo 'usage: run_b2_full_coadapt.sh PYTHON MODEL_CACHE TOKENS OUTPUT_ROOT GPU_INDEX PROJECT_ROOT' >&2
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

"$python_bin" -m mla_pair.train_full_coadapt --variant A_TOKEN \
  --cache-dir "$model_cache" --tokens "$tokens" \
  --output "$output_root/a_full" --steps 4096 --sequence-length 256 \
  --eval-sequences 32 --eval-every 1024 --rope-dim-per-head 96 \
  --backbone-lr 2e-5 --compiler-lr 2e-4 --seed 20260923

"$python_bin" -m mla_pair.train_full_coadapt --variant B2_HEADWISE \
  --cache-dir "$model_cache" --tokens "$tokens" \
  --output "$output_root/b2_headwise" --steps 4096 --sequence-length 256 \
  --eval-sequences 32 --eval-every 1024 --rope-dim-per-head 96 \
  --backbone-lr 2e-5 --compiler-lr 2e-4 --seed 20260923

"$python_bin" -m mla_pair.compare_b2 \
  --a "$output_root/a_full/summary.json" \
  --b "$output_root/b2_headwise/summary.json" \
  --output "$output_root/paired_summary.json"
