#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: run_standard_kv_v02.sh RUN_ROOT TOKEN_CACHE MODEL_CACHE PHYSICAL_GPU CONDA_BIN SOURCE_REV" >&2
  exit 2
fi

run_root="$1"
token_cache="$2"
model_cache="$3"
physical_gpu="$4"
conda_bin="$5"
source_rev="$6"
cd "$run_root"
mkdir -p runs logs
export CUDA_VISIBLE_DEVICES="$physical_gpu"

printf 'host=%s\npid=%s\nphysical_gpu=%s\nrun_root=%s\nsource_rev=%s\nstarted_utc=%s\n' \
  "$(hostname)" "$$" "$physical_gpu" "$run_root" "$source_rev" "$(date -u +%FT%TZ)" \
  > runs/launch_receipt.txt

"$conda_bin" run --no-capture-output -n latent-hub python scripts/train_variant.py \
  --variant A_TOKEN --value-head-multiplier 1 \
  --steps 256 --sequence-length 256 --producer-layers 4 --reader-layers 4 \
  --local-window 16 --seed 20260922 \
  --cache-dir "$model_cache" --data "$token_cache" \
  --output runs/a_token_standard_kv_s256_l256_v02 \
  > logs/a_standard_kv.log 2>&1

if ! grep -q '"verdict": "A_STANDARD_TRAINED"' \
  runs/a_token_standard_kv_s256_l256_v02/summary.json; then
  echo "A standard-KV substrate did not qualify; see A summary" >&2
  exit 20
fi

"$conda_bin" run --no-capture-output -n latent-hub python scripts/train_variant.py \
  --variant B_PAIR_WIDE --value-head-multiplier 1 \
  --steps 256 --sequence-length 256 --producer-layers 4 --reader-layers 4 \
  --local-window 16 --seed 20260922 \
  --cache-dir "$model_cache" --data "$token_cache" \
  --reference-summary runs/a_token_standard_kv_s256_l256_v02/summary.json \
  --output runs/b_pair_standard_kv_s256_l256_v02 \
  > logs/b_standard_kv.log 2>&1

"$conda_bin" run --no-capture-output -n latent-hub python scripts/compare_checkpoints.py \
  --value-head-multiplier 1 --count 64 --sequence-length 256 \
  --cache-dir "$model_cache" --data "$token_cache" \
  --a-checkpoint runs/a_token_standard_kv_s256_l256_v02/checkpoint.pt \
  --b-checkpoint runs/b_pair_standard_kv_s256_l256_v02/checkpoint.pt \
  --output runs/paired_eval_n64_standard_kv_v02.json \
  > logs/paired_eval_standard_kv.log 2>&1

printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >> runs/launch_receipt.txt
echo "STANDARD_KV_V02_COMPLETE"
