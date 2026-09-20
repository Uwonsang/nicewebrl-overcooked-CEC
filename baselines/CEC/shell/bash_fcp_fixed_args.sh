#!/usr/bin/env bash

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: bash $0 <gpu> <layout> [\"seed ...\"]" >&2
    exit 1
fi

gpu=$1
layout=$2
shift 2

if (( $# > 1 )); then
    echo "Seeds must be passed as one quoted string, e.g. \"0 1 2\"." >&2
    exit 1
elif (( $# == 1 )); then
    read -r -a SEEDS <<< "$1"
else
    SEEDS=(0 1 2 3 4 5)
fi

FCP_PATH_ARGS=()
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
    FCP_PATH_ARGS+=(FCP_filepath="${CHECKPOINT_ROOT}")
fi

echo "Using GPU ${gpu}, layout ${layout}, seeds: ${SEEDS[*]}"
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
    echo "Using FCP checkpoint pool: ${CHECKPOINT_ROOT}"
fi

for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/fcp_general_fixed.py \
      ENV_KWARGS.layout="${layout}" \
      SEED="${seed}" \
      NUM_ENVS=32 \
      "${FCP_PATH_ARGS[@]}" \
      PROJECT=crossenv_baseline_v2 \
      WANDB_MODE=online
done
