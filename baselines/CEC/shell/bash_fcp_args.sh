#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

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
    CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/fcp_general.py \
      ENV_KWARGS.layout="${layout}" \
      SEED="${seed}" \
      NUM_ENVS=32 \
      "${FCP_PATH_ARGS[@]}" \
      WANDB_MODE=online
done
