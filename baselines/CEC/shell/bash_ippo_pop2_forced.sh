#! /bin/bash

DEFAULT_GPU=3
GPU_ID="${1:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ $# -eq 0 ]]; then
    SEEDS=(0 1 2 3 4 5)
else
    SEEDS=()
    for SEED_GROUP in "$@"; do
        read -r -a PARSED_SEEDS <<< "${SEED_GROUP}"
        SEEDS+=("${PARSED_SEEDS[@]}")
    done
fi

if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "No seeds were provided" >&2
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    if [[ ! "${SEED}" =~ ^[0-5]$ ]]; then
        echo "Invalid seed: ${SEED} (expected 0-5)" >&2
        exit 2
    fi
done

export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU $CUDA_VISIBLE_DEVICES"
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# cramped_room_9
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2

# # coord_ring_9
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2

# # asymm_advantages_9
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2


# # forced_coord_9
for SEED in "${SEEDS[@]}"; do
    echo "Running forced_coord_9 with seed ${SEED}"
    python baselines/CEC/ippo_general_population_v2.py \
        ENV_KWARGS.layout=forced_coord_9 \
        SEED="${SEED}" \
        WANDB_MODE=online \
        PROJECT=crossenv_baseline_v2 \
        NUM_ENVS=32
done

# # counter_circuit_9
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2
