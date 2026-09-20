#! /bin/bash
# Trains BC on one layout across 6 seeds (0-5).
# Usage: bash baselines/human_proxy/shell/bash_bc_sweep.sh <gpu_id> <layout>
# layouts: cramped_room coord_ring asymm_advantages forced_coord counter_circuit

gpu=$1
layout=$2

# orthogonal() init (QR decomposition) uses cuSolver, which can fail to grab a handle
# if JAX has already preallocated most of the GPU memory.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

SEEDS=(0 1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu} python -m baselines.human_proxy.bc_agent \
      LAYOUT=${layout} \
      SEED=$seed \
      WANDB_MODE=online
done
