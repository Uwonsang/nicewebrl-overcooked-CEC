#! /bin/bash

DEFAULT_GPU=3
GPU_ID="${1:-${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Using GPU $CUDA_VISIBLE_DEVICES"
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)

# cramped_room_9
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=cramped_room_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64

# # coord_ring_9
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=coord_ring_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64

# # asymm_advantages_9
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# # CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=asymm_advantages_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64

# # forced_coord_9
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32
python baselines/CEC/e3t.py ENV_KWARGS.layout=forced_coord_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=32

# # counter_circuit_9
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
# CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py ENV_KWARGS.layout=counter_circuit_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2 NUM_ENVS=64
