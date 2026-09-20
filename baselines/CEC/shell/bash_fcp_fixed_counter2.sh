#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=3 PROJECT=crossenv_baseline_v2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=4 PROJECT=crossenv_baseline_v2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=5 PROJECT=crossenv_baseline_v2 WANDB_MODE=online