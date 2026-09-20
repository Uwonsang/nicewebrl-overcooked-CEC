#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

CUDA_VISIBLE_DEVICES=0 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=0 PROJECT=crossenv_baseline_v2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=0 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=1 PROJECT=crossenv_baseline_v2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=0 python baselines/CEC/fcp_general_fixed.py ENV_KWARGS.layout=counter_circuit_9 SEED=2 PROJECT=crossenv_baseline_v2 WANDB_MODE=online