#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=cramped_room_9 SEED=0 SCALE_CLIP_EPS=True WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=cramped_room_9 SEED=0 WANDB_MODE=online

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=coord_ring_9 SEED=0 SCALE_CLIP_EPS=True WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=coord_ring_9 SEED=0 WANDB_MODE=online

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=asymm_advantages_9 SEED=0 SCALE_CLIP_EPS=True WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=asymm_advantages_9 SEED=0 WANDB_MODE=online

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=forced_coord_9 SEED=0 SCALE_CLIP_EPS=True WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=forced_coord_9 SEED=0 WANDB_MODE=online

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=counter_circuit_9 SEED=0 SCALE_CLIP_EPS=True WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py  ENV_KWARGS.layout=counter_circuit_9 SEED=0 WANDB_MODE=online