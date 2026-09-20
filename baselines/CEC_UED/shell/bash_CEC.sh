#! /bin/bash
gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=0 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=1 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=3 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=4 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=5 WANDB_MODE=online
