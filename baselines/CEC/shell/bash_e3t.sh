#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
gpu=$1
layout=$2

SEEDS=(0 1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py \
      ENV_KWARGS.layout=${layout} \
      SEED=$seed \
      WANDB_MODE=online
done
