#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=0 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=1 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=2 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=3 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=4 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/fcp_general_fixed.py --config-name=fcp_final_toy.yaml SEED=5 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
