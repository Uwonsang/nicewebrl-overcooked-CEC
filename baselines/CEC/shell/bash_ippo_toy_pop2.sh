#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=1 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=2 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=3 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=4 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=5 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online

CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=1 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=2 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=3 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=4 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=5 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online