#! /bin/bash
srun --gres=gpu:1 --nodelist=dgx-a100-n3 --ntasks=1 --pty -J ws1 singularity exec --cleanenv --nv /scratch/cilab/uwonsang/jax_ued_v4.sif bash -lc '
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=0 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=1 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=0 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=2 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=0 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=3 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=1 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=4 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=1 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=6 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=5 UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 ENV_KWARGS.incentivize_strat=1 WANDB_MODE=online'