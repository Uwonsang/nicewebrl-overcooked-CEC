#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# cramped_room_9
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=cramped_room_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2

# coord_ring_9
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=coord_ring_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2

# asymm_advantages_9
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=asymm_advantages_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2


# forced_coord_9
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=forced_coord_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2

# counter_circuit_9
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=0 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=1 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=2 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=3 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=4 WANDB_MODE=online PROJECT=crossenv_baseline_v2
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py ENV_KWARGS.layout=counter_circuit_9 SEED=5 WANDB_MODE=online PROJECT=crossenv_baseline_v2
