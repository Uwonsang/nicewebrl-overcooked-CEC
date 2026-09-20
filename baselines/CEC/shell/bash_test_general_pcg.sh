#!/usr/bin/env bash

# Usage: bash_test_general_pcg.sh [GPU] [MODEL|MODEL,...|all] [MODEL_NUM_ENVS] [MODEL_PATH] [OUTPUT_DIR]
# Example: bash_test_general_pcg.sh 0 CEC,CEC_IDAAC 128,256 /app/nas/models/ICRL

gpu="${1:-0}"
model_arg="${2:-all}"
model_num_envs_arg="${3:-256}"
model_path="${4:-}"

if [[ -z "${model_path}" ]]; then
  if [[ -d /app/nas/models/ICRL ]]; then
    model_path=/app/nas/models/ICRL
  elif [[ -d /mnt/nas/wonsang/crossenv_ued/models/ICRL ]]; then
    model_path=/mnt/nas/wonsang/crossenv_ued/models/ICRL
  else
    echo "Could not find the ICRL model directory." >&2
    echo "Pass MODEL_PATH as the fourth argument." >&2
    exit 2
  fi
fi

output_dir="${5:-${model_path%/}/pcg_xp_results}"

all_models=(
  CEC
  CEC_Finetune
  CEC_IDAAC
  CEC_IDAAC_Finetune
  E3T
  FCP
  FCP_Fixed
  IPPO
)

if [[ "${model_arg}" == "all" ]]; then
  models=("${all_models[@]}")
else
  IFS=',' read -r -a models <<< "${model_arg}"
  for model in "${models[@]}"; do
    case " ${all_models[*]} " in
      *" ${model} "*) ;;
      *)
        echo "Unknown model: ${model}" >&2
        echo "Available: all, ${all_models[*]}" >&2
        exit 2
        ;;
    esac
  done
fi

IFS=',' read -r -a model_num_envs_values <<< "${model_num_envs_arg}"
for model_num_envs in "${model_num_envs_values[@]}"; do
  case "${model_num_envs}" in
    32|64|128|256) ;;
    *)
      echo "MODEL_NUM_ENVS values must be selected from: 32, 64, 128, 256" >&2
      exit 2
      ;;
  esac
done

all_layouts=(
  asymm_advantages_9
  coord_ring_9
  counter_circuit_9
  cramped_room_9
  forced_coord_9
)

for model in "${models[@]}"; do
  if [[ "${model}" == "CEC" || "${model}" == "CEC_IDAAC" ]]; then
    selected_envs=("${model_num_envs_values[@]}")
    selected_layouts=(cramped_room_9)
  else
    selected_envs=("${model_num_envs_values[0]}")
    selected_layouts=("${all_layouts[@]}")
  fi

  if [[ "${model}" == "IPPO" ]]; then
    model_seeds="[0,1,2,3,5,6]"
    model_seed_label="0,1,2,3,5,6"
  else
    model_seeds="[0,1,2,3,4,5]"
    model_seed_label="0,1,2,3,4,5"
  fi

  for model_num_envs in "${selected_envs[@]}"; do
    model_output_dir="${output_dir%/}/${model}"
    if [[ "${model}" == "CEC" || "${model}" == "CEC_IDAAC" ]]; then
      model_output_dir="${model_output_dir}/envs${model_num_envs}"
    fi

    for layout in "${selected_layouts[@]}"; do
      echo "PCG XP evaluation: model=${model}, checkpoint_layout=${layout}, seeds=${model_seed_label}, model_num_envs=${model_num_envs}"
      echo "Output directory: ${model_output_dir}"
      CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general_pcg.py \
        "model_name=${model}" \
        "ENV_KWARGS.layout=${layout}" \
        "++MODEL_PATH=${model_path}" \
        "++MODEL_NUM_ENVS=${model_num_envs}" \
        "++MODEL_SEEDS=${model_seeds}" \
        NUM_MODELS=6 \
        XP_ONLY=true \
        "++OUTPUT_DIR=${model_output_dir}"
    done
  done
done
