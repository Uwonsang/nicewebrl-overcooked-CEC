#!/usr/bin/env bash

# Usage: bash_test_general_human_proxy.sh [GPU] [MODEL|MODEL,...|all] [MODEL_NUM_ENVS] [MODEL_PATH] [OUTPUT_DIR]
# Example: bash_test_general_human_proxy.sh 0 CEC_IDAAC 128,256 /app/nas/models/ICRL

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

output_dir="${5:-${model_path%/}/human_proxy_results}"

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

layouts=(
  asymm_advantages_9
  coord_ring_9
  counter_circuit_9
  cramped_room_9
  forced_coord_9
)

for layout in "${layouts[@]}"; do
  for model in "${models[@]}"; do
    if [[ "${model}" == "CEC" || "${model}" == "CEC_IDAAC" ]]; then
      selected_envs=("${model_num_envs_values[@]}")
    else
      selected_envs=("${model_num_envs_values[0]}")
    fi
    for model_num_envs in "${selected_envs[@]}"; do
      if [[ "${model}" == "IPPO" ]]; then
        model_seeds="[0,1,2,3,5,6]"
        model_seed_label="0,1,2,3,5,6"
      else
        model_seeds="[0,1,2,3,4,5]"
        model_seed_label="0,1,2,3,4,5"
      fi
      model_output_dir=""
      if [[ -n "${output_dir}" ]]; then
        model_output_dir="${output_dir%/}/${model}"
        if [[ "${model}" == "CEC" || "${model}" == "CEC_IDAAC" ]]; then
          model_output_dir="${model_output_dir}/envs${model_num_envs}"
        fi
      fi

      echo "Human-proxy evaluation: layout=${layout}, model=${model}, model_num_envs=${model_num_envs}, model seeds=${model_seed_label}, BC seeds=0-4"
      if [[ -n "${model_output_dir}" ]]; then
        echo "Output directory: ${model_output_dir}"
      fi

      command=(python baselines/CEC/test_general_human_proxy.py
        "model_name=${model}"
        "ENV_KWARGS.layout=${layout}"
        "++MODEL_PATH=${model_path}"
        "++MODEL_NUM_ENVS=${model_num_envs}"
        "++MODEL_SEEDS=${model_seeds}"
        NUM_MODELS=6
        ++HUMAN_PROXY_NUM_SEEDS=5
        ++HUMAN_PROXY_CKPT_DIR=baselines/human_proxy/checkpoints
      )
      if [[ -n "${model_output_dir}" ]]; then
        command+=("++OUTPUT_DIR=${model_output_dir}")
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}"
    done
  done
done
