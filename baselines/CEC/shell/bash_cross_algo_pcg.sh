#!/usr/bin/env bash
set -euo pipefail

# Run cross-algorithm evaluation on the same 100 PCG tasks for one or more
# checkpoint-layout cohorts.
#
# Usage:
#   bash baselines/CEC/shell/bash_cross_algo_pcg.sh \
#     [GPU] [LAYOUTS|all] [MODEL_PATH] [NUM_TRAJS] [OVERWRITE]
#
# Examples:
#   bash baselines/CEC/shell/bash_cross_algo_pcg.sh 0 all
#   bash baselines/CEC/shell/bash_cross_algo_pcg.sh \
#     1 coord_ring_9,cramped_room_9 /app/nas/models/ICRL 5 false
#
# Set PYTHON_BIN when a specific Python executable is required:
#   PYTHON_BIN=/path/to/python bash baselines/CEC/shell/bash_cross_algo_pcg.sh 0 all

gpu="${1:-0}"
layout_arg="${2:-all}"
model_path="${3:-}"
num_trajs="${4:-5}"
overwrite="${5:-false}"
python_bin="${PYTHON_BIN:-python}"

if [[ ! "${gpu}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU must be an integer or comma-separated integers: ${gpu}" >&2
  exit 2
fi

if [[ ! "${num_trajs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_TRAJS must be a positive integer: ${num_trajs}" >&2
  exit 2
fi

if [[ "${overwrite}" != "true" && "${overwrite}" != "false" ]]; then
  echo "OVERWRITE must be true or false: ${overwrite}" >&2
  exit 2
fi

if [[ -z "${model_path}" ]]; then
  if [[ -d /app/nas/models/ICRL ]]; then
    model_path=/app/nas/models/ICRL
  elif [[ -d /mnt/nas/wonsang/crossenv_ued/models/ICRL ]]; then
    model_path=/mnt/nas/wonsang/crossenv_ued/models/ICRL
  else
    echo "Could not find the ICRL model directory." >&2
    echo "Pass MODEL_PATH as the third argument." >&2
    exit 2
  fi
fi

if [[ ! -d "${model_path}" ]]; then
  echo "MODEL_PATH does not exist: ${model_path}" >&2
  exit 2
fi

all_layouts=(
  asymm_advantages_9
  coord_ring_9
  counter_circuit_9
  cramped_room_9
  forced_coord_9
)

if [[ "${layout_arg}" == "all" ]]; then
  layouts=("${all_layouts[@]}")
else
  IFS=',' read -r -a layouts <<< "${layout_arg}"
  if [[ "${#layouts[@]}" -eq 0 ]]; then
    echo "At least one layout is required." >&2
    exit 2
  fi
  for layout in "${layouts[@]}"; do
    valid=false
    for candidate in "${all_layouts[@]}"; do
      if [[ "${layout}" == "${candidate}" ]]; then
        valid=true
        break
      fi
    done
    if [[ "${valid}" != "true" ]]; then
      echo "Unknown layout: ${layout}" >&2
      echo "Available: all or $(IFS=,; echo "${all_layouts[*]}")" >&2
      exit 2
    fi
  done
fi

output_dir="${model_path%/}/pcg_xp_results_diff_algo"

for layout in "${layouts[@]}"; do
  echo "PCG cross-algorithm evaluation"
  echo "  GPU: ${gpu}"
  echo "  checkpoint layout: ${layout}"
  echo "  trajectories per seed pair/task: ${num_trajs}"
  echo "  output: ${output_dir}/${layout}_pcg_cross_algo_results.csv"

  env CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" \
    baselines/CEC/cross_algo_pcg.py \
    "ENV_KWARGS.layout=${layout}" \
    "MODEL_PATH=${model_path}" \
    "TEST_KWARGS.num_trajs=${num_trajs}" \
    "OVERWRITE=${overwrite}"
done
