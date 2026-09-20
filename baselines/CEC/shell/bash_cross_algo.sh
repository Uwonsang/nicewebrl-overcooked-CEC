#!/usr/bin/env bash
set -euo pipefail

# Usage: bash_cross_algo.sh [GPU] [MODEL_PATH] [NUM_TRAJS]
# Example: bash baselines/CEC/shell/bash_cross_algo.sh \
#   0 /app/nas/models/ICRL 10

gpu="${1:-0}"
model_path="${2:-}"
num_trajs="${3:-10}"

if [[ -z "${model_path}" ]]; then
  if [[ -d /app/nas/models/ICRL ]]; then
    model_path=/app/nas/models/ICRL
  elif [[ -d /mnt/nas/wonsang/crossenv_ued/models/ICRL ]]; then
    model_path=/mnt/nas/wonsang/crossenv_ued/models/ICRL
  else
    echo "Could not find the ICRL model directory." >&2
    echo "Pass MODEL_PATH as the second argument." >&2
    exit 2
  fi
fi

output_dir="${model_path%/}/xp_results_diff_algo"

layouts=(
  asymm_advantages_9
  coord_ring_9
  counter_circuit_9
  cramped_room_9
  forced_coord_9
)

for layout in "${layouts[@]}"; do
  echo "Cross-algorithm XP: layout=${layout}, trajectories=${num_trajs}"
  echo "Output directory: ${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/cross_algo.py \
    "ENV_KWARGS.layout=${layout}" \
    "MODEL_PATH=${model_path}" \
    "SAVE_PATH=${output_dir}" \
    "TEST_KWARGS.num_trajs=${num_trajs}" \
    XP_ONLY=False
done
