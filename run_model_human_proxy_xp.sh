#!/usr/bin/env bash

set -euo pipefail

# Single-algorithm form:
#   ./run_model_human_proxy_xp.sh ippo /path/to/ICRL
#
# Override any of these from the shell, for example:
#   EPISODES=5 ALGORITHMS="ippo cec_64" ./run_model_human_proxy_xp.sh
MODEL_ROOT="${MODEL_ROOT:-/mnt/nas/wonsang/crossenv_ued/models/ICRL}"
HUMAN_PROXY_ROOT="${HUMAN_PROXY_ROOT:-human_proxy/checkpoints}"
OUTPUT_DIR="${OUTPUT_DIR:-data/xp_human_proxy}"
EPISODES="${EPISODES:-1}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-200}"
WORLD_SEED="${WORLD_SEED:-1}"

CHECKPOINT_PATH=""
if [[ $# -ge 1 && "${1}" != -* ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <algorithm> <checkpoint-path> [extra Python arguments]" >&2
    exit 2
  fi
  ALGORITHMS_TEXT="${1}"
  CHECKPOINT_PATH="${2}"
  shift 2
else
  ALGORITHMS_TEXT="${ALGORITHMS:-ippo e3t fcp cec_64 cec_idaac_32 cec_idaac_256}"
fi
LAYOUTS_TEXT="${LAYOUTS:-counter_circuit coord_ring asymm_advantages forced_coord cramped_room}"

read -r -a ALGORITHM_LIST <<< "${ALGORITHMS_TEXT}"
read -r -a LAYOUT_LIST <<< "${LAYOUTS_TEXT}"

mkdir -p "${OUTPUT_DIR}/run_logs"

for algorithm in "${ALGORITHM_LIST[@]}"; do
  for layout in "${LAYOUT_LIST[@]}"; do
    run_name="${algorithm}_${layout}"
    log_path="${OUTPUT_DIR}/run_logs/${run_name}.log"

    echo "[$(date --iso-8601=seconds)] starting ${run_name}" | tee -a "${log_path}"

    command=(
      uv run --no-sync python model_human_proxy_xp.py
      --model-root "${MODEL_ROOT}"
      --human-proxy-root "${HUMAN_PROXY_ROOT}"
      --output-dir "${OUTPUT_DIR}"
      --models "${algorithm}"
      --layouts "${layout}"
      --episodes "${EPISODES}"
      --max-timesteps "${MAX_TIMESTEPS}"
      --world-seed "${WORLD_SEED}"
    )
    if [[ -n "${CHECKPOINT_PATH}" ]]; then
      command+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi
    command+=("$@")

    "${command[@]}" 2>&1 | tee -a "${log_path}"

    # model_human_proxy_xp.py records the latest invocation in manifest.json.
    # Preserve a per-pair copy before the next invocation replaces it.
    cp \
      "${OUTPUT_DIR}/manifest.json" \
      "${OUTPUT_DIR}/run_logs/${run_name}_manifest.json"

    echo "[$(date --iso-8601=seconds)] finished ${run_name}" | tee -a "${log_path}"
  done
done

echo "All requested algorithm-layout evaluations finished: ${OUTPUT_DIR}"
