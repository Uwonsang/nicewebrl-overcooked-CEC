#!/usr/bin/env bash
# Plot early/middle/final CEC and CEC_IDDAC critic loss surfaces.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPOSITORY_ROOT}"

MODEL_ROOT="${MODEL_ROOT:-/app/nas/models/ICRL}"
MODELS="${MODELS:-CEC CEC_IDDAC}"
TRAINING_NUM_ENVS="${TRAINING_NUM_ENVS:-}"
SEEDS="${SEEDS:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_DIR:-${REPOSITORY_ROOT}/baselines/CEC/figures/results/critic_loss_surfaces}}"
GRID_SIZE="${GRID_SIZE:-21}"
RADIUS="${RADIUS:-0.5}"
DIRECTION_SEED="${DIRECTION_SEED:-0}"
PARAMETER_CASE="${PARAMETER_CASE:-all}"

read -r -a model_args <<< "${MODELS}"
read -r -a seed_args <<< "${SEEDS}"
read -r -a training_num_env_args <<< "${TRAINING_NUM_ENVS}"

includes_training_num_envs() {
    local candidate="$1"
    if (( ${#training_num_env_args[@]} == 0 )); then
        return 0
    fi
    local allowed
    for allowed in "${training_num_env_args[@]}"; do
        if [[ "${candidate}" == "${allowed}" ]]; then
            return 0
        fi
    done
    return 1
}

echo "CEC/CEC_IDDAC critic loss-surface evaluation"
echo "  model root:     ${MODEL_ROOT}"
echo "  models:         ${MODELS}"
echo "  train envs:     ${TRAINING_NUM_ENVS:-all discovered}"
echo "  seeds:          ${SEEDS}"
echo "  output root:    ${OUTPUT_ROOT}"
echo "  grid:           ${GRID_SIZE}x${GRID_SIZE}"
echo "  radius:         ${RADIUS}"
echo "  parameter case: ${PARAMETER_CASE}"

shopt -s nullglob
evaluation_count=0
for model in "${model_args[@]}"; do
    case "${model}" in
        CEC|CEC_IDDAC) ;;
        *)
            echo "Unsupported model: ${model} (expected CEC or CEC_IDDAC)" >&2
            exit 2
            ;;
    esac
    model_dir="${MODEL_ROOT}/${model}"
    if [[ ! -d "${model_dir}" ]]; then
        echo "Model directory not found: ${model_dir}" >&2
        exit 2
    fi

    for env_dir in "${model_dir}"/*; do
        [[ -d "${env_dir}" ]] || continue
        training_num_envs="$(basename -- "${env_dir}")"
        [[ "${training_num_envs}" =~ ^[0-9]+$ ]] || continue
        includes_training_num_envs "${training_num_envs}" || continue

        for seed in "${seed_args[@]}"; do
            snapshot_dir="${env_dir}/seed${seed}/seed${seed}_mid_ckpts/critic_loss_surface"
            if [[ ! -d "${snapshot_dir}" ]]; then
                echo "Skipping missing snapshots: ${model}, envs=${training_num_envs}, seed=${seed}"
                continue
            fi
            output_dir="${OUTPUT_ROOT}/${model}/env${training_num_envs}/seed${seed}"
            evaluation_count=$((evaluation_count + 1))
            echo
            echo "[${evaluation_count}] ${model}, training NUM_ENVS=${training_num_envs}, seed=${seed}"
            python -m baselines.CEC_UED.plot_cec_critic_loss_surfaces \
                --snapshot-dir "${snapshot_dir}" \
                --output-dir "${output_dir}" \
                --grid-size "${GRID_SIZE}" \
                --radius "${RADIUS}" \
                --direction-seed "${DIRECTION_SEED}" \
                --parameter-case "${PARAMETER_CASE}" \
                "$@"
        done
    done
done

if (( evaluation_count == 0 )); then
    echo "No matching critic-loss-surface snapshot directories found." >&2
    exit 2
fi

echo
echo "Completed ${evaluation_count} model/env/seed evaluation(s)."
