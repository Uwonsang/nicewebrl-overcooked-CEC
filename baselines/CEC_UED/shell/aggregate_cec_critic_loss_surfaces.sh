#!/usr/bin/env bash
# Aggregate already-computed loss-surface NPZ files without reevaluating models.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPOSITORY_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/baselines/CEC/figures/results/critic_loss_surfaces}"
MODELS="${MODELS:-CEC CEC_IDDAC}"
TRAINING_NUM_ENVS="${TRAINING_NUM_ENVS:-}"
SEEDS="${SEEDS:-}"
PARAMETER_CASES="${PARAMETER_CASES:-encoder_rnn critic_mlp critic_full}"

read -r -a model_args <<< "${MODELS}"
read -r -a training_num_env_args <<< "${TRAINING_NUM_ENVS}"
read -r -a seed_args <<< "${SEEDS}"
read -r -a case_args <<< "${PARAMETER_CASES}"

aggregate_args=(
    --output-root "${OUTPUT_ROOT}"
    --models "${model_args[@]}"
    --cases "${case_args[@]}"
)
if (( ${#training_num_env_args[@]} > 0 )); then
    aggregate_args+=(--training-num-envs "${training_num_env_args[@]}")
fi
if (( ${#seed_args[@]} > 0 )); then
    aggregate_args+=(--seeds "${seed_args[@]}")
fi

echo "CEC/CEC_IDDAC loss-surface NPZ aggregation"
echo "  input/output root: ${OUTPUT_ROOT}"
echo "  models:            ${MODELS}"
echo "  training num envs: ${TRAINING_NUM_ENVS:-all discovered}"
echo "  seeds:             ${SEEDS:-all discovered}"
echo "  parameter cases:   ${PARAMETER_CASES}"

exec python -m baselines.CEC_UED.aggregate_cec_critic_loss_surfaces \
    "${aggregate_args[@]}" \
    "$@"
