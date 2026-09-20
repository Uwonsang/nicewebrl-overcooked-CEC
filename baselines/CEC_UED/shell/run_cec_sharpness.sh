#!/usr/bin/env bash
# Evaluate Keskar loss-landscape sharpness for CEC and CEC_IDDAC only.
#
# Run from anywhere:
#   bash baselines/CEC_UED/shell/run_cec_sharpness.sh
#
# Common overrides:
#   CUDA_VISIBLE_DEVICES=0 \
#   PYTHON_BIN=/path/to/python \
#   TRAINING_NUM_ENVS="128 256" \
#   SEEDS="0 1" \
#   OUTPUT=results/sharpness.json \
#   bash baselines/CEC_UED/shell/run_cec_sharpness.sh
#
# Extra arguments are forwarded to measure_cec_sharpness.py:
#   bash baselines/CEC_UED/shell/run_cec_sharpness.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPOSITORY_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-CEC CEC_IDDAC}"
TRAINING_NUM_ENVS="${TRAINING_NUM_ENVS:-${TRAINING_ROLLOUT_SIZES:-32 256}}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
EPSILONS="${EPSILONS:-0.001 0.0005}"
MAXITER="${MAXITER:-10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-32}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-400}"
SAMPLED_ACTORS="${SAMPLED_ACTORS:-16}"
ROLLOUT_SEED="${ROLLOUT_SEED:-20260823}"
OUTPUT="${OUTPUT:-critic_sharpness_cec_32_256.json}"

read -r -a model_args <<< "${MODELS}"
read -r -a training_num_env_args <<< "${TRAINING_NUM_ENVS}"
read -r -a seed_args <<< "${SEEDS}"
read -r -a epsilon_args <<< "${EPSILONS}"

echo "CEC sharpness evaluation"
echo "  models:                 ${MODELS}"
echo "  training num envs:      ${TRAINING_NUM_ENVS}"
echo "  seeds:                  ${SEEDS}"
echo "  epsilons:               ${EPSILONS}"
echo "  output:                 ${OUTPUT}"
echo "  loss scope:             critic only"

exec "${PYTHON_BIN}" -m baselines.CEC_UED.measure_cec_sharpness \
    --models "${model_args[@]}" \
    --training-num-envs "${training_num_env_args[@]}" \
    --seeds "${seed_args[@]}" \
    --epsilons "${epsilon_args[@]}" \
    --maxiter "${MAXITER}" \
    --eval-num-envs "${EVAL_NUM_ENVS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --sampled-actors "${SAMPLED_ACTORS}" \
    --rollout-seed "${ROLLOUT_SEED}" \
    --output "${OUTPUT}" \
    --loss-scope critic \
    "$@"
