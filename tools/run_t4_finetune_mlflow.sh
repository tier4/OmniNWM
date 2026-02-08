#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train/t4_finetune.py}"
shift || true

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlruns_omninwm/mlflow.db}"
MLFLOW_ARTIFACT_LOCATION="${MLFLOW_ARTIFACT_LOCATION:-mlruns_omninwm/artifacts}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-omninwm_t4_shared}"

# Fixed run naming defaults
if [[ "${CONFIG}" == *"phaseB"* ]]; then
  DEFAULT_RUN_NAME="omninwm_t4_phaseB"
else
  DEFAULT_RUN_NAME="omninwm_t4_phaseA"
fi
EXP_NAME="${EXP_NAME:-${DEFAULT_RUN_NAME}}"
MLFLOW_RUN_NAME="${MLFLOW_RUN_NAME:-${DEFAULT_RUN_NAME}}"

if [[ "${MLFLOW_TRACKING_URI}" == sqlite:///* ]]; then
  DB_PATH="${MLFLOW_TRACKING_URI#sqlite:///}"
  mkdir -p "$(dirname "${DB_PATH}")"
fi

if [[ "${MLFLOW_ARTIFACT_LOCATION}" != *"://"* ]]; then
  mkdir -p "${MLFLOW_ARTIFACT_LOCATION}"
fi

echo "Config: ${CONFIG}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MLFLOW_TRACKING_URI: ${MLFLOW_TRACKING_URI}"
echo "MLFLOW_ARTIFACT_LOCATION: ${MLFLOW_ARTIFACT_LOCATION}"
echo "MLFLOW_EXPERIMENT: ${MLFLOW_EXPERIMENT}"
echo "EXP_NAME: ${EXP_NAME}"
echo "MLFLOW_RUN_NAME: ${MLFLOW_RUN_NAME}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if [[ -x ".venv/bin/torchrun" ]]; then
  TORCHRUN_BIN=".venv/bin/torchrun"
fi
echo "TORCHRUN_BIN: ${TORCHRUN_BIN}"

"${TORCHRUN_BIN}" --nproc-per-node "${NPROC_PER_NODE}" \
  tools/train.py "${CONFIG}" \
  --exp_name "${EXP_NAME}" \
  --mlflow true \
  --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
  --mlflow_run_name "${MLFLOW_RUN_NAME}" \
  --mlflow_tracking_uri "${MLFLOW_TRACKING_URI}" \
  --mlflow_artifact_location "${MLFLOW_ARTIFACT_LOCATION}" \
  "$@"
