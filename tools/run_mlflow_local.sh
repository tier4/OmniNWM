#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/nvme2/OmniNWM"
# SQLite 数据库存储后端 - 使用绝对路径
DB_URI="sqlite:///${PROJECT_ROOT}/mlruns_omninwm/mlflow.db"
ART_ROOT="${PROJECT_ROOT}/mlruns"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5001}"

mkdir -p "${PROJECT_ROOT}/mlruns_omninwm" "${ART_ROOT}"

echo "Starting MLflow server..."
echo "  Database: ${DB_URI}"
echo "  Artifacts: ${ART_ROOT}"
echo "  Host: ${HOST}:${PORT}"

"${PROJECT_ROOT}/.venv/bin/mlflow" server \
  --backend-store-uri "${DB_URI}" \
  --default-artifact-root "${ART_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --allowed-hosts '*'
