#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_URI="sqlite:///${PROJECT_ROOT}/mlruns_omninwm/mlflow.db"
ART_ROOT="${PROJECT_ROOT}/mlruns_omninwm/artifacts"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5001}"

mkdir -p "${PROJECT_ROOT}/mlruns_omninwm" "${ART_ROOT}"

".venv/bin/mlflow" ui \
  --backend-store-uri "${DB_URI}" \
  --default-artifact-root "${ART_ROOT}" \
  --host "${HOST}" \
  --port "${PORT}"
