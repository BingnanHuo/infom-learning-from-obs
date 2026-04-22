#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

echo "[setup] creating virtualenv at ${VENV_DIR} using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

PIP_BIN="${VENV_DIR}/bin/pip"
PY_BIN="${VENV_DIR}/bin/python"

echo "[setup] upgrading pip"
"${PIP_BIN}" install --upgrade pip

echo "[setup] installing project dependencies"
"${PIP_BIN}" install -r requirements.txt

echo "[setup] verifying core imports"
"${PY_BIN}" - <<'PY'
from importlib.metadata import version

import jax
import flax
import gymnasium
import ogbench

print('jax', jax.__version__)
print('flax', flax.__version__)
print('gymnasium', gymnasium.__version__)
print('ogbench', version('ogbench'))
PY

echo "[setup] done"
