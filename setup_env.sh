#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup] error: conda is required but was not found in PATH"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR=${REPO_DIR:-"${SCRIPT_DIR}"}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-infom}
PYTHON_VERSION=${PYTHON_VERSION:-3.10.16}

echo "[setup] creating conda env ${CONDA_ENV_NAME} (python=${PYTHON_VERSION})"
conda create -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "[setup] installing conda dependencies"
conda install -n "${CONDA_ENV_NAME}" -c conda-forge glew -y
conda install -n "${CONDA_ENV_NAME}" -c conda-forge mesalib -y

echo "[setup] installing project dependencies"
conda run -n "${CONDA_ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${CONDA_ENV_NAME}" python -m pip install -r "${REPO_DIR}/requirements.txt"

echo "[setup] configuring environment variables"
conda env config vars set -n "${CONDA_ENV_NAME}" \
  PYTHONPATH="${REPO_DIR}" \
  MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl

echo "[setup] verifying core imports"
conda run -n "${CONDA_ENV_NAME}" python - <<'PY'
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
echo "[setup] run: conda activate ${CONDA_ENV_NAME}"
echo "[setup] if this env already existed, run: conda deactivate && conda activate ${CONDA_ENV_NAME} to apply updated env vars"
