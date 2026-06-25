#!/bin/bash
# cluster/env_setup.sh — one-time environment setup on the Phoenix (HUJI SLURM) cluster.
#
# Creates a conda env (preferred, via miniforge3) or a venv fallback, and installs the
# model dependencies. Run ONCE on a login/compute node, then the .slurm wrappers just
# activate it. Re-running is safe (idempotent).
#
#   bash cluster/env_setup.sh
#
set -uo pipefail

# ---- configurable -----------------------------------------------------------
ENV_NAME="bikedemand"
# Point this at your miniforge/conda profile if you have one (HUJI lab convention).
# Leave as-is to auto-detect; override by exporting CONDA_PROFILE before running.
CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
PKGS="pandas numpy scikit-learn lightgbm xgboost joblib"
# ----------------------------------------------------------------------------

if [ -f "$CONDA_PROFILE" ]; then
    echo "[env_setup] using conda profile: $CONDA_PROFILE"
    source "$CONDA_PROFILE"
    if ! conda env list | grep -q "/${ENV_NAME}\$"; then
        conda create -y -n "$ENV_NAME" python=3.11
    fi
    conda activate "$ENV_NAME"
    pip install --upgrade pip >/dev/null
    pip install $PKGS
    echo "[env_setup] conda env '$ENV_NAME' ready. Activate with:"
    echo "    source $CONDA_PROFILE && conda activate $ENV_NAME"
else
    echo "[env_setup] no conda profile at $CONDA_PROFILE — falling back to venv"
    python3 -m venv "$HOME/${ENV_NAME}-venv"
    source "$HOME/${ENV_NAME}-venv/bin/activate"
    pip install --upgrade pip >/dev/null
    pip install $PKGS
    echo "[env_setup] venv ready. Activate with:"
    echo "    source $HOME/${ENV_NAME}-venv/bin/activate"
fi

python -c "import pandas,numpy,sklearn,lightgbm,xgboost,joblib; print('[env_setup] OK — all imports succeed')"
