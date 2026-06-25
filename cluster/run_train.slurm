#!/bin/bash
# cluster/run_train.slurm — train a submission's model on Phoenix (HUJI SLURM).
#
#   sbatch cluster/run_train.slurm submissions/challenge_1_ensamble
#
# CPU-only job (LightGBM/XGBoost). Do NOT request a GPU.
# >>> FILL IN / CONFIRM the placeholders below for your Phoenix account. <<<
#SBATCH --job-name=bike-train
#SBATCH --account=nirf                 # <PLACEHOLDER> e.g. nirf  (or killable-cs for the killable partition)
#SBATCH --partition=short              # <PLACEHOLDER> short (<=3h) | killable (long, preemptible)
#SBATCH --time=03:00:00                # raise for full-year enriched data
#SBATCH --cpus-per-task=8              # <PLACEHOLDER> match your allocation
#SBATCH --mem=32G                      # <PLACEHOLDER>
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
# #SBATCH --mail-user=ron.drori@mail.huji.ac.il   # uncomment to get emails
# #SBATCH --mail-type=END,FAIL

set -uo pipefail
SUB="${1:?usage: sbatch cluster/run_train.slurm submissions/<folder>}"
mkdir -p cluster/logs

# Match library thread pools to the SLURM allocation (avoid oversubscription).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

# Activate the env created by env_setup.sh (conda preferred, venv fallback).
CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"; conda activate bikedemand
else
    source "$HOME/bikedemand-venv/bin/activate"
fi

echo "[run_train] $(date) | sub=$SUB | cpus=$OMP_NUM_THREADS | host=$(hostname)"
test -f dataset/train_set.csv || { echo "ERROR: dataset/train_set.csv missing (copy it onto Phoenix; it is NOT in git)"; exit 2; }

cd "$SUB"
python train.py
echo "[run_train] $(date) | done -> $SUB/weights.joblib"
