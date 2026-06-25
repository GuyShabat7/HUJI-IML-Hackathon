#!/bin/bash
# cluster/run_eval.slurm — evaluate a submission with tools/eval_local.py on Phoenix.
#
#   sbatch cluster/run_eval.slurm submissions/challenge_1_ensamble
#
# CPU-only. Trains on the held-out train split and reports per-city + blended MAE.
# >>> FILL IN / CONFIRM the placeholders below for your Phoenix account. <<<
#SBATCH --job-name=bike-eval
#SBATCH --account=nirf                 # <PLACEHOLDER>
#SBATCH --partition=short              # <PLACEHOLDER> short | killable
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8              # <PLACEHOLDER>
#SBATCH --mem=32G                      # <PLACEHOLDER>
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -uo pipefail
SUB="${1:?usage: sbatch cluster/run_eval.slurm submissions/<folder>}"
mkdir -p cluster/logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_PROFILE" ]; then
    source "$CONDA_PROFILE"; conda activate bikedemand
else
    source "$HOME/bikedemand-venv/bin/activate"
fi

echo "[run_eval] $(date) | sub=$SUB | cpus=$OMP_NUM_THREADS | host=$(hostname)"
test -f dataset/train_set.csv || { echo "ERROR: dataset/train_set.csv missing (copy it onto Phoenix; it is NOT in git)"; exit 2; }

python tools/eval_local.py "$SUB" --n-jobs "$OMP_NUM_THREADS"
echo "[run_eval] $(date) | done"
