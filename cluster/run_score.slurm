#!/bin/bash
# cluster/run_score.slurm — score the model on the unseen 4th city on Phoenix (HUJI SLURM).
#
#   sbatch cluster/run_score.slurm submissions/challenge_1_ensamble dataset/chicago_enriched.csv
#
# CPU-only. Trains on c1+c2 (held-out protocol) and reports MAE for city1/2/3 + city4.
# >>> FILL IN / CONFIRM the placeholders below. <<<
#SBATCH --job-name=bike-score
#SBATCH --account=nirf                 # <PLACEHOLDER>
#SBATCH --partition=short              # <PLACEHOLDER> short | killable
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8              # <PLACEHOLDER>
#SBATCH --mem=32G                      # <PLACEHOLDER>
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -uo pipefail
SUB="${1:?usage: sbatch cluster/run_score.slurm submissions/<folder> <enriched.csv>}"
ENR="${2:?need enriched city-4 csv}"
mkdir -p cluster/logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"; export MKL_NUM_THREADS="$OMP_NUM_THREADS"

CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_PROFILE" ]; then source "$CONDA_PROFILE"; conda activate bikedemand
else source "$HOME/bikedemand-venv/bin/activate"; fi

echo "[run_score] $(date) | sub=$SUB enriched=$ENR | host=$(hostname)"
test -f dataset/train_set.csv || { echo "ERROR: dataset/train_set.csv missing on Phoenix"; exit 2; }
python tools/score_holdout.py --submission "$SUB" --enriched "$ENR" --n-jobs "$OMP_NUM_THREADS"
echo "[run_score] $(date) | done"
