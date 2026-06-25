#!/bin/bash
# cluster/run_enrich.slurm — enrich a start-side ride file on Phoenix (HUJI SLURM).
#
#   sbatch cluster/run_enrich.slurm dataset/chicago_holdout_start.csv chicago
#
# NOTE: enrichment calls the Open-Meteo and Overpass (or reads a local .osm.pbf) APIs.
# Phoenix COMPUTE nodes are often firewalled — run this where outbound internet is allowed
# (a login/interactive node per cluster policy), OR pre-download a local .osm.pbf and pass
# --poi-pbf so only the (cached) weather call needs the network.
# >>> FILL IN / CONFIRM the placeholders below. <<<
#SBATCH --job-name=bike-enrich
#SBATCH --account=nirf                 # <PLACEHOLDER>
#SBATCH --partition=short              # <PLACEHOLDER> short | killable
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4              # <PLACEHOLDER>
#SBATCH --mem=16G                      # <PLACEHOLDER>
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err

set -uo pipefail
IN="${1:?usage: sbatch cluster/run_enrich.slurm <start_side.csv> [name] [extra args...]}"
NAME="${2:-$(basename "${IN%.*}")}"
shift 2 2>/dev/null || shift $#
mkdir -p cluster/logs
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

CONDA_PROFILE="${CONDA_PROFILE:-$HOME/miniforge3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_PROFILE" ]; then source "$CONDA_PROFILE"; conda activate bikedemand
else source "$HOME/bikedemand-venv/bin/activate"; fi

echo "[run_enrich] $(date) | in=$IN name=$NAME | host=$(hostname)"
python tools/enrich_city.py "$IN" --name "$NAME" "$@"
echo "[run_enrich] $(date) | done -> dataset/${NAME}_enriched.csv"
