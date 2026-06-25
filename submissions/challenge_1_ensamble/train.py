#!/usr/bin/env python3
"""train.py — train the missingness-aware XGBoost ensemble and save weights.joblib.

Run from inside this folder:

    cd submissions/challenge_1_ensamble
    python train.py

Reads:  ../../dataset/train_set.csv   (ride-level data; built into a station-hour table)
Writes: weights.joblib               (3 base models + orchestrator + fallback)

The evaluator never runs this file; it loads weights.joblib via predict.py.
"""
import os

import joblib

from data import build_or_load_table        # shared station-hour table builder (zero-recon)
from model import train_artifacts           # shared model code (single source of truth)

OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:
    # Build the labelled station-hour table from train_set.csv (all available cities).
    table = build_or_load_table()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or -1

    print(f"training rows={len(table):,}  cities={sorted(table['city_key'].unique())}  "
          f"mean_demand={table['demand'].mean():.3f}  n_jobs={n_jobs}")
    artifacts = train_artifacts(table, n_jobs=n_jobs)

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(f"Saved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
