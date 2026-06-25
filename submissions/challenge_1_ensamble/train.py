#!/usr/bin/env python3
"""train.py — train the missingness-aware XGBoost ensemble and save weights.joblib.

Run from inside this folder:

    cd submissions/challenge_1_ensamble
    python train.py

Reads:  ../../dataset/train_set.csv   (ride-level data; built into a station-hour table)
Writes: weights.joblib               (3 base models + orchestrator + fallback)

The evaluator never runs this file; it loads weights.joblib via predict.py.
"""
import argparse
import os

import joblib

from data import build_or_load_table        # shared station-hour table builder (zero-recon)
from model import train_artifacts           # shared model code (single source of truth)

OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the XGBoost ensemble -> weights.joblib")
    ap.add_argument("--fast", action="store_true",
                    help="quick iteration: small trees + no mask augmentation (NOT for submission)")
    ap.add_argument("--n-estimators", type=int, default=None,
                    help="override base-model tree count (default: full recipe)")
    ap.add_argument("--n-mask-augment", type=int, default=0,
                    help="missingness-augmentation rounds for the orchestrator (0 = off)")
    args = ap.parse_args()

    n_estimators = args.n_estimators
    n_mask_augment = args.n_mask_augment
    if args.fast:                            # fast preset, unless explicitly overridden
        n_estimators = n_estimators or 150
        n_mask_augment = 0

    # Build the labelled station-hour table from train_set.csv (all available cities).
    table = build_or_load_table()
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or -1

    print(f"training rows={len(table):,}  cities={sorted(table['city_key'].unique())}  "
          f"mean_demand={table['demand'].mean():.3f}  n_jobs={n_jobs}  "
          f"fast={args.fast}  n_estimators={n_estimators or 'default'}  n_mask_augment={n_mask_augment}")
    artifacts = train_artifacts(table, n_jobs=n_jobs,
                                n_estimators=n_estimators, n_mask_augment=n_mask_augment)

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(f"Saved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
