#!/usr/bin/env python3
"""
Create a local train/validation split for offline evaluation.

Splits dataset/train_set.csv by TIME within each city (holds out the latest
~20% of each city as validation). A time-based split is essential: a random
split would leak future information into the past and inflate your local score.

Run once from the project root:

    python make_local_split.py

Outputs:
    dataset/local_train_set.csv
    dataset/local_validation_set.csv

Then convert the validation set into the evaluator's station-hour format:

    python build_station_hour_eval_data.py \
        --input_csv dataset/local_validation_set.csv \
        --public_targets_csv dataset/public_validation_targets.csv \
        --private_labels_csv dataset/private_labels.csv
"""

from pathlib import Path

import pandas as pd

DATASET_DIR = Path("dataset")
TRAIN_CSV = DATASET_DIR / "train_set.csv"
VAL_FRACTION = 0.20  # fraction of the latest time per city held out as validation


def main() -> None:
    df = pd.read_csv(TRAIN_CSV, low_memory=False)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")

    train_parts, val_parts = [], []
    for city, g in df.groupby("city"):
        g = g.sort_values("started_at")
        cut = g["started_at"].quantile(1.0 - VAL_FRACTION)
        train_parts.append(g[g["started_at"] <= cut])
        val_parts.append(g[g["started_at"] > cut])

    train = pd.concat(train_parts).sort_index()
    val = pd.concat(val_parts).sort_index()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATASET_DIR / "local_train_set.csv", index=False)
    val.to_csv(DATASET_DIR / "local_validation_set.csv", index=False)

    print(f"local_train_set.csv      rows={len(train):>10,}")
    print(f"local_validation_set.csv rows={len(val):>10,}")


if __name__ == "__main__":
    main()
