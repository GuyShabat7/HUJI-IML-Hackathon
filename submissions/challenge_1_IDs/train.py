#!/usr/bin/env python3
"""
train.py — trains the bike-demand model and saves weights.joblib.

Run from inside this folder:

    cd submissions/challenge_1_IDs
    python train.py

Reads:  ../../dataset/train_set.csv   (ride-level data)
Writes: weights.joblib               (everything predict.py needs)

The evaluator never runs this file; it only loads weights.joblib via predict.py.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "lightgbm is required to train. Install with: pip install lightgbm"
    ) from e

# Shared feature logic — identical code runs at train and predict time.
from model import (
    DAYTIME_HOURS, WEATHER_COLS, META_COLS, FLAG_COLS,
    add_keys, build_features, fit_encodings,
)

DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def build_training_table(rides: pd.DataFrame) -> pd.DataFrame:
    """Ride-level -> station-hour table WITH reconstructed zero hours."""
    keyed = add_keys(rides)

    # 1) observed counts: rides per (city, station, hour)
    demand = (
        keyed.groupby(["city_key", "station_key", "ts"])
        .size().reset_index(name="demand")
    )

    # 2) reconstruct the active-window grid — this is where the zeros come from
    grids = []
    for (ck, sk), g in keyed.groupby(["city_key", "station_key"]):
        lo, hi = g["ts"].min(), g["ts"].max()
        if pd.isna(lo) or pd.isna(hi):
            continue
        hours = pd.date_range(lo, hi, freq="h")
        hours = hours[hours.hour.isin(DAYTIME_HOURS)]  # daytime only, like evaluator
        if len(hours):
            grids.append(pd.DataFrame({"city_key": ck, "station_key": sk, "ts": hours}))
    grid = pd.concat(grids, ignore_index=True)

    # 3) attach demand; missing grid hours are TRUE ZEROS
    table = grid.merge(demand, on=["city_key", "station_key", "ts"], how="left")
    table["demand"] = table["demand"].fillna(0).astype(int)

    # 4) attach per-(city,hour) weather/flags and per-station metadata
    wcols = [c for c in WEATHER_COLS + FLAG_COLS if c in keyed.columns]
    if wcols:
        city_hour = keyed.groupby(["city_key", "ts"])[wcols].first().reset_index()
        table = table.merge(city_hour, on=["city_key", "ts"], how="left")
    mcols = [c for c in META_COLS if c in keyed.columns]
    if mcols:
        station_meta = keyed.groupby(["city_key", "station_key"])[mcols].first().reset_index()
        table = table.merge(station_meta, on=["city_key", "station_key"], how="left")

    # time parts on the grid timestamps
    table["hour"] = table["ts"].dt.hour
    table["weekday"] = table["ts"].dt.weekday
    table["how"] = table["weekday"] * 24 + table["hour"]
    table["is_weekend"] = table["weekday"].isin([5, 6]).astype(int)
    return table


def train_one(X: pd.DataFrame, y: pd.Series, objective: str):
    """A single LightGBM regressor. objective: 'poisson' or 'regression_l1'."""
    model = lgb.LGBMRegressor(
        objective=objective,
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model


def main() -> None:
    rides = pd.read_csv(TRAIN_CSV, low_memory=False)

    table = build_training_table(rides)          # station-hours with zeros
    enc = fit_encodings(table)                    # hierarchical fallback stats
    X = build_features(table, enc)                # shared feature matrix
    y = table["demand"].astype(float)
    feature_cols = list(X.columns)

    print(f"training rows={len(X):,}  features={len(feature_cols)}  "
          f"mean_demand={y.mean():.3f}  zero_rate={(y == 0).mean():.3f}")

    m_pois = train_one(X, y, "poisson")
    m_mae = train_one(X, y, "regression_l1")

    joblib.dump(
        {
            "model_poisson": m_pois,
            "model_mae": m_mae,
            "encodings": enc,
            "feature_cols": feature_cols,
        },
        OUTPUT_WEIGHTS,
    )
    print(f"Saved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()
