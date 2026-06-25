"""
model.py — feature engineering and prediction logic for the bike-demand model.

This file defines `BikeDemandModel`, which is instantiated by the fixed
evaluator wrapper in predict.py. The SAME feature-building code here is imported
and reused by train.py, so training and inference always see identical features.

Do NOT load weights.joblib here — predict.py loads it and passes the artifacts
into `load_artifacts`.
"""

import numpy as np
import pandas as pd

# Daytime hours kept by the evaluator (build_station_hour_eval_data.py uses
# range(6, 23), i.e. 06:00..22:00 inclusive). Match it so training data shares
# the same character (daytime hours, zeros included) as the test data.
DAYTIME_HOURS = list(range(6, 23))

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]
META_COLS = [
    "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
    "restaurant_cafe_count_500m", "transit_stop_count_500m",
    "distance_to_nearest_rail_station", "distance_to_city_center",
]
FLAG_COLS = ["weekend", "holiday", "working_day"]


def normalize_station_id(s: pd.Series) -> pd.Series:
    """Make 31631, 31631.0 and '31631.0' all become '31631' (matches evaluator)."""
    raw = s.astype("string").str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    int_like = num.notna() & np.isfinite(num) & (num % 1 == 0)
    out = raw.copy()
    out.loc[int_like] = num.loc[int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add city_key, station_key, a floored hourly timestamp, and time parts."""
    out = df.copy()

    out["city_key"] = (
        out["city"].astype("string").fillna("__missing_city__")
        if "city" in out.columns else "__all__"
    )

    stn = "start_station_id" if "start_station_id" in out.columns else "station_id"
    out["station_key"] = normalize_station_id(out[stn])

    if "hour_ts" in out.columns:
        ts = pd.to_datetime(out["hour_ts"], errors="coerce")
    elif "target_hour_start" in out.columns:
        ts = pd.to_datetime(out["target_hour_start"], errors="coerce")
    elif "started_at" in out.columns:
        ts = pd.to_datetime(out["started_at"], errors="coerce")
    else:  # test rows that only carry date + hour
        ts = pd.to_datetime(out["date"], errors="coerce") + pd.to_timedelta(
            pd.to_numeric(out["hour"], errors="coerce").fillna(0), unit="h"
        )

    ts = ts.dt.floor("h")
    out["ts"] = ts
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday
    out["how"] = out["weekday"] * 24 + out["hour"]  # hour-of-week (0..167)
    out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)
    return out


def fit_encodings(table: pd.DataFrame) -> dict:
    """Hierarchical mean-demand lookups, from specific to general."""
    g = table.groupby
    return {
        "station_how": g(["station_key", "how"])["demand"].mean().to_dict(),
        "station": g("station_key")["demand"].mean().to_dict(),
        "city_how": g(["city_key", "how"])["demand"].mean().to_dict(),
        "city": g("city_key")["demand"].mean().to_dict(),
        "global": float(table["demand"].mean()),
    }


def build_features(df_keyed: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Build the numeric feature matrix. Shared by train and predict."""
    X = pd.DataFrame(index=df_keyed.index)

    # cyclical time encodings (23:00 is adjacent to 00:00)
    X["hour_sin"] = np.sin(2 * np.pi * df_keyed["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df_keyed["hour"] / 24)
    X["dow_sin"] = np.sin(2 * np.pi * df_keyed["weekday"] / 7)
    X["dow_cos"] = np.cos(2 * np.pi * df_keyed["weekday"] / 7)
    X["hour"] = df_keyed["hour"]
    X["weekday"] = df_keyed["weekday"]
    X["is_weekend"] = df_keyed["is_weekend"]

    # weather + calendar flags + station metadata (whatever is present)
    for c in WEATHER_COLS + FLAG_COLS + META_COLS:
        X[c] = pd.to_numeric(df_keyed[c], errors="coerce") if c in df_keyed.columns else np.nan

    # hierarchical target features with graceful fallback to the next-more-general level
    sk, ck, how = df_keyed["station_key"], df_keyed["city_key"], df_keyed["how"]
    g = enc["global"]
    city_te = ck.map(enc["city"]).fillna(g)
    city_how = pd.Series(list(zip(ck, how)), index=df_keyed.index).map(enc["city_how"]).fillna(city_te)
    station_te = sk.map(enc["station"]).fillna(city_te)
    stn_how = pd.Series(list(zip(sk, how)), index=df_keyed.index).map(enc["station_how"]).fillna(station_te)

    X["te_city"] = city_te
    X["te_city_how"] = city_how
    X["te_station"] = station_te
    X["te_station_how"] = stn_how  # usually the single most predictive feature
    return X


class BikeDemandModel:
    """Gradient-boosted-trees demand model with hierarchical fallback features."""

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")
        a = self.artifacts

        keyed = add_keys(test_df)                       # same keying as training
        X = build_features(keyed, a["encodings"])
        X = X.reindex(columns=a["feature_cols"], fill_value=np.nan)  # exact column order

        p1 = a["model_poisson"].predict(X)
        p2 = a["model_mae"].predict(X)
        preds = 0.5 * (p1 + p2)                         # small ensemble
        return np.maximum(0.0, preds)                   # demand is non-negative
