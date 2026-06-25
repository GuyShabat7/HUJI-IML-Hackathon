"""features.py — shared feature builders for the bike-demand contenders.

Single source of truth used by both the Step-2 bake-off (tools/bakeoff.py) and the
Step-3 submission (model.py), so train and predict always see identical features.

Two feature sets:
  * build_no_identity(df)            — MODELPLAN "B": weather + calendar + built-environment
                                       only; NO city/station identity, NO lat/lng, NO target
                                       encodings. Missingness-aware (see apply_missingness).
  * build_fallback_te_* (df, enc)    — MODELPLAN "C": B's features PLUS hierarchical
                                       target encodings with fallback
                                       (station×how → station → city×how → city → global),
                                       fit OUT-OF-FOLD on train to avoid leakage.

Self-contained (numpy/pandas/sklearn only) so the submission's model.py can import it at
predict time without dragging in the data-building pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]
CALENDAR_FLAGS = ["weekend", "holiday", "working_day"]
# Built-environment columns (station metadata MINUS lat/lng identity).
POI_COLS = [
    "bike_lane_length_500m", "park_area_500m", "university_count_1000m",
    "office_poi_count_1000m", "retail_poi_count_1000m", "restaurant_cafe_count_500m",
    "transit_stop_count_500m", "distance_to_nearest_rail_station", "distance_to_city_center",
]
# The "presence" POI columns: if ALL of these are 0 the station metadata is a sentinel
# (London), not real signal -> treat as missing. distance_to_city_center is excluded
# because it is a real non-zero value even where the POI counts are absent.
_POI_PRESENCE = [
    "bike_lane_length_500m", "park_area_500m", "university_count_1000m",
    "office_poi_count_1000m", "retail_poi_count_1000m", "restaurant_cafe_count_500m",
    "transit_stop_count_500m",
]
TE_COLS = ["te_city", "te_city_how", "te_station", "te_station_how"]


def apply_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """NaN-out sentinels so XGBoost/LightGBM treat them as missing, not as real values.

    - distance_to_nearest_rail_station == -1  -> NaN (it's a 'not found' sentinel).
    - rows where every presence-POI column is 0 -> all POI counts NaN (London: the course
      never computed station metadata, so it's uniformly 0, i.e. uninformative).
    """
    out = df.copy()
    if "distance_to_nearest_rail_station" in out.columns:
        rail = pd.to_numeric(out["distance_to_nearest_rail_station"], errors="coerce")
        out["distance_to_nearest_rail_station"] = rail.mask(rail == -1)
    present = [c for c in _POI_PRESENCE if c in out.columns]
    if present:
        vals = out[present].apply(pd.to_numeric, errors="coerce").fillna(0)
        all_zero = (vals == 0).all(axis=1)
        out.loc[all_zero, present] = np.nan
    return out


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical + raw temporal features. Works on the station-hour table (has ts/hour/...)."""
    ts = pd.to_datetime(df["ts"]) if "ts" in df.columns else pd.to_datetime(df["hour_ts"])
    hour = pd.to_numeric(df["hour"], errors="coerce") if "hour" in df.columns else ts.dt.hour
    wd = pd.to_numeric(df["weekday"], errors="coerce") if "weekday" in df.columns else ts.dt.weekday
    X = pd.DataFrame(index=df.index)
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * wd / 7)
    X["dow_cos"] = np.cos(2 * np.pi * wd / 7)
    X["hour"] = hour
    X["weekday"] = wd
    X["is_weekend"] = (df["is_weekend"] if "is_weekend" in df.columns
                       else wd.isin([5, 6]).astype(int))
    X["month"] = ts.dt.month
    X["day_of_year"] = ts.dt.dayofyear
    return X


def build_no_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set B: transferable features only (no identity / lat-lng / TE)."""
    d = apply_missingness(df)
    X = _time_features(d)
    for c in CALENDAR_FLAGS + WEATHER_COLS + POI_COLS:
        X[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return X


# --------------------------------------------------------------------------- #
# Hierarchical target encodings with fallback (feature set C)
# --------------------------------------------------------------------------- #
def te_fit(df: pd.DataFrame) -> dict:
    """Mean-demand lookups from specific to general (mirror challenge_1_IDs fit_encodings)."""
    g = df.groupby
    return {
        "station_how": g(["station_key", "how"])["demand"].mean().to_dict(),
        "station": g("station_key")["demand"].mean().to_dict(),
        "city_how": g(["city_key", "how"])["demand"].mean().to_dict(),
        "city": g("city_key")["demand"].mean().to_dict(),
        "global": float(df["demand"].mean()),
    }


def te_transform(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Apply TE with graceful fallback to the next-more-general level."""
    sk, ck, how = df["station_key"], df["city_key"], df["how"]
    g = enc["global"]
    city_te = ck.map(enc["city"]).fillna(g)
    city_how = pd.Series(list(zip(ck, how)), index=df.index).map(enc["city_how"]).fillna(city_te)
    station_te = sk.map(enc["station"]).fillna(city_te)
    stn_how = pd.Series(list(zip(sk, how)), index=df.index).map(enc["station_how"]).fillna(station_te)
    return pd.DataFrame(
        {"te_city": city_te, "te_city_how": city_how,
         "te_station": station_te, "te_station_how": stn_how},
        index=df.index,
    )


def te_oof(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Out-of-fold TE for TRAIN rows: each fold encoded from the OTHER folds only."""
    from sklearn.model_selection import KFold
    out = pd.DataFrame(index=df.index, columns=TE_COLS, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos = np.arange(len(df))
    for tr, va in kf.split(pos):
        enc = te_fit(df.iloc[tr])
        out.iloc[va] = te_transform(df.iloc[va], enc).to_numpy()
    return out


def build_fallback_te_train(train_df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Feature set C for TRAIN: B + out-of-fold TE."""
    base = build_no_identity(train_df)
    te = te_oof(train_df, n_splits=n_splits, seed=seed)
    return pd.concat([base, te], axis=1)


def build_fallback_te_eval(eval_df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    """Feature set C for VAL/TEST: B + TE transformed with encodings fit on full TRAIN."""
    base = build_no_identity(eval_df)
    te = te_transform(eval_df, enc)
    return pd.concat([base, te], axis=1)


# --------------------------------------------------------------------------- #
# Per-category feature matrices for the missingness-aware XGBoost ensemble
# (MODELPLAN §2-3): three base models, each on its own annotation category.
# --------------------------------------------------------------------------- #
def build_weather(df: pd.DataFrame) -> pd.DataFrame:
    """M_weather inputs — the 8 Open-Meteo columns only."""
    X = pd.DataFrame(index=df.index)
    for c in WEATHER_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return X


def build_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """M_calendar inputs — cyclical/temporal + calendar flags (no weather, no POI)."""
    X = _time_features(df)
    for c in CALENDAR_FLAGS:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return X


def build_station(df: pd.DataFrame) -> pd.DataFrame:
    """M_station inputs — built-environment POI only (missingness-aware)."""
    d = apply_missingness(df)
    X = pd.DataFrame(index=df.index)
    for c in POI_COLS:
        X[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return X


def category_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row missingness masks the orchestrator gates on (1 = category absent/uninformative).

    Catches 'present-but-uninformative' too (MODELPLAN §4): a station row whose POI is the
    all-zero London sentinel counts as missing even though it is not literally NaN.
    """
    d = apply_missingness(df)
    M = pd.DataFrame(index=df.index)
    wx = [c for c in WEATHER_COLS if c in d.columns]
    M["miss_weather"] = (d[wx].apply(pd.to_numeric, errors="coerce").isna().all(axis=1).astype(int)
                         if wx else 1)
    poi = [c for c in _POI_PRESENCE if c in d.columns]
    M["miss_station"] = (d[poi].apply(pd.to_numeric, errors="coerce").isna().all(axis=1).astype(int)
                         if poi else 1)
    flags = [c for c in CALENDAR_FLAGS if c in d.columns]
    M["miss_calendar"] = (d[flags].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).astype(int)
                          if flags else 1)
    return M
