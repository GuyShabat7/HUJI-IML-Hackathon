#!/usr/bin/env python3
"""
data.py — Dev 1 data & validation harness for the missingness-aware ensemble.

This is the single data entry point the other devs import. It:

  1. Builds the *labeled* station-hour table (with reconstructed zero hours) from
     the raw ride-level data, mirroring the proven
     ``challenge_1_IDs/train.py:build_training_table`` pattern and the helpers in
     ``build_station_hour_eval_data.py`` (MODELPLAN.md §5).
  2. Splits it into the three sets the project agreed on (MODELPLAN.md §6):
        - TRAIN  : ``city 1`` + ``city 2``, earliest ``1 - val_fraction`` by time
        - VAL    : ``city 1`` + ``city 2``, latest ``val_fraction`` by time
                   (the in-distribution temporal holdout used for early stopping,
                   model selection, and to produce the base preds that train the
                   orchestrator)
        - TEST   : *all* of ``city 3`` — the never-trained-on "brand-new city"
                   generalization probe
  3. Caches the built table under ``_cache/`` (gitignored) so repeated calls are
     cheap, and exposes ``load_splits()`` + ``city_balanced_weights()`` helpers.

Only depends on numpy / pandas / joblib (no xgboost), so it is cheap to import.

Run directly to (re)build + cache the table and print a summary:

    cd submissions/challenge_1_ensamble
    python data.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DAYTIME_HOURS", "HOLDOUT_CITY", "VAL_FRACTION",
    "WEATHER_COLS", "CALENDAR_COLS", "STATION_META_COLS",
    "normalize_station_id", "add_keys", "build_station_hour_table",
    "build_or_load_table", "load_splits", "city_balanced_weights", "Splits",
]

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DATA_ROOT = (HERE / ".." / ".." / "dataset").resolve()
TRAIN_CSV = DATA_ROOT / "train_set.csv"
CACHE_DIR = HERE / "_cache"
TABLE_CACHE = CACHE_DIR / "station_hour_table.pkl"
TABLE_CACHE_META = CACHE_DIR / "station_hour_table.meta.json"

# Daytime hours kept by the evaluator (build_station_hour_eval_data.py uses
# range(6, 23) == 06:00..22:00 inclusive). Match it so the reconstructed grid
# shares the same character (daytime hours, zeros included) as graded test data.
DAYTIME_HOURS = list(range(6, 23))

# The never-trained-on city: held out entirely as the "brand-new city" probe.
# This harness reads ONLY dataset/train_set.csv — it never pulls the supplementary
# full-year London release (README "Supplementary Data"). That data is London-only,
# opt-in, and rules-gated; there is no supplementary data for city 3, so city 3 stays
# a genuine unknown. Any future London enrichment (MODELPLAN §4) must be confined to
# city 1 *train* rows and must NEVER touch the val or unseen-city sets — load_splits
# enforces this with a hard leakage guard.
HOLDOUT_CITY = "city 3"

# Fraction of the latest time, *per city*, used as the in-distribution holdout.
VAL_FRACTION = 0.20

# Column groups carried through the table (mirrored from
# build_station_hour_eval_data.py). The harness makes everything available;
# feature *selection* (and the identity-feature exclusions) happen later in
# model.py — see MODELPLAN.md §3.
WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]
CALENDAR_COLS = ["weekend", "holiday", "working_day", "holiday_name"]
STATION_META_COLS = [
    "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
    "restaurant_cafe_count_500m", "transit_stop_count_500m",
    "distance_to_nearest_rail_station", "distance_to_city_center",
]


# --------------------------------------------------------------------------- #
# Keying helpers (kept self-contained; mirrors challenge_1_IDs/model.py)
# --------------------------------------------------------------------------- #
def normalize_station_id(s: pd.Series) -> pd.Series:
    """Make 31631, 31631.0 and '31631.0' all become '31631' (matches evaluator)."""
    raw = s.astype("string").str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    int_like = num.notna() & np.isfinite(num) & (num % 1 == 0)
    out = raw.copy()
    out.loc[int_like] = num.loc[int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add city_key, station_key, and a floored hourly timestamp ``ts``.

    Accepts ride-level rows (``hour_ts`` / ``started_at``) or already-built
    station-hour rows (``target_hour_start``), so it works at both train and
    eval time.
    """
    out = df.copy()

    out["city_key"] = (
        out["city"].astype("string").fillna("__missing_city__")
        if "city" in out.columns else pd.Series("__all__", index=out.index, dtype="string")
    )

    stn = "start_station_id" if "start_station_id" in out.columns else "station_id"
    out["station_key"] = normalize_station_id(out[stn])

    if "hour_ts" in out.columns:
        ts = pd.to_datetime(out["hour_ts"], errors="coerce")
    elif "target_hour_start" in out.columns:
        ts = pd.to_datetime(out["target_hour_start"], errors="coerce")
    elif "started_at" in out.columns:
        ts = pd.to_datetime(out["started_at"], errors="coerce")
    else:  # rows that only carry date + hour
        ts = pd.to_datetime(out["date"], errors="coerce") + pd.to_timedelta(
            pd.to_numeric(out["hour"], errors="coerce").fillna(0), unit="h"
        )

    out["ts"] = ts.dt.floor("h")
    return out


# --------------------------------------------------------------------------- #
# Labeled station-hour table with zero reconstruction
# --------------------------------------------------------------------------- #
def build_station_hour_table(rides: pd.DataFrame) -> pd.DataFrame:
    """Ride-level rows -> labeled station-hour table WITH reconstructed zeros.

    Steps (mirrors challenge_1_IDs build_training_table):
      1. observed counts per (city, station, hour),
      2. an active-window daytime grid per station — this is where the zero
         hours come from,
      3. join demand onto the grid (missing grid hours are TRUE zeros),
      4. attach per-(city,hour) weather/calendar and per-station metadata,
      5. add convenience time parts.
    """
    keyed = add_keys(rides)
    keyed = keyed.dropna(subset=["ts"])

    # 1) observed demand: rides per (city, station, hour)
    demand = (
        keyed.groupby(["city_key", "station_key", "ts"], dropna=False)
        .size().reset_index(name="demand")
    )

    # 2) active-window daytime grid per station -> the source of the zeros
    grids = []
    for (ck, sk), g in keyed.groupby(["city_key", "station_key"], dropna=False):
        lo, hi = g["ts"].min(), g["ts"].max()
        if pd.isna(lo) or pd.isna(hi):
            continue
        hours = pd.date_range(lo, hi, freq="h")
        hours = hours[hours.hour.isin(DAYTIME_HOURS)]  # daytime only, like evaluator
        if len(hours):
            grids.append(pd.DataFrame({"city_key": ck, "station_key": sk, "ts": hours}))
    grid = pd.concat(grids, ignore_index=True)
    # Match key dtypes so the merges below align cleanly (date_range/loop produce
    # object columns; the keyed frame uses pandas "string").
    grid["city_key"] = grid["city_key"].astype("string")
    grid["station_key"] = grid["station_key"].astype("string")

    # 3) attach demand; missing grid hours are TRUE ZEROS
    table = grid.merge(demand, on=["city_key", "station_key", "ts"], how="left")
    table["demand"] = table["demand"].fillna(0).astype("int64")

    # 4) per-(city,hour) weather/calendar + per-station metadata (first non-null)
    wcols = [c for c in WEATHER_COLS + CALENDAR_COLS if c in keyed.columns]
    if wcols:
        city_hour = keyed.groupby(["city_key", "ts"], dropna=False)[wcols].first().reset_index()
        table = table.merge(city_hour, on=["city_key", "ts"], how="left")
    mcols = [c for c in STATION_META_COLS if c in keyed.columns]
    if mcols:
        station_meta = (
            keyed.groupby(["city_key", "station_key"], dropna=False)[mcols]
            .first().reset_index()
        )
        table = table.merge(station_meta, on=["city_key", "station_key"], how="left")

    # 5) convenience columns (kept for splitting/reporting; NOT all are features)
    table["city"] = table["city_key"]                 # human-readable, train-time only
    table["hour"] = table["ts"].dt.hour
    table["weekday"] = table["ts"].dt.weekday
    table["is_weekend"] = table["weekday"].isin([5, 6]).astype("int64")
    table["how"] = table["weekday"] * 24 + table["hour"]  # hour-of-week (0..167)

    return table.sort_values(["city_key", "station_key", "ts"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def _source_signature(path: Path) -> dict:
    """Cheap staleness check: rebuild the cache if the source csv changes."""
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def build_or_load_table(
    train_csv: str | Path = TRAIN_CSV,
    use_cache: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Build the station-hour table, reusing ``_cache/`` when the source is unchanged."""
    train_csv = Path(train_csv).resolve()

    if use_cache and not rebuild and TABLE_CACHE.exists() and TABLE_CACHE_META.exists():
        try:
            cached_sig = json.loads(TABLE_CACHE_META.read_text())
        except (OSError, ValueError):
            cached_sig = None
        if cached_sig == _source_signature(train_csv):
            return pd.read_pickle(TABLE_CACHE)

    rides = pd.read_csv(train_csv, low_memory=False)
    table = build_station_hour_table(rides)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_pickle(TABLE_CACHE)
        TABLE_CACHE_META.write_text(json.dumps(_source_signature(train_csv)))

    return table


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    """The three agreed evaluation sets (MODELPLAN.md §6)."""

    train: pd.DataFrame        # c1+c2, earliest (1 - val_fraction) by time
    val: pd.DataFrame          # c1+c2, latest val_fraction by time (in-dist holdout)
    test_unseen: pd.DataFrame  # all of holdout_city (unseen-city probe)
    holdout_city: str
    val_fraction: float
    cutoffs: dict = field(default_factory=dict)  # per-city temporal cut used

    def per_city_counts(self) -> pd.DataFrame:
        """Tidy row/demand stats per split per city — handy for sanity checks."""
        rows = []
        for name, frame in (("train", self.train), ("val", self.val),
                            ("test", self.test_unseen)):
            for ck, g in frame.groupby("city_key", dropna=False):
                rows.append({
                    "split": name, "city": str(ck), "rows": len(g),
                    "mean_demand": float(g["demand"].mean()),
                    "zero_rate": float((g["demand"] == 0).mean()),
                    "max_demand": int(g["demand"].max()),
                })
        return pd.DataFrame(rows)


def load_splits(
    train_csv: str | Path = TRAIN_CSV,
    val_fraction: float = VAL_FRACTION,
    holdout_city: str = HOLDOUT_CITY,
    use_cache: bool = True,
    rebuild: bool = False,
) -> Splits:
    """Build (or load) the table and return the train / val / unseen-city splits.

    The temporal cut is taken *per city* (matching make_local_split.py) so each
    in-distribution city contributes the same latest-``val_fraction`` slice; a
    random split would leak the future. The holdout city is removed *before*
    the temporal split and returned whole as ``test_unseen``.
    """
    table = build_or_load_table(train_csv, use_cache=use_cache, rebuild=rebuild)

    test_unseen = table[table["city_key"] == holdout_city].copy()
    in_dist = table[table["city_key"] != holdout_city]

    train_parts, val_parts, cutoffs = [], [], {}
    for ck, g in in_dist.groupby("city_key", dropna=False):
        g = g.sort_values("ts")
        cut = g["ts"].quantile(1.0 - val_fraction)
        cutoffs[str(ck)] = cut
        train_parts.append(g[g["ts"] <= cut])
        val_parts.append(g[g["ts"] > cut])

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)

    # Hard invariant: the holdout city must stay a genuine unknown. This catches a
    # mistyped city name *and* any future enriched/extended source (e.g. full-year
    # London, §4) that might accidentally carry city-3 rows into train/val.
    if len(test_unseen) == 0:
        raise ValueError(
            f"holdout_city {holdout_city!r} not found in the table "
            f"(present: {sorted(table['city_key'].unique())})."
        )
    leaked = set(train["city_key"]) | set(val["city_key"])
    if holdout_city in leaked:
        raise AssertionError(f"holdout_city {holdout_city!r} leaked into train/val.")

    return Splits(
        train=train, val=val, test_unseen=test_unseen,
        holdout_city=holdout_city, val_fraction=val_fraction, cutoffs=cutoffs,
    )


# --------------------------------------------------------------------------- #
# City-balanced sample weights (MODELPLAN.md §6)
# --------------------------------------------------------------------------- #
def city_balanced_weights(df: pd.DataFrame, city_col: str = "city_key") -> np.ndarray:
    """Inverse per-city-frequency weights, normalized to mean 1.

    London (``city 1``) supplies far more rows than the others; without this the
    loss would be dominated by it. Mean-1 normalization keeps the effective
    learning rate comparable to the unweighted case.
    """
    counts = df[city_col].map(df[city_col].value_counts())
    w = 1.0 / counts.astype(float)
    return (w / w.mean()).to_numpy()


# --------------------------------------------------------------------------- #
# Script entry point: (re)build + cache the table and print a summary
# --------------------------------------------------------------------------- #
def _print_summary(sp: Splits) -> None:
    print(f"Splits  holdout_city={sp.holdout_city!r}  val_fraction={sp.val_fraction}\n")
    counts = sp.per_city_counts()
    with pd.option_context("display.width", 120):
        print(counts.to_string(index=False))
    print("\nPer-city temporal cutoffs (<= cut -> train ; > cut -> val):")
    for ck, cut in sp.cutoffs.items():
        print(f"  {ck}: {cut}")
    w = city_balanced_weights(sp.train)
    print(f"\nTrain city-balanced weights: min={w.min():.3f} max={w.max():.3f} "
          f"mean={w.mean():.3f}  (rows={len(w):,})")


if __name__ == "__main__":
    splits = load_splits(rebuild=True)  # force a fresh build + cache
    _print_summary(splits)
    print(f"\nCached station-hour table -> {TABLE_CACHE}")
