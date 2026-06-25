#!/usr/bin/env python3
"""
data.py — Dev 1 data & validation harness for the missingness-aware ensemble.

This is the single data entry point the other devs import. It:

  1. Builds the *labeled* station-hour table (with reconstructed zero hours) from
     the raw ride-level data, mirroring the proven
     ``challenge_1_IDs/train.py:build_training_table`` pattern and the helpers in
     ``build_station_hour_eval_data.py`` (MODELPLAN.md §5).
  2. Pools the official data with any approved **supplemental** sources (external
     data is course-approved and ON by default — auto-discovered from
     ``dataset/supplemental/``; see load_splits) and splits into (MODELPLAN.md §6):
        - TRAIN  : in-distribution cities (``city 1`` + ``city 2``), earliest
                   ``1 - val_fraction`` by time — official + supplemental
        - VAL    : in-distribution cities, latest ``val_fraction`` by time —
                   official + supplemental (early stopping, model selection, and
                   the base preds that train the orchestrator)
        - TEST   : *all* of ``city 3`` — the never-trained-on "brand-new city"
                   probe. The one hard rule: **city 3 is hidden during training.**
  3. Caches built tables under ``_cache/`` (gitignored) so repeated calls are
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
    "DAYTIME_HOURS", "HOLDOUT_CITY", "VAL_FRACTION", "SUPPLEMENTAL_DIR",
    "WEATHER_COLS", "CALENDAR_COLS", "STATION_META_COLS",
    "normalize_station_id", "add_keys", "build_station_hour_table",
    "build_or_load_table", "discover_supplemental_sources",
    "load_splits", "city_balanced_weights", "Splits",
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

# Supplemental sources are ENABLED by default (course approved external data for
# training + validation). Any train_set.csv-schema CSV dropped here is auto-merged
# into the pool and split like official data — see load_splits(). The directory is
# gitignored and may be empty/absent (then the harness falls back to train_set.csv
# only). Produce sources with tools/build_supplementary_london.py (any city).
SUPPLEMENTAL_DIR = DATA_ROOT / "supplemental"

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


def _cache_paths_for(train_csv: Path) -> tuple[Path, Path]:
    """Per-source cache filenames so different sources don't clobber each other.

    The canonical train_set.csv keeps the original, stable cache names; any other
    source (e.g. an enriched London CSV) gets its own pair derived from its stem.
    """
    if train_csv == TRAIN_CSV.resolve():
        return TABLE_CACHE, TABLE_CACHE_META
    return CACHE_DIR / f"table_{train_csv.stem}.pkl", CACHE_DIR / f"table_{train_csv.stem}.meta.json"


def build_or_load_table(
    train_csv: str | Path = TRAIN_CSV,
    use_cache: bool = True,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Build the station-hour table, reusing ``_cache/`` when the source is unchanged."""
    train_csv = Path(train_csv).resolve()
    cache_path, meta_path = _cache_paths_for(train_csv)

    if use_cache and not rebuild and cache_path.exists() and meta_path.exists():
        try:
            cached_sig = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            cached_sig = None
        if cached_sig == _source_signature(train_csv):
            return pd.read_pickle(cache_path)

    rides = pd.read_csv(train_csv, low_memory=False)
    table = build_station_hour_table(rides)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_pickle(cache_path)
        meta_path.write_text(json.dumps(_source_signature(train_csv)))

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


def discover_supplemental_sources(directory: str | Path = SUPPLEMENTAL_DIR) -> list[Path]:
    """All ``*.csv`` supplemental sources in ``directory`` (sorted; [] if absent).

    Drop any number of train_set.csv-schema CSVs here (London, D.C., …) and they
    are auto-merged by ``load_splits``. Empty/missing directory ⇒ official-only.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.csv") if p.is_file())


def load_splits(
    train_csv: str | Path = TRAIN_CSV,
    val_fraction: float = VAL_FRACTION,
    holdout_city: str = HOLDOUT_CITY,
    use_cache: bool = True,
    rebuild: bool = False,
    supplemental: str | list | None = "auto",
) -> Splits:
    """Build (or load) the pooled table and return train / val / unseen-city splits.

    Supplemental external data is **enabled by default** (course approved it for
    training + validation). The official ``train_set.csv`` and every supplemental
    source are concatenated into one pool, exact ``(city, station, hour)`` overlaps
    are de-duplicated keeping the official row, then:

      - **city 3 (``holdout_city``) is pulled out whole** → ``test_unseen``. It is
        the never-trained-on "brand-new city" probe and is used for testing/
        validation only. The single hard rule — *city 3 stays hidden during
        training* — is enforced by removing it before the split and by a guard.
      - every other city is split by a **per-city temporal cut** (latest
        ``val_fraction`` by time → ``val``; the rest → ``train``), matching
        make_local_split.py. Supplemental rows take part in this split, so they
        appear in **both train and val** (a clean temporal split on the union ⇒ no
        train/val leakage). Every row carries an ``is_supplementary`` flag.

    ``supplemental``:
        ``"auto"`` (default) — auto-discover CSVs in ``SUPPLEMENTAL_DIR``;
        ``list`` of paths — use exactly those; ``None`` / ``[]`` — official only
        (use this for the honest train_set-only baseline comparison, §9).
    """
    official = build_or_load_table(train_csv, use_cache=use_cache, rebuild=rebuild)
    official["is_supplementary"] = False

    if supplemental == "auto":
        sources = discover_supplemental_sources()
    elif supplemental in (None, []):
        sources = []
    else:
        sources = [Path(p) for p in supplemental]

    frames = [official]
    for src in sources:
        extra = build_or_load_table(src, use_cache=use_cache, rebuild=rebuild)
        extra["is_supplementary"] = True
        frames.append(extra)

    # Pool, then drop exact station-hour duplicates keeping the official copy
    # (official is first, so keep="first" prefers it and its ground-truth demand).
    keys = ["city_key", "station_key", "ts"]
    pool = pd.concat(frames, ignore_index=True).drop_duplicates(subset=keys, keep="first")

    if not (pool["city_key"] == holdout_city).any():
        raise ValueError(
            f"holdout_city {holdout_city!r} not found "
            f"(present: {sorted(pool['city_key'].unique())})."
        )

    # Hard rule: city 3 is hidden during training. Pull it out whole *before* the
    # split → only ever in test_unseen, never in train or val, regardless of source.
    test_unseen = pool[pool["city_key"] == holdout_city].copy().reset_index(drop=True)
    in_dist = pool[pool["city_key"] != holdout_city]

    train_parts, val_parts, cutoffs = [], [], {}
    for ck, g in in_dist.groupby("city_key", dropna=False):
        g = g.sort_values("ts")
        cut = g["ts"].quantile(1.0 - val_fraction)
        cutoffs[str(ck)] = cut
        train_parts.append(g[g["ts"] <= cut])
        val_parts.append(g[g["ts"] > cut])

    train = pd.concat(train_parts).reset_index(drop=True)
    val = pd.concat(val_parts).reset_index(drop=True)

    if holdout_city in (set(train["city_key"]) | set(val["city_key"])):
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
    sup = int(sp.train.get("is_supplementary", pd.Series(dtype=bool)).sum())
    val_sup = int(sp.val.get("is_supplementary", pd.Series(dtype=bool)).sum())
    print(f"\nSupplemental rows: train={sup:,}  val={val_sup:,}  "
          f"(0 ⇒ no sources in {SUPPLEMENTAL_DIR.name}/ ; train_set.csv only)")
    print("Per-city temporal cutoffs (<= cut -> train ; > cut -> val):")
    for ck, cut in sp.cutoffs.items():
        print(f"  {ck}: {cut}")
    w = city_balanced_weights(sp.train)
    print(f"\nTrain city-balanced weights: min={w.min():.3f} max={w.max():.3f} "
          f"mean={w.mean():.3f}  (rows={len(w):,})")


if __name__ == "__main__":
    splits = load_splits(rebuild=True)  # force a fresh build + cache
    _print_summary(splits)
    print(f"\nCached station-hour table -> {TABLE_CACHE}")
