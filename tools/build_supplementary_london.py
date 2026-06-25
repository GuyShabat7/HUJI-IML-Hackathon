#!/usr/bin/env python3
"""
build_supplementary_london.py — enricher for supplemental external ride data.

Turns a raw start-side ride file (e.g. the `london-2025-data` GitHub release via
`tools/fetch_london_tfl.py`, or any city's raw rides reduced to `started_at` +
`start_station_id`) into a CSV that matches `dataset/train_set.csv`'s schema, and
writes it to `dataset/supplemental/` where the harness auto-discovers it.

External data is course-approved for training + validation, and supplemental
sources are **on by default** in the harness (see data.py `load_splits`). So once
you run this, no extra wiring is needed:

    python tools/build_supplementary_london.py --raw dataset/london_2025_full_year_start.csv
    # then, anywhere:
    from data import load_splits;  sp = load_splits()   # picks it up automatically

Use `--city "city 2"` (etc.) for other cities; despite the filename this is not
London-specific. The single hard rule still holds: **city 3 stays hidden during
training** — the harness routes any city-3 rows to the unseen-city test, never train.

What it reconstructs (raw releases have no weather/POI/calendar):
  - calendar  : derived locally from the timestamp; `holiday`/`holiday_name`
                replicate the course's US-federal-holiday convention (README §171).
  - station   : lat/lng + POI columns back-filled from train_set.csv by station id
                (London's 807 ids are identical to TfL numbers; POIs are 0 there).
  - weather   : `--weather trainset` joins train_set's weather on the overlap range;
                `--weather openmeteo` fetches the FULL range from Open-Meteo (the
                transferable win — a warm-season temperature->demand curve for
                M_weather); `--weather none` leaves it NaN (the gate handles it).

Self-contained: numpy / pandas + stdlib urllib. Run from the repo root.

    python tools/build_supplementary_london.py \
        --raw dataset/london_2025_full_year_start.csv \
        --weather trainset                       # -> dataset/supplemental/..._enriched.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CITY = "city 1"  # London; override with --city for D.C. etc.

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]
STATION_META_COLS = [
    "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
    "restaurant_cafe_count_500m", "transit_stop_count_500m",
    "distance_to_nearest_rail_station", "distance_to_city_center",
]

# US federal holidays for 2025 (the whole release is 2025). Names match the
# Python `holidays` US set, which is what train_set.csv uses — verified against
# the three present in Jan-Feb: New Year's Day / MLK / Washington's Birthday.
# None of 2025's federal holidays fall on a weekend, so no observed-day shifts.
US_FEDERAL_HOLIDAYS_2025 = {
    "2025-01-01": "New Year's Day",
    "2025-01-20": "Martin Luther King Jr. Day",
    "2025-02-17": "Washington's Birthday",
    "2025-05-26": "Memorial Day",
    "2025-06-19": "Juneteenth National Independence Day",
    "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day",
    "2025-10-13": "Columbus Day",
    "2025-11-11": "Veterans Day",
    "2025-11-27": "Thanksgiving",
    "2025-12-25": "Christmas Day",
}

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# --------------------------------------------------------------------------- #
# Keying (kept consistent with submissions/.../data.py)
# --------------------------------------------------------------------------- #
def normalize_station_id(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    num = pd.to_numeric(raw, errors="coerce")
    int_like = num.notna() & np.isfinite(num) & (num % 1 == 0)
    out = raw.copy()
    out.loc[int_like] = num.loc[int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


# --------------------------------------------------------------------------- #
# Calendar (local, deterministic)
# --------------------------------------------------------------------------- #
def derive_calendar(ts: pd.Series) -> pd.DataFrame:
    """Rebuild train_set's calendar columns from a timestamp series."""
    ts = pd.to_datetime(ts, errors="coerce")
    date = ts.dt.date.astype("string")
    weekday = ts.dt.weekday  # Mon=0 .. Sun=6
    weekend = weekday.isin([5, 6]).astype("int64")
    hol_name = date.map(US_FEDERAL_HOLIDAYS_2025).astype("string")
    holiday = hol_name.notna().astype("int64")
    non_2025 = ts.dt.year.ne(2025)
    if bool(non_2025.any()):
        print(f"  [warn] {int(non_2025.sum())} rows are not 2025; "
              f"holiday flags only cover 2025.", file=sys.stderr)
    working_day = ((weekend == 0) & (holiday == 0)).astype("int64")
    return pd.DataFrame({
        "date": date, "weekday": weekday, "weekend": weekend,
        "holiday": holiday, "holiday_name": hol_name, "working_day": working_day,
    }, index=ts.index)


# --------------------------------------------------------------------------- #
# Station metadata + weather back-fill from the official train_set
# --------------------------------------------------------------------------- #
def load_station_meta(train_csv: Path, city: str) -> pd.DataFrame:
    """One row per station in ``city``: lat/lng + POI columns, keyed by station_key."""
    cols = ["city", "start_station_id"] + STATION_META_COLS
    df = pd.read_csv(train_csv, usecols=lambda c: c in cols, low_memory=False)
    df = df[df["city"] == city].copy()
    df["station_key"] = normalize_station_id(df["start_station_id"])
    keep = [c for c in STATION_META_COLS if c in df.columns]
    meta = df.groupby("station_key", dropna=False)[keep].first().reset_index()
    return meta


def load_hourly_weather(train_csv: Path, city: str) -> pd.DataFrame:
    """One row per hour in ``city``: weather columns, keyed by hour_ts (train range)."""
    cols = ["city", "hour_ts"] + WEATHER_COLS
    df = pd.read_csv(train_csv, usecols=lambda c: c in cols, low_memory=False)
    df = df[df["city"] == city].copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce").dt.floor("h")
    keep = [c for c in WEATHER_COLS if c in df.columns]
    wx = df.groupby("hour_ts", dropna=False)[keep].first().reset_index()
    return wx


# --------------------------------------------------------------------------- #
# Optional Open-Meteo fetch (the transferable, warm-season weather win)
# --------------------------------------------------------------------------- #
def fetch_open_meteo_hourly(lat: float, lng: float,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly weather for a single point from the Open-Meteo archive API.

    Variable names already match WEATHER_COLS, so the result joins on hour_ts
    with no renaming. Network call — only used with `--weather openmeteo`.
    """
    params = {
        "latitude": f"{lat:.4f}", "longitude": f"{lng:.4f}",
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_COLS), "timezone": "auto",
    }
    url = f"{OPEN_METEO_ARCHIVE}?{urllib.parse.urlencode(params)}"
    print(f"  [openmeteo] GET {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (trusted host)
        payload = json.loads(resp.read().decode("utf-8"))
    hourly = payload["hourly"]
    wx = pd.DataFrame(hourly)
    wx["hour_ts"] = pd.to_datetime(wx.pop("time"), errors="coerce").dt.floor("h")
    return wx[["hour_ts"] + [c for c in WEATHER_COLS if c in wx.columns]]


# --------------------------------------------------------------------------- #
# Enrich
# --------------------------------------------------------------------------- #
def enrich(raw: pd.DataFrame, train_csv: Path,
           city: str = DEFAULT_CITY, weather: str = "trainset") -> pd.DataFrame:
    """Raw start-side rides for ``city`` -> train_set.csv-schema, ride-level DataFrame."""
    out = pd.DataFrame(index=raw.index)
    out["started_at"] = pd.to_datetime(raw["started_at"], errors="coerce")
    out = out.dropna(subset=["started_at"]).copy()
    out["start_station_id"] = raw.loc[out.index, "start_station_id"]
    out["city"] = city
    out["hour_ts"] = out["started_at"].dt.floor("h")
    station_key = normalize_station_id(out["start_station_id"])

    # calendar
    cal = derive_calendar(out["started_at"])
    for c in cal.columns:
        out[c] = cal[c]

    # station metadata
    meta = load_station_meta(train_csv, city).set_index("station_key")
    for c in STATION_META_COLS:
        out[c] = station_key.map(meta[c]) if c in meta.columns else np.nan

    # weather
    if weather == "none":
        for c in WEATHER_COLS:
            out[c] = np.nan
    else:
        if weather == "openmeteo":
            lat, lng = float(meta["start_lat"].median()), float(meta["start_lng"].median())
            if not (np.isfinite(lat) and np.isfinite(lng)):
                raise SystemExit(
                    f"--weather openmeteo needs station coords, but {city!r} has none "
                    f"in train_set.csv. Use --weather trainset/none instead."
                )
            d0 = out["hour_ts"].min().date().isoformat()
            d1 = out["hour_ts"].max().date().isoformat()
            wx = fetch_open_meteo_hourly(lat, lng, d0, d1).set_index("hour_ts")
        else:  # "trainset"
            wx = load_hourly_weather(train_csv, city).set_index("hour_ts")
        for c in WEATHER_COLS:
            out[c] = out["hour_ts"].map(wx[c]) if c in wx.columns else np.nan
        missing = out[WEATHER_COLS].isna().all(axis=1).mean()
        if missing:
            print(f"  [info] {missing:.1%} of rows have no weather match "
                  f"(expected outside the source's date range).")

    # serialize timestamps like train_set.csv
    out["hour_ts"] = out["hour_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out["started_at"] = out["started_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True,
                    help="Raw start-side rides CSV (needs started_at + start_station_id).")
    ap.add_argument("--city", default=DEFAULT_CITY,
                    help="City label to stamp + back-fill against (default 'city 1' = London).")
    ap.add_argument("--train-csv", default="dataset/train_set.csv",
                    help="Official train_set.csv (source for station-meta + train-range weather).")
    ap.add_argument("--out", default=None,
                    help="Output CSV. Default: dataset/supplemental/<raw_stem>_enriched.csv "
                         "(auto-discovered by load_splits).")
    ap.add_argument("--weather", choices=["trainset", "openmeteo", "none"], default="trainset",
                    help="How to fill weather. 'openmeteo' = fetch full range (network).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only read the first N raw rows (for quick tests).")
    args = ap.parse_args()

    raw_path, train_csv = Path(args.raw), Path(args.train_csv)
    out_path = (Path(args.out) if args.out
                else Path("dataset/supplemental") / f"{raw_path.stem}_enriched.csv")
    if not raw_path.exists():
        raise SystemExit(
            f"Raw file not found: {raw_path}\n"
            f"Get it from the london-2025-data release or run tools/fetch_london_tfl.py."
        )

    raw = pd.read_csv(raw_path, nrows=args.limit, low_memory=False)
    print(f"raw rows={len(raw):,}  city={args.city!r}  weather={args.weather}")
    enriched = enrich(raw, train_csv, city=args.city, weather=args.weather)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(out_path, index=False)
    print(f"wrote {len(enriched):,} enriched {args.city} rows -> {out_path}")
    print("It will be auto-merged by the harness (supplemental is on by default):")
    print("    from data import load_splits;  load_splits()")


if __name__ == "__main__":
    main()
