#!/usr/bin/env python3
"""tools/enrich_city.py — add the train_set.csv feature columns to a start-side ride file.

Reproduces EXACTLY the columns the course added, so any coordinate-bearing raw ride file
becomes schema-compatible with dataset/train_set.csv and can be scored by our model:

  * Calendar/keys : date, weekday, weekend, holiday, holiday_name, working_day, hour_ts
                    (US federal holidays — the course used the US calendar for every city).
  * Weather       : 8 Open-Meteo Archive columns, per (city, hour) at the city centroid.
  * Station meta  : 9 OpenStreetMap columns (Overpass), computed once per unique station.

Caches Open-Meteo + Overpass responses under dataset/_cache/ (gitignored) so re-runs are
cheap and deterministic. Degrades gracefully: --weather none / --poi none leave those
columns NaN (calendar is always produced).

    python tools/enrich_city.py dataset/chicago_holdout_start.csv --name chicago
    python tools/enrich_city.py <in.csv> --weather openmeteo --poi osm --city-centroid 41.88,-87.63

Output: dataset/<name>_enriched.csv  (superset of train_set.csv columns).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "enrichment"))
import osm_poi  # noqa: E402  — Overpass bbox fetch + local KD-tree/shapely POI compute

import holidays as _holidays  # noqa: E402

CACHE = ROOT / "dataset" / "_cache"
WEATHER_COLS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m"]
POI8 = ["bike_lane_length_500m", "park_area_500m", "university_count_1000m",
        "office_poi_count_1000m", "retail_poi_count_1000m", "restaurant_cafe_count_500m",
        "transit_stop_count_500m", "distance_to_nearest_rail_station"]
# Exact train_set.csv column order (output is a superset; ride-level leakage cols are
# stubbed as NaN so the column SET matches by name).
TRAIN_COLS = [
    "started_at", "ended_at", "start_station_id", "end_station_id", "usage_time_minutes",
    "distance_meters", "user_type", "start_lat", "start_lng",
    "temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation",
    "rain", "snowfall", "cloud_cover", "wind_speed_10m", "city",
    "bike_lane_length_500m", "park_area_500m", "university_count_1000m",
    "office_poi_count_1000m", "retail_poi_count_1000m", "restaurant_cafe_count_500m",
    "transit_stop_count_500m", "distance_to_nearest_rail_station", "distance_to_city_center",
    "date", "weekday", "weekend", "holiday", "holiday_name", "working_day", "hour_ts",
]
STUB_LEAKAGE = ["ended_at", "end_station_id", "usage_time_minutes", "distance_meters", "user_type"]


def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1); dlng = np.radians(lng2 - lng1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """date, weekday, weekend, holiday, holiday_name (US), working_day, hour_ts."""
    t = pd.to_datetime(df["started_at"], errors="coerce")
    yrs = list(range(int(t.dt.year.min()), int(t.dt.year.max()) + 1))
    us = _holidays.US(years=yrs)
    df["date"] = t.dt.strftime("%Y-%m-%d")
    df["weekday"] = t.dt.weekday                       # Mon=0 .. Sun=6
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    hol = t.dt.normalize().dt.date.map(lambda d: us.get(d) if pd.notna(d) else None)
    df["holiday"] = hol.notna().astype(int)
    df["holiday_name"] = hol
    df["working_day"] = ((df["weekend"] == 0) & (df["holiday"] == 0)).astype(int)
    df["hour_ts"] = t.dt.floor("h").dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def fetch_weather(centroid, start_date, end_date) -> pd.DataFrame:
    """Open-Meteo Archive hourly weather at the city centroid, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{centroid[0]:.3f},{centroid[1]:.3f},{start_date},{end_date}".encode()).hexdigest()[:12]
    cf = CACHE / f"weather_{key}.json"
    if cf.exists():
        h = json.loads(cf.read_text())
    else:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={centroid[0]}&longitude={centroid[1]}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&hourly={','.join(WEATHER_COLS)}&timezone=auto")
        h = requests.get(url, timeout=120).json()["hourly"]
        cf.write_text(json.dumps(h))
    w = pd.DataFrame(h).rename(columns={"time": "hour_ts"})
    w["hour_ts"] = pd.to_datetime(w["hour_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return w[["hour_ts"] + WEATHER_COLS]


def compute_poi(stations: pd.DataFrame, centroid, pbf: str | None = None) -> pd.DataFrame:
    """Per unique station: 8 OSM columns + distance_to_city_center (haversine).

    Cached on disk keyed by the set of rounded (lat,lng) so re-runs don't recompute.
    POI source: Overpass API (default) OR a local .osm.pbf extract (``pbf``) — the latter
    is far faster/more reliable for whole-city bounding boxes (big cities make the Overpass
    bbox query slow), reusing enrichment/osm_poi_local.py.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    coords = stations.dropna(subset=["start_lat", "start_lng"]).copy()
    sig = hashlib.md5(
        (("pbf|" if pbf else "ovp|") +
         "|".join(sorted(f"{a:.4f},{b:.4f}" for a, b in
                         zip(coords["start_lat"], coords["start_lng"])))).encode()
    ).hexdigest()[:12]
    pf = CACHE / f"poi_{sig}.pkl"
    if pf.exists():
        poi = pd.read_pickle(pf)
    elif pbf:
        import osm_poi_local
        poi = osm_poi_local.compute_from_pbf(pbf, coords["start_station_id"].tolist(),
                                             coords["start_lat"].values, coords["start_lng"].values)
        poi.to_pickle(pf)
    else:
        rows = osm_poi.compute(coords["start_station_id"].tolist(),
                               coords["start_lat"].values, coords["start_lng"].values)
        poi = pd.DataFrame(rows)
        poi.to_pickle(pf)
    poi = stations.merge(poi, on="start_station_id", how="left")
    poi["distance_to_city_center"] = _haversine_m(
        poi["start_lat"], poi["start_lng"], centroid[0], centroid[1]).round(1)
    return poi[["start_station_id"] + POI8 + ["distance_to_city_center"]]


def enrich(df: pd.DataFrame, weather_mode: str, poi_mode: str, centroid, pbf=None) -> pd.DataFrame:
    df = df.copy()
    if "city" not in df.columns:
        df["city"] = "city ?"
    for c in ("start_lat", "start_lng"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = add_calendar(df)

    if weather_mode == "openmeteo":
        t = pd.to_datetime(df["started_at"], errors="coerce")
        w = fetch_weather(centroid, t.min().strftime("%Y-%m-%d"), t.max().strftime("%Y-%m-%d"))
        df = df.merge(w, on="hour_ts", how="left")
    else:
        for c in WEATHER_COLS:
            df[c] = np.nan

    if poi_mode == "osm":
        # one row per station (median coords) -> compute once -> map back
        st = (df.groupby("start_station_id", dropna=False)[["start_lat", "start_lng"]]
              .median().reset_index())
        feats = compute_poi(st, centroid, pbf=pbf)
        df = df.merge(feats, on="start_station_id", how="left")
    else:
        for c in POI8 + ["distance_to_city_center"]:
            df[c] = np.nan

    for c in STUB_LEAKAGE:
        if c not in df.columns:
            df[c] = np.nan
    return df.reindex(columns=TRAIN_COLS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="start-side CSV (started_at, start_station_id, start_lat, start_lng)")
    ap.add_argument("--name", default=None, help="output stem (default: input stem)")
    ap.add_argument("--weather", choices=["openmeteo", "none"], default="openmeteo")
    ap.add_argument("--poi", choices=["osm", "none"], default="osm")
    ap.add_argument("--poi-pbf", default=None,
                    help="path to a local .osm.pbf extract; faster/more reliable than Overpass "
                         "for whole-city bounding boxes (uses enrichment/osm_poi_local.py)")
    ap.add_argument("--city-centroid", default=None, help="LAT,LNG override for weather + distance")
    ap.add_argument("--sample", type=int, default=0, help="enrich only the first N rows (for verification)")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    if args.sample:
        df = df.head(args.sample).copy()

    if args.city_centroid:
        centroid = tuple(float(x) for x in args.city_centroid.split(","))
    else:
        centroid = (float(pd.to_numeric(df["start_lat"], errors="coerce").median()),
                    float(pd.to_numeric(df["start_lng"], errors="coerce").median()))
    print(f"[enrich] rows={len(df):,}  centroid={centroid[0]:.4f},{centroid[1]:.4f}  "
          f"weather={args.weather}  poi={args.poi}", flush=True)

    out = enrich(df, args.weather, args.poi, centroid, pbf=args.poi_pbf)

    stem = args.name or Path(args.input).stem
    out_path = ROOT / "dataset" / f"{stem}_enriched.csv"
    out.to_csv(out_path, index=False)
    print(f"[enrich] wrote {out_path}  shape={out.shape}")
    print(f"[enrich] calendar NaN: "
          f"{int(out[['date','weekday','weekend','holiday','working_day','hour_ts']].isna().any(axis=1).sum())} rows")


if __name__ == "__main__":
    main()
