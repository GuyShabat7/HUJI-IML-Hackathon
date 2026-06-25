#!/usr/bin/env python3
"""tools/fetch_chicago_divvy.py — fetch Chicago (Divvy) rides as a TEST-ONLY 4th city.

Chicago is a genuinely-unseen city used ONLY to estimate performance on the hidden
grading city. It is labelled ``city 4`` and must NEVER enter train/val (the leakage guard
in submissions/challenge_1_ensamble/data.py and tools/score_holdout.py enforce this).

Source : https://divvy-tripdata.s3.amazonaws.com/<YYYYMM>-divvy-tripdata.zip
Format : Lyft/Motivate schema (same as Capital Bikeshare) — ride_id, started_at,
         start_station_id, start_station_name, start_lat, start_lng, ...
Default: Jan–Feb 2025 (matches the train_set.csv winter window).

Output : dataset/chicago_holdout_start.csv  (start-side fields + city='city 4')

    python tools/fetch_chicago_divvy.py
    python tools/fetch_chicago_divvy.py --months 202501 202502 --out dataset/chicago_holdout_start.csv
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

BASE = "https://divvy-tripdata.s3.amazonaws.com/"
ROOT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = ROOT / "dataset" / "chicago_holdout_start.csv"
RAW_DIR = ROOT / "dataset" / "_cache" / "divvy_raw"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# start-side columns, matching the other cities' start-side files
KEEP = ["city", "started_at", "start_station_id", "start_station_name", "start_lat", "start_lng"]


def _download(month: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{month}-divvy-tripdata.zip"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = f"{BASE}{month}-divvy-tripdata.zip"
    print(f"[divvy] downloading {url}", flush=True)
    rc = subprocess.call(["curl", "-sSfL", "-A", UA, "-o", str(dest), url])
    if rc != 0 or not (dest.exists() and dest.stat().st_size > 1000):
        raise SystemExit(f"download failed for {url} (curl rc={rc}). "
                         f"Check the month exists on the Divvy S3 bucket.")
    return dest


def _read_month(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")][0]
        with z.open(name) as f:
            df = pd.read_csv(f, dtype=str,
                             usecols=["ride_id", "started_at", "start_station_id",
                                      "start_station_name", "start_lat", "start_lng"])
    df = df.rename(columns={"ride_id": "rental_id"})
    df["city"] = "city 4"
    # numeric coords; drop rows with no coords (can't be POI-enriched)
    for c in ("start_lat", "start_lng"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", default=["202501", "202502"],
                    help="YYYYMM Divvy months (default Jan+Feb 2025 to match train_set.csv)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    frames = []
    for m in args.months:
        df = _read_month(_download(m))
        print(f"[divvy] {m}: {len(df):,} rides", flush=True)
        frames.append(df)
    rides = pd.concat(frames, ignore_index=True)

    # keep the start-side schema (order matches the other cities' files)
    rides = rides[[c for c in KEEP if c in rides.columns]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rides.to_csv(args.out, index=False)

    n_coord = int(rides["start_lat"].notna().sum())
    print(f"\n[divvy] wrote {args.out}  rows={len(rides):,}  "
          f"stations={rides['start_station_id'].nunique():,}  with_coords={n_coord:,}")
    print(rides.head(3).to_string())


if __name__ == "__main__":
    main()
