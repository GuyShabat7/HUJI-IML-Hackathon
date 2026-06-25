"""Fetch full-year London (TfL Santander Cycles) ride data and trim to start-side fields.

This is the SAME upstream source the hackathon used for `city 1` (London). It downloads
the 2025 bi-weekly journey extracts, keeps only the non-removed start-side fields, and
writes:
    dataset/london_2025_full_year_start.csv   (Jan-Dec)
    dataset/london_2025_summer_start.csv      (Jun-Aug subset)

Output columns: rental_id, started_at, start_station_id, start_station_name
(station ids are TfL station numbers; in train_set.csv they appear as the same numbers
as floats).

NOTE: raw ride data only -- no weather/POI/calendar enrichment (the course added that).
NOTE: using data beyond the provided train_set.csv may be against hackathon rules.

Usage:
    python tools/fetch_london_tfl.py
Run from the repo root. Requires `curl` on PATH.
"""
import os, csv, subprocess

BASE = "https://cycling.data.tfl.gov.uk/usage-stats/"
# 2025 bi-weekly extracts, Jan -> Dec (files 411..434)
FILES = [
    "411JourneyDataExtract01Jan2025-14Jan2025.csv", "412JourneyDataExtract15Jan2025-31Jan2025.csv",
    "413JourneyDataExtract01Feb2025-14Feb2025.csv", "414JourneyDataExtract15Feb2025-28Feb2025.csv",
    "415JourneyDataExtract01Mar2025-14Mar2025.csv", "416JourneyDataExtract15Mar2025-31Mar2025.csv",
    "417JourneyDataExtract01Apr2025-14Apr2025.csv", "418JourneyDataExtract15Apr2025-30Apr2025.csv",
    "419JourneyDataExtract01May2025-14May2025.csv", "420JourneyDataExtract14May2025-31May2025.csv",
    "421JourneyDataExtract01Jun2025-15Jun2025.csv", "422JourneyDataExtract15Jun2025-30Jun2025.csv",
    "423JourneyDataExtract01Jul2025-15Jul2025.csv", "424JourneyDataExtract16Jul2025-31Jul2025.csv",
    "425JourneyDataExtract01Aug2025-15Aug2025.csv", "426JourneyDataExtract16Aug2025-31Aug2025.csv",
    "427JourneyDataExtract01Sep2025-15Sep2025.csv", "428JourneyDataExtract16Sep2025-30Sep2025.csv",
    "429JourneyDataExtract01Oct2025-15Oct2025.csv", "430JourneyDataExtract16Oct2025-31Oct2025.csv",
    "431JourneyDataExtract01Nov2025-15Nov2025.csv", "432JourneyDataExtract16Nov2025-30Nov2025.csv",
    "433JourneyDataExtract01Dec2025-15Dec2025.csv", "434JourneyDataExtract16Dec2025-31Dec2025.csv",
]

OUTDIR = "dataset"
RAW = os.path.join(OUTDIR, "_raw")
FULL = os.path.join(OUTDIR, "london_2025_full_year_start.csv")
SUMMER = os.path.join(OUTDIR, "london_2025_summer_start.csv")
OUT_COLS = ["rental_id", "started_at", "start_station_id", "start_station_name"]


def col(header, *names):
    """Resolve a column by any of its known header names (order differs between files)."""
    for n in names:
        if n in header:
            return header.index(n)
    raise SystemExit(f"missing column {names} in header={header}")


def main():
    os.makedirs(RAW, exist_ok=True)
    total = 0
    per_month = {}
    with open(FULL, "w", newline="", encoding="utf-8") as cf, \
         open(SUMMER, "w", newline="", encoding="utf-8") as sf:
        cw, sw = csv.writer(cf), csv.writer(sf)
        cw.writerow(OUT_COLS)
        sw.writerow(OUT_COLS)
        for i, fn in enumerate(FILES, 1):
            dest = os.path.join(RAW, fn)
            if not (os.path.exists(dest) and os.path.getsize(dest) > 1000):
                print(f"[{i}/{len(FILES)}] downloading {fn}", flush=True)
                rc = subprocess.call(["curl", "-sSfL", "-A", "Mozilla/5.0", "-o", dest, BASE + fn])
                if rc != 0 or not (os.path.exists(dest) and os.path.getsize(dest) > 1000):
                    raise SystemExit(f"download failed for {fn} (curl rc={rc})")
            print(f"[{i}/{len(FILES)}] processing {fn}", flush=True)
            with open(dest, encoding="utf-8", errors="replace", newline="") as f:
                r = csv.reader(f)
                header = next(r)
                ri = col(header, "Number", "Rental Id")
                ti = col(header, "Start date", "Start Date")
                si = col(header, "Start station number", "StartStation Id")
                ni = col(header, "Start station", "StartStation Name")
                n = max(ri, ti, si, ni)
                for row in r:
                    if len(row) <= n:
                        continue
                    ts = row[ti]
                    rec = [row[ri], ts, row[si], row[ni]]
                    cw.writerow(rec)
                    if ts[5:7] in ("06", "07", "08"):
                        sw.writerow(rec)
                    total += 1
                    per_month[ts[:7]] = per_month.get(ts[:7], 0) + 1
            os.remove(dest)
    try:
        os.rmdir(RAW)
    except OSError:
        pass
    print("\n===== DONE =====")
    print(f"full year: {FULL}  ({os.path.getsize(FULL) / 1e6:.0f} MB)")
    print(f"summer:    {SUMMER}  ({os.path.getsize(SUMMER) / 1e6:.0f} MB)")
    print(f"total rides: {total:,}")
    for m in sorted(per_month):
        print(f"  {m}: {per_month[m]:,}")


if __name__ == "__main__":
    main()
