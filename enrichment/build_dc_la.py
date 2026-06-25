"""Fetch full-year 2025 ride data for D.C. (Capital Bikeshare) and L.A. (Metro Bike Share),
trim to start-side fields (matching the London files), concat per city + summer subset.

Output columns: rental_id, started_at, start_station_id, start_station_name
(L.A. has no station names -> left blank; timestamps normalised to 'YYYY-MM-DD HH:MM:SS')
"""
import os, csv, io, zipfile, subprocess, datetime as dt
import pandas as pd

OUT = "dataset"; os.makedirs(OUT, exist_ok=True)
RAW = os.path.join(OUT, "_raw"); os.makedirs(RAW, exist_ok=True)
COLS = ["rental_id", "started_at", "start_station_id", "start_station_name"]

DC_BASE = "https://s3.amazonaws.com/capitalbikeshare-data/"
DC_FILES = [f"2025{m:02d}-capitalbikeshare-tripdata.zip" for m in range(1, 13)]
LA = {  # quarter -> url
    "q1": "https://bikeshare.metro.net/wp-content/uploads/2025/04/metro-trips-2025-q1.zip",
    "q2": "https://bikeshare.metro.net/wp-content/uploads/2025/07/metro-trips-2025-q2.zip",
    "q3": "https://bikeshare.metro.net/wp-content/uploads/2025/10/metro-trips-2025-q3.zip",
    "q4": "https://bikeshare.metro.net/wp-content/uploads/2026/01/metro-trips-2025-q4.zip",
}

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    rc = subprocess.call(["curl", "-sSfL", "-A", ua, "-o", dest, url])
    if rc != 0 or not (os.path.exists(dest) and os.path.getsize(dest) > 1000):
        raise SystemExit(f"download failed: {url} (rc={rc})")

def main():
    summary = {}

    # ---------- D.C. ----------
    full = open(os.path.join(OUT, "dc_2025_full_year_start.csv"), "w", newline="", encoding="utf-8")
    summ = open(os.path.join(OUT, "dc_2025_summer_start.csv"), "w", newline="", encoding="utf-8")
    fw, sw = csv.writer(full), csv.writer(summ); fw.writerow(COLS); sw.writerow(COLS)
    n = 0; months = {}; st = set()
    for fn in DC_FILES:
        dest = os.path.join(RAW, fn); dl(DC_BASE + fn, dest)
        print(f"[DC] {fn}", flush=True)
        with zipfile.ZipFile(dest) as z:
            name = [x for x in z.namelist() if x.endswith(".csv") and not x.startswith("__MACOSX")][0]
            with z.open(name) as raw:
                r = csv.reader(io.TextIOWrapper(raw, "utf-8", errors="replace"))
                h = next(r)
                ri, ti = h.index("ride_id"), h.index("started_at")
                si, ni = h.index("start_station_id"), h.index("start_station_name")
                mx = max(ri, ti, si, ni)
                for row in r:
                    if len(row) <= mx: continue
                    ts = row[ti][:19]
                    rec = [row[ri], ts, row[si], row[ni]]
                    fw.writerow(rec); n += 1
                    mo = ts[5:7]; months[mo] = months.get(mo, 0) + 1; st.add(row[si])
                    if mo in ("06", "07", "08"): sw.writerow(rec)
        os.remove(dest)
    full.close(); summ.close()
    summary["DC"] = (n, len(st), months)

    # ---------- L.A. ----------
    full = open(os.path.join(OUT, "la_2025_full_year_start.csv"), "w", newline="", encoding="utf-8")
    summ = open(os.path.join(OUT, "la_2025_summer_start.csv"), "w", newline="", encoding="utf-8")
    fw, sw = csv.writer(full), csv.writer(summ); fw.writerow(COLS); sw.writerow(COLS)
    n = 0; months = {}; st = set()
    for q, url in LA.items():
        dest = os.path.join(RAW, f"la-2025-{q}.zip"); dl(url, dest)
        print(f"[LA] {q}", flush=True)
        with zipfile.ZipFile(dest) as z:
            name = [x for x in z.namelist() if x.endswith(".csv") and not x.startswith("__MACOSX")][0]
            df = pd.read_csv(z.open(name), usecols=["trip_id", "start_time", "start_station"],
                             dtype=str)
        ts = pd.to_datetime(df["start_time"], errors="coerce")
        df = df[ts.notna()]; ts = ts[ts.notna()]
        df["started_at"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        for rid, t, sid in zip(df["trip_id"], df["started_at"], df["start_station"]):
            rec = [rid, t, sid, ""]
            fw.writerow(rec); n += 1
            mo = t[5:7]; months[mo] = months.get(mo, 0) + 1; st.add(sid)
            if mo in ("06", "07", "08"): sw.writerow(rec)
        os.remove(dest)
    full.close(); summ.close()
    summary["LA"] = (n, len(st), months)

    try: os.rmdir(RAW)
    except OSError: pass

    print("\n===== DONE =====")
    for city, (n, nst, months) in summary.items():
        print(f"\n{city}: {n:,} rides, {nst:,} start stations")
        for m in sorted(months): print(f"   2025-{m}: {months[m]:,}")

if __name__ == "__main__":
    main()
