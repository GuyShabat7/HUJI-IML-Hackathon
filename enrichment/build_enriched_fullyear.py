"""Produce model-ready ENRICHED full-year ride data for each city, matching the
columns train.py/model.py expect (so it is a drop-in for dataset/train_set.csv).

For each city it:
  1. downloads the full-year start-side ride file from its GitHub release,
  2. joins per-station features (lat/lng + 9 POI) and per-hour weather,
  3. computes the calendar fields (US-holiday encoding, like the course),
  4. writes <city>_2025_enriched.csv.gz   (chunked, low memory).

Output columns (subset of train_set.csv that the model uses):
  started_at, hour_ts, city, start_station_id, start_station_name,
  start_lat, start_lng, <8 weather>, <9 POI/infra>,
  date, weekday, weekend, holiday, holiday_name, working_day
"""
import os, gzip, subprocess, numpy as np, pandas as pd
import holidays as _holidays

REPO = "GuyShabat7/HUJI-IML-Hackathon"
WX = ['temperature_2m','relative_humidity_2m','apparent_temperature','precipitation',
      'rain','snowfall','cloud_cover','wind_speed_10m']
POI = ['bike_lane_length_500m','park_area_500m','university_count_1000m','office_poi_count_1000m',
       'retail_poi_count_1000m','restaurant_cafe_count_500m','transit_stop_count_500m',
       'distance_to_nearest_rail_station','distance_to_city_center']
OUT_COLS = (['started_at','hour_ts','city','start_station_id','start_station_name',
             'start_lat','start_lng'] + WX + POI +
            ['date','weekday','weekend','holiday','holiday_name','working_day'])

CITIES = {
    "london": dict(city="city 1", tag="london-2025-data",
                   ride="london_2025_full_year_start.csv.gz",
                   feat="london_station_features.csv", weather="london_weather_2025.csv"),
    "dc":     dict(city="city 2", tag="dc-la-2025-data",
                   ride="dc_2025_full_year_start.csv.gz",
                   feat="dc_station_features.csv", weather="dc_weather_2025.csv"),
    "la":     dict(city="city 3", tag="dc-la-2025-data",
                   ride="la_2025_full_year_start.csv.gz",
                   feat="la_station_features.csv", weather="la_weather_2025.csv"),
}
US = _holidays.US(years=[2025])
TMP = "_tmp"; os.makedirs(TMP, exist_ok=True)

def add_calendar(df):
    t = pd.to_datetime(df['started_at'], errors='coerce')
    df['hour_ts'] = t.dt.floor('h')
    df['date'] = t.dt.strftime('%Y-%m-%d')
    df['weekday'] = t.dt.weekday
    df['weekend'] = (df['weekday'] >= 5).astype('Int64')
    hol = t.dt.normalize().dt.date.map(lambda x: US.get(x) if pd.notna(x) else None)
    df['holiday'] = hol.notna().astype('Int64')
    df['holiday_name'] = hol
    df['working_day'] = (((df['weekend'] == 0) & (df['holiday'] == 0)).astype('Int64'))
    return df

def run_city(key):
    c = CITIES[key]
    gz = os.path.join(TMP, c['ride'])
    if not os.path.exists(gz):
        print(f"[{key}] downloading {c['ride']} ...", flush=True)
        subprocess.check_call(["gh","release","download",c['tag'],"--repo",REPO,
                               "--pattern",c['ride'],"--dir",TMP,"--clobber"])
    feat = pd.read_csv(c['feat']).rename(columns={'lat':'start_lat','lng':'start_lng'})
    feat['start_station_id'] = pd.to_numeric(feat['start_station_id'], errors='coerce')
    w = pd.read_csv(c['weather']); w['hour_ts'] = pd.to_datetime(w['hour_ts']).dt.floor('h')

    out = os.path.join("..","dataset",f"{key}_2025_enriched.csv.gz")
    os.makedirs(os.path.join("..","dataset"), exist_ok=True)
    n=0; first=True
    with gzip.open(out, "wt", encoding="utf-8", newline="") as fo:
        for chunk in pd.read_csv(gz, compression="gzip", chunksize=1_000_000, dtype=str):
            chunk['start_station_id'] = pd.to_numeric(chunk['start_station_id'], errors='coerce')
            chunk['city'] = c['city']
            chunk = add_calendar(chunk)
            chunk = chunk.merge(feat[['start_station_id','start_lat','start_lng']+POI],
                                on='start_station_id', how='left')
            chunk = chunk.merge(w[['hour_ts']+WX], on='hour_ts', how='left')
            chunk['hour_ts'] = chunk['hour_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')
            for col in OUT_COLS:
                if col not in chunk.columns: chunk[col] = np.nan
            chunk[OUT_COLS].to_csv(fo, index=False, header=first)
            first=False; n+=len(chunk)
            print(f"[{key}] {n:,} rows", flush=True)
    print(f"[{key}] DONE -> {out}  ({os.path.getsize(out)/1e6:.0f} MB)  rows={n:,}", flush=True)
    return out

if __name__ == "__main__":
    import sys
    keys = sys.argv[1:] or ["london","dc","la"]
    for k in keys:
        run_city(k)
