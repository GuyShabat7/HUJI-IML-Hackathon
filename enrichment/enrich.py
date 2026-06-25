"""Enrich a London ride/demand table to (approximately) the train_set.csv schema.

Joins:
  * per-station features  (lat/lng + 8 OSM POI columns + distance_to_city_center)
                          from enrichment/london_station_features.csv   (by start_station_id)
  * per-hour weather      (8 Open-Meteo columns) from enrichment/london_weather_2025.csv (by hour_ts)
  * calendar fields       (weekday/weekend/holiday/holiday_name/working_day/date/hour_ts)
                          computed to match the course encoding (US federal holidays; Mon=0).

Input must have a `start_station_id` column and a timestamp column (`started_at` or `hour_ts`).
This reproduces every train_set.csv column EXCEPT the ride-level leakage fields that vanish
after aggregation (ended_at, end_station_id, usage_time_minutes, distance_meters, user_type).

Usage:
    python enrich.py <input.csv> <output.csv> [--city "city 1"]
"""
import sys, argparse, pandas as pd, numpy as np
import holidays as _holidays

WX = ['temperature_2m','relative_humidity_2m','apparent_temperature','precipitation',
      'rain','snowfall','cloud_cover','wind_speed_10m']
POI = ['bike_lane_length_500m','park_area_500m','university_count_1000m','office_poi_count_1000m',
       'retail_poi_count_1000m','restaurant_cafe_count_500m','transit_stop_count_500m',
       'distance_to_nearest_rail_station','distance_to_city_center']
HERE = __file__.rsplit('/',1)[0].rsplit('\\',1)[0] if ('/' in __file__ or '\\' in __file__) else '.'

def add_calendar(df, ts):
    t = pd.to_datetime(df[ts])
    yrs = range(int(t.dt.year.min()), int(t.dt.year.max())+1)
    us = _holidays.US(years=list(yrs))
    d = t.dt.normalize()
    df['date'] = t.dt.strftime('%Y-%m-%d')
    df['weekday'] = t.dt.weekday                      # Mon=0 .. Sun=6
    df['weekend'] = (df['weekday'] >= 5).astype(int)
    hol = d.dt.date.map(lambda x: us.get(x))
    df['holiday'] = hol.notna().astype(int)
    df['holiday_name'] = hol
    df['working_day'] = ((df['weekend'] == 0) & (df['holiday'] == 0)).astype(int)
    df['hour_ts'] = t.dt.floor('h').dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--city', default='city 1')
    ap.add_argument('--feat', default=HERE+'/london_station_features.csv')
    ap.add_argument('--weather', default=HERE+'/london_weather_2025.csv')
    a = ap.parse_args()

    df = pd.read_csv(a.inp)
    ts = 'started_at' if 'started_at' in df.columns else 'hour_ts'
    df['start_station_id'] = pd.to_numeric(df['start_station_id'], errors='coerce').astype('Int64')

    feat = pd.read_csv(a.feat).rename(columns={'lat':'start_lat','lng':'start_lng'})
    feat['start_station_id'] = feat['start_station_id'].astype('Int64')
    df = df.merge(feat, on='start_station_id', how='left')

    df = add_calendar(df, ts)
    w = pd.read_csv(a.weather)
    w['hour_ts'] = pd.to_datetime(w['hour_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df.merge(w, on='hour_ts', how='left')

    df['city'] = a.city
    df.to_csv(a.out, index=False)
    miss_f = df['office_poi_count_1000m'].isna().mean()*100
    miss_w = df['temperature_2m'].isna().mean()*100
    print(f"wrote {a.out}  rows={len(df):,}  cols={df.shape[1]}")
    print(f"  station-feature coverage: {100-miss_f:.1f}%   weather coverage: {100-miss_w:.1f}%")

if __name__ == '__main__':
    main()
