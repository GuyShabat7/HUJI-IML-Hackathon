"""Compute REAL OSM POI features for every London station and write them into
enrichment/london_station_features.csv (the course left these blank for city 1).

Stations are processed in small spatial tiles so each Overpass query stays fast.
Keeps the course's real distance_to_city_center and lat/lng; replaces the 8 blank
OSM columns with computed values. Validated against city 2 (D.C.) — see README.
"""
import pandas as pd, numpy as np
import osm_poi

POI_COLS = ['bike_lane_length_500m','park_area_500m','university_count_1000m','office_poi_count_1000m',
            'retail_poi_count_1000m','restaurant_cafe_count_500m','transit_stop_count_500m',
            'distance_to_nearest_rail_station']
PATH = "london_station_features.csv"
TILE = 0.035  # ~3.5 km tiles -> small bbox per Overpass query

def main():
    f = pd.read_csv(PATH)
    have = f.dropna(subset=['lat','lng']).copy()
    print(f"{len(have)}/{len(f)} stations have coords; tiling at {TILE} deg", flush=True)
    have['gx'] = (have['lat']/TILE).round().astype(int)
    have['gy'] = (have['lng']/TILE).round().astype(int)
    tiles = list(have.groupby(['gx','gy']))
    print(f"{len(tiles)} tiles", flush=True)

    out = {}
    for i,(key,grp) in enumerate(tiles,1):
        bbox = osm_poi._bbox(grp['lat'].values, grp['lng'].values)
        print(f"[{i}/{len(tiles)}] tile {key}: {len(grp)} stations, bbox span "
              f"{bbox[2]-bbox[0]:.3f}x{bbox[3]-bbox[1]:.3f}", flush=True)
        layers = osm_poi.fetch_layers(bbox)
        rows = osm_poi.compute(grp['start_station_id'].tolist(), grp['lat'].values, grp['lng'].values,
                               layers=layers)
        for r in rows:
            out[r['start_station_id']] = r
    comp = pd.DataFrame(out.values())

    # merge: overwrite the blank POI columns, keep lat/lng + distance_to_city_center
    base = f.drop(columns=POI_COLS)
    merged = base.merge(comp[['start_station_id']+POI_COLS], on='start_station_id', how='left')
    # reorder to original column layout
    cols = ['start_station_id','lat','lng'] + POI_COLS + ['distance_to_city_center']
    merged = merged[[c for c in cols if c in merged.columns]]
    merged.to_csv(PATH, index=False)
    print("\nUPDATED", PATH, merged.shape)
    filled = merged['office_poi_count_1000m'].notna().sum()
    print(f"stations with computed POI: {filled}/{len(merged)}")
    print(merged[POI_COLS].describe().loc[['min','50%','max']].to_string())

if __name__ == "__main__":
    main()
