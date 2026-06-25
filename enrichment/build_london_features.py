"""Fill London's blank POI columns using the local OSM extract, calibrated to the
course's scale (factors fit against city 2 / D.C.), and write london_station_features.csv.
"""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import osm_poi_local as L

POI=['bike_lane_length_500m','park_area_500m','university_count_1000m','office_poi_count_1000m',
     'retail_poi_count_1000m','restaurant_cafe_count_500m','transit_stop_count_500m',
     'distance_to_nearest_rail_station']
TRAIN=r"C:\Users\schrs\Downloads\train_set.csv"

# 1) scale factors = ratio of medians (course / local) on DC-proper stations
df=pd.read_csv(TRAIN, usecols=['city','start_station_id','start_lat','start_lng']+POI)
c2=df[df['city']=='city 2'].groupby('start_station_id').first().reset_index().dropna(subset=['start_lat'])
core=c2[(c2.start_lat.between(38.79,39.00))&(c2.start_lng.between(-77.12,-76.91))].reset_index(drop=True)
dc=L.compute_from_pbf("dc.osm.pbf", core.start_station_id.tolist(), core.start_lat.values, core.start_lng.values)
m=core.merge(dc,on='start_station_id',suffixes=('_t','_m'))
scale={}
for c in POI:
    if c=='distance_to_nearest_rail_station':
        scale[c]=1.0; continue   # distance: keep raw metres
    mt=m[c+'_t'][m[c+'_t']>0].median(); mk=m[c+'_m'][m[c+'_m']>0].median()
    scale[c]=float(mt/mk) if mk and mk>0 else 1.0
json.dump(scale, open("poi_scale_factors.json","w"), indent=2)
print("scale factors (course/local):", {k:round(v,2) for k,v in scale.items()})

# 2) compute London POI locally (complete greater-london coverage)
feat=pd.read_csv("london_station_features.csv")
have=feat.dropna(subset=['lat','lng']).copy()
print(f"computing London POI for {len(have)} stations ...", flush=True)
lp=L.compute_from_pbf("greater-london.osm.pbf", have.start_station_id.tolist(),
                      have.lat.values, have.lng.values)

# 3) apply scale
for c in POI:
    if c in lp.columns:
        lp[c]=(lp[c]*scale[c]).round(1) if scale[c]!=1.0 else lp[c]
for c in ['university_count_1000m','office_poi_count_1000m','retail_poi_count_1000m',
          'restaurant_cafe_count_500m','transit_stop_count_500m']:
    lp[c]=lp[c].round().astype('Int64')   # counts back to integers after scaling

# 4) merge: replace blank POI cols, keep lat/lng + distance_to_city_center
base=feat.drop(columns=POI)
out=base.merge(lp[['start_station_id']+POI], on='start_station_id', how='left')
cols=['start_station_id','lat','lng']+POI+['distance_to_city_center']
out=out[[c for c in cols if c in out.columns]]
out.to_csv("london_station_features.csv", index=False)
print("\nUPDATED london_station_features.csv", out.shape)
print("stations with POI:", out['office_poi_count_1000m'].notna().sum(), "/", len(out))

# 5) sanity vs course D.C./L.A. medians
print("\nLondon (mine) vs course medians:")
ca=df[df.city=='city 2'].groupby('start_station_id').first()
cl=df[df.city=='city 3'].groupby('start_station_id').first()
for c in POI:
    lo=out[c][out[c]>0].median() if (out[c]>0).any() else 0
    dcm=ca[c][ca[c]>0].median() if (ca[c]>0).any() else 0
    lam=cl[c][cl[c]>0].median() if (cl[c]>0).any() else 0
    print(f"  {c:35s} London={lo:9.1f}  DC={dcm:9.1f}  LA={lam:9.1f}")
