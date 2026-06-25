"""Compute per-station POI / infrastructure features from a LOCAL OpenStreetMap
extract (.osm.pbf, e.g. Geofabrik) using pyrosm — no API, no rate limits.

Same 8 columns as train_set.csv. Used because the Overpass API is too slow/flaky for
city-scale extraction. Reproduces the spatial definitions in osm_poi.py.
"""
import numpy as np, pandas as pd, pyrosm
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely.ops import transform as shp_transform

CATS = {
    "university": {"amenity": ["university"]},
    "office":     {"office": True},
    "retail":     {"shop": True},
    "food":       {"amenity": ["restaurant", "cafe"]},
    "transit":    {"highway": ["bus_stop"], "railway": ["tram_stop"], "public_transport": ["platform"]},
    "rail":       {"railway": ["station"]},
    "park":       {"leisure": ["park", "garden", "nature_reserve", "common"],
                   "landuse": ["recreation_ground", "grass", "meadow"]},
    "cycle":      {"highway": ["cycleway"], "cycleway": True, "bicycle": ["designated"]},
}

def _extract(osm, cf):
    try:
        g = osm.get_data_by_custom_criteria(custom_filter=cf, filter_type="keep",
                                            keep_nodes=True, keep_ways=True, keep_relations=True)
    except Exception:
        g = None
    return g if g is not None and len(g) else None

def compute_from_pbf(pbf, station_ids, lats, lngs, lat0=None, lng0=None, bbox=None):
    """``bbox`` = [west, south, east, north] limits parsing to a region (e.g. one city) —
    essential when the extract covers a whole state/country, else pyrosm loads everything."""
    lats = np.asarray(lats, float); lngs = np.asarray(lngs, float)
    if lat0 is None:
        lat0, lng0 = float(np.nanmean(lats)), float(np.nanmean(lngs))
    proj = f"+proj=aeqd +lat_0={lat0} +lon_0={lng0} +datum=WGS84 +units=m"
    fwd = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    to_m = lambda geom: shp_transform(lambda x, y, z=None: fwd.transform(x, y), geom)
    sx, sy = fwd.transform(lngs, lats)

    osm = pyrosm.OSM(pbf, bounding_box=bbox) if bbox else pyrosm.OSM(pbf)
    layers = {k: _extract(osm, cf) for k, cf in CATS.items()}

    def centroids_tree(gdf):
        if gdf is None: return None
        pts = gdf.geometry.centroid
        xy = np.array([fwd.transform(p.x, p.y) for p in pts])
        return cKDTree(xy) if len(xy) else None
    trees = {k: centroids_tree(layers[k]) for k in ["university","office","retail","food","transit","rail"]}

    def geoms_index(gdf):
        if gdf is None: return None, None
        gs = [to_m(g) for g in gdf.geometry if g is not None and not g.is_empty]
        return (STRtree(gs), gs) if gs else (None, None)
    park_tree, parks = geoms_index(layers["park"])
    cyc_tree,  cycles = geoms_index(layers["cycle"])

    def cnt(t, x, y, r): return 0 if t is None else len(t.query_ball_point([x, y], r))
    def near(t, x, y):
        if t is None or t.n == 0: return -1.0
        d, _ = t.query([x, y]); return round(float(d), 1)

    rows = []
    for sid, x, y in zip(station_ids, sx, sy):
        if not np.isfinite(x):
            rows.append({"start_station_id": sid}); continue
        buf = Point(x, y).buffer(500)
        parea = 0.0
        if park_tree is not None:
            for j in park_tree.query(buf):
                g = parks[j]
                if g.intersects(buf): parea += g.intersection(buf).area
        clen = 0.0
        if cyc_tree is not None:
            for j in cyc_tree.query(buf):
                g = cycles[j]
                if g.intersects(buf): clen += g.intersection(buf).length
        rows.append({
            "start_station_id": sid,
            "bike_lane_length_500m": round(clen, 1),
            "park_area_500m": round(parea, 1),
            "university_count_1000m": cnt(trees["university"], x, y, 1000),
            "office_poi_count_1000m": cnt(trees["office"], x, y, 1000),
            "retail_poi_count_1000m": cnt(trees["retail"], x, y, 1000),
            "restaurant_cafe_count_500m": cnt(trees["food"], x, y, 500),
            "transit_stop_count_500m": cnt(trees["transit"], x, y, 500),
            "distance_to_nearest_rail_station": near(trees["rail"], x, y),
        })
    return pd.DataFrame(rows)
