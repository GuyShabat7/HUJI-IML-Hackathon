"""Compute per-station POI / infrastructure features from OpenStreetMap (Overpass API).

Reproduces the 8 OSM-derived columns in train_set.csv for any city given station
coordinates:
    bike_lane_length_500m, park_area_500m, university_count_1000m, office_poi_count_1000m,
    retail_poi_count_1000m, restaurant_cafe_count_500m, transit_stop_count_500m,
    distance_to_nearest_rail_station

Approach: one Overpass query per category over the station bounding box, then a local
spatial computation (azimuthal-equidistant projection -> meters; KD-tree for point
counts; shapely for park area and cycleway length within radius).

Used to fill London (city 1), whose POI columns are all blank in train_set.csv, and to
enrich any newly added city. Calibrated against city 2 (D.C.) known values.
"""
import time, requests, numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform as shp_transform

OVERPASS_ENDPOINTS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]
HEADERS = {"User-Agent": "huji-iml-hackathon-bikedemand/1.0 (station POI enrichment; contact: student)"}

def overpass(query, retries=5):
    last = None
    for a in range(retries):
        ep = OVERPASS_ENDPOINTS[a % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(ep, data={"data": query}, headers=HEADERS, timeout=300)
            if r.status_code == 200:
                return r.json()["elements"]
            last = f"{r.status_code} @ {ep}"
        except requests.RequestException as e:
            last = f"{e.__class__.__name__} @ {ep}"
        time.sleep(6 * (a + 1))
    raise RuntimeError(f"Overpass failed after {retries} tries: {last}")

def _bbox(lats, lngs, margin=0.013):
    return (min(lats)-margin, min(lngs)-margin, max(lats)+margin, max(lngs)+margin)

def _q(bbox, body):
    s, w, n, e = bbox
    return f"[out:json][timeout:280];({body.format(b=f'{s},{w},{n},{e}')});out center;"

def fetch_layers(bbox):
    """Return dict of raw OSM elements per category."""
    L = {}
    L["university"] = overpass(_q(bbox, 'node["amenity"="university"]({b});way["amenity"="university"]({b});'))
    L["office"]     = overpass(_q(bbox, 'node["office"]({b});way["office"]({b});'))
    L["retail"]     = overpass(_q(bbox, 'node["shop"]({b});way["shop"]({b});'))
    L["food"]       = overpass(_q(bbox, 'node["amenity"~"^(restaurant|cafe)$"]({b});way["amenity"~"^(restaurant|cafe)$"]({b});'))
    L["transit"]    = overpass(_q(bbox, 'node["highway"="bus_stop"]({b});node["railway"="tram_stop"]({b});node["public_transport"="platform"]({b});'))
    L["rail"]       = overpass(_q(bbox, 'node["railway"="station"]({b});way["railway"="station"]({b});'))
    s, w, n, e = bbox
    bs = f"{s},{w},{n},{e}"
    L["park"]   = overpass(f'[out:json][timeout:280];(way["leisure"~"^(park|garden|nature_reserve|common)$"]({bs});way["landuse"~"^(recreation_ground|grass|meadow)$"]({bs}););out geom;')
    L["cycle"]  = overpass(f'[out:json][timeout:280];(way["highway"="cycleway"]({bs});way["cycleway"~"lane|track|opposite"]({bs});way["cycleway:left"~"lane|track"]({bs});way["cycleway:right"~"lane|track"]({bs});way["cycleway:both"~"lane|track"]({bs});way["bicycle"="designated"]({bs}););out geom;')
    return L

def _pts(elements):
    out = []
    for el in elements:
        if "lat" in el and "lon" in el:
            out.append((el["lat"], el["lon"]))
        elif "center" in el:
            out.append((el["center"]["lat"], el["center"]["lon"]))
    return np.array(out) if out else np.empty((0, 2))

def compute(station_ids, lats, lngs, layers=None, lat0=None, lng0=None):
    """Return list of dicts with the 8 POI columns for each station."""
    lats = np.asarray(lats, float); lngs = np.asarray(lngs, float)
    if lat0 is None: lat0, lng0 = float(np.nanmean(lats)), float(np.nanmean(lngs))
    if layers is None: layers = fetch_layers(_bbox(lats[~np.isnan(lats)], lngs[~np.isnan(lngs)]))
    # local metric projection (meters) centred on the city
    fwd = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lng0} +units=m", always_xy=True)
    def to_m(lon, lat): return fwd.transform(lon, lat)
    sx, sy = to_m(lngs, lats)

    def proj_pts(arr):
        if len(arr) == 0: return np.empty((0, 2))
        x, y = to_m(arr[:, 1], arr[:, 0]); return np.column_stack([x, y])
    trees = {}
    for k in ["university", "office", "retail", "food", "transit", "rail"]:
        p = proj_pts(_pts(layers[k]))
        trees[k] = cKDTree(p) if len(p) else None

    # park polygons & cycleway lines in projected meters
    parks, cycles = [], []
    for el in layers["park"]:
        g = el.get("geometry")
        if g and len(g) >= 3:
            try: parks.append(Polygon([to_m(p["lon"], p["lat"]) for p in g]).buffer(0))
            except Exception: pass
    for el in layers["cycle"]:
        g = el.get("geometry")
        if g and len(g) >= 2:
            try: cycles.append(LineString([to_m(p["lon"], p["lat"]) for p in g]))
            except Exception: pass

    def count(tree, x, y, r):
        return 0 if tree is None else len(tree.query_ball_point([x, y], r))
    def nearest(tree, x, y):
        if tree is None or tree.n == 0: return -1.0
        d, _ = tree.query([x, y]); return round(float(d), 1)

    rows = []
    for sid, x, y in zip(station_ids, sx, sy):
        if not np.isfinite(x):
            rows.append({"start_station_id": sid}); continue
        pt500 = Point(x, y).buffer(500)
        parea = sum(p.intersection(pt500).area for p in parks if p.distance(Point(x, y)) < 500)
        clen = sum(c.intersection(pt500).length for c in cycles if c.distance(Point(x, y)) < 500)
        rows.append({
            "start_station_id": sid,
            "bike_lane_length_500m": round(clen, 1),
            "park_area_500m": round(parea, 1),
            "university_count_1000m": count(trees["university"], x, y, 1000),
            "office_poi_count_1000m": count(trees["office"], x, y, 1000),
            "retail_poi_count_1000m": count(trees["retail"], x, y, 1000),
            "restaurant_cafe_count_500m": count(trees["food"], x, y, 500),
            "transit_stop_count_500m": count(trees["transit"], x, y, 500),
            "distance_to_nearest_rail_station": nearest(trees["rail"], x, y),
        })
    return rows
