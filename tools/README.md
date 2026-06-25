# tools/ — evaluation, enrichment & unseen-city scoring

Helper scripts around the submissions. None of them commit data
(`dataset/*` is gitignored); they read `dataset/train_set.csv` (copied in locally).

| Script | What it does |
|---|---|
| `eval_local.py` | Train any submission on the held-out split (c1+c2; city 3 held out) and print per-city + **blended** MAE. `python tools/eval_local.py submissions/<folder>` |
| `bakeoff.py` | Feature bake-off — A baseline / B no_identity / C fallback_te, same recipe, side-by-side blended MAE. `python tools/bakeoff.py` |
| `fetch_chicago_divvy.py` | Download Chicago/Divvy rides as the TEST-ONLY 4th city (`city 4`). → `dataset/chicago_holdout_start.csv` |
| `enrich_city.py` | Add the exact `train_set.csv` feature columns (Open-Meteo weather + OSM POI + US-holiday calendar) to any start-side ride file. → `dataset/<name>_enriched.csv` |
| `score_holdout.py` | Score the model on the unseen 4th city; prints city1/2/3 + **city4** MAE with a hard leakage guard. |
| `fetch_london_tfl.py`, `build_supplementary_london.py` | Full-year London supplementary data (rules-gated; see repo README). |

## Unseen-city workflow (Chicago = city 4)
```
python tools/fetch_chicago_divvy.py                                  # raw start-side rides
python tools/enrich_city.py dataset/chicago_holdout_start.csv --name chicago \
       --poi-pbf dataset/_cache/illinois.osm.pbf                     # add features (fast local OSM)
python tools/score_holdout.py --submission submissions/challenge_1_ensamble \
       --enriched dataset/chicago_enriched.csv                       # city4 MAE, leakage-guarded
```
`city 4` (and `city 3`) can NEVER enter train/val — `data.load_splits` + `score_holdout`
assert this and fail loudly otherwise.

## Enrichment notes
- **Weather:** Open-Meteo Archive API (free, no key), per city-hour at the city centroid;
  cached under `dataset/_cache/`.
- **POI:** OpenStreetMap. Default `--poi osm` uses the **Overpass API**, but whole-city
  bounding boxes make Overpass slow/flaky — for big cities pass **`--poi-pbf <file>.osm.pbf`**
  (a Geofabrik extract) to compute locally via `enrichment/osm_poi_local.py` (fast, reliable).
- `--weather none` / `--poi none` degrade to calendar-only (those columns become NaN).
- Column names/units are verified to match `train_set.csv` exactly.

## Running on Phoenix (HUJI SLURM)
`cluster/run_enrich.slurm` and `cluster/run_score.slurm` wrap these (CPU-only; threads bound
to `$SLURM_CPUS_PER_TASK`; logs in `cluster/logs/`). **Enrichment needs outbound internet**
(Open-Meteo / Overpass) — Phoenix compute nodes are often firewalled, so run enrichment on a
login/interactive node, or pre-download the `.osm.pbf` and use `--poi-pbf` so only the cached
weather call touches the network. Confirm `--account` / `--partition` placeholders for your
account (defaults: `nirf` / `short`; `killable-cs` / `killable` for long jobs).
