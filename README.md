<h1 align="center">Bike Demand Forecasting</h1>

<p align="center">
  <em>HUJI — Introduction to Machine Learning (67577)</em><br>
  <em>Hackathon 2026 · Challenge 1</em>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Model" src="https://img.shields.io/badge/Model-LightGBM-success">
  <img alt="Metric" src="https://img.shields.io/badge/Metric-MAE-orange">
</p>

---

## Overview

Predict **hourly bike-share demand per station**. Acting as the data-science team for
a fictional rental company operating in several cities, we forecast — for a given
`(city, station, hour)` — how many rides will *start* from that station during that
hour.

The challenge tests both predictive accuracy on the two main cities and
**generalization** to a third city with very little history (and an unseen fourth
city used only for grading).

> **Key idea:** the prediction target does not exist in the raw data. The dataset is
> *ride-level* (one row per trip); the target is *station-hour demand*, which we
> construct by counting rides into station-hour buckets — including the many hours
> with zero rides.

---

## Table of Contents

1. [Data Provenance](#data-provenance)
2. [Supplementary Data — Full-Year London](#supplementary-data--full-year-london)
3. [Dataset at a Glance](#dataset-at-a-glance)
4. [Column Schema](#column-schema)
5. [Inference Contract](#inference-contract)
6. [Working with the Data](#working-with-the-data)
7. [Project Structure](#project-structure)
8. [Reproducing the Model](#reproducing-the-model)
9. [Evaluation](#evaluation)
10. [Notes & Caveats](#notes--caveats)

---

## Data Provenance

`train_set.csv` was assembled from three public bike-share open-data portals, then
anonymized (cities renamed `city 1/2/3`) and enriched with weather and neighborhood
features. **All trips are from early 2025.**

| Anonymized | Real system | Period in the data | Source file(s) |
|:----------:|-------------|--------------------|----------------|
| `city 1` | **London** — TfL Santander Cycles | 2025-01-01 → 2025-02-28 *(full Jan + Feb)* | `usage-stats/411…01Jan2025-14Jan2025.csv` → `414…15Feb2025-28Feb2025.csv` (4 bi-weekly extracts) |
| `city 2` | **Washington, D.C.** — Capital Bikeshare | 2025-01-01 → 2025-02-28 *(full Jan + Feb)* | `202501-capitalbikeshare-tripdata.zip`, `202502-capitalbikeshare-tripdata.zip` |
| `city 3` | **Los Angeles** — Metro Bike Share | 2025-01-08 → 2025-01-12 *(5-day slice)* | `metro-trips-2025-q1.zip` (Q1 = Jan–Mar 2025), sub-sampled |

**Enrichment**

| Feature group | Source | How it was joined |
|---------------|--------|-------------------|
| Weather (`*_2m`, `*_10m`, precipitation, …) | [Open-Meteo](https://open-meteo.com/) | per city-hour |
| POI / infrastructure (`*_500m`, `*_1000m`, distances) | [OpenStreetMap](https://www.openstreetmap.org/) | per station |

**Source portals** — London: <https://cycling.data.tfl.gov.uk/> · D.C.:
<https://capitalbikeshare.com/system-data> · L.A.:
<https://bikeshare.metro.net/about/data/>

> The held-out grading city ("city 4") is almost certainly another comparable public
> system processed through the same pipeline — likely an early-2025 slice, so training
> seasonality should roughly match the grading distribution.

---

## Supplementary Data — Full-Year London

The provided `train_set.csv` only covers **Jan–Feb 2025** for London (`city 1`). For
experiments that need warm-season / full-year coverage, we pulled the **entire 2025
calendar year** for London directly from the original TfL portal and host it as a
GitHub **Release** (the files are too large for the git repo itself):

**➡ [Release: `london-2025-data`](https://github.com/GuyShabat7/HUJI-IML-Hackathon/releases/tag/london-2025-data)**

| Asset | Coverage | Rides | Size (gz) |
|-------|----------|-------|-----------|
| `london_2025_full_year_start.csv.gz` | Jan–Dec 2025 | 9,068,241 | 100 MB |
| `london_2025_summer_start.csv.gz` | Jun–Aug 2025 | 2,562,071 | 28 MB |

**Verified same source.** Reconstructing the Jan–Feb slice from these files reproduces
`train_set.csv`'s `city 1` **exactly**: 807/807 stations identical, 1,142,318 vs
1,142,317 rides (a 1-row boundary difference). London `start_station_id` values in
`train_set.csv` are simply the TfL station numbers in numeric form (e.g. TfL `001037`
→ `1037.0`).

**Format — start-side fields only** (the fields that survive to inference; the
"removed" leakage fields — end station, duration, bike, user type — are dropped):

```
rental_id,started_at,start_station_id,start_station_name
146970868,2025-03-14 23:59,001035,"Boston Place, Marylebone"
```

> **Not enriched.** This is the raw upstream ride data — it has **no** weather, POI,
> lat/lng, or calendar columns (those were added by the course pipeline). It is enough
> to **reconstruct the station-hour demand target**; to use it as model input you must
> re-join weather/POI yourself.

**Reproduce / extend** (other months, full columns, or D.C./L.A.):

```bash
python tools/fetch_london_tfl.py     # downloads, trims to start-side, writes dataset/*.csv
```

> ⚠️ **Hackathon rules:** training on data beyond the provided `train_set.csv` may be
> disallowed — confirm with course staff before using this in a submission.

---

## Dataset at a Glance

| Property | Value |
|----------|-------|
| Total ride records | ~1,605,208 |
| `city 1` (London) | ~1.14M rides · 807 stations |
| `city 2` (D.C.) | ~461K rides · 783 stations |
| `city 3` (L.A.) | ~2,249 rides · 211 stations *(intentionally tiny)* |
| Time span | 2025-01-01 → 2025-02-28 |
| Typical demand (non-zero hours) | median ≈ 2, 90th pct ≈ 5, max ≈ 138 |

**Implications**

- Demand is **small counts with a long tail** → favors a count/median-style loss
  (Poisson or L1) and makes MAE the natural metric.
- **Missingness is real and uneven:** `distance_meters` ~100% empty, `user_type`
  ~71% empty, coordinates 100% empty for `city 3`. Cities differ in available fields —
  never assume a column is populated.

---

## Column Schema

**Raw `train_set.csv`**

| Group | Columns |
|-------|---------|
| Ride timing & identity | `started_at`, `ended_at`, `start_station_id`, `end_station_id`, `usage_time_minutes`, `distance_meters`, `user_type` |
| Station location | `start_lat`, `start_lng` |
| Weather (Open-Meteo, per city-hour) | `temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `precipitation`, `rain`, `snowfall`, `cloud_cover`, `wind_speed_10m` |
| Station metadata (OSM, per station) | `bike_lane_length_500m`, `park_area_500m`, `university_count_1000m`, `office_poi_count_1000m`, `retail_poi_count_1000m`, `restaurant_cafe_count_500m`, `transit_stop_count_500m`, `distance_to_nearest_rail_station`, `distance_to_city_center` |
| Calendar & keys | `city`, `date`, `weekday`, `weekend`, `holiday`, `holiday_name`, `working_day`, `hour_ts` |

---

## Inference Contract

At evaluation, the model receives **station-hour target rows** (not ride data) and
returns one demand prediction per row.

**Guaranteed present** (they define a prediction):
`city`, `start_station_id`, and the hour timestamp (`hour_ts` / `target_hour_start`,
plus derivable `date` / `weekday` / `hour`). Weather and station-metadata columns are
typically present but **may be empty** for some cities.

**Never present** (ride-level fields that vanish after aggregation — do **not** use as
features): `demand`, `started_at`, `ended_at`, `end_station_id`, `usage_time_minutes`,
`distance_meters`, `user_type`.

> `holiday_name` lists **US** federal holidays for *every* city, including London — so
> it does **not** encode true local (UK) holidays. Treat it as a generic
> low-demand-day flag, not a country signal.

---

## Working with the Data

1. **Build a local validation harness first.** Split `train_set.csv` by *time* within
   each city (hold out the latest ~20% — never a random split, which leaks the
   future). Run `build_station_hour_eval_data.py` to convert the validation slice into
   the evaluator's format (public targets + private labels).
2. **Aggregate rides into station-hours and reconstruct zeros.** Count rides per
   `(city, station, hour)`, then add back empty daytime hours (06:00–22:00) inside
   each station's active window as `demand = 0`. *This is the single most important
   step — the test set is full of genuine zeros.*
3. **Engineer shared features** (identical code for train and predict): cyclical
   hour/weekday (sin/cos), weather, calendar flags, station metadata, and
   **hierarchical demand averages** with graceful fallback
   (`station×hour-of-week → station → city×hour-of-week → city → global`). The
   fallback chain is what lets the model survive unseen stations/cities.
4. **Train gradient-boosted trees** (LightGBM). Because the target is a count and the
   metric is MAE, train both `objective="poisson"` and `objective="regression_l1"` and
   average them.
5. **Score locally with `evaluate.py`**, watching per-city MAE separately — a change
   that helps cities 1–2 can hurt city 3. Iterate.

---

## Project Structure

```
HUJI-IML-Hackathon/
├── dataset/
│   └── train_set.csv                 # place the provided data here (not committed)
├── submissions/
│   └── challenge_1_IDs/
│       ├── train.py                  # trains from ../../dataset/train_set.csv → weights.joblib
│       ├── model.py                  # BikeDemandModel: feature engineering + prediction
│       ├── predict.py                # evaluator wrapper — DO NOT modify
│       ├── weights.joblib            # all fitted artifacts
│       └── README                    # team names / IDs / model description
├── tools/
│   └── fetch_london_tfl.py           # fetch full-year London data (see Supplementary Data)
├── evaluate.py                       # local evaluator
├── base_model.py                     # base interface the model inherits
├── build_station_hour_eval_data.py   # ride-level → station-hour eval format
├── check_submission_format.py        # validates a submission folder
└── README.md
```

---

## Reproducing the Model

```bash
# 1. Place the provided dataset
mkdir -p dataset && cp /path/to/train_set.csv dataset/

# 2. Install dependencies
pip install pandas numpy scikit-learn lightgbm joblib

# 3. Build the local validation harness (time-based split per city)
python make_local_split.py
python build_station_hour_eval_data.py \
    --input_csv dataset/local_validation_set.csv \
    --public_targets_csv dataset/public_validation_targets.csv \
    --private_labels_csv dataset/private_labels.csv

# 4. Train (writes weights.joblib inside the submission folder)
cd submissions/challenge_1_IDs && python train.py && cd -

# 5. Score locally
python evaluate.py --eval_dir dataset --submissions_dir submissions --output_csv mae_by_city.csv
```

---

## Evaluation

- **Metric:** Mean Absolute Error — `MAE = (1/N) Σ |yᵢ − ŷᵢ|`.
- Predictions are clipped to non-negative before scoring: `ŷ = max(0, ŷ)`.
- The evaluator reports **per-city MAE** plus an overall row; test data is assumed to
  be evenly distributed across the tested cities.

---

## Notes & Caveats

- **External data:** using bike-share data beyond the provided `train_set.csv`
  (other months/cities from the source portals) **may be against hackathon rules** —
  confirm with course staff before training on anything extra.
- **No absolute paths** (e.g. `C:/Users/...`) in any submitted file; `train.py` must
  read from `../../dataset/train_set.csv` and run from inside the submission folder.
- Run `python check_submission_format.py` before submitting.

---

<p align="center"><sub>Data © Transport for London, Capital Bikeshare, and LA Metro Bike Share under their respective open-data licenses · Weather © Open-Meteo · POI data © OpenStreetMap contributors.</sub></p>
