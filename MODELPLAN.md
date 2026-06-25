<h1 align="center">MODELPLAN — Missingness-Aware XGBoost Ensemble</h1>

<p align="center"><em>Implementation plan & task board for the bike-demand model. Tick the boxes as you go.</em></p>

> **Status:** planning / not yet implemented. This document is the team's task list for building
> the architecture agreed with the group.
>
> **Working folder for the final model:** **`submissions/challenge_1_ensamble/`** (to be created).
> The existing LightGBM model in [`submissions/challenge_1_IDs/`](submissions/challenge_1_IDs/) is an
> intentional **safety-net baseline** (so we don't fail the assignment); the ensemble below is the
> intended final submission — see [§9 Relationship to the safety-net baseline](#9-relationship-to-the-safety-net-baseline).

---

## 1. Goal & metric

Predict **station-hour demand** = number of rides that *start* at a `(city, station, hour)`.
Scored by **MAE, reported per city** (see [evaluate.py](evaluate.py)). Demand is a small, zero-heavy
count (median ≈ 1–2, long tail to 138).

The challenge explicitly rewards **generalization to a brand-new city**: `city 3` (L.A.) has only ~5
days of data, and a hidden **`city 4` is used only for grading**. This is the reason the architecture
below uses **no location/identity features**.

---

## 2. Architecture — 3 category models + 1 learned-gate orchestrator (4 XGBoost models)

```
                test row (station-hour)
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   M_weather         M_calendar         M_station
  (weather only)   (time/calendar)   (built-environment)
        │                 │                 │
     pred_W            pred_C            pred_S
        └───────┬─────────┴─────────┬───────┘
                ▼                    ▼
        per-category missingness masks (miss_W, miss_C, miss_S)
                          │
                          ▼
                 ORCHESTRATOR  (XGBoost learned gate)
        inputs: [pred_W, pred_C, pred_S, miss_W, miss_C, miss_S]   ← NO city/location
                          │
                          ▼
              max(0, ŷ)  →  final demand
```

- Each **base model** is an `XGBRegressor` trained only on its own annotation category.
- The **orchestrator** is a 4th `XGBRegressor` that blends the three predictions using the
  per-category missingness masks, so it learns to **down-weight a base model whose inputs are absent
  (or uninformative) for a given row**. XGBoost handles NaN natively, so missing inputs flow through.

---

## 3. Feature categories (exact lists)

Source of truth for the constants lives in
[build_station_hour_eval_data.py](build_station_hour_eval_data.py).

| Model | Category | Features |
|------|----------|----------|
| **M_weather** | Weather | `temperature_2m`, `relative_humidity_2m`, `apparent_temperature`, `precipitation`, `rain`, `snowfall`, `cloud_cover`, `wind_speed_10m` |
| **M_calendar** | Calendar / temporal | derived `hour`, `weekday`, `is_weekend`, `month`, `day_of_year`, cyclical `sin/cos(hour)`, `sin/cos(weekday)`; flags `weekend`, `holiday`, `working_day` |
| **M_station** | Built-environment / land-use | `bike_lane_length_500m`, `park_area_500m`, `university_count_1000m`, `office_poi_count_1000m`, `retail_poi_count_1000m`, `restaurant_cafe_count_500m`, `transit_stop_count_500m`, `distance_to_city_center`, `distance_to_nearest_rail_station` |

### Excluded EVERYWHERE (never a model input, never seen at inference)
`city`, `start_station_id`, `start_lat`, `start_lng`, raw `date`/year, and **any station/city target
("mean-demand") encoding**. These encode *location identity* and would not transfer to a new city.

> `city` may be used **at train time only** for sample-balancing, split stratification, and per-city
> MAE reporting — never as a feature.

---

## 4. Data-driven caveats (from [eda_report.txt](eda_report.txt)) — READ BEFORE CODING

These come straight from the EDA and directly shape the implementation:

- [ ] **`city 1` (London) has NO usable station metadata.** Every POI column is constant `0.0` and
  `distance_to_nearest_rail_station` is constant `-1.0`. → As-is, M_station has ~zero signal for London;
  the orchestrator must learn to ignore it there. Only `city 2`/`city 3` carry real POI signal.
  - [ ] **Planned fix — external enrichment:** backfill London (and any other gap) station metadata from
        the **same upstream source the course used (OpenStreetMap per station)** so M_station becomes
        useful everywhere instead of being gated off. London station coords are recoverable
        (TfL station numbers ↔ `start_station_id`, see [README.md](README.md#supplementary-data--full-year-london)),
        which lets us recompute the `*_500m`/`*_1000m` counts and distances.
  - [ ] **⚠️ Rules check first:** confirm with course staff that external feature enrichment is allowed
        (README warns extra data *may* be disallowed). Keep enrichment reproducible (a script under
        `tools/`) and **never overwrite `dataset/train_set.csv`**. If disallowed, fall back to the
        orchestrator-gating behavior above.
- [ ] **Treat `distance_to_nearest_rail_station == -1` as missing (NaN)** — it's a sentinel, not a value.
- [ ] **`city 3` (L.A.) has `start_lat`/`start_lng` 100% missing** (already excluded) and constant
  `precipitation/rain/snowfall/holiday`.
- [ ] **Define the missingness mask to also catch "present-but-uninformative"** (e.g. all POI columns
  `0` ⇒ `miss_station ≈ 1`), not just literal NaN — otherwise London looks "complete" but is useless.
- [ ] **Demand is small counts, long tail** (median ≈ 1–2, max 138, very zero-heavy) → use
  `count:poisson` and/or `reg:absoluteerror`/`reg:squarederror`; **MAE is the selection metric**.
- [ ] **Hour/weekday patterns differ by city** (city1 peak 08:00 commuter, city2 peak 17:00, city3
  peak 12:00 leisure; weekend ratio 4.25× vs 1.24×). Temporal features carry most of the transferable
  signal once identity is removed.
- [ ] **Accuracy trade-off, decided deliberately:** the existing baseline's strongest feature is the
  *station × hour-of-week mean-demand* target encoding (identity). Excluding it (for generalization)
  will likely **lower in-distribution MAE on cities 1–2** vs. that baseline. Justified by the unseen-city
  goal — **but benchmark both before choosing what to submit** (§9).

---

## 5. Data pipeline (reuse existing infra — do not reinvent)

The target does not exist in the raw data; **build it by aggregation**, reconstructing zero hours.

- [ ] **Local split:** reuse [make_local_split.py](make_local_split.py) — per-city time-based holdout
  (latest 20% per city). A random split leaks the future.
- [ ] **Station-hour + zero reconstruction:** follow the pattern already in
  [`submissions/challenge_1_IDs/train.py`](submissions/challenge_1_IDs/train.py) `build_training_table()`
  / the helpers in [build_station_hour_eval_data.py](build_station_hour_eval_data.py)
  (`make_active_window_station_hour_grid`, `add_station_hour_keys`, `normalize_station_id`). Keep daytime
  hours `06:00–22:00` and add back empty hours inside each station's active window as `demand = 0`.
- [ ] **Labeled validation set:** convert the held-out slice with `build_station_hour_eval_data.py`
  (`--public_targets_csv` / `--private_labels_csv`) and **join on `id`** to get features + labels.
- [ ] Cache intermediate frames under `submissions/challenge_1_ensamble/_cache/` (gitignored).

---

## 6. Validation protocol

- [ ] **Train on `city 1` + `city 2` only.** Hold out **all of `city 3`** as the "never-before-seen
  city" generalization test.
- [ ] **In-distribution temporal holdout:** within c1 & c2, latest ~20% by time → validation
  (early stopping, model selection, and to produce the base predictions that train the orchestrator).
- [ ] **City-balanced `sample_weight`** (inverse to per-city volume) so London doesn't dominate.
- [ ] **Report per-city MAE & RMSE** (mirror [evaluate.py](evaluate.py)) separately for: c1/c2 temporal
  holdout *and* c3 (headline generalization number).
- [ ] **Missingness stress test:** zero-out a whole category in validation and confirm the orchestrator
  reweights and MAE degrades gracefully.

---

## 7. Submission contract (must pass [check_submission_format.py](check_submission_format.py))

Target folder: **`submissions/challenge_1_ensamble/`** (the team's final-model working folder; rename to
`challenge_1_<ID1>_<ID2>` for the actual course submission and fill names/IDs in its `README`). Copy the
fixed `predict.py` from `challenge_1_IDs` unchanged.

- [ ] `train.py` reads `../../dataset/train_set.csv`, writes `weights.joblib`.
- [ ] `model.py` defines `BikeDemandModel` + **all shared feature code** (single source of truth,
  imported by `train.py`). Self-contained: `numpy/pandas/xgboost/joblib` only.
- [ ] `predict.py` defines `Model(BaseModel)` — **leave as-is**.
- [ ] `weights.joblib` holds the 3 base models + orchestrator + `CATEGORY_FEATURES` + feature-column
  order + non-spatial global fallback table + chosen objectives. **No identity/location lookups.**
- [ ] **`predict.py`/`model.py` must NOT read any data file**; predictions must be **deterministic**,
  **non-negative**, length `== len(test_df)`, and must **not mutate** `test_df`.
- [ ] Robust timestamp handling: accept `hour_ts` **or** `target_hour_start`.
- [ ] Cold-start (all categories missing) → fill from a **non-spatial global fallback** (e.g. median
  demand by hour-of-day × weekday computed over c1+c2), then clip to ≥ 0.

---

## 8. Implementation task list (by collaborator)

> 3 base models + orchestrator map cleanly onto the team. Claim a section by adding your name.

### Dev 1 — Data & validation harness  ([data pipeline §5], [validation §6])
> 🚧 **IN PROGRESS — claude (2026-06-25).** Building `submissions/challenge_1_ensamble/data.py`.
- [ ] Labeled station-hour builder with zero reconstruction (reuse `build_training_table` pattern).
- [ ] Per-city time split (reuse `make_local_split.py`); city-3-held-out + c1/c2 temporal holdout.
- [ ] Caching + a `load_splits()` helper the others can import.

### Dev 2 — Shared features + M_weather + M_calendar  ([§3], [§4])
- [ ] `CATEGORY_FEATURES`, `build_category_matrix(df, cat)`, `category_missingness(df, cat)` in `model.py`
      (incl. the −1 sentinel and "all-POI-zero ⇒ missing" rules).
- [ ] Derived/cyclical temporal features; `holiday_name` treated as a generic low-demand flag (it's US
      holidays for every city — not a country signal).
- [ ] Train + tune `M_weather` and `M_calendar` (objective by val MAE).

### Dev 3 — M_station + orchestrator + packaging  ([§2], [§7])
- [ ] Train + tune `M_station` (expect near-zero signal on city 1 — that's expected).
- [ ] Build orchestrator training matrix from base preds on the holdout; train the gate.
- [ ] Assemble & `joblib.dump` `weights.joblib`; implement `BikeDemandModel.predict` end-to-end.

### Dev 4 — Eval harness, ablations, format check, git  ([§6], [§7])
- [ ] `eval_local.py` mirroring `evaluate.py` (per-city MAE/RMSE; c1/c2 vs c3).
- [ ] Ablations (each base alone vs simple average vs orchestrator) + missingness stress test.
- [ ] `python check_submission_format.py challenge_1_ensamble` green; sync + push.

---

## 9. Relationship to the safety-net baseline

[`submissions/challenge_1_IDs/`](submissions/challenge_1_IDs/) is our **safety-net baseline** — a working
LightGBM model (two regressors, `poisson` + `regression_l1`, averaged) kept so we **don't fail the
assignment** while the ensemble is in progress. It **uses** the features the ensemble excludes —
`start_lat`/`start_lng` and hierarchical station/city **target encodings** (its README calls
`te_station_how` "usually the single most predictive feature").

- [ ] **Build the ensemble in its own folder** `submissions/challenge_1_ensamble/` — leave the baseline
      untouched.
- [ ] **Compare on the same validation harness**, especially the held-out city 3 / unseen-city metric.
- [ ] **The ensemble is the intended final submission**; the baseline is the fallback. Submit the
      ensemble once it matches/beats the baseline (or clearly wins on the unseen city). The README notes
      test data is assumed evenly distributed across tested cities, with a hidden city 4 graded.

---

## 10. Definition of done

- [ ] `cd submissions/challenge_1_ensamble && python train.py` → writes `weights.joblib`, prints per-city val MAE.
- [ ] `eval_local.py`: orchestrator beats every single base model and the safety-net baseline; graceful
      under missingness; city-3 MAE recorded.
- [ ] `check_submission_format.py challenge_1_ensamble` passes.
- [ ] `predict()` is deterministic, non-negative, correct length, non-mutating.
- [ ] Decision logged in §9; repo synced and pushed (dataset never committed).
