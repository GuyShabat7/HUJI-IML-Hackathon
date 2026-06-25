#!/usr/bin/env python3
"""tools/bakeoff.py — Step 2: three-way feature bake-off (decide identity with data).

Trains the SAME LightGBM recipe (poisson + L1, averaged — the baseline settings) on three
feature sets that differ ONLY in features, evaluated on the SAME splits as eval_local:
  A baseline    — existing challenge_1_IDs (identity + lat/lng + in-sample TE), via eval_local
  B no_identity — weather + calendar + built-environment only (features.build_no_identity)
  C fallback_te — B + out-of-fold hierarchical target encodings (features.build_fallback_te_*)

Prints per-city + blended MAE side by side and recommends a winner on the BLENDED metric
(≈ 2/3 mean(city1,city2) + 1/3 unseen city 3).

    python tools/bakeoff.py --n-jobs 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "submissions" / "challenge_1_ensamble"))

import eval_local  # noqa: E402  (scoring + baseline evaluate + data harness loader)
import features as F  # noqa: E402

# Recorded Step-1 baseline (challenge_1_IDs, full 1500-tree poisson+L1 recipe) measured by
# tools/eval_local.py — reused so the bake-off doesn't re-train the baseline every time.
# Re-measure anytime with:  python tools/eval_local.py submissions/challenge_1_IDs
BASELINE_STEP1 = {
    "city 1": {"n_rows": 158935, "mae": 0.9846, "rmse": 1.5340, "mean_true": 1.553, "mean_pred": 1.296},
    "city 2": {"n_rows": 147885, "mae": 0.5684, "rmse": 1.1676, "mean_true": 0.822, "mean_pred": 0.565},
    "city 3": {"n_rows": 10866,  "mae": 1.1415, "rmse": 1.2037, "mean_true": 0.200, "mean_pred": 1.236},
}

# Base recipe = baseline settings; n_estimators is overridable (the bake-off uses a
# lighter count for a fast *ranking* — the final Step-3 model uses the full count).
# force_col_wise removes the row-wise overhead test and is faster for this row count.
LGBM_PARAMS = dict(
    learning_rate=0.03, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, force_col_wise=True,
)


def _train_lgbm(X, y, objective, n_jobs, weight, n_estimators):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(objective=objective, n_estimators=n_estimators, n_jobs=n_jobs,
                          verbose=-1, **LGBM_PARAMS)
    m.fit(X, y, sample_weight=weight)
    return m


def _fit_predict(Xtr, ytr, w, Xva, Xte, n_jobs, n_estimators, objectives):
    """Train the requested objective(s), average, clip to >= 0.

    objectives='poisson'    -> fast, gradient-based, multi-threaded (default for ranking).
    objectives='poisson_l1' -> baseline-faithful poisson + regression_l1 average (slower:
                               weighted-median L1 is single-threaded).
    """
    Xva = Xva.reindex(columns=Xtr.columns)
    Xte = Xte.reindex(columns=Xtr.columns)
    objs = {"poisson": ["poisson"], "poisson_l1": ["poisson", "regression_l1"]}[objectives]
    models = [_train_lgbm(Xtr, ytr, o, n_jobs, w, n_estimators) for o in objs]
    pv = np.mean([m.predict(Xva) for m in models], axis=0)
    pt = np.mean([m.predict(Xte) for m in models], axis=0)
    return np.maximum(0.0, pv), np.maximum(0.0, pt)


def _scores_for(name, splits, val_pred, test_pred):
    s = eval_local.per_city_scores(splits.val, val_pred)
    s.update(eval_local.per_city_scores(splits.test_unseen, test_pred))
    return {"name": name, "holdout_city": splits.holdout_city, "scores": s}


def run(n_jobs, val_fraction, rebuild, n_estimators, sample, objectives, recompute_baseline):
    data = eval_local.load_data_harness()
    splits = data.load_splits(val_fraction=val_fraction, rebuild=rebuild)

    # Optional train subsample for a fast ranking (val/test stay full).
    train_fit = splits.train
    if sample < 1.0:
        train_fit = splits.train.sample(frac=sample, random_state=42).reset_index(drop=True)
    print(f"[bakeoff] recipe: objectives={objectives}  n_estimators={n_estimators}  "
          f"train_sample={sample}  train_rows={len(train_fit):,}/{len(splits.train):,}", flush=True)
    w = data.city_balanced_weights(train_fit)
    ytr = train_fit["demand"].astype(float)

    results = []

    # A — real baseline (full recipe). Reuse the recorded Step-1 number unless asked to
    # recompute (re-training it is the slow part and it never changes).
    if recompute_baseline:
        print("[bakeoff] A baseline (re-measuring full recipe) ...", flush=True)
        a = eval_local.evaluate(ROOT / "submissions" / "challenge_1_IDs",
                                n_jobs=n_jobs, val_fraction=val_fraction, rebuild=False)
        results.append({"name": "A baseline*", "holdout_city": a["holdout_city"], "scores": a["scores"]})
    else:
        print("[bakeoff] A baseline* (recorded Step-1 numbers)", flush=True)
        results.append({"name": "A baseline*", "holdout_city": splits.holdout_city,
                        "scores": BASELINE_STEP1})

    # B — no identity.
    print("[bakeoff] B no_identity ...", flush=True)
    Xtr = F.build_no_identity(train_fit)
    pv, pt = _fit_predict(Xtr, ytr, w,
                          F.build_no_identity(splits.val),
                          F.build_no_identity(splits.test_unseen), n_jobs, n_estimators, objectives)
    results.append(_scores_for("B no_identity", splits, pv, pt))

    # C — no identity + out-of-fold fallback target encodings.
    print("[bakeoff] C fallback_te (OOF) ...", flush=True)
    Xtr_c = F.build_fallback_te_train(train_fit)
    enc_full = F.te_fit(train_fit)
    pv, pt = _fit_predict(Xtr_c, ytr, w,
                          F.build_fallback_te_eval(splits.val, enc_full),
                          F.build_fallback_te_eval(splits.test_unseen, enc_full),
                          n_jobs, n_estimators, objectives)
    results.append(_scores_for("C fallback_te", splits, pv, pt))

    _print_table(results)
    return results


def _print_table(results):
    hc = results[0]["holdout_city"]
    print("\n" + "=" * 78)
    print(f"STEP 2 — feature bake-off   (unseen city = {hc};  blended = 2/3*mean(c1,c2)+1/3*c3)")
    print("=" * 78)
    cities = ["city 1", "city 2", "city 3"]
    hdr = ["contender", "MAE c1", "MAE c2", "MAE c3(unseen)", "mean(c1,c2)", "BLENDED"]
    rows = []
    for r in results:
        sc = r["scores"]
        c1 = sc.get("city 1", {}).get("mae", float("nan"))
        c2 = sc.get("city 2", {}).get("mae", float("nan"))
        mean_indist, unseen, bl = eval_local.blended(sc, hc)
        rows.append([r["name"], f"{c1:.4f}", f"{c2:.4f}", f"{unseen:.4f}",
                     f"{mean_indist:.4f}", f"{bl:.4f}"])
    widths = [max(len(str(x[i])) for x in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*hdr))
    for x in rows:
        print(fmt.format(*x))
    print("-" * 78)
    best = min(results, key=lambda r: eval_local.blended(r["scores"], hc)[2])
    bb = eval_local.blended(best["scores"], hc)[2]
    print(f"RECOMMENDED WINNER (lowest blended): {best['name']}  (blended MAE = {bb:.4f})")
    print("* A baseline = the real submitted model (full 1500-tree recipe, no sample weight) "
          "= the bar.\n  B/C use the fast ranking recipe; the Step-3 model retrains the winner "
          "on the full recipe.")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=500,
                    help="trees for B/C fast ranking (baseline A uses its full 1500)")
    ap.add_argument("--sample", type=float, default=0.35,
                    help="train subsample fraction for B/C fast ranking (val/test stay full)")
    ap.add_argument("--objectives", choices=["poisson", "poisson_l1"], default="poisson",
                    help="poisson = fast ranking (default); poisson_l1 = baseline-faithful but slow")
    ap.add_argument("--recompute-baseline", action="store_true",
                    help="re-train the baseline instead of using recorded Step-1 numbers")
    args = ap.parse_args()
    run(args.n_jobs, args.val_fraction, args.rebuild, args.n_estimators, args.sample,
        args.objectives, args.recompute_baseline)


if __name__ == "__main__":
    main()
