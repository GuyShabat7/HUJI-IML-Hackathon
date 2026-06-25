#!/usr/bin/env python3
"""tools/score_holdout.py — score the model on a genuinely-unseen city (Chicago = city 4).

Trains a submission's model on the in-distribution TRAIN split (cities 1+2 only, via the
shared eval_local/data harness — city 3 AND city 4 held out), then reports MAE for:
  * city 1, city 2  — in-distribution temporal holdout (val)
  * city 3          — unseen-city probe already inside train_set.csv
  * city 4          — Chicago/Divvy, a brand-new unseen city (this file)
all from the SAME model, so c3 and c4 are directly comparable as unseen cities.

Chicago demand is reconstructed with the SAME zero-reconstruction as training
(data.build_station_hour_table -> daytime active-window grid), so the target rows match
the evaluator's logic. Predictions are clipped to >= 0.

Hard leakage guard: asserts city 3 and city 4 never appear in train/val (fails loudly).

    python tools/score_holdout.py --enriched dataset/chicago_enriched.csv
    python tools/score_holdout.py --submission submissions/challenge_1_IDs --enriched dataset/chicago_enriched.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import eval_local  # noqa: E402

FORBIDDEN_IN_TRAIN = {"city 3", "city 4"}


def _assert_no_leak(splits, extra_city: str):
    seen = set(splits.train["city_key"].astype(str)) | set(splits.val["city_key"].astype(str))
    leaked = (FORBIDDEN_IN_TRAIN | {extra_city}) & seen
    if leaked:
        raise AssertionError(f"LEAKAGE: holdout cities {sorted(leaked)} found in train/val!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="submissions/challenge_1_IDs")
    ap.add_argument("--enriched", required=True, help="enriched held-out ride file (city 4)")
    ap.add_argument("--holdout-city", default="city 4")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    args = ap.parse_args()

    folder = (ROOT / args.submission).resolve()
    data = eval_local.load_data_harness()
    splits = data.load_splits(val_fraction=args.val_fraction)
    _assert_no_leak(splits, args.holdout_city)  # loud failure if c3/c4 leaked

    # Train the model on c1+c2 (held-out protocol) and score the in-dist + c3 sets.
    model_mod, train_mod = eval_local.import_submission(folder)
    artifacts = eval_local.build_artifacts(model_mod, train_mod, splits.train, args.n_jobs)
    val_pred = eval_local.predict_demand(model_mod, artifacts, splits.val)
    c3_pred = eval_local.predict_demand(model_mod, artifacts, splits.test_unseen)
    scores = eval_local.per_city_scores(splits.val, val_pred)
    scores.update(eval_local.per_city_scores(splits.test_unseen, c3_pred))

    # --- city 4 (Chicago): reconstruct station-hours, then score with the SAME model ---
    rides = pd.read_csv(args.enriched, low_memory=False)
    assert (rides["city"].astype(str) == args.holdout_city).all(), \
        f"enriched file must be all {args.holdout_city!r}"
    c4_table = data.build_station_hour_table(rides)
    if args.holdout_city in (set(splits.train["city_key"]) | set(splits.val["city_key"])):
        raise AssertionError(f"LEAKAGE: {args.holdout_city} reached train/val!")
    c4_pred = eval_local.predict_demand(model_mod, artifacts, c4_table)
    scores.update(eval_local.per_city_scores(c4_table, c4_pred))

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print(f"HELD-OUT SCORING — model trained on c1+c2 only; c3 & c4 are unseen")
    print(f"submission: {folder.name}   |   city 4 source: {Path(args.enriched).name}")
    print("=" * 74)
    order = ["city 1", "city 2", "city 3", "city 4"]
    hdr = ["city", "role", "n_rows", "MAE", "RMSE", "mean_true", "mean_pred"]
    rows = []
    for c in order:
        if c not in scores:
            continue
        s = scores[c]
        role = {"city 1": "in-dist", "city 2": "in-dist",
                "city 3": "UNSEEN", "city 4": "UNSEEN(new)"}[c]
        rows.append([c, role, s["n_rows"], round(s["mae"], 4), round(s["rmse"], 4),
                     round(s["mean_true"], 3), round(s["mean_pred"], 3)])
    widths = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*hdr))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    mean12 = np.mean([scores[c]["mae"] for c in ("city 1", "city 2") if c in scores])
    print("-" * 74)
    print(f"mean(city1,city2)={mean12:.4f}   city3(unseen)={scores.get('city 3',{}).get('mae',float('nan')):.4f}"
          f"   city4(unseen-new)={scores.get('city 4',{}).get('mae',float('nan')):.4f}")
    print("leakage guard: PASSED (city 3 & city 4 absent from train/val)")
    print("=" * 74)


if __name__ == "__main__":
    main()
