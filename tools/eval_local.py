#!/usr/bin/env python3
"""tools/eval_local.py — one reusable local evaluation harness for any submission.

Given a submission folder it:
  1. builds the agreed splits via ``submissions/challenge_1_ensamble/data.py``
     (TRAIN = c1+c2 earliest 80% by time, VAL = c1+c2 latest 20%, TEST_UNSEEN = all
     of city 3 — held out whole, never trained on);
  2. trains the submission's model on TRAIN only;
  3. predicts through the submission's real ``BikeDemandModel.predict`` path on VAL
     and TEST_UNSEEN;
  4. reports per-city MAE/RMSE plus a blended headline number.

City 3 is a genuine unknown here: the model is fit on c1+c2 only, so its city-3 MAE is
an honest unseen-city probe. Mirrors evaluate.py's scoring (per-city MAE, preds clipped
to >= 0).

Training contract (submission-agnostic):
  * If the submission's ``model.py`` defines ``train_artifacts(train_df, n_jobs=...)``
    -> use it (the clean path for the new contenders).
  * Otherwise fall back to the baseline adapter: ``model.fit_encodings`` +
    ``model.build_features`` + ``train.train_one`` (so the existing challenge_1_IDs
    baseline works UNCHANGED).

Usage:
    python tools/eval_local.py submissions/challenge_1_IDs
    python tools/eval_local.py submissions/challenge_1_ensamble --n-jobs 8

Blended metric (grading is ~2/3 cities 1+2, 1/3 the unseen city):
    blended = (2/3) * mean(MAE_city1, MAE_city2) + (1/3) * MAE_city3
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ENSEMBLE_DIR = ROOT / "submissions" / "challenge_1_ensamble"

# Grading weights (see module docstring / prompt context).
W_INDIST = 2.0 / 3.0   # weight on mean(city1, city2)
W_UNSEEN = 1.0 / 3.0   # weight on the unseen city


# --------------------------------------------------------------------------- #
# Module loading
# --------------------------------------------------------------------------- #
def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register before exec so @dataclass / typing lookups work
    spec.loader.exec_module(mod)
    return mod


def load_data_harness():
    """Import the shared data.py (splits + station-hour table) by file path."""
    if str(ENSEMBLE_DIR) not in sys.path:
        sys.path.insert(0, str(ENSEMBLE_DIR))
    return _load_module(ENSEMBLE_DIR / "data.py", "eval_data_harness")


def import_submission(folder: Path):
    """Import a submission's model.py (+ train.py) the way the grader resolves imports."""
    for p in (str(ROOT), str(folder)):
        if p not in sys.path:
            sys.path.insert(0, p)
    # fresh modules so two submissions never cross-contaminate
    for m in ("model", "train"):
        sys.modules.pop(m, None)
    model_mod = _load_module(folder / "model.py", "model")
    sys.modules["model"] = model_mod  # so train.py's `from model import ...` resolves
    train_mod = None
    if (folder / "train.py").exists():
        train_mod = _load_module(folder / "train.py", "train")
    return model_mod, train_mod


# --------------------------------------------------------------------------- #
# Train / predict adapters
# --------------------------------------------------------------------------- #
def build_artifacts(model_mod, train_mod, train_df: pd.DataFrame, n_jobs: int) -> dict:
    """Fit the submission's model on the TRAIN split and return its artifacts dict."""
    if hasattr(model_mod, "train_artifacts"):
        return model_mod.train_artifacts(train_df, n_jobs=n_jobs)

    # Baseline adapter (challenge_1_IDs): replicate train.py on the given table.
    if not (hasattr(model_mod, "fit_encodings") and hasattr(model_mod, "build_features")
            and train_mod is not None and hasattr(train_mod, "train_one")):
        raise RuntimeError(
            "submission exposes neither train_artifacts(...) nor the baseline "
            "fit_encodings/build_features/train_one API; cannot evaluate it.")
    enc = model_mod.fit_encodings(train_df)
    X = model_mod.build_features(train_df, enc)
    y = train_df["demand"].astype(float)
    artifacts = {
        "model_poisson": train_mod.train_one(X, y, "poisson"),
        "model_mae": train_mod.train_one(X, y, "regression_l1"),
        "encodings": enc,
        "feature_cols": list(X.columns),
    }
    return artifacts


def to_eval_input(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a station-hour table (data.py schema) into a grader-style predict input.

    The model's predict()/add_keys expects ``start_station_id`` + ``city`` + a
    timestamp (``hour_ts``). The split frames carry ``station_key``/``city_key``/``ts``.
    """
    out = df.copy()
    out["start_station_id"] = df["station_key"].astype(str)
    out["city"] = df["city_key"].astype(str)
    out["hour_ts"] = pd.to_datetime(df["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    return out


def predict_demand(model_mod, artifacts: dict, df: pd.DataFrame) -> np.ndarray:
    bm = model_mod.BikeDemandModel()
    bm.load_artifacts(artifacts)
    preds = np.asarray(bm.predict(to_eval_input(df))).reshape(-1).astype(float)
    return np.maximum(0.0, preds)  # mirror the evaluator's clip


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def per_city_scores(df: pd.DataFrame, preds: np.ndarray) -> dict[str, dict]:
    y = df["demand"].to_numpy(dtype=float)
    err = np.abs(y - preds)
    sq = (y - preds) ** 2
    city = df["city_key"].astype(str).to_numpy()
    out = {}
    for c in sorted(set(city)):
        m = city == c
        out[c] = {
            "n_rows": int(m.sum()),
            "mae": float(err[m].mean()),
            "rmse": float(np.sqrt(sq[m].mean())),
            "mean_true": float(y[m].mean()),
            "mean_pred": float(preds[m].mean()),
        }
    return out


def evaluate(folder: Path, n_jobs: int, val_fraction: float, rebuild: bool) -> dict:
    data = load_data_harness()
    splits = data.load_splits(val_fraction=val_fraction, rebuild=rebuild)
    model_mod, train_mod = import_submission(folder)

    artifacts = build_artifacts(model_mod, train_mod, splits.train, n_jobs)

    val_pred = predict_demand(model_mod, artifacts, splits.val)
    test_pred = predict_demand(model_mod, artifacts, splits.test_unseen)

    scores = per_city_scores(splits.val, val_pred)            # city1 + city2 (in-dist)
    scores.update(per_city_scores(splits.test_unseen, test_pred))  # city3 (unseen)
    return {"folder": folder.name, "holdout_city": splits.holdout_city, "scores": scores}


def blended(scores: dict[str, dict], holdout_city: str) -> tuple[float, float, float]:
    indist = [v["mae"] for c, v in scores.items() if c != holdout_city]
    mean_indist = float(np.mean(indist)) if indist else float("nan")
    unseen = scores.get(holdout_city, {}).get("mae", float("nan"))
    return mean_indist, unseen, W_INDIST * mean_indist + W_UNSEEN * unseen


def print_report(result: dict) -> None:
    scores, hc = result["scores"], result["holdout_city"]
    print("\n" + "=" * 72)
    print(f"eval_local — submission: {result['folder']}   (unseen city = {hc})")
    print("=" * 72)
    rows = []
    for c in sorted(scores):
        s = scores[c]
        tag = "UNSEEN" if c == hc else "in-dist"
        rows.append([c, tag, s["n_rows"], round(s["mae"], 4), round(s["rmse"], 4),
                     round(s["mean_true"], 3), round(s["mean_pred"], 3)])
    hdr = ["city", "role", "n_rows", "MAE", "RMSE", "mean_true", "mean_pred"]
    widths = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*hdr))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    mean_indist, unseen, bl = blended(scores, hc)
    print("-" * 72)
    print(f"mean(city1,city2) MAE = {mean_indist:.4f}   |   unseen({hc}) MAE = {unseen:.4f}")
    print(f"BLENDED MAE = (2/3)*{mean_indist:.4f} + (1/3)*{unseen:.4f} = {bl:.4f}")
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local per-city + blended MAE for a submission.")
    ap.add_argument("folder", help="submission folder, e.g. submissions/challenge_1_IDs")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--rebuild", action="store_true", help="force rebuild of the cached table")
    args = ap.parse_args()

    folder = (ROOT / args.folder).resolve() if not Path(args.folder).is_absolute() else Path(args.folder)
    if not folder.exists():
        # allow passing just the folder name
        folder = (ROOT / "submissions" / args.folder).resolve()
    if not folder.exists():
        raise SystemExit(f"submission folder not found: {args.folder}")

    result = evaluate(folder, n_jobs=args.n_jobs, val_fraction=args.val_fraction, rebuild=args.rebuild)
    print_report(result)


if __name__ == "__main__":
    main()
