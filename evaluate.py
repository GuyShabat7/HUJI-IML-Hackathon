#!/usr/bin/env python3
"""
Instructor-side MAE evaluator for bike-demand submissions.

This is an actual evaluation script, not a grading/points script.

Expected split submission format:

    submissions/
      team_a/
        train.py
        model.py
        predict.py
        weights.joblib

Expected hidden evaluation files:

    private_eval/
      public_test_targets.csv      # hidden test features, no demand column
      private_test_labels.csv      # hidden labels, contains demand and city

Despite the name public_test_targets.csv, this file can remain private.
It is only the feature dataframe passed into Model.predict(...).

Output:
    mae_by_city.csv

Run example:

    python evaluate_mae.py ^
      --eval_dir private_eval ^
      --submissions_dir submissions ^
      --output_csv mae_by_city.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EVAL_DIR = PROJECT_ROOT / "dataset"
DEFAULT_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
DEFAULT_WEIGHTS_FILENAME = "weights.joblib"

FORBIDDEN_TEST_COLUMNS = {
    "demand",
    "started_at",
    "ended_at",
    "end_station_id",
    "usage_time_minutes",
    "distance_meters",
    "user_type",
    "prediction",
    "predicted_demand",
}


def require_columns(df: pd.DataFrame, required: Iterable[str], file_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def assert_test_targets_are_safe(test_df: pd.DataFrame) -> None:
    """
    Prevent accidental evaluation on raw ride-level test data or leaked labels.
    """
    bad = sorted(FORBIDDEN_TEST_COLUMNS.intersection(test_df.columns))
    if bad:
        raise ValueError(
            "Test targets contain leakage/raw-ride columns: "
            f"{bad}. The evaluator should receive station-hour target rows only."
        )

    require_columns(test_df, ["id"], "test targets")

    has_station = (
        "start_station_id" in test_df.columns
        or "station_id" in test_df.columns
        or "station_key" in test_df.columns
    )
    if not has_station:
        raise ValueError(
            "Test targets must contain a station identifier, for example start_station_id."
        )

    has_time = (
        "target_hour_start" in test_df.columns
        or "hour_ts" in test_df.columns
        or {"date", "hour"}.issubset(set(test_df.columns))
    )
    if not has_time:
        raise ValueError(
            "Test targets must contain target time: target_hour_start, hour_ts, or date+hour."
        )


def safe_module_suffix(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def require_submission_files(team_dir: Path, weights_filename: str) -> dict[str, Path]:
    """
    Validate the split submission layout.
    """
    paths = {
        "train.py": team_dir / "train.py",
        "model.py": team_dir / "model.py",
        "predict.py": team_dir / "predict.py",
        weights_filename: team_dir / weights_filename,
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Submission {team_dir.name} is missing required files: {missing}"
        )

    return paths


def load_submission(team_dir: Path, weights_filename: str = DEFAULT_WEIGHTS_FILENAME):
    """
    Load submissions/<team>/predict.py and call Model().load(weights_path).

    The grader-facing class Model must live in predict.py.
    """
    paths = require_submission_files(team_dir, weights_filename=weights_filename)

    predict_path = paths["predict.py"]
    weights_path = paths[weights_filename]

    original_sys_path = list(sys.path)
    old_model_module = sys.modules.pop("model", None)
    old_predict_module = sys.modules.pop("predict", None)

    try:
        # Allow:
        #   from base_model import BaseModel
        #   from model import BikeDemandModel
        sys.path.insert(0, str(PROJECT_ROOT))
        sys.path.insert(0, str(team_dir))

        module_name = f"submission_{safe_module_suffix(team_dir.name)}_predict"
        spec = importlib.util.spec_from_file_location(module_name, predict_path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {predict_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "Model"):
            raise AttributeError(f"{predict_path} must define a class named Model.")

        model = module.Model()

        if not hasattr(model, "load") or not callable(model.load):
            raise AttributeError("Model must implement load(weights_path).")

        if not hasattr(model, "predict") or not callable(model.predict):
            raise AttributeError("Model must implement predict(test_df).")

        model.load(str(weights_path))
        return model

    finally:
        sys.path = original_sys_path

        # Prevent one team's imports from leaking into the next team.
        sys.modules.pop("model", None)
        sys.modules.pop("predict", None)

        if old_model_module is not None:
            sys.modules["model"] = old_model_module
        if old_predict_module is not None:
            sys.modules["predict"] = old_predict_module


def predict_submission(model, test_df: pd.DataFrame) -> np.ndarray:
    """
    Calls model.predict(test_df) and validates the returned predictions.
    """
    preds = model.predict(test_df.copy())
    preds = np.asarray(preds)

    if preds.ndim != 1:
        preds = preds.reshape(-1)

    if len(preds) != len(test_df):
        raise ValueError(
            f"predict() returned {len(preds)} predictions, "
            f"but test set has {len(test_df)} rows."
        )

    preds = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float)

    if np.isnan(preds).any():
        n_bad = int(np.isnan(preds).sum())
        raise ValueError(f"predict() returned {n_bad} NaN/non-numeric predictions.")

    return preds


def load_eval_files(
    eval_dir: Path,
    test_targets_csv: str,
    test_labels_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_targets_path = eval_dir / test_targets_csv
    test_labels_path = eval_dir / test_labels_csv

    if not test_targets_path.exists():
        raise FileNotFoundError(
            f"Could not find test targets file: {test_targets_path}\n"
            f"Your eval folder contains: "
            f"{[p.name for p in eval_dir.iterdir()] if eval_dir.exists() else 'MISSING EVAL DIR'}\n"
            f"Use --eval_dir or --test_targets_csv if your files are elsewhere."
        )

    if not test_labels_path.exists():
        raise FileNotFoundError(
            f"Could not find test labels file: {test_labels_path}\n"
            f"Use --eval_dir or --test_labels_csv if your files are elsewhere."
        )

    test_df = pd.read_csv(test_targets_path, low_memory=False)
    labels_df = pd.read_csv(test_labels_path, low_memory=False)

    assert_test_targets_are_safe(test_df)
    require_columns(labels_df, ["id", "city", "demand"], str(test_labels_path))

    labels_df["city"] = labels_df["city"].astype(str)

    if test_df["id"].duplicated().any():
        raise ValueError(f"{test_targets_path} contains duplicated ids.")

    if labels_df["id"].duplicated().any():
        raise ValueError(f"{test_labels_path} contains duplicated ids.")

    if set(test_df["id"]) != set(labels_df["id"]):
        raise ValueError("Target ids and label ids do not match.")

    return test_df, labels_df


def score_mae_by_city(
    model_name: str,
    test_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        scored_rows:
            one row per test target with demand, prediction, abs_error

        summary:
            MAE summary with one row per city plus an OVERALL row
    """
    pred_df = pd.DataFrame(
        {
            "id": test_df["id"].to_numpy(),
            "prediction": predictions,
        }
    )

    scored = labels_df.merge(pred_df, on="id", how="left", validate="one_to_one")

    if scored["prediction"].isna().any():
        missing = int(scored["prediction"].isna().sum())
        raise ValueError(f"{model_name}: missing predictions for {missing} ids.")

    scored["prediction"] = scored["prediction"].astype(float)
    scored["prediction_clipped"] = np.maximum(0.0, scored["prediction"])

    scored["abs_error"] = np.abs(
        scored["demand"].astype(float) - scored["prediction_clipped"]
    )

    scored["squared_error"] = (
        scored["demand"].astype(float) - scored["prediction_clipped"]
    ) ** 2

    scored["model"] = model_name

    by_city = (
        scored
        .groupby("city", dropna=False)
        .agg(
            n_rows=("abs_error", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            mean_true_demand=("demand", "mean"),
            mean_prediction=("prediction_clipped", "mean"),
            # negative_predictions_before_clipping=("prediction", lambda x: int((x < 0).sum())),
        )
        .reset_index()
    )

    overall = pd.DataFrame([
        {
            "city": "OVERALL",
            "n_rows": int(len(scored)),
            "mae": float(scored["abs_error"].mean()),
            "rmse": float(np.sqrt(scored["squared_error"].mean())),
            "mean_true_demand": float(scored["demand"].astype(float).mean()),
            "mean_prediction": float(scored["prediction_clipped"].mean()),
            "negative_predictions_before_clipping": int((scored["prediction"] < 0).sum()),
        }
    ])

    summary = pd.concat([by_city, overall], ignore_index=True)
    summary.insert(0, "model", model_name)

    return scored, summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval_dir", default=DEFAULT_EVAL_DIR, type=Path)
    parser.add_argument("--submissions_dir", default=DEFAULT_SUBMISSIONS_DIR, type=Path)

    parser.add_argument(
        "--test_targets_csv",
        default="public_test_targets.csv",
        help="Station-hour target file inside eval_dir. Default: public_test_targets.csv",
    )
    parser.add_argument(
        "--test_labels_csv",
        default="private_test_labels.csv",
        help="Private labels file inside eval_dir. Default: private_test_labels.csv",
    )

    parser.add_argument(
        "--weights_filename",
        default=DEFAULT_WEIGHTS_FILENAME,
        help="Required weights/artifact filename. Default: weights.joblib",
    )

    parser.add_argument(
        "--output_csv",
        default=Path("mae_by_city.csv"),
        type=Path,
        help="Output CSV path. Default: mae_by_city.csv",
    )

    parser.add_argument(
        "--save_scored_rows",
        default=None,
        type=Path,
        help="Optional path to save row-level predictions and errors.",
    )

    args = parser.parse_args()

    test_df, labels_df = load_eval_files(
        eval_dir=args.eval_dir,
        test_targets_csv=args.test_targets_csv,
        test_labels_csv=args.test_labels_csv,
    )

    if not args.submissions_dir.exists():
        raise FileNotFoundError(f"Submissions directory does not exist: {args.submissions_dir}")

    team_dirs = sorted(d for d in args.submissions_dir.iterdir() if d.is_dir())

    if not team_dirs:
        raise FileNotFoundError(f"No team folders found in {args.submissions_dir}")

    summary_frames = []
    scored_frames = []
    failures = []

    for team_dir in team_dirs:
        team_name = team_dir.name
        print(f"Evaluating {team_name}...", end=" ", flush=True)

        try:
            model = load_submission(
                team_dir,
                weights_filename=args.weights_filename,
            )

            preds = predict_submission(model, test_df)

            scored, summary = score_mae_by_city(
                model_name=team_name,
                test_df=test_df,
                labels_df=labels_df,
                predictions=preds,
            )

            summary_frames.append(summary)
            scored_frames.append(scored)

            overall_mae = float(summary.loc[summary["city"] == "OVERALL", "mae"].iloc[0])
            print(f"OK | overall MAE={overall_mae:.6f}")

        except Exception as e:
            print(f"FAILED | {e}")
            failures.append(
                {
                    "model": team_name,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    if not summary_frames:
        raise RuntimeError("All submissions failed. No MAE table was created.")

    results = pd.concat(summary_frames, ignore_index=True)

    # Sort by overall MAE first, then model, then city.
    overall_mae_lookup = (
        results[results["city"] == "OVERALL"]
        .set_index("model")["mae"]
        .to_dict()
    )
    results["_overall_mae_for_sort"] = results["model"].map(overall_mae_lookup)
    results["_city_sort"] = np.where(results["city"] == "OVERALL", "ZZZ_OVERALL", results["city"])
    results = (
        results
        .sort_values(["_overall_mae_for_sort", "model", "_city_sort"], ascending=[True, True, True])
        .drop(columns=["_overall_mae_for_sort", "_city_sort"])
        .reset_index(drop=True)
    )

    numeric_cols = [
        "mae",
        "rmse",
        "mean_true_demand",
        "mean_prediction",
    ]
    results[numeric_cols] = results[numeric_cols].round(6)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)

    if args.save_scored_rows is not None:
        scored_all = pd.concat(scored_frames, ignore_index=True)
        args.save_scored_rows.parent.mkdir(parents=True, exist_ok=True)
        scored_all.to_csv(args.save_scored_rows, index=False)

    if failures:
        print()
        print("Failed submissions:")
        for failure in failures:
            print(f"  {failure['model']}: {failure['error']}")

    print()
    print("--- MAE by city ---")
    print(results.to_string(index=False))
    print()
    print(f"Wrote: {args.output_csv}")

    if args.save_scored_rows is not None:
        print(f"Wrote scored rows: {args.save_scored_rows}")


if __name__ == "__main__":
    main()