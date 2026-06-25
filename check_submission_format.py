#!/usr/bin/env python3
"""
Format checker for bike-demand submissions.

Expected submission layout:

    submissions/
      team_a/
        train.py
        model.py
        predict.py
        weights.joblib
        README.md              # README, README.md, README.txt, or readme.md accepted

Contract:

    train.py
        Student training script.
        Should define main() and produce weights.joblib when run.

    model.py
        Student model logic.
        Should define BikeDemandModel.
        Should not define the grader-facing Model class.

    predict.py
        Grader-facing wrapper.
        Must define class Model(BaseModel) with:
            load(weights_path)
            predict(test_df)

    weights.joblib
        Required trained artifact.

Run:

    python check_submission_format.py
    python check_submission_format.py team_name
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

WEIGHTS_FILENAME = "weights.joblib"

REQUIRED_FILES = [
    "train.py",
    "model.py",
    "predict.py",
    WEIGHTS_FILENAME,
]

README_CANDIDATES = [
    "README.md",
    "README.txt",
    "README",
    "readme.md",
    "readme.txt",
    "readme",
]

COMMON_WEIGHT_PATTERNS = [
    "weights.joblib",
    "weights.pkl",
    "weights.pt",
    "model.joblib",
    "model.pkl",
    "*.joblib",
    "*.pkl",
    "*.pt",
]

USE_COLOR = True

ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_RESET = "\033[0m"


def colorize(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{ANSI_RESET}"

def fail(message: str) -> bool:
    print(colorize(f"[FAIL] {message}", ANSI_RED))
    return False


def warn(message: str) -> bool:
    print(colorize(f"[WARN] {message}", ANSI_YELLOW))
    return True


def pass_check(message: str) -> bool:
    print(colorize(f"[ OK ] {message}", ANSI_GREEN))
    return True


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_readme(team_dir: Path) -> Path | None:
    for name in README_CANDIDATES:
        path = team_dir / name
        if path.exists() and path.is_file():
            return path
    return None


def list_weight_like_files(team_dir: Path) -> list[Path]:
    found: dict[Path, None] = {}

    for pattern in COMMON_WEIGHT_PATTERNS:
        for path in team_dir.glob(pattern):
            if path.is_file():
                found[path] = None

    return sorted(found.keys(), key=lambda p: p.name)


def require_files(team_dir: Path, filenames: Iterable[str]) -> bool:
    ok = True

    for filename in filenames:
        file_path = team_dir / filename
        if file_path.exists() and file_path.is_file():
            pass_check(f"Found {filename}")
        else:
            ok = fail(f"Missing required file: {filename}") and ok

    return ok


def make_smoke_test_df() -> pd.DataFrame:
    """
    Small station-hour dataframe that mimics the hidden evaluator input.
    """
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "city": ["city 1", "city 1", "city 2", "city 3"],
            "start_station_id": ["101", "101.0", 202, "station_303"],
            "target_hour_start": [
                "2026-01-01 08:00:00",
                "2026-01-01 09:00:00",
                "2026-01-02 17:00:00",
                "2026-01-03 23:00:00",
            ],
            "hour_ts": [
                "2026-01-01 08:00:00",
                "2026-01-01 09:00:00",
                "2026-01-02 17:00:00",
                "2026-01-03 23:00:00",
            ],
            "date": [
                "2026-01-01",
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "weekday": [3, 3, 4, 5],
            "weekend": [0, 0, 0, 1],
            "holiday": [0, 0, 0, 0],
            "working_day": [1, 1, 1, 0],
            "temperature_2m": [15.0, 16.0, 12.5, 10.0],
            "relative_humidity_2m": [60.0, 58.0, 70.0, 80.0],
            "precipitation": [0.0, 0.0, 0.2, 0.0],
            "rain": [0.0, 0.0, 0.2, 0.0],
            "cloud_cover": [20.0, 25.0, 80.0, 40.0],
            "wind_speed_10m": [5.0, 6.0, 9.0, 3.0],
            "start_lat": [32.08, 32.08, 31.78, 32.79],
            "start_lng": [34.78, 34.78, 35.21, 34.99],
            "bike_lane_length_500m": [1000.0, 1000.0, 500.0, 750.0],
            "park_area_500m": [200.0, 200.0, 150.0, 300.0],
            "university_count_1000m": [1, 1, 0, 0],
            "office_poi_count_1000m": [20, 20, 5, 10],
            "retail_poi_count_1000m": [30, 30, 15, 8],
            "restaurant_cafe_count_500m": [10, 10, 5, 3],
            "transit_stop_count_500m": [4, 4, 2, 1],
            "distance_to_nearest_rail_station": [500.0, 500.0, 1200.0, 900.0],
            "distance_to_city_center": [1000.0, 1000.0, 2500.0, 3000.0],
        }
    )


def predict_imports_bike_demand_model(predict_path: Path) -> bool:
    """
    Static check:
    predict.py should explicitly import BikeDemandModel from model.py.

    Accepted:
        from model import BikeDemandModel
    """
    source = predict_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "model":
                imported_names = {alias.name for alias in node.names}
                if "BikeDemandModel" in imported_names:
                    return True

    return False


def warn_if_predict_has_suspicious_io(predict_path: Path) -> None:
    """
    Warning-only check.

    predict.py should normally not read train/test CSVs or access dataset paths.
    It should receive test_df from the evaluator and predict on it.

    This check is AST-based to avoid false positives from words inside class names
    such as BikeDemandModel.
    """
    source = predict_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    suspicious_calls = {
        "read_csv",
        "read_parquet",
        "read_excel",
        "read_json",
        "to_csv",
        "to_parquet",
        "to_excel",
        "open",
    }

    suspicious_string_fragments = [
        "train_set.csv",
        "validation_set.csv",
        "test_set.csv",
        "public_test_targets",
        "private_test_labels",
        "dataset/",
        "dataset\\",
        "private_eval",
    ]

    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func

            if isinstance(func, ast.Attribute):
                if func.attr in suspicious_calls:
                    found.add(func.attr)

            elif isinstance(func, ast.Name):
                if func.id in suspicious_calls:
                    found.add(func.id)

        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            for fragment in suspicious_string_fragments:
                if fragment.lower() in lowered:
                    found.add(fragment)

    if found:
        warn(
            "predict.py contains suspicious file/data access references: "
            f"{sorted(found)}. This may be fine, but predict.py should usually only use "
            "the input test_df and the loaded weights."
        )


def find_inner_bike_model_attribute(wrapper_model):
    """
    Find the attribute inside predict.Model that looks like a BikeDemandModel.

    We intentionally do NOT rely on isinstance(...), because the checker imports
    model.py once under a custom module name, while predict.py imports it again
    as plain 'model'. That creates two different BikeDemandModel class objects.
    """
    for attr_name, value in vars(wrapper_model).items():
        cls = value.__class__

        if cls.__name__ == "BikeDemandModel":
            return attr_name

    return None


def check_predict_delegates_to_bike_model(wrapper_model) -> bool:
    """
    Runtime check:
    Replace the inner BikeDemandModel with a sentinel object and verify that
    wrapper_model.predict(test_df) calls sentinel.predict(test_df).

    This verifies that predict.py actually delegates to the model in model.py.
    """
    attr_name = find_inner_bike_model_attribute(wrapper_model)

    if attr_name is None:
        visible_attrs = {
            name: value.__class__.__name__
            for name, value in vars(wrapper_model).items()
        }

        return fail(
            "predict.Model does not contain an attribute whose class is named "
            "BikeDemandModel. Expected something like: self.model = BikeDemandModel(). "
            f"Found attributes: {visible_attrs}"
        )

    pass_check(f"predict.Model contains BikeDemandModel-like instance at self.{attr_name}")

    class SentinelBikeDemandModel:
        def __init__(self):
            self.called = False
            self.received_rows = None

        def predict(self, test_df: pd.DataFrame):
            self.called = True
            self.received_rows = len(test_df)
            return np.zeros(len(test_df), dtype=float)

    sentinel = SentinelBikeDemandModel()
    setattr(wrapper_model, attr_name, sentinel)

    test_df = make_smoke_test_df()

    try:
        preds = wrapper_model.predict(test_df)
    except Exception as e:
        return fail(
            "predict.Model.predict(test_df) failed when BikeDemandModel was replaced "
            f"with a sentinel object. This suggests predict.py is not a clean wrapper. "
            f"Error: {e}"
        )

    if not sentinel.called:
        return fail(
            "predict.Model.predict(test_df) did not call BikeDemandModel.predict(test_df)."
        )

    if sentinel.received_rows != len(test_df):
        return fail(
            "BikeDemandModel.predict received the wrong number of rows. "
            f"Expected {len(test_df)}, got {sentinel.received_rows}."
        )

    if not validate_predictions(preds, n_expected=len(test_df)):
        return False

    pass_check("predict.Model.predict(...) delegates to BikeDemandModel.predict(...)")
    return True


def make_test_format_from_training_set(
    train_csv: Path,
    max_rows: int = 128,
) -> pd.DataFrame:
    """
    Build a hidden-test-style dataframe from train_set.csv.

    This does NOT include demand.
    It simulates the evaluator input format:
        one row per station-hour target
    """
    raw = pd.read_csv(train_csv, low_memory=False)

    if "start_station_id" not in raw.columns:
        raise ValueError("train_set.csv must contain start_station_id")

    df = raw.copy()

    if "city" not in df.columns:
        df["city"] = "__all_cities__"

    if "hour_ts" in df.columns:
        ts = pd.to_datetime(df["hour_ts"], errors="coerce")
    elif "started_at" in df.columns:
        ts = pd.to_datetime(df["started_at"], errors="coerce")
    else:
        raise ValueError("train_set.csv must contain hour_ts or started_at")

    df["target_hour_start"] = ts.dt.floor("h")
    df = df.dropna(subset=["target_hour_start"])

    if len(df) == 0:
        raise ValueError("Could not build smoke test: no valid timestamps in train_set.csv")

    candidate_feature_cols = [
        "city",
        "start_station_id",
        "target_hour_start",
        "start_lat",
        "start_lng",
        "bike_lane_length_500m",
        "park_area_500m",
        "university_count_1000m",
        "office_poi_count_1000m",
        "retail_poi_count_1000m",
        "restaurant_cafe_count_500m",
        "transit_stop_count_500m",
        "distance_to_nearest_rail_station",
        "distance_to_city_center",
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "wind_speed_10m",
        "holiday",
        "holiday_name",
        "working_day",
    ]

    existing_cols = [c for c in candidate_feature_cols if c in df.columns]

    test_like = (
        df[existing_cols]
        .groupby(["city", "start_station_id", "target_hour_start"], dropna=False)
        .first()
        .reset_index()
    )

    test_like = test_like.sort_values(
        ["city", "start_station_id", "target_hour_start"]
    ).head(max_rows).reset_index(drop=True)

    test_like.insert(0, "id", np.arange(len(test_like)))

    test_like["hour_ts"] = test_like["target_hour_start"]
    test_like["hour"] = test_like["target_hour_start"].dt.hour
    test_like["weekday"] = test_like["target_hour_start"].dt.weekday
    test_like["date"] = test_like["target_hour_start"].dt.date.astype(str)

    if "weekend" not in test_like.columns:
        test_like["weekend"] = test_like["weekday"].isin([5, 6]).astype(int)

    test_like["target_hour_start"] = pd.to_datetime(
        test_like["target_hour_start"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    test_like["hour_ts"] = pd.to_datetime(
        test_like["hour_ts"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    test_like = test_like.drop(
        columns=["demand", "prediction", "predicted_demand"],
        errors="ignore",
    )

    return test_like


def validate_predictions(preds, n_expected: int) -> bool:
    preds = np.asarray(preds)

    if preds.ndim != 1:
        try:
            preds = preds.reshape(-1)
            warn("predict(test_df) did not return a 1D object, but it could be flattened.")
        except Exception:
            return fail("predict(test_df) must return a 1D array-like object.")

    if len(preds) != n_expected:
        return fail(
            f"predict(test_df) returned {len(preds)} predictions, "
            f"but expected {n_expected}."
        )

    numeric = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float)

    if np.isnan(numeric).any():
        n_bad = int(np.isnan(numeric).sum())
        return fail(f"predict(test_df) returned {n_bad} NaN/non-numeric predictions.")

    pass_check("predict(test_df) returns numeric predictions")

    if np.isinf(numeric).any():
        return fail("predict(test_df) returned infinite predictions.")

    pass_check("Predictions are finite")

    if (numeric < 0).any():
        n_negative = int((numeric < 0).sum())
        return fail(
            f"predict(test_df) returned {n_negative} negative predictions. "
            "Bike demand predictions must be non-negative."
        )

    pass_check("Predictions are non-negative")

    return True


def check_prediction_behavior(model, test_df: pd.DataFrame, label: str) -> bool:
    """
    Run prediction sanity checks on a provided test-like dataframe.
    """
    print(f"[INFO] Running prediction sanity checks on {label} ({len(test_df)} rows)")

    before = test_df.copy(deep=True)

    try:
        preds1 = model.predict(test_df)
        pass_check(f"predict(test_df) runs successfully on {label}")
    except Exception as e:
        return fail(f"predict(test_df) failed on {label}: {e}")

    try:
        assert_frame_equal(test_df, before, check_dtype=False)
        pass_check("predict(test_df) does not mutate input dataframe")
    except AssertionError:
        return fail("predict(test_df) mutates the input dataframe in-place")

    if not validate_predictions(preds1, n_expected=len(test_df)):
        return False

    try:
        preds2 = model.predict(test_df.copy())
    except Exception as e:
        return fail(f"Second predict(test_df) call failed on {label}: {e}")

    preds1_arr = pd.to_numeric(
        pd.Series(np.asarray(preds1).reshape(-1)),
        errors="coerce",
    ).to_numpy(dtype=float)

    preds2_arr = pd.to_numeric(
        pd.Series(np.asarray(preds2).reshape(-1)),
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.allclose(preds1_arr, preds2_arr, equal_nan=False):
        return fail("predict(test_df) is not deterministic across repeated calls")

    pass_check("predict(test_df) is deterministic across repeated calls")

    shuffled = test_df.sample(frac=1.0, random_state=123).reset_index(drop=True)

    try:
        shuffled_preds = model.predict(shuffled.copy())
    except Exception as e:
        return fail(f"predict(shuffled_test_df) failed on {label}: {e}")

    if not validate_predictions(shuffled_preds, n_expected=len(shuffled)):
        return False

    pass_check("predict(test_df) works when rows are shuffled")

    return True

def static_find_dangerous_file_writes(py_path: Path) -> list[str]:
    """
    Static check for attempts to overwrite instructor-owned data files.

    This catches common patterns such as:
        pd.DataFrame(...).to_csv("../../dataset/train_set.csv")
        open("dataset/test_set.csv", "w")
        Path("private_test_labels.csv").write_text(...)
        shutil.copy(..., "dataset/train_set.csv")

    It is intentionally conservative. It may not catch every dynamic path, but it
    catches the common dangerous cases.
    """
    source = py_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source)

    protected_fragments = [
        "train_set.csv",
        "validation_set.csv",
        "test_set.csv",
        "public_test_targets.csv",
        "private_test_labels.csv",
        "model_scores.csv",
        "grading_results.csv",
        "dataset/train",
        "dataset\\train",
        "dataset/test",
        "dataset\\test",
        "dataset/validation",
        "dataset\\validation",
        "private_eval",
    ]

    write_methods = {
        "to_csv",
        "to_parquet",
        "to_excel",
        "to_json",
        "to_pickle",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "rmdir",
        "mkdir",
    }

    dangerous_calls = {
        "open",
        "remove",
        "unlink",
        "rename",
        "replace",
        "rmdir",
    }

    dangerous_modules = {
        "os",
        "shutil",
    }

    findings: list[str] = []

    def string_value(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def path_looks_protected(text: str) -> bool:
        normalized = text.lower().replace("\\\\", "/").replace("\\", "/")
        return any(fragment.lower().replace("\\", "/") in normalized for fragment in protected_fragments)

    def record_if_protected(text: str | None, lineno: int, reason: str) -> None:
        if text and path_looks_protected(text):
            findings.append(f"line {lineno}: {reason}: {text}")

    for node in ast.walk(tree):
        # Catch string literals anywhere in write calls.
        if isinstance(node, ast.Call):
            func = node.func

            # open("...", "w"/"a"/"x"/"+")
            if isinstance(func, ast.Name) and func.id == "open":
                if node.args:
                    path_text = string_value(node.args[0])
                    mode_text = None

                    if len(node.args) >= 2:
                        mode_text = string_value(node.args[1])

                    for keyword in node.keywords:
                        if keyword.arg == "mode":
                            mode_text = string_value(keyword.value)

                    mode_text = mode_text or "r"

                    if any(flag in mode_text for flag in ["w", "a", "x", "+"]):
                        record_if_protected(
                            path_text,
                            node.lineno,
                            f"open(..., mode={mode_text!r}) may overwrite protected data",
                        )

            # os.remove("..."), os.rename("...", "..."), shutil.copy(..., protected)
            if isinstance(func, ast.Attribute):
                attr = func.attr

                # Something like df.to_csv("...")
                if attr in write_methods:
                    for arg in node.args:
                        record_if_protected(
                            string_value(arg),
                            node.lineno,
                            f"{attr}(...) writes/removes protected data",
                        )

                    for keyword in node.keywords:
                        record_if_protected(
                            string_value(keyword.value),
                            node.lineno,
                            f"{attr}(...) writes/removes protected data",
                        )

                # os.* / shutil.* dangerous operations.
                if isinstance(func.value, ast.Name):
                    module_name = func.value.id

                    if module_name in dangerous_modules:
                        if attr in {"remove", "unlink", "rename", "replace", "rmdir"}:
                            for arg in node.args:
                                record_if_protected(
                                    string_value(arg),
                                    node.lineno,
                                    f"{module_name}.{attr}(...) modifies protected data",
                                )

                        if module_name == "shutil" and attr in {"copy", "copy2", "copyfile", "move"}:
                            # Destination is usually the second positional arg.
                            for arg in node.args[1:]:
                                record_if_protected(
                                    string_value(arg),
                                    node.lineno,
                                    f"shutil.{attr}(...) may overwrite protected data",
                                )

        # Catch Path("protected").write_text(...) / Path("protected").unlink()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr

            if attr in write_methods:
                receiver = node.func.value

                if (
                    isinstance(receiver, ast.Call)
                    and isinstance(receiver.func, ast.Name)
                    and receiver.func.id == "Path"
                    and receiver.args
                ):
                    record_if_protected(
                        string_value(receiver.args[0]),
                        node.lineno,
                        f"Path(...).{attr}(...) modifies protected data",
                    )

    return findings


def check_no_protected_data_overwrite(team_dir: Path) -> bool:
    """
    Prevent submissions from overwriting instructor-owned train/test/eval files.

    Hard failure:
        model.py, predict.py

    Warning:
        train.py, because it is expected to write weights.joblib, but it still
        should not write dataset/train_set.csv or hidden test files.
    """
    ok = True

    hard_fail_files = [
        team_dir / "model.py",
        team_dir / "predict.py",
    ]

    warning_files = [
        team_dir / "train.py",
    ]

    for py_path in hard_fail_files:
        findings = static_find_dangerous_file_writes(py_path)

        if findings:
            ok = fail(
                f"{py_path.name} appears to write/delete/overwrite protected data files."
            ) and ok
            for finding in findings:
                print(f"       - {finding}")
        else:
            pass_check(f"{py_path.name} does not appear to overwrite protected data files")

    for py_path in warning_files:
        findings = static_find_dangerous_file_writes(py_path)

        if findings:
            warn(
                f"{py_path.name} may write/delete/overwrite protected data files. "
                "Review these lines manually:"
            )
            for finding in findings:
                print(f"       - {finding}")
        else:
            pass_check(f"{py_path.name} does not appear to overwrite protected data files")

    return ok


def check_team_submission(team_dir: Path) -> bool:
    print()
    print("=" * 70)
    print(f"Checking submission: {team_dir.name}")
    print("=" * 70)

    ok = True

    if not team_dir.exists():
        return fail(f"Submission folder does not exist: {team_dir}")

    if not team_dir.is_dir():
        return fail(f"Submission path is not a folder: {team_dir}")

    pass_check("Submission folder exists")

    # ------------------------------------------------------------------
    # Required files
    # ------------------------------------------------------------------
    if not require_files(team_dir, REQUIRED_FILES):
        ok = False

    readme_path = find_readme(team_dir)
    if readme_path is None:
        ok = fail(
            "Missing README file. Accepted names: "
            + ", ".join(README_CANDIDATES)
        ) and ok
    else:
        if readme_path.stat().st_size == 0:
            ok = fail(f"README exists but is empty: {readme_path.name}") and ok
        else:
            pass_check(f"Found non-empty README: {readme_path.name}")

            readme_text = readme_path.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if len(readme_text) < 50:
                warn(
                    f"README is very short ({len(readme_text)} characters). "
                    "It should briefly explain the approach and how to run train.py."
                )

    # ------------------------------------------------------------------
    # Weights / artifacts
    # ------------------------------------------------------------------
    weight_like_files = list_weight_like_files(team_dir)
    if weight_like_files:
        print("[INFO] Weight-like files found:")
        for path in weight_like_files:
            print(f"       - {path.name}")

    extra_weight_files = [
        p for p in weight_like_files
        if p.name != WEIGHTS_FILENAME
    ]

    if extra_weight_files:
        warn(
            "Extra weight/model artifact files found. "
            f"The grader will use only {WEIGHTS_FILENAME}."
        )

    weights_path = team_dir / WEIGHTS_FILENAME

    if weights_path.exists() and weights_path.is_file():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        pass_check(f"{WEIGHTS_FILENAME} exists ({size_mb:.2f} MB)")

        if size_mb > 100:
            warn(
                f"{WEIGHTS_FILENAME} is large ({size_mb:.2f} MB). "
                "This may be acceptable, but large artifacts can slow grading."
            )

    if not ok:
        return False

    # ------------------------------------------------------------------
    # Protected data overwrite checks
    # ------------------------------------------------------------------
    if not check_no_protected_data_overwrite(team_dir):
        ok = False

    if not ok:
        return False

    train_path = team_dir / "train.py"
    model_path = team_dir / "model.py"
    predict_path = team_dir / "predict.py"

    dataset_dir = PROJECT_ROOT / "dataset"
    train_csv = dataset_dir / "train_set.csv"

    original_sys_path = list(sys.path)

    old_model_module = sys.modules.pop("model", None)
    old_predict_module = sys.modules.pop("predict", None)

    try:
        # Make project root importable for base_model.py.
        sys.path.insert(0, str(PROJECT_ROOT))

        # Make this team folder importable, so predict.py can do:
        # from model import BikeDemandModel
        sys.path.insert(0, str(team_dir))

        # --------------------------------------------------------------
        # train.py checks
        # --------------------------------------------------------------
        train_before_mtime = weights_path.stat().st_mtime if weights_path.exists() else None

        train_module = load_module(train_path, f"{team_dir.name}_train")
        pass_check("train.py imports successfully")

        train_after_mtime = weights_path.stat().st_mtime if weights_path.exists() else None

        if train_before_mtime != train_after_mtime:
            warn(
                "Importing train.py appears to modify weights.joblib. "
                "Training should normally happen only inside main(), not at import time."
            )

        if hasattr(train_module, "main") and callable(train_module.main):
            pass_check("train.py defines callable main()")
        else:
            warn(
                "train.py does not define callable main(). "
                "It may still run, but main() is recommended."
            )

        train_source = train_path.read_text(encoding="utf-8", errors="ignore")
        if WEIGHTS_FILENAME in train_source:
            pass_check(f"train.py mentions {WEIGHTS_FILENAME}")
        else:
            warn(
                f"train.py does not mention {WEIGHTS_FILENAME}. "
                "Make sure running train.py creates the required artifact."
            )

        # --------------------------------------------------------------
        # model.py checks
        # --------------------------------------------------------------
        model_module = load_module(model_path, f"{team_dir.name}_model")
        pass_check("model.py imports successfully")

        if hasattr(model_module, "BikeDemandModel"):
            pass_check("Found BikeDemandModel class in model.py")
        else:
            ok = fail("model.py must define a class named BikeDemandModel") and ok

        if hasattr(model_module, "Model"):
            ok = fail(
                "model.py should not define the grader-facing class Model. "
                "Move Model(BaseModel) to predict.py."
            ) and ok
        else:
            pass_check("model.py does not define grader-facing Model class")

        # --------------------------------------------------------------
        # predict.py static checks
        # --------------------------------------------------------------
        if predict_imports_bike_demand_model(predict_path):
            pass_check("predict.py imports BikeDemandModel from model.py")
        else:
            ok = fail(
                "predict.py must import BikeDemandModel from model.py, "
                "for example: from model import BikeDemandModel"
            ) and ok

        warn_if_predict_has_suspicious_io(predict_path)

        # --------------------------------------------------------------
        # predict.py import / wrapper checks
        # --------------------------------------------------------------
        predict_module = load_module(predict_path, f"{team_dir.name}_predict")
        pass_check("predict.py imports successfully")

        if not hasattr(predict_module, "Model"):
            ok = fail("predict.py must define a class named Model") and ok
        else:
            pass_check("Found Model class in predict.py")

        if not ok:
            return False

        model = predict_module.Model()
        pass_check("Model() can be constructed")

        # --------------------------------------------------------------
        # Runtime delegation check:
        # predict.Model must contain and call BikeDemandModel.
        # --------------------------------------------------------------
        if not check_predict_delegates_to_bike_model(wrapper_model=model):
            ok = False

        if not ok:
            return False

        # The delegation check replaces the inner model with a sentinel.
        # Construct a fresh wrapper before checking real weights.
        model = predict_module.Model()
        pass_check("Fresh Model() can be constructed after delegation check")

    except Exception as e:
        return fail(f"Could not import/construct submission files: {e}")

    finally:
        sys.path = original_sys_path

        # Prevent one team's modules from leaking into the next team.
        sys.modules.pop("model", None)
        sys.modules.pop("predict", None)

        if old_model_module is not None:
            sys.modules["model"] = old_model_module
        if old_predict_module is not None:
            sys.modules["predict"] = old_predict_module

    # ------------------------------------------------------------------
    # Public API checks
    # ------------------------------------------------------------------
    if not hasattr(model, "load") or not callable(model.load):
        return fail("Model must implement load(weights_path)")

    pass_check("Model has load(...) method")

    if not hasattr(model, "predict") or not callable(model.predict):
        return fail("Model must implement predict(test_df)")

    pass_check("Model has predict(...) method")

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------
    try:
        model.load(str(weights_path))
        pass_check(f"{WEIGHTS_FILENAME} loads successfully")
    except Exception as e:
        return fail(
            f"Could not load {WEIGHTS_FILENAME}. "
            "Most likely, the artifact file does not match the logic in model.py.\n"
            f"Error: {e}"
        )

    # ------------------------------------------------------------------
    # Built-in synthetic smoke test
    # ------------------------------------------------------------------
    if not check_prediction_behavior(
        model=model,
        test_df=make_smoke_test_df(),
        label="built-in synthetic station-hour format",
    ):
        return False

    # ------------------------------------------------------------------
    # Training-set-derived smoke test
    # ------------------------------------------------------------------
    if train_csv.exists():
        try:
            train_like_test_df = make_test_format_from_training_set(train_csv)
            pass_check(f"Built simulated test format from {train_csv}")
        except Exception as e:
            return fail(f"Could not build simulated test format from train_set.csv: {e}")

        if not check_prediction_behavior(
            model=model,
            test_df=train_like_test_df,
            label="train_set-derived station-hour format",
        ):
            return False
    else:
        warn(
            f"Could not find {train_csv}. "
            "Skipped train_set-derived prediction smoke test."
        )

    print()
    print(f"[SUCCESS] {team_dir.name} passed all bike submission format checks.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "team_name",
        nargs="?",
        help="Name of the team folder inside submissions/. If omitted, checks all teams.",
    )
    args = parser.parse_args()

    if not SUBMISSIONS_DIR.exists():
        print(f"[FAIL] Could not find submissions folder: {SUBMISSIONS_DIR}")
        sys.exit(1)

    if args.team_name:
        team_dirs = [SUBMISSIONS_DIR / args.team_name]
    else:
        team_dirs = sorted(
            d for d in SUBMISSIONS_DIR.iterdir()
            if d.is_dir()
        )

    if not team_dirs:
        print("[FAIL] No submission folders found.")
        sys.exit(1)

    all_ok = True

    for team_dir in team_dirs:
        team_ok = check_team_submission(team_dir)
        all_ok = all_ok and team_ok

    print()
    print("=" * 70)

    if all_ok:
        print("[SUCCESS] All checked bike submissions passed.")
        sys.exit(0)
    else:
        print("[FAIL] At least one bike submission has problems.")
        sys.exit(1)


if __name__ == "__main__":
    main()