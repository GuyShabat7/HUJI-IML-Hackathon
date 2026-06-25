#!/usr/bin/env python3
"""
Convert ride-level bike data into station-hour demand format.

Input ride-level columns may include:
    started_at, ended_at, start_station_id, end_station_id, usage_time_minutes,
    distance_meters, user_type, start_lat, start_lng, weather columns, city,
    POI/station metadata columns, date, weekday, weekend, holiday, holiday_name,
    working_day, hour_ts

Output 1: public_test_targets.csv
    Given to students. Contains one row per station-hour target and NO demand column.

Output 2: private_test_labels.csv
    Kept by the instructor. Contains the true station-hour demand.

Core target definition:
    demand = number of rides starting from station s during hour h

Default target-row generation:
    grid_mode = active_window
    active_buffer_hours_before = 0
    active_buffer_hours_after = 0
    min_active_window_hours = 72
    keep hours = 07:00 through 21:00
    drop hours = 00:00 through 06:00 and 22:00 through 23:00
    max_eval_hours_by_city = {"city 1": 300, "city 2": 300, "city 3": 200}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RIDE_ONLY_COLUMNS = {
    "started_at",
    "ended_at",
    "end_station_id",
    "usage_time_minutes",
    "distance_meters",
    "user_type",
}

WEATHER_TIME_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "wind_speed_10m",
]

CALENDAR_COLUMNS = [
    "date",
    "weekday",
    "weekend",
    "holiday",
    "holiday_name",
    "working_day",
]

STATION_METADATA_COLUMNS = [
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
]


# Reduced-test defaults.
# Keep 07:00 through 21:00 by default.
# Drop 00:00-06:00 and 22:00-23:00.
DEFAULT_KEEP_HOURS = tuple(range(6, 23))

# Active-window defaults chosen to reduce redundant station-hour rows.
DEFAULT_ACTIVE_BUFFER_HOURS_BEFORE = 0
DEFAULT_ACTIVE_BUFFER_HOURS_AFTER = 0
DEFAULT_MIN_ACTIVE_WINDOW_HOURS = 72

# Cap the number of unique evaluated timestamps per city.
# All station rows are kept for the selected city-hours.
DEFAULT_MAX_EVAL_HOURS_BY_CITY = {
    "city 1": 300,
    "city 2": 300,
    "city 3": 200,
}
DEFAULT_EVAL_HOUR_CAP_RANDOM_STATE = 42


def normalize_station_id(s: pd.Series) -> pd.Series:
    """
    Normalize station IDs so 31631, 31631.0, and '31631.0' become '31631'.

    Non-numeric IDs such as 'station_123' are preserved.
    """
    raw = s.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    is_int_like = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)

    out = raw.copy()
    out.loc[is_int_like] = numeric.loc[is_int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


def _first_non_null(x: pd.Series):
    non_null = x.dropna()
    if len(non_null) == 0:
        return np.nan
    return non_null.iloc[0]


def _agg_dict_for_columns(columns: Iterable[str]) -> dict[str, str | callable]:
    """
    Use median for numeric columns and first non-null value for non-numeric columns.
    """
    return {col: _first_non_null for col in columns}


def add_station_hour_keys(
    df: pd.DataFrame,
    start_time_col: str = "started_at",
    hour_col: str = "hour_ts",
    station_col: str = "start_station_id",
    city_col: str = "city",
) -> pd.DataFrame:
    """
    Adds:
        city_key
        station_key
        target_hour_start

    If hour_ts exists, it is used. Otherwise target_hour_start is derived from started_at.
    """
    out = df.copy()

    if station_col not in out.columns:
        raise ValueError(f"Missing required station column: {station_col}")

    if city_col in out.columns:
        out["city_key"] = out[city_col].astype("string").fillna("__missing_city__")
    else:
        out["city_key"] = "__all_cities__"

    out["station_key"] = normalize_station_id(out[station_col])

    if hour_col in out.columns:
        ts = pd.to_datetime(out[hour_col], errors="coerce")
    elif start_time_col in out.columns:
        ts = pd.to_datetime(out[start_time_col], errors="coerce")
    else:
        raise ValueError(
            f"Need either '{hour_col}' or '{start_time_col}' to define the target hour."
        )

    out["target_hour_start"] = ts.dt.floor("h")

    bad = int(out["target_hour_start"].isna().sum())
    if bad:
        raise ValueError(f"{bad} rows have invalid/missing timestamps.")

    return out

def make_full_city_station_hour_grid(
    keyed: pd.DataFrame,
    station_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Old behavior:
    for each city, create all city x station x hour combinations
    between the first and last observed city hour.
    """
    grids = []

    for city_key, city_stations in station_meta.groupby("city_key", dropna=False):
        city_hours = keyed.loc[
            keyed["city_key"] == city_key,
            "target_hour_start",
        ]

        min_hour = city_hours.min()
        max_hour = city_hours.max()

        if pd.isna(min_hour) or pd.isna(max_hour):
            continue

        hours = pd.date_range(min_hour, max_hour, freq="h")

        stations = city_stations[["city_key", "station_key"]].drop_duplicates()

        grid = (
            stations.assign(_join_key=1)
            .merge(
                pd.DataFrame(
                    {
                        "target_hour_start": hours,
                        "_join_key": 1,
                    }
                ),
                on="_join_key",
                how="outer",
            )
            .drop(columns="_join_key")
        )

        grids.append(grid)

    if not grids:
        return pd.DataFrame(
            columns=["city_key", "station_key", "target_hour_start"]
        )

    return pd.concat(grids, ignore_index=True)


def make_active_window_station_hour_grid(
    keyed: pd.DataFrame,
    station_meta: pd.DataFrame,
    buffer_hours_before: int = 24,
    buffer_hours_after: int = 24,
    min_window_hours: int = DEFAULT_MIN_ACTIVE_WINDOW_HOURS,
) -> pd.DataFrame:
    """
    Recommended behavior:
    for each city + station, create station-hour rows only around the
    station's observed active period.

    Example:
        first observed ride from station: 2026-04-03 10:00
        last observed ride from station:  2026-04-20 22:00
        buffer before: 24 hours
        buffer after:  24 hours

    Evaluated window:
        2026-04-02 10:00 through 2026-04-21 22:00

    This keeps zero-demand hours inside the station's plausible operational
    period, but avoids evaluating stations for long periods where they likely
    were not active.
    """
    if buffer_hours_before < 0:
        raise ValueError("buffer_hours_before must be non-negative")

    if buffer_hours_after < 0:
        raise ValueError("buffer_hours_after must be non-negative")

    if min_window_hours < 1:
        raise ValueError("min_window_hours must be at least 1")

    keyed = keyed.copy()

    keyed["target_hour_start"] = pd.to_datetime(
        keyed["target_hour_start"],
        errors="coerce",
    )

    keyed = keyed.dropna(subset=["target_hour_start"])

    # City-level hidden period bounds.
    city_bounds = (
        keyed.groupby("city_key", dropna=False)["target_hour_start"]
        .agg(city_min_hour="min", city_max_hour="max")
        .reset_index()
    )

    # Station-level observed activity windows.
    station_windows = (
        keyed.groupby(["city_key", "station_key"], dropna=False)["target_hour_start"]
        .agg(
            first_active_hour="min",
            last_active_hour="max",
            observed_ride_rows="size",
            observed_active_hours="nunique",
        )
        .reset_index()
    )

    station_windows = station_windows.merge(
        city_bounds,
        on="city_key",
        how="left",
        validate="many_to_one",
    )

    # Make sure we only create windows for stations that have metadata.
    station_windows = station_meta[["city_key", "station_key"]].drop_duplicates().merge(
        station_windows,
        on=["city_key", "station_key"],
        how="inner",
        validate="one_to_one",
    )

    rows = []

    for row in station_windows.itertuples(index=False):
        city_key = row.city_key
        station_key = row.station_key

        first_active = row.first_active_hour
        last_active = row.last_active_hour
        city_min = row.city_min_hour
        city_max = row.city_max_hour

        if pd.isna(first_active) or pd.isna(last_active):
            continue

        start = first_active - pd.Timedelta(hours=buffer_hours_before)
        end = last_active + pd.Timedelta(hours=buffer_hours_after)

        # Ensure a minimum window length, useful for stations with only one ride.
        current_hours = int((end - start).total_seconds() // 3600) + 1

        if current_hours < min_window_hours:
            missing = min_window_hours - current_hours
            extra_before = missing // 2
            extra_after = missing - extra_before

            start = start - pd.Timedelta(hours=extra_before)
            end = end + pd.Timedelta(hours=extra_after)

        # Clip to the city's actual hidden period.
        start = max(start, city_min)
        end = min(end, city_max)

        if pd.isna(start) or pd.isna(end) or start > end:
            continue

        hours = pd.date_range(start, end, freq="h")

        if len(hours) == 0:
            continue

        part = pd.DataFrame(
            {
                "city_key": city_key,
                "station_key": station_key,
                "target_hour_start": hours,
            }
        )

        rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=["city_key", "station_key", "target_hour_start"]
        )

    targets = pd.concat(rows, ignore_index=True)

    targets = targets.drop_duplicates(
        ["city_key", "station_key", "target_hour_start"]
    ).reset_index(drop=True)

    return targets


def make_observed_station_hour_grid(demand: pd.DataFrame) -> pd.DataFrame:
    """
    Observed-only behavior:
    only station-hours with at least one ride.

    This is usually not recommended for final evaluation because it removes all
    zero-demand rows.
    """
    return demand[["city_key", "station_key", "target_hour_start"]].copy()


def filter_target_hours(
    targets: pd.DataFrame,
    keep_hours: Iterable[int] | None = DEFAULT_KEEP_HOURS,
) -> pd.DataFrame:
    """
    Keep only selected hours of day.

    Default behavior:
        keep 06:00 through 23:00
        drop 00:00 through 05:00

    This is applied after the station-hour grid is created and before demand
    labels are attached.
    """
    if keep_hours is None:
        return targets

    keep_hours = set(int(h) for h in keep_hours)

    invalid = sorted(h for h in keep_hours if h < 0 or h > 23)
    if invalid:
        raise ValueError(f"Invalid keep_hours values: {invalid}")

    out = targets.copy()

    out["target_hour_start"] = pd.to_datetime(
        out["target_hour_start"],
        errors="coerce",
    )

    before = len(out)

    out = out[
        out["target_hour_start"].dt.hour.isin(keep_hours)
    ].copy()

    after = len(out)

    # print(
    #     f"[INFO] Hour filter kept {after:,}/{before:,} target rows "
    #     f"({after / max(before, 1):.1%}). "
    #     f"Kept hours: {sorted(keep_hours)}"
    # )

    return out.reset_index(drop=True)




def cap_eval_hours_per_city(
    targets: pd.DataFrame,
    max_eval_hours_by_city: dict[str, int] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce test size by limiting the number of unique evaluated hours per city.

    This samples city-level target hours and then keeps all station-hour rows
    for those selected hours. That is usually better than random row sampling,
    because all stations in a city are evaluated at the same selected timestamps.

    Example:
        max_eval_hours_by_city={"city 1": 300, "city 2": 300, "city 3": 200}

    Notes:
        - The cap is applied after the hour-of-day filter.
        - If a city has fewer unique hours than the cap, all its hours are kept.
        - City keys are matched against city_key, which is usually the original
          city string when the input has a city column.
    """
    if not max_eval_hours_by_city:
        return targets.reset_index(drop=True)

    required_cols = {"city_key", "target_hour_start"}
    missing = required_cols - set(targets.columns)
    if missing:
        raise ValueError(
            "Cannot cap evaluation hours; missing required target columns: "
            f"{sorted(missing)}"
        )

    invalid_caps = {
        city: cap
        for city, cap in max_eval_hours_by_city.items()
        if cap is None or int(cap) < 1
    }
    if invalid_caps:
        raise ValueError(
            "max_eval_hours_by_city values must be positive integers. "
            f"Invalid values: {invalid_caps}"
        )

    tmp = targets.copy()
    tmp["city_key"] = tmp["city_key"].astype("string")
    tmp["target_hour_start"] = pd.to_datetime(
        tmp["target_hour_start"],
        errors="coerce",
    )

    bad = int(tmp["target_hour_start"].isna().sum())
    if bad:
        raise ValueError(f"{bad} target rows have invalid/missing target_hour_start.")

    rng = np.random.default_rng(random_state)
    out_parts = []

    for city_key, g in tmp.groupby("city_key", dropna=False):
        city_key_str = str(city_key)
        max_hours = max_eval_hours_by_city.get(city_key_str)

        if max_hours is None:
            out_parts.append(g)
            continue

        unique_hours = np.array(
            sorted(g["target_hour_start"].drop_duplicates().to_numpy())
        )

        before_hours = len(unique_hours)

        if before_hours > int(max_hours):
            selected_hours = rng.choice(
                unique_hours,
                size=int(max_hours),
                replace=False,
            )
            selected_hours = pd.to_datetime(selected_hours)
            g = g[g["target_hour_start"].isin(selected_hours)].copy()

        after_hours = g["target_hour_start"].nunique()


        out_parts.append(g)

    if not out_parts:
        return tmp.iloc[0:0].reset_index(drop=True)

    out = pd.concat(out_parts, ignore_index=True)
    return out.reset_index(drop=True)


def format_max_eval_hours_by_city(value: dict[str, int] | None) -> str | None:
    """Format a per-city hour cap dict for argparse defaults/help output."""
    if not value:
        return None
    return ",".join(f"{city}={hours}" for city, hours in value.items())


def parse_max_eval_hours_by_city(value: str | None) -> dict[str, int] | None:
    """
    Parse CLI values like:
        "city 1=300,city 2=300,city 3=200"
        "city 1:300, city 2:300, city 3:200"

    Returns None when no value is supplied.
    """
    if value is None:
        return None

    value = value.strip()
    if value == "":
        return None

    out: dict[str, int] = {}

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" in item:
            city, cap = item.split("=", 1)
        elif ":" in item:
            city, cap = item.split(":", 1)
        else:
            raise ValueError(
                "Invalid --max_eval_hours_by_city item. Use city=hours, e.g. "
                f"'city 1=300'. Got: {item!r}"
            )

        city = city.strip()
        cap = cap.strip()

        if city == "":
            raise ValueError(
                f"Invalid --max_eval_hours_by_city item with empty city: {item!r}"
            )

        try:
            cap_int = int(cap)
        except ValueError as exc:
            raise ValueError(
                f"Invalid max-hours value for city {city!r}: {cap!r}"
            ) from exc

        if cap_int < 1:
            raise ValueError(
                f"Max-hours value for city {city!r} must be >= 1. Got: {cap_int}"
            )

        out[city] = cap_int

    return out or None

def make_station_hour_test_format(
    rides: pd.DataFrame,
    grid_mode: str = "active_window",
    active_buffer_hours_before: int = DEFAULT_ACTIVE_BUFFER_HOURS_BEFORE,
    active_buffer_hours_after: int = DEFAULT_ACTIVE_BUFFER_HOURS_AFTER,
    min_active_window_hours: int = DEFAULT_MIN_ACTIVE_WINDOW_HOURS,
    keep_hours: Iterable[int] | None = DEFAULT_KEEP_HOURS,
    max_eval_hours_by_city: dict[str, int] | None = DEFAULT_MAX_EVAL_HOURS_BY_CITY,
    eval_hour_cap_random_state: int = DEFAULT_EVAL_HOUR_CAP_RANDOM_STATE,
    start_time_col: str = "started_at",
    hour_col: str = "hour_ts",
    station_col: str = "start_station_id",
    city_col: str = "city",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert ride-level test data into:

    1. public_targets:
        File to give students.
        Includes station-hour rows and features, but NOT demand.

    2. private_labels:
        File to keep private for scoring.
        Includes station-hour rows and true demand.

    grid_mode:
        "active_window":
            Recommended.
            Creates station-hour rows only around each station's observed active
            period, with configurable buffer before/after.

        "full":
            Creates all city x station x hour combinations between the first
            and last observed test hour for that city.

        "observed_only":
            Only station-hours with at least one ride.
            Not recommended for evaluation because it removes zero-demand rows.

    active_buffer_hours_before:
        Number of hours to include before the station's first observed ride.

    active_buffer_hours_after:
        Number of hours to include after the station's last observed ride.

    min_active_window_hours:
        Minimum number of hours to include for a station.
        Useful for stations with very few rides.

    max_eval_hours_by_city:
        Optional cap on unique evaluated target hours per city, applied after
        grid creation and hour-of-day filtering. Example:
        {"city 1": 300, "city 2": 300, "city 3": 200}.

    eval_hour_cap_random_state:
        Random seed used when sampling unique target hours per city.
    """
    keyed = add_station_hour_keys(
        rides,
        start_time_col=start_time_col,
        hour_col=hour_col,
        station_col=station_col,
        city_col=city_col,
    )

    # Count rides per city-station-hour.
    demand = (
        keyed
        .groupby(["city_key", "station_key", "target_hour_start"], dropna=False)
        .size()
        .reset_index(name="demand")
    )

    # Station metadata: one row per city-station.
    station_feature_cols = [
        c for c in STATION_METADATA_COLUMNS
        if c in keyed.columns
    ]

    station_base_cols = ["city_key", "station_key"]
    if city_col in keyed.columns:
        station_base_cols.append(city_col)
    if station_col in keyed.columns:
        station_base_cols.append(station_col)

    station_meta = (
        keyed[station_base_cols + station_feature_cols]
        .groupby(["city_key", "station_key"], dropna=False)
        .agg(_agg_dict_for_columns([c for c in station_base_cols + station_feature_cols if c not in ["city_key", "station_key"]]))
        .reset_index()
    )

    # City-hour metadata: weather/calendar features are naturally city-hour-level.
    city_hour_feature_cols = [
        c for c in WEATHER_TIME_COLUMNS + CALENDAR_COLUMNS
        if c in keyed.columns
    ]

    city_hour_meta = (
        keyed[["city_key", "target_hour_start"] + city_hour_feature_cols]
        .groupby(["city_key", "target_hour_start"], dropna=False)
        .agg(_agg_dict_for_columns(city_hour_feature_cols))
        .reset_index()
    )

    grid_mode = grid_mode.lower().strip()

    if grid_mode == "full":
        targets = make_full_city_station_hour_grid(
            keyed=keyed,
            station_meta=station_meta,
        )

    elif grid_mode == "active_window":
        targets = make_active_window_station_hour_grid(
            keyed=keyed,
            station_meta=station_meta,
            buffer_hours_before=active_buffer_hours_before,
            buffer_hours_after=active_buffer_hours_after,
            min_window_hours=min_active_window_hours,
        )

    elif grid_mode == "observed_only":
        targets = make_observed_station_hour_grid(demand)

    else:
        raise ValueError(
            "grid_mode must be one of: 'active_window', 'full', 'observed_only'. "
            f"Got: {grid_mode!r}"
        )

    # Default: remove only deep-night hours 00:00 through 05:00.
    targets = filter_target_hours(targets, keep_hours=keep_hours)

    # Optional: reduce the number of unique evaluated hours per city while
    # preserving all station rows for the selected city-hours.
    targets = cap_eval_hours_per_city(
        targets,
        max_eval_hours_by_city=max_eval_hours_by_city,
        random_state=eval_hour_cap_random_state,
    )

    # Attach metadata.
    targets = targets.merge(
        station_meta,
        on=["city_key", "station_key"],
        how="left",
        validate="many_to_one",
    )

    targets = targets.merge(
        city_hour_meta,
        on=["city_key", "target_hour_start"],
        how="left",
        validate="many_to_one",
    )

    # Attach demand; missing station-hours are true zero-demand rows.
    labels = targets.merge(
        demand,
        on=["city_key", "station_key", "target_hour_start"],
        how="left",
        validate="one_to_one",
    )
    labels["demand"] = labels["demand"].fillna(0).astype(int)

    # Stable id: same id in public targets and private labels.
    labels = labels.sort_values(["city_key", "station_key", "target_hour_start"]).reset_index(drop=True)
    labels.insert(0, "id", np.arange(len(labels)))

    # Nice feature columns for model input.
    labels["hour_ts"] = labels["target_hour_start"]
    labels["hour"] = labels["target_hour_start"].dt.hour
    labels["weekday"] = labels["target_hour_start"].dt.weekday
    labels["date"] = labels["target_hour_start"].dt.date.astype(str)

    if "weekend" not in labels.columns:
        labels["weekend"] = labels["weekday"].isin([5, 6]).astype(int)

    # Public targets: remove private label and internal keys.
    public_targets = labels.drop(columns=["demand"], errors="ignore").copy()

    # Private labels: enough to grade + useful metadata.
    label_cols = [
        "id",
        city_col if city_col in labels.columns else None,
        station_col if station_col in labels.columns else None,
        "target_hour_start",
        "demand",
    ]
    label_cols = [c for c in label_cols if c is not None and c in labels.columns]

    private_labels = labels[label_cols].copy()

    # Format datetime for CSV readability.
    for df in [public_targets, private_labels]:
        if "target_hour_start" in df.columns:
            df["target_hour_start"] = pd.to_datetime(df["target_hour_start"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        if "hour_ts" in df.columns:
            df["hour_ts"] = pd.to_datetime(df["hour_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Remove internal helper keys from public file unless you want them for debugging.
    public_targets = public_targets.drop(columns=["city_key", "station_key"], errors="ignore")

    return public_targets, private_labels


def write_station_hour_test_files(
    input_csv: str | Path,
    public_targets_csv: str | Path = "dataset/public_test_targets.csv",
    private_labels_csv: str | Path = "dataset/private_test_labels.csv",
    grid_mode: str = "active_window",
    active_buffer_hours_before: int = DEFAULT_ACTIVE_BUFFER_HOURS_BEFORE,
    active_buffer_hours_after: int = DEFAULT_ACTIVE_BUFFER_HOURS_AFTER,
    min_active_window_hours: int = DEFAULT_MIN_ACTIVE_WINDOW_HOURS,
    keep_hours: Iterable[int] | None = DEFAULT_KEEP_HOURS,
    max_eval_hours_by_city: dict[str, int] | None = DEFAULT_MAX_EVAL_HOURS_BY_CITY,
    eval_hour_cap_random_state: int = DEFAULT_EVAL_HOUR_CAP_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: read raw rides CSV, write the two output CSV files.
    """
    rides = pd.read_csv(input_csv, low_memory=False)

    public_targets, private_labels = make_station_hour_test_format(
        rides,
        grid_mode=grid_mode,
        active_buffer_hours_before=active_buffer_hours_before,
        active_buffer_hours_after=active_buffer_hours_after,
        min_active_window_hours=min_active_window_hours,
        keep_hours=keep_hours,
        max_eval_hours_by_city=max_eval_hours_by_city,
        eval_hour_cap_random_state=eval_hour_cap_random_state,
    )

    public_targets_csv = Path(public_targets_csv)
    private_labels_csv = Path(private_labels_csv)

    public_targets_csv.parent.mkdir(parents=True, exist_ok=True)
    private_labels_csv.parent.mkdir(parents=True, exist_ok=True)

    public_targets.to_csv(public_targets_csv, index=False)
    private_labels.to_csv(private_labels_csv, index=False)

    return public_targets, private_labels



def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument(
        "--public_targets_csv",
        default=Path("dataset/public_test_targets.csv"),
        type=Path,
    )
    parser.add_argument(
        "--private_labels_csv",
        default=Path("dataset/private_test_labels.csv"),
        type=Path,
    )

    parser.add_argument(
        "--grid_mode",
        choices=["active_window", "full", "observed_only"],
        default="active_window",
        help=(
            "Target-row generation mode. "
            "'active_window' is recommended; "
            "'full' is old city x station x hour behavior; "
            "'observed_only' removes zero-demand rows and is not recommended."
        ),
    )

    parser.add_argument(
        "--active_buffer_hours_before",
        type=int,
        default=DEFAULT_ACTIVE_BUFFER_HOURS_BEFORE,
        help=(
            "For grid_mode=active_window, include this many hours before each "
            "station's first observed active hour."
        ),
    )

    parser.add_argument(
        "--active_buffer_hours_after",
        type=int,
        default=DEFAULT_ACTIVE_BUFFER_HOURS_AFTER,
        help=(
            "For grid_mode=active_window, include this many hours after each "
            "station's last observed active hour."
        ),
    )

    parser.add_argument(
        "--min_active_window_hours",
        type=int,
        default=DEFAULT_MIN_ACTIVE_WINDOW_HOURS,
        help=(
            "For grid_mode=active_window, minimum number of station-hours to "
            "evaluate per station. Default is 72, i.e. three days."
        ),
    )

    parser.add_argument(
        "--max_eval_hours_by_city",
        default=format_max_eval_hours_by_city(DEFAULT_MAX_EVAL_HOURS_BY_CITY),
        type=str,
        help=(
            "Optional per-city cap on unique evaluated target hours, e.g. "
            "'city 1=300,city 2=300,city 3=200'. "
            "Applied after grid generation and the hour-of-day filter. "
            "Rows are kept for all stations at each selected city-hour."
        ),
    )

    parser.add_argument(
        "--eval_hour_cap_random_state",
        default=DEFAULT_EVAL_HOUR_CAP_RANDOM_STATE,
        type=int,
        help="Random seed used when sampling unique target hours per city.",
    )

    parser.add_argument(
        "--keep_all_hours",
        action="store_true",
        help=(
            "If set, keep all 24 hours. By default, the script drops only "
            "00:00 through 06:00 and 22:00 through 23:00; keeps 07:00 through 21:00."
        ),
    )

    parser.add_argument(
        "--keep_hours",
        default=None,
        type=str,
        help=(
            "Optional comma-separated list of hours to keep, e.g. "
            "'7,8,9,10,11,12,13,14,15,16,17,18,19,20,21'. "
            "Overrides the default 07:00-21:00 filter."
        ),
    )

    # Backward-compatible flag.
    parser.add_argument(
        "--observed_only",
        action="store_true",
        help=(
            "Deprecated shortcut for --grid_mode observed_only. "
            "Only includes station-hours where at least one ride occurred."
        ),
    )

    # Backward-compatible flag.
    parser.add_argument(
        "--full_grid",
        action="store_true",
        help=(
            "Shortcut for --grid_mode full. "
            "Creates the old full city x station x hour grid."
        ),
    )

    args = parser.parse_args()

    grid_mode = args.grid_mode

    if args.observed_only:
        grid_mode = "observed_only"

    if args.full_grid:
        grid_mode = "full"

    if args.keep_all_hours:
        keep_hours = None
    elif args.keep_hours is not None:
        keep_hours = tuple(
            int(x.strip())
            for x in args.keep_hours.split(",")
            if x.strip() != ""
        )

        invalid = sorted(h for h in keep_hours if h < 0 or h > 23)
        if invalid:
            raise ValueError(f"Invalid values in --keep_hours: {invalid}")
    else:
        keep_hours = DEFAULT_KEEP_HOURS

    max_eval_hours_by_city = parse_max_eval_hours_by_city(
        args.max_eval_hours_by_city
    )

    public_targets, private_labels = write_station_hour_test_files(
        input_csv=args.input_csv,
        public_targets_csv=args.public_targets_csv,
        private_labels_csv=args.private_labels_csv,
        grid_mode=grid_mode,
        active_buffer_hours_before=args.active_buffer_hours_before,
        active_buffer_hours_after=args.active_buffer_hours_after,
        min_active_window_hours=args.min_active_window_hours,
        keep_hours=keep_hours,
        max_eval_hours_by_city=max_eval_hours_by_city,
        eval_hour_cap_random_state=args.eval_hour_cap_random_state,
    )

    # if keep_hours is None:
    #     print("Hour filter: keeping all 24 hours")
    # else:
    #     print(f"Hour filter: keeping hours {list(keep_hours)}")
    #     print("Dropped hours:", sorted(set(range(24)) - set(keep_hours)))
    #
    # if max_eval_hours_by_city:
    #     print(f"Max evaluation hours by city: {max_eval_hours_by_city}")
    #     print(f"Evaluation-hour cap random state: {args.eval_hour_cap_random_state}")
    # else:
    #     print("Max evaluation hours by city: no cap")
    # print()
    #
    # if "city" in private_labels.columns:
    #     print("Rows by city:")
    #     print(private_labels["city"].value_counts().to_string())
    #     print()
    #
    if "target_hour_start" in private_labels.columns:
        ts = pd.to_datetime(private_labels["target_hour_start"], errors="coerce")
    #     print("Evaluation time range:")
    #     print(f"  min: {ts.min()}")
    #     print(f"  max: {ts.max()}")
    #     print(f"  unique hours: {ts.nunique()}")
    #     print()
    #
    #     hour_counts = ts.dt.hour.value_counts().sort_index()
    #     print("Rows by hour:")
    #     print(hour_counts.to_string())
    #     print()

    # print("Demand summary:")
    # print(private_labels["demand"].describe().to_string())
    # print()

    # print("Private labels preview:")
    # print(private_labels.head(10).to_string(index=False))

if __name__ == "__main__":
    main()