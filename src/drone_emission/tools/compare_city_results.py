"""Compare an isolated city rerun with an authoritative existing result."""

from __future__ import annotations

import argparse
import json
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

CORE_STATS = {
    "total_merchants", "total_monthly_sales", "mean_monthly_sales",
    "building_count", "building_height_mean", "building_height_max",
    "building_height_min", "building_volume_total", "building_area_total",
    "total_store_annual_sales", "total_odpair_annual_sales", "odpair_count",
    "roadnet_node_count", "roadnet_edge_count", "roadnet_length_total",
    "avg_road_distance_m", "avg_straight_distance_m", "detour_ratio",
    "valid_path_count", "invalid_path_count", "total_ebike_lifecycle",
    "total_drone_lifecycle", "oracle_min_lifecycle_kg",
    "oracle_min_reduction_vs_ebike_pct", "oracle_drone_order_ratio_pct",
    "ebike_kappa", "province",
}


def _compare_value(name: str, expected, actual, differences: list[str]) -> None:
    if isinstance(expected, Number) and isinstance(actual, Number):
        if not np.isclose(expected, actual, rtol=1e-6, atol=1e-9, equal_nan=True):
            differences.append(f"{name}: expected {expected!r}, got {actual!r}")
    elif expected != actual:
        differences.append(f"{name}: expected {expected!r}, got {actual!r}")


def compare_city(expected_dir: Path, actual_dir: Path) -> dict:
    city = expected_dir.name
    expected_csv = expected_dir / f"{city}_od_pairs.csv"
    actual_csv = actual_dir / f"{city}_od_pairs.csv"
    expected_stats = expected_dir / "region_stats.json"
    actual_stats = actual_dir / "region_stats.json"
    for path in (expected_csv, actual_csv, expected_stats, actual_stats):
        if not path.is_file():
            raise FileNotFoundError(path)

    expected = pd.read_csv(expected_csv)
    actual = pd.read_csv(actual_csv)
    differences: list[str] = []
    if list(expected.columns) != list(actual.columns):
        differences.append(
            f"OD columns differ: expected={list(expected.columns)!r}, actual={list(actual.columns)!r}"
        )
    if len(expected) != len(actual):
        differences.append(f"OD row count differs: expected={len(expected)}, actual={len(actual)}")
    if not differences:
        for column in expected.columns:
            left = expected[column]
            right = actual[column]
            if pd.api.types.is_integer_dtype(left.dtype):
                if not left.equals(right):
                    differences.append(f"OD integer column differs: {column}")
            elif pd.api.types.is_numeric_dtype(left.dtype):
                if not np.allclose(
                    left.to_numpy(dtype=float),
                    right.to_numpy(dtype=float),
                    rtol=1e-6,
                    atol=1e-9,
                    equal_nan=True,
                ):
                    differences.append(f"OD numeric column differs: {column}")
            elif not left.fillna("").equals(right.fillna("")):
                differences.append(f"OD value column differs: {column}")

    expected_payload = json.loads(expected_stats.read_text(encoding="utf-8"))
    actual_payload = json.loads(actual_stats.read_text(encoding="utf-8"))
    expected_values = expected_payload.get("stats", {})
    actual_values = actual_payload.get("stats", {})
    missing_core = sorted(CORE_STATS - set(actual_values))
    unexpected = sorted(set(actual_values) - set(expected_values))
    if missing_core:
        differences.append(f"active core stats missing: {missing_core}")
    if unexpected:
        differences.append(f"isolated result has unknown stats: {unexpected}")
    compared_keys = sorted(set(actual_values) & set(expected_values))
    for key in compared_keys:
        _compare_value(f"stats.{key}", expected_values[key], actual_values[key], differences)

    return {
        "schema_version": 1,
        "city": city,
        "rtol": 1e-6,
        "atol": 1e-9,
        "od_rows_expected": len(expected),
        "od_rows_actual": len(actual),
        "stats_key_count_expected": len(expected_values),
        "stats_key_count_actual": len(actual_values),
        "authoritative_retired_stats_ignored": sorted(set(expected_values) - set(actual_values)),
        "ok": not differences,
        "differences": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected_dir", type=Path)
    parser.add_argument("actual_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = compare_city(args.expected_dir, args.actual_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.report)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
