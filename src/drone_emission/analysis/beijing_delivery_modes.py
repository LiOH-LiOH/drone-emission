"""Retained Beijing lifecycle totals for single-mode delivery scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.emission import DELIVERY_MODES, lifecycle_emission_per_order
from drone_emission.utils.atomic import atomic_open


ANALYSIS_VERSION = 1
DEFAULT_CHUNK_ROWS = 250_000
BEIJING_GRID_EF_KG_PER_KWH = 0.558
# DESNZ 2026 mineral petrol, Net CV: direct + well-to-tank emissions.
PETROL_WTW_EF_KG_PER_MJ = 0.08901666666666667
REQUIRED_OD_COLUMNS = ("distance_km", "road_distance_m", "annual_sales")
OUTPUT_COLUMNS = (
    "region_key",
    "mode",
    "distance_basis",
    "annual_orders",
    "weighted_mean_distance_km",
    "weighted_mean_g_per_order",
    "annual_lifecycle_kg",
    "annual_lifecycle_kt",
)


def _affine_coefficients(mode: str, exp_config: dict) -> tuple[float, float]:
    """Return slope and intercept for the existing distance-only model."""
    kwargs = {
        "exp_config": exp_config,
        "grid_ef_kg_per_kwh": BEIJING_GRID_EF_KG_PER_KWH,
        "fuel_wtw_ef_kg_per_mj": PETROL_WTW_EF_KG_PER_MJ,
    }
    intercept = lifecycle_emission_per_order(mode, 0.0, **kwargs)
    at_one_km = lifecycle_emission_per_order(mode, 1.0, **kwargs)
    return float(at_one_km - intercept), float(intercept)


def calculate_beijing_delivery_mode_totals(
    od_csv: str | Path,
    exp_config: dict,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate annual lifecycle totals from a common valid Beijing OD set."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    od_csv = Path(od_csv)
    header = set(pd.read_csv(od_csv, nrows=0).columns)
    missing = sorted(set(REQUIRED_OD_COLUMNS) - header)
    if missing:
        raise ValueError(f"{od_csv} is missing columns: {missing}")
    od_stat = od_csv.stat()

    input_rows = 0
    valid_rows = 0
    excluded_rows = 0
    annual_orders = 0.0
    excluded_positive_orders = 0.0
    weighted_air_distance_km = 0.0
    weighted_road_distance_km = 0.0

    for chunk in pd.read_csv(
        od_csv,
        usecols=list(REQUIRED_OD_COLUMNS),
        chunksize=chunk_rows,
    ):
        input_rows += len(chunk)
        air_km = pd.to_numeric(chunk["distance_km"], errors="coerce").to_numpy(
            dtype=float
        )
        road_km = (
            pd.to_numeric(chunk["road_distance_m"], errors="coerce").to_numpy(
                dtype=float
            )
            / 1000.0
        )
        weights = pd.to_numeric(chunk["annual_sales"], errors="coerce").to_numpy(
            dtype=float
        )
        valid = (
            np.isfinite(air_km)
            & np.isfinite(road_km)
            & np.isfinite(weights)
            & (air_km > 0)
            & (road_km > 0)
            & (weights > 0)
        )
        valid_rows += int(valid.sum())
        excluded_rows += int((~valid).sum())
        positive_weight = np.isfinite(weights) & (weights > 0)
        excluded_positive_orders += float(weights[positive_weight & ~valid].sum())
        if not valid.any():
            continue
        selected_weights = weights[valid]
        annual_orders += float(selected_weights.sum())
        weighted_air_distance_km += float(
            np.sum(air_km[valid] * selected_weights, dtype=np.float64)
        )
        weighted_road_distance_km += float(
            np.sum(road_km[valid] * selected_weights, dtype=np.float64)
        )

    if valid_rows == 0 or annual_orders <= 0:
        raise ValueError(f"{od_csv} contains no common valid weighted OD rows")

    rows = []
    coefficients = {}
    for mode in DELIVERY_MODES:
        distance_basis = "straight" if mode == "drone" else "road"
        weighted_distance = (
            weighted_air_distance_km
            if distance_basis == "straight"
            else weighted_road_distance_km
        )
        slope, intercept = _affine_coefficients(mode, exp_config)
        annual_lifecycle_g = slope * weighted_distance + intercept * annual_orders
        coefficients[mode] = {
            "slope_g_per_order_per_one_way_km": slope,
            "intercept_g_per_order": intercept,
        }
        rows.append(
            {
                "region_key": "beijing",
                "mode": mode,
                "distance_basis": distance_basis,
                "annual_orders": annual_orders,
                "weighted_mean_distance_km": weighted_distance / annual_orders,
                "weighted_mean_g_per_order": annual_lifecycle_g / annual_orders,
                "annual_lifecycle_kg": annual_lifecycle_g / 1000.0,
                "annual_lifecycle_kt": annual_lifecycle_g / 1_000_000_000.0,
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    report = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "region_key": "beijing",
        "functional_unit": "all common-valid Beijing OD annual orders served by one mode",
        "calculation_method": "distance-affine coefficients applied to weighted OD sums",
        "orders_per_trip": 1,
        "od_filter": {
            "required_columns": list(REQUIRED_OD_COLUMNS),
            "rule": "finite positive straight distance, road distance, and annual sales",
            "input_rows": input_rows,
            "valid_rows": valid_rows,
            "excluded_rows": excluded_rows,
            "annual_orders": annual_orders,
            "excluded_positive_annual_orders": excluded_positive_orders,
        },
        "distance": {
            "drone": "straight one-way distance_km",
            "ground_modes": "road one-way road_distance_m / 1000",
            "weighted_mean_air_km": weighted_air_distance_km / annual_orders,
            "weighted_mean_road_km": weighted_road_distance_km / annual_orders,
            "ebike_kappa": 1.0,
        },
        "scenario": {
            "grid_ef_kg_per_kwh": BEIJING_GRID_EF_KG_PER_KWH,
            "fuel_wtw_ef_kg_per_mj": PETROL_WTW_EF_KG_PER_MJ,
        },
        "input_signature": {
            "source": str(od_csv),
            "size_bytes": od_stat.st_size,
            "mtime_ns": od_stat.st_mtime_ns,
        },
        "coefficients": coefficients,
        "boundary": {
            "walking_human_metabolism_included": False,
            "shared_operations_and_rebalancing_included": False,
            "road_and_charging_infrastructure_included": False,
        },
    }
    return result, report


def run_beijing_delivery_mode_analysis(
    config: dict,
    *,
    force: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[Path, Path]:
    """Write retained source data and audit metadata for the Beijing figure."""
    output_dir = Path(config["work_dir"]) / "analysis" / "beijing_delivery_modes"
    csv_path = output_dir / "beijing_all_mode_lifecycle.csv"
    report_path = output_dir / "beijing_all_mode_lifecycle_report.json"
    if not force and csv_path.is_file() and report_path.is_file():
        return csv_path, report_path

    od_csv = Path(config["data_dir"]) / "city" / "beijing" / "beijing_od_pairs.csv"
    frame, report = calculate_beijing_delivery_mode_totals(
        od_csv,
        config,
        chunk_rows=chunk_rows,
    )
    report["input_signature"]["source"] = "city/beijing/beijing_od_pairs.csv"
    with atomic_open(csv_path, "w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    with atomic_open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return csv_path, report_path
