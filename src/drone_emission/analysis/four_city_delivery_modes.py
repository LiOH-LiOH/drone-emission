"""Oracle drone substitution against four incumbent delivery modes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.emission import lifecycle_emission_per_order
from drone_emission.paper_cohort import cohort_hash
from drone_emission.utils.atomic import atomic_open


ANALYSIS_VERSION = 1
DEFAULT_CHUNK_ROWS = 250_000
CITY_ORDER = ("chengdu", "guangzhou", "beijing", "shanghai")
BASELINE_MODES = (
    "pedelec_ebike",
    "fuel_moped",
    "battery_electric_car",
    "fuel_car",
)
PETROL_WTW_EF_KG_PER_MJ = 0.08901666666666667
REQUIRED_OD_COLUMNS = (
    "distance_km",
    "road_distance_m",
    "annual_sales",
    "drone_lifecycle_g",
    "ebike_lifecycle_g",
    "ebike_kappa",
    "grid_emission_factor",
)
OUTPUT_COLUMNS = (
    "region_key",
    "region_name_en",
    "baseline_mode",
    "annual_orders",
    "grid_emission_factor_kg_per_kwh",
    "ground_distance_kappa",
    "baseline_lifecycle_kg",
    "oracle_mixed_lifecycle_kg",
    "drone_replaced_annual_orders",
    "drone_replacement_rate_pct",
    "absolute_reduction_kg",
    "absolute_reduction_kt",
    "relative_reduction_pct",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ground_slope(mode: str, exp_config: dict, gef: float) -> float:
    return float(
        lifecycle_emission_per_order(
            mode,
            1.0,
            exp_config=exp_config,
            grid_ef_kg_per_kwh=gef,
            fuel_wtw_ef_kg_per_mj=PETROL_WTW_EF_KG_PER_MJ,
        )
    )


def _require_constant(values: np.ndarray, name: str, previous: float | None) -> float:
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError(f"{name} must be finite for every OD row")
    value = float(values[0]) if previous is None else previous
    if not np.allclose(values, value, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be constant within each city")
    return value


def _validate_main_experiment(row: dict, stats: pd.Series) -> None:
    expected = {
        "baseline_lifecycle_kg": "total_ebike_lifecycle",
        "oracle_mixed_lifecycle_kg": "oracle_min_lifecycle_kg",
        "drone_replacement_rate_pct": "oracle_drone_order_ratio_pct",
        "relative_reduction_pct": "oracle_min_reduction_vs_ebike_pct",
    }
    for observed_name, expected_name in expected.items():
        observed = float(row[observed_name])
        reference = float(stats[expected_name])
        if not math.isclose(observed, reference, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError(
                f"{row['region_key']} E-bike {observed_name} does not match "
                f"main experiment {expected_name}: {observed} != {reference}"
            )


def calculate_city_mode_oracle(
    od_csv: str | Path,
    stats: pd.Series,
    exp_config: dict,
    *,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate four incumbent-mode oracle scenarios for one city."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    od_csv = Path(od_csv)
    columns = set(pd.read_csv(od_csv, nrows=0).columns)
    missing = sorted(set(REQUIRED_OD_COLUMNS) - columns)
    if missing:
        raise ValueError(f"{od_csv} is missing columns: {missing}")

    totals = {
        mode: {"baseline_g": 0.0, "mixed_g": 0.0, "drone_orders": 0.0}
        for mode in BASELINE_MODES
    }
    annual_orders = 0.0
    row_count = 0
    gef = None
    kappa = None
    ground_slopes = None
    for chunk in pd.read_csv(
        od_csv,
        usecols=list(REQUIRED_OD_COLUMNS),
        chunksize=chunk_rows,
    ):
        numeric = {
            name: pd.to_numeric(chunk[name], errors="coerce").to_numpy(dtype=float)
            for name in REQUIRED_OD_COLUMNS
        }
        invalid = (
            ~np.isfinite(numeric["distance_km"])
            | (numeric["distance_km"] <= 0)
            | ~np.isfinite(numeric["road_distance_m"])
            | (numeric["road_distance_m"] <= 0)
            | ~np.isfinite(numeric["annual_sales"])
            | (numeric["annual_sales"] <= 0)
            | ~np.isfinite(numeric["drone_lifecycle_g"])
            | (numeric["drone_lifecycle_g"] < 0)
            | ~np.isfinite(numeric["ebike_lifecycle_g"])
            | (numeric["ebike_lifecycle_g"] < 0)
            | ~np.isfinite(numeric["ebike_kappa"])
            | (numeric["ebike_kappa"] <= 0)
            | ~np.isfinite(numeric["grid_emission_factor"])
            | (numeric["grid_emission_factor"] < 0)
        )
        if invalid.any():
            raise ValueError(
                f"{od_csv} contains {int(invalid.sum())} invalid oracle input rows"
            )
        gef = _require_constant(
            numeric["grid_emission_factor"], "grid_emission_factor", gef
        )
        kappa = _require_constant(numeric["ebike_kappa"], "ebike_kappa", kappa)
        if ground_slopes is None:
            ground_slopes = {
                mode: _ground_slope(mode, exp_config, gef)
                for mode in BASELINE_MODES[1:]
            }
        weights = numeric["annual_sales"]
        drone_g = numeric["drone_lifecycle_g"]
        road_km = numeric["road_distance_m"] / 1000.0 * kappa
        annual_orders += float(weights.sum())
        row_count += len(chunk)

        baselines = {"pedelec_ebike": numeric["ebike_lifecycle_g"]}
        for mode in BASELINE_MODES[1:]:
            baselines[mode] = road_km * ground_slopes[mode]
        for mode, baseline_g in baselines.items():
            use_drone = drone_g < baseline_g
            totals[mode]["baseline_g"] += float(np.sum(baseline_g * weights))
            totals[mode]["mixed_g"] += float(
                np.sum(np.where(use_drone, drone_g, baseline_g) * weights)
            )
            totals[mode]["drone_orders"] += float(weights[use_drone].sum())

    if (
        row_count == 0
        or annual_orders <= 0
        or gef is None
        or kappa is None
        or ground_slopes is None
    ):
        raise ValueError(f"{od_csv} contains no positive OD rows")

    rows = []
    for mode in BASELINE_MODES:
        baseline_kg = totals[mode]["baseline_g"] / 1000.0
        mixed_kg = totals[mode]["mixed_g"] / 1000.0
        reduction_kg = baseline_kg - mixed_kg
        rows.append(
            {
                "region_key": str(stats["region_key"]),
                "region_name_en": str(stats["region_name_en"]),
                "baseline_mode": mode,
                "annual_orders": annual_orders,
                "grid_emission_factor_kg_per_kwh": gef,
                "ground_distance_kappa": kappa,
                "baseline_lifecycle_kg": baseline_kg,
                "oracle_mixed_lifecycle_kg": mixed_kg,
                "drone_replaced_annual_orders": totals[mode]["drone_orders"],
                "drone_replacement_rate_pct": (
                    totals[mode]["drone_orders"] / annual_orders * 100.0
                ),
                "absolute_reduction_kg": reduction_kg,
                "absolute_reduction_kt": reduction_kg / 1_000_000.0,
                "relative_reduction_pct": reduction_kg / baseline_kg * 100.0,
            }
        )
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    _validate_main_experiment(
        result[result["baseline_mode"].eq("pedelec_ebike")].iloc[0].to_dict(),
        stats,
    )
    return result, {
        "region_key": str(stats["region_key"]),
        "region_name_en": str(stats["region_name_en"]),
        "od_rows": row_count,
        "invalid_od_rows": 0,
        "annual_orders": annual_orders,
        "grid_emission_factor_kg_per_kwh": gef,
        "ground_distance_kappa": kappa,
        "ground_mode_g_per_effective_road_km": ground_slopes,
        "od_input": {
            "path": f"city/{stats['region_key']}/{stats['region_key']}_od_pairs.csv",
            "size_bytes": od_csv.stat().st_size,
            "sha256": _sha256(od_csv),
        },
    }


def run_four_city_delivery_mode_analysis(
    config: dict,
    *,
    force: bool = False,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
) -> tuple[Path, Path]:
    """Write the retained four-city mode-oracle table and audit report."""
    output_dir = Path(config["work_dir"]) / "analysis" / "four_city_delivery_modes"
    csv_path = output_dir / "four_city_mode_oracle_summary.csv"
    report_path = output_dir / "four_city_mode_oracle_report.json"
    if not force and csv_path.is_file() and report_path.is_file():
        return csv_path, report_path

    stats_path = Path(config["data_dir"]) / "all_region_stats.csv"
    stats = pd.read_csv(stats_path)
    if "region_level" in stats.columns:
        stats = stats[stats["region_level"].eq("city")].copy()
    required_stats = {
        "region_key",
        "region_name_en",
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
        "oracle_drone_order_ratio_pct",
        "oracle_min_reduction_vs_ebike_pct",
    }
    missing = sorted(required_stats - set(stats.columns))
    if missing:
        raise ValueError(f"{stats_path} is missing columns: {missing}")
    selected = stats[stats["region_key"].astype(str).isin(CITY_ORDER)].copy()
    actual = set(selected["region_key"].astype(str))
    if len(selected) != len(CITY_ORDER) or actual != set(CITY_ORDER):
        raise ValueError(
            "four-city stats contract failed: "
            f"missing={sorted(set(CITY_ORDER) - actual)}, "
            f"extra={sorted(actual - set(CITY_ORDER))}"
        )
    selected = selected.set_index("region_key", drop=False)

    frames = []
    city_reports = []
    for region_key in CITY_ORDER:
        city_frame, city_report = calculate_city_mode_oracle(
            Path(config["data_dir"])
            / "city"
            / region_key
            / f"{region_key}_od_pairs.csv",
            selected.loc[region_key],
            config,
            chunk_rows=chunk_rows,
        )
        frames.append(city_frame)
        city_reports.append(city_report)
    result = pd.concat(frames, ignore_index=True)
    report = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "paper_cohort_sha256": cohort_hash(),
        "functional_unit": "annual weighted orders in each city-mode scenario",
        "city_order": list(CITY_ORDER),
        "baseline_mode_order": list(BASELINE_MODES),
        "assignment_rule": (
            "Assign an OD to drone only when stored main-experiment drone "
            "lifecycle emissions are strictly below the incumbent mode."
        ),
        "ground_distance_rule": "road_distance_m / 1000 * stored ebike_kappa",
        "energy_factors": {
            "electric_modes": "stored city grid_emission_factor",
            "fuel_modes_petrol_wtw_kg_per_mj": PETROL_WTW_EF_KG_PER_MJ,
        },
        "main_experiment_validation": (
            "E-bike baseline, oracle total, replacement rate, and relative "
            "reduction must match all_region_stats.csv."
        ),
        "cities": city_reports,
        "aggregate_input": {
            "path": "all_region_stats.csv",
            "size_bytes": stats_path.stat().st_size,
            "sha256": _sha256(stats_path),
        },
        "plotted_values": result.to_dict(orient="records"),
    }
    with atomic_open(csv_path, "w", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    with atomic_open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return csv_path, report_path
