"""Prepare cohort-wide pathway scenario data outside formal renderers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.emission.cal_single_delivery import (
    create_drone_params,
    create_ebike_params,
)
from drone_emission.paper_cohort import cohort_hash, locked_region_keys
from drone_emission.sensitivity.vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)
from drone_emission.utils.grid_emission_factor import fetch_factor


PATHWAY_ANALYSIS_VERSION = 1
PATHWAY_CURVE_N_POINTS = 21
PATHWAY_CURVE_MAX_ROWS_PER_CITY = 1500
PATHWAY_CURVE_RANDOM_STATE = 42
PATHWAY_KAPPA_END = 0.65
OD_COLUMNS = ("road_distance_m", "distance_km", "annual_sales")


def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _analysis_signature(config: dict, city_rows: pd.DataFrame) -> dict:
    data_root = Path(config["data_dir"])
    emission = config.get("emission_params", {})
    return {
        "analysis_version": PATHWAY_ANALYSIS_VERSION,
        "paper_cohort_sha256": cohort_hash(),
        "stats": _file_signature(data_root / "all_region_stats.csv"),
        "named": _file_signature(data_root / "all_sensitivity_named.csv"),
        "gef": _file_signature(Path(config["gef_path"])),
        "emission_params": emission,
        "od_files": [
            _file_signature(
                data_root / "city" / key / f"{key}_od_pairs.csv"
            )
            for key in city_rows["region_key"].astype(str)
        ],
        "curve": {
            "n_points": PATHWAY_CURVE_N_POINTS,
            "max_rows_per_city": PATHWAY_CURVE_MAX_ROWS_PER_CITY,
            "random_state": PATHWAY_CURVE_RANDOM_STATE,
            "kappa_end": PATHWAY_KAPPA_END,
        },
    }


def _named_baseline_kappa(named: pd.DataFrame) -> float:
    kappas = np.array(sorted(named["kappa"].dropna().unique()), dtype=float)
    if kappas.size == 0:
        return 1.0
    non_comparison = kappas[~np.isclose(kappas, PATHWAY_KAPPA_END)]
    return float(non_comparison.max() if non_comparison.size else kappas.max())


def _baseline_kappa(named: pd.DataFrame, stats: pd.DataFrame) -> float:
    named_kappas = np.array(sorted(named["kappa"].dropna().unique()), dtype=float)
    if "ebike_kappa" in stats.columns:
        values = pd.to_numeric(stats["ebike_kappa"], errors="coerce").dropna()
        if not values.empty:
            candidate = float(values.median())
            if named_kappas.size and np.any(np.isclose(named_kappas, candidate)):
                return candidate
    return _named_baseline_kappa(named)


def _scenario_definitions(baseline_kappa: float) -> list[tuple]:
    return [
        ("a", "Today", baseline_kappa, 1.0, "optimal_mixed"),
        ("b", "Halved GEF", baseline_kappa, 0.5, "optimal_mixed_2030"),
        ("c", "Operator bundling", PATHWAY_KAPPA_END, 1.0, "optimal_mixed"),
        (
            "d",
            "Halved GEF + bundling",
            PATHWAY_KAPPA_END,
            0.5,
            "optimal_mixed_2030",
        ),
    ]


def _valid_od(od: pd.DataFrame) -> pd.DataFrame:
    missing = set(OD_COLUMNS) - set(od.columns)
    if missing:
        raise KeyError(f"missing OD columns {sorted(missing)}")
    valid = od.loc[:, list(OD_COLUMNS)].copy()
    for column in OD_COLUMNS:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid[
        (valid["road_distance_m"] >= 0)
        & (valid["distance_km"] > 0)
        & (valid["annual_sales"] > 0)
    ].dropna()
    if valid.empty:
        raise ValueError("no valid OD rows with positive annual sales")
    return valid


def _sample_od(od: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, float]:
    total_sales = float(od["annual_sales"].sum())
    sample = od
    if len(sample) > PATHWAY_CURVE_MAX_ROWS_PER_CITY:
        sample = sample.sample(
            n=PATHWAY_CURVE_MAX_ROWS_PER_CITY,
            random_state=random_state,
        )
    sample_sales = float(sample["annual_sales"].sum())
    return sample, total_sales / sample_sales


def _evaluate_oracle(
    od: pd.DataFrame,
    ebike_params,
    drone_params,
    province_gef: float,
    *,
    kappa: float,
    gef_mult: float,
    baseline_kappa: float,
    rider_weight: float,
    payload_kg: float,
) -> dict:
    road_km = od["road_distance_m"].to_numpy(dtype=float) / 1000.0
    air_km = od["distance_km"].to_numpy(dtype=float)
    annual_sales = od["annual_sales"].to_numpy(dtype=float)
    ebike = replace(ebike_params, e_use=province_gef)
    drone = replace(drone_params, e_use=province_gef)
    ebike_g = compute_ebike_per_order_g(
        road_km,
        ebike,
        EbikeScenario(
            kappa=kappa,
            gef_mult=gef_mult,
            payload_kg=payload_kg,
            rider_weight=rider_weight,
        ),
    )
    drone_g = compute_drone_per_order_g(
        air_km,
        drone,
        DroneScenario(gef_mult=gef_mult, payload_kg=payload_kg),
    )
    anchor_g = compute_ebike_per_order_g(
        road_km,
        ebike,
        EbikeScenario(
            kappa=baseline_kappa,
            gef_mult=1.0,
            payload_kg=payload_kg,
            rider_weight=rider_weight,
        ),
    )
    use_drone = drone_g < ebike_g
    oracle_kg = float(
        np.sum(np.where(use_drone, drone_g, ebike_g) * annual_sales) / 1000.0
    )
    anchor_kg = float(np.sum(anchor_g * annual_sales) / 1000.0)
    ebike_kg = float(np.sum(ebike_g * annual_sales) / 1000.0)
    total_orders = float(annual_sales.sum())
    return {
        "annual_lifecycle_kg": oracle_kg,
        "anchor_annual_kg": anchor_kg,
        "scenario_ebike_annual_kg": ebike_kg,
        "drone_ratio_pct": float(annual_sales[use_drone].sum())
        / total_orders
        * 100.0,
    }


def _evaluate_curve_point(
    od: pd.DataFrame,
    scale: float,
    ebike_params,
    drone_params,
    province_gef: float,
    *,
    kappa: float,
    gef_mult: float,
    baseline_kappa: float,
    rider_weight: float,
    payload_kg: float,
) -> dict:
    metrics = _evaluate_oracle(
        od,
        ebike_params,
        drone_params,
        province_gef,
        kappa=kappa,
        gef_mult=gef_mult,
        baseline_kappa=baseline_kappa,
        rider_weight=rider_weight,
        payload_kg=payload_kg,
    )
    return {
        "saved_kg": (
            metrics["anchor_annual_kg"] - metrics["annual_lifecycle_kg"]
        )
        * scale,
        "drone_share_pct": metrics["drone_ratio_pct"],
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_pathway_analysis(config: dict, *, force: bool = False) -> tuple[Path, Path, Path]:
    """Build exact oracle and sampled pathway tables for the locked cohort."""
    data_root = Path(config["data_dir"])
    scenario_path = data_root / "all_pathway_scenario_city.csv"
    curve_path = data_root / "all_pathway_curve_points.csv"
    report_path = data_root / "all_pathway_scenarios_report.json"
    stats = pd.read_csv(data_root / "all_region_stats.csv")
    named = pd.read_csv(data_root / "all_sensitivity_named.csv")
    city_rows = (
        stats[stats["region_level"].eq("city")].copy()
        if "region_level" in stats.columns
        else stats.copy()
    )
    expected = set(locked_region_keys())
    actual = set(city_rows["region_key"].astype(str))
    if len(city_rows) != len(expected) or actual != expected:
        raise RuntimeError(
            "pathway analysis does not match the locked cohort: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    signature = _analysis_signature(config, city_rows)
    if not force and scenario_path.is_file() and curve_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("signature") == signature:
            return scenario_path, curve_path, report_path

    baseline_kappa = _baseline_kappa(named, city_rows)
    panels = _scenario_definitions(baseline_kappa)
    kappa_values = np.linspace(
        baseline_kappa,
        PATHWAY_KAPPA_END,
        PATHWAY_CURVE_N_POINTS,
    )
    gef_values = np.linspace(1.0, 0.5, PATHWAY_CURVE_N_POINTS)
    curve_saved = {
        "kappa": np.zeros(len(kappa_values), dtype=float),
        "gef": np.zeros(len(gef_values), dtype=float),
    }
    curve_shares = {
        "kappa": [[] for _ in kappa_values],
        "gef": [[] for _ in gef_values],
    }
    ebike_params = create_ebike_params(config)
    drone_params = create_drone_params(config)
    emission = config.get("emission_params", {})
    rider_weight = float(emission.get("rider_weight", 70.0))
    payload_kg = float(emission.get("payload_kg", 1.0))
    rows: list[dict] = []

    for source_index, city in city_rows.iterrows():
        region_key = str(city["region_key"])
        od_path = data_root / "city" / region_key / f"{region_key}_od_pairs.csv"
        try:
            od = _valid_od(pd.read_csv(od_path, usecols=list(OD_COLUMNS)))
            province_gef = float(fetch_factor(city["province"], config["gef_path"]))
            for label, _title, kappa, gef_mult, _strategy in panels:
                metrics = _evaluate_oracle(
                    od,
                    ebike_params,
                    drone_params,
                    province_gef,
                    kappa=kappa,
                    gef_mult=gef_mult,
                    baseline_kappa=baseline_kappa,
                    rider_weight=rider_weight,
                    payload_kg=payload_kg,
                )
                rows.append(
                    {
                        "panel_label": label,
                        "region_key": region_key,
                        "province": city["province"],
                        "kappa": kappa,
                        "strategy": "oracle_min_mixed",
                        "gef_mult": gef_mult,
                        "threshold_km": np.nan,
                        "drone_ratio_pct": metrics["drone_ratio_pct"],
                        "annual_lifecycle_kg": metrics["annual_lifecycle_kg"],
                        "anchor_annual_kg": metrics["anchor_annual_kg"],
                        "scenario_ebike_annual_kg": metrics[
                            "scenario_ebike_annual_kg"
                        ],
                        "reduction_vs_anchor_ebike_pct": (
                            1.0
                            - metrics["annual_lifecycle_kg"]
                            / metrics["anchor_annual_kg"]
                        )
                        * 100.0,
                    }
                )

            sample, scale = _sample_od(
                od,
                PATHWAY_CURVE_RANDOM_STATE + int(source_index),
            )
            for index, value in enumerate(kappa_values):
                point = _evaluate_curve_point(
                    sample,
                    scale,
                    ebike_params,
                    drone_params,
                    province_gef,
                    kappa=float(value),
                    gef_mult=1.0,
                    baseline_kappa=baseline_kappa,
                    rider_weight=rider_weight,
                    payload_kg=payload_kg,
                )
                curve_saved["kappa"][index] += point["saved_kg"]
                curve_shares["kappa"][index].append(point["drone_share_pct"])
            for index, value in enumerate(gef_values):
                point = _evaluate_curve_point(
                    sample,
                    scale,
                    ebike_params,
                    drone_params,
                    province_gef,
                    kappa=baseline_kappa,
                    gef_mult=float(value),
                    baseline_kappa=baseline_kappa,
                    rider_weight=rider_weight,
                    payload_kg=payload_kg,
                )
                curve_saved["gef"][index] += point["saved_kg"]
                curve_shares["gef"][index].append(point["drone_share_pct"])
        except Exception as exc:
            raise RuntimeError(
                f"pathway scenario analysis failed for {region_key}: {exc}"
            ) from exc

    curve_rows = []
    for pathway, values in (("kappa", kappa_values), ("gef", gef_values)):
        for index, value in enumerate(values):
            curve_rows.append(
                {
                    "pathway": pathway,
                    "x": float(value),
                    "saved_kt": float(curve_saved[pathway][index]) / 1e6,
                    "drone_share_pct": float(
                        np.nanmedian(curve_shares[pathway][index])
                    ),
                }
            )

    _atomic_csv(pd.DataFrame(rows), scenario_path)
    _atomic_csv(pd.DataFrame(curve_rows), curve_path)
    _atomic_json(
        {
            "schema_version": 1,
            "analysis_version": PATHWAY_ANALYSIS_VERSION,
            "paper_cohort_sha256": cohort_hash(),
            "city_count": len(city_rows),
            "baseline_kappa": baseline_kappa,
            "signature": signature,
        },
        report_path,
    )
    return scenario_path, curve_path, report_path
