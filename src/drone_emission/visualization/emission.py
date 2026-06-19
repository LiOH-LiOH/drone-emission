"""
排放分析可视化模块
"""

import json
import os
import time
import zlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from ._utils import _add_title_suffix


CONTRIBUTION_SAMPLE_LIMIT = 2000
CONTRIBUTION_SAMPLE_SEED = 42
CONTRIBUTION_COMPONENT_SPECS = (
    {
        "key": "production_eol_advantage",
        "label": "Production / EOL advantage",
        "short_label": "Production / EOL",
        "color": "#4C78A8",
    },
    {
        "key": "road_detour_benefit",
        "label": "Road-detour benefit",
        "short_label": "Road detour",
        "color": "#72B7B2",
    },
    {
        "key": "drone_energy_penalty",
        "label": "Drone energy penalty",
        "short_label": "Drone energy",
        "color": "#E45756",
    },
    {
        "key": "bundling_penalty",
        "label": "Bundling penalty",
        "short_label": "Bundling",
        "color": "#B279A2",
    },
)
CONTRIBUTION_COMPONENT_KEYS = tuple(
    spec["key"] for spec in CONTRIBUTION_COMPONENT_SPECS
)
BREAK_EVEN_REPRESENTATIVE_AIR_DISTANCE_KM = 2.0
BREAK_EVEN_CURVE_AIR_DISTANCES_KM = (0.5, 1.0, 2.0, 3.0, 5.0)
BREAK_EVEN_DETOUR_LIMITS = (1.0, 2.4)
BREAK_EVEN_GEF_LIMITS = (0.05, 0.80)


def _json_safe(value):
    """Convert numpy/pandas values into JSON-safe Python scalars."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _report_number(value):
    """Return a JSON-safe scalar with NaN/Inf represented as None."""
    return _json_safe(value)


def _report_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_report_number(value) for value in values]


def _report_axis(ax):
    """Return figure-fraction position and data scale for an axis."""
    pos = ax.get_position()
    return {
        "bbox_figure_fraction": {
            "x0": _report_number(pos.x0),
            "y0": _report_number(pos.y0),
            "x1": _report_number(pos.x1),
            "y1": _report_number(pos.y1),
            "width": _report_number(pos.width),
            "height": _report_number(pos.height),
        },
        "xlim_data": _report_array(ax.get_xlim()),
        "ylim_data": _report_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _report_array(ax.get_xticks()),
        "yticks_data": _report_array(ax.get_yticks()),
    }


def _array_summary(values):
    """Return compact numeric summary stats for report payloads."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "q05": None,
            "q95": None,
        }
    q05, median, q95 = np.percentile(arr, [5, 50, 95])
    return {
        "count": int(arr.size),
        "mean": _report_number(arr.mean()),
        "median": _report_number(median),
        "min": _report_number(arr.min()),
        "max": _report_number(arr.max()),
        "q05": _report_number(q05),
        "q95": _report_number(q95),
    }


def _stable_seed(region_key: str, base_seed: int) -> int:
    """Return a deterministic per-city seed from a shared base seed."""
    return int((base_seed + zlib.crc32(region_key.encode("utf-8"))) % (2 ** 32))


def _resolve_contribution_od_path(data_dir: str, region_key: str) -> str | None:
    """Return the first available OD CSV path for a region key."""
    candidates = []
    for level in ("city", "district"):
        for key in (region_key, region_key.lower()):
            candidates.append(
                os.path.join(data_dir, level, key, f"{key}_od_pairs.csv")
            )
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.exists(path):
            return path
    return None


def _distance_strata_codes(distance_values: np.ndarray, n_strata: int = 5) -> np.ndarray:
    """Return quantile-based distance strata codes for proportional sampling."""
    if len(distance_values) == 0:
        return np.zeros(0, dtype=int)
    series = pd.Series(distance_values, copy=False)
    finite = series[np.isfinite(series)]
    if finite.empty or finite.nunique() <= 1:
        return np.zeros(len(series), dtype=int)
    q = int(min(n_strata, len(finite), finite.nunique()))
    if q <= 1:
        return np.zeros(len(series), dtype=int)
    try:
        strata = pd.qcut(finite, q=q, labels=False, duplicates="drop")
    except ValueError:
        return np.zeros(len(series), dtype=int)
    codes = np.zeros(len(series), dtype=int)
    codes[finite.index.to_numpy(dtype=int)] = strata.to_numpy(dtype=int)
    return codes


def _allocate_stratified_sample_sizes(
    stratum_sizes: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    """Allocate a proportional sample size across non-empty strata."""
    sizes = np.asarray(stratum_sizes, dtype=int)
    total = int(sizes.sum())
    if total <= 0 or sample_size >= total:
        return sizes.copy()

    raw = sample_size * sizes / total
    alloc = np.floor(raw).astype(int)
    nonempty = np.flatnonzero(sizes > 0)
    if sample_size >= len(nonempty):
        zero_alloc = nonempty[alloc[nonempty] == 0]
        alloc[zero_alloc] = 1

    max_keep = np.where(sizes > 0, sizes, 0)
    alloc = np.minimum(alloc, max_keep)

    remainder = int(sample_size - alloc.sum())
    fractions = raw - np.floor(raw)

    while remainder < 0:
        removable = np.flatnonzero(alloc > 1)
        if len(removable) == 0:
            removable = np.flatnonzero(alloc > 0)
        if len(removable) == 0:
            break
        order = removable[np.argsort(fractions[removable])]
        idx = int(order[0])
        alloc[idx] -= 1
        remainder += 1

    while remainder > 0:
        capacity = sizes - alloc
        candidates = np.flatnonzero(capacity > 0)
        if len(candidates) == 0:
            break
        order = candidates[np.argsort(-fractions[candidates])]
        idx = int(order[0])
        alloc[idx] += 1
        remainder -= 1

    return alloc


def _sample_city_valid_rows(
    valid_df: pd.DataFrame,
    sample_distance_km: np.ndarray,
    region_key: str,
    max_sample_rows: int | None,
    sample_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Sample valid OD rows with proportional distance stratification."""
    total_rows = len(valid_df)
    if total_rows == 0:
        return valid_df.copy(), {
            "valid_rows": 0,
            "sampled_rows": 0,
            "distance_strata": [],
            "sample_seed": int(sample_seed),
        }

    if max_sample_rows is None or total_rows <= max_sample_rows:
        sampled = valid_df.copy()
        sampled["_sample_expansion_weight"] = 1.0
        sampled["_sample_stratum"] = _distance_strata_codes(sample_distance_km)
        stratum_counts = (
            sampled["_sample_stratum"].value_counts().sort_index().to_dict()
        )
        return sampled, {
            "valid_rows": int(total_rows),
            "sampled_rows": int(total_rows),
            "distance_strata": [
                {
                    "stratum": int(code),
                    "valid_rows": int(count),
                    "sampled_rows": int(count),
                    "expansion_weight": 1.0,
                }
                for code, count in stratum_counts.items()
            ],
            "sample_seed": int(sample_seed),
            "sampling_mode": "full" if max_sample_rows is None else "sample_or_full",
        }

    strata = _distance_strata_codes(sample_distance_km)
    sampled_parts = []
    stratum_report = []
    rng = np.random.default_rng(_stable_seed(region_key, sample_seed))
    unique_codes = np.array(sorted(pd.unique(strata)), dtype=int)
    stratum_sizes = np.array([(strata == code).sum() for code in unique_codes], dtype=int)
    allocations = _allocate_stratified_sample_sizes(stratum_sizes, max_sample_rows)

    for code, stratum_size, draw_size in zip(unique_codes, stratum_sizes, allocations):
        indices = np.flatnonzero(strata == code)
        if draw_size <= 0 or len(indices) == 0:
            stratum_report.append(
                {
                    "stratum": int(code),
                    "valid_rows": int(stratum_size),
                    "sampled_rows": 0,
                    "expansion_weight": None,
                }
            )
            continue
        if draw_size >= len(indices):
            chosen = indices
        else:
            chosen = np.sort(rng.choice(indices, size=int(draw_size), replace=False))
        part = valid_df.iloc[chosen].copy()
        expansion_weight = float(stratum_size / len(chosen))
        part["_sample_expansion_weight"] = expansion_weight
        part["_sample_stratum"] = int(code)
        sampled_parts.append(part)
        stratum_report.append(
            {
                "stratum": int(code),
                "valid_rows": int(stratum_size),
                "sampled_rows": int(len(chosen)),
                "expansion_weight": expansion_weight,
            }
        )

    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else valid_df.iloc[0:0].copy()
    return sampled, {
        "valid_rows": int(total_rows),
        "sampled_rows": int(len(sampled)),
        "distance_strata": stratum_report,
        "sample_seed": int(sample_seed),
    }


def _resolve_threshold_from_named(
    named_df: pd.DataFrame,
    region_key: str,
    baseline_kappa: float,
    stats_kappa: float | None = None,
) -> tuple[float | None, dict]:
    """Return the preferred precomputed optimal-mixed threshold for a city."""
    if named_df is None or named_df.empty:
        return None, {"source": "missing_named_csv"}

    subset = named_df.copy()
    for col in ("kappa", "gef_mult", "threshold_km"):
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset[
        (subset["region_key"] == region_key)
        & (subset["strategy"] == "optimal_mixed")
        & np.isclose(subset["gef_mult"], 1.0)
        & subset["threshold_km"].notna()
    ]
    if subset.empty:
        return None, {"source": "missing_city_threshold"}

    candidate_specs = [("configured_kappa", baseline_kappa)]
    if stats_kappa is not None and np.isfinite(stats_kappa):
        if not np.isclose(stats_kappa, baseline_kappa):
            candidate_specs.append(("stats_kappa", float(stats_kappa)))

    unique_kappas = np.array(sorted(subset["kappa"].dropna().unique()), dtype=float)
    if len(unique_kappas) == 1:
        only_kappa = float(unique_kappas[0])
        if not any(np.isclose(only_kappa, value) for _, value in candidate_specs):
            candidate_specs.append(("only_available_kappa", only_kappa))

    for source, kappa_value in candidate_specs:
        matched = subset[np.isclose(subset["kappa"], kappa_value)]
        if matched.empty:
            continue
        threshold = float(matched.iloc[0]["threshold_km"])
        return threshold, {
            "source": source,
            "matched_kappa": float(kappa_value),
            "available_kappas": unique_kappas.tolist(),
        }

    return None, {
        "source": "no_matching_kappa",
        "available_kappas": unique_kappas.tolist(),
    }


def _sample_threshold_sweep(
    road_km: np.ndarray,
    ebike_total_g: np.ndarray,
    drone_total_g: np.ndarray,
    order_weight: np.ndarray = None,
) -> tuple[float, dict]:
    """Select the sampled optimal threshold using the standard vectorized sweep."""
    from sensitivity.vectorized_lca import threshold_sweep_km

    if order_weight is None:
        order_weight = np.ones_like(road_km, dtype=float)
    else:
        order_weight = np.asarray(order_weight, dtype=float)
    thresholds = threshold_sweep_km(road_km)
    best_total = None
    best_threshold = float("inf")
    for threshold in thresholds:
        use_drone = road_km >= threshold
        mixed_total = float(
            (np.where(use_drone, drone_total_g, ebike_total_g) * order_weight).sum()
        )
        if best_total is None or mixed_total < best_total:
            best_total = mixed_total
            best_threshold = float(threshold)
    return best_threshold, {
        "source": "sampled_threshold_sweep",
        "tested_thresholds_km": thresholds.tolist(),
    }


def _contribution_row_template(label: str) -> dict:
    """Create a zero-filled aggregate row for a plotted contribution group."""
    row = {"label": label, "sampled_drone_rows": 0}
    for key in CONTRIBUTION_COMPONENT_KEYS:
        row[f"{key}_kg"] = 0.0
        row[f"{key}_pp"] = 0.0
    row["total_net_kg"] = 0.0
    row["total_net_pp"] = 0.0
    return row


def _top_city_component_rows(
    city_reports: list[dict],
    national_baseline_kg: float,
    top_n: int = 20,
) -> list[dict]:
    """Return top-city contribution rows ranked by total sampled net reduction."""
    if national_baseline_kg <= 0:
        return []

    rows = []
    for city in city_reports:
        label = str(city.get("region_name_en", city.get("region_key", "")))
        row = _contribution_row_template(label)
        row["region_key"] = str(city.get("region_key", ""))
        row["sampled_drone_rows"] = int(city.get("drone_assigned_sampled_rows", 0))
        component_kg = city.get("component_kg", {}) or {}
        for key in CONTRIBUTION_COMPONENT_KEYS:
            kg_value = float(component_kg.get(key, 0.0))
            row[f"{key}_kg"] = kg_value
            row[f"{key}_pp"] = kg_value / national_baseline_kg * 100.0
        row["total_net_kg"] = float(city.get("net_saving_kg", 0.0))
        row["total_net_pp"] = row["total_net_kg"] / national_baseline_kg * 100.0
        rows.append(row)

    rows.sort(key=lambda row: (row["total_net_kg"], row["label"]), reverse=True)
    return rows[:top_n]


def prepare_contribution_decomposition_data(
    city_df: pd.DataFrame,
    data_dir: str,
    named_df: pd.DataFrame = None,
    exp_config: dict = None,
    max_sample_rows: int | None = CONTRIBUTION_SAMPLE_LIMIT,
    sample_seed: int = CONTRIBUTION_SAMPLE_SEED,
) -> Dict:
    """Prepare current-mechanism contribution data for the paper figure."""
    from dataclasses import replace

    from emission.cal_single_delivery import (
        create_drone_params,
        create_ebike_params,
        get_emission_params,
    )
    from sensitivity.vectorized_lca import (
        DroneScenario,
        EbikeScenario,
        compute_drone_per_order_parts_g,
        compute_ebike_per_order_parts_g,
    )

    if exp_config is None:
        print("Contribution decomposition skipped — exp_config missing")
        return None

    start_time = time.perf_counter()
    city_rows = city_df.copy()
    if "region_level" in city_rows.columns:
        city_rows = city_rows[city_rows["region_level"] == "city"].copy()

    required_city_cols = {
        "region_key",
        "region_name_en",
        "province",
        "total_ebike_lifecycle",
        "gef",
    }
    missing_city_cols = sorted(required_city_cols - set(city_rows.columns))
    if missing_city_cols:
        print(
            "Contribution decomposition skipped — city stats missing columns "
            f"{missing_city_cols}"
        )
        return None

    for col in ("total_ebike_lifecycle", "gef", "ebike_kappa"):
        if col in city_rows.columns:
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
    city_rows = city_rows[
        city_rows["total_ebike_lifecycle"].notna()
        & (city_rows["total_ebike_lifecycle"] > 0)
        & city_rows["gef"].notna()
    ].copy()
    if city_rows.empty:
        print("Contribution decomposition skipped — no eligible city rows")
        return None

    baseline_kappa = float(getattr(create_ebike_params(exp_config), "kappa", 1.0))
    emission_cfg = get_emission_params(exp_config)
    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_actual_scenario = EbikeScenario(
        kappa=baseline_kappa,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )
    ebike_no_bundle_scenario = EbikeScenario(
        kappa=1.0,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )
    drone_actual_scenario = DroneScenario(
        payload_kg=emission_cfg["payload_kg"],
    )

    assigned_frames = []
    city_reports = []
    skipped_cities = []
    national_components_kg = {key: 0.0 for key in CONTRIBUTION_COMPONENT_KEYS}
    national_baseline_kg = 0.0

    for city_row in city_rows.sort_values("region_key").itertuples(index=False):
        region_key = str(city_row.region_key)
        od_path = _resolve_contribution_od_path(data_dir, region_key)
        if od_path is None:
            skipped_cities.append(
                {"region_key": region_key, "reason": "missing_od_csv"}
            )
            continue

        od_df = pd.read_csv(
            od_path,
            usecols=lambda col: col in {"distance_km", "annual_sales", "road_distance_m"},
        )
        if "distance_km" not in od_df.columns or "annual_sales" not in od_df.columns:
            skipped_cities.append(
                {
                    "region_key": region_key,
                    "reason": "missing_required_od_columns",
                    "columns_present": sorted(od_df.columns.tolist()),
                }
            )
            continue

        valid = od_df.copy()
        valid["distance_km"] = pd.to_numeric(valid["distance_km"], errors="coerce")
        valid["annual_sales"] = pd.to_numeric(valid["annual_sales"], errors="coerce")
        has_road_distance = "road_distance_m" in valid.columns
        if has_road_distance:
            valid["road_distance_m"] = pd.to_numeric(
                valid["road_distance_m"], errors="coerce"
            )
            valid["_road_km"] = valid["road_distance_m"] / 1000.0
        else:
            valid["_road_km"] = valid["distance_km"]

        valid = valid[
            valid["distance_km"].notna()
            & (valid["distance_km"] > 0)
            & valid["_road_km"].notna()
            & (valid["_road_km"] > 0)
            & valid["annual_sales"].notna()
            & (valid["annual_sales"] > 0)
        ].copy()
        if valid.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "no_valid_od_rows"}
            )
            continue

        sample_distance_km = (
            valid["_road_km"].to_numpy(dtype=float)
            if has_road_distance
            else valid["distance_km"].to_numpy(dtype=float)
        )
        sampled, sample_meta = _sample_city_valid_rows(
            valid_df=valid,
            sample_distance_km=sample_distance_km,
            region_key=region_key,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )
        if sampled.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "sampling_returned_no_rows"}
            )
            continue

        gef = float(city_row.gef)
        ebike_pp = replace(ebike_params, e_use=gef)
        drone_pp = replace(drone_params, e_use=gef)
        road_km = sampled["_road_km"].to_numpy(dtype=float)
        air_km = sampled["distance_km"].to_numpy(dtype=float)
        annual_sales = sampled["annual_sales"].to_numpy(dtype=float)
        expansion_weight = sampled["_sample_expansion_weight"].to_numpy(dtype=float)
        annual_weight = annual_sales * expansion_weight

        ebike_actual_parts = compute_ebike_per_order_parts_g(
            road_km,
            ebike_pp,
            ebike_actual_scenario,
        )
        ebike_no_bundle_road_parts = compute_ebike_per_order_parts_g(
            road_km,
            ebike_pp,
            ebike_no_bundle_scenario,
        )
        ebike_no_bundle_air_parts = compute_ebike_per_order_parts_g(
            air_km,
            ebike_pp,
            ebike_no_bundle_scenario,
        )
        drone_actual_parts = compute_drone_per_order_parts_g(
            air_km,
            drone_pp,
            drone_actual_scenario,
        )

        use_drone = drone_actual_parts["total_g"] < ebike_actual_parts["total_g"]
        assignment_meta = {
            "rule": "oracle_per_od_lifecycle_minimum",
            "description": (
                "Assign each OD to drone only when drone lifecycle emissions "
                "are lower than the e-bike alternative."
            ),
        }

        production_eol_advantage_g = (
            ebike_no_bundle_air_parts["production_g"]
            + ebike_no_bundle_air_parts["end_of_life_g"]
            - drone_actual_parts["production_g"]
            - drone_actual_parts["end_of_life_g"]
        )
        road_detour_benefit_g = (
            ebike_no_bundle_road_parts["total_g"]
            - ebike_no_bundle_air_parts["total_g"]
        )
        drone_energy_penalty_g = (
            ebike_no_bundle_air_parts["use_g"] - drone_actual_parts["use_g"]
        )
        bundling_penalty_g = (
            ebike_actual_parts["total_g"] - ebike_no_bundle_road_parts["total_g"]
        )
        net_saving_g = ebike_actual_parts["total_g"] - drone_actual_parts["total_g"]
        sum_check_g = (
            production_eol_advantage_g
            + road_detour_benefit_g
            + drone_energy_penalty_g
            + bundling_penalty_g
            - net_saving_g
        )

        assigned = sampled.loc[use_drone].copy()
        assigned["_annual_weight"] = annual_weight[use_drone]
        for key, values in (
            ("production_eol_advantage", production_eol_advantage_g),
            ("road_detour_benefit", road_detour_benefit_g),
            ("drone_energy_penalty", drone_energy_penalty_g),
            ("bundling_penalty", bundling_penalty_g),
        ):
            assigned[f"{key}_kg"] = values[use_drone] * annual_weight[use_drone] / 1000.0
        assigned["net_saving_kg"] = net_saving_g[use_drone] * annual_weight[use_drone] / 1000.0
        assigned["sum_check_error_g"] = sum_check_g[use_drone]
        assigned["gef"] = gef
        assigned["region_key"] = region_key
        assigned["region_name_en"] = str(city_row.region_name_en)
        assigned["city_baseline_kg"] = float(city_row.total_ebike_lifecycle)

        city_components_kg = {
            key: float(assigned[f"{key}_kg"].sum())
            for key in CONTRIBUTION_COMPONENT_KEYS
        }
        city_net_kg = float(sum(city_components_kg.values()))
        city_baseline_kg = float(city_row.total_ebike_lifecycle)
        city_reports.append(
            {
                "region_key": region_key,
                "region_name_en": str(city_row.region_name_en),
                "province": str(city_row.province),
                "gef": gef,
                "baseline_ebike_lifecycle_kg": city_baseline_kg,
                "assignment_rule": assignment_meta["rule"],
                "assignment_meta": assignment_meta,
                "od_path": od_path,
                "has_road_distance_m": bool(has_road_distance),
                "sample": sample_meta,
                "drone_assigned_sampled_rows": int(use_drone.sum()),
                "sampled_drone_ratio_pct": float(use_drone.mean() * 100.0),
                "component_kg": city_components_kg,
                "component_pct_points": {
                    key: (
                        city_components_kg[key] / city_baseline_kg * 100.0
                        if city_baseline_kg > 0
                        else 0.0
                    )
                    for key in CONTRIBUTION_COMPONENT_KEYS
                },
                "net_saving_kg": city_net_kg,
                "net_saving_pct_points": (
                    city_net_kg / city_baseline_kg * 100.0
                    if city_baseline_kg > 0
                    else 0.0
                ),
                "sum_check": {
                    "max_abs_error_g_per_order": float(
                        np.abs(sum_check_g[use_drone]).max()
                    )
                    if use_drone.any()
                    else 0.0,
                    "aggregate_error_kg": float(
                        assigned["sum_check_error_g"].mul(assigned["_annual_weight"]).sum()
                        / 1000.0
                    )
                    if not assigned.empty
                    else 0.0,
                },
            }
        )

        national_baseline_kg += city_baseline_kg
        for key, value in city_components_kg.items():
            national_components_kg[key] += value
        if not assigned.empty:
            assigned_frames.append(
                assigned[
                    [
                        "_annual_weight",
                        "sum_check_error_g",
                    ]
                ].copy()
            )

    if national_baseline_kg <= 0 or not city_reports:
        print("Contribution decomposition skipped — no city contributions available")
        return None

    assigned_df = (
        pd.concat(assigned_frames, ignore_index=True)
        if assigned_frames
        else pd.DataFrame(
            columns=[
                "_annual_weight",
                "sum_check_error_g",
            ]
        )
    )

    national = {
        f"{key}_kg": float(national_components_kg[key])
        for key in CONTRIBUTION_COMPONENT_KEYS
    }
    for key in CONTRIBUTION_COMPONENT_KEYS:
        national[f"{key}_pp"] = (
            national[f"{key}_kg"] / national_baseline_kg * 100.0
            if national_baseline_kg > 0
            else 0.0
        )
    national["baseline_ebike_lifecycle_kg"] = float(national_baseline_kg)
    national["total_net_kg"] = float(
        sum(national[f"{key}_kg"] for key in CONTRIBUTION_COMPONENT_KEYS)
    )
    national["total_net_pp"] = float(
        sum(national[f"{key}_pp"] for key in CONTRIBUTION_COMPONENT_KEYS)
    )
    national["drone_assigned_sampled_rows"] = int(len(assigned_df))
    national["sum_check"] = {
        "max_abs_error_g_per_order": float(np.abs(assigned_df["sum_check_error_g"]).max())
        if not assigned_df.empty
        else 0.0,
        "aggregate_error_kg": float(
            assigned_df["sum_check_error_g"].mul(assigned_df["_annual_weight"]).sum()
            / 1000.0
        )
        if not assigned_df.empty
        else 0.0,
    }

    runtime_seconds = time.perf_counter() - start_time
    return {
        "component_specs": list(CONTRIBUTION_COMPONENT_SPECS),
        "sampling": {
            "max_sample_rows_per_city": (
                None if max_sample_rows is None else int(max_sample_rows)
            ),
            "sample_seed": int(sample_seed),
            "mode": "full_real_data" if max_sample_rows is None else "stratified_sample",
            "distance_stratification": "road_distance_m if available else distance_km",
        },
        "runtime": {
            "prep_runtime_seconds": float(runtime_seconds),
            "city_count_input": int(len(city_rows)),
            "city_count_processed": int(len(city_reports)),
            "city_count_skipped": int(len(skipped_cities)),
        },
        "national": national,
        "top_cities": _top_city_component_rows(
            city_reports=city_reports,
            national_baseline_kg=national_baseline_kg,
            top_n=15,
        ),
        "cities": city_reports,
        "skipped_cities": skipped_cities,
    }


def _format_component_annotation(pp_value: float, kg_value: float) -> str:
    """Format a pp / kt annotation for on-figure labels."""
    return f"{pp_value:+.2f} pp\n({kg_value / 1e6:+.2f} kt)"


def _add_panel_letter(
    ax,
    letter: str,
    fontsize: int = 13,
    position: tuple[float, float] = (-0.10, 1.03),
):
    """Add a bold panel letter just above the top-left corner."""
    text_kwargs = {
        "fontsize": fontsize,
        "fontweight": "bold",
        "ha": "right",
        "va": "bottom",
    }
    if hasattr(ax, "text2D"):
        ax.text2D(position[0], position[1], letter, transform=ax.transAxes, **text_kwargs)
    else:
        ax.text(position[0], position[1], letter, transform=ax.transAxes, **text_kwargs)


def _plot_contribution_waterfall(ax, national: dict, rotate_xticks: bool = False):
    """Draw the national contribution waterfall."""
    x = np.arange(len(CONTRIBUTION_COMPONENT_SPECS) + 1, dtype=float)
    cumulative = 0.0
    xticklabels = []
    for idx, spec in enumerate(CONTRIBUTION_COMPONENT_SPECS):
        value = float(national[f"{spec['key']}_pp"])
        start = cumulative
        end = cumulative + value
        bottom = min(start, end)
        height = abs(value)
        ax.bar(
            x[idx],
            height,
            bottom=bottom,
            width=0.72,
            color=spec["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            x[idx],
            end + (0.12 if value >= 0 else -0.12),
            _format_component_annotation(
                pp_value=value,
                kg_value=float(national[f"{spec['key']}_kg"]),
            ),
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
        if idx < len(CONTRIBUTION_COMPONENT_SPECS) - 1:
            ax.plot(
                [x[idx] + 0.36, x[idx] + 0.64],
                [end, end],
                color="#666666",
                linewidth=0.9,
            )
        cumulative = end
        xticklabels.append(spec["short_label"])

    total_pp = float(national["total_net_pp"])
    total_bottom = min(0.0, total_pp)
    ax.bar(
        x[-1],
        abs(total_pp),
        bottom=total_bottom,
        width=0.72,
        color="#2F4B7C",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.text(
        x[-1],
        total_pp + (0.12 if total_pp >= 0 else -0.12),
        _format_component_annotation(
            pp_value=total_pp,
            kg_value=float(national["total_net_kg"]),
        ),
        ha="center",
        va="bottom" if total_pp >= 0 else "top",
        fontsize=9,
        fontweight="bold",
    )
    xticklabels.append("Net reduction")

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(
        x,
        xticklabels,
        rotation=45 if rotate_xticks else 0,
        ha="right" if rotate_xticks else "center",
    )
    ax.set_ylabel("Contribution to national all-e-bike baseline (pp)", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(x) - 0.4)


def _plot_top_city_contribution_bars(ax, rows: list[dict]):
    """Draw top-city stacked contribution bars ranked by total net reduction."""
    if not rows:
        ax.text(
            0.5,
            0.5,
            "no sampled city contributions",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return

    x = np.arange(len(rows), dtype=float)
    pos_acc = np.zeros(len(rows), dtype=float)
    neg_acc = np.zeros(len(rows), dtype=float)

    for spec in CONTRIBUTION_COMPONENT_SPECS:
        values = np.array([float(row[f"{spec['key']}_pp"]) for row in rows], dtype=float)
        positive = np.clip(values, 0.0, None)
        negative = np.clip(values, None, 0.0)
        ax.bar(
            x,
            positive,
            bottom=pos_acc,
            width=0.72,
            color=spec["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.bar(
            x,
            -negative,
            bottom=neg_acc + negative,
            width=0.72,
            color=spec["color"],
            edgecolor="white",
            linewidth=0.8,
        )
        pos_acc += positive
        neg_acc += negative

    total_pp = np.array([float(row["total_net_pp"]) for row in rows], dtype=float)
    total_kg = np.array([float(row["total_net_kg"]) for row in rows], dtype=float)
    max_abs = max(
        0.2,
        float(np.max(np.abs(total_pp))) if len(total_pp) else 0.0,
        float(np.max(pos_acc)) if len(pos_acc) else 0.0,
        float(np.max(np.abs(neg_acc))) if len(neg_acc) else 0.0,
    )
    label_offset = max_abs * 0.025
    for idx, (pp_value, kg_value) in enumerate(zip(total_pp, total_kg)):
        ax.text(
            x[idx],
            pp_value + (label_offset if pp_value >= 0 else -label_offset),
            f"{kg_value / 1e6:.2f} kt",
            ha="center",
            va="bottom" if pp_value >= 0 else "top",
            fontsize=8,
        )

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [row["label"] for row in rows], rotation=45, ha="right")
    ax.set_ylabel("Contribution to national all-e-bike baseline (pp)", fontsize=12)
    ax.set_xlabel("City", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ymin = min(neg_acc.min(), total_pp.min(), 0.0) * 1.15
    ymax = max(pos_acc.max(), total_pp.max(), 0.0) * 1.15
    ax.set_ylim(ymin, ymax)


def _plot_contribution_decomposition_panels(
    fig,
    ax_a,
    ax_b,
    prepared: Dict,
    plot_title: bool = False,
    panel_letters: tuple[str, str] = ("a", "b"),
    legend_anchor: tuple[float, float] = (0.5, 0.98),
    panel_letter_fontsize: int = 13,
    rotate_waterfall_xticks: bool = False,
    panel_letter_position: tuple[float, float] = (-0.10, 1.03),
):
    """Draw contribution-decomposition panels on caller-provided axes."""
    _plot_contribution_waterfall(
        ax_a,
        prepared["national"],
        rotate_xticks=rotate_waterfall_xticks,
    )
    _plot_top_city_contribution_bars(ax_b, prepared.get("top_cities", [])[:15])

    if plot_title:
        ax_a.set_title(
            "Sampled Current-Mechanism Contribution Decomposition",
            fontsize=14,
        )
        ax_b.set_title("Top 15 cities by total emission reduction", fontsize=12)
    _add_panel_letter(
        ax_a,
        panel_letters[0],
        fontsize=panel_letter_fontsize,
        position=panel_letter_position,
    )
    _add_panel_letter(
        ax_b,
        panel_letters[1],
        fontsize=panel_letter_fontsize,
        position=panel_letter_position,
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=spec["color"],
            edgecolor="white",
            label=spec["label"],
        )
        for spec in CONTRIBUTION_COMPONENT_SPECS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=legend_anchor,
    )


def _write_contribution_decomposition_report(prepared: Dict, save_path: str):
    """Write the sampled contribution decomposition audit JSON."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    payload = {
        "figure": "contribution_decomposition",
        **prepared,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Contribution decomposition report saved to {report_path}")


def _break_even_panel_report(prepared: Dict, axes: tuple, panel_a_mode: str = "lines"):
    """Return compact report data for break-even phase panels."""
    ax_a, ax_b, ax_c = axes
    panel_a = prepared["panel_a"]
    panel_b = prepared["panel_b"]
    observed = prepared.get("panel_c", {})
    observed_samples = observed.get("samples", pd.DataFrame())
    city_reports = observed.get("city_reports", [])

    break_even_lines = panel_a.get("break_even_lines", [])
    line_gefs = np.array(
        [line.get("gef", np.nan) for line in break_even_lines],
        dtype=float,
    )
    observed_mask = panel_a.get("observed_mask") or {}
    mask_values = np.asarray(observed_mask.get("mask"), dtype=float)
    finite_mask = mask_values[np.isfinite(mask_values)] if mask_values.size else np.array([])

    gef_panel_reports = []
    for letter, ax, gef_panel in zip(("b", "c"), (ax_b, ax_c), panel_b["gef_panels"]):
        delta_g = np.asarray(gef_panel["delta_g"], dtype=float)
        zero_contour_present = bool(np.nanmin(delta_g) <= 0.0 <= np.nanmax(delta_g))
        gef_panel_reports.append(
            {
                "panel_letter": letter,
                "axis": _report_axis(ax),
                "label": gef_panel["label"],
                "grid_emission_factor_kg_co2e_per_kwh": _report_number(gef_panel["gef"]),
                "delta_g_summary_g_co2e_per_order": _array_summary(delta_g),
                "zero_contour_delta_g_equals_0_present": zero_contour_present,
                "color_encoding": "Delta g = E_drone - E_ebike; negative means drone lower-emission.",
            }
        )

    panel_a_common = {
        "axis": _report_axis(ax_a),
        "zlim_data": _report_array(ax_a.get_zlim()),
        "distance_grid_km": {
            "count": int(len(prepared["air_distance_grid_km"])),
            "min": _report_number(np.min(prepared["air_distance_grid_km"])),
            "max": _report_number(np.max(prepared["air_distance_grid_km"])),
        },
        "detour_ratio_grid": {
            "count": int(len(prepared["detour_ratios"])),
            "min": _report_number(np.min(prepared["detour_ratios"])),
            "max": _report_number(np.max(prepared["detour_ratios"])),
        },
        "observed_context": {
            "city_count": int(len(city_reports)),
            "sample_count": int(len(observed_samples))
            if hasattr(observed_samples, "__len__")
            else 0,
            "mask_nonzero_cells": int(np.count_nonzero(finite_mask > 0))
            if finite_mask.size
            else 0,
            "annotation": observed.get("annotation"),
        },
        "interpretation": (
            "Break-even surface where drone and e-bike lifecycle emissions are "
            "equal across distance, detour ratio, and grid carbon intensity."
        ),
    }
    if panel_a_mode == "marching_cubes":
        panel_a_key = "a_break_even_marching_cubes_surface"
        panel_a_report = {
            **panel_a_common,
            "surface_method": "skimage.measure.marching_cubes on Delta g = 0",
        }
    else:
        panel_a_key = "a_break_even_gef_lines"
        panel_a_report = {
            **panel_a_common,
            "line_count": int(len(break_even_lines)),
            "line_gef_summary_kg_co2e_per_kwh": _array_summary(line_gefs),
        }

    return {
        panel_a_key: panel_a_report,
        "b_c_delta_g_heatmaps": gef_panel_reports,
    }


def _contribution_panel_report(prepared: Dict, axes: tuple):
    """Return compact report data for contribution-decomposition panels."""
    ax_d, ax_e = axes
    national = prepared["national"]
    national_components = []
    for idx, spec in enumerate(CONTRIBUTION_COMPONENT_SPECS):
        national_components.append(
            {
                "x_index": idx,
                "key": spec["key"],
                "label": spec["label"],
                "short_label": spec["short_label"],
                "color": spec["color"],
                "kg_co2e_per_year": _report_number(national[f"{spec['key']}_kg"]),
                "percentage_points_of_national_ebike_baseline": _report_number(
                    national[f"{spec['key']}_pp"]
                ),
            }
        )
    national_components.append(
        {
            "x_index": len(CONTRIBUTION_COMPONENT_SPECS),
            "key": "total_net",
            "label": "Net reduction",
            "color": "#2F4B7C",
            "kg_co2e_per_year": _report_number(national["total_net_kg"]),
            "percentage_points_of_national_ebike_baseline": _report_number(
                national["total_net_pp"]
            ),
        }
    )

    top_city_rows = []
    for idx, row in enumerate(prepared.get("top_cities", [])[:15]):
        top_city_rows.append(
            {
                "x_index": idx,
                "region_key": row.get("region_key"),
                "label": row.get("label"),
                "baseline_ebike_lifecycle_kg": _report_number(
                    row.get("baseline_ebike_lifecycle_kg")
                ),
                "total_net_kg": _report_number(row.get("total_net_kg")),
                "total_net_pp": _report_number(row.get("total_net_pp")),
                "component_pct_points": {
                    spec["key"]: _report_number(row.get(f"{spec['key']}_pp"))
                    for spec in CONTRIBUTION_COMPONENT_SPECS
                },
                "component_kg": {
                    spec["key"]: _report_number(row.get(f"{spec['key']}_kg"))
                    for spec in CONTRIBUTION_COMPONENT_SPECS
                },
            }
        )

    return {
        "d_national_contribution_waterfall": {
            "axis": _report_axis(ax_d),
            "baseline_ebike_lifecycle_kg": _report_number(
                national["baseline_ebike_lifecycle_kg"]
            ),
            "drone_assigned_sampled_rows": _report_number(
                national["drone_assigned_sampled_rows"]
            ),
            "sum_check": _json_safe(national.get("sum_check", {})),
            "bars": national_components,
        },
        "e_top_city_contribution_bars": {
            "axis": _report_axis(ax_e),
            "city_count_plotted": len(top_city_rows),
            "bars": top_city_rows,
            "legend_components": _json_safe(list(CONTRIBUTION_COMPONENT_SPECS)),
        },
        "sampling": _json_safe(prepared.get("sampling", {})),
        "runtime": _json_safe(prepared.get("runtime", {})),
    }


def _write_break_even_contribution_decomposition_report(
    break_even_prepared: Dict,
    contribution_prepared: Dict,
    save_path: str,
    figsize: tuple,
    axes: tuple,
    figure_name: str = "break_even_contribution_decomposition",
    panel_a_mode: str = "lines",
):
    """Write compact JSON and Markdown reports for the combined figure."""
    report_base = save_path.rsplit(".", 1)[0] + "_report"
    json_path = report_base + ".json"
    md_path = report_base + ".md"
    ax_a, ax_b, ax_c, ax_d, ax_e = axes
    break_even_report = _break_even_panel_report(
        break_even_prepared,
        axes=(ax_a, ax_b, ax_c),
        panel_a_mode=panel_a_mode,
    )
    contribution_report = _contribution_panel_report(
        contribution_prepared,
        axes=(ax_d, ax_e),
    )
    payload = {
        "figure_name": figure_name,
        "figure_path": save_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "coordinate_convention": (
            "Plotted x/y/z values are Matplotlib data coordinates for the stated "
            "axis. Axis bbox values are figure-fraction coordinates."
        ),
        "report_scope": (
            "Compact LLM-readable report: full plotted grids are summarized, "
            "while all waterfall bars and top-city stacked bars are listed."
        ),
        "panels": {
            **break_even_report,
            **contribution_report,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False, allow_nan=False)

    national = contribution_prepared["national"]
    lines = [
        f"# {figure_name}",
        "",
        f"- Figure path: `{save_path}`",
        f"- Figure size: {figsize[0]} x {figsize[1]} inches",
        "- Layout: panels a-c are the break-even phase diagram; panels d-e are the contribution decomposition.",
        f"- National net reduction: {national['total_net_pp']:.3f} pp "
        f"({national['total_net_kg'] / 1e6:.3f} kt CO2e/yr).",
        f"- National all-e-bike baseline: {national['baseline_ebike_lifecycle_kg'] / 1e6:.3f} kt CO2e/yr.",
        f"- Top-city bars plotted: {len(contribution_prepared.get('top_cities', [])[:15])}.",
        "",
        "## Panel Reading Guide",
        "",
        "- a: 3D break-even surface where drone and e-bike lifecycle emissions are equal across air distance, road detour ratio, and grid carbon intensity.",
        "- b-c: heatmaps at representative GEF levels; negative Delta g means drone has lower per-order emissions.",
        "- d: national waterfall of mechanism contributions relative to the all-e-bike baseline.",
        "- e: stacked contribution bars for the top 15 cities by net reduction.",
        "",
        "## National Components",
    ]
    for spec in CONTRIBUTION_COMPONENT_SPECS:
        key = spec["key"]
        pp_key = f"{key}_pp"
        kg_key = f"{key}_kg"
        lines.append(
            f"- {spec['label']}: {national[pp_key]:.3f} pp, "
            f"{national[kg_key] / 1e6:.3f} kt CO2e/yr"
        )
    lines.extend(["", "## Top Cities"])
    for row in contribution_prepared.get("top_cities", [])[:15]:
        lines.append(
            f"- {row.get('label')}: {row.get('total_net_pp', 0.0):.3f} pp, "
            f"{row.get('total_net_kg', 0.0) / 1e6:.3f} kt CO2e/yr"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Break-even contribution decomposition report saved to {md_path}")
    print(f"Break-even contribution decomposition report data saved to {json_path}")


def plot_contribution_decomposition(
    prepared: Dict,
    save_path: str = None,
    figsize: tuple = (18, 7),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Two-panel sampled contribution decomposition for the paper figure.

    Args:
        prepared: Output of ``prepare_contribution_decomposition_data``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if not prepared:
        print("Contribution decomposition skipped — prepared data missing")
        return

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.85, 1.35],
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    _plot_contribution_decomposition_panels(
        fig,
        ax_a,
        ax_b,
        prepared,
        plot_title=plot_title,
    )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        _write_contribution_decomposition_report(prepared, save_path)
        plt.close(fig)
        print(f"Contribution decomposition saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def _build_break_even_scenarios(exp_config: dict = None):
    """Create shared LCA params and scenarios for break-even calculations."""
    from emission.cal_single_delivery import (
        create_drone_params,
        create_ebike_params,
        get_emission_params,
    )
    from sensitivity.vectorized_lca import DroneScenario, EbikeScenario

    emission_cfg = get_emission_params(exp_config)
    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_scenario = EbikeScenario(
        kappa=float(getattr(ebike_params, "kappa", 1.0)),
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )
    drone_scenario = DroneScenario(
        payload_kg=emission_cfg["payload_kg"],
    )
    return ebike_params, drone_params, ebike_scenario, drone_scenario


def _break_even_linear_coefficients(
    detour_ratios: np.ndarray,
    air_distance_km: float,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delta_g intercept and GEF slope for one air distance."""
    from dataclasses import replace

    from sensitivity.vectorized_lca import (
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    detour_ratios = np.asarray(detour_ratios, dtype=float)
    air_distance_km = float(air_distance_km)
    road_distances_km = detour_ratios * air_distance_km
    air_distances_km = np.full_like(road_distances_km, air_distance_km, dtype=float)

    ebike_zero = compute_ebike_per_order_g(
        road_distances_km,
        replace(ebike_params, e_use=0.0),
        ebike_scenario,
    )
    drone_zero = compute_drone_per_order_g(
        air_distances_km,
        replace(drone_params, e_use=0.0),
        drone_scenario,
    )
    ebike_one = compute_ebike_per_order_g(
        road_distances_km,
        replace(ebike_params, e_use=1.0),
        ebike_scenario,
    )
    drone_one = compute_drone_per_order_g(
        air_distances_km,
        replace(drone_params, e_use=1.0),
        drone_scenario,
    )

    delta_zero = drone_zero - ebike_zero
    delta_slope = (drone_one - ebike_one) - delta_zero
    return delta_zero, delta_slope


def _break_even_grid_from_coefficients(
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    grid_factors: np.ndarray,
) -> np.ndarray:
    """Evaluate delta_g on a detour-ratio x GEF grid."""
    return (
        delta_zero[np.newaxis, :]
        + np.asarray(grid_factors, dtype=float)[:, np.newaxis]
        * delta_slope[np.newaxis, :]
    )


def _break_even_zero_curve(
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
) -> np.ndarray:
    """Return the GEF at which delta_g equals zero for each detour ratio."""
    gef_min, gef_max = map(float, gef_limits)
    roots = np.full_like(delta_zero, np.nan, dtype=float)
    valid = np.abs(delta_slope) > 1e-12
    roots[valid] = -delta_zero[valid] / delta_slope[valid]
    roots[(roots < gef_min) | (roots > gef_max)] = np.nan
    return roots


def _break_even_zero_surface_raw(
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
) -> np.ndarray:
    """Return the unconstrained GEF at which delta_g equals zero."""
    roots = np.full_like(delta_zero, np.nan, dtype=float)
    valid = np.abs(delta_slope) > 1e-12
    roots[valid] = -delta_zero[valid] / delta_slope[valid]
    return roots


def _sample_break_even_observed_context(
    city_df: pd.DataFrame,
    data_dir: str,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
    max_sample_rows: int,
    sample_seed: int,
) -> dict:
    """Sample observed OD/city contexts for the phase-diagram panel."""
    from dataclasses import replace

    from sensitivity.vectorized_lca import (
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    empty_samples = pd.DataFrame(
        columns=[
            "region_key",
            "gef",
            "air_distance_km",
            "road_distance_km",
            "detour_ratio",
            "delta_g",
        ]
    )
    if city_df is None or city_df.empty:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [],
            "annotation": "No observed city sample available",
        }

    city_rows = city_df.copy()
    if "region_level" in city_rows.columns:
        city_rows = city_rows[city_rows["region_level"] == "city"].copy()

    required_cols = {"region_key", "gef"}
    missing = sorted(required_cols - set(city_rows.columns))
    if missing:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [
                {
                    "reason": "missing_city_columns",
                    "columns": missing,
                }
            ],
            "annotation": "No observed city sample available",
        }

    city_rows["gef"] = pd.to_numeric(city_rows["gef"], errors="coerce")
    city_rows = city_rows[city_rows["region_key"].notna() & city_rows["gef"].notna()]
    if city_rows.empty:
        return {
            "samples": empty_samples,
            "city_reports": [],
            "skipped_cities": [],
            "annotation": "No observed city sample available",
        }

    sample_frames = []
    city_reports = []
    skipped_cities = []

    for city_row in city_rows.sort_values("region_key").itertuples(index=False):
        region_key = str(city_row.region_key)
        od_path = _resolve_contribution_od_path(data_dir, region_key)
        if od_path is None:
            skipped_cities.append(
                {"region_key": region_key, "reason": "missing_od_csv"}
            )
            continue

        od_df = pd.read_csv(
            od_path,
            usecols=lambda col: col in {"distance_km", "road_distance_m"},
        )
        required_od_cols = {"distance_km", "road_distance_m"}
        if not required_od_cols.issubset(od_df.columns):
            skipped_cities.append(
                {
                    "region_key": region_key,
                    "reason": "missing_required_od_columns",
                    "columns_present": sorted(od_df.columns.tolist()),
                }
            )
            continue

        valid = od_df.copy()
        valid["distance_km"] = pd.to_numeric(valid["distance_km"], errors="coerce")
        valid["road_distance_m"] = pd.to_numeric(
            valid["road_distance_m"], errors="coerce"
        )
        valid["_road_km"] = valid["road_distance_m"] / 1000.0
        valid = valid[
            valid["distance_km"].notna()
            & (valid["distance_km"] > 0)
            & valid["_road_km"].notna()
            & (valid["_road_km"] > 0)
        ].copy()
        if valid.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "no_valid_od_rows"}
            )
            continue

        sampled, sample_meta = _sample_city_valid_rows(
            valid_df=valid,
            sample_distance_km=valid["_road_km"].to_numpy(dtype=float),
            region_key=region_key,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )
        if sampled.empty:
            skipped_cities.append(
                {"region_key": region_key, "reason": "sampling_returned_no_rows"}
            )
            continue

        gef = float(city_row.gef)
        air_distance_km = sampled["distance_km"].to_numpy(dtype=float)
        road_distance_km = sampled["_road_km"].to_numpy(dtype=float)
        delta_g = compute_drone_per_order_g(
            air_distance_km,
            replace(drone_params, e_use=gef),
            drone_scenario,
        ) - compute_ebike_per_order_g(
            road_distance_km,
            replace(ebike_params, e_use=gef),
            ebike_scenario,
        )
        sample_frames.append(
            pd.DataFrame(
                {
                    "region_key": region_key,
                    "gef": gef,
                    "air_distance_km": air_distance_km,
                    "road_distance_km": road_distance_km,
                    "detour_ratio": road_distance_km / air_distance_km,
                    "delta_g": delta_g,
                }
            )
        )
        city_reports.append(
            {
                "region_key": region_key,
                "gef": gef,
                "sample": sample_meta,
                "od_path": od_path,
            }
        )

    samples = (
        pd.concat(sample_frames, ignore_index=True)
        if sample_frames
        else empty_samples
    )
    if samples.empty:
        annotation = "No observed OD/city sample available"
    else:
        annotation = (
            f"Observed sample: {len(city_reports)} cities, "
            f"{len(samples):,} sampled OD pairs"
        )
    return {
        "samples": samples,
        "city_reports": city_reports,
        "skipped_cities": skipped_cities,
        "annotation": annotation,
    }


def _phase_label_position(
    delta_g_grid: np.ndarray,
    detour_ratios: np.ndarray,
    grid_factors: np.ndarray,
    drone_wins: bool,
) -> tuple[float, float] | None:
    """Return a representative in-region label position."""
    if drone_wins:
        mask = delta_g_grid < 0
    else:
        mask = delta_g_grid > 0
    if not np.any(mask):
        return None
    row_idx, col_idx = np.where(mask)
    x = float(np.median(detour_ratios[col_idx]))
    y = float(np.median(grid_factors[row_idx]))
    return x, y


def _break_even_mesh_coefficients(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    ebike_params,
    drone_params,
    ebike_scenario,
    drone_scenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Return delta_g intercept and GEF slope over a distance-detour mesh."""
    from dataclasses import replace

    from sensitivity.vectorized_lca import (
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    distance_axis = np.asarray(air_distance_grid_km, dtype=float)
    detour_axis = np.asarray(detour_ratios, dtype=float)
    distance_mesh, detour_mesh = np.meshgrid(distance_axis, detour_axis)
    road_distance_mesh = distance_mesh * detour_mesh

    air_distance_flat = distance_mesh.ravel()
    road_distance_flat = road_distance_mesh.ravel()
    drone_zero = compute_drone_per_order_g(
        air_distance_flat,
        replace(drone_params, e_use=0.0),
        drone_scenario,
    )
    ebike_zero = compute_ebike_per_order_g(
        road_distance_flat,
        replace(ebike_params, e_use=0.0),
        ebike_scenario,
    )
    drone_one = compute_drone_per_order_g(
        air_distance_flat,
        replace(drone_params, e_use=1.0),
        drone_scenario,
    )
    ebike_one = compute_ebike_per_order_g(
        road_distance_flat,
        replace(ebike_params, e_use=1.0),
        ebike_scenario,
    )

    delta_zero = (drone_zero - ebike_zero).reshape(distance_mesh.shape)
    delta_slope = ((drone_one - ebike_one) - (drone_zero - ebike_zero)).reshape(
        distance_mesh.shape
    )
    return delta_zero, delta_slope


def _break_even_observed_mask_data(
    observed_samples: pd.DataFrame,
    distance_limits: tuple[float, float],
    detour_limits: tuple[float, float],
    bin_width_km: float = 0.25,
) -> dict | None:
    """Return smoothed observed detour quantile bands for masking."""
    valid = observed_samples.copy()
    air_col = "air_distance_km" if "air_distance_km" in valid.columns else "distance_km"
    detour_col = "detour_ratio" if "detour_ratio" in valid.columns else "detour"
    if air_col not in valid.columns or detour_col not in valid.columns:
        return None

    distance_min, distance_max = map(float, distance_limits)
    detour_min, detour_max = map(float, detour_limits)
    valid[air_col] = pd.to_numeric(valid[air_col], errors="coerce")
    valid[detour_col] = pd.to_numeric(valid[detour_col], errors="coerce")
    valid = valid.dropna(subset=[air_col, detour_col]).copy()
    valid = valid[
        (valid[air_col] >= distance_min)
        & (valid[air_col] <= distance_max)
        & (valid[detour_col] >= detour_min)
        & (valid[detour_col] <= detour_max)
    ]
    if valid.empty:
        return None

    from scipy.ndimage import gaussian_filter1d

    x_edges = np.arange(distance_min, distance_max + bin_width_km, bin_width_km)
    if len(x_edges) < 2:
        return None
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    mask_x = np.linspace(distance_min, distance_max, 700)
    quantile_levels = np.array([0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95])
    quantile_curves = np.full((len(quantile_levels), len(x_centers)), np.nan)
    quantile_bands = [
        (None, 0, 0.66),
        (0, 1, 0.54),
        (1, 2, 0.42),
        (2, 3, 0.24),
        (3, 4, 0.00),
        (4, 5, 0.24),
        (5, 6, 0.42),
        (6, 7, 0.54),
        (7, None, 0.66),
    ]

    for x_i, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        if right >= distance_max:
            mask = (valid[air_col] >= left) & (valid[air_col] <= right)
        else:
            mask = (valid[air_col] >= left) & (valid[air_col] < right)
        values = valid.loc[mask, detour_col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        quantile_curves[:, x_i] = np.quantile(values, quantile_levels)

    observed = np.flatnonzero(np.isfinite(quantile_curves[0]))
    if observed.size <= 1:
        return None

    smooth_curves = np.full((len(quantile_levels), len(mask_x)), np.nan)
    for q_i in range(len(quantile_levels)):
        curve = quantile_curves[q_i]
        curve = np.interp(x_centers, x_centers[observed], curve[observed])
        curve = gaussian_filter1d(curve, sigma=1.0, mode="nearest")
        smooth_curves[q_i] = np.interp(mask_x, x_centers, curve)

    return {
        "mask_x": mask_x,
        "smooth_curves": smooth_curves,
        "quantile_bands": quantile_bands,
        "detour_limits": (detour_min, detour_max),
    }


def _draw_observed_detour_mask_2d(ax, mask_data: dict, zorder: int = 5):
    """Draw observed detour quantile mask bands on a 2D axis."""
    if not mask_data:
        return

    mask_x = mask_data["mask_x"]
    smooth_curves = mask_data["smooth_curves"]
    detour_min, detour_max = mask_data["detour_limits"]
    for low_i, high_i, alpha in mask_data["quantile_bands"]:
        if alpha <= 0:
            continue
        y_low = (
            np.full_like(mask_x, detour_min)
            if low_i is None
            else smooth_curves[low_i]
        )
        y_high = (
            np.full_like(mask_x, detour_max)
            if high_i is None
            else smooth_curves[high_i]
        )
        ax.fill_between(
            mask_x,
            y_low,
            y_high,
            color="white",
            alpha=alpha,
            linewidth=0,
            zorder=zorder,
        )


def _select_break_even_gef_levels(
    city_gefs: np.ndarray,
    gef_limits: tuple[float, float],
) -> list[dict]:
    """Choose low/high GEF slices from the strongest low/high city-distribution peaks."""
    gef_min, gef_max = map(float, gef_limits)
    values = np.asarray(city_gefs, dtype=float)
    values = values[np.isfinite(values)]

    def _pick_peak_value(
        subset: np.ndarray,
        lower: float,
        upper: float,
        fallback: float,
    ) -> float:
        subset = np.asarray(subset, dtype=float)
        subset = subset[np.isfinite(subset)]
        subset = subset[(subset >= lower) & (subset < upper)]
        if subset.size == 0:
            return float(fallback)
        if subset.size == 1:
            return float(subset[0])

        bin_width = 0.025
        edges = np.arange(lower, upper + bin_width, bin_width)
        if len(edges) < 2:
            return float(np.median(subset))
        counts, _ = np.histogram(subset, bins=edges)
        peak_idx = int(np.argmax(counts))
        peak_low = edges[peak_idx]
        peak_high = edges[peak_idx + 1]
        if peak_idx == len(counts) - 1:
            mask = (subset >= peak_low) & (subset <= peak_high)
        else:
            mask = (subset >= peak_low) & (subset < peak_high)
        peak_values = np.sort(subset[mask])
        if peak_values.size == 0:
            return float(np.median(subset))
        return float(np.median(peak_values))

    if len(values) >= 8:
        low_peak = _pick_peak_value(
            values,
            lower=gef_min,
            upper=min(0.30, gef_max),
            fallback=0.14,
        )
        high_peak = _pick_peak_value(
            values,
            lower=max(0.30, gef_min),
            upper=gef_max + 1e-9,
            fallback=0.49,
        )
        selected = np.array([low_peak, high_peak], dtype=float)
    else:
        selected = np.array([0.14, 0.49], dtype=float)

    selected = np.clip(selected, gef_min, gef_max)
    return [
        {"label": "Low GEF", "gef": float(selected[0])},
        {"label": "High GEF", "gef": float(selected[1])},
    ]


def _build_break_even_zero_lines(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
    gef_step: float = 0.05,
    line_points: int = 320,
) -> list[dict]:
    """Build fixed-GEF break-even curves in 3D space."""
    from scipy.interpolate import PchipInterpolator

    gef_min, gef_max = map(float, gef_limits)
    x_axis = np.asarray(air_distance_grid_km, dtype=float)
    y_axis = np.asarray(detour_ratios, dtype=float)
    gef_axis = np.arange(gef_min, gef_max + 1e-12, float(gef_step))
    gef_axis = gef_axis[gef_axis <= gef_max + 1e-12]
    delta_zero = np.asarray(delta_zero, dtype=float)
    delta_slope = np.asarray(delta_slope, dtype=float)

    def _smooth_segment(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_segment = np.asarray(segment[:, 0], dtype=float)
        y_segment = np.asarray(segment[:, 1], dtype=float)
        finite = np.isfinite(x_segment) & np.isfinite(y_segment)
        x_segment = x_segment[finite]
        y_segment = y_segment[finite]
        if len(x_segment) < 2:
            return x_segment, y_segment

        deltas = np.hypot(np.diff(x_segment), np.diff(y_segment))
        keep = np.concatenate(([True], deltas > 1e-10))
        x_segment = x_segment[keep]
        y_segment = y_segment[keep]
        if len(x_segment) < 4:
            return x_segment, y_segment

        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(x_segment), np.diff(y_segment))))
        )
        if distance[-1] <= 1e-10:
            return x_segment, y_segment

        line_distance = np.linspace(0.0, float(distance[-1]), line_points)
        x_line = PchipInterpolator(distance, x_segment)(line_distance)
        y_line = PchipInterpolator(distance, y_segment)(line_distance)
        return x_line, y_line

    lines = []
    for gef in gef_axis:
        delta_g = delta_zero + gef * delta_slope
        finite_delta = delta_g[np.isfinite(delta_g)]
        if not len(finite_delta) or finite_delta.min() > 0.0 or finite_delta.max() < 0.0:
            continue

        contour_fig, contour_ax = plt.subplots(figsize=(1, 1))
        try:
            contour = contour_ax.contour(
                x_axis,
                y_axis,
                delta_g,
                levels=[0.0],
            )
            segments = [
                np.asarray(segment, dtype=float)
                for segment in contour.allsegs[0]
                if len(segment) >= 2
            ]
        finally:
            plt.close(contour_fig)

        if not segments:
            continue

        segment = max(segments, key=len)
        x_line, y_line = _smooth_segment(segment)
        if len(x_line) < 2:
            continue
        lines.append(
            {
                "gef": float(gef),
                "x": np.asarray(x_line, dtype=float),
                "y": np.asarray(y_line, dtype=float),
                "z": np.full_like(x_line, float(gef), dtype=float),
            }
        )

    return lines


def _build_break_even_marching_cubes_surface(
    air_distance_grid_km: np.ndarray,
    detour_ratios: np.ndarray,
    delta_zero: np.ndarray,
    delta_slope: np.ndarray,
    gef_limits: tuple[float, float],
    gef_points: int = 72,
) -> dict | None:
    """Extract the Delta g = 0 surface with skimage marching cubes."""
    from skimage.measure import marching_cubes

    x_axis = np.asarray(air_distance_grid_km, dtype=float)
    y_axis = np.asarray(detour_ratios, dtype=float)
    z_axis = np.linspace(float(gef_limits[0]), float(gef_limits[1]), int(gef_points))
    delta_zero = np.asarray(delta_zero, dtype=float)
    delta_slope = np.asarray(delta_slope, dtype=float)
    volume = (
        delta_zero[np.newaxis, :, :]
        + z_axis[:, np.newaxis, np.newaxis] * delta_slope[np.newaxis, :, :]
    )
    finite_volume = volume[np.isfinite(volume)]
    if not len(finite_volume) or finite_volume.min() > 0.0 or finite_volume.max() < 0.0:
        return None

    dz = float(z_axis[1] - z_axis[0]) if len(z_axis) > 1 else 1.0
    dy = float(y_axis[1] - y_axis[0]) if len(y_axis) > 1 else 1.0
    dx = float(x_axis[1] - x_axis[0]) if len(x_axis) > 1 else 1.0
    vertices, faces, _, _ = marching_cubes(
        volume,
        level=0.0,
        spacing=(dz, dy, dx),
        step_size=6,
        allow_degenerate=False,
    )
    if not len(vertices) or not len(faces):
        return None

    z = vertices[:, 0] + float(z_axis[0])
    y = vertices[:, 1] + float(y_axis[0])
    x = vertices[:, 2] + float(x_axis[0])
    return {
        "kind": "marching_cubes_surface",
        "x": x,
        "y": y,
        "z": z,
        "triangles": faces,
    }


def prepare_break_even_phase_diagram_data(
    data_dir: str = None,
    city_df: pd.DataFrame = None,
    exp_config: dict = None,
    representative_air_distance_km: float = BREAK_EVEN_REPRESENTATIVE_AIR_DISTANCE_KM,
    curve_air_distances_km: tuple[float, ...] = BREAK_EVEN_CURVE_AIR_DISTANCES_KM,
    max_sample_rows: int = CONTRIBUTION_SAMPLE_LIMIT,
    sample_seed: int = CONTRIBUTION_SAMPLE_SEED,
) -> Dict:
    """Prepare break-even phase-diagram inputs for the paper figure.

    Args:
        data_dir: Root data directory for optional observed OD sampling.
        city_df: Optional city-level stats with ``region_key`` and ``gef``.
        exp_config: Experiment config dict for parameter overrides.
        representative_air_distance_km: Air distance retained for compatibility and
            surface annotation.
        curve_air_distances_km: Retained for API compatibility.
        max_sample_rows: Per-city OD sampling cap.
        sample_seed: Deterministic sampling seed.
    """
    ebike_params, drone_params, ebike_scenario, drone_scenario = (
        _build_break_even_scenarios(exp_config)
    )

    observed = {
        "samples": pd.DataFrame(
            columns=[
                "region_key",
                "gef",
                "air_distance_km",
                "road_distance_km",
                "detour_ratio",
                "delta_g",
            ]
        ),
        "city_reports": [],
        "skipped_cities": [],
        "annotation": "No observed city sample available",
    }
    if data_dir:
        observed = _sample_break_even_observed_context(
            city_df=city_df,
            data_dir=data_dir,
            ebike_params=ebike_params,
            drone_params=drone_params,
            ebike_scenario=ebike_scenario,
            drone_scenario=drone_scenario,
            max_sample_rows=max_sample_rows,
            sample_seed=sample_seed,
        )

    distance_min, distance_max = 0.2, 5.0
    detour_min, detour_max = BREAK_EVEN_DETOUR_LIMITS
    detour_ratios = np.linspace(detour_min, detour_max, 260)

    gef_min, gef_max = BREAK_EVEN_GEF_LIMITS
    grid_factors = np.linspace(gef_min, gef_max, 220)
    air_distance_grid_km = np.linspace(distance_min, distance_max, 260)
    delta_zero_grid, delta_slope_grid = _break_even_mesh_coefficients(
        air_distance_grid_km=air_distance_grid_km,
        detour_ratios=detour_ratios,
        ebike_params=ebike_params,
        drone_params=drone_params,
        ebike_scenario=ebike_scenario,
        drone_scenario=drone_scenario,
    )
    break_even_surface_raw = _break_even_zero_surface_raw(
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
    )
    break_even_surface = _break_even_zero_curve(
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
        gef_limits=BREAK_EVEN_GEF_LIMITS,
    )
    break_even_lines = _build_break_even_zero_lines(
        air_distance_grid_km=air_distance_grid_km,
        detour_ratios=detour_ratios,
        delta_zero=delta_zero_grid,
        delta_slope=delta_slope_grid,
        gef_limits=BREAK_EVEN_GEF_LIMITS,
    )
    observed_mask = _break_even_observed_mask_data(
        observed_samples=observed["samples"],
        distance_limits=(distance_min, distance_max),
        detour_limits=(detour_min, detour_max),
    )
    city_gefs = np.array(
        [report["gef"] for report in observed.get("city_reports", [])],
        dtype=float,
    )
    gef_panels = []
    for level in _select_break_even_gef_levels(city_gefs, BREAK_EVEN_GEF_LIMITS):
        gef_panels.append(
            {
                "label": level["label"],
                "gef": level["gef"],
                "delta_g": delta_zero_grid + float(level["gef"]) * delta_slope_grid,
            }
        )

    return {
        "axes": {
            "distance_limits": (float(distance_min), float(distance_max)),
            "detour_limits": (float(detour_min), float(detour_max)),
            "gef_limits": (float(gef_min), float(gef_max)),
        },
        "representative_air_distance_km": float(representative_air_distance_km),
        "curve_air_distances_km": tuple(float(v) for v in curve_air_distances_km),
        "air_distance_grid_km": air_distance_grid_km,
        "detour_ratios": detour_ratios,
        "grid_factors": grid_factors,
        "panel_a": {
            "break_even_gef": break_even_surface,
            "break_even_gef_raw": break_even_surface_raw,
            "delta_zero": delta_zero_grid,
            "delta_slope": delta_slope_grid,
            "break_even_lines": break_even_lines,
            "observed_mask": observed_mask,
        },
        "panel_b": {
            "gef_panels": gef_panels,
        },
        "panel_c": observed,
        "meta": {
            "sample_seed": int(sample_seed),
            "max_sample_rows_per_city": int(max_sample_rows),
            "ebike_kappa": float(ebike_scenario.kappa),
            "city_gefs": city_gefs,
        },
    }


def _plot_break_even_phase_diagram_panels(
    fig,
    ax_a,
    ax_b,
    ax_c,
    prepared: Dict,
    plot_title: bool = False,
    panel_letters: tuple[str, str, str] = ("a", "b", "c"),
    font_scale: float = 1.0,
    panel_letter_fontsize: int = 13,
    show_surface_annotations: bool = True,
    panel_a_mode: str = "lines",
):
    """Draw break-even phase-diagram panels on caller-provided axes."""
    air_distance_grid_km = prepared["air_distance_grid_km"]
    detour_ratios = prepared["detour_ratios"]
    panel_a = prepared["panel_a"]
    panel_b = prepared["panel_b"]
    distance_limits = prepared["axes"]["distance_limits"]
    detour_limits = prepared["axes"]["detour_limits"]
    gef_limits = prepared["axes"]["gef_limits"]

    break_even_lines = panel_a.get("break_even_lines", [])

    finite_blocks = []
    for gef_panel in panel_b["gef_panels"]:
        values = np.asarray(gef_panel["delta_g"], dtype=float)
        finite_blocks.append(values[np.isfinite(values)])
    finite_delta = np.concatenate([block for block in finite_blocks if len(block)])
    max_abs = (
        float(np.nanpercentile(np.abs(finite_delta), 90))
        if len(finite_delta)
        else 1.0
    )
    max_abs = max(max_abs, 10.0)

    delta_bounds = np.linspace(-max_abs, max_abs, 9)
    delta_cmap = plt.get_cmap("RdBu_r", len(delta_bounds) - 1)
    delta_norm = BoundaryNorm(delta_bounds, ncolors=delta_cmap.N, clip=False)
    line_cmap = plt.get_cmap("viridis")
    line_norm = plt.Normalize(vmin=gef_limits[0], vmax=gef_limits[1])
    surface_colorbar_mappable = None
    X, Y = np.meshgrid(air_distance_grid_km, detour_ratios)

    if panel_a_mode == "marching_cubes":
        surface_mesh = _build_break_even_marching_cubes_surface(
            air_distance_grid_km=air_distance_grid_km,
            detour_ratios=detour_ratios,
            delta_zero=panel_a["delta_zero"],
            delta_slope=panel_a["delta_slope"],
            gef_limits=gef_limits,
        )
        if surface_mesh:
            ax_a.plot_trisurf(
                surface_mesh["x"],
                surface_mesh["y"],
                surface_mesh["z"],
                triangles=surface_mesh["triangles"],
                cmap=line_cmap,
                norm=line_norm,
                linewidth=0.12,
                antialiased=False,
                shade=False,
                alpha=1.0,
                edgecolor=(0.25, 0.25, 0.25, 0.22),
                rasterized=True,
            )
            surface_colorbar_mappable = ScalarMappable(
                cmap=line_cmap,
                norm=line_norm,
            )
            surface_colorbar_mappable.set_array([])
    else:
        for line in break_even_lines:
            gef = float(line["gef"])
            line_color = line_cmap(line_norm(gef))
            ax_a.plot(
                line["x"],
                line["y"],
                line["z"],
                color=line_color,
                linewidth=1.55,
                alpha=0.95,
            )

    if show_surface_annotations:
        if panel_a_mode == "marching_cubes":
            annotation_specs = (
                (0.90, "Break-even surface from marching cubes", 9.5, "#444444"),
                (0.82, "Delta g = 0 implicit surface", 8.8, "#555555"),
                (0.05, "Surface marks Delta g = 0", 8.4, "#555555"),
            )
        else:
            annotation_specs = (
                (0.90, "Break-even curves at fixed GEF", 9.5, "#444444"),
                (0.82, "GEF spacing = 0.05 kg CO2e/kWh", 8.8, "#555555"),
                (0.05, "Each line marks Delta g = 0", 8.4, "#555555"),
            )
        for y, text, fontsize, color in annotation_specs:
            ax_a.text2D(
                0.02,
                y,
                text,
                transform=ax_a.transAxes,
                fontsize=fontsize * font_scale,
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "alpha": 0.84,
                    "edgecolor": "none",
                },
            )
    ax_a.set_xlim(*distance_limits)
    ax_a.set_ylim(*detour_limits)
    ax_a.set_zlim(*gef_limits)
    ax_a.set_xlabel("Air distance (km)", fontsize=10.5 * font_scale, labelpad=8)
    ax_a.set_ylabel("Detour ratio (road / air)", fontsize=10.5 * font_scale, labelpad=8)
    ax_a.set_zlabel("GEF (kg CO2e/kWh)", fontsize=10.5 * font_scale, labelpad=8)
    ax_a.set_proj_type("ortho")
    if panel_a_mode == "marching_cubes":
        ax_a.view_init(elev=20, azim=225)
    else:
        ax_a.view_init(elev=28, azim=-128)
    ax_a.grid(alpha=0.2)
    ax_a.xaxis.pane.set_alpha(0.0)
    ax_a.yaxis.pane.set_alpha(0.0)
    ax_a.zaxis.pane.set_alpha(0.0)
    if surface_colorbar_mappable is not None:
        surface_colorbar = fig.colorbar(
            surface_colorbar_mappable,
            ax=ax_a,
            fraction=0.035,
            pad=0.02,
            shrink=0.62,
            ticks=np.linspace(gef_limits[0], gef_limits[1], 4),
        )
        surface_colorbar.set_label(
            "GEF (kg CO2e/kWh)",
            fontsize=9.2 * font_scale,
        )
        surface_colorbar.ax.tick_params(labelsize=8.5 * font_scale)

    heatmap_axes = [ax_b, ax_c]
    heatmap_detour_limits = (detour_limits[0], min(detour_limits[1], 2.4))
    heatmap_mappable = None
    for ax, gef_panel in zip(heatmap_axes, panel_b["gef_panels"]):
        delta_g = np.asarray(gef_panel["delta_g"], dtype=float)
        im = ax.imshow(
            delta_g,
            cmap=delta_cmap,
            aspect="auto",
            origin="lower",
            extent=[
                distance_limits[0],
                distance_limits[1],
                detour_limits[0],
                detour_limits[1],
            ],
            norm=delta_norm,
            interpolation="nearest",
        )
        heatmap_mappable = im
        if np.nanmin(delta_g) <= 0.0 <= np.nanmax(delta_g):
            contour = ax.contour(
                X,
                Y,
                delta_g,
                levels=[0.0],
                colors="black",
                linewidths=1.3,
                zorder=6,
            )
            ax.clabel(
                contour,
                fmt={0.0: "Delta g = 0"},
                fontsize=8 * font_scale,
                inline=True,
            )
        ax.set_xlim(*distance_limits)
        ax.set_ylim(*heatmap_detour_limits)
        ax.set_xlabel("Air distance (km)", fontsize=11 * font_scale)
        if ax is ax_b:
            ax.set_ylabel("Detour ratio (road / air)", fontsize=11 * font_scale)
        else:
            ax.set_ylabel("")
        ax.grid(alpha=0.15)
        if plot_title:
            ax.set_title(
                f"{gef_panel['label']} ({gef_panel['gef']:.2f})",
                fontsize=11.5 * font_scale,
            )
        ax.text(
            0.03,
            0.97,
            f"{gef_panel['label']}\nGEF = {gef_panel['gef']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2 * font_scale,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "alpha": 0.82,
                "edgecolor": "none",
            },
            zorder=7,
        )

    colorbar = fig.colorbar(
        heatmap_mappable,
        ax=heatmap_axes,
        fraction=0.022,
        pad=0.01,
        boundaries=delta_bounds,
        ticks=delta_bounds[::2],
        extend="both",
        spacing="proportional",
    )
    colorbar.set_label(
        "Delta g = E_drone - E_ebike (g CO2e/order)",
        fontsize=10 * font_scale,
    )

    if plot_title:
        if panel_a_mode == "marching_cubes":
            title = "Break-even surface by marching cubes"
        else:
            title = "Break-even curves by GEF"
        ax_a.set_title(title, fontsize=12 * font_scale, pad=10)

    _add_panel_letter(ax_a, panel_letters[0], fontsize=panel_letter_fontsize)
    _add_panel_letter(ax_b, panel_letters[1], fontsize=panel_letter_fontsize)
    _add_panel_letter(ax_c, panel_letters[2], fontsize=panel_letter_fontsize)


def plot_break_even_phase_diagram(
    prepared: Dict,
    save_path: str = None,
    figsize: tuple = (14.5, 5.0),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Lifecycle break-even phase diagram for drone vs e-bike delivery.

    Args:
        prepared: Output of ``prepare_break_even_phase_diagram_data``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if not prepared:
        print("Break-even phase diagram skipped — prepared data missing")
        return

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    _plot_break_even_phase_diagram_panels(
        fig,
        ax_a,
        ax_b,
        ax_c,
        prepared,
        plot_title=plot_title,
    )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Break-even phase diagram saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def plot_break_even_contribution_decomposition(
    break_even_prepared: Dict,
    contribution_prepared: Dict,
    save_path: str = None,
    figsize: tuple = (15.0, 10.2),
    for_latex: bool = False,
    plot_title: bool = False,
    panel_a_mode: str = "lines",
    figure_name: str = "break_even_contribution_decomposition",
):
    """Stack break-even phase diagram above contribution decomposition.

    Args:
        break_even_prepared: Output of ``prepare_break_even_phase_diagram_data``.
        contribution_prepared: Output of ``prepare_contribution_decomposition_data``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if not break_even_prepared or not contribution_prepared:
        print("Break-even contribution decomposition skipped — prepared data missing")
        return

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 0.92],
        hspace=0.42,
    )
    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=[1.2, 1.0, 1.0],
        wspace=0.18,
    )
    bottom = outer[1].subgridspec(
        1,
        2,
        width_ratios=[0.85, 1.35],
        wspace=0.22,
    )

    ax_a = fig.add_subplot(top[0, 0], projection="3d")
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(top[0, 2])
    ax_d = fig.add_subplot(bottom[0, 0])
    ax_e = fig.add_subplot(bottom[0, 1])

    _plot_break_even_phase_diagram_panels(
        fig,
        ax_a,
        ax_b,
        ax_c,
        break_even_prepared,
        plot_title=plot_title,
        panel_letters=("a", "b", "c"),
        font_scale=0.92,
        panel_letter_fontsize=12,
        show_surface_annotations=False,
        panel_a_mode=panel_a_mode,
    )
    _plot_contribution_decomposition_panels(
        fig,
        ax_d,
        ax_e,
        contribution_prepared,
        plot_title=plot_title,
        panel_letters=("d", "e"),
        legend_anchor=(0.5, 0.455),
        panel_letter_fontsize=12,
        rotate_waterfall_xticks=True,
        panel_letter_position=(-0.10, 1.22),
    )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        _write_break_even_contribution_decomposition_report(
            break_even_prepared,
            contribution_prepared,
            save_path,
            figsize,
            axes=(ax_a, ax_b, ax_c, ax_d, ax_e),
            figure_name=figure_name,
            panel_a_mode=panel_a_mode,
        )
        plt.close(fig)
        print(f"{figure_name.replace('_', ' ').capitalize()} saved to {save_path}")
    else:
        plt.show()
        plt.close(fig)


def plot_break_even_contribution_decomposition_marching_cubes(
    break_even_prepared: Dict,
    contribution_prepared: Dict,
    save_path: str = None,
    figsize: tuple = (15.0, 10.2),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Stack contribution decomposition with a marching-cubes break-even surface.

    Args:
        break_even_prepared: Output of ``prepare_break_even_phase_diagram_data``.
        contribution_prepared: Output of ``prepare_contribution_decomposition_data``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    plot_break_even_contribution_decomposition(
        break_even_prepared,
        contribution_prepared,
        save_path=save_path,
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
        panel_a_mode="marching_cubes",
        figure_name="break_even_contribution_decomposition_marching_cubes",
    )


def add_experiment_info_box(
    ax,
    experiment_settings: Dict[str, str],
    loc: str = "upper right",
    fontsize: int = 9,
):
    """
    在图表上添加实验设定信息文本框

    参数:
        ax: Matplotlib axes 对象
        experiment_settings: 实验设定字典，键为设定名称，值为设定值
                            例如: {"Distance": "3 km", "Payload": "1 kg", "Route Factor": "1.4"}
        loc: 文本框位置，可选 "upper right", "upper left", "lower right", "lower left"
        fontsize: 字体大小
    """
    if not experiment_settings:
        return

    # 构建文本内容
    lines = [f"{k}: {v}" for k, v in experiment_settings.items()]
    text = "\n".join(lines)

    # 根据位置设置坐标
    loc_map = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }

    x, y, ha, va = loc_map.get(loc, loc_map["upper right"])

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        zorder=100,
    )


def _draw_emission_reduction_vs_drone_ratio(ax, od_df, plot_title: bool = False, n_thresholds: int = 10):
    """Draw the sales-weighted emission reduction vs drone-order ratio curve."""
    required = ["road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g", "annual_sales"]
    valid = od_df.dropna(subset=required).copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=required)
    valid = valid[
        (valid["road_distance_m"] >= 0)
        & (valid["annual_sales"] > 0)
    ]
    if valid.empty:
        ax.text(0.5, 0.5, "weighted lifecycle data\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return

    road_km = valid["road_distance_m"].to_numpy(dtype=float) / 1000.0
    drone_lc = valid["drone_lifecycle_g"].to_numpy(dtype=float)
    ebike_lc = valid["ebike_lifecycle_g"].to_numpy(dtype=float)
    annual_sales = valid["annual_sales"].to_numpy(dtype=float)
    total_orders = annual_sales.sum()
    ebike_total = np.sum(ebike_lc * annual_sales)

    middle = np.linspace(
        max(0.3, np.percentile(road_km, 5)),
        np.percentile(road_km, 95),
        n_thresholds,
    )
    thresholds = np.concatenate([[-0.01], middle, [np.inf]])

    drone_ratios = []
    reductions = []
    for t in thresholds:
        use_drone = road_km >= t
        mixed = np.sum(np.where(use_drone, drone_lc, ebike_lc) * annual_sales)
        drone_ratios.append(annual_sales[use_drone].sum() / total_orders * 100)
        reductions.append((ebike_total - mixed) / ebike_total * 100)

    ax.plot(drone_ratios, reductions, "b-s", linewidth=2, markersize=6)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Drone Order Ratio (%)", fontsize=12)
    ax.set_ylabel("Weighted Emission Reduction (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    if plot_title:
        ax.set_title("Emission Reduction vs Drone Ratio", fontsize=14)


def plot_emission_reduction_vs_drone_ratio(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (8, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_thresholds: int = 10,
):
    """Emission reduction (%) vs drone order ratio (%).

    Sweeps road-distance thresholds: orders with road_distance >= threshold
    are served by drone, others by e-bike. Plots resulting lifecycle reduction
    against annual-sales-weighted drone order share.

    Args:
        od_df: OD pairs DataFrame with road_distance_m, drone_lifecycle_g,
               ebike_lifecycle_g, annual_sales columns.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_thresholds: Number of interior thresholds between 5th and 95th
                      percentile of road distance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_emission_reduction_vs_drone_ratio(ax, od_df, plot_title=plot_title, n_thresholds=n_thresholds)
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Emission reduction vs drone ratio saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_by_road_distance_bar(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.5,
):
    """Stacked histogram: annual lifecycle emission per road-distance bin.

    Each bin holds two stacked segments:
      - bottom: optimal emission (per OD, the lower of drone vs e-bike)
      - top:    reduction (e-bike origin minus optimal)
    Total bar height = e-bike origin emission for that bin.

    Per-OD totals are weighted by ``annual_sales`` and reported in
    tonnes CO2e per year.

    Args:
        od_df: OD pairs DataFrame with ``road_distance_m``,
               ``drone_lifecycle_g``, ``ebike_lifecycle_g``, ``annual_sales``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each road-distance bin in km. Bins span 0 to
                      the observed max road distance — no overflow bin.
    """
    valid = od_df[od_df["road_distance_m"] >= 0].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    drone_lc = valid["drone_lifecycle_g"].values
    ebike_lc = valid["ebike_lifecycle_g"].values
    sales = valid["annual_sales"].values

    max_km = float(np.ceil(road_km.max() / bin_width_km) * bin_width_km)
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2

    optimal_lc = np.minimum(drone_lc, ebike_lc)
    ebike_per_od = ebike_lc * sales / 1e6  # g → tonnes
    optimal_per_od = optimal_lc * sales / 1e6

    bin_idx = np.clip(np.digitize(road_km, edges) - 1, 0, n_bins - 1)
    ebike_bin = np.zeros(n_bins)
    optimal_bin = np.zeros(n_bins)
    for k in range(n_bins):
        m = bin_idx == k
        ebike_bin[k] = ebike_per_od[m].sum()
        optimal_bin[k] = optimal_per_od[m].sum()
    reduction_bin = ebike_bin - optimal_bin

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(centers, optimal_bin, width=bin_width_km, color="#2ca02c",
           edgecolor="white", linewidth=0.3, label="Optimal (drone+e-bike)",
           align="center")
    ax.bar(centers, reduction_bin, width=bin_width_km, bottom=optimal_bin,
           color="#ff7f0e", edgecolor="white", linewidth=0.3,
           label="Emission reduction", align="center")

    ax.set_xlabel("Road Distance (km)", fontsize=12)
    ax.set_ylabel("Annual Lifecycle Emission (t CO2e)", fontsize=12)
    ax.set_xlim(0, max_km)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    if plot_title:
        ax.set_title("Annual Emission by Road Distance: Optimal + Reduction", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Emission by road distance bar saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_reduction_pct_and_detour_by_distance(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.5,
):
    """Two-panel figure binned by road distance.

    Top panel: boxplot of per-OD detour ratio (road / air) per bin.
    Bottom panel: sales-weighted emission reduction (%) per bin, defined as
    ``(sum(ebike_lc * sales) - sum(min(drone_lc, ebike_lc) * sales)) / sum(ebike_lc * sales)``.

    Args:
        od_df: OD pairs DataFrame with ``road_distance_m``, ``distance_km``,
               ``drone_lifecycle_g``, ``ebike_lifecycle_g``, ``annual_sales``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each road-distance bin in km. Bins span 0 to
                      the observed max road distance — no overflow bin.
    """
    valid = od_df[(od_df["road_distance_m"] >= 0) & (od_df["distance_km"] > 0)].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    air_km = valid["distance_km"].values
    detour = road_km / air_km
    drone_lc = valid["drone_lifecycle_g"].values
    ebike_lc = valid["ebike_lifecycle_g"].values
    sales = valid["annual_sales"].values

    max_km = float(np.ceil(road_km.max() / bin_width_km) * bin_width_km)
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2

    optimal_lc = np.minimum(drone_lc, ebike_lc)
    bin_idx = np.clip(np.digitize(road_km, edges) - 1, 0, n_bins - 1)

    reduction_pct = np.full(n_bins, np.nan)
    ebike_total_t = np.zeros(n_bins)
    detour_by_bin = []
    for k in range(n_bins):
        m = bin_idx == k
        if not m.any():
            detour_by_bin.append(np.array([]))
            continue
        ebike_tot = (ebike_lc[m] * sales[m]).sum()
        optimal_tot = (optimal_lc[m] * sales[m]).sum()
        ebike_total_t[k] = ebike_tot / 1e6
        if ebike_tot > 0:
            reduction_pct[k] = (ebike_tot - optimal_tot) / ebike_tot * 100
        detour_by_bin.append(detour[m])
    x_ticks = np.arange(0, max_km + 1e-6, 1.0)
    x_tick_labels = [f"{int(t)}" for t in x_ticks]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    nonempty = [(c, d) for c, d in zip(centers, detour_by_bin) if d.size > 0]
    box_positions = [c for c, _ in nonempty]
    box_data = [d for _, d in nonempty]
    ax_top.boxplot(
        box_data, positions=box_positions, widths=bin_width_km * 0.8,
        showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f4e79", linewidth=0.6),
        medianprops=dict(color="#1f4e79", linewidth=1.2),
        whiskerprops=dict(color="#1f4e79", linewidth=0.6),
        capprops=dict(color="#1f4e79", linewidth=0.6),
    )
    ax_top.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_top.set_ylabel("Detour Ratio (road / air)", fontsize=12)
    ax_top.set_xticks(x_ticks, labels=x_tick_labels)
    ax_top.set_xlim(0, max_km)
    ax_top.grid(axis="y", alpha=0.3)

    ax_bot.bar(
        centers, reduction_pct, width=bin_width_km,
        color="#2ca02c", edgecolor="white", linewidth=0.3, align="center",
        label="Reduction %",
    )
    ax_bot.axhline(0, color="gray", linewidth=0.6)
    ax_bot.set_xlabel("Road Distance (km)", fontsize=12)
    ax_bot.set_ylabel("Emission Reduction (%)", fontsize=12, color="#2ca02c")
    ax_bot.set_xlim(0, max_km)
    ax_bot.set_xticks(x_ticks, labels=x_tick_labels)
    ax_bot.tick_params(axis="y", labelcolor="#2ca02c")
    ax_bot.grid(axis="y", alpha=0.3)

    ax_abs = ax_bot.twinx()
    ax_abs.plot(
        centers, ebike_total_t,
        color="#555555", alpha=0.5, linewidth=1.6, marker="o", markersize=4,
        label="E-bike total (t CO2e/yr)",
    )
    ax_abs.set_ylabel("E-bike total (t CO2e/yr)", fontsize=12, color="#555555")
    ax_abs.tick_params(axis="y", labelcolor="#555555")
    ax_abs.set_ylim(bottom=0)

    if plot_title:
        fig.suptitle("Detour Ratio and Emission Reduction by Road Distance", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reduction pct and detour by distance saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _select_median_trip_emission_sample(od_df: pd.DataFrame):
    """Return an OD row close to both median air and road distances."""
    required = [
        "distance_km",
        "road_distance_m",
        "drone_emission_kg",
        "ebike_emission_kg",
    ]
    valid = od_df.dropna(subset=required).copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=required)
    valid = valid[
        (valid["distance_km"] > 0)
        & (valid["road_distance_m"] > 0)
        & (valid["drone_emission_kg"] >= 0)
        & (valid["ebike_emission_kg"] >= 0)
    ]
    if valid.empty:
        return None

    if "annual_sales" in valid.columns:
        valid["annual_sales"] = pd.to_numeric(
            valid["annual_sales"],
            errors="coerce",
        )
        weighted = valid.dropna(subset=["annual_sales"])
        weighted = weighted[weighted["annual_sales"] > 0].sort_values("distance_km")
    else:
        weighted = pd.DataFrame()

    def _weighted_or_plain_median(value_col):
        if weighted.empty:
            return valid[value_col].median()
        sorted_weighted = weighted.sort_values(value_col)
        cumulative_sales = sorted_weighted["annual_sales"].cumsum()
        median_idx = cumulative_sales.searchsorted(cumulative_sales.iloc[-1] / 2.0)
        return sorted_weighted.iloc[min(median_idx, len(sorted_weighted) - 1)][value_col]

    target_air = _weighted_or_plain_median("distance_km")
    target_road = _weighted_or_plain_median("road_distance_m")

    air_scale = valid["distance_km"].quantile(0.75) - valid["distance_km"].quantile(0.25)
    road_scale = (
        valid["road_distance_m"].quantile(0.75)
        - valid["road_distance_m"].quantile(0.25)
    )
    if pd.isna(air_scale) or air_scale <= 0:
        air_scale = max(float(valid["distance_km"].median()), 1.0)
    if pd.isna(road_scale) or road_scale <= 0:
        road_scale = max(float(valid["road_distance_m"].median()), 1.0)

    valid["_median_trip_score"] = (
        (valid["distance_km"] - target_air).abs() / air_scale
        + (valid["road_distance_m"] - target_road).abs() / road_scale
    )
    valid["_air_tiebreak"] = (valid["distance_km"] - target_air).abs()
    valid["_road_tiebreak"] = (valid["road_distance_m"] - target_road).abs()
    return valid.sort_values(
        ["_median_trip_score", "_air_tiebreak", "_road_tiebreak"]
    ).iloc[0]


def _draw_median_trip_emission_process(ax, od_df: pd.DataFrame, plot_title: bool = False):
    """Draw cumulative use-phase emission for a median-distance delivery leg."""
    sample = _select_median_trip_emission_sample(od_df)
    if sample is None:
        ax.text(
            0.5,
            0.5,
            "median trip emission data\nnot available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return None

    air_km = float(sample["distance_km"])
    road_km = float(sample["road_distance_m"]) / 1000.0
    drone_use_cycle_total = float(sample["drone_emission_kg"]) * 1000.0
    ebike_use_cycle_total = float(sample["ebike_emission_kg"]) * 1000.0

    def _drone_delivery_leg_profile(distance_km, use_cycle_total_g):
        """Scale modeled drone use-phase shares to the cached OD total."""
        try:
            from emission.cal_single_delivery import DroneEnergyModel

            energy = DroneEnergyModel().calculate_energy_consumption(
                distance_km,
                payload_kg=1.0,
            )
            parts = {
                "takeoff": energy["energy_takeoff_kwh"],
                "level_out": energy["energy_level_out_kwh"],
                "landing_out": energy["energy_landing_out_kwh"],
                "hover": energy["energy_hover_kwh"],
                "takeoff_return": energy["energy_takeoff_return_kwh"],
                "level_return": energy["energy_level_return_kwh"],
                "landing_return": energy["energy_landing_return_kwh"],
            }
            cycle_energy = sum(parts.values())
            if cycle_energy <= 0:
                raise ValueError("non-positive drone cycle energy")

            scale = use_cycle_total_g / cycle_energy
            takeoff_g = parts["takeoff"] * scale
            cruise_g = parts["level_out"] * scale
            landing_g = (parts["landing_out"] + parts["hover"]) * scale
            y_takeoff = takeoff_g
            y_cruise_end = takeoff_g + cruise_g
            y_leg_end = y_cruise_end + landing_g
            return (
                np.array([0.0, 0.0, distance_km, distance_km]),
                np.array([0.0, y_takeoff, y_cruise_end, y_leg_end]),
                {"takeoff_g": takeoff_g, "landing_g": landing_g},
            )
        except Exception:
            outbound_g = use_cycle_total_g / 2.0
            return (
                np.array([0.0, distance_km]),
                np.array([0.0, outbound_g]),
                {"takeoff_g": 0.0, "landing_g": 0.0},
            )

    def _ebike_delivery_leg_profile(distance_km, use_cycle_total_g):
        """Scale modeled e-bike outbound use-phase share to the cached OD total."""
        try:
            from emission.cal_single_delivery import EbikeEnergyModel

            energy = EbikeEnergyModel().calculate_energy_consumption(
                distance_km,
                payload_kg=1.0,
                rider_weight=70,
            )
            energy_go = energy["power_kw_go"] * energy["time_hours"]
            total_energy = energy["energy_kwh"]
            if total_energy <= 0:
                raise ValueError("non-positive e-bike cycle energy")
            outbound_g = use_cycle_total_g * energy_go / total_energy
        except Exception:
            outbound_g = use_cycle_total_g / 2.0
        return np.array([0.0, distance_km]), np.array([0.0, outbound_g])

    drone_x, drone_y, drone_parts = _drone_delivery_leg_profile(
        air_km,
        drone_use_cycle_total,
    )
    ebike_x, ebike_y = _ebike_delivery_leg_profile(
        road_km,
        ebike_use_cycle_total,
    )
    max_x = max(air_km, road_km)
    max_y = max(drone_y.max(), ebike_y.max(), 1e-6)

    drone_color = "#2b78b8"
    ebike_color = "#d95f35"
    gap_color = "#8a8f98"

    ax.plot(
        drone_x,
        drone_y,
        color=drone_color,
        linewidth=2.6,
        marker="o",
        markevery=[len(drone_x) - 1],
        markersize=6,
        label="Drone (air path)",
    )
    ax.plot(
        ebike_x,
        ebike_y,
        color=ebike_color,
        linewidth=2.6,
        marker="s",
        markevery=[len(ebike_x) - 1],
        markersize=5.5,
        label="E-bike (road route)",
    )

    ax.axvline(
        air_km,
        color=drone_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )
    ax.axvline(
        road_km,
        color=ebike_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.55,
    )

    if road_km > air_km:
        ax.annotate(
            "",
            xy=(road_km, max_y * 0.10),
            xytext=(air_km, max_y * 0.10),
            arrowprops=dict(arrowstyle="<->", color=gap_color, linewidth=1.0),
        )
        ax.text(
            (air_km + road_km) / 2,
            max_y * 0.13,
            f"route detour +{road_km - air_km:.2f} km",
            ha="center",
            va="bottom",
            fontsize=9,
            color=gap_color,
        )

    if drone_parts["takeoff_g"] > 0:
        ax.text(
            air_km * 0.015,
            drone_y[1] / 2,
            "takeoff",
            ha="left",
            va="center",
            fontsize=8.5,
            color=drone_color,
        )
        ax.text(
            air_km * 0.985,
            drone_y[-2] + drone_parts["landing_g"] / 2,
            "landing + hover",
            ha="right",
            va="center",
            fontsize=8.5,
            color=drone_color,
        )

    label_offset = max_y * 0.045
    ax.text(
        air_km,
        drone_y[-1] + label_offset,
        f"{drone_y[-1]:.1f} g\n{air_km:.2f} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color=drone_color,
    )
    ax.text(
        road_km,
        ebike_y[-1] + label_offset,
        f"{ebike_y[-1]:.1f} g\n{road_km:.2f} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color=ebike_color,
    )

    ax.set_xlim(0, max_x * 1.08)
    ax.set_ylim(0, max_y * 1.22)
    ax.set_xlabel("Actual one-way distance traveled (km)", fontsize=12)
    ax.set_ylabel(
        "Accumulated use-phase emission (g CO$_2$e / delivery leg)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.text(
        0.98,
        0.04,
        "Beijing median trip",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )

    if plot_title:
        ax.set_title(
            "Median-Air-Distance Delivery-Leg Emission Process",
            fontsize=14,
        )
    return {
        "sample": sample,
        "sample_index": sample.name,
        "air_km": air_km,
        "road_km": road_km,
        "drone_x": drone_x,
        "drone_y": drone_y,
        "ebike_x": ebike_x,
        "ebike_y": ebike_y,
    }


def plot_median_trip_emission_process(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (7.2, 4.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Cumulative use-phase emission along a median-distance delivery leg.

    Args:
        od_df: OD pairs DataFrame with distance_km, road_distance_m,
               drone_emission_kg, and ebike_emission_kg columns. If
               annual_sales is present, it is used to select the median trip.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    _draw_median_trip_emission_process(ax, od_df, plot_title=plot_title)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Median trip emission process saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_reduction_analysis(
    results_df: pd.DataFrame,
    bundle_results: dict = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制排放减少分析图

    参数:
        results_df: 包含 threshold, drone_ratio, reduction 列的 DataFrame
        bundle_results: 束配送场景结果字典 {coef: DataFrame}
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Color scheme for bundle scenarios
    bundle_colors = {
        0.9: "#2ca02c",  # green
        0.8: "#ff7f0e",  # orange
        0.7: "#9467bd",  # purple
        0.6: "#8c564b",  # brown
    }

    # 图1: 阈值 vs 减排百分比
    ax1 = axes[0]
    ax1.plot(
        results_df["threshold"],
        results_df["reduction"],
        "b-o",
        linewidth=2,
        markersize=8,
        label="No bundling",
    )

    # Plot bundle scenarios
    if bundle_results:
        for coef, df in bundle_results.items():
            color = bundle_colors.get(coef, "gray")
            ax1.plot(
                df["threshold"],
                df["reduction"],
                "-s",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=f"Bundle {coef}",
            )

    ax1.set_xlabel("Distance Threshold (km)", fontsize=12)
    ax1.set_ylabel("Emission Reduction (%)", fontsize=12)
    if plot_title:
        ax1.set_title("Emission Reduction vs Distance Threshold", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # 图2: 无人机占比 vs 减排百分比
    ax2 = axes[1]
    ax2.plot(
        results_df["drone_ratio"],
        results_df["reduction"],
        "b-s",
        linewidth=2,
        markersize=8,
        label="No bundling",
    )

    # Plot bundle scenarios
    if bundle_results:
        for coef, df in bundle_results.items():
            color = bundle_colors.get(coef, "gray")
            ax2.plot(
                df["drone_ratio"],
                df["reduction"],
                "-s",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=f"Bundle {coef}",
            )

    ax2.set_xlabel("Drone Delivery Ratio (%)", fontsize=12)
    ax2.set_ylabel("Emission Reduction (%)", fontsize=12)
    if plot_title:
        ax2.set_title("Emission Reduction vs Drone Ratio", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_road_distance_distribution(
    road_distances: "np.ndarray",
    annual_sales: "np.ndarray",
    save_path: str = None,
    dpi: int = 150,
    n_bins: int = 50,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制道路距离分布图（按年销量加权）

    参数:
        road_distances: 道路距离数组 (km)
        annual_sales: 年销量数组
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率
        n_bins: 直方图的柱数

    返回:
        保存的图像路径
    """
    # Filter out invalid distances (negative values)
    valid_mask = road_distances >= 0
    valid_distances = road_distances[valid_mask]
    valid_sales = annual_sales[valid_mask]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 图1: 未加权的距离分布
    ax1 = axes[0]
    ax1.hist(
        valid_distances,
        bins=n_bins,
        color="#2196F3",
        edgecolor="white",
        alpha=0.7,
    )
    ax1.set_xlabel("Road Distance (km)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    if plot_title:
        ax1.set_title("Road Distance Distribution (Unweighted)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # 添加统计信息
    mean_dist = valid_distances.mean()
    median_dist = np.median(valid_distances)
    ax1.axvline(
        mean_dist,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dist:.2f} km",
    )
    ax1.axvline(
        median_dist,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_dist:.2f} km",
    )
    if plot_title:
        ax1.legend(loc="upper right")

    # 图2: 按年销量加权的距离分布
    ax2 = axes[1]
    ax2.hist(
        valid_distances,
        bins=n_bins,
        weights=valid_sales,
        color="#4CAF50",
        edgecolor="white",
        alpha=0.7,
    )
    ax2.set_xlabel("Road Distance (km)", fontsize=12)
    ax2.set_ylabel("Weighted Count (by Annual Sales)", fontsize=12)
    if plot_title:
        ax2.set_title(
            "Road Distance Distribution (Weighted by Annual Sales)", fontsize=14
        )
    ax2.grid(True, alpha=0.3, axis="y")

    # 添加加权统计信息
    weighted_mean = np.average(valid_distances, weights=valid_sales)
    # 计算加权中位数
    sorted_idx = np.argsort(valid_distances)
    sorted_distances = valid_distances[sorted_idx]
    sorted_weights = valid_sales[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    weighted_median = sorted_distances[np.searchsorted(cumsum, cumsum[-1] / 2)]

    ax2.axvline(
        weighted_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Weighted Mean: {weighted_mean:.2f} km",
    )
    ax2.axvline(
        weighted_median,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Weighted Median: {weighted_median:.2f} km",
    )
    if plot_title:
        ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_road_air_distance_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 10.0,
):
    """Road distance weighted distribution with air distance boxplot per bin.

    Top panel: bar chart of road distance weighted by annual sales.
    Bottom panel: boxplot of air (haversine) distance for each road distance bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles.
        n_bins: Number of distance bins.
        max_km: Upper bound of distance axis in km.
    """
    valid = df[df["road_distance_m"] >= 0].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    air_km = valid["distance_km"].values
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Weighted bar heights
    weights, _ = np.histogram(road_km, bins=bins, weights=sales)

    # Air distance groups per bin
    bin_idx = np.digitize(road_km, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    air_groups = [air_km[bin_idx == i] for i in range(n_bins)]

    fig, (ax_bar, ax_box) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # --- Top: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.85,
    )
    ax_bar.set_ylabel("Weighted Count\n(Annual Sales)", fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}×10⁶" if x >= 1e6 else f"{x:.0f}")
    )
    if plot_title:
        ax_bar.set_title("Road Distance Distribution (Weighted by Annual Sales)", fontsize=13)

    # --- Bottom: boxplot of air distance per bin ---
    nonempty = [(i, g) for i, g in enumerate(air_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        bp = ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.7,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.8),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        _ = bp  # suppress unused warning

    ax_box.set_xlabel("Road Distance (km)", fontsize=11)
    ax_box.set_ylabel("Air Distance (km)", fontsize=11)
    ax_box.set_xlim(0, max_km)
    ax_box.grid(True, alpha=0.3, axis="y")
    if plot_title:
        ax_box.set_title("Air Distance Distribution per Road Distance Bin", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Road air distance distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_road_detour_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 10.0,
):
    """Road distance weighted bar chart with detour ratio boxplot on twin y-axis.

    Left y-axis: bar chart of road distance weighted by annual sales.
    Right y-axis: boxplot of detour ratio (road / air distance) per bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_bins: Number of distance bins.
        max_km: Upper bound of distance axis in km.
    """
    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    detour = road_km / valid["distance_km"].values
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    weights, _ = np.histogram(road_km, bins=bins, weights=sales)

    bin_idx = np.clip(np.digitize(road_km, bins) - 1, 0, n_bins - 1)
    detour_groups = [detour[bin_idx == i] for i in range(n_bins)]

    fig, ax_bar = plt.subplots(figsize=figsize)
    ax_box = ax_bar.twinx()

    # --- Left: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.7,
        zorder=2,
    )
    ax_bar.set_xlabel("Road Distance (km)", fontsize=11)
    ax_bar.set_ylabel("Weighted Count (Annual Sales)", fontsize=11)
    ax_bar.set_xlim(0, max_km)
    ax_bar.grid(True, alpha=0.3, axis="y", zorder=1)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    )

    # --- Right: boxplot of detour ratio per bin ---
    nonempty = [(i, g) for i, g in enumerate(detour_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.85),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            zorder=3,
        )

    ax_box.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)
    ax_box.set_ylabel("Detour Ratio (road / air)", fontsize=11)

    if plot_title:
        ax_bar.set_title("Road Distance Distribution & Detour Ratio", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Road detour distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_air_detour_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 8.0,
):
    """Air distance weighted bar chart with detour ratio boxplot on twin y-axis.

    Left y-axis: bar chart of air distance weighted by annual sales.
    Right y-axis: boxplot of detour ratio (road / air) per air distance bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_bins: Number of distance bins.
        max_km: Upper bound of air distance axis in km.
    """
    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    air_km = valid["distance_km"].values
    detour = (valid["road_distance_m"].values / 1000.0) / air_km
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    weights, _ = np.histogram(air_km, bins=bins, weights=sales)

    bin_idx = np.clip(np.digitize(air_km, bins) - 1, 0, n_bins - 1)
    detour_groups = [detour[bin_idx == i] for i in range(n_bins)]

    fig, ax_bar = plt.subplots(figsize=figsize)
    ax_box = ax_bar.twinx()

    # --- Left: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.7,
        zorder=2,
    )
    ax_bar.set_xlabel("Air Distance (km)", fontsize=11)
    ax_bar.set_ylabel("Weighted Count (Annual Sales)", fontsize=11)
    ax_bar.set_xlim(0, max_km)
    ax_bar.grid(True, alpha=0.3, axis="y", zorder=1)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    )

    # --- Right: boxplot of detour ratio per air distance bin ---
    nonempty = [(i, g) for i, g in enumerate(detour_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.85),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            zorder=3,
        )

    ax_box.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)
    ax_box.set_ylabel("Detour Ratio (road / air)", fontsize=11)

    if plot_title:
        ax_bar.set_title("Air Distance Distribution & Detour Ratio", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Air detour distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_air_road_joint(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = None,
    for_latex: bool = False,
    plot_title: bool = False,
    max_road_km: float = None,
    max_air_km: float = None,
    gridsize: int = 40,
):
    """Joint distribution of road (x) and air (y) distance with sales-weighted density.

    Centre: hexbin of road (x) vs air (y) distance weighted by annual sales.
    Top marginal: road distance distribution weighted by annual sales.
    Right marginal: air distance distribution weighted by annual sales.
    Reference lines show detour ratio = 1.2×, 1.5×, 2.0× (air = road / ratio).

    Args:
        df: DataFrame with 'distance_km', 'road_distance_m', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        max_road_km: Upper bound for road distance axis; None = 99th percentile of data.
        max_air_km: Upper bound for air distance axis; None = 99th percentile of data.
        gridsize: Hexbin grid resolution.
    """
    import matplotlib.gridspec as gridspec

    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    air_km = valid["distance_km"].values
    road_km = valid["road_distance_m"].values / 1000.0
    sales = valid["annual_sales"].values

    if max_road_km is None:
        max_road_km = float(np.percentile(road_km, 99))
    if max_air_km is None:
        max_air_km = float(np.percentile(air_km, 99))

    if figsize is None:
        # Size the figure so 1 km = 1 km on both axes (equal aspect).
        # GridSpec: width_ratios=[4,1,0.15], height_ratios=[1,4]
        main_h = 5.0  # inches for the main panel
        main_w = main_h * (max_road_km / max_air_km)
        total_w = main_w * (4 + 1 + 0.15) / 4
        total_h = main_h * (4 + 1) / 4
        figsize = (total_w, total_h)

    mask = (air_km <= max_air_km) & (road_km <= max_road_km)
    air_km, road_km, sales = air_km[mask], road_km[mask], sales[mask]

    # Equal bin width for both marginals
    n_bars = 30
    road_bins = np.linspace(0, max_road_km, n_bars + 1)
    air_bins = np.linspace(0, max_air_km, n_bars + 1)
    road_centers = (road_bins[:-1] + road_bins[1:]) / 2
    air_centers = (air_bins[:-1] + air_bins[1:]) / 2
    road_bw = road_bins[1] - road_bins[0]
    air_bw = air_bins[1] - air_bins[0]
    road_weights, _ = np.histogram(road_km, bins=road_bins, weights=sales)
    air_weights, _ = np.histogram(air_km, bins=air_bins, weights=sales)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1, 4], width_ratios=[4, 1, 0.15],
        hspace=0.05, wspace=0.25,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])
    for ax in [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]:
        ax.set_visible(False)

    # --- Centre: hexbin (x=road, y=air) ---
    hb = ax_main.hexbin(
        road_km, air_km,
        C=sales,
        reduce_C_function=np.sum,
        gridsize=gridsize,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, max_road_km, 0, max_air_km],
    )
    ax_main.set_xlabel("Road Distance (km)", fontsize=11)
    ax_main.set_ylabel("Air Distance (km)", fontsize=11)
    ax_main.set_xlim(0, max_road_km)
    ax_main.set_ylim(0, max_air_km)
    ax_main.set_aspect("equal")

    # Detour ratio reference lines: detour = road/air → y = x/ratio
    ref_ratios = [(1.2, "--", "#555555"), (1.5, "-.", "#333333"), (2.0, ":", "#111111")]
    for ratio, ls, color in ref_ratios:
        y_end = min(max_air_km, max_road_km / ratio)
        x_end = y_end * ratio
        ax_main.plot(
            [0, x_end], [0, y_end],
            linestyle=ls, color=color, linewidth=1.0, alpha=0.7,
            label=f"detour = {ratio}×",
        )
    ax_main.legend(fontsize=9, loc="upper left")

    # Colourbar in its own dedicated axes
    cb = fig.colorbar(hb, cax=ax_cbar)
    cb.set_label("Annual Sales (sum)", fontsize=9)
    cb.formatter = plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    cb.update_ticks()

    # --- Top marginal: road distance ---
    ax_top.bar(road_centers, road_weights, width=road_bw * 0.85, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax_top.set_ylabel("Sales (M)", fontsize=9)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(0, max_road_km)
    ax_top.grid(True, alpha=0.3, axis="y")

    # --- Right marginal: air distance (horizontal) ---
    ax_right.barh(air_centers, air_weights, height=air_bw * 0.85, color="#2196F3", edgecolor="white", alpha=0.8)
    ax_right.set_xlabel("Sales (M)", fontsize=9)
    ax_right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax_right.tick_params(labelleft=False)
    ax_right.set_ylim(0, max_air_km)
    ax_right.grid(True, alpha=0.3, axis="x")

    if plot_title:
        fig.suptitle("Air vs Road Distance Joint Distribution (Shanghai)", fontsize=13)

    # After equal-aspect shrinks ax_main, snap marginals to its actual bounds.
    fig.canvas.draw()
    m = ax_main.get_position()
    r = ax_right.get_position()
    t = ax_top.get_position()
    c = ax_cbar.get_position()
    ax_right.set_position([r.x0, m.y0, r.width, m.height])
    ax_top.set_position([m.x0, t.y0, m.width, t.height])
    ax_cbar.set_position([c.x0, m.y0, c.width, m.height])

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Air road joint distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_comparison_bar(
    ebike_emissions: float,
    drone_emissions: float,
    mixed_emissions: float = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制不同配送方式排放对比柱状图

    参数:
        ebike_emissions: 电动自行车总排放 (kg CO2e)
        drone_emissions: 无人机总排放 (kg CO2e)
        mixed_emissions: 混合模式总排放 (kg CO2e)，可选
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["E-bike", "Drone"]
    values = [ebike_emissions, drone_emissions]
    colors = ["#4CAF50", "#2196F3"]

    if mixed_emissions is not None:
        labels.append("Mixed")
        values.append(mixed_emissions)
        colors.append("#FF9800")

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Total Emissions (kg CO2e)", fontsize=12)
    if plot_title:
        ax.set_title("Emission Comparison by Delivery Mode", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_distance_distribution(
    road_distances,
    thresholds: list = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制配送距离分布直方图

    参数:
        road_distances: 道路距离数组
        thresholds: 阈值列表，用于标记垂直线
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(road_distances, bins=50, color="#2196F3", alpha=0.7, edgecolor="black")

    if thresholds:
        colors = plt.cm.Reds(range(50, 250, int(200 / len(thresholds))))
        for i, threshold in enumerate(thresholds):
            ax.axvline(
                x=threshold,
                color=colors[i],
                linestyle="--",
                linewidth=1.5,
                label=f"{threshold:.1f} km",
            )
        ax.legend(title="Thresholds", loc="upper right")

    ax.set_xlabel("Road Distance (km)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    if plot_title:
        ax.set_title("Distribution of Delivery Distances", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_sensitivity_analysis(
    param_values: list,
    ebike_emissions: list,
    drone_emissions: list,
    param_name: str,
    param_unit: str = "",
    experiment_settings: Dict[str, str] = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制单参数敏感性分析图

    参数:
        param_values: 参数值列表
        ebike_emissions: 对应的电动自行车排放列表 (g CO2e)
        drone_emissions: 对应的无人机排放列表 (g CO2e)
        param_name: 参数名称
        param_unit: 参数单位
        experiment_settings: 实验设定字典，用于显示在图表上
        save_path: 保存路径
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 排放值对比
    ax1 = axes[0]
    ax1.plot(
        param_values, ebike_emissions, "g-o", linewidth=2, markersize=6, label="E-bike"
    )
    ax1.plot(
        param_values, drone_emissions, "b-s", linewidth=2, markersize=6, label="Drone"
    )
    ax1.set_xlabel(
        f"{param_name} ({param_unit})" if param_unit else param_name, fontsize=12
    )
    ax1.set_ylabel("Emissions (g CO2e)", fontsize=12)
    if plot_title:
        ax1.set_title(f"Emissions vs {param_name}", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 添加实验设定信息
    if experiment_settings and plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    # 图2: 无人机相对电动车的排放变化百分比
    ax2 = axes[1]
    relative_diff = [
        (d / e - 1) * 100 for d, e in zip(drone_emissions, ebike_emissions)
    ]
    ax2.plot(param_values, relative_diff, "r-^", linewidth=2, markersize=6)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(
        f"{param_name} ({param_unit})" if param_unit else param_name, fontsize=12
    )
    ax2.set_ylabel("Drone Emissions Relative to E-bike (%)", fontsize=12)
    if plot_title:
        ax2.set_title(f"Relative Emission Difference vs {param_name}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标记交叉点
    for i in range(len(relative_diff) - 1):
        if relative_diff[i] * relative_diff[i + 1] < 0:
            # 线性插值找交叉点
            x_cross = param_values[i] + (param_values[i + 1] - param_values[i]) * (
                -relative_diff[i] / (relative_diff[i + 1] - relative_diff[i])
            )
            ax2.axvline(x=x_cross, color="orange", linestyle=":", linewidth=2)
            ax2.annotate(
                f"Crossover: {x_cross:.2f}",
                xy=(x_cross, 0),
                xytext=(x_cross, np.max(relative_diff) * 0.3),
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="orange"),
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_distance_route_factor_heatmap(
    distances: list,
    route_factors: list,
    emission_diff_matrix,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
    ylabel: str = "Route Factor (road/straight-line)",
    xlabel: str = "Delivery Distance (km)",
) -> str:
    """
    绘制配送距离 × 路径系数的二维热力图

    参数:
        distances: 配送距离列表 (km)
        route_factors: 路径系数列表
        emission_diff_matrix: 排放差异矩阵 (无人机相对电动车的百分比变化)
                              shape: (len(route_factors), len(distances))
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建热力图
    abs_max = max(abs(emission_diff_matrix.min()), abs(emission_diff_matrix.max()))
    im = ax.imshow(
        emission_diff_matrix,
        cmap="RdYlGn_r",  # 红色=无人机排放高，绿色=无人机排放低
        aspect="auto",
        origin="lower",
        extent=[distances[0], distances[-1], route_factors[0], route_factors[-1]],
        vmin=-abs_max,
        vmax=abs_max,
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Drone Emissions Relative to E-bike (%)", fontsize=12)

    # 添加等值线 (类似等高线，显示不同排放差异水平)
    X, Y = np.meshgrid(distances, route_factors)

    # 根据数据范围动态生成等值线水平，间隔20%
    data_min = np.floor(emission_diff_matrix.min() / 20) * 20
    data_max = np.ceil(emission_diff_matrix.max() / 20) * 20
    contour_levels = np.arange(data_min, data_max + 1, 20)

    # 绘制所有等值线
    linewidths = [3 if lvl == 0 else 2 for lvl in contour_levels]
    linestyles = ["-" if lvl == 0 else "--" for lvl in contour_levels]
    contour = ax.contour(
        X,
        Y,
        emission_diff_matrix,
        levels=contour_levels,
        colors="black",
        linewidths=linewidths,
        linestyles=linestyles,
    )
    ax.clabel(contour, inline=True, fontsize=9, fmt=lambda x: f"{x:+.0f}%")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if plot_title:
        ax.set_title(
            "Emission Comparison: Drone vs E-bike\n(Negative = Drone Better)",
            fontsize=14,
        )

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax, experiment_settings, loc="upper left")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_velocity_power_emission(
    velocities: list,
    powers: list,
    emissions: list,
    vehicle_type: str = "Drone",
    distance_info: str = "",
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制速度-功率-排放敏感性分析图

    参数:
        velocities: 速度列表 (km/h)
        powers: 对应的功率列表 (kW)
        emissions: 对应的排放列表 (g CO2e)
        vehicle_type: 车辆类型 ("Drone" 或 "E-bike")
        distance_info: 距离信息字符串，用于图表标题
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 根据车辆类型选择颜色
    if vehicle_type.lower() == "drone":
        color_power = "b-o"
        color_emission = "r-s"
        xlabel = "Drone Horizontal Velocity (km/h)"
    else:
        color_power = "g-o"
        color_emission = "m-s"
        xlabel = "E-bike Velocity (km/h)"

    # 图1: 速度 vs 功率
    ax1 = axes[0]
    ax1.plot(velocities, powers, color_power, linewidth=2, markersize=6)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Power (kW)", fontsize=12)
    if plot_title:
        ax1.set_title(f"{vehicle_type} Power vs Velocity", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 图2: 速度 vs 排放
    ax2 = axes[1]
    ax2.plot(velocities, emissions, color_emission, linewidth=2, markersize=6)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Emissions (g CO2e)", fontsize=12)
    if plot_title:
        title = f"{vehicle_type} Emissions vs Velocity"
        if distance_info:
            title += f" ({distance_info})"
        ax2.set_title(title, fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_energy_consumption_vs_speed(
    velocities: list,
    energy_per_km: list,
    vehicle_type: str = "E-bike",
    empirical_range: tuple = None,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制能耗(Wh/km)与速度的关系图

    参数:
        velocities: 速度列表 (km/h)
        energy_per_km: 每公里能耗列表 (Wh/km)
        vehicle_type: 车辆类型
        empirical_range: 经验值范围元组 (min, max)，用于对比
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # 选择颜色
    if vehicle_type.lower() == "drone":
        color = "#2196F3"
        xlabel = "Drone Horizontal Velocity (km/h)"
    else:
        color = "#4CAF50"
        xlabel = "E-bike Velocity (km/h)"

    # 绘制模型预测曲线
    ax.plot(
        velocities,
        energy_per_km,
        "o-",
        color=color,
        linewidth=2,
        markersize=8,
        label=f"{vehicle_type} Model",
    )

    # 添加经验值范围
    if empirical_range:
        min_val, max_val = empirical_range
        ax.axhspan(
            min_val,
            max_val,
            alpha=0.2,
            color="orange",
            label=f"Empirical Range ({min_val}-{max_val} Wh/km)",
        )
        ax.axhline(
            y=(min_val + max_val) / 2,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Energy Consumption (Wh/km)", fontsize=12)
    if plot_title:
        ax.set_title(f"{vehicle_type} Energy Consumption vs Velocity", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_combined_velocity_power(
    drone_velocities: list,
    drone_powers: list,
    ebike_velocities: list,
    ebike_powers: list,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    将无人机和电动车的速度-功率曲线绘制在同一张图上（双X轴）

    参数:
        drone_velocities: 无人机速度列表 (km/h)
        drone_powers: 无人机功率列表 (kW)
        ebike_velocities: 电动车速度列表 (km/h)
        ebike_powers: 电动车功率列表 (kW)
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 电动车曲线 (底部X轴)
    color_ebike = "#4CAF50"
    line1 = ax1.plot(
        ebike_velocities,
        ebike_powers,
        "o-",
        color=color_ebike,
        linewidth=2,
        markersize=6,
        label="E-bike",
    )
    ax1.set_xlabel("E-bike Velocity (km/h)", fontsize=12, color=color_ebike)
    ax1.set_ylabel("Power (kW)", fontsize=12)
    ax1.tick_params(axis="x", labelcolor=color_ebike)
    ax1.grid(True, alpha=0.3)

    # 无人机曲线 (顶部X轴)
    ax2 = ax1.twiny()
    color_drone = "#2196F3"
    line2 = ax2.plot(
        drone_velocities,
        drone_powers,
        "s-",
        color=color_drone,
        linewidth=2,
        markersize=6,
        label="Drone",
    )
    ax2.set_xlabel("Drone Horizontal Velocity (km/h)", fontsize=12, color=color_drone)
    ax2.tick_params(axis="x", labelcolor=color_drone)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11)

    if plot_title:
        ax1.set_title("Power vs Velocity Comparison", fontsize=14)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_velocity_power_stacked(
    drone_velocities: list,
    drone_power_breakdowns: list,
    ebike_velocities: list,
    ebike_power_breakdowns: list,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制速度-功率对比的堆叠柱状图，显示各功率分量

    参数:
        drone_velocities: 无人机速度列表 (km/h)
        drone_power_breakdowns: 无人机功率分解列表，每个元素是包含
            P_induced, P_profile, P_drag, P_climb, P_aux 的字典
        ebike_velocities: 电动车速度列表 (km/h)
        ebike_power_breakdowns: 电动车功率分解列表，每个元素是包含
            P_rolling, P_air, P_grade, P_acc 的字典
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 无人机堆叠柱状图 ==========
    ax1 = axes[0]
    x_drone = np.arange(len(drone_velocities))
    bar_width = 0.7

    # 无人机功率分量 (按显示顺序)
    drone_components = ["P_induced", "P_profile", "P_drag", "P_climb", "P_aux"]
    drone_labels = ["Induced", "Profile", "Drag", "Climb", "Auxiliary"]
    drone_colors = ["#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD"]

    # 提取各分量数据
    drone_data = {comp: [] for comp in drone_components}
    for breakdown in drone_power_breakdowns:
        for comp in drone_components:
            drone_data[comp].append(breakdown.get(comp, 0) * 1000)  # kW -> W

    # 绘制堆叠柱状图
    bottom = np.zeros(len(drone_velocities))
    for comp, label, color in zip(drone_components, drone_labels, drone_colors):
        values = np.array(drone_data[comp])
        ax1.bar(
            x_drone,
            values,
            bar_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax1.set_xlabel("Drone Horizontal Velocity (km/h)", fontsize=12)
    ax1.set_ylabel("Power (W)", fontsize=12)
    if plot_title:
        ax1.set_title("Drone Power Breakdown vs Velocity", fontsize=14)
    ax1.set_xticks(x_drone)
    ax1.set_xticklabels([f"{v:.0f}" for v in drone_velocities])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ========== 电动车堆叠柱状图 ==========
    ax2 = axes[1]
    x_ebike = np.arange(len(ebike_velocities))

    # 电动车功率分量 (按显示顺序)
    ebike_components = ["P_rolling", "P_air", "P_grade", "P_acc"]
    ebike_labels = ["Rolling", "Air Drag", "Grade", "Acceleration"]
    ebike_colors = ["#388E3C", "#66BB6A", "#A5D6A7", "#C8E6C9"]

    # 提取各分量数据
    ebike_data = {comp: [] for comp in ebike_components}
    for breakdown in ebike_power_breakdowns:
        for comp in ebike_components:
            ebike_data[comp].append(breakdown.get(comp, 0) * 1000)  # kW -> W

    # 绘制堆叠柱状图
    bottom = np.zeros(len(ebike_velocities))
    for comp, label, color in zip(ebike_components, ebike_labels, ebike_colors):
        values = np.array(ebike_data[comp])
        ax2.bar(
            x_ebike,
            values,
            bar_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax2.set_xlabel("E-bike Velocity (km/h)", fontsize=12)
    ax2.set_ylabel("Power (W)", fontsize=12)
    if plot_title:
        ax2.set_title("E-bike Power Breakdown vs Velocity", fontsize=14)
    ax2.set_xticks(x_ebike)
    ax2.set_xticklabels([f"{v:.0f}" for v in ebike_velocities])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax2, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_distance_detour_scatter(
    straight_distances: "np.ndarray",
    road_distances: "np.ndarray",
    annual_sales: "np.ndarray" = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制直线距离与绕行比的关系图（中位数线 + 10-90%区间）
    """
    # 计算绕行比
    valid_mask = (straight_distances > 0) & (road_distances > 0)
    distances = straight_distances[valid_mask]
    detour_ratios = road_distances[valid_mask] / distances

    fig, ax = plt.subplots(figsize=(10, 6))

    # 分箱计算统计量
    n_bins = 20
    d_min, d_max = distances.min(), distances.max()
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 首尾使用边界值而非中心
    bin_centers[0] = d_min
    bin_centers[-1] = d_max
    bin_medians = []
    bin_p10 = []
    bin_p90 = []

    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_medians.append(np.median(detour_ratios[mask]))
            bin_p10.append(np.percentile(detour_ratios[mask], 10))
            bin_p90.append(np.percentile(detour_ratios[mask], 90))
        else:
            bin_medians.append(np.nan)
            bin_p10.append(np.nan)
            bin_p90.append(np.nan)

    bin_medians = np.array(bin_medians)
    bin_p10 = np.array(bin_p10)
    bin_p90 = np.array(bin_p90)

    # 绘制中位数线和10-90%区间
    valid_bins = ~np.isnan(bin_medians)
    ax.fill_between(
        bin_centers[valid_bins],
        bin_p10[valid_bins],
        bin_p90[valid_bins],
        alpha=0.3,
        color="#2196F3",
        label="10-90% Range",
    )
    ax.plot(bin_centers[valid_bins], bin_medians[valid_bins], "#1565C0", linewidth=2.5, label="Median")

    # 添加参考线
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, label="Direct (ratio=1)")

    ax.set_xlabel("Straight-line Distance (km)", fontsize=12)
    ax.set_ylabel("Detour Ratio (road / straight-line)", fontsize=12)
    if plot_title:
        ax.set_title("Distance vs Detour Ratio", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d_min, d_max)

    # 设置 y 轴范围
    ax.set_ylim(0.9, np.nanmax(bin_p90) * 1.05)

    # 添加统计信息
    if plot_title:
        stats_text = (
            f"n = {len(distances):,}\n"
            f"Mean: {detour_ratios.mean():.2f}\n"
            f"Median: {np.median(detour_ratios):.2f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def _save_or_show(fig, save_path, dpi, for_latex, plot_title):
    """Helper to save or show a figure."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def plot_time_vs_distance(
    road_distances_km,
    ebike_time_min,
    drone_time_min,
    optimal_threshold_km: float,
    save_path: str = None,
    dpi: int = 300,
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 50,
):
    """Plot trip time vs trip distance for all-ebike and optimal mix scenarios."""
    road_distances_km = np.asarray(road_distances_km)
    ebike_time_min = np.asarray(ebike_time_min)
    drone_time_min = np.asarray(drone_time_min)

    mix_time = np.where(road_distances_km >= optimal_threshold_km, drone_time_min, ebike_time_min)

    bin_edges = np.linspace(road_distances_km.min(), np.percentile(road_distances_km, 99), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(road_distances_km, bin_edges) - 1, 0, n_bins - 1)

    ebike_means = np.array([ebike_time_min[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(n_bins)])
    mix_means = np.array([mix_time[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(n_bins)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, ebike_means, color="#e74c3c", linewidth=2, label="All E-bike")
    ax.plot(bin_centers, mix_means, color="#3498db", linewidth=2, label="Optimal Mix")
    ax.axvline(optimal_threshold_km, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Threshold: {optimal_threshold_km:.1f} km")

    ax.set_xlabel("Trip Distance (km)", fontsize=12)
    ax.set_ylabel("Trip Time (min)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if plot_title:
        ax.set_title("Trip Time vs Distance: All E-bike vs Optimal Mix", fontsize=14)

    plt.tight_layout()
    return _save_or_show(fig, save_path, dpi, for_latex, plot_title)


def plot_distance_vs_emission(
    road_distances_km,
    ebike_lifecycle_g,
    drone_lifecycle_g,
    optimal_threshold_km: float,
    annual_sales=None,
    save_path: str = None,
    dpi: int = 300,
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 50,
):
    """Plot lifecycle emission vs trip distance for all-ebike and optimal mix scenarios.

    When ``annual_sales`` is provided, binned means are weighted by annual OD
    order counts.
    """
    road_distances_km = np.asarray(road_distances_km)
    ebike_lifecycle_g = np.asarray(ebike_lifecycle_g)
    drone_lifecycle_g = np.asarray(drone_lifecycle_g)
    weights = None if annual_sales is None else np.asarray(annual_sales, dtype=float)

    mix_emission = np.where(road_distances_km >= optimal_threshold_km, drone_lifecycle_g, ebike_lifecycle_g)

    bin_edges = np.linspace(road_distances_km.min(), np.percentile(road_distances_km, 99), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(road_distances_km, bin_edges) - 1, 0, n_bins - 1)

    def _bin_mean(values, mask):
        if not mask.any():
            return np.nan
        if weights is None:
            return values[mask].mean()
        w = weights[mask]
        total_w = w.sum()
        return np.average(values[mask], weights=w) if total_w > 0 else np.nan

    ebike_means = np.array([_bin_mean(ebike_lifecycle_g, bin_idx == i) for i in range(n_bins)])
    mix_means = np.array([_bin_mean(mix_emission, bin_idx == i) for i in range(n_bins)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, ebike_means, color="#e74c3c", linewidth=2, label="All E-bike")
    ax.plot(bin_centers, mix_means, color="#3498db", linewidth=2, label="Optimal Mix")
    ax.axvline(optimal_threshold_km, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Threshold: {optimal_threshold_km:.1f} km")

    ax.set_xlabel("Trip Distance (km)", fontsize=12)
    ax.set_ylabel("Lifecycle Emission (g CO\u2082e)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if plot_title:
        ax.set_title("Emission vs Distance: All E-bike vs Optimal Mix", fontsize=14)

    plt.tight_layout()
    return _save_or_show(fig, save_path, dpi, for_latex, plot_title)


def _draw_lifecycle_distance_route_factor(
    ax,
    fig,
    od_df=None,
    plot_title: bool = False,
    cax=None,
    draw_heatmap: bool = True,
    contour_percentiles=None,
    mark_center: bool = False,
) -> None:
    """Draw lifecycle distance/route-factor comparison onto *ax*."""
    distances = np.linspace(0.5, 4, 30)
    detour_ratios = np.linspace(1.0, 1.8, 20)
    if contour_percentiles is None:
        contour_percentiles = (0.20, 0.40, 0.60, 0.80)

    if draw_heatmap:
        from emission.cal_single_delivery import LifeCycleEmissions

        lca = LifeCycleEmissions()
        emission_diff_matrix = np.zeros((len(detour_ratios), len(distances)))
        for i, r in enumerate(detour_ratios):
            for j, d in enumerate(distances):
                ebike_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d * r, n_orders_per_delivery=1, vehicle_type="ebike"
                )
                drone_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d, n_orders_per_delivery=1, vehicle_type="drone"
                )
                diff = (drone_result["total_per_order_g"] / ebike_result["total_per_order_g"] - 1) * 100
                emission_diff_matrix[i, j] = diff

        abs_max = max(abs(emission_diff_matrix.min()), abs(emission_diff_matrix.max()))
        im = ax.imshow(
            emission_diff_matrix,
            cmap="RdYlGn_r",
            aspect="auto",
            origin="lower",
            extent=[distances[0], distances[-1], detour_ratios[0], detour_ratios[-1]],
            vmin=-abs_max,
            vmax=abs_max,
        )

        cbar = fig.colorbar(im, cax=cax) if cax is not None else fig.colorbar(im, ax=ax)
        cbar.set_label("Drone emissions relative to e-bike (%)", fontsize=12)
    else:
        ax.set_xlim(distances[0], distances[-1])
        ax.set_ylim(detour_ratios[0], detour_ratios[-1])
        ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Air distance (km)", fontsize=12)
    ax.set_ylabel("Detour ratio (road / air distance)", fontsize=12)
    if plot_title:
        title = "Emission comparison: drone vs e-bike" if draw_heatmap else "Shanghai order distribution"
        ax.set_title(title, fontsize=14)

    if od_df is not None:
        from scipy.ndimage import gaussian_filter

        valid = od_df.dropna(subset=["distance_km", "road_distance_m", "annual_sales"]).copy()
        for col in ["distance_km", "road_distance_m", "annual_sales"]:
            valid[col] = pd.to_numeric(valid[col], errors="coerce")
        valid = valid.dropna(subset=["distance_km", "road_distance_m", "annual_sales"])
        valid = valid[valid["annual_sales"] > 0]
        valid = valid[(valid["distance_km"] >= distances[0]) & (valid["distance_km"] <= distances[-1])].copy()
        valid["detour"] = valid["road_distance_m"] / (valid["distance_km"] * 1000)
        valid = valid[(valid["detour"] >= detour_ratios[0]) & (valid["detour"] <= detour_ratios[-1])]

        if len(valid) > 10:
            n_bins = 100
            hist, xedges, yedges = np.histogram2d(
                valid["distance_km"].values,
                valid["detour"].values,
                bins=n_bins,
                range=[[distances[0], distances[-1]], [detour_ratios[0], detour_ratios[-1]]],
                weights=valid["annual_sales"].values,
            )
            Zi = gaussian_filter(hist.T, sigma=4.0)

            xi = 0.5 * (xedges[:-1] + xedges[1:])
            yi = 0.5 * (yedges[:-1] + yedges[1:])
            Xi, Yi = np.meshgrid(xi, yi)

            z_sorted = np.sort(Zi.ravel())[::-1]
            z_cumsum = np.cumsum(z_sorted) / z_sorted.sum()
            pcts = tuple(contour_percentiles)
            level_pairs = [
                (z_sorted[np.searchsorted(z_cumsum, p)], p)
                for p in pcts
            ]
            level_pairs = sorted(level_pairs, key=lambda pair: pair[0])
            kde_levels = [level for level, _ in level_pairs]

            cs = ax.contour(Xi, Yi, Zi, levels=kde_levels, colors="black",
                            linewidths=1.2 if not draw_heatmap else 1.0,
                            linestyles="solid", alpha=0.85 if not draw_heatmap else 0.7)
            fmt = {level: f"{int(round(p * 100))}%" for level, p in level_pairs}
            ax.clabel(cs, fmt=fmt, fontsize=7, inline=True)

            if mark_center:
                weights = valid["annual_sales"].to_numpy(dtype=float)
                center_x = np.average(valid["distance_km"].to_numpy(dtype=float), weights=weights)
                center_y = np.average(valid["detour"].to_numpy(dtype=float), weights=weights)
                ax.scatter(
                    center_x,
                    center_y,
                    s=48,
                    marker="o",
                    facecolor="#d73027",
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=6,
                    label="Weighted center",
                )

            ax.plot([], [], color="black", linewidth=1.0, linestyle="solid",
                    label="Order distribution (Shanghai)")
            ax.legend(loc="upper right", fontsize=9, framealpha=0.7)


def plot_lifecycle_distance_route_factor(
    od_df=None,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Shanghai order-density contours over air distance and detour ratio.

    Draws only the weighted center point and 30%/60% order-density contours,
    without the lifecycle background heatmap or colorbar.

    Args:
        od_df: OD pairs DataFrame with columns distance_km, road_distance_m, annual_sales.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_lifecycle_distance_route_factor(
        ax,
        fig,
        od_df=od_df,
        plot_title=plot_title,
        draw_heatmap=False,
        contour_percentiles=(0.30, 0.60),
        mark_center=True,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Lifecycle distance route factor heatmap saved to {save_path}")
    else:
        plt.show()


def _production_phase_parts_per_km(distance_km: float):
    """Return body/battery production amortization per round-trip vehicle-km."""
    from emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()
    rows = []

    specs = [
        (
            "E-bike",
            lca.ebike_p,
            lca.calculate_ebike_production_emissions(),
            lca.ebike_p.velocity,
        ),
        (
            "Drone",
            lca.drone_p,
            lca.calculate_drone_production_emissions(),
            lca.drone_p.v_level,
        ),
    ]
    for vehicle, p, prod, speed_kmh in specs:
        delivery_time_min = (2 * distance_km / speed_kmh) * 60 + p.loading_time_min
        deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
        annual_deliveries = deliveries_per_day * p.working_days_per_year
        round_trip_km = 2 * distance_km

        body_g_per_km = (
            prod["body_production_kg"] / p.lifespan
            / annual_deliveries
            * 1000
            / round_trip_km
        )
        battery_g_per_km = (
            prod["battery_production_kg"] / p.battery_lifespan
            / annual_deliveries
            * 1000
            / round_trip_km
        )
        rows.append(
            {
                "vehicle": vehicle,
                "body_g_per_km": body_g_per_km,
                "battery_g_per_km": battery_g_per_km,
            }
        )

    return pd.DataFrame(rows)


def plot_production_phase_per_km(
    save_path: str = None,
    figsize: tuple = (5.2, 3.6),
    for_latex: bool = False,
    plot_title: bool = False,
    distance_km: float = 1.0,
):
    """Stacked bar plot of body and battery production emissions per vehicle-km.

    Args:
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        distance_km: One-way delivery distance used for delivery-time amortization.
    """
    df = _production_phase_parts_per_km(distance_km)
    x = np.arange(len(df))
    body = df["body_g_per_km"].to_numpy()
    battery = df["battery_g_per_km"].to_numpy()
    totals = body + battery

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    body_color = "#7f8c8d"
    battery_color = "#2ca25f"
    edge_color = "#2f3437"

    ax.bar(
        x,
        body,
        width=0.56,
        color=body_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Body production",
    )
    ax.bar(
        x,
        battery,
        width=0.56,
        bottom=body,
        color=battery_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Battery production",
    )

    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + max(totals) * 0.035,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["vehicle"])
    ax.set_xlabel(f"Amortized using {distance_km:g} km one-way trips")
    ax.set_ylabel("Production emission (g CO$_2$e / vehicle-km)")
    ax.set_ylim(0, max(totals) * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=9,
    )
    if plot_title:
        ax.set_title("Production Phase Emissions per Kilometer")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Production phase per km barplot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def _end_of_life_phase_parts_per_km(distance_km: float):
    """Return body/battery end-of-life amortization per round-trip vehicle-km."""
    from emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()
    rows = []

    specs = [
        (
            "E-bike",
            lca.ebike_p,
            lca.calculate_ebike_production_emissions(),
            lca.ebike_p.velocity,
        ),
        (
            "Drone",
            lca.drone_p,
            lca.calculate_drone_production_emissions(),
            lca.drone_p.v_level,
        ),
    ]
    for vehicle, p, prod, speed_kmh in specs:
        delivery_time_min = (2 * distance_km / speed_kmh) * 60 + p.loading_time_min
        deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
        annual_deliveries = deliveries_per_day * p.working_days_per_year
        round_trip_km = 2 * distance_km

        body_g_per_km = (
            prod["body_disposal_kg"] / p.lifespan
            / annual_deliveries
            * 1000
            / round_trip_km
        )
        battery_g_per_km = (
            prod["battery_disposal_kg"] / p.battery_lifespan
            / annual_deliveries
            * 1000
            / round_trip_km
        )
        rows.append(
            {
                "vehicle": vehicle,
                "body_g_per_km": body_g_per_km,
                "battery_g_per_km": battery_g_per_km,
            }
        )

    return pd.DataFrame(rows)


def plot_end_of_life_phase_per_km(
    save_path: str = None,
    figsize: tuple = (5.2, 3.6),
    for_latex: bool = False,
    plot_title: bool = False,
    distance_km: float = 1.0,
):
    """Stacked bar plot of body and battery end-of-life emissions per vehicle-km.

    Args:
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        distance_km: One-way delivery distance used for delivery-time amortization.
    """
    df = _end_of_life_phase_parts_per_km(distance_km)
    x = np.arange(len(df))
    body = df["body_g_per_km"].to_numpy()
    battery = df["battery_g_per_km"].to_numpy()
    totals = body + battery

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    body_color = "#8b8f97"
    battery_color = "#5b8db8"
    edge_color = "#2f3437"

    ax.bar(
        x,
        body,
        width=0.56,
        color=body_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Body end-of-life",
    )
    ax.bar(
        x,
        battery,
        width=0.56,
        bottom=body,
        color=battery_color,
        edgecolor=edge_color,
        linewidth=0.8,
        label="Battery end-of-life",
    )

    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + max(totals) * 0.035,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["vehicle"])
    ax.set_xlabel(f"Amortized using {distance_km:g} km one-way trips")
    ax.set_ylabel("End-of-life emission (g CO$_2$e / vehicle-km)")
    ax.set_ylim(0, max(totals) * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=9,
    )
    if plot_title:
        ax.set_title("End-of-Life Phase Emissions per Kilometer")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"End-of-life phase per km barplot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_lifecycle_distance_detour_boxplot(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.25,
):
    """Heatmap of lifecycle emission difference with detour quantile mask bands.

    Args:
        od_df: OD pairs DataFrame with distance_km and road_distance_m columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each shared air-distance bin in km.
    """
    from emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()

    distance_min, distance_max = 0.2, 5.0
    heatmap_distance_min = 0.2
    detour_min, detour_max = 1.0, 1.8
    x_edges = np.arange(distance_min, distance_max + bin_width_km, bin_width_km)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    distances = np.linspace(heatmap_distance_min, distance_max, 140)
    detour_ratios = np.linspace(detour_min, detour_max, 90)

    def _emission_diff_matrix(distance_grid):
        matrix = np.zeros((len(detour_ratios), len(distance_grid)))
        for i, r in enumerate(detour_ratios):
            for j, d in enumerate(distance_grid):
                ebike_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d * r, n_orders_per_delivery=1, vehicle_type="ebike"
                )
                drone_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d, n_orders_per_delivery=1, vehicle_type="drone"
                )
                diff = (drone_result["total_per_order_g"] / ebike_result["total_per_order_g"] - 1) * 100
                matrix[i, j] = diff
        return matrix

    emission_diff_matrix = _emission_diff_matrix(distances)
    original_distances = np.linspace(heatmap_distance_min, 4, 30)
    original_emission_diff_matrix = _emission_diff_matrix(original_distances)

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list(
        "teal_yellow_orange",
        ["#006b6b", "#48c4b2", "#e5dfb6", "#f0a33a", "#df3b20"],
    )
    norm = TwoSlopeNorm(
        vmin=float(original_emission_diff_matrix.min()),
        vcenter=0.0,
        vmax=float(original_emission_diff_matrix.max()),
    )
    im = ax.imshow(
        emission_diff_matrix,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[heatmap_distance_min, distance_max, detour_min, detour_max],
        norm=norm,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Drone emissions relative to e-bike (%)", fontsize=12)

    X, Y = np.meshgrid(distances, detour_ratios)
    data_min = np.floor(emission_diff_matrix.min() / 10) * 10
    negative_levels = np.arange(data_min, -9, 10)
    contour_levels = [*negative_levels, 0, 40, 80]
    contour = ax.contour(
        X,
        Y,
        emission_diff_matrix,
        levels=contour_levels,
        colors="black",
        linewidths=[1.5 if lvl == 0 else 1.0 for lvl in contour_levels],
        linestyles=["-" if lvl == 0 else "--" for lvl in contour_levels],
        alpha=0.65,
        zorder=4,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt=lambda x: f"{x:+.0f}%")

    valid = od_df.dropna(subset=["distance_km", "road_distance_m"]).copy()
    valid = valid[
        (valid["distance_km"] > distance_min)
        & (valid["distance_km"] <= distance_max)
        & (valid["road_distance_m"] >= 0)
    ]
    valid["detour"] = valid["road_distance_m"] / (valid["distance_km"] * 1000)
    valid = valid[np.isfinite(valid["detour"])]

    if len(valid) > 0:
        from scipy.ndimage import gaussian_filter1d

        mask_x = np.linspace(distance_min, distance_max, 700)
        quantile_levels = np.array([0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95])
        quantile_curves = np.full((len(quantile_levels), len(x_centers)), np.nan)
        quantile_bands = [
            (None, 0, 0.66),
            (0, 1, 0.54),
            (1, 2, 0.42),
            (2, 3, 0.24),
            (3, 4, 0.00),
            (4, 5, 0.24),
            (5, 6, 0.42),
            (6, 7, 0.54),
            (7, None, 0.66),
        ]
        for x_i, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            if right >= distance_max:
                mask = (valid["distance_km"] >= left) & (valid["distance_km"] <= right)
            else:
                mask = (valid["distance_km"] >= left) & (valid["distance_km"] < right)
            values = valid.loc[mask, "detour"].to_numpy()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            quantile_curves[:, x_i] = np.quantile(values, quantile_levels)

        observed = np.flatnonzero(np.isfinite(quantile_curves[0]))
        if observed.size > 1:
            smooth_curves = np.full((len(quantile_levels), len(mask_x)), np.nan)
            for q_i in range(len(quantile_levels)):
                curve = quantile_curves[q_i]
                curve = np.interp(x_centers, x_centers[observed], curve[observed])
                curve = gaussian_filter1d(curve, sigma=1.0, mode="nearest")
                smooth_curves[q_i] = np.interp(mask_x, x_centers, curve)

            for low_i, high_i, alpha in quantile_bands:
                if alpha <= 0:
                    continue
                y_low = np.full_like(mask_x, detour_min) if low_i is None else smooth_curves[low_i]
                y_high = np.full_like(mask_x, detour_max) if high_i is None else smooth_curves[high_i]
                ax.fill_between(
                    mask_x,
                    y_low,
                    y_high,
                    color="white",
                    alpha=alpha,
                    linewidth=0,
                    zorder=5,
                )

    ax.set_xlim(distance_min, distance_max)
    ax.set_ylim(detour_min, detour_max)
    ax.set_xlabel("Air distance (km)", fontsize=12)
    ax.set_ylabel("Detour ratio (road / air distance)", fontsize=12)
    if plot_title:
        ax.set_title("Emission comparison with observed detour distribution", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Lifecycle distance detour quantile mask heatmap saved to {save_path}")
    else:
        plt.show()
