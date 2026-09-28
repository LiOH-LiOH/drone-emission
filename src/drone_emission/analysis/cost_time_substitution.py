"""Independent OD-level emission, transport-cost and transport-time optima."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    compute_drone_per_order_parts_g,
    compute_ebike_per_order_parts_g,
)
from drone_emission.utils.atomic import atomic_open


ANALYSIS_VERSION = 1
STRATEGIES = ("all_ebike", "all_drone", "emission_optimal", "cost_optimal", "time_optimal")
OD_COLUMNS = (
    "distance_km", "road_distance_m", "annual_sales", "ebike_kappa",
    "ebike_lifecycle_g", "drone_lifecycle_g", "ebike_time_min", "drone_time_min",
)
TOTAL_COLUMNS = (
    "annual_orders", "drone_annual_orders", "annual_lifecycle_kg",
    "annual_transport_cost_cny", "annual_transport_minutes",
)


@dataclass(frozen=True)
class CostParameters:
    """Li 2025 transport boundary; S7 battery stock is per vehicle, not cells."""

    electricity_cny_per_kwh: float = 0.5
    charging_efficiency: float = 0.81
    rider_fee_share: float = 0.7
    base_delivery_fee_cny: float = 2.8
    base_delivery_distance_km: float = 3.0
    extra_delivery_fee_cny_per_km: float = 1.4
    battery_life_years: float = 2.0
    battery_salvage_fraction: float = 0.05
    ebike_battery_price_cny: float = 400.0
    drone_battery_price_cny: float = 1000.0
    ebike_battery_stock_per_vehicle: float = 2.0
    drone_battery_stock_per_vehicle: float = 3.0

    def __post_init__(self):
        values = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Transport cost parameters must be finite and nonnegative")
        if not 0 < self.charging_efficiency <= 1 or self.battery_life_years <= 0:
            raise ValueError("Charging efficiency must be in (0, 1]; battery life > 0")
        if self.battery_salvage_fraction > 1 or self.rider_fee_share > 1:
            raise ValueError("Salvage and rider shares cannot exceed one")


def transport_cost_parts(road_km, air_km, kappa, config: dict) -> dict:
    """Return CNY/order components using existing round-trip resource allocation."""
    cost = CostParameters(**config.get("transport_cost", {}))
    ebike, drone = create_ebike_params(config), create_drone_params(config)
    emission = config.get("emission_params", {})
    payload = emission.get("payload_kg", 1.0)
    ebike_parts = compute_ebike_per_order_parts_g(
        road_km, ebike,
        EbikeScenario(kappa=kappa, payload_kg=payload,
                      rider_weight=emission.get("rider_weight", 70.0)),
    )
    drone_parts = compute_drone_per_order_parts_g(
        air_km, drone, DroneScenario(payload_kg=payload),
    )
    rider = cost.rider_fee_share * (
        cost.base_delivery_fee_cny
        + cost.extra_delivery_fee_cny_per_km
        * np.maximum(np.asarray(road_km) - cost.base_delivery_distance_km, 0)
    )
    result = {}
    for mode, params, distance, speed, parts, labor in (
        ("ebike", ebike, np.asarray(road_km) * kappa, ebike.velocity, ebike_parts, rider),
        ("drone", drone, np.asarray(air_km), drone.v_level, drone_parts, np.zeros_like(rider)),
    ):
        cycle_minutes = 2 * distance / speed * 60 + params.loading_time_min
        annual_orders = params.working_hours_per_day * 60 * params.working_days_per_year / cycle_minutes
        battery = (
            (1 - cost.battery_salvage_fraction)
            * getattr(cost, f"{mode}_battery_price_cny")
            * getattr(cost, f"{mode}_battery_stock_per_vehicle")
            / cost.battery_life_years / annual_orders
        )
        electricity = parts["energy_kwh"] / cost.charging_efficiency * cost.electricity_cny_per_kwh
        result[mode] = {
            "rider_cny": labor, "electricity_cny": electricity,
            "battery_cny": battery, "total_cny": labor + electricity + battery,
        }
    return result


def summarize_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive order-weighted means and changes from five strategy totals."""
    frame = frame.copy()
    orders = frame["annual_orders"]
    frame["drone_replacement_rate_pct"] = frame["drone_annual_orders"] / orders * 100
    frame["mean_transport_cost_cny"] = frame["annual_transport_cost_cny"] / orders
    frame["mean_transport_time_min"] = frame["annual_transport_minutes"] / orders
    baseline = frame.loc[frame["strategy"].eq("all_ebike")].iloc[0]
    for metric in ("annual_lifecycle_kg", "annual_transport_cost_cny", "mean_transport_time_min"):
        frame[f"{metric}_change_vs_ebike_pct"] = (
            (frame[metric] / baseline[metric] - 1) * 100
            if baseline[metric] != 0 else np.nan
        )
    return frame


def calculate_city_strategies(od_path: Path, stats: pd.Series, config: dict, *, chunk_rows=250_000):
    """Reduce all objectives on the same valid, sales-weighted OD sample."""
    totals = np.zeros((len(STRATEGIES), len(TOTAL_COLUMNS)))
    row_count = excluded_rows = 0
    excluded_orders = 0.0
    for chunk in pd.read_csv(od_path, usecols=list(OD_COLUMNS), chunksize=chunk_rows):
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        row_count += len(chunk)
        valid = np.isfinite(chunk.to_numpy()).all(axis=1)
        valid &= (chunk[["distance_km", "road_distance_m", "annual_sales", "ebike_kappa"]] > 0).all(axis=1)
        valid &= (chunk[["ebike_lifecycle_g", "drone_lifecycle_g", "ebike_time_min", "drone_time_min"]] >= 0).all(axis=1)
        excluded_rows += int((~valid).sum())
        sales = chunk["annual_sales"].to_numpy()
        excluded_orders += float(sales[(~valid) & np.isfinite(sales) & (sales > 0)].sum())
        chunk = chunk.loc[valid]
        if chunk.empty:
            continue
        weights = chunk["annual_sales"].to_numpy()
        costs = transport_cost_parts(
            chunk["road_distance_m"].to_numpy() / 1000,
            chunk["distance_km"].to_numpy(), chunk["ebike_kappa"].to_numpy(), config,
        )
        ebike = np.column_stack((chunk["ebike_lifecycle_g"] / 1000,
                                 costs["ebike"]["total_cny"], chunk["ebike_time_min"]))
        drone = np.column_stack((chunk["drone_lifecycle_g"] / 1000,
                                 costs["drone"]["total_cny"], chunk["drone_time_min"]))
        if not np.isfinite(ebike).all() or not np.isfinite(drone).all():
            raise ValueError(f"Non-finite calculated metrics: {od_path}")
        masks = (np.zeros(len(chunk), dtype=bool), np.ones(len(chunk), dtype=bool),
                 *(drone[:, index] < ebike[:, index] for index in range(3)))
        for index, choose_drone in enumerate(masks):
            totals[index, 0] += weights.sum()
            totals[index, 1] += weights[choose_drone].sum()
            totals[index, 2:] += (np.where(choose_drone[:, None], drone, ebike) * weights[:, None]).sum(axis=0)
    if totals[0, 0] <= 0:
        raise ValueError(f"No valid weighted orders: {od_path}")
    frame = pd.DataFrame(totals, columns=TOTAL_COLUMNS)
    frame.insert(0, "strategy", STRATEGIES)
    frame = summarize_totals(frame)
    frame.insert(0, "region_name_en", stats["region_name_en"])
    frame.insert(0, "region_key", stats["region_key"])
    indexed = frame.set_index("strategy")
    comparisons = {}
    for strategy, metric, reference in (
        ("all_ebike", "annual_lifecycle_kg", "total_ebike_lifecycle"),
        ("emission_optimal", "annual_lifecycle_kg", "oracle_min_lifecycle_kg"),
        ("emission_optimal", "drone_replacement_rate_pct", "oracle_drone_order_ratio_pct"),
    ):
        observed, expected = float(indexed.loc[strategy, metric]), float(stats[reference])
        matches = bool(np.isclose(observed, expected, rtol=1e-9, atol=1e-6))
        comparisons[reference] = {"observed": observed, "reference": expected, "matches": matches}
        if excluded_rows == 0 and not matches:
            raise ValueError(f"{stats['region_key']}: {reference} differs from main experiment")
    return frame, {
        "region_key": str(stats["region_key"]), "od_rows": row_count,
        "excluded_rows": excluded_rows, "excluded_positive_annual_orders": excluded_orders,
        "main_experiment_comparable": excluded_rows == 0,
        "main_experiment_comparison": comparisons,
    }


def run_cost_time_analysis(config: dict, *, force=False, chunk_rows=250_000) -> tuple[Path, Path, Path]:
    """Write city/cohort tables and a compact source and validation report."""
    root = Path(config["work_dir"]) / "analysis" / "cost_time_substitution"
    paths = tuple(root / name for name in ("city_strategies.csv", "cohort_strategies.csv", "report.json"))
    data_root = Path(config["data_dir"])
    stats_path = data_root / "all_region_stats.csv"
    stats = pd.read_csv(stats_path)
    if "region_level" in stats:
        stats = stats.loc[stats["region_level"].eq("city")]
    keys = locked_region_keys()
    stats = stats.loc[stats["region_key"].isin(keys)].set_index("region_key", drop=False)
    if len(stats) != len(keys) or set(stats.index) != set(keys):
        raise ValueError("Cost/time analysis inputs do not match the locked paper cohort")
    od_paths = {
        key: data_root / "city" / str(stats.loc[key, "region_name_en"]).lower()
        / f"{str(stats.loc[key, 'region_name_en']).lower()}_od_pairs.csv"
        for key in keys
    }
    inputs = [stats_path, *od_paths.values()]
    signature = {
        "analysis_version": ANALYSIS_VERSION, "cohort_sha256": cohort_hash(),
        "transport_cost": asdict(CostParameters(**config.get("transport_cost", {}))),
        "emission_params": config.get("emission_params", {}),
        "inputs": [{"path": str(path.resolve()), "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns} for path in inputs],
    }
    if not force and all(path.is_file() for path in paths):
        if json.loads(paths[2].read_text())["signature"] == signature:
            return paths
    frames, reports = [], []
    for index, key in enumerate(keys, 1):
        frame, report = calculate_city_strategies(od_paths[key], stats.loc[key], config, chunk_rows=chunk_rows)
        frames.append(frame)
        reports.append(report)
        print(f"Cost/time analysis: {index}/{len(keys)} {key}", flush=True)
    cities = pd.concat(frames, ignore_index=True)
    cohort = cities.groupby("strategy", sort=False)[list(TOTAL_COLUMNS)].sum().reset_index()
    cohort = summarize_totals(cohort)
    cohort.insert(0, "scope", "locked_paper_cohort")
    report = {
        "signature": signature, "city_count": len(keys),
        "source_doi": "10.1016/j.scs.2025.106134",
        "cost_boundary": "Transport only: rider remuneration, charging electricity, replacement batteries. No capital, common collection, cabinets, separate supervision or repair.",
        "battery_stock_mapping": "S7 economic stock / vehicle count: ebike 200/100=2; drone 300/100=3. Not S4 physical battery counts.",
        "allocation": "Existing LCA round-trip cycle plus loading time; working hours and days from existing vehicle params. Battery net price annualized over economic lifetime.",
        "time_boundary": "Stored one-way transport time, including drone vertical flight and hover; excludes return, waiting and handling.",
        "assignment": "Independent per-OD minima; ties stay with ebike; no fleet-capacity constraints.",
        "weighting": "annual_sales; all three objectives share the same valid OD sample; cohort totals are not a national extrapolation.",
        "change_columns": "Percent change relative to all-ebike; negative means lower. Blank when baseline is zero.",
        "cities": reports,
    }
    for path, frame in zip(paths[:2], (cities, cohort)):
        with atomic_open(path, "w", newline="") as handle:
            frame.to_csv(handle, index=False)
    with atomic_open(paths[2], "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return paths
