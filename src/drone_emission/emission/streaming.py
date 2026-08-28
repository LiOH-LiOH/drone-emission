"""Bounded-memory OD emission calculation and durable CSV publication."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.utils.atomic import atomic_open


CHUNK_ROWS = 250_000
OUTPUT_COLUMNS = (
    "drone_emission_kg",
    "ebike_emission_kg",
    "drone_lifecycle_g",
    "ebike_lifecycle_g",
    "drone_time_min",
    "ebike_time_min",
    "ebike_kappa",
    "grid_emission_factor",
    "ebike_lifecycle_production_g",
    "ebike_lifecycle_use_g",
    "ebike_lifecycle_eol_g",
    "drone_lifecycle_production_g",
    "drone_lifecycle_use_g",
    "drone_lifecycle_eol_g",
)


def _calculate_chunk(
    frame: pd.DataFrame,
    *,
    lca,
    payload_kg: float,
    rider_weight: float,
    ebike_params,
    drone_params,
    ebike_kappa: float,
    grid_factor: float,
) -> pd.DataFrame:
    values = {column: [] for column in OUTPUT_COLUMNS}
    for row in frame.itertuples(index=False):
        distance_km = float(row.distance_km)
        road_distance_km = float(row.road_distance_m) / 1000.0
        ebike_distance_km = road_distance_km * ebike_kappa
        drone_use = lca.calculate_drone_use_phase_emissions(distance_km, payload_kg)
        ebike_use = lca.calculate_ebike_use_phase_emissions(
            ebike_distance_km, payload_kg, rider_weight
        )
        drone_lifecycle = lca.calculate_lifecycle_emissions_per_order_simple(
            distance_km, n_orders_per_delivery=1, vehicle_type="drone"
        )
        ebike_lifecycle = lca.calculate_lifecycle_emissions_per_order_simple(
            ebike_distance_km, n_orders_per_delivery=1, vehicle_type="ebike"
        )
        drone_time = (
            2 * (drone_params.h_takeoff / 1000) / drone_params.v_vertical
            + distance_km / drone_params.v_level
            + drone_params.hover_time / 3600
        ) * 60
        values["drone_emission_kg"].append(drone_use["emissions_kg_co2e"])
        values["ebike_emission_kg"].append(ebike_use["emissions_kg_co2e"])
        values["drone_lifecycle_g"].append(drone_lifecycle["total_per_order_g"])
        values["ebike_lifecycle_g"].append(ebike_lifecycle["total_per_order_g"])
        values["drone_time_min"].append(drone_time)
        values["ebike_time_min"].append(road_distance_km / ebike_params.velocity * 60)
        values["ebike_kappa"].append(ebike_kappa)
        values["grid_emission_factor"].append(grid_factor)
        values["ebike_lifecycle_production_g"].append(
            ebike_lifecycle["production_per_order_g"]
        )
        values["ebike_lifecycle_use_g"].append(ebike_lifecycle["use_per_order_g"])
        values["ebike_lifecycle_eol_g"].append(
            ebike_lifecycle["end_of_life_per_order_g"]
        )
        values["drone_lifecycle_production_g"].append(
            drone_lifecycle["production_per_order_g"]
        )
        values["drone_lifecycle_use_g"].append(drone_lifecycle["use_per_order_g"])
        values["drone_lifecycle_eol_g"].append(
            drone_lifecycle["end_of_life_per_order_g"]
        )
    for column in OUTPUT_COLUMNS:
        frame[column] = values[column]
    return frame


def process_emission_csv(
    path: str | Path,
    *,
    region_info,
    lca,
    payload_kg: float,
    rider_weight: float,
    ebike_params,
    drone_params,
    ebike_kappa: float,
    grid_factor: float,
    chunk_rows: int = CHUNK_ROWS,
) -> pd.DataFrame:
    """Recompute emission columns by chunk and reduce weighted city statistics."""
    path = Path(path)
    totals = {column: 0.0 for column in OUTPUT_COLUMNS if column.endswith("_g")}
    total_drone_use = 0.0
    total_ebike_use = 0.0
    total_orders = 0.0
    oracle_total_g = 0.0
    oracle_drone_orders = 0.0
    min_road_km = np.inf
    max_road_km = -np.inf
    row_count = 0

    with atomic_open(path, "w", encoding="utf-8", newline="") as handle:
        first = True
        for frame in pd.read_csv(path, chunksize=chunk_rows):
            required = {"distance_km", "road_distance_m", "annual_sales"}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise ValueError(f"{path} is missing emission inputs: {missing}")
            frame = _calculate_chunk(
                frame,
                lca=lca,
                payload_kg=payload_kg,
                rider_weight=rider_weight,
                ebike_params=ebike_params,
                drone_params=drone_params,
                ebike_kappa=ebike_kappa,
                grid_factor=grid_factor,
            )
            annual_sales = pd.to_numeric(frame["annual_sales"], errors="coerce").to_numpy()
            if not np.isfinite(annual_sales).all() or (annual_sales <= 0).any():
                raise ValueError(f"{path} has invalid annual_sales")
            total_orders += float(annual_sales.sum())
            row_count += len(frame)
            road_km = pd.to_numeric(frame["road_distance_m"], errors="coerce").to_numpy() / 1000
            if not np.isfinite(road_km).all():
                raise ValueError(f"{path} has non-finite road distances")
            min_road_km = min(min_road_km, float(road_km.min()))
            max_road_km = max(max_road_km, float(road_km.max()))
            total_drone_use += float(np.sum(frame["drone_emission_kg"] * annual_sales))
            total_ebike_use += float(np.sum(frame["ebike_emission_kg"] * annual_sales))
            for column in totals:
                totals[column] += float(np.sum(frame[column].to_numpy() * annual_sales))
            drone_lifecycle = frame["drone_lifecycle_g"].to_numpy()
            ebike_lifecycle = frame["ebike_lifecycle_g"].to_numpy()
            choose_drone = drone_lifecycle < ebike_lifecycle
            oracle_total_g += float(
                np.sum(np.where(choose_drone, drone_lifecycle, ebike_lifecycle) * annual_sales)
            )
            oracle_drone_orders += float(annual_sales[choose_drone].sum())
            frame.to_csv(handle, index=False, header=first)
            first = False
        if row_count == 0 or total_orders <= 0:
            raise ValueError(f"{path} has no positive OD rows")

    total_drone_lifecycle = totals["drone_lifecycle_g"] / 1000
    total_ebike_lifecycle = totals["ebike_lifecycle_g"] / 1000
    oracle_lifecycle = oracle_total_g / 1000
    lifecycle_reduction = (
        (total_ebike_lifecycle - total_drone_lifecycle) / total_ebike_lifecycle * 100
    )
    oracle_reduction = (
        (total_ebike_lifecycle - oracle_lifecycle) / total_ebike_lifecycle * 100
    )
    oracle_ratio = oracle_drone_orders / total_orders * 100
    stats = {
        "total_ebike_lifecycle": total_ebike_lifecycle,
        "total_drone_lifecycle": total_drone_lifecycle,
        "total_ebike_production_emission_kg": totals["ebike_lifecycle_production_g"] / 1000,
        "total_ebike_use_phase_emission_kg": totals["ebike_lifecycle_use_g"] / 1000,
        "total_ebike_end_of_life_emission_kg": totals["ebike_lifecycle_eol_g"] / 1000,
        "total_drone_production_emission_kg": totals["drone_lifecycle_production_g"] / 1000,
        "total_drone_use_phase_emission_kg": totals["drone_lifecycle_use_g"] / 1000,
        "total_drone_end_of_life_emission_kg": totals["drone_lifecycle_eol_g"] / 1000,
        "drone_minus_ebike_lifecycle_kg": total_drone_lifecycle - total_ebike_lifecycle,
        "lifecycle_all_drone_reduction_pct": lifecycle_reduction,
        "ebike_kappa": ebike_kappa,
        "oracle_min_lifecycle_kg": oracle_lifecycle,
        "oracle_min_reduction_vs_ebike_pct": oracle_reduction,
        "oracle_drone_order_ratio_pct": oracle_ratio,
    }
    for key, value in stats.items():
        region_info.set_stat(key, value)
    for stale_key in (
        "max_lifecycle_reduction_pct",
        "optimal_threshold_km",
        "optimal_drone_ratio_pct",
    ):
        region_info.stats.pop(stale_key, None)
    print(f"Emission and time data saved to {path} ({row_count:,} rows)")
    print(f"All-drone use-phase emissions: {total_drone_use:.2f} kg CO2e")
    print(f"All-ebike use-phase emissions: {total_ebike_use:.2f} kg CO2e")
    print(f"Road distance range: {min_road_km:.2f} - {max_road_km:.2f} km")
    return pd.DataFrame(
        [{
            "total_drone_lifecycle_kg": total_drone_lifecycle,
            "total_ebike_lifecycle_kg": total_ebike_lifecycle,
            "total_drone_production_emission_kg": stats["total_drone_production_emission_kg"],
            "total_drone_use_phase_emission_kg": stats["total_drone_use_phase_emission_kg"],
            "total_drone_end_of_life_emission_kg": stats["total_drone_end_of_life_emission_kg"],
            "total_ebike_production_emission_kg": stats["total_ebike_production_emission_kg"],
            "total_ebike_use_phase_emission_kg": stats["total_ebike_use_phase_emission_kg"],
            "total_ebike_end_of_life_emission_kg": stats["total_ebike_end_of_life_emission_kg"],
            "drone_minus_ebike_lifecycle_kg": stats["drone_minus_ebike_lifecycle_kg"],
            "oracle_min_lifecycle_kg": oracle_lifecycle,
            "oracle_min_reduction_vs_ebike_pct": oracle_reduction,
            "oracle_drone_order_ratio_pct": oracle_ratio,
        }]
    )
