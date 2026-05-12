"""Named policy/grid scenarios per region.

Six strategies × 2 anchor kappa values = 12 rows per city. Strategies are:

1. ``all_ebike`` — threshold = +inf (no drones)
2. ``all_drone`` — threshold = 0
3. ``optimal_mixed`` — best of the 12-point threshold sweep, current GEF
4. ``conservative_2km`` — fixed threshold = 2.0 km, current GEF
5. ``optimal_mixed_2030`` — best threshold under GEF × 0.5 (TODO confirm
   IEA WEO 2024 STEPS multiplier)
6. ``optimal_mixed_2060`` — best threshold under GEF × 0.2 (TODO same)

Reductions are reported against:

* ``reduction_vs_all_ebike_same_kappa_pct`` — vs strategy 1 at the same kappa
  (operational baseline a courier would compare against)
* ``reduction_vs_anchor_ebike_pct`` — vs the all-ebike total at kappa=1.0,
  current GEF (the "status quo" anchor)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from emission.cal_single_delivery import DroneParams, EbikeParams

from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)

KAPPA_ANCHORS = (0.7, 1.0)
GEF_MULT_2030 = 0.5  # TODO replace with IEA WEO 2024 STEPS China power-sector multiplier
GEF_MULT_2060 = 0.2  # TODO same


def _threshold_sweep(road_distances_km: np.ndarray) -> np.ndarray:
    middle = np.linspace(
        max(0.3, np.percentile(road_distances_km, 5)),
        np.percentile(road_distances_km, 95),
        10,
    )
    return np.concatenate([[-0.01], middle, [np.inf]])


def _per_od_emissions(
    road_distances_km: np.ndarray,
    straight_distances_km: np.ndarray,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    province_gef_kwh: float,
    kappa: float,
    gef_mult: float,
    payload_kg: float,
    rider_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)
    ebike_g = compute_ebike_per_order_g(
        road_distances_km,
        ebike_pp,
        EbikeScenario(kappa=kappa, gef_mult=gef_mult,
                       payload_kg=payload_kg, rider_weight=rider_weight),
    )
    drone_g = compute_drone_per_order_g(
        straight_distances_km,
        drone_pp,
        DroneScenario(gef_mult=gef_mult, payload_kg=payload_kg),
    )
    return ebike_g, drone_g


def _evaluate_strategy(
    strategy: str,
    road_distances_km: np.ndarray,
    ebike_g: np.ndarray,
    drone_g: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Apply the strategy decision rule, return total mixed kg + threshold + drone ratio."""
    n = len(road_distances_km)
    if strategy == "all_ebike":
        chosen_threshold = float("inf")
        use_drone = np.zeros(n, dtype=bool)
    elif strategy == "all_drone":
        chosen_threshold = 0.0
        use_drone = road_distances_km >= 0.0  # all True
    elif strategy == "conservative_2km":
        chosen_threshold = 2.0
        use_drone = road_distances_km >= 2.0
    elif strategy.startswith("optimal_mixed"):
        # Pick the threshold that minimizes mixed total
        best_total = None
        chosen_threshold = float("nan")
        chosen_use_drone = None
        for t in thresholds:
            ud = road_distances_km >= t
            mixed = float(np.where(ud, drone_g, ebike_g).sum())
            if best_total is None or mixed < best_total:
                best_total = mixed
                chosen_threshold = float(t)
                chosen_use_drone = ud
        use_drone = chosen_use_drone
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    mixed_total_g = float(np.where(use_drone, drone_g, ebike_g).sum())
    drone_ratio_pct = float(use_drone.sum()) / n * 100.0
    return {
        "threshold_km": chosen_threshold,
        "drone_ratio_pct": drone_ratio_pct,
        "total_lifecycle_kg": mixed_total_g / 1000.0,
        "_total_g": mixed_total_g,
    }


def run_named_scenarios_for_region(
    od_df: pd.DataFrame,
    region_key: str,
    province: str,
    province_gef_kwh: float,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    rider_weight: float = 70.0,
    payload_kg: float = 1.0,
) -> pd.DataFrame:
    """Compute named-strategy table for a single region (12 rows)."""
    road_distances_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
    straight_distances_km = od_df["distance_km"].to_numpy(dtype=float)
    if len(road_distances_km) == 0:
        return pd.DataFrame()
    thresholds = _threshold_sweep(road_distances_km)

    # Anchor reference: all-ebike total at kappa=1.0, current GEF.
    anchor_ebike_g, _ = _per_od_emissions(
        road_distances_km, straight_distances_km,
        ebike_params, drone_params, province_gef_kwh,
        kappa=1.0, gef_mult=1.0,
        payload_kg=payload_kg, rider_weight=rider_weight,
    )
    anchor_total_g = float(anchor_ebike_g.sum())

    rows = []
    for kappa in KAPPA_ANCHORS:
        # Pre-compute per-OD emissions for current-GEF and the two future-GEF cases.
        gef_cases = [
            ("current", 1.0),
            ("2030", GEF_MULT_2030),
            ("2060", GEF_MULT_2060),
        ]
        emissions_by_gef = {}
        for label, gef_mult in gef_cases:
            ebike_g, drone_g = _per_od_emissions(
                road_distances_km, straight_distances_km,
                ebike_params, drone_params, province_gef_kwh,
                kappa=kappa, gef_mult=gef_mult,
                payload_kg=payload_kg, rider_weight=rider_weight,
            )
            emissions_by_gef[label] = (ebike_g, drone_g)

        # Same-kappa all-ebike denominator
        same_kappa_ebike_total_g = float(emissions_by_gef["current"][0].sum())

        strategies = [
            ("all_ebike", "current", 1.0),
            ("all_drone", "current", 1.0),
            ("optimal_mixed", "current", 1.0),
            ("conservative_2km", "current", 1.0),
            ("optimal_mixed_2030", "2030", GEF_MULT_2030),
            ("optimal_mixed_2060", "2060", GEF_MULT_2060),
        ]
        for strategy, gef_label, gef_mult in strategies:
            ebike_g, drone_g = emissions_by_gef[gef_label]
            res = _evaluate_strategy(strategy, road_distances_km, ebike_g, drone_g, thresholds)
            mixed_g = res["_total_g"]
            red_anchor = (
                (anchor_total_g - mixed_g) / anchor_total_g * 100.0
                if anchor_total_g > 0 else 0.0
            )
            red_same_kappa = (
                (same_kappa_ebike_total_g - mixed_g) / same_kappa_ebike_total_g * 100.0
                if same_kappa_ebike_total_g > 0 else 0.0
            )
            rows.append({
                "region_key": region_key,
                "province": province,
                "kappa": kappa,
                "strategy": strategy,
                "gef_mult": gef_mult,
                "threshold_km": res["threshold_km"],
                "drone_ratio_pct": res["drone_ratio_pct"],
                "total_lifecycle_kg": res["total_lifecycle_kg"],
                "reduction_vs_anchor_ebike_pct": float(red_anchor),
                "reduction_vs_all_ebike_same_kappa_pct": float(red_same_kappa),
            })

    return pd.DataFrame(rows)
