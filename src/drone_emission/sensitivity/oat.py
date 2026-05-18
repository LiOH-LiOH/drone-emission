"""One-at-a-time (OAT) sensitivity sweep, per region.

Produces one row per (region, scenario_id) — a baseline plus deviations on
each of 11 features:

* kappa (e-bike bundling)               levels: 0.6, 0.7, 0.8, 0.9 (baseline 1.0)
* gef_mult                              levels: 0.5, 0.8, 1.2 (baseline 1.0)
* drone_energy_mult                     levels: 0.75, 1.25, 1.5 (baseline 1.0)
* payload_kg                            levels: 0.5, 2.0 (baseline 1.0)
* drone_lifespan_yr                     levels: 2, 6 (baseline 4)
* ebike_lifespan_yr                     levels: 2, 6 (baseline 4)
* drone_cruise_kmh                      levels: 40, 80 (baseline 60)
* drone_body_kg                         levels: 6.0, 14.0 (baseline 10.1)
* drone_hover_s                         levels: 15, 60 (baseline 30)
* drone_battery_lifespan_yr             levels: 1, 3 (baseline 2)
* headwind_mps (cruise-leg constant)    levels: 3.5, 7.0 (baseline 0.0)

For each scenario the threshold sweep matches ``emission/api.py``: 12 points
(-0.01, p5..p95 of road distances, +inf). Reductions are reported against:

* the e-bike-only total under the SAME scenario (`reduction_vs_scenario_pct`)
* the e-bike-only total under the ANCHOR scenario (all features at baseline) —
  `reduction_vs_anchor_pct`. The anchor uses the city's actual provincial GEF;
  only the multiplier is fixed at 1.

Aggregation matches the existing emission stage: each OD pair contributes
once, unweighted by ``annual_sales`` (the reduction ratio is invariant under
sales weighting since both modes share the OD set).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np
import pandas as pd

from emission.cal_single_delivery import DroneParams, EbikeParams

from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)

# Baseline values (the OAT anchor). Defaults reflect DroneParams / EbikeParams
# defaults so the anchor exactly matches the main pipeline's headline scenario.
BASELINE = {
    "kappa": 1.0,
    "gef_mult": 1.0,
    "drone_energy_mult": 1.0,
    "payload_kg": 1.0,
    "drone_lifespan_yr": 4,
    "ebike_lifespan_yr": 4,
    "drone_cruise_kmh": 60.0,             # = DroneParams.v_level
    "drone_body_kg": 10.1,                # = DroneParams.w_body
    "drone_hover_s": 30.0,                # = DroneParams.hover_time
    "drone_battery_lifespan_yr": 2,       # = DroneParams.battery_lifespan
    "headwind_mps": 0.0,                  # = DroneParams.headwind_speed_mps
}

# Deviating levels per feature (baseline value omitted)
FEATURE_LEVELS = {
    "kappa": [0.6, 0.7, 0.8, 0.9],
    "gef_mult": [0.5, 0.8, 1.2],
    "drone_energy_mult": [0.75, 1.25, 1.5],
    "payload_kg": [0.5, 2.0],
    "drone_lifespan_yr": [2, 6],
    "ebike_lifespan_yr": [2, 6],
    "drone_cruise_kmh": [40.0, 80.0],
    "drone_body_kg": [6.0, 14.0],
    "drone_hover_s": [15.0, 60.0],
    "drone_battery_lifespan_yr": [1, 3],
    "headwind_mps": [3.5, 7.0],
}


def _threshold_sweep(road_distances_km: np.ndarray) -> np.ndarray:
    """12-point threshold list matching emission/api.py:206-216."""
    middle = np.linspace(
        max(0.3, np.percentile(road_distances_km, 5)),
        np.percentile(road_distances_km, 95),
        10,
    )
    return np.concatenate([[-0.01], middle, [np.inf]])


def _best_threshold_metrics(
    road_distances_km: np.ndarray,
    ebike_g: np.ndarray,
    drone_g: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Iterate the threshold sweep, return optimal threshold + reduction metrics.

    Reduction here is *vs the e-bike-only total under the same per-OD emissions*
    (i.e. the scenario-specific baseline). The "vs anchor" comparison is done
    by the caller using a fixed anchor total.
    """
    n = len(road_distances_km)
    ebike_only_total_g = ebike_g.sum()
    drone_only_total_g = drone_g.sum()

    best = {
        "best_threshold_km": float("nan"),
        "best_drone_ratio_pct": float("nan"),
        "best_mixed_total_g": float("nan"),
        "best_reduction_vs_scenario_pct": float("-inf"),
    }
    for t in thresholds:
        use_drone = road_distances_km >= t
        mixed = float(np.where(use_drone, drone_g, ebike_g).sum())
        drone_ratio = float(use_drone.sum()) / n * 100.0
        if ebike_only_total_g > 0:
            reduction = (ebike_only_total_g - mixed) / ebike_only_total_g * 100.0
        else:
            reduction = 0.0
        if reduction > best["best_reduction_vs_scenario_pct"]:
            best["best_threshold_km"] = float(t)
            best["best_drone_ratio_pct"] = drone_ratio
            best["best_mixed_total_g"] = mixed
            best["best_reduction_vs_scenario_pct"] = reduction
    best["ebike_only_total_g"] = float(ebike_only_total_g)
    best["drone_only_total_g"] = float(drone_only_total_g)
    return best


def _emissions_for_scenario(
    road_distances_km: np.ndarray,
    straight_distances_km: np.ndarray,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    rider_weight: float,
    kappa: float,
    gef_mult: float,
    drone_energy_mult: float,
    payload_kg: float,
    ebike_lifespan_yr: float,
    drone_lifespan_yr: float,
    drone_cruise_kmh: float,
    drone_body_kg: float,
    drone_hover_s: float,
    drone_battery_lifespan_yr: float,
    headwind_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Per-scenario drone params: cruise speed, body mass, hover time, and
    # headwind enter the physics directly; battery lifespan is passed through
    # the DroneScenario override field instead.
    drone_params_eff = replace(
        drone_params,
        v_level=drone_cruise_kmh,
        w_body=drone_body_kg,
        hover_time=drone_hover_s,
        headwind_speed_mps=headwind_mps,
    )
    e_scen = EbikeScenario(
        kappa=kappa,
        gef_mult=gef_mult,
        energy_mult=1.0,
        payload_kg=payload_kg,
        rider_weight=rider_weight,
        lifespan_yr=ebike_lifespan_yr,
    )
    d_scen = DroneScenario(
        gef_mult=gef_mult,
        energy_mult=drone_energy_mult,
        payload_kg=payload_kg,
        lifespan_yr=drone_lifespan_yr,
        battery_lifespan_yr=drone_battery_lifespan_yr,
    )
    ebike_g = compute_ebike_per_order_g(road_distances_km, ebike_params, e_scen)
    drone_g = compute_drone_per_order_g(straight_distances_km, drone_params_eff, d_scen)
    return ebike_g, drone_g


def _row_for_scenario(
    feature_varied: str,
    kappa: float,
    gef_mult: float,
    drone_energy_mult: float,
    payload_kg: float,
    drone_lifespan_yr: float,
    ebike_lifespan_yr: float,
    drone_cruise_kmh: float,
    drone_body_kg: float,
    drone_hover_s: float,
    drone_battery_lifespan_yr: float,
    headwind_mps: float,
    road_distances_km: np.ndarray,
    straight_distances_km: np.ndarray,
    thresholds: np.ndarray,
    ebike_params_provincial: EbikeParams,
    drone_params_provincial: DroneParams,
    rider_weight: float,
    anchor_ebike_only_total_g: float,
    region_key: str,
    province: str,
    scenario_id: str,
) -> dict:
    ebike_g, drone_g = _emissions_for_scenario(
        road_distances_km, straight_distances_km,
        ebike_params_provincial, drone_params_provincial, rider_weight,
        kappa=kappa, gef_mult=gef_mult, drone_energy_mult=drone_energy_mult,
        payload_kg=payload_kg, ebike_lifespan_yr=ebike_lifespan_yr,
        drone_lifespan_yr=drone_lifespan_yr,
        drone_cruise_kmh=drone_cruise_kmh,
        drone_body_kg=drone_body_kg,
        drone_hover_s=drone_hover_s,
        drone_battery_lifespan_yr=drone_battery_lifespan_yr,
        headwind_mps=headwind_mps,
    )
    metrics = _best_threshold_metrics(road_distances_km, ebike_g, drone_g, thresholds)
    if anchor_ebike_only_total_g > 0:
        reduction_vs_anchor = (
            (anchor_ebike_only_total_g - metrics["best_mixed_total_g"])
            / anchor_ebike_only_total_g * 100.0
        )
    else:
        reduction_vs_anchor = 0.0
    return {
        "region_key": region_key,
        "province": province,
        "scenario_id": scenario_id,
        "feature_varied": feature_varied,
        "kappa": kappa,
        "gef_mult": gef_mult,
        "drone_energy_mult": drone_energy_mult,
        "payload_kg": payload_kg,
        "drone_lifespan_yr": drone_lifespan_yr,
        "ebike_lifespan_yr": ebike_lifespan_yr,
        "drone_cruise_kmh": drone_cruise_kmh,
        "drone_body_kg": drone_body_kg,
        "drone_hover_s": drone_hover_s,
        "drone_battery_lifespan_yr": drone_battery_lifespan_yr,
        "headwind_mps": headwind_mps,
        "optimal_threshold_km": metrics["best_threshold_km"],
        "optimal_drone_ratio_pct": metrics["best_drone_ratio_pct"],
        "mixed_lifecycle_kg": metrics["best_mixed_total_g"] / 1000.0,
        "ebike_only_lifecycle_kg": metrics["ebike_only_total_g"] / 1000.0,
        "drone_only_lifecycle_kg": metrics["drone_only_total_g"] / 1000.0,
        "reduction_vs_scenario_ebike_pct": metrics["best_reduction_vs_scenario_pct"],
        "reduction_vs_anchor_ebike_pct": float(reduction_vs_anchor),
        "beneficial": bool(metrics["best_reduction_vs_scenario_pct"] > 0),
    }


def run_oat_for_region(
    od_df: pd.DataFrame,
    region_key: str,
    province: str,
    province_gef_kwh: float,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    rider_weight: float = 70.0,
) -> pd.DataFrame:
    """Compute the OAT sensitivity sweep for a single region.

    ``province_gef_kwh`` is multiplied by each scenario's ``gef_mult`` to
    produce that scenario's effective grid emission factor. The ebike/drone
    params passed in have their ``e_use`` overridden to ``province_gef_kwh``
    so all gef_mult-relative scaling stays consistent.
    """
    road_distances_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
    straight_distances_km = od_df["distance_km"].to_numpy(dtype=float)
    if len(road_distances_km) == 0:
        return pd.DataFrame()

    # Override e_use to the province baseline; gef_mult scales it per scenario.
    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)

    thresholds = _threshold_sweep(road_distances_km)

    def _kwargs_for(level_overrides: dict) -> dict:
        """Build the full feature kwargs dict for _row_for_scenario."""
        merged = dict(BASELINE)
        merged.update(level_overrides)
        return merged

    # Anchor reference: ebike-only total at all-baseline values.
    anchor_ebike_g, _ = _emissions_for_scenario(
        road_distances_km, straight_distances_km,
        ebike_pp, drone_pp, rider_weight,
        kappa=BASELINE["kappa"], gef_mult=BASELINE["gef_mult"],
        drone_energy_mult=BASELINE["drone_energy_mult"],
        payload_kg=BASELINE["payload_kg"],
        ebike_lifespan_yr=BASELINE["ebike_lifespan_yr"],
        drone_lifespan_yr=BASELINE["drone_lifespan_yr"],
        drone_cruise_kmh=BASELINE["drone_cruise_kmh"],
        drone_body_kg=BASELINE["drone_body_kg"],
        drone_hover_s=BASELINE["drone_hover_s"],
        drone_battery_lifespan_yr=BASELINE["drone_battery_lifespan_yr"],
        headwind_mps=BASELINE["headwind_mps"],
    )
    anchor_ebike_only_total_g = float(anchor_ebike_g.sum())

    rows = []

    def _append_row(feature_varied: str, scenario_id: str, kwargs: dict) -> None:
        rows.append(_row_for_scenario(
            feature_varied=feature_varied,
            kappa=kwargs["kappa"], gef_mult=kwargs["gef_mult"],
            drone_energy_mult=kwargs["drone_energy_mult"],
            payload_kg=kwargs["payload_kg"],
            drone_lifespan_yr=kwargs["drone_lifespan_yr"],
            ebike_lifespan_yr=kwargs["ebike_lifespan_yr"],
            drone_cruise_kmh=kwargs["drone_cruise_kmh"],
            drone_body_kg=kwargs["drone_body_kg"],
            drone_hover_s=kwargs["drone_hover_s"],
            drone_battery_lifespan_yr=kwargs["drone_battery_lifespan_yr"],
            headwind_mps=kwargs["headwind_mps"],
            road_distances_km=road_distances_km,
            straight_distances_km=straight_distances_km,
            thresholds=thresholds,
            ebike_params_provincial=ebike_pp, drone_params_provincial=drone_pp,
            rider_weight=rider_weight,
            anchor_ebike_only_total_g=anchor_ebike_only_total_g,
            region_key=region_key, province=province,
            scenario_id=scenario_id,
        ))

    # Baseline row
    _append_row("baseline", "baseline", _kwargs_for({}))

    # Per-feature deviations
    for feature, levels in FEATURE_LEVELS.items():
        for level in levels:
            _append_row(feature, f"{feature}={level}", _kwargs_for({feature: level}))

    return pd.DataFrame(rows)
