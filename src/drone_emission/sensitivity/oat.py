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
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from emission.cal_single_delivery import DroneParams, EbikeParams

from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
    optimal_reduction_metrics,
    threshold_sweep_km,
)

# Baseline values (the OAT anchor).
#
# The first six features have fixed numeric baselines that fully define the
# anchor. The remaining five (drone_cruise_kmh, drone_body_kg, drone_hover_s,
# drone_battery_lifespan_yr, headwind_mps) default to ``None``, meaning
# "inherit whatever value is already on the passed-in drone_params". This
# matters because ``create_drone_params(exp_config)`` reads values from
# ``experiment.toml`` (e.g. hover_time=10s rather than the DroneParams default
# of 30s), and the OAT baseline must exactly reproduce ``cal_emission``'s
# headline numbers — hardcoding a value here would silently override the
# user's experimental configuration.
BASELINE = {
    "kappa": 1.0,
    "gef_mult": 1.0,
    "drone_energy_mult": 1.0,
    "payload_kg": 1.0,
    "drone_lifespan_yr": 4,
    "ebike_lifespan_yr": 4,
    "drone_cruise_kmh": None,             # inherit drone_params.v_level
    "drone_body_kg": None,                # inherit drone_params.w_body
    "drone_hover_s": None,                # inherit drone_params.hover_time
    "drone_battery_lifespan_yr": None,    # inherit drone_params.battery_lifespan
    "headwind_mps": None,                 # inherit drone_params.headwind_speed_mps
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
    drone_cruise_kmh: Optional[float] = None,
    drone_body_kg: Optional[float] = None,
    drone_hover_s: Optional[float] = None,
    drone_battery_lifespan_yr: Optional[float] = None,
    headwind_mps: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Per-scenario drone params: cruise speed, body mass, hover time, and
    # headwind enter the physics through DroneParams; battery lifespan goes
    # through the DroneScenario override field. None means "use the value
    # already on drone_params" so the baseline anchor honors experiment.toml.
    overrides = {}
    if drone_cruise_kmh is not None:
        overrides["v_level"] = drone_cruise_kmh
    if drone_body_kg is not None:
        overrides["w_body"] = drone_body_kg
    if drone_hover_s is not None:
        overrides["hover_time"] = drone_hover_s
    if headwind_mps is not None:
        overrides["headwind_speed_mps"] = headwind_mps
    drone_params_eff = replace(drone_params, **overrides) if overrides else drone_params

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
    drone_cruise_kmh: Optional[float],
    drone_body_kg: Optional[float],
    drone_hover_s: Optional[float],
    drone_battery_lifespan_yr: Optional[float],
    headwind_mps: Optional[float],
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
    metrics = optimal_reduction_metrics(road_distances_km, ebike_g, drone_g, thresholds)
    if anchor_ebike_only_total_g > 0:
        reduction_vs_anchor = (
            (anchor_ebike_only_total_g - metrics["best_mixed_total_g"])
            / anchor_ebike_only_total_g * 100.0
        )
    else:
        reduction_vs_anchor = 0.0
    # Resolve effective values for the CSV (so each row's columns are
    # interpretable without needing to know the runtime drone_params).
    eff_cruise = drone_cruise_kmh if drone_cruise_kmh is not None else drone_params_provincial.v_level
    eff_body = drone_body_kg if drone_body_kg is not None else drone_params_provincial.w_body
    eff_hover = drone_hover_s if drone_hover_s is not None else drone_params_provincial.hover_time
    eff_batt_life = (
        drone_battery_lifespan_yr
        if drone_battery_lifespan_yr is not None
        else drone_params_provincial.battery_lifespan
    )
    eff_wind = headwind_mps if headwind_mps is not None else drone_params_provincial.headwind_speed_mps
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
        "drone_cruise_kmh": eff_cruise,
        "drone_body_kg": eff_body,
        "drone_hover_s": eff_hover,
        "drone_battery_lifespan_yr": eff_batt_life,
        "headwind_mps": eff_wind,
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

    thresholds = threshold_sweep_km(road_distances_km)

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
