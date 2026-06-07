"""Vectorized lifecycle-emission helpers for the sensitivity stage.

Per-OD lifecycle emissions reduce to closed-form `alpha + beta * d` per scenario
for both modes (use-phase + production amortization both linear in distance with
intercept). This module exposes vectorized numpy implementations that produce
numerically identical results to ``LifeCycleEmissions.calculate_lifecycle_emissions_per_order_simple``
when run with matching parameters. See ``tests/test_vectorized_lca.py``.

The scalar reference (in emission/cal_single_delivery.py) has these structural
properties that we mirror:

* E-bike use phase: ``energy_kwh = (P_go + P_return) * d / velocity`` already
  accounts for round-trip via the sum of loaded + empty one-way powers.
* Drone use phase: takeoff/landing/hover energies are scenario-constants;
  level-flight energy scales linearly with one-way distance.
* Production amortization: ``delivery_time_min`` is computed from per-OD
  distance (``2*d/v*60 + loading_time_min``), so ``deliveries_per_day`` and the
  resulting ``prod_per_delivery_kg`` vary per OD. The scalar code uses
  ``2*d/v*60 + loading`` as the round-trip flight time even for drone (it does
  NOT add takeoff/landing/hover); we mirror that simplification exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from emission.cal_single_delivery import (
    DroneEnergyModel,
    DroneParams,
    EbikeEnergyModel,
    EbikeParams,
)


@dataclass
class EbikeScenario:
    """Per-scenario knobs for the ebike vectorized LCA.

    Attributes:
        kappa: Bundling factor applied to each OD distance.
        gef_mult: Multiplier applied to the use-phase grid emission factor.
        energy_mult: Multiplier on use-phase energy (e.g. for drone scenarios
            this maps to drone aero uncertainty; ebike rarely varies).
        payload_kg: Payload weight per order (kg).
        rider_weight: Rider weight (kg).
        lifespan_yr: Override for vehicle body lifespan (years). ``None``
            keeps the params.lifespan default.
        battery_lifespan_yr: Override for battery lifespan (years).
        n_orders_per_delivery: Orders amortized per delivery trip.
        include_production: If False, zero out production amortization.
        include_use: If False, zero out use-phase emissions.
    """

    kappa: float = 1.0
    gef_mult: float = 1.0
    energy_mult: float = 1.0
    payload_kg: float = 1.0
    rider_weight: float = 70.0
    lifespan_yr: Optional[float] = None
    battery_lifespan_yr: Optional[float] = None
    n_orders_per_delivery: int = 1
    include_production: bool = True
    include_use: bool = True


@dataclass
class DroneScenario:
    """Per-scenario knobs for the drone vectorized LCA."""

    gef_mult: float = 1.0
    energy_mult: float = 1.0
    payload_kg: float = 1.0
    lifespan_yr: Optional[float] = None
    battery_lifespan_yr: Optional[float] = None
    n_orders_per_delivery: int = 1
    include_production: bool = True
    include_use: bool = True
    utilization_mult: float = 1.0


def _ebike_production_lifetime_kg(p: EbikeParams, lifespan: float, battery_lifespan: float) -> float:
    """Per-year-equivalent production+disposal mass for an ebike.

    Mirrors ``LifeCycleEmissions.calculate_ebike_production_emissions`` plus
    the body/lifespan + battery/battery_lifespan amortization at lines
    1219-1225 of cal_single_delivery.py.
    """
    body_prod = p.e_body * p.w_body
    body_dispose = p.e_body_dispose * p.w_body
    total_battery_kwh = p.v_battery * p.n_battery
    battery_prod = p.e_battery * total_battery_kwh
    battery_dispose = p.e_battery_dispose * total_battery_kwh
    body_lifetime = (body_prod + body_dispose) / lifespan
    battery_lifetime = (battery_prod + battery_dispose) / battery_lifespan
    return body_lifetime + battery_lifetime


def _ebike_lifecycle_annual_parts_kg(
    p: EbikeParams, lifespan: float, battery_lifespan: float
) -> tuple[float, float]:
    body_prod = p.e_body * p.w_body
    total_battery_kwh = p.v_battery * p.n_battery
    battery_prod = p.e_battery * total_battery_kwh
    body_dispose = p.e_body_dispose * p.w_body
    battery_dispose = p.e_battery_dispose * total_battery_kwh
    production_kg = body_prod / lifespan + battery_prod / battery_lifespan
    end_of_life_kg = body_dispose / lifespan + battery_dispose / battery_lifespan
    return production_kg, end_of_life_kg


def _drone_production_lifetime_kg(p: DroneParams, lifespan: float, battery_lifespan: float) -> float:
    body_prod = p.e_body * p.w_body
    body_dispose = p.e_body_dispose * p.w_body
    total_battery_kwh = p.v_battery * p.n_battery
    battery_prod = p.e_battery * total_battery_kwh
    battery_dispose = p.e_battery_dispose * total_battery_kwh
    body_lifetime = (body_prod + body_dispose) / lifespan
    battery_lifetime = (battery_prod + battery_dispose) / battery_lifespan
    return body_lifetime + battery_lifetime


def _drone_lifecycle_annual_parts_kg(
    p: DroneParams, lifespan: float, battery_lifespan: float
) -> tuple[float, float]:
    body_prod = p.e_body * p.w_body
    total_battery_kwh = p.v_battery * p.n_battery
    battery_prod = p.e_battery * total_battery_kwh
    body_dispose = p.e_body_dispose * p.w_body
    battery_dispose = p.e_battery_dispose * total_battery_kwh
    production_kg = body_prod / lifespan + battery_prod / battery_lifespan
    end_of_life_kg = body_dispose / lifespan + battery_dispose / battery_lifespan
    return production_kg, end_of_life_kg


def compute_ebike_per_order_parts_g(
    distances_km: np.ndarray,
    ebike_params: EbikeParams,
    scenario: EbikeScenario,
) -> dict[str, np.ndarray]:
    """Vectorized e-bike lifecycle components per order (g CO2e)."""
    p = ebike_params
    s = scenario
    lifespan = s.lifespan_yr if s.lifespan_yr is not None else p.lifespan
    battery_lifespan = (
        s.battery_lifespan_yr if s.battery_lifespan_yr is not None else p.battery_lifespan
    )

    d_eff = np.asarray(distances_km, dtype=float) * s.kappa

    eng = EbikeEnergyModel(p)
    velocity_kmh = p.velocity
    velocity_ms = velocity_kmh / 3.6
    w_go = p.w_body + p.w_battery + s.rider_weight + s.payload_kg
    w_return = p.w_body + p.w_battery + s.rider_weight
    P_go, _ = eng.calculate_power(w_go, velocity_ms)
    P_return, _ = eng.calculate_power(w_return, velocity_ms)
    energy_kwh = (P_go + P_return) * d_eff / velocity_kmh * s.energy_mult
    use_g = energy_kwh * p.e_use * s.gef_mult * 1000.0
    if not s.include_use:
        use_g = np.zeros_like(d_eff)

    delivery_time_min = 2.0 * d_eff / velocity_kmh * 60.0 + p.loading_time_min
    deliveries_per_day = p.working_hours_per_day * 60.0 / delivery_time_min
    production_annual_kg, eol_annual_kg = _ebike_lifecycle_annual_parts_kg(
        p, lifespan, battery_lifespan
    )
    prod_g = (
        production_annual_kg
        / (deliveries_per_day * p.working_days_per_year)
        * 1000.0
    )
    eol_g = eol_annual_kg / (deliveries_per_day * p.working_days_per_year) * 1000.0
    if not s.include_production:
        prod_g = np.zeros_like(d_eff)
        eol_g = np.zeros_like(d_eff)

    divisor = s.n_orders_per_delivery
    prod_g = prod_g / divisor
    use_g = use_g / divisor
    eol_g = eol_g / divisor
    return {
        "production_g": prod_g,
        "use_g": use_g,
        "end_of_life_g": eol_g,
        "total_g": prod_g + use_g + eol_g,
    }


def compute_ebike_per_order_g(
    distances_km: np.ndarray,
    ebike_params: EbikeParams,
    scenario: EbikeScenario,
) -> np.ndarray:
    """Vectorized per-order lifecycle emissions for the ebike mode (g CO2e).

    ``distances_km`` is the one-way distance per OD pair. ``scenario.kappa``
    is applied internally so the caller can pass a single distance array
    regardless of bundling assumption.

    Returns an array shaped like ``distances_km``.
    """
    return compute_ebike_per_order_parts_g(
        distances_km, ebike_params, scenario
    )["total_g"]


def compute_drone_per_order_parts_g(
    distances_km: np.ndarray,
    drone_params: DroneParams,
    scenario: DroneScenario,
) -> dict[str, np.ndarray]:
    """Vectorized drone lifecycle components per order (g CO2e)."""
    p = drone_params
    s = scenario
    lifespan = s.lifespan_yr if s.lifespan_yr is not None else p.lifespan
    battery_lifespan = (
        s.battery_lifespan_yr if s.battery_lifespan_yr is not None else p.battery_lifespan
    )

    d = np.asarray(distances_km, dtype=float)

    eng = DroneEnergyModel(p)
    w_loaded = p.w_body + p.w_battery + s.payload_kg
    w_empty = p.w_body + p.w_battery
    v_level_ms = p.v_level / 3.6
    v_vertical_ms = p.v_vertical / 3.6

    P_takeoff_loaded, _ = eng.calculate_flight_angle_power(w_loaded, v_vertical_ms, 90)
    P_landing_loaded, _ = eng.calculate_flight_angle_power(w_loaded, v_vertical_ms, -90)
    P_hover_loaded, _ = eng.calculate_flight_angle_power(w_loaded, 0.0, 0)
    P_level_loaded, _ = eng.calculate_flight_angle_power(w_loaded, v_level_ms, 0)
    P_takeoff_empty, _ = eng.calculate_flight_angle_power(w_empty, v_vertical_ms, 90)
    P_landing_empty, _ = eng.calculate_flight_angle_power(w_empty, v_vertical_ms, -90)
    P_level_empty, _ = eng.calculate_flight_angle_power(w_empty, v_level_ms, 0)

    t_takeoff_h = (p.h_takeoff / 1000.0) / p.v_vertical
    t_landing_h = t_takeoff_h
    t_hover_h = p.hover_time / 3600.0
    E_const = (
        P_takeoff_loaded * t_takeoff_h
        + P_landing_loaded * t_landing_h
        + P_hover_loaded * t_hover_h
        + P_takeoff_empty * t_takeoff_h
        + P_landing_empty * t_landing_h
    )
    energy_kwh = (E_const + (P_level_loaded + P_level_empty) * d / p.v_level) * s.energy_mult
    use_g = energy_kwh * p.e_use * s.gef_mult * 1000.0
    if not s.include_use:
        use_g = np.zeros_like(d)

    delivery_time_min = 2.0 * d / p.v_level * 60.0 + p.loading_time_min
    deliveries_per_day = p.working_hours_per_day * 60.0 / delivery_time_min
    production_annual_kg, eol_annual_kg = _drone_lifecycle_annual_parts_kg(
        p, lifespan, battery_lifespan
    )
    denominator = deliveries_per_day * p.working_days_per_year * s.utilization_mult
    prod_g = production_annual_kg / denominator * 1000.0
    eol_g = eol_annual_kg / denominator * 1000.0
    if not s.include_production:
        prod_g = np.zeros_like(d)
        eol_g = np.zeros_like(d)

    divisor = s.n_orders_per_delivery
    prod_g = prod_g / divisor
    use_g = use_g / divisor
    eol_g = eol_g / divisor
    return {
        "production_g": prod_g,
        "use_g": use_g,
        "end_of_life_g": eol_g,
        "total_g": prod_g + use_g + eol_g,
    }


def compute_drone_per_order_g(
    distances_km: np.ndarray,
    drone_params: DroneParams,
    scenario: DroneScenario,
) -> np.ndarray:
    """Vectorized per-order lifecycle emissions for the drone mode (g CO2e)."""
    return compute_drone_per_order_parts_g(
        distances_km, drone_params, scenario
    )["total_g"]


def threshold_sweep_km(road_distances_km: np.ndarray) -> np.ndarray:
    """12-point threshold list matching emission/api.py:206-216."""
    middle = np.linspace(
        max(0.3, np.percentile(road_distances_km, 5)),
        np.percentile(road_distances_km, 95),
        10,
    )
    return np.concatenate([[-0.01], middle, [np.inf]])


def optimal_reduction_metrics(
    road_distances_km: np.ndarray,
    ebike_g: np.ndarray,
    drone_g: np.ndarray,
    thresholds: np.ndarray,
) -> dict:
    """Iterate the threshold sweep, return optimal threshold + reduction metrics.

    Reduction is vs the e-bike-only total under the same per-OD emissions
    (the scenario-specific baseline). The "vs anchor" comparison is the
    caller's responsibility.
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
