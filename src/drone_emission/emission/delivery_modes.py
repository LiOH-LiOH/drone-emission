"""Comparable per-order lifecycle emissions for delivery transport modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from .cal_single_delivery import (
    LifeCycleEmissions,
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)


DeliveryMode = Literal[
    "drone",
    "walking",
    "pedal_bicycle",
    "pedelec_ebike",
    "fuel_moped",
    "electric_moped",
    "fuel_car",
    "battery_electric_car",
]

DELIVERY_MODES: tuple[DeliveryMode, ...] = (
    "drone",
    "walking",
    "pedal_bicycle",
    "pedelec_ebike",
    "fuel_moped",
    "electric_moped",
    "fuel_car",
    "battery_electric_car",
)

_ELECTRIC_MODES = {
    "drone",
    "pedelec_ebike",
    "electric_moped",
    "battery_electric_car",
}
_FUEL_MODES = {"fuel_moped", "fuel_car"}


@dataclass(frozen=True)
class _GroundModeParameters:
    """OECD/ITF inventory allocated over source-reported vehicle-km."""

    lifecycle_g_per_vehicle: float
    lifetime_vehicle_km: float
    electricity_kwh_per_vehicle_km: float = 0.0
    fuel_mj_per_vehicle_km: float = 0.0

    @property
    def fixed_g_per_vehicle_km(self) -> float:
        return self.lifecycle_g_per_vehicle / self.lifetime_vehicle_km


# OECD/ITF (2020), Good to Go? life-cycle assessment workbook.
# Fixed inventories sum vehicle/battery manufacturing, assembly, disposal,
# fluids, and delivery at point of purchase. Infrastructure and operational
# services are excluded. Exact source cells:
#   0_Total!AG/AK/AL/AO/AX113:114 (inventory components)
#   0_Total!AG/AK/AL/AO/AX22 (lifetime vehicle-km)
#   3_Use!AK/AO35 (fuel MJ/km), AH/AL/AX4 (electricity kWh/km)
_ITF_GROUND_MODE_PARAMETERS: Mapping[str, _GroundModeParameters] = {
    "pedal_bicycle": _GroundModeParameters(
        lifecycle_g_per_vehicle=100_357.6974652074,
        lifetime_vehicle_km=13_440.0,
    ),
    "fuel_moped": _GroundModeParameters(
        lifecycle_g_per_vehicle=391_272.1646562054,
        lifetime_vehicle_km=49_000.0,
        fuel_mj_per_vehicle_km=0.6493506493506495,
    ),
    "electric_moped": _GroundModeParameters(
        lifecycle_g_per_vehicle=480_144.80662530754,
        lifetime_vehicle_km=49_000.0,
        electricity_kwh_per_vehicle_km=0.03501400560224091,
    ),
    "fuel_car": _GroundModeParameters(
        lifecycle_g_per_vehicle=6_496_824.912821,
        lifetime_vehicle_km=181_500.0,
        fuel_mj_per_vehicle_km=2.277184,
    ),
    "battery_electric_car": _GroundModeParameters(
        lifecycle_g_per_vehicle=11_339_014.877200084,
        lifetime_vehicle_km=181_500.0,
        electricity_kwh_per_vehicle_km=0.19034624761904761,
    ),
}


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return parsed


def lifecycle_emission_per_order(
    mode: DeliveryMode,
    distance_km: float,
    *,
    exp_config: Mapping[str, Any] | None,
    grid_ef_kg_per_kwh: float | None = None,
    fuel_wtw_ef_kg_per_mj: float | None = None,
) -> float:
    """Return lifecycle emissions for one unbundled round-trip delivery.

    ``distance_km`` is the common one-way travelled distance. The result is in
    g CO2e/order. Energy factors are required only for modes that use them;
    irrelevant factors are deliberately ignored to support uniform batch calls.
    """
    if mode not in DELIVERY_MODES:
        choices = ", ".join(DELIVERY_MODES)
        raise ValueError(f"Unknown delivery mode {mode!r}; expected one of: {choices}")
    distance = _finite_nonnegative(distance_km, "distance_km")

    grid_ef = None
    if mode in _ELECTRIC_MODES:
        if grid_ef_kg_per_kwh is None:
            raise ValueError(f"grid_ef_kg_per_kwh is required for {mode}")
        grid_ef = _finite_nonnegative(
            grid_ef_kg_per_kwh,
            "grid_ef_kg_per_kwh",
        )

    fuel_ef = None
    if mode in _FUEL_MODES:
        if fuel_wtw_ef_kg_per_mj is None:
            raise ValueError(f"fuel_wtw_ef_kg_per_mj is required for {mode}")
        fuel_ef = _finite_nonnegative(
            fuel_wtw_ef_kg_per_mj,
            "fuel_wtw_ef_kg_per_mj",
        )

    if mode == "walking":
        return 0.0

    if mode in {"drone", "pedelec_ebike"}:
        if exp_config is None:
            raise ValueError(f"exp_config is required for {mode}")
        config = dict(exp_config)
        common = get_emission_params(config)
        ebike_params = create_ebike_params(config)
        drone_params = create_drone_params(config)
        ebike_params.e_use = grid_ef
        drone_params.e_use = grid_ef
        lca = LifeCycleEmissions(
            ebike_params=ebike_params,
            drone_params=drone_params,
        )
        vehicle_type = "drone" if mode == "drone" else "ebike"
        # The legacy method uses a one-way distance and internally models the
        # return trip. One order per trip keeps the paper e-bike model at kappa=1.
        result = lca.calculate_lifecycle_emissions_per_order_simple(
            distance,
            n_orders_per_delivery=1,
            vehicle_type=vehicle_type,
        )
        if mode == "pedelec_ebike":
            # Preserve the configured paper payload/rider assumptions used by
            # the direct use-phase path. The lifecycle method's production
            # allocation is unchanged; use is recomputed only when assumptions
            # differ from its historical defaults.
            payload_kg = float(common["payload_kg"])
            rider_weight = float(common["rider_weight"])
            if payload_kg != 1.0 or rider_weight != 70.0:
                old_use_g = float(result["use_per_order_g"])
                new_use_g = lca.calculate_ebike_use_phase_emissions(
                    distance,
                    payload_kg=payload_kg,
                    rider_weight=rider_weight,
                )["emissions_g_co2e"]
                return float(result["total_per_order_g"] - old_use_g + new_use_g)
        elif float(common["payload_kg"]) != 1.0:
            old_use_g = float(result["use_per_order_g"])
            new_use_g = lca.calculate_drone_use_phase_emissions(
                distance,
                payload_kg=float(common["payload_kg"]),
            )["emissions_g_co2e"]
            return float(result["total_per_order_g"] - old_use_g + new_use_g)
        return float(result["total_per_order_g"])

    params = _ITF_GROUND_MODE_PARAMETERS[mode]
    per_vehicle_km_g = params.fixed_g_per_vehicle_km
    if params.electricity_kwh_per_vehicle_km:
        per_vehicle_km_g += (
            params.electricity_kwh_per_vehicle_km * grid_ef * 1000.0
        )
    if params.fuel_mj_per_vehicle_km:
        per_vehicle_km_g += params.fuel_mj_per_vehicle_km * fuel_ef * 1000.0
    return float(2.0 * distance * per_vehicle_km_g)
