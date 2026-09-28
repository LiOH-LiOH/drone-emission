"""Citywide full-cost pilot using demand-sized Li 2025 deployment modules.

Fleet and facility counts are capacity estimates, not spatial siting results.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drone_emission.analysis.cost_time_substitution import (
    CostParameters, OD_COLUMNS, transport_cost_parts,
)
from drone_emission.emission.cal_single_delivery import create_drone_params, create_ebike_params
from drone_emission.paper_cohort import cohort_hash, locked_region_keys
from drone_emission.utils.atomic import atomic_open


@dataclass(frozen=True)
class ReferenceSystem:
    fleet_per_mode: int = 100
    distribution_centers: int = 10
    destination_cabinets: int = 100
    common_dc_cabinets: int = 20
    staff_count: int = 10
    staff_daily_cny: float = 200
    working_days: int = 300
    cabinet_hours_daily: float = 24
    vehicle_life_years: float = 4
    cabinet_life_years: float = 5
    ebike_price_cny: float = 4000
    drone_price_cny: float = 60000
    ordinary_cabinet_price_cny: float = 10000
    drone_cabinet_price_cny: float = 30000
    ordinary_cabinet_kw: float = 0.2
    drone_cabinet_kw: float = 0.4
    distance_bin_km: float = 0.1
    drone_hours_daily: float = 12
    pilots_per_drone: float = 0
    pilot_daily_cny: float = 200


def annual_fixed_cost(system: ReferenceSystem, cost: CostParameters, deployment: str) -> dict:
    """Annual fleet, battery, cabinet and staff costs, including idle capacity."""
    bike_on = deployment in {"ebike", "mixed"}
    drone_on = deployment in {"drone", "mixed"}
    if not bike_on and not drone_on:
        raise ValueError(f"Unknown deployment: {deployment}")
    n = system.fleet_per_mode
    net = 1 - cost.battery_salvage_fraction
    vehicle = net * n * (
        bike_on * system.ebike_price_cny + drone_on * system.drone_price_cny
    ) / system.vehicle_life_years
    battery = net * n * (
        bike_on * cost.ebike_battery_price_cny * cost.ebike_battery_stock_per_vehicle
        + drone_on * cost.drone_battery_price_cny * cost.drone_battery_stock_per_vehicle
    ) / cost.battery_life_years
    # Both modes can use an upgraded cabinet. Mixed deployment pays once.
    cabinet_price = system.drone_cabinet_price_cny if drone_on else system.ordinary_cabinet_price_cny
    cabinet_kw = system.drone_cabinet_kw if drone_on else system.ordinary_cabinet_kw
    cabinets = net * (
        system.common_dc_cabinets * system.drone_cabinet_price_cny
        + system.destination_cabinets * cabinet_price
    ) / system.cabinet_life_years
    cabinet_electricity = (
        system.common_dc_cabinets * system.drone_cabinet_kw
        + system.destination_cabinets * cabinet_kw
    ) * system.cabinet_hours_daily * system.working_days * cost.electricity_cny_per_kwh / cost.charging_efficiency
    return {
        "annual_vehicle_cny": vehicle,
        "annual_battery_cny": battery,
        "annual_cabinet_capital_cny": cabinets,
        "annual_staff_cny": system.staff_count * system.staff_daily_cny * system.working_days,
        "annual_pilot_cny": drone_on * n * system.pilots_per_drone * system.pilot_daily_cny * system.working_days,
        "annual_cabinet_electricity_cny": cabinet_electricity,
    }


def build_profiles(od_path: Path, config: dict, system: ReferenceSystem, chunk_rows=250_000):
    """Pool weighted OD metrics by 100 m distance classes; preserve their totals."""
    bike, drone = create_ebike_params(config), create_drone_params(config)
    fields = [name for name in OD_COLUMNS if "lifecycle" not in name]
    grouped = []
    rows = 0
    for chunk in pd.read_csv(od_path, usecols=fields, chunksize=chunk_rows):
        chunk = chunk.apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(chunk.to_numpy()).all() or (chunk <= 0).any().any():
            raise ValueError("Pilot requires finite, positive OD distances, weights and times")
        road = chunk.road_distance_m.to_numpy() / 1000
        air = chunk.distance_km.to_numpy()
        kappa = chunk.ebike_kappa.to_numpy()
        q = chunk.annual_sales.to_numpy()
        parts = transport_cost_parts(road, air, kappa, config)
        metrics = pd.DataFrame({
            "road_bin": np.floor(road / system.distance_bin_km).astype(int),
            "air_bin": np.floor(air / system.distance_bin_km).astype(int),
            "orders": q,
            "ebike_variable": parts["ebike"]["rider_cny"] + parts["ebike"]["electricity_cny"],
            "drone_variable": parts["drone"]["electricity_cny"],
            "ebike_cycle": 2 * road * kappa / bike.velocity * 60 + bike.loading_time_min,
            "drone_cycle": (
                2 * air / drone.v_level * 60
                + 4 * drone.h_takeoff / 1000 / drone.v_vertical * 60
                + drone.hover_time / 60 + drone.loading_time_min
            ),
            "ebike_time": chunk.ebike_time_min.to_numpy(),
            "drone_time": chunk.drone_time_min.to_numpy(),
        })
        values = list(metrics.columns[3:])
        metrics[values] = metrics[values].mul(q, axis=0)
        grouped.append(metrics.groupby(["road_bin", "air_bin"]).sum())
        rows += len(chunk)
    if not grouped:
        raise ValueError("Empty city OD input")
    profile = pd.concat(grouped).groupby(level=[0, 1]).sum().reset_index()
    values = list(profile.columns[3:])
    profile[values] = profile[values].div(profile.orders, axis=0)
    profile["share"] = profile.orders / profile.orders.sum()
    return profile, {"od_rows": rows, "profiles": len(profile), "city_annual_orders": float(profile.orders.sum())}


def development_stages(config: dict, system: ReferenceSystem):
    """Chinese price endpoints combined with supervised/autonomous operation."""
    stages = config.get("full_cost_pilot", {}).get("cost_stages", [
        {"stage": "conservative", "drone_price_cny": 60000.,
         "drone_battery_price_cny": 1000., "drone_hours_daily": 8., "pilots_per_drone": 1.,
         "basis": "Li 2025 baseline prices; Baldisseri 2022 conservative automation; project 8 h shift"},
        {"stage": "optimistic", "drone_price_cny": 40000.,
         "drone_battery_price_cny": 600., "drone_hours_daily": 12., "pilots_per_drone": 0.,
         "basis": "Li 2025 ideal prices; Baldisseri 2022 future automation; project 12 h operation"},
    ])
    for stage in stages:
        stage_config = {
            **config,
            "transport_cost": {
                **config.get("transport_cost", {}),
                "drone_battery_price_cny": stage["drone_battery_price_cny"],
            },
        }
        yield stage, stage_config, replace(
            system, drone_price_cny=stage["drone_price_cny"],
            drone_hours_daily=stage["drone_hours_daily"], pilots_per_drone=stage["pilots_per_drone"],
        )


def solve_assignment(profile, annual_orders, config, system, rate=None):
    """Jointly choose OD assignments and integer citywide deployment modules.

    A module supplies 100 vehicles of a mode. Network modules are shared by
    the two modes; drone modules require upgraded destination cabinets.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csc_matrix

    if annual_orders <= 0:
        raise ValueError("Annual orders must be positive")
    bike = create_ebike_params(config)
    cost = CostParameters(**config.get("transport_cost", {}))
    share = profile.share.to_numpy()
    ebike_cycle = profile.ebike_cycle.to_numpy()
    drone_cycle = profile.drone_cycle.to_numpy()
    ebike_variable = profile.ebike_variable.to_numpy()
    drone_variable = profile.drone_variable.to_numpy()
    daily_orders = annual_orders / system.working_days
    n = len(share)
    base_bike = annual_fixed_cost(system, cost, "ebike")
    base_drone = annual_fixed_cost(system, cost, "drone")
    asset_bike = base_bike["annual_vehicle_cny"] + base_bike["annual_battery_cny"]
    asset_drone = (base_drone["annual_vehicle_cny"] + base_drone["annual_battery_cny"]
                   + base_drone["annual_pilot_cny"])
    network = sum(base_bike.values()) - asset_bike
    upgrade = sum(base_drone.values()) - asset_drone - network
    bike_capacity = system.fleet_per_mode * bike.working_hours_per_day * 60
    drone_capacity = system.fleet_per_mode * system.drone_hours_daily * 60
    # Use daily order counts, not tiny fractions: small OD-share upper bounds
    # otherwise disappear in MIP presolve and spuriously require a whole fleet.
    # Remaining variables are integer bike, drone and shared network modules.
    objective = np.r_[drone_variable - ebike_variable,
                      np.array([asset_bike, asset_drone + upgrade, network]) / system.working_days]
    matrix = np.zeros((4 + (rate is not None), n + 3))
    matrix[0, :n], matrix[0, n + 1] = drone_cycle, -drone_capacity
    matrix[1, :n], matrix[1, n] = -ebike_cycle, -bike_capacity
    matrix[2, n], matrix[2, n + 2] = 1, -1
    matrix[3, n + 1], matrix[3, n + 2] = 1, -1
    lower = np.full(len(matrix), -np.inf)
    upper = np.array([0., -np.dot(share, ebike_cycle) * daily_orders, 0., 0.]
                     + ([] if rate is None else [rate * daily_orders]))
    if rate is not None:
        matrix[4, :n] = 1
        lower[4] = rate * daily_orders
    variable_upper = np.r_[share * daily_orders, [np.inf] * 3]
    if rate == 0:
        variable_upper[:n] = 0
        variable_upper[n + 1] = 0
    elif rate == 1:
        variable_upper[n] = 0
    result = milp(
        objective, integrality=np.r_[np.zeros(n), np.ones(3)],
        bounds=Bounds(np.zeros(n + 3), variable_upper),
        constraints=LinearConstraint(csc_matrix(matrix), lower, upper),
        options={"mip_rel_gap": 1e-6, "time_limit": 60},
    )
    if not result.success:
        raise RuntimeError(f"Citywide cost solve did not converge: {result.message}")
    x = result.x[:n] / daily_orders
    bike_blocks, drone_blocks, network_blocks = np.rint(result.x[n:]).astype(int)
    fixed = {}
    for key in base_bike:
        if key in {"annual_vehicle_cny", "annual_battery_cny", "annual_pilot_cny"}:
            fixed[key] = base_bike[key] * bike_blocks + base_drone[key] * drone_blocks
        else:
            fixed[key] = base_bike[key] * network_blocks + (base_drone[key] - base_bike[key]) * drone_blocks
    variable = float(np.dot(share - x, ebike_variable) + np.dot(x, drone_variable))
    bike_work = float(np.dot(share - x, ebike_cycle) * daily_orders)
    drone_work = float(np.dot(x, drone_cycle) * daily_orders)
    total = sum(fixed.values()) + variable * annual_orders
    return {
        "requested_drone_rate": rate,
        "drone_replacement_rate_pct": float(x.sum() * 100),
        "annual_orders": annual_orders, "daily_orders": daily_orders,
        **fixed, "annual_variable_cny": variable * annual_orders,
        "annual_total_cny": total, "mean_total_cost_cny": total / annual_orders,
        "mean_transport_time_min": float(np.dot(share - x, profile.ebike_time) + np.dot(x, profile.drone_time)),
        "ebike_count": int(bike_blocks * system.fleet_per_mode),
        "drone_count": int(drone_blocks * system.fleet_per_mode),
        "pilot_positions": int(drone_blocks * system.fleet_per_mode * system.pilots_per_drone),
        "network_modules": int(network_blocks),
        "distribution_centers": int(network_blocks * system.distribution_centers),
        "ground_staff": int(network_blocks * system.staff_count),
        "common_dc_cabinets": int(network_blocks * system.common_dc_cabinets),
        "ordinary_destination_cabinets": int((network_blocks - drone_blocks) * system.destination_cabinets),
        "drone_destination_cabinets": int(drone_blocks * system.destination_cabinets),
        "ebike_available_capacity_used_pct": bike_work / (bike_capacity * bike_blocks) * 100 if bike_blocks else 0,
        "drone_available_capacity_used_pct": drone_work / (drone_capacity * drone_blocks) * 100 if drone_blocks else 0,
        "solver_relative_gap": float(result.mip_gap),
    }


def run_full_cost_pilot(config: dict, region_key: str) -> tuple[Path, Path, Path]:
    """Size citywide fleets and facilities for one explicitly named cohort city."""
    if region_key not in locked_region_keys():
        raise ValueError("Pilot city must belong to the locked paper cohort")
    settings = config.get("full_cost_pilot", {})
    system = ReferenceSystem(**settings.get("system", {}))
    rates = settings.get("replacement_rates", [i / 10 for i in range(11)])
    if any(not 0 <= r <= 1 for r in rates) or system.distance_bin_km <= 0:
        raise ValueError("Replacement rates must be in [0, 1]; bin width must be positive")
    data = Path(config["data_dir"])
    stats = pd.read_csv(data / "all_region_stats.csv")
    city = stats.loc[stats.region_key.eq(region_key)]
    if len(city) != 1:
        raise ValueError(f"Expected one stats row for {region_key}")
    directory = str(city.iloc[0].region_name_en).lower()
    od_path = data / "city" / directory / f"{directory}_od_pairs.csv"
    profile, audit = build_profiles(od_path, config, system)
    expected_orders = float(city.iloc[0].total_odpair_annual_sales)
    if not np.isclose(audit["city_annual_orders"], expected_orders, rtol=1e-10):
        raise ValueError("Pilot OD annual demand differs from the city aggregate")
    curves, optima = [], []
    stages = list(development_stages(config, system))
    annual_orders = audit["city_annual_orders"]
    for stage, stage_config, stage_system in stages:
        best = solve_assignment(profile, annual_orders, stage_config, stage_system)
        optima.append({**stage, **best})
        scenario_curve = []
        for rate in rates:
            row = solve_assignment(profile, annual_orders, stage_config, stage_system, rate=rate)
            scenario_curve.append(row)
            curves.append({**stage, **row})
        curve_minimum = min(row["annual_total_cny"] for row in scenario_curve)
        if best["annual_total_cny"] > curve_minimum + max(1., curve_minimum * 2e-6):
            raise RuntimeError("Free-rate optimum is worse than a fixed-rate solution")
        print(f"Full-cost pilot {region_key}: stage={stage['stage']}", flush=True)
    root = Path(config["work_dir"]) / "analysis" / "full_cost_pilot" / region_key
    paths = tuple(root / name for name in ("cost_curves.csv", "optimal_strategies.csv", "report.json"))
    report = {
        "analysis_version": 3,
        "region_key": region_key, "scope": "citywide_development_cost_stages",
        "cohort_sha256": cohort_hash(), "source_doi": "10.1016/j.scs.2025.106134",
        "automation_source_doi": "10.1016/j.trd.2022.103296",
        "source_module": asdict(system), "transport_cost": asdict(CostParameters(**config.get("transport_cost", {}))),
        "emission_params": config.get("emission_params", {}),
        "cost_stages": [stage for stage, _, _ in stages],
        "capacity_basis": {
            "ebike_hours_per_day": create_ebike_params(config).working_hours_per_day,
            "drone_hours_per_day_by_stage": {stage["stage"]: s.drone_hours_daily for stage, _, s in stages},
            "working_days": system.working_days,
        },
        "replacement_rates": rates, "input": {"path": str(od_path.resolve()),
        "size_bytes": od_path.stat().st_size, "mtime_ns": od_path.stat().st_mtime_ns},
        **audit,
        "interpretation": [
            "Both scenarios use identical actual city annual orders; no demand or availability sweep.",
            "Joint scenarios change drone/battery prices, automation and daily drone operating hours (8 conservative, 12 optimistic). Electric bikes retain configured 8 h capacity.",
            "Automation follows Baldisseri 2022 conservative/future endpoints, not its partially automated optimistic scenario. Prices follow Li 2025 Table 1 endpoints.",
            "Conservative pilots are paid for full configured shifts at 200 CNY/day, 300 days/year, one position per drone including idle capacity. Optimistic has no pilots; ground staff remain. Wage is a Li 2025 local proxy, not measured Nanjing pilot pay. Full-shift payment and 8/12 h operation are project assumptions; Baldisseri uses 7.5 h shifts.",
            "Cabinet prices/power, staff wages, energy prices, speeds and lifetimes remain fixed. The source ideal cabinet price includes solar equipment and is not adopted in this isolated equipment-cost experiment.",
            "Jointly optimize assignments and integer 100-vehicle modules. Network modules equal the larger mode-module count; mixed modes can share each network.",
            "Per network module: 10 distribution centers, 20 common DC drone cabinets, 100 destination cabinets and 10 staff. Drone modules upgrade that many destination modules; other destination modules use ordinary cabinets.",
            "Facility counts are a source-ratio capacity estimate, not a spatial siting or actual facility inventory. Source ratios and modular expansion are project assumptions.",
            "All capital and batteries annualized with 5% salvage; fleet batteries are not charged again per order. No solar or separate repair cost.",
            "Capacity uses project speeds/workday lengths and round-trip tasks. Drone cycle includes both legs' vertical flight, one loaded hover and existing loading allowance.",
            "Fleet and facility quantities are reoptimized for each scenario with its operating hours; daily hours are planning assumptions, not measured utilization or a queue simulation.",
            "OD metrics are averaged in 100 m road/air classes; fractional within-class assignment is approximate. Weighted baseline totals are preserved.",
            "A free-rate mixed-integer solve determines the optimum; the rate grid only displays the curve. Optimality is within the modular cost and aggregated-hour capacity model.",
            "Cost includes facilities; reported time remains transport-only. No full-system lifecycle-emissions claim is made.",
        ],
    }
    for path, frame in zip(paths[:2], (pd.DataFrame(curves), pd.DataFrame(optima))):
        with atomic_open(path, "w", newline="") as handle:
            frame.to_csv(handle, index=False)
    with atomic_open(paths[2], "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return paths
