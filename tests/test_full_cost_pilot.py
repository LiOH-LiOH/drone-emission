"""Targeted economic and capacity checks for the opt-in citywide pilot."""
import json

import numpy as np
import pandas as pd
import pytest

from drone_emission.analysis import full_cost_pilot as pilot
from drone_emission.analysis.cost_time_substitution import CostParameters


def profiles():
    return pd.DataFrame({
        "share": [0.4, 0.6], "ebike_cycle": [10., 20.], "drone_cycle": [5., 10.],
        "ebike_variable": [2., 3.], "drone_variable": [0.1, 0.2],
        "ebike_time": [4., 8.], "drone_time": [2., 4.],
    })


def test_source_module_costs_and_no_duplicate_battery_charge():
    system = pilot.ReferenceSystem()
    expected = {"ebike": 1161444.4444444445, "drone": 3064833.3333333335}
    for mode, rate in (("ebike", 0), ("drone", 1)):
        fixed = pilot.annual_fixed_cost(system, CostParameters(), mode)
        assert sum(fixed.values()) == pytest.approx(expected[mode])
        row = pilot.solve_assignment(profiles(), 300000, {}, system, rate)
        variable = 2.6 if mode == "ebike" else 0.16
        assert row["annual_total_cny"] == pytest.approx(expected[mode] + variable * 300000)
        assert row["drone_replacement_rate_pct"] == pytest.approx(rate * 100)


def test_citywide_demand_sizes_fleet_and_shared_network():
    system = pilot.ReferenceSystem()
    for rate, count, networks in ((0, 400, 4), (1, 200, 2)):
        row = pilot.solve_assignment(profiles(), 3000000, {}, system, rate)
        assert row["ebike_count" if rate == 0 else "drone_count"] == count
        assert row["network_modules"] == networks
        assert row["common_dc_cabinets"] == 20 * networks
        assert row["ground_staff"] == 10 * networks
        assert row["ebike_available_capacity_used_pct"] <= 100.0001
        assert row["drone_available_capacity_used_pct"] <= 100.0001
    mixed = pilot.solve_assignment(profiles(), 3000000, {}, system, 0.5)
    assert mixed["network_modules"] == max(mixed["ebike_count"], mixed["drone_count"]) / 100
    assert mixed["ordinary_destination_cabinets"] + mixed["drone_destination_cabinets"] == 100 * mixed["network_modules"]


def test_free_optimum_dominates_grid_and_cost_stages_reduce_equipment_costs():
    system = pilot.ReferenceSystem()
    best = pilot.solve_assignment(profiles(), 3000000, {}, system)
    for rate in (0, 0.5, 1):
        row = pilot.solve_assignment(profiles(), 3000000, {}, system, rate)
        assert best["annual_total_cny"] <= row["annual_total_cny"] + 10
    stages = list(pilot.development_stages({}, system))
    lower_cost = stages[-1]
    fixed = []
    for stage, config, stage_system in stages:
        fixed.append(pilot.annual_fixed_cost(
            stage_system, CostParameters(**config["transport_cost"]), "drone"
        ))
    assert len(stages) == 2
    assert sum(fixed[0].values()) - sum(fixed[1].values()) == pytest.approx(6532000)
    assert fixed[0]["annual_pilot_cny"] == 6000000
    assert fixed[1]["annual_pilot_cny"] == 0
    for key in ("annual_staff_cny", "annual_cabinet_capital_cny", "annual_cabinet_electricity_cny"):
        assert fixed[0][key] == fixed[1][key]
    for (_, config, scenario_system), count, hours in zip(stages, (200, 200), (8, 12)):
        assert scenario_system.drone_hours_daily == hours
        row = pilot.solve_assignment(profiles(), 3000000, config, scenario_system, 1)
        assert row["drone_count"] == count
        assert row["pilot_positions"] == count * scenario_system.pilots_per_drone
        assert row["annual_pilot_cny"] == row["pilot_positions"] * 60000
        assert row["drone_available_capacity_used_pct"] == pytest.approx(80000 / (count * hours * 60) * 100)
        optimum = pilot.solve_assignment(profiles(), 3000000, config, scenario_system)
        for rate in (0, 0.5, 1):
            constrained = pilot.solve_assignment(profiles(), 3000000, config, scenario_system, rate)
            assert optimum["annual_total_cny"] <= constrained["annual_total_cny"] + 10
    cheaper = pilot.solve_assignment(profiles(), 3000000, lower_cost[1], lower_cost[2])
    assert cheaper["annual_total_cny"] <= best["annual_total_cny"]
    # Real-city tiny OD weights must not create a phantom 100-bike fleet.
    tiny = profiles().copy()
    tiny.loc[0, "share"], tiny.loc[1, "share"] = 1e-8, 1 - 1e-8
    free = pilot.solve_assignment(tiny, 84779304, {}, system)
    endpoint = pilot.solve_assignment(tiny, 84779304, {}, system, 1)
    assert free["annual_total_cny"] <= endpoint["annual_total_cny"] + 10


def test_single_city_pilot_preserves_total_demand_and_profiles(tmp_path, monkeypatch):
    city = tmp_path / "data/city/nanjing"
    city.mkdir(parents=True)
    od = pd.DataFrame({
        "distance_km": [1.01, 1.02, 2.], "road_distance_m": [2010., 2020., 4000.],
        "annual_sales": [10000., 30000., 60000.], "ebike_kappa": [0.75] * 3,
        "ebike_time_min": [4.02, 4.04, 8.], "drone_time_min": [2., 2.01, 3.],
    })
    path = city / "nanjing_od_pairs.csv"
    od.to_csv(path, index=False)
    pd.DataFrame([{"region_key": "nanjing", "region_name_en": "Nanjing",
                   "total_odpair_annual_sales": 100000}]).to_csv(tmp_path / "data/all_region_stats.csv", index=False)
    config = {"data_dir": str(tmp_path / "data"), "work_dir": str(tmp_path / "work"),
              "full_cost_pilot": {"replacement_rates": [0., 1.]}}
    grouped, audit = pilot.build_profiles(path, config, pilot.ReferenceSystem(), chunk_rows=1)
    assert audit["profiles"] == 2
    assert np.dot(grouped.share, grouped.ebike_time) == pytest.approx(np.average(od.ebike_time_min, weights=od.annual_sales))
    monkeypatch.setattr(pilot, "locked_region_keys", lambda: ["nanjing", "another_city"])
    monkeypatch.setattr(pilot, "cohort_hash", lambda: "test-cohort")
    paths = pilot.run_full_cost_pilot(config, "nanjing")
    result = pd.read_csv(paths[1])
    assert len(result) == 2
    assert result.annual_orders.eq(100000).all()
    assert "demand_multiplier" not in result and "available_time_fraction" not in result
    assert result.stage.tolist() == ["conservative", "optimistic"]
    assert (result.mean_total_cost_cny.diff().dropna() <= 1e-6).all()
    endpoints = pd.read_csv(paths[0])
    ebike_only = endpoints.loc[endpoints.requested_drone_rate.eq(0), "annual_total_cny"]
    assert ebike_only.max() == pytest.approx(ebike_only.min())
    assert json.loads(paths[2].read_text())["city_annual_orders"] == 100000
    with pytest.raises(ValueError, match="locked paper cohort"):
        pilot.run_full_cost_pilot(config, "outside")
