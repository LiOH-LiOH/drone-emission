"""Focused checks for transport economics and independent weighted optima."""

from dataclasses import asdict
import json

import numpy as np
import pandas as pd
import pytest

from drone_emission.analysis import cost_time_substitution as analysis
from drone_emission.emission.cal_single_delivery import (
    LifeCycleEmissions, create_drone_params, create_ebike_params,
)


def test_cost_components_match_scalar_energy_and_annual_allocation():
    config = {"emission_params": {"e_use": 0.0}}
    road, air, kappa = np.array([2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0]), 0.75
    actual = analysis.transport_cost_parts(road, air, kappa, config)
    ebike, drone = create_ebike_params(config), create_drone_params(config)
    lca = LifeCycleEmissions(ebike_params=ebike, drone_params=drone)
    np.testing.assert_allclose(actual["ebike"]["rider_cny"], [1.96, 1.96, 2.94])
    for mode, params, distance, speed, price, stock in (
        ("ebike", ebike, road * kappa, ebike.velocity, 400, 2),
        ("drone", drone, air, drone.v_level, 1000, 3),
    ):
        scalar = getattr(lca, f"calculate_{mode}_use_phase_emissions")
        energies = np.array([scalar(float(d))["total_energy_kwh"] for d in distance])
        np.testing.assert_allclose(actual[mode]["electricity_cny"], energies * 0.5 / 0.81)
        annual_orders = params.working_hours_per_day * 60 * params.working_days_per_year / (
            2 * distance / speed * 60 + params.loading_time_min
        )
        np.testing.assert_allclose(actual[mode]["battery_cny"], 0.95 * price * stock / 2 / annual_orders)
        assert (actual[mode]["electricity_cny"] > 0).all()
    longer = analysis.transport_cost_parts(road, air, kappa, {
        **config, "transport_cost": {"battery_life_years": 4.0},
    })
    np.testing.assert_allclose(longer["drone"]["battery_cny"], actual["drone"]["battery_cny"] / 2)


def write_fixture(path):
    pd.DataFrame({
        "distance_km": [1., 2., 3.], "road_distance_m": [1000., 2000., 3000.],
        "annual_sales": [10., 30., 60.], "ebike_kappa": [0.75] * 3,
        "ebike_lifecycle_g": [10., 20., 5.], "drone_lifecycle_g": [5., 20., 10.],
        "ebike_time_min": [4., 4., 4.], "drone_time_min": [5., 4., 2.],
    }).to_csv(path, index=False)
    return pd.Series({
        "region_key": "beijing", "region_name_en": "Beijing",
        "total_ebike_lifecycle": 1., "oracle_min_lifecycle_kg": 0.95,
        "oracle_drone_order_ratio_pct": 10.,
    })


def fake_costs(road, air, kappa, config):
    return {
        "ebike": {"total_cny": np.full(len(road), 3.)},
        "drone": {"total_cny": np.where(road == 2, 1., 3.)},
    }


def test_weighted_objectives_ties_chunking_and_cohort_means(tmp_path, monkeypatch):
    path = tmp_path / "od.csv"
    stats = write_fixture(path)
    monkeypatch.setattr(analysis, "transport_cost_parts", fake_costs)
    frame, report = analysis.calculate_city_strategies(path, stats, {}, chunk_rows=1)
    whole, _ = analysis.calculate_city_strategies(path, stats, {}, chunk_rows=100)
    pd.testing.assert_frame_equal(frame, whole)
    rows = frame.set_index("strategy")
    np.testing.assert_allclose(rows["drone_replacement_rate_pct"], [0, 100, 10, 30, 60])
    assert rows.loc["time_optimal", "mean_transport_time_min"] == pytest.approx(2.8)
    for strategy, metric in (
        ("emission_optimal", "annual_lifecycle_kg"),
        ("cost_optimal", "annual_transport_cost_cny"),
        ("time_optimal", "mean_transport_time_min"),
    ):
        assert rows.loc[strategy, metric] == pytest.approx(rows[metric].min())
    assert all(item["matches"] for item in report["main_experiment_comparison"].values())
    second = frame.copy()
    second[list(analysis.TOTAL_COLUMNS)] *= 3
    second["annual_transport_minutes"] *= 2
    combined = pd.concat([frame, second]).groupby("strategy", sort=False)[list(analysis.TOTAL_COLUMNS)].sum().reset_index()
    cohort = analysis.summarize_totals(combined).set_index("strategy")
    assert cohort.loc["time_optimal", "mean_transport_time_min"] == pytest.approx(2.8 * 1.75)


def test_invalid_time_excludes_same_orders_from_every_objective(tmp_path):
    path = tmp_path / "od.csv"
    stats = write_fixture(path)
    data = pd.read_csv(path)
    data.loc[0, "drone_time_min"] = np.nan
    data.to_csv(path, index=False)
    frame, report = analysis.calculate_city_strategies(path, stats, {})
    assert frame["annual_orders"].eq(90).all()
    assert report["excluded_positive_annual_orders"] == 10
    assert not report["main_experiment_comparable"]


def test_main_experiment_mismatch_is_not_silently_accepted(tmp_path):
    path = tmp_path / "od.csv"
    stats = write_fixture(path)
    stats["oracle_min_lifecycle_kg"] = 10
    with pytest.raises(ValueError, match="differs from main experiment"):
        analysis.calculate_city_strategies(path, stats, {})


def test_outputs_cohort_contract_and_economic_cache_invalidation(tmp_path, monkeypatch):
    data = tmp_path / "data"
    city = data / "city" / "beijing"
    city.mkdir(parents=True)
    stats = write_fixture(city / "beijing_od_pairs.csv")
    stats.to_frame().T.to_csv(data / "all_region_stats.csv", index=False)
    monkeypatch.setattr(analysis, "locked_region_keys", lambda: ["beijing"])
    monkeypatch.setattr(analysis, "cohort_hash", lambda: "test-cohort")
    config = {"data_dir": str(data), "work_dir": str(tmp_path / "work")}
    paths = analysis.run_cost_time_analysis(config)
    first_cost = pd.read_csv(paths[1])["annual_transport_cost_cny"].to_numpy()
    times = [path.stat().st_mtime_ns for path in paths]
    analysis.run_cost_time_analysis(config)
    assert times == [path.stat().st_mtime_ns for path in paths]
    config["transport_cost"] = {"electricity_cny_per_kwh": 1.0}
    analysis.run_cost_time_analysis(config)
    assert (pd.read_csv(paths[1])["annual_transport_cost_cny"].to_numpy() > first_cost).all()
    assert json.loads(paths[2].read_text())["city_count"] == 1
    monkeypatch.setattr(analysis, "locked_region_keys", lambda: ["beijing", "missing"])
    with pytest.raises(ValueError, match="locked paper cohort"):
        analysis.run_cost_time_analysis(config)


def test_workflow_registration_and_config_defaults():
    from drone_emission.analysis_workflow import ANALYSIS_MODULES, _module_outputs
    from drone_emission.paths import PROJECT_ROOT
    from drone_emission.utils.config import load_config

    assert "cost-time-substitution" in ANALYSIS_MODULES
    config = load_config(str(PROJECT_ROOT / "experiment.toml"), env="remote")
    assert config["transport_cost"] == asdict(analysis.CostParameters())
    assert len(_module_outputs(config)["cost-time-substitution"]) == 3
