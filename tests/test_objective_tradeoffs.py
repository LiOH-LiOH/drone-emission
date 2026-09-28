"""Cross-objective decisions, accounting and locked-cohort render contract."""
import numpy as np
import pandas as pd
import pytest

from drone_emission.analysis.full_cost_pilot import ReferenceSystem, build_profiles, development_stages, solve_assignment
from drone_emission.analysis.objective_tradeoffs import evaluate_city, derive_metrics


def profile():
    return pd.DataFrame({
        "orders": [1200000., 1800000.], "share": [.4, .6],
        "ebike_cycle": [10., 20.], "drone_cycle": [5., 10.],
        "ebike_variable": [2., 3.], "drone_variable": [.1, .2],
        "ebike_time": [2., 8.], "drone_time": [4., 3.],
        "ebike_lifecycle_kg": [.02, .03], "drone_lifecycle_kg": [.01, .04],
        "emission_choice": [True, False], "time_choice": [False, True],
    })


def test_exact_objective_masks_survive_distance_grouping(tmp_path):
    od = pd.DataFrame({
        "distance_km": [1.01, 1.02], "road_distance_m": [2010, 2020],
        "annual_sales": [100., 300.], "ebike_kappa": [.75, .75],
        "ebike_time_min": [4., 2.], "drone_time_min": [3., 3.],
        "ebike_lifecycle_g": [10., 10.], "drone_lifecycle_g": [8., 12.],
    })
    path = tmp_path / "od.csv"
    od.to_csv(path, index=False)
    grouped, audit = build_profiles(path, {}, ReferenceSystem(), chunk_rows=1, include_objectives=True)
    assert audit["profiles"] == 2
    selected = np.where(grouped.emission_choice, grouped.drone_lifecycle_kg, grouped.ebike_lifecycle_kg)
    assert np.dot(grouped.orders, selected) == pytest.approx(3.8)
    assert np.dot(grouped.share, grouped.time_choice) == .25


def test_cross_evaluation_fixes_allocation_and_prices_capacity():
    result = evaluate_city(profile(), {}, ReferenceSystem())
    assert len(result) == 12
    assert result.groupby("strategy").drone_replacement_rate_pct.nunique().eq(1).all()
    for scenario, rows in result.groupby("scenario"):
        rows = rows.set_index("strategy")
        assert rows.loc["emission_optimal", "drone_replacement_rate_pct"] == pytest.approx(40)
        assert rows.loc["time_optimal", "drone_replacement_rate_pct"] == pytest.approx(60)
        assert rows.loc["emission_optimal", "annual_lifecycle_kg"] == rows.annual_lifecycle_kg.min()
        assert rows.loc["time_optimal", "mean_transport_time_min"] == rows.mean_transport_time_min.min()
        assert rows.loc[f"cost_{scenario}", "annual_total_cny"] <= rows.annual_total_cny.min() + 1
        if scenario == "conservative":
            assert (rows.annual_pilot_cny == rows.drone_count * 60000).all()
        else:
            assert rows.annual_pilot_cny.eq(0).all()
        assert rows.annual_staff_cny.gt(0).all()
    for _, config, system in development_stages({}, ReferenceSystem()):
        x = profile().share.to_numpy() * .3
        fixed = solve_assignment(profile(), 3000000, config, system, fixed_assignment=x)
        assert fixed["drone_replacement_rate_pct"] == pytest.approx(30)
        assert fixed["network_modules"] == max(fixed["ebike_count"], fixed["drone_count"]) / 100


def test_cohort_ratios_use_totals_and_zero_mitigation_is_undefined():
    first = evaluate_city(profile(), {}, ReferenceSystem())
    second = first.copy()
    total_cols = [c for c in first if c.startswith("annual_") or c == "drone_annual_orders"]
    second[total_cols] *= 3
    second["drone_annual_orders"] *= .5
    combined = pd.concat([first, second]).groupby(["scenario", "strategy"])[total_cols].sum().reset_index()
    totals = derive_metrics(combined)
    row = totals.loc[totals.scenario.eq("conservative") & totals.strategy.eq("emission_optimal")].iloc[0]
    assert row.drone_replacement_rate_pct == pytest.approx(25)
    combined["annual_lifecycle_kg"] = 1.
    assert derive_metrics(combined).mitigation_retained_pct.isna().all()


def test_candidate_manifest_and_registry_outputs_match():
    from drone_emission.paper_workflow import load_manifest
    from drone_emission.visualization.objective_tradeoffs import PANELS
    manifest = load_manifest()
    request = next(r for r in manifest["requests"] if r["id"] == "objective_tradeoffs")
    assert set(request["outputs"]) == {f"objective_{name}.pdf" for name in PANELS}
    candidate = next(f for f in manifest["figures"] if f["id"] == "candidate_objective_tradeoffs")
    assert not candidate["publish"].startswith("docs/paper/")
