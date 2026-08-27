"""Targeted tests for fixed-threshold probabilistic resilience."""

from __future__ import annotations

import os
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))

from drone_emission.emission.cal_single_delivery import (  # noqa: E402
    DroneParams,
    EbikeParams,
    create_drone_params,
    create_ebike_params,
)
from drone_emission.sensitivity.threshold_resilience import (  # noqa: E402
    PARAMETER_ORDER,
    ThresholdResilienceContext,
    TriangularSpec,
    aggregate_fixed_threshold,
    evaluate_samples,
    mode_sample,
    resolve_distributions,
    sample_parameters,
    spearman_drivers,
    summarize_samples,
    wilson_interval,
)
from drone_emission.sensitivity import threshold_resilience as resilience  # noqa: E402
from drone_emission.sensitivity.vectorized_lca import (  # noqa: E402
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
    compute_ebike_per_order_parts_g,
)
from drone_emission.utils import load_config  # noqa: E402
from drone_emission.utils.region_config import RegionConfig  # noqa: E402
from drone_emission.tools import run_threshold_resilience as runner  # noqa: E402


def _specs() -> dict[str, TriangularSpec]:
    modes = {
        "kappa": 0.7692307692307692,
        "payload_kg": 1.0,
        "drone_hover_s": 10.0,
        "headwind_mps": 0.0,
        "drone_cruise_kmh": 60.0,
        "drone_energy_mult": 1.0,
        "ebike_energy_mult": 1.0,
        "drone_lifespan_yr": 4.0,
        "drone_battery_lifespan_yr": 2.0,
        "ebike_lifespan_yr": 4.0,
        "ebike_battery_lifespan_yr": 2.0,
        "drone_utilization_rate": 1.0,
        "ebike_utilization_rate": 1.0,
        "drone_production_co2_per_kg": 9.3,
        "drone_battery_co2_per_kwh": 141.0,
        "ebike_production_co2_per_kg": 5.65,
        "ebike_battery_co2_per_kwh": 107.0,
    }
    widths = {name: max(abs(mode) * 0.1, 0.1) for name, mode in modes.items()}
    return {
        name: TriangularSpec(
            modes[name] - widths[name], modes[name], modes[name] + widths[name]
        )
        for name in PARAMETER_ORDER
    }


def test_sampling_reproducible_and_bounded() -> None:
    specs = _specs()
    first = sample_parameters(specs, 500, 42)
    second = sample_parameters(specs, 500, 42)
    pd.testing.assert_frame_equal(first, second)
    for name, spec in specs.items():
        assert first[name].between(spec.minimum, spec.maximum).all()


def _context(n_samples: int = 250) -> ThresholdResilienceContext:
    specs = _specs()
    return ThresholdResilienceContext.create(
        specs,
        EbikeParams(kappa=specs["kappa"].mode),
        DroneParams(hover_time=specs["drone_hover_s"].mode),
        70.0,
        n_samples,
        42,
        correlated_stress=True,
        stress_spearman_rho=0.5,
        config_sha256="test-config",
    )


def test_shared_context_preserves_seeded_samples_and_results() -> None:
    context = _context()
    pd.testing.assert_frame_equal(
        context.samples_by_scenario["independent"],
        sample_parameters(context.specs, 250, 42),
    )
    pd.testing.assert_frame_equal(
        context.samples_by_scenario["correlated_adverse_stress"],
        sample_parameters(
            context.specs, 250, 43, stress_spearman_rho=0.5
        ),
    )
    aggregates_by_city = (
        aggregate_fixed_threshold(
            [0.5, 1.0, 2.0], [0.4, 0.8, 1.5], [10.0, 20.0, 30.0], 1.0
        ),
        aggregate_fixed_threshold(
            [0.7, 1.4, 3.2], [0.5, 1.0, 2.2], [7.0, 13.0, 31.0], 1.5
        ),
    )
    for province_gef in (0.42, 0.71):
        for aggregates in aggregates_by_city:
            baseline = context.evaluate("mode", aggregates, province_gef)
            legacy_baseline = evaluate_samples(
                context.mode_samples,
                aggregates,
                context.ebike_params,
                context.drone_params,
                province_gef,
                context.rider_weight,
            )
            pd.testing.assert_frame_equal(baseline, legacy_baseline)
            baseline_reduction = float(baseline.iloc[0]["reduction_pct"])
            for scenario, samples in context.samples_by_scenario.items():
                shared = context.evaluate(scenario, aggregates, province_gef)
                legacy = evaluate_samples(
                    samples,
                    aggregates,
                    context.ebike_params,
                    context.drone_params,
                    province_gef,
                    context.rider_weight,
                )
                pd.testing.assert_frame_equal(shared, legacy)
                assert summarize_samples(
                    shared, scenario, aggregates, baseline_reduction
                ) == summarize_samples(
                    legacy, scenario, aggregates, baseline_reduction
                )


def test_shared_coefficients_are_cached_by_exact_gef_and_scenario() -> None:
    context = _context(n_samples=25)
    aggregates = aggregate_fixed_threshold(
        [0.5, 1.0, 2.0], [0.4, 0.8, 1.5], [10.0, 20.0, 30.0], 1.0
    )
    calls = []
    original = resilience._affine_coefficients

    def counted(*args, **kwargs):
        calls.append((float(args[3]), len(args[0])))
        return original(*args, **kwargs)

    resilience._affine_coefficients = counted
    try:
        for province_gef in (0.42, 0.42, 0.71):
            for scenario in (
                "mode",
                "independent",
                "correlated_adverse_stress",
            ):
                context.evaluate(scenario, aggregates, province_gef)
    finally:
        resilience._affine_coefficients = original

    assert len(calls) == 2 * 3
    assert set(context.coefficient_cache) == {
        (province_gef, scenario)
        for province_gef in (0.42, 0.71)
        for scenario in (
            "mode",
            "independent",
            "correlated_adverse_stress",
        )
    }


def test_shared_spearman_ranks_match_individual_scipy_calls() -> None:
    context = _context()
    aggregates = aggregate_fixed_threshold(
        [0.5, 1.0, 2.0], [0.4, 0.8, 1.5], [10.0, 20.0, 30.0], 1.0
    )
    evaluated = context.evaluate("independent", aggregates, 0.42)
    actual = spearman_drivers(
        evaluated, context.independent_parameter_ranks
    ).set_index("parameter")
    for name in PARAMETER_ORDER:
        rho, pvalue = spearmanr(evaluated[name], evaluated["reduction_pct"])
        assert np.isclose(actual.loc[name, "spearman_rho"], rho, rtol=0.0, atol=1e-15)
        assert np.isclose(
            actual.loc[name, "spearman_pvalue"], pvalue, rtol=1e-13, atol=1e-300
        )
        assert actual.loc[name, "direction"] == (
            "increases_reduction" if rho > 0 else "decreases_reduction"
        )
    expected_order = sorted(
        PARAMETER_ORDER,
        key=lambda name: (-abs(float(actual.loc[name, "spearman_rho"])), name),
    )
    assert actual.sort_values("rank").index.tolist() == expected_order


def test_project_config_matches_runtime_modes() -> None:
    config = load_config(os.path.join(HERE, "..", "experiment.toml"), env="local")
    specs = resolve_distributions(
        config, create_ebike_params(config), create_drone_params(config)
    )
    assert tuple(specs) == PARAMETER_ORDER


def test_stress_copula_risk_directions() -> None:
    samples = sample_parameters(
        _specs(), 20000, 43, stress_spearman_rho=0.5
    )
    same, _ = spearmanr(samples["drone_energy_mult"], samples["drone_hover_s"])
    opposite, _ = spearmanr(samples["drone_energy_mult"], samples["kappa"])
    independent, _ = spearmanr(
        samples["drone_energy_mult"], samples["drone_cruise_kmh"]
    )
    assert abs(same - 0.5) < 0.03
    assert abs(opposite + 0.5) < 0.03
    assert abs(independent) < 0.03


def test_threshold_equality_goes_to_drone() -> None:
    aggregates = aggregate_fixed_threshold(
        road_distance_km=[0.5, 1.0, 2.0],
        straight_distance_km=[0.4, 0.8, 1.5],
        annual_sales=[10.0, 20.0, 30.0],
        threshold_km=1.0,
    )
    assert aggregates.ebike_orders == 10.0
    assert aggregates.drone_orders == 50.0
    assert aggregates.drone_air_order_km == 61.0


def test_affine_aggregation_matches_direct_od_evaluation() -> None:
    road = np.array([0.5, 1.0, 2.0, 3.5])
    straight = np.array([0.4, 0.8, 1.5, 2.4])
    sales = np.array([10.0, 20.0, 30.0, 40.0])
    threshold = 1.0
    aggregates = aggregate_fixed_threshold(road, straight, sales, threshold)
    specs = _specs()
    samples = mode_sample(specs)
    samples.loc[0, "payload_kg"] = 1.05
    samples.loc[0, "headwind_mps"] = 0.05
    ebike_params = EbikeParams(kappa=specs["kappa"].mode, e_use=0.558)
    drone_params = DroneParams(hover_time=10.0, e_use=0.558)
    evaluated = evaluate_samples(
        samples, aggregates, ebike_params, drone_params, 0.558, 70.0
    ).iloc[0]
    row = samples.iloc[0]
    ep = replace(
        ebike_params,
        e_body=row.ebike_production_co2_per_kg,
        e_battery=row.ebike_battery_co2_per_kwh,
    )
    dp = replace(
        drone_params,
        v_level=row.drone_cruise_kmh,
        hover_time=row.drone_hover_s,
        headwind_speed_mps=row.headwind_mps,
        e_body=row.drone_production_co2_per_kg,
        e_battery=row.drone_battery_co2_per_kwh,
    )
    ebike_g = compute_ebike_per_order_g(
        road,
        ep,
        EbikeScenario(
            kappa=row.kappa,
            energy_mult=row.ebike_energy_mult,
            payload_kg=row.payload_kg,
            rider_weight=70.0,
            lifespan_yr=row.ebike_lifespan_yr,
            battery_lifespan_yr=row.ebike_battery_lifespan_yr,
            utilization_mult=row.ebike_utilization_rate,
        ),
    )
    drone_g = compute_drone_per_order_g(
        straight,
        dp,
        DroneScenario(
            energy_mult=row.drone_energy_mult,
            payload_kg=row.payload_kg,
            lifespan_yr=row.drone_lifespan_yr,
            battery_lifespan_yr=row.drone_battery_lifespan_yr,
            utilization_mult=row.drone_utilization_rate,
        ),
    )
    use_drone = road >= threshold
    expected_ebike_kg = float(np.sum(ebike_g * sales) / 1000.0)
    expected_mixed_kg = float(
        np.sum(np.where(use_drone, drone_g, ebike_g) * sales) / 1000.0
    )
    assert np.isclose(evaluated.all_ebike_lifecycle_kg, expected_ebike_kg)
    assert np.isclose(evaluated.mixed_lifecycle_kg, expected_mixed_kg)


def test_ebike_utilization_only_scales_embodied_parts() -> None:
    distances = np.array([0.5, 2.0])
    baseline = compute_ebike_per_order_parts_g(
        distances, EbikeParams(), EbikeScenario(utilization_mult=1.0)
    )
    half = compute_ebike_per_order_parts_g(
        distances, EbikeParams(), EbikeScenario(utilization_mult=0.5)
    )
    assert np.allclose(half["production_g"], baseline["production_g"] * 2.0)
    assert np.allclose(half["end_of_life_g"], baseline["end_of_life_g"] * 2.0)
    assert np.allclose(half["use_g"], baseline["use_g"])


def test_wilson_interval_extremes() -> None:
    low_zero, high_zero = wilson_interval(0, 100)
    low_all, high_all = wilson_interval(100, 100)
    assert low_zero == 0.0 and 0.0 < high_zero < 0.05
    assert high_all == 1.0 and 0.95 < low_all < 1.0


def test_runner_inventory_classifies_and_orders_cities() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_dir = root / "data"
        regions_path = root / "regions.json"
        regions_path.write_text(
            json.dumps(
                {
                    "large": {
                        "region_name_cn": "大城",
                        "region_name_en": "Large",
                        "region_level": "city",
                    },
                    "stale": {
                        "region_name_cn": "旧城",
                        "region_name_en": "Stale",
                        "region_level": "city",
                    },
                    "missing": {
                        "region_name_cn": "缺城",
                        "region_name_en": "Missing",
                        "region_level": "city",
                    },
                }
            ),
            encoding="utf-8",
        )
        gef_path = root / "gef.csv"
        gef_path.write_text("province,factor\n测试,0.5\n", encoding="utf-8")
        config = {
            "data_dir": str(data_dir),
            "gef_path": str(gef_path),
            "threshold_resilience": {"distance_basis": "road"},
            "emission_params": {},
        }

        for name, rows in (("large", 3), ("stale", 1)):
            city_dir = data_dir / "city" / name
            city_dir.mkdir(parents=True)
            od_path = city_dir / f"{name}_od_pairs.csv"
            pd.DataFrame(
                {
                    "distance_km": np.arange(rows, dtype=float) + 1.0,
                    "road_distance_m": np.arange(rows, dtype=float) + 1000.0,
                    "annual_sales": np.ones(rows),
                }
            ).to_csv(od_path, index=False)
            stat = od_path.stat()
            threshold_dir = city_dir / "analysis" / "distance_threshold"
            threshold_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "distance_basis": ["road"],
                    "optimal_threshold_km": [1.0],
                    "annual_reduction_pct": [2.0],
                    "input_size_bytes": [stat.st_size],
                    "input_mtime_ns": [stat.st_mtime_ns if name == "large" else 0],
                }
            ).to_csv(threshold_dir / "summary.csv", index=False)

        old_regions_json = runner.REGIONS_JSON
        runner.REGIONS_JSON = regions_path
        try:
            inventory = runner.inventory_regions(
                config, require_samples=True, force=False
            )
        finally:
            runner.REGIONS_JSON = old_regions_json

        assert inventory["region_key"].tolist() == ["large", "stale", "missing"]
        statuses = inventory.set_index("region_key")["status"].to_dict()
        assert statuses == {
            "large": "ready",
            "stale": "stale_threshold",
            "missing": "missing_od",
        }


def test_analysis_fingerprint_tracks_gef_and_emission_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        gef_path = Path(temporary) / "gef.csv"
        gef_path.write_text("province,factor\n测试,0.5\n", encoding="utf-8")
        config = {
            "gef_path": str(gef_path),
            "threshold_resilience": {"n_samples": 10},
            "emission_params": {"payload_kg": 1.0},
        }
        baseline = runner._analysis_config_digest(config)
        config["emission_params"]["payload_kg"] = 1.1
        assert runner._analysis_config_digest(config) != baseline
        config["emission_params"]["payload_kg"] = 1.0
        gef_path.write_text("province,factor\n测试,0.6\n", encoding="utf-8")
        assert runner._analysis_config_digest(config) != baseline


def test_runner_reuses_context_and_invalidates_old_analysis_version() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        config = load_config(
            os.path.join(HERE, "..", "experiment.toml"), env="local"
        )
        config["data_dir"] = str(root / "data")
        config["gef_path"] = str(root / "gef.csv")
        config["threshold_resilience"]["n_samples"] = 25
        Path(config["gef_path"]).write_text(
            "城市,因子\n测试,0.42\n", encoding="gbk"
        )

        regions = {
            key: RegionConfig(
                region_name_cn=f"测试{index}",
                region_name_en=key.title(),
                region_level="city",
                stats={"province": "测试"},
            )
            for index, key in enumerate(("alpha", "beta"), start=1)
        }
        regions_path = root / "regions.json"
        regions_path.write_text(
            json.dumps(
                {
                    key: {
                        "region_name_cn": region.region_name_cn,
                        "region_name_en": region.region_name_en,
                        "region_level": region.region_level,
                        "stats": region.stats,
                    }
                    for key, region in regions.items()
                }
            ),
            encoding="utf-8",
        )
        context = runner.build_execution_context(config)
        for key, region in regions.items():
            city_dir = Path(config["data_dir"]) / "city" / key
            city_dir.mkdir(parents=True)
            od_path = city_dir / f"{key}_od_pairs.csv"
            od_frame = pd.DataFrame(
                {
                    "distance_km": [0.4, 0.8, 1.5],
                    "road_distance_m": [500.0, 1000.0, 2000.0],
                    "annual_sales": [10.0, 20.0, 30.0],
                }
            )
            od_frame.to_csv(od_path, index=False)
            aggregates = aggregate_fixed_threshold(
                od_frame["road_distance_m"].to_numpy() / 1000.0,
                od_frame["distance_km"].to_numpy(),
                od_frame["annual_sales"].to_numpy(),
                1.0,
            )
            baseline = float(
                context.evaluate("mode", aggregates, 0.42).iloc[0][
                    "reduction_pct"
                ]
            )
            stat = od_path.stat()
            threshold_dir = city_dir / "analysis" / "distance_threshold"
            threshold_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "distance_basis": ["road"],
                    "optimal_threshold_km": [1.0],
                    "annual_reduction_pct": [baseline],
                    "input_size_bytes": [stat.st_size],
                    "input_mtime_ns": [stat.st_mtime_ns],
                }
            ).to_csv(threshold_dir / "summary.csv", index=False)

        first = runner.analyze_region("alpha", regions["alpha"], config)
        assert first["status"] == "ok"
        cached = runner.analyze_region("alpha", regions["alpha"], config)
        assert cached["status"] == "cached"

        metadata_path = (
            Path(config["data_dir"])
            / "city"
            / "alpha"
            / "analysis"
            / "threshold_resilience"
            / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["analysis_version"] = "1.1.0"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        refreshed = runner.analyze_region("alpha", regions["alpha"], config)
        assert refreshed["status"] == "ok"

        for key, region in regions.items():
            result = runner.analyze_region(
                key,
                region,
                config,
                force=True,
                execution_context=context,
            )
            assert result["status"] == "ok"
        assert len(context.coefficient_cache) == 3
        old_regions_json = runner.REGIONS_JSON
        runner.REGIONS_JSON = regions_path
        try:
            summary_out, drivers_out = runner.consolidate(config)
        finally:
            runner.REGIONS_JSON = old_regions_json
        assert pd.read_csv(summary_out).shape[0] == 4
        assert pd.read_csv(drivers_out).shape[0] == 2 * len(PARAMETER_ORDER)


def main() -> int:
    tests = (
        test_sampling_reproducible_and_bounded,
        test_shared_context_preserves_seeded_samples_and_results,
        test_shared_coefficients_are_cached_by_exact_gef_and_scenario,
        test_shared_spearman_ranks_match_individual_scipy_calls,
        test_project_config_matches_runtime_modes,
        test_stress_copula_risk_directions,
        test_threshold_equality_goes_to_drone,
        test_affine_aggregation_matches_direct_od_evaluation,
        test_ebike_utilization_only_scales_embodied_parts,
        test_wilson_interval_extremes,
        test_runner_inventory_classifies_and_orders_cities,
        test_analysis_fingerprint_tracks_gef_and_emission_inputs,
        test_runner_reuses_context_and_invalidates_old_analysis_version,
    )
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"All {len(tests)} threshold-resilience tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
