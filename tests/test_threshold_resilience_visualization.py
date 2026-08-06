"""Targeted tests for fixed-threshold resilience visualizations."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src" / "drone_emission"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sensitivity.threshold_resilience import (  # noqa: E402
    PARAMETER_ORDER,
    THRESHOLD_RESILIENCE_QUADRANT_IDS,
    derive_threshold_resilience_typology,
)
from tools import plot_threshold_resilience as plot_runner  # noqa: E402
from visualization.sensitivity import (  # noqa: E402
    THRESHOLD_RESILIENCE_SCENARIOS,
    THRESHOLD_RESILIENCE_TYPICAL_CITIES,
    classify_threshold_resilience,
    plot_threshold_resilience_driver_forest,
    plot_threshold_resilience_typology,
    plot_threshold_resilience_typical_city_distributions,
    prepare_threshold_resilience_drivers,
    validate_threshold_resilience_samples,
    validate_threshold_resilience_summary,
)


def _summary() -> pd.DataFrame:
    rows = []
    for city_index, city in enumerate(THRESHOLD_RESILIENCE_TYPICAL_CITIES):
        for scenario_index, scenario in enumerate(THRESHOLD_RESILIENCE_SCENARIOS):
            rows.append({
                "region_key": city,
                "region_name_cn": city,
                "region_name_en": city.title(),
                "scenario": scenario,
                "n_samples": 10000,
                "prob_reduction_positive": 0.96 - 0.08 * city_index - 0.05 * scenario_index,
                "baseline_fixed_threshold_reduction_pct": 18.0 - 3.0 * city_index,
            })
    return pd.DataFrame(rows)


def _samples() -> dict[str, pd.DataFrame]:
    samples = {}
    quantiles = np.linspace(-2.5, 2.5, 10000)
    for city_index, city in enumerate(THRESHOLD_RESILIENCE_TYPICAL_CITIES):
        frames = []
        for scenario_index, scenario in enumerate(THRESHOLD_RESILIENCE_SCENARIOS):
            frames.append(pd.DataFrame({
                "scenario": scenario,
                "reduction_pct": (
                    20.0 - 4.0 * city_index - 2.0 * scenario_index
                    + (4.0 + scenario_index) * quantiles
                ),
            }))
        samples[city] = pd.concat(frames, ignore_index=True)
    return samples


def _drivers() -> pd.DataFrame:
    rows = []
    for city_index, city in enumerate(("beijing", "shanghai", "guangzhou")):
        for parameter_index, parameter in enumerate(PARAMETER_ORDER):
            rows.append({
                "region_key": city,
                "parameter": parameter,
                "spearman_rho": np.sin(parameter_index + city_index) * 0.65,
            })
    for parameter in PARAMETER_ORDER:
        rows.append({
            "region_key": "linzhi",
            "parameter": parameter,
            "spearman_rho": np.nan,
        })
    return pd.DataFrame(rows)


def _typology_input() -> pd.DataFrame:
    baselines = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    probabilities = {
        "independent": {"a": 0.1, "b": 0.9, "c": 0.2, "d": 0.8},
        "correlated_adverse_stress": {
            "a": 0.8, "b": 0.2, "c": 0.9, "d": 0.1,
        },
    }
    rows = []
    for scenario, city_probabilities in probabilities.items():
        for city, probability in city_probabilities.items():
            rows.append({
                "region_key": city,
                "region_name_cn": city,
                "region_name_en": city.upper(),
                "scenario": scenario,
                "n_samples": 10000,
                "baseline_fixed_threshold_reduction_pct": baselines[city],
                "prob_reduction_positive": probability,
                "prob_positive_ci95_low": max(0.0, probability - 0.02),
                "prob_positive_ci95_high": min(1.0, probability + 0.02),
            })
    return pd.DataFrame(rows)


def test_probability_class_boundaries() -> None:
    actual = classify_threshold_resilience([0.0, 0.4999, 0.5, 0.8999, 0.9, 1.0])
    np.testing.assert_array_equal(actual, [0, 0, 1, 1, 2, 2])
    assert plot_runner._parse_only("typology") == {"typology"}


def test_summary_and_samples_validate_complete_inputs() -> None:
    wide = validate_threshold_resilience_summary(_summary(), expected_cities=6)
    assert set(wide["region_key"]) == set(THRESHOLD_RESILIENCE_TYPICAL_CITIES)
    samples = _samples()
    validate_threshold_resilience_samples(samples)
    broken = dict(samples)
    broken["beijing"] = broken["beijing"].iloc[:-1]
    try:
        validate_threshold_resilience_samples(broken)
    except ValueError as exc:
        assert "expected 10000 samples" in str(exc)
    else:
        raise AssertionError("incomplete samples should be rejected")


def test_driver_preparation_uses_only_complete_finite_cities() -> None:
    finite, rank_one, n_cities = prepare_threshold_resilience_drivers(_drivers())
    assert n_cities == 3
    assert set(finite["region_key"]) == {"beijing", "shanghai", "guangzhou"}
    assert int(rank_one["rank_one_count"].sum()) == 3


def test_typology_derives_separate_four_plus_four_classes() -> None:
    typology, summary = derive_threshold_resilience_typology(
        _typology_input(), expected_cities=4
    )
    assert len(typology) == 8
    assert len(summary) == 8
    assert set(summary["quadrant_id"]) == set(THRESHOLD_RESILIENCE_QUADRANT_IDS)
    assert (summary["city_count"] == 1).all()
    np.testing.assert_allclose(
        summary.groupby("scenario")["city_share"].sum().to_numpy(),
        [1.0, 1.0],
    )
    medians = typology.groupby("scenario")[
        "resilience_probability_median"
    ].first()
    assert medians["independent"] == 0.5
    assert medians["correlated_adverse_stress"] == 0.5
    assert typology["baseline_median"].eq(2.5).all()


def test_typology_marks_ci_overlap_without_creating_a_fifth_class() -> None:
    source = _typology_input()
    mask = (
        (source["region_key"] == "a")
        & (source["scenario"] == "independent")
    )
    source.loc[mask, "prob_positive_ci95_high"] = 0.55
    typology, summary = derive_threshold_resilience_typology(source)
    row = typology[
        (typology["region_key"] == "a")
        & (typology["scenario"] == "independent")
    ].iloc[0]
    assert bool(row["boundary_uncertain"])
    assert row["quadrant_id"] == "low_baseline_low_resilience"
    assert len(summary) == 8


def test_typology_rejects_incomplete_or_inconsistent_scenarios() -> None:
    incomplete = _typology_input().iloc[:-1]
    try:
        derive_threshold_resilience_typology(incomplete)
    except ValueError as exc:
        assert "missing one resilience scenario" in str(exc)
    else:
        raise AssertionError("incomplete city scenarios should be rejected")

    inconsistent = _typology_input()
    mask = (
        (inconsistent["region_key"] == "a")
        & (inconsistent["scenario"] == "correlated_adverse_stress")
    )
    inconsistent.loc[mask, "baseline_fixed_threshold_reduction_pct"] = 1.1
    try:
        derive_threshold_resilience_typology(inconsistent)
    except ValueError as exc:
        assert "baselines differ" in str(exc)
    else:
        raise AssertionError("scenario-specific city baselines should be rejected")


def test_distribution_and_driver_figures_render() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        out_dir = Path(temporary_dir)
        city_path = out_dir / "cities.png"
        driver_path = out_dir / "drivers.png"
        typology_path = out_dir / "typology.png"
        plot_threshold_resilience_typical_city_distributions(
            _summary(), _samples(), save_path=str(city_path)
        )
        plot_threshold_resilience_driver_forest(
            _drivers(), save_path=str(driver_path)
        )
        typology, _ = derive_threshold_resilience_typology(_typology_input())
        plot_threshold_resilience_typology(typology, save_path=str(typology_path))
        for path in (city_path, driver_path, typology_path):
            assert path.exists()
            assert path.stat().st_size > 10000


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
