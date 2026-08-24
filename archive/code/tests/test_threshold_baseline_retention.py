"""Targeted tests for city baseline-retention robustness."""

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

from sensitivity.baseline_retention import (  # noqa: E402
    RETENTION_CURVE_TARGETS,
    RETENTION_SCENARIOS,
    derive_city_baseline_retention,
    summarize_baseline_retention,
)
from visualization.sensitivity import (  # noqa: E402
    THRESHOLD_RESILIENCE_TYPICAL_CITIES,
    classify_threshold_baseline_retention,
    plot_threshold_baseline_retention_curves,
    plot_threshold_baseline_retention_typology,
    validate_threshold_baseline_retention_curve,
)


def _city_source(city: str, baseline: float, n_samples: int) -> pd.DataFrame:
    rows = []
    for scenario in RETENTION_SCENARIOS:
        rows.append({
            "region_key": city,
            "region_name_cn": city,
            "region_name_en": city.title(),
            "province": "test",
            "scenario": scenario,
            "n_samples": n_samples,
            "baseline_fixed_threshold_reduction_pct": baseline,
            "prob_reduction_positive": 0.5,
        })
    return pd.DataFrame(rows)


def _six_city_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    curves = []
    base_values = np.linspace(-12.0, 28.0, 101)
    for city_index, city in enumerate(THRESHOLD_RESILIENCE_TYPICAL_CITIES):
        baseline = 4.0 + city_index * 3.0
        samples = []
        for scenario_index, scenario in enumerate(RETENTION_SCENARIOS):
            samples.append(pd.DataFrame({
                "scenario": scenario,
                "reduction_pct": base_values + city_index - 2.0 * scenario_index,
            }))
        summary, curve = derive_city_baseline_retention(
            _city_source(city, baseline, len(base_values)),
            pd.concat(samples, ignore_index=True),
        )
        summaries.append(summary)
        curves.append(curve)
    return pd.concat(summaries, ignore_index=True), pd.concat(curves, ignore_index=True)


def test_signed_retention_quantiles_and_key_probabilities() -> None:
    reduction = np.array([-10.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5])
    summary, curve = summarize_baseline_retention(
        reduction, 10.0, scenario="independent"
    )
    expected = np.percentile(reduction / 10.0, [5.0, 50.0, 95.0])
    np.testing.assert_allclose(
        [
            summary["baseline_retention_p05"],
            summary["baseline_retention_p50"],
            summary["baseline_retention_p95"],
        ],
        expected,
    )
    assert summary["retain_025pct_count"] == 5
    assert summary["retain_050pct_count"] == 4
    assert summary["retain_100pct_count"] == 2
    zero_target = curve[curve["retention_target_fraction"] == 0.0].iloc[0]
    assert zero_target["success_count"] == 5


def test_zero_baseline_is_auditable_and_undefined() -> None:
    summary, curve = summarize_baseline_retention(
        [0.0, 0.0, 0.0], 0.0, scenario="independent"
    )
    assert not summary["retention_eligible"]
    assert summary["retention_exclusion_reason"] == "nonpositive_baseline"
    assert np.isnan(summary["baseline_retention_p05"])
    assert not curve["retention_eligible"].any()
    assert curve["prob_retain_target"].isna().all()


def test_city_derivation_preserves_paired_scenario_counts() -> None:
    source = _city_source("beijing", 10.0, 4)
    samples = pd.DataFrame({
        "scenario": ["independent"] * 4 + ["correlated_adverse_stress"] * 4,
        "reduction_pct": [1.0, 5.0, 10.0, 15.0, -5.0, 0.0, 5.0, 10.0],
    })
    summary, curve = derive_city_baseline_retention(source, samples)
    assert len(summary) == 2
    assert len(curve) == 2 * len(RETENTION_CURVE_TARGETS)
    independent = summary[summary["scenario"] == "independent"].iloc[0]
    assert independent["prob_retain_050pct_baseline"] == 0.75
    existing_positive = curve[
        (curve["scenario"] == "independent")
        & (curve["retention_target_fraction"] == 0.0)
    ].iloc[0]
    assert existing_positive["prob_retain_target"] == 1.0


def test_curve_validation_and_class_boundaries() -> None:
    _, curve = _six_city_outputs()
    eligible = validate_threshold_baseline_retention_curve(curve)
    assert len(eligible) == len(curve)
    classes = classify_threshold_baseline_retention(
        [-2.0, -0.0001, 0.0, 0.4999, 0.5, 0.9999, 1.0, 4.0, np.nan]
    )
    np.testing.assert_allclose(
        classes[:-1], [0, 0, 1, 1, 2, 2, 3, 3]
    )
    assert np.isnan(classes[-1])


def test_curve_and_typology_figures_render() -> None:
    summary, curve = _six_city_outputs()
    with tempfile.TemporaryDirectory() as temporary_dir:
        output_dir = Path(temporary_dir)
        curve_path = output_dir / "curves.png"
        typology_path = output_dir / "typology.png"
        plot_threshold_baseline_retention_curves(curve, save_path=str(curve_path))
        plot_threshold_baseline_retention_typology(
            summary, save_path=str(typology_path)
        )
        for path in (curve_path, typology_path):
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
