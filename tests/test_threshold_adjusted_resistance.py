"""Targeted tests for baseline-adjusted perturbation resistance."""

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

from sensitivity.adjusted_resistance import (  # noqa: E402
    derive_adjusted_resistance,
    summarize_city_downside,
    validate_downside_table,
)
from visualization.sensitivity import (  # noqa: E402
    plot_threshold_adjusted_resistance_diagnostic,
)


def _downside_table(n_cities: int = 48) -> pd.DataFrame:
    rows = []
    residual_pattern = np.array([-2.4, -1.5, -0.8, -0.2, 0.3, 0.9, 1.6, 2.5])
    for index in range(n_cities):
        baseline = 2.0 + 0.8 * index
        residual = residual_pattern[index % len(residual_pattern)]
        for scenario_index, scenario in enumerate(
            ("independent", "correlated_adverse_stress")
        ):
            typical = 8.0 + scenario_index * 9.0 + 0.45 * baseline
            downside_p05 = typical + residual * (1.0 + 0.3 * scenario_index)
            downside_p10 = typical - 1.4 + residual * (0.8 + 0.2 * scenario_index)
            rows.append({
                "region_key": f"city_{index:03d}",
                "region_name_cn": f"城市{index}",
                "region_name_en": f"City {index}",
                "province": "test",
                "scenario": scenario,
                "n_samples": 10000,
                "baseline_fixed_threshold_reduction_pct": baseline,
                "reduction_p05_pct": baseline - downside_p05,
                "downside_p05_pp": downside_p05,
                "reduction_p10_pct": baseline - downside_p10,
                "downside_p10_pp": downside_p10,
            })
    return pd.DataFrame(rows)


def test_absolute_downside_is_translation_invariant() -> None:
    original = summarize_city_downside([-4.0, 2.0, 8.0, 14.0], 10.0)
    shifted = summarize_city_downside([6.0, 12.0, 18.0, 24.0], 20.0)
    assert original["downside_p05_pp"] == shifted["downside_p05_pp"]
    assert original["downside_p10_pp"] == shifted["downside_p10_pp"]


def test_derivation_produces_three_plus_three_classes() -> None:
    detail, summary, diagnostics, amplification = derive_adjusted_resistance(
        _downside_table(), expected_cities=48, bootstrap_replicates=0
    )
    assert len(detail) == 96
    assert len(summary) == 6
    assert set(summary["resistance_class"]) == {"high", "typical", "low"}
    assert len(diagnostics) == 4
    assert len(amplification) == 48
    assert detail["resistance_eligible"].all()
    assert np.isfinite(detail["adjusted_resistance_score_p05"]).all()
    assert (
        amplification["stress_amplification_p05_pp"] > 0
    ).all()


def test_nonpositive_baseline_is_auditable_and_excluded() -> None:
    source = _downside_table()
    mask = source["region_key"] == "city_000"
    for suffix in ("p05", "p10"):
        source.loc[mask, f"reduction_{suffix}_pct"] = (
            -source.loc[mask, f"downside_{suffix}_pp"]
        )
    source.loc[mask, "baseline_fixed_threshold_reduction_pct"] = 0.0
    detail, _, _, amplification = derive_adjusted_resistance(
        source, expected_cities=48, bootstrap_replicates=0
    )
    excluded = detail[detail["region_key"] == "city_000"]
    assert not excluded["resistance_eligible"].any()
    assert excluded["adjusted_resistance_score_p05"].isna().all()
    city = amplification[amplification["region_key"] == "city_000"].iloc[0]
    assert not city["resistance_eligible"]
    assert np.isnan(city["stress_amplification_p05_pp"])


def test_validation_rejects_inconsistent_downside() -> None:
    broken = _downside_table()
    broken.loc[0, "downside_p05_pp"] += 1.0
    try:
        validate_downside_table(broken)
    except ValueError as error:
        assert "inconsistent" in str(error)
    else:
        raise AssertionError("inconsistent downside was not rejected")


def test_bootstrap_is_deterministic() -> None:
    first = derive_adjusted_resistance(
        _downside_table(), bootstrap_replicates=12, bootstrap_seed=73
    )[0]
    second = derive_adjusted_resistance(
        _downside_table(), bootstrap_replicates=12, bootstrap_seed=73
    )[0]
    np.testing.assert_allclose(
        first["class_stability"], second["class_stability"], equal_nan=True
    )
    np.testing.assert_allclose(
        first["adjusted_score_ci95_low"],
        second["adjusted_score_ci95_low"],
        equal_nan=True,
    )


def test_diagnostic_figure_renders() -> None:
    detail = derive_adjusted_resistance(
        _downside_table(), bootstrap_replicates=0
    )[0]
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "adjusted_resistance.png"
        plot_threshold_adjusted_resistance_diagnostic(detail, save_path=str(path))
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
