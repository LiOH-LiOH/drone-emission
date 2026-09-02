import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drone_emission.visualization.distance_threshold import (
    DISTANCE_THRESHOLD_TYPICAL_CITIES,
    _draw_distance_threshold_representative_city_panels,
    _prepare_distance_threshold_all_city_data,
    _prepare_distance_threshold_representative_city_data,
)


def test_road_policy_uses_full_city_baseline_for_oracle_comparison() -> None:
    summary = pd.DataFrame(
        {
            "region_key": ["example", "example"],
            "region_name_en": ["Example", "Example"],
            "distance_basis": ["road", "straight"],
            "optimal_threshold_km": [2.0, 1.5],
            "annual_reduction_kg": [10.0, 12.0],
            "annual_reduction_pct": [20.0, 24.0],
            "n_od_pairs_total": [12, 12],
            "n_od_pairs_unreachable": [2, 2],
            "unreachable_annual_orders": [10.0, 10.0],
            "total_annual_orders": [90.0, 90.0],
        }
    )
    oracle = pd.DataFrame(
        {
            "region_key": ["example"],
            "region_level": ["city"],
            "total_ebike_lifecycle": [100.0],
            "oracle_min_lifecycle_kg": [85.0],
        }
    )

    wide = _prepare_distance_threshold_all_city_data(summary, oracle)

    row = wide.iloc[0]
    assert row["road_annual_reduction_pct"] == 20.0
    assert row["road_full_domain_reduction_pct"] == 10.0
    assert row["oracle_annual_reduction_pct"] == 15.0
    assert row["road_oracle_gap_pp"] == 5.0


def _representative_wide() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region_key": list(DISTANCE_THRESHOLD_TYPICAL_CITIES),
            "road_full_domain_reduction_pct": [34.3, 18.6, 11.9, 7.6],
            "oracle_annual_reduction_pct": [34.4, 19.9, 14.0, 10.2],
            "road_optimal_threshold_km": [0.31, 1.32, 1.55, 1.71],
        }
    )


def test_representative_city_retention_preserves_locked_order() -> None:
    representative = _prepare_distance_threshold_representative_city_data(
        _representative_wide()
    )

    assert representative["region_key"].tolist() == list(
        DISTANCE_THRESHOLD_TYPICAL_CITIES
    )
    np.testing.assert_allclose(
        representative["retention_pct"],
        [
            34.3 / 34.4 * 100.0,
            18.6 / 19.9 * 100.0,
            11.9 / 14.0 * 100.0,
            7.6 / 10.2 * 100.0,
        ],
    )


def test_representative_city_retention_axis_uses_full_percentage_range() -> None:
    representative = _prepare_distance_threshold_representative_city_data(
        _representative_wide()
    )
    fig, axes = plt.subplots(1, 3)
    try:
        _draw_distance_threshold_representative_city_panels(
            axes, representative
        )

        assert axes[1].get_xlim() == (0.0, 100.0)
        np.testing.assert_allclose(
            axes[1].get_xticks(), [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        )
        assert all(
            "axis starts" not in text.get_text().lower()
            for text in axes[1].texts
        )
    finally:
        plt.close(fig)
