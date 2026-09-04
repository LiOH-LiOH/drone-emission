import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from drone_emission.visualization.distance_threshold import (
    DISTANCE_THRESHOLD_LANDSCAPE_AXIS_LABELS,
    DISTANCE_THRESHOLD_ORACLE_AXIS_LABELS,
    DISTANCE_THRESHOLD_TYPICAL_CITIES,
    _add_distance_threshold_macro_region_legend,
    _draw_distance_threshold_representative_city_panels,
    _plot_distance_threshold_all_city_scatter,
    _prepare_distance_threshold_all_city_data,
    _prepare_distance_threshold_representative_city_data,
)
from drone_emission.visualization.macro_regions import (
    CHINA_MACRO_REGION_SPECS,
)


MACRO_REGION_COLOR_BY_KEY = {
    spec["key"]: spec["color"] for spec in CHINA_MACRO_REGION_SPECS
}


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
            "province": ["北京"],
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
    assert row["macro_region_key"] == "north_china"
    assert (
        row["macro_region_color"]
        == MACRO_REGION_COLOR_BY_KEY["north_china"]
    )


def test_unmapped_province_fails_macro_region_assignment() -> None:
    summary = pd.DataFrame(
        {
            "region_key": ["example", "example"],
            "region_name_en": ["Example", "Example"],
            "distance_basis": ["road", "straight"],
            "optimal_threshold_km": [2.0, 1.5],
            "annual_reduction_kg": [10.0, 12.0],
            "annual_reduction_pct": [20.0, 24.0],
            "n_od_pairs_total": [12, 12],
            "n_od_pairs_unreachable": [0, 0],
            "unreachable_annual_orders": [0.0, 0.0],
            "total_annual_orders": [100.0, 100.0],
        }
    )
    oracle = pd.DataFrame(
        {
            "region_key": ["example"],
            "province": ["Unknown"],
            "total_ebike_lifecycle": [100.0],
            "oracle_min_lifecycle_kg": [85.0],
        }
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "without a China macro-region",
    ):
        _prepare_distance_threshold_all_city_data(summary, oracle)


def _representative_wide() -> pd.DataFrame:
    macro_region_keys = [
        "southwest_china",
        "south_china",
        "north_china",
        "east_china",
    ]
    return pd.DataFrame(
        {
            "region_key": list(DISTANCE_THRESHOLD_TYPICAL_CITIES),
            "province": ["四川", "广东", "北京", "上海"],
            "macro_region_key": macro_region_keys,
            "macro_region_label": [
                "Southwest",
                "South China",
                "North China",
                "East China",
            ],
            "macro_region_color": [
                MACRO_REGION_COLOR_BY_KEY[key] for key in macro_region_keys
            ],
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
        assert len(axes[1].texts) == len(representative)
        for label, bar in zip(axes[1].texts, axes[1].patches):
            assert label.get_color() == "white"
            assert label.get_ha() == "right"
            assert label.get_position()[0] < 0
            assert label.xy[0] == bar.get_width()
        expected_colors = representative["macro_region_color"].tolist()
        for axis in axes[1:]:
            actual_colors = [bar.get_facecolor() for bar in axis.patches]
            expected_rgba = [
                to_rgba(color)
                for color in expected_colors
            ]
            np.testing.assert_allclose(actual_colors, expected_rgba)
        road_marker = axes[0].collections[1].get_paths()[0].vertices
        oracle_marker = axes[0].collections[2].get_paths()[0].vertices
        assert not np.array_equal(road_marker, oracle_marker)
    finally:
        plt.close(fig)


def test_all_city_scatter_and_shared_legend_use_macro_region_colors() -> None:
    wide = _representative_wide()
    wide["road_annual_reduction_pct"] = [34.3, 18.6, 11.9, 7.6]
    wide["road_oracle_gap_pp"] = [0.1, 1.3, 2.1, 2.6]
    fig, ax = plt.subplots()
    try:
        _plot_distance_threshold_all_city_scatter(
            ax,
            wide,
            "road_optimal_threshold_km",
            "road_annual_reduction_pct",
            "Threshold",
            "Reduction",
            panel_idx=0,
            diagonal=False,
        )
        actual = ax.collections[1].get_facecolors()
        expected = [
            to_rgba(color, alpha=0.9)
            for color in wide["macro_region_color"]
        ]
        np.testing.assert_allclose(actual, expected)

        legend = _add_distance_threshold_macro_region_legend(fig, wide)
        assert [text.get_text() for text in legend.get_texts()] == [
            "North China (1)",
            "East China (1)",
            "South China (1)",
            "Southwest (1)",
        ]
    finally:
        plt.close(fig)


def test_distance_threshold_scatter_axis_labels_are_explicit() -> None:
    assert DISTANCE_THRESHOLD_LANDSCAPE_AXIS_LABELS == (
        "Road delivery distance threshold (km)",
        "Maximum lifecycle-emissions reduction (%)",
    )
    assert DISTANCE_THRESHOLD_ORACLE_AXIS_LABELS == (
        "Distance-based lifecycle-emissions\nreduction (%)",
        "Additional reduction under trip-level allocation\n(percentage points)",
    )
