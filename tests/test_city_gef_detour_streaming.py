"""Equivalence checks for bounded GEF-detour binning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drone_emission.visualization.city_gef_support import (
    _bin_reduction_and_detour,
    _bin_reduction_and_detour_chunks,
)
from drone_emission.visualization.city_gef_detour import (
    _filter_tail_rows_by_merchant,
)


def _frame():
    return pd.DataFrame(
        {
            "road_distance_m": [100.0, 300.0, -1.0, 800.0, 1200.0, 500.0],
            "distance_km": [0.1, 0.2, 0.3, 0.4, 0.61, 0.0],
            "drone_lifecycle_g": [4.0, 8.0, 2.0, 10.0, 25.0, 3.0],
            "ebike_lifecycle_g": [6.0, 7.0, 3.0, 14.0, 20.0, 4.0],
            "annual_sales": [12.0, 24.0, 36.0, 48.0, 60.0, 72.0],
        }
    )


def _assert_equivalent(expected, actual):
    for expected_values, actual_values in zip(expected[:3], actual[:3]):
        assert np.allclose(expected_values, actual_values, equal_nan=True)
    assert expected[4] == actual[4]
    assert np.allclose(expected[5], actual[5], equal_nan=True)
    assert len(expected[3]) == len(actual[3])
    for expected_values, actual_values in zip(expected[3], actual[3]):
        assert np.allclose(expected_values, actual_values, equal_nan=True)


def test_chunked_binning_matches_full_frame_for_both_metrics():
    frame = _frame()
    chunks = [frame.iloc[:2], frame.iloc[2:4], frame.iloc[4:]]
    max_air = float(frame.loc[frame["distance_km"] > 0, "distance_km"].max())

    for metric in ("total_t", "per_order_g"):
        expected = _bin_reduction_and_detour(frame, 0.2, metric)
        actual = _bin_reduction_and_detour_chunks(
            iter(chunks), 0.2, max_air, metric
        )
        _assert_equivalent(expected, actual)


def test_chunked_binning_rejects_invalid_maximum():
    with np.testing.assert_raises_regex(ValueError, "finite and positive"):
        _bin_reduction_and_detour_chunks([], 0.2, np.nan)


def test_shanghai_merchant_filter_only_changes_excluded_tail_rows():
    frame = pd.DataFrame(
        {
            "from_lat": [31.0, 31.0, 31.1, 31.2],
            "from_lng": [121.0, 121.0, 121.1, 121.2],
            "road_distance_m": [3_100.0, 6_400.0, 3_200.0, 3_300.0],
            "distance_km": [3.1, 3.2, 3.2, 3.3],
            "drone_lifecycle_g": [5.0, 5.0, 20.0, 8.0],
            "ebike_lifecycle_g": [10.0, 10.0, 10.0, 16.0],
            "annual_sales": [10.0, 20.0, 30.0, 40.0],
        }
    )
    excluded = {(31.0, 121.0)}

    filtered, mask, filtered_sales = _filter_tail_rows_by_merchant(
        frame, excluded
    )

    assert mask.tolist() == [False, True, False, False]
    assert filtered.index.tolist() == [0, 2, 3]
    assert filtered_sales == 20.0

    actual = (
        _bin_reduction_and_detour_chunks(
            [filtered], 0.2, frame["distance_km"].max(), "per_order_g"
        )
    )
    expected = _bin_reduction_and_detour(filtered, 0.2, "per_order_g")
    _assert_equivalent(expected, actual)

    centres, _, reduction_abs, _, _, drone_share = actual
    bin_33 = np.flatnonzero(np.isclose(centres, 3.3))[0]
    assert np.isclose(reduction_abs[bin_33], 20.0 / 70.0)
    assert np.isclose(drone_share[bin_33], 40.0 / 70.0 * 100.0)

    visible = np.isfinite(reduction_abs) & (centres >= 0.2) & (centres <= 5.0)
    actual_fit = np.polyfit(centres[visible], reduction_abs[visible], 1)
    expected_fit = np.polyfit(
        expected[0][visible], expected[2][visible], 1
    )
    assert np.allclose(actual_fit, expected_fit)

    other_city, other_mask, other_sales = _filter_tail_rows_by_merchant(
        frame, set()
    )
    pd.testing.assert_frame_equal(other_city, frame)
    assert not other_mask.any()
    assert other_sales == 0.0

    full_city_detour = (
        (frame["road_distance_m"] * frame["annual_sales"]).sum()
        / (
            frame["distance_km"]
            * 1_000.0
            * frame["annual_sales"]
        ).sum()
    )
    filtered_city_detour = (
        (filtered["road_distance_m"] * filtered["annual_sales"]).sum()
        / (
            filtered["distance_km"]
            * 1_000.0
            * filtered["annual_sales"]
        ).sum()
    )
    assert full_city_detour != filtered_city_detour
    assert np.isclose(full_city_detour, 387.0 / 323.0)
