"""Equivalence checks for bounded GEF-detour binning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drone_emission.visualization.city_gef_support import (
    _bin_reduction_and_detour,
    _bin_reduction_and_detour_chunks,
)
from drone_emission.visualization.city_gef_detour import (
    _filter_rows_by_merchant,
    _finalize_merchant_filter_bin_audit,
    _merchant_snap_outliers,
    _new_merchant_filter_bin_audit,
    _unique_finite_coordinates,
    _update_merchant_filter_bin_audit,
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


def test_p99_uses_unique_finite_merchants_and_strict_upper_tail():
    raw_coordinates = np.array(
        [[31.0, 121.0], [31.0, 121.0], [np.nan, 121.2], [31.1, 121.1]]
    )
    unique = _unique_finite_coordinates(raw_coordinates)
    assert unique.tolist() == [[31.0, 121.0], [31.1, 121.1]]

    coordinates = np.column_stack(
        [np.arange(100, dtype=float), np.arange(100, dtype=float) + 100.0]
    )
    excluded, threshold_m = _merchant_snap_outliers(
        coordinates, np.arange(100, dtype=float)
    )
    assert np.isclose(threshold_m, 98.01)
    assert excluded == {(99.0, 199.0)}

    boundary_coordinates = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    boundary_excluded, boundary_m = _merchant_snap_outliers(
        boundary_coordinates, np.array([0.0, 1.0, 2.0]), quantile=0.5
    )
    assert boundary_m == 1.0
    assert boundary_excluded == {(2.0, 2.0)}


def test_shanghai_p99_filter_removes_excluded_merchants_at_all_distances():
    frame = pd.DataFrame(
        {
            "from_lat": [31.0, 31.0, 31.1, 31.2],
            "from_lng": [121.0, 121.0, 121.1, 121.2],
            "road_distance_m": [1_000.0, 6_400.0, 3_200.0, 3_300.0],
            "distance_km": [1.0, 4.0, 3.2, 3.3],
            "drone_lifecycle_g": [5.0, 5.0, 20.0, 8.0],
            "ebike_lifecycle_g": [10.0, 10.0, 10.0, 16.0],
            "annual_sales": [10.0, 20.0, 30.0, 40.0],
        }
    )
    excluded = {(31.0, 121.0)}

    filtered, mask, filtered_sales = _filter_rows_by_merchant(
        frame, excluded
    )

    assert mask.tolist() == [True, True, False, False]
    assert filtered.index.tolist() == [2, 3]
    assert filtered_sales == 30.0

    actual = (
        _bin_reduction_and_detour_chunks(
            [filtered], 0.2, frame["distance_km"].max(), "per_order_g"
        )
    )
    expected = _bin_reduction_and_detour(filtered, 0.2, "per_order_g")
    expected_bin_count = len(expected[0])
    for expected_values, actual_values in zip(expected[:3], actual[:3]):
        assert np.allclose(
            expected_values,
            actual_values[:expected_bin_count],
            equal_nan=True,
        )
    assert np.allclose(
        expected[5], actual[5][:expected_bin_count], equal_nan=True
    )

    centres, _, reduction_abs, _, _, drone_share = actual
    bin_33 = np.flatnonzero(np.isclose(centres, 3.3))[0]
    assert np.isclose(reduction_abs[bin_33], 20.0 / 70.0)
    assert np.isclose(drone_share[bin_33], 40.0 / 70.0 * 100.0)

    visible = np.isfinite(reduction_abs) & (centres >= 0.2) & (centres <= 5.0)
    actual_fit = np.polyfit(centres[visible], reduction_abs[visible], 1)
    expected_reduction_abs = np.zeros_like(centres)
    expected_reduction_abs[:expected_bin_count] = expected[2]
    expected_fit = np.polyfit(
        centres[np.isfinite(expected_reduction_abs) & (centres >= 0.2)],
        expected_reduction_abs[
            np.isfinite(expected_reduction_abs) & (centres >= 0.2)
        ],
        1,
    )
    assert np.allclose(actual_fit, expected_fit)

    other_city, other_mask, other_sales = _filter_rows_by_merchant(
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
    assert np.isclose(full_city_detour, 366.0 / 318.0)

    audit = _new_merchant_filter_bin_audit(
        frame["distance_km"].max(), 0.2
    )
    _update_merchant_filter_bin_audit(audit, "before", frame)
    _update_merchant_filter_bin_audit(audit, "after", filtered)
    records = _finalize_merchant_filter_bin_audit(audit)
    bin_33_audit = next(
        record
        for record in records
        if np.isclose(record["bin_center_air_km"], 3.3)
    )
    assert bin_33_audit["before"] == bin_33_audit["after"]
    assert np.isclose(
        bin_33_audit["after"]["optimal_drone_share_pct"],
        drone_share[bin_33],
    )
    assert np.isclose(
        bin_33_audit["after"][
            "sales_weighted_ebike_minus_drone_g_per_order"
        ],
        reduction_abs[bin_33],
    )
