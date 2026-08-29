"""Equivalence checks for bounded GEF-detour binning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drone_emission.visualization.city_gef_support import (
    _bin_reduction_and_detour,
    _bin_reduction_and_detour_chunks,
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
