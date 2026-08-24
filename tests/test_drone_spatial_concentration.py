"""Targeted tests for drone-favorable order spatial concentration."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))

from drone_emission.analysis.drone_spatial_concentration import (  # noqa: E402
    _assign_classes_and_representatives,
    _eligible_mask,
    compute_city_pilot_capture,
)


def _od_frame(rows) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "from_lat",
            "from_lng",
            "to_lat",
            "to_lng",
            "annual_sales",
            "drone_lifecycle_g",
            "ebike_lifecycle_g",
        ],
    )


def test_colocated_orders_are_fully_captured() -> None:
    od_df = _od_frame(
        [
            (22.50, 113.90, 22.50, 113.90, 100, 40, 50),
            (22.501, 113.901, 22.502, 113.902, 50, 30, 50),
        ]
    )
    result = compute_city_pilot_capture(od_df)
    assert result["eligible_annual_orders"] == 150
    assert result["pilot_concentration_score_pct"] > 99.99


def test_sales_weights_not_od_rows_define_capture() -> None:
    od_df = _od_frame(
        [
            (22.50, 113.90, 22.50, 113.90, 900, 40, 50),
            (23.50, 114.90, 23.50, 114.90, 100, 40, 50),
        ]
    )
    result = compute_city_pilot_capture(od_df, radii_km=(2.0,))
    assert 89.9 <= result["capture_2km_pct"] <= 90.1
    assert 89.9 <= result["pilot_concentration_score_pct"] <= 90.1


def test_both_endpoints_must_fit_inside_circle() -> None:
    od_df = _od_frame(
        [(22.50, 113.90, 22.50, 114.00, 100, 40, 50)]
    )
    result = compute_city_pilot_capture(od_df, radii_km=(2.0,))
    assert result["capture_2km_pct"] == 0.0


def test_emission_margin_filters_borderline_orders() -> None:
    od_df = _od_frame(
        [
            (22.50, 113.90, 22.50, 113.90, 100, 49, 50),
            (22.51, 113.91, 22.51, 113.91, 200, 40, 50),
            (22.52, 113.92, 22.52, 113.92, 300, 51, 50),
        ]
    )
    assert _eligible_mask(od_df, 0).tolist() == [True, True, False]
    assert _eligible_mask(od_df, 5).tolist() == [False, True, False]
    assert _eligible_mask(od_df, 10).tolist() == [False, True, False]


def test_coordinate_translation_preserves_score() -> None:
    base = _od_frame(
        [
            (22.50, 113.90, 22.505, 113.905, 100, 40, 50),
            (22.51, 113.91, 22.515, 113.915, 200, 40, 50),
        ]
    )
    shifted = base.copy()
    shifted[["from_lat", "to_lat"]] += 0.1
    shifted[["from_lng", "to_lng"]] += 0.1
    first = compute_city_pilot_capture(base)
    second = compute_city_pilot_capture(shifted)
    assert np.isclose(
        first["pilot_concentration_score_pct"],
        second["pilot_concentration_score_pct"],
        atol=0.1,
    )


def test_city_classes_and_representatives_are_deterministic() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "region_key": f"city_{index:02d}",
                "status": "ok",
                "pilot_concentration_score_pct": 20.0 + index * 5.0,
                "eligible_annual_orders": float(10 ** (3 + (index % 4))),
            }
        )
    rows.append(
        {
            "region_key": "missing",
            "status": "skipped",
            "pilot_concentration_score_pct": np.nan,
            "eligible_annual_orders": np.nan,
        }
    )
    first = _assign_classes_and_representatives(pd.DataFrame(rows))
    second = _assign_classes_and_representatives(pd.DataFrame(rows))
    assert first["concentration_class"].value_counts().to_dict() == {
        "Low": 4,
        "Medium": 4,
        "High": 4,
        "": 1,
    }
    assert first["is_representative"].sum() == 4
    pd.testing.assert_series_equal(
        first["is_representative"], second["is_representative"]
    )
