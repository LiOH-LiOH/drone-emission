"""Narrow tests for demand-conserving OD allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from shapely.geometry import Point

from drone_emission.demand import api


class _Region:
    def __init__(self):
        self.stats = {}

    def to_dict(self):
        return {
            "region_name_en": "Test",
            "region_level": "city",
        }

    def set_stat(self, key, value):
        self.stats[key] = value


def _buildings(volumes=(100.0, 80.0, 60.0, 40.0)):
    return pd.DataFrame(
        {
            "centroid": [
                Point(114.001, 22.500),
                Point(114.002, 22.500),
                Point(114.003, 22.500),
                Point(114.004, 22.500),
            ],
            "volume_m3": volumes,
        }
    )


def test_largest_remainder_conserves_orders_and_breaks_ties_stably():
    allocation = api._allocate_integer_orders(
        np.array([0.25, 0.25, 0.25, 0.25]),
        2,
        stable_keys=np.array([30, 10, 40, 20]),
    )

    assert allocation.tolist() == [0, 1, 0, 1]
    assert int(allocation.sum()) == 2


def test_store_generators_conserve_monthly_orders_and_agree():
    buildings = _buildings()
    params = {
        "impedance_type": "tanner",
        "alpha": 0.4219672366,
        "tanner_a": 0.3255233472,
        "tanner_b": 1.650088692,
    }

    scalar = api.generate_one_store_od_pairs(
        buildings, 3, (22.5, 114.0), params
    )
    vectorized = api.generate_one_store_od_pairs_vectorized(
        buildings, 3, (22.5, 114.0), params
    )

    def allocated_sales(rows):
        return {
            (row["to_lat"], row["to_lng"]): row["annual_sales"] for row in rows
        }

    assert allocated_sales(scalar) == allocated_sales(vectorized)
    assert np.allclose(
        sorted(row["distance_km"] for row in scalar),
        sorted(row["distance_km"] for row in vectorized),
    )
    assert sum(row["annual_sales"] for row in vectorized) == 36
    assert len(vectorized) <= 3
    assert all(row["annual_sales"] > 0 for row in vectorized)
    assert all(row["annual_sales"] % 12 == 0 for row in vectorized)


def test_invalid_candidates_are_not_allocated():
    buildings = _buildings(volumes=(np.nan, 0.0, -1.0, np.inf))

    result = api.generate_one_store_od_pairs_vectorized(
        buildings,
        10,
        (22.5, 114.0),
        {"impedance_type": "tanner"},
    )

    assert result == []


def test_region_statistics_separate_unallocatable_and_invalid_sales(
    tmp_path, monkeypatch
):
    buildings = _buildings()
    stores = pd.DataFrame(
        {
            "month_sales": [2.0, 1.5, 1.0, 3.0],
            "lat": [22.5] * 4,
            "lng": [114.0] * 4,
        }
    )
    indices = [
        np.array([0, 1, 2, 3]),
        np.array([0, 1]),
        np.array([], dtype=int),
        np.array([0, 1, 2, 3]),
    ]
    store_coords = np.array([[22.5, 114.0]] * 4)
    monkeypatch.setattr(
        api,
        "prepare_store_building_matching",
        lambda region_info, exp_config: (
            indices,
            stores,
            buildings,
            store_coords,
        ),
    )
    output_dir = tmp_path / "city" / "test"
    output_dir.mkdir(parents=True)
    region = _Region()

    api.generate_od_pairs(
        region,
        {
            "data_dir": str(tmp_path),
            "od_params": {"impedance_type": "tanner"},
        },
    )

    output = pd.read_csv(output_dir / "test_od_pairs.csv")
    assert int(output["annual_sales"].sum()) == 60
    assert region.stats["total_store_annual_sales"] == 72
    assert region.stats["allocatable_store_annual_sales"] == 60
    assert region.stats["unallocatable_store_annual_sales"] == 12
    assert region.stats["invalid_sales_store_count"] == 1
    assert region.stats["allocatable_store_count"] == 2
    assert region.stats["unallocatable_store_count"] == 1
    assert region.stats["od_allocation_annual_sales_error"] == 0
