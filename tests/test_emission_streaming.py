"""Bounded emission CSV equivalence and atomicity checks."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from drone_emission.emission.streaming import OUTPUT_COLUMNS, process_emission_csv


class Region:
    def __init__(self):
        self.stats = {}

    def set_stat(self, key, value):
        self.stats[key] = value


class Lca:
    def __init__(self, fail_distance=None):
        self.fail_distance = fail_distance

    def calculate_drone_use_phase_emissions(self, distance, payload):
        if distance == self.fail_distance:
            raise RuntimeError("injected chunk failure")
        return {"emissions_kg_co2e": distance * payload}

    def calculate_ebike_use_phase_emissions(self, distance, payload, rider):
        return {"emissions_kg_co2e": distance * (payload + rider)}

    def calculate_lifecycle_emissions_per_order_simple(
        self, distance, n_orders_per_delivery, vehicle_type
    ):
        multiplier = 2 if vehicle_type == "drone" else 3
        total = distance * multiplier
        return {
            "total_per_order_g": total,
            "production_per_order_g": total * 0.2,
            "use_per_order_g": total * 0.7,
            "end_of_life_per_order_g": total * 0.1,
        }


def _input(path):
    pd.DataFrame(
        {
            "from_lat": [1.0, 2.0, 3.0],
            "from_lng": [1.0, 2.0, 3.0],
            "to_lat": [1.1, 2.1, 3.1],
            "to_lng": [1.1, 2.1, 3.1],
            "distance_km": [1.0, 2.0, 3.0],
            "annual_sales": [12, 24, 36],
            "road_distance_m": [1500.0, 2500.0, 3500.0],
        }
    ).to_csv(path, index=False)


def _run(path, region, lca):
    return process_emission_csv(
        path,
        region_info=region,
        lca=lca,
        payload_kg=2.0,
        rider_weight=1.0,
        ebike_params=SimpleNamespace(velocity=20.0),
        drone_params=SimpleNamespace(
            h_takeoff=100.0, v_vertical=10.0, v_level=60.0, hover_time=30.0
        ),
        ebike_kappa=1.2,
        grid_factor=0.5,
        chunk_rows=2,
    )


def test_streaming_emissions_preserve_order_and_weighted_totals(tmp_path):
    path = tmp_path / "od.csv"
    _input(path)
    region = Region()
    result = _run(path, region, Lca())
    output = pd.read_csv(path)

    assert output.columns.tolist()[-len(OUTPUT_COLUMNS):] == list(OUTPUT_COLUMNS)
    assert output["distance_km"].tolist() == [1.0, 2.0, 3.0]
    expected_drone_g = sum(
        distance * 2 * sales
        for distance, sales in zip([1.0, 2.0, 3.0], [12, 24, 36])
    )
    assert region.stats["total_drone_lifecycle"] == expected_drone_g / 1000
    assert result.iloc[0]["total_drone_lifecycle_kg"] == expected_drone_g / 1000


def test_midstream_emission_failure_preserves_old_csv(tmp_path):
    path = tmp_path / "od.csv"
    _input(path)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        _run(path, Region(), Lca(fail_distance=3.0))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))
