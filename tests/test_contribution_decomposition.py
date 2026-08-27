"""Focused identities for the formal contribution decomposition."""

from __future__ import annotations

import numpy as np

from drone_emission.emission.cal_single_delivery import DroneParams, EbikeParams
from drone_emission.sensitivity.vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_parts_g,
    compute_ebike_per_order_parts_g,
)


def test_drone_use_phase_splits_into_fixed_and_cruise_parts() -> None:
    distances = np.array([0.5, 2.0, 5.0])
    parts = compute_drone_per_order_parts_g(
        distances,
        DroneParams(),
        DroneScenario(),
    )

    assert np.allclose(parts["fixed_use_g"], parts["fixed_use_g"][0])
    assert np.allclose(
        parts["cruise_use_g"] / distances,
        parts["cruise_use_g"][0] / distances[0],
    )
    assert np.allclose(
        parts["fixed_use_g"] + parts["cruise_use_g"],
        parts["use_g"],
    )


def test_disabling_drone_use_phase_zeros_both_parts() -> None:
    parts = compute_drone_per_order_parts_g(
        np.array([0.5, 2.0]),
        DroneParams(),
        DroneScenario(include_use=False),
    )

    assert np.allclose(parts["fixed_use_g"], 0.0)
    assert np.allclose(parts["cruise_use_g"], 0.0)
    assert np.allclose(parts["use_g"], 0.0)


def test_four_contributions_sum_to_od_lifecycle_saving() -> None:
    road_distances = np.array([1.0, 2.5, 6.0])
    air_distances = np.array([0.8, 1.5, 3.5])
    ebike = compute_ebike_per_order_parts_g(
        road_distances,
        EbikeParams(),
        EbikeScenario(kappa=0.7692307692307692),
    )
    drone = compute_drone_per_order_parts_g(
        air_distances,
        DroneParams(),
        DroneScenario(),
    )
    embodied_balance = (
        ebike["production_g"]
        + ebike["end_of_life_g"]
        - drone["production_g"]
        - drone["end_of_life_g"]
    )
    ledger = (
        embodied_balance
        + ebike["use_g"]
        - drone["fixed_use_g"]
        - drone["cruise_use_g"]
    )
    expected = ebike["total_g"] - drone["total_g"]

    assert np.allclose(ledger, expected)
