"""Deterministic checks for the kappa VRP exact solver and plateau rule.

Run remotely with:
    .venv/bin/python tests/test_kappa_vrp.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from analysis.kappa_vrp import (  # noqa: E402
    RouteOrder,
    detect_plateau,
    solve_batch_exact,
)


def _orders(deadline=60.0):
    return [
        RouteOrder(0, 1, 0.0, 0.0, deadline),
        RouteOrder(1, 2, 0.0, 0.0, deadline),
    ]


def _line_matrix():
    return np.array(
        [
            [0.0, 1000.0, 2000.0],
            [1000.0, 0.0, 1000.0],
            [2000.0, 1000.0, 0.0],
        ]
    )


def _solve(capacity, orders=None, constrained=True):
    return solve_batch_exact(
        orders=_orders() if orders is None else orders,
        matrix=_line_matrix(),
        max_bundle=capacity,
        dispatch_min=0.0,
        speed_kmh=30.0,
        pickup_service_min=0.0,
        drop_service_min=0.0,
        constrained=constrained,
    )


def test_capacity_one_control():
    solution = _solve(1)
    assert solution.status == "optimal"
    assert math.isclose(solution.distance_m, 6000.0)
    assert len(solution.routes) == 2


def test_two_order_bundle_known_kappa():
    solution = _solve(2)
    assert solution.status == "optimal"
    assert math.isclose(solution.distance_m, 4000.0)
    assert math.isclose(solution.distance_m / 6000.0, 2.0 / 3.0)
    assert len(solution.routes) == 1


def test_capacity_curve_is_monotone():
    distances = [_solve(capacity).distance_m for capacity in (1, 2)]
    assert distances[1] <= distances[0]


def test_impossible_deadline_is_infeasible():
    orders = [
        RouteOrder(0, 1, 0.0, 0.0, 1.0),
        RouteOrder(1, 2, 0.0, 0.0, 1.0),
    ]
    solution = _solve(2, orders=orders)
    assert solution.status == "infeasible"


def test_plateau_detection():
    curve = pd.DataFrame(
        {
            "capacity": [1, 2, 3, 4, 5],
            "kappa": [1.0, 0.75, 0.70, 0.6965, 0.6940],
        }
    )
    plateau = detect_plateau(curve, threshold_pct=1.0, consecutive_steps=2)
    assert plateau["reached"]
    assert plateau["bundle_size_star"] == 3
    assert math.isclose(plateau["future_extreme_kappa"], 0.70)


def test_plateau_not_reached():
    curve = pd.DataFrame(
        {
            "capacity": [1, 2, 3, 4],
            "kappa": [1.0, 0.8, 0.7, 0.6],
        }
    )
    plateau = detect_plateau(curve, threshold_pct=1.0, consecutive_steps=2)
    assert not plateau["reached"]
    assert plateau["bundle_size_star"] is None


def main():
    checks = [
        test_capacity_one_control,
        test_two_order_bundle_known_kappa,
        test_capacity_curve_is_monotone,
        test_impossible_deadline_is_infeasible,
        test_plateau_detection,
        test_plateau_not_reached,
    ]
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
    print("All kappa VRP checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
