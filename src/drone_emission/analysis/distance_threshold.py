"""Optimize distance-threshold drone adoption policies.

For a threshold ``d``, routes shorter than ``d`` use an e-bike and routes at
or above ``d`` use a drone.  The optimizer evaluates every distinct observed
distance boundary using a sort followed by suffix sums, so it is exact for the
available OD sample without constructing a route-by-threshold matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANALYSIS_VERSION = "1.0.0"
TIE_RTOL = 1e-12
TIE_ATOL_KG = 1e-6


@dataclass(frozen=True)
class ThresholdInputs:
    """Validated positive-volume inputs for one city."""

    road_distance_km: np.ndarray
    straight_distance_km: np.ndarray
    annual_sales: np.ndarray
    ebike_lifecycle_g: np.ndarray
    drone_lifecycle_g: np.ndarray
    n_od_pairs_total: int
    n_od_pairs_unreachable: int
    unreachable_annual_orders: float


def validate_inputs(
    road_distance_km,
    straight_distance_km,
    annual_sales,
    ebike_lifecycle_g,
    drone_lifecycle_g,
) -> ThresholdInputs:
    """Validate arrays and remove zero-volume OD pairs.

    Zero annual sales do not affect the objective and are excluded before
    sorting.  Other invalid values fail explicitly because silently dropping
    them could change the optimal policy.
    """

    arrays = {
        "road_distance_km": np.asarray(road_distance_km, dtype=np.float64),
        "straight_distance_km": np.asarray(
            straight_distance_km, dtype=np.float64
        ),
        "annual_sales": np.asarray(annual_sales, dtype=np.float64),
        "ebike_lifecycle_g": np.asarray(ebike_lifecycle_g, dtype=np.float64),
        "drone_lifecycle_g": np.asarray(drone_lifecycle_g, dtype=np.float64),
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("threshold input arrays must have the same length")
    n_total = lengths.pop()
    if n_total == 0:
        raise ValueError("OD data is empty")

    for name, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")

    if (arrays["annual_sales"] < 0).any():
        raise ValueError("annual_sales contains negative values")

    unreachable = arrays["road_distance_km"] < 0
    if unreachable.any() and not np.allclose(
        arrays["road_distance_km"][unreachable], -0.001
    ):
        raise ValueError(
            "road_distance_km contains a negative value other than the "
            "documented -1 m unreachable sentinel"
        )
    reachable = ~unreachable
    for name in (
        "road_distance_km",
        "straight_distance_km",
        "ebike_lifecycle_g",
        "drone_lifecycle_g",
    ):
        if (arrays[name][reachable] < 0).any():
            raise ValueError(f"{name} contains negative values")

    positive = (arrays["annual_sales"] > 0) & reachable
    if not positive.any():
        raise ValueError("OD data has no positive annual_sales")

    if positive.all():
        selected = arrays
    else:
        selected = {name: values[positive] for name, values in arrays.items()}

    return ThresholdInputs(
        road_distance_km=selected["road_distance_km"],
        straight_distance_km=selected["straight_distance_km"],
        annual_sales=selected["annual_sales"],
        ebike_lifecycle_g=selected["ebike_lifecycle_g"],
        drone_lifecycle_g=selected["drone_lifecycle_g"],
        n_od_pairs_total=n_total,
        n_od_pairs_unreachable=int(unreachable.sum()),
        unreachable_annual_orders=float(arrays["annual_sales"][unreachable].sum()),
    )


def _sorted_policy_arrays(
    distance_km: np.ndarray,
    annual_sales: np.ndarray,
    ebike_lifecycle_g: np.ndarray,
    drone_lifecycle_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return sorted distances and suffix changes/orders for one basis."""

    order = np.argsort(distance_km, kind="quicksort")
    sorted_distance = distance_km[order]
    annual_change_g = (
        (drone_lifecycle_g - ebike_lifecycle_g) * annual_sales
    )[order]
    sorted_sales = annual_sales[order]

    n = len(order)
    suffix_change_g = np.empty(n + 1, dtype=np.float64)
    suffix_change_g[n] = 0.0
    suffix_change_g[:n] = np.cumsum(annual_change_g[::-1], dtype=np.float64)[::-1]

    suffix_orders = np.empty(n + 1, dtype=np.float64)
    suffix_orders[n] = 0.0
    suffix_orders[:n] = np.cumsum(sorted_sales[::-1], dtype=np.float64)[::-1]

    baseline_g = float(
        np.sum(ebike_lifecycle_g * annual_sales, dtype=np.float64)
    )
    total_orders = float(np.sum(annual_sales, dtype=np.float64))
    if baseline_g <= 0:
        raise ValueError("all-e-bike annual lifecycle emissions must be positive")
    return (
        sorted_distance,
        suffix_change_g,
        suffix_orders,
        baseline_g,
        total_orders,
    )


def _policy_case(drone_orders: float, total_orders: float) -> str:
    if np.isclose(drone_orders, total_orders, rtol=TIE_RTOL, atol=1e-9):
        return "all_drone"
    if np.isclose(drone_orders, 0.0, rtol=0.0, atol=1e-9):
        return "all_ebike"
    return "mixed"


def optimize_threshold(
    distance_km: np.ndarray,
    annual_sales: np.ndarray,
    ebike_lifecycle_g: np.ndarray,
    drone_lifecycle_g: np.ndarray,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """Find the exact observed-distance optimum for one distance basis.

    The returned working arrays can be reused by :func:`sample_curve` to avoid
    a second sort.  Threshold equality is assigned to the drone.  Numerical
    ties are resolved toward the smallest threshold (more drone orders).
    """

    working = _sorted_policy_arrays(
        distance_km,
        annual_sales,
        ebike_lifecycle_g,
        drone_lifecycle_g,
    )
    sorted_distance, suffix_change_g, suffix_orders, baseline_g, total_orders = (
        working
    )
    n = len(sorted_distance)

    first_of_value = np.empty(n, dtype=bool)
    first_of_value[0] = True
    first_of_value[1:] = sorted_distance[1:] != sorted_distance[:-1]
    observed_indices = np.flatnonzero(first_of_value)

    thresholds = sorted_distance[observed_indices]
    indices = observed_indices
    if thresholds[0] > 0:
        thresholds = np.concatenate(([0.0], thresholds))
        indices = np.concatenate(([0], indices))

    all_ebike_threshold = np.nextafter(sorted_distance[-1], np.inf)
    thresholds = np.concatenate((thresholds, [all_ebike_threshold]))
    indices = np.concatenate((indices, [n]))

    mixed_total_kg = (baseline_g + suffix_change_g[indices]) / 1000.0
    minimum_kg = float(np.min(mixed_total_kg))
    tied = np.flatnonzero(
        np.isclose(
            mixed_total_kg,
            minimum_kg,
            rtol=TIE_RTOL,
            atol=TIE_ATOL_KG,
        )
    )
    best_position = int(tied[np.argmin(thresholds[tied])])
    best_index = int(indices[best_position])
    best_threshold = float(thresholds[best_position])
    best_total_kg = float(mixed_total_kg[best_position])
    drone_orders = float(suffix_orders[best_index])
    baseline_kg = baseline_g / 1000.0
    reduction_kg = baseline_kg - best_total_kg

    result = {
        "policy_case": _policy_case(drone_orders, total_orders),
        "optimal_threshold_km": best_threshold,
        "max_observed_distance_km": float(sorted_distance[-1]),
        "all_ebike_lifecycle_kg": baseline_kg,
        "optimal_lifecycle_kg": best_total_kg,
        "annual_reduction_kg": reduction_kg,
        "annual_reduction_pct": reduction_kg / baseline_kg * 100.0,
        "drone_annual_orders": drone_orders,
        "drone_order_ratio_pct": drone_orders / total_orders * 100.0,
        "n_od_pairs_optimized": n,
        "total_annual_orders": total_orders,
    }
    return result, working


def sample_curve(
    optimum: dict,
    working: tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    n_uniform: int = 201,
) -> list[dict]:
    """Return a compact, uniformly spaced threshold-response curve."""

    if n_uniform < 2:
        raise ValueError("n_uniform must be at least 2")
    sorted_distance, suffix_change_g, suffix_orders, baseline_g, total_orders = (
        working
    )
    maximum = float(sorted_distance[-1])
    all_ebike_threshold = float(np.nextafter(maximum, np.inf))
    thresholds = np.linspace(0.0, maximum, n_uniform, dtype=np.float64)
    thresholds = np.concatenate(
        (thresholds, [all_ebike_threshold, optimum["optimal_threshold_km"]])
    )
    thresholds = np.unique(thresholds)

    indices = np.searchsorted(sorted_distance, thresholds, side="left")
    mixed_kg = (baseline_g + suffix_change_g[indices]) / 1000.0
    drone_orders = suffix_orders[indices]
    baseline_kg = baseline_g / 1000.0

    rows = []
    for threshold, total_kg, orders in zip(
        thresholds, mixed_kg, drone_orders, strict=True
    ):
        reduction_kg = baseline_kg - float(total_kg)
        rows.append(
            {
                "threshold_km": float(threshold),
                "policy_case": _policy_case(float(orders), total_orders),
                "mixed_lifecycle_kg": float(total_kg),
                "annual_reduction_kg": reduction_kg,
                "annual_reduction_pct": reduction_kg / baseline_kg * 100.0,
                "drone_annual_orders": float(orders),
                "drone_order_ratio_pct": float(orders) / total_orders * 100.0,
                "is_exact_optimum": bool(
                    threshold == optimum["optimal_threshold_km"]
                ),
            }
        )
    return rows


def analyze_city_inputs(
    inputs: ThresholdInputs,
    include_curve: bool = False,
    n_uniform: int = 201,
) -> tuple[list[dict], list[dict]]:
    """Analyze road- and straight-distance policies for one city."""

    summaries: list[dict] = []
    curves: list[dict] = []
    for basis, distance in (
        ("road", inputs.road_distance_km),
        ("straight", inputs.straight_distance_km),
    ):
        optimum, working = optimize_threshold(
            distance,
            inputs.annual_sales,
            inputs.ebike_lifecycle_g,
            inputs.drone_lifecycle_g,
        )
        optimum["distance_basis"] = basis
        optimum["n_od_pairs_total"] = inputs.n_od_pairs_total
        optimum["n_od_pairs_unreachable"] = inputs.n_od_pairs_unreachable
        optimum["unreachable_annual_orders"] = inputs.unreachable_annual_orders
        summaries.append(optimum)
        if include_curve:
            basis_curve = sample_curve(optimum, working, n_uniform=n_uniform)
            for row in basis_curve:
                row["distance_basis"] = basis
            curves.extend(basis_curve)
    return summaries, curves
