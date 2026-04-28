"""Shapley decomposition of the drone-vs-ebike emission reduction.

Four binary mechanisms toggle the model between baseline (all ON) and a
counterfactual residual (all OFF):

* **M1 routing geometry** — ON: e-bike uses road distance; OFF: e-bike uses
  straight-line distance (matching drone routing).
* **M2 bundling** — ON: kappa = 0.7 (Eleme-derived realistic bundling);
  OFF: kappa = 1.0 (no bundling).
* **M3 production amortization** — ON: include production+disposal; OFF:
  zero out for both modes.
* **M4 grid carbon intensity** — ON: actual provincial GEF; OFF: clean-grid
  baseline (province GEF × 0.10).

Coalition value v(S) = optimal-threshold mixed-strategy reduction (vs ebike
under the same state) when only mechanisms in S are ON. Shapley values sum
to v({M1,M2,M3,M4}) − v(∅).
"""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations, product
from math import factorial

import numpy as np
import pandas as pd

from emission.cal_single_delivery import DroneParams, EbikeParams

from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)

MECHANISMS = ("M1_routing", "M2_bundling", "M3_production", "M4_gef")
KAPPA_BUNDLING_ON = 0.7
KAPPA_BUNDLING_OFF = 1.0
GEF_CLEAN_MULT = 0.10  # OFF state for M4


def _coalition_reduction_pct(
    state: tuple[bool, bool, bool, bool],
    road_distances_km: np.ndarray,
    straight_distances_km: np.ndarray,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    province_gef_kwh: float,
    rider_weight: float,
    payload_kg: float,
) -> float:
    """v(S) — optimal reduction (%) under the given mechanism state."""
    m1_routing, m2_bundling, m3_production, m4_gef = state

    # Distances seen by ebike depend on M1
    ebike_d = road_distances_km if m1_routing else straight_distances_km

    # Effective GEF multiplier from M4
    gef_mult = 1.0 if m4_gef else GEF_CLEAN_MULT

    # Bundling kappa from M2
    kappa = KAPPA_BUNDLING_ON if m2_bundling else KAPPA_BUNDLING_OFF

    e_scen = EbikeScenario(
        kappa=kappa,
        gef_mult=gef_mult,
        payload_kg=payload_kg,
        rider_weight=rider_weight,
        include_production=m3_production,
    )
    d_scen = DroneScenario(
        gef_mult=gef_mult,
        payload_kg=payload_kg,
        include_production=m3_production,
    )

    # Override e_use to the province baseline; gef_mult scales it.
    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)

    ebike_g = compute_ebike_per_order_g(ebike_d, ebike_pp, e_scen)
    drone_g = compute_drone_per_order_g(straight_distances_km, drone_pp, d_scen)

    # Threshold sweep on road distance (operationally observed)
    middle = np.linspace(
        max(0.3, np.percentile(road_distances_km, 5)),
        np.percentile(road_distances_km, 95),
        10,
    )
    thresholds = np.concatenate([[-0.01], middle, [np.inf]])
    n = len(road_distances_km)
    ebike_total = float(ebike_g.sum())
    if ebike_total <= 0:
        return 0.0
    best = float("-inf")
    for t in thresholds:
        use_drone = road_distances_km >= t
        mixed = float(np.where(use_drone, drone_g, ebike_g).sum())
        red = (ebike_total - mixed) / ebike_total * 100.0
        if red > best:
            best = red
    return best


def _shapley_values(coalition_values: dict) -> dict:
    """Compute Shapley value for each of the 4 mechanisms.

    coalition_values: dict mapping state-tuple (4-bool) -> v(state)
    Returns dict of mechanism index -> Shapley value.
    """
    n = 4
    shapley = {k: 0.0 for k in range(n)}
    perms = list(permutations(range(n)))
    for perm in perms:
        state = [False] * n
        before = coalition_values[tuple(state)]
        for k in perm:
            state[k] = True
            after = coalition_values[tuple(state)]
            shapley[k] += (after - before)
            before = after
    for k in range(n):
        shapley[k] /= len(perms)
    return shapley


def run_shapley_for_region(
    od_df: pd.DataFrame,
    region_key: str,
    province: str,
    province_gef_kwh: float,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    rider_weight: float = 70.0,
    payload_kg: float = 1.0,
) -> pd.DataFrame:
    """Compute the 4-mechanism Shapley decomposition for a single region.

    Output rows (per region):
      - one row per mechanism with column ``shapley_pct_pts``
      - one row labeled ``residual`` with v(∅)
      - one row labeled ``total_baseline`` with v(full)
    """
    road_distances_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
    straight_distances_km = od_df["distance_km"].to_numpy(dtype=float)
    if len(road_distances_km) == 0:
        return pd.DataFrame()

    # Enumerate all 2^4 = 16 mechanism states
    states = list(product([False, True], repeat=4))
    coalition_values = {}
    for state in states:
        v = _coalition_reduction_pct(
            state,
            road_distances_km, straight_distances_km,
            ebike_params, drone_params,
            province_gef_kwh, rider_weight, payload_kg,
        )
        coalition_values[state] = v

    shapley = _shapley_values(coalition_values)

    residual = coalition_values[(False, False, False, False)]
    total = coalition_values[(True, True, True, True)]

    rows = []
    for k, name in enumerate(MECHANISMS):
        rows.append({
            "region_key": region_key,
            "province": province,
            "mechanism": name,
            "shapley_pct_pts": shapley[k],
        })
    rows.append({
        "region_key": region_key,
        "province": province,
        "mechanism": "residual_all_off",
        "shapley_pct_pts": residual,
    })
    rows.append({
        "region_key": region_key,
        "province": province,
        "mechanism": "total_baseline",
        "shapley_pct_pts": total,
    })

    # Sanity: sum of M1..M4 + residual should equal total to within float epsilon
    sum_decomposed = sum(shapley.values()) + residual
    rows.append({
        "region_key": region_key,
        "province": province,
        "mechanism": "_sum_check_residual",
        "shapley_pct_pts": float(total - sum_decomposed),  # should be ~0
    })
    return pd.DataFrame(rows)
