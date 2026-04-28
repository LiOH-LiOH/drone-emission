"""Numerical-equivalence test for sensitivity/vectorized_lca.py.

Verifies that vectorized ebike + drone lifecycle emissions match the scalar
reference (``LifeCycleEmissions.calculate_lifecycle_emissions_per_order_simple``)
to within 1e-6 absolute / 1e-9 relative across a range of parameter perturbations
and 100 random distances per parameter set.

Run with:
    cd src/drone_emission && ../../.venv/bin/python -m tests.test_vectorized_lca
or:
    .venv/bin/python tests/test_vectorized_lca.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

# Ensure src/drone_emission is importable when running this file directly.
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from emission.cal_single_delivery import (  # noqa: E402
    DroneParams,
    EbikeParams,
    LifeCycleEmissions,
)
from sensitivity.vectorized_lca import (  # noqa: E402
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
)

ABS_TOL = 1e-6
REL_TOL = 1e-9
RNG = np.random.default_rng(42)
N_DISTANCES = 100


def _scalar_ebike(distances_km, ebike_params, payload_kg, kappa, gef_mult, energy_mult,
                  lifespan, battery_lifespan, include_production, include_use,
                  rider_weight, n_orders_per_delivery):
    """Reference: scalar LifeCycleEmissions, kappa applied externally."""
    p = replace(
        ebike_params,
        e_use=ebike_params.e_use * gef_mult,
        lifespan=int(round(lifespan)) if float(lifespan).is_integer() else lifespan,
        battery_lifespan=int(round(battery_lifespan)) if float(battery_lifespan).is_integer() else battery_lifespan,
    )
    # Note: `lifespan` and `battery_lifespan` are ints in EbikeParams; replace preserves values.
    # When fractional values are tested, we monkeypatch via a non-frozen dataclass copy below.
    # Easier path: rebuild via direct field overrides.
    pp = EbikeParams(**{**ebike_params.__dict__, **{
        "e_use": ebike_params.e_use * gef_mult,
        "lifespan": lifespan,
        "battery_lifespan": battery_lifespan,
    }})
    lca = LifeCycleEmissions(ebike_params=pp)
    out = np.zeros_like(distances_km)
    for i, d in enumerate(distances_km):
        d_eff = d * kappa
        # Use phase
        use = lca.calculate_ebike_use_phase_emissions(d_eff, payload_kg, rider_weight)
        use_g = use["emissions_g_co2e"] * energy_mult
        # Production amortization (mirror calculate_lifecycle_emissions_per_order_simple)
        delivery_time_min = (2 * d_eff / pp.velocity) * 60 + pp.loading_time_min
        deliveries_per_day = pp.working_hours_per_day * 60 / delivery_time_min
        lifetime_deliveries = deliveries_per_day * pp.working_days_per_year * pp.lifespan
        prod = lca.calculate_ebike_production_emissions()
        body_lifetime_kg = (prod["body_production_kg"] + prod["body_disposal_kg"]) / pp.lifespan
        battery_lifetime_kg = (prod["battery_production_kg"] + prod["battery_disposal_kg"]) / pp.battery_lifespan
        annual_prod_kg = body_lifetime_kg + battery_lifetime_kg
        prod_per_delivery_kg = annual_prod_kg * pp.lifespan / lifetime_deliveries
        prod_g = prod_per_delivery_kg * 1000.0
        if not include_production:
            prod_g = 0.0
        if not include_use:
            use_g = 0.0
        out[i] = (prod_g + use_g) / n_orders_per_delivery
    return out


def _scalar_drone(distances_km, drone_params, payload_kg, gef_mult, energy_mult,
                  lifespan, battery_lifespan, include_production, include_use,
                  n_orders_per_delivery):
    pp = DroneParams(**{**drone_params.__dict__, **{
        "e_use": drone_params.e_use * gef_mult,
        "lifespan": lifespan,
        "battery_lifespan": battery_lifespan,
    }})
    lca = LifeCycleEmissions(drone_params=pp)
    out = np.zeros_like(distances_km)
    for i, d in enumerate(distances_km):
        use = lca.calculate_drone_use_phase_emissions(d, payload_kg)
        use_g = use["emissions_g_co2e"] * energy_mult
        delivery_time_min = (2 * d / pp.v_level) * 60 + pp.loading_time_min
        deliveries_per_day = pp.working_hours_per_day * 60 / delivery_time_min
        lifetime_deliveries = deliveries_per_day * pp.working_days_per_year * pp.lifespan
        prod = lca.calculate_drone_production_emissions()
        body_lifetime_kg = (prod["body_production_kg"] + prod["body_disposal_kg"]) / pp.lifespan
        battery_lifetime_kg = (prod["battery_production_kg"] + prod["battery_disposal_kg"]) / pp.battery_lifespan
        annual_prod_kg = body_lifetime_kg + battery_lifetime_kg
        prod_per_delivery_kg = annual_prod_kg * pp.lifespan / lifetime_deliveries
        prod_g = prod_per_delivery_kg * 1000.0
        if not include_production:
            prod_g = 0.0
        if not include_use:
            use_g = 0.0
        out[i] = (prod_g + use_g) / n_orders_per_delivery
    return out


def _check(label, expected, actual):
    diff = np.abs(expected - actual)
    rel = diff / (np.abs(expected) + 1e-30)
    max_abs = float(diff.max())
    max_rel = float(rel.max())
    ok = max_abs < ABS_TOL and max_rel < REL_TOL
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {label}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    if not ok:
        # Print the worst few to aid debugging.
        idx = np.argsort(diff)[-3:][::-1]
        for k in idx:
            print(f"        d_idx={k}: expected={expected[k]:.6f}, actual={actual[k]:.6f}, abs={diff[k]:.3e}")
    return ok


def test_ebike_baseline():
    distances = RNG.uniform(0.1, 8.0, N_DISTANCES)
    p = EbikeParams()
    s = EbikeScenario()
    actual = compute_ebike_per_order_g(distances, p, s)
    expected = _scalar_ebike(
        distances, p,
        payload_kg=s.payload_kg, kappa=s.kappa, gef_mult=s.gef_mult,
        energy_mult=s.energy_mult, lifespan=p.lifespan, battery_lifespan=p.battery_lifespan,
        include_production=s.include_production, include_use=s.include_use,
        rider_weight=s.rider_weight, n_orders_per_delivery=s.n_orders_per_delivery,
    )
    return _check("ebike baseline", expected, actual)


def test_ebike_perturbations():
    distances = RNG.uniform(0.1, 8.0, N_DISTANCES)
    p = EbikeParams()
    cases = [
        EbikeScenario(kappa=0.7),
        EbikeScenario(gef_mult=0.5, payload_kg=2.0),
        EbikeScenario(energy_mult=1.25, lifespan_yr=6),
        EbikeScenario(include_production=False),
        EbikeScenario(include_use=False, kappa=0.8),
    ]
    all_ok = True
    for i, s in enumerate(cases):
        actual = compute_ebike_per_order_g(distances, p, s)
        expected = _scalar_ebike(
            distances, p,
            payload_kg=s.payload_kg, kappa=s.kappa, gef_mult=s.gef_mult,
            energy_mult=s.energy_mult,
            lifespan=s.lifespan_yr if s.lifespan_yr is not None else p.lifespan,
            battery_lifespan=s.battery_lifespan_yr if s.battery_lifespan_yr is not None else p.battery_lifespan,
            include_production=s.include_production, include_use=s.include_use,
            rider_weight=s.rider_weight, n_orders_per_delivery=s.n_orders_per_delivery,
        )
        ok = _check(f"ebike perturbation {i+1}: {s}", expected, actual)
        all_ok = all_ok and ok
    return all_ok


def test_drone_baseline():
    distances = RNG.uniform(0.1, 8.0, N_DISTANCES)
    p = DroneParams()
    s = DroneScenario()
    actual = compute_drone_per_order_g(distances, p, s)
    expected = _scalar_drone(
        distances, p,
        payload_kg=s.payload_kg, gef_mult=s.gef_mult, energy_mult=s.energy_mult,
        lifespan=p.lifespan, battery_lifespan=p.battery_lifespan,
        include_production=s.include_production, include_use=s.include_use,
        n_orders_per_delivery=s.n_orders_per_delivery,
    )
    return _check("drone baseline", expected, actual)


def test_drone_perturbations():
    distances = RNG.uniform(0.1, 8.0, N_DISTANCES)
    p = DroneParams()
    cases = [
        DroneScenario(gef_mult=0.5),
        DroneScenario(payload_kg=2.0, energy_mult=1.5),
        DroneScenario(lifespan_yr=2),
        DroneScenario(include_production=False, gef_mult=1.2),
        DroneScenario(include_use=False),
    ]
    all_ok = True
    for i, s in enumerate(cases):
        actual = compute_drone_per_order_g(distances, p, s)
        expected = _scalar_drone(
            distances, p,
            payload_kg=s.payload_kg, gef_mult=s.gef_mult, energy_mult=s.energy_mult,
            lifespan=s.lifespan_yr if s.lifespan_yr is not None else p.lifespan,
            battery_lifespan=s.battery_lifespan_yr if s.battery_lifespan_yr is not None else p.battery_lifespan,
            include_production=s.include_production, include_use=s.include_use,
            n_orders_per_delivery=s.n_orders_per_delivery,
        )
        ok = _check(f"drone perturbation {i+1}: {s}", expected, actual)
        all_ok = all_ok and ok
    return all_ok


def main():
    print("Vectorized LCA equivalence tests (atol=1e-6, rtol=1e-9):")
    results = [
        test_ebike_baseline(),
        test_ebike_perturbations(),
        test_drone_baseline(),
        test_drone_perturbations(),
    ]
    if all(results):
        print("All checks passed.")
        return 0
    print("FAILURES detected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
