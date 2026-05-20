"""Global Sobol sensitivity (first + total order) and Monte Carlo bounds.

Replaces the 4-mechanism Shapley decomposition as the main-text §4 robustness
artifact. Where ``shapley.py`` runs a 2^4 coalition enumeration of binary
mechanism toggles, this module performs Saltelli sampling over a continuous
13-parameter design (mirroring OAT FEATURE_LEVELS after the 2026-05-20
pivot) and reports:

* **Sobol first-order index (S1)** per parameter — fraction of the city's
  reduction-% variance attributable to that parameter alone.
* **Sobol total-order index (ST)** per parameter — fraction including all
  interactions involving that parameter.
* **Monte Carlo bounds** on the reduction itself — mean, median, and 2.5/97.5
  percentiles taken across the Saltelli sample for each city.

The 13 features mirror ``oat.py:FEATURE_LEVELS`` exactly:
- 8 drone-tech use/lifespan features (energy_mult, body_kg, battery_mass,
  hover, cruise, headwind, payload, drone/battery lifespans, utilization)
- 2 production-intensity features (e_body, e_battery)
- 1 anchor assumption (κ) and 1 lifespan-amortization (drone_lifespan_yr)

Notably **gef_mult is excluded** — it's a measured value (provincial grid
data), not a new-tech uncertainty, and is treated as a §5 forward-pathway
lever instead. Ranges match OAT FEATURE_LEVELS, with Sobol extending
asymmetric ranges (e.g. hover, headwind) down to the baseline so that
uniform Saltelli sampling covers the full operating envelope.

A separate anchor row reports the all-baseline (κ=1.0) reduction so the §2
headline (29.2%) reconciles with the Sobol/MC summary without the κ=0.7
inconsistency that plagued the Shapley decomposition.

Output CSV schema (per region)::

    row_kind     mechanism/feature        S1   S1_conf  ST   ST_conf  value
    sobol_index  gef_mult                 ...  ...      ...  ...
    sobol_index  drone_energy_mult        ...  ...      ...  ...
    ...                                                                    (9 rows)
    mc_summary   mc_mean_reduction_pct                                 ...
    mc_summary   mc_median_reduction_pct                               ...
    mc_summary   mc_p2.5                                               ...
    mc_summary   mc_p97.5                                              ...
    mc_summary   mc_n_samples                                          ...
    mc_summary   anchor_kappa1_reduction_pct                           ...
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

from emission.cal_single_delivery import DroneParams, EbikeParams

from .oat import BASELINE
from .vectorized_lca import (
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_g,
    compute_ebike_per_order_g,
    optimal_reduction_metrics,
    threshold_sweep_km,
)


# 13-feature Sobol input space (2026-05-20): mirrors OAT exactly under the
# new §4 framing (drone-tech parametric uncertainty + κ anchor robustness;
# gef_mult moved to §5 as a forward-pathway lever, not §4 robustness).
# Ranges match OAT FEATURE_LEVELS; where OAT levels exclude baseline (e.g.
# headwind, hover), Sobol extends the range down to the baseline so that
# uniform sampling covers the full operating envelope.
SOBOL_RANGES: dict[str, tuple[float, float]] = {
    # Anchor assumption (e-bike side)
    "kappa":                       (0.7, 1.0),

    # Use phase
    "drone_energy_mult":           (0.85, 1.15),
    "drone_cruise_kmh":            (50.0, 70.0),
    "drone_hover_s":               (10.0, 60.0),   # baseline 10s included
    "headwind_mps":                (0.0, 7.0),    # baseline 0 included
    "payload_kg":                  (0.7, 1.3),

    # Production (mass)
    "drone_body_kg":               (9.0, 11.5),
    "drone_battery_mass":          (2.7, 3.3),    # NEW (w_battery)

    # Production (intensity) — NEW
    "drone_production_co2_per_kg": (7.0, 11.6),   # NEW (e_body)
    "drone_battery_co2_per_kwh":   (100.0, 180.0), # NEW (e_battery)

    # Lifespan / amortization
    "drone_lifespan_yr":           (2.0, 6.0),
    "drone_battery_lifespan_yr":   (1.0, 3.0),
    "drone_utilization_rate":      (0.5, 1.2),    # NEW (utilization_mult)
}


def _resolve_baseline(feature: str, drone_params: DroneParams) -> float:
    """Return the effective baseline value of an OAT feature.

    Features whose BASELINE entry is None inherit from drone_params (which in
    turn comes from experiment.toml).
    """
    base = BASELINE.get(feature)
    if base is not None:
        return float(base)
    if feature == "drone_cruise_kmh":
        return float(drone_params.v_level)
    if feature == "drone_body_kg":
        return float(drone_params.w_body)
    if feature == "drone_hover_s":
        return float(drone_params.hover_time)
    if feature == "drone_battery_lifespan_yr":
        return float(drone_params.battery_lifespan)
    if feature == "headwind_mps":
        return float(drone_params.headwind_speed_mps)
    if feature == "drone_battery_mass":
        return float(drone_params.w_battery)
    if feature == "drone_production_co2_per_kg":
        return float(drone_params.e_body)
    if feature == "drone_battery_co2_per_kwh":
        return float(drone_params.e_battery)
    raise KeyError(f"no baseline for feature {feature}")


def _reduction_for_sample(
    sample_values: dict,
    road_distances_km: np.ndarray,
    straight_distances_km: np.ndarray,
    ebike_pp: EbikeParams,
    drone_pp: DroneParams,
    rider_weight: float,
    thresholds: np.ndarray,
    fixed: dict,
) -> float:
    """Optimal mixed-strategy reduction (%) for one Saltelli row.

    ``fixed`` carries baseline values for features that are NOT in
    SOBOL_RANGES (notably gef_mult, ebike_lifespan_yr). They are held at
    baseline through the entire Sobol sweep.
    """
    # Build effective DroneParams overrides for params that flow through
    # the dataclass rather than the scenario knobs.
    overrides: dict = {}
    if "drone_cruise_kmh" in sample_values:
        overrides["v_level"] = sample_values["drone_cruise_kmh"]
    if "drone_body_kg" in sample_values:
        overrides["w_body"] = sample_values["drone_body_kg"]
    if "drone_hover_s" in sample_values:
        overrides["hover_time"] = sample_values["drone_hover_s"]
    if "headwind_mps" in sample_values:
        overrides["headwind_speed_mps"] = sample_values["headwind_mps"]
    if "drone_battery_mass" in sample_values:
        overrides["w_battery"] = sample_values["drone_battery_mass"]
    if "drone_production_co2_per_kg" in sample_values:
        overrides["e_body"] = sample_values["drone_production_co2_per_kg"]
    if "drone_battery_co2_per_kwh" in sample_values:
        overrides["e_battery"] = sample_values["drone_battery_co2_per_kwh"]
    drone_params_eff = replace(drone_pp, **overrides) if overrides else drone_pp

    kappa = sample_values.get("kappa", fixed["kappa"])
    gef_mult = sample_values.get("gef_mult", fixed["gef_mult"])
    drone_energy_mult = sample_values.get("drone_energy_mult", fixed["drone_energy_mult"])
    payload_kg = sample_values.get("payload_kg", fixed["payload_kg"])
    drone_lifespan_yr: Optional[float] = sample_values.get(
        "drone_lifespan_yr", fixed["drone_lifespan_yr"])
    drone_battery_lifespan_yr: Optional[float] = sample_values.get("drone_battery_lifespan_yr")
    utilization_mult = sample_values.get("drone_utilization_rate", 1.0)

    e_scen = EbikeScenario(
        kappa=kappa,
        gef_mult=gef_mult,
        energy_mult=1.0,
        payload_kg=payload_kg,
        rider_weight=rider_weight,
        lifespan_yr=fixed["ebike_lifespan_yr"],
    )
    d_scen = DroneScenario(
        gef_mult=gef_mult,
        energy_mult=drone_energy_mult,
        payload_kg=payload_kg,
        lifespan_yr=drone_lifespan_yr,
        battery_lifespan_yr=drone_battery_lifespan_yr,
        utilization_mult=utilization_mult,
    )
    ebike_g = compute_ebike_per_order_g(road_distances_km, ebike_pp, e_scen)
    drone_g = compute_drone_per_order_g(straight_distances_km, drone_params_eff, d_scen)
    metrics = optimal_reduction_metrics(road_distances_km, ebike_g, drone_g, thresholds)
    return float(metrics["best_reduction_vs_scenario_pct"])


def run_sobol_for_region(
    od_df: pd.DataFrame,
    region_key: str,
    province: str,
    province_gef_kwh: float,
    ebike_params: EbikeParams,
    drone_params: DroneParams,
    rider_weight: float = 70.0,
    payload_kg: float = 1.0,
    n: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """Saltelli/Sobol global SA + MC bounds for a single region.

    Returns a DataFrame with rows for the 9 Sobol indices plus 6 MC-summary
    rows. See module docstring for the schema.
    """
    road_distances_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
    straight_distances_km = od_df["distance_km"].to_numpy(dtype=float)
    if len(road_distances_km) == 0:
        return pd.DataFrame()

    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)
    thresholds = threshold_sweep_km(road_distances_km)

    # Features held constant across the Sobol sweep at their OAT baseline.
    # gef_mult is fixed (it moved to §5 as a pathway lever, not a §4
    # robustness feature). ebike_lifespan_yr is fixed (mature tech, sub-1pp).
    # The other entries shadow SOBOL_RANGES keys so that sample_values.get(...)
    # always has a fallback — features inside SOBOL_RANGES overwrite the
    # fixed value via sample_values on each Saltelli row.
    fixed = {
        "kappa":                BASELINE["kappa"],
        "gef_mult":             BASELINE["gef_mult"],
        "drone_energy_mult":    BASELINE["drone_energy_mult"],
        "payload_kg":           BASELINE["payload_kg"],
        "ebike_lifespan_yr":    BASELINE["ebike_lifespan_yr"],
        "drone_lifespan_yr":    BASELINE["drone_lifespan_yr"],
    }

    # Build the SALib problem dict in deterministic column order.
    feature_names = list(SOBOL_RANGES.keys())
    bounds = [list(SOBOL_RANGES[f]) for f in feature_names]
    problem = {
        "num_vars": len(feature_names),
        "names": feature_names,
        "bounds": bounds,
    }

    # Import inside the function so a missing SALib import does not break
    # the OAT/Shapley/named pipeline when only those are wanted.
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import saltelli

    # calc_second_order=False keeps the sample size at N*(d+2) and we only
    # need S1 + ST. The seed lets us reproduce CSVs byte-for-byte.
    param_values = saltelli.sample(problem, n, calc_second_order=False, skip_values=1024)
    # SALib emits a DeprecationWarning about saltelli with non-power-of-2 N;
    # 512 is power-of-2 so we are clean.

    Y = np.empty(param_values.shape[0], dtype=float)
    for i, row in enumerate(param_values):
        sample_values = {name: float(row[j]) for j, name in enumerate(feature_names)}
        Y[i] = _reduction_for_sample(
            sample_values,
            road_distances_km, straight_distances_km,
            ebike_pp, drone_pp, rider_weight,
            thresholds, fixed,
        )

    Si = sobol_analyze.analyze(
        problem, Y, calc_second_order=False, num_resamples=100,
        conf_level=0.95, seed=seed, print_to_console=False,
    )

    rows = []
    for j, fname in enumerate(feature_names):
        rows.append({
            "region_key": region_key,
            "province": province,
            "row_kind": "sobol_index",
            "feature": fname,
            "S1": float(Si["S1"][j]),
            "S1_conf": float(Si["S1_conf"][j]),
            "ST": float(Si["ST"][j]),
            "ST_conf": float(Si["ST_conf"][j]),
            "value": float("nan"),
        })

    # Anchor (all-baseline, κ=1.0) reduction for reconciling with the §2
    # headline. Same call as a Sobol sample with all values at baseline.
    anchor_sample = {f: _resolve_baseline(f, drone_pp) for f in feature_names}
    anchor_red = _reduction_for_sample(
        anchor_sample,
        road_distances_km, straight_distances_km,
        ebike_pp, drone_pp, rider_weight,
        thresholds, fixed,
    )

    mc_rows = [
        ("mc_mean_reduction_pct",       float(np.mean(Y))),
        ("mc_median_reduction_pct",     float(np.median(Y))),
        ("mc_p2.5",                     float(np.percentile(Y, 2.5))),
        ("mc_p97.5",                    float(np.percentile(Y, 97.5))),
        ("mc_n_samples",                float(Y.size)),
        ("anchor_kappa1_reduction_pct", float(anchor_red)),
    ]
    for name, val in mc_rows:
        rows.append({
            "region_key": region_key,
            "province": province,
            "row_kind": "mc_summary",
            "feature": name,
            "S1": float("nan"),
            "S1_conf": float("nan"),
            "ST": float("nan"),
            "ST_conf": float("nan"),
            "value": val,
        })

    return pd.DataFrame(rows)
