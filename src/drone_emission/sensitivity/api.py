"""Sensitivity stage entry point.

Runs OAT sweep, 4-mechanism Shapley decomposition, and named-scenarios for a
single region; writes three CSV outputs alongside the existing per-region
data, and runs a runtime sanity check that the baseline scenario reproduces
the values written by ``cal_emission`` to within 0.1%.
"""

from __future__ import annotations

import os

import pandas as pd

from emission.cal_single_delivery import (
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from utils.grid_emission_factor import fetch_factor, normalize_province_name
from utils.province import get_province_from_district

from .named_scenarios import run_named_scenarios_for_region
from .oat import run_oat_for_region
from .shapley import run_shapley_for_region
from .sobol_mc import run_sobol_for_region

SANITY_TOL_PCT = 0.1  # max % deviation from cached baseline before aborting


def _resolve_province(region_info, exp_config) -> str:
    province = region_info.get_stat("province")
    if province:
        return province
    region_name_cn = region_info.to_dict()["region_name_cn"]
    sales_path = exp_config["sales_path"]
    province_raw = get_province_from_district(region_name_cn, sales_path)
    province = normalize_province_name(province_raw)
    region_info.set_stat("province", province)
    return province


def _baseline_sanity_check(region_info, oat_df: pd.DataFrame) -> None:
    """Compare baseline OAT row against ``cal_emission``'s cached stats.

    Aborts (raises) on > SANITY_TOL_PCT deviation. Print-only for missing stats.
    """
    cached_ebike_kg = region_info.get_stat("total_ebike_lifecycle")
    cached_drone_kg = region_info.get_stat("total_drone_lifecycle")
    if cached_ebike_kg is None or cached_drone_kg is None:
        print("  [sanity] cached lifecycle stats missing; skipping comparison")
        return

    baseline_row = oat_df[oat_df["scenario_id"] == "baseline"]
    if baseline_row.empty:
        print("  [sanity] no baseline row in OAT output; skipping comparison")
        return
    new_ebike_kg = float(baseline_row.iloc[0]["ebike_only_lifecycle_kg"])
    new_drone_kg = float(baseline_row.iloc[0]["drone_only_lifecycle_kg"])

    def _pct_diff(a, b):
        denom = max(abs(b), 1e-12)
        return abs(a - b) / denom * 100.0

    e_diff = _pct_diff(new_ebike_kg, cached_ebike_kg)
    d_diff = _pct_diff(new_drone_kg, cached_drone_kg)
    print(
        f"  [sanity] ebike_only: cached={cached_ebike_kg:.3f} kg, "
        f"new={new_ebike_kg:.3f} kg, diff={e_diff:.4f}%"
    )
    print(
        f"  [sanity] drone_only: cached={cached_drone_kg:.3f} kg, "
        f"new={new_drone_kg:.3f} kg, diff={d_diff:.4f}%"
    )
    if e_diff > SANITY_TOL_PCT or d_diff > SANITY_TOL_PCT:
        raise AssertionError(
            f"Sensitivity baseline diverges from cal_emission cache by > {SANITY_TOL_PCT}% "
            f"(ebike {e_diff:.4f}%, drone {d_diff:.4f}%). "
            "Refusing to write outputs."
        )


def _outputs_exist(out_dir: str, region_name_en: str) -> bool:
    suffixes = ("_oat.csv", "_shapley.csv", "_named.csv", "_sobol.csv")
    return all(
        os.path.exists(os.path.join(out_dir, f"{region_name_en}{s}"))
        for s in suffixes
    )


def run_sensitivity(region_info, exp_config) -> None:
    """Compute OAT + Shapley + named-scenarios for a single region.

    Reads the region's OD CSV produced by ``cal_emission`` and writes three
    CSVs (``{region}_oat.csv``, ``{region}_shapley.csv``, ``{region}_named.csv``)
    next to it.
    """
    cfg = region_info.to_dict()
    region_name_en = cfg["region_name_en"].lower()
    region_level = cfg["region_level"]
    region_key = cfg.get("region_key") or region_name_en

    sens_cfg = exp_config.get("sensitivity", {}) or {}
    if not sens_cfg.get("enable", True):
        print(f"Sensitivity stage disabled for {region_name_en}; skipping.")
        return

    out_dir = os.path.join(exp_config["data_dir"], region_level, region_name_en)
    if sens_cfg.get("skip_if_done", True) and _outputs_exist(out_dir, region_name_en):
        print(f"Sensitivity outputs already exist for {region_name_en}; skipping.")
        return

    od_path = os.path.join(out_dir, f"{region_name_en}_od_pairs.csv")
    if not os.path.exists(od_path):
        print(f"  [sensitivity] OD CSV not found for {region_name_en}: {od_path}; skipping")
        return
    od_df = pd.read_csv(od_path)
    if len(od_df) == 0:
        print(f"  [sensitivity] empty OD CSV for {region_name_en}; skipping")
        return

    province = _resolve_province(region_info, exp_config)

    gef_path = exp_config.get("gef_path")
    emission_cfg = get_emission_params(exp_config)
    if gef_path:
        province_gef = fetch_factor(province, gef_path)
    else:
        province_gef = emission_cfg.get("e_use", 0.4)

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    rider_weight = emission_cfg["rider_weight"]
    payload_kg = emission_cfg["payload_kg"]

    print(f"Sensitivity for {region_name_en} ({province}, GEF={province_gef:.4f}): "
          f"{len(od_df)} OD pairs")

    oat_df = run_oat_for_region(
        od_df, region_key=region_key, province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight,
    )
    _baseline_sanity_check(region_info, oat_df)

    shapley_df = run_shapley_for_region(
        od_df, region_key=region_key, province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight, payload_kg=payload_kg,
    )

    named_df = run_named_scenarios_for_region(
        od_df, region_key=region_key, province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight, payload_kg=payload_kg,
    )

    sobol_n = int(sens_cfg.get("sobol_n", 512))
    sobol_seed = int(sens_cfg.get("sobol_seed", 42))
    sobol_df = run_sobol_for_region(
        od_df, region_key=region_key, province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight, payload_kg=payload_kg,
        n=sobol_n, seed=sobol_seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    oat_path = os.path.join(out_dir, f"{region_name_en}_oat.csv")
    shapley_path = os.path.join(out_dir, f"{region_name_en}_shapley.csv")
    named_path = os.path.join(out_dir, f"{region_name_en}_named.csv")
    sobol_path = os.path.join(out_dir, f"{region_name_en}_sobol.csv")
    oat_df.to_csv(oat_path, index=False)
    shapley_df.to_csv(shapley_path, index=False)
    named_df.to_csv(named_path, index=False)
    sobol_df.to_csv(sobol_path, index=False)
    print(f"  [sensitivity] wrote {len(oat_df)} OAT rows -> {oat_path}")
    print(f"  [sensitivity] wrote {len(shapley_df)} shapley rows -> {shapley_path}")
    print(f"  [sensitivity] wrote {len(named_df)} named rows -> {named_path}")
    print(f"  [sensitivity] wrote {len(sobol_df)} sobol rows -> {sobol_path}")
