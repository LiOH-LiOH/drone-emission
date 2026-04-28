"""Smoke test the sensitivity stage on the local nanshan district data.

Uses the existing nanshan OD CSV to verify OAT, Shapley and named-scenarios
modules end-to-end. Skips the registry-based path (nanshan is a district and
not in regions.json) and calls the per-region computations directly.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "src", "drone_emission"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from emission.cal_single_delivery import create_drone_params, create_ebike_params, get_emission_params  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.grid_emission_factor import fetch_factor  # noqa: E402

from sensitivity.named_scenarios import run_named_scenarios_for_region  # noqa: E402
from sensitivity.oat import run_oat_for_region  # noqa: E402
from sensitivity.shapley import run_shapley_for_region  # noqa: E402


def main():
    project_root = os.path.normpath(os.path.join(HERE, ".."))
    config = load_config(os.path.join(project_root, "experiment.toml"))
    od_path = os.path.join(config["data_dir"], "district", "nanshan", "nanshan_od_pairs.csv")
    print(f"Loading OD CSV: {od_path}")
    od_df = pd.read_csv(od_path)
    print(f"  rows: {len(od_df):,}")

    province = "广东"  # Nanshan is in Shenzhen, Guangdong
    province_gef = fetch_factor(province, config["gef_path"])
    print(f"  province={province}, GEF={province_gef:.4f} kg CO2e/kWh")

    ebike_params = create_ebike_params(config)
    drone_params = create_drone_params(config)
    emission_cfg = get_emission_params(config)
    rider_weight = emission_cfg["rider_weight"]
    payload_kg = emission_cfg["payload_kg"]

    t0 = time.time()
    oat_df = run_oat_for_region(
        od_df, region_key="nanshan", province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight,
    )
    print(f"\nOAT: {len(oat_df)} rows, {time.time() - t0:.2f}s")
    print(oat_df[["scenario_id", "feature_varied", "optimal_threshold_km",
                  "optimal_drone_ratio_pct", "reduction_vs_scenario_ebike_pct",
                  "reduction_vs_anchor_ebike_pct", "beneficial"]].to_string(index=False))

    t0 = time.time()
    sh_df = run_shapley_for_region(
        od_df, region_key="nanshan", province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight, payload_kg=payload_kg,
    )
    print(f"\nShapley: {len(sh_df)} rows, {time.time() - t0:.2f}s")
    print(sh_df.to_string(index=False))

    t0 = time.time()
    nm_df = run_named_scenarios_for_region(
        od_df, region_key="nanshan", province=province,
        province_gef_kwh=province_gef,
        ebike_params=ebike_params, drone_params=drone_params,
        rider_weight=rider_weight, payload_kg=payload_kg,
    )
    print(f"\nNamed scenarios: {len(nm_df)} rows, {time.time() - t0:.2f}s")
    print(nm_df.to_string(index=False))

    # Persist outputs so the consolidation tool + plotting can pick them up.
    out_dir = os.path.dirname(od_path)
    oat_df.to_csv(os.path.join(out_dir, "nanshan_oat.csv"), index=False)
    sh_df.to_csv(os.path.join(out_dir, "nanshan_shapley.csv"), index=False)
    nm_df.to_csv(os.path.join(out_dir, "nanshan_named.csv"), index=False)
    print(f"\nOutputs persisted under {out_dir}")


if __name__ == "__main__":
    main()
