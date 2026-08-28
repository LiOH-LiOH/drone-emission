"""City-level lifecycle emission calculation."""

from __future__ import annotations

import os

import pandas as pd

from drone_emission.utils.grid_emission_factor import fetch_factor, normalize_province_name
from drone_emission.utils.province import get_province_from_district

from .cal_single_delivery import (
    LifeCycleEmissions,
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from .streaming import process_emission_csv


LIFECYCLE_COMPONENT_COLUMNS = (
    "drone_lifecycle_production_g",
    "drone_lifecycle_use_g",
    "drone_lifecycle_eol_g",
    "ebike_lifecycle_production_g",
    "ebike_lifecycle_use_g",
    "ebike_lifecycle_eol_g",
)


def cal_emission(region_info, exp_config):
    """Calculate and persist emissions while keeping OD memory bounded."""
    region_config = region_info.to_dict()
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    columns = pd.read_csv(od_save_path, nrows=0).columns.tolist()
    print(f"Calculating emissions in bounded chunks for {region_name_en}...")
    print("OD pair columns:", columns)

    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]
    province = region_info.get_stat("province")
    if not province:
        province_raw = get_province_from_district(
            region_config["region_name_cn"], exp_config["sales_path"]
        )
        province = normalize_province_name(province_raw)
        region_info.set_stat("province", province)

    gef_path = exp_config.get("gef_path")
    if gef_path:
        e_use = fetch_factor(province, gef_path)
    else:
        e_use = emission_cfg.get("e_use", 0.4)
        print(f"Warning: gef_path not found; using e_use={e_use}")

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_params.e_use = e_use
    drone_params.e_use = e_use
    ebike_kappa = float(getattr(ebike_params, "kappa", 1.0))
    lca = LifeCycleEmissions(ebike_params=ebike_params, drone_params=drone_params)
    print(
        f"payload={payload_kg} kg rider={rider_weight} kg "
        f"grid_factor={e_use:.4f} kgCO2e/kWh ebike_kappa={ebike_kappa:.6f}"
    )
    return process_emission_csv(
        od_save_path,
        region_info=region_info,
        lca=lca,
        payload_kg=payload_kg,
        rider_weight=rider_weight,
        ebike_params=ebike_params,
        drone_params=drone_params,
        ebike_kappa=ebike_kappa,
        grid_factor=e_use,
    )
