import os

import numpy as np
import pandas as pd
from utils.province import get_province_from_district
from utils.grid_emission_factor import fetch_factor, normalize_province_name

from .cal_single_delivery import (
    LifeCycleEmissions,
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)

LIFECYCLE_COMPONENT_COLUMNS = (
    "drone_lifecycle_production_g",
    "drone_lifecycle_use_g",
    "drone_lifecycle_eol_g",
    "ebike_lifecycle_production_g",
    "ebike_lifecycle_use_g",
    "ebike_lifecycle_eol_g",
)


def cal_emission(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]

    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    od_df = pd.read_csv(od_save_path)
    print(f"Calculating emissions for {len(od_df)} OD pairs in {region_name_en}...")
    print("OD pair columns:", end=" ")
    print(od_df.columns.tolist())

    # 从配置文件读取排放模型参数
    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    # 获取省份名称（用于电网排放因子查询）
    province = region_info.get_stat("province")
    if not province:
        # 从 combined_data.csv 查询省份
        region_name_cn = region_config["region_name_cn"]
        sales_path = exp_config["sales_path"]
        province_raw = get_province_from_district(region_name_cn, sales_path)
        # 转换为 gef_china.csv 中的格式
        province = normalize_province_name(province_raw)
        region_info.set_stat("province", province)
    print(f"Region province: {province}")

    # 获取省份对应的电网排放因子
    gef_path = exp_config.get("gef_path")
    if gef_path:
        e_use = fetch_factor(province, gef_path)
        print(f"CO2 emission factor for {province}: {e_use} kgCO2eq/kWh")
    else:
        e_use = emission_cfg.get("e_use", 0.4)
        print(f"Warning: gef_path not found in config, using default e_use = {e_use}")

    # 创建参数时覆盖 e_use
    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    ebike_params.e_use = e_use
    drone_params.e_use = e_use
    ebike_kappa = float(getattr(ebike_params, "kappa", 1.0))

    lca = LifeCycleEmissions(
        ebike_params=ebike_params,
        drone_params=drone_params,
    )

    print(f"Using payload_kg = {payload_kg} kg, rider_weight = {rider_weight} kg")
    print(
        f"Grid emission factor (e_use) = {e_use:.4f} kg CO2e/kWh ({e_use * 1000:.1f} g CO2e/kWh)"
    )
    print(
        f"E-bike velocity = {ebike_params.velocity} km/h, "
        f"Drone velocity = {drone_params.v_level} km/h"
    )
    print(f"E-bike kappa = {ebike_kappa:.6f}")

    # 检查是否已有排放列，避免重复计算
    has_emission_cols = (
        "drone_emission_kg" in od_df.columns and "ebike_emission_kg" in od_df.columns
    )
    has_lifecycle_cols = (
        "drone_lifecycle_g" in od_df.columns and "ebike_lifecycle_g" in od_df.columns
    )
    has_lifecycle_component_cols = all(
        col in od_df.columns for col in LIFECYCLE_COMPONENT_COLUMNS
    )
    has_time_cols = (
        "drone_time_min" in od_df.columns and "ebike_time_min" in od_df.columns
    )
    cached_ebike_kappas = (
        od_df["ebike_kappa"].dropna().to_numpy(dtype=float)
        if "ebike_kappa" in od_df.columns
        else np.array([])
    )
    has_matching_ebike_kappa = (
        cached_ebike_kappas.size == len(od_df)
        and np.allclose(cached_ebike_kappas, ebike_kappa)
    )

    if (
        has_emission_cols
        and has_lifecycle_cols
        and has_lifecycle_component_cols
        and has_time_cols
        and has_matching_ebike_kappa
    ):
        print("Emission and time columns already exist, loading from CSV...")
        drone_emissions = od_df["drone_emission_kg"].values
        ebike_emissions = od_df["ebike_emission_kg"].values
        drone_lifecycle = od_df["drone_lifecycle_g"].values
        ebike_lifecycle = od_df["ebike_lifecycle_g"].values
        drone_lifecycle_production = od_df["drone_lifecycle_production_g"].values
        drone_lifecycle_use = od_df["drone_lifecycle_use_g"].values
        drone_lifecycle_eol = od_df["drone_lifecycle_eol_g"].values
        ebike_lifecycle_production = od_df["ebike_lifecycle_production_g"].values
        ebike_lifecycle_use = od_df["ebike_lifecycle_use_g"].values
        ebike_lifecycle_eol = od_df["ebike_lifecycle_eol_g"].values
        drone_time = od_df["drone_time_min"].values
        ebike_time = od_df["ebike_time_min"].values
        road_distances = od_df["road_distance_m"].values / 1000.0
    else:
        print("Calculating emissions and delivery times for each OD pair...")
        drone_emissions_list = []
        ebike_emissions_list = []
        drone_lifecycle_list = []
        ebike_lifecycle_list = []
        drone_lifecycle_production_list = []
        drone_lifecycle_use_list = []
        drone_lifecycle_eol_list = []
        ebike_lifecycle_production_list = []
        ebike_lifecycle_use_list = []
        ebike_lifecycle_eol_list = []
        drone_time_list = []
        ebike_time_list = []
        road_distances_list = []

        # Get velocity parameters for one-way time calculation
        ebike_velocity = ebike_params.velocity  # km/h
        drone_v_level = drone_params.v_level  # km/h
        drone_v_vertical = drone_params.v_vertical  # km/h
        drone_h_takeoff = drone_params.h_takeoff  # m
        drone_hover_time = drone_params.hover_time  # seconds

        for row in od_df.itertuples():
            distance_km = row.distance_km  # 直线距离，用于无人机
            road_distance_km = row.road_distance_m / 1000.0  # 道路距离，用于电动自行车
            ebike_emission_distance_km = road_distance_km * ebike_kappa

            # Use-phase emissions
            result_drone = lca.calculate_drone_use_phase_emissions(
                distance_km, payload_kg
            )
            result_ebike = lca.calculate_ebike_use_phase_emissions(
                ebike_emission_distance_km, payload_kg, rider_weight
            )

            drone_emissions_list.append(result_drone["emissions_kg_co2e"])
            ebike_emissions_list.append(result_ebike["emissions_kg_co2e"])

            # One-way delivery times (in minutes)
            # Ebike: road_distance / velocity
            ebike_oneway_min = (road_distance_km / ebike_velocity) * 60
            ebike_time_list.append(ebike_oneway_min)

            # Drone: takeoff + level_flight + landing + hover
            t_takeoff_h = (drone_h_takeoff / 1000) / drone_v_vertical  # hours
            t_level_h = distance_km / drone_v_level  # hours
            t_landing_h = (drone_h_takeoff / 1000) / drone_v_vertical  # hours
            t_hover_h = drone_hover_time / 3600  # hours
            drone_oneway_min = (t_takeoff_h + t_level_h + t_landing_h + t_hover_h) * 60
            drone_time_list.append(drone_oneway_min)

            # Lifecycle emissions (per order, g CO2e)
            lifecycle_drone = lca.calculate_lifecycle_emissions_per_order_simple(
                distance_km, n_orders_per_delivery=1, vehicle_type="drone"
            )
            lifecycle_ebike = lca.calculate_lifecycle_emissions_per_order_simple(
                ebike_emission_distance_km, n_orders_per_delivery=1, vehicle_type="ebike"
            )

            drone_lifecycle_list.append(lifecycle_drone["total_per_order_g"])
            ebike_lifecycle_list.append(lifecycle_ebike["total_per_order_g"])
            drone_lifecycle_production_list.append(
                lifecycle_drone["production_per_order_g"]
            )
            drone_lifecycle_use_list.append(lifecycle_drone["use_per_order_g"])
            drone_lifecycle_eol_list.append(
                lifecycle_drone["end_of_life_per_order_g"]
            )
            ebike_lifecycle_production_list.append(
                lifecycle_ebike["production_per_order_g"]
            )
            ebike_lifecycle_use_list.append(lifecycle_ebike["use_per_order_g"])
            ebike_lifecycle_eol_list.append(
                lifecycle_ebike["end_of_life_per_order_g"]
            )
            road_distances_list.append(road_distance_km)

        drone_emissions = np.array(drone_emissions_list)
        ebike_emissions = np.array(ebike_emissions_list)
        drone_lifecycle = np.array(drone_lifecycle_list)
        ebike_lifecycle = np.array(ebike_lifecycle_list)
        drone_lifecycle_production = np.array(drone_lifecycle_production_list)
        drone_lifecycle_use = np.array(drone_lifecycle_use_list)
        drone_lifecycle_eol = np.array(drone_lifecycle_eol_list)
        ebike_lifecycle_production = np.array(ebike_lifecycle_production_list)
        ebike_lifecycle_use = np.array(ebike_lifecycle_use_list)
        ebike_lifecycle_eol = np.array(ebike_lifecycle_eol_list)
        drone_time = np.array(drone_time_list)
        ebike_time = np.array(ebike_time_list)
        road_distances = np.array(road_distances_list)

        # 保存排放和时间数据到 CSV
        od_df["drone_emission_kg"] = drone_emissions
        od_df["ebike_emission_kg"] = ebike_emissions
        od_df["drone_lifecycle_g"] = drone_lifecycle
        od_df["ebike_lifecycle_g"] = ebike_lifecycle
        od_df["drone_lifecycle_production_g"] = drone_lifecycle_production
        od_df["drone_lifecycle_use_g"] = drone_lifecycle_use
        od_df["drone_lifecycle_eol_g"] = drone_lifecycle_eol
        od_df["ebike_lifecycle_production_g"] = ebike_lifecycle_production
        od_df["ebike_lifecycle_use_g"] = ebike_lifecycle_use
        od_df["ebike_lifecycle_eol_g"] = ebike_lifecycle_eol
        od_df["drone_time_min"] = drone_time
        od_df["ebike_time_min"] = ebike_time
        od_df["ebike_kappa"] = ebike_kappa
        od_df.to_csv(od_save_path, index=False)
        print(f"Emission and time data saved to {od_save_path}")

    if "annual_sales" not in od_df.columns:
        raise ValueError(
            f"{od_save_path} is missing annual_sales; city-level emissions must be "
            "weighted by OD-pair order counts."
        )
    annual_sales = pd.to_numeric(od_df["annual_sales"], errors="coerce").fillna(0).to_numpy()
    total_annual_orders = annual_sales.sum()
    if total_annual_orders <= 0:
        raise ValueError(
            f"{od_save_path} has no positive annual_sales; cannot compute weighted emissions."
        )

    # ==================== Use-phase Emission Analysis ====================
    # 基准：全部电动自行车排放
    total_ebike_emissions = np.sum(ebike_emissions * annual_sales)
    total_drone_emissions = np.sum(drone_emissions * annual_sales)

    print(f"\n===== Use-phase Emissions =====")
    print(f"全部使用无人机排放: {total_drone_emissions:.2f} kg CO2e")
    print(f"全部使用电动自行车排放: {total_ebike_emissions:.2f} kg CO2e")

    # ==================== Lifecycle Emission Analysis ====================
    total_ebike_lifecycle = np.sum(ebike_lifecycle * annual_sales) / 1000  # g -> kg
    total_drone_lifecycle = np.sum(drone_lifecycle * annual_sales) / 1000  # g -> kg
    total_ebike_production_emission = (
        np.sum(ebike_lifecycle_production * annual_sales) / 1000
    )
    total_ebike_use_phase_emission = np.sum(ebike_lifecycle_use * annual_sales) / 1000
    total_ebike_end_of_life_emission = (
        np.sum(ebike_lifecycle_eol * annual_sales) / 1000
    )
    total_drone_production_emission = (
        np.sum(drone_lifecycle_production * annual_sales) / 1000
    )
    total_drone_use_phase_emission = np.sum(drone_lifecycle_use * annual_sales) / 1000
    total_drone_end_of_life_emission = (
        np.sum(drone_lifecycle_eol * annual_sales) / 1000
    )

    print(f"\n===== Lifecycle Emissions (Production + Use + End of Life) =====")
    print(f"全部使用无人机排放: {total_drone_lifecycle:.2f} kg CO2e")
    print(f"全部使用电动自行车排放: {total_ebike_lifecycle:.2f} kg CO2e")
    print(
        "电动自行车分项: "
        f"production={total_ebike_production_emission:.2f}, "
        f"use={total_ebike_use_phase_emission:.2f}, "
        f"end_of_life={total_ebike_end_of_life_emission:.2f} kg CO2e"
    )
    print(
        "无人机分项: "
        f"production={total_drone_production_emission:.2f}, "
        f"use={total_drone_use_phase_emission:.2f}, "
        f"end_of_life={total_drone_end_of_life_emission:.2f} kg CO2e"
    )
    lifecycle_reduction = (
        (total_ebike_lifecycle - total_drone_lifecycle) / total_ebike_lifecycle * 100
    )
    print(f"无人机相对电动自行车减排: {lifecycle_reduction:.1f}%")

    region_info.set_stat("total_ebike_lifecycle", total_ebike_lifecycle)
    region_info.set_stat("total_drone_lifecycle", total_drone_lifecycle)
    region_info.set_stat(
        "total_ebike_production_emission_kg",
        total_ebike_production_emission,
    )
    region_info.set_stat(
        "total_ebike_use_phase_emission_kg",
        total_ebike_use_phase_emission,
    )
    region_info.set_stat(
        "total_ebike_end_of_life_emission_kg",
        total_ebike_end_of_life_emission,
    )
    region_info.set_stat(
        "total_drone_production_emission_kg",
        total_drone_production_emission,
    )
    region_info.set_stat(
        "total_drone_use_phase_emission_kg",
        total_drone_use_phase_emission,
    )
    region_info.set_stat(
        "total_drone_end_of_life_emission_kg",
        total_drone_end_of_life_emission,
    )
    region_info.set_stat(
        "drone_minus_ebike_lifecycle_kg",
        total_drone_lifecycle - total_ebike_lifecycle,
    )
    region_info.set_stat(
        "lifecycle_all_drone_reduction_pct", float(lifecycle_reduction)
    )
    region_info.set_stat("ebike_kappa", ebike_kappa)

    use_drone_oracle = drone_lifecycle < ebike_lifecycle
    oracle_lifecycle = (
        np.sum(np.where(use_drone_oracle, drone_lifecycle, ebike_lifecycle) * annual_sales)
        / 1000
    )
    oracle_reduction = (
        (total_ebike_lifecycle - oracle_lifecycle) / total_ebike_lifecycle * 100
    )
    oracle_drone_order_ratio = annual_sales[use_drone_oracle].sum() / total_annual_orders * 100
    region_info.set_stat("oracle_min_lifecycle_kg", oracle_lifecycle)
    region_info.set_stat("oracle_min_reduction_vs_ebike_pct", oracle_reduction)
    region_info.set_stat("oracle_drone_order_ratio_pct", oracle_drone_order_ratio)

    for stale_key in (
        "max_lifecycle_reduction_pct",
        "optimal_threshold_km",
        "optimal_drone_ratio_pct",
    ):
        region_info.stats.pop(stale_key, None)

    min_dist = road_distances.min()
    max_dist = road_distances.max()
    print(f"道路距离范围: {min_dist:.2f} - {max_dist:.2f} km")
    print(
        "All-drone minus all-ebike lifecycle: "
        f"{total_drone_lifecycle - total_ebike_lifecycle:.2f} kg CO2e"
    )
    print(f"Oracle minimum lifecycle: {oracle_lifecycle:.2f} kg CO2e")
    print(f"Oracle reduction vs all-ebike: {oracle_reduction:.2f}%")

    return pd.DataFrame(
        [
            {
                "total_drone_lifecycle_kg": total_drone_lifecycle,
                "total_ebike_lifecycle_kg": total_ebike_lifecycle,
                "total_drone_production_emission_kg": total_drone_production_emission,
                "total_drone_use_phase_emission_kg": total_drone_use_phase_emission,
                "total_drone_end_of_life_emission_kg": total_drone_end_of_life_emission,
                "total_ebike_production_emission_kg": total_ebike_production_emission,
                "total_ebike_use_phase_emission_kg": total_ebike_use_phase_emission,
                "total_ebike_end_of_life_emission_kg": total_ebike_end_of_life_emission,
                "drone_minus_ebike_lifecycle_kg": total_drone_lifecycle
                - total_ebike_lifecycle,
                "oracle_min_lifecycle_kg": oracle_lifecycle,
                "oracle_min_reduction_vs_ebike_pct": oracle_reduction,
                "oracle_drone_order_ratio_pct": oracle_drone_order_ratio,
            }
        ]
    )
