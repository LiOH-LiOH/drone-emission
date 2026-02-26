import os

import numpy as np
import pandas as pd
from utils.province import get_province_from_district
from utils.grid_emission_factor import fetch_factor, normalize_province_name
from visualization.emission import (
    plot_distance_detour_scatter,
    plot_emission_reduction_analysis,
    plot_road_distance_distribution,
)

from .cal_single_delivery import (
    LifeCycleEmissions,
    create_drone_params,
    create_ebike_params,
    get_emission_params,
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

    lca = LifeCycleEmissions(
        ebike_params=ebike_params,
        drone_params=drone_params,
    )
    total_orders = len(od_df)

    print(f"Using payload_kg = {payload_kg} kg, rider_weight = {rider_weight} kg")
    print(
        f"Grid emission factor (e_use) = {e_use:.4f} kg CO2e/kWh ({e_use * 1000:.1f} g CO2e/kWh)"
    )
    print(
        f"E-bike velocity = {ebike_params.velocity} km/h, Drone velocity = {drone_params.v_level} km/h"
    )

    # 检查是否已有排放列，避免重复计算
    has_emission_cols = (
        "drone_emission_kg" in od_df.columns and "ebike_emission_kg" in od_df.columns
    )
    has_lifecycle_cols = (
        "drone_lifecycle_g" in od_df.columns and "ebike_lifecycle_g" in od_df.columns
    )
    has_time_cols = (
        "drone_time_min" in od_df.columns and "ebike_time_min" in od_df.columns
    )

    if has_emission_cols and has_lifecycle_cols and has_time_cols:
        print("Emission and time columns already exist, loading from CSV...")
        drone_emissions = od_df["drone_emission_kg"].values
        ebike_emissions = od_df["ebike_emission_kg"].values
        drone_lifecycle = od_df["drone_lifecycle_g"].values
        ebike_lifecycle = od_df["ebike_lifecycle_g"].values
        drone_time = od_df["drone_time_min"].values
        ebike_time = od_df["ebike_time_min"].values
        road_distances = od_df["road_distance_m"].values / 1000.0
    else:
        print("Calculating emissions and delivery times for each OD pair...")
        drone_emissions_list = []
        ebike_emissions_list = []
        drone_lifecycle_list = []
        ebike_lifecycle_list = []
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

            # Use-phase emissions
            result_drone = lca.calculate_drone_use_phase_emissions(
                distance_km, payload_kg
            )
            result_ebike = lca.calculate_ebike_use_phase_emissions(
                road_distance_km, payload_kg, rider_weight
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
                road_distance_km, n_orders_per_delivery=1, vehicle_type="ebike"
            )

            drone_lifecycle_list.append(lifecycle_drone["total_per_order_g"])
            ebike_lifecycle_list.append(lifecycle_ebike["total_per_order_g"])
            road_distances_list.append(road_distance_km)

        drone_emissions = np.array(drone_emissions_list)
        ebike_emissions = np.array(ebike_emissions_list)
        drone_lifecycle = np.array(drone_lifecycle_list)
        ebike_lifecycle = np.array(ebike_lifecycle_list)
        drone_time = np.array(drone_time_list)
        ebike_time = np.array(ebike_time_list)
        road_distances = np.array(road_distances_list)

        # 保存排放和时间数据到 CSV
        od_df["drone_emission_kg"] = drone_emissions
        od_df["ebike_emission_kg"] = ebike_emissions
        od_df["drone_lifecycle_g"] = drone_lifecycle
        od_df["ebike_lifecycle_g"] = ebike_lifecycle
        od_df["drone_time_min"] = drone_time
        od_df["ebike_time_min"] = ebike_time
        od_df.to_csv(od_save_path, index=False)
        print(f"Emission and time data saved to {od_save_path}")

    # ==================== Use-phase Emission Analysis ====================
    # 基准：全部电动自行车排放
    total_ebike_emissions = ebike_emissions.sum()
    total_drone_emissions = drone_emissions.sum()

    print(f"\n===== Use-phase Emissions =====")
    print(f"全部使用无人机排放: {total_drone_emissions:.2f} kg CO2e")
    print(f"全部使用电动自行车排放: {total_ebike_emissions:.2f} kg CO2e")

    # ==================== Lifecycle Emission Analysis ====================
    total_ebike_lifecycle = ebike_lifecycle.sum() / 1000  # g -> kg
    total_drone_lifecycle = drone_lifecycle.sum() / 1000  # g -> kg

    print(f"\n===== Lifecycle Emissions (Production + Use) =====")
    print(f"全部使用无人机排放: {total_drone_lifecycle:.2f} kg CO2e")
    print(f"全部使用电动自行车排放: {total_ebike_lifecycle:.2f} kg CO2e")
    lifecycle_reduction = (
        (total_ebike_lifecycle - total_drone_lifecycle) / total_ebike_lifecycle * 100
    )
    print(f"无人机相对电动自行车减排: {lifecycle_reduction:.1f}%")

    region_info.set_stat("total_ebike_lifecycle", total_ebike_lifecycle)
    region_info.set_stat("total_drone_lifecycle", total_drone_lifecycle)

    # 根据道路距离分布确定阈值范围
    min_dist = road_distances.min()
    max_dist = road_distances.max()
    print(f"道路距离范围: {min_dist:.2f} - {max_dist:.2f} km")

    # 生成10个合理的阈值，前后加上0和无限大
    middle_thresholds = np.linspace(
        max(0.3, np.percentile(road_distances, 5)),
        np.percentile(road_distances, 95),
        10,
    )
    thresholds = np.concatenate([[-0.01], middle_thresholds, [np.inf]])

    # 计算每个阈值下的结果 (使用生命周期排放)
    results = []
    for threshold in thresholds:
        # road_distance >= threshold 用无人机，否则用电动自行车
        use_drone = road_distances >= threshold

        mixed_lifecycle = np.where(use_drone, drone_lifecycle, ebike_lifecycle).sum()
        drone_ratio = use_drone.sum() / total_orders * 100
        reduction = (
            (ebike_lifecycle.sum() - mixed_lifecycle) / ebike_lifecycle.sum() * 100
        )

        results.append(
            {
                "threshold": threshold,
                "drone_count": use_drone.sum(),
                "ebike_count": (~use_drone).sum(),
                "drone_ratio": drone_ratio,
                "emissions_g": mixed_lifecycle,
                "reduction": reduction,
            }
        )

    results_df = pd.DataFrame(results)

    # Bundle delivery scenarios
    bundle_coeffs = [0.9, 0.8, 0.7, 0.6]
    bundle_results = {}

    for coef in bundle_coeffs:
        # Recalculate e-bike emissions with bundled distances
        bundled_ebike_lifecycle = []
        for row in od_df.itertuples():
            road_distance_km = row.road_distance_m / 1000.0
            bundled_distance = road_distance_km * coef
            lifecycle_ebike = lca.calculate_lifecycle_emissions_per_order_simple(
                bundled_distance, n_orders_per_delivery=1, vehicle_type="ebike"
            )
            bundled_ebike_lifecycle.append(lifecycle_ebike["total_per_order_g"])

        bundled_ebike_lifecycle = np.array(bundled_ebike_lifecycle)

        # Calculate mixed scenario results with bundled e-bike
        coef_results = []
        for threshold in thresholds:
            use_drone = road_distances >= threshold
            mixed_lifecycle = np.where(
                use_drone, drone_lifecycle, bundled_ebike_lifecycle
            ).sum()
            drone_ratio = use_drone.sum() / total_orders * 100
            reduction = (
                (bundled_ebike_lifecycle.sum() - mixed_lifecycle)
                / bundled_ebike_lifecycle.sum()
                * 100
            )
            coef_results.append(
                {
                    "threshold": threshold,
                    "drone_ratio": drone_ratio,
                    "reduction": reduction,
                }
            )

        bundle_results[coef] = pd.DataFrame(coef_results)

    # Find max emission reduction and save stats
    max_idx = results_df["reduction"].idxmax()
    max_row = results_df.loc[max_idx]
    region_info.set_stat("max_lifecycle_reduction_pct", float(max_row["reduction"]))
    region_info.set_stat("optimal_threshold_km", float(max_row["threshold"]))
    region_info.set_stat("optimal_drone_ratio_pct", float(max_row["drone_ratio"]))
    region_info.set_stat(
        "lifecycle_all_drone_reduction_pct", float(lifecycle_reduction)
    )

    # 打印结果表格
    print("\n" + "=" * 80)
    print(
        "混合场景对比 - 生命周期排放 (road_distance > 阈值 用无人机，否则用电动自行车)"
    )
    print("=" * 80)
    print(f"{'阈值(km)':<10} {'无人机占比%':<12} {'排放(g CO2e)':<18} {'减排%':<10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<10.2f} {row['drone_ratio']:<12.1f} "
            f"{row['emissions_g']:<18.2f} {row['reduction']:<10.2f}"
        )
    print("=" * 80)

    # 绘制图形
    fig_save_dir = os.path.join(
        exp_config["figures_dir"],
        region_level,
        region_name_en.lower(),
    )
    save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_emission_reduction.png"
    )
    for_latex = exp_config.get("for_latex", False)
    fig_path = plot_emission_reduction_analysis(
        results_df,
        bundle_results=bundle_results,
        save_path=save_path,
        for_latex=for_latex,
    )
    print(f"\n图表已保存: {fig_path}")

    # 绘制道路距离分布图（按年销量加权）
    annual_sales = od_df["annual_sales"].values
    dist_save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_road_distance_distribution.png"
    )
    dist_fig_path = plot_road_distance_distribution(
        road_distances, annual_sales, save_path=dist_save_path, for_latex=for_latex
    )
    print(f"道路距离分布图已保存: {dist_fig_path}")

    # 绘制距离与绕行比散点图
    straight_distances = od_df["distance_km"].values
    detour_save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_distance_detour_scatter.png"
    )
    detour_fig_path = plot_distance_detour_scatter(
        straight_distances,
        road_distances,
        annual_sales,
        save_path=detour_save_path,
        for_latex=for_latex,
    )
    print(f"距离-绕行比散点图已保存: {detour_fig_path}")

    return results_df
