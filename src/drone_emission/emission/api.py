import os

import numpy as np
import pandas as pd
from utils.grid_emission_factor import get_factor
from visualization.emission import (
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

    # parent_city 是小写英文城市名 (如 "shenzhen")，与 CITY_EN_TO_CN 键匹配
    parent_city = region_config["parent_city"]
    print(f"Region parent city: {parent_city}")

    # 获取城市对应的电网排放因子
    gef_path = exp_config.get("gef_path")
    if gef_path:
        e_use = get_factor(parent_city, gef_path)
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

    # 先计算每个订单的无人机和电动自行车排放
    drone_emissions_list = []
    ebike_emissions_list = []
    road_distances_list = []

    for row in od_df.itertuples():
        distance_km = row.distance_km  # 直线距离，用于无人机
        road_distance_km = row.road_distance_m / 1000.0  # 道路距离，用于电动自行车

        result_drone = lca.calculate_drone_use_phase_emissions(distance_km, payload_kg)
        result_ebike = lca.calculate_ebike_use_phase_emissions(
            road_distance_km, payload_kg, rider_weight
        )

        drone_emissions_list.append(result_drone["emissions_kg_co2e"])
        ebike_emissions_list.append(result_ebike["emissions_kg_co2e"])
        road_distances_list.append(road_distance_km)

    drone_emissions = np.array(drone_emissions_list)
    ebike_emissions = np.array(ebike_emissions_list)
    road_distances = np.array(road_distances_list)

    # 基准：全部电动自行车排放
    total_ebike_emissions = ebike_emissions.sum()
    total_drone_emissions = drone_emissions.sum()

    print(f"\n全部使用无人机排放: {total_drone_emissions:.2f} kg CO2e")
    print(f"全部使用电动自行车排放: {total_ebike_emissions:.2f} kg CO2e")

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

    # 计算每个阈值下的结果
    results = []
    for threshold in thresholds:
        # road_distance >= threshold 用无人机，否则用电动自行车
        use_drone = road_distances >= threshold

        mixed_emissions = np.where(use_drone, drone_emissions, ebike_emissions).sum()
        drone_ratio = use_drone.sum() / total_orders * 100
        reduction = (
            (total_ebike_emissions - mixed_emissions) / total_ebike_emissions * 100
        )

        results.append(
            {
                "threshold": threshold,
                "drone_count": use_drone.sum(),
                "ebike_count": (~use_drone).sum(),
                "drone_ratio": drone_ratio,
                "emissions": mixed_emissions,
                "reduction": reduction,
            }
        )

    results_df = pd.DataFrame(results)

    # 打印结果表格
    print("\n" + "=" * 80)
    print("混合场景对比 (road_distance > 阈值 用无人机，否则用电动自行车)")
    print("=" * 80)
    print(f"{'阈值(km)':<10} {'无人机占比%':<12} {'排放(kg CO2e)':<18} {'减排%':<10}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(
            f"{row['threshold']:<10.2f} {row['drone_ratio']:<12.1f} "
            f"{row['emissions']:<18.2f} {row['reduction']:<10.2f}"
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
    fig_path = plot_emission_reduction_analysis(results_df, save_path=save_path)
    print(f"\n图表已保存: {fig_path}")

    # 绘制道路距离分布图（按年销量加权）
    annual_sales = od_df["annual_sales"].values
    dist_save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_road_distance_distribution.png"
    )
    dist_fig_path = plot_road_distance_distribution(
        road_distances, annual_sales, save_path=dist_save_path
    )
    print(f"道路距离分布图已保存: {dist_fig_path}")

    return results_df
