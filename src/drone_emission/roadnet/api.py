import os

import numpy as np
import osmnx as ox
import pandas as pd
from visualization.roadnet import visualize_distance_comparison, visualize_roadnet

from .osm_raodnet import fetch_bikeable_roadnet
from .router import RoadNetRouter


def load_roadnet(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_name_cn = region_config.get("region_name_cn")
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict")
    large_bounds_dict = region_info.get_cache("enlarged_boundaries")

    # Setup data directory and file path
    data_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    roadnet_data_path = os.path.join(
        data_dir, f"{region_name_en.lower()}_roadnet.graphml"
    )

    # Fetch road network if requested or file doesn't exist
    if exp_config.get("fetch_roadnet", False) or not os.path.exists(roadnet_data_path):
        fetch_bikeable_roadnet(
            bbox_4326=large_bounds_dict,
            save_path=roadnet_data_path,
        )

    # Load road network from file
    print(f"Reading {roadnet_data_path}...")
    try:
        G = ox.load_graphml(roadnet_data_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print(f"Loaded road network: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    region_info.set_stat("roadnet_node_count", len(G.nodes))
    region_info.set_stat("roadnet_edge_count", len(G.edges))

    # Visualize road network
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    roadnet_fig_path = os.path.join(fig_dir, f"{region_name_en.lower()}_roadnet.png")

    for_latex = exp_config.get("for_latex", False)
    visualize_roadnet(
        G,
        city_name_cn=region_name_cn,
        boundaries_dict=large_bounds_dict,
        original_boundaries_dict=boundaries_dict,
        save_path=roadnet_fig_path,
        for_latex=for_latex,
    )

    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    od_df = pd.read_csv(od_save_path)
    print(f"Loaded {len(od_df)} OD pairs")

    router = RoadNetRouter(G)

    # Only compute road distances for rows with NaN values
    if "road_distance_m" not in od_df.columns:
        od_df["road_distance_m"] = np.nan

    nan_mask = od_df["road_distance_m"].isna()
    nan_count = nan_mask.sum()

    if nan_count > 0:
        print(f"Computing road distances for {nan_count} rows with NaN values...")

        # Get only the rows that need computation
        nan_df = od_df[nan_mask]
        origins = nan_df[["from_lat", "from_lng"]].values
        destinations = nan_df[["to_lat", "to_lng"]].values

        road_distances = router.compute_od_distances_batch(origins, destinations)

        # Update only the NaN rows
        od_df.loc[nan_mask, "road_distance_m"] = road_distances

        # Save updated OD pairs
        od_df.to_csv(od_save_path, index=False)
        print(f"Updated OD pairs with road distances saved to {od_save_path}")
    else:
        print("All road distances already computed, skipping...")

    road_distances = od_df["road_distance_m"].values

    # Statistics
    valid_mask = road_distances >= 0
    if valid_mask.sum() > 0:
        # use weighted mean of road_distance_m and annual_sales
        avg_road_dist = (
            road_distances[valid_mask] * od_df.loc[valid_mask, "annual_sales"]
        ).sum() / od_df.loc[valid_mask, "annual_sales"].sum()
        avg_straight_dist = (
            (
                od_df.loc[valid_mask, "distance_km"]
                * od_df.loc[valid_mask, "annual_sales"]
            ).sum()
            / od_df.loc[valid_mask, "annual_sales"].sum()
            * 1000
        )
        detour_ratio = avg_road_dist / avg_straight_dist if avg_straight_dist > 0 else 0

        region_info.set_stat("avg_road_distance_m", avg_road_dist)
        region_info.set_stat("avg_straight_distance_m", avg_straight_dist)
        region_info.set_stat("detour_ratio", detour_ratio)
        region_info.set_stat("valid_path_count", int(valid_mask.sum()))
        region_info.set_stat("invalid_path_count", int((~valid_mask).sum()))

        print(f"\n=== Road Distance Statistics ===")
        print(f"Valid paths: {valid_mask.sum():,} / {len(od_df):,}")
        print(f"Avg straight-line distance: {avg_straight_dist:.1f} m")
        print(f"Avg road network distance: {avg_road_dist:.1f} m")
        print(f"Detour ratio: {detour_ratio:.2f}")

        # Visualize distance comparison with 10 samples
        _visualize_sample_paths(
            od_df, router, valid_mask, boundaries_dict, fig_dir, region_name_en, exp_config
        )


def _visualize_sample_paths(
    od_df, router, valid_mask, boundaries_dict, fig_dir, region_name_en, exp_config, n_samples=10
):
    """抽取样本并绘制直线距离与路网距离对比图"""
    # 筛选有效路径，并确保有一定的路网距离
    valid_df = od_df[valid_mask & (od_df["road_distance_m"] > 100)].copy()

    if len(valid_df) < n_samples:
        print(f"Warning: Only {len(valid_df)} valid samples available")
        n_samples = len(valid_df)

    if n_samples == 0:
        print("No valid samples for distance comparison visualization")
        return

    # 按绕行比分层抽样，获取不同绕行程度的样本
    valid_df["detour_ratio"] = valid_df["road_distance_m"] / (
        valid_df["distance_km"] * 1000
    )

    # 分4个分位数区间，每个区间抽1个样本
    samples = []
    for q in [0.2, 0.4, 0.6, 0.8]:
        quantile_val = valid_df["detour_ratio"].quantile(q)
        # 找到最接近该分位数的样本
        closest_idx = (valid_df["detour_ratio"] - quantile_val).abs().nsmallest(1).index
        samples.extend(closest_idx.tolist())

    # 去重并取前n_samples个
    samples = list(dict.fromkeys(samples))[:n_samples]
    sample_df = valid_df.loc[samples]

    print(f"\n=== Visualizing {len(sample_df)} sample paths ===")

    # 计算每个样本的完整路径
    sample_paths = []
    for _, row in sample_df.iterrows():
        origin = (row["from_lat"], row["from_lng"])
        destination = (row["to_lat"], row["to_lng"])

        path_result = router.find_shortest_path(origin, destination)

        sample_paths.append(
            {
                "origin": origin,
                "destination": destination,
                "straight_distance": row["distance_km"] * 1000,
                "road_distance": row["road_distance_m"],
                "path_result": path_result,
            }
        )

        print(
            f"  Sample: straight={row['distance_km']*1000:.0f}m, "
            f"road={row['road_distance_m']:.0f}m, "
            f"detour={row['detour_ratio']:.2f}"
        )

    # 计算样本的边界范围
    all_lats = []
    all_lngs = []
    for sp in sample_paths:
        all_lats.extend([sp["origin"][0], sp["destination"][0]])
        all_lngs.extend([sp["origin"][1], sp["destination"][1]])

    padding = 0.005
    sample_bounds = {
        "min_lat": min(all_lats) - padding,
        "max_lat": max(all_lats) + padding,
        "min_lng": min(all_lngs) - padding,
        "max_lng": max(all_lngs) + padding,
    }

    # 绘制对比图
    comparison_fig_path = os.path.join(
        fig_dir, f"{region_name_en.lower()}_distance_comparison.png"
    )

    for_latex = exp_config.get("for_latex", False)
    visualize_distance_comparison(
        router.G,
        sample_paths,
        boundaries_dict=sample_bounds,
        save_path=comparison_fig_path,
        for_latex=for_latex,
    )
