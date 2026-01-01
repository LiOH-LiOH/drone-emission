import os

import osmnx as ox
import pandas as pd
from visualization.roadnet import visualize_roadnet

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

    visualize_roadnet(
        G,
        city_name_cn=region_name_cn,
        boundaries_dict=large_bounds_dict,
        original_boundaries_dict=boundaries_dict,
        save_path=roadnet_fig_path,
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

    # Compute road network distances for all OD pairs
    import numpy as np

    origins = od_df[["from_lat", "from_lng"]].values
    destinations = od_df[["to_lat", "to_lng"]].values

    road_distances = router.compute_od_distances_batch(origins, destinations)

    # Add road distance to dataframe
    od_df["road_distance_m"] = road_distances

    # Save updated OD pairs
    od_df.to_csv(od_save_path, index=False)
    print(f"Updated OD pairs with road distances saved to {od_save_path}")

    # Statistics
    valid_mask = road_distances >= 0
    if valid_mask.sum() > 0:
        avg_road_dist = road_distances[valid_mask].mean()
        avg_straight_dist = od_df.loc[valid_mask, "distance_km"].mean() * 1000
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
