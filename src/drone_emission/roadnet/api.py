import os

import osmnx as ox
from visualization.roadnet import visualize_roadnet

from .osm_raodnet import fetch_bikeable_roadnet


def add_roadnet_info(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_name_cn = region_config.get("region_name_cn")
    region_level = region_config["region_level"]
    # boundaries_dict 使用 WGS84 (EPSG:4326) 坐标系
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
