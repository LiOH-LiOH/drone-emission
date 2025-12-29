import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .cal_area import preprocess_buildings

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0
# Search radius in kilometers
SEARCH_RADIUS_KM = 5.0


def generate_od_pairs(region_config, exp_config):
    """
    Generate Origin-Destination pairs by matching stores to nearby buildings.

    For each store (origin), find all buildings (potential destinations)
    within SEARCH_RADIUS_KM using BallTree with Haversine distance.
    """
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]

    raw_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_buildings.geojson",
    )

    building_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_buildings_preprocessed.geojson",
    )

    preprocess_buildings(raw_path, building_path)
    print("Preprocessing of building data completed.")

    # === Load store data ===
    store_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_sales.csv",
    )
    stores_df = pd.read_csv(store_path)
    print(f"Loaded {len(stores_df)} stores from {store_path}")

    # === Load building data ===
    buildings_gdf = gpd.read_file(building_path)
    print(f"Loaded {len(buildings_gdf)} buildings from {building_path}")

    # === Extract building centroids (convert to WGS84 if needed) ===
    if buildings_gdf.crs and buildings_gdf.crs.to_epsg() != 4326:
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    # Get centroids
    buildings_gdf["centroid"] = buildings_gdf.geometry.centroid
    building_coords = np.array(
        [[geom.y, geom.x] for geom in buildings_gdf["centroid"]]
    )  # [lat, lng]

    # === Build BallTree with Haversine metric ===
    building_coords_rad = np.radians(building_coords)
    tree = BallTree(building_coords_rad, metric="haversine")

    # === Query for each store ===
    store_coords = stores_df[["lat", "lng"]].values  # [lat, lng]
    store_coords_rad = np.radians(store_coords)

    # Query radius in radians
    radius_rad = SEARCH_RADIUS_KM / EARTH_RADIUS_KM

    # Find all buildings within radius for each store
    indices_list = tree.query_radius(store_coords_rad, r=radius_rad)

    print(f"\n=== OD Generation Summary ===")
    total_pairs = sum(len(idx) for idx in indices_list)
    stores_with_buildings = sum(1 for idx in indices_list if len(idx) > 0)
    print(f"Total stores: {len(stores_df)}")
    print(f"Stores with nearby buildings: {stores_with_buildings}")
    print(f"Total OD pairs: {total_pairs}")
    print(f"Avg buildings per store: {total_pairs / len(stores_df):.1f}")

    # === Process each store and its nearby buildings ===
    for store_idx, building_indices in enumerate(indices_list):
        if len(building_indices) == 0:
            continue

        store_row = stores_df.iloc[store_idx]
        nearby_buildings = buildings_gdf.iloc[building_indices]

        # TODO: Process store and nearby buildings
        # - Calculate distances
        # - Distribute demand based on building attributes (height, area, etc.)
        # - Generate OD matrix entries
        pass

    return indices_list
