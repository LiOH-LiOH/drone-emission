import os

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from utils.coordinates import gcj02_to_wgs84
from utils.tools import get_epsg_code
from visualization.buildings import visualize_store_coverage

from .cal_area import preprocess_buildings

# Earth radius in kilometers (for Haversine distance)
EARTH_RADIUS_KM = 6371.0


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

    # === Extract building centroids ===
    # Convert to projected CRS for accurate centroid calculation, then back to WGS84
    if buildings_gdf.crs and buildings_gdf.crs.to_epsg() != 4326:
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    epsg_code = get_epsg_code(region_name_en.lower())

    # Use UTM zone for accurate centroid calculation (zone 50N for Shenzhen area)
    buildings_projected = buildings_gdf.to_crs(epsg=epsg_code)
    centroids_projected = buildings_projected.geometry.centroid
    # Convert centroids back to WGS84
    centroids_gdf = gpd.GeoSeries(centroids_projected, crs=f"EPSG:{epsg_code}").to_crs(
        epsg=4326
    )
    buildings_gdf["centroid"] = centroids_gdf

    building_coords = np.array(
        [[geom.y, geom.x] for geom in buildings_gdf["centroid"]]
    )  # [lat, lng]

    # === Build BallTree with Haversine metric ===
    building_coords_rad = np.radians(building_coords)
    tree = BallTree(building_coords_rad, metric="haversine")

    # === Query for each store ===
    # Convert store coords from GCJ02 to WGS84 to match building coords
    store_coords_gcj = stores_df[["lat", "lng"]].values  # [lat, lng] in GCJ02
    store_coords = np.array(
        [gcj02_to_wgs84(lng, lat)[::-1] for lat, lng in store_coords_gcj]
    )
    store_coords_rad = np.radians(store_coords)

    # Query radius in radians (convert from km using Earth radius)
    delivery_radius_km = exp_config.get("delivery_radius_km", 5.0)
    radius_rad = delivery_radius_km / EARTH_RADIUS_KM

    # Find all buildings within radius for each store
    indices_list = tree.query_radius(store_coords_rad, r=radius_rad)

    print(f"\n=== OD Generation Summary ===")
    total_pairs = sum(len(idx) for idx in indices_list)
    stores_with_buildings = sum(1 for idx in indices_list if len(idx) > 0)
    print(f"Total stores: {len(stores_df)}")
    print(f"Stores with nearby buildings: {stores_with_buildings}")
    print(f"Total OD pairs: {total_pairs}")
    print(f"Avg buildings per store: {total_pairs / len(stores_df):.1f}")

    # === Plot store coverage example ===
    # Find a store with many nearby buildings for visualization
    nearby_counts = [len(idx) for idx in indices_list]
    example_idx = np.argmax(nearby_counts)  # Store with most nearby buildings

    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Get store coord in WGS84 (already converted)
    example_store_coord = (
        store_coords[example_idx][1],
        store_coords[example_idx][0],
    )  # (lng, lat)

    # Get boundaries_dict from region_config
    boundaries_dict = region_config.get("boundaries_dict", None)

    # Convert km to degrees for boundary padding
    # Latitude: 1° ≈ 111.32 km (constant)
    # Longitude: 1° ≈ 111.32 × cos(lat) km (varies with latitude)
    center_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2
    km_per_deg_lat = EARTH_RADIUS_KM * np.pi / 180  # ≈ 111.32 km
    km_per_deg_lng = km_per_deg_lat * np.cos(np.radians(center_lat))

    padding_lat = delivery_radius_km / km_per_deg_lat
    padding_lng = delivery_radius_km / km_per_deg_lng

    large_bounds_dict = {
        "min_lat": boundaries_dict["min_lat"] - padding_lat,
        "max_lat": boundaries_dict["max_lat"] + padding_lat,
        "min_lng": boundaries_dict["min_lng"] - padding_lng,
        "max_lng": boundaries_dict["max_lng"] + padding_lng,
    }

    visualize_store_coverage(
        buildings_gdf=buildings_gdf,
        store_coord=example_store_coord,
        nearby_indices=indices_list[example_idx],
        city_name=region_name_en,
        city_name_cn=region_config.get("region_name_cn"),
        boundaries_dict=large_bounds_dict,
        radius_km=delivery_radius_km,
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_store_coverage.png"),
    )

    # === Process each store and its nearby buildings ===
    # for store_idx, building_indices in enumerate(indices_list):
    #     if len(building_indices) == 0:
    #         continue

    #     store_row = stores_df.iloc[store_idx]
    #     nearby_buildings = buildings_gdf.iloc[building_indices]

    #     # TODO: Process store and nearby buildings
    #     # - Calculate distances
    #     # - Distribute demand based on building attributes (height, area, etc.)
    #     # - Generate OD matrix entries
    #     pass

    return indices_list
