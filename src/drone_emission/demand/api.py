import os

import geopandas as gpd
import numpy as np
import pandas as pd
from haversine import Unit, haversine, haversine_vector
from sklearn.neighbors import BallTree
from utils.coordinates import gcj02_to_wgs84, haversine_distance
from utils.tools import get_epsg_code
from visualization.buildings import visualize_store_coverage

from .cal_area import preprocess_buildings

# Earth radius in kilometers (for Haversine distance)
EARTH_RADIUS_KM = 6371.0


def prepare_store_building_matching(region_info, exp_config):
    """Prepare data for store-building matching.

    Loads store and building data, builds spatial index, and finds nearby buildings
    for each store within delivery radius.

    Returns:
        Tuple of (indices_list, stores_df, buildings_gdf, store_coords)
        - indices_list: List of arrays, each containing building indices near a store
        - stores_df: DataFrame of store data
        - buildings_gdf: GeoDataFrame of buildings with centroid column
        - store_coords: Array of store coordinates in WGS84 (lat, lng)
    """
    region_config = region_info.to_dict()

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

    print(f"\n=== Store-Building Matching Summary ===")
    total_pairs = sum(len(idx) for idx in indices_list)
    stores_with_buildings = sum(1 for idx in indices_list if len(idx) > 0)
    print(f"Total stores: {len(stores_df)}")
    print(f"Stores with nearby buildings: {stores_with_buildings}")
    print(f"Total store-building pairs: {total_pairs}")
    print(f"Avg buildings per store: {total_pairs / len(stores_df):.1f}")

    return indices_list, stores_df, buildings_gdf, store_coords


def generate_od_pairs(region_info, exp_config):
    """
    Generate Origin-Destination pairs by matching stores to nearby buildings.

    For each store (origin), find all buildings (potential destinations)
    within SEARCH_RADIUS_KM using BallTree with Haversine distance.
    """
    region_config = region_info.to_dict()
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    delivery_radius_km = exp_config.get("delivery_radius_km", 5.0)

    # Prepare store-building matching data
    indices_list, stores_df, buildings_gdf, store_coords = (
        prepare_store_building_matching(region_info, exp_config)
    )

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

    large_bounds_dict = region_info.get_stat("enlarged_boundaries")

    for_latex = exp_config.get("for_latex", False)
    visualize_store_coverage(
        buildings_gdf=buildings_gdf,
        store_coord=example_store_coord,
        nearby_indices=indices_list[example_idx],
        city_name=region_name_en,
        city_name_cn=region_config.get("region_name_cn"),
        boundaries_dict=large_bounds_dict,
        original_boundaries_dict=region_config.get("boundaries_dict"),
        radius_km=delivery_radius_km,
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_store_coverage.png"),
        for_latex=for_latex,
    )

    # === Process each store and its nearby buildings ===
    od_params = exp_config.get("od_params", {})
    od_pairs = []
    total_sales = 0
    output_total_sales = 0
    for store_idx, building_indices in enumerate(indices_list):
        if len(building_indices) == 0:
            continue

        store_row = stores_df.iloc[store_idx]
        nearby_buildings = buildings_gdf.iloc[building_indices]
        # Use WGS84 coordinates for store (converted from GCJ02)
        store_coord_wgs84 = store_coords[store_idx]  # (lat, lng)

        tmp_od_pairs = generate_one_store_od_pairs_vectorized(
            nearby_buildings, store_row["month_sales"], store_coord_wgs84, od_params
        )
        od_pairs.extend(tmp_od_pairs)

        total_sales += store_row["month_sales"] * 12  # Annual sales
        output_total_sales += sum([pair["annual_sales"] for pair in tmp_od_pairs])

    print(f"\nTotal store annual sales: {total_sales}")
    print(f"Total output annual sales from OD pairs: {output_total_sales}")

    region_info.set_stat("total_store_annual_sales", total_sales)
    region_info.set_stat("total_odpair_annual_sales", output_total_sales)
    region_info.set_stat("odpair_count", len(od_pairs))

    od_df = pd.DataFrame(od_pairs)
    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    od_df.to_csv(od_save_path, index=False)
    print(f"Saved OD pairs to {od_save_path}")


def generate_one_store_od_pairs(
    nearby_buildings, month_sales, store_coord_wgs84, od_params
):
    """Generate OD pairs for one store.

    Args:
        nearby_buildings: GeoDataFrame of nearby buildings
        month_sales: Monthly sales of the store
        store_coord_wgs84: Store coordinates in WGS84 (lat, lng)
        od_params: Dict with g_param, min_distance_km, min_sales
    """
    impedance_type = od_params.get("impedance_type", "power_law")
    min_sales = od_params.get("min_sales", 5)

    total_demand = month_sales * 12  # Annual demand
    store_lat, store_lng = store_coord_wgs84

    ratios = {}
    distances = {}
    for idx, building in nearby_buildings.iterrows():
        distance_km = haversine(
            (store_lat, store_lng),
            (building["centroid"].y, building["centroid"].x),
            unit=Unit.KILOMETERS,
        )
        distances[idx] = distance_km
        volume = building.get("volume_m3")

        if impedance_type == "tanner":
            alpha = od_params.get("alpha", 1.7)
            tanner_a = od_params.get("tanner_a", 1.2)
            tanner_b = od_params.get("tanner_b", 0.8)
            tmp_ratio = (volume ** alpha) * (distance_km ** tanner_a) * np.exp(-tanner_b * distance_km)
        else:  # power_law (default, backward compatible)
            g_param = od_params.get("g_param", [1, 1, 2])
            min_distance_km = od_params.get("min_distance_km", 0.05)
            tmp_ratio = (
                g_param[0]
                * volume ** g_param[1]
                / (distance_km + min_distance_km) ** g_param[2]
            )
        ratios[idx] = tmp_ratio

    sales_dict = {}
    total_ratio = sum(ratios.values())
    for k, v in ratios.items():
        tmp_sales = int((v / total_ratio) * total_demand)
        if tmp_sales >= min_sales:
            sales_dict[k] = tmp_sales

    od_pairs = []
    for b_idx, sales in sales_dict.items():
        od_pairs.append(
            {
                "from_lat": store_lat,
                "from_lng": store_lng,
                "to_lat": nearby_buildings.loc[b_idx]["centroid"].y,
                "to_lng": nearby_buildings.loc[b_idx]["centroid"].x,
                "distance_km": distances[b_idx],
                "annual_sales": sales,
            }
        )

    return od_pairs


def generate_one_store_od_pairs_vectorized(
    nearby_buildings, month_sales, store_coord_wgs84, od_params
):
    """Vectorized version of generate_one_store_od_pairs.

    Uses haversine_vector and numpy operations for faster computation.

    Args:
        nearby_buildings: GeoDataFrame of nearby buildings
        month_sales: Monthly sales of the store
        store_coord_wgs84: Store coordinates in WGS84 (lat, lng)
        od_params: Dict with g_param, min_distance_km, min_sales
    """
    impedance_type = od_params.get("impedance_type", "power_law")
    min_sales = od_params.get("min_sales", 5)

    total_demand = month_sales * 12  # Annual demand
    store_lat, store_lng = store_coord_wgs84

    n_buildings = len(nearby_buildings)
    if n_buildings == 0:
        return []

    # Extract building coordinates as numpy array
    building_lats = np.array([c.y for c in nearby_buildings["centroid"]])
    building_lngs = np.array([c.x for c in nearby_buildings["centroid"]])
    building_coords = np.column_stack([building_lats, building_lngs])

    # Store coordinates repeated for vectorized distance calculation
    store_coords_arr = np.array([[store_lat, store_lng]] * n_buildings)

    # Vectorized haversine distance calculation
    distances_km = haversine_vector(
        store_coords_arr, building_coords, unit=Unit.KILOMETERS
    )

    # Extract volumes as numpy array
    volumes = nearby_buildings["volume_m3"].values

    # Vectorized ratio calculation
    if impedance_type == "tanner":
        alpha = od_params.get("alpha", 1.7)
        tanner_a = od_params.get("tanner_a", 1.2)
        tanner_b = od_params.get("tanner_b", 0.8)
        ratios = (volumes ** alpha) * (distances_km ** tanner_a) * np.exp(-tanner_b * distances_km)
    else:  # power_law (default, backward compatible)
        g_param = od_params.get("g_param", [1, 1, 2])
        min_distance_km = od_params.get("min_distance_km", 0.05)
        ratios = (
            g_param[0]
            * (volumes ** g_param[1])
            / ((distances_km + min_distance_km) ** g_param[2])
        )

    # Normalize and calculate sales
    total_ratio = ratios.sum()
    if total_ratio == 0:
        return []

    sales = (ratios / total_ratio * total_demand).astype(int)

    # Filter by minimum sales threshold
    mask = sales >= min_sales
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        return []

    # Build result using vectorized operations
    od_pairs = [
        {
            "from_lat": store_lat,
            "from_lng": store_lng,
            "to_lat": building_lats[i],
            "to_lng": building_lngs[i],
            "distance_km": distances_km[i],
            "annual_sales": sales[i],
        }
        for i in valid_indices
    ]

    return od_pairs
