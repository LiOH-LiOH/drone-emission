import os
import csv

import geopandas as gpd
import numpy as np
import pandas as pd
from haversine import Unit, haversine, haversine_vector
from sklearn.neighbors import BallTree
from drone_emission.utils.coordinates import gcj02_to_wgs84, haversine_distance
from drone_emission.utils.tools import get_epsg_from_bounds
from drone_emission.utils.atomic import atomic_open

from .cal_area import preprocess_buildings

# Earth radius in kilometers (for Haversine distance)
EARTH_RADIUS_KM = 6371.0
STORE_QUERY_BATCH_SIZE = 128
OD_COLUMNS = (
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    "distance_km",
    "annual_sales",
)


class _BatchedRadiusQueries:
    """Re-iterable radius-query results without retaining every candidate index."""

    def __init__(self, tree, coordinates, radius, batch_size=STORE_QUERY_BATCH_SIZE):
        self.tree = tree
        self.coordinates = coordinates
        self.radius = radius
        self.batch_size = batch_size

    def __len__(self):
        return len(self.coordinates)

    def __iter__(self):
        for start in range(0, len(self.coordinates), self.batch_size):
            batch = self.tree.query_radius(
                self.coordinates[start : start + self.batch_size], r=self.radius
            )
            yield from batch


def _allocate_integer_orders(probabilities, total_orders, stable_keys=None):
    """Allocate integer orders with the largest-remainder method."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError("probabilities must be finite and positive")
    if not isinstance(total_orders, (int, np.integer)) or total_orders <= 0:
        raise ValueError("total_orders must be a positive integer")

    probabilities = probabilities / probabilities.sum()
    quotas = probabilities * total_orders
    allocations = np.floor(quotas).astype(np.int64)
    remaining = int(total_orders - allocations.sum())
    if remaining == 0:
        return allocations

    fractions = quotas - allocations
    threshold_index = len(fractions) - remaining
    threshold = np.partition(fractions, threshold_index)[threshold_index]
    above_threshold = fractions > threshold
    allocations[above_threshold] += 1

    tie_slots = remaining - int(above_threshold.sum())
    if tie_slots:
        tied = np.flatnonzero(fractions == threshold)
        if stable_keys is None:
            stable_keys = np.arange(len(fractions), dtype=np.int64)
        stable_keys = np.asarray(stable_keys)
        if tie_slots < len(tied):
            selected = np.argpartition(stable_keys[tied], tie_slots - 1)[:tie_slots]
            tied = tied[selected]
        allocations[tied] += 1

    if int(allocations.sum()) != total_orders:
        raise RuntimeError("integer order allocation failed to conserve demand")
    return allocations


def _monthly_orders(month_sales):
    """Return a valid positive integer monthly demand, or ``None``."""
    try:
        value = float(month_sales)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0 or value != np.floor(value):
        return None
    return int(value)


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

    region_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    base = region_name_en.lower()

    # Prefer .gpkg, fall back to legacy .geojson
    raw_gpkg = os.path.join(region_dir, f"{base}_buildings.gpkg")
    raw_geojson = os.path.join(region_dir, f"{base}_buildings.geojson")
    raw_path = raw_gpkg if os.path.exists(raw_gpkg) else raw_geojson

    prep_gpkg = os.path.join(region_dir, f"{base}_buildings_preprocessed.gpkg")
    prep_geojson = os.path.join(region_dir, f"{base}_buildings_preprocessed.geojson")
    building_path = prep_gpkg if os.path.exists(prep_gpkg) else prep_geojson

    building_path, buildings_gdf = preprocess_buildings(raw_path, building_path)
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

    print(f"Loaded {len(buildings_gdf)} buildings from {building_path}")

    # === Extract building centroids ===
    # Convert to projected CRS for accurate centroid calculation, then back to WGS84
    if buildings_gdf.crs and buildings_gdf.crs.to_epsg() != 4326:
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    epsg_code = get_epsg_from_bounds(region_info.boundaries_dict)

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
    indices_list = _BatchedRadiusQueries(tree, store_coords_rad, radius_rad)

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
    # Prepare store-building matching data
    indices_list, stores_df, buildings_gdf, store_coords = (
        prepare_store_building_matching(region_info, exp_config)
    )

    # === Process each store and its nearby buildings ===
    od_params = exp_config.get("od_params", {})
    total_sales = 0
    allocatable_sales = 0
    unallocatable_sales = 0
    output_total_sales = 0
    invalid_sales_store_count = 0
    allocatable_store_count = 0
    unallocatable_store_count = 0
    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    odpair_count = 0
    with atomic_open(od_save_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OD_COLUMNS)
        writer.writeheader()
        for store_idx, building_indices in enumerate(indices_list):
            store_row = stores_df.iloc[store_idx]
            monthly_orders = _monthly_orders(store_row["month_sales"])
            if monthly_orders is None:
                invalid_sales_store_count += 1
                continue

            annual_orders = monthly_orders * 12
            total_sales += annual_orders
            if len(building_indices) == 0:
                unallocatable_store_count += 1
                unallocatable_sales += annual_orders
                continue

            nearby_buildings = buildings_gdf.iloc[building_indices]
            store_coord_wgs84 = store_coords[store_idx]
            tmp_od_pairs = generate_one_store_od_pairs_vectorized(
                nearby_buildings, monthly_orders, store_coord_wgs84, od_params
            )
            if not tmp_od_pairs:
                unallocatable_store_count += 1
                unallocatable_sales += annual_orders
                continue

            store_output_sales = sum(pair["annual_sales"] for pair in tmp_od_pairs)
            if store_output_sales != annual_orders:
                raise RuntimeError(
                    "OD allocation failed to conserve store demand: "
                    f"store_index={store_idx}, expected={annual_orders}, "
                    f"actual={store_output_sales}"
                )
            writer.writerows(tmp_od_pairs)
            odpair_count += len(tmp_od_pairs)
            allocatable_store_count += 1
            allocatable_sales += annual_orders
            output_total_sales += store_output_sales
        allocation_error = output_total_sales - allocatable_sales
        if total_sales != allocatable_sales + unallocatable_sales:
            raise RuntimeError("store demand accounting is inconsistent")
        if allocation_error != 0:
            raise RuntimeError("regional OD allocation failed to conserve demand")

    print(f"\nTotal store annual sales: {total_sales}")
    print(f"Total output annual sales from OD pairs: {output_total_sales}")

    region_info.set_stat("total_store_annual_sales", total_sales)
    region_info.set_stat("invalid_sales_store_count", invalid_sales_store_count)
    region_info.set_stat("allocatable_store_count", allocatable_store_count)
    region_info.set_stat("unallocatable_store_count", unallocatable_store_count)
    region_info.set_stat("allocatable_store_annual_sales", allocatable_sales)
    region_info.set_stat("unallocatable_store_annual_sales", unallocatable_sales)
    region_info.set_stat("od_allocation_annual_sales_error", allocation_error)
    region_info.set_stat("total_odpair_annual_sales", output_total_sales)
    region_info.set_stat("odpair_count", odpair_count)
    print(f"Saved OD pairs to {od_save_path}")


def generate_one_store_od_pairs(
    nearby_buildings, month_sales, store_coord_wgs84, od_params
):
    """Generate OD pairs for one store.

    Args:
        nearby_buildings: GeoDataFrame of nearby buildings
        month_sales: Monthly sales of the store
        store_coord_wgs84: Store coordinates in WGS84 (lat, lng)
        od_params: Dict with gravity-model parameters
    """
    impedance_type = od_params.get("impedance_type", "power_law")

    monthly_orders = _monthly_orders(month_sales)
    if monthly_orders is None:
        return []
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

    valid_keys = [
        key
        for key, ratio in ratios.items()
        if np.isfinite(ratio) and ratio > 0 and np.isfinite(distances[key])
    ]
    if not valid_keys:
        return []
    valid_ratios = np.array([ratios[key] for key in valid_keys], dtype=float)
    total_ratio = valid_ratios.sum()
    if not np.isfinite(total_ratio) or total_ratio <= 0:
        return []
    valid_ratios /= valid_ratios.sum()
    monthly_sales = _allocate_integer_orders(
        valid_ratios,
        monthly_orders,
        stable_keys=np.asarray(valid_keys),
    )

    od_pairs = []
    for b_idx, sales in zip(valid_keys, monthly_sales):
        if sales == 0:
            continue
        od_pairs.append(
            {
                "from_lat": store_lat,
                "from_lng": store_lng,
                "to_lat": nearby_buildings.loc[b_idx]["centroid"].y,
                "to_lng": nearby_buildings.loc[b_idx]["centroid"].x,
                "distance_km": distances[b_idx],
                "annual_sales": int(sales) * 12,
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
        od_params: Dict with gravity-model parameters
    """
    impedance_type = od_params.get("impedance_type", "power_law")

    monthly_orders = _monthly_orders(month_sales)
    if monthly_orders is None:
        return []
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
    with np.errstate(all="ignore"):
        if impedance_type == "tanner":
            alpha = od_params.get("alpha", 1.7)
            tanner_a = od_params.get("tanner_a", 1.2)
            tanner_b = od_params.get("tanner_b", 0.8)
            ratios = (
                (volumes**alpha)
                * (distances_km**tanner_a)
                * np.exp(-tanner_b * distances_km)
            )
        else:  # power_law (default, backward compatible)
            g_param = od_params.get("g_param", [1, 1, 2])
            min_distance_km = od_params.get("min_distance_km", 0.05)
            ratios = (
                g_param[0]
                * (volumes ** g_param[1])
                / ((distances_km + min_distance_km) ** g_param[2])
            )

    valid = (
        np.isfinite(building_lats)
        & np.isfinite(building_lngs)
        & np.isfinite(distances_km)
        & np.isfinite(volumes)
        & (volumes > 0)
        & np.isfinite(ratios)
        & (ratios > 0)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        return []

    valid_ratios = ratios[valid_indices]
    total_ratio = valid_ratios.sum()
    if not np.isfinite(total_ratio) or total_ratio <= 0:
        return []
    monthly_sales = _allocate_integer_orders(
        valid_ratios / total_ratio,
        monthly_orders,
        stable_keys=nearby_buildings.index.to_numpy()[valid_indices],
    )

    # Build result using vectorized operations
    od_pairs = [
        {
            "from_lat": store_lat,
            "from_lng": store_lng,
            "to_lat": building_lats[i],
            "to_lng": building_lngs[i],
            "distance_km": distances_km[i],
            "annual_sales": int(monthly_sales[position]) * 12,
        }
        for position, i in enumerate(valid_indices)
        if monthly_sales[position] > 0
    ]

    return od_pairs
