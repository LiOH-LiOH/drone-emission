import os

import geopandas as gpd
from shapely.geometry import box

from utils.coordinates import wgs84_to_3857


def load_buildings_from_gba(
    gba_data_path: str,
    bbox_4326: tuple,
    save_path: str,
) -> str:
    """
    Load buildings from pre-downloaded GBA (GlobalBuildingAtlas) GeoJSON,
    filtering to the given bounding box.

    GBA data is in EPSG:3857. The bbox is converted from WGS84 before filtering.

    Args:
        gba_data_path: Path to the full GBA LoD1 GeoJSON file.
        bbox_4326: Bounding box in WGS84 (min_lng, min_lat, max_lng, max_lat).
        save_path: Path to save the filtered region GeoJSON.

    Returns:
        Path to the saved GeoJSON file.
    """
    if not os.path.exists(gba_data_path):
        raise FileNotFoundError(f"GBA data not found at {gba_data_path}")

    # Convert WGS84 bbox to EPSG:3857 for spatial filtering
    min_x, min_y = wgs84_to_3857(bbox_4326[0], bbox_4326[1])
    max_x, max_y = wgs84_to_3857(bbox_4326[2], bbox_4326[3])
    filter_box = box(min_x, min_y, max_x, max_y)

    print(f"Loading GBA buildings from {gba_data_path}")
    print(f"  Bbox (WGS84): {bbox_4326}")
    print(f"  Bbox (3857):  ({min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f})")

    # Use bbox parameter for efficient spatial filtering during read
    gdf = gpd.read_file(gba_data_path, bbox=filter_box)

    if gdf.empty:
        print("No buildings found in the specified area.")
        return save_path

    print(f"  Loaded {len(gdf)} buildings")

    # Save filtered result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gdf.to_file(save_path, driver="GeoJSON")

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  Saved to {save_path} ({file_size_mb:.1f} MB)")

    return save_path
