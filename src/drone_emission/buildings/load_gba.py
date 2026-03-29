import math
import os

import geopandas as gpd
from shapely.geometry import box

from utils.coordinates import wgs84_to_3857


def get_gba_tile_info(boundaries_dict: dict) -> tuple[str, str]:
    """Compute (tile_name, gba_region) from a region's bounding box.

    Uses center of bounding box to determine the 5x5 degree GBA tile.
    Region is 'asiawest' for lng < 110E, 'asiaeast' for lng >= 110E.

    Args:
        boundaries_dict: Dict with min_lat, max_lat, min_lng, max_lng.

    Returns:
        Tuple of (tile_name, gba_region), e.g. ("e110_n25_e115_n20", "asiaeast").
    """
    center_lng = (boundaries_dict["min_lng"] + boundaries_dict["max_lng"]) / 2
    center_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2

    # Snap to 5-degree grid
    min_lng = int(math.floor(center_lng / 5) * 5)
    max_lng = min_lng + 5
    min_lat = int(math.floor(center_lat / 5) * 5)
    max_lat = min_lat + 5

    tile_name = f"e{min_lng:03d}_n{max_lat:02d}_e{max_lng:03d}_n{min_lat:02d}"
    gba_region = "asiawest" if center_lng < 110 else "asiaeast"

    return tile_name, gba_region


def _resolve_gba_source(gba_data_path: str) -> str:
    """Return the .gpkg path if it exists, otherwise fall back to the original."""
    gpkg_path = os.path.splitext(gba_data_path)[0] + ".gpkg"
    if os.path.exists(gpkg_path):
        return gpkg_path
    return gba_data_path


def load_buildings_from_gba(
    gba_data_path: str,
    bbox_4326: tuple,
    save_path: str,
) -> str:
    """
    Load buildings from pre-downloaded GBA (GlobalBuildingAtlas) data,
    filtering to the given bounding box.

    Prefers GeoPackage (.gpkg) over GeoJSON for efficient spatial queries.
    GBA data is in EPSG:3857. The bbox is converted from WGS84 before filtering.

    Args:
        gba_data_path: Path to the full GBA LoD1 file (.geojson or .gpkg).
        bbox_4326: Bounding box in WGS84 (min_lng, min_lat, max_lng, max_lat).
        save_path: Path to save the filtered region GeoPackage.

    Returns:
        Path to the saved file.
    """
    source_path = _resolve_gba_source(gba_data_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"GBA data not found at {source_path}")

    # Convert WGS84 bbox to EPSG:3857 for spatial filtering
    min_x, min_y = wgs84_to_3857(bbox_4326[0], bbox_4326[1])
    max_x, max_y = wgs84_to_3857(bbox_4326[2], bbox_4326[3])
    filter_box = box(min_x, min_y, max_x, max_y)

    print(f"Loading GBA buildings from {source_path}")
    print(f"  Bbox (WGS84): {bbox_4326}")
    print(f"  Bbox (3857):  ({min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f})")

    # Use bbox parameter for efficient spatial filtering during read
    gdf = gpd.read_file(source_path, bbox=filter_box)

    if gdf.empty:
        print("No buildings found in the specified area.")
        return save_path

    print(f"  Loaded {len(gdf)} buildings")

    # Save filtered result as GeoPackage for fast subsequent reads
    save_path = os.path.splitext(save_path)[0] + ".gpkg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gdf.to_file(save_path, driver="GPKG")

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  Saved to {save_path} ({file_size_mb:.1f} MB)")

    return save_path
