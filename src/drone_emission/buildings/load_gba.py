import math
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from drone_emission.utils.coordinates import wgs84_to_3857
from drone_emission.utils.atomic import atomic_replace


def _atomic_gpkg(frame, save_path: str) -> None:
    target = Path(save_path)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.gpkg")
    try:
        frame.to_file(temporary, layer=target.stem, driver="GPKG")
        atomic_replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def get_gba_tile_infos(boundaries_dict: dict) -> list[tuple[str, str]]:
    """Compute all (tile_name, gba_region) pairs that overlap a bounding box.

    GBA data is stored in 5x5 degree tiles. A city bbox may span multiple tiles.
    Region is 'asiawest' for tile lng < 110E, 'asiaeast' for tile lng >= 110E.

    Args:
        boundaries_dict: Dict with min_lat, max_lat, min_lng, max_lng.

    Returns:
        List of (tile_name, gba_region) tuples covering the bbox.
    """
    # Compute range of 5-degree grid cells overlapping the bbox
    lng_min_tile = int(math.floor(boundaries_dict["min_lng"] / 5)) * 5
    lng_max_tile = int(math.floor(boundaries_dict["max_lng"] / 5)) * 5
    lat_min_tile = int(math.floor(boundaries_dict["min_lat"] / 5)) * 5
    lat_max_tile = int(math.floor(boundaries_dict["max_lat"] / 5)) * 5

    tiles = []
    for tile_min_lng in range(lng_min_tile, lng_max_tile + 1, 5):
        for tile_min_lat in range(lat_min_tile, lat_max_tile + 1, 5):
            tile_max_lng = tile_min_lng + 5
            tile_max_lat = tile_min_lat + 5
            tile_name = f"e{tile_min_lng:03d}_n{tile_max_lat:02d}_e{tile_max_lng:03d}_n{tile_min_lat:02d}"
            gba_region = "asiawest" if tile_min_lng < 110 else "asiaeast"
            tiles.append((tile_name, gba_region))

    return tiles


def get_gba_tile_info(boundaries_dict: dict) -> tuple[str, str]:
    """Deprecated: returns only the first tile. Use get_gba_tile_infos()."""
    return get_gba_tile_infos(boundaries_dict)[0]


def _resolve_gba_source(gba_data_path: str) -> str:
    """Return the .gpkg path if it exists, otherwise fall back to the original."""
    gpkg_path = os.path.splitext(gba_data_path)[0] + ".gpkg"
    if os.path.exists(gpkg_path):
        return gpkg_path
    return gba_data_path


def load_buildings_from_gba(
    gba_data_paths: list[str],
    bbox_4326: tuple,
    save_path: str,
) -> str:
    """
    Load buildings from pre-downloaded GBA (GlobalBuildingAtlas) data,
    filtering to the given bounding box.  Reads from multiple tile files
    when a city spans more than one 5-degree tile.

    Prefers GeoPackage (.gpkg) over GeoJSON for efficient spatial queries.
    GBA data is in EPSG:3857. The bbox is converted from WGS84 before filtering.

    Args:
        gba_data_paths: Paths to GBA LoD1 tile files (.geojson or .gpkg).
        bbox_4326: Bounding box in WGS84 (min_lng, min_lat, max_lng, max_lat).
        save_path: Path to save the filtered region GeoPackage.

    Returns:
        Path to the saved file.
    """
    # Convert WGS84 bbox to EPSG:3857 for spatial filtering (once for all tiles)
    min_x, min_y = wgs84_to_3857(bbox_4326[0], bbox_4326[1])
    max_x, max_y = wgs84_to_3857(bbox_4326[2], bbox_4326[3])
    filter_box = box(min_x, min_y, max_x, max_y)

    print(f"  Bbox (WGS84): {bbox_4326}")
    print(f"  Bbox (3857):  ({min_x:.0f}, {min_y:.0f}, {max_x:.0f}, {max_y:.0f})")

    gdfs = []
    total_loaded = 0
    for path in gba_data_paths:
        source_path = _resolve_gba_source(path)
        if not os.path.exists(source_path):
            print(f"  WARNING: GBA tile not found at {source_path}, skipping")
            continue

        print(f"  Reading {os.path.basename(source_path)}...")
        gdf = gpd.read_file(source_path, bbox=filter_box)
        if not gdf.empty:
            gdfs.append(gdf)
            total_loaded += len(gdf)
            print(f"    {len(gdf)} buildings")
        else:
            print(f"    0 buildings (no data in bbox)")

    if not gdfs:
        print("  No buildings found in the specified area from any tile.")
        save_path = os.path.splitext(save_path)[0] + ".gpkg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _atomic_gpkg(gpd.GeoDataFrame(), save_path)
        return save_path

    # Concatenate results from all tiles
    if len(gdfs) == 1:
        gdf = gdfs[0]
    else:
        gdf = pd.concat(gdfs, ignore_index=True)
        before = len(gdf)
        gdf = gdf.drop_duplicates(subset="geometry")
        dupes = before - len(gdf)
        if dupes > 0:
            print(f"  Removed {dupes} duplicate buildings at tile edges")

    print(f"  Loaded {len(gdf)} buildings total from {len(gdfs)} tile(s)")

    # Save filtered result as GeoPackage for fast subsequent reads
    save_path = os.path.splitext(save_path)[0] + ".gpkg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _atomic_gpkg(gdf, save_path)

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  Saved to {save_path} ({file_size_mb:.1f} MB)")

    return save_path
