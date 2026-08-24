import fcntl
import os
import subprocess

import geopandas as gpd
import numpy as np
from drone_emission.utils.boundaries import resolve_enlarged_boundaries

from .load_gba import get_gba_tile_infos, load_buildings_from_gba


def _convert_geojson_to_gpkg(geojson_path: str) -> str:
    """Convert a GeoJSON file to GeoPackage for efficient spatial queries.

    Returns the .gpkg path. Skips if already converted.
    """
    gpkg_path = os.path.splitext(geojson_path)[0] + ".gpkg"
    if os.path.exists(gpkg_path):
        return gpkg_path

    import fiona
    import time

    print(f"Converting {os.path.basename(geojson_path)} to GeoPackage...")
    t0 = time.time()
    tmp_path = gpkg_path + ".tmp"
    with fiona.open(geojson_path) as src:
        schema = src.schema.copy()
        crs = src.crs
        n = 0
        with fiona.open(tmp_path, "w", driver="GPKG", schema=schema, crs=crs) as dst:
            for feat in src:
                dst.write(feat)
                n += 1
                if n % 500_000 == 0:
                    print(f"  ... {n:,} features")
    os.rename(tmp_path, gpkg_path)
    elapsed = time.time() - t0
    print(f"  Converted {n:,} features in {elapsed:.0f}s")
    return gpkg_path


def _ensure_tile_available(tile_name: str, gba_region: str, gba_data_dir: str):
    """Download and convert a single GBA tile if needed.

    Returns the .gpkg path, or None if the tile doesn't exist in GBA.
    """
    tile_dir = os.path.join(gba_data_dir, "LoD1_GeoJSON", gba_region)
    geojson_path = os.path.join(tile_dir, f"{tile_name}.geojson")
    gpkg_path = os.path.join(tile_dir, f"{tile_name}.gpkg")

    # Already converted
    if os.path.exists(gpkg_path):
        return gpkg_path

    # Download if missing
    if not os.path.exists(geojson_path):
        lock_path = os.path.join(gba_data_dir, f".lock_{gba_region}_{tile_name}")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if os.path.exists(gpkg_path):
                    return gpkg_path
                if not os.path.exists(geojson_path):
                    print(f"  Downloading GBA tile {tile_name} ({gba_region})...")
                    script_path = os.path.join(
                        os.path.dirname(__file__), "download_gba.sh"
                    )
                    result = subprocess.run(
                        ["bash", script_path, tile_name, gba_region, gba_data_dir],
                        check=False,
                    )
                    if result.returncode != 0 or not os.path.exists(geojson_path):
                        print(f"  WARNING: GBA tile {tile_name} not available, skipping")
                        return None
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    # Convert to GeoPackage
    lock_path = os.path.join(gba_data_dir, f".lock_convert_{gba_region}_{tile_name}")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            gpkg_path = _convert_geojson_to_gpkg(geojson_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return gpkg_path


def _resolve_gba_data_paths(boundaries_dict: dict, exp_config: dict) -> list[str]:
    """Resolve GBA data paths for a region, downloading and converting if needed.

    Returns a list of .gpkg paths covering all tiles that overlap the bbox.
    """
    gba_data_dir = exp_config.get("gba_data_dir")

    if gba_data_dir:
        tile_infos = get_gba_tile_infos(boundaries_dict)
        print(f"GBA tiles needed: {len(tile_infos)}")
        for tn, rg in tile_infos:
            print(f"  {tn} ({rg})")

        paths = []
        for tile_name, gba_region in tile_infos:
            result = _ensure_tile_available(tile_name, gba_region, gba_data_dir)
            if result is not None:
                paths.append(result)

        if not paths:
            raise FileNotFoundError(
                f"No GBA tiles found for bbox {boundaries_dict}. "
                f"Checked tiles: {tile_infos}"
            )
        return paths

    # Fallback to legacy gba_data_path
    gba_data_path = exp_config.get("gba_data_path")
    if not gba_data_path:
        raise ValueError(
            "Neither gba_data_dir nor gba_data_path set in experiment.toml"
        )
    return [gba_data_path]


def load_buildings(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict", None)
    if not boundaries_dict:
        print(
            f"No boundaries_dict found for {region_name_en}. Skipping building fetch."
        )
        return

    large_bounds_dict = resolve_enlarged_boundaries(region_info, exp_config)

    # Enlarge bbox slightly to ensure full coverage
    bbox_4326 = (
        large_bounds_dict["min_lng"],
        large_bounds_dict["min_lat"],
        large_bounds_dict["max_lng"],
        large_bounds_dict["max_lat"],
    )

    region_save_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(region_save_dir):
        os.makedirs(region_save_dir)

    base_name = f"{region_name_en.lower()}_buildings"
    gpkg_file = os.path.join(region_save_dir, f"{base_name}.gpkg")
    geojson_file = os.path.join(region_save_dir, f"{base_name}.geojson")

    # Prefer .gpkg, fall back to legacy .geojson
    buildings_file = gpkg_file if os.path.exists(gpkg_file) else geojson_file

    # Fetch buildings if requested or no file exists
    if exp_config.get("fetch_buildings", False) or not os.path.exists(buildings_file):
        gba_data_paths = _resolve_gba_data_paths(boundaries_dict, exp_config)

        # load_buildings_from_gba now saves as .gpkg
        buildings_file = load_buildings_from_gba(
            gba_data_paths=gba_data_paths,
            bbox_4326=bbox_4326,
            save_path=gpkg_file,
        )

    # Load buildings from file
    print(f"Reading {buildings_file}...")
    try:
        gdf = gpd.read_file(buildings_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if gdf.empty:
        print("GeoJSON file is empty.")
        return

    print(f"Loaded {len(gdf)} buildings.")

    region_info.set_stat("building_count", len(gdf))
    region_info.set_stat("building_height_mean", gdf["height"].mean())
    region_info.set_stat("building_height_max", gdf["height"].max())
    region_info.set_stat("building_height_min", gdf["height"].min())

    # Calculate and save total building volume and area if not present
    if "area_m2" not in gdf.columns:
        # Calculate area from geometry (need to project to a metric CRS first)
        gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator for area calculation
        gdf["area_m2"] = gdf_projected.geometry.area
    if "volume_m3" not in gdf.columns:
        gdf["volume_m3"] = gdf["area_m2"] * gdf["height"]

    region_info.set_stat("building_volume_total", gdf["volume_m3"].sum())
    region_info.set_stat("building_area_total", gdf["area_m2"].sum())

    print(f"Total building volume: {region_info.get_stat('building_volume_total')}")
