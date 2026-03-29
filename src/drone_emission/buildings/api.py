import fcntl
import os
import subprocess

import geopandas as gpd
import numpy as np
from visualization.buildings import visualize_building_heights, visualize_buildings

from .load_gba import get_gba_tile_info, load_buildings_from_gba

# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def enlarge_boundaries(boundaries_dict, enlarge_km):
    center_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2
    km_per_deg_lat = EARTH_RADIUS_KM * np.pi / 180
    km_per_deg_lng = km_per_deg_lat * np.cos(np.radians(center_lat))

    enlarge_lat = enlarge_km / km_per_deg_lat
    enlarge_lng = enlarge_km / km_per_deg_lng

    large_bounds_dict = {
        "min_lat": boundaries_dict["min_lat"] - enlarge_lat,
        "max_lat": boundaries_dict["max_lat"] + enlarge_lat,
        "min_lng": boundaries_dict["min_lng"] - enlarge_lng,
        "max_lng": boundaries_dict["max_lng"] + enlarge_lng,
    }

    return large_bounds_dict


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


def _resolve_gba_data_path(boundaries_dict: dict, exp_config: dict) -> str:
    """Resolve the GBA data path for a region, downloading and converting if needed.

    Prefers GeoPackage (.gpkg) over GeoJSON for efficient spatial queries.
    Prefers gba_data_dir (dynamic tile resolution) over gba_data_path (legacy).
    """
    gba_data_dir = exp_config.get("gba_data_dir")

    if gba_data_dir:
        tile_name, gba_region = get_gba_tile_info(boundaries_dict)
        tile_dir = os.path.join(gba_data_dir, "LoD1_GeoJSON", gba_region)
        geojson_path = os.path.join(tile_dir, f"{tile_name}.geojson")
        gpkg_path = os.path.join(tile_dir, f"{tile_name}.gpkg")
        print(f"GBA tile: {tile_name} (region: {gba_region})")

        # If .gpkg already exists, use it directly
        if os.path.exists(gpkg_path):
            return gpkg_path

        if not os.path.exists(geojson_path):
            # Use file lock to prevent parallel processes from downloading the same tile
            lock_path = os.path.join(gba_data_dir, f".lock_{gba_region}_{tile_name}")
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "w") as lock_file:
                print(f"GBA tile not found locally, acquiring lock...")
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Re-check after acquiring lock (another process may have finished)
                    if os.path.exists(gpkg_path):
                        print(f"GBA tile already converted by another process.")
                        return gpkg_path
                    if not os.path.exists(geojson_path):
                        print(f"Downloading GBA tile...")
                        script_path = os.path.join(
                            os.path.dirname(__file__), "download_gba.sh"
                        )
                        subprocess.run(
                            ["bash", script_path, tile_name, gba_region, gba_data_dir],
                            check=True,
                        )
                        if not os.path.exists(geojson_path):
                            raise FileNotFoundError(
                                f"Download completed but GBA tile not found at {geojson_path}"
                            )
                    else:
                        print(f"GBA tile already downloaded by another process.")
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        # Convert to GeoPackage (lock to prevent parallel conversion)
        lock_path = os.path.join(gba_data_dir, f".lock_convert_{gba_region}_{tile_name}")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                gpkg_path = _convert_geojson_to_gpkg(geojson_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        return gpkg_path

    # Fallback to legacy gba_data_path
    gba_data_path = exp_config.get("gba_data_path")
    if not gba_data_path:
        raise ValueError(
            "Neither gba_data_dir nor gba_data_path set in experiment.toml"
        )
    return gba_data_path


def load_buildings(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_name_cn = region_config["region_name_cn"]
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict", None)
    if not boundaries_dict:
        print(
            f"No boundaries_dict found for {region_name_en}. Skipping building fetch."
        )
        return

    large_bounds_dict = region_info.get_stat("enlarged_boundaries")

    if not large_bounds_dict:
        enlarge_km = exp_config.get("delivery_radius_km", 5.0)
        large_bounds_dict = enlarge_boundaries(boundaries_dict, enlarge_km)
        region_info.set_stat("enlarged_boundaries", large_bounds_dict)

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
        gba_data_path = _resolve_gba_data_path(boundaries_dict, exp_config)

        # load_buildings_from_gba now saves as .gpkg
        buildings_file = load_buildings_from_gba(
            gba_data_path=gba_data_path,
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

    # Visualization
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for_latex = exp_config.get("for_latex", False)
    plot_title = exp_config.get("plot_title", False)
    save_path = os.path.join(fig_dir, f"{region_name_en.lower()}_buildings_vis.png")

    visualize_buildings(
        gdf,
        city_name=region_name_en,
        city_name_cn=region_name_cn,
        boundaries_dict=large_bounds_dict,
        original_boundaries_dict=boundaries_dict,
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )

    # Height heatmap visualization
    height_save_path = os.path.join(
        fig_dir, f"{region_name_en.lower()}_buildings_height.png"
    )

    visualize_building_heights(
        gdf,
        city_name=region_name_en,
        city_name_cn=region_name_cn,
        boundaries_dict=large_bounds_dict,
        original_boundaries_dict=boundaries_dict,
        save_path=height_save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )
