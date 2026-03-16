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


def _resolve_gba_data_path(boundaries_dict: dict, exp_config: dict) -> str:
    """Resolve the GBA GeoJSON path for a region, downloading if needed.

    Prefers gba_data_dir (dynamic tile resolution) over gba_data_path (legacy).
    """
    gba_data_dir = exp_config.get("gba_data_dir")

    if gba_data_dir:
        tile_name, gba_region = get_gba_tile_info(boundaries_dict)
        gba_data_path = os.path.join(
            gba_data_dir, "LoD1_GeoJSON", gba_region, f"{tile_name}.geojson"
        )
        print(f"GBA tile: {tile_name} (region: {gba_region})")

        if not os.path.exists(gba_data_path):
            # Use file lock to prevent parallel processes from downloading the same tile
            lock_path = os.path.join(gba_data_dir, f".lock_{gba_region}_{tile_name}")
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "w") as lock_file:
                print(f"GBA tile not found locally, acquiring lock...")
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Re-check after acquiring lock (another process may have finished)
                    if not os.path.exists(gba_data_path):
                        print(f"Downloading GBA tile...")
                        script_path = os.path.join(
                            os.path.dirname(__file__), "download_gba.sh"
                        )
                        subprocess.run(
                            ["bash", script_path, tile_name, gba_region, gba_data_dir],
                            check=True,
                        )
                        if not os.path.exists(gba_data_path):
                            raise FileNotFoundError(
                                f"Download completed but GBA tile not found at {gba_data_path}"
                            )
                    else:
                        print(f"GBA tile already downloaded by another process.")
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)

        return gba_data_path

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

    geojson_file = os.path.join(
        region_save_dir, f"{region_name_en.lower()}_buildings.geojson"
    )

    # Fetch buildings if requested or file doesn't exist
    if exp_config.get("fetch_buildings", False) or not os.path.exists(geojson_file):
        gba_data_path = _resolve_gba_data_path(boundaries_dict, exp_config)

        load_buildings_from_gba(
            gba_data_path=gba_data_path,
            bbox_4326=bbox_4326,
            save_path=geojson_file,
        )

    # Load buildings from file
    print(f"Reading {geojson_file}...")
    try:
        gdf = gpd.read_file(geojson_file)
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
