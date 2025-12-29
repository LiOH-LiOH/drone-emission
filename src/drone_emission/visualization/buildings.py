import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from ._utils import add_basemap_and_boundaries, finalize_plot, setup_figure


def visualize_buildings(
    geojson_path, city_name, city_name_cn=None, boundaries_dict=None, save_path=None
):
    """
    Read a GeoJSON file of buildings and visualize it.

    Args:
        geojson_path (str): Path to the GeoJSON file.
        city_name_cn (str, optional): Chinese name of the city for basemap and boundaries.
        boundaries_dict (dict, optional): Dictionary with min_lng, max_lng, min_lat, max_lat.
        save_path (str, optional): Path to save the visualization. If None, shows the plot.
    """
    if not os.path.exists(geojson_path):
        print(f"File not found: {geojson_path}")
        return

    print(f"Reading {geojson_path}...")
    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if gdf.empty:
        print("GeoJSON file is empty.")
        return

    print(f"Loaded {len(gdf)} buildings.")

    # Convert to EPSG:4326 (WGS84) if needed for basemap
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    # Get bounds from boundaries_dict
    lng = np.array([boundaries_dict["min_lng"], boundaries_dict["max_lng"]])
    lat = np.array([boundaries_dict["min_lat"], boundaries_dict["max_lat"]])

    fig, ax, _, _, _ = setup_figure(lng, lat)
    gdf.plot(ax=ax, color="blue", edgecolor=None, alpha=0.7, zorder=10)

    # Add basemap and boundaries if city_name_cn is provided
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn)

    title = f"{city_name} Buildings Visualization"
    stats_text = f"Total Buildings: {len(gdf)}"

    finalize_plot(fig, ax, lng, lat, save_path, title, stats_text)
    plt.close(fig)
