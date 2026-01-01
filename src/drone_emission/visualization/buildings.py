import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
from utils.tools import get_epsg_code

from ._utils import (
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    plot_boundary_box,
    setup_figure,
)


def visualize_buildings(
    gdf,
    city_name,
    city_name_cn=None,
    boundaries_dict=None,
    original_boundaries_dict=None,
    save_path=None,
):
    """
    Read a GeoJSON file of buildings and visualize it.

    Args:
        gdf: GeoDataFrame of buildings.
        city_name (str): English name of the city.
        city_name_cn (str, optional): Chinese name of the city for basemap and boundaries.
        boundaries_dict (dict, optional): Dictionary with min_lng, max_lng, min_lat, max_lat (enlarged).
        original_boundaries_dict (dict, optional): Original boundary dict for drawing dashed box.
        save_path (str, optional): Path to save the visualization. If None, shows the plot.
    """
    # Convert to EPSG:4326 (WGS84) if needed for basemap
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    fig, ax, _, _, _ = setup_figure(None, None, boundaries_dict)
    gdf.plot(ax=ax, color="blue", edgecolor=None, alpha=0.7, zorder=10)

    # Add basemap and boundaries if city_name_cn is provided
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn, boundaries_dict)

    # Plot original boundary as dashed box
    plot_boundary_box(ax, original_boundaries_dict)

    title = f"{city_name} Buildings Visualization"
    stats_text = f"Total Buildings: {len(gdf)}"

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict)
    plt.close(fig)


def visualize_building_heights(
    gdf,
    city_name,
    city_name_cn=None,
    boundaries_dict=None,
    original_boundaries_dict=None,
    save_path=None,
):
    """
    Visualize building heights using a heatmap.

    Args:
        gdf: GeoDataFrame of buildings.
        city_name (str): English name of the city.
        city_name_cn (str, optional): Chinese name of the city for basemap and boundaries.
        boundaries_dict (dict, optional): Dictionary with min_lng, max_lng, min_lat, max_lat (enlarged).
        original_boundaries_dict (dict, optional): Original boundary dict for drawing dashed box.
        save_path (str, optional): Path to save the visualization. If None, shows the plot.
    """
    if "height" not in gdf.columns:
        print("No 'height' column found in GeoJSON.")
        return

    # Convert to EPSG:4326 (WGS84) if needed for basemap
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {gdf.crs} to EPSG:4326...")
        gdf = gdf.to_crs(epsg=4326)

    fig, ax, aspect_ratio, _, _ = setup_figure(None, None, boundaries_dict)

    # Get custom colormap
    cmap = get_custom_cmap(0)

    # Plot buildings colored by height
    gdf.plot(
        ax=ax,
        column="height",
        cmap=cmap,
        edgecolor=None,
        alpha=1,
        zorder=10,
        legend=True,
        legend_kwds={"label": "Building Height (m)", "shrink": 0.8},
    )

    # Add basemap and boundaries if city_name_cn is provided
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn, boundaries_dict)

    # Plot original boundary as dashed box
    plot_boundary_box(ax, original_boundaries_dict)

    title = f"{city_name} Building Heights"
    heights = gdf["height"].dropna()
    stats_text = (
        f"Total Buildings: {len(gdf)}\n"
        f"Avg Height: {heights.mean():.1f}m\n"
        f"Max Height: {heights.max():.1f}m"
    )

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict)
    plt.close(fig)


def visualize_store_coverage(
    buildings_gdf,
    store_coord,
    nearby_indices,
    city_name,
    city_name_cn=None,
    boundaries_dict=None,
    original_boundaries_dict=None,
    radius_km=5.0,
    save_path=None,
):
    """
    Visualize a store and its nearby buildings within delivery radius.

    Args:
        buildings_gdf: GeoDataFrame of all buildings.
        store_coord: Tuple of (lng, lat) in WGS84 for the store location.
        nearby_indices: Array of indices for buildings within delivery radius.
        city_name: English name of the city/region.
        city_name_cn: Chinese name for basemap and boundaries.
        boundaries_dict: Dictionary with min_lng, max_lng, min_lat, max_lat (enlarged).
        original_boundaries_dict: Original boundary dict for drawing dashed box.
        radius_km: Delivery radius in kilometers.
        save_path: Path to save the visualization.
    """
    # Convert to EPSG:4326 (WGS84) if needed
    if buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {buildings_gdf.crs} to EPSG:4326...")
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    fig, ax, aspect_ratio, _, _ = setup_figure(None, None, boundaries_dict)

    # Split buildings into nearby and others
    nearby_mask = np.zeros(len(buildings_gdf), dtype=bool)
    nearby_mask[nearby_indices] = True

    other_buildings = buildings_gdf[~nearby_mask]
    nearby_buildings = buildings_gdf[nearby_mask]

    # Plot other buildings in blue (background)
    if len(other_buildings) > 0:
        other_buildings.plot(ax=ax, color="blue", edgecolor=None, alpha=0.5, zorder=10)

    # Plot nearby buildings in red (highlighted)
    if len(nearby_buildings) > 0:
        nearby_buildings.plot(ax=ax, color="red", edgecolor=None, alpha=0.8, zorder=11)

    # Plot the store location as a star marker
    store_lng, store_lat = store_coord
    ax.scatter(
        store_lng,
        store_lat,
        c="yellow",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=1,
        zorder=20,
        label="Store",
    )

    # Draw delivery radius circle using projected CRS for accuracy
    # Get appropriate UTM EPSG code for the city
    epsg_code = get_epsg_code(city_name)

    # Create a point in WGS84, project to UTM, buffer, then back to WGS84
    store_point = gpd.GeoSeries([Point(store_lng, store_lat)], crs="EPSG:4326")
    store_point_utm = store_point.to_crs(epsg=epsg_code)

    # Convert radius from km to meters
    radius_m = radius_km * 1000

    # Create buffer circle in projected CRS
    circle_utm = store_point_utm.buffer(radius_m)
    circle_wgs84 = circle_utm.to_crs(epsg=4326)

    # Plot the circle boundary
    circle_coords = np.array(circle_wgs84.iloc[0].exterior.coords)
    ax.plot(
        circle_coords[:, 0],
        circle_coords[:, 1],
        color="green",
        linewidth=2,
        linestyle="--",
        zorder=15,
        label=f"Delivery Radius ({radius_km}km)",
    )

    # Add basemap and boundaries
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn, boundaries_dict)

    # Plot original boundary as dashed box
    plot_boundary_box(ax, original_boundaries_dict)

    # Add legend with high zorder to ensure visibility
    legend = ax.legend(loc="upper right", fontsize=10)
    legend.set_zorder(100)

    title = f"{city_name} Store Coverage Example"
    stats_text = (
        f"Total Buildings: {len(buildings_gdf)}\n"
        f"Nearby Buildings: {len(nearby_buildings)}\n"
        f"Radius: {radius_km}km"
    )

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict)
    plt.close(fig)
