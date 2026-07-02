import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point
from utils.coordinates import gcj02_to_wgs84
from utils.tools import get_epsg_code, get_epsg_from_bounds

from ._utils import (
    _add_title_suffix,
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    plot_boundary_box,
    setup_figure,
)


MERCHANT_MAP_DEGREE_SIZE = 0.1
MERCHANT_MAP_RANDOM_SEED = 20260703
MERCHANT_MAP_TOP_SALES_FRACTION = 0.01


def visualize_buildings(
    gdf,
    city_name,
    city_name_cn=None,
    boundaries_dict=None,
    original_boundaries_dict=None,
    save_path=None,
    for_latex=False,
    plot_title=False,
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

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict, for_latex, plot_title)
    plt.close(fig)


def visualize_building_heights(
    gdf,
    city_name,
    city_name_cn=None,
    boundaries_dict=None,
    original_boundaries_dict=None,
    save_path=None,
    for_latex=False,
    plot_title=False,
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

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict, for_latex, plot_title)
    plt.close(fig)


def plot_building_height_merchant_map(
    buildings_gdf,
    stores_df,
    region_name,
    region_name_cn,
    save_path=None,
    figsize=(10, 6),
    for_latex=False,
    plot_title=False,
):
    """Show building footprints colored by height around a top-sales merchant.

    Args:
        buildings_gdf: GeoDataFrame with building geometries and a height column.
        stores_df: DataFrame with lat, lng, and month_sales columns.
        region_name: English region name.
        region_name_cn: Chinese region name for basemap and boundaries.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required_store_cols = {"lat", "lng", "month_sales"}
    missing_store_cols = required_store_cols - set(stores_df.columns)
    if missing_store_cols:
        print(
            "Building height merchant map skipped - stores data missing "
            f"columns {sorted(missing_store_cols)}"
        )
        return

    if "height" not in buildings_gdf.columns:
        print(
            "Building height merchant map skipped - buildings data missing "
            "height column"
        )
        return

    valid_stores = stores_df.dropna(subset=["lat", "lng", "month_sales"]).copy()
    valid_stores["month_sales"] = pd.to_numeric(
        valid_stores["month_sales"], errors="coerce"
    )
    valid_stores = valid_stores.dropna(subset=["month_sales"])
    if valid_stores.empty:
        print("Building height merchant map skipped - no valid merchant sales rows")
        return

    top_n = max(
        1,
        int(np.ceil(len(valid_stores) * MERCHANT_MAP_TOP_SALES_FRACTION)),
    )
    top_merchants = valid_stores.nlargest(top_n, "month_sales")
    merchant = top_merchants.sample(
        n=1,
        random_state=MERCHANT_MAP_RANDOM_SEED,
    ).iloc[0]
    merchant_lng, merchant_lat = gcj02_to_wgs84(
        float(merchant["lng"]),
        float(merchant["lat"]),
    )

    half_size = MERCHANT_MAP_DEGREE_SIZE / 2
    boundaries_dict = {
        "min_lng": merchant_lng - half_size,
        "max_lng": merchant_lng + half_size,
        "min_lat": merchant_lat - half_size,
        "max_lat": merchant_lat + half_size,
    }

    if buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {buildings_gdf.crs} to EPSG:4326...")
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    buildings_in_frame = buildings_gdf.cx[
        boundaries_dict["min_lng"]:boundaries_dict["max_lng"],
        boundaries_dict["min_lat"]:boundaries_dict["max_lat"],
    ].copy()
    if buildings_in_frame.empty:
        print(
            "Building height merchant map skipped - no buildings in 0.1 "
            "degree frame"
        )
        return

    buildings_in_frame["height"] = pd.to_numeric(
        buildings_in_frame["height"], errors="coerce"
    )
    buildings_in_frame = buildings_in_frame.dropna(subset=["height"])
    if buildings_in_frame.empty:
        print(
            "Building height merchant map skipped - no valid building heights "
            "in frame"
        )
        return

    height_cap = buildings_in_frame["height"].quantile(0.95)
    if not np.isfinite(height_cap) or height_cap <= 0:
        print(
            "Building height merchant map skipped - invalid building heights "
            "in frame"
        )
        return

    buildings_in_frame["_height_capped_m"] = buildings_in_frame["height"].clip(
        upper=height_cap
    )
    height_min = float(buildings_in_frame["_height_capped_m"].min())
    height_max = float(buildings_in_frame["_height_capped_m"].max())
    if height_min == height_max:
        height_min = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    avg_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])

    add_basemap_and_boundaries(
        ax,
        region_name_cn,
        boundaries_dict,
        zoom=14,
        alpha=0.45,
    )

    cmap = get_custom_cmap(0)
    buildings_in_frame.plot(
        ax=ax,
        column="_height_capped_m",
        cmap=cmap,
        vmin=height_min,
        vmax=height_max,
        edgecolor="none",
        alpha=0.92,
        zorder=10,
        legend=True,
        legend_kwds={
            "label": "Building height (m, capped at 95th pct)",
            "shrink": 0.75,
        },
    )

    ax.scatter(
        merchant_lng,
        merchant_lat,
        c="#ffd400",
        s=260,
        marker="*",
        edgecolors="black",
        linewidths=1.2,
        zorder=30,
        label="Selected top-sales merchant",
    )

    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, alpha=0.35, linestyle="--", color="white", zorder=20)

    legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    legend.set_zorder(100)

    if plot_title:
        ax.set_title(
            f"{region_name} building heights around a top-sales merchant",
            fontsize=16,
            fontweight="bold",
            pad=18,
        )
        ax.text(
            0.02,
            0.98,
            (
                f"Frame: {MERCHANT_MAP_DEGREE_SIZE:.1f} deg x "
                f"{MERCHANT_MAP_DEGREE_SIZE:.1f} deg\n"
                f"Buildings: {len(buildings_in_frame):,}\n"
                f"Merchant monthly sales: {merchant['month_sales']:.0f}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            zorder=100,
        )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".pdf"):
            ax.set_rasterization_zorder(12)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Building height merchant map saved to {save_path}")
    else:
        plt.show()

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
    for_latex=False,
    plot_title=False,
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
    # Get appropriate UTM EPSG code from boundaries if available, else fall back to Nominatim
    if boundaries_dict:
        epsg_code = get_epsg_from_bounds(boundaries_dict)
    else:
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

    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict, for_latex, plot_title)
    plt.close(fig)
