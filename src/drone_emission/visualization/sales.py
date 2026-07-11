import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils.coordinates import convert_coords_array

from ._utils import (
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    saved_tight_axis_report,
    setup_figure,
    write_figure_report,
    _add_title_suffix,
)

# Padding coefficient for heatmap Gaussian smoothing boundary extension
# This ensures heat diffusion doesn't get clipped at data boundaries
HEATMAP_PADDING_COEFF = 0.05
HEATMAP_SMOOTHING_DEG = 0.0135


def plot_sales_distribution(df, region_name, log_scale=False, save_path=None, for_latex=False, plot_title=False):
    """Plot sales distribution histogram.

    Args:
        df: DataFrame with 'month_sales' column
        region_name: Name of the region for title
        log_scale: If True, use logarithmic x-axis scale
        save_path: Path to save the figure
        for_latex: If True, save as PDF format
        plot_title: If True, show title
    """
    plt.figure(figsize=(10, 6))

    sales = df["month_sales"]
    if log_scale:
        # Filter out zero or negative values for log scale
        sales = sales[sales > 0]
        bins = np.logspace(np.log10(sales.min()), np.log10(sales.max()), 30)
    else:
        bins = 30

    plt.hist(sales, bins=bins, color="skyblue", edgecolor="black")

    if log_scale:
        plt.xscale("log")
        if plot_title:
            plt.title(f"The Distribution of Monthly Sales in {region_name} (Log Scale)")
        plt.xlabel("Monthly Sales (log scale)")
    else:
        if plot_title:
            plt.title(f"The Distribution of Monthly Sales in {region_name}")
        plt.xlabel("Monthly Sales")

    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def _prepare_data(df):
    lat = df["lat"].values
    lng = df["lng"].values
    sales = df["month_sales"].values
    lng, lat = convert_coords_array(lng, lat)
    return lng, lat, sales


def _sales_heatmap_geometry(lng, lat, boundaries_dict=None, lng_range=None, lat_range=None):
    """Return heatmap bounds and map aspect geometry."""
    if lng_range is not None and lat_range is not None:
        min_lng, max_lng = lng_range
        min_lat, max_lat = lat_range
        heatmap_range = [list(lng_range), list(lat_range)]
    elif boundaries_dict:
        min_lng = boundaries_dict["min_lng"]
        max_lng = boundaries_dict["max_lng"]
        min_lat = boundaries_dict["min_lat"]
        max_lat = boundaries_dict["max_lat"]
        heatmap_range = [[min_lng, max_lng], [min_lat, max_lat]]
    else:
        pad_x = (lng.max() - lng.min()) * HEATMAP_PADDING_COEFF
        pad_y = (lat.max() - lat.min()) * HEATMAP_PADDING_COEFF
        min_lng, max_lng = lng.min() - pad_x, lng.max() + pad_x
        min_lat, max_lat = lat.min() - pad_y, lat.max() + pad_y
        heatmap_range = [[min_lng, max_lng], [min_lat, max_lat]]

    avg_lat = (min_lat + max_lat) / 2
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    plot_height = (max_lat - min_lat) * aspect_ratio
    plot_width = max_lng - min_lng
    return heatmap_range, aspect_ratio, plot_height, plot_width


def _sales_heatmap_sigma(heatmap_range, bins_x, bins_y, geographic_scale=False):
    """Return Gaussian sigma in grid cells, optionally preserving map bandwidth."""
    if not geographic_scale:
        return 15

    lng_span = float(heatmap_range[0][1] - heatmap_range[0][0])
    lat_span = float(heatmap_range[1][1] - heatmap_range[1][0])
    return (
        HEATMAP_SMOOTHING_DEG / (lng_span / bins_x),
        HEATMAP_SMOOTHING_DEG / (lat_span / bins_y),
    )


def _compact_sales_cmap():
    """Return a sequential warm colormap with a transparent low-density tail."""
    color_array = plt.cm.YlOrRd(np.linspace(0.12, 0.95, 256))
    density_fraction = np.linspace(0.0, 1.0, 256)
    color_array[:, -1] = 0.12 + 0.83 * np.power(density_fraction, 0.65)
    return LinearSegmentedColormap.from_list(
        name="sales_density_warm_alpha",
        colors=color_array,
    )


def _plot_sales_heatmap_on_ax(
    ax,
    df,
    region_name_cn,
    boundaries_dict=None,
    lng_range=None,
    lat_range=None,
    zoom=14,
    cax=None,
):
    """Draw a sales heatmap on an existing axes and return the image artist."""
    lng, lat, sales = _prepare_data(df)
    heatmap_range, aspect_ratio, plot_height, plot_width = _sales_heatmap_geometry(
        lng, lat, boundaries_dict, lng_range, lat_range
    )

    bins_x = 1000
    bins_y = max(1, int(bins_x * (plot_height / plot_width)))
    heatmap, xedges, yedges = np.histogram2d(
        lng, lat, bins=[bins_x, bins_y], weights=sales, range=heatmap_range
    )
    heatmap_smooth = gaussian_filter(heatmap, sigma=15)
    mask_threshold = heatmap_smooth.max() * 0.001
    heatmap_masked = np.ma.masked_where(heatmap_smooth < mask_threshold, heatmap_smooth)

    map_cmap = get_custom_cmap(0.1)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.set_aspect(aspect_ratio)
    im = ax.imshow(
        heatmap_masked.T,
        extent=extent,
        origin="lower",
        cmap=map_cmap,
        aspect=aspect_ratio,
        interpolation="bilinear",
        zorder=10,
    )

    ax.set_xlim(heatmap_range[0])
    ax.set_ylim(heatmap_range[1])
    add_basemap_and_boundaries(ax, region_name_cn, boundaries_dict, zoom=zoom)
    ax.set_xlim(heatmap_range[0])
    ax.set_ylim(heatmap_range[1])

    if cax is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, label="Sales Density (Weighted)")
    cbar.ax.tick_params(labelsize=10)
    return im


def plot_sales_heatmap(
    df, region_name, region_name_cn, boundaries_dict=None, save_path=None,
    for_latex=False, plot_title=False, lng_range=None, lat_range=None, zoom=14,
    compact=False, figsize=None,
):
    """Plot sales heatmap for a region.

    Args:
        df: DataFrame with 'lat', 'lng', 'month_sales' columns.
        region_name: English region name.
        region_name_cn: Chinese region name.
        boundaries_dict: Optional boundary dict for axis limits.
        save_path: Path to save figure. Shows plot if None.
        for_latex: If True, save as PDF.
        plot_title: If True, show title and stats text.
        lng_range: Optional (min_lng, max_lng) to fix histogram and view extent.
        lat_range: Optional (min_lat, max_lat) to fix histogram and view extent.
        zoom: Basemap tile zoom level (lower = fewer tiles to download).
        compact: If True, use a qualitative layout for a small composite panel.
        figsize: Optional figure size tuple. Compact mode defaults to (1.25, 1.05).
    """
    lng, lat, sales = _prepare_data(df)
    heatmap_range, range_aspect, range_height, range_width = (
        _sales_heatmap_geometry(
            lng,
            lat,
            boundaries_dict=boundaries_dict if compact else None,
            lng_range=lng_range,
            lat_range=lat_range,
        )
    )

    if compact:
        fig, ax = plt.subplots(figsize=figsize or (1.25, 1.05))
        aspect_ratio = range_aspect
        plot_height = range_height
        plot_width = range_width
        ax.set_aspect(aspect_ratio)
    else:
        fig, ax, aspect_ratio, plot_height, plot_width = setup_figure(
            lng, lat, boundaries_dict
        )
        if figsize is not None:
            fig.set_size_inches(figsize)

    # Create weighted 2D histogram (weighted by sales)
    bins_x = 1000
    bins_y = int(bins_x * (plot_height / plot_width))

    # Generate weighted 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        lng, lat, bins=[bins_x, bins_y], weights=sales, range=heatmap_range
    )

    smoothing_sigma = _sales_heatmap_sigma(
        heatmap_range,
        bins_x,
        bins_y,
        geographic_scale=compact,
    )
    heatmap_smooth = gaussian_filter(heatmap, sigma=smoothing_sigma)

    # Mask the background (low values) to make them transparent
    # Use a very low threshold so the alpha gradient handles the fade
    mask_threshold = heatmap_smooth.max() * 0.001
    heatmap_masked = np.ma.masked_where(heatmap_smooth < mask_threshold, heatmap_smooth)

    map_cmap = _compact_sales_cmap() if compact else get_custom_cmap(0.1)

    # Plot heatmap
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(
        heatmap_masked.T,
        extent=extent,
        origin="lower",
        cmap=map_cmap,  # Use custom transparent colormap
        aspect=aspect_ratio,  # Maintain geographic proportions
        interpolation="bilinear",
        zorder=10,  # Ensure heatmap is on top of basemap
    )

    ax.set_xlim(heatmap_range[0])
    ax.set_ylim(heatmap_range[1])
    add_basemap_and_boundaries(
        ax,
        region_name_cn,
        boundaries_dict,
        zoom=zoom,
        alpha=0.28 if compact else 0.6,
        boundary_color="#777777" if compact else "gray",
        boundary_linewidth=0.35 if compact else 2,
        boundary_alpha=0.65 if compact else 0.8,
        attribution_size=3.0 if compact else 8,
        attribution_text="(C) OpenStreetMap contributors" if compact else None,
        attribution_wrap=not compact,
    )
    ax.set_xlim(heatmap_range[0])
    ax.set_ylim(heatmap_range[1])

    if compact:
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.20)
        cax = fig.add_axes((0.24, 0.06, 0.56, 0.035))
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks([im.norm.vmin, im.norm.vmax])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(labelsize=4.5, length=0, pad=1)
        cbar.ax.set_title("Weighted sales density", fontsize=4.5, pad=1)
        cbar.outline.set_linewidth(0.25)
        colorbar_label = "Weighted sales density"
        colorbar_orientation = "horizontal"
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(
            im,
            cax=cax,
            orientation="vertical",
            label="Sales Density (Weighted)",
        )
        cbar.set_label("Sales Density (Weighted)", fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        colorbar_label = "Sales Density (Weighted)"
        colorbar_orientation = "vertical"

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    def _write_report(final_save_path):
        report_path = final_save_path.rsplit(".", 1)[0] + "_report.json"
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        write_figure_report(
            report_path,
            {
                "figure_name": "shanghai_heatmap",
                "figure_path": final_save_path,
                "layout_mode": "compact" if compact else "full",
                "native_size_inches": [
                    float(fig.get_size_inches()[0]),
                    float(fig.get_size_inches()[1]),
                ],
                "palette": (
                    "sequential warm yellow-to-red"
                    if compact
                    else "Spectral_r with transparent low-density tail"
                ),
                "smoothing": {
                    "geographic_bandwidth_degrees": (
                        HEATMAP_SMOOTHING_DEG if compact else None
                    ),
                    "sigma_grid_cells": (
                        [float(smoothing_sigma[0]), float(smoothing_sigma[1])]
                        if compact
                        else [float(smoothing_sigma), float(smoothing_sigma)]
                    ),
                },
                "basemap": {
                    "provider": "OpenStreetMap.Mapnik",
                    "zoom": int(zoom),
                    "alpha": 0.28 if compact else 0.6,
                    "attribution_fontsize_pt": 3.0 if compact else 8.0,
                    "attribution_wrap": not compact,
                },
                "coordinate_convention": (
                    "bbox_saved_tight_fraction values are fractions of the "
                    "saved page after bbox_inches='tight'."
                ),
                "map": {
                    "axis": saved_tight_axis_report(
                        fig, ax, renderer=renderer, tight_bbox=tight_bbox
                    ),
                    "lng_range": [float(heatmap_range[0][0]), float(heatmap_range[0][1])],
                    "lat_range": [float(heatmap_range[1][0]), float(heatmap_range[1][1])],
                },
                "colorbar": {
                    "axis": saved_tight_axis_report(
                        fig, cax, renderer=renderer, tight_bbox=tight_bbox
                    ),
                    "label": colorbar_label,
                    "orientation": colorbar_orientation,
                    "ticks": "qualitative Low-High" if compact else "numeric",
                },
            },
        )
        print(f"Sales heatmap report saved to {report_path}")

    finalize_plot(
        fig,
        ax,
        save_path,
        f"{region_name} Food Delivery Heatmap by Monthly Sales",
        stats_text,
        boundaries_dict,
        for_latex,
        plot_title and not compact,
        show_geo_axes=False,
        show_grid=False,
        show_frame=False,
        before_save=_write_report if save_path else None,
    )


def plot_merchant_monthly_sales_map(
    stores_df,
    buildings_gdf,
    region_name,
    save_path=None,
    figsize=(1.25, 0.95),
    for_latex=False,
    plot_title=False,
):
    """Show merchant monthly sales over a local building-footprint context.

    Args:
        stores_df: DataFrame with lat, lng, and month_sales columns.
        buildings_gdf: GeoDataFrame with building geometries.
        region_name: English region name used for an optional title.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    from .buildings import (
        MERCHANT_MAP_DEGREE_SIZE,
        MERCHANT_MAP_RANDOM_SEED,
        MERCHANT_MAP_TOP_SALES_FRACTION,
    )

    required_cols = {"lat", "lng", "month_sales"}
    missing_cols = required_cols - set(stores_df.columns)
    if missing_cols:
        print(
            "Merchant monthly sales map skipped - stores data missing "
            f"columns {sorted(missing_cols)}"
        )
        return
    if buildings_gdf is None or buildings_gdf.empty:
        print("Merchant monthly sales map skipped - no building geometries")
        return

    valid_stores = stores_df.dropna(subset=list(required_cols)).copy()
    for col in required_cols:
        valid_stores[col] = pd.to_numeric(valid_stores[col], errors="coerce")
    valid_stores = valid_stores.dropna(subset=list(required_cols))
    valid_stores = valid_stores[valid_stores["month_sales"] >= 0].copy()
    if valid_stores.empty:
        print("Merchant monthly sales map skipped - no valid merchant sales rows")
        return

    top_n = max(
        1,
        int(np.ceil(len(valid_stores) * MERCHANT_MAP_TOP_SALES_FRACTION)),
    )
    center_merchant = valid_stores.nlargest(top_n, "month_sales").sample(
        n=1,
        random_state=MERCHANT_MAP_RANDOM_SEED,
    ).iloc[0]
    center_lng, center_lat = convert_coords_array(
        np.array([float(center_merchant["lng"])]),
        np.array([float(center_merchant["lat"])]),
    )
    center_lng = float(center_lng[0])
    center_lat = float(center_lat[0])

    map_rect = (0.03, 0.23, 0.94, 0.74)
    map_physical_ratio = (
        float(figsize[0]) * map_rect[2]
        / (float(figsize[1]) * map_rect[3])
    )
    aspect_ratio = 1.0 / np.cos(np.radians(center_lat))
    lat_span = MERCHANT_MAP_DEGREE_SIZE
    lng_span = map_physical_ratio * lat_span * aspect_ratio
    boundaries = {
        "min_lng": center_lng - lng_span / 2,
        "max_lng": center_lng + lng_span / 2,
        "min_lat": center_lat - lat_span / 2,
        "max_lat": center_lat + lat_span / 2,
    }

    if buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() != 4326:
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)
    buildings_in_frame = buildings_gdf.cx[
        boundaries["min_lng"]:boundaries["max_lng"],
        boundaries["min_lat"]:boundaries["max_lat"],
    ].copy()
    if buildings_in_frame.empty:
        print("Merchant monthly sales map skipped - no buildings in local frame")
        return

    merchant_lng, merchant_lat = convert_coords_array(
        valid_stores["lng"].to_numpy(),
        valid_stores["lat"].to_numpy(),
    )
    valid_stores["_lng_wgs84"] = merchant_lng
    valid_stores["_lat_wgs84"] = merchant_lat
    in_frame = (
        valid_stores["_lng_wgs84"].between(
            boundaries["min_lng"], boundaries["max_lng"]
        )
        & valid_stores["_lat_wgs84"].between(
            boundaries["min_lat"], boundaries["max_lat"]
        )
    )
    merchants_in_frame = valid_stores[in_frame].copy()
    if merchants_in_frame.empty:
        print("Merchant monthly sales map skipped - no merchants in local frame")
        return

    sqrt_sales = np.sqrt(merchants_in_frame["month_sales"].to_numpy(dtype=float))
    marker_min, marker_max = 5.0, 38.0
    if float(sqrt_sales.max()) > float(sqrt_sales.min()):
        marker_sizes = marker_min + (
            (sqrt_sales - sqrt_sales.min())
            / (sqrt_sales.max() - sqrt_sales.min())
            * (marker_max - marker_min)
        )
    else:
        marker_sizes = np.full(len(merchants_in_frame), (marker_min + marker_max) / 2)
    merchants_in_frame["_marker_size"] = marker_sizes
    merchants_in_frame = merchants_in_frame.sort_values("_marker_size")

    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes(map_rect)
    ax.set_facecolor("#fafafa")
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries["min_lng"], boundaries["max_lng"])
    ax.set_ylim(boundaries["min_lat"], boundaries["max_lat"])
    buildings_in_frame.plot(
        ax=ax,
        facecolor="#d7d9dc",
        edgecolor="none",
        alpha=0.86,
        zorder=1,
    )
    merchant_color = "#d94b3f"
    ax.scatter(
        merchants_in_frame["_lng_wgs84"],
        merchants_in_frame["_lat_wgs84"],
        s=merchants_in_frame["_marker_size"],
        c=merchant_color,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.22,
        zorder=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_ax = fig.add_axes((0.0, 0.02, 1.0, 0.16))
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")
    legend_ax.text(
        0.5,
        0.82,
        "Merchant monthly sales",
        ha="center",
        va="center",
        fontsize=4.5,
    )
    legend_ax.scatter(
        [0.38, 0.62],
        [0.31, 0.31],
        s=[marker_min, marker_max],
        c=merchant_color,
        edgecolors="white",
        linewidths=0.22,
    )
    legend_ax.text(0.33, 0.31, "Low", ha="right", va="center", fontsize=4.5)
    legend_ax.text(0.69, 0.31, "High", ha="left", va="center", fontsize=4.5)

    if plot_title:
        ax.set_title(
            f"{region_name} merchant monthly sales",
            fontsize=7,
            fontweight="bold",
            pad=2,
        )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        report_path = save_path.rsplit(".", 1)[0] + "_report.json"
        sales_values = merchants_in_frame["month_sales"].to_numpy(dtype=float)
        write_figure_report(
            report_path,
            {
                "figure_name": "shanghai_merchant_monthly_sales",
                "figure_path": save_path,
                "source_field": "month_sales",
                "observation_period": "2021-08",
                "native_size_inches": [float(figsize[0]), float(figsize[1])],
                "selection": {
                    "center_rule": "fixed-seed sample from top 1% monthly sales",
                    "top_sales_fraction": MERCHANT_MAP_TOP_SALES_FRACTION,
                    "random_seed": MERCHANT_MAP_RANDOM_SEED,
                    "center_merchant": {
                        "lng": center_lng,
                        "lat": center_lat,
                        "month_sales": float(center_merchant["month_sales"]),
                    },
                },
                "map": {
                    "axis": saved_tight_axis_report(
                        fig, ax, renderer=renderer, tight_bbox=tight_bbox
                    ),
                    "lng_range": [boundaries["min_lng"], boundaries["max_lng"]],
                    "lat_range": [boundaries["min_lat"], boundaries["max_lat"]],
                    "merchant_count": int(len(merchants_in_frame)),
                    "building_count": int(len(buildings_in_frame)),
                },
                "monthly_sales": {
                    "min": float(np.min(sales_values)),
                    "median": float(np.median(sales_values)),
                    "max": float(np.max(sales_values)),
                },
                "size_encoding": {
                    "transform": "sqrt without percentile cap",
                    "marker_area_pt2": [marker_min, marker_max],
                    "legend": "qualitative Low-High",
                },
            },
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Merchant monthly sales map saved to {save_path}")
        print(f"Merchant monthly sales map report saved to {report_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_store_scatter(
    df, region_name, region_name_cn, boundaries_dict=None, save_path=None, for_latex=False, plot_title=False
):
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, _, _ = setup_figure(lng, lat, boundaries_dict)

    # Get custom colormap
    map_cmap = get_custom_cmap(0.1)

    # Scatter plot
    sc = ax.scatter(
        lng,
        lat,
        c=sales,
        cmap=map_cmap,
        s=8,
        alpha=1.0,  # Alpha is handled by the colormap
        zorder=10,
        edgecolors="none",
    )

    add_basemap_and_boundaries(ax, region_name_cn, boundaries_dict)

    # Add colorbar aligned with map height using make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(sc, cax=cax, label="Monthly Sales")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    finalize_plot(
        fig,
        ax,
        save_path,
        f"{region_name} Food Delivery Store Locations",
        stats_text,
        boundaries_dict,
        for_latex,
        plot_title,
    )
