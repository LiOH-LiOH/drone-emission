import os

import matplotlib.pyplot as plt
import numpy as np
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
        attribution_size=3.5 if compact else 8,
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
                    "attribution_fontsize_pt": 3.5 if compact else 8.0,
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
