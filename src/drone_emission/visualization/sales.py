import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils.coordinates import convert_coords_array

from ._utils import (
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    setup_figure,
    _add_title_suffix,
)

# Padding coefficient for heatmap Gaussian smoothing boundary extension
# This ensures heat diffusion doesn't get clipped at data boundaries
HEATMAP_PADDING_COEFF = 0.05


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
    """
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, plot_height, plot_width = setup_figure(
        lng, lat, boundaries_dict
    )

    # Create weighted 2D histogram (weighted by sales)
    bins_x = 1000
    bins_y = int(bins_x * (plot_height / plot_width))

    if lng_range is not None and lat_range is not None:
        heatmap_range = [list(lng_range), list(lat_range)]
    else:
        pad_x = (lng.max() - lng.min()) * HEATMAP_PADDING_COEFF
        pad_y = (lat.max() - lat.min()) * HEATMAP_PADDING_COEFF
        heatmap_range = [
            [lng.min() - pad_x, lng.max() + pad_x],
            [lat.min() - pad_y, lat.max() + pad_y],
        ]

    # Generate weighted 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        lng, lat, bins=[bins_x, bins_y], weights=sales, range=heatmap_range
    )

    # Apply Gaussian filter to spread heat around each point
    # Larger sigma value means wider spread
    # Scaled sigma for higher resolution (approx 5x bins -> 5x sigma)
    heatmap_smooth = gaussian_filter(heatmap, sigma=15)

    # Mask the background (low values) to make them transparent
    # Use a very low threshold so the alpha gradient handles the fade
    mask_threshold = heatmap_smooth.max() * 0.001
    heatmap_masked = np.ma.masked_where(heatmap_smooth < mask_threshold, heatmap_smooth)

    # Get custom colormap
    map_cmap = get_custom_cmap(0.1)

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

    add_basemap_and_boundaries(ax, region_name_cn, boundaries_dict, zoom=zoom)

    # Place colorbar above the map to preserve horizontal figure space.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="4%", pad=0.1)
    cbar = plt.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        label="Sales Density (Weighted)",
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    finalize_plot(
        fig,
        ax,
        save_path,
        f"{region_name} Food Delivery Heatmap by Monthly Sales",
        stats_text,
        boundaries_dict,
        for_latex,
        plot_title,
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
