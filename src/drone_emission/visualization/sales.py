import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from utils.coordinates import convert_coords_array

from ._utils import (
    PADDING_COEFF,
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    setup_figure,
)


def plot_sales_distribution(df, region_name, save_path=None):
    plt.figure(figsize=(10, 6))

    plt.hist(df["month_sales"], bins=30, color="skyblue", edgecolor="black")
    plt.title("The Distribution of Monthly Sales in " + region_name)
    plt.xlabel("Monthly Sales")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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


def plot_sales_heatmap(df, region_name, region_name_cn, save_path=None):
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, plot_height, plot_width = setup_figure(lng, lat)

    # Create weighted 2D histogram (weighted by sales)
    # Increase resolution (approx 5x previous)
    bins_x = 1000
    bins_y = int(bins_x * (plot_height / plot_width))

    # Add padding to prevent heatmap clipping at data boundaries
    pad_x = (lng.max() - lng.min()) * PADDING_COEFF
    pad_y = (lat.max() - lat.min()) * PADDING_COEFF
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

    add_basemap_and_boundaries(ax, region_name_cn)

    # Add colorbar aligned with map height using make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, label="Sales Density (Weighted)")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    finalize_plot(
        fig,
        ax,
        lng,
        lat,
        save_path,
        f"{region_name} Food Delivery Heatmap by Monthly Sales",
        stats_text,
    )


def plot_store_scatter(df, region_name, region_name_cn, save_path=None):
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, _, _ = setup_figure(lng, lat)

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

    add_basemap_and_boundaries(ax, region_name_cn)

    # Add colorbar aligned with map height using make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cbar = plt.colorbar(sc, cax=cax, label="Monthly Sales")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    finalize_plot(
        fig,
        ax,
        lng,
        lat,
        save_path,
        f"{region_name} Food Delivery Store Locations",
        stats_text,
    )
