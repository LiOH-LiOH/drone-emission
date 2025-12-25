import json
import os

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

from ._utils import convert_coords_array, gcj02_to_wgs84


def get_adcode(city_name):
    # Simple mapping for major cities
    mapping = {
        "beijing": 110000,
        "shanghai": 310000,
        "guangzhou": 440100,
        "shenzhen": 440300,
    }
    return mapping.get(city_name.lower())


def fetch_boundaries(adcode):
    # Define cache directory
    cache_dir = os.path.join(os.getcwd(), "cache", "boundaries")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{adcode}.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                print(f"Loading boundaries from cache: {cache_file}")
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache {cache_file}: {e}")

    # Download if not in cache
    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    try:
        print(f"Downloading boundaries for adcode {adcode}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return data
    except Exception as e:
        print(f"Failed to fetch boundaries: {e}")
        return None


def plot_geojson_boundaries(ax, geojson_data, zorder=15):
    if not geojson_data:
        return

    print("Plotting administrative boundaries...")
    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        if geom_type == "Polygon":
            for ring in coords:
                # ring is a list of [x, y]
                ring_wgs = [gcj02_to_wgs84(pt[0], pt[1]) for pt in ring]
                xs, ys = zip(*ring_wgs)
                ax.plot(xs, ys, color="gray", linewidth=2, alpha=0.8, zorder=zorder)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    ring_wgs = [gcj02_to_wgs84(pt[0], pt[1]) for pt in ring]
                    xs, ys = zip(*ring_wgs)
                    ax.plot(xs, ys, color="gray", linewidth=2, alpha=0.8, zorder=zorder)


def plot_sales_distribution(df, city_name, save_dir):
    plt.figure(figsize=(10, 6))

    plt.hist(df["month_sales"], bins=30, color="skyblue", edgecolor="black")
    plt.title("The Distribution of Monthly Sales in " + city_name)
    plt.xlabel("Monthly Sales")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    if save_dir:
        save_path = os.path.join(
            save_dir, f"{city_name.lower()}_month_sales_distribution.png"
        )
        plt.savefig(save_path)


def _prepare_data(df):
    lat = df["lat"].values
    lng = df["lng"].values
    sales = df["month_sales"].values
    lng, lat = convert_coords_array(lng, lat)
    return lng, lat, sales


def _setup_figure(lng, lat):
    # Calculate aspect ratio to maintain geographic proportions
    avg_lat = np.mean(lat)
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))

    # Calculate suitable figsize based on data bounds
    padding = 0.04
    lat_range = (lat.max() - lat.min()) + padding
    lng_range = (lng.max() - lng.min()) + padding

    plot_height = lat_range * aspect_ratio
    plot_width = lng_range

    base_width = 15
    fig_height = base_width * (plot_height / plot_width)

    fig, ax = plt.subplots(figsize=(base_width, fig_height), dpi=150)
    return fig, ax, aspect_ratio, plot_height, plot_width


# Global flag for cache configuration
_CONTEXTILY_CACHE_CONFIGURED = False


def _configure_contextily_cache():
    global _CONTEXTILY_CACHE_CONFIGURED
    if not _CONTEXTILY_CACHE_CONFIGURED:
        try:
            cache_dir = os.path.join(os.getcwd(), "cache", "tiles")
            os.makedirs(cache_dir, exist_ok=True)
            ctx.set_cache_dir(cache_dir)
            print(f"Contextily cache configured at: {cache_dir}")
            _CONTEXTILY_CACHE_CONFIGURED = True
        except Exception as e:
            print(f"Failed to configure contextily cache: {e}")


def _add_basemap_and_boundaries(ax, city_name):
    # Add basemap with administrative boundaries
    try:
        # Ensure cache is configured
        _configure_contextily_cache()

        # Add basemap
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha=1.0,  # Map fully opaque (or slightly transparent if desired)
            zorder=0,  # Map at bottom
        )
        print("Basemap added successfully")
    except Exception as e:
        print(f"Could not add basemap: {e}")

    # Add administrative boundaries (counties/districts)
    adcode = get_adcode(city_name)
    if adcode:
        geojson_data = fetch_boundaries(adcode)
        plot_geojson_boundaries(ax, geojson_data, zorder=15)
    else:
        print(f"No adcode found for {city_name}, skipping boundaries.")


def _finalize_plot(fig, ax, lng, lat, city_name, save_dir, filename, title, stats_text):
    # Set title and labels
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", color="white")

    # Add statistics
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Set axis limits
    ax.set_xlim(lng.min() - 0.02, lng.max() + 0.02)
    ax.set_ylim(lat.min() - 0.02, lat.max() + 0.02)

    plt.tight_layout()

    if save_dir:
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as {save_path}")


def _get_custom_cmap():
    # Create custom colormap with transparency gradient
    # Base colormap: Spectral_r (Blue -> Green -> Yellow -> Red)
    ncolors = 256
    color_array = plt.cm.Spectral_r(np.linspace(0, 1, ncolors))

    # Create alpha gradient
    # "Rapid decay": Alpha should be high (1.0) for most values, but drop quickly to 0 at the low end.
    # Below 10% of the range, it fades from 0 to 1.0. Above 10%, it stays fully opaque (1.0).
    threshold_idx = int(ncolors * 0.1)
    alphas = np.concatenate(
        (
            np.linspace(0, 1.0, threshold_idx),  # Rapid rise from 0 to 1.0
            np.ones(ncolors - threshold_idx),  # Stable high opacity (1.0)
        )
    )
    color_array[:, -1] = alphas

    # Create new colormap
    return LinearSegmentedColormap.from_list(
        name="Spectral_r_alpha", colors=color_array
    )


def plot_sales_heatmap(df, city_name, save_dir):
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, plot_height, plot_width = _setup_figure(lng, lat)

    # Create weighted 2D histogram (weighted by sales)
    # Increase resolution (approx 5x previous)
    bins_x = 1000
    bins_y = int(bins_x * (plot_height / plot_width))

    # Generate weighted 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        lng, lat, bins=[bins_x, bins_y], weights=sales
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
    map_cmap = _get_custom_cmap()

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

    _add_basemap_and_boundaries(ax, city_name)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Sales Density (Weighted)")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    _finalize_plot(
        fig,
        ax,
        lng,
        lat,
        city_name,
        save_dir,
        f"{city_name.lower()}_heatmap.png",
        f"{city_name} Food Delivery Heatmap by Monthly Sales",
        stats_text,
    )


def plot_store_scatter(df, city_name, save_dir):
    lng, lat, sales = _prepare_data(df)
    fig, ax, aspect_ratio, _, _ = _setup_figure(lng, lat)

    # Get custom colormap
    map_cmap = _get_custom_cmap()

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

    _add_basemap_and_boundaries(ax, city_name)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, label="Monthly Sales")
    cbar.ax.tick_params(labelsize=10)

    stats_text = f"Total Stores: {len(df)}\nAvg Sales: {sales.mean():.1f}\nMax Sales: {sales.max():.0f}"

    _finalize_plot(
        fig,
        ax,
        lng,
        lat,
        city_name,
        save_dir,
        f"{city_name.lower()}_scatter.png",
        f"{city_name} Food Delivery Store Locations",
        stats_text,
    )
