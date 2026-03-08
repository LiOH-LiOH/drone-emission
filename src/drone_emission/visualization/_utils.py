import json
import os

import contextily as ctx
import cpca
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.colors import LinearSegmentedColormap
from utils.coordinates import gcj02_to_wgs84

# Global padding coefficient for plotting
PADDING_COEFF = 0.1

# Global flag for cache configuration
_CONTEXTILY_CACHE_CONFIGURED = False

# Project root directory (consistent cache location)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def get_adcode(city_name_cn):
    """Get adcode using cpca library"""
    try:
        df = cpca.transform([city_name_cn])
        if not df.empty:
            return df.iloc[0]["adcode"]
    except Exception as e:
        print(f"Error getting adcode for {city_name_cn}: {e}")
    return None


def fetch_boundaries(adcode):
    # Define cache directory (use project root for consistency)
    cache_dir = os.path.join(_PROJECT_ROOT, "cache", "boundaries")
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


def configure_contextily_cache():
    global _CONTEXTILY_CACHE_CONFIGURED
    if not _CONTEXTILY_CACHE_CONFIGURED:
        try:
            cache_dir = os.path.join(_PROJECT_ROOT, "cache", "tiles")
            os.makedirs(cache_dir, exist_ok=True)
            ctx.set_cache_dir(cache_dir)
            print(f"Contextily cache configured at: {cache_dir}")
            _CONTEXTILY_CACHE_CONFIGURED = True
        except Exception as e:
            print(f"Failed to configure contextily cache: {e}")


def plot_boundary_box(
    ax,
    boundaries_dict,
    color="orange",
    linestyle="--",
    linewidth=2,
    label="Original Boundary",
    zorder=16,
):
    """Plot a rectangular boundary box on the axes.

    Args:
        ax: Matplotlib axes
        boundaries_dict: Dictionary with min_lng, max_lng, min_lat, max_lat
        color: Line color
        linestyle: Line style (default dashed)
        linewidth: Line width
        label: Label for legend
        zorder: Z-order for layering
    """
    if not boundaries_dict:
        return

    min_lng = boundaries_dict["min_lng"]
    max_lng = boundaries_dict["max_lng"]
    min_lat = boundaries_dict["min_lat"]
    max_lat = boundaries_dict["max_lat"]

    # Draw rectangle (closed path)
    rect_x = [min_lng, max_lng, max_lng, min_lng, min_lng]
    rect_y = [min_lat, min_lat, max_lat, max_lat, min_lat]

    ax.plot(
        rect_x,
        rect_y,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=zorder,
    )


def add_basemap_and_boundaries(
    ax, city_name_cn, boundaries_dict=None, zoom=14, alpha=0.6
):
    # Set axis limits BEFORE adding basemap to ensure correct coverage
    if boundaries_dict:
        min_lng = boundaries_dict["min_lng"]
        max_lng = boundaries_dict["max_lng"]
        min_lat = boundaries_dict["min_lat"]
        max_lat = boundaries_dict["max_lat"]

        ax.set_xlim(min_lng, max_lng)
        ax.set_ylim(min_lat, max_lat)

    # Add basemap with administrative boundaries
    try:
        # Ensure cache is configured
        configure_contextily_cache()

        # Add basemap with higher resolution (zoom level) and reduced opacity
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=zoom,  # Higher zoom = higher resolution
            alpha=alpha,  # Reduced opacity for better data visibility
            zorder=0,  # Map at bottom
        )
        print(f"Basemap added successfully (zoom={zoom}, alpha={alpha})")
    except Exception as e:
        print(f"Could not add basemap: {e}")

    # Add administrative boundaries (counties/districts)
    adcode = get_adcode(city_name_cn)
    if adcode:
        geojson_data = fetch_boundaries(adcode)
        plot_geojson_boundaries(ax, geojson_data, zorder=15)
    else:
        print(f"No adcode found for {city_name_cn}, skipping boundaries.")


def _add_title_suffix(save_path):
    """Insert '_t' before the file extension in save_path."""
    if not save_path:
        return save_path
    base, ext = save_path.rsplit('.', 1)
    return f"{base}_t.{ext}"


def finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict=None, for_latex=False, plot_title=False):
    """Finalize plot with title, labels, and axis limits.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        save_path: Path to save the figure (or None to show)
        title: Plot title
        stats_text: Statistics text to display
        boundaries_dict: Optional dict with min_lng, max_lng, min_lat, max_lat
                        If provided, uses fixed boundaries for axis limits
        for_latex: If True, save as PDF format
        plot_title: If True, show title and stats text
    """
    # Set labels
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    if plot_title:
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", color="white")

    # Disable scientific notation for coordinates
    ax.ticklabel_format(useOffset=False, style="plain")

    # Add statistics
    if plot_title:
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            zorder=100,  # Ensure text is on top of all plot elements
        )

    # Set axis limits using boundaries_dict
    if boundaries_dict:
        min_lng = boundaries_dict["min_lng"]
        max_lng = boundaries_dict["max_lng"]
        min_lat = boundaries_dict["min_lat"]
        max_lat = boundaries_dict["max_lat"]

        print("Using fixed boundaries:")
        print(f"Longitude: {min_lng} to {max_lng}")
        print(f"Latitude: {min_lat} to {max_lat}")

        ax.set_xlim(min_lng, max_lng)
        ax.set_ylim(min_lat, max_lat)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Use PDF for LaTeX, PNG otherwise
        if for_latex and not save_path.endswith('.pdf'):
            save_path = save_path.rsplit('.', 1)[0] + '.pdf'
        # Add _t suffix for titled figures
        if plot_title:
            save_path = _add_title_suffix(save_path)
        # Rasterize heavy elements (basemap, buildings, roads) in PDF output
        # Elements with zorder < 12 are rasterized, others remain vector
        if save_path.endswith('.pdf'):
            ax.set_rasterization_zorder(12)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as {save_path}")
    else:
        plt.show()


def setup_figure(lng, lat, boundaries_dict=None):
    """Setup figure with proper aspect ratio for geographic data.

    Args:
        lng: Array of longitude values (used if boundaries_dict is None)
        lat: Array of latitude values (used if boundaries_dict is None)
        boundaries_dict: Optional dict with min_lng, max_lng, min_lat, max_lat
                        If provided, uses fixed boundaries instead of data range

    Returns:
        fig, ax, aspect_ratio, plot_height, plot_width
    """
    # Use boundaries_dict if provided, otherwise use data range
    if boundaries_dict:
        min_lng = boundaries_dict["min_lng"]
        max_lng = boundaries_dict["max_lng"]
        min_lat = boundaries_dict["min_lat"]
        max_lat = boundaries_dict["max_lat"]
    else:
        min_lng, max_lng = lng.min(), lng.max()
        min_lat, max_lat = lat.min(), lat.max()

    # Calculate aspect ratio to maintain geographic proportions
    avg_lat = (min_lat + max_lat) / 2
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))

    # Calculate suitable figsize based on bounds
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng

    pad_lat = lat_range * PADDING_COEFF
    pad_lng = lng_range * PADDING_COEFF

    lat_range_padded = lat_range + 2 * pad_lat
    lng_range_padded = lng_range + 2 * pad_lng

    plot_height = lat_range_padded * aspect_ratio
    plot_width = lng_range_padded

    base_width = 15
    fig_height = base_width * (plot_height / plot_width)

    fig, ax = plt.subplots(figsize=(base_width, fig_height))

    # Set aspect ratio to ensure consistent geometry across different plot types
    ax.set_aspect(aspect_ratio)

    return fig, ax, aspect_ratio, plot_height, plot_width


def get_custom_cmap(threshold):
    """Create custom colormap with transparency gradient.

    Base colormap: Spectral_r (Blue -> Green -> Yellow -> Red)
    Alpha gradient: rapid rise from 0 to 1.0 at the low end.
    """
    ncolors = 256
    color_array = plt.cm.Spectral_r(np.linspace(0, 1, ncolors))

    # Create alpha gradient
    if threshold > 0:
        threshold_idx = int(ncolors * threshold)
        alphas = np.concatenate(
            (
                np.linspace(0, 1.0, threshold_idx),  # Rapid rise from 0 to 1.0
                np.ones(ncolors - threshold_idx),  # Stable high opacity (1.0)
            )
        )
        color_array[:, -1] = alphas
    elif threshold == 0:
        color_array[:, -1] = 1.0  # Fully opaque
    else:
        raise ValueError("Threshold must be non-negative.")

    return LinearSegmentedColormap.from_list(
        name="Spectral_r_alpha", colors=color_array
    )
