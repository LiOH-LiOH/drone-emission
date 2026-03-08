import json
import os

import matplotlib.pyplot as plt
import numpy as np
import requests

# Project root directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _fetch_china_boundaries():
    """Fetch China provincial boundaries from DataV API."""
    # China adcode is 100000
    adcode = "100000"
    cache_dir = os.path.join(_PROJECT_ROOT, "cache", "boundaries")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{adcode}.json")

    # Try to load from cache
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")

    # Download if not in cache
    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    try:
        print("Downloading China boundaries...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        # Save to cache
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        return data
    except Exception as e:
        print(f"Failed to fetch boundaries: {e}")
        return None


def _plot_china_boundaries(ax, geojson_data):
    """Plot China boundaries on the axes."""
    if not geojson_data:
        return

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates")

        def plot_polygon(rings):
            for ring in rings:
                if isinstance(ring[0][0], list):
                    # Nested polygon
                    plot_polygon(ring)
                else:
                    xs, ys = zip(*[(pt[0], pt[1]) for pt in ring])
                    ax.plot(xs, ys, color="gray", linewidth=0.5, alpha=0.8, zorder=1)

        if geom_type == "Polygon":
            plot_polygon(coords)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                plot_polygon(polygon)


def plot_regions_on_china_map(
    region_keys: list = None,
    regions_json_path: str = None,
    save_path: str = None,
    figsize: tuple = (12, 10),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot study regions as scatter points on a China map.

    Args:
        region_keys: List of region keys to plot. If None, plots all regions.
        regions_json_path: Path to regions.template.json. If None, uses default path.
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size tuple (width, height).
        for_latex: If True, skip title for LaTeX export and save as PDF.
    """
    # Default path to regions.template.json
    if regions_json_path is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        regions_json_path = os.path.join(project_root, "regions.template.json")

    # Load regions data
    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    # Extract center coordinates for each region
    region_names = []
    lats = []
    lngs = []

    for region_key, region_info in regions_data.items():
        # Filter by region_keys if provided
        if region_keys is not None and region_key not in region_keys:
            continue

        boundaries = region_info.get("boundaries_dict", {})
        if not boundaries:
            continue

        center_lat = (boundaries["min_lat"] + boundaries["max_lat"]) / 2
        center_lng = (boundaries["min_lng"] + boundaries["max_lng"]) / 2

        region_names.append(region_info.get("region_name_en", region_key))
        lats.append(center_lat)
        lngs.append(center_lng)

    lats = np.array(lats)
    lngs = np.array(lngs)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # China approximate boundaries for context
    china_lat_range = (18, 54)
    china_lng_range = (73, 135)

    # Fetch and plot China base map
    china_geojson = _fetch_china_boundaries()
    _plot_china_boundaries(ax, china_geojson)

    # Plot scatter points
    ax.scatter(
        lngs,
        lats,
        c="red",
        s=100,
        alpha=0.8,
        edgecolors="darkred",
        linewidths=1.5,
        zorder=10,
    )

    # Add region labels
    for name, lat, lng in zip(region_names, lats, lngs):
        ax.annotate(
            name,
            (lng, lat),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="darkblue",
            zorder=11,
        )

    # Set axis limits to show China context
    ax.set_xlim(china_lng_range)
    ax.set_ylim(china_lat_range)

    # Labels
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    if plot_title:
        ax.set_title("Study Regions in China", fontsize=14, fontweight="bold")

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Adjust aspect ratio for geographic data
    avg_lat = np.mean(china_lat_range)
    ax.set_aspect(1.0 / np.cos(np.radians(avg_lat)))

    plt.tight_layout()

    # Save or show
    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            from ._utils import _add_title_suffix
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    return fig, ax
