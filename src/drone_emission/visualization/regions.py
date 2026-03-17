import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    csv_path: str = None,
    save_path: str = None,
    figsize: tuple = (12, 10),
    for_latex: bool = False,
    plot_title: bool = False,
    color_by: str = "max_lifecycle_reduction_pct",
):
    """Plot study regions as scatter points on a China map.

    Supports two data sources: CSV (preferred for large-scale analysis) or
    regions.json (legacy). When using CSV, points are color-coded by the
    specified variable.

    Args:
        region_keys: List of region keys to plot. If None, plots all regions.
        regions_json_path: Path to regions.json (legacy mode).
        csv_path: Path to all_region_stats.csv (preferred for large-scale analysis).
        save_path: Path to save the figure. If None, displays the plot.
        figsize: Figure size tuple (width, height).
        for_latex: If True, skip title for LaTeX export and save as PDF.
        plot_title: If True, show title.
        color_by: Column name to color-code points by (only used with csv_path).
    """
    # Load data from CSV or JSON
    if csv_path is not None:
        df = pd.read_csv(csv_path)
        # Filter to rows with boundary data
        df = df[df[["min_lat", "max_lat", "min_lng", "max_lng"]].notna().all(axis=1)].copy()
        if region_keys is not None:
            df = df[df["region_key"].isin(region_keys)]

        # Compute center coordinates
        lats = ((df["min_lat"] + df["max_lat"]) / 2).values
        lngs = ((df["min_lng"] + df["max_lng"]) / 2).values
        region_names = df["region_name_en"].values
        color_values = df[color_by].values if color_by in df.columns else None
    else:
        # Legacy JSON mode
        if regions_json_path is None:
            regions_json_path = os.path.join(_PROJECT_ROOT, "regions.json")

        with open(regions_json_path, "r", encoding="utf-8") as f:
            regions_data = json.load(f)

        region_names = []
        lats = []
        lngs = []

        for region_key, region_info in regions_data.items():
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
        region_names = np.array(region_names)
        color_values = None

    n_points = len(lats)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # China approximate boundaries for context
    china_lat_range = (18, 54)
    china_lng_range = (73, 135)

    # Fetch and plot China base map
    china_geojson = _fetch_china_boundaries()
    _plot_china_boundaries(ax, china_geojson)

    # Plot scatter points
    if color_values is not None and not np.all(np.isnan(color_values.astype(float))):
        # Color-coded scatter with colorbar
        valid_mask = ~pd.isna(color_values)
        color_vals = color_values.astype(float)

        scatter = ax.scatter(
            lngs[valid_mask],
            lats[valid_mask],
            c=color_vals[valid_mask],
            cmap="RdYlGn",
            s=40 if n_points > 20 else 100,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            zorder=10,
        )
        # Points without color data
        if (~valid_mask).any():
            ax.scatter(
                lngs[~valid_mask],
                lats[~valid_mask],
                c="gray",
                s=20,
                alpha=0.5,
                edgecolors="black",
                linewidths=0.3,
                zorder=9,
            )
        from analysis.constants import VARIABLE_LABELS
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(VARIABLE_LABELS.get(color_by, color_by), fontsize=10)
    else:
        ax.scatter(
            lngs,
            lats,
            c="red",
            s=40 if n_points > 20 else 100,
            alpha=0.8,
            edgecolors="darkred",
            linewidths=1.0,
            zorder=10,
        )

    # Add region labels: all if <=20, otherwise only top/bottom 5 outliers
    if n_points <= 20:
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
    elif color_values is not None:
        # Annotate top/bottom 5 outliers
        valid_mask = ~pd.isna(color_values)
        if valid_mask.sum() > 0:
            valid_idx = np.where(valid_mask)[0]
            valid_vals = color_values.astype(float)[valid_idx]
            sorted_idx = valid_idx[np.argsort(valid_vals)]
            n_annotate = min(5, len(sorted_idx) // 2)
            annotate_idx = np.concatenate([sorted_idx[:n_annotate], sorted_idx[-n_annotate:]])
            for idx in annotate_idx:
                ax.annotate(
                    region_names[idx],
                    (lngs[idx], lats[idx]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
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
