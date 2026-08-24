import json
import os

import contextily as ctx
import cpca
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from drone_emission.utils.config import get_cache_dir
from drone_emission.utils.coordinates import gcj02_to_wgs84

# Global padding coefficient for plotting
PADDING_COEFF = 0.1

# Global flag for cache configuration
_CONTEXTILY_CACHE_CONFIGURED = False


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
    cache_file = os.path.join(get_cache_dir("boundaries"), f"{adcode}.json")

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


def plot_geojson_boundaries(
    ax,
    geojson_data,
    zorder=15,
    color="gray",
    linewidth=2,
    alpha=0.8,
):
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
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=zorder,
                )
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    ring_wgs = [gcj02_to_wgs84(pt[0], pt[1]) for pt in ring]
                    xs, ys = zip(*ring_wgs)
                    ax.plot(
                        xs,
                        ys,
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha,
                        zorder=zorder,
                    )


def configure_contextily_cache():
    global _CONTEXTILY_CACHE_CONFIGURED
    if not _CONTEXTILY_CACHE_CONFIGURED:
        try:
            tiles_dir = get_cache_dir("tiles")
            ctx.set_cache_dir(tiles_dir)
            print(f"Contextily cache configured at: {tiles_dir}")
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
    ax,
    city_name_cn,
    boundaries_dict=None,
    zoom=14,
    alpha=0.6,
    boundary_color="gray",
    boundary_linewidth=2,
    boundary_alpha=0.8,
    attribution_size=8,
    attribution_text=None,
    attribution_wrap=True,
    basemap_provider="openstreetmap",
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

        providers = {
            "openstreetmap": (
                ctx.providers.OpenStreetMap.Mapnik,
                "(C) OpenStreetMap contributors",
            ),
            "cartodb_positron": (
                ctx.providers.CartoDB.Positron,
                "(C) OpenStreetMap contributors (C) CARTO",
            ),
        }
        if basemap_provider not in providers:
            raise ValueError(f"Unsupported basemap provider: {basemap_provider}")
        basemap_source, default_attribution = providers[basemap_provider]

        # Add basemap with higher resolution (zoom level) and reduced opacity
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=basemap_source,
            zoom=zoom,  # Higher zoom = higher resolution
            alpha=alpha,  # Reduced opacity for better data visibility
            attribution=False if not attribution_wrap else attribution_text,
            attribution_size=attribution_size,
            zorder=0,  # Map at bottom
        )
        if not attribution_wrap:
            ax.text(
                0.005,
                0.005,
                attribution_text or default_attribution,
                transform=ax.transAxes,
                fontsize=attribution_size,
                va="bottom",
                ha="left",
                wrap=False,
                path_effects=[
                    patheffects.withStroke(linewidth=0.6, foreground="white")
                ],
                zorder=100,
            )
        print(
            "Basemap added successfully "
            f"(provider={basemap_provider}, zoom={zoom}, alpha={alpha})"
        )
    except Exception as e:
        print(f"Could not add basemap: {e}")

    # Add administrative boundaries (counties/districts)
    adcode = get_adcode(city_name_cn)
    if adcode:
        geojson_data = fetch_boundaries(adcode)
        plot_geojson_boundaries(
            ax,
            geojson_data,
            zorder=15,
            color=boundary_color,
            linewidth=boundary_linewidth,
            alpha=boundary_alpha,
        )
    else:
        print(f"No adcode found for {city_name_cn}, skipping boundaries.")


def _add_title_suffix(save_path):
    """Insert '_t' before the file extension in save_path."""
    if not save_path:
        return save_path
    base, ext = save_path.rsplit('.', 1)
    return f"{base}_t.{ext}"


def _json_safe_scalar(value):
    """Return a scalar that can be written with strict JSON settings."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_safe_array(values):
    """Return a JSON-safe list from a numeric sequence."""
    return [_json_safe_scalar(value) for value in values]


def _bbox_report(bbox):
    """Return JSON-safe bbox coordinates."""
    return {
        "x0": _json_safe_scalar(bbox.x0),
        "y0": _json_safe_scalar(bbox.y0),
        "x1": _json_safe_scalar(bbox.x1),
        "y1": _json_safe_scalar(bbox.y1),
        "width": _json_safe_scalar(bbox.width),
        "height": _json_safe_scalar(bbox.height),
    }


def saved_tight_axis_report(fig, ax, renderer=None, tight_bbox=None, pad_inches=None):
    """Return axis position in both figure and saved tight-page coordinates."""
    if renderer is None:
        renderer = fig.canvas.get_renderer()
    if tight_bbox is None:
        tight_bbox = fig.get_tightbbox(renderer)
    if pad_inches is None:
        pad_inches = plt.rcParams.get("savefig.pad_inches", 0.1)
    tight_x0 = tight_bbox.x0 - pad_inches
    tight_y0 = tight_bbox.y0 - pad_inches
    tight_width = tight_bbox.width + 2 * pad_inches
    tight_height = tight_bbox.height + 2 * pad_inches
    axis_inches = ax.get_window_extent(renderer).transformed(
        fig.dpi_scale_trans.inverted()
    )
    saved_fraction = {
        "x0": (axis_inches.x0 - tight_x0) / tight_width,
        "y0": (axis_inches.y0 - tight_y0) / tight_height,
        "x1": (axis_inches.x1 - tight_x0) / tight_width,
        "y1": (axis_inches.y1 - tight_y0) / tight_height,
    }
    saved_fraction["width"] = saved_fraction["x1"] - saved_fraction["x0"]
    saved_fraction["height"] = saved_fraction["y1"] - saved_fraction["y0"]
    return {
        "bbox_figure_fraction": _bbox_report(ax.get_position()),
        "saved_tight_pad_inches": _json_safe_scalar(pad_inches),
        "bbox_saved_tight_fraction": {
            key: _json_safe_scalar(value)
            for key, value in saved_fraction.items()
        },
        "xlim_data": _json_safe_array(ax.get_xlim()),
        "ylim_data": _json_safe_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _json_safe_array(ax.get_xticks()),
        "yticks_data": _json_safe_array(ax.get_yticks()),
    }


def write_figure_report(report_path, payload):
    """Write a JSON report for figure composition metadata."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def finalize_plot(
    fig,
    ax,
    save_path,
    title,
    stats_text,
    boundaries_dict=None,
    for_latex=False,
    plot_title=False,
    show_geo_axes=True,
    show_grid=True,
    show_frame=True,
    before_save=None,
    save_tight=True,
):
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
        show_geo_axes: If False, hide longitude/latitude labels and ticks.
        show_grid: If False, skip the plot grid.
        show_frame: If False, hide the axes frame/spines.
        before_save: Optional callback receiving the final save path before savefig.
        save_tight: If True, crop output to the tight content bounding box.
    """
    # Set labels
    if show_geo_axes:
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False,
            labelbottom=False,
        )

    if plot_title:
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

    # Add grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="--", color="white")
    else:
        ax.grid(False)

    if not show_frame:
        for spine in ax.spines.values():
            spine.set_visible(False)

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
        if before_save is not None:
            fig.canvas.draw()
            before_save(save_path)
        save_kwargs = {"bbox_inches": "tight"} if save_tight else {}
        plt.savefig(save_path, dpi=300, **save_kwargs)
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
