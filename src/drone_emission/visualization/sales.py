from .palette import opacity as paper_alpha

from .prepared_bundle import prepared_plot, prepared_data

from .palette import color as paper_color
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from drone_emission.utils.coordinates import convert_coords_array

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
@prepared_plot()
def plot_merchant_monthly_sales_map(
    stores_df,
    buildings_gdf,
    region_name,
    save_path=None,
    figsize=(1.25, 1.17),
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

    fig = plt.figure(figsize=figsize, facecolor=paper_color("ink.background"))
    ax = fig.add_axes(map_rect)
    ax.set_facecolor(paper_color("sales.fafafa"))
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries["min_lng"], boundaries["max_lng"])
    ax.set_ylim(boundaries["min_lat"], boundaries["max_lat"])
    buildings_in_frame.plot(
        ax=ax,
        facecolor=paper_color("sales.d7d9dc"),
        edgecolor="none",
        alpha=paper_alpha("sales.alpha_0_86"),
        zorder=1,
    )
    merchant_color = paper_color("sales.d94b3f")
    ax.scatter(
        merchants_in_frame["_lng_wgs84"],
        merchants_in_frame["_lat_wgs84"],
        s=merchants_in_frame["_marker_size"],
        c=merchant_color,
        alpha=paper_alpha("sales.alpha_0_88"),
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
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
                        fig,
                        ax,
                        renderer=renderer,
                        tight_bbox=tight_bbox,
                        pad_inches=0.02,
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
        print(f"Merchant monthly sales map saved to {save_path}")
        print(f"Merchant monthly sales map report saved to {report_path}")
    else:
        plt.show()

    plt.close(fig)
