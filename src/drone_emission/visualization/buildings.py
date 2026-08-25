import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point
from drone_emission.utils.coordinates import gcj02_to_wgs84
from drone_emission.utils.tools import get_epsg_code, get_epsg_from_bounds

from ._utils import (
    _add_title_suffix,
    add_basemap_and_boundaries,
    finalize_plot,
    get_custom_cmap,
    plot_boundary_box,
    saved_tight_axis_report,
    setup_figure,
    write_figure_report,
)


MERCHANT_MAP_DEGREE_SIZE = 0.01
MERCHANT_MAP_RANDOM_SEED = 20260703
MERCHANT_MAP_TOP_SALES_FRACTION = 0.01
MERCHANT_MAP_ORIGIN_TOLERANCE_DEG = 1e-6
MERCHANT_MAP_OD_LINE_COUNT = 15
MERCHANT_MAP_HEIGHT_BINS = 6


def plot_building_height_merchant_map(
    buildings_gdf,
    stores_df,
    region_name,
    region_name_cn,
    od_df=None,
    road_graph=None,
    save_path=None,
    figsize=(10, 6),
    for_latex=False,
    plot_title=False,
    lat_span=None,
    compact=False,
):
    """Show building footprints colored by height around a top-sales merchant.

    Args:
        buildings_gdf: GeoDataFrame with building geometries and a height column.
        stores_df: DataFrame with lat, lng, and month_sales columns.
        od_df: Optional DataFrame with OD demand columns from_lat, from_lng,
            to_lat, to_lng, and annual_sales.
        road_graph: Optional SparseGraph used to draw local roadnet background.
        region_name: English region name.
        region_name_cn: Chinese region name for basemap and boundaries.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        lat_span: Optional latitude span in degrees.
        compact: If True, use styling and padding for a narrow paper panel.
    """
    required_store_cols = {"lat", "lng", "month_sales"}
    missing_store_cols = required_store_cols - set(stores_df.columns)
    if missing_store_cols:
        print(
            "Building height merchant map skipped - stores data missing "
            f"columns {sorted(missing_store_cols)}"
        )
        return

    if "height" not in buildings_gdf.columns:
        print(
            "Building height merchant map skipped - buildings data missing "
            "height column"
        )
        return

    valid_stores = stores_df.dropna(subset=["lat", "lng", "month_sales"]).copy()
    valid_stores["month_sales"] = pd.to_numeric(
        valid_stores["month_sales"], errors="coerce"
    )
    valid_stores = valid_stores.dropna(subset=["month_sales"])
    if valid_stores.empty:
        print("Building height merchant map skipped - no valid merchant sales rows")
        return

    top_n = max(
        1,
        int(np.ceil(len(valid_stores) * MERCHANT_MAP_TOP_SALES_FRACTION)),
    )
    top_merchants = valid_stores.nlargest(top_n, "month_sales")
    merchant = top_merchants.sample(
        n=1,
        random_state=MERCHANT_MAP_RANDOM_SEED,
    ).iloc[0]
    merchant_lng, merchant_lat = gcj02_to_wgs84(
        float(merchant["lng"]),
        float(merchant["lat"]),
    )

    lat_span = MERCHANT_MAP_DEGREE_SIZE if lat_span is None else float(lat_span)
    if lat_span <= 0:
        print("Building height merchant map skipped - lat_span must be positive")
        return
    lng_span = MERCHANT_MAP_DEGREE_SIZE
    boundaries_dict = {
        "min_lng": merchant_lng - lng_span / 2,
        "max_lng": merchant_lng + lng_span / 2,
        "min_lat": merchant_lat - lat_span / 2,
        "max_lat": merchant_lat + lat_span / 2,
    }

    if buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {buildings_gdf.crs} to EPSG:4326...")
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    buildings_in_frame = buildings_gdf.cx[
        boundaries_dict["min_lng"]:boundaries_dict["max_lng"],
        boundaries_dict["min_lat"]:boundaries_dict["max_lat"],
    ].copy()
    if buildings_in_frame.empty:
        print(
            "Building height merchant map skipped - no buildings in "
            f"{lng_span:.3f} x {lat_span:.3f} degree frame"
        )
        return

    buildings_in_frame["height"] = pd.to_numeric(
        buildings_in_frame["height"], errors="coerce"
    )
    buildings_in_frame = buildings_in_frame.dropna(subset=["height"])
    if buildings_in_frame.empty:
        print(
            "Building height merchant map skipped - no valid building heights "
            "in frame"
        )
        return

    height_cap = buildings_in_frame["height"].quantile(0.95)
    if not np.isfinite(height_cap) or height_cap <= 0:
        print(
            "Building height merchant map skipped - invalid building heights "
            "in frame"
        )
        return

    buildings_in_frame["_height_capped_m"] = buildings_in_frame["height"].clip(
        upper=height_cap
    )
    height_max = float(buildings_in_frame["_height_capped_m"].max())
    height_bins = np.linspace(0.0, height_max, MERCHANT_MAP_HEIGHT_BINS + 1)
    if np.unique(height_bins).size < 2:
        print(
            "Building height merchant map skipped - invalid segmented "
            "height bins"
        )
        return

    fig, ax = plt.subplots(figsize=figsize)
    if compact:
        ax.set_position([0.0, 0.19, 1.0, 0.81])
    avg_lat = (boundaries_dict["min_lat"] + boundaries_dict["max_lat"]) / 2
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ax.set_facecolor("#f7f7f5")

    road_edge_count = 0
    if road_graph is None:
        print("Building height merchant map roadnet background skipped - missing roadnet")
    else:
        edge_coords = road_graph.get_local_edges(boundaries_dict, margin_deg=0.0)
        road_edge_count = len(edge_coords)
        if road_edge_count == 0:
            print(
                "Building height merchant map roadnet background skipped - "
                "no road edges in frame"
            )
        else:
            segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords]
            roads = LineCollection(
                segments,
                colors="#aeb4b9",
                linewidths=0.45 if compact else 0.75,
                alpha=0.6 if compact else 0.75,
                zorder=2,
            )
            ax.add_collection(roads)

    base_cmap = plt.get_cmap("Spectral_r")
    cmap = ListedColormap(
        base_cmap(np.linspace(0.12, 0.92, MERCHANT_MAP_HEIGHT_BINS))
    )
    norm = BoundaryNorm(height_bins, cmap.N, clip=True)
    buildings_in_frame.plot(
        ax=ax,
        column="_height_capped_m",
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        alpha=0.92,
        zorder=10,
        legend=False,
    )
    if compact:
        cbar_ax = fig.add_axes([0.12, 0.075, 0.76, 0.025])
    else:
        cbar_ax = ax.inset_axes(
            [0.18, -0.24, 0.64, 0.035],
            transform=ax.transAxes,
        )
    cbar_ticks = np.linspace(0.0, height_max, 5)
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        boundaries=height_bins,
        ticks=cbar_ticks,
        orientation="horizontal",
    )
    cbar_label = "Building height (m; 95th pct cap)" if compact else (
        "Building height (m, capped at 95th pct)"
    )
    cbar.set_label(
        cbar_label,
        fontsize=6.5 if compact else 13,
        labelpad=1.0 if compact else 4.0,
    )
    cbar.ax.tick_params(
        labelsize=5.5 if compact else 12,
        length=1.5 if compact else 3.5,
        pad=1.0 if compact else 3.5,
    )

    od_line_count = 0
    if od_df is None:
        print("Building height merchant map OD overlay skipped - missing OD data")
    else:
        required_od_cols = {
            "from_lat",
            "from_lng",
            "to_lat",
            "to_lng",
            "annual_sales",
        }
        missing_od_cols = required_od_cols - set(od_df.columns)
        if missing_od_cols:
            print(
                "Building height merchant map OD overlay skipped - OD data "
                f"missing columns {sorted(missing_od_cols)}"
            )
        else:
            od_plot = od_df[list(required_od_cols)].copy()
            for col in required_od_cols:
                od_plot[col] = pd.to_numeric(od_plot[col], errors="coerce")
            od_plot = od_plot.dropna(subset=list(required_od_cols))
            same_origin = (
                (od_plot["from_lat"] - merchant_lat).abs()
                <= MERCHANT_MAP_ORIGIN_TOLERANCE_DEG
            ) & (
                (od_plot["from_lng"] - merchant_lng).abs()
                <= MERCHANT_MAP_ORIGIN_TOLERANCE_DEG
            )
            in_frame = (
                od_plot["from_lng"].between(
                    boundaries_dict["min_lng"],
                    boundaries_dict["max_lng"],
                )
                & od_plot["from_lat"].between(
                    boundaries_dict["min_lat"],
                    boundaries_dict["max_lat"],
                )
                & od_plot["to_lng"].between(
                    boundaries_dict["min_lng"],
                    boundaries_dict["max_lng"],
                )
                & od_plot["to_lat"].between(
                    boundaries_dict["min_lat"],
                    boundaries_dict["max_lat"],
                )
            )
            top_od = (
                od_plot[same_origin & in_frame]
                .nlargest(MERCHANT_MAP_OD_LINE_COUNT, "annual_sales")
                .copy()
            )
            if top_od.empty:
                print(
                    "Building height merchant map OD overlay skipped - no "
                    "selected-merchant OD pairs fully inside frame"
                )
            else:
                sales_min = float(top_od["annual_sales"].min())
                sales_max = float(top_od["annual_sales"].max())
                for _, row in top_od.iterrows():
                    if sales_max == sales_min:
                        line_width = 3.25 if compact else 2.5
                    else:
                        fraction = (
                            (float(row["annual_sales"]) - sales_min)
                            / (sales_max - sales_min)
                        )
                        if compact:
                            line_width = 2.5 + fraction * 1.5
                        else:
                            line_width = 4.0 + fraction * 6.0
                    ax.plot(
                        [row["from_lng"], row["to_lng"]],
                        [row["from_lat"], row["to_lat"]],
                        color="#005f9e",
                        linewidth=line_width,
                        alpha=0.8 if compact else 0.88,
                        solid_capstyle="round",
                        zorder=25,
                    )
                od_line_count = len(top_od)

    ax.scatter(
        merchant_lng,
        merchant_lat,
        c="#ffd400",
        s=95 if compact else 260,
        marker="*",
        edgecolors="black",
        linewidths=0.6 if compact else 1.2,
        zorder=30,
        label="Merchant",
    )
    if od_line_count:
        ax.plot(
            [],
            [],
            color="#005f9e",
            linewidth=3.25 if compact else 5,
            label="OD demands",
        )

    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(
        loc="center" if compact else "upper center",
        bbox_to_anchor=(0.5, 0.145) if compact else (0.5, -0.08),
        bbox_transform=fig.transFigure if compact else ax.transAxes,
        borderaxespad=0.0,
        fontsize=6.5 if compact else 12,
        frameon=False,
        ncol=2 if od_line_count else 1,
        markerscale=1.0 if compact else 1.4,
        handlelength=1.8 if compact else 2.4,
        handletextpad=0.4 if compact else 0.7,
        columnspacing=1.0 if compact else 1.6,
    )
    legend.set_zorder(100)

    if plot_title:
        ax.text(
            0.02,
            0.98,
            (
                f"Frame: {lng_span:.3f} deg x {lat_span:.3f} deg\n"
                f"Buildings: {len(buildings_in_frame):,}\n"
                f"Road edges: {road_edge_count:,}\n"
                f"OD lines: {od_line_count}\n"
                f"Merchant monthly sales: {merchant['month_sales']:.0f}"
            ),
            transform=ax.transAxes,
            fontsize=6.5 if compact else 10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            zorder=100,
        )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".pdf"):
            ax.set_rasterization_zorder(12)
        fig.canvas.draw()
        report_path = save_path.rsplit(".", 1)[0] + "_report.json"
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        report = {
            "figure_name": "shanghai_building_height_merchant",
            "figure_path": save_path,
            "layout_mode": "compact" if compact else "full",
            "native_size_inches": [float(figsize[0]), float(figsize[1])],
            "configured_lng_span_degrees": float(lng_span),
            "configured_lat_span_degrees": float(lat_span),
            "od_demands": {
                "count": int(od_line_count),
                "selection_limit": MERCHANT_MAP_OD_LINE_COUNT,
                "configured_line_width_range_points": (
                    [2.5, 4.0] if compact else [4.0, 10.0]
                ),
                "equal_sales_line_width_points": 3.25 if compact else 2.5,
                "alpha": 0.8 if compact else 0.88,
                "scaling": "linear min-max within selected OD demands",
            },
            "coordinate_convention": (
                "bbox_saved_tight_fraction values are fractions of the saved "
                "page after bbox_inches='tight'."
            ),
            "map": {
                "axis": saved_tight_axis_report(
                    fig,
                    ax,
                    renderer=renderer,
                    tight_bbox=tight_bbox,
                    pad_inches=0.02 if compact else None,
                ),
                "lng_range": [
                    float(boundaries_dict["min_lng"]),
                    float(boundaries_dict["max_lng"]),
                ],
                "lat_range": [
                    float(boundaries_dict["min_lat"]),
                    float(boundaries_dict["max_lat"]),
                ],
                "merchant": {
                    "lng": float(merchant_lng),
                    "lat": float(merchant_lat),
                    "month_sales": float(merchant["month_sales"]),
                },
            },
            "colorbar": {
                "axis": saved_tight_axis_report(
                    fig,
                    cbar.ax,
                    renderer=renderer,
                    tight_bbox=tight_bbox,
                    pad_inches=0.02 if compact else None,
                ),
                "label": cbar_label,
                "orientation": "horizontal",
            },
        }
        if legend is not None:
            legend_bbox = legend.get_window_extent(renderer).transformed(
                fig.dpi_scale_trans.inverted()
            )
            pad_inches = (
                0.02
                if compact
                else plt.rcParams.get("savefig.pad_inches", 0.1)
            )
            tight_x0 = tight_bbox.x0 - pad_inches
            tight_y0 = tight_bbox.y0 - pad_inches
            tight_width = tight_bbox.width + 2 * pad_inches
            tight_height = tight_bbox.height + 2 * pad_inches
            saved_fraction = {
                "x0": float((legend_bbox.x0 - tight_x0) / tight_width),
                "y0": float((legend_bbox.y0 - tight_y0) / tight_height),
                "x1": float((legend_bbox.x1 - tight_x0) / tight_width),
                "y1": float((legend_bbox.y1 - tight_y0) / tight_height),
            }
            saved_fraction["width"] = saved_fraction["x1"] - saved_fraction["x0"]
            saved_fraction["height"] = saved_fraction["y1"] - saved_fraction["y0"]
            report["legend"] = {
                "bbox_saved_tight_fraction": saved_fraction,
                "labels": [text.get_text() for text in legend.get_texts()],
            }
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02 if compact else None,
        )
        write_figure_report(report_path, report)
        print(f"Building height merchant map report saved to {report_path}")
        print(f"Building height merchant map saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_building_footprint_heatmap(
    buildings_gdf,
    stores_df,
    region_name,
    region_name_cn,
    save_path=None,
    figsize=(6.4, 3.2),
    for_latex=False,
    plot_title=False,
    lat_span=None,
    compact=False,
):
    """Show building footprints colored by height around a top-sales merchant.

    Args:
        buildings_gdf: GeoDataFrame with building geometries and a height column.
        stores_df: DataFrame with lat, lng, and month_sales columns.
        region_name: English region name.
        region_name_cn: Chinese region name.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        lat_span: Optional latitude span in degrees.
        compact: If True, use paper-panel export padding.
    """
    required_store_cols = {"lat", "lng", "month_sales"}
    missing_store_cols = required_store_cols - set(stores_df.columns)
    if missing_store_cols:
        print(
            "Building footprint heatmap skipped - stores data missing "
            f"columns {sorted(missing_store_cols)}"
        )
        return

    if "height" not in buildings_gdf.columns:
        print(
            "Building footprint heatmap skipped - buildings data missing "
            "height column"
        )
        return

    valid_stores = stores_df.dropna(subset=["lat", "lng", "month_sales"]).copy()
    valid_stores["month_sales"] = pd.to_numeric(
        valid_stores["month_sales"], errors="coerce"
    )
    valid_stores = valid_stores.dropna(subset=["month_sales"])
    if valid_stores.empty:
        print("Building footprint heatmap skipped - no valid merchant sales rows")
        return

    top_n = max(
        1,
        int(np.ceil(len(valid_stores) * MERCHANT_MAP_TOP_SALES_FRACTION)),
    )
    top_merchants = valid_stores.nlargest(top_n, "month_sales")
    merchant = top_merchants.sample(
        n=1,
        random_state=MERCHANT_MAP_RANDOM_SEED,
    ).iloc[0]
    merchant_lng, merchant_lat = gcj02_to_wgs84(
        float(merchant["lng"]),
        float(merchant["lat"]),
    )

    lat_span = MERCHANT_MAP_DEGREE_SIZE if lat_span is None else float(lat_span)
    if lat_span <= 0:
        print("Building footprint heatmap skipped - lat_span must be positive")
        return
    avg_lat = merchant_lat
    figure_ratio = float(figsize[0]) / float(figsize[1])
    lng_span = figure_ratio * lat_span / np.cos(np.radians(avg_lat))
    boundaries_dict = {
        "min_lng": merchant_lng - lng_span / 2,
        "max_lng": merchant_lng + lng_span / 2,
        "min_lat": merchant_lat - lat_span / 2,
        "max_lat": merchant_lat + lat_span / 2,
    }

    if buildings_gdf.crs is not None and buildings_gdf.crs.to_epsg() != 4326:
        print(f"Converting CRS from {buildings_gdf.crs} to EPSG:4326...")
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    buildings_in_frame = buildings_gdf.cx[
        boundaries_dict["min_lng"]:boundaries_dict["max_lng"],
        boundaries_dict["min_lat"]:boundaries_dict["max_lat"],
    ].copy()
    if buildings_in_frame.empty:
        print("Building footprint heatmap skipped - no buildings in frame")
        return

    buildings_in_frame["height"] = pd.to_numeric(
        buildings_in_frame["height"], errors="coerce"
    )
    buildings_in_frame = buildings_in_frame.dropna(subset=["height"])
    if buildings_in_frame.empty:
        print("Building footprint heatmap skipped - no valid building heights in frame")
        return

    height_cap = buildings_in_frame["height"].quantile(0.95)
    if not np.isfinite(height_cap) or height_cap <= 0:
        print("Building footprint heatmap skipped - invalid building heights in frame")
        return

    buildings_in_frame["_height_capped_m"] = buildings_in_frame["height"].clip(
        upper=height_cap
    )
    height_max = float(buildings_in_frame["_height_capped_m"].max())
    height_bins = np.linspace(0.0, height_max, MERCHANT_MAP_HEIGHT_BINS + 1)
    if np.unique(height_bins).size < 2:
        print("Building footprint heatmap skipped - invalid segmented height bins")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ax.set_facecolor("#f7f7f5")

    base_cmap = plt.get_cmap("Spectral_r")
    cmap = ListedColormap(
        base_cmap(np.linspace(0.12, 0.92, MERCHANT_MAP_HEIGHT_BINS))
    )
    norm = BoundaryNorm(height_bins, cmap.N, clip=True)
    buildings_in_frame.plot(
        ax=ax,
        column="_height_capped_m",
        cmap=cmap,
        norm=norm,
        edgecolor="none",
        alpha=0.94,
        zorder=10,
        legend=False,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if plot_title:
        ax.text(
            0.02,
            0.98,
            (
                f"{region_name} building footprints\n"
                f"Buildings: {len(buildings_in_frame):,}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            zorder=100,
        )

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".pdf"):
            ax.set_rasterization_zorder(12)
        fig.canvas.draw()
        report_path = save_path.rsplit(".", 1)[0] + "_report.json"
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        report = {
            "figure_name": "shanghai_building_footprint_heatmap",
            "figure_path": save_path,
            "layout_mode": "compact" if compact else "full",
            "native_size_inches": [float(figsize[0]), float(figsize[1])],
            "configured_lat_span_degrees": float(lat_span),
            "save_padding_inches": 0.02 if compact else 0.0,
            "coordinate_convention": (
                "bbox_saved_tight_fraction values are fractions of the saved "
                "page after bbox_inches='tight'."
            ),
            "map": {
                "axis": saved_tight_axis_report(
                    fig,
                    ax,
                    renderer=renderer,
                    tight_bbox=tight_bbox,
                    pad_inches=0.02 if compact else 0.0,
                ),
                "lng_range": [
                    float(boundaries_dict["min_lng"]),
                    float(boundaries_dict["max_lng"]),
                ],
                "lat_range": [
                    float(boundaries_dict["min_lat"]),
                    float(boundaries_dict["max_lat"]),
                ],
                "building_count": int(len(buildings_in_frame)),
                "height_cap_m": float(height_cap),
                "merchant_center": {
                    "lng": float(merchant_lng),
                    "lat": float(merchant_lat),
                    "month_sales": float(merchant["month_sales"]),
                },
            },
        }
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02 if compact else 0,
        )
        write_figure_report(report_path, report)
        print(f"Building footprint heatmap report saved to {report_path}")
        print(f"Building footprint heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
