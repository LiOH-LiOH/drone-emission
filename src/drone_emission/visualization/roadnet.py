
from .prepared_bundle import prepared_plot, prepared_data, local_road_edges, shortest_path

from .palette import color as paper_color
import ast
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import os
import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from ._utils import (
    add_basemap_and_boundaries,
    configure_contextily_cache,
    finalize_plot,
    plot_boundary_box,
    saved_tight_axis_report,
    setup_figure,
    _add_title_suffix,
    write_figure_report,
)


ROAD_NETWORK_CONTEXT_BACKGROUND = paper_color("roadnet.eef0eb")
def get_road_network_subfigure_frame(
    center_lng,
    center_lat,
    figsize=(1.25, 1.11),
    lat_span=0.018,
):
    """Return a deterministic north-up frame around an explicit center."""
    center_lng = float(center_lng)
    center_lat = float(center_lat)
    lat_span = float(lat_span)
    if not all(np.isfinite([center_lng, center_lat, lat_span])):
        print("Road network subfigure skipped - frame values must be finite")
        return None
    if lat_span <= 0 or float(figsize[0]) <= 0 or float(figsize[1]) <= 0:
        print("Road network subfigure skipped - frame dimensions must be positive")
        return None
    figure_ratio = float(figsize[0]) / float(figsize[1])
    lng_span = figure_ratio * lat_span / np.cos(np.radians(center_lat))
    return {
        "bounds": {
            "min_lng": center_lng - lng_span / 2,
            "max_lng": center_lng + lng_span / 2,
            "min_lat": center_lat - lat_span / 2,
            "max_lat": center_lat + lat_span / 2,
        },
        "center_lng": center_lng,
        "center_lat": center_lat,
        "lat_span": lat_span,
    }


def _deduplicate_road_edges(edge_coords):
    """Return unique undirected road segments."""
    source_nodes = edge_coords[:, :2]
    target_nodes = edge_coords[:, 2:]
    swap_endpoints = (
        (source_nodes[:, 0] > target_nodes[:, 0])
        | (
            (source_nodes[:, 0] == target_nodes[:, 0])
            & (source_nodes[:, 1] > target_nodes[:, 1])
        )
    )

    undirected_edges = np.empty_like(edge_coords)
    undirected_edges[:, :2] = np.where(
        swap_endpoints[:, None], target_nodes, source_nodes
    )
    undirected_edges[:, 2:] = np.where(
        swap_endpoints[:, None], source_nodes, target_nodes
    )
    return np.unique(undirected_edges, axis=0)


def _add_context_road_lines(ax, edge_coords, compact):
    """Draw road lines with the context-map styling."""
    segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords]
    if compact:
        ax.add_collection(
            LineCollection(
                segments,
                colors=paper_color("ink.background"),
                linewidths=1.05,
                alpha=0.92,
                zorder=15,
            )
        )
        roads = LineCollection(
            segments,
            colors=paper_color("roadnet.3f4852"),
            linewidths=0.5,
            alpha=0.98,
            zorder=20,
        )
    else:
        roads = LineCollection(
            segments,
            colors=paper_color("roadnet.4f5963"),
            linewidths=0.55,
            alpha=0.95,
            zorder=10,
        )
    ax.add_collection(roads)


@prepared_plot(omit=('graph',))
def plot_road_network_subfigure(
    graph,
    region_name,
    center_lng,
    center_lat,
    save_path=None,
    figsize=(6.4, 3.2),
    for_latex=False,
    plot_title=False,
    lat_span=None,
    compact=False,
    buildings_gdf=None,
    figure_name="road_network_subfigure",
):
    """Show a road network over subdued building footprints.

    Args:
        graph: SparseGraph with get_local_edges support.
        region_name: English region name.
        center_lng: Longitude of the fixed map center in WGS84.
        center_lat: Latitude of the fixed map center in WGS84.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        lat_span: Optional latitude span in degrees.
        compact: If True, use paper-panel styling and export padding.
        buildings_gdf: Optional GeoDataFrame with building geometries.
        figure_name: Stable figure identifier written to the report.
    """
    lat_span = 0.018 if lat_span is None else float(lat_span)
    frame = get_road_network_subfigure_frame(
        center_lng,
        center_lat,
        figsize=figsize,
        lat_span=lat_span,
    )
    if frame is None:
        return
    boundaries_dict = frame["bounds"]
    avg_lat = frame["center_lat"]

    edge_coords = local_road_edges(graph, boundaries_dict, margin_deg=0.0)
    if len(edge_coords) == 0:
        print("Road network subfigure skipped - no road edges in frame")
        return

    unique_edges = _deduplicate_road_edges(edge_coords)
    displayed_edges = unique_edges if compact else edge_coords

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    aspect_ratio = 1.0 / np.cos(np.radians(avg_lat))
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    background_color = (
        ROAD_NETWORK_CONTEXT_BACKGROUND if compact else paper_color("roadnet.f7f7f5")
    )
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    building_color = paper_color("roadnet.c8c1b3")
    building_alpha = 0.65
    building_count = 0
    building_context_available = buildings_gdf is not None
    if buildings_gdf is None:
        print(
            "Road network subfigure building background skipped - missing "
            "building geometry"
        )
    else:
        if (
            buildings_gdf.crs is not None
            and buildings_gdf.crs.to_epsg() != 4326
        ):
            print(f"Converting CRS from {buildings_gdf.crs} to EPSG:4326...")
            buildings_gdf = buildings_gdf.to_crs(epsg=4326)
        buildings_in_frame = buildings_gdf.cx[
            boundaries_dict["min_lng"]:boundaries_dict["max_lng"],
            boundaries_dict["min_lat"]:boundaries_dict["max_lat"],
        ].copy()
        building_count = len(buildings_in_frame)
        if building_count == 0:
            print(
                "Road network subfigure building background skipped - no "
                "buildings in frame"
            )
        else:
            buildings_in_frame.plot(
                ax=ax,
                color=building_color,
                edgecolor="none",
                alpha=building_alpha,
                zorder=2,
            )

    _add_context_road_lines(ax, displayed_edges, compact=compact)

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

    if plot_title:
        ax.text(
            0.02,
            0.98,
            (
                f"{region_name} road network\n"
                f"Buildings: {building_count:,}\n"
                f"Edges: {len(edge_coords):,}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=paper_color("ink.background"), alpha=0.85),
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
        map_report = {
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
            "road_edge_count": int(len(edge_coords)),
            "undirected_road_segment_count": int(len(unique_edges)),
            "background_color": background_color,
            "selected_center": {
                "lng": frame["center_lng"],
                "lat": frame["center_lat"],
                "selection": "manual_fixed_center",
                "orientation": "north_up",
            },
            "building_context": {
                "available": bool(building_context_available),
                "building_count": int(building_count),
                "color": building_color,
                "alpha": float(building_alpha),
                "fallback_to_road_only": bool(building_count == 0),
            },
        }
        report = {
            "figure_name": figure_name,
            "region_name": region_name,
            "variant": "context",
            "layout_mode": "compact" if compact else "full",
            "native_size_inches": [float(figsize[0]), float(figsize[1])],
            "configured_lat_span_degrees": float(lat_span),
            "save_padding_inches": 0.02 if compact else 0.0,
            "coordinate_convention": (
                "bbox_saved_tight_fraction values are fractions of the saved "
                "page after bbox_inches='tight'."
            ),
            "map": map_report,
        }
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02 if compact else 0,
            facecolor=background_color,
        )
        write_figure_report(report_path, report)
        print(f"Road network subfigure report saved to {report_path}")
        print(f"Road network subfigure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _draw_distance_panel(
    ax,
    sample,
    graph,
    show_legend=False,
    show_stats_box=True,
    show_basemap=True,
    padding=0.003,
    target_aspect=1.0,
    rotate_air_diagonal=False,
    compact=False,
    route_legend_only=False,
    show_endpoint_labels=False,
):
    """Draw one distance comparison sample onto *ax*."""
    from matplotlib.collections import LineCollection
    from ._utils import configure_contextily_cache

    path_result = sample["path_result"]
    origin = sample["origin"]
    destination = sample["destination"]
    straight_dist = sample["straight_distance"]
    road_dist = sample["road_distance"]
    detour = road_dist / straight_dist if straight_dist > 0 else 0

    lats = [origin[0], destination[0]]
    lngs = [origin[1], destination[1]]
    if path_result.get("success") and len(path_result.get("path_coordinates", [])) > 1:
        lats.extend(c[0] for c in path_result["path_coordinates"])
        lngs.extend(c[1] for c in path_result["path_coordinates"])
    local_bounds = {
        "min_lat": min(lats) - padding,
        "max_lat": max(lats) + padding,
        "min_lng": min(lngs) - padding,
        "max_lng": max(lngs) + padding,
    }

    if rotate_air_diagonal:
        lat_center = (local_bounds["min_lat"] + local_bounds["max_lat"]) / 2
        lng_center = (local_bounds["min_lng"] + local_bounds["max_lng"]) / 2
        cos_lat = max(float(np.cos(np.radians(lat_center))), 1e-6)
        target_aspect = max(float(target_aspect), 1e-6)

        def _project(lat, lng):
            return np.array([(lng - lng_center) * cos_lat, lat - lat_center])

        origin_xy = _project(origin[0], origin[1])
        dest_xy = _project(destination[0], destination[1])
        air_vec = dest_xy - origin_xy
        target_angle = np.arctan2(1.0, target_aspect)
        current_angle = (
            np.arctan2(air_vec[1], air_vec[0])
            if np.hypot(air_vec[0], air_vec[1]) > 0
            else target_angle
        )
        theta = target_angle - current_angle
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        def _rotate_xy(xy):
            return np.array([
                xy[0] * cos_t - xy[1] * sin_t,
                xy[0] * sin_t + xy[1] * cos_t,
            ])

        def _transform(lat, lng):
            return _rotate_xy(_project(lat, lng))

        def _inverse_transform(x, y):
            xr = x * cos_t + y * sin_t
            yr = -x * sin_t + y * cos_t
            return yr + lat_center, xr / cos_lat + lng_center

        primary_points = [
            _transform(origin[0], origin[1]),
            _transform(destination[0], destination[1]),
        ]
        if path_result.get("success") and len(path_result.get("path_coordinates", [])) > 1:
            primary_points.extend(
                _transform(lat, lng)
                for lat, lng in path_result["path_coordinates"]
            )
        primary_points = np.vstack(primary_points)
        x_min, y_min = primary_points.min(axis=0)
        x_max, y_max = primary_points.max(axis=0)
        pad = max(max(x_max - x_min, y_max - y_min) * 0.12, padding)
        x_min, x_max = x_min - pad, x_max + pad
        y_min, y_max = y_min - pad, y_max + pad
        x_span = x_max - x_min
        y_span = y_max - y_min
        current_aspect = x_span / y_span if y_span > 0 else target_aspect
        if current_aspect < target_aspect:
            extra = (target_aspect * y_span - x_span) / 2
            x_min -= extra
            x_max += extra
        else:
            extra = (x_span / target_aspect - y_span) / 2
            y_min -= extra
            y_max += extra

        corners = [
            _inverse_transform(x_min, y_min),
            _inverse_transform(x_min, y_max),
            _inverse_transform(x_max, y_min),
            _inverse_transform(x_max, y_max),
        ]
        query_bounds = {
            "min_lat": min(lat for lat, _ in corners),
            "max_lat": max(lat for lat, _ in corners),
            "min_lng": min(lng for _, lng in corners),
            "max_lng": max(lng for _, lng in corners),
        }

        if show_basemap:
            try:
                from matplotlib.transforms import Affine2D

                configure_contextily_cache()
                img, basemap_extent = ctx.bounds2img(
                    query_bounds["min_lng"],
                    query_bounds["min_lat"],
                    query_bounds["max_lng"],
                    query_bounds["max_lat"],
                    zoom=16,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    ll=True,
                )

                def _mercator_to_lng_lat(x, y):
                    radius = 6378137.0
                    lng = np.degrees(x / radius)
                    lat = np.degrees(2 * np.arctan(np.exp(y / radius)) - np.pi / 2)
                    return lng, lat

                west_m, east_m, south_m, north_m = basemap_extent
                west, south = _mercator_to_lng_lat(west_m, south_m)
                east, north = _mercator_to_lng_lat(east_m, north_m)
                sw = _transform(south, west)
                se = _transform(south, east)
                nw = _transform(north, west)
                v_x = se - sw
                v_y = nw - sw
                image_transform = (
                    Affine2D.from_values(
                        v_x[0] / img.shape[1],
                        v_x[1] / img.shape[1],
                        v_y[0] / img.shape[0],
                        v_y[1] / img.shape[0],
                        sw[0],
                        sw[1],
                    )
                    + ax.transData
                )
                ax.imshow(
                    np.flipud(img),
                    extent=(0, img.shape[1], 0, img.shape[0]),
                    origin="lower",
                    transform=image_transform,
                    alpha=0.5,
                    zorder=0,
                )
            except Exception as e:
                print(f"Could not add rotated basemap: {e}")

        edge_coords = local_road_edges(graph, query_bounds)
        if len(edge_coords) > 0:
            segments = [
                [
                    tuple(_transform(ec[0], ec[1])),
                    tuple(_transform(ec[2], ec[3])),
                ]
                for ec in edge_coords
            ]
            network_color = paper_color("roadnet.cfd4d1") if compact else paper_color("ink.hex_ddd")
            network_width = 0.35 if compact else 0.3
            lc = LineCollection(
                segments,
                colors=network_color,
                linewidths=network_width,
                zorder=1,
            )
            ax.add_collection(lc)

        if path_result.get("success") and len(path_result.get("path_coordinates", [])) > 1:
            path_xy = np.array([
                _transform(lat, lng)
                for lat, lng in path_result["path_coordinates"]
            ])
            road_width = 1.5 if compact else 2.5
            ax.plot(path_xy[:, 0], path_xy[:, 1], color=paper_color("roadnet.d95f35"), linewidth=road_width,
                    alpha=0.9, zorder=10, label="E-bike (road)")

        origin_rot = _transform(origin[0], origin[1])
        dest_rot = _transform(destination[0], destination[1])
        drone_width = 1.2 if compact else 2
        marker_size = 22 if compact else 90
        marker_edge_width = 0.7 if compact else 1.5
        ax.plot(
            [origin_rot[0], dest_rot[0]], [origin_rot[1], dest_rot[1]],
            color=paper_color("roadnet.1f77b4"), linewidth=drone_width, linestyle="--", alpha=0.9, zorder=20,
            label="Drone (straight)",
        )
        ax.scatter(origin_rot[0], origin_rot[1], c=paper_color("roadnet.2f6f4e"), s=marker_size, marker="o",
                   zorder=25, edgecolors="white", linewidth=marker_edge_width, label="Merchant")
        ax.scatter(dest_rot[0], dest_rot[1], c=paper_color("roadnet.3f4b64"), s=marker_size, marker="s",
                   zorder=25, edgecolors="white", linewidth=marker_edge_width, label="Customer")

        if show_endpoint_labels:
            from matplotlib import patheffects

            label_effects = [
                patheffects.withStroke(linewidth=1.5, foreground=paper_color("ink.background"))
            ]
            ax.annotate(
                "Merchant",
                xy=origin_rot,
                xytext=(3, -2),
                textcoords="offset points",
                fontsize=5.5,
                ha="left",
                va="top",
                zorder=30,
                path_effects=label_effects,
            )
            ax.annotate(
                "Customer",
                xy=dest_rot,
                xytext=(-3, 2),
                textcoords="offset points",
                fontsize=5.5,
                ha="right",
                va="bottom",
                zorder=30,
                path_effects=label_effects,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        if show_basemap:
            ax.set_facecolor(paper_color("roadnet.f7f7f7"))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if show_stats_box:
            ax.text(
                0.97, 0.03,
                f"Straight: {straight_dist:.0f} m\nRoad: {road_dist:.0f} m\nDetour: {detour:.2f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right", zorder=80,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=paper_color("ink.background"), edgecolor=paper_color("roadnet.d3d8d5"), alpha=0.92),
            )
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            if route_legend_only:
                route_items = [
                    (handle, label)
                    for handle, label in zip(handles, labels)
                    if label in {"E-bike (road)", "Drone (straight)"}
                ]
                handles, labels = zip(*route_items) if route_items else ([], [])
            ax.legend(
                handles,
                labels,
                loc="upper left",
                ncol=2 if compact else 1,
                fontsize=5.5 if compact else 9,
                handlelength=1.5 if compact else 2,
                handletextpad=0.4 if compact else 0.8,
                columnspacing=0.8 if compact else 2,
                borderpad=0.25 if compact else 0.4,
                framealpha=0.85 if compact else 0.9,
            )
        return

    # Expand the tighter axis to match the target display shape while keeping
    # longitude foreshortening at this latitude, so the map is not stretched.
    lat_center = (local_bounds["min_lat"] + local_bounds["max_lat"]) / 2
    cos_lat = max(float(np.cos(np.radians(lat_center))), 1e-6)
    target_aspect = max(float(target_aspect), 1e-6)
    lat_span = local_bounds["max_lat"] - local_bounds["min_lat"]
    lon_span = local_bounds["max_lng"] - local_bounds["min_lng"]
    current_aspect = lon_span * cos_lat / lat_span if lat_span > 0 else target_aspect
    if current_aspect < target_aspect:
        target_lon_span = target_aspect * lat_span / cos_lat
        extra = (target_lon_span - lon_span) / 2
        local_bounds["min_lng"] -= extra
        local_bounds["max_lng"] += extra
    else:
        target_lat_span = lon_span * cos_lat / target_aspect
        extra = (target_lat_span - lat_span) / 2
        local_bounds["min_lat"] -= extra
        local_bounds["max_lat"] += extra

    edge_coords = local_road_edges(graph, local_bounds)
    if len(edge_coords) > 0:
        segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords]
        lc = LineCollection(segments, colors=paper_color("ink.hex_ddd"), linewidths=0.3, zorder=1)
        ax.add_collection(lc)

    if path_result["success"] and len(path_result["path_coordinates"]) > 1:
        path_lngs = [c[1] for c in path_result["path_coordinates"]]
        path_lats = [c[0] for c in path_result["path_coordinates"]]
        ax.plot(path_lngs, path_lats, color=paper_color("roadnet.d95f35"), linewidth=2.5, alpha=0.9, zorder=10,
                label="E-bike (road)")

    ax.plot(
        [origin[1], destination[1]], [origin[0], destination[0]],
        color=paper_color("roadnet.1f77b4"), linewidth=2, linestyle="--", alpha=0.9, zorder=20,
        label="Drone (straight)",
    )
    ax.scatter(origin[1], origin[0], c=paper_color("roadnet.2f6f4e"), s=90, marker="o",
               zorder=25, edgecolors="white", linewidth=1.5, label="Merchant")
    ax.scatter(destination[1], destination[0], c=paper_color("roadnet.3f4b64"), s=90, marker="s",
               zorder=25, edgecolors="white", linewidth=1.5, label="Customer")

    ax.set_xlim(local_bounds["min_lng"], local_bounds["max_lng"])
    ax.set_ylim(local_bounds["min_lat"], local_bounds["max_lat"])
    ax.set_aspect(1 / cos_lat, adjustable="box")

    if show_basemap:
        try:
            configure_contextily_cache()
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.OpenStreetMap.Mapnik,
                            zoom=16, alpha=0.5, zorder=0)
        except Exception as e:
            print(f"Could not add basemap: {e}")

    ax.set_xlim(local_bounds["min_lng"], local_bounds["max_lng"])
    ax.set_ylim(local_bounds["min_lat"], local_bounds["max_lat"])
    ax.set_aspect(1 / cos_lat, adjustable="box")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if show_stats_box:
        ax.text(
            0.97, 0.03,
            f"Straight: {straight_dist:.0f} m\nRoad: {road_dist:.0f} m\nDetour: {detour:.2f}",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right", zorder=80,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=paper_color("ink.background"), edgecolor=paper_color("roadnet.d3d8d5"), alpha=0.92),
        )
    if show_legend:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


@prepared_plot(omit=('graph', 'router'))
def plot_beijing_p75_trip_route_comparison(
    selected_trip,
    selection_report,
    graph,
    router,
    save_path: Optional[str] = None,
    figsize: tuple = (2.4, 1.2),
    for_latex: bool = False,
    plot_title: bool = False,
) -> None:
    """Plot the selected Beijing P75 trip's road and air routes.

    Args:
        selected_trip: Shared representative OD row.
        selection_report: Shared representative-trip provenance.
        graph: SparseGraph built from the region's graphml file.
        router: TileRouter wrapping the SparseGraph.
        save_path: Path to save figure. Shows plot if None.
        figsize: Final figure size in inches.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    origin = (float(selected_trip["from_lat"]), float(selected_trip["from_lng"]))
    destination = (float(selected_trip["to_lat"]), float(selected_trip["to_lng"]))
    path_result = shortest_path(router, origin, destination)
    if not path_result.get("success"):
        raise RuntimeError(
            "Beijing P75 representative trip could not be reconstructed on the "
            f"current road network: {path_result.get('message', 'unknown error')}"
        )
    sample = {
        "origin": origin,
        "destination": destination,
        "straight_distance": float(selected_trip["distance_km"]) * 1000.0,
        "road_distance": float(selected_trip["road_distance_m"]),
        "path_result": path_result,
    }

    fig, ax = plt.subplots(figsize=figsize)
    _draw_distance_panel(
        ax,
        sample,
        graph,
        show_legend=True,
        show_stats_box=False,
        show_basemap=False,
        target_aspect=2.0,
        rotate_air_diagonal=True,
        compact=True,
        route_legend_only=True,
        show_endpoint_labels=True,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    if plot_title:
        fig.suptitle(
            "Straight-line vs Road Network Distance",
            fontsize=7,
            fontweight="bold",
            y=0.98,
        )
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.84)
    else:
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        if save_path.endswith(".pdf"):
            ax.set_rasterization_zorder(2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=paper_color("ink.background"))
        straight_distance = float(sample["straight_distance"])
        road_distance = float(sample["road_distance"])
        path_result = sample.get("path_result", {})
        write_figure_report(
            save_path.rsplit(".", 1)[0] + "_report.json",
            {
                "schema_version": 1,
                "figure_name": "beijing_p75_trip_route_comparison",
                "figure_path": save_path,
                "selection": selection_report,
                "selected_trip": {
                    "origin_lat_lng": [float(value) for value in sample["origin"]],
                    "destination_lat_lng": [
                        float(value) for value in sample["destination"]
                    ],
                    "straight_distance_m": straight_distance,
                    "road_distance_m": road_distance,
                    "detour_ratio": road_distance / straight_distance,
                    "road_path_success": bool(path_result.get("success", False)),
                    "road_path_message": path_result.get("message"),
                    "reconstructed_road_distance_m": float(
                        path_result.get("total_distance", 0.0)
                    ),
                    "road_path_coordinate_count": len(
                        path_result.get("path_coordinates", [])
                    ),
                },
                "route_definitions": {
                    "E-bike": "shortest available road-network route",
                    "Drone": "straight line between the same endpoints",
                },
            },
        )
        print(f"Beijing P75 trip route comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
