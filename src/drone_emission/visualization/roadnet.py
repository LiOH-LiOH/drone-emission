from typing import Any, Dict, List, Optional, Tuple
import os
import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection
from utils.coordinates import gcj02_to_wgs84

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


MERCHANT_MAP_DEGREE_SIZE = 0.01
MERCHANT_MAP_RANDOM_SEED = 20260703
MERCHANT_MAP_TOP_SALES_FRACTION = 0.01


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


def visualize_roadnet(
    G: nx.MultiDiGraph,
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    original_boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> None:
    """Visualize the road network.

    Args:
        G: The road network graph
        city_name_cn: Chinese name of the city for basemap and boundaries
        boundaries_dict: Dictionary with min_lng, max_lng, min_lat, max_lat (enlarged)
        original_boundaries_dict: Original boundary dict for drawing dashed box
        save_path: If provided, save the figure to this path
    """
    fig, ax, _, _, _ = setup_figure(None, None, boundaries_dict)

    ox.plot_graph(
        G,
        ax=ax,
        show=False,
        close=False,
        node_size=2,
        node_color="#333",
        edge_color="#888",
        edge_linewidth=0.8,
        bgcolor="white",
    )

    # Add basemap and boundaries if city_name_cn is provided
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn, boundaries_dict)

    # Plot original boundary as dashed box
    plot_boundary_box(ax, original_boundaries_dict)

    title = "Road Network Visualization"
    stats_text = f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}"
    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict, for_latex, plot_title)
    plt.close(fig)


def visualize_path(
    G: nx.MultiDiGraph,
    path_result: Dict[str, Any],
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    original_boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> None:
    """Visualize the shortest path on the road network.

    Args:
        G: The road network graph
        path_result: Result from find_shortest_path function
        city_name_cn: Chinese name of the city for basemap and boundaries
        boundaries_dict: Dictionary with min_lng, max_lng, min_lat, max_lat (enlarged)
        original_boundaries_dict: Original boundary dict for drawing dashed box
        save_path: If provided, save the figure to this path
    """
    if not path_result["success"]:
        print(f"Cannot visualize: {path_result['message']}")
        return

    fig, ax, _, _, _ = setup_figure(None, None, boundaries_dict)

    ox.plot_graph(
        G,
        ax=ax,
        show=False,
        close=False,
        node_size=2,
        node_color="#333",
        edge_color="#bbb",
        edge_linewidth=0.7,
        bgcolor="white",
    )

    route_nodes = path_result["path_nodes"]
    ox.plot_graph_route(
        G,
        route_nodes,
        ax=ax,
        route_color="red",
        route_linewidth=3,
        node_size=0,
        show=False,
        close=False,
    )

    # Add basemap and boundaries if city_name_cn is provided
    if city_name_cn:
        add_basemap_and_boundaries(ax, city_name_cn, boundaries_dict)

    # Plot original boundary as dashed box
    plot_boundary_box(ax, original_boundaries_dict)

    title = f"Shortest Path: {path_result['total_distance']:.2f}m ({path_result['total_distance']/1000:.3f}km)"
    stats_text = (
        f"Nodes: {len(G.nodes)}, Edges: {len(G.edges)}\nPath Length: {len(route_nodes)}"
    )
    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict, for_latex, plot_title)
    plt.close(fig)


def plot_road_network_subfigure(
    graph,
    stores_df,
    region_name,
    save_path=None,
    figsize=(6.4, 3.2),
    for_latex=False,
    plot_title=False,
    lat_span=None,
    compact=False,
    buildings_gdf=None,
):
    """Show a road network over subdued building footprints.

    Args:
        graph: SparseGraph with get_local_edges support.
        stores_df: DataFrame with lat, lng, and month_sales columns.
        region_name: English region name.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        lat_span: Optional latitude span in degrees.
        compact: If True, use paper-panel styling and export padding.
        buildings_gdf: Optional GeoDataFrame with building geometries.
    """
    required_store_cols = {"lat", "lng", "month_sales"}
    missing_store_cols = required_store_cols - set(stores_df.columns)
    if missing_store_cols:
        print(
            "Road network subfigure skipped - stores data missing "
            f"columns {sorted(missing_store_cols)}"
        )
        return

    valid_stores = stores_df.dropna(subset=["lat", "lng", "month_sales"]).copy()
    valid_stores["month_sales"] = pd.to_numeric(
        valid_stores["month_sales"], errors="coerce"
    )
    valid_stores = valid_stores.dropna(subset=["month_sales"])
    if valid_stores.empty:
        print("Road network subfigure skipped - no valid merchant sales rows")
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
        print("Road network subfigure skipped - lat_span must be positive")
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

    edge_coords = graph.get_local_edges(boundaries_dict, margin_deg=0.0)
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
    ax.set_facecolor("white" if compact else "#f7f7f5")

    building_color = "#d8d2c5"
    building_alpha = 0.5
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

    segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in displayed_edges]
    if compact:
        road_casing = LineCollection(
            segments,
            colors="white",
            linewidths=1.05,
            alpha=0.92,
            zorder=15,
        )
        ax.add_collection(road_casing)
        roads = LineCollection(
            segments,
            colors="#3f4852",
            linewidths=0.5,
            alpha=0.98,
            zorder=20,
        )
    else:
        roads = LineCollection(
            segments,
            colors="#4f5963",
            linewidths=0.55,
            alpha=0.95,
            zorder=10,
        )
    ax.add_collection(roads)

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
            "figure_name": "shanghai_road_network_subfigure",
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
                "road_edge_count": int(len(edge_coords)),
                "undirected_road_segment_count": int(len(unique_edges)),
                "building_context": {
                    "available": bool(building_context_available),
                    "building_count": int(building_count),
                    "color": building_color,
                    "alpha": float(building_alpha),
                    "fallback_to_road_only": bool(building_count == 0),
                },
                "merchant_center": {
                    "lng": float(merchant_lng),
                    "lat": float(merchant_lat),
                    "month_sales": float(merchant["month_sales"]),
                },
            },
        }
        write_figure_report(report_path, report)
        print(f"Road network subfigure report saved to {report_path}")
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.02 if compact else 0,
        )
        print(f"Road network subfigure saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _compute_distance_samples(od_df, router, n_samples=2):
    """Sample OD pairs at detour-ratio quantiles and compute shortest paths."""
    valid_df = od_df[
        (od_df["road_distance_m"] >= 0) & (od_df["road_distance_m"] > 100)
    ].copy()
    if len(valid_df) == 0:
        return []

    valid_df["detour_ratio"] = valid_df["road_distance_m"] / (valid_df["distance_km"] * 1000)

    quantiles = np.linspace(0.25, 0.75, n_samples)
    samples = []
    for q in quantiles:
        qval = valid_df["detour_ratio"].quantile(q)
        closest_idx = (valid_df["detour_ratio"] - qval).abs().nsmallest(1).index
        samples.extend(closest_idx.tolist())

    samples = list(dict.fromkeys(samples))[:n_samples]
    sample_df = valid_df.loc[samples]

    sample_paths = []
    for _, row in sample_df.iterrows():
        origin = (row["from_lat"], row["from_lng"])
        destination = (row["to_lat"], row["to_lng"])
        path_result = router.find_shortest_path(origin, destination)
        sample_paths.append({
            "origin": origin,
            "destination": destination,
            "straight_distance": row["distance_km"] * 1000,
            "road_distance": row["road_distance_m"],
            "path_result": path_result,
        })
    return sample_paths


def _compute_distance_sample_at_quantile(od_df, router, quantile=0.5):
    """Return the OD sample closest to one detour-ratio quantile."""
    valid_df = od_df[
        (od_df["road_distance_m"] > 100) & (od_df["distance_km"] > 0)
    ].copy()
    if len(valid_df) == 0:
        return None

    valid_df["detour_ratio"] = (
        valid_df["road_distance_m"] / (valid_df["distance_km"] * 1000)
    )
    quantile_value = valid_df["detour_ratio"].quantile(quantile)
    sample_idx = (valid_df["detour_ratio"] - quantile_value).abs().idxmin()
    row = valid_df.loc[sample_idx]
    origin = (row["from_lat"], row["from_lng"])
    destination = (row["to_lat"], row["to_lng"])
    return {
        "origin": origin,
        "destination": destination,
        "straight_distance": row["distance_km"] * 1000,
        "road_distance": row["road_distance_m"],
        "path_result": router.find_shortest_path(origin, destination),
    }


def _compute_distance_samples_extreme(od_df, router, n_samples, extreme,
                                       air_km_min=None, air_km_max=None):
    """Sample OD pairs from the extreme tail of the detour-ratio distribution.

    ``extreme="high"`` draws from the deep upper tail (top ~1% up to the max);
    ``extreme="low"`` from the impossible (< 1.0) detour tail for debugging.
    Optional ``air_km_min`` / ``air_km_max`` restrict the candidate pool to a
    specific straight-line distance band before quantile sampling.
    """
    valid_df = od_df[
        (od_df["road_distance_m"] >= 0) & (od_df["road_distance_m"] > 100) &
        (od_df["distance_km"] > 0)
    ].copy()
    if air_km_min is not None:
        valid_df = valid_df[valid_df["distance_km"] >= air_km_min]
    if air_km_max is not None:
        valid_df = valid_df[valid_df["distance_km"] <= air_km_max]
    if len(valid_df) == 0:
        return []

    valid_df["detour_ratio"] = valid_df["road_distance_m"] / (valid_df["distance_km"] * 1000)

    if extreme == "high":
        # Span the deep upper tail; 1.0 picks the absolute maximum.
        qs = np.linspace(0.99, 1.0, n_samples)
    elif extreme == "low":
        # Inspect the impossible-detour tail (< 1.0) for debugging coord issues.
        valid_df = valid_df[valid_df["detour_ratio"] < 1.0]
        if len(valid_df) == 0:
            return []
        qs = np.linspace(0.0, 0.99, n_samples)
    else:
        raise ValueError(f"extreme must be 'high' or 'low', got {extreme!r}")

    picked = []
    used = set()
    for q in qs:
        qval = valid_df["detour_ratio"].quantile(q)
        order = (valid_df["detour_ratio"] - qval).abs().sort_values().index
        for idx in order:
            if idx not in used:
                picked.append(idx)
                used.add(idx)
                break
    sample_df = valid_df.loc[picked]

    sample_paths = []
    for _, row in sample_df.iterrows():
        origin = (row["from_lat"], row["from_lng"])
        destination = (row["to_lat"], row["to_lng"])
        path_result = router.find_shortest_path(origin, destination)
        sample_paths.append({
            "origin": origin,
            "destination": destination,
            "straight_distance": row["distance_km"] * 1000,
            "road_distance": row["road_distance_m"],
            "path_result": path_result,
        })
    return sample_paths


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

        edge_coords = graph.get_local_edges(query_bounds)
        if len(edge_coords) > 0:
            segments = [
                [
                    tuple(_transform(ec[0], ec[1])),
                    tuple(_transform(ec[2], ec[3])),
                ]
                for ec in edge_coords
            ]
            network_color = "#cfd4d1" if compact else "#ddd"
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
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="#d95f35", linewidth=road_width,
                    alpha=0.9, zorder=10, label="E-bike (road)")

        origin_rot = _transform(origin[0], origin[1])
        dest_rot = _transform(destination[0], destination[1])
        drone_width = 1.2 if compact else 2
        marker_size = 22 if compact else 90
        marker_edge_width = 0.7 if compact else 1.5
        ax.plot(
            [origin_rot[0], dest_rot[0]], [origin_rot[1], dest_rot[1]],
            color="#1f77b4", linewidth=drone_width, linestyle="--", alpha=0.9, zorder=20,
            label="Drone (straight)",
        )
        ax.scatter(origin_rot[0], origin_rot[1], c="#2f6f4e", s=marker_size, marker="o",
                   zorder=25, edgecolors="white", linewidth=marker_edge_width, label="Merchant")
        ax.scatter(dest_rot[0], dest_rot[1], c="#3f4b64", s=marker_size, marker="s",
                   zorder=25, edgecolors="white", linewidth=marker_edge_width, label="Customer")

        if show_endpoint_labels:
            from matplotlib import patheffects

            label_effects = [
                patheffects.withStroke(linewidth=1.5, foreground="white")
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
            ax.set_facecolor("#f7f7f7")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if show_stats_box:
            ax.text(
                0.97, 0.03,
                f"Straight: {straight_dist:.0f} m\nRoad: {road_dist:.0f} m\nDetour: {detour:.2f}",
                transform=ax.transAxes, fontsize=9, va="bottom", ha="right", zorder=80,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d3d8d5", alpha=0.92),
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

    edge_coords = graph.get_local_edges(local_bounds)
    if len(edge_coords) > 0:
        segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords]
        lc = LineCollection(segments, colors="#ddd", linewidths=0.3, zorder=1)
        ax.add_collection(lc)

    if path_result["success"] and len(path_result["path_coordinates"]) > 1:
        path_lngs = [c[1] for c in path_result["path_coordinates"]]
        path_lats = [c[0] for c in path_result["path_coordinates"]]
        ax.plot(path_lngs, path_lats, color="#d95f35", linewidth=2.5, alpha=0.9, zorder=10,
                label="E-bike (road)")

    ax.plot(
        [origin[1], destination[1]], [origin[0], destination[0]],
        color="#1f77b4", linewidth=2, linestyle="--", alpha=0.9, zorder=20,
        label="Drone (straight)",
    )
    ax.scatter(origin[1], origin[0], c="#2f6f4e", s=90, marker="o",
               zorder=25, edgecolors="white", linewidth=1.5, label="Merchant")
    ax.scatter(destination[1], destination[0], c="#3f4b64", s=90, marker="s",
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
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d3d8d5", alpha=0.92),
        )
    if show_legend:
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)


def plot_distance_comparison(
    od_df,
    graph,
    router,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    for_latex: bool = False,
    plot_title: bool = False,
    n_samples: int = 2,
    n_cols: int = 1,
    show_stats_box: bool = True,
    show_basemap: bool = True,
    target_aspect: float = 1.0,
) -> None:
    """Visualize sampled delivery paths comparing straight-line vs road network distance.

    Samples OD pairs at detour-ratio quantiles and plots each as a square panel
    showing the road path and straight-line distance.

    Args:
        od_df: OD DataFrame with columns from_lat, from_lng, to_lat, to_lng,
               distance_km, road_distance_m, annual_sales.
        graph: SparseGraph built from the region's graphml file.
        router: TileRouter wrapping the SparseGraph.
        save_path: Path to save figure (with .png extension). Shows plot if None.
        figsize: Total figure size tuple. Defaults to a 4-inch square per panel.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_samples: Number of sample panels to show.
        n_cols: Number of subplot columns.
        show_stats_box: If True, show per-panel distance summary text boxes.
        show_basemap: If True, add the OpenStreetMap tile background.
        target_aspect: Desired displayed panel width-to-height ratio.
    """
    sample_paths = _compute_distance_samples(od_df, router, n_samples)
    if not sample_paths:
        print("No valid samples for distance comparison visualization")
        return

    n_cols = max(1, min(int(n_cols), len(sample_paths)))
    n_rows = int(np.ceil(len(sample_paths) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (4 * n_cols, 4 * n_rows),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    panel_axes = axes.flatten()

    for idx, (ax, sample) in enumerate(zip(panel_axes, sample_paths)):
        _draw_distance_panel(
            ax,
            sample,
            graph,
            show_legend=(idx == 0),
            show_stats_box=show_stats_box,
            show_basemap=show_basemap,
            target_aspect=target_aspect,
        )

    for idx in range(len(sample_paths), len(panel_axes)):
        panel_axes[idx].set_visible(False)

    if plot_title:
        plt.suptitle("Straight-line vs Road Network Distance Comparison",
                     fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        if save_path.endswith(".pdf"):
            for ax in panel_axes[:len(sample_paths)]:
                ax.set_rasterization_zorder(12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Distance comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_distance_comparison_single_panel(
    od_df,
    graph,
    router,
    save_path: Optional[str] = None,
    figsize: tuple = (2.4, 1.2),
    for_latex: bool = False,
    plot_title: bool = False,
) -> None:
    """Plot a compact median-trip road-versus-air distance comparison.

    Args:
        od_df: OD DataFrame with coordinates and road/air distances.
        graph: SparseGraph built from the region's graphml file.
        router: TileRouter wrapping the SparseGraph.
        save_path: Path to save figure. Shows plot if None.
        figsize: Final figure size in inches.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    sample = _compute_distance_sample_at_quantile(od_df, router, quantile=0.5)
    if sample is None:
        print("No valid sample for single-panel distance comparison")
        return

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
        fig.savefig(save_path, dpi=300, facecolor="white")
        print(f"Single-panel distance comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_distance_comparison_grid(
    od_df,
    graph,
    router,
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    for_latex: bool = False,
    plot_title: bool = False,
    n_rows: int = 3,
    n_cols: int = 3,
    extreme: str = "high",
    air_km_min: Optional[float] = None,
    air_km_max: Optional[float] = None,
) -> None:
    """Grid of OD samples drawn from the extreme high or low detour-ratio tail.

    Each panel shows the e-bike road path and straight-line drone path for one
    OD pair, with a small text box listing straight/road distance and detour.

    Args:
        od_df: OD DataFrame with from_lat/from_lng/to_lat/to_lng/distance_km/
               road_distance_m columns.
        graph: SparseGraph for the region.
        router: TileRouter wrapping the SparseGraph.
        save_path: Path to save figure (with .png extension). Shows plot if None.
        figsize: Total figure size. Defaults to (4 * n_cols, 4 * n_rows).
        for_latex: If True, save as PDF.
        plot_title: If True, show suptitle.
        n_rows: Grid rows.
        n_cols: Grid cols.
        extreme: ``"high"`` samples the top tail of detour ratio, ``"low"`` the bottom.
    """
    n_samples = n_rows * n_cols
    sample_paths = _compute_distance_samples_extreme(
        od_df, router, n_samples, extreme,
        air_km_min=air_km_min, air_km_max=air_km_max,
    )
    if not sample_paths:
        print(f"No valid samples for extreme={extreme} distance comparison grid")
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize or (4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, sample in enumerate(sample_paths):
        _draw_distance_panel(axes[idx], sample, graph, show_legend=(idx == 0))
    for idx in range(len(sample_paths), len(axes)):
        axes[idx].set_visible(False)

    if plot_title:
        tail = "high" if extreme == "high" else "low"
        plt.suptitle(
            f"Distance comparison — extreme {tail} detour ratio",
            fontsize=14, fontweight="bold", y=1.02,
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        if save_path.endswith(".pdf"):
            for ax in axes[:len(sample_paths)]:
                ax.set_rasterization_zorder(12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Distance comparison grid ({extreme}) saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def visualize_distance_comparison(
    G: nx.MultiDiGraph,
    sample_paths: List[Dict[str, Any]],
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> None:
    """Visualize comparison between straight-line and road network distances.

    Args:
        G: The road network graph
        sample_paths: List of path results with 'origin', 'destination',
                     'straight_distance', 'road_distance', 'path_result' keys
        city_name_cn: Chinese name of the city for basemap
        boundaries_dict: Dictionary with min_lng, max_lng, min_lat, max_lat
        save_path: If provided, save the figure to this path
    """
    n_samples = len(sample_paths)

    # Create figure with subplots: one map per sample
    n_cols = min(2, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sample in enumerate(sample_paths):
        ax = axes[idx]

        path_result = sample["path_result"]
        origin = sample["origin"]  # (lat, lng)
        destination = sample["destination"]  # (lat, lng)
        straight_dist = sample["straight_distance"]
        road_dist = sample["road_distance"]
        detour = road_dist / straight_dist if straight_dist > 0 else 0

        # Plot road network in light gray
        ox.plot_graph(
            G,
            ax=ax,
            show=False,
            close=False,
            node_size=0,
            edge_color="#ddd",
            edge_linewidth=0.3,
            bgcolor="white",
        )

        # Plot road path if available
        if path_result["success"] and len(path_result["path_nodes"]) > 1:
            ox.plot_graph_route(
                G,
                path_result["path_nodes"],
                ax=ax,
                route_color="blue",
                route_linewidth=2,
                route_alpha=0.8,
                node_size=0,
                show=False,
                close=False,
            )

        # Plot straight line (origin to destination)
        ax.plot(
            [origin[1], destination[1]],  # lng
            [origin[0], destination[0]],  # lat
            color="red",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            zorder=20,
            label="Straight line",
        )

        # Plot origin and destination points
        ax.scatter(
            origin[1],
            origin[0],
            c="green",
            s=80,
            marker="o",
            zorder=25,
            edgecolors="white",
            linewidth=1.5,
            label="Origin",
        )
        ax.scatter(
            destination[1],
            destination[0],
            c="red",
            s=80,
            marker="s",
            zorder=25,
            edgecolors="white",
            linewidth=1.5,
            label="Destination",
        )

        # Calculate bounds from origin and destination with padding
        padding = 0.003  # ~300m padding
        local_bounds = {
            "min_lat": min(origin[0], destination[0]) - padding,
            "max_lat": max(origin[0], destination[0]) + padding,
            "min_lng": min(origin[1], destination[1]) - padding,
            "max_lng": max(origin[1], destination[1]) + padding,
        }

        # Add basemap with local bounds
        ax.set_xlim(local_bounds["min_lng"], local_bounds["max_lng"])
        ax.set_ylim(local_bounds["min_lat"], local_bounds["max_lat"])

        try:
            configure_contextily_cache()
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=16,  # Higher zoom for smaller area
                alpha=0.5,
                zorder=0,
            )
        except Exception as e:
            print(f"Could not add basemap for sample {idx}: {e}")

        # Title with distance info
        ax.set_title(
            f"Sample {idx + 1}\n"
            f"Straight: {straight_dist:.0f}m | Road: {road_dist:.0f}m\n"
            f"Detour ratio: {detour:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)

        # Only show legend on first subplot
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].set_visible(False)

    if plot_title:
        plt.suptitle(
            "Straight-line vs Road Network Distance Comparison",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        # Rasterize heavy elements (basemap, roads) in PDF output
        # Elements with zorder < 12 are rasterized, others remain vector
        if save_path.endswith(".pdf"):
            for ax in axes[:n_samples]:
                ax.set_rasterization_zorder(12)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
