from typing import Any, Dict, List, Optional, Tuple
import os
import contextily as ctx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox

from ._utils import (
    add_basemap_and_boundaries,
    configure_contextily_cache,
    finalize_plot,
    plot_boundary_box,
    setup_figure,
    _add_title_suffix,
)


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
    padding=0.003,
    target_aspect=1.0,
    rotate_air_diagonal=False,
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

        edge_coords = graph.get_local_edges(query_bounds)
        if len(edge_coords) > 0:
            segments = [
                [
                    tuple(_transform(ec[1], ec[0])),
                    tuple(_transform(ec[3], ec[2])),
                ]
                for ec in edge_coords
            ]
            lc = LineCollection(segments, colors="#ddd", linewidths=0.3, zorder=1)
            ax.add_collection(lc)

        if path_result.get("success") and len(path_result.get("path_coordinates", [])) > 1:
            path_xy = np.array([
                _transform(lat, lng)
                for lat, lng in path_result["path_coordinates"]
            ])
            ax.plot(path_xy[:, 0], path_xy[:, 1], color="#d95f35", linewidth=2.5,
                    alpha=0.9, zorder=10, label="E-bike (road)")

        origin_rot = _transform(origin[0], origin[1])
        dest_rot = _transform(destination[0], destination[1])
        ax.plot(
            [origin_rot[0], dest_rot[0]], [origin_rot[1], dest_rot[1]],
            color="#1f77b4", linewidth=2, linestyle="--", alpha=0.9, zorder=20,
            label="Drone (straight)",
        )
        ax.scatter(origin_rot[0], origin_rot[1], c="#2f6f4e", s=90, marker="o",
                   zorder=25, edgecolors="white", linewidth=1.5, label="Merchant")
        ax.scatter(dest_rot[0], dest_rot[1], c="#3f4b64", s=90, marker="s",
                   zorder=25, edgecolors="white", linewidth=1.5, label="Customer")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
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
            ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
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
        figsize: Total figure size tuple. Defaults to (4, 4*n_samples).
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_samples: Number of sample panels to show.
    """
    sample_paths = _compute_distance_samples(od_df, router, n_samples)
    if not sample_paths:
        print("No valid samples for distance comparison visualization")
        return

    n_rows = len(sample_paths)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize or (4, 4 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, sample) in enumerate(zip(axes, sample_paths)):
        _draw_distance_panel(ax, sample, graph, show_legend=(idx == 0))

    for idx in range(len(sample_paths), len(axes)):
        axes[idx].set_visible(False)

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
            for ax in axes[:len(sample_paths)]:
                ax.set_rasterization_zorder(12)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Distance comparison saved to {save_path}")
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
