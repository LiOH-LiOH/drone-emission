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


def plot_distance_comparison(
    od_df,
    graph,
    router,
    save_path: Optional[str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
    n_samples: int = 4,
) -> None:
    """Visualize sampled delivery paths comparing straight-line vs road network distance.

    Samples OD pairs at detour-ratio quantiles (20/40/60/80th percentile) and plots
    each as a panel showing the road path (blue) and straight-line (red dashed).
    Uses SparseGraph local edges for background roads — no full NetworkX load needed.

    Args:
        od_df: OD DataFrame with columns from_lat, from_lng, to_lat, to_lng,
               distance_km, road_distance_m, annual_sales.
        graph: SparseGraph built from the region's graphml file.
        router: TileRouter wrapping the SparseGraph.
        save_path: Path to save figure (with .png extension). Shows plot if None.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_samples: Maximum number of sample panels to show.
    """
    from matplotlib.collections import LineCollection

    from ._utils import configure_contextily_cache

    valid_df = od_df[
        (od_df["road_distance_m"] >= 0) & (od_df["road_distance_m"] > 100)
    ].copy()

    if len(valid_df) == 0:
        print("No valid samples for distance comparison visualization")
        return

    valid_df["detour_ratio"] = valid_df["road_distance_m"] / (valid_df["distance_km"] * 1000)

    samples = []
    for q in [0.2, 0.4, 0.6, 0.8]:
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

    n_cols = min(2, len(sample_paths))
    n_rows = (len(sample_paths) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if len(sample_paths) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    padding = 0.003
    for idx, sample in enumerate(sample_paths):
        ax = axes[idx]
        path_result = sample["path_result"]
        origin = sample["origin"]
        destination = sample["destination"]
        straight_dist = sample["straight_distance"]
        road_dist = sample["road_distance"]
        detour = road_dist / straight_dist if straight_dist > 0 else 0

        local_bounds = {
            "min_lat": min(origin[0], destination[0]) - padding,
            "max_lat": max(origin[0], destination[0]) + padding,
            "min_lng": min(origin[1], destination[1]) - padding,
            "max_lng": max(origin[1], destination[1]) + padding,
        }

        edge_coords = graph.get_local_edges(local_bounds)
        if len(edge_coords) > 0:
            segments = [[(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords]
            lc = LineCollection(segments, colors="#ddd", linewidths=0.3, zorder=1)
            ax.add_collection(lc)

        if path_result["success"] and len(path_result["path_coordinates"]) > 1:
            path_lngs = [c[1] for c in path_result["path_coordinates"]]
            path_lats = [c[0] for c in path_result["path_coordinates"]]
            ax.plot(path_lngs, path_lats, color="blue", linewidth=2, alpha=0.8, zorder=10)

        ax.plot(
            [origin[1], destination[1]], [origin[0], destination[0]],
            color="red", linewidth=2, linestyle="--", alpha=0.8, zorder=20, label="Straight line",
        )
        ax.scatter(
            origin[1], origin[0], c="green", s=80, marker="o",
            zorder=25, edgecolors="white", linewidth=1.5, label="Origin",
        )
        ax.scatter(
            destination[1], destination[0], c="red", s=80, marker="s",
            zorder=25, edgecolors="white", linewidth=1.5, label="Destination",
        )

        ax.set_xlim(local_bounds["min_lng"], local_bounds["max_lng"])
        ax.set_ylim(local_bounds["min_lat"], local_bounds["max_lat"])

        try:
            configure_contextily_cache()
            ctx.add_basemap(
                ax, crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=16, alpha=0.5, zorder=0,
            )
        except Exception as e:
            print(f"Could not add basemap for sample {idx}: {e}")

        ax.set_title(
            f"Sample {idx + 1}\n"
            f"Straight: {straight_dist:.0f}m | Road: {road_dist:.0f}m\n"
            f"Detour ratio: {detour:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

    for idx in range(len(sample_paths), len(axes)):
        axes[idx].set_visible(False)

    if plot_title:
        plt.suptitle(
            "Straight-line vs Road Network Distance Comparison",
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
        print(f"Distance comparison saved to {save_path}")
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
