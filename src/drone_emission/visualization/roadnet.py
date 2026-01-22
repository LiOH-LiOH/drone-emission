from typing import Any, Dict, List, Optional, Tuple

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
)


def visualize_roadnet(
    G: nx.MultiDiGraph,
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    original_boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
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
    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict)
    plt.close(fig)


def visualize_path(
    G: nx.MultiDiGraph,
    path_result: Dict[str, Any],
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    original_boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
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
    finalize_plot(fig, ax, save_path, title, stats_text, boundaries_dict)
    plt.close(fig)


def visualize_distance_comparison(
    G: nx.MultiDiGraph,
    sample_paths: List[Dict[str, Any]],
    city_name_cn: Optional[str] = None,
    boundaries_dict: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
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

    plt.suptitle(
        "Straight-line vs Road Network Distance Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
