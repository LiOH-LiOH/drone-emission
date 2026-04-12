import os

import numpy as np
import pandas as pd
from visualization.roadnet import visualize_roadnet

from ._compat import load_networkx_graph
from .osm_roadnet import fetch_bikeable_roadnet
from .sparse_graph import SparseGraph
from .tile_router import TileRouter

# Threshold for "large" graphs where we skip full-graph NetworkX visualization
_LARGE_GRAPH_THRESHOLD = 500_000


def load_roadnet(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_name_cn = region_config.get("region_name_cn")
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict")
    large_bounds_dict = region_info.get_stat("enlarged_boundaries")

    # Setup data directory and file path
    data_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    roadnet_data_path = os.path.join(
        data_dir, f"{region_name_en.lower()}_roadnet.graphml"
    )

    # Fetch road network if requested or file doesn't exist
    if exp_config.get("fetch_roadnet", False) or not os.path.exists(roadnet_data_path):
        fetch_bikeable_roadnet(
            bbox_4326=large_bounds_dict,
            save_path=roadnet_data_path,
        )

    # Build sparse graph (NetworkX loaded temporarily then freed)
    try:
        graph = SparseGraph.from_graphml(roadnet_data_path)
    except Exception as e:
        print(f"Error loading road network: {e}")
        return

    region_info.set_stat("roadnet_node_count", graph.num_nodes)
    region_info.set_stat("roadnet_edge_count", graph.num_edges)
    region_info.set_stat("roadnet_length_total", graph.total_length)

    # Compute OSMnx network metrics (skip if already cached)
    if region_info.get_stat("circuity_avg") is None:
        _compute_network_metrics(roadnet_data_path, region_info)

    # Visualize road network (expensive at city scale — can be disabled)
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    roadnet_fig_path = os.path.join(fig_dir, f"{region_name_en.lower()}_roadnet.png")
    for_latex = exp_config.get("for_latex", False)
    plot_title = exp_config.get("plot_title", False)
    roadnet_vis = exp_config.get("roadnet_visualization", True)

    if roadnet_vis:
        if graph.num_nodes < _LARGE_GRAPH_THRESHOLD:
            G_vis = load_networkx_graph(roadnet_data_path)
            visualize_roadnet(
                G_vis,
                city_name_cn=region_name_cn,
                boundaries_dict=large_bounds_dict,
                original_boundaries_dict=boundaries_dict,
                save_path=roadnet_fig_path,
                for_latex=for_latex,
                plot_title=plot_title,
            )
            del G_vis
        else:
            print(
                f"Graph too large for full visualization ({graph.num_nodes:,} nodes), "
                f"generating scatter plot instead..."
            )
            _visualize_roadnet_scatter(
                graph, large_bounds_dict, roadnet_fig_path, for_latex, plot_title
            )
    else:
        print("Road network visualization disabled (roadnet_visualization = false)")

    # Load OD pairs
    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    od_df = pd.read_csv(od_save_path)
    print(f"Loaded {len(od_df)} OD pairs")

    # Build tile router
    router = TileRouter(graph)

    # Only compute road distances for rows with NaN values
    if "road_distance_m" not in od_df.columns:
        od_df["road_distance_m"] = np.nan

    nan_mask = od_df["road_distance_m"].isna()
    nan_count = nan_mask.sum()

    if nan_count > 0:
        print(f"Computing road distances for {nan_count} rows with NaN values...")

        nan_df = od_df[nan_mask]
        origins = nan_df[["from_lat", "from_lng"]].values
        destinations = nan_df[["to_lat", "to_lng"]].values

        road_distances = router.compute_od_distances_batch(origins, destinations)

        od_df.loc[nan_mask, "road_distance_m"] = road_distances

        od_df.to_csv(od_save_path, index=False)
        print(f"Updated OD pairs with road distances saved to {od_save_path}")
    else:
        print("All road distances already computed, skipping...")

    road_distances = od_df["road_distance_m"].values

    # Statistics
    valid_mask = road_distances >= 0
    if valid_mask.sum() > 0:
        avg_road_dist = (
            road_distances[valid_mask] * od_df.loc[valid_mask, "annual_sales"]
        ).sum() / od_df.loc[valid_mask, "annual_sales"].sum()
        avg_straight_dist = (
            (
                od_df.loc[valid_mask, "distance_km"]
                * od_df.loc[valid_mask, "annual_sales"]
            ).sum()
            / od_df.loc[valid_mask, "annual_sales"].sum()
            * 1000
        )
        detour_ratio = avg_road_dist / avg_straight_dist if avg_straight_dist > 0 else 0

        region_info.set_stat("avg_road_distance_m", avg_road_dist)
        region_info.set_stat("avg_straight_distance_m", avg_straight_dist)
        region_info.set_stat("detour_ratio", detour_ratio)
        region_info.set_stat("valid_path_count", int(valid_mask.sum()))
        region_info.set_stat("invalid_path_count", int((~valid_mask).sum()))

        print(f"\n=== Road Distance Statistics ===")
        print(f"Valid paths: {valid_mask.sum():,} / {len(od_df):,}")
        print(f"Avg straight-line distance: {avg_straight_dist:.1f} m")
        print(f"Avg road network distance: {avg_road_dist:.1f} m")
        print(f"Detour ratio: {detour_ratio:.2f}")

        # Sample path visualization (always runs — uses local data only, no full graph load)
        _visualize_sample_paths(
            od_df,
            graph,
            router,
            valid_mask,
            fig_dir,
            region_name_en,
            exp_config,
        )


def _compute_network_metrics(graphml_path, region_info):
    """Compute OSMnx network metrics and save to region stats.

    Loads the full NetworkX graph temporarily, computes structural
    metrics (circuity, dead-end proportion, 4-way proportion,
    orientation entropy, avg street length), then frees memory.
    """
    import osmnx as ox

    G = ox.load_graphml(graphml_path)
    Gu = ox.convert.to_undirected(G)

    # OSMnx basic stats
    stats = ox.stats.basic_stats(G)
    region_info.set_stat("circuity_avg", stats.get("circuity_avg"))
    region_info.set_stat("avg_street_length", stats.get("street_length_avg"))

    # Node degree proportions (from street_count attribute)
    spn = ox.stats.streets_per_node(G)
    total_nodes = len(spn)
    if total_nodes > 0:
        dead_ends = sum(1 for v in spn.values() if v == 1)
        intersections = sum(1 for v in spn.values() if v >= 2)
        four_way_plus = sum(1 for v in spn.values() if v >= 4)
        region_info.set_stat("dead_end_proportion", dead_ends / total_nodes)
        region_info.set_stat(
            "four_way_proportion",
            four_way_plus / intersections if intersections > 0 else 0,
        )

    # Orientation entropy
    try:
        entropy = ox.bearing.orientation_entropy(Gu)
        region_info.set_stat("orientation_entropy", entropy)
    except Exception:
        pass  # Some graphs may lack bearing data

    circuity = stats.get("circuity_avg", "N/A")
    dead_pct = region_info.get_stat("dead_end_proportion")
    entropy = region_info.get_stat("orientation_entropy")
    print(
        f"  Network metrics: circuity={circuity}, "
        f"dead_end_pct={f'{dead_pct:.3f}' if dead_pct is not None else 'N/A'}, "
        f"entropy={f'{entropy:.3f}' if entropy is not None else 'N/A'}"
    )

    del G, Gu


def _visualize_sample_paths(
    od_df,
    graph,
    router,
    valid_mask,
    fig_dir,
    region_name_en,
    exp_config,
    n_samples=10,
):
    """Sample paths visualization using SparseGraph local edges (no NetworkX needed)."""
    import contextily as ctx
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from visualization._utils import configure_contextily_cache

    valid_df = od_df[valid_mask & (od_df["road_distance_m"] > 100)].copy()

    if len(valid_df) < n_samples:
        print(f"Warning: Only {len(valid_df)} valid samples available")
        n_samples = len(valid_df)

    if n_samples == 0:
        print("No valid samples for distance comparison visualization")
        return

    valid_df["detour_ratio"] = valid_df["road_distance_m"] / (
        valid_df["distance_km"] * 1000
    )

    samples = []
    for q in [0.2, 0.4, 0.6, 0.8]:
        quantile_val = valid_df["detour_ratio"].quantile(q)
        closest_idx = (valid_df["detour_ratio"] - quantile_val).abs().nsmallest(1).index
        samples.extend(closest_idx.tolist())

    samples = list(dict.fromkeys(samples))[:n_samples]
    sample_df = valid_df.loc[samples]

    print(f"\n=== Visualizing {len(sample_df)} sample paths ===")

    sample_paths = []
    for _, row in sample_df.iterrows():
        origin = (row["from_lat"], row["from_lng"])
        destination = (row["to_lat"], row["to_lng"])

        path_result = router.find_shortest_path(origin, destination)

        sample_paths.append(
            {
                "origin": origin,
                "destination": destination,
                "straight_distance": row["distance_km"] * 1000,
                "road_distance": row["road_distance_m"],
                "path_result": path_result,
            }
        )

        print(
            f"  Sample: straight={row['distance_km']*1000:.0f}m, "
            f"road={row['road_distance_m']:.0f}m, "
            f"detour={row['detour_ratio']:.2f}"
        )

    # Plot
    n_cols = min(2, len(sample_paths))
    n_rows = (len(sample_paths) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if len(sample_paths) == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, sample in enumerate(sample_paths):
        ax = axes[idx]
        path_result = sample["path_result"]
        origin = sample["origin"]
        destination = sample["destination"]
        straight_dist = sample["straight_distance"]
        road_dist = sample["road_distance"]
        detour = road_dist / straight_dist if straight_dist > 0 else 0

        # Local bounds for this panel
        padding = 0.003
        local_bounds = {
            "min_lat": min(origin[0], destination[0]) - padding,
            "max_lat": max(origin[0], destination[0]) + padding,
            "min_lng": min(origin[1], destination[1]) - padding,
            "max_lng": max(origin[1], destination[1]) + padding,
        }

        # Draw local road edges from SparseGraph (light gray background)
        edge_coords = graph.get_local_edges(local_bounds)
        if len(edge_coords) > 0:
            segments = [
                [(ec[1], ec[0]), (ec[3], ec[2])] for ec in edge_coords
            ]
            lc = LineCollection(
                segments, colors="#ddd", linewidths=0.3, zorder=1
            )
            ax.add_collection(lc)

        # Draw road path
        if path_result["success"] and len(path_result["path_coordinates"]) > 1:
            path_coords = path_result["path_coordinates"]
            path_lngs = [c[1] for c in path_coords]
            path_lats = [c[0] for c in path_coords]
            ax.plot(
                path_lngs, path_lats,
                color="blue", linewidth=2, alpha=0.8, zorder=10,
            )

        # Draw straight line
        ax.plot(
            [origin[1], destination[1]], [origin[0], destination[0]],
            color="red", linewidth=2, linestyle="--",
            alpha=0.8, zorder=20, label="Straight line",
        )

        # Draw origin and destination
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

        # Add basemap
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

    for_latex = exp_config.get("for_latex", False)
    plot_title = exp_config.get("plot_title", False)
    if plot_title:
        plt.suptitle(
            "Straight-line vs Road Network Distance Comparison",
            fontsize=14, fontweight="bold", y=1.02,
        )
    plt.tight_layout()

    comparison_fig_path = os.path.join(
        fig_dir, f"{region_name_en.lower()}_distance_comparison.png"
    )
    os.makedirs(os.path.dirname(comparison_fig_path), exist_ok=True)
    if for_latex and not comparison_fig_path.endswith(".pdf"):
        comparison_fig_path = comparison_fig_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        from visualization._utils import _add_title_suffix
        comparison_fig_path = _add_title_suffix(comparison_fig_path)
    if comparison_fig_path.endswith(".pdf"):
        for ax in axes[:len(sample_paths)]:
            ax.set_rasterization_zorder(12)
    plt.savefig(comparison_fig_path, dpi=200, bbox_inches="tight")
    print(f"Comparison plot saved to {comparison_fig_path}")
    plt.close(fig)


def _visualize_roadnet_scatter(
    graph, large_bounds_dict, save_path, for_latex, plot_title=False
):
    """Fallback scatter-plot visualization for large road networks."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    coords = graph._node_coords
    # Subsample if too many nodes
    if len(coords) > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), 200_000, replace=False)
        coords = coords[idx]

    ax.scatter(coords[:, 1], coords[:, 0], s=0.1, c="#888", alpha=0.3)

    if large_bounds_dict:
        ax.set_xlim(large_bounds_dict["min_lng"], large_bounds_dict["max_lng"])
        ax.set_ylim(large_bounds_dict["min_lat"], large_bounds_dict["max_lat"])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Road Network ({graph.num_nodes:,} nodes, {graph.num_edges:,} edges)")
    ax.set_aspect("equal")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            from visualization._utils import _add_title_suffix
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Road network scatter plot saved to {save_path}")

    plt.close(fig)
