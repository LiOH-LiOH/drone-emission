import os

import numpy as np
import pandas as pd
from visualization.roadnet import visualize_distance_comparison, visualize_roadnet

from ._compat import load_networkx_graph, load_subgraph_for_visualization
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

    # Visualize road network
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    roadnet_fig_path = os.path.join(fig_dir, f"{region_name_en.lower()}_roadnet.png")
    for_latex = exp_config.get("for_latex", False)

    if graph.num_nodes < _LARGE_GRAPH_THRESHOLD:
        # Small enough to load full NetworkX graph for visualization
        G_vis = load_networkx_graph(roadnet_data_path)
        visualize_roadnet(
            G_vis,
            city_name_cn=region_name_cn,
            boundaries_dict=large_bounds_dict,
            original_boundaries_dict=boundaries_dict,
            save_path=roadnet_fig_path,
            for_latex=for_latex,
        )
        del G_vis
    else:
        print(
            f"Graph too large for full visualization ({graph.num_nodes:,} nodes), "
            f"generating scatter plot instead..."
        )
        _visualize_roadnet_scatter(
            graph, large_bounds_dict, boundaries_dict, roadnet_fig_path, for_latex
        )

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

        _visualize_sample_paths(
            od_df,
            router,
            valid_mask,
            fig_dir,
            region_name_en,
            exp_config,
            roadnet_data_path,
        )


def _visualize_sample_paths(
    od_df,
    router,
    valid_mask,
    fig_dir,
    region_name_en,
    exp_config,
    roadnet_data_path,
    n_samples=10,
):
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

    all_lats = []
    all_lngs = []
    for sp in sample_paths:
        all_lats.extend([sp["origin"][0], sp["destination"][0]])
        all_lngs.extend([sp["origin"][1], sp["destination"][1]])

    padding = 0.005
    sample_bounds = {
        "min_lat": min(all_lats) - padding,
        "max_lat": max(all_lats) + padding,
        "min_lng": min(all_lngs) - padding,
        "max_lng": max(all_lngs) + padding,
    }

    # Load a local subgraph for visualization (covers just the sample area)
    G_vis = load_subgraph_for_visualization(roadnet_data_path, sample_bounds)

    comparison_fig_path = os.path.join(
        fig_dir, f"{region_name_en.lower()}_distance_comparison.png"
    )

    for_latex = exp_config.get("for_latex", False)
    visualize_distance_comparison(
        G_vis,
        sample_paths,
        boundaries_dict=sample_bounds,
        save_path=comparison_fig_path,
        for_latex=for_latex,
    )
    del G_vis


def _visualize_roadnet_scatter(
    graph, large_bounds_dict, boundaries_dict, save_path, for_latex
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

    if boundaries_dict:
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
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Road network scatter plot saved to {save_path}")

    plt.close(fig)
