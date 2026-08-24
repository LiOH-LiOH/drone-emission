import os

import numpy as np
import pandas as pd
from drone_emission.utils.boundaries import graphml_covers_bbox, resolve_enlarged_boundaries

from ._compat import load_networkx_graph
from .osm_roadnet import fetch_bikeable_roadnet
from .sparse_graph import SparseGraph
from .tile_router import TileRouter

def load_roadnet(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    large_bounds_dict = resolve_enlarged_boundaries(region_info, exp_config)

    # Setup data directory and file path
    data_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    roadnet_data_path = os.path.join(
        data_dir, f"{region_name_en.lower()}_roadnet.graphml"
    )

    # Fetch road network if requested, file missing, or existing file's
    # bbox doesn't cover the expected enlarged boundaries (a stale graphml
    # from an earlier run with a smaller boundaries_dict).
    needs_fetch = (
        exp_config.get("fetch_roadnet", False)
        or not os.path.exists(roadnet_data_path)
    )
    if not needs_fetch and large_bounds_dict is not None:
        try:
            if not graphml_covers_bbox(roadnet_data_path, large_bounds_dict):
                print(
                    f"Existing roadnet bbox does not cover expected bounds; "
                    f"refetching for {region_name_en}."
                )
                needs_fetch = True
        except Exception as e:
            print(f"Could not verify existing roadnet bbox ({e}); refetching.")
            needs_fetch = True

    if needs_fetch:
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
    if region_info.get_stat("orientation_entropy") is None:
        _compute_network_metrics(roadnet_data_path, region_info)

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



def _compute_network_metrics(graphml_path, region_info):
    """Compute OSMnx network metrics and save to region stats.

    Loads the full NetworkX graph temporarily, computes structural
    metrics (dead-end proportion, 4-way proportion, orientation entropy,
    avg street length), then frees memory.

    Note: circuity_avg is NOT computed because the unsimplified graph
    stores edge length = great-circle distance, making circuity = 1.0.
    """
    import gc

    import osmnx as ox

    G = ox.load_graphml(graphml_path)
    Gu = ox.convert.to_undirected(G)

    # Avg street length (manual — avoids slow basic_stats)
    total_len = sum(float(d["length"]) for u, v, d in Gu.edges(data=True))
    n_edges = Gu.number_of_edges()
    region_info.set_stat("avg_street_length", total_len / n_edges if n_edges > 0 else 0)

    # Node degree proportions
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

    # Orientation entropy (requires bearings on edges)
    try:
        Gu = ox.bearing.add_edge_bearings(Gu)
        entropy = ox.bearing.orientation_entropy(Gu)
        region_info.set_stat("orientation_entropy", entropy)
    except Exception:
        pass

    dead_pct = region_info.get_stat("dead_end_proportion")
    entropy = region_info.get_stat("orientation_entropy")
    print(
        f"  Network metrics: "
        f"dead_end={f'{dead_pct:.3f}' if dead_pct is not None else 'N/A'}, "
        f"entropy={f'{entropy:.3f}' if entropy is not None else 'N/A'}"
    )

    del G, Gu
    gc.collect()
