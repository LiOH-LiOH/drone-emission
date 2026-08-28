import os
from pathlib import Path

import numpy as np
import pandas as pd
from drone_emission.utils.boundaries import graphml_covers_bbox, resolve_enlarged_boundaries
from drone_emission.utils.atomic import atomic_open


OD_CHUNK_ROWS = 250_000
OD_BASE_COLUMNS = (
    "from_lat",
    "from_lng",
    "to_lat",
    "to_lng",
    "distance_km",
    "annual_sales",
)


def validate_od_checkpoint(path: str | Path, chunk_rows: int = OD_CHUNK_ROWS) -> dict:
    """Validate the complete atomic OD checkpoint accepted by stage repair."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"OD checkpoint is missing: {path}")
    columns = tuple(pd.read_csv(path, nrows=0).columns)
    missing = [column for column in OD_BASE_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"OD checkpoint is missing columns {missing}: {path}")
    row_count = 0
    annual_sales_total = 0
    for chunk in pd.read_csv(path, usecols=list(OD_BASE_COLUMNS), chunksize=chunk_rows):
        sales = pd.to_numeric(chunk["annual_sales"], errors="coerce").to_numpy()
        coordinates = chunk[list(OD_BASE_COLUMNS[:-1])].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy()
        if not np.isfinite(coordinates).all():
            raise ValueError(f"OD checkpoint has non-finite coordinates/distances: {path}")
        if (
            not np.isfinite(sales).all()
            or (sales <= 0).any()
            or (sales != np.floor(sales)).any()
            or (sales % 12 != 0).any()
        ):
            raise ValueError(f"OD checkpoint violates annual_sales contract: {path}")
        row_count += len(chunk)
        annual_sales_total += int(sales.sum())
    if row_count == 0:
        raise ValueError(f"OD checkpoint contains no rows: {path}")
    return {
        "path": str(path.resolve()),
        "rows": row_count,
        "annual_sales": annual_sales_total,
        "columns": list(columns),
    }


def _fetch_bikeable_roadnet(*, bbox_4326, save_path):
    from .osm_roadnet import fetch_bikeable_roadnet

    return fetch_bikeable_roadnet(bbox_4326=bbox_4326, save_path=save_path)


def _load_sparse_graph(path):
    from .sparse_graph import SparseGraph

    return SparseGraph.from_graphml(path)


def _make_tile_router(graph):
    from .tile_router import TileRouter

    return TileRouter(graph)


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
    strict_no_fetch = exp_config.get("strict_no_fetch", False)
    roadnet_exists = os.path.exists(roadnet_data_path)
    if strict_no_fetch and not roadnet_exists:
        raise FileNotFoundError(
            f"Road network is missing and --no-fetch is active: {roadnet_data_path}"
        )
    needs_fetch = exp_config.get("fetch_roadnet", False) or not roadnet_exists
    if not needs_fetch and large_bounds_dict is not None:
        try:
            covers_expected = graphml_covers_bbox(
                roadnet_data_path, large_bounds_dict
            )
        except Exception as e:
            if strict_no_fetch:
                raise RuntimeError(
                    "Could not verify road network bbox while --no-fetch is active: "
                    f"{roadnet_data_path}"
                ) from e
            print(f"Could not verify existing roadnet bbox ({e}); refetching.")
            needs_fetch = True
        else:
            if not covers_expected:
                if strict_no_fetch:
                    raise RuntimeError(
                        "Existing road network bbox does not cover expected bounds "
                        f"and --no-fetch is active: {roadnet_data_path}"
                    )
                print(
                    f"Existing roadnet bbox does not cover expected bounds; "
                    f"refetching for {region_name_en}."
                )
                needs_fetch = True

    if needs_fetch:
        _fetch_bikeable_roadnet(
            bbox_4326=large_bounds_dict,
            save_path=roadnet_data_path,
        )

    # Build sparse graph (NetworkX loaded temporarily then freed)
    try:
        graph = _load_sparse_graph(roadnet_data_path)
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
    checkpoint = validate_od_checkpoint(od_save_path)
    print(f"Loaded {checkpoint['rows']} OD pairs")
    region_info.set_stat("odpair_count", checkpoint["rows"])
    region_info.set_stat("total_odpair_annual_sales", checkpoint["annual_sales"])

    # Build tile router
    router = _make_tile_router(graph)

    valid_count = 0
    invalid_count = 0
    valid_sales = 0.0
    weighted_road = 0.0
    weighted_straight = 0.0
    computed_count = 0
    with atomic_open(od_save_path, "w", encoding="utf-8", newline="") as handle:
        first = True
        for od_df in pd.read_csv(od_save_path, chunksize=OD_CHUNK_ROWS):
            if "road_distance_m" not in od_df.columns:
                od_df["road_distance_m"] = np.nan
            nan_mask = od_df["road_distance_m"].isna()
            if nan_mask.any():
                nan_df = od_df.loc[nan_mask]
                distances = router.compute_od_distances_batch(
                    nan_df[["from_lat", "from_lng"]].values,
                    nan_df[["to_lat", "to_lng"]].values,
                )
                od_df.loc[nan_mask, "road_distance_m"] = distances
                computed_count += int(nan_mask.sum())
            road_distances = pd.to_numeric(
                od_df["road_distance_m"], errors="coerce"
            ).to_numpy()
            annual_sales = pd.to_numeric(
                od_df["annual_sales"], errors="coerce"
            ).to_numpy()
            straight = pd.to_numeric(od_df["distance_km"], errors="coerce").to_numpy()
            valid_mask = np.isfinite(road_distances) & (road_distances >= 0)
            valid_count += int(valid_mask.sum())
            invalid_count += int((~valid_mask).sum())
            valid_sales += float(annual_sales[valid_mask].sum())
            weighted_road += float(
                np.sum(road_distances[valid_mask] * annual_sales[valid_mask])
            )
            weighted_straight += float(
                np.sum(straight[valid_mask] * annual_sales[valid_mask] * 1000)
            )
            od_df.to_csv(handle, index=False, header=first)
            first = False
        if valid_count == 0 or valid_sales <= 0:
            raise ValueError(f"No valid routed OD rows in {od_save_path}")

    print(
        f"Updated {computed_count} road distances in bounded chunks; "
        f"saved {od_save_path}"
    )
    if valid_count > 0:
        avg_road_dist = weighted_road / valid_sales
        avg_straight_dist = weighted_straight / valid_sales
        detour_ratio = avg_road_dist / avg_straight_dist if avg_straight_dist > 0 else 0

        region_info.set_stat("avg_road_distance_m", avg_road_dist)
        region_info.set_stat("avg_straight_distance_m", avg_straight_dist)
        region_info.set_stat("detour_ratio", detour_ratio)
        region_info.set_stat("valid_path_count", valid_count)
        region_info.set_stat("invalid_path_count", invalid_count)

        print(f"\n=== Road Distance Statistics ===")
        print(f"Valid paths: {valid_count:,} / {valid_count + invalid_count:,}")
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
