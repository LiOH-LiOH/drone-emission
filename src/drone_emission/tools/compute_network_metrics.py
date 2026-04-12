"""
Compute OSMnx network metrics for all regions from existing GraphML files.

Lightweight script — does NOT re-run the full pipeline. Just loads each
city's GraphML, computes structural metrics, and saves to region_stats.json.

Usage:
    cd src/drone_emission
    python -m tools.compute_network_metrics --env remote
    python -m tools.compute_network_metrics --env remote --force
    python -m tools.compute_network_metrics --env remote --region shenzhen
"""

import argparse
import gc
import os
import time

import osmnx as ox

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils import load_config
from utils.region_config import RegionRegistry


def _file_size_mb(path):
    try:
        return os.path.getsize(path) / 1e6
    except OSError:
        return 0


def compute_metrics_for_region(region, data_dir, force=False):
    """Compute OSMnx metrics for a single region from its GraphML file."""
    level = region.region_level
    name_en = region.region_name_en.lower()

    # Check cache
    if not force and region.get_stat("circuity_avg") is not None:
        return "cached", 0

    # Find GraphML file
    graphml_path = os.path.join(data_dir, level, name_en, f"{name_en}_roadnet.graphml")
    if not os.path.exists(graphml_path):
        return "no_graphml", 0

    size_mb = _file_size_mb(graphml_path)
    if size_mb < 0.001:
        return "empty_file", 0

    print(f"    Loading {name_en}_roadnet.graphml ({size_mb:.0f} MB)...", flush=True)

    t0 = time.time()
    try:
        G = ox.load_graphml(graphml_path)
        t_load = time.time() - t0
        print(f"    Loaded in {t_load:.0f}s ({G.number_of_nodes():,} nodes)", flush=True)

        Gu = ox.convert.to_undirected(G)

        # Basic stats (circuity is the slow part)
        stats = ox.stats.basic_stats(G)
        region.set_stat("circuity_avg", stats.get("circuity_avg"))
        region.set_stat("avg_street_length", stats.get("street_length_avg"))

        # Node degree proportions
        spn = ox.stats.streets_per_node(G)
        total_nodes = len(spn)
        if total_nodes > 0:
            dead_ends = sum(1 for v in spn.values() if v == 1)
            intersections = sum(1 for v in spn.values() if v >= 2)
            four_way_plus = sum(1 for v in spn.values() if v >= 4)
            region.set_stat("dead_end_proportion", dead_ends / total_nodes)
            region.set_stat(
                "four_way_proportion",
                four_way_plus / intersections if intersections > 0 else 0,
            )

        # Orientation entropy (requires bearings on edges)
        try:
            Gu = ox.bearing.add_edge_bearings(Gu)
            entropy = ox.bearing.orientation_entropy(Gu)
            region.set_stat("orientation_entropy", entropy)
        except Exception:
            pass

        del G, Gu
        gc.collect()
        region.save_stats()
        elapsed = time.time() - t0
        return "computed", elapsed

    except Exception as e:
        elapsed = time.time() - t0
        return f"error: {e}", elapsed


def process_single_region(region_key, env, force=False):
    """Entry point for single-region processing (used by parallel xargs)."""
    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=env)
    data_dir = config["data_dir"]

    regions_json = os.path.join(PROJECT_ROOT, "regions.json")
    registry = RegionRegistry(regions_json, data_dir=data_dir)

    if region_key not in registry:
        print(f"{region_key}: not found in registry")
        return

    region = registry[region_key]
    result, elapsed = compute_metrics_for_region(region, data_dir, force=force)

    if result == "computed":
        circ = region.get_stat("circuity_avg")
        ent = region.get_stat("orientation_entropy")
        print(
            f"{region_key}: done in {elapsed:.0f}s — "
            f"circuity={f'{circ:.4f}' if circ else 'N/A'}, "
            f"entropy={f'{ent:.3f}' if ent else 'N/A'}",
            flush=True,
        )
    elif result == "cached":
        print(f"{region_key}: cached", flush=True)
    else:
        print(f"{region_key}: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute OSMnx network metrics from existing GraphML files"
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if already cached"
    )
    parser.add_argument(
        "--region", type=str, default=None,
        help="Process single region (for parallel xargs usage)",
    )
    args = parser.parse_args()

    # Single-region mode (for parallel execution)
    if args.region:
        process_single_region(args.region, args.env, force=args.force)
        return

    # Batch mode
    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    data_dir = config["data_dir"]

    regions_json = os.path.join(PROJECT_ROOT, "regions.json")
    registry = RegionRegistry(regions_json, data_dir=data_dir)

    regions = list(registry.list_by_level("city").items())
    total = len(regions)
    print(f"Found {total} city-level regions", flush=True)

    computed = 0
    cached = 0
    missing = 0
    errors = 0
    total_time = 0

    for i, (key, region) in enumerate(sorted(regions), 1):
        result, elapsed = compute_metrics_for_region(region, data_dir, force=args.force)

        if result == "computed":
            computed += 1
            total_time += elapsed
            circ = region.get_stat("circuity_avg")
            ent = region.get_stat("orientation_entropy")
            print(
                f"  [{i}/{total}] {key}: done in {elapsed:.0f}s — "
                f"circuity={f'{circ:.4f}' if circ else 'N/A'}, "
                f"entropy={f'{ent:.3f}' if ent else 'N/A'}",
                flush=True,
            )
        elif result == "cached":
            cached += 1
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] ... ({cached} cached so far)", flush=True)
        elif result in ("no_graphml", "empty_file"):
            missing += 1
            print(f"  [{i}/{total}] {key}: {result}", flush=True)
        else:
            errors += 1
            print(f"  [{i}/{total}] {key}: {result}", flush=True)

    print(f"\nDone: {computed} computed, {cached} cached, {missing} missing/empty, {errors} errors")
    if computed > 0:
        print(f"Total compute time: {total_time:.0f}s (avg {total_time/computed:.1f}s per city)")


if __name__ == "__main__":
    main()
