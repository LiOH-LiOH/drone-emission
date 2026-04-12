"""
Compute OSMnx network metrics for all regions from existing GraphML files.

Lightweight script — does NOT re-run the full pipeline. Just loads each
city's GraphML, computes structural metrics, and saves to region_stats.json.

Usage:
    cd src/drone_emission
    python -m tools.compute_network_metrics
    python -m tools.compute_network_metrics --env remote
    python -m tools.compute_network_metrics --force   # recompute even if cached
"""

import argparse
import os
import sys

import osmnx as ox

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from utils import load_config
from utils.region_config import RegionRegistry


def compute_metrics_for_region(region, data_dir, force=False):
    """Compute OSMnx metrics for a single region from its GraphML file."""
    level = region.region_level
    name_en = region.region_name_en.lower()

    # Check cache
    if not force and region.get_stat("circuity_avg") is not None:
        return "cached"

    # Find GraphML file
    graphml_path = os.path.join(data_dir, level, name_en, f"{name_en}_roadnet.graphml")
    if not os.path.exists(graphml_path):
        return "no_graphml"

    try:
        G = ox.load_graphml(graphml_path)
        Gu = ox.convert.to_undirected(G)

        # Basic stats
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

        # Orientation entropy
        try:
            entropy = ox.bearing.orientation_entropy(Gu)
            region.set_stat("orientation_entropy", entropy)
        except Exception:
            pass

        del G, Gu
        region.save_stats()
        return "computed"

    except Exception as e:
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute OSMnx network metrics from existing GraphML files"
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if already cached"
    )
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    data_dir = config["data_dir"]

    regions_json = os.path.join(PROJECT_ROOT, "regions.json")
    registry = RegionRegistry(regions_json, data_dir=data_dir)

    regions = list(registry.list_by_level("city").items())
    total = len(regions)
    print(f"Found {total} city-level regions")

    computed = 0
    cached = 0
    missing = 0
    errors = 0

    for i, (key, region) in enumerate(sorted(regions), 1):
        result = compute_metrics_for_region(region, data_dir, force=args.force)

        if result == "computed":
            computed += 1
            circ = region.get_stat("circuity_avg")
            ent = region.get_stat("orientation_entropy")
            print(
                f"  [{i}/{total}] {key}: computed — "
                f"circuity={circ:.4f}, "
                f"entropy={f'{ent:.3f}' if ent else 'N/A'}"
            )
        elif result == "cached":
            cached += 1
            if i % 50 == 0 or i == total:
                print(f"  [{i}/{total}] ... ({cached} cached so far)")
        elif result == "no_graphml":
            missing += 1
            print(f"  [{i}/{total}] {key}: no GraphML file")
        else:
            errors += 1
            print(f"  [{i}/{total}] {key}: {result}")

    print(f"\nDone: {computed} computed, {cached} cached, {missing} no GraphML, {errors} errors")


if __name__ == "__main__":
    main()
