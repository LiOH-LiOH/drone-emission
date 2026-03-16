"""
Consolidate per-region stats into a single summary.

Reads static config from regions.json and per-region stats from
data/{level}/{city}/region_stats.json, then outputs:
  - data/all_region_stats.json  (full merged format)
  - data/all_region_stats.csv   (one row per region, flattened)

Usage:
    cd src/drone_emission
    python consolidate_stats.py                  # consolidate
    python consolidate_stats.py --migrate        # one-time migration from regions.json
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REGIONS_JSON_PATH = os.path.join(PROJECT_ROOT, "regions.json")

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config


def stats_path_for(data_dir: str, level: str, name_en: str) -> str:
    return os.path.join(data_dir, level, name_en.lower(), "region_stats.json")


def migrate(data_dir: str):
    """One-time migration: copy stats/cache from regions.json to per-region files."""
    with open(REGIONS_JSON_PATH, "r", encoding="utf-8") as f:
        regions = json.load(f)

    migrated = 0
    skipped = 0
    for key, region in regions.items():
        stats = region.get("stats", {})
        cache = region.get("cache", {})
        if not stats and not cache:
            skipped += 1
            continue

        level = region["region_level"]
        name_en = region["region_name_en"]
        path = stats_path_for(data_dir, level, name_en)

        # Merge with existing per-region file if present
        existing_stats = {}
        existing_cache = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing_stats = existing.get("stats", {})
            existing_cache = existing.get("cache", {})

        # Per-region file takes precedence; only fill in missing keys
        merged_stats = {**stats, **existing_stats}
        merged_cache = {**cache, **existing_cache}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stats": merged_stats, "cache": merged_cache}, f, ensure_ascii=False, indent=2)
        migrated += 1

    print(f"Migrated: {migrated}, Skipped (empty): {skipped}")


def consolidate(data_dir: str):
    """Merge per-region stats into consolidated JSON and CSV."""
    with open(REGIONS_JSON_PATH, "r", encoding="utf-8") as f:
        regions = json.load(f)

    merged = {}
    completed = 0
    total = len(regions)

    # Collect all stat keys for CSV header
    all_stat_keys = set()

    for key, region in sorted(regions.items()):
        level = region["region_level"]
        name_en = region["region_name_en"]
        path = stats_path_for(data_dir, level, name_en)

        stats = {}
        cache = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            stats = data.get("stats", {})
            cache = data.get("cache", {})

        merged[key] = {**region, "stats": stats, "cache": cache}
        all_stat_keys.update(stats.keys())

        if "total_ebike_lifecycle" in stats:
            completed += 1

    # Write consolidated JSON
    out_json = os.path.join(data_dir, "all_region_stats.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Write consolidated CSV (one row per region, flattened stats)
    out_csv = os.path.join(data_dir, "all_region_stats.csv")
    stat_keys = sorted(all_stat_keys)
    fieldnames = ["region_key", "region_name_cn", "region_name_en", "region_level"] + stat_keys

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, region in sorted(merged.items()):
            row = {
                "region_key": key,
                "region_name_cn": region["region_name_cn"],
                "region_name_en": region["region_name_en"],
                "region_level": region["region_level"],
            }
            for sk in stat_keys:
                row[sk] = region["stats"].get(sk, "")
            writer.writerow(row)

    print(f"Consolidated: {completed}/{total} regions completed")
    print(f"  JSON: {out_json}")
    print(f"  CSV:  {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate per-region stats")
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="One-time migration: copy stats from regions.json to per-region files",
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    data_dir = config["data_dir"]

    if args.migrate:
        migrate(data_dir)
    else:
        consolidate(data_dir)
