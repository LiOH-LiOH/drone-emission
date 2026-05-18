"""Consolidate per-region sensitivity outputs into cross-region tables.

Reads each region's ``{region}_oat.csv``, ``{region}_shapley.csv``,
``{region}_named.csv``, ``{region}_sobol.csv`` and concatenates them into:

  data/all_sensitivity_oat.csv
  data/all_sensitivity_shapley.csv
  data/all_sensitivity_named.csv
  data/all_sensitivity_sobol.csv

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.consolidate_sensitivity
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission"))

from utils import load_config  # noqa: E402

SUFFIXES = (
    ("_oat.csv", "all_sensitivity_oat.csv"),
    ("_shapley.csv", "all_sensitivity_shapley.csv"),
    ("_named.csv", "all_sensitivity_named.csv"),
    ("_sobol.csv", "all_sensitivity_sobol.csv"),
)


def _is_sensitivity_csv(filename: str, suffix: str) -> bool:
    return filename.endswith(suffix) and not filename.startswith("all_")


def consolidate(data_dir: str, level_filter: str | None = None) -> None:
    by_suffix: dict[str, list[pd.DataFrame]] = {s: [] for s, _ in SUFFIXES}
    region_count = 0

    levels = (
        [level_filter]
        if level_filter
        else [d for d in ("city", "district") if os.path.isdir(os.path.join(data_dir, d))]
    )

    for level in levels:
        level_dir = os.path.join(data_dir, level)
        if not os.path.isdir(level_dir):
            continue
        for region_name in sorted(os.listdir(level_dir)):
            region_dir = os.path.join(level_dir, region_name)
            if not os.path.isdir(region_dir):
                continue
            has_outputs = False
            for suffix, _ in SUFFIXES:
                pattern = os.path.join(region_dir, f"{region_name}{suffix}")
                if not os.path.exists(pattern):
                    continue
                df = pd.read_csv(pattern)
                if "region_key" not in df.columns:
                    df.insert(0, "region_key", region_name)
                by_suffix[suffix].append(df)
                if suffix == "_oat.csv":
                    has_outputs = True
            if has_outputs:
                region_count += 1

    print(f"Regions with sensitivity outputs: {region_count}")
    for suffix, out_name in SUFFIXES:
        frames = by_suffix[suffix]
        if not frames:
            print(f"  no rows for {suffix}; skipping")
            continue
        merged = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(data_dir, out_name)
        merged.to_csv(out_path, index=False)
        print(f"  wrote {len(merged):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--level", type=str, default=None,
                         help="Optional region_level filter (city/district)")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "experiment.toml"), env=args.env)
    consolidate(config["data_dir"], level_filter=args.level)


if __name__ == "__main__":
    main()
