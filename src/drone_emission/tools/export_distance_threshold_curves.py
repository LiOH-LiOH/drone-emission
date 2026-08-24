"""Manually export compact threshold-response curves for named cities.

This command is deliberately separate from the all-city threshold workflow.
It never scans every city and requires one or more explicit ``--region``
arguments.
"""

from __future__ import annotations

import argparse
import sys

from drone_emission.tools.run_distance_threshold_analysis import (
    PROJECT_ROOT,
    _city_regions,
    _lookup_region,
    analyze_region,
)
from drone_emission.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export visualization-ready threshold curves"
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--region", action="append", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    regions = _city_regions(config)
    failed = False

    for requested in args.region:
        try:
            region_key, region = _lookup_region(regions, requested)
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            failed = True
            continue
        result = analyze_region(
            region_key,
            region,
            config,
            force=args.force,
            include_curve=True,
            n_uniform=201,
        )
        print(
            f"{region_key}: {result['status']} "
            f"({result['elapsed_seconds']:.1f}s) {result['message']}",
            flush=True,
        )
        failed |= result["status"] != "ok"

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
