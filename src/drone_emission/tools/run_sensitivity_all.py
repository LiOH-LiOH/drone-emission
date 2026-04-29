"""Standalone sensitivity-stage runner.

Iterates over all regions in ``regions.json`` (plus any districts present
under ``data/district/``), skips regions whose 3 sensitivity CSVs already
exist, and calls ``sensitivity.api.run_sensitivity`` for the rest. Designed
to be invoked once cal_emission has populated the per-region OD CSV.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.run_sensitivity_all                # all regions in registry
    ../../.venv/bin/python -m tools.run_sensitivity_all --region linzhi
    ../../.venv/bin/python -m tools.run_sensitivity_all --env remote
    ../../.venv/bin/python -m tools.run_sensitivity_all --include-districts

Returns exit 0 on completion (per-region failures are logged but don't
abort the run).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission"))

from sensitivity.api import run_sensitivity  # noqa: E402
from utils import load_config  # noqa: E402
from utils.region_config import RegionConfig, init_registry  # noqa: E402

REGIONS_JSON = os.path.join(PROJECT_ROOT, "regions.json")


def _outputs_exist(out_dir: str, region_name_en: str) -> bool:
    suffixes = ("_oat.csv", "_shapley.csv", "_named.csv")
    return all(
        os.path.exists(os.path.join(out_dir, f"{region_name_en}{s}")) for s in suffixes
    )


def _od_csv_exists(out_dir: str, region_name_en: str) -> bool:
    return os.path.exists(os.path.join(out_dir, f"{region_name_en}_od_pairs.csv"))


def _scan_districts(data_dir: str) -> list[RegionConfig]:
    """Build RegionConfig stubs for each district directory present locally.

    Districts may not appear in regions.json (e.g. nanshan); this lets the
    standalone runner cover them.
    """
    out: list[RegionConfig] = []
    district_root = os.path.join(data_dir, "district")
    if not os.path.isdir(district_root):
        return out
    for name in sorted(os.listdir(district_root)):
        path = os.path.join(district_root, name)
        if not os.path.isdir(path):
            continue
        rc = RegionConfig(
            region_name_cn=name,
            region_name_en=name,
            region_level="district",
        )
        rc._stats_path = os.path.join(path, "region_stats.json")
        rc.load_stats()
        out.append(rc)
    return out


def _process_region(region_info, config: dict) -> tuple[str, str, float]:
    """Return (region_key, status, elapsed_seconds). status ∈ {ok, skipped, missing_od, error}."""
    region_dict = region_info.to_dict()
    region_name_en = region_dict["region_name_en"].lower()
    region_level = region_dict["region_level"]
    region_key = region_dict.get("region_key") or region_name_en
    out_dir = os.path.join(config["data_dir"], region_level, region_name_en)

    if _outputs_exist(out_dir, region_name_en):
        return region_key, "skipped", 0.0
    if not _od_csv_exists(out_dir, region_name_en):
        return region_key, "missing_od", 0.0

    t0 = time.time()
    try:
        run_sensitivity(region_info=region_info, exp_config=config)
        return region_key, "ok", time.time() - t0
    except Exception as e:
        print(f"  [error] {region_key}: {e}")
        traceback.print_exc()
        return region_key, "error", time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--region", type=str, default=None,
                         help="Only run this region")
    parser.add_argument("--include-districts", action="store_true",
                         help="Also process districts found in data/district/")
    parser.add_argument("--level", type=str, default=None,
                         help="Only process regions at this level (city/district)")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "experiment.toml"), env=args.env)
    registry = init_registry(REGIONS_JSON, data_dir=config["data_dir"])

    regions: list = []
    if args.region:
        if args.region in registry:
            regions = [(args.region, registry[args.region])]
        else:
            # Try district lookup
            for rc in _scan_districts(config["data_dir"]):
                if rc.region_name_en.lower() == args.region.lower():
                    regions = [(rc.region_name_en, rc)]
                    break
        if not regions:
            print(f"Region '{args.region}' not found in registry or district scan.")
            sys.exit(1)
    else:
        regions = list(registry.items())
        if args.include_districts:
            for rc in _scan_districts(config["data_dir"]):
                regions.append((rc.region_name_en, rc))
        if args.level:
            regions = [(k, r) for (k, r) in regions if r.region_level == args.level]

    n = len(regions)
    print(f"Processing {n} regions...")

    counts = {"ok": 0, "skipped": 0, "missing_od": 0, "error": 0}
    errors: list[str] = []
    elapsed_total = 0.0

    for i, (region_key, region_info) in enumerate(regions, 1):
        rk, status, elapsed = _process_region(region_info, config)
        counts[status] += 1
        elapsed_total += elapsed
        print(f"[{i}/{n}] {rk}: {status} ({elapsed:.1f}s)", flush=True)
        if status == "error":
            errors.append(rk)

    print()
    print("=" * 60)
    print(f"Summary: ok={counts['ok']}, skipped={counts['skipped']}, "
          f"missing_od={counts['missing_od']}, error={counts['error']}")
    print(f"Elapsed (sum of per-region run times): {elapsed_total:.1f}s")
    if errors:
        print(f"Errors: {', '.join(errors[:20])}{' ...' if len(errors) > 20 else ''}")


if __name__ == "__main__":
    main()
