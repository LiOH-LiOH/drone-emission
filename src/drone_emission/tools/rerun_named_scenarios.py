"""Recompute only the named-scenarios portion of the sensitivity stage.

OAT and Shapley don't depend on the GEF_MULT_2030 / GEF_MULT_2060 constants,
so when those change we only need to rebuild the per-region ``{region}_named.csv``
files and then re-consolidate. This tool does exactly that for every region in
the registry (plus optional districts), then calls consolidate to refresh
``data/all_sensitivity_named.csv``.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.rerun_named_scenarios
    ../../.venv/bin/python -m tools.rerun_named_scenarios --env remote
    ../../.venv/bin/python -m tools.rerun_named_scenarios --env remote --level city
    ../../.venv/bin/python -m tools.rerun_named_scenarios --include-districts
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission"))

from emission.cal_single_delivery import (  # noqa: E402
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from sensitivity.named_scenarios import run_named_scenarios_for_region  # noqa: E402
from utils import load_config  # noqa: E402
from utils.grid_emission_factor import fetch_factor, normalize_province_name  # noqa: E402
from utils.province import get_province_from_district  # noqa: E402
from utils.region_config import RegionConfig, init_registry  # noqa: E402

REGIONS_JSON = os.path.join(PROJECT_ROOT, "regions.json")


def _count_od_csvs(data_dir: str) -> int:
    """Count available OD-pair CSVs under a configured data directory."""
    count = 0
    for root, _dirs, files in os.walk(data_dir):
        count += sum(1 for name in files if name.endswith("_od_pairs.csv"))
    return count


def _scan_districts(data_dir: str) -> list[RegionConfig]:
    out: list[RegionConfig] = []
    district_root = os.path.join(data_dir, "district")
    if not os.path.isdir(district_root):
        return out
    for name in sorted(os.listdir(district_root)):
        path = os.path.join(district_root, name)
        if not os.path.isdir(path):
            continue
        rc = RegionConfig(
            region_name_cn=name, region_name_en=name, region_level="district",
        )
        rc._stats_path = os.path.join(path, "region_stats.json")
        rc.load_stats()
        out.append(rc)
    return out


def _resolve_province(region_info, exp_config) -> str:
    province = region_info.get_stat("province")
    if province:
        return province
    region_name_cn = region_info.to_dict()["region_name_cn"]
    province_raw = get_province_from_district(region_name_cn, exp_config["sales_path"])
    return normalize_province_name(province_raw)


def _process(region_info, exp_config) -> tuple[str, str, float]:
    cfg = region_info.to_dict()
    region_name_en = cfg["region_name_en"].lower()
    region_level = cfg["region_level"]
    region_key = cfg.get("region_key") or region_name_en
    out_dir = os.path.join(exp_config["data_dir"], region_level, region_name_en)
    od_path = os.path.join(out_dir, f"{region_name_en}_od_pairs.csv")
    if not os.path.exists(od_path):
        return region_key, "missing_od", 0.0
    t0 = time.time()
    try:
        od_df = pd.read_csv(od_path)
        if len(od_df) == 0:
            return region_key, "empty_od", time.time() - t0
        province = _resolve_province(region_info, exp_config)
        emission_cfg = get_emission_params(exp_config)
        gef_path = exp_config.get("gef_path")
        province_gef = (
            fetch_factor(province, gef_path) if gef_path else emission_cfg.get("e_use", 0.4)
        )
        named_df = run_named_scenarios_for_region(
            od_df,
            region_key=region_key,
            province=province,
            province_gef_kwh=province_gef,
            ebike_params=create_ebike_params(exp_config),
            drone_params=create_drone_params(exp_config),
            rider_weight=emission_cfg["rider_weight"],
            payload_kg=emission_cfg["payload_kg"],
        )
        named_df.to_csv(os.path.join(out_dir, f"{region_name_en}_named.csv"), index=False)
        return region_key, "ok", time.time() - t0
    except Exception as e:
        print(f"  [error] {region_key}: {e}")
        traceback.print_exc()
        return region_key, "error", time.time() - t0


def _select_regions(
    registry,
    data_dir: str,
    level: str | None,
    region: str | None,
    include_districts: bool,
) -> list[tuple[str, RegionConfig]]:
    regions = list(registry.items())
    if include_districts:
        for rc in _scan_districts(data_dir):
            regions.append((rc.region_name_en, rc))

    if level is not None:
        regions = [
            (key, region_info)
            for key, region_info in regions
            if region_info.to_dict().get("region_level") == level
        ]

    if region is not None:
        region_lower = region.lower()
        regions = [
            (key, region_info)
            for key, region_info in regions
            if key.lower() == region_lower
            or region_info.to_dict().get("region_name_en", "").lower() == region_lower
        ]

    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--include-districts", action="store_true")
    parser.add_argument("--level", type=str, default=None, help="Only process one level, e.g. city or district")
    parser.add_argument("--region", type=str, default=None, help="Only process one region key or English name")
    args = parser.parse_args()

    config = load_config(os.path.join(PROJECT_ROOT, "experiment.toml"), env=args.env)
    registry = init_registry(REGIONS_JSON, data_dir=config["data_dir"])
    od_count = _count_od_csvs(config["data_dir"]) if os.path.isdir(config["data_dir"]) else 0
    print(f"env={args.env or config.get('active_env', 'config default')}, data_dir={config['data_dir']}")
    print(f"Found {od_count} OD-pair CSVs under data_dir")
    if od_count == 0 and args.env is None:
        print(
            "No OD files found for the default environment. If this is running "
            "on hku-sto, rerun with --env remote."
        )

    regions = _select_regions(
        registry=registry,
        data_dir=config["data_dir"],
        level=args.level,
        region=args.region,
        include_districts=args.include_districts,
    )

    n = len(regions)
    print(f"Recomputing named scenarios for {n} regions...")
    counts = {"ok": 0, "missing_od": 0, "empty_od": 0, "error": 0}
    elapsed_total = 0.0
    for i, (_, region_info) in enumerate(regions, 1):
        rk, status, elapsed = _process(region_info, config)
        counts[status] += 1
        elapsed_total += elapsed
        if status != "ok" or i % 25 == 0:
            print(f"[{i}/{n}] {rk}: {status} ({elapsed:.2f}s)", flush=True)
    print("=" * 60)
    print(f"Summary: {counts}  total {elapsed_total:.1f}s")

    print("Consolidating named CSVs ...")
    from tools.consolidate_sensitivity import consolidate
    consolidate(config["data_dir"])


if __name__ == "__main__":
    main()
