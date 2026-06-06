"""Surgically refresh kappa-dependent city emission metrics.

This tool uses existing OD-pair CSVs and does not run demand, buildings,
roadnet, equity, profit, or the full city pipeline.

Usage:
    cd src/drone_emission
    python -m tools.recompute_kappa_city_metrics --env remote --dry-run
    python -m tools.recompute_kappa_city_metrics --env remote --consolidate --enrich
    python -m tools.recompute_kappa_city_metrics --region shanghai --env remote
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "drone_emission", "emission"))

from cal_single_delivery import (  # noqa: E402
    LifeCycleEmissions,
    create_ebike_params,
    get_emission_params,
)
from utils import load_config  # noqa: E402
from utils.grid_emission_factor import fetch_factor, normalize_province_name  # noqa: E402
from utils.province import get_province_from_district  # noqa: E402
from utils.region_config import RegionConfig, init_registry  # noqa: E402

REGIONS_JSON = os.path.join(PROJECT_ROOT, "regions.json")
STALE_THRESHOLD_STATS = (
    "max_lifecycle_reduction_pct",
    "optimal_threshold_km",
    "optimal_drone_ratio_pct",
)
REQUIRED_OD_COLUMNS = (
    "road_distance_m",
    "distance_km",
    "annual_sales",
    "drone_lifecycle_g",
)


@dataclass
class ProcessResult:
    region_key: str
    status: str
    elapsed_s: float
    recomputed_ebike: bool = False


def _od_path(exp_config: dict, region_info: RegionConfig) -> str:
    name_en = region_info.region_name_en.lower()
    return os.path.join(
        exp_config["data_dir"],
        region_info.region_level,
        name_en,
        f"{name_en}_od_pairs.csv",
    )


def _resolve_province(region_info: RegionConfig, exp_config: dict) -> str:
    province = region_info.get_stat("province")
    if province:
        return province
    province_raw = get_province_from_district(
        region_info.region_name_cn,
        exp_config["sales_path"],
    )
    return normalize_province_name(province_raw)


def _has_matching_kappa(od_df: pd.DataFrame, ebike_kappa: float) -> bool:
    if "ebike_kappa" not in od_df.columns:
        return False
    cached = pd.to_numeric(od_df["ebike_kappa"], errors="coerce").dropna().to_numpy()
    return cached.size == len(od_df) and np.allclose(cached, ebike_kappa)


def _annual_sales(od_df: pd.DataFrame) -> np.ndarray:
    sales = pd.to_numeric(od_df["annual_sales"], errors="coerce").fillna(0).to_numpy()
    if sales.sum() <= 0:
        raise ValueError("annual_sales has no positive total")
    return sales


def _recompute_ebike_columns(
    od_df: pd.DataFrame,
    exp_config: dict,
    province_gef: float,
    ebike_kappa: float,
) -> pd.DataFrame:
    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    ebike_params = create_ebike_params(exp_config)
    ebike_params.e_use = province_gef
    road_distances_km = pd.to_numeric(
        od_df["road_distance_m"], errors="coerce"
    ).to_numpy() / 1000.0
    ebike_effective_km = road_distances_km * ebike_kappa

    lca = LifeCycleEmissions(ebike_params=ebike_params)
    ebike_emissions = np.array(
        [
            lca.calculate_ebike_use_phase_emissions(
                distance_km,
                payload_kg=payload_kg,
                rider_weight=rider_weight,
            )["emissions_kg_co2e"]
            for distance_km in ebike_effective_km
        ],
        dtype=float,
    )
    ebike_lifecycle = np.array(
        [
            lca.calculate_lifecycle_emissions_per_order_simple(
                distance_km,
                n_orders_per_delivery=1,
                vehicle_type="ebike",
            )["total_per_order_g"]
            for distance_km in ebike_effective_km
        ],
        dtype=float,
    )

    od_df["ebike_emission_kg"] = ebike_emissions
    od_df["ebike_lifecycle_g"] = ebike_lifecycle
    od_df["ebike_kappa"] = ebike_kappa
    return od_df


def _update_weighted_stats(
    region_info: RegionConfig,
    od_df: pd.DataFrame,
    ebike_kappa: float,
) -> None:
    annual_sales = _annual_sales(od_df)
    drone_lifecycle = pd.to_numeric(
        od_df["drone_lifecycle_g"], errors="coerce"
    ).to_numpy()
    ebike_lifecycle = pd.to_numeric(
        od_df["ebike_lifecycle_g"], errors="coerce"
    ).to_numpy()
    if np.isnan(drone_lifecycle).any() or np.isnan(ebike_lifecycle).any():
        raise ValueError("lifecycle columns contain NaN")

    total_ebike_lifecycle = np.sum(ebike_lifecycle * annual_sales) / 1000.0
    total_drone_lifecycle = np.sum(drone_lifecycle * annual_sales) / 1000.0
    lifecycle_reduction = (
        (total_ebike_lifecycle - total_drone_lifecycle)
        / total_ebike_lifecycle
        * 100.0
        if total_ebike_lifecycle > 0
        else 0.0
    )

    use_drone_oracle = drone_lifecycle < ebike_lifecycle
    oracle_lifecycle = (
        np.sum(np.where(use_drone_oracle, drone_lifecycle, ebike_lifecycle) * annual_sales)
        / 1000.0
    )
    oracle_reduction = (
        (total_ebike_lifecycle - oracle_lifecycle) / total_ebike_lifecycle * 100.0
        if total_ebike_lifecycle > 0
        else 0.0
    )
    oracle_drone_order_ratio = (
        annual_sales[use_drone_oracle].sum() / annual_sales.sum() * 100.0
    )

    region_info.set_stat("total_ebike_lifecycle", total_ebike_lifecycle)
    region_info.set_stat("total_drone_lifecycle", total_drone_lifecycle)
    region_info.set_stat(
        "drone_minus_ebike_lifecycle_kg",
        total_drone_lifecycle - total_ebike_lifecycle,
    )
    region_info.set_stat(
        "lifecycle_all_drone_reduction_pct",
        lifecycle_reduction,
    )
    region_info.set_stat("oracle_min_lifecycle_kg", oracle_lifecycle)
    region_info.set_stat(
        "oracle_min_reduction_vs_ebike_pct",
        oracle_reduction,
    )
    region_info.set_stat("oracle_drone_order_ratio_pct", oracle_drone_order_ratio)
    region_info.set_stat("ebike_kappa", ebike_kappa)

    for key in STALE_THRESHOLD_STATS:
        region_info.stats.pop(key, None)


def _process_region(
    region_key: str,
    region_info: RegionConfig,
    exp_config: dict,
    dry_run: bool,
    write_od: bool,
    force_ebike: bool,
) -> ProcessResult:
    t0 = time.time()
    od_path = _od_path(exp_config, region_info)
    if not os.path.exists(od_path):
        return ProcessResult(region_key, "missing_od", time.time() - t0)

    try:
        od_df = pd.read_csv(od_path)
        if len(od_df) == 0:
            return ProcessResult(region_key, "empty_od", time.time() - t0)
        missing = [col for col in REQUIRED_OD_COLUMNS if col not in od_df.columns]
        if missing:
            return ProcessResult(
                region_key,
                f"missing_columns:{','.join(missing)}",
                time.time() - t0,
            )

        province = _resolve_province(region_info, exp_config)
        gef_path = exp_config.get("gef_path")
        emission_cfg = get_emission_params(exp_config)
        province_gef = (
            fetch_factor(province, gef_path)
            if gef_path
            else emission_cfg.get("e_use", 0.4)
        )
        ebike_kappa = float(getattr(create_ebike_params(exp_config), "kappa", 1.0))

        needs_ebike = (
            force_ebike
            or "ebike_lifecycle_g" not in od_df.columns
            or "ebike_emission_kg" not in od_df.columns
            or not _has_matching_kappa(od_df, ebike_kappa)
        )
        if needs_ebike:
            od_df = _recompute_ebike_columns(
                od_df,
                exp_config,
                province_gef,
                ebike_kappa,
            )

        _update_weighted_stats(region_info, od_df, ebike_kappa)
        region_info.set_stat("province", province)

        if not dry_run:
            if needs_ebike and write_od:
                od_df.to_csv(od_path, index=False)
            region_info.save_stats()

        status = "ok_recomputed_ebike" if needs_ebike else "ok_stats_only"
        if dry_run:
            status = f"dry_run_{status}"
        return ProcessResult(region_key, status, time.time() - t0, needs_ebike)
    except Exception as exc:
        print(f"[error] {region_key}: {exc}", flush=True)
        traceback.print_exc()
        return ProcessResult(region_key, "error", time.time() - t0)


def _select_regions(registry, level: str, region: str | None) -> list[tuple[str, RegionConfig]]:
    if region is None:
        return sorted(registry.list_by_level(level).items())
    if region in registry:
        return [(region, registry[region])]
    region_lower = region.lower()
    for key, region_info in registry.items():
        if region_info.region_name_en.lower() == region_lower:
            return [(key, region_info)]
    raise KeyError(f"Region '{region}' not found")


def _maybe_consolidate(exp_config: dict, run_enrich: bool) -> None:
    from tools.consolidate_stats import consolidate

    consolidate(exp_config["data_dir"])
    if run_enrich:
        from tools.enrich_stats import enrich

        enrich(
            data_dir=exp_config["data_dir"],
            sales_path=exp_config["sales_path"],
            gef_path=exp_config["gef_path"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute only kappa-dependent OD columns and weighted city stats."
    )
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--level", type=str, default="city")
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-write-od",
        action="store_true",
        help="Refresh stats only; do not write recomputed OD CSV columns.",
    )
    parser.add_argument(
        "--force-ebike",
        action="store_true",
        help="Recompute e-bike OD columns even when ebike_kappa already matches.",
    )
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Run tools.consolidate_stats after per-city updates.",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Run tools.enrich_stats after consolidation.",
    )
    args = parser.parse_args()

    exp_config = load_config(
        os.path.join(PROJECT_ROOT, "experiment.toml"),
        env=args.env,
    )
    registry = init_registry(REGIONS_JSON, data_dir=exp_config["data_dir"])
    regions = _select_regions(registry, args.level, args.region)

    print(
        f"Processing {len(regions)} {args.level} region(s); "
        f"dry_run={args.dry_run}, write_od={not args.no_write_od}"
    )
    counts: dict[str, int] = {}
    recomputed = 0
    elapsed_total = 0.0
    for idx, (region_key, region_info) in enumerate(regions, 1):
        result = _process_region(
            region_key,
            region_info,
            exp_config,
            dry_run=args.dry_run,
            write_od=not args.no_write_od,
            force_ebike=args.force_ebike,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        recomputed += int(result.recomputed_ebike)
        elapsed_total += result.elapsed_s
        if result.status != "ok_stats_only" or idx % 25 == 0:
            print(
                f"[{idx}/{len(regions)}] {result.region_key}: "
                f"{result.status} ({result.elapsed_s:.2f}s)",
                flush=True,
            )

    print("=" * 60)
    print(f"Summary: {counts}")
    print(f"E-bike OD columns recomputed: {recomputed}")
    print(f"Elapsed: {elapsed_total:.1f}s")

    if args.consolidate and not args.dry_run:
        _maybe_consolidate(exp_config, run_enrich=args.enrich)
    elif args.enrich:
        print("--enrich ignored without --consolidate")


if __name__ == "__main__":
    main()
