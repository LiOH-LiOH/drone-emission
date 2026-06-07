"""Refresh per-city lifecycle component emissions from existing OD CSVs.

This tool does not run demand, buildings, roadnet, equity, profit, or the full
city pipeline. It recomputes lifecycle production/use/end-of-life columns and
weighted city stats from data/{level}/{city}/{city}_od_pairs.csv.

Usage:
    cd src/drone_emission
    python -m tools.recompute_lifecycle_parts --env remote --dry-run
    python -m tools.recompute_lifecycle_parts --env remote --consolidate --enrich
    python -m tools.recompute_lifecycle_parts --region shanghai --env remote
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

from emission.cal_single_delivery import (  # noqa: E402
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from sensitivity.vectorized_lca import (  # noqa: E402
    DroneScenario,
    EbikeScenario,
    compute_drone_per_order_parts_g,
    compute_ebike_per_order_parts_g,
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
)
LIFECYCLE_COLUMNS = (
    "ebike_lifecycle_g",
    "ebike_lifecycle_production_g",
    "ebike_lifecycle_use_g",
    "ebike_lifecycle_eol_g",
    "drone_lifecycle_g",
    "drone_lifecycle_production_g",
    "drone_lifecycle_use_g",
    "drone_lifecycle_eol_g",
)


@dataclass
class ProcessResult:
    region_key: str
    status: str
    elapsed_s: float


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


def _annual_sales(od_df: pd.DataFrame) -> np.ndarray:
    sales = pd.to_numeric(od_df["annual_sales"], errors="coerce").fillna(0).to_numpy()
    if sales.sum() <= 0:
        raise ValueError("annual_sales has no positive total")
    return sales


def _weighted_kg(values_g: np.ndarray, annual_sales: np.ndarray) -> float:
    return float(np.sum(values_g * annual_sales) / 1000.0)


def _write_lifecycle_columns(
    od_df: pd.DataFrame,
    ebike_parts: dict[str, np.ndarray],
    drone_parts: dict[str, np.ndarray],
) -> None:
    od_df["ebike_lifecycle_g"] = ebike_parts["total_g"]
    od_df["ebike_lifecycle_production_g"] = ebike_parts["production_g"]
    od_df["ebike_lifecycle_use_g"] = ebike_parts["use_g"]
    od_df["ebike_lifecycle_eol_g"] = ebike_parts["end_of_life_g"]
    od_df["drone_lifecycle_g"] = drone_parts["total_g"]
    od_df["drone_lifecycle_production_g"] = drone_parts["production_g"]
    od_df["drone_lifecycle_use_g"] = drone_parts["use_g"]
    od_df["drone_lifecycle_eol_g"] = drone_parts["end_of_life_g"]


def _update_weighted_stats(
    region_info: RegionConfig,
    od_df: pd.DataFrame,
    annual_sales: np.ndarray,
    ebike_kappa: float,
) -> None:
    values = {
        col: pd.to_numeric(od_df[col], errors="coerce").to_numpy()
        for col in LIFECYCLE_COLUMNS
    }
    nan_cols = [col for col, arr in values.items() if np.isnan(arr).any()]
    if nan_cols:
        raise ValueError(f"lifecycle columns contain NaN: {','.join(nan_cols)}")

    total_ebike_lifecycle = _weighted_kg(values["ebike_lifecycle_g"], annual_sales)
    total_drone_lifecycle = _weighted_kg(values["drone_lifecycle_g"], annual_sales)
    lifecycle_reduction = (
        (total_ebike_lifecycle - total_drone_lifecycle)
        / total_ebike_lifecycle
        * 100.0
        if total_ebike_lifecycle > 0
        else 0.0
    )

    use_drone_oracle = values["drone_lifecycle_g"] < values["ebike_lifecycle_g"]
    oracle_lifecycle = (
        np.sum(
            np.where(
                use_drone_oracle,
                values["drone_lifecycle_g"],
                values["ebike_lifecycle_g"],
            )
            * annual_sales
        )
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
        "total_ebike_production_emission_kg",
        _weighted_kg(values["ebike_lifecycle_production_g"], annual_sales),
    )
    region_info.set_stat(
        "total_ebike_use_phase_emission_kg",
        _weighted_kg(values["ebike_lifecycle_use_g"], annual_sales),
    )
    region_info.set_stat(
        "total_ebike_end_of_life_emission_kg",
        _weighted_kg(values["ebike_lifecycle_eol_g"], annual_sales),
    )
    region_info.set_stat(
        "total_drone_production_emission_kg",
        _weighted_kg(values["drone_lifecycle_production_g"], annual_sales),
    )
    region_info.set_stat(
        "total_drone_use_phase_emission_kg",
        _weighted_kg(values["drone_lifecycle_use_g"], annual_sales),
    )
    region_info.set_stat(
        "total_drone_end_of_life_emission_kg",
        _weighted_kg(values["drone_lifecycle_eol_g"], annual_sales),
    )
    region_info.set_stat(
        "drone_minus_ebike_lifecycle_kg",
        total_drone_lifecycle - total_ebike_lifecycle,
    )
    region_info.set_stat(
        "lifecycle_all_drone_reduction_pct",
        lifecycle_reduction,
    )
    region_info.set_stat("oracle_min_lifecycle_kg", float(oracle_lifecycle))
    region_info.set_stat(
        "oracle_min_reduction_vs_ebike_pct",
        float(oracle_reduction),
    )
    region_info.set_stat(
        "oracle_drone_order_ratio_pct",
        float(oracle_drone_order_ratio),
    )
    region_info.set_stat("ebike_kappa", ebike_kappa)

    for key in STALE_THRESHOLD_STATS:
        region_info.stats.pop(key, None)


def _process_region(
    region_key: str,
    region_info: RegionConfig,
    exp_config: dict,
    dry_run: bool,
    write_od: bool,
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

        emission_cfg = get_emission_params(exp_config)
        province = _resolve_province(region_info, exp_config)
        gef_path = exp_config.get("gef_path")
        province_gef = (
            fetch_factor(province, gef_path)
            if gef_path
            else emission_cfg.get("e_use", 0.4)
        )

        ebike_params = create_ebike_params(exp_config)
        drone_params = create_drone_params(exp_config)
        ebike_params.e_use = province_gef
        drone_params.e_use = province_gef
        ebike_kappa = float(getattr(ebike_params, "kappa", 1.0))

        road_distances_km = pd.to_numeric(
            od_df["road_distance_m"], errors="coerce"
        ).to_numpy() / 1000.0
        drone_distances_km = pd.to_numeric(
            od_df["distance_km"], errors="coerce"
        ).to_numpy()
        if np.isnan(road_distances_km).any() or np.isnan(drone_distances_km).any():
            raise ValueError("distance columns contain NaN")

        ebike_parts = compute_ebike_per_order_parts_g(
            road_distances_km,
            ebike_params,
            EbikeScenario(kappa=ebike_kappa),
        )
        drone_parts = compute_drone_per_order_parts_g(
            drone_distances_km,
            drone_params,
            DroneScenario(),
        )
        _write_lifecycle_columns(od_df, ebike_parts, drone_parts)
        od_df["ebike_kappa"] = ebike_kappa

        annual_sales = _annual_sales(od_df)
        _update_weighted_stats(region_info, od_df, annual_sales, ebike_kappa)
        region_info.set_stat("province", province)

        if not dry_run:
            if write_od:
                od_df.to_csv(od_path, index=False)
            region_info.save_stats()

        status = "ok"
        if dry_run:
            status = f"dry_run_{status}"
        elif not write_od:
            status = "ok_stats_only"
        return ProcessResult(region_key, status, time.time() - t0)
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
        description="Recompute lifecycle component OD columns and weighted city stats."
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
    elapsed_total = 0.0
    for idx, (region_key, region_info) in enumerate(regions, 1):
        result = _process_region(
            region_key,
            region_info,
            exp_config,
            dry_run=args.dry_run,
            write_od=not args.no_write_od,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        elapsed_total += result.elapsed_s
        if result.status != "ok" or idx % 25 == 0:
            print(
                f"[{idx}/{len(regions)}] {result.region_key}: "
                f"{result.status} ({result.elapsed_s:.2f}s)",
                flush=True,
            )

    print("=" * 60)
    print(f"Summary: {counts}")
    print(f"Elapsed: {elapsed_total:.1f}s")

    if args.consolidate and not args.dry_run:
        _maybe_consolidate(exp_config, run_enrich=args.enrich)
    elif args.enrich:
        print("--enrich ignored without --consolidate")


if __name__ == "__main__":
    main()
