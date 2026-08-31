"""Read locked distance-threshold inputs without exposing a plotting CLI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY_REQUIRED_COLUMNS = {
    "region_key",
    "region_name_en",
    "distance_basis",
    "policy_case",
    "optimal_threshold_km",
    "annual_reduction_kg",
    "annual_reduction_pct",
    "drone_order_ratio_pct",
    "n_od_pairs_total",
    "n_od_pairs_optimized",
    "n_od_pairs_unreachable",
    "unreachable_annual_orders",
    "total_annual_orders",
    "analysis_version",
    "input_size_bytes",
    "input_mtime_ns",
}
def _resolve_summary_path(data_dir: Path, requested: str | None) -> Path:
    if requested:
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(f"Threshold summary not found: {path}")
        return path

    partial = data_dir / "all_city_distance_threshold_summary.partial.csv"
    complete = data_dir / "all_city_distance_threshold_summary.csv"
    if complete.exists():
        return complete
    if partial.exists():
        raise FileNotFoundError(
            f"Distance-threshold analysis is partial: {partial}. "
            "Run `python -m drone_emission analyze` for the locked cohort."
        )
    raise FileNotFoundError(
        "No city distance-threshold summary found. Run the standalone "
        "distance-threshold analysis first."
    )


def _load_summary(path: Path, data_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(path)
    missing = SUMMARY_REQUIRED_COLUMNS.difference(summary.columns)
    if missing:
        raise ValueError(f"Threshold summary is missing columns: {sorted(missing)}")
    if summary.duplicated(["region_key", "distance_basis"]).any():
        raise ValueError("Threshold summary contains duplicate city/basis rows")
    basis_sets = summary.groupby("region_key")["distance_basis"].agg(set)
    incomplete = basis_sets[basis_sets != {"road", "straight"}]
    if not incomplete.empty:
        raise ValueError(
            "Threshold summary has incomplete city records: "
            + ", ".join(incomplete.index.astype(str))
        )

    status_path = data_dir / "all_city_distance_threshold_status.csv"
    configured_count = summary["region_key"].nunique()
    if status_path.exists():
        status = pd.read_csv(status_path, usecols=["region_key"])
        configured_count = status["region_key"].nunique()
    summary.attrs["configured_city_count"] = int(configured_count)
    summary.attrs["source_name"] = path.name
    return summary


def _load_oracle_results(data_dir: Path) -> pd.DataFrame:
    """Load city-level trip-optimal lifecycle results."""
    path = data_dir / "all_region_stats.csv"
    if not path.exists():
        raise FileNotFoundError(f"All-region statistics not found: {path}")
    oracle = pd.read_csv(path)
    if "region_key" not in oracle.columns:
        raise ValueError("All-region statistics are missing region_key")
    has_baseline = "total_ebike_lifecycle" in oracle.columns
    has_direct = "oracle_min_reduction_vs_ebike_pct" in oracle.columns
    has_components = {
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
    }.issubset(oracle.columns)
    if not has_baseline or (not has_direct and not has_components):
        raise ValueError(
            "All-region statistics are missing the all-e-bike baseline or "
            "trip-level Oracle metrics"
        )
    return oracle
