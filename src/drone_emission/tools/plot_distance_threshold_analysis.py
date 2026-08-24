"""Render exploratory distance-threshold analysis figures.

Curve generation remains an explicit prerequisite. This command reads the
compact summary and curve CSVs only; it never opens the large OD-pair files.

Usage:
    cd src/drone_emission
    ../../.venv/bin/python -m tools.plot_distance_threshold_analysis --env remote
    ../../.venv/bin/python -m tools.plot_distance_threshold_analysis --only all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from drone_emission.utils import load_config
from drone_emission.visualization.analysis import (
    DISTANCE_THRESHOLD_TYPICAL_CITIES,
    plot_distance_threshold_city_landscape,
    plot_distance_threshold_oracle_performance,
    plot_distance_threshold_typical_city_curves,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_REQUIRED_COLUMNS = {
    "region_key",
    "region_name_en",
    "distance_basis",
    "policy_case",
    "optimal_threshold_km",
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
CURVE_REQUIRED_COLUMNS = {
    "region_key",
    "region_name_en",
    "distance_basis",
    "threshold_km",
    "annual_reduction_pct",
    "drone_order_ratio_pct",
    "is_exact_optimum",
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
    if partial.exists():
        return partial
    if complete.exists():
        return complete
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
    has_direct = "oracle_min_reduction_vs_ebike_pct" in oracle.columns
    has_components = {
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
    }.issubset(oracle.columns)
    if not has_direct and not has_components:
        raise ValueError(
            "All-region statistics are missing trip-level Oracle metrics"
        )
    return oracle


def _curve_generation_command(env: str | None) -> str:
    env_flag = f" --env {env}" if env else ""
    regions = " ".join(
        f"--region {city}" for city in DISTANCE_THRESHOLD_TYPICAL_CITIES
    )
    return (
        "python -m tools.export_distance_threshold_curves"
        f"{env_flag} {regions}"
    )


def _coerce_exact_optimum(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false"})
    if invalid.any():
        raise ValueError("is_exact_optimum must contain only true/false values")
    return normalized == "true"


def _load_typical_curves(
    data_dir: Path,
    summary: pd.DataFrame,
    env: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    problems: list[str] = []
    for city in DISTANCE_THRESHOLD_TYPICAL_CITIES:
        curve_path = (
            data_dir
            / "city"
            / city
            / "analysis"
            / "distance_threshold"
            / "curve.csv"
        )
        if not curve_path.exists():
            problems.append(f"{city}: curve.csv is missing")
            continue
        curve = pd.read_csv(curve_path)
        missing = CURVE_REQUIRED_COLUMNS.difference(curve.columns)
        if missing:
            problems.append(f"{city}: missing columns {sorted(missing)}")
            continue
        if set(curve["region_key"]) != {city}:
            problems.append(f"{city}: region_key does not match directory")
            continue
        if set(curve["distance_basis"]) != {"road", "straight"}:
            problems.append(f"{city}: road/straight curves are incomplete")
            continue

        city_summary = summary[summary["region_key"] == city]
        if len(city_summary) != 2:
            problems.append(f"{city}: summary does not contain two basis rows")
            continue
        for metadata in (
            "analysis_version",
            "input_size_bytes",
            "input_mtime_ns",
        ):
            expected = set(city_summary[metadata].astype(str))
            observed = set(curve[metadata].astype(str))
            if len(expected) != 1 or observed != expected:
                problems.append(f"{city}: stale curve metadata for {metadata}")
                break
        else:
            curve["is_exact_optimum"] = _coerce_exact_optimum(
                curve["is_exact_optimum"]
            )
            optimum_counts = curve.groupby("distance_basis")[
                "is_exact_optimum"
            ].sum()
            if not (optimum_counts == 1).all():
                problems.append(f"{city}: each basis must mark one exact optimum")
                continue
            frames.append(curve)

    if problems:
        command = _curve_generation_command(env)
        raise RuntimeError(
            "Typical-city curve prerequisites are incomplete:\n- "
            + "\n- ".join(problems)
            + f"\nGenerate fresh curves with:\n  {command}"
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render exploratory city distance-threshold figures"
    )
    parser.add_argument("--env", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument(
        "--only",
        choices=("typical", "all", "both"),
        default="both",
    )
    parser.add_argument(
        "--untitled",
        action="store_true",
        help="Omit exploratory figure titles and the _t filename suffix.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    analysis_figures_dir = (
        Path(config["figures_dir"]) / "analysis" / "distance_threshold"
    )
    paper_subfigure_dir = Path(config["figures_dir"]) / "paper" / "subfigure"
    summary_path = _resolve_summary_path(data_dir, args.summary_path)
    summary = _load_summary(summary_path, data_dir)
    plot_title = not args.untitled

    if args.only in {"typical", "both"}:
        curves = _load_typical_curves(data_dir, summary, args.env)
        plot_distance_threshold_typical_city_curves(
            curves,
            summary,
            save_path=str(
                analysis_figures_dir
                / "distance_threshold_typical_city_curves.png"
            ),
            plot_title=plot_title,
        )

    if args.only in {"all", "both"}:
        oracle = _load_oracle_results(data_dir)
        plot_distance_threshold_city_landscape(
            summary,
            oracle,
            save_path=str(
                paper_subfigure_dir / "distance_threshold_city_landscape.pdf"
            ),
            for_latex=True,
            plot_title=plot_title,
        )
        plot_distance_threshold_oracle_performance(
            summary,
            oracle,
            save_path=str(
                paper_subfigure_dir
                / "distance_threshold_oracle_performance.pdf"
            ),
            for_latex=True,
            plot_title=plot_title,
        )

    print(
        f"Rendered {args.only} distance-threshold figure set from "
        f"{summary.attrs['source_name']}"
    )


if __name__ == "__main__":
    main()
