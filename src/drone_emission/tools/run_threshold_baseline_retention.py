"""Derive and plot city baseline-retention robustness from saved MC samples.

Examples:
    python -m tools.run_threshold_baseline_retention --env remote
    python -m tools.run_threshold_baseline_retention --env remote --derive-only
    python -m tools.run_threshold_baseline_retention --env remote --plot-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "drone_emission"))

from sensitivity.baseline_retention import (  # noqa: E402
    RETENTION_CURVE_TARGETS,
    RETENTION_SCENARIOS,
    derive_city_baseline_retention,
)
from utils import load_config  # noqa: E402
from visualization.sensitivity import (  # noqa: E402
    plot_threshold_baseline_retention_curves,
    plot_threshold_baseline_retention_national,
    plot_threshold_baseline_retention_typology,
)

SUMMARY_FILENAME = "all_threshold_baseline_retention.csv"
CURVE_FILENAME = "all_threshold_baseline_retention_curve.csv"
FIGURE_FILENAMES = {
    "national": "threshold_baseline_retention_national.png",
    "curves": "threshold_baseline_retention_curves.png",
    "typology": "threshold_baseline_retention_typology.png",
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_path(data_dir: Path, region_key: str) -> Path:
    return (
        data_dir
        / "city"
        / region_key
        / "analysis"
        / "threshold_resilience"
        / "samples.csv.gz"
    )


def derive_all_threshold_baseline_retention(
    data_dir: Path,
    source_summary_path: Path,
    output_summary_path: Path,
    output_curve_path: Path,
    *,
    positive_tolerance_pct: float = 1e-9,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read each saved city sample once and write strict derived outputs."""
    if not source_summary_path.exists():
        raise FileNotFoundError(f"threshold-resilience summary missing: {source_summary_path}")
    source = pd.read_csv(source_summary_path)
    required = {
        "region_key",
        "region_name_cn",
        "region_name_en",
        "province",
        "scenario",
        "n_samples",
        "baseline_fixed_threshold_reduction_pct",
        "prob_reduction_positive",
        "reduction_p05_pct",
        "reduction_p50_pct",
        "reduction_p95_pct",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise KeyError(f"threshold-resilience summary missing columns: {missing}")
    source = source[source["scenario"].isin(RETENTION_SCENARIOS)].copy()
    if source["region_key"].nunique() != 314:
        raise ValueError(
            f"expected 314 valid cities, found {source['region_key'].nunique()}"
        )
    counts = source.groupby("region_key")["scenario"].nunique()
    incomplete = sorted(counts[counts != len(RETENTION_SCENARIOS)].index)
    if incomplete:
        raise ValueError(f"cities missing one retention scenario: {incomplete}")
    if not (pd.to_numeric(source["n_samples"], errors="coerce") == 10000).all():
        raise ValueError("every city/scenario must contain 10,000 Monte Carlo draws")

    summary_frames = []
    curve_frames = []
    for region_key in sorted(source["region_key"].unique()):
        path = _sample_path(data_dir, region_key)
        if not path.exists():
            raise FileNotFoundError(f"saved Monte Carlo samples missing: {path}")
        samples = pd.read_csv(path, usecols=["scenario", "reduction_pct"])
        city_summary, city_curve = derive_city_baseline_retention(
            source[source["region_key"] == region_key],
            samples,
            curve_targets=RETENTION_CURVE_TARGETS,
            positive_tolerance_pct=positive_tolerance_pct,
        )
        summary_frames.append(city_summary)
        curve_frames.append(city_curve)

    summary = pd.concat(summary_frames, ignore_index=True)
    curve = pd.concat(curve_frames, ignore_index=True)
    expected_summary_rows = 314 * len(RETENTION_SCENARIOS)
    expected_curve_rows = expected_summary_rows * len(RETENTION_CURVE_TARGETS)
    if len(summary) != expected_summary_rows or len(curve) != expected_curve_rows:
        raise AssertionError(
            f"derived row-count mismatch: summary={len(summary)}, curve={len(curve)}"
        )
    eligible_rows = int(summary["retention_eligible"].sum())
    if eligible_rows != 626:
        raise AssertionError(f"expected 626 eligible city/scenario rows, found {eligible_rows}")
    eligible = summary[summary["retention_eligible"]]
    if not eligible["baseline_retention_p05"].notna().all():
        raise AssertionError("eligible retention summaries contain missing quantiles")
    source_quantiles = source[[
        "region_key", "scenario", "reduction_p05_pct", "reduction_p50_pct",
        "reduction_p95_pct",
    ]]
    checked = eligible.merge(
        source_quantiles, on=["region_key", "scenario"], validate="one_to_one"
    )
    baseline = checked["baseline_fixed_threshold_reduction_pct"].to_numpy()
    for percentile in ("p05", "p50", "p95"):
        expected = checked[f"reduction_{percentile}_pct"].to_numpy() / baseline
        if not np.allclose(
            checked[f"baseline_retention_{percentile}"],
            expected,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise AssertionError(
                f"baseline-retention {percentile} differs from source quantile scaling"
            )
    zero_target = curve[
        curve["retention_eligible"]
        & (curve["retention_target_fraction"] == 0.0)
    ]
    if not np.allclose(
        zero_target["prob_retain_target"],
        zero_target["prob_reduction_positive"],
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError(
            "zero-target retention probability differs from paired positive probability"
        )
    _atomic_csv(summary, output_summary_path)
    _atomic_csv(curve, output_curve_path)
    return summary, curve


def render_threshold_baseline_retention(
    summary_path: Path,
    curve_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Render all exploratory baseline-retention figures from derived CSVs."""
    if not summary_path.exists():
        raise FileNotFoundError(f"baseline-retention summary missing: {summary_path}")
    if not curve_path.exists():
        raise FileNotFoundError(f"baseline-retention curve missing: {curve_path}")
    summary = pd.read_csv(summary_path)
    curve = pd.read_csv(curve_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    calls = (
        ("national", plot_threshold_baseline_retention_national, (summary,)),
        ("curves", plot_threshold_baseline_retention_curves, (curve,)),
        ("typology", plot_threshold_baseline_retention_typology, (summary,)),
    )
    for name, function, inputs in calls:
        output = output_dir / FIGURE_FILENAMES[name]
        function(*inputs, save_path=str(output))
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"baseline-retention figure was not written: {output}")
        outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive and plot fixed-threshold baseline-retention robustness"
    )
    parser.add_argument("--env", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--derive-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--curve-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    figures_dir = Path(config["figures_dir"])
    retention_config = config.get("threshold_resilience", {}) or {}
    positive_tolerance = float(
        retention_config.get("positive_tolerance_pct", 1e-9)
    )
    source_summary = args.source_summary or data_dir / "all_threshold_resilience.csv"
    summary_output = args.summary_output or data_dir / SUMMARY_FILENAME
    curve_output = args.curve_output or data_dir / CURVE_FILENAME
    output_dir = (
        args.output_dir
        or figures_dir / "analysis" / "threshold_baseline_retention"
    )

    if not args.plot_only:
        summary, curve = derive_all_threshold_baseline_retention(
            data_dir,
            source_summary,
            summary_output,
            curve_output,
            positive_tolerance_pct=positive_tolerance,
        )
        print(
            f"Verified derived retention: {summary_output} ({len(summary)} rows; "
            f"{int(summary['retention_eligible'].sum())} eligible)"
        )
        print(f"Verified retention curve: {curve_output} ({len(curve)} rows)")
    if not args.derive_only:
        outputs = render_threshold_baseline_retention(
            summary_output, curve_output, output_dir
        )
        for output in outputs:
            print(f"Verified output: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
