"""Derive and plot baseline-adjusted perturbation resistance from saved samples.

Examples:
    python -m tools.run_threshold_adjusted_resistance --env remote
    python -m tools.run_threshold_adjusted_resistance --env remote --derive-only
    python -m tools.run_threshold_adjusted_resistance --env remote --plot-only
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

from sensitivity.adjusted_resistance import (  # noqa: E402
    ADJUSTED_RESISTANCE_BOOTSTRAP_REPLICATES,
    derive_adjusted_resistance,
    summarize_city_downside,
)
from sensitivity.threshold_resilience import (  # noqa: E402
    THRESHOLD_RESILIENCE_SCENARIOS,
)
from utils import load_config  # noqa: E402
from visualization.sensitivity import (  # noqa: E402
    plot_threshold_adjusted_resistance_diagnostic,
    plot_threshold_adjusted_resistance_national,
)


DETAIL_FILENAME = "all_threshold_adjusted_resistance.csv"
SUMMARY_FILENAME = "all_threshold_adjusted_resistance_summary.csv"
AMPLIFICATION_FILENAME = "all_threshold_stress_amplification.csv"
DIAGNOSTICS_FILENAME = "threshold_adjusted_resistance_diagnostics.csv"
FIGURE_FILENAMES = {
    "diagnostic": "threshold_adjusted_resistance_diagnostic.png",
    "national": "threshold_adjusted_resistance_national.png",
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
        data_dir / "city" / region_key / "analysis"
        / "threshold_resilience" / "samples.csv.gz"
    )


def derive_all_threshold_adjusted_resistance(
    data_dir: Path,
    source_summary_path: Path,
    *,
    bootstrap_replicates: int = ADJUSTED_RESISTANCE_BOOTSTRAP_REPLICATES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all saved samples once and derive strict national outputs."""
    if not source_summary_path.exists():
        raise FileNotFoundError(
            f"threshold-resilience summary missing: {source_summary_path}"
        )
    source = pd.read_csv(source_summary_path)
    required = {
        "region_key", "region_name_cn", "region_name_en", "province",
        "scenario", "n_samples", "baseline_fixed_threshold_reduction_pct",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise KeyError(f"threshold-resilience summary missing columns: {missing}")
    source = source[source["scenario"].isin(THRESHOLD_RESILIENCE_SCENARIOS)].copy()
    if source["region_key"].nunique() != 314:
        raise ValueError(
            f"expected 314 valid cities, found {source['region_key'].nunique()}"
        )
    if source.duplicated(["region_key", "scenario"]).any():
        raise ValueError("threshold-resilience summary contains duplicate rows")
    counts = source.groupby("region_key")["scenario"].nunique()
    incomplete = sorted(counts[counts != len(THRESHOLD_RESILIENCE_SCENARIOS)].index)
    if incomplete:
        raise ValueError(f"cities missing one threshold scenario: {incomplete}")
    if not (pd.to_numeric(source["n_samples"], errors="coerce") == 10000).all():
        raise ValueError("every city/scenario must contain 10,000 Monte Carlo draws")

    rows = []
    identifier_columns = (
        "region_key", "region_name_cn", "region_name_en", "province",
        "scenario", "n_samples", "baseline_fixed_threshold_reduction_pct",
    )
    for region_key in sorted(source["region_key"].unique()):
        path = _sample_path(data_dir, region_key)
        if not path.exists():
            raise FileNotFoundError(f"saved Monte Carlo samples missing: {path}")
        samples = pd.read_csv(path, usecols=["scenario", "reduction_pct"])
        city_source = source[source["region_key"] == region_key]
        unexpected = sorted(
            set(samples["scenario"].dropna()) - set(THRESHOLD_RESILIENCE_SCENARIOS)
        )
        if unexpected:
            raise ValueError(f"{region_key} samples contain unknown scenarios: {unexpected}")
        for scenario in THRESHOLD_RESILIENCE_SCENARIOS:
            source_row = city_source[city_source["scenario"] == scenario].iloc[0]
            reduction = pd.to_numeric(
                samples.loc[samples["scenario"] == scenario, "reduction_pct"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            expected_n = int(source_row["n_samples"])
            if len(reduction) != expected_n:
                raise ValueError(
                    f"{region_key}/{scenario} expected {expected_n} samples, "
                    f"found {len(reduction)}"
                )
            downside = summarize_city_downside(
                reduction, float(source_row["baseline_fixed_threshold_reduction_pct"])
            )
            rows.append({
                **{column: source_row[column] for column in identifier_columns},
                **downside,
            })
    downside_frame = pd.DataFrame(rows)
    if len(downside_frame) != 628:
        raise AssertionError(f"expected 628 city-scenario rows, found {len(downside_frame)}")
    return derive_adjusted_resistance(
        downside_frame,
        expected_cities=314,
        bootstrap_replicates=bootstrap_replicates,
    )


def render_threshold_adjusted_resistance(
    detail_path: Path,
    amplification_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Render the diagnostic and national adjusted-resistance figures."""
    if not detail_path.exists():
        raise FileNotFoundError(f"adjusted-resistance detail missing: {detail_path}")
    if not amplification_path.exists():
        raise FileNotFoundError(
            f"stress-amplification output missing: {amplification_path}"
        )
    detail = pd.read_csv(detail_path)
    amplification = pd.read_csv(amplification_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = output_dir / FIGURE_FILENAMES["diagnostic"]
    national = output_dir / FIGURE_FILENAMES["national"]
    plot_threshold_adjusted_resistance_diagnostic(detail, save_path=str(diagnostic))
    plot_threshold_adjusted_resistance_national(
        detail, amplification, save_path=str(national)
    )
    outputs = [diagnostic, national]
    for output in outputs:
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"adjusted-resistance figure was not written: {output}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive baseline-adjusted fixed-threshold perturbation resistance"
    )
    parser.add_argument("--env", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--derive-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--bootstrap-replicates", type=int,
        default=ADJUSTED_RESISTANCE_BOOTSTRAP_REPLICATES,
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "experiment.toml", env=args.env)
    data_dir = Path(config["data_dir"])
    figures_dir = Path(config["figures_dir"])
    source_summary = args.source_summary or data_dir / "all_threshold_resilience.csv"
    detail_path = data_dir / DETAIL_FILENAME
    summary_path = data_dir / SUMMARY_FILENAME
    amplification_path = data_dir / AMPLIFICATION_FILENAME
    diagnostics_path = data_dir / DIAGNOSTICS_FILENAME
    output_dir = (
        args.output_dir
        or figures_dir / "analysis" / "threshold_adjusted_resistance"
    )

    if not args.plot_only:
        detail, summary, diagnostics, amplification = (
            derive_all_threshold_adjusted_resistance(
                data_dir,
                source_summary,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        )
        _atomic_csv(detail, detail_path)
        _atomic_csv(summary, summary_path)
        _atomic_csv(amplification, amplification_path)
        _atomic_csv(diagnostics, diagnostics_path)
        print(
            f"Verified adjusted resistance: {detail_path} ({len(detail)} rows; "
            f"{int(detail['resistance_eligible'].sum())} eligible)"
        )
        print(f"Verified class summary: {summary_path} ({len(summary)} rows)")
        print(
            f"Verified stress amplification: {amplification_path} "
            f"({len(amplification)} rows)"
        )
        print(f"Verified model diagnostics: {diagnostics_path} ({len(diagnostics)} rows)")
    if not args.derive_only:
        outputs = render_threshold_adjusted_resistance(
            detail_path, amplification_path, output_dir
        )
        for output in outputs:
            print(f"Verified output: {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
