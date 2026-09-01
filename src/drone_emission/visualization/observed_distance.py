"""Formal Beijing distance renderer used only by the manifest registry."""

import json
from pathlib import Path

import pandas as pd


def _render_beijing_order_distance_distribution(
    data_dir,
    out,
    for_latex,
    plot_title,
):
    """Render the accepted gravity-model fit against observed Beijing orders."""
    from drone_emission.visualization.observed_distance_plot import (
        plot_beijing_order_distance_distribution,
    )

    calibration_dir = Path(__file__).resolve().parent.parent / "calibration"
    observed_path = calibration_dir / "observed_delivery_distance_histogram.csv"
    summary_path = calibration_dir / "gravity_calibration_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_hash = str(summary["run_hash"])
    model_path = (
        calibration_dir
        / f"beijing_gravity_distance_distribution_{run_hash}.csv"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            "Missing accepted Beijing gravity distance source: "
            f"{model_path}. Restore the source from calibration run {run_hash}."
        )

    print(
        "beijing_order_distance_distribution: using locked observed histogram "
        f"{observed_path} and accepted calibration source {model_path}"
    )
    plot_beijing_order_distance_distribution(
        pd.read_csv(observed_path),
        pd.read_csv(model_path),
        save_path=str(Path(out) / "beijing_order_distance_distribution.pdf"),
        for_latex=for_latex,
        plot_title=plot_title,
        calibration_run_hash=run_hash,
    )
