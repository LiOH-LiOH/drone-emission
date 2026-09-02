"""Plot retained Beijing annual lifecycle totals for all delivery modes.

This candidate plot consumes only the source-data CSV produced by
``analyze --only beijing-delivery-modes`` and is not in the formal manifest.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


MODE_LABELS = {
    "pedal_bicycle": "Pedal bicycle",
    "pedelec_ebike": "Pedelec e-bike",
    "electric_moped": "Electric moped",
    "fuel_moped": "Fuel moped",
    "battery_electric_car": "Battery-electric car",
    "fuel_car": "Fuel car",
    "drone": "Drone",
}
MODE_COLORS = {
    "pedal_bicycle": "#4f8f62",
    "pedelec_ebike": "#76a96d",
    "electric_moped": "#5f9fbd",
    "fuel_moped": "#c58a52",
    "battery_electric_car": "#4678a8",
    "fuel_car": "#a95f43",
    "drone": "#7b6aa8",
}
SOURCE_MODES = (
    "drone",
    "walking",
    "pedal_bicycle",
    "pedelec_ebike",
    "fuel_moped",
    "electric_moped",
    "fuel_car",
    "battery_electric_car",
)
REQUIRED_COLUMNS = {
    "mode",
    "distance_basis",
    "annual_orders",
    "weighted_mean_distance_km",
    "weighted_mean_g_per_order",
    "annual_lifecycle_kg",
    "annual_lifecycle_kt",
}
WIDTH_MM = 180.0
HEIGHT_MM = 92.0


def _next_review_directory(project_root: Path) -> Path:
    date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    stem = (
        project_root
        / "figures"
        / "review"
        / f"beijing-all-mode-emissions-no-walking_{date}"
    )
    candidate = stem
    suffix = 2
    while candidate.exists():
        candidate = Path(f"{stem}_{suffix:02d}")
        suffix += 1
    return candidate


def load_plot_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the eight-row retained analysis table."""
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"analysis CSV is missing columns: {missing}")
    unknown = sorted(set(frame["mode"].astype(str)) - set(SOURCE_MODES))
    if unknown:
        raise ValueError(f"analysis CSV contains unknown modes: {unknown}")
    if (
        len(frame) != len(SOURCE_MODES)
        or frame["mode"].nunique() != len(SOURCE_MODES)
    ):
        raise ValueError("analysis CSV must contain each delivery mode exactly once")
    frame = frame.copy()
    frame["annual_lifecycle_kt"] = pd.to_numeric(
        frame["annual_lifecycle_kt"], errors="raise"
    )
    frame = frame[frame["mode"].isin(MODE_LABELS)].copy()
    return frame.sort_values(
        ["annual_lifecycle_kt", "mode"], kind="stable"
    ).reset_index(drop=True)


def load_analysis_report(path: str | Path) -> dict:
    """Load the retained analysis metadata used by the figure report."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_figure_report(
    frame: pd.DataFrame,
    analysis_report: dict,
    *,
    figure_path: str | Path,
) -> dict:
    """Build a seven-mode machine-readable report for the rendered figure."""
    displayed_modes = frame["mode"].astype(str).tolist()
    coefficients = analysis_report.get("coefficients", {})
    boundary = {
        key: value
        for key, value in analysis_report.get("boundary", {}).items()
        if not key.startswith("walking_")
    }
    return {
        "schema_version": 1,
        "figure_name": "beijing_all_mode_lifecycle_candidate",
        "figure_path": Path(figure_path).name,
        "formal_manifest_registered": False,
        "figure_size_mm": {"width": WIDTH_MM, "height": HEIGHT_MM},
        "functional_unit": analysis_report.get("functional_unit"),
        "displayed_modes": displayed_modes,
        "excluded_modes": [
            {"mode": "walking", "reason": "excluded from display by request"}
        ],
        "analysis": {
            "analysis_version": analysis_report.get("analysis_version"),
            "calculation_method": analysis_report.get("calculation_method"),
            "input_signature": analysis_report.get("input_signature"),
            "od_filter": analysis_report.get("od_filter"),
            "distance": analysis_report.get("distance"),
            "scenario": analysis_report.get("scenario"),
            "boundary": boundary,
            "coefficients": {
                mode: coefficients[mode]
                for mode in displayed_modes
                if mode in coefficients
            },
        },
        "plotted_values": frame.to_dict(orient="records"),
    }


def plot_beijing_all_mode_lifecycle(
    frame: pd.DataFrame,
    *,
    save_path: str | Path,
    plot_title: bool = False,
    figsize: tuple[float, float] = (WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
) -> Path:
    """Plot annual lifecycle totals for seven all-Beijing mode scenarios."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.linewidth": 0.8,
        }
    )
    save_path = Path(save_path)
    labels = [MODE_LABELS[mode] for mode in frame["mode"]]
    values = frame["annual_lifecycle_kt"].to_numpy(dtype=float)
    colors = [MODE_COLORS[mode] for mode in frame["mode"]]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    bars = ax.barh(labels, values, color=colors, height=0.68)
    max_value = float(values.max())
    label_offset = max(max_value * 0.012, 0.05)
    for bar, value in zip(bars, values):
        ax.text(
            value + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="0.2",
        )

    ax.set_xlim(0, max_value * 1.15 + label_offset)
    ax.set_xlabel("Annual lifecycle emissions (kt CO$_2$e yr$^{-1}$)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    if plot_title:
        ax.set_title(
            "All Beijing deliveries served by one transport mode",
            fontsize=9,
            loc="left",
        )
        save_path = save_path.with_name(f"{save_path.stem}_t{save_path.suffix}")
    fig.tight_layout(pad=0.8)
    save_path.parent.mkdir(parents=True, exist_ok=False)
    fig.savefig(save_path, dpi=300, facecolor="white")
    fig.savefig(save_path.with_suffix(".svg"), facecolor="white")
    fig.savefig(save_path.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--analysis-report", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-title", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _next_review_directory(project_root)
    )
    frame = load_plot_data(args.input_csv)
    figure = plot_beijing_all_mode_lifecycle(
        frame,
        save_path=output_dir / "beijing_all_mode_lifecycle.pdf",
        plot_title=args.plot_title,
    )
    report = build_figure_report(
        frame,
        load_analysis_report(args.analysis_report),
        figure_path=figure,
    )
    report_path = figure.with_name(f"{figure.stem}_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate figure saved to {figure}")
    print(f"Candidate report saved to {report_path}")


if __name__ == "__main__":
    main()
