"""City-level climate-mitigation screening visualization."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drone_emission.visualization._utils import (
    _add_title_suffix,
    write_figure_report,
)

ABSOLUTE_REDUCTION_COL = "annual_absolute_reduction_t_co2e"
RELATIVE_REDUCTION_COL = "oracle_min_reduction_vs_ebike_pct"


def _reduction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "total_ebike_lifecycle",
        "oracle_min_lifecycle_kg",
        RELATIVE_REDUCTION_COL,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"City reduction scatter missing columns: {missing}")

    frame = df.copy()
    if "region_level" in frame:
        frame = frame[frame["region_level"].eq("city")].copy()
    ebike_kg = pd.to_numeric(frame["total_ebike_lifecycle"], errors="coerce")
    mixed_kg = pd.to_numeric(frame["oracle_min_lifecycle_kg"], errors="coerce")
    frame[ABSOLUTE_REDUCTION_COL] = (ebike_kg - mixed_kg) / 1_000.0
    frame[RELATIVE_REDUCTION_COL] = pd.to_numeric(
        frame[RELATIVE_REDUCTION_COL], errors="coerce"
    )
    valid = (
        np.isfinite(frame[ABSOLUTE_REDUCTION_COL])
        & (frame[ABSOLUTE_REDUCTION_COL] > 0)
        & np.isfinite(frame[RELATIVE_REDUCTION_COL])
    )
    result = frame.loc[valid].copy()
    if result.empty:
        raise ValueError("No finite positive city reduction results to plot")
    return result


def plot_city_reduction_potential_scatter(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.965, 3.43),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot absolute against relative city emission-reduction potential."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    plot_df = _reduction_metrics(df)
    absolute_median = float(plot_df[ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[RELATIVE_REDUCTION_COL].median())

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0.31,
        right=0.97,
        bottom=0.16,
        top=0.91 if plot_title else 0.98,
    )
    ax.scatter(
        plot_df[ABSOLUTE_REDUCTION_COL],
        plot_df[RELATIVE_REDUCTION_COL],
        s=18,
        alpha=0.68,
        color="#737373",
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    ax.axhline(
        relative_median,
        color="#666666",
        linestyle="--",
        linewidth=0.75,
        zorder=2,
    )
    ax.axvline(
        absolute_median,
        color="#666666",
        linestyle="--",
        linewidth=0.75,
        zorder=2,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 50.0)

    ax.set_xlabel(
        "Annual absolute reduction\n(t CO$_2$e yr$^{-1}$)",
        fontsize=7.5,
    )
    ax.set_ylabel("Relative lifecycle reduction (%)", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    if plot_title:
        ax.set_title("City mitigation priorities", fontsize=9, fontweight="bold")

    if not save_path:
        fig.show()
        return None
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches=fig.bbox_inches,
        pad_inches=0.0,
    )
    name_col = "region_name_en" if "region_name_en" in plot_df else "region_key"
    point_records = []
    for _, row in plot_df.iterrows():
        city_name = str(row[name_col])
        point_records.append(
            {
                "region_key": str(row.get("region_key", "")),
                "region_name_en": city_name,
                ABSOLUTE_REDUCTION_COL: float(row[ABSOLUTE_REDUCTION_COL]),
                RELATIVE_REDUCTION_COL: float(row[RELATIVE_REDUCTION_COL]),
                "highlighted": False,
            }
        )
    axis_position = ax.get_position()
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    write_figure_report(
        report_path,
        {
            "schema_version": 1,
            "figure_name": "city_reduction_potential_scatter",
            "figure_path": save_path,
            "figure_size_inches": {
                "width": float(figsize[0]),
                "height": float(figsize[1]),
            },
            "data": {
                "input_row_count": int(len(df)),
                "plotted_city_count": int(len(plot_df)),
                "absolute_reduction_field": ABSOLUTE_REDUCTION_COL,
                "relative_reduction_field": RELATIVE_REDUCTION_COL,
                "points": point_records,
            },
            "axes": {
                "x_scale": "log",
                "x_label": "Annual absolute reduction (t CO2e yr-1)",
                "x_limits": [float(value) for value in ax.get_xlim()],
                "y_scale": "linear",
                "y_label": "Relative lifecycle reduction (%)",
                "y_limits": [float(value) for value in ax.get_ylim()],
                "bbox_figure_fraction": {
                    "x0": float(axis_position.x0),
                    "y0": float(axis_position.y0),
                    "width": float(axis_position.width),
                    "height": float(axis_position.height),
                },
            },
            "median_reference_lines": {
                "absolute_reduction_t_co2e": absolute_median,
                "relative_reduction_pct": relative_median,
                "legend_drawn": False,
            },
            "highlighted_cities": [],
            "city_name_labels_drawn": False,
            "highlight_rings_drawn": False,
            "label_leaders_drawn": False,
        },
    )
    plt.close(fig)
    print(f"City reduction-potential scatter saved to {save_path}")
    return save_path
