"""Visualization functions for cross-region correlation and emission analysis."""

import json
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

from drone_emission.visualization._utils import (
    _add_title_suffix,
    saved_tight_axis_report,
    write_figure_report,
)
from drone_emission.visualization.macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    macro_region_for_province,
)

# Import from constants module (no circular dependency)
from drone_emission.analysis.constants import (
    DRONE_SPATIAL_SCATTER_LABELED_CITIES,
    PROVINCE_NAME_EN,
    VARIABLE_LABELS,
)
from drone_emission.analysis.city_attribution import R2_ABSOLUTE_OUTCOME, R2_OUTCOME


PILOT_ABSOLUTE_REDUCTION_COL = "oracle_absolute_reduction_t_co2e"
PILOT_RELATIVE_REDUCTION_COL = "oracle_relative_reduction_pct"
def plot_city_reduction_concentration_scatter(
    df: pd.DataFrame,
    scatter_save_path: str = None,
    distribution_save_path: str = None,
    scatter_figsize: tuple = (130.5 / 25.4, 100 / 25.4),
    distribution_figsize: tuple = (43.5 / 25.4, 100 / 25.4),
    for_latex: bool = False,
):
    """Plot city reduction potential and MPCS distribution separately.

    Args:
        df: Successful city concentration results with Oracle reduction metrics.
        scatter_save_path: Path for the reduction-potential scatter figure.
        distribution_save_path: Path for the MPCS distribution figure.
        scatter_figsize: Scatter figure size tuple.
        distribution_figsize: Distribution figure size tuple.
        for_latex: If True, save as PDF.
    """
    required = [
        "pilot_concentration_score_pct",
        "concentration_class",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"City reduction scatter missing columns: {missing}")
    source_df = df.copy()
    for column in (
        "pilot_concentration_score_pct",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ):
        source_df[column] = pd.to_numeric(source_df[column], errors="coerce")
    distribution_df = source_df[
        np.isfinite(source_df["pilot_concentration_score_pct"])
        & source_df["concentration_class"].isin(("Low", "Medium", "High"))
    ].copy()
    plot_df = distribution_df[
        np.isfinite(distribution_df[PILOT_ABSOLUTE_REDUCTION_COL])
        & (distribution_df[PILOT_ABSOLUTE_REDUCTION_COL] > 0)
        & np.isfinite(distribution_df[PILOT_RELATIVE_REDUCTION_COL])
    ].copy()
    if distribution_df.empty:
        raise ValueError("No finite city concentration results to plot")
    if plot_df.empty:
        raise ValueError("No finite positive city reduction results to plot")
    absolute_median = float(plot_df[PILOT_ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[PILOT_RELATIVE_REDUCTION_COL].median())
    class_colors = {"Low": "#D95F0E", "Medium": "#7570B3", "High": "#1B9E77"}

    fig, ax = plt.subplots(figsize=scatter_figsize)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)
    for concentration_class in ("Low", "Medium", "High"):
        group = plot_df[plot_df["concentration_class"].eq(concentration_class)]
        if group.empty:
            continue
        ax.scatter(
            group[PILOT_ABSOLUTE_REDUCTION_COL],
            group[PILOT_RELATIVE_REDUCTION_COL],
            s=30,
            alpha=0.72,
            color=class_colors.get(concentration_class, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=f"{concentration_class} MPCS",
            zorder=3,
        )
    ax.axhline(relative_median, color="#666666", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(absolute_median, color="#666666", linestyle="--", linewidth=1, zorder=1)
    ax.set_xscale("log")

    name_col = "region_name_en" if "region_name_en" in plot_df else "region_key"
    labeled_city_order = tuple(
        city_name for city_name, _ in DRONE_SPATIAL_SCATTER_LABELED_CITIES
    )
    labeled = plot_df[plot_df[name_col].isin(labeled_city_order)].copy()
    labeled["_label_order"] = pd.Categorical(
        labeled[name_col], categories=labeled_city_order, ordered=True
    )
    labeled = labeled.sort_values("_label_order")
    ax.scatter(
        labeled[PILOT_ABSOLUTE_REDUCTION_COL],
        labeled[PILOT_RELATIVE_REDUCTION_COL],
        s=30,
        facecolors="none",
        edgecolors="#555555",
        linewidth=0.8,
        zorder=5,
    )
    label_texts = [
        ax.text(
            row[PILOT_ABSOLUTE_REDUCTION_COL],
            row[PILOT_RELATIVE_REDUCTION_COL],
            row[name_col],
            fontsize=7.5,
            color="#202020",
            zorder=6,
        )
        for _, row in labeled.iterrows()
    ]
    if label_texts:
        from adjustText import adjust_text

        adjust_text(
            label_texts,
            x=labeled[PILOT_ABSOLUTE_REDUCTION_COL].to_numpy(dtype=float),
            y=labeled[PILOT_RELATIVE_REDUCTION_COL].to_numpy(dtype=float),
            ax=ax,
            expand=(1.12, 1.22),
            force_text=(0.15, 0.25),
            force_static=(0.08, 0.15),
            ensure_inside_axes=True,
            arrowprops=None,
        )

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1),
    )
    labels.append("Median values")
    ax.set_xlabel(
        "Annual absolute emission reduction (t CO$_2$e yr$^{-1}$)",
        fontsize=9,
    )
    ax.set_ylabel("Relative lifecycle emission reduction (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig_distribution, ax_distribution = plt.subplots(figsize=distribution_figsize)
    fig_distribution.subplots_adjust(left=0.32, right=0.96, bottom=0.16, top=0.98)
    concentration_values = distribution_df[
        "pilot_concentration_score_pct"
    ].to_numpy(dtype=float)
    if concentration_values.size < 2 or np.unique(concentration_values).size < 2:
        raise ValueError(
            "At least two distinct city concentration values are required for KDE"
        )
    lower_tertile, upper_tertile = np.quantile(
        concentration_values,
        (1 / 3, 2 / 3),
    )
    concentration_kde = stats.gaussian_kde(
        concentration_values,
        bw_method="scott",
    )
    concentration_support = np.linspace(0.0, 100.0, 512)
    concentration_density = concentration_kde(concentration_support)
    concentration_regions = (
        (0.0, lower_tertile, class_colors["Low"]),
        (lower_tertile, upper_tertile, class_colors["Medium"]),
        (upper_tertile, 100.0, class_colors["High"]),
    )
    for lower_bound, upper_bound, color in concentration_regions:
        region_support = concentration_support[
            (concentration_support >= lower_bound)
            & (concentration_support <= upper_bound)
        ]
        region_support = np.unique(
            np.concatenate(([lower_bound], region_support, [upper_bound]))
        )
        ax_distribution.fill_betweenx(
            region_support,
            0.0,
            concentration_kde(region_support),
            color=color,
            alpha=0.65,
            linewidth=0,
            zorder=1,
        )
    for quantile in (lower_tertile, upper_tertile):
        ax_distribution.axhline(
            quantile,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
    ax_distribution.plot(
        concentration_density,
        concentration_support,
        color="#3F3F3F",
        linewidth=1.2,
        zorder=3,
    )
    ax_distribution.set_ylim(0, 100)
    ax_distribution.set_xlim(left=0)
    ax_distribution.set_yticks([0, 25, 50, 75, 100])
    ax_distribution.set_xlabel("Probability density", fontsize=9)
    ax_distribution.set_ylabel(
        "Multi-scale Pilot Capture Score (MPCS, %)",
        fontsize=9,
    )
    ax_distribution.tick_params(axis="both", labelsize=8)
    ax_distribution.spines[["top", "right"]].set_visible(False)

    def _save_panel(panel_fig, save_path, description):
        if not save_path:
            panel_fig.show()
            return None
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        panel_fig.savefig(
            save_path,
            dpi=300,
            bbox_inches=panel_fig.bbox_inches,
            pad_inches=0.0,
        )
        print(f"{description} saved to {save_path}")
        return save_path

    scatter_path = _save_panel(
        fig,
        scatter_save_path,
        "City reduction-potential scatter",
    )
    distribution_path = _save_panel(
        fig_distribution,
        distribution_save_path,
        "City concentration distribution",
    )
    plt.close(fig)
    plt.close(fig_distribution)
    return scatter_path, distribution_path
