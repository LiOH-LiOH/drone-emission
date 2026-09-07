"""Visualization functions for cross-region correlation and emission analysis."""
from .palette import opacity as paper_alpha

from .prepared_bundle import prepared_plot, prepared_data, evaluate_kde, fit_polynomial

from .palette import color as paper_color

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
    PROVINCE_NAME_EN,
    VARIABLE_LABELS,
)
from drone_emission.analysis.city_attribution import R2_ABSOLUTE_OUTCOME, R2_OUTCOME


def _r2_save_figure(
    fig,
    save_path: str,
    for_latex: bool,
    plot_title: bool,
    figure_name: str,
    report: dict,
):
    """Save an R2 figure and its structured sidecar report."""
    out = save_path
    if out:
        if for_latex and not out.endswith(".pdf"):
            out = out.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        svg_path = out.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, dpi=300, bbox_inches="tight")
        report_payload = {
            "schema_version": 2,
            "figure_name": figure_name,
            "figure_path": out,
            "editable_vector_path": svg_path,
            "shared_analysis_report": (
                os.path.join(os.path.dirname(out), "r2_city_attribution_analysis_report.json")
            ),
            **report,
        }
        write_figure_report(out.rsplit(".", 1)[0] + "_report.json", report_payload)
        print(f"{figure_name} saved to {out}")
    else:
        plt.show()
    plt.close(fig)


@prepared_plot()
def plot_r2_population_reduction_contrast(
    df: pd.DataFrame,
    analysis: dict,
    save_dir: str = None,
    figsize: tuple = (3.53, 3.92),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot separate population correlations for absolute and relative reduction.

    Args:
        df: Prepared city-attribution DataFrame.
        analysis: Shared R2 analysis result dictionary.
        save_dir: Directory for both figures. Shows plots if None.
        figsize: Figure size tuple for each standalone figure.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    specs = (
        (
            R2_ABSOLUTE_OUTCOME,
            "Annual optimal reduction (t CO$_2$e yr$^{-1}$)",
            analysis["population_contrast"]["absolute_reduction"],
            "r2_population_absolute_reduction",
            "Population and annual absolute reduction",
        ),
        (
            R2_OUTCOME,
            "Optimal lifecycle reduction (%)",
            analysis["population_contrast"]["relative_reduction"],
            "r2_population_relative_reduction",
            "Population and relative lifecycle reduction",
        ),
    )
    for outcome, y_label, record, figure_name, title in specs:
        fig, ax = plt.subplots(figsize=figsize)
        plot_df = df[["region_key", "population_10k", outcome]].dropna().copy()
        population_million = plot_df["population_10k"] / 100.0
        ax.scatter(population_million, plot_df[outcome], s=20, color=paper_color("city_attribution_figures.4c78a8"),
                   alpha=paper_alpha("city_attribution_figures.alpha_0_52"), edgecolor=paper_color("ink.background"), linewidth=0.3)
        slope, intercept = fit_polynomial(population_million, plot_df[outcome], 1)
        x_line = np.linspace(population_million.min(), population_million.max(), 120)
        ax.plot(x_line, slope * x_line + intercept, color=paper_color("city_attribution_figures.d95f02"), linewidth=1.5)
        label_x = population_million.min() + 0.65 * (
            population_million.max() - population_million.min()
        )
        label_y = slope * label_x + intercept
        intercept_sign = "+" if intercept >= 0 else "-"
        label_below_line = outcome == R2_ABSOLUTE_OUTCOME
        ax.annotate(
            f"$y = {slope:.2f}x {intercept_sign} {abs(intercept):.2f}$\n"
            f"$r = {record['coefficient']:.2f}$",
            xy=(label_x, label_y),
            xytext=(0, -6 if label_below_line else 6),
            textcoords="offset points",
            ha="center",
            va="top" if label_below_line else "bottom",
            fontsize=7.5,
            color=paper_color("city_attribution_figures.444444"),
            linespacing=1.15,
            zorder=5,
        )
        ax.set_xlabel("Population (million)")
        ax.set_ylabel(y_label)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        if plot_title:
            ax.set_title(title)
        fig.tight_layout()
        report = {
            "outcome": outcome,
            "correlation": record,
            "linear_fit": {"slope": float(slope), "intercept": float(intercept)},
        }
        save_path = (
            os.path.join(save_dir, f"{figure_name}.png")
            if save_dir
            else None
        )
        _r2_save_figure(
            fig,
            save_path,
            for_latex,
            plot_title,
            figure_name,
            report,
        )
