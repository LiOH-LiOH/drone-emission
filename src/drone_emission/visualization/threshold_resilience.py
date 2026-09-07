"""Formal national threshold-resilience renderers."""

from __future__ import annotations

from .prepared_bundle import prepared_plot, prepared_data, evaluate_kde, fit_polynomial

from .palette import color as paper_color, colormap as paper_cmap

import json
import os
import sys
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    TwoSlopeNorm,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

from drone_emission.sensitivity.threshold_resilience import (
    THRESHOLD_RESILIENCE_MEGACITIES,
    THRESHOLD_RESILIENCE_QUADRANT_IDS,
    THRESHOLD_RESILIENCE_QUADRANT_LABELS,
    THRESHOLD_RESILIENCE_SCENARIOS,
)

def validate_threshold_resilience_summary(
    summary: pd.DataFrame,
    expected_cities: int | None = None,
) -> pd.DataFrame:
    """Validate and pivot the two-scenario city resilience summary."""
    required = {
        "region_key",
        "region_name_cn",
        "region_name_en",
        "scenario",
        "n_samples",
        "prob_reduction_positive",
        "baseline_fixed_threshold_reduction_pct",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise KeyError(f"threshold-resilience summary missing columns: {missing}")
    frame = summary[summary["scenario"].isin(THRESHOLD_RESILIENCE_SCENARIOS)].copy()
    duplicates = frame.duplicated(["region_key", "scenario"], keep=False)
    if duplicates.any():
        keys = sorted(frame.loc[duplicates, "region_key"].unique())
        raise ValueError(f"duplicate city/scenario rows: {keys}")
    counts = frame.groupby("region_key")["scenario"].nunique()
    incomplete = sorted(counts[counts != len(THRESHOLD_RESILIENCE_SCENARIOS)].index)
    if incomplete:
        raise ValueError(f"cities missing one resilience scenario: {incomplete}")
    n_cities = frame["region_key"].nunique()
    if expected_cities is not None and n_cities != expected_cities:
        raise ValueError(f"expected {expected_cities} cities, found {n_cities}")
    if not (pd.to_numeric(frame["n_samples"], errors="coerce") == 10000).all():
        raise ValueError("every city/scenario must contain 10,000 Monte Carlo draws")
    probabilities = pd.to_numeric(
        frame["prob_reduction_positive"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("resilience probabilities must be finite and within [0, 1]")
    frame["prob_reduction_positive"] = probabilities

    identifiers = frame[
        ["region_key", "region_name_cn", "region_name_en"]
    ].drop_duplicates("region_key")
    probability = frame.pivot(
        index="region_key", columns="scenario", values="prob_reduction_positive"
    ).reset_index()
    baseline = frame[frame["scenario"] == "independent"][[
        "region_key", "baseline_fixed_threshold_reduction_pct"
    ]]
    wide = identifiers.merge(probability, on="region_key", validate="one_to_one")
    return wide.merge(baseline, on="region_key", validate="one_to_one")


def _style_resilience_map_axis(ax) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


@prepared_plot()
def plot_threshold_resilience_national_subfigures(
    summary: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (88 / 25.4, 75 / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
    expected_cities: int | None = None,
):
    """Write the two national resilience maps and a shared colorbar.

    ``save_path`` is a base path. Its stem receives ``_independent``,
    ``_stress``, ``_colorbar`` and ``_report`` suffixes. The two map PDFs are
    fixed at 88 x 75 mm so that an external two-column assembly is about
    180 mm wide with a 4 mm gutter.

    Args:
        summary: Consolidated two-scenario city resilience summary.
        save_path: Base path for the outputs. Shows plots if None.
        figsize: Map figure size in inches (kept for the standard plot API).
        for_latex: If True, write PDF outputs; otherwise write PNG maps.
        plot_title: Accepted for the standard plot API; paper subfigures stay
            untitled by design.
    """
    del plot_title
    from drone_emission.visualization.regions import _batch_city_to_adcode, plot_regions_on_china_map

    wide = validate_threshold_resilience_summary(
        summary, expected_cities=expected_cities
    )
    unresolved = sorted(
        set(wide["region_name_cn"])
        - set(_batch_city_to_adcode(wide["region_name_cn"].tolist()))
    )
    if unresolved:
        raise ValueError(f"valid resilience cities unresolved for mapping: {unresolved}")

    cmap = paper_cmap("threshold_resilience.probability")
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    scenarios = (
        ("independent", "Independent perturbations"),
        ("correlated_adverse_stress", "Correlated adverse stress"),
    )

    if save_path is None:
        output_stem = None
        extension = ".pdf" if for_latex else ".png"
    else:
        output_stem = os.path.splitext(save_path)[0]
        extension = ".pdf" if for_latex else ".png"

    output_paths = {}
    for scenario, _label in scenarios:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        map_df = wide[["region_key", "region_name_cn", scenario]].copy()
        plot_regions_on_china_map(
            df=map_df,
            color_by=scenario,
            ax=ax,
            plot_title=False,
            cmap=cmap,
            norm=norm,
            draw_colorbar=False,
            draw_legend=False,
        )
        _style_resilience_map_axis(ax)
        if output_stem is None:
            plt.show()
        else:
            suffix = "independent" if scenario == "independent" else "stress"
            path = f"{output_stem}_{suffix}{extension}"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=300, facecolor=paper_color("ink.background"))
            output_paths[scenario] = path
        plt.close(fig)

    colorbar_figsize = (90 / 25.4, 12 / 25.4)
    colorbar_fig = plt.figure(figsize=colorbar_figsize)
    colorbar_ax = colorbar_fig.add_axes((0.07, 0.53, 0.86, 0.18))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = colorbar_fig.colorbar(sm, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
    colorbar.ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.0%}")
    )
    colorbar.set_label(
        "Probability of positive reduction",
        fontsize=6.5,
        labelpad=1.5,
    )
    colorbar.ax.tick_params(labelsize=6.0, length=2, pad=1)
    colorbar.outline.set_linewidth(0.4)
    if output_stem is None:
        plt.show()
    else:
        colorbar_path = f"{output_stem}_colorbar{extension}"
        os.makedirs(os.path.dirname(colorbar_path), exist_ok=True)
        colorbar_fig.savefig(colorbar_path, dpi=300, facecolor=paper_color("ink.background"))
        output_paths["colorbar"] = colorbar_path
    plt.close(colorbar_fig)

    if output_stem is None:
        return None

    def _summary_stats(values: pd.Series) -> dict[str, float]:
        quantiles = values.quantile([0.25, 0.5, 0.75])
        return {
            "min": float(values.min()),
            "q1": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.5]),
            "mean": float(values.mean()),
            "q3": float(quantiles.loc[0.75]),
            "max": float(values.max()),
        }

    report_path = f"{output_stem}_report.json"
    report = {
        "schema_version": 1,
        "figure_name": "threshold_resilience_national_overview",
        "figure_paths": output_paths,
        "layout": {
            "map_size_mm": {"width": 88.0, "height": 75.0},
            "combined_width_mm": 180.0,
            "gutter_mm": 4.0,
            "colorbar_size_mm": {"width": 90.0, "height": 12.0},
        },
        "sample": {
            "city_count": int(wide["region_key"].nunique()),
            "configured_city_count": 328,
            "fallback_gef_note": (
                "Tibet, Hong Kong and Macao use the national fallback GEF."
            ),
        },
        "global_encodings": {
            "colormap": "viridis",
            "probability_range": [0.0, 1.0],
            "colorbar_label": "Probability of positive reduction",
            "colorbar_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        },
        "scenarios": {
            scenario: {
                "label": label,
                "probability_summary": _summary_stats(wide[scenario]),
            }
            for scenario, label in scenarios
        },
    }
    temporary = f".{os.path.basename(report_path)}.tmp"
    temporary_path = os.path.join(os.path.dirname(report_path), temporary)
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, report_path)
    output_paths["report"] = report_path
    print(
        "Threshold resilience national subfigures saved to "
        f"{output_stem}_*.{{pdf,png}}"
    )
    print(f"Threshold resilience national subfigure report saved to {report_path}")
    return output_paths


@prepared_plot()
def plot_threshold_resilience_national_kdes(
    summary: pd.DataFrame,
    save_dir: str,
    expected_cities: int | None = None,
):
    """Save independent and stress city-probability KDE subfigures."""
    wide = validate_threshold_resilience_summary(
        summary, expected_cities=expected_cities
    )
    os.makedirs(save_dir, exist_ok=True)
    outputs = {}
    for scenario, color, suffix in (
        ("independent", paper_color("threshold_resilience.3b75af"), "independent"),
        ("correlated_adverse_stress", paper_color("threshold_resilience.c44e52"), "stress"),
    ):
        values = wide[scenario].to_numpy(float)
        x = np.linspace(0, 1, 500)
        density = evaluate_kde(values, x)
        fig, ax = plt.subplots(figsize=(88 / 25.4, 55 / 25.4))
        ax.plot(x, density, color=color, linewidth=1.4)
        ax.fill_between(x, density, color=color, alpha=0.18)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xlabel("Probability of positive reduction")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.subplots_adjust(left=0.18, right=0.92, bottom=0.22, top=0.96)
        path = os.path.join(
            save_dir,
            f"threshold_resilience_national_overview_{suffix}_kde.pdf",
        )
        fig.savefig(path, dpi=300, facecolor=paper_color("ink.background"))
        plt.close(fig)
        outputs[scenario] = {"path": path, "bandwidth_factor": float(kde.factor)}
    return outputs
