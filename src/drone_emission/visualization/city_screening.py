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
POPULATION_COL = "population_10k"
POPULATION_THRESHOLDS = (50.0, 100.0, 500.0, 1000.0)
POPULATION_LABELS = ("<0.5", "0.5–1", "1–5", "5–10", "≥10")
POPULATION_COLORS = ("#9692B6", "#7973A3", "#5E548E", "#44336F", "#29164F")
BACKGROUND_COLORS = {
    "upper_left": "#EAF2F8",
    "upper_right": "#FBEEDD",
    "lower": "#F1F1F1",
}


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


def _validate_population(plot_df: pd.DataFrame) -> pd.Series:
    """Return positive finite population for every plotted city."""
    if POPULATION_COL not in plot_df.columns:
        raise ValueError(
            f"City reduction scatter missing columns: ['{POPULATION_COL}']"
        )
    values = pd.to_numeric(plot_df[POPULATION_COL], errors="coerce")
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        if "region_key" in plot_df.columns:
            names = plot_df.loc[invalid, "region_key"].astype(str).tolist()
        else:
            names = [str(value) for value in plot_df.index[invalid]]
        raise ValueError(
            "City reduction scatter requires positive finite population_10k for: "
            + ", ".join(names)
        )
    return values.astype(float)


def _population_classes(values) -> np.ndarray:
    """Assign threshold values to the higher population interval."""
    return np.searchsorted(POPULATION_THRESHOLDS, values, side="right")


def plot_city_reduction_potential_scatter(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.965, 3.43),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot absolute against relative city emission-reduction potential."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

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
    population = _validate_population(plot_df)
    plot_df[POPULATION_COL] = population
    population_classes = _population_classes(population)
    absolute_median = float(plot_df[ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[RELATIVE_REDUCTION_COL].median())
    color_norm = BoundaryNorm(np.arange(6) - 0.5, ncolors=5)
    color_map = ListedColormap(POPULATION_COLORS)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(
        left=0.31,
        right=0.97,
        bottom=0.31,
        top=0.91 if plot_title else 0.98,
    )
    scatter = ax.scatter(
        plot_df[ABSOLUTE_REDUCTION_COL],
        plot_df[RELATIVE_REDUCTION_COL],
        s=18,
        alpha=1.0,
        c=population_classes,
        cmap=color_map,
        norm=color_norm,
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
    # Freeze data-derived limits before adding background rectangles on log x.
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max)
    ax.axhspan(0.0, relative_median, color=BACKGROUND_COLORS["lower"],
               linewidth=0, zorder=0)
    for left, right, key in (
        (x_min, absolute_median, "upper_left"),
        (absolute_median, x_max, "upper_right"),
    ):
        ax.fill_between([left, right], relative_median, 50.0,
                        color=BACKGROUND_COLORS[key], linewidth=0, zorder=0)

    ax.set_xlabel(
        "Annual absolute reduction\n(t CO$_2$e yr$^{-1}$)",
        fontsize=7.5,
    )
    ax.set_ylabel("Relative lifecycle reduction (%)", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar_ax = fig.add_axes((0.31, 0.11, 0.66, 0.018))
    colorbar = fig.colorbar(
        scatter,
        cax=colorbar_ax,
        orientation="horizontal",
        ticks=np.arange(5),
        spacing="uniform",
        drawedges=True,
    )
    colorbar.ax.set_xticklabels(POPULATION_LABELS)
    colorbar.ax.tick_params(labelsize=5.5, length=1.5, pad=1.0)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label(
        "Population (million)",
        fontsize=6.0,
        labelpad=1.5,
    )
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
                POPULATION_COL: float(row[POPULATION_COL]),
                "population_class_index": int(_population_classes(row[POPULATION_COL])),
                "highlighted": False,
            }
        )
    axis_position = ax.get_position()
    colorbar_position = colorbar_ax.get_position()
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
                "color_field": POPULATION_COL,
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
            "color_encoding": {
                "field": POPULATION_COL,
                "colors": list(POPULATION_COLORS),
                "normalization": "discrete",
                "thresholds_10k": list(POPULATION_THRESHOLDS),
                "interval_closure": "left",
                "point_alpha": 1.0,
                "population_basis": "Existing socioeconomic population_10k; not uniform urban population",
                "official_city_size_classification": False,
            },
            "colorbar": {
                "orientation": "horizontal",
                "label": "Population (million)",
                "display_scale_10k_per_unit": 100.0,
                "tick_labels": list(POPULATION_LABELS),
                "spacing": "uniform",
                "bbox_figure_fraction": {
                    "x0": float(colorbar_position.x0),
                    "y0": float(colorbar_position.y0),
                    "width": float(colorbar_position.width),
                    "height": float(colorbar_position.height),
                },
            },
            "median_reference_lines": {
                "absolute_reduction_t_co2e": absolute_median,
                "relative_reduction_pct": relative_median,
                "legend_drawn": False,
            },
            "background_regions": {
                "colors": BACKGROUND_COLORS,
                "split_x": absolute_median,
                "split_y": relative_median,
                "lower_quadrants_share_color": True,
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
