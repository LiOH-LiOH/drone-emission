"""Visualization functions for cross-region correlation and emission analysis."""
from .palette import opacity as paper_alpha

from .prepared_bundle import evaluate_kde, fit_polynomial, linear_fit_interval, prepared_data

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


WEIGHTED_REDUCTION_COLUMNS = (
    "oracle_min_reduction_vs_ebike_pct",
    "max_lifecycle_reduction_pct",
    "lifecycle_all_drone_reduction_pct",
)
def _first_available_column(df: pd.DataFrame, candidates: tuple[str, ...], purpose: str) -> str:
    """Return the first available candidate column, preferring weighted metrics."""
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError(f"No {purpose} column found; tried {', '.join(candidates)}")


def _weighted_reduction_column(df: pd.DataFrame) -> str:
    """Return the preferred weighted reduction metric column for city stats."""
    if {"total_ebike_lifecycle", "oracle_min_lifecycle_kg"}.issubset(df.columns):
        ebike = pd.to_numeric(df["total_ebike_lifecycle"], errors="coerce")
        oracle = pd.to_numeric(df["oracle_min_lifecycle_kg"], errors="coerce")
        df["oracle_min_reduction_vs_ebike_pct"] = np.where(
            ebike > 0,
            (1.0 - oracle / ebike) * 100.0,
            np.nan,
        )
    return _first_available_column(df, WEIGHTED_REDUCTION_COLUMNS, "weighted reduction")


def _plot_grouped_scatter_split_panels_axes(
    fig,
    subplot_spec,
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    group_var: str,
    threshold: float,
    plot_title: bool = False,
    group_labels: tuple = None,
    panel_letters: tuple = ("a", "b"),
    compact: bool = False,
    horizontal_colorbar: bool = False,
    external_colorbar_ax=None,
    colorbar_boundaries: tuple = None,
    colorbar_label: str = None,
    scatter_size: float = None,
    scatter_alpha: float = 0.68,
    fit_linewidth: float = None,
    show_panel_letters: bool = True,
    axis_label_fontsize: float = None,
    tick_fontsize: float = None,
    group_label_fontsize: float = None,
    colorbar_fontsize: float = None,
    grayscale: bool = False,
):
    """Draw threshold-split scatter panels into a parent subplot spec."""
    valid_mask = df[[feature_var, result_var, group_var]].notna().all(axis=1)
    plot_df = df.loc[valid_mask].copy()
    if plot_df.empty:
        print("Grouped scatter split panels skipped - no valid rows")
        return None

    low_mask = plot_df[group_var] <= threshold
    high_mask = ~low_mask

    if group_labels is None:
        group_labels = (
            f"Low GEF ($\\leq$ {threshold})",
            f"High GEF (> {threshold})",
        )

    x_data_min = plot_df[feature_var].min()
    x_data_max = plot_df[feature_var].max()
    y_data_min = plot_df[result_var].min()
    y_data_max = plot_df[result_var].max()
    x_pad = (x_data_max - x_data_min) * 0.03 or 0.1
    y_pad = (y_data_max - y_data_min) * 0.06 or 1.0
    x_global_min = x_data_min - x_pad
    x_global_max = x_data_max + x_pad
    y_global_min = y_data_min - y_pad
    y_global_max = y_data_max + y_pad

    from matplotlib.colors import (
        BoundaryNorm,
        LinearSegmentedColormap,
        ListedColormap,
        Normalize,
    )
    from matplotlib.cm import ScalarMappable

    grayscale_colors = (
        paper_color("scale.gef.bin0"),
        paper_color("scale.gef.bin1"),
        paper_color("scale.gef.bin2"),
        paper_color("scale.gef.bin3"),
        paper_color("scale.gef.bin4"),
        paper_color("scale.gef.bin5"),
        paper_color("scale.gef.bin6"),
    )
    base_cmap = LinearSegmentedColormap.from_list(
        "gef_grayscale" if grayscale else "gef",
        grayscale_colors
        if grayscale
        else [paper_color("scale.gef.low"), paper_color("scale.gef.mid"), paper_color("scale.gef.high")],
    )
    gef_vals = plot_df[group_var].values
    if colorbar_boundaries is not None:
        boundaries = np.asarray(colorbar_boundaries, dtype=float)
        if grayscale and len(boundaries) - 1 == len(grayscale_colors):
            colors = grayscale_colors
        else:
            colors = base_cmap(np.linspace(0.0, 1.0, len(boundaries) - 1))
        cmap = ListedColormap(colors, name="gef_discrete")
        norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    else:
        boundaries = None
        cmap = base_cmap
        norm = Normalize(vmin=gef_vals.min(), vmax=gef_vals.max())

    if external_colorbar_ax is not None:
        panels_grid = subplot_spec.subgridspec(1, 2, wspace=0.20)
        cbar_ax = external_colorbar_ax
        ax_low = fig.add_subplot(panels_grid[0, 0])
        ax_high = fig.add_subplot(
            panels_grid[0, 1], sharex=ax_low, sharey=ax_low
        )
    elif horizontal_colorbar:
        root = subplot_spec.subgridspec(
            2,
            1,
            height_ratios=[0.10, 0.90],
            hspace=0.13,
        )
        cbar_grid = root[0].subgridspec(
            1,
            3,
            width_ratios=[1.0, 0.72, 1.0],
        )
        panels_grid = root[1].subgridspec(1, 2, wspace=0.20)
        cbar_ax = fig.add_subplot(cbar_grid[0, 1])
        ax_low = fig.add_subplot(panels_grid[0, 0])
        ax_high = fig.add_subplot(
            panels_grid[0, 1], sharex=ax_low, sharey=ax_low
        )
    else:
        gs = subplot_spec.subgridspec(
            1,
            3,
            width_ratios=[1, 0.055, 1],
            wspace=0.22,
        )
        ax_low = fig.add_subplot(gs[0, 0])
        cbar_ax = fig.add_subplot(gs[0, 1])
        ax_high = fig.add_subplot(gs[0, 2], sharex=ax_low, sharey=ax_low)
    axes = [ax_low, ax_high]
    masks = [low_mask, high_mask]
    line_colors = (
        [paper_color("city_gef_support.4d4d4d"), paper_color("city_gef_support.4d4d4d")]
        if grayscale
        else [paper_color("city_gef_support.006837"), paper_color("city_gef_support.a50f15")]
    )

    for ax, mask, line_color, label, panel_letter in zip(
        axes, masks, line_colors, group_labels, panel_letters
    ):
        panel_df = plot_df.loc[mask]
        ax.scatter(
            panel_df[feature_var].values,
            panel_df[result_var].values,
            s=(
                scatter_size
                if scatter_size is not None
                else (12 if compact else 24)
            ),
            alpha=scatter_alpha,
            c=panel_df[group_var].values,
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            zorder=10,
        )

        x = panel_df[feature_var].values
        y = panel_df[result_var].values
        if len(x) >= 3 and np.sum((x - np.mean(x)) ** 2) > 0:
            x_line, y_line, ci = linear_fit_interval(x, y, x_global_min, x_global_max)
            ax.plot(
                x_line,
                y_line,
                color=line_color,
                linewidth=(
                    fit_linewidth
                    if fit_linewidth is not None
                    else (1.35 if compact else 2.5)
                ),
                alpha=paper_alpha("city_gef_support.alpha_0_42"),
                zorder=15,
            )
            ax.fill_between(
                x_line,
                y_line - ci,
                y_line + ci,
                alpha=paper_alpha("city_gef_support.alpha_0_11"),
                color=line_color,
                zorder=5,
            )

        if show_panel_letters and panel_letter:
            ax.text(
                0.0 if compact else -0.10,
                1.01 if compact else 1.02,
                panel_letter,
                transform=ax.transAxes,
                fontsize=9 if compact else 14,
                fontweight="bold",
                va="bottom",
                ha="left" if compact else "right",
            )
        ax.text(
            0.04,
            0.95,
            label,
            transform=ax.transAxes,
            fontsize=(
                group_label_fontsize
                if group_label_fontsize is not None
                else (7 if compact else 9)
            ),
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.14" if compact else "round,pad=0.22",
                facecolor=paper_color("ink.background"),
                edgecolor=paper_color("ink.0_7"),
                alpha=paper_alpha("city_gef_support.alpha_0_9"),
            ),
        )
        ax.set_xlim(x_global_min, x_global_max)
        ax.set_ylim(y_global_min, y_global_max)
        ax.set_xlabel(
            VARIABLE_LABELS.get(feature_var, feature_var),
            fontsize=(
                axis_label_fontsize
                if axis_label_fontsize is not None
                else (7.5 if compact else 12)
            ),
        )
        if compact:
            ax.tick_params(
                axis="both",
                labelsize=(
                    tick_fontsize if tick_fontsize is not None else 7
                ),
                length=2.5,
                pad=1.5,
            )
        ax.grid(
            True,
            alpha=paper_alpha("city_gef_support.alpha_0_18") if compact else paper_alpha("city_gef_support.alpha_0_3"),
            linestyle="--",
            linewidth=0.45 if compact else 0.8,
        )

    ax_low.set_ylabel(
        VARIABLE_LABELS.get(result_var, result_var),
        fontsize=(
            axis_label_fontsize
            if axis_label_fontsize is not None
            else (7.5 if compact else 12)
        ),
    )
    ax_high.set_ylabel("")
    ax_high.tick_params(axis="y", left=False, labelleft=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation=(
            "horizontal"
            if horizontal_colorbar and external_colorbar_ax is None
            else "vertical"
        ),
        boundaries=boundaries,
        ticks=boundaries,
        spacing="uniform",
    )
    resolved_colorbar_label = (
        colorbar_label
        if colorbar_label is not None
        else VARIABLE_LABELS.get(group_var, group_var)
    )
    if compact:
        cbar.set_label(
            (
                "GEF (kg CO$_2$e/kWh)"
                if (
                    horizontal_colorbar
                    and external_colorbar_ax is None
                    and group_var == "grid_emission_factor"
                )
                else resolved_colorbar_label
            ),
            fontsize=(
                colorbar_fontsize
                if colorbar_fontsize is not None
                else 7
            ),
            labelpad=1.5,
        )
        if horizontal_colorbar:
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.tick_params(
            labelsize=(
                colorbar_fontsize
                if colorbar_fontsize is not None
                else 6.8
            ),
            length=2.2,
        )
        if boundaries is not None:
            tick_labels = [f"{value:.1f}" for value in boundaries]
            if horizontal_colorbar and external_colorbar_ax is None:
                cbar.ax.set_xticklabels(tick_labels)
            else:
                cbar.ax.set_yticklabels(tick_labels)
    else:
        cbar.set_label(resolved_colorbar_label, fontsize=10)

    if plot_title:
        fig.suptitle(
            f"{VARIABLE_LABELS.get(feature_var, feature_var)} vs "
            f"{VARIABLE_LABELS.get(result_var, result_var)}",
            fontsize=12,
            fontweight="bold",
        )

    return ax_low, cbar_ax, ax_high


@prepared_data
def _prepare_city_gef_component_data(
    df: pd.DataFrame,
    grayscale: bool = False,
) -> dict:
    """Prepare shared GEF-bin lifecycle-component values."""
    gef_col = next(
        (
            candidate
            for candidate in ("grid_emission_factor", "gef")
            if candidate in df.columns and df[candidate].notna().any()
        ),
        None,
    )
    if gef_col is None:
        raise ValueError(
            "City GEF lifecycle components require "
            "grid_emission_factor or gef"
        )

    vehicle_specs = [
        {
            "label": "E-bike",
            "color": paper_color("mode.pedelec_ebike"),
            "offset": -0.18,
            "use_col": "total_ebike_use_phase_emission_kg",
            "production_col": "total_ebike_production_emission_kg",
            "eol_col": "total_ebike_end_of_life_emission_kg",
        },
        {
            "label": "Drone",
            "color": paper_color("mode.drone"),
            "offset": 0.18,
            "use_col": "total_drone_use_phase_emission_kg",
            "production_col": "total_drone_production_emission_kg",
            "eol_col": "total_drone_end_of_life_emission_kg",
        },
    ]
    required_cols = ["total_odpair_annual_sales"]
    for spec in vehicle_specs:
        required_cols.extend(
            [spec["use_col"], spec["production_col"], spec["eol_col"]]
        )
    missing = sorted(set(required_cols) - set(df.columns))
    if missing:
        raise ValueError(
            "City GEF lifecycle components require columns: "
            + ", ".join(missing)
        )

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"].eq("city")].copy()
    input_city_count = int(len(plot_df))
    for col in [gef_col, *required_cols]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    complete_mask = plot_df[[gef_col, *required_cols]].notna().all(axis=1)
    valid_mask = (
        complete_mask
        & plot_df[gef_col].gt(0)
        & plot_df[gef_col].le(0.9)
        & plot_df["total_odpair_annual_sales"].gt(0)
    )
    for spec in vehicle_specs:
        valid_mask = (
            valid_mask
            & plot_df[spec["use_col"]].gt(0)
            & plot_df[spec["production_col"]].ge(0)
            & plot_df[spec["eol_col"]].ge(0)
        )
    plot_df = plot_df.loc[valid_mask].copy()
    if plot_df.empty:
        raise ValueError(
            "City GEF lifecycle components have no valid city rows"
        )

    bin_width = 0.1
    bin_edges = np.linspace(0.0, 0.9, 10)
    labels = [f"{right:.1f}" for right in bin_edges[1:]]
    x_all = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    plot_df["gef_range"] = pd.cut(
        plot_df[gef_col],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    bins = []
    bars = []
    for index, label in enumerate(labels):
        group = plot_df[plot_df["gef_range"].eq(label)]
        if group.empty:
            continue
        orders_sum = float(group["total_odpair_annual_sales"].sum())
        bins.append(
            {
                "index": index,
                "label_right_edge": label,
                "bin_left": float(bin_edges[index]),
                "bin_right": float(bin_edges[index + 1]),
                "bin_center": float(x_all[index]),
                "city_count": int(len(group)),
                "orders_sum": orders_sum,
            }
        )
        for spec in vehicle_specs:
            production_eol = float(
                (
                    group[spec["production_col"]].sum()
                    + group[spec["eol_col"]].sum()
                )
                / orders_sum
                * 1000.0
            )
            use_phase = float(
                group[spec["use_col"]].sum() / orders_sum * 1000.0
            )
            bars.append(
                {
                    "vehicle": spec["label"],
                    "color": spec["color"],
                    "bin_index": index,
                    "bin_label_right_edge": label,
                    "x_center": float(
                        x_all[index] + spec["offset"] * bin_width
                    ),
                    "production_eol": production_eol,
                    "use_phase": use_phase,
                    "total": production_eol + use_phase,
                    "orders_sum": orders_sum,
                    "city_count": int(len(group)),
                }
            )

    if not bins:
        raise ValueError(
            "City GEF lifecycle components have no populated GEF bins"
        )

    density_x = plot_df[gef_col].to_numpy(dtype=float)
    if len(density_x) > 5 and np.unique(density_x).size > 1:
        x_density = np.linspace(bin_edges[0], bin_edges[-1], 240)
        density = evaluate_kde(density_x, x_density)
    else:
        x_density = x_all
        density = np.array(
            [
                len(plot_df[plot_df["gef_range"].eq(label)])
                for label in labels
            ],
            dtype=float,
        )
    if density.size and density.max() > 0:
        density = density / density.max()

    return {
        "gef_col": gef_col,
        "input_city_count": input_city_count,
        "valid_city_count": int(len(plot_df)),
        "excluded_city_count": input_city_count - int(len(plot_df)),
        "plot_df": plot_df,
        "vehicle_specs": vehicle_specs,
        "required_columns": required_cols,
        "bin_width": bin_width,
        "bin_edges": bin_edges,
        "labels": labels,
        "x_all": x_all,
        "bins": bins,
        "bars": bars,
        "density_x": density_x,
        "density_curve_x": x_density,
        "density_curve_y": density,
    }


def _plot_city_gef_scatter_components_axes(
    fig,
    subplot_spec,
    df: pd.DataFrame,
    plot_title: bool = False,
    compact: bool = False,
    show_legend: bool = True,
    grayscale: bool = False,
):
    """Draw GEF-binned per-order emission components into a parent subplot spec."""
    try:
        component_data = _prepare_city_gef_component_data(
            df,
            grayscale=grayscale,
        )
    except ValueError as error:
        print(f"City GEF stacked components skipped - {error}")
        return None

    gef_col = component_data["gef_col"]
    vehicle_specs = component_data["vehicle_specs"]
    bin_width = component_data["bin_width"]
    bin_edges = component_data["bin_edges"]
    bars = component_data["bars"]

    gs = subplot_spec.subgridspec(
        2,
        1,
        height_ratios=[3.6, 1.0] if compact else [4.0, 1.0],
        hspace=0.0,
    )
    ax = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1], sharex=ax)
    tick_labels = [f"{edge:.1f}" for edge in bin_edges]
    width = bin_width * 0.28
    max_total = 0.0
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=paper_color("city_gef_support.777777"),
            alpha=paper_alpha("city_gef_support.alpha_0_35"),
            edgecolor=paper_color("city_gef_support.555555"),
            label="Production + EOL",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=paper_color("city_gef_support.777777"),
            alpha=paper_alpha("city_gef_support.alpha_0_9"),
            edgecolor=paper_color("city_gef_support.555555"),
            label="Use phase",
        ),
    ]

    for spec in vehicle_specs:
        vehicle_bars = [
            bar for bar in bars if bar["vehicle"] == spec["label"]
        ]
        positions = np.array(
            [bar["x_center"] for bar in vehicle_bars], dtype=float
        )
        fixed_values = np.array(
            [bar["production_eol"] for bar in vehicle_bars], dtype=float
        )
        use_values = np.array(
            [bar["use_phase"] for bar in vehicle_bars], dtype=float
        )
        totals = np.array(
            [bar["total"] for bar in vehicle_bars], dtype=float
        )
        if totals.size:
            max_total = max(max_total, float(totals.max()))

        ax.bar(
            positions,
            fixed_values,
            width=width,
            color=spec["color"],
            alpha=paper_alpha("city_gef_support.alpha_1_0") if grayscale else paper_alpha("city_gef_support.alpha_0_38"),
            edgecolor=paper_color("city_gef_support.222222") if grayscale else "white",
            linewidth=0.45 if grayscale else 0.6,
            hatch="////" if grayscale else None,
            zorder=3,
        )
        ax.bar(
            positions,
            use_values,
            width=width,
            bottom=fixed_values,
            color=spec["color"],
            alpha=paper_alpha("city_gef_support.alpha_1_0") if grayscale else paper_alpha("city_gef_support.alpha_0_88"),
            edgecolor=paper_color("city_gef_support.222222") if grayscale else "white",
            linewidth=0.45 if grayscale else 0.6,
            zorder=3,
            label=spec["label"],
        )
        for xpos, total in zip(positions, totals):
            if total <= 0:
                continue
            ax.text(
                xpos,
                total,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=paper_color("ink.0_25"),
                rotation=90,
                zorder=5,
            )

    gef_label = VARIABLE_LABELS.get(gef_col, "Grid Emission Factor (kg CO2e/kWh)")
    ax.set_ylabel(
        "Lifecycle emission\n(g CO2e/order)",
        fontsize=7.5 if compact else 12,
    )
    ax.set_xticks(bin_edges)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0, max_total * 1.22)
    ax.grid(
        True,
        axis="y",
        alpha=paper_alpha("city_gef_support.alpha_0_18") if compact else paper_alpha("city_gef_support.alpha_0_3"),
        linestyle="--",
        linewidth=0.45 if compact else 0.8,
    )
    if compact:
        ax.tick_params(axis="y", labelsize=7, length=2.5, pad=1.5)
    ax.set_axisbelow(True)

    vehicle_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=spec["color"],
            alpha=paper_alpha("city_gef_support.alpha_1_0") if grayscale else paper_alpha("city_gef_support.alpha_0_88"),
            edgecolor=paper_color("city_gef_support.222222") if grayscale else "white",
            label=spec["label"],
        )
        for spec in vehicle_specs
    ]
    if grayscale:
        legend_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=paper_color("city_gef_support.bdbdbd"),
                edgecolor=paper_color("city_gef_support.222222"),
                linewidth=0.45,
                hatch="////",
                label="Production + EOL",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=paper_color("city_gef_support.bdbdbd"),
                edgecolor=paper_color("city_gef_support.222222"),
                linewidth=0.45,
                label="Use phase",
            ),
        ]
    if show_legend:
        legend1 = ax.legend(
            handles=vehicle_handles,
            loc="upper left",
            fontsize=7 if compact else 9,
            title="Vehicle",
            title_fontsize=7 if compact else 9,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=legend_handles,
            loc="center left",
            fontsize=7 if compact else 9,
            title="Component",
            title_fontsize=7 if compact else 9,
        )

    density_x = component_data["density_x"]
    x_violin = component_data["density_curve_x"]
    density = component_data["density_curve_y"]

    ax_dist.fill_between(
        x_violin,
        0,
        density,
        color=paper_color("city_gef_support.7a7a7a"),
        alpha=paper_alpha("city_gef_support.alpha_0_35"),
        linewidth=0,
        zorder=1,
    )
    ax_dist.plot(x_violin, density, color=paper_color("city_gef_support.555555"), linewidth=0.8, zorder=2)
    ax_dist.scatter(
        density_x,
        np.zeros_like(density_x),
        marker="|",
        s=16,
        color=paper_color("city_gef_support.333333"),
        alpha=paper_alpha("city_gef_support.alpha_0_35"),
        linewidths=0.6,
        zorder=3,
    )
    ax_dist.axhline(0, color=paper_color("city_gef_support.555555"), linewidth=0.8, zorder=2)
    ax_dist.set_ylim(-0.12, 1.15)
    ax_dist.set_yticks([])
    ax_dist.set_ylabel("Cities", fontsize=7 if compact else 10)
    ax_dist.set_xlabel(gef_label, fontsize=7.5 if compact else 12)
    ax_dist.set_xticks(bin_edges)
    ax_dist.set_xticklabels(tick_labels)
    ax_dist.tick_params(axis="x", which="both", bottom=True, labelbottom=True, pad=2)
    if compact:
        ax_dist.tick_params(axis="x", labelsize=7, length=2.5, pad=1.5)
    ax_dist.set_xlim(0.05, 0.85)
    ax_dist.spines["left"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(False)

    if plot_title:
        ax.set_title(
            "City Per-order Emission Components by Grid Emission Factor",
            fontsize=9 if compact else 14,
            fontweight="bold",
        )

    return ax, ax_dist


def _bin_reduction_and_detour(
    od_df: pd.DataFrame,
    bin_width_km: float,
    absolute_reduction_metric: str = "total_t",
):
    """Bin OD rows by air distance.

    Returns (centres, reduction_pct, reduction_abs, detour_by_bin, max_km,
    oracle_drone_share_pct).
    ``reduction_pct`` is (e-bike − drone) / e-bike per bin in percent,
    negative when the drone exceeds e-bike for the bin's OD pairs.
    ``reduction_abs`` is annual absolute emission reduction per bin in tonnes
    CO2e when ``absolute_reduction_metric`` is ``total_t``. When it is
    ``per_order_g``, it is (all e-bike − all drone) divided by total orders
    in the distance range, in g CO2e/order.
    """
    if absolute_reduction_metric not in {"total_t", "per_order_g"}:
        raise ValueError(
            "absolute_reduction_metric must be 'total_t' or 'per_order_g'"
        )

    valid = od_df[(od_df["road_distance_m"] >= 0) & (od_df["distance_km"] > 0)].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    air_km = valid["distance_km"].values
    detour = road_km / air_km
    drone_lc = valid["drone_lifecycle_g"].values
    ebike_lc = valid["ebike_lifecycle_g"].values
    sales = valid["annual_sales"].values

    max_km = float(np.ceil(air_km.max() / bin_width_km) * bin_width_km)
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centres = (edges[:-1] + edges[1:]) / 2

    bin_idx = np.clip(np.digitize(air_km, edges) - 1, 0, n_bins - 1)
    reduction_pct = np.full(n_bins, np.nan)
    reduction_abs = np.zeros(n_bins)
    oracle_drone_share_pct = np.full(n_bins, np.nan)
    detour_by_bin = []
    for k in range(n_bins):
        m = bin_idx == k
        if not m.any():
            detour_by_bin.append(np.array([]))
            continue
        ebike_tot = (ebike_lc[m] * sales[m]).sum()
        drone_tot = (drone_lc[m] * sales[m]).sum()
        total_orders = sales[m].sum()
        if total_orders > 0:
            oracle_drone_share_pct[k] = (
                sales[m][drone_lc[m] < ebike_lc[m]].sum() / total_orders * 100.0
            )
        if absolute_reduction_metric == "per_order_g":
            reduction_abs[k] = (
                (ebike_tot - drone_tot) / total_orders
                if total_orders > 0
                else np.nan
            )
        else:
            reduction_abs[k] = (ebike_tot - drone_tot) / 1e6  # g -> tonnes
        if ebike_tot > 0:
            reduction_pct[k] = (ebike_tot - drone_tot) / ebike_tot * 100
        detour_by_bin.append(detour[m])
    return centres, reduction_pct, reduction_abs, detour_by_bin, max_km, oracle_drone_share_pct


def _bin_reduction_and_detour_chunks(
    chunks,
    bin_width_km: float,
    max_air_distance_km: float,
    absolute_reduction_metric: str = "total_t",
):
    """Aggregate the reduction-detour bins from bounded OD chunks."""
    if absolute_reduction_metric not in {"total_t", "per_order_g"}:
        raise ValueError(
            "absolute_reduction_metric must be 'total_t' or 'per_order_g'"
        )
    if not np.isfinite(max_air_distance_km) or max_air_distance_km <= 0:
        raise ValueError("max_air_distance_km must be finite and positive")

    max_km = float(
        np.ceil(max_air_distance_km / bin_width_km) * bin_width_km
    )
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centres = (edges[:-1] + edges[1:]) / 2
    ebike_totals = np.zeros(n_bins, dtype=np.float64)
    drone_totals = np.zeros(n_bins, dtype=np.float64)
    order_totals = np.zeros(n_bins, dtype=np.float64)
    oracle_drone_orders = np.zeros(n_bins, dtype=np.float64)
    detour_chunks = [[] for _ in range(n_bins)]

    for od_df in chunks:
        valid = od_df[
            (od_df["road_distance_m"] >= 0) & (od_df["distance_km"] > 0)
        ]
        if valid.empty:
            continue
        road_km = valid["road_distance_m"].to_numpy(dtype=float) / 1000.0
        air_km = valid["distance_km"].to_numpy(dtype=float)
        detour = road_km / air_km
        drone_lc = valid["drone_lifecycle_g"].to_numpy(dtype=float)
        ebike_lc = valid["ebike_lifecycle_g"].to_numpy(dtype=float)
        sales = valid["annual_sales"].to_numpy(dtype=float)
        bin_idx = np.clip(np.digitize(air_km, edges) - 1, 0, n_bins - 1)

        for index in np.unique(bin_idx):
            mask = bin_idx == index
            index = int(index)
            bin_sales = sales[mask]
            bin_ebike = ebike_lc[mask]
            bin_drone = drone_lc[mask]
            ebike_totals[index] += np.sum(bin_ebike * bin_sales)
            drone_totals[index] += np.sum(bin_drone * bin_sales)
            order_totals[index] += np.sum(bin_sales)
            oracle_drone_orders[index] += np.sum(
                bin_sales[bin_drone < bin_ebike]
            )
            detour_chunks[index].append(detour[mask])

    reduction_pct = np.full(n_bins, np.nan)
    reduction_abs = np.zeros(n_bins)
    oracle_drone_share_pct = np.full(n_bins, np.nan)
    populated = np.array([bool(parts) for parts in detour_chunks])
    positive_ebike = populated & (ebike_totals > 0)
    positive_orders = populated & (order_totals > 0)
    difference = ebike_totals - drone_totals
    reduction_pct[positive_ebike] = (
        difference[positive_ebike] / ebike_totals[positive_ebike] * 100.0
    )
    oracle_drone_share_pct[positive_orders] = (
        oracle_drone_orders[positive_orders]
        / order_totals[positive_orders]
        * 100.0
    )
    if absolute_reduction_metric == "per_order_g":
        reduction_abs[populated & ~positive_orders] = np.nan
        reduction_abs[positive_orders] = (
            difference[positive_orders] / order_totals[positive_orders]
        )
    else:
        reduction_abs = difference / 1e6
    detour_by_bin = [
        np.concatenate(parts) if parts else np.array([])
        for parts in detour_chunks
    ]
    return (
        centres,
        reduction_pct,
        reduction_abs,
        detour_by_bin,
        max_km,
        oracle_drone_share_pct,
    )
