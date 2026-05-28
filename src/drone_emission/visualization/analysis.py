"""Visualization functions for cross-region correlation and emission analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from visualization._utils import _add_title_suffix

# Import from constants module (no circular dependency)
from analysis.constants import PROVINCE_NAME_EN, VARIABLE_LABELS


def _build_reduction_cmap(v_max: float):
    """Discrete green sequential cmap with gray 'Unstudied' band below 0.

    Returns (cmap, norm, bin_edges, reduction_edges). Use with
    `BoundaryNorm`-compatible plotters; place a vertical "Unstudied" label
    next to the gray band when drawing the colorbar.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap

    step = 5 if v_max <= 25 else (10 if v_max <= 80 else 15)
    top = max(int(np.ceil(v_max / step) * step), step * 5)
    reduction_edges = np.arange(0, top + step, step, dtype=float)
    bin_edges = np.concatenate([[-step], reduction_edges])
    palette = ["#ffffe0", "#f7fcb9", "#d9f0a3", "#addd8e",
               "#78c679", "#41ab5d", "#238443", "#005a32"]
    cmap_colors = ["#cccccc"] + palette[:len(reduction_edges) - 1]
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(bin_edges, ncolors=cmap.N, clip=True)
    return cmap, norm, bin_edges, reduction_edges


def _draw_discrete_reduction_colorbar(fig, cbar_ax, cmap, norm, bin_edges,
                                      reduction_edges, label_text):
    """Draw the discrete reduction colorbar with a vertical 'Unstudied'
    label aligned with the numeric tick labels. Returns (cbar, n_bands).
    Call AFTER fig.tight_layout / final layout so positions are settled.
    """
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, cax=cbar_ax,
        boundaries=bin_edges, ticks=reduction_edges,
        spacing="uniform",
    )
    cbar.ax.set_yticklabels([f"{int(b)}" for b in reduction_edges])
    plt.setp(cbar.ax.get_yticklabels(), rotation=0)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.set_label(label_text, fontsize=11)
    return cbar, len(bin_edges) - 1


def _annotate_unstudied(fig, cbar, cbar_ax, n_bands):
    """Add a vertical 'Unstudied' label next to the gray band on a discrete
    colorbar, x-anchored to the existing numeric tick label positions."""
    fig.canvas.draw()
    cbar_pos = cbar_ax.get_position()
    gray_y_fig = cbar_pos.y0 + cbar_pos.height * (0.5 / n_bands)
    ref_label = cbar.ax.yaxis.get_majorticklabels()[0]
    ref_disp_x = ref_label.get_window_extent().x0
    ref_x_fig = fig.transFigure.inverted().transform((ref_disp_x, 0))[0]
    fig.text(
        ref_x_fig, gray_y_fig, "Unstudied",
        ha="left", va="center", fontsize=9, rotation=90,
    )


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    title: str = "Feature-Result Correlations",
):
    """Plot correlation heatmap.

    Args:
        corr_df: Correlation DataFrame (features x results).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get feature labels
    row_labels = [VARIABLE_LABELS.get(f, f) for f in corr_df.index]
    col_labels = [VARIABLE_LABELS.get(c, c) for c in corr_df.columns]

    # Create heatmap
    im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(
        "Correlation Coefficient", rotation=-90, va="bottom", fontsize=11
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    if plot_title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_feature_scatter(
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    save_path: str = None,
    figsize: tuple = (8, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot scatter plot of feature vs result with regression line and 95% CI.

    Scales to large datasets: small markers, no per-point labels,
    annotates only top/bottom 5 outliers by residual.

    Args:
        df: DataFrame with region statistics.
        feature_var: Feature variable name.
        result_var: Result variable name.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    # Filter valid data
    valid_mask = df[[feature_var, result_var]].notna().all(axis=1)
    x = df.loc[valid_mask, feature_var].values
    y = df.loc[valid_mask, result_var].values
    labels = df.loc[valid_mask, "region_name"].values

    if len(x) < 2:
        print(f"Not enough data points for {feature_var} vs {result_var}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot — smaller markers for large datasets
    marker_size = 100 if len(x) <= 20 else 30
    ax.scatter(
        x, y, s=marker_size, alpha=0.6, edgecolors="darkblue", linewidths=0.8, zorder=10
    )

    # Add regression line with 95% CI band
    if len(x) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            "r--",
            alpha=0.8,
            linewidth=2,
            label=f"r = {r_value:.3f}, p = {p_value:.1e}",
        )

        # 95% confidence interval
        n = len(x)
        x_mean = np.mean(x)
        se = np.sqrt(
            np.sum((y - (slope * x + intercept)) ** 2) / (n - 2)
        )
        t_val = stats.t.ppf(0.975, n - 2)
        ci = t_val * se * np.sqrt(
            1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
        )
        ax.fill_between(
            x_line, y_line - ci, y_line + ci, alpha=0.15, color="red", zorder=5
        )

        ax.legend(loc="best", fontsize=10)

        # Annotate top/bottom 5 outliers by residual (only when many points)
        if len(x) > 20:
            residuals = y - (slope * x + intercept)
            n_outliers = min(5, len(x) // 10)
            if n_outliers > 0:
                outlier_idx = np.argsort(np.abs(residuals))[-n_outliers:]
                for idx in outlier_idx:
                    ax.annotate(
                        labels[idx],
                        (x[idx], y[idx]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=7,
                        alpha=0.7,
                    )
    elif len(x) == 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, "r--", alpha=0.8, linewidth=2, label=f"r = {r_value:.3f}"
        )
        ax.legend(loc="best", fontsize=10)

    # Add per-point labels only for small datasets
    if len(x) <= 20:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (x[i], y[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

    # Labels
    ax.set_xlabel(VARIABLE_LABELS.get(feature_var, feature_var), fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_var, result_var), fontsize=12)

    if plot_title:
        title = f"{VARIABLE_LABELS.get(feature_var, feature_var)} vs {VARIABLE_LABELS.get(result_var, result_var)}"
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_grouped_scatter(
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    group_var: str,
    threshold: float,
    save_path: str = None,
    figsize: tuple = (8, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    group_labels: tuple = None,
    ax: "plt.Axes" = None,
):
    """Plot scatter with two groups split by threshold, each with its own regression line.

    Args:
        df: DataFrame with region statistics.
        feature_var: X-axis variable name.
        result_var: Y-axis variable name.
        group_var: Variable to split groups on.
        threshold: Split threshold for group_var.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        group_labels: Tuple of (low_label, high_label). Auto-generated if None.
        ax: External axes to draw into. When provided, skips figure creation and save.
    """
    valid_mask = df[[feature_var, result_var, group_var]].notna().all(axis=1)
    plot_df = df.loc[valid_mask]

    low_mask = plot_df[group_var] <= threshold
    high_mask = ~low_mask

    if group_labels is None:
        group_labels = (
            f"Low GEF ($\\leq$ {threshold})",
            f"High GEF (> {threshold})",
        )

    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Global x-range with padding so regression lines extend to plot edges
    x_data_min = plot_df[feature_var].min()
    x_data_max = plot_df[feature_var].max()
    x_pad = (x_data_max - x_data_min) * 0.03
    x_global_min = x_data_min - x_pad
    x_global_max = x_data_max + x_pad

    # Continuous colormap: dark green (clean) -> dark orange -> dark red (dirty)
    # Skips yellow — all colors saturated and visible on white background
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list(
        "gef", ["#006837", "#31A354", "#D95F0E", "#A50F15"]
    )
    gef_vals = plot_df[group_var].values
    norm = Normalize(vmin=gef_vals.min(), vmax=gef_vals.max())

    # Scatter all points colored by GEF
    sc = ax.scatter(
        plot_df[feature_var].values, plot_df[result_var].values,
        s=22, alpha=0.6, c=gef_vals, cmap=cmap, norm=norm,
        edgecolors="none", zorder=10,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label(VARIABLE_LABELS.get(group_var, group_var), fontsize=10)

    # Two regression lines by group (dark green / dark red to match colormap ends)
    line_colors = ["#006837", "#A50F15"]
    masks = [low_mask, high_mask]
    for mask, color, label in zip(masks, line_colors, group_labels):
        x = plot_df.loc[mask, feature_var].values
        y = plot_df.loc[mask, result_var].values
        if len(x) < 3:
            continue
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x_global_min, x_global_max, 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, color=color, linewidth=2.5, alpha=0.35, zorder=15,
            label=label,
        )
        n = len(x)
        x_mean = np.mean(x)
        se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
        t_val = stats.t.ppf(0.975, n - 2)
        ci = t_val * se * np.sqrt(
            1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
        )
        ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.10, color=color, zorder=5)

    # Annotate selected cities
    if "region_name" in plot_df.columns:
        highlight = [
            "Kunming", "Chengdu", "Panzhihua", "Yuxi",
            "Beijing", "Shanghai", "Chongqing", "Shenzhen",
        ]
        offsets = {
            "Kunming": (-12, 10), "Chengdu": (-12, -14),
            "Panzhihua": (-50, -8), "Yuxi": (-8, -14),
            "Beijing": (8, 8), "Shanghai": (8, -12),
            "Chongqing": (-14, -14), "Shenzhen": (8, 8),
        }
        for city in highlight:
            row = plot_df[plot_df["region_name"] == city]
            if row.empty:
                continue
            cx = row[feature_var].iloc[0]
            cy = row[result_var].iloc[0]
            gef = row[group_var].iloc[0]
            dot_color = cmap(norm(gef))
            ax.scatter(
                cx, cy, s=70, marker="o",
                facecolors=dot_color, edgecolors="black", linewidths=1.2,
                zorder=20, alpha=0.95,
            )
            ox, oy = offsets.get(city, (8, 8))
            ax.annotate(
                city, (cx, cy), xytext=(ox, oy),
                textcoords="offset points", fontsize=8,
                fontweight="bold", color="black",
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="0.6", alpha=0.85),
                zorder=25,
            )

    ax.set_xlim(x_global_min, x_global_max)
    ax.set_xlabel(VARIABLE_LABELS.get(feature_var, feature_var), fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_var, result_var), fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    if plot_title:
        ax.set_title(
            f"{VARIABLE_LABELS.get(feature_var, feature_var)} vs "
            f"{VARIABLE_LABELS.get(result_var, result_var)}",
            fontsize=12, fontweight="bold",
        )

    if not _external_ax:
        plt.tight_layout()

    if _external_ax:
        return
    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grouped scatter saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_and_gef_scatter(
    df_stats: "pd.DataFrame",
    df_enriched: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (18, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    gef_scenario: str = "current",
    named_csv_path: str = None,
):
    """Two-panel figure: (a) China choropleth map of reduction, (b) detour ratio vs reduction by GEF.

    Args:
        df_stats: all_region_stats DataFrame (for map coloring).
        df_enriched: all_region_stats_enriched DataFrame (has grid_emission_factor).
        stats_csv_path: Path to all_region_stats.csv (passed to map function).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        gef_scenario: "current" (default), "2030", or "2060". Future scenarios pull
            reductions from the `optimal_mixed_<year>` strategy at kappa=1.0 in the
            named-sensitivity CSV.
        named_csv_path: Path to all_sensitivity_named.csv. Required when
            gef_scenario != "current".
    """
    from visualization.regions import plot_regions_on_china_map

    valid_scenarios = ("current", "2030", "2060")
    if gef_scenario not in valid_scenarios:
        raise ValueError(f"gef_scenario must be one of {valid_scenarios}, got {gef_scenario!r}")

    if gef_scenario == "current":
        map_df = None
        map_color_by = "max_lifecycle_reduction_pct"
        scatter_df = df_enriched
        scatter_result_var = "max_lifecycle_reduction_pct"
        cbar_label = None
        y_label_override = None
        scatter_group_labels = None
    else:
        if named_csv_path is None:
            raise ValueError("named_csv_path is required when gef_scenario != 'current'")
        named = pd.read_csv(named_csv_path)
        scen_strategy = f"optimal_mixed_{gef_scenario}"
        reductions = (
            named[(named["strategy"] == scen_strategy) & (named["kappa"] == 1.0)]
            [["region_key", "reduction_vs_anchor_ebike_pct"]]
            .rename(columns={"reduction_vs_anchor_ebike_pct": "max_lifecycle_reduction_pct"})
        )
        map_df = df_stats.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            reductions, on="region_key", how="left"
        )
        scatter_df = df_enriched.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            reductions, on="region_key", how="left"
        )
        map_color_by = "max_lifecycle_reduction_pct"
        scatter_result_var = "max_lifecycle_reduction_pct"
        cbar_label = f"Max Lifecycle Reduction under {gef_scenario} GEF (%)"
        y_label_override = cbar_label
        scatter_group_labels = (
            "Low current GEF ($\\leq$ 0.3)",
            "High current GEF (> 0.3)",
        )

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.6, 1]})

    plot_regions_on_china_map(
        csv_path=stats_csv_path if map_df is None else None,
        df=map_df,
        color_by=map_color_by,
        colorbar_label=cbar_label,
        ax=ax_a,
        plot_title=plot_title,
    )
    plot_grouped_scatter(
        scatter_df,
        feature_var="detour_ratio",
        result_var=scatter_result_var,
        group_var="grid_emission_factor",
        threshold=0.3,
        ax=ax_b,
        plot_title=plot_title,
        group_labels=scatter_group_labels,
    )
    if y_label_override is not None:
        ax_b.set_ylabel(y_label_override, fontsize=12)

    ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")
    ax_b.text(-0.08, 1.02, "b", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold", va="bottom", ha="right")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map and GEF scatter saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_kappa_panels(
    oat_csv_path: str,
    stats_csv_path: str,
    kappas: tuple = (0.9, 0.8, 0.7, 0.6),
    save_path: str = None,
    figsize: tuple = (16, 12),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """2x2 panel grid: China region choropleth of drone-vs-ebike reduction across kappa.

    Reduction values come from `all_sensitivity_oat.csv` rows where
    `feature_varied == "kappa"`, using `reduction_vs_scenario_ebike_pct` —
    i.e. the optimal mix vs all-ebike at the SAME kappa. This isolates the
    drone-substitution benefit from the e-bike bundling benefit. All four
    panels share a single diverging colour scale and one figure-level
    colorbar so the maps are directly comparable.

    Args:
        oat_csv_path: Path to all_sensitivity_oat.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        kappas: Four kappa values to render (one per panel, row-major order).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show figure-level title.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    from visualization.regions import plot_regions_on_china_map

    if len(kappas) != 4:
        raise ValueError(f"plot_regions_map_kappa_panels expects 4 kappa values, got {len(kappas)}")

    oat = pd.read_csv(oat_csv_path)
    base = pd.read_csv(stats_csv_path)

    reduction_col = "reduction_vs_scenario_ebike_pct"
    kappa_dfs = {}
    drone_share_median = {}
    all_values = []
    for k in kappas:
        kappa_rows = oat[(oat["feature_varied"] == "kappa") & (np.isclose(oat["kappa"], k))]
        sub = kappa_rows[["region_key", reduction_col]].rename(
            columns={reduction_col: "max_lifecycle_reduction_pct"}
        )
        merged = base.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            sub, on="region_key", how="left"
        )
        kappa_dfs[k] = merged
        all_values.extend(merged["max_lifecycle_reduction_pct"].dropna().tolist())
        drone_share_median[k] = float(kappa_rows["optimal_drone_ratio_pct"].median())

    if not all_values:
        print("plot_regions_map_kappa_panels: no kappa OAT values matched any region in stats — "
              "rendering empty maps. Run sensitivity across all regions to populate.")
        max_abs = 1.0
    else:
        max_abs = max(abs(min(all_values)), abs(max(all_values)))
    shared_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    shared_cmap = LinearSegmentedColormap.from_list("green_diverging", [
        "#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641",
    ])

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    panel_labels = ("a", "b", "c", "d")

    for ax, k, label in zip(axes.flat, kappas, panel_labels):
        plot_regions_on_china_map(
            df=kappa_dfs[k],
            color_by="max_lifecycle_reduction_pct",
            ax=ax,
            plot_title=False,
            norm=shared_norm,
            draw_colorbar=False,
            draw_legend=False,
        )
        ax.set_title(f"$\\kappa = {k}$", fontsize=13)
        ax.text(-0.02, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right")
        ax.text(
            0.02, 0.98,
            f"Median drone share: {drone_share_median[k]:.0f}%",
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )

    fig.tight_layout(rect=(0, 0.06, 0.92, 1.0))

    sm = plt.cm.ScalarMappable(cmap=shared_cmap, norm=shared_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes((0.93, 0.20, 0.015, 0.6))
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Reduction vs same-$\kappa$ all-ebike (%)", fontsize=11)

    fig.legend(
        handles=[Patch(facecolor="#cccccc", edgecolor="none", label="Unstudied regions")],
        loc="lower center", fontsize=11, framealpha=0.8, ncol=1,
        bbox_to_anchor=(0.46, 0.02),
    )

    if plot_title:
        fig.suptitle("Lifecycle reduction across drone-share $\\kappa$", fontsize=14, fontweight="bold")

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map kappa panels saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_pathway_scenarios(
    named_csv_path: str,
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (16, 18),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """2x2 panel grid: China region choropleth + per-province drone-share boxplot.

    Each of four scenarios occupies a (map, boxplot) pair, with the boxplot
    directly under its map. The boxplot has one box per province (drone-share
    distribution across cities within that province). Provinces ordered by the
    today-baseline median drone share, consistent across all four panels.

    Panels (all from `all_sensitivity_named.csv`):
        (a) today                    kappa=1.0, gef_mult=1.0, optimal_mixed
        (b) halved GEF               kappa=1.0, gef_mult=0.5, optimal_mixed_2030
        (c) operator bundling        kappa=0.7, gef_mult=1.0, optimal_mixed
        (d) halved GEF + bundling    kappa=0.7, gef_mult=0.5, optimal_mixed_2030

    Reduction values (map fill) use `reduction_vs_anchor_ebike_pct` (vs today's
    no-bundling all-ebike anchor). Boxplot y-axis is drone_ratio_pct in 0..100.

    Args:
        named_csv_path: Path to all_sensitivity_named.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show figure-level title.
    """
    from visualization.regions import plot_regions_on_china_map

    named = pd.read_csv(named_csv_path)
    base = pd.read_csv(stats_csv_path)

    panels = [
        ("a", "Today",                    1.0, 1.0, "optimal_mixed"),
        ("b", "Halved GEF",               1.0, 0.5, "optimal_mixed_2030"),
        ("c", "Operator bundling",        0.7, 1.0, "optimal_mixed"),
        ("d", "Halved GEF + bundling",    0.7, 0.5, "optimal_mixed_2030"),
    ]

    reduction_col = "reduction_vs_anchor_ebike_pct"
    has_annual = "annual_lifecycle_kg" in named.columns
    if has_annual:
        anchor_rows = named[
            np.isclose(named["kappa"], 1.0)
            & np.isclose(named["gef_mult"], 1.0)
            & (named["strategy"] == "all_ebike")
        ][["region_key", "annual_lifecycle_kg"]].rename(
            columns={"annual_lifecycle_kg": "anchor_annual_kg"}
        )
    panel_dfs = {}
    panel_named = {}
    drone_share_median = {}
    annual_saved_total_kg = {}
    all_values = []
    for label, _title, k, g, strat in panels:
        rows = named[
            np.isclose(named["kappa"], k)
            & np.isclose(named["gef_mult"], g)
            & (named["strategy"] == strat)
        ].copy()
        rows["province_en"] = rows["province"].map(PROVINCE_NAME_EN).fillna(rows["province"])
        panel_named[label] = rows
        sub = rows[["region_key", reduction_col]].rename(
            columns={reduction_col: "max_lifecycle_reduction_pct"}
        )
        merged = base.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            sub, on="region_key", how="left"
        )
        panel_dfs[label] = merged
        all_values.extend(merged["max_lifecycle_reduction_pct"].dropna().tolist())
        drone_share_median[label] = float(rows["drone_ratio_pct"].median()) if len(rows) else float("nan")
        if has_annual:
            joined = rows[["region_key", "annual_lifecycle_kg"]].merge(anchor_rows, on="region_key", how="inner")
            annual_saved_total_kg[label] = float((joined["anchor_annual_kg"] - joined["annual_lifecycle_kg"]).sum())

    if not all_values:
        print("plot_regions_map_pathway_scenarios: no scenario rows matched any region in stats — "
              "rendering empty maps. Run sensitivity --stages named first.")
        v_max = 1.0
    else:
        v_max = max(max(all_values), 0.0)

    shared_cmap, shared_norm, bin_edges, reduction_edges = _build_reduction_cmap(v_max)

    # Province order: total annual lifecycle emission under today's all-ebike
    # baseline (kappa=1, gef=1, strategy=all_ebike). Largest emitters on the left.
    anchor_named = named[
        np.isclose(named["kappa"], 1.0)
        & np.isclose(named["gef_mult"], 1.0)
        & (named["strategy"] == "all_ebike")
    ].copy()
    anchor_named["province_en"] = anchor_named["province"].map(PROVINCE_NAME_EN).fillna(anchor_named["province"])
    province_emission_baseline_kt = (
        anchor_named.groupby("province_en")["annual_lifecycle_kg"].sum() / 1e6
    )
    province_order = province_emission_baseline_kt.sort_values(ascending=False).index.tolist()

    # Per-panel province totals (kt CO2/yr) for the overlay line.
    panel_province_emission_kt = {
        lbl: panel_named[lbl].groupby("province_en")["annual_lifecycle_kg"].sum() / 1e6
        for lbl in panel_named
    }
    line_y_max = max(
        float(s.max()) if len(s) else 0.0 for s in panel_province_emission_kt.values()
    ) if panel_province_emission_kt else 0.0
    line_y_max = max(line_y_max, 1.0) * 1.05

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2, height_ratios=[5, 2.2, 5, 2.2], hspace=0.25, wspace=0.20)
    pos = {"a": (0, 0), "b": (0, 1), "c": (2, 0), "d": (2, 1)}
    ax_pairs = []

    box_lines = []  # parallel list of twin axes for line overlay
    for label, title, k, g, _s in panels:
        r, c = pos[label]
        ax_map = fig.add_subplot(gs[r, c])
        ax_box = fig.add_subplot(gs[r + 1, c])
        ax_pairs.append((ax_map, ax_box))

        plot_regions_on_china_map(
            df=panel_dfs[label],
            color_by="max_lifecycle_reduction_pct",
            ax=ax_map,
            plot_title=False,
            norm=shared_norm,
            cmap=shared_cmap,
            draw_colorbar=False,
            draw_legend=False,
        )
        ax_map.set_xlabel("")
        ax_map.set_ylabel("")
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(False)
        ax_map.set_title(title, fontsize=12)
        ax_map.text(-0.02, 1.02, label, transform=ax_map.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="right")

        from matplotlib.patches import Rectangle
        table_rows = [
            (r"$\kappa$", f"{k:.1f}"),
            ("gef", f"{g:.1f}"),
            ("Drone share", f"{drone_share_median[label]:.0f}%"),
        ]
        if label in annual_saved_total_kg:
            saved_kt = annual_saved_total_kg[label] / 1e6  # kg -> kt
            table_rows.append((r"CO$_2$ saved", f"{saved_kt:.0f} kt/yr"))

        n_rows = len(table_rows)
        t_w, t_h = 0.34, 0.045 * n_rows
        t_x = (1.0 - t_w) / 2  # centered, sits under the title
        t_y = 0.98
        col_split = 0.55  # fraction of table width for col 1
        # Background + outer border
        ax_map.add_patch(Rectangle(
            (t_x, t_y - t_h), t_w, t_h,
            transform=ax_map.transAxes, facecolor="white", edgecolor="gray",
            linewidth=0.4, alpha=0.9, zorder=3,
        ))
        # Row separators + cell text
        for i, (lbl, val) in enumerate(table_rows):
            row_y_top = t_y - (i / n_rows) * t_h
            row_y_mid = t_y - ((i + 0.5) / n_rows) * t_h
            if i > 0:
                ax_map.plot(
                    [t_x, t_x + t_w], [row_y_top, row_y_top],
                    transform=ax_map.transAxes, color="gray",
                    linewidth=0.3, zorder=4,
                )
            ax_map.text(
                t_x + 0.01, row_y_mid, lbl,
                transform=ax_map.transAxes, fontsize=9,
                ha="left", va="center", zorder=5,
            )
            ax_map.text(
                t_x + col_split * t_w + 0.005, row_y_mid, val,
                transform=ax_map.transAxes, fontsize=9,
                ha="left", va="center", zorder=5,
            )

        rows = panel_named[label]
        data = [rows.loc[rows["province_en"] == p, "drone_ratio_pct"].values for p in province_order]
        bp = ax_box.boxplot(
            data, positions=range(len(province_order)), widths=0.65,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.2),
            boxprops=dict(facecolor="#a6d96a", edgecolor="#1a9641", linewidth=0.6),
            whiskerprops=dict(color="#1a9641", linewidth=0.6),
            capprops=dict(color="#1a9641", linewidth=0.6),
        )
        ax_box.set_ylim(0, 102)
        ax_box.set_xlim(-0.7, len(province_order) - 0.3)
        ax_box.set_xticks(range(len(province_order)))
        ax_box.set_xticklabels(province_order, rotation=90, fontsize=7)
        ax_box.set_ylabel("Drone share (%)", fontsize=9)
        ax_box.tick_params(axis="y", labelsize=8)
        ax_box.grid(axis="y", linestyle=":", alpha=0.4)
        ax_box.axhline(drone_share_median[label], color="#1a9641", linestyle="--", linewidth=0.7, alpha=0.8)

        # Overlay: total annual lifecycle emission per province (gray line, right axis).
        emission_series = panel_province_emission_kt[label]
        emission_y = [float(emission_series.get(p, 0.0)) for p in province_order]
        ax_line = ax_box.twinx()
        ax_line.plot(
            range(len(province_order)), emission_y,
            color="dimgray", linewidth=1.0, marker="o", markersize=2.5,
            alpha=0.85, zorder=5,
        )
        ax_line.set_ylim(0, line_y_max)
        ax_line.set_ylabel("Annual lifecycle (kt CO$_2$)", fontsize=9, color="dimgray")
        ax_line.tick_params(axis="y", labelsize=8, colors="dimgray")
        box_lines.append(ax_line)

    cbar_ax = fig.add_axes((0.93, 0.28, 0.015, 0.44))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig, cbar_ax, shared_cmap, shared_norm, bin_edges, reduction_edges,
        r"Reduction vs today's all-ebike anchor (%)",
    )

    if plot_title:
        fig.suptitle("Forward-pathway scenarios: per-delivery reduction across 314 cities",
                     fontsize=14, fontweight="bold")

    fig.tight_layout(rect=(0, 0.03, 0.92, 0.98))

    # Align each boxplot's width with its map (the map shrinks horizontally due to
    # its enforced geographic aspect ratio; gridspec leaves the boxplot wider).
    # Also pull the boxplot up to sit right under the map.
    fig.canvas.draw()
    vgap = 0.015
    for (ax_map, ax_box), ax_line in zip(ax_pairs, box_lines):
        map_box = ax_map.get_position()
        box_box = ax_box.get_position()
        new_y0 = map_box.y0 - vgap - box_box.height
        new_pos = [map_box.x0, new_y0, map_box.width, box_box.height]
        ax_box.set_position(new_pos)
        ax_line.set_position(new_pos)

    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map pathway scenarios saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_city_province_ranks(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (18, 14),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Three-panel figure: (a) China choropleth, (b) top-10 city ebike rank, (c) province ebike ranking.

    Top row: regions map | city_ebike_rank_top10. Bottom row spans full width with
    province_total_ebike_lifecycle.

    Args:
        df_stats: all_region_stats DataFrame.
        stats_csv_path: Path to all_region_stats.csv (for map function).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles on panels.
    """
    from visualization.regions import plot_regions_on_china_map

    map_df = pd.read_csv(stats_csv_path)
    v_max = max(float(map_df["max_lifecycle_reduction_pct"].max(skipna=True)), 0.0)
    shared_cmap, shared_norm, bin_edges, reduction_edges = _build_reduction_cmap(v_max)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[0.7, 0.35],
        width_ratios=[1.6, 1.0],
        hspace=0.25, wspace=0.30,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    plot_regions_on_china_map(
        df=map_df, color_by="max_lifecycle_reduction_pct",
        ax=ax_a, plot_title=plot_title,
        cmap=shared_cmap, norm=shared_norm,
        draw_colorbar=False, draw_legend=False,
    )
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    plot_city_ebike_rank_top10(df_stats, ax=ax_b, plot_title=plot_title)
    plot_province_total_ebike_lifecycle(df_stats, ax=ax_c, plot_title=plot_title)

    for ax_panel, label in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        ax_panel.text(-0.05, 1.02, label, transform=ax_panel.transAxes,
                      fontsize=16, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()

    # Place colorbar to the right of ax_a, aligned with its vertical extent.
    fig.canvas.draw()
    a_pos = ax_a.get_position()
    cbar_ax = fig.add_axes((a_pos.x1 + 0.005, a_pos.y0,
                            0.012, a_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig, cbar_ax, shared_cmap, shared_norm, bin_edges, reduction_edges,
        r"Reduction vs today's all-ebike anchor (%)",
    )
    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map + city/province ranks saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_top_correlations_bar(
    corr_df: pd.DataFrame,
    result_var: str,
    top_n: int = 5,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot bar chart of top correlated features for a result variable.

    Args:
        corr_df: Correlation DataFrame (features x results).
        result_var: Result variable to analyze.
        top_n: Number of top correlations to show.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if result_var not in corr_df.columns:
        print(f"Result variable '{result_var}' not in correlation DataFrame")
        return

    # Get correlations for this result, sorted by absolute value
    correlations = corr_df[result_var].dropna()
    sorted_corr = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )
    top_corr = sorted_corr.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar colors based on positive/negative
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_corr.values]

    # Get labels
    feature_labels = [VARIABLE_LABELS.get(f, f) for f in top_corr.index]

    # Bar plot
    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, i, f"{val:.3f}", va="center", ha=ha, fontsize=10)

    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel("Correlation Coefficient", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)

    if plot_title:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"Top Correlations with {result_label}", fontsize=12, fontweight="bold"
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_annual_emissions_comparison(
    df: pd.DataFrame = None,
    csv_path: str = None,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    top_n: int = None,
):
    """Plot grouped bar chart comparing annual emissions across regions.

    For large datasets, requires top_n to show only top-N and bottom-N cities
    by emission reduction percentage.

    Args:
        df: Pre-loaded emissions DataFrame. If None, loads from csv_path.
        csv_path: Path to all_region_stats.csv (used if df is None).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        top_n: Show only top-N and bottom-N cities by reduction. Required if >20 cities.
    """
    if df is None:
        from analysis.correlation import load_annual_emissions
        df = load_annual_emissions(csv_path)

    if df.empty:
        print("No emission data available")
        return

    # Guard: skip if too many cities without top_n
    if len(df) > 20 and top_n is None:
        print(
            f"Warning: {len(df)} cities too many for bar chart. "
            f"Use top_n parameter to show subset. Skipping."
        )
        return

    # Filter to top/bottom N if specified
    if top_n is not None and len(df) > top_n * 2:
        df_sorted = df.sort_values("reduction_pct", ascending=False)
        top = df_sorted.head(top_n)
        bottom = df_sorted.tail(top_n)
        df = pd.concat([top, bottom]).drop_duplicates()
        # Re-sort for display
        df = df.sort_values("reduction_pct", ascending=False)

    # Check if using actual data
    has_drone_data = (
        "all_drone_tonnes" in df.columns and df["all_drone_tonnes"].notna().any()
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))

    if has_drone_data:
        # Three bars: E-bike, Drone, Optimal Mix
        width = 0.25
        ebike_color = "#e74c3c"
        drone_color = "#3498db"
        mix_color = "#2ecc71"

        ax.bar(
            x - width,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color=ebike_color,
            alpha=0.8,
        )
        ax.bar(
            x,
            df["all_drone_tonnes"],
            width,
            label="All Drone",
            color=drone_color,
            alpha=0.8,
        )
        bars3 = ax.bar(
            x + width,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color=mix_color,
            alpha=0.8,
        )

        # Add reduction percentage labels on optimal mix bars
        for bar, pct in zip(bars3, df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"-{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ylabel = "Annual Lifecycle Emissions (tonnes CO₂e)"
    else:
        # Two bars: E-bike vs Optimal Mix (fallback)
        width = 0.35

        ax.bar(
            x - width / 2,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color="#e74c3c",
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color="#2ecc71",
            alpha=0.8,
        )

        for bar, pct in zip(bars2, df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"-{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

        ylabel = "Annual Lifecycle Emissions (tonnes CO₂e)"

    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["region_name"], fontsize=8, rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10)

    if plot_title:
        ax.set_title(
            "Annual Food Delivery Lifecycle Emissions by Scenario",
            fontsize=14,
            fontweight="bold",
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, df["all_ebike_tonnes"].max() * 1.25)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Annual emissions comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    return df


def plot_emission_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot distribution of emission reduction and optimal drone ratio across all cities.

    Two subplots: histogram+KDE of max_lifecycle_reduction_pct, and histogram of
    optimal_drone_ratio_pct. Annotated with n, mean, median, std.

    Args:
        df: DataFrame with region statistics (from load_region_stats).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Subplot 1: max_lifecycle_reduction_pct
    ax1 = axes[0]
    col1 = "max_lifecycle_reduction_pct"
    if col1 in df.columns:
        data1 = df[col1].dropna()
        ax1.hist(data1, bins=30, alpha=0.7, color="#3498db", edgecolor="white", density=True)
        # KDE
        if len(data1) > 5:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data1)
            x_kde = np.linspace(data1.min(), data1.max(), 200)
            ax1.plot(x_kde, kde(x_kde), color="darkblue", linewidth=2)
        # Vertical lines
        ax1.axvline(data1.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {data1.mean():.1f}%")
        ax1.axvline(data1.median(), color="green", linestyle="--", linewidth=1.5, label=f"Median: {data1.median():.1f}%")
        ax1.legend(fontsize=9)
        # Text annotation
        ax1.text(
            0.97, 0.95,
            f"n={len(data1)}\nstd={data1.std():.1f}%",
            transform=ax1.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
    ax1.set_xlabel(VARIABLE_LABELS.get(col1, col1), fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # Subplot 2: optimal_drone_ratio_pct
    ax2 = axes[1]
    col2 = "optimal_drone_ratio_pct"
    if col2 in df.columns:
        data2 = df[col2].dropna()
        ax2.hist(data2, bins=30, alpha=0.7, color="#2ecc71", edgecolor="white", density=True)
        if len(data2) > 5:
            from scipy.stats import gaussian_kde
            kde2 = gaussian_kde(data2)
            x_kde2 = np.linspace(data2.min(), data2.max(), 200)
            ax2.plot(x_kde2, kde2(x_kde2), color="darkgreen", linewidth=2)
        ax2.axvline(data2.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {data2.mean():.1f}%")
        ax2.axvline(data2.median(), color="blue", linestyle="--", linewidth=1.5, label=f"Median: {data2.median():.1f}%")
        ax2.legend(fontsize=9)
        ax2.text(
            0.97, 0.95,
            f"n={len(data2)}\nstd={data2.std():.1f}%",
            transform=ax2.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )
    ax2.set_xlabel(VARIABLE_LABELS.get(col2, col2), fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle="--")

    if plot_title:
        fig.suptitle("Distribution of Emission Results Across Cities", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Distribution plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_city_emission_ranking(
    df: pd.DataFrame,
    variable: str = "max_lifecycle_reduction_pct",
    save_path: str = None,
    figsize: tuple = (12, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot cities ranked by emission reduction with percentile highlights.

    Args:
        df: DataFrame with region statistics.
        variable: Variable to rank by.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if variable not in df.columns or "region_name" not in df.columns:
        print(f"Required columns not found for city ranking plot")
        return

    plot_df = df[["region_name", variable]].dropna().copy()
    if plot_df.empty:
        print("No data for city ranking plot")
        return

    # Sort descending by reduction
    plot_df = plot_df.sort_values(variable, ascending=False).reset_index(drop=True)
    plot_df["rank"] = plot_df.index + 1
    n = len(plot_df)

    fig, ax = plt.subplots(figsize=figsize)

    # Highlight indices
    highlight_indices = {
        "1st": 0,
        "25%": max(0, int(n * 0.25) - 1),
        "50%": max(0, int(n * 0.50) - 1),
    }
    highlight_colors = {"1st": "#e74c3c", "25%": "#f39c12", "50%": "#2ecc71"}
    highlight_set = set(highlight_indices.values())

    # Bar colors: highlight bars get their color, rest get default
    bar_colors = []
    for i in range(n):
        assigned = False
        for label, idx in highlight_indices.items():
            if i == idx:
                bar_colors.append(highlight_colors[label])
                assigned = True
                break
        if not assigned:
            bar_colors.append("#3498db")

    # Draw bars
    ax.bar(
        plot_df["rank"], plot_df[variable],
        color=bar_colors, width=1.0, edgecolor="none", alpha=0.85, zorder=2,
    )

    # Annotate highlighted bars
    for label, idx in highlight_indices.items():
        row = plot_df.iloc[idx]
        city = row["region_name"]
        value = row[variable]
        ax.annotate(
            f"{label}: {city} ({value:.1f}%)",
            xy=(row["rank"], value),
            xytext=(15, 12), textcoords="offset points",
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=highlight_colors[label], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=highlight_colors[label], alpha=0.9),
        )

    ax.set_xlabel("City Rank", fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(variable, variable), fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add summary stats
    ax.text(
        0.97, 0.95,
        f"n={n}\nMean: {plot_df[variable].mean():.1f}%\nMedian: {plot_df[variable].median():.1f}%",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    if plot_title:
        ax.set_title(
            f"City Ranking by {VARIABLE_LABELS.get(variable, variable)}",
            fontsize=14, fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"City ranking plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_province_boxplot(
    df: pd.DataFrame,
    variable: str,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    min_cities: int = 3,
):
    """Plot horizontal boxplot grouped by province, sorted by median.

    Provinces with fewer than min_cities are grouped as "Other".

    Args:
        df: DataFrame with region statistics.
        variable: Variable to plot (e.g., 'max_lifecycle_reduction_pct').
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        min_cities: Minimum cities required per province (rest grouped as "Other").
    """
    if "province" not in df.columns or variable not in df.columns:
        print(f"Required columns not found for province boxplot")
        return

    plot_df = df[["province", variable]].dropna().copy()
    if plot_df.empty:
        print("No data for province boxplot")
        return

    # Translate province names to English
    plot_df["province"] = plot_df["province"].map(
        lambda x: PROVINCE_NAME_EN.get(x, x)
    )

    # Group small provinces as "Other"
    province_counts = plot_df["province"].value_counts()
    small_provinces = province_counts[province_counts < min_cities].index
    plot_df.loc[plot_df["province"].isin(small_provinces), "province"] = "Other"

    # Sort by median
    medians = plot_df.groupby("province")[variable].median().sort_values()
    ordered_provinces = medians.index.tolist()

    # Prepare data for boxplot
    data = [plot_df[plot_df["province"] == p][variable].values for p in ordered_provinces]

    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=ordered_provinces,
    )

    # Color boxes by median value using RdYlGn colormap
    cmap = plt.cm.RdYlGn
    norm = plt.Normalize(medians.min(), medians.max())
    for patch, province in zip(bp["boxes"], ordered_provinces):
        color = cmap(norm(medians[province]))
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlabel(VARIABLE_LABELS.get(variable, variable), fontsize=12)
    ax.set_ylabel("Province", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    if plot_title:
        ax.set_title(
            f"{VARIABLE_LABELS.get(variable, variable)} by Province",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Province boxplot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regression_coefficients(
    results,
    result_var: str,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot horizontal bar chart of standardized regression coefficients with 95% CI.

    Args:
        results: statsmodels OLS RegressionResults.
        result_var: Name of dependent variable (for title).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    # Exclude intercept
    params = results.params.drop("const", errors="ignore")
    conf = results.conf_int().drop("const", errors="ignore")
    pvalues = results.pvalues.drop("const", errors="ignore")

    if params.empty:
        print("No coefficients to plot")
        return

    # Sort by absolute value
    order = params.abs().sort_values(ascending=True).index
    params = params[order]
    conf = conf.loc[order]
    pvalues = pvalues[order]

    labels = [VARIABLE_LABELS.get(v, v) for v in params.index]
    y_pos = np.arange(len(params))

    # Error bars (distance from coefficient to CI bounds)
    ci_lower = params.values - conf[0].values
    ci_upper = conf[1].values - params.values
    errors = np.array([ci_lower, ci_upper])

    # Color by significance
    colors = []
    for p in pvalues.values:
        if p < 0.01:
            colors.append("#2ecc71")  # Green — highly significant
        elif p < 0.05:
            colors.append("#3498db")  # Blue — significant
        elif p < 0.1:
            colors.append("#f39c12")  # Orange — marginal
        else:
            colors.append("#95a5a6")  # Gray — not significant

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y_pos, params.values, xerr=errors, color=colors, alpha=0.8,
            ecolor="black", capsize=3, height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Standardized Coefficient (β)", fontsize=12)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    # Add R² and N annotation
    ax.text(
        0.97, 0.03,
        f"N={int(results.nobs)}, R²={results.rsquared:.3f}, Adj R²={results.rsquared_adj:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    # Legend for significance colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="p < 0.01"),
        Patch(facecolor="#3498db", label="p < 0.05"),
        Patch(facecolor="#f39c12", label="p < 0.1"),
        Patch(facecolor="#95a5a6", label="n.s."),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.8)

    if plot_title:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"OLS Coefficients: {result_label}",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regression coefficient plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_hierarchical_r2_comparison(
    hier_results: list[dict],
    result_var: str,
    lasso_r2: float = None,
    save_path: str = None,
    figsize: tuple = (10, 5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot grouped bar chart comparing R² across hierarchical models.

    Shows cumulative R² (bars) with delta-R² labeled, plus optional
    Lasso R² as a horizontal reference line.

    Args:
        hier_results: List of dicts from run_hierarchical_regression().
        result_var: Dependent variable name (for labels).
        lasso_r2: Optional Lasso R² to show as reference line.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    if not hier_results:
        return

    names = [r["name"] for r in hier_results]
    r2_vals = [r["r2"] for r in hier_results]
    r2_adj_vals = [r["r2_adj"] for r in hier_results]
    delta_vals = [r["delta_r2"] for r in hier_results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    bars1 = ax.bar(x - width / 2, r2_vals, width, label="R²", color="#3498db", alpha=0.85)
    bars2 = ax.bar(x + width / 2, r2_adj_vals, width, label="Adjusted R²", color="#2ecc71", alpha=0.85)

    # Label delta-R² on each R² bar
    for bar, delta, r2 in zip(bars1, delta_vals, r2_vals):
        label_text = f"Δ={delta:.3f}" if delta > 0.001 else f"R²={r2:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            label_text, ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    # Lasso reference line
    if lasso_r2 is not None:
        ax.axhline(
            y=lasso_r2, color="#e74c3c", linestyle="--", linewidth=1.5,
            label=f"Lasso R² = {lasso_r2:.3f}",
        )

    ax.set_ylabel("R²", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, max(r2_vals + ([lasso_r2] if lasso_r2 else [])) * 1.15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    if plot_title:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"Hierarchical Regression: {result_label}",
            fontsize=12, fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Hierarchical R² comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_geographic_pattern(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Scatter plot of latitude vs max_lifecycle_reduction_pct with regression.

    Args:
        df: DataFrame with region statistics (must have min_lat/max_lat columns).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    result_col = "max_lifecycle_reduction_pct"

    # Compute center latitude from boundary columns
    if "min_lat" in df.columns and "max_lat" in df.columns:
        lat = (df["min_lat"] + df["max_lat"]) / 2
    else:
        print("No latitude columns found for geographic pattern plot")
        return

    valid = lat.notna() & df[result_col].notna()
    lat_vals = lat[valid].values
    result_vals = df.loc[valid, result_col].values

    if len(lat_vals) < 3:
        print("Not enough data for geographic pattern plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(lat_vals, result_vals, s=30, alpha=0.6, edgecolors="darkblue", linewidths=0.5)

    # Regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(lat_vals, result_vals)
    x_line = np.linspace(lat_vals.min(), lat_vals.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(
        x_line, y_line, "r--", linewidth=2, alpha=0.8,
        label=f"r = {r_value:.3f}, p = {p_value:.1e}",
    )

    # 95% CI band
    n = len(lat_vals)
    x_mean = np.mean(lat_vals)
    se = np.sqrt(np.sum((result_vals - (slope * lat_vals + intercept)) ** 2) / (n - 2))
    t_val = stats.t.ppf(0.975, n - 2)
    ci = t_val * se * np.sqrt(
        1.0 / n + (x_line - x_mean) ** 2 / np.sum((lat_vals - x_mean) ** 2)
    )
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color="red")

    ax.set_xlabel("Latitude (°N)", fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_col, result_col), fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    if plot_title:
        ax.set_title("Geographic Pattern: Latitude vs Emission Reduction", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Geographic pattern plot saved to {save_path}")
    else:
        plt.show()


def plot_province_orders_emission(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    dpi: int = 300,
):
    """Bar (orders) + scatter (e-bike emission) by province, sorted by order volume.

    Args:
        df: all_region_stats DataFrame with columns province, total_odpair_annual_sales,
            total_ebike_lifecycle.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        dpi: Output resolution.
    """
    sub = df[["province", "total_odpair_annual_sales", "total_ebike_lifecycle"]].dropna()
    prov = (
        sub.groupby("province")
        .agg(
            total_orders=("total_odpair_annual_sales", "sum"),
            total_emission=("total_ebike_lifecycle", "sum"),
        )
        .reset_index()
        .sort_values("total_orders", ascending=False)
    )

    prov["province_en"] = prov["province"].map(PROVINCE_NAME_EN).fillna(prov["province"])

    orders_m = prov["total_orders"] / 1e6
    emission_kt = prov["total_emission"] / 1e3

    x = np.arange(len(prov))
    bar_color = "#4C8CBF"
    scatter_color = "#E05C2B"

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(x, orders_m, color=bar_color, alpha=0.8, label="Annual orders")
    ax1.set_xlabel("Province", fontsize=12)
    ax1.set_ylabel("Annual orders (millions)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(prov["province_en"], rotation=45, ha="right", fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.scatter(x, emission_kt, color=scatter_color, s=50, zorder=5, label="E-bike emission")
    ax2.set_ylabel("Total e-bike lifecycle emission (t CO₂eq)", fontsize=12, color=scatter_color)
    ax2.tick_params(axis="y", labelcolor=scatter_color)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=0.8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=scatter_color, markersize=8),
    ]
    ax1.legend(handles, ["Annual orders", "E-bike lifecycle emission"], loc="upper right", fontsize=10)

    if plot_title:
        fig.suptitle("Province-level Delivery Orders and E-bike Lifecycle Emissions", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_province_orders_both_emissions(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    dpi: int = 300,
):
    """Bar (orders) + two scatters (e-bike and drone emission) by province, sorted by order volume.

    Args:
        df: all_region_stats DataFrame with columns province, total_odpair_annual_sales,
            total_ebike_lifecycle, total_drone_lifecycle.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        dpi: Output resolution.
    """
    sub = df[["province", "total_odpair_annual_sales", "total_ebike_lifecycle", "total_drone_lifecycle"]].dropna()
    prov = (
        sub.groupby("province")
        .agg(
            total_orders=("total_odpair_annual_sales", "sum"),
            total_ebike=("total_ebike_lifecycle", "sum"),
            total_drone=("total_drone_lifecycle", "sum"),
        )
        .reset_index()
        .sort_values("total_orders", ascending=False)
    )

    prov["province_en"] = prov["province"].map(PROVINCE_NAME_EN).fillna(prov["province"])

    orders_m = prov["total_orders"] / 1e6
    ebike_kt = prov["total_ebike"] / 1e3
    drone_kt = prov["total_drone"] / 1e3

    x = np.arange(len(prov))
    bar_color = "#4C8CBF"
    ebike_color = "#E05C2B"
    drone_color = "#2CA02C"

    fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(x, orders_m, color=bar_color, alpha=0.8)
    ax1.set_xlabel("Province", fontsize=12)
    ax1.set_ylabel("Annual orders (millions)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(prov["province_en"], rotation=45, ha="right", fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.scatter(x, ebike_kt, color=ebike_color, s=50, zorder=5, marker="o")
    ax2.scatter(x, drone_kt, color=drone_color, s=50, zorder=5, marker="^")
    ax2.set_ylabel("Total lifecycle emission (t CO₂eq)", fontsize=12)
    ax2.tick_params(axis="y")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=0.8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ebike_color, markersize=8),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=drone_color, markersize=8),
    ]
    ax1.legend(handles, ["Annual orders", "E-bike emission", "Drone emission"], loc="upper right", fontsize=10)

    if plot_title:
        fig.suptitle("Province-level Delivery Orders and Lifecycle Emissions", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_province_total_ebike_lifecycle(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (14, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
):
    """Province bar chart ranked by total annual e-bike emission, with reduction% boxplot overlay.

    Bars (left axis) show summed total_ebike_lifecycle per province, sorted descending.
    Boxplot (right axis) shows per-city distribution of max_lifecycle_reduction_pct
    within each province.

    Args:
        df: all_region_stats DataFrame with province, total_ebike_lifecycle,
            max_lifecycle_reduction_pct.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
    """
    needed = ["province", "total_ebike_lifecycle", "max_lifecycle_reduction_pct"]
    sub = df[needed].dropna().copy()
    if sub.empty:
        print("Province total ebike lifecycle skipped — no rows with required columns")
        return

    sub["province_en"] = sub["province"].map(PROVINCE_NAME_EN).fillna(sub["province"])

    prov = (
        sub.groupby("province_en")
        .agg(total_ebike=("total_ebike_lifecycle", "sum"))
        .reset_index()
        .sort_values("total_ebike", ascending=False)
    )

    ordered = prov["province_en"].tolist()
    ebike_t = prov["total_ebike"].values / 1e3
    reduction_data = [
        sub.loc[sub["province_en"] == p, "max_lifecycle_reduction_pct"].values
        for p in ordered
    ]

    x = np.arange(len(ordered))
    bar_color = "#B0B0B0"
    box_color = "#1F6E2D"

    _external_ax = ax is not None
    if _external_ax:
        ax1 = ax
        fig = ax1.get_figure()
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    ax1.bar(x, ebike_t, color=bar_color, alpha=0.85, zorder=2, label="Total e-bike emission")
    ax1.set_xlabel("Province", fontsize=12)
    ax1.set_ylabel("Total e-bike lifecycle emission (t CO₂eq / yr)", fontsize=12, color=bar_color)
    ax1.tick_params(axis="y", labelcolor=bar_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered, rotation=45, ha="right", fontsize=9)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    bp = ax2.boxplot(
        reduction_data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor(box_color)
        patch.set_alpha(0.35)
    for whisker in bp["whiskers"]:
        whisker.set_color(box_color)
    for cap in bp["caps"]:
        cap.set_color(box_color)
    for median in bp["medians"]:
        median.set_color(box_color)
        median.set_linewidth(1.5)
    ax2.set_ylabel("Max lifecycle reduction (%)", fontsize=12, color=box_color)
    ax2.tick_params(axis="y", labelcolor=box_color)
    ax2.set_xlim(ax1.get_xlim())

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor=box_color, alpha=0.35, edgecolor=box_color),
    ]
    ax1.legend(
        handles,
        ["Total e-bike emission", "Max reduction % (per-city distribution)"],
        loc="upper right",
        fontsize=10,
    )

    if plot_title:
        title = "Province e-bike emission ranking with reduction-% distribution"
        if _external_ax:
            ax1.set_title(title, fontsize=12, fontweight="bold")
        else:
            fig.suptitle(title, fontsize=14, fontweight="bold")

    if _external_ax:
        return

    plt.tight_layout()

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Province total ebike lifecycle saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def _plot_city_ebike_rank(
    df: pd.DataFrame,
    mode: str,
    n: int,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
):
    """Shared horizontal grouped-bar plot for top/bottom city ebike rankings."""
    needed = ["region_name_en", "total_ebike_lifecycle", "total_drone_lifecycle"]
    sub = df[needed].dropna()
    if sub.empty:
        print(f"City ebike rank ({mode}) skipped — no rows with ebike + drone emissions")
        return

    ascending = mode == "bottom"
    ranked = sub.sort_values("total_ebike_lifecycle", ascending=ascending).head(n)
    # Largest bar at top for readability
    ranked = ranked.sort_values("total_ebike_lifecycle", ascending=True).reset_index(drop=True)

    cities = ranked["region_name_en"].tolist()
    # Raw stats are kg CO₂eq; convert to tonnes for display.
    ebike_t = ranked["total_ebike_lifecycle"].values / 1e3
    drone_t = ranked["total_drone_lifecycle"].values / 1e3

    y = np.arange(len(cities))
    bar_h = 0.6
    ebike_color = "#E05C2B"
    drone_color = "#2CA02C"

    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    # E-bike (orange) is the full bar; drone (green) overlays from 0, covering the avoided portion.
    ax.barh(y, ebike_t, height=bar_h, color=ebike_color, alpha=0.85, label="E-bike (total)", zorder=2)
    ax.barh(y, drone_t, height=bar_h, color=drone_color, alpha=0.95, label="Drone (residual)", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_xlabel("Annual lifecycle emission (t CO₂eq)", fontsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10)

    xmax = ebike_t.max()
    pad = xmax * 0.01
    for i, (e, d) in enumerate(zip(ebike_t, drone_t)):
        reduction = e - d
        fmt = "{:,.2f}" if reduction < 10 else ("{:,.1f}" if reduction < 100 else "{:,.0f}")
        reduction_pct = reduction / e * 100 if e > 0 else 0
        ax.text(e + pad, i, f"{fmt.format(reduction)} (−{reduction_pct:.1f}%)",
                va="center", ha="left",
                fontsize=9, color="black", zorder=4)
    ax.set_xlim(0, xmax * 1.22)

    if plot_title:
        label = "Top" if mode == "top" else "Bottom"
        title = f"{label} {n} Cities by Annual E-bike Lifecycle Emission"
        if _external_ax:
            ax.set_title(title, fontsize=12, fontweight="bold")
        else:
            fig.suptitle(title, fontsize=14, fontweight="bold")

    if _external_ax:
        return

    plt.tight_layout()

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"City ebike rank ({mode} {n}) saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def plot_city_ebike_rank_top10(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
):
    """Horizontal grouped bars: top-10 cities by annual e-bike lifecycle emission, with drone alongside.

    Args:
        df: all_region_stats DataFrame with region_name_en, total_ebike_lifecycle, total_drone_lifecycle.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        ax: External axes. When provided, skips figure creation and save.
    """
    _plot_city_ebike_rank(df, "top", 10, save_path, figsize, for_latex, plot_title, ax=ax)


def plot_city_ebike_rank_bottom10(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
):
    """Horizontal grouped bars: bottom-10 cities by annual e-bike lifecycle emission, with drone alongside.

    Args:
        df: all_region_stats DataFrame with region_name_en, total_ebike_lifecycle, total_drone_lifecycle.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        ax: External axes. When provided, skips figure creation and save.
    """
    _plot_city_ebike_rank(df, "bottom", 10, save_path, figsize, for_latex, plot_title, ax=ax)


def plot_city_reduction_rank_top10(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Horizontal bars: top-10 cities by absolute emission reduction value (e-bike − drone).

    Each bar's length is the avoided annual lifecycle emission (t CO₂eq). Labels show
    the absolute reduction and the percentage reduction relative to e-bike baseline.

    Args:
        df: all_region_stats DataFrame with region_name_en, total_ebike_lifecycle, total_drone_lifecycle.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
    """
    needed = ["region_name_en", "total_ebike_lifecycle", "total_drone_lifecycle"]
    sub = df[needed].dropna().copy()
    if sub.empty:
        print("City reduction rank skipped — no rows with ebike + drone emissions")
        return

    # Raw stats are kg CO₂eq; convert to tonnes.
    sub["ebike_t"] = sub["total_ebike_lifecycle"] / 1e3
    sub["drone_t"] = sub["total_drone_lifecycle"] / 1e3
    sub["reduction_t"] = sub["ebike_t"] - sub["drone_t"]

    ranked = sub.sort_values("reduction_t", ascending=False).head(10)
    ranked = ranked.sort_values("reduction_t", ascending=True).reset_index(drop=True)

    cities = ranked["region_name_en"].tolist()
    reduction = ranked["reduction_t"].values
    ebike = ranked["ebike_t"].values

    y = np.arange(len(cities))
    bar_color = "#1F77B4"

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(y, reduction, height=0.65, color=bar_color, alpha=0.85, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=10)
    ax.set_xlabel("Annual emission reduction (t CO₂eq)", fontsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    xmax = reduction.max()
    pad = xmax * 0.01
    for i, (r, e) in enumerate(zip(reduction, ebike)):
        fmt = "{:,.2f}" if r < 10 else ("{:,.1f}" if r < 100 else "{:,.0f}")
        pct = r / e * 100 if e > 0 else 0
        ax.text(r + pad, i, f"{fmt.format(r)}  (−{pct:.1f}%)",
                va="center", fontsize=9, color=bar_color, fontweight="bold", zorder=3)
    ax.set_xlim(0, xmax * 1.22)

    if plot_title:
        fig.suptitle(
            "Top 10 Cities by Annual Emission Reduction",
            fontsize=14, fontweight="bold",
        )

    plt.tight_layout()

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"City reduction rank (top 10) saved to {out}")
    else:
        plt.show()

    plt.close(fig)


def _bin_reduction_and_detour(od_df: pd.DataFrame, bin_width_km: float):
    """Bin OD rows by air distance.

    Returns (centres, reduction_pct, reduction_total_t, detour_by_bin, max_km).
    ``reduction_pct`` is (e-bike − drone) / e-bike per bin in percent,
    negative when the drone exceeds e-bike for the bin's OD pairs.
    ``reduction_total_t`` is annual absolute emission reduction per bin in
    tonnes CO2e, defined as all e-bike minus all drone.
    """
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
    reduction_total_t = np.zeros(n_bins)
    detour_by_bin = []
    for k in range(n_bins):
        m = bin_idx == k
        if not m.any():
            detour_by_bin.append(np.array([]))
            continue
        ebike_tot = (ebike_lc[m] * sales[m]).sum()
        drone_tot = (drone_lc[m] * sales[m]).sum()
        reduction_total_t[k] = (ebike_tot - drone_tot) / 1e6  # g -> tonnes
        if ebike_tot > 0:
            reduction_pct[k] = (ebike_tot - drone_tot) / ebike_tot * 100
        detour_by_bin.append(detour[m])
    return centres, reduction_pct, reduction_total_t, detour_by_bin, max_km


def plot_reduction_detour_grid(
    od_dfs: dict,
    city_labels: dict = None,
    save_path: str = None,
    figsize: tuple = (12, 14),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.25,
    n_cols: int = 2,
    max_air_km: float = 5.0,
):
    """Multi-city grid of detour-ratio boxplot + emission-reduction bars per air-distance bin.

    Args:
        od_dfs: Mapping ``{city_key: od_dataframe}``. Each frame needs
                ``road_distance_m``, ``distance_km``, ``drone_lifecycle_g``,
                ``ebike_lifecycle_g``, ``annual_sales``. Insertion order
                determines panel order.
        city_labels: Optional ``{city_key: display_name}``. Defaults to key.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show suptitle.
        bin_width_km: Width of each road-distance bin in km.
        n_cols: Number of city columns in the grid.
        max_air_km: Maximum air distance shown on the x-axis.
    """
    keys = list(od_dfs.keys())
    n_cities = len(keys)
    n_rows = int(np.ceil(n_cities / n_cols))
    city_labels = city_labels or {}

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(n_rows, n_cols, hspace=0.45, wspace=0.22)

    for i, key in enumerate(keys):
        r, c = divmod(i, n_cols)
        inner = outer[r, c].subgridspec(2, 1, hspace=0.08, height_ratios=[1, 1])
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)

        centres, reduction_pct, reduction_total_t, detour_by_bin, max_km = _bin_reduction_and_detour(
            od_dfs[key], bin_width_km
        )
        x_max = min(max_air_km, max_km)
        x_ticks = np.arange(0, x_max + 1e-6, 1.0)
        x_tick_labels = [f"{int(t)}" for t in x_ticks]

        nonempty = [(c_, d_) for c_, d_ in zip(centres, detour_by_bin) if d_.size > 0]
        if nonempty:
            box_positions = [c_ for c_, _ in nonempty]
            box_data = [d_ for _, d_ in nonempty]
            ax_top.boxplot(
                box_data, positions=box_positions, widths=bin_width_km * 0.8,
                showfliers=False, patch_artist=True,
                boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f4e79", linewidth=0.5),
                medianprops=dict(color="#1f4e79", linewidth=1.0),
                whiskerprops=dict(color="#1f4e79", linewidth=0.5),
                capprops=dict(color="#1f4e79", linewidth=0.5),
            )
        ax_top.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_top.set_ylabel("Detour (road / air)", fontsize=9)
        ax_top.set_xticks(x_ticks, labels=x_tick_labels)
        ax_top.set_xlim(0, x_max)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="both", labelsize=8)
        ax_top.grid(axis="y", alpha=0.3)
        ax_top.set_title(city_labels.get(key, key), fontsize=11, fontweight="bold")

        bar_colors = [
            "#2ca02c" if (np.isfinite(v) and v >= 0) else "#d62728" for v in reduction_pct
        ]
        ax_bot.bar(
            centres, reduction_pct, width=0.1,
            color=bar_colors, edgecolor="white", linewidth=0.2, align="center",
            label="(E-bike − Drone) / E-bike",
        )
        ax_bot.axhline(0, color="gray", linewidth=0.5)
        ax_bot.set_xlabel("Air Distance (km)", fontsize=9)
        ax_bot.set_ylabel("(E-bike − Drone) / E-bike (%)", fontsize=9, color="#2ca02c")
        ax_bot.set_xlim(0, x_max)
        ax_bot.set_xticks(x_ticks, labels=x_tick_labels)
        ax_bot.tick_params(axis="x", labelsize=8)
        ax_bot.tick_params(axis="y", labelsize=8, labelcolor="#2ca02c")
        ax_bot.grid(axis="y", alpha=0.3)

        ax_abs = ax_bot.twinx()
        ax_abs.plot(
            centres, reduction_total_t,
            color="#555555", alpha=0.5, linewidth=1.2, marker="o", markersize=2.5,
            label="E-bike - drone (t CO2e)",
        )
        ax_abs.axhline(0, color="#555555", linewidth=0.5, alpha=0.35)
        ax_abs.set_ylabel("E-bike - Drone (t CO2e/yr)", fontsize=9, color="#555555")
        ax_abs.tick_params(axis="y", labelsize=8, labelcolor="#555555")

    # Hide any unused outer slots (when n_cities < n_rows * n_cols)
    for j in range(n_cities, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        ax_empty = fig.add_subplot(outer[r, c])
        ax_empty.set_axis_off()

    if plot_title:
        fig.suptitle("Detour Ratio and Relative Emission Reduction by Air Distance", fontsize=14)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reduction detour grid saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
