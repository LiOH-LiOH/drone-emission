"""Entry point for rendering slide-ready Results figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.slides
    ../../.venv/bin/python -m visualization.slides --png
    ../../.venv/bin/python -m visualization.slides --only slide_city_reduction_map,slide_reduction_distribution

Outputs are written to ``figures/slides/``.
"""

import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
from scipy import stats

from analysis.constants import VARIABLE_LABELS
from visualization._utils import _add_title_suffix
from visualization.analysis import (
    _annotate_unstudied,
    _build_absolute_reduction_log_cmap,
    _build_blue_reduction_cmap,
    _build_reduction_cmap,
    _draw_discrete_reduction_colorbar,
    _draw_trip_phase_barplot,
    _finer_percentage_bins,
    _oracle_reduction_tonnes_column,
    _override_pathway_curve_endpoint,
    _pathway_curve_metric_limits,
    _plot_city_gef_scatter_components_axes,
    _plot_city_oracle_reduction_rank,
    _plot_distribution_inset,
    _plot_pathway_curve_panel,
    _plot_reduction_absolute_scatter_panel,
    _plot_reduction_detour_panel,
    _prepare_pathway_scenario_data,
    _selected_trip_distance_sample,
    _selected_trip_phase_parts,
    _weighted_reduction_column,
    plot_grouped_scatter_split_panels,
)
from visualization.concepts import _draw_detour_ratio_concept
from visualization.emission import (
    _build_break_even_marching_cubes_surface,
    _draw_median_trip_emission_process,
    _plot_contribution_waterfall,
    _plot_top_city_contribution_bars,
    _select_median_trip_emission_sample,
    _select_sparse_break_even_overlay_curves,
)
from visualization.paper import (
    CITY_CONTEXT_FIGURES,
    _attach_enrichment_columns,
    _build_reduction_detour_binned_by_city,
    _plot_power_generation_mix_pie,
    _prepare_break_even_phase_diagram,
    _prepare_contribution_decomposition_city_stats,
    _prepare_contribution_decomposition_full,
    _prepare_reduction_detour_city_stats,
    _render_city_context,
)
from visualization.regions import plot_regions_on_china_map


SLIDE_CONTEXTS = {
    "slide_city_context_dali": CITY_CONTEXT_FIGURES["dali_city_context"],
    "slide_city_context_chengdu": CITY_CONTEXT_FIGURES["chengdu_city_context"],
    "slide_city_context_tianjin": CITY_CONTEXT_FIGURES["tianjin_city_context"],
    "slide_city_context_guangzhou": CITY_CONTEXT_FIGURES["guangzhou_city_context"],
}

SLIDE_CITY_CONTEXT_PAIRS = {
    "slide_low_gef_city_contexts": (
        CITY_CONTEXT_FIGURES["dali_city_context"],
        CITY_CONTEXT_FIGURES["chengdu_city_context"],
    ),
    "slide_high_gef_city_contexts": (
        CITY_CONTEXT_FIGURES["tianjin_city_context"],
        CITY_CONTEXT_FIGURES["guangzhou_city_context"],
    ),
}

PAPER_FIGURE_REUSE = {
    "slide_gef_component_bins": "city_gef_scatter_components",
    "slide_city_context_dali": "dali_city_context",
    "slide_city_context_chengdu": "chengdu_city_context",
    "slide_city_context_tianjin": "tianjin_city_context",
    "slide_city_context_guangzhou": "guangzhou_city_context",
}

PAPER_CACHE_BACKED_SLIDES = {
    "slide_break_even_surface",
    "slide_break_even_low_gef",
    "slide_break_even_high_gef",
    "slide_contribution_waterfall",
    "slide_top_city_contributions",
}


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def _city_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return city-level rows when region_level is present."""
    if "region_level" in df.columns:
        return df[df["region_level"] == "city"].copy()
    return df.copy()


def _save_figure(fig, save_path: str, for_latex: bool, plot_title: bool, message: str):
    """Save a figure using the project's title and PDF conventions."""
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{message} saved to {save_path}")


def _figure_path(directory: str, figure_name: str, for_latex: bool, plot_title: bool) -> str:
    """Return the resolved figure path using paper/slide naming conventions."""
    ext = ".pdf" if for_latex else ".png"
    path = os.path.join(directory, f"{figure_name}{ext}")
    return _add_title_suffix(path) if plot_title else path


def _copy_paper_figure_if_available(
    figures_dir: str,
    paper_figure_name: str,
    slide_figure_name: str,
    for_latex: bool,
    plot_title: bool,
) -> bool:
    """Copy an already-rendered paper figure when it is slide-equivalent."""
    paper_dir = os.path.join(figures_dir, "paper")
    slides_dir = os.path.join(figures_dir, "slides")
    src = _figure_path(paper_dir, paper_figure_name, for_latex, plot_title)
    dst = _figure_path(slides_dir, slide_figure_name, for_latex, plot_title)
    if not os.path.exists(src):
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{slide_figure_name}: reused paper figure {src} -> {dst}")
    return True


def _placeholder_axis(ax, text: str):
    """Show a muted placeholder when required data are unavailable."""
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#777777",
    )
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_city_reduction_map(
    df_stats: pd.DataFrame,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render absolute and percentage city-level reduction maps."""
    city_df = _city_rows(df_stats)
    plot_df = city_df.copy()
    try:
        reduction_col = _oracle_reduction_tonnes_column(plot_df)
        reduction_pct_col = _weighted_reduction_column(plot_df)
    except KeyError as exc:
        fig, ax = plt.subplots(figsize=(8.5, 7.0))
        _placeholder_axis(ax, f"City reduction map unavailable\n{exc}")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide city reduction map")
        return

    valid = pd.to_numeric(plot_df[reduction_col], errors="coerce").dropna()
    valid_pct = pd.to_numeric(plot_df[reduction_pct_col], errors="coerce").dropna()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.0),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.28},
    )
    if valid.empty or valid_pct.empty:
        _placeholder_axis(ax_a, "City reduction map unavailable\nno valid reduction values")
        ax_b.axis("off")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide city reduction map")
        return

    cmap, norm, bin_edges, reduction_edges = _build_absolute_reduction_log_cmap(valid)
    plot_regions_on_china_map(
        df=plot_df,
        color_by=reduction_col,
        ax=ax_a,
        plot_title=False,
        cmap=cmap,
        norm=norm,
        draw_colorbar=False,
        draw_legend=False,
    )
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.grid(False)
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    inset_ax = ax_a.inset_axes([0.03, 0.04, 0.34, 0.38], zorder=10)
    _plot_city_oracle_reduction_rank(
        plot_df,
        10,
        plot_title=False,
        ax=inset_ax,
        invert_vertical=True,
        compact=True,
    )
    inset_ax.set_facecolor("none")
    inset_ax.patch.set_alpha(0.0)

    v_max = max(float(valid_pct.max()), 0.0)
    pct_cmap, pct_norm, pct_bin_edges, pct_reduction_edges = _build_blue_reduction_cmap(v_max)
    plot_regions_on_china_map(
        df=plot_df,
        color_by=reduction_pct_col,
        ax=ax_b,
        plot_title=False,
        cmap=pct_cmap,
        norm=pct_norm,
        draw_colorbar=False,
        draw_legend=False,
    )
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    ax_b.grid(False)
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    distribution_ax = ax_b.inset_axes([0.03, 0.04, 0.42, 0.38], zorder=10)
    _plot_distribution_inset(
        plot_df,
        reduction_pct_col,
        ax=distribution_ax,
        bins=_finer_percentage_bins(pct_reduction_edges),
        xlabel="Reduction (%)",
    )

    ax_a.text(-0.03, 1.02, "a", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_b.text(-0.03, 1.02, "b", transform=ax_b.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    fig.canvas.draw()
    ax_a_pos = ax_a.get_position()
    cbar_a_ax = fig.add_axes((ax_a_pos.x1 + 0.006, ax_a_pos.y0, 0.012, ax_a_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_a_ax,
        cmap,
        norm,
        bin_edges,
        reduction_edges,
        "Annual reduction vs all-e-bike baseline (t CO2e/yr)",
    )
    _annotate_unstudied(fig, cbar, cbar_a_ax, n_bands)

    ax_b_pos = ax_b.get_position()
    cbar_b_ax = fig.add_axes((ax_b_pos.x1 + 0.006, ax_b_pos.y0, 0.012, ax_b_pos.height))
    cbar_b, n_bands_b = _draw_discrete_reduction_colorbar(
        fig,
        cbar_b_ax,
        pct_cmap,
        pct_norm,
        pct_bin_edges,
        pct_reduction_edges,
        "Lifecycle reduction vs all-e-bike baseline (%)",
    )
    _annotate_unstudied(fig, cbar_b, cbar_b_ax, n_bands_b)

    if plot_title:
        fig.suptitle(
            "Annual lifecycle emission reduction under the optimal allocation",
            fontsize=14,
            fontweight="bold",
        )

    _save_figure(fig, save_path, for_latex, plot_title, "Slide city reduction map")


def _render_reduction_distribution(
    df_stats: pd.DataFrame,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the city-level percentage reduction distribution."""
    city_df = _city_rows(df_stats).copy()
    try:
        reduction_col = _weighted_reduction_column(city_df)
    except KeyError as exc:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        _placeholder_axis(ax, f"Reduction distribution unavailable\n{exc}")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide reduction distribution")
        return

    values = pd.to_numeric(city_df[reduction_col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    if values.empty:
        _placeholder_axis(ax, "Reduction distribution unavailable\nno valid city reductions")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide reduction distribution")
        return

    bins = np.arange(0, max(50.0, float(np.ceil(values.max() / 5.0) * 5.0)) + 2.5, 2.5)
    counts, edges, _ = ax.hist(
        values,
        bins=bins,
        color="#9ecae1",
        edgecolor="#08519c",
        linewidth=0.6,
        alpha=0.90,
        zorder=2,
    )
    if len(values) >= 5 and np.unique(values).size > 1:
        density = stats.gaussian_kde(values)
        x_curve = np.linspace(float(edges[0]), float(edges[-1]), 240)
        y_curve = density(x_curve)
        if np.nanmax(y_curve) > 0:
            y_curve = y_curve / np.nanmax(y_curve) * max(counts.max(), 1)
            ax.plot(x_curve, y_curve, color="#08306b", linewidth=1.5, zorder=3)

    mean_value = float(values.mean())
    median_value = float(values.median())
    ax.axvspan(5.0, 20.0, color="#e5f5e0", alpha=0.55, zorder=0)
    ax.axvline(mean_value, color="#1b7837", linewidth=1.4, linestyle="--", zorder=4)
    ax.axvline(median_value, color="#762a83", linewidth=1.4, linestyle="-.", zorder=4)

    ax.set_xlabel("Lifecycle reduction vs all-e-bike baseline (%)", fontsize=11)
    ax.set_ylabel("Cities", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(0, max(counts.max() * 1.18, 1.0))

    if plot_title:
        ax.set_title(
            "City-level distribution of lifecycle reduction",
            fontsize=14,
            fontweight="bold",
        )

    _save_figure(fig, save_path, for_latex, plot_title, "Slide reduction distribution")


def _render_detour_route_mechanism(
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the route-detour mechanism concept panel."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _draw_detour_ratio_concept(ax, plot_title=False)
    if plot_title:
        ax.set_title("Direct drone routing versus detoured e-bike routing", fontsize=14)
    _save_figure(fig, save_path, for_latex, plot_title, "Slide detour route mechanism")


def _render_selected_trip_lifecycle(
    od_df: pd.DataFrame | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render lifecycle components for the selected median Beijing trip."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    if od_df is None:
        _placeholder_axis(ax, "Selected trip lifecycle unavailable\nBeijing OD data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide selected trip lifecycle")
        return

    sample = _select_median_trip_emission_sample(od_df)
    required = {
        "distance_km",
        "road_distance_m",
        "drone_emission_kg",
        "ebike_emission_kg",
        "drone_lifecycle_production_g",
        "ebike_lifecycle_production_g",
        "drone_lifecycle_eol_g",
        "ebike_lifecycle_eol_g",
    }
    if sample is None or not required.issubset(sample.index):
        _placeholder_axis(
            ax,
            "Selected trip lifecycle unavailable\nmissing lifecycle component columns",
        )
        _save_figure(fig, save_path, for_latex, plot_title, "Slide selected trip lifecycle")
        return

    production = np.array(
        [
            float(sample["ebike_lifecycle_production_g"]),
            float(sample["drone_lifecycle_production_g"]),
        ]
    )
    use_phase = np.array(
        [
            float(sample["ebike_emission_kg"]) * 1000.0,
            float(sample["drone_emission_kg"]) * 1000.0,
        ]
    )
    eol = np.array(
        [
            float(sample["ebike_lifecycle_eol_g"]),
            float(sample["drone_lifecycle_eol_g"]),
        ]
    )
    totals = production + use_phase + eol
    x = np.arange(2, dtype=float)

    colors = {
        "Production": "#6b7280",
        "Use phase": "#3b82f6",
        "End of life": "#cbd5e1",
    }
    ax.bar(x, production, width=0.58, color=colors["Production"], label="Production")
    ax.bar(x, use_phase, width=0.58, bottom=production, color=colors["Use phase"], label="Use phase")
    ax.bar(
        x,
        eol,
        width=0.58,
        bottom=production + use_phase,
        color=colors["End of life"],
        label="End of life",
    )

    for idx, total in enumerate(totals):
        ax.text(
            x[idx],
            total + totals.max() * 0.035,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["E-bike", "Drone"])
    ax.set_ylabel("Lifecycle emission (g CO2e/order)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=3, bbox_to_anchor=(0.0, 1.10))

    if plot_title:
        ax.set_title("Selected-trip lifecycle trade-off", fontsize=14, fontweight="bold")

    _save_figure(fig, save_path, for_latex, plot_title, "Slide selected trip lifecycle")


def _draw_missing_trip_panel(ax, text: str):
    """Render a compact placeholder for one selected-trip subpanel."""
    _placeholder_axis(ax, text)


def _render_trip_level_mechanism(
    od_df: pd.DataFrame | None,
    graph,
    router,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
    city: str = "beijing",
):
    """Render c--g trip-level mechanism panels for the selected median trip."""
    city_label = city.replace("_", " ").title()
    fig = plt.figure(figsize=(13.8, 6.6))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.08, 1.28, 1.08],
        height_ratios=[1.0, 0.96],
        hspace=0.42,
        wspace=0.34,
    )
    ax_c = fig.add_subplot(gs[0, 0])
    ax_d = fig.add_subplot(gs[0, 1:])
    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    ax_g = fig.add_subplot(gs[1, 2])

    _draw_detour_ratio_concept(ax_c, plot_title=plot_title)

    selected_trip = _select_median_trip_emission_sample(od_df) if od_df is not None else None
    distance_sample = _selected_trip_distance_sample(selected_trip, router)
    if distance_sample is not None and graph is not None:
        from visualization.roadnet import _draw_distance_panel

        _draw_distance_panel(
            ax_d,
            distance_sample,
            graph,
            show_legend=True,
            show_stats_box=False,
            target_aspect=2.0,
            rotate_air_diagonal=True,
        )
    else:
        _draw_missing_trip_panel(
            ax_d,
            f"{city_label} median trip distance comparison\nnot available",
        )

    production_df = _selected_trip_phase_parts(selected_trip, "production")
    eol_df = _selected_trip_phase_parts(selected_trip, "eol")
    if selected_trip is not None:
        air_km = float(selected_trip["distance_km"])
        road_km = float(selected_trip["road_distance_m"]) / 1000.0
        ebike_km = road_km
        if "ebike_kappa" in selected_trip.index and pd.notna(selected_trip["ebike_kappa"]):
            ebike_km *= float(selected_trip["ebike_kappa"])
        distance_label = f"E-bike {ebike_km:.2f} km; drone {air_km:.2f} km"
    else:
        distance_label = "Median trip"

    ebike_body_color = "#d95f35"
    ebike_battery_color = "#f2a07f"
    drone_body_color = "#1f77b4"
    drone_battery_color = "#9ecae1"
    phase_body_color = "#5f6368"
    phase_battery_color = "#c7c9cc"
    phase_body_colors = {"E-bike": ebike_body_color, "Drone": drone_body_color}
    phase_battery_colors = {"E-bike": ebike_battery_color, "Drone": drone_battery_color}

    if production_df is None:
        _draw_missing_trip_panel(
            ax_e,
            f"{city_label} median trip production data\nnot available",
        )
    else:
        _draw_trip_phase_barplot(
            ax_e,
            production_df,
            ylabel="Production\n(g CO$_2$e/order)",
            body_label="Body",
            battery_label="Battery",
            body_color=phase_body_color,
            battery_color=phase_battery_color,
            distance_label=distance_label,
            body_colors_by_vehicle=phase_body_colors,
            battery_colors_by_vehicle=phase_battery_colors,
            legend_body_color=phase_body_color,
            legend_battery_color=phase_battery_color,
        )

    if od_df is None:
        _draw_missing_trip_panel(
            ax_f,
            f"{city_label} median trip emission data\nnot available",
        )
    else:
        _draw_median_trip_emission_process(ax_f, od_df, plot_title=plot_title)
        ax_f.set_xlabel("Distance (km)", fontsize=11)
        ax_f.set_ylabel("Use phase\n(g CO$_2$e/leg)", fontsize=11)

    if eol_df is None:
        _draw_missing_trip_panel(
            ax_g,
            f"{city_label} median trip EOL data\nnot available",
        )
    else:
        _draw_trip_phase_barplot(
            ax_g,
            eol_df,
            ylabel="End of life\n(g CO$_2$e/order)",
            body_label="Body",
            battery_label="Battery",
            body_color=phase_body_color,
            battery_color=phase_battery_color,
            distance_label=distance_label,
            body_colors_by_vehicle=phase_body_colors,
            battery_colors_by_vehicle=phase_battery_colors,
            legend_body_color=phase_body_color,
            legend_battery_color=phase_battery_color,
        )

    for label, ax in zip(("a", "b", "c", "d", "e"), (ax_c, ax_d, ax_e, ax_f, ax_g)):
        ax.text(
            -0.03,
            1.02,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    if plot_title:
        fig.suptitle("Trip-level lifecycle mechanism", fontsize=14, fontweight="bold")

    _save_figure(fig, save_path, for_latex, plot_title, "Slide trip-level mechanism")


def _render_city_context_pair(
    data_dir: str,
    city_df: pd.DataFrame,
    exp_config: dict | None,
    contexts: tuple,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
    title: str | None = None,
):
    """Render two city-context blocks as one slide figure."""
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: i for i, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    if selected.empty:
        fig, ax = plt.subplots(figsize=(11.6, 6.4))
        _placeholder_axis(ax, "City contexts unavailable\nselected city rows missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide city context pair")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=os.path.splitext(os.path.basename(save_path))[0],
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=False,
        bin_width_km=0.1,
    )
    if built is None:
        fig, ax = plt.subplots(figsize=(11.6, 6.4))
        _placeholder_axis(ax, "City contexts unavailable\nbinned data build failed")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide city context pair")
        return

    binned_by_city, labels = built
    available_contexts = [
        context for context in contexts if context["region_key"] in binned_by_city
    ]
    if not available_contexts:
        fig, ax = plt.subplots(figsize=(11.6, 6.4))
        _placeholder_axis(ax, "City contexts unavailable\nbinned city data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide city context pair")
        return

    fig = plt.figure(figsize=(12.8, 6.4))
    outer = fig.add_gridspec(
        len(contexts),
        3,
        width_ratios=[1.0, 1.0, 0.92],
        hspace=0.42,
        wspace=0.30,
    )

    for i, context in enumerate(contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        if region_key not in binned_by_city:
            ax = fig.add_subplot(outer[i, :])
            _placeholder_axis(
                ax,
                f"{display_name} context unavailable\nbinned data missing",
            )
            continue

        _, ax_reduction, _ = _plot_reduction_detour_panel(
            fig,
            outer[i, 0],
            binned_by_city[region_key],
            labels.get(region_key, display_name),
            bin_width_km=0.1,
            min_air_km=0.2,
            absolute_reduction_metric="per_order_g",
            show_detour_boxplot=False,
            mark_reduction_break_even=False,
            show_city_label=False,
            bar_width_fraction=0.78,
            show_absolute_axis=False,
            show_saturation_fit=True,
        )
        ax_scatter = _plot_reduction_absolute_scatter_panel(
            fig,
            outer[i, 1],
            binned_by_city[region_key],
            labels.get(region_key, display_name),
            min_air_km=0.2,
            absolute_reduction_metric="per_order_g",
            show_city_label=False,
        )
        power_grid = outer[i, 2].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.72],
            wspace=0.12,
        )
        ax_power_mix = fig.add_subplot(power_grid[0])
        ax_power_legend = fig.add_subplot(power_grid[1])
        _plot_power_generation_mix_pie(
            ax_power_mix,
            power_region_key,
            legend_ax=ax_power_legend,
            radius=1.12,
            legend_fontsize=8.0,
            legend_handlelength=1.05,
            legend_handleheight=0.60,
            legend_labelspacing=0.32,
        )
        legend = ax_power_legend.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(8.0)

        for ax in (ax_reduction, ax_scatter):
            ax.xaxis.label.set_size(8.5)
            ax.yaxis.label.set_size(8.5)
            ax.tick_params(axis="both", labelsize=7.5)

        panel_label_x = ax_reduction.get_position().x0 - 0.045
        city_panel_labels = "abcdef"[3 * i: 3 * i + 3]
        for label, ax in zip(
            city_panel_labels,
            (ax_reduction, ax_scatter, ax_power_mix),
        ):
            pos = ax.get_position()
            fig.text(
                panel_label_x,
                pos.y1 + 0.008,
                label,
                fontsize=13,
                fontweight="bold",
                va="bottom",
            )
        city_pos = ax_reduction.get_position()
        fig.text(
            city_pos.x0,
            city_pos.y1 + 0.034,
            display_name,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    if plot_title:
        fig.suptitle(title or "City Contexts", fontsize=14, fontweight="bold")

    _save_figure(fig, save_path, for_latex, plot_title, "Slide city context pair")


def _render_gef_component_bins(
    df_enriched: pd.DataFrame | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render GEF-binned lifecycle components."""
    fig = plt.figure(figsize=(10.6, 4.8))
    if df_enriched is None:
        ax = fig.add_subplot(111)
        _placeholder_axis(ax, "GEF component bins unavailable\nmissing enriched stats")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide GEF component bins")
        return

    gs = fig.add_gridspec(1, 1)
    drawn = _plot_city_gef_scatter_components_axes(
        fig=fig,
        subplot_spec=gs[0],
        df=df_enriched,
        plot_title=plot_title,
    )
    if drawn is None:
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10.6, 4.8))
        _placeholder_axis(ax, "GEF component bins unavailable\nno valid GEF/component data")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide GEF component bins")
        return

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.95)
    _save_figure(fig, save_path, for_latex, plot_title, "Slide GEF component bins")


def _render_detour_reduction_regime(
    df_enriched: pd.DataFrame | None,
    regime: str,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the low- or high-GEF detour-reduction scatter as a standalone panel."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if df_enriched is None:
        _placeholder_axis(ax, "Detour-reduction scatter unavailable\nmissing enriched stats")
        _save_figure(
            fig,
            save_path,
            for_latex,
            plot_title,
            f"Slide detour reduction {regime} GEF",
        )
        return

    plot_df = df_enriched.copy()
    try:
        reduction_col = _weighted_reduction_column(plot_df)
    except KeyError as exc:
        _placeholder_axis(ax, f"Detour-reduction scatter unavailable\n{exc}")
        _save_figure(
            fig,
            save_path,
            for_latex,
            plot_title,
            f"Slide detour reduction {regime} GEF",
        )
        return

    required = ["detour_ratio", reduction_col, "grid_emission_factor"]
    valid = plot_df.dropna(subset=required).copy()
    if valid.empty:
        _placeholder_axis(ax, "Detour-reduction scatter unavailable\nno valid rows")
        _save_figure(
            fig,
            save_path,
            for_latex,
            plot_title,
            f"Slide detour reduction {regime} GEF",
        )
        return

    threshold = 0.3
    if regime == "low":
        valid = valid[valid["grid_emission_factor"] <= threshold].copy()
        line_color = "#006837"
        title_text = "Low GEF"
    else:
        valid = valid[valid["grid_emission_factor"] > threshold].copy()
        line_color = "#A50F15"
        title_text = "High GEF"

    if valid.empty:
        _placeholder_axis(ax, f"Detour-reduction scatter unavailable\nno {title_text.lower()} rows")
        _save_figure(
            fig,
            save_path,
            for_latex,
            plot_title,
            f"Slide detour reduction {regime} GEF",
        )
        return

    cmap = LinearSegmentedColormap.from_list(
        "gef_slide", ["#006837", "#31A354", "#D95F0E", "#A50F15"]
    )
    norm = Normalize(
        vmin=float(valid["grid_emission_factor"].min()),
        vmax=float(valid["grid_emission_factor"].max()),
    )
    ax.scatter(
        valid["detour_ratio"].to_numpy(dtype=float),
        valid[reduction_col].to_numpy(dtype=float),
        s=28,
        alpha=0.72,
        c=valid["grid_emission_factor"].to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        zorder=5,
    )

    x = valid["detour_ratio"].to_numpy(dtype=float)
    y = valid[reduction_col].to_numpy(dtype=float)
    if len(x) >= 3 and np.sum((x - np.mean(x)) ** 2) > 0:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(float(x.min()), float(x.max()), 120)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=line_color, linewidth=2.3, alpha=0.90, zorder=6)

        n = len(x)
        se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / max(n - 2, 1))
        t_val = stats.t.ppf(0.975, max(n - 2, 1))
        x_mean = np.mean(x)
        denom = np.sum((x - x_mean) ** 2)
        if denom > 0 and np.isfinite(se):
            ci = t_val * se * np.sqrt(1.0 / n + (x_line - x_mean) ** 2 / denom)
            ax.fill_between(
                x_line,
                y_line - ci,
                y_line + ci,
                alpha=0.12,
                color=line_color,
                zorder=4,
            )
    ax.set_xlabel(VARIABLE_LABELS.get("detour_ratio", "Detour ratio"), fontsize=11)
    ax.set_ylabel(VARIABLE_LABELS.get(reduction_col, reduction_col), fontsize=11)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label(VARIABLE_LABELS.get("grid_emission_factor", "Grid emission factor"), fontsize=10)

    if plot_title:
        ax.set_title(f"Detour ratio and lifecycle reduction: {title_text.lower()}-GEF cities", fontsize=14)

    _save_figure(
        fig,
        save_path,
        for_latex,
        plot_title,
        f"Slide detour reduction {regime} GEF",
    )


def _render_detour_reduction_by_gef(
    df_enriched: pd.DataFrame | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the slide detour-reduction figure using the paper scatter function."""
    if df_enriched is None:
        fig, ax = plt.subplots(figsize=(10.8, 4.6))
        _placeholder_axis(ax, "Detour-reduction scatter unavailable\nmissing enriched stats")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide detour reduction by GEF")
        return

    plot_df = df_enriched.copy()
    try:
        reduction_col = _weighted_reduction_column(plot_df)
    except KeyError as exc:
        fig, ax = plt.subplots(figsize=(10.8, 4.6))
        _placeholder_axis(ax, f"Detour-reduction scatter unavailable\n{exc}")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide detour reduction by GEF")
        return

    plot_grouped_scatter_split_panels(
        plot_df,
        feature_var="detour_ratio",
        result_var=reduction_col,
        group_var="grid_emission_factor",
        threshold=0.3,
        save_path=save_path,
        figsize=(10.8, 4.6),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _break_even_delta_style(prepared: dict):
    """Return shared heatmap styling for the break-even slices."""
    finite_blocks = []
    for gef_panel in prepared["panel_b"]["gef_panels"]:
        values = np.asarray(gef_panel["delta_g"], dtype=float)
        finite_blocks.append(values[np.isfinite(values)])
    nonempty_blocks = [block for block in finite_blocks if len(block)]
    finite_delta = (
        np.concatenate(nonempty_blocks)
        if nonempty_blocks
        else np.array([0.0], dtype=float)
    )
    max_abs = (
        float(np.nanpercentile(np.abs(finite_delta), 90))
        if len(finite_delta)
        else 1.0
    )
    max_abs = max(max_abs, 10.0)
    delta_bounds = np.linspace(-max_abs, max_abs, 9)
    delta_cmap = plt.get_cmap("RdBu_r", len(delta_bounds) - 1)
    delta_norm = BoundaryNorm(delta_bounds, ncolors=delta_cmap.N, clip=False)
    return delta_bounds, delta_cmap, delta_norm


def _render_break_even_surface(
    prepared: dict | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the stand-alone break-even surface."""
    fig = plt.figure(figsize=(7.6, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    if not prepared:
        _placeholder_axis(ax, "Break-even surface unavailable\nprepared data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide break-even surface")
        return

    air_distance_grid_km = prepared["air_distance_grid_km"]
    detour_ratios = prepared["detour_ratios"]
    panel_a = prepared["panel_a"]
    gef_limits = prepared["axes"]["gef_limits"]
    distance_limits = prepared["axes"]["distance_limits"]
    detour_limits = prepared["axes"]["detour_limits"]
    break_even_lines = panel_a.get("break_even_lines", [])

    line_cmap = plt.get_cmap("viridis")
    line_norm = plt.Normalize(vmin=gef_limits[0], vmax=gef_limits[1])
    surface_mesh = _build_break_even_marching_cubes_surface(
        air_distance_grid_km=air_distance_grid_km,
        detour_ratios=detour_ratios,
        delta_zero=panel_a["delta_zero"],
        delta_slope=panel_a["delta_slope"],
        gef_limits=gef_limits,
    )
    if surface_mesh:
        ax.plot_trisurf(
            surface_mesh["x"],
            surface_mesh["y"],
            surface_mesh["z"],
            triangles=surface_mesh["triangles"],
            cmap=line_cmap,
            norm=line_norm,
            linewidth=0.0,
            antialiased=False,
            shade=False,
            alpha=1.0,
            edgecolor="none",
            rasterized=True,
        )
        for overlay_line in _select_sparse_break_even_overlay_curves(
            break_even_lines,
            max_curves=5,
        ):
            ax.plot(
                overlay_line["x"],
                overlay_line["y"],
                overlay_line["z"],
                color=(0.16, 0.16, 0.16, 0.45),
                linewidth=0.8,
                solid_capstyle="round",
            )

    ax.set_xlim(*distance_limits)
    ax.set_ylim(*detour_limits)
    ax.set_zlim(*gef_limits)
    ax.set_xlabel("Air distance (km)", fontsize=10, labelpad=8)
    ax.set_ylabel("Detour ratio (road / air)", fontsize=10, labelpad=8)
    ax.set_zlabel("GEF (kg CO2e/kWh)", fontsize=10, labelpad=8)
    ax.set_proj_type("ortho")
    ax.view_init(elev=20, azim=225)
    ax.grid(alpha=0.20)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    surface_colorbar = fig.colorbar(
        ScalarMappable(cmap=line_cmap, norm=line_norm),
        ax=ax,
        fraction=0.038,
        pad=0.06,
        shrink=0.70,
        ticks=np.linspace(gef_limits[0], gef_limits[1], 4),
    )
    surface_colorbar.ax.set_title("GEF", fontsize=8.5, pad=4)
    surface_colorbar.ax.tick_params(labelsize=8.5)

    if plot_title:
        ax.set_title("Break-even operating envelope", fontsize=13, pad=8)

    _save_figure(fig, save_path, for_latex, plot_title, "Slide break-even surface")


def _render_break_even_heatmap(
    prepared: dict | None,
    panel_idx: int,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render one break-even Delta g slice for the selected GEF regime."""
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if not prepared:
        _placeholder_axis(ax, "Break-even heatmap unavailable\nprepared data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide break-even heatmap")
        return

    gef_panels = prepared["panel_b"]["gef_panels"]
    if panel_idx >= len(gef_panels):
        _placeholder_axis(ax, "Break-even heatmap unavailable\nGEF slice missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide break-even heatmap")
        return

    gef_panel = gef_panels[panel_idx]
    delta_g = np.asarray(gef_panel["delta_g"], dtype=float)
    distance_limits = prepared["axes"]["distance_limits"]
    detour_limits = prepared["axes"]["detour_limits"]
    X, Y = np.meshgrid(prepared["air_distance_grid_km"], prepared["detour_ratios"])
    delta_bounds, delta_cmap, delta_norm = _break_even_delta_style(prepared)

    im = ax.imshow(
        delta_g,
        cmap=delta_cmap,
        aspect="auto",
        origin="lower",
        extent=[
            distance_limits[0],
            distance_limits[1],
            detour_limits[0],
            detour_limits[1],
        ],
        norm=delta_norm,
        interpolation="nearest",
    )
    if np.nanmin(delta_g) <= 0.0 <= np.nanmax(delta_g):
        contour = ax.contour(X, Y, delta_g, levels=[0.0], colors="black", linewidths=1.3, zorder=6)
        ax.clabel(contour, fmt={0.0: "Delta g = 0"}, fontsize=8, inline=True)

    ax.set_xlim(*distance_limits)
    ax.set_ylim(detour_limits[0], min(detour_limits[1], 2.4))
    ax.set_xlabel("Air distance (km)", fontsize=11)
    ax.set_ylabel("Detour ratio (road / air)", fontsize=11)
    ax.grid(alpha=0.15)
    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.02,
        boundaries=delta_bounds,
        ticks=delta_bounds[::2],
        extend="both",
        spacing="proportional",
    )
    cbar.set_label("Delta g = E_drone - E_ebike (g CO2e/order)", fontsize=10)

    if plot_title:
        ax.set_title(f"Break-even slice: {gef_panel['label'].lower()}", fontsize=13)

    _save_figure(fig, save_path, for_latex, plot_title, "Slide break-even heatmap")


def _render_contribution_waterfall(
    prepared: dict | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the national contribution waterfall."""
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    if not prepared:
        _placeholder_axis(ax, "Contribution waterfall unavailable\nprepared data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide contribution waterfall")
        return

    _plot_contribution_waterfall(ax, prepared["national"], rotate_xticks=True, ylim_expand=0.10)
    if plot_title:
        ax.set_title("National contribution decomposition", fontsize=14, fontweight="bold")
    _save_figure(fig, save_path, for_latex, plot_title, "Slide contribution waterfall")


def _render_top_city_contributions(
    prepared: dict | None,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render stacked mechanism bars for the top city contributors."""
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    if not prepared:
        _placeholder_axis(ax, "Top-city contributions unavailable\nprepared data missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide top city contributions")
        return

    _plot_top_city_contribution_bars(ax, prepared.get("top_cities", [])[:15])
    if plot_title:
        ax.set_title("Top city contributors to the national net reduction", fontsize=14, fontweight="bold")
    _save_figure(fig, save_path, for_latex, plot_title, "Slide top city contributions")


def _prepare_pathway_bundle(
    named_csv_path: str | None,
    stats_csv_path: str | None,
    data_dir: str,
    exp_config: dict,
):
    """Return pathway scenario tables and exact curve endpoints for slide plots."""
    if not named_csv_path or not os.path.exists(named_csv_path):
        return None

    scenario_data = _prepare_pathway_scenario_data(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        data_dir=data_dir,
        exp_config=exp_config,
    )
    baseline_kappa = scenario_data["baseline_kappa"]
    annual_saved_total_kg = scenario_data["annual_saved_total_kg"]
    drone_share_median = scenario_data["drone_share_median"]
    curve_data = scenario_data["curve_data"]
    if "kappa" in curve_data:
        curve_data["kappa"] = _override_pathway_curve_endpoint(
            curve_data["kappa"],
            baseline_kappa,
            annual_saved_total_kg.get("a", np.nan) / 1e6,
            drone_share_median.get("a", np.nan),
        )
        curve_data["kappa"] = _override_pathway_curve_endpoint(
            curve_data["kappa"],
            0.65,
            annual_saved_total_kg.get("c", np.nan) / 1e6,
            drone_share_median.get("c", np.nan),
        )
    if "gef" in curve_data:
        curve_data["gef"] = _override_pathway_curve_endpoint(
            curve_data["gef"],
            1.0,
            annual_saved_total_kg.get("a", np.nan) / 1e6,
            drone_share_median.get("a", np.nan),
        )
        curve_data["gef"] = _override_pathway_curve_endpoint(
            curve_data["gef"],
            0.5,
            annual_saved_total_kg.get("b", np.nan) / 1e6,
            drone_share_median.get("b", np.nan),
        )
    return scenario_data


def _render_pathway_reduction_map(
    scenario_data: dict | None,
    scenario_label: str,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render one pathway reduction map with the share/saved inset."""
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    if not scenario_data:
        _placeholder_axis(ax, "Pathway reduction map unavailable\nmissing named scenarios")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide pathway reduction map")
        return

    panel_df = scenario_data["panel_dfs"].get(scenario_label)
    if panel_df is None or panel_df.empty:
        _placeholder_axis(ax, "Pathway reduction map unavailable\nscenario rows missing")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide pathway reduction map")
        return

    all_values = scenario_data["all_values"]
    v_max = max(max(all_values), 0.0) if all_values else 1.0
    cmap, norm, bin_edges, reduction_edges = _build_reduction_cmap(v_max)
    plot_regions_on_china_map(
        df=panel_df,
        color_by="max_lifecycle_reduction_pct",
        ax=ax,
        plot_title=False,
        cmap=cmap,
        norm=norm,
        draw_colorbar=False,
        draw_legend=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    scenario_meta = {
        label: title
        for label, title, kappa, gef_mult, _strategy in scenario_data["panels"]
    }
    title = scenario_meta[scenario_label]

    fig.canvas.draw()
    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes((ax_pos.x1 + 0.01, ax_pos.y0, 0.016, ax_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_ax,
        cmap,
        norm,
        bin_edges,
        reduction_edges,
        "Reduction vs today's all-e-bike anchor (%)",
    )
    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if plot_title:
        fig.suptitle(f"Pathway map: {title}", fontsize=14, fontweight="bold")

    _save_figure(fig, save_path, for_latex, plot_title, "Slide pathway reduction map")


def _render_pathway_curve(
    scenario_data: dict | None,
    curve_key: str,
    save_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render the GEF or kappa pathway response curve as a stand-alone slide figure."""
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    if not scenario_data:
        _placeholder_axis(ax, "Pathway curve unavailable\nmissing named scenarios")
        _save_figure(fig, save_path, for_latex, plot_title, "Slide pathway curve")
        return

    curve_data = scenario_data["curve_data"]
    saved_limits, share_limits = _pathway_curve_metric_limits(curve_data)
    if curve_key == "gef":
        axis_label = "Grid emission factor multiplier"
        curve_df = curve_data.get("gef")
        mirror_x = True
    else:
        axis_label = r"E-bike bundling factor $\kappa$"
        curve_df = curve_data.get("kappa")
        mirror_x = False

    _plot_pathway_curve_panel(
        ax,
        curve_df,
        panel_label="",
        axis_label=axis_label,
        orientation="horizontal",
        saved_limits=saved_limits,
        share_limits=share_limits,
        saved_as_bar=True,
        mirror_x=mirror_x,
    )

    if plot_title:
        title = "Grid decarbonization pathway" if curve_key == "gef" else "Bundling pathway"
        ax.set_title(title, fontsize=14, fontweight="bold")

    _save_figure(fig, save_path, for_latex, plot_title, "Slide pathway curve")


def render_all(
    data_dir: str,
    figures_dir: str,
    for_latex: bool = False,
    plot_title: bool = False,
    only: set | None = None,
    exp_config: dict | None = None,
):
    """Render all slide-ready Results figures."""
    out = os.path.join(figures_dir, "slides")
    os.makedirs(out, exist_ok=True)

    def _want(name: str) -> bool:
        return only is None or name in only

    copied_from_paper = set()
    for slide_name, paper_name in PAPER_FIGURE_REUSE.items():
        if _want(slide_name) and _copy_paper_figure_if_available(
            figures_dir,
            paper_name,
            slide_name,
            for_latex,
            plot_title,
        ):
            copied_from_paper.add(slide_name)

    def _need(name: str) -> bool:
        return _want(name) and name not in copied_from_paper

    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    if not os.path.exists(stats_path):
        print(f"Missing {stats_path}; slide rendering skipped")
        return

    df_stats = pd.read_csv(stats_path)
    city_df = None
    contribution_prepared = None
    break_even_prepared = None
    df_enriched = None
    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")

    if any(
        _need(name)
        for name in (
            "slide_city_context_dali",
            "slide_city_context_chengdu",
            "slide_city_context_tianjin",
            "slide_city_context_guangzhou",
            *SLIDE_CITY_CONTEXT_PAIRS,
            *PAPER_CACHE_BACKED_SLIDES,
        )
    ):
        raw_city_df = _city_rows(df_stats)
        gef_path = os.path.join(data_dir, "gef_china.csv")
        gef_df = None
        if os.path.exists(gef_path):
            gef_df = pd.read_csv(gef_path, encoding="gbk")
            gef_df.columns = ["province", "gef"]
        else:
            print(
                f"Missing {gef_path}; slide figures will use paper caches or "
                "GEF fallbacks where available"
            )
        if gef_df is not None:
            city_df = raw_city_df.merge(gef_df, on="province", how="left")
            city_df = _prepare_reduction_detour_city_stats(city_df)
        else:
            city_df = None
        contribution_city_df = _prepare_contribution_decomposition_city_stats(
            raw_city_df,
            gef_df,
            exp_config,
        )
        if any(
            _need(name)
            for name in (
                "slide_contribution_waterfall",
                "slide_top_city_contributions",
            )
        ):
            named_df = pd.read_csv(named_path) if os.path.exists(named_path) else None
            contribution_prepared = _prepare_contribution_decomposition_full(
                data_dir=data_dir,
                city_df=contribution_city_df,
                named_df=named_df,
                exp_config=exp_config,
            )
        if any(
            _need(name)
            for name in (
                "slide_break_even_surface",
                "slide_break_even_low_gef",
                "slide_break_even_high_gef",
            )
        ):
            break_even_prepared = _prepare_break_even_phase_diagram(
                data_dir=data_dir,
                exp_config=exp_config,
            )

    if any(
        _need(name)
        for name in (
            "slide_gef_component_bins",
            "slide_detour_reduction_by_gef",
            "slide_detour_reduction_low_gef",
            "slide_detour_reduction_high_gef",
        )
    ):
        enriched_path = os.path.join(data_dir, "all_region_stats_enriched.csv")
        if os.path.exists(enriched_path):
            df_enriched = _attach_enrichment_columns(df_stats, pd.read_csv(enriched_path))
        else:
            print(f"Missing {enriched_path}; enriched-stat slide figures may show placeholders")

    if _need("slide_city_reduction_map"):
        _render_city_reduction_map(
            df_stats,
            save_path=os.path.join(out, "slide_city_reduction_map.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_reduction_distribution"):
        _render_reduction_distribution(
            df_stats,
            save_path=os.path.join(out, "slide_reduction_distribution.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_detour_route_mechanism"):
        _render_detour_route_mechanism(
            save_path=os.path.join(out, "slide_detour_route_mechanism.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_selected_trip_lifecycle"):
        beijing_od_path = os.path.join(data_dir, "city", "beijing", "beijing_od_pairs.csv")
        beijing_od_df = pd.read_csv(beijing_od_path) if os.path.exists(beijing_od_path) else None
        _render_selected_trip_lifecycle(
            beijing_od_df,
            save_path=os.path.join(out, "slide_selected_trip_lifecycle.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_trip_level_mechanism"):
        beijing_od_path = os.path.join(data_dir, "city", "beijing", "beijing_od_pairs.csv")
        beijing_graphml_path = os.path.join(data_dir, "city", "beijing", "beijing_roadnet.graphml")
        beijing_od_df = pd.read_csv(beijing_od_path) if os.path.exists(beijing_od_path) else None
        beijing_graph, beijing_router = None, None
        if os.path.exists(beijing_graphml_path):
            from roadnet.sparse_graph import SparseGraph
            from roadnet.tile_router import TileRouter

            beijing_graph = SparseGraph.from_graphml(beijing_graphml_path)
            beijing_router = TileRouter(beijing_graph)
        else:
            print(
                "slide_trip_level_mechanism: Beijing roadnet missing, "
                f"panel d will show placeholder ({beijing_graphml_path})"
            )
        _render_trip_level_mechanism(
            beijing_od_df,
            beijing_graph,
            beijing_router,
            save_path=os.path.join(out, "slide_trip_level_mechanism.png"),
            for_latex=for_latex,
            plot_title=plot_title,
            city="beijing",
        )

    if _need("slide_gef_component_bins"):
        _render_gef_component_bins(
            df_enriched,
            save_path=os.path.join(out, "slide_gef_component_bins.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if city_df is not None:
        for figure_name, contexts in SLIDE_CITY_CONTEXT_PAIRS.items():
            if not _need(figure_name):
                continue
            _render_city_context_pair(
                data_dir=data_dir,
                city_df=city_df,
                exp_config=exp_config,
                contexts=contexts,
                save_path=os.path.join(out, f"{figure_name}.png"),
                for_latex=for_latex,
                plot_title=plot_title,
                title=figure_name.replace("slide_", "").replace("_", " ").title(),
            )

        for figure_name, context in SLIDE_CONTEXTS.items():
            if not _need(figure_name):
                continue
            _render_city_context(
                data_dir=data_dir,
                out=out,
                for_latex=for_latex,
                plot_title=plot_title,
                city_df=city_df,
                exp_config=exp_config,
                figure_name=figure_name,
                **context,
            )

    if _need("slide_detour_reduction_by_gef"):
        _render_detour_reduction_by_gef(
            df_enriched,
            save_path=os.path.join(out, "slide_detour_reduction_by_gef.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if only is not None and _need("slide_detour_reduction_low_gef"):
        _render_detour_reduction_regime(
            df_enriched,
            regime="low",
            save_path=os.path.join(out, "slide_detour_reduction_low_gef.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if only is not None and _need("slide_detour_reduction_high_gef"):
        _render_detour_reduction_regime(
            df_enriched,
            regime="high",
            save_path=os.path.join(out, "slide_detour_reduction_high_gef.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_break_even_surface"):
        _render_break_even_surface(
            break_even_prepared,
            save_path=os.path.join(out, "slide_break_even_surface.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_break_even_low_gef"):
        _render_break_even_heatmap(
            break_even_prepared,
            panel_idx=0,
            save_path=os.path.join(out, "slide_break_even_low_gef.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_break_even_high_gef"):
        _render_break_even_heatmap(
            break_even_prepared,
            panel_idx=1,
            save_path=os.path.join(out, "slide_break_even_high_gef.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_contribution_waterfall"):
        _render_contribution_waterfall(
            contribution_prepared,
            save_path=os.path.join(out, "slide_contribution_waterfall.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_top_city_contributions"):
        _render_top_city_contributions(
            contribution_prepared,
            save_path=os.path.join(out, "slide_top_city_contributions.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    scenario_data = None
    if any(
        _need(name)
        for name in (
            "slide_pathway_current_reduction",
            "slide_pathway_clean_grid_reduction",
            "slide_pathway_bundling_reduction",
            "slide_pathway_gef_curve",
            "slide_pathway_kappa_curve",
        )
    ):
        scenario_data = _prepare_pathway_bundle(
            named_csv_path=named_path if os.path.exists(named_path) else None,
            stats_csv_path=stats_path,
            data_dir=data_dir,
            exp_config=exp_config,
        )

    if _need("slide_pathway_current_reduction"):
        _render_pathway_reduction_map(
            scenario_data,
            scenario_label="a",
            save_path=os.path.join(out, "slide_pathway_current_reduction.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_pathway_clean_grid_reduction"):
        _render_pathway_reduction_map(
            scenario_data,
            scenario_label="b",
            save_path=os.path.join(out, "slide_pathway_clean_grid_reduction.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_pathway_bundling_reduction"):
        _render_pathway_reduction_map(
            scenario_data,
            scenario_label="c",
            save_path=os.path.join(out, "slide_pathway_bundling_reduction.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_pathway_gef_curve"):
        _render_pathway_curve(
            scenario_data,
            curve_key="gef",
            save_path=os.path.join(out, "slide_pathway_gef_curve.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _need("slide_pathway_kappa_curve"):
        _render_pathway_curve(
            scenario_data,
            curve_key="kappa",
            save_path=os.path.join(out, "slide_pathway_kappa_curve.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )


def main():
    _bootstrap_paths()
    from utils import load_config

    parser = argparse.ArgumentParser(description="Render slide-ready Results figures")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--png", action="store_true", help="Force PNG output")
    parser.add_argument("--titled", action="store_true", help="Add titles to figures")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated figure names to render")
    args = parser.parse_args()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    for_latex = (not args.png) and config.get("for_latex", False)
    plot_title = args.titled or config.get("plot_title", False)
    only = set(args.only.split(",")) if args.only else None

    print(f"for_latex={for_latex}, plot_title={plot_title}, only={only or 'all'}")
    render_all(
        config["data_dir"],
        config["figures_dir"],
        for_latex=for_latex,
        plot_title=plot_title,
        only=only,
        exp_config=config,
    )


if __name__ == "__main__":
    main()
