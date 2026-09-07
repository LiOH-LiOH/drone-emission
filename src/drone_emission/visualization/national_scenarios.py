"""Visualization functions for cross-region correlation and emission analysis."""

from .prepared_bundle import prepared_plot, prepared_data, box_statistics

from .palette import color as paper_color

import json
import os

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


def _first_available_column(df: pd.DataFrame, candidates: tuple[str, ...], purpose: str) -> str:
    """Return the first available candidate column, preferring weighted metrics."""
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError(f"No {purpose} column found; tried {', '.join(candidates)}")


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
    palette = [paper_color("national_scenarios.ffffe0"), paper_color("national_scenarios.f7fcb9"), paper_color("national_scenarios.d9f0a3"), paper_color("national_scenarios.addd8e"),
               paper_color("national_scenarios.78c679"), paper_color("national_scenarios.41ab5d"), paper_color("national_scenarios.238443"), paper_color("national_scenarios.005a32")]
    cmap_colors = [paper_color("national_scenarios.cccccc")] + palette[:len(reduction_edges) - 1]
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
    def _format_tick(value):
        if np.isclose(value, 0):
            return "0"
        if abs(value) >= 1000:
            scaled = value / 1000
            return f"{int(scaled)}k" if np.isclose(scaled, round(scaled)) else f"{scaled:g}k"
        return f"{int(value)}" if np.isclose(value, round(value)) else f"{value:g}"

    tick_labels = [_format_tick(b) for b in reduction_edges]
    cbar.ax.set_yticklabels(tick_labels)
    plt.setp(cbar.ax.get_yticklabels(), rotation=0)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=9, length=2)
    cbar.set_label(label_text, fontsize=11)
    return cbar, len(bin_edges) - 1




def _format_kappa_value(value: float) -> str:
    """Format kappa compactly without hiding configured non-round values."""
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _scenario_drone_share_column(df: pd.DataFrame) -> str:
    """Return the preferred drone-share column for named scenario rows."""
    return _first_available_column(
        df,
        (
            "drone_order_ratio_pct",
            "weighted_drone_ratio_pct",
            "oracle_drone_order_ratio_pct",
            "drone_ratio_pct",
        ),
        "scenario drone share",
    )



def _pathway_scenario_definitions(baseline_kappa: float) -> list[tuple]:
    """Return the standard pathway scenario definitions."""
    return [
        ("a", "Today", baseline_kappa, 1.0, "optimal_mixed"),
        ("b", "Halved GEF", baseline_kappa, 0.5, "optimal_mixed_2030"),
        ("c", "Operator bundling", 0.65, 1.0, "optimal_mixed"),
        ("d", "Halved GEF + bundling", 0.65, 0.5, "optimal_mixed_2030"),
    ]


@prepared_data
def _prepare_pathway_scenario_data(
    stats_csv_path: str,
    scenario_csv_path: str,
    curve_csv_path: str,
    analysis_report_path: str,
):
    """Load cohort-bound pathway analysis tables used by map figures."""
    from drone_emission.analysis.pathway_scenarios import PATHWAY_ANALYSIS_VERSION
    from drone_emission.paper_cohort import cohort_hash, locked_region_keys

    base = pd.read_csv(stats_csv_path)
    oracle_rows = pd.read_csv(scenario_csv_path)
    curve_rows = pd.read_csv(curve_csv_path)
    with open(analysis_report_path, "r", encoding="utf-8") as handle:
        analysis_report = json.load(handle)
    if analysis_report.get("analysis_version") != PATHWAY_ANALYSIS_VERSION:
        raise RuntimeError("pathway analysis version is stale; rerun analyze")
    if analysis_report.get("paper_cohort_sha256") != cohort_hash():
        raise RuntimeError("pathway analysis belongs to a different cohort")
    if analysis_report.get("city_count") != len(locked_region_keys()):
        raise RuntimeError("pathway analysis has the wrong city count")
    baseline_kappa = float(analysis_report["baseline_kappa"])
    panels = _pathway_scenario_definitions(baseline_kappa)
    expected_rows = len(locked_region_keys()) * len(panels)
    if len(oracle_rows) != expected_rows:
        raise RuntimeError(
            f"pathway scenario table has {len(oracle_rows)} rows; "
            f"expected {expected_rows}"
        )
    curve_data = {
        pathway: rows.drop(columns=["pathway"]).reset_index(drop=True)
        for pathway, rows in curve_rows.groupby("pathway", sort=False)
    }
    anchor_rows = (
        oracle_rows[["region_key", "anchor_annual_kg"]]
        .dropna()
        .drop_duplicates(subset=["region_key"])
    )
    has_annual = True

    panel_dfs = {}
    panel_named = {}
    drone_share_median = {}
    annual_saved_total_kg = {}
    all_values = []
    for label, _title, _k, _g, _strategy in panels:
        rows = oracle_rows[oracle_rows["panel_label"] == label].copy()
        rows["province_en"] = rows["province"].map(PROVINCE_NAME_EN).fillna(
            rows["province"]
        )
        panel_named[label] = rows
        share_col = (
            _scenario_drone_share_column(rows)
            if not rows.empty
            else "drone_ratio_pct"
        )
        sub = rows[["region_key", "annual_lifecycle_kg"]].merge(
            anchor_rows, on="region_key", how="left"
        )
        sub["max_lifecycle_reduction_pct"] = np.where(
            sub["anchor_annual_kg"] > 0,
            (
                1.0
                - pd.to_numeric(sub["annual_lifecycle_kg"], errors="coerce")
                / sub["anchor_annual_kg"]
            ) * 100.0,
            np.nan,
        )
        sub = sub[["region_key", "max_lifecycle_reduction_pct"]]
        merged = base.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            sub, on="region_key", how="left"
        )
        panel_dfs[label] = merged
        all_values.extend(merged["max_lifecycle_reduction_pct"].dropna().tolist())
        drone_share_median[label] = float(rows[share_col].median()) if len(rows) else float("nan")
        joined = rows[["region_key", "annual_lifecycle_kg"]].merge(
            anchor_rows, on="region_key", how="inner"
        )
        annual_saved_total_kg[label] = float(
            (joined["anchor_annual_kg"] - joined["annual_lifecycle_kg"]).sum()
        )

    return {
        "base": base,
        "baseline_kappa": baseline_kappa,
        "panels": panels,
        "has_annual": has_annual,
        "anchor_rows": anchor_rows,
        "panel_dfs": panel_dfs,
        "panel_named": panel_named,
        "drone_share_median": drone_share_median,
        "annual_saved_total_kg": annual_saved_total_kg,
        "all_values": all_values,
        "curve_data": curve_data,
    }


def _plot_pathway_curve_panel(
    ax,
    curve_df: pd.DataFrame,
    panel_label: str,
    axis_label: str,
    orientation: str = "horizontal",
    saved_limits: tuple[float, float] = None,
    share_limits: tuple[float, float] = None,
    saved_as_bar: bool = False,
    show_saved_axis: bool = True,
    show_share_axis: bool = True,
    mirror_x: bool = False,
):
    """Draw total saved CO2 and drone share on dual axes."""
    if curve_df is None or curve_df.empty:
        ax.text(0.5, 0.5, "Pathway curve unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="dimgray")
        ax.set_axis_off()
        return None

    x = curve_df["x"].to_numpy(dtype=float)
    saved = curve_df["saved_kt"].to_numpy(dtype=float)
    share = curve_df["drone_share_pct"].to_numpy(dtype=float)
    if saved_limits is None:
        saved_limits = _padded_visible_ylim(saved)
    if saved_as_bar and saved_limits is not None:
        saved_limits = (
            0.0,
            max(0.0, float(saved_limits[1])),
        )
    if share_limits is None:
        share_limits = (0, min(105, max(100, np.nanmax(share) * 1.08)))

    finite_x = x[np.isfinite(x)]
    if finite_x.size > 1:
        x_steps = np.diff(np.sort(np.unique(finite_x)))
        x_steps = x_steps[x_steps > 0]
        bar_width = float(x_steps.min() * 0.72) if x_steps.size else 0.08
    else:
        bar_width = 0.08

    if orientation == "vertical":
        if saved_as_bar:
            ax.barh(x, saved, height=bar_width, left=0.0, color=paper_color("national_scenarios.1b7837"), alpha=0.75)
        else:
            ax.plot(saved, x, color=paper_color("national_scenarios.1b7837"), marker="o", linewidth=1.8, markersize=4)
        ax.set_xlabel(r"CO$_2$ saved (kt/yr)", fontsize=10, color=paper_color("national_scenarios.1b7837"))
        ax.set_ylabel(axis_label, fontsize=10)
        ax.tick_params(axis="x", labelcolor=paper_color("national_scenarios.1b7837"), labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        if saved_limits is not None:
            ax.set_xlim(*saved_limits)
        if saved_as_bar:
            ax.margins(x=0)
            ax.set_xlim(left=0.0)

        ax_share = ax.twiny()
        ax_share.plot(share, x, color=paper_color("national_scenarios.2166ac"), marker="s", linewidth=1.6, markersize=3.5)
        ax_share.set_xlabel("Drone share (%)", fontsize=10, color=paper_color("national_scenarios.2166ac"))
        ax_share.tick_params(axis="x", labelcolor=paper_color("national_scenarios.2166ac"), labelsize=8)
        if share_limits is not None:
            ax_share.set_xlim(*share_limits)
    else:
        if saved_as_bar:
            ax.bar(x, saved, width=bar_width, bottom=0.0, color=paper_color("national_scenarios.1b7837"), alpha=0.75)
        else:
            ax.plot(x, saved, color=paper_color("national_scenarios.1b7837"), marker="o", linewidth=1.8, markersize=4)
        ax.set_xlabel(axis_label, fontsize=10)
        ax.set_ylabel(r"CO$_2$ saved (kt/yr)", fontsize=10, color=paper_color("national_scenarios.1b7837"))
        ax.tick_params(axis="y", labelcolor=paper_color("national_scenarios.1b7837"), labelsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        if x[0] > x[-1]:
            ax.invert_xaxis()
        if saved_limits is not None:
            ax.set_ylim(*saved_limits)
        if saved_as_bar:
            ax.margins(y=0)
            ax.set_ylim(bottom=0.0)
            ax.spines["bottom"].set_position(("data", 0.0))

        ax_share = ax.twinx()
        ax_share.plot(x, share, color=paper_color("national_scenarios.2166ac"), marker="s", linewidth=1.6, markersize=3.5)
        ax_share.set_ylabel("Drone share (%)", fontsize=10, color=paper_color("national_scenarios.2166ac"))
        ax_share.tick_params(axis="y", labelcolor=paper_color("national_scenarios.2166ac"), labelsize=8)
        if share_limits is not None:
            ax_share.set_ylim(*share_limits)
        if mirror_x:
            ax.invert_xaxis()
        if not show_saved_axis:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
        if not show_share_axis:
            ax_share.set_ylabel("")
            ax_share.tick_params(axis="y", right=False, labelright=False)
            ax_share.spines["right"].set_visible(False)

    ax.text(-0.08, 1.03, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="right")
    return ax_share


def _override_pathway_curve_endpoint(
    curve_df: pd.DataFrame,
    x_value: float,
    saved_kt: float,
    drone_share_pct: float,
) -> pd.DataFrame:
    """Replace the curve point at x_value with exact panel summary values."""
    if curve_df is None or curve_df.empty:
        return curve_df
    out = curve_df.copy()
    idx = np.flatnonzero(np.isclose(out["x"].to_numpy(dtype=float), x_value))
    if idx.size == 0:
        return out
    row_idx = out.index[idx[0]]
    if np.isfinite(saved_kt):
        out.loc[row_idx, "saved_kt"] = saved_kt
    if np.isfinite(drone_share_pct):
        out.loc[row_idx, "drone_share_pct"] = drone_share_pct
    return out


def _pathway_curve_metric_limits(curve_data: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return shared CO2-saved and drone-share limits for pathway curve panels."""
    frames = [
        frame
        for frame in (curve_data or {}).values()
        if frame is not None and not frame.empty
    ]
    if not frames:
        return None, None

    saved = np.concatenate([
        frame["saved_kt"].to_numpy(dtype=float)
        for frame in frames
        if "saved_kt" in frame.columns
    ])
    share = np.concatenate([
        frame["drone_share_pct"].to_numpy(dtype=float)
        for frame in frames
        if "drone_share_pct" in frame.columns
    ])
    saved_limits = _padded_visible_ylim(saved)
    share_finite = share[np.isfinite(share)]
    if share_finite.size == 0:
        share_limits = None
    else:
        share_limits = (
            max(0.0, float(share_finite.min()) - 5.0),
            min(105.0, max(100.0, float(share_finite.max()) + 5.0)),
        )
    return saved_limits, share_limits


def _add_map_parameter_bars(
    ax_map,
    gef_mult: float,
    kappa: float,
):
    """Add axis-free GEF-multiplier and kappa bars inside a map panel."""
    inset = ax_map.inset_axes([0.035, 0.06, 0.25, 0.20], zorder=10)
    values = [gef_mult, kappa]
    colors = [paper_color("national_scenarios.2166ac"), paper_color("national_scenarios.e08214")]
    value_labels = [f"{gef_mult:.1f}", _format_kappa_value(kappa)]

    inset.patch.set_alpha(0.0)
    inset.bar([0, 1], values, width=0.55, color=colors, alpha=0.9, zorder=3)

    inset.set_xlim(-0.6, 1.6)
    inset.set_ylim(0.0, 1.14)
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["GEF", r"$\kappa$"], fontsize=6.5)
    inset.set_yticks([])
    inset.tick_params(axis="x", length=0, pad=1)
    inset.grid(False)
    for spine in inset.spines.values():
        spine.set_visible(False)

    for x, value, value_label, color in zip(
        [0, 1], values, value_labels, colors
    ):
        inset.text(
            x,
            value,
            value_label,
            ha="center",
            va="bottom",
            fontsize=6.5,
            color=color,
            zorder=5,
        )
    return inset


def _pathway_curve_report(
    ax,
    ax_twin,
    curve_df: pd.DataFrame,
    orientation: str,
    co2_saved_mark: str = "line",
    co2_saved_baseline: float = None,
):
    """Return plotted pathway curve points and axis metadata."""
    points = []
    if curve_df is not None and not curve_df.empty:
        for _, row in curve_df.iterrows():
            points.append(
                {
                    "pathway_x": _json_safe_scalar(row.get("x")),
                    "co2_saved_kt_per_year": _json_safe_scalar(row.get("saved_kt")),
                    "drone_share_pct": _json_safe_scalar(row.get("drone_share_pct")),
                }
            )
    return {
        "axis_primary": _axis_report(ax),
        "axis_secondary": _axis_report(ax_twin) if ax_twin is not None else None,
        "orientation": orientation,
        "co2_saved_mark": co2_saved_mark,
        "co2_saved_baseline": _json_safe_scalar(co2_saved_baseline),
        "co2_saved_color": paper_color("national_scenarios.1b7837"),
        "drone_share_mark": "line",
        "drone_share_color": paper_color("national_scenarios.2166ac"),
        "points": points,
    }


def _pathway_scenario_panel_report(
    rows: pd.DataFrame,
    panel_df: pd.DataFrame,
    color_col: str,
    ax_map,
    ax_box,
    ax_line,
    parameter_bar_ax,
    share_col: str,
    province_order: list[str],
    panel_label: str,
    title: str,
    kappa: float,
    gef_mult: float,
    annual_saved_kg: float,
    drone_share_median: float,
):
    """Return compact LLM-readable metadata for one pathway scenario panel."""
    def compact_city_records(frame):
        records = []
        for _, row in frame.iterrows():
            records.append(
                {
                    "region_key": _json_safe_scalar(row.get("region_key")),
                    "region_name_en": _json_safe_scalar(row.get("region_name_en")),
                    "region_name_cn": _json_safe_scalar(row.get("region_name_cn")),
                    "province": _json_safe_scalar(row.get("province")),
                    color_col: _json_safe_scalar(row.get(color_col)),
                }
            )
        return records

    city_values = panel_df.reindex(
        columns=[
            "region_key",
            "region_name_en",
            "region_name_cn",
            "province",
            color_col,
        ]
    ).copy()
    city_values[color_col] = pd.to_numeric(city_values[color_col], errors="coerce")
    ranked = city_values.dropna(subset=[color_col]).sort_values(color_col, ascending=False)

    province_boxes = []
    if rows is not None and not rows.empty and share_col in rows.columns:
        for province in province_order:
            values = rows.loc[rows["province_en"] == province, share_col].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            q1, median, q3 = np.percentile(finite, [25, 50, 75])
            province_boxes.append(
                {
                    "province": province,
                    "n_cities": int(finite.size),
                    "drone_share_pct_q1": _json_safe_scalar(q1),
                    "drone_share_pct_median": _json_safe_scalar(median),
                    "drone_share_pct_q3": _json_safe_scalar(q3),
                    "drone_share_pct_min": _json_safe_scalar(finite.min()),
                    "drone_share_pct_max": _json_safe_scalar(finite.max()),
                }
            )

    parameter_bar_inset = None
    if parameter_bar_ax is not None:
        parameter_bar_inset = {
            "axis": _axis_report(parameter_bar_ax),
            "scale": [0.0, 1.0],
            "bars": [
                {
                    "label": "GEF multiplier",
                    "value": _json_safe_scalar(gef_mult),
                    "unit": "multiplier",
                    "color": paper_color("national_scenarios.2166ac"),
                },
                {
                    "label": "E-bike bundling factor kappa",
                    "value": _json_safe_scalar(kappa),
                    "unit": "dimensionless",
                    "color": paper_color("national_scenarios.e08214"),
                },
            ],
        }

    return {
        "panel_letter": panel_label,
        "title": title,
        "scenario": {
            "kappa": _json_safe_scalar(kappa),
            "gef_multiplier": _json_safe_scalar(gef_mult),
            "annual_co2_saved_kt_per_year": _json_safe_scalar(annual_saved_kg / 1e6),
            "median_city_drone_share_pct": _json_safe_scalar(drone_share_median),
        },
        "map_axis": _axis_report(ax_map),
        "boxplot_axis": _axis_report(ax_box),
        "province_lifecycle_axis": _axis_report(ax_line),
        "map_color_by": color_col,
        "top_10_cities_by_map_value": compact_city_records(ranked.head(10)),
        "bottom_10_cities_by_map_value": compact_city_records(
            ranked.tail(10).sort_values(color_col)
        ),
        "province_drone_share_boxplots": province_boxes,
        "parameter_bar_inset": parameter_bar_inset,
    }


def _write_regions_map_pathway_scenarios_report(
    report_path: str,
    output_paths: dict,
    figsize: tuple,
    colorbar_figsize: tuple,
    scenario_csv_path: str,
    curve_csv_path: str,
    stats_csv_path: str,
    panels: list[tuple],
    panel_dfs: dict,
    panel_named: dict,
    panel_axes: dict,
    parameter_bar_axes: dict,
    province_order: list[str],
    annual_saved_total_kg: dict,
    drone_share_median: dict,
    shared_cmap,
    bin_edges,
    reduction_edges,
    cbar_ax,
):
    """Write a compact JSON report for the pathway scenario subfigures."""
    scenario_reports = {}
    color_col = "max_lifecycle_reduction_pct"
    for label, title, kappa, gef_mult, _strategy in panels:
        rows = panel_named.get(label, pd.DataFrame())
        share_col = _scenario_drone_share_column(rows) if not rows.empty else "drone_ratio_pct"
        ax_map, ax_box, ax_line = panel_axes[label]
        scenario_reports[title.lower().replace(" ", "_").replace("+", "plus")] = (
            _pathway_scenario_panel_report(
                rows=rows,
                panel_df=panel_dfs[label],
                color_col=color_col,
                ax_map=ax_map,
                ax_box=ax_box,
                ax_line=ax_line,
                parameter_bar_ax=parameter_bar_axes.get(label),
                share_col=share_col,
                province_order=province_order,
                panel_label=None,
                title=title,
                kappa=kappa,
                gef_mult=gef_mult,
                annual_saved_kg=annual_saved_total_kg.get(label, np.nan),
                drone_share_median=drone_share_median.get(label, np.nan),
            )
        )

    payload = {
        "figure_name": "regions_map_pathway_scenarios",
        "figure_paths": output_paths,
        "source_scenario_csv_path": scenario_csv_path,
        "source_curve_csv_path": curve_csv_path,
        "source_stats_csv_path": stats_csv_path,
        "scenario_figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "colorbar_figure_size_inches": {
            "width": colorbar_figsize[0],
            "height": colorbar_figsize[1],
        },
        "coordinate_convention": (
            "Axis bbox values are figure-fraction coordinates. Map colors encode "
            "reduction vs today's all-ebike anchor; gray indicates unstudied regions."
        ),
        "layout": {
            "scenario_subfigure": (
                "Map with GEF/kappa inset above province boxplot with "
                "lifecycle-emission overlay"
            ),
            "suggested_assembly": [
                ["today", "operator_bundling"],
                ["halved_gef", "halved_gef_bundling"],
            ],
            "shared_colorbar": "Vertical strip placed to the right of the 2x2 scenario grid",
        },
        "global_encodings": {
            "map_colorbar": _colorbar_report(
                cbar_ax,
                bin_edges,
                reduction_edges,
                "Reduction vs all-e-bike baseline (%)",
                shared_cmap,
            ),
            "scenario_parameter_bar_colors": {
                "gef_multiplier": paper_color("national_scenarios.2166ac"),
                "kappa": paper_color("national_scenarios.e08214"),
            },
        },
        "panels": scenario_reports,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map pathway scenarios report saved to {report_path}")


def _write_regions_map_pathway_scenario_maps_report(
    save_path: str,
    output_paths: dict,
    figsize: tuple,
    scenario_csv_path: str,
    curve_csv_path: str,
    stats_csv_path: str,
    curve_data: dict,
    curve_axes: list,
    curve_twins: list,
):
    """Write a compact JSON report for the standalone pathway curves."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    payload = {
        "figure_name": "regions_map_pathway_scenario_maps",
        "figure_paths": output_paths,
        "source_scenario_csv_path": scenario_csv_path,
        "source_curve_csv_path": curve_csv_path,
        "source_stats_csv_path": stats_csv_path,
        "panel_size_inches": {"width": figsize[0], "height": figsize[1]},
        "panel_size_mm": {
            "width": figsize[0] * 25.4,
            "height": figsize[1] * 25.4,
        },
        "coordinate_convention": (
            "Axis bbox values are figure-fraction coordinates. Both panels run "
            "left-to-right from today's parameter value toward the improved scenario."
        ),
        "layout": {
            "assembly": "Standalone g and h panels intended for vertical stacking",
            "top": "g: grid emission factor pathway",
            "bottom": "h: e-bike bundling pathway",
            "suggested_inter_panel_gap_mm": 6.0,
        },
        "global_encodings": {
            "curve_colors": {
                "co2_saved": paper_color("national_scenarios.1b7837"),
                "drone_share": paper_color("national_scenarios.2166ac"),
            },
            "curve_marks": {
                "co2_saved": "bar",
                "drone_share": "line",
            },
            "curve_baselines": {
                "co2_saved": 0.0,
            },
        },
        "panels": {
            "g_gef_pathway_curve": {
                **_pathway_curve_report(
                    curve_axes[0],
                    curve_twins[0],
                    curve_data.get("gef"),
                    "horizontal",
                    co2_saved_mark="bar",
                    co2_saved_baseline=0.0,
                ),
                "left_to_right_direction": "GEF multiplier 1.0 to 0.5",
                "mirrored_x_axis": False,
                "visible_y_axes": ["co2_saved", "drone_share"],
            },
            "h_kappa_pathway_curve": {
                **_pathway_curve_report(
                    curve_axes[1],
                    curve_twins[1],
                    curve_data.get("kappa"),
                    "horizontal",
                    co2_saved_mark="bar",
                    co2_saved_baseline=0.0,
                ),
                "left_to_right_direction": "baseline kappa to 0.65",
                "mirrored_x_axis": False,
                "visible_y_axes": ["co2_saved", "drone_share"],
            },
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map pathway scenario maps report saved to {report_path}")


@prepared_plot()
def plot_regions_map_pathway_scenario_maps(
    stats_csv_path: str,
    scenario_csv_path: str,
    curve_csv_path: str,
    analysis_report_path: str,
    save_path: str = None,
    figsize: tuple = (120 / 25.4, 72 / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Render standalone GEF and kappa pathway-curve panels.

    Panel g shows the grid-emission-factor pathway and panel h shows the e-bike
    bundling pathway. Both run left-to-right from today's parameter value to the
    improved scenario, use shared y-axis limits, and are sized for vertical
    assembly into an approximately 120 mm by 150 mm paper figure.

    Args:
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        scenario_csv_path: Prepared city-by-scenario analysis table.
        curve_csv_path: Prepared national pathway curve table.
        analysis_report_path: Cohort and analysis-version metadata.
        save_path: Base output path. Panel suffixes are added before extension.
            Shows plots if None.
        figsize: Size tuple for each standalone panel.
        for_latex: If True, save as PDF.
        plot_title: If True, show a title in each panel.
    """
    scenario_data = _prepare_pathway_scenario_data(
        stats_csv_path=stats_csv_path,
        scenario_csv_path=scenario_csv_path,
        curve_csv_path=curve_csv_path,
        analysis_report_path=analysis_report_path,
    )
    curve_data = scenario_data["curve_data"]
    drone_share_median = scenario_data["drone_share_median"]
    annual_saved_total_kg = scenario_data["annual_saved_total_kg"]
    baseline_kappa = scenario_data["baseline_kappa"]

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

    curve_saved_limits, curve_share_limits = _pathway_curve_metric_limits(curve_data)
    panel_specs = (
        (
            "g_gef_pathway",
            "gef",
            "g",
            "Grid emission factor multiplier",
            "Grid decarbonization pathway",
        ),
        (
            "h_kappa_pathway",
            "kappa",
            "h",
            r"E-bike bundling factor $\kappa$",
            "E-bike bundling pathway",
        ),
    )
    curve_axes = []
    curve_twins = []
    curve_figures = []
    output_paths = {}

    if save_path:
        save_root, save_ext = os.path.splitext(save_path)
        if not save_ext:
            save_ext = ".png"
        if for_latex:
            save_ext = ".pdf"

    for suffix, curve_key, panel_label, axis_label, title in panel_specs:
        fig, ax = plt.subplots(figsize=figsize)
        ax_twin = _plot_pathway_curve_panel(
            ax,
            curve_data.get(curve_key),
            "",
            axis_label,
            orientation="horizontal",
            saved_limits=curve_saved_limits,
            share_limits=curve_share_limits,
            saved_as_bar=True,
            show_saved_axis=True,
            show_share_axis=True,
            mirror_x=False,
        )
        ax.grid(False)
        if plot_title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        fig.subplots_adjust(left=0.17, right=0.83, bottom=0.22, top=0.88)
        curve_axes.append(ax)
        curve_twins.append(ax_twin)
        curve_figures.append(fig)

        if save_path:
            panel_path = f"{save_root}_{suffix}{save_ext}"
            if plot_title:
                panel_path = _add_title_suffix(panel_path)
            os.makedirs(os.path.dirname(panel_path), exist_ok=True)
            fig.savefig(panel_path, dpi=300)
            output_paths[suffix] = panel_path
            print(f"Pathway curve panel {panel_label} saved to {panel_path}")
        else:
            plt.show()

    if save_path:
        report_base_path = f"{save_root}{save_ext}"
        if plot_title:
            report_base_path = _add_title_suffix(report_base_path)
        _write_regions_map_pathway_scenario_maps_report(
            save_path=report_base_path,
            output_paths=output_paths,
            figsize=figsize,
            scenario_csv_path=scenario_csv_path,
            curve_csv_path=curve_csv_path,
            stats_csv_path=stats_csv_path,
            curve_data=curve_data,
            curve_axes=curve_axes,
            curve_twins=curve_twins,
        )

    for fig in curve_figures:
        plt.close(fig)


@prepared_plot()
def plot_regions_map_pathway_scenarios(
    stats_csv_path: str,
    scenario_csv_path: str,
    curve_csv_path: str,
    analysis_report_path: str,
    save_path: str = None,
    figsize: tuple = (3.05, 3.75),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Write four standalone pathway-scenario map and boxplot subfigures.

    Each scenario is saved as a separate subfigure with a map above a province
    boxplot. The boxplot has one box per province (drone-share distribution
    across cities within that province), plus a province lifecycle-emission
    overlay. Provinces and all numeric scales are consistent across subfigures.
    An axis-free inset identifies each scenario's GEF multiplier and kappa.
    A fifth standalone vertical colorbar is written for shared use when the four
    scenarios are assembled into a 2x2 paper figure; its height matches two
    stacked scenario canvases.

    Panels use the cohort-bound per-OD oracle-min mixed allocation prepared by
    the analyze stage:
        today                    baseline kappa, gef_mult=1.0
        halved GEF               baseline kappa, gef_mult=0.5
        operator bundling        kappa=0.65, gef_mult=1.0
        halved GEF + bundling    kappa=0.65, gef_mult=0.5

    Reduction values (map fill) use annual lifecycle totals against the weighted
    all-ebike city baseline from `all_region_stats.csv`. Boxplot y-axis is drone
    order share in 0..100.

    Args:
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        scenario_csv_path: Prepared city-by-scenario analysis table.
        curve_csv_path: Prepared national pathway curve table.
        analysis_report_path: Cohort and analysis-version metadata.
        save_path: Base output path. Scenario suffixes are added before extension.
            Shows plots if None.
        figsize: Size tuple for each standalone scenario subfigure.
        for_latex: If True, save as PDF.
        plot_title: If True, show the scenario title in each subfigure.
    """
    from drone_emission.visualization.regions import plot_regions_on_china_map

    scenario_data = _prepare_pathway_scenario_data(
        stats_csv_path=stats_csv_path,
        scenario_csv_path=scenario_csv_path,
        curve_csv_path=curve_csv_path,
        analysis_report_path=analysis_report_path,
    )
    base = scenario_data["base"]
    anchor_rows = scenario_data["anchor_rows"]
    panels = scenario_data["panels"]
    panel_dfs = scenario_data["panel_dfs"]
    panel_named = scenario_data["panel_named"]
    drone_share_median = scenario_data["drone_share_median"]
    annual_saved_total_kg = scenario_data["annual_saved_total_kg"]
    all_values = scenario_data["all_values"]
    has_annual = scenario_data["has_annual"]

    if not all_values:
        raise RuntimeError(
            "pathway scenario analysis contains no rows matching region stats"
        )
        v_max = 1.0
    else:
        v_max = max(max(all_values), 0.0)

    shared_cmap, shared_norm, bin_edges, reduction_edges = _build_reduction_cmap(v_max)

    # Province order: total annual lifecycle emission under today's all-ebike
    # baseline (configured kappa, gef=1, strategy=all_ebike). Largest emitters on the left.
    anchor_named = anchor_rows.merge(
        base[["region_key", "province"]],
        on="region_key",
        how="left",
    )
    anchor_named["province_en"] = anchor_named["province"].map(PROVINCE_NAME_EN).fillna(anchor_named["province"])
    province_emission_baseline_kt = (
        anchor_named.groupby("province_en")["anchor_annual_kg"].sum() / 1e6
    )
    province_order = province_emission_baseline_kt.sort_values(ascending=False).index.tolist()
    if not province_order:
        province_order = sorted(
            base["province"].map(PROVINCE_NAME_EN).fillna(base["province"]).dropna().unique()
        )

    # Per-panel province totals (kt CO2/yr) for the overlay line.
    if has_annual:
        panel_province_emission_kt = {
            lbl: panel_named[lbl].groupby("province_en")["annual_lifecycle_kg"].sum() / 1e6
            for lbl in panel_named
        }
    else:
        panel_province_emission_kt = {
            lbl: pd.Series(dtype=float)
            for lbl in panel_named
        }
    line_y_max = max(
        float(s.max()) if len(s) else 0.0 for s in panel_province_emission_kt.values()
    ) if panel_province_emission_kt else 0.0
    line_y_max = max(line_y_max, 1.0) * 1.05
    output_suffixes = {
        "a": "today",
        "b": "halved_gef",
        "c": "operator_bundling",
        "d": "halved_gef_bundling",
    }
    panel_axes = {}
    parameter_bar_axes = {}
    scenario_figures = []
    output_paths = {}

    if save_path:
        save_root, save_ext = os.path.splitext(save_path)
        if not save_ext:
            save_ext = ".png"
        if for_latex:
            save_ext = ".pdf"

    for label, title, k, g, _s in panels:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[6.2, 2.2], hspace=0.12)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_box = fig.add_subplot(gs[1, 0])

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
        if plot_title:
            ax_map.set_title(title, fontsize=9)

        parameter_bar_axes[label] = _add_map_parameter_bars(ax_map, g, k)

        rows = panel_named[label]
        share_col = _scenario_drone_share_column(rows) if not rows.empty else "drone_ratio_pct"
        data = [rows.loc[rows["province_en"] == p, share_col].values for p in province_order]
        ax_box.bxp(
            box_statistics(data), positions=range(len(province_order)), widths=0.65,
            patch_artist=True, showfliers=False,
            medianprops=dict(color=paper_color("ink.black"), linewidth=0.8),
            boxprops=dict(facecolor=paper_color("national_scenarios.a6d96a"), edgecolor=paper_color("national_scenarios.1a9641"), linewidth=0.45),
            whiskerprops=dict(color=paper_color("national_scenarios.1a9641"), linewidth=0.45),
            capprops=dict(color=paper_color("national_scenarios.1a9641"), linewidth=0.45),
        )
        ax_box.set_ylim(0, 102)
        ax_box.set_xlim(-0.7, len(province_order) - 0.3)
        ax_box.set_xticks(range(len(province_order)))
        ax_box.set_xticklabels(province_order, rotation=90, fontsize=5.2)
        ax_box.set_ylabel("Drone share (%)", fontsize=6.5)
        ax_box.tick_params(axis="y", labelsize=5.8, length=2)
        ax_box.grid(axis="y", linestyle=":", alpha=0.4)
        ax_box.axhline(drone_share_median[label], color=paper_color("national_scenarios.1a9641"), linestyle="--", linewidth=0.55, alpha=0.8)

        # Overlay: total annual lifecycle emission per province (gray line, right axis).
        emission_series = panel_province_emission_kt[label]
        emission_y = [float(emission_series.get(p, 0.0)) for p in province_order]
        ax_line = ax_box.twinx()
        ax_line.plot(
            range(len(province_order)), emission_y,
            color="dimgray", linewidth=0.7, marker="o", markersize=1.7,
            alpha=0.85, zorder=5,
        )
        ax_line.set_ylim(0, line_y_max)
        ax_line.set_ylabel("Annual lifecycle (kt CO$_2$)", fontsize=6.5, color="dimgray")
        ax_line.tick_params(axis="y", labelsize=5.8, colors="dimgray", length=2)
        panel_axes[label] = (ax_map, ax_box, ax_line)
        fig.subplots_adjust(
            left=0.08,
            right=0.92,
            bottom=0.08,
            top=0.98,
            hspace=0.12,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box_tight = ax_box.get_tightbbox(renderer).transformed(
            fig.transFigure.inverted()
        )
        line_tight = ax_line.get_tightbbox(renderer).transformed(
            fig.transFigure.inverted()
        )
        map_box = ax_map.get_position()
        map_width = max(box_tight.x1, line_tight.x1) - min(
            box_tight.x0, line_tight.x0
        )
        map_height = map_width * map_box.height / map_box.width
        map_x0 = min(box_tight.x0, line_tight.x0)
        map_y1 = 0.99
        ax_map.set_position(
            [map_x0, map_y1 - map_height, map_width, map_height],
            which="both",
        )

        if save_path:
            scenario_path = f"{save_root}_{output_suffixes[label]}{save_ext}"
            if plot_title:
                scenario_path = _add_title_suffix(scenario_path)
            os.makedirs(os.path.dirname(scenario_path), exist_ok=True)
            fig.savefig(scenario_path, dpi=300, bbox_inches="tight")
            output_paths[output_suffixes[label]] = scenario_path
            print(f"Pathway scenario {title} saved to {scenario_path}")
        scenario_figures.append(fig)

    colorbar_figsize = (0.60, figsize[1] * 2)
    colorbar_fig = plt.figure(figsize=colorbar_figsize)
    cbar_ax = colorbar_fig.add_axes((0.08, 0.03, 0.22, 0.94))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        colorbar_fig,
        cbar_ax,
        shared_cmap,
        shared_norm,
        bin_edges,
        reduction_edges,
        r"Reduction vs all-e-bike baseline (%)",
    )
    cbar.ax.tick_params(pad=3)
    colorbar_fig.canvas.draw()
    numeric_tick = cbar.ax.yaxis.get_majorticklabels()[0]
    numeric_left_px = numeric_tick.get_window_extent().x0
    unstudied_text = cbar_ax.text(
        1.0,
        0.5 / n_bands,
        "No data",
        transform=cbar_ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        rotation=90,
    )
    colorbar_fig.canvas.draw()
    unstudied_left_px = unstudied_text.get_window_extent().x0
    x_shift_axes = (
        cbar_ax.transAxes.inverted().transform(
            (numeric_left_px, 0)
        )[0]
        - cbar_ax.transAxes.inverted().transform(
            (unstudied_left_px, 0)
        )[0]
    )
    unstudied_text.set_x(unstudied_text.get_position()[0] + x_shift_axes)

    if save_path:
        colorbar_path = f"{save_root}_colorbar{save_ext}"
        os.makedirs(os.path.dirname(colorbar_path), exist_ok=True)
        colorbar_fig.savefig(colorbar_path, dpi=300, bbox_inches="tight")
        output_paths["colorbar"] = colorbar_path
        print(f"Pathway scenarios shared colorbar saved to {colorbar_path}")

        report_path = f"{save_root}_report.json"
        if plot_title:
            report_path = _add_title_suffix(report_path)
        output_paths["report"] = report_path
        _write_regions_map_pathway_scenarios_report(
            report_path=report_path,
            output_paths=output_paths,
            figsize=figsize,
            colorbar_figsize=colorbar_figsize,
            scenario_csv_path=scenario_csv_path,
            curve_csv_path=curve_csv_path,
            stats_csv_path=stats_csv_path,
            panels=panels,
            panel_dfs=panel_dfs,
            panel_named=panel_named,
            panel_axes=panel_axes,
            parameter_bar_axes=parameter_bar_axes,
            province_order=province_order,
            annual_saved_total_kg=annual_saved_total_kg,
            drone_share_median=drone_share_median,
            shared_cmap=shared_cmap,
            bin_edges=bin_edges,
            reduction_edges=reduction_edges,
            cbar_ax=cbar_ax,
        )
    else:
        plt.show()

    for fig in scenario_figures:
        plt.close(fig)
    plt.close(colorbar_fig)


def _json_safe_scalar(value):
    """Return a JSON-safe scalar with missing numeric values represented as None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _json_safe_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_json_safe_scalar(value) for value in values]


def _axis_report(ax):
    """Return final axis position and scale metadata."""
    pos = ax.get_position()
    return {
        "bbox_figure_fraction": {
            "x0": _json_safe_scalar(pos.x0),
            "y0": _json_safe_scalar(pos.y0),
            "x1": _json_safe_scalar(pos.x1),
            "y1": _json_safe_scalar(pos.y1),
            "width": _json_safe_scalar(pos.width),
            "height": _json_safe_scalar(pos.height),
        },
        "xlim_data": _json_safe_array(ax.get_xlim()),
        "ylim_data": _json_safe_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _json_safe_array(ax.get_xticks()),
        "yticks_data": _json_safe_array(ax.get_yticks()),
    }


def _colorbar_report(ax, bin_edges, tick_edges, label_text, cmap):
    """Return colorbar axis and discrete scale metadata."""
    from matplotlib.colors import to_hex

    return {
        "axis": _axis_report(ax),
        "label": label_text,
        "bin_edges": _json_safe_array(bin_edges),
        "tick_edges": _json_safe_array(tick_edges),
        "colors": [to_hex(cmap(i)) for i in range(cmap.N)],
        "unstudied_band": {
            "label": "No data",
            "color": paper_color("national_scenarios.cccccc"),
            "bin_index": 0,
        },
    }


def _padded_visible_ylim(values: np.ndarray, include_zero: bool = True):
    """Return padded y-limits for finite visible values."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    y_min = float(finite.min())
    y_max = float(finite.max())
    if include_zero:
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 0.0)
    if np.isclose(y_min, y_max):
        pad = max(abs(y_min) * 0.1, 1.0)
    else:
        pad = (y_max - y_min) * 0.08
    return y_min - pad, y_max + pad
