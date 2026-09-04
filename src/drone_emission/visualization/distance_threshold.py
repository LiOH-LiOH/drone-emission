"""Visualization functions for cross-region correlation and emission analysis."""

import json
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def _writing_numeric_summary(values) -> dict:
    """Return compact finite-value statistics for a writing-oriented report."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    series = series[np.isfinite(series)]
    if series.empty:
        return {"count": 0}
    return {
        "count": int(len(series)),
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


DISTANCE_THRESHOLD_TYPICAL_CITIES = (
    "chengdu",
    "guangzhou",
    "beijing",
    "shanghai",
)
DISTANCE_THRESHOLD_CITY_LABELS = {
    "chengdu": "Chengdu",
    "guangzhou": "Guangzhou",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
}
DISTANCE_THRESHOLD_LABEL_OFFSETS = {
    "chengdu": (6, 7),
    "guangzhou": (6, 7),
    "beijing": (6, 7),
    "shanghai": (6, -11),
}
DISTANCE_THRESHOLD_COMPARISON_LABEL_OFFSETS = {
    1: {
        "beijing": (12, 12),
        "shanghai": (-12, 20),
    },
}
DISTANCE_THRESHOLD_POINT_SIZE = 28
DISTANCE_THRESHOLD_POINT_ALPHA = 0.9
DISTANCE_THRESHOLD_LANDSCAPE_AXIS_LABELS = (
    "Road delivery distance threshold (km)",
    "Maximum lifecycle-emissions reduction (%)",
)
DISTANCE_THRESHOLD_ORACLE_AXIS_LABELS = (
    "Distance-based lifecycle-emissions\nreduction (%)",
    "Additional reduction under trip-level allocation\n(percentage points)",
)


def _save_distance_threshold_figure(
    fig,
    save_path: str,
    for_latex: bool,
    plot_title: bool,
    description: str,
    report_payload: dict,
    save_tight: bool = True,
):
    """Save one threshold figure and its compact JSON report."""
    if not save_path:
        plt.show()
        plt.close(fig)
        return None
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_kwargs = {"bbox_inches": "tight"} if save_tight else {}
    fig.savefig(save_path, dpi=300, facecolor="white", **save_kwargs)
    report_payload = {
        "figure_path": os.path.relpath(save_path, start=os.getcwd()),
        **report_payload,
    }
    write_figure_report(
        save_path.rsplit(".", 1)[0] + "_report.json",
        report_payload,
    )
    print(f"{description} saved to {save_path}")
    plt.close(fig)
    return save_path


def _style_distance_threshold_axis(ax):
    """Apply restrained exploratory-analysis styling."""
    ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _prepare_distance_threshold_all_city_data(
    summary_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and reshape all-city threshold and Oracle results."""
    required = {
        "region_key",
        "region_name_en",
        "distance_basis",
        "optimal_threshold_km",
        "annual_reduction_kg",
        "annual_reduction_pct",
        "n_od_pairs_total",
        "n_od_pairs_unreachable",
        "unreachable_annual_orders",
        "total_annual_orders",
    }
    missing = required.difference(summary_df.columns)
    if missing:
        raise KeyError(f"Missing all-city threshold columns: {sorted(missing)}")

    duplicates = summary_df.duplicated(["region_key", "distance_basis"])
    if duplicates.any():
        raise ValueError("Duplicate city/distance-basis rows in threshold summary")
    basis_sets = summary_df.groupby("region_key")["distance_basis"].agg(set)
    incomplete = basis_sets[basis_sets != {"road", "straight"}]
    if not incomplete.empty:
        raise ValueError(
            "Cities without both distance bases: "
            + ", ".join(incomplete.index.astype(str))
        )

    oracle_required = {"region_key", "province"}
    missing_oracle_columns = oracle_required.difference(oracle_df.columns)
    if missing_oracle_columns:
        raise KeyError(
            "Missing Oracle city columns: "
            f"{sorted(missing_oracle_columns)}"
        )
    oracle = oracle_df.copy()
    if "region_level" in oracle.columns:
        oracle = oracle[oracle["region_level"] == "city"].copy()
    if oracle.duplicated("region_key").any():
        raise ValueError("Duplicate city rows in Oracle results")
    oracle["macro_region"] = oracle["province"].map(
        macro_region_for_province
    )
    unmapped = oracle[oracle["macro_region"].isna()]
    if not unmapped.empty:
        details = ", ".join(
            f"{row.region_key} ({row.province})"
            for row in unmapped[["region_key", "province"]].itertuples(
                index=False
            )
        )
        raise ValueError(
            "Oracle cities without a China macro-region mapping: " + details
        )
    oracle["macro_region_key"] = oracle["macro_region"].map(
        lambda spec: spec["key"]
    )
    oracle["macro_region_label"] = oracle["macro_region"].map(
        lambda spec: spec["label"]
    )
    oracle["macro_region_color"] = oracle["macro_region"].map(
        lambda spec: spec["color"]
    )
    if {"total_ebike_lifecycle", "oracle_min_lifecycle_kg"}.issubset(
        oracle.columns
    ):
        ebike = pd.to_numeric(
            oracle["total_ebike_lifecycle"], errors="coerce"
        )
        oracle_lifecycle = pd.to_numeric(
            oracle["oracle_min_lifecycle_kg"], errors="coerce"
        )
        oracle["oracle_annual_reduction_pct"] = np.where(
            ebike > 0,
            (1.0 - oracle_lifecycle / ebike) * 100.0,
            np.nan,
        )
    elif "oracle_min_reduction_vs_ebike_pct" in oracle.columns:
        oracle["oracle_annual_reduction_pct"] = pd.to_numeric(
            oracle["oracle_min_reduction_vs_ebike_pct"], errors="coerce"
        )
    else:
        raise KeyError(
            "Oracle results require oracle_min_reduction_vs_ebike_pct or "
            "total_ebike_lifecycle plus oracle_min_lifecycle_kg"
        )
    if "total_ebike_lifecycle" not in oracle.columns:
        raise KeyError(
            "Oracle results require total_ebike_lifecycle for the full-city "
            "road-policy baseline"
        )
    oracle["total_ebike_lifecycle"] = pd.to_numeric(
        oracle["total_ebike_lifecycle"], errors="coerce"
    )
    oracle = oracle[
        [
            "region_key",
            "province",
            "macro_region_key",
            "macro_region_label",
            "macro_region_color",
            "oracle_annual_reduction_pct",
            "total_ebike_lifecycle",
        ]
    ]

    identity = summary_df[
        ["region_key", "region_name_en"]
    ].drop_duplicates("region_key")
    metric_columns = (
        "optimal_threshold_km",
        "annual_reduction_kg",
        "annual_reduction_pct",
    )
    wide = identity.set_index("region_key")
    for basis in ("road", "straight"):
        basis_df = summary_df[summary_df["distance_basis"] == basis].set_index(
            "region_key"
        )
        for column in metric_columns:
            wide[f"{basis}_{column}"] = pd.to_numeric(
                basis_df[column], errors="coerce"
            )
    wide = wide.join(oracle.set_index("region_key"), how="left")
    wide = wide.reset_index()
    wide["road_full_domain_reduction_pct"] = np.where(
        wide["total_ebike_lifecycle"] > 0,
        wide["road_annual_reduction_kg"]
        / wide["total_ebike_lifecycle"]
        * 100.0,
        np.nan,
    )
    wide["road_oracle_gap_pp"] = (
        wide["oracle_annual_reduction_pct"]
        - wide["road_full_domain_reduction_pct"]
    )
    numeric = [
        f"{basis}_{column}"
        for basis in ("road", "straight")
        for column in metric_columns
    ] + [
        "oracle_annual_reduction_pct",
        "total_ebike_lifecycle",
        "road_full_domain_reduction_pct",
        "road_oracle_gap_pp",
    ]
    if not np.isfinite(wide[numeric].to_numpy(dtype=float)).all():
        missing_oracle = wide.loc[
            ~np.isfinite(wide["oracle_annual_reduction_pct"]), "region_key"
        ].tolist()
        raise ValueError(
            "All-city threshold and Oracle metrics must be finite; missing "
            "Oracle cities: " + ", ".join(map(str, missing_oracle))
        )

    return wide


def _plot_distance_threshold_all_city_scatter(
    ax,
    wide: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    panel_idx: int,
    diagonal: bool,
    zero_baseline: bool = False,
):
    """Draw one all-city threshold scatter panel on an existing axis."""
    highlight = wide["region_key"].isin(DISTANCE_THRESHOLD_TYPICAL_CITIES)
    ax.scatter(
        wide.loc[~highlight, x_col],
        wide.loc[~highlight, y_col],
        color=wide.loc[~highlight, "macro_region_color"],
        s=DISTANCE_THRESHOLD_POINT_SIZE,
        alpha=DISTANCE_THRESHOLD_POINT_ALPHA,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.scatter(
        wide.loc[highlight, x_col],
        wide.loc[highlight, y_col],
        color=wide.loc[highlight, "macro_region_color"],
        s=DISTANCE_THRESHOLD_POINT_SIZE,
        alpha=DISTANCE_THRESHOLD_POINT_ALPHA,
        edgecolor="#151515",
        linewidth=0.8,
        zorder=4,
    )
    if diagonal:
        lower = min(float(wide[x_col].min()), float(wide[y_col].min()))
        upper = max(float(wide[x_col].max()), float(wide[y_col].max()))
        padding = max((upper - lower) * 0.04, 0.1)
        lower -= padding
        upper += padding
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="#666666",
            linewidth=0.9,
            linestyle="--",
            zorder=0,
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    elif zero_baseline:
        x_upper = max(float(wide[x_col].max()) * 1.06, 0.1)
        y_upper = max(float(wide[y_col].max()) * 1.08, 0.1)
        ax.axhline(
            0.0,
            color="#8A8A8A",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )
        ax.set_xlim(0.0, x_upper)
        ax.set_ylim(0.0, y_upper)
    for _, row in wide[highlight].iterrows():
        offset = DISTANCE_THRESHOLD_COMPARISON_LABEL_OFFSETS.get(
            panel_idx, {}
        ).get(
            row["region_key"],
            DISTANCE_THRESHOLD_LABEL_OFFSETS[row["region_key"]],
        )
        ax.annotate(
            DISTANCE_THRESHOLD_CITY_LABELS[row["region_key"]],
            (row[x_col], row[y_col]),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.5,
            fontweight="semibold",
            color="#222222",
            ha="right" if offset[0] < 0 else "left",
            bbox={
                "boxstyle": "square,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.5,
            },
            zorder=5,
        )
    ax.set_xlabel(x_label, fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    _style_distance_threshold_axis(ax)
    ax.grid(False)
    ax.tick_params(labelsize=7)


def _distance_threshold_all_city_report(
    summary_df: pd.DataFrame,
    wide: pd.DataFrame,
    road_reduction_col: str = "road_annual_reduction_pct",
) -> dict:
    """Return shared report fields for the split all-city figures."""
    city_count = int(wide["region_key"].nunique())
    configured_count = int(
        summary_df.attrs.get("configured_city_count", city_count)
    )
    road = summary_df[summary_df["distance_basis"] == "road"]
    unreachable_rows = int(road["n_od_pairs_unreachable"].sum())
    total_rows = int(road["n_od_pairs_total"].sum())
    unreachable_orders = float(road["unreachable_annual_orders"].sum())
    reachable_orders = float(road["total_annual_orders"].sum())
    report = {
        "city_count": city_count,
        "configured_city_count": configured_count,
        "highlighted_cities": list(DISTANCE_THRESHOLD_TYPICAL_CITIES),
        "macro_regions": _distance_threshold_macro_region_records(wide),
        "road_threshold_km": _writing_numeric_summary(
            wide["road_optimal_threshold_km"]
        ),
        "road_reduction_pct": _writing_numeric_summary(
            wide[road_reduction_col]
        ),
        "oracle_reduction_pct": _writing_numeric_summary(
            wide["oracle_annual_reduction_pct"]
        ),
        "excluded_unreachable_routes": {
            "od_pair_count": unreachable_rows,
            "od_pair_pct": unreachable_rows / total_rows * 100.0,
            "annual_orders": unreachable_orders,
            "annual_order_pct": (
                unreachable_orders
                / (unreachable_orders + reachable_orders)
                * 100.0
            ),
        },
    }
    if road_reduction_col == "road_full_domain_reduction_pct":
        report["oracle_minus_road_gap_pp"] = _writing_numeric_summary(
            wide["road_oracle_gap_pp"]
        )
        report["road_policy_comparison_scope"] = {
            "reduction_denominator": (
                "full-city all-e-bike lifecycle emissions"
            ),
            "unreachable_road_assignment": "e-bike",
            "threshold_optimization": "valid road-distance OD pairs",
        }
    return report


def _distance_threshold_macro_region_records(
    wide: pd.DataFrame,
) -> list[dict]:
    """Summarize the shared macro-region color encoding."""
    records = []
    for spec in CHINA_MACRO_REGION_SPECS:
        count = int((wide["macro_region_key"] == spec["key"]).sum())
        if count:
            records.append(
                {
                    "macro_region": spec["key"],
                    "label": spec["label"],
                    "color": spec["color"],
                    "city_count": count,
                }
            )
    return records


def _add_distance_threshold_macro_region_legend(fig, wide: pd.DataFrame):
    """Add one shared macro-region legend for the full Figure 4 panel."""
    handles = []
    for record in _distance_threshold_macro_region_records(wide):
        handles.append(
            Patch(
                facecolor=record["color"],
                edgecolor="none",
                label=f"{record['label']} ({record['city_count']})",
            )
        )
    return fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(handles),
        frameon=False,
        fontsize=6.0,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
        columnspacing=0.65,
    )


def _prepare_distance_threshold_representative_city_data(
    wide: pd.DataFrame,
) -> pd.DataFrame:
    """Select representative cities and calculate road-policy retention."""
    indexed = wide.set_index("region_key", drop=False)
    missing = [
        city for city in DISTANCE_THRESHOLD_TYPICAL_CITIES
        if city not in indexed.index
    ]
    if missing:
        raise ValueError(
            "Representative cities missing from distance-threshold results: "
            + ", ".join(missing)
        )

    representative = indexed.loc[list(DISTANCE_THRESHOLD_TYPICAL_CITIES)].copy()
    oracle = representative["oracle_annual_reduction_pct"].to_numpy(dtype=float)
    road = representative["road_full_domain_reduction_pct"].to_numpy(dtype=float)
    if np.any(oracle <= 0.0):
        invalid = representative.loc[
            oracle <= 0.0, "region_key"
        ].astype(str).tolist()
        raise ValueError(
            "Representative-city Oracle reductions must be positive: "
            + ", ".join(invalid)
        )
    retention = road / oracle * 100.0
    if np.any((retention < -1e-6) | (retention > 100.0 + 1e-6)):
        invalid = representative.loc[
            (retention < -1e-6) | (retention > 100.0 + 1e-6),
            "region_key",
        ].astype(str).tolist()
        raise ValueError(
            "Representative-city policy retention must be within 0--100%: "
            + ", ".join(invalid)
        )
    representative["retention_pct"] = np.clip(retention, 0.0, 100.0)
    return representative.reset_index(drop=True)


def _draw_distance_threshold_representative_city_panels(
    axes,
    representative: pd.DataFrame,
) -> None:
    """Draw the three aligned representative-city summary panels."""
    city_colors = representative["macro_region_color"].tolist()
    city_labels = [
        DISTANCE_THRESHOLD_CITY_LABELS[city]
        for city in representative["region_key"]
    ]
    y = np.arange(len(representative), dtype=float)

    for ax in axes:
        for separator in y[:-1] + 0.5:
            ax.axhline(separator, color="#D5DADF", linewidth=0.65, zorder=0)
        ax.set_ylim(len(representative) - 0.5, -0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=7)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)

    road = representative["road_full_domain_reduction_pct"].to_numpy(dtype=float)
    oracle = representative["oracle_annual_reduction_pct"].to_numpy(dtype=float)
    reduction_ax, retention_ax, threshold_ax = axes
    reduction_ax.hlines(y, road, oracle, color="#9AA3AA", linewidth=0.8, zorder=1)
    reduction_ax.scatter(
        road,
        y,
        color=city_colors,
        marker="o",
        s=30,
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    reduction_ax.scatter(
        oracle,
        y,
        color=city_colors,
        marker="D",
        s=28,
        edgecolor="#303030",
        linewidth=0.55,
        zorder=4,
    )
    reduction_upper = max(40.0, np.ceil(float(oracle.max()) / 10.0) * 10.0)
    reduction_ax.set_xlim(0.0, reduction_upper)
    reduction_ax.set_xticks(np.arange(0.0, reduction_upper + 0.1, 10.0))
    reduction_ax.set_yticks(y, labels=city_labels)
    reduction_ax.tick_params(axis="y", labelsize=8.5, pad=4)
    reduction_ax.set_xlabel("Reduction (%)", fontsize=9, loc="left", labelpad=16)
    reduction_ax.xaxis.set_label_position("top")
    reduction_ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="none", color="#777777",
                markersize=5.5, label="Road policy",
            ),
            Line2D(
                [], [], marker="D", linestyle="none", color="#777777",
                markersize=5.5, label="Trip-level optimum",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.17),
        borderaxespad=0.0,
        frameon=False,
        fontsize=7.5,
        handletextpad=0.4,
        columnspacing=1.0,
        ncol=2,
    )
    for y_value, road_value, oracle_value, city_color in zip(
        y, road, oracle, city_colors
    ):
        if oracle_value - road_value >= 0.75:
            reduction_ax.annotate(
                f"{road_value:.1f}",
                (road_value, y_value),
                xytext=(-4, 6),
                textcoords="offset points",
                ha="right",
                color=city_color,
                fontsize=7,
            )
        reduction_ax.annotate(
            f"{oracle_value:.1f}",
            (oracle_value, y_value),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center" if oracle_value - road_value < 0.75 else "left",
            color=city_color,
            fontsize=7,
        )

    retention = representative["retention_pct"].to_numpy(dtype=float)
    retention_ax.barh(y, retention, color=city_colors, height=0.42, zorder=2)
    retention_ax.set_xlim(0.0, 100.0)
    retention_ax.set_xticks(np.arange(0.0, 100.1, 20.0))
    retention_ax.set_yticks([])
    retention_ax.set_xlabel("Retention (%)", fontsize=9, loc="left", labelpad=16)
    retention_ax.xaxis.set_label_position("top")
    for y_value, value in zip(y, retention):
        retention_ax.annotate(
            f"{value:.1f}",
            (value, y_value),
            xytext=(-3, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color="white",
            fontsize=7,
        )

    thresholds = representative["road_optimal_threshold_km"].to_numpy(dtype=float)
    threshold_ax.barh(y, thresholds, color=city_colors, height=0.42, zorder=2)
    threshold_upper = max(2.0, np.ceil(float(thresholds.max()) / 0.5) * 0.5)
    threshold_ax.set_xlim(0.0, threshold_upper)
    threshold_ax.set_xticks(np.arange(0.0, threshold_upper + 0.01, 0.5))
    threshold_ax.set_yticks([])
    threshold_ax.set_xlabel(
        "Road threshold (km)", fontsize=9, loc="left", labelpad=16
    )
    threshold_ax.xaxis.set_label_position("top")
    for y_value, value in zip(y, thresholds):
        threshold_ax.annotate(
            f"{value:.2f}",
            (value, y_value),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color="#303030",
            fontsize=7,
        )


def plot_distance_threshold_representative_city_summary(
    summary_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (175.0 / 25.4, 55.0 / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot reduction, retention, and thresholds for representative cities."""
    wide = _prepare_distance_threshold_all_city_data(summary_df, oracle_df)
    representative = _prepare_distance_threshold_representative_city_data(wide)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": (1.45, 1.0, 1.0), "wspace": 0.30},
    )
    _draw_distance_threshold_representative_city_panels(axes, representative)
    _add_distance_threshold_macro_region_legend(fig, wide)
    fig.subplots_adjust(
        left=0.13,
        right=0.98,
        bottom=0.27,
        top=0.73 if plot_title else 0.79,
    )
    if plot_title:
        fig.suptitle(
            "Representative-city road-distance policies",
            fontsize=10,
            fontweight="bold",
            y=0.98,
        )

    records = representative[
        [
            "region_key",
            "province",
            "macro_region_key",
            "macro_region_label",
            "macro_region_color",
            "road_full_domain_reduction_pct",
            "oracle_annual_reduction_pct",
            "retention_pct",
            "road_optimal_threshold_km",
        ]
    ].to_dict(orient="records")
    return _save_distance_threshold_figure(
        fig,
        save_path,
        for_latex,
        plot_title,
        "Distance-threshold representative-city summary",
        {
            "figure_name": "distance_threshold_representative_city_summary_preview",
            "core_conclusion": (
                "Representative cities retain different shares of the trip-level "
                "optimum under city-specific road-distance policies."
            ),
            "native_size_mm": [175.0, 55.0],
            "retention_axis_pct": [0.0, 100.0],
            "macro_regions": _distance_threshold_macro_region_records(wide),
            "representative_cities": records,
        },
        save_tight=False,
    )


def plot_distance_threshold_city_landscape(
    summary_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (84.0 / 25.4, 78.0 / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the all-city road-threshold landscape.

    Args:
        summary_df: Long-form exact optimum records for all complete cities.
        oracle_df: City results for the trip-level Oracle allocation.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    wide = _prepare_distance_threshold_all_city_data(summary_df, oracle_df)
    fig, ax = plt.subplots(figsize=figsize)
    _plot_distance_threshold_all_city_scatter(
        ax,
        wide,
        "road_optimal_threshold_km",
        "road_annual_reduction_pct",
        *DISTANCE_THRESHOLD_LANDSCAPE_AXIS_LABELS,
        panel_idx=0,
        diagonal=False,
    )
    fig.subplots_adjust(
        left=0.20,
        right=0.97,
        bottom=0.18,
        top=0.79 if plot_title else 0.96,
    )
    if plot_title:
        fig.suptitle(
            "City-specific carbon benefits emerge from road thresholds",
            fontsize=10,
            fontweight="bold",
            y=0.97,
        )
    report = _distance_threshold_all_city_report(summary_df, wide)
    return _save_distance_threshold_figure(
        fig,
        save_path,
        for_latex,
        plot_title,
        "Distance-threshold city landscape",
        {
            "figure_name": "distance_threshold_city_landscape",
            "core_conclusion": (
                "Road-distance thresholds produce heterogeneous carbon benefits "
                "across cities."
            ),
            "native_size_mm": [84.0, 78.0],
            **report,
        },
        save_tight=False,
    )


def plot_distance_threshold_oracle_performance(
    summary_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (84.0 / 25.4, 78.0 / 25.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot road-threshold performance against the trip-level Oracle.

    Args:
        summary_df: Long-form exact optimum records for all complete cities.
        oracle_df: City results for the trip-level Oracle allocation.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    wide = _prepare_distance_threshold_all_city_data(summary_df, oracle_df)
    fig, ax = plt.subplots(figsize=figsize)
    _plot_distance_threshold_all_city_scatter(
        ax,
        wide,
        "road_full_domain_reduction_pct",
        "road_oracle_gap_pp",
        *DISTANCE_THRESHOLD_ORACLE_AXIS_LABELS,
        panel_idx=1,
        diagonal=False,
        zero_baseline=True,
    )
    fig.subplots_adjust(
        left=0.22,
        right=0.97,
        bottom=0.18,
        top=0.79 if plot_title else 0.96,
    )
    if plot_title:
        fig.suptitle(
            "Road-policy benefits retain a limited gap to trip-level performance",
            fontsize=10,
            fontweight="bold",
            y=0.97,
        )
    report = _distance_threshold_all_city_report(
        summary_df,
        wide,
        road_reduction_col="road_full_domain_reduction_pct",
    )
    return _save_distance_threshold_figure(
        fig,
        save_path,
        for_latex,
        plot_title,
        "Distance-threshold Oracle performance",
        {
            "figure_name": "distance_threshold_oracle_performance",
            "core_conclusion": (
                "The remaining trip-level performance gap is shown against "
                "the full-city reduction achieved by each road policy."
            ),
            "native_size_mm": [84.0, 78.0],
            **report,
        },
        save_tight=False,
    )
