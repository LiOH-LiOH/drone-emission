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
        "beijing": (-7, 9),
        "shanghai": (7, -14),
    },
}


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
        "annual_reduction_pct",
        "drone_order_ratio_pct",
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

    if "region_key" not in oracle_df.columns:
        raise KeyError("Missing Oracle city key: region_key")
    oracle = oracle_df.copy()
    if "region_level" in oracle.columns:
        oracle = oracle[oracle["region_level"] == "city"].copy()
    if oracle.duplicated("region_key").any():
        raise ValueError("Duplicate city rows in Oracle results")
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
    oracle = oracle[["region_key", "oracle_annual_reduction_pct"]]

    identity = summary_df[
        ["region_key", "region_name_en"]
    ].drop_duplicates("region_key")
    metric_columns = (
        "optimal_threshold_km",
        "annual_reduction_pct",
        "drone_order_ratio_pct",
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
    numeric = [
        f"{basis}_{column}"
        for basis in ("road", "straight")
        for column in metric_columns
    ] + ["oracle_annual_reduction_pct"]
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
):
    """Draw one all-city threshold scatter panel on an existing axis."""
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap("Blues")
    norm = Normalize(vmin=0, vmax=100)
    highlight = wide["region_key"].isin(DISTANCE_THRESHOLD_TYPICAL_CITIES)
    color_values = wide["road_drone_order_ratio_pct"]
    ax.scatter(
        wide.loc[~highlight, x_col],
        wide.loc[~highlight, y_col],
        c=color_values[~highlight],
        cmap=cmap,
        norm=norm,
        s=18,
        alpha=0.62,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.scatter(
        wide.loc[highlight, x_col],
        wide.loc[highlight, y_col],
        c=color_values[highlight],
        cmap=cmap,
        norm=norm,
        s=46,
        alpha=1.0,
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
    return norm, cmap


def _distance_threshold_all_city_report(
    summary_df: pd.DataFrame,
    wide: pd.DataFrame,
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
    return {
        "city_count": city_count,
        "configured_city_count": configured_count,
        "highlighted_cities": list(DISTANCE_THRESHOLD_TYPICAL_CITIES),
        "road_threshold_km": _writing_numeric_summary(
            wide["road_optimal_threshold_km"]
        ),
        "road_reduction_pct": _writing_numeric_summary(
            wide["road_annual_reduction_pct"]
        ),
        "oracle_reduction_pct": _writing_numeric_summary(
            wide["oracle_annual_reduction_pct"]
        ),
        "road_drone_order_share_pct": _writing_numeric_summary(
            wide["road_drone_order_ratio_pct"]
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
        "Road threshold (km)",
        "Maximum annual lifecycle reduction (%)",
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
    figsize: tuple = (92.0 / 25.4, 78.0 / 25.4),
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
    norm, cmap = _plot_distance_threshold_all_city_scatter(
        ax,
        wide,
        "oracle_annual_reduction_pct",
        "road_annual_reduction_pct",
        "Trip-level optimal reduction (%)\n"
        "(drone for every lower-emission trip)",
        "Road-policy reduction (%)",
        panel_idx=1,
        diagonal=True,
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    fig.subplots_adjust(
        left=0.17,
        right=0.78,
        bottom=0.18,
        top=0.79 if plot_title else 0.96,
    )
    colorbar_top = 0.79 if plot_title else 0.96
    colorbar_ax = fig.add_axes([0.84, 0.18, 0.025, colorbar_top - 0.18])
    colorbar = fig.colorbar(
        scalar,
        cax=colorbar_ax,
    )
    colorbar.set_label("Road-policy drone order share (%)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    if plot_title:
        fig.suptitle(
            "Road thresholds approach but remain below trip-level performance",
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
        "Distance-threshold Oracle performance",
        {
            "figure_name": "distance_threshold_oracle_performance",
            "core_conclusion": (
                "Road-distance threshold performance remains below the "
                "trip-level Oracle allocation."
            ),
            "native_size_mm": [92.0, 78.0],
            **report,
        },
        save_tight=False,
    )
