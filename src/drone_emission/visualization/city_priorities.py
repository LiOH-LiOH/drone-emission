"""Visualization functions for cross-region correlation and emission analysis."""
from .palette import opacity as paper_alpha

from .prepared_bundle import prepared_plot, prepared_data, read_plot_csv, box_statistics

from .palette import color as paper_color, colormap as paper_cmap

import json
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    ORACLE_REDUCTION_LABEL_CITIES,
    PROVINCE_NAME_EN,
    VARIABLE_LABELS,
)
from drone_emission.analysis.city_attribution import R2_ABSOLUTE_OUTCOME, R2_OUTCOME


WEIGHTED_REDUCTION_COLUMNS = (
    "oracle_min_reduction_vs_ebike_pct",
    "max_lifecycle_reduction_pct",
    "lifecycle_all_drone_reduction_pct",
)
ORACLE_REDUCTION_KG_COL = "oracle_min_reduction_vs_ebike_kg"
ORACLE_REDUCTION_T_COL = "oracle_min_reduction_vs_ebike_t"
ORACLE_REDUCTION_MAP_ONLY_FIGURE = "regions_map_oracle_reduction_rank_map_only"
CITY_REDUCTION_RANK_LABEL_Y = {
    "Chengdu": 44.0,
    "Chongqing": 37.0,
    "Guangzhou": 30.0,
    "Wuhan": 24.0,
    "Shenzhen": 26.0,
    "Beijing": 33.0,
    "Hangzhou": 19.0,
    "Shanghai": 13.0,
}
ORACLE_REDUCTION_LABEL_OFFSETS = {
    "Beijing": (8.0, 8.0),
    "Shanghai": (10.0, -14.0),
    "Guangzhou": (-46.0, -16.0),
    "Hangzhou": (12.0, 10.0),
    "Shenzhen": (12.0, 8.0),
    "Wuhan": (10.0, -14.0),
    "Chongqing": (-54.0, -6.0),
    "Chengdu": (-46.0, 14.0),
}
ORACLE_REDUCTION_LABEL_LAYOUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "layouts",
    "regions_map_oracle_reduction_rank_map_only_labels.json",
)


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


def _first_available_column(df: pd.DataFrame, candidates: tuple[str, ...], purpose: str) -> str:
    """Return the first available candidate column, preferring weighted metrics."""
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    raise KeyError(f"No {purpose} column found; tried {', '.join(candidates)}")


@prepared_data
def load_oracle_reduction_label_layout(
    layout_path: str = None,
) -> dict[str, tuple[float, float]]:
    """Load manual label offsets, falling back to built-in defaults."""
    offsets = {
        city: tuple(value)
        for city, value in ORACLE_REDUCTION_LABEL_OFFSETS.items()
    }
    path = layout_path or ORACLE_REDUCTION_LABEL_LAYOUT_PATH
    if not path or not os.path.exists(path):
        return offsets

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not load Oracle map label layout {path}: {exc}")
        return offsets

    raw_offsets = payload.get("offsets_points", payload)
    if not isinstance(raw_offsets, dict):
        return offsets

    for city, value in raw_offsets.items():
        if city not in offsets or not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            offsets[city] = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            continue
    return offsets


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


def _oracle_reduction_tonnes_column(df: pd.DataFrame) -> str:
    """Add and return the absolute Oracle-vs-e-bike reduction column in tonnes."""
    if "total_ebike_lifecycle" not in df.columns:
        raise KeyError("Missing column for Oracle reduction: total_ebike_lifecycle")
    ebike = pd.to_numeric(df["total_ebike_lifecycle"], errors="coerce")

    if "oracle_min_lifecycle_kg" in df.columns:
        oracle = pd.to_numeric(df["oracle_min_lifecycle_kg"], errors="coerce")
        df[ORACLE_REDUCTION_KG_COL] = ebike - oracle
    else:
        reduction_pct_col = _first_available_column(
            df, WEIGHTED_REDUCTION_COLUMNS, "weighted reduction"
        )
        reduction_pct = pd.to_numeric(df[reduction_pct_col], errors="coerce")
        df[ORACLE_REDUCTION_KG_COL] = ebike * reduction_pct / 100.0

    df[ORACLE_REDUCTION_T_COL] = df[ORACLE_REDUCTION_KG_COL] / 1e3
    return ORACLE_REDUCTION_T_COL


def _build_bubble_reduction_cmap(v_max: float):
    """Return a visible blue cmap for percentage-colored bubbles."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    step = 5 if v_max <= 25 else (10 if v_max <= 80 else 15)
    top = max(int(np.ceil(v_max / step) * step), step * 5)
    reduction_edges = np.arange(0, top + step, step, dtype=float)
    bin_edges = np.concatenate([[-step], reduction_edges])
    base_colors = [
        paper_color("city_priorities.9bd3e6"),
        paper_color("city_priorities.66b7d2"),
        paper_color("city_priorities.2f91b8"),
        paper_color("city_priorities.126782"),
        paper_color("city_priorities.064d6b"),
        paper_color("city_priorities.023047"),
    ]
    if len(reduction_edges) - 1 <= len(base_colors):
        value_colors = base_colors[:len(reduction_edges) - 1]
    else:
        base = paper_cmap("city_priorities.reduction")
        value_colors = [
            base(x)
            for x in np.linspace(0.38, 0.95, len(reduction_edges) - 1)
        ]
    cmap = ListedColormap([paper_color("city_priorities.cccccc"), *value_colors])
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


def _geometry_outer_rings(geometry):
    """Yield outer rings from a GeoJSON Polygon or MultiPolygon geometry."""
    if not geometry:
        return
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return
    if geom_type == "Polygon":
        if coords and coords[0]:
            yield coords[0]
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            if polygon and polygon[0]:
                yield polygon[0]


def _ring_area_and_centroid(ring):
    """Return planar lon-lat ring area magnitude and centroid."""
    pts = np.asarray(ring, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return 0.0, None
    pts = pts[:, :2]
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    signed_area = 0.5 * np.sum(cross)
    if np.isclose(signed_area, 0.0):
        return 0.0, (float(np.nanmean(x[:-1])), float(np.nanmean(y[:-1])))
    cx = np.sum((x[:-1] + x[1:]) * cross) / (6.0 * signed_area)
    cy = np.sum((y[:-1] + y[1:]) * cross) / (6.0 * signed_area)
    return abs(float(signed_area)), (float(cx), float(cy))


def _geometry_largest_ring_centroid(geometry):
    """Return centroid of the largest outer ring in a GeoJSON geometry."""
    best_area = -1.0
    best_centroid = None
    for ring in _geometry_outer_rings(geometry):
        area, centroid = _ring_area_and_centroid(ring)
        if centroid is not None and area > best_area:
            best_area = area
            best_centroid = centroid
    return best_centroid


def _bubble_sizes_from_values(values, size_min=6.0, size_max=130.0):
    """Map positive values to scatter marker areas with square-root scaling."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    sizes = np.full(values.shape, np.nan, dtype=float)
    if positive.size == 0:
        return sizes
    vmax = float(np.nanmax(positive))
    if np.isclose(vmax, 0.0):
        sizes[np.isfinite(values)] = (size_min + size_max) / 2.0
        return sizes
    scaled = np.sqrt(np.clip(values, 0.0, None) / vmax)
    sizes = size_min + scaled * (size_max - size_min)
    sizes[~np.isfinite(values) | (values <= 0)] = np.nan
    return sizes


def _regions_oracle_bubble_df(map_df, reduction_col, reduction_pct_col):
    """Build bubble positions and encodings for study cities on the China map."""
    from drone_emission.visualization.regions import _batch_city_to_adcode, _ensure_all_china_city_areas

    required = ["region_name_cn", reduction_col, reduction_pct_col]
    missing = [col for col in required if col not in map_df.columns]
    if missing:
        raise KeyError(f"Missing bubble map columns: {missing}")

    city_names = map_df["region_name_cn"].dropna().unique().tolist()
    name_to_adcode = _batch_city_to_adcode(city_names)
    geometry_by_adcode = {
        str(adcode): geometry
        for adcode, _name, geometry in _ensure_all_china_city_areas()
    }

    rows = []
    for _, row in map_df.iterrows():
        name_cn = row.get("region_name_cn")
        adcode = name_to_adcode.get(name_cn)
        geometry = geometry_by_adcode.get(adcode)
        centroid = _geometry_largest_ring_centroid(geometry)
        if centroid is None:
            continue
        reduction_t = pd.to_numeric(pd.Series([row.get(reduction_col)]), errors="coerce").iloc[0]
        reduction_pct = pd.to_numeric(pd.Series([row.get(reduction_pct_col)]), errors="coerce").iloc[0]
        if not (np.isfinite(reduction_t) and reduction_t > 0 and np.isfinite(reduction_pct)):
            continue
        rows.append(
            {
                "region_key": row.get("region_key"),
                "region_name_cn": name_cn,
                "region_name_en": row.get("region_name_en"),
                "region_level": row.get("region_level"),
                "province": row.get("province"),
                "adcode": adcode,
                "lng": centroid[0],
                "lat": centroid[1],
                reduction_col: float(reduction_t),
                reduction_pct_col: float(reduction_pct),
            }
        )

    bubble_df = pd.DataFrame(rows)
    if not bubble_df.empty:
        bubble_df["_bubble_size"] = _bubble_sizes_from_values(bubble_df[reduction_col])
    return bubble_df


def _draw_bubble_size_legend(
    ax,
    values,
    size_min=6.0,
    size_max=130.0,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.31),
    fontsize=8,
    title_fontsize=8,
    framealpha=paper_alpha("city_priorities.alpha_0_92"),
    borderpad=0.7,
    labelspacing=1.0,
    handletextpad=1.2,
    ncol=1,
    columnspacing=1.0,
    title="E-bike - Optimal (t CO2eq / yr)",
    quantiles=(0.25, 0.5, 0.9),
):
    """Draw an in-map legend for absolute reduction bubble sizes."""
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    if values.empty:
        return None
    legend_values = np.nanquantile(values, quantiles)
    legend_values = np.unique(np.round(legend_values, 1))
    handles = []
    labels = []
    vmax = float(values.max())
    for value in legend_values:
        if value <= 0:
            continue
        marker_size = size_min + np.sqrt(value / vmax) * (size_max - size_min)
        handles.append(
            ax.scatter(
                [],
                [],
                s=marker_size,
                facecolor=paper_color("city_priorities.bdbdbd"),
                edgecolor=paper_color("city_priorities.ffffff"),
                linewidth=0.65,
                alpha=paper_alpha("city_priorities.alpha_0_82"),
            )
        )
        labels.append(f"{value:,.0f}" if value >= 100 else f"{value:,.1f}")
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=framealpha,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        borderpad=borderpad,
        labelspacing=labelspacing,
        handletextpad=handletextpad,
        ncol=ncol,
        columnspacing=columnspacing,
    )


def _draw_bubble_color_legend(
    ax,
    cmap,
    _bin_edges,
    reduction_edges,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    fontsize=7.2,
    title_fontsize=7.5,
    framealpha=paper_alpha("city_priorities.alpha_0_92"),
    borderpad=0.55,
    labelspacing=0.35,
    handletextpad=0.55,
    include_unstudied=True,
    ncol=1,
    columnspacing=1.0,
    title="Optimal reduction",
):
    """Draw an in-map legend for bubble color bins."""
    handles = []
    labels = []
    if include_unstudied:
        handles.append(Patch(facecolor=paper_color("city_priorities.cccccc"), edgecolor="none", label="Unstudied"))
        labels.append("Unstudied")
    value_edges = list(reduction_edges)
    for i in range(len(value_edges) - 1):
        lo = value_edges[i]
        hi = value_edges[i + 1]
        handles.append(
            Patch(
                facecolor=cmap(i + 1),
                edgecolor=paper_color("ink.0_65"),
                linewidth=0.4,
            )
        )
        labels.append(f"{lo:.0f}-{hi:.0f}%")

    return ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=framealpha,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        borderpad=borderpad,
        labelspacing=labelspacing,
        handlelength=1.1,
        handleheight=0.8,
        handletextpad=handletextpad,
        ncol=ncol,
        columnspacing=columnspacing,
    )


@prepared_plot()
def plot_province_total_ebike_lifecycle(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (7.62, 2.84),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
    reduction_metric: str = "absolute",
):
    """Province bar chart ranked by total annual e-bike emission, with Oracle reduction overlay.

    Bars (left axis) show summed annual-sales-weighted total_ebike_lifecycle
    per province, sorted descending. Boxplot (right axis) shows the per-city
    distribution of absolute or percentage Oracle reduction versus the
    kappa-adjusted e-bike baseline.

    Args:
        df: all_region_stats DataFrame with province, total_ebike_lifecycle,
            and oracle_min_lifecycle_kg.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        ax: External axes. When provided, skips figure creation and save.
        reduction_metric: ``"absolute"`` for annual tonnes of CO2-equivalent
            reduction or ``"percent"`` for reduction relative to e-bike emissions.
    """
    if reduction_metric not in {"absolute", "percent"}:
        raise ValueError(
            "reduction_metric must be either 'absolute' or 'percent', "
            f"got {reduction_metric!r}"
        )

    needed = ["province", "total_ebike_lifecycle", "oracle_min_lifecycle_kg"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        print(f"Province total ebike lifecycle skipped — missing columns {missing}")
        return

    source = df
    if "region_level" in source.columns:
        source = source[source["region_level"] == "city"]
    sub = source[needed].copy()
    sub["total_ebike_lifecycle"] = pd.to_numeric(
        sub["total_ebike_lifecycle"], errors="coerce"
    )
    sub["oracle_min_lifecycle_kg"] = pd.to_numeric(
        sub["oracle_min_lifecycle_kg"], errors="coerce"
    )
    if reduction_metric == "absolute":
        reduction_col = _oracle_reduction_tonnes_column(sub)
        reduction_axis_label = "City optimal reduction\n(t CO$_2$eq / yr)"
        reduction_legend_label = "City optimal reduction (per city)"
    else:
        sub["oracle_min_reduction_vs_ebike_pct"] = np.where(
            sub["total_ebike_lifecycle"] > 0,
            (
                1.0
                - sub["oracle_min_lifecycle_kg"] / sub["total_ebike_lifecycle"]
            )
            * 100.0,
            np.nan,
        )
        reduction_col = "oracle_min_reduction_vs_ebike_pct"
        reduction_axis_label = "City optimal reduction vs e-bike (%)"
        reduction_legend_label = "Optimal reduction vs e-bike (per city)"

    sub = sub.dropna(subset=[*needed, reduction_col])
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
        sub.loc[sub["province_en"] == p, reduction_col].values
        for p in ordered
    ]

    x = np.arange(len(ordered))
    bar_color = paper_color("city_priorities.b0b0b0")
    bar_axis_color = paper_color("city_priorities.4a4a4a")
    box_color = paper_color("city_priorities.1f6e2d")

    _external_ax = ax is not None
    if _external_ax:
        ax1 = ax
        fig = ax1.get_figure()
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    axis_label_size = 12 if _external_ax else 9
    x_tick_size = 9 if _external_ax else 7
    ebike_axis_label = (
        "Kappa-adjusted e-bike\nlifecycle emission\n(t CO$_2$eq / yr)"
        if _external_ax
        else "Annual e-bike lifecycle emissions\n(t CO$_2$eq yr$^{-1}$)"
    )

    ax1.bar(
        x,
        ebike_t,
        color=bar_color,
        alpha=paper_alpha("city_priorities.alpha_0_85"),
        zorder=2,
        label="Kappa-adjusted e-bike emission",
    )
    ax1.set_xlabel("Provincial-level region", fontsize=axis_label_size)
    ax1.set_ylabel(
        ebike_axis_label,
        fontsize=axis_label_size,
        color=bar_axis_color,
    )
    ax1.tick_params(axis="y", labelcolor=bar_axis_color)
    if not _external_ax:
        ax1.tick_params(axis="y", labelsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered, rotation=45, ha="right", fontsize=x_tick_size)
    ax1.yaxis.grid(True, linestyle="--", alpha=paper_alpha("city_priorities.alpha_0_4"))
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    bp = ax2.bxp(
        box_statistics(reduction_data),
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=reduction_metric == "absolute",
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "none",
            "markeredgecolor": box_color,
            "alpha": paper_alpha("city_priorities.alpha_0_75"),
        },
        manage_ticks=False,
        zorder=3,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(box_color)
        patch.set_edgecolor(box_color)
        patch.set_alpha(paper_alpha("city_priorities.alpha_0_35"))
    for whisker in bp["whiskers"]:
        whisker.set_color(box_color)
    for cap in bp["caps"]:
        cap.set_color(box_color)
    for median in bp["medians"]:
        median.set_color(box_color)
        median.set_linewidth(1.5)
    if reduction_metric == "absolute":
        ax2.axhline(0.0, color=box_color, linewidth=0.8, alpha=paper_alpha("city_priorities.alpha_0_55"), zorder=1)
    ax2.set_ylabel(
        reduction_axis_label,
        fontsize=axis_label_size,
        color=box_color,
    )
    ax2.tick_params(axis="y", labelcolor=box_color)
    if not _external_ax:
        ax2.tick_params(axis="y", labelsize=8)
    ax2.set_xlim(ax1.get_xlim())

    if _external_ax:
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=paper_alpha("city_priorities.alpha_0_85")),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=box_color,
                alpha=paper_alpha("city_priorities.alpha_0_35"),
                edgecolor=box_color,
            ),
        ]
        ax1.legend(
            handles,
            [
                "Provincial-level kappa-adjusted e-bike emission",
                reduction_legend_label,
            ],
            loc="upper right",
            fontsize=10,
        )

    if plot_title:
        reduction_title = (
            "absolute optimal reduction"
            if reduction_metric == "absolute"
            else "optimal reduction percentage"
        )
        title = (
            "Provincial-level e-bike emission ranking with city "
            f"{reduction_title}"
        )
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
        province_records = []
        for rank, province in enumerate(ordered, start=1):
            province_rows = sub[sub["province_en"] == province]
            province_records.append(
                {
                    "rank": rank,
                    "province": province,
                    "city_count": int(len(province_rows)),
                    "annual_ebike_lifecycle_t_co2e": float(
                        prov.loc[
                            prov["province_en"] == province,
                            "total_ebike",
                        ].iloc[0]
                        / 1e3
                    ),
                    "city_optimal_reduction": _writing_numeric_summary(
                        province_rows[reduction_col]
                    ),
                }
            )
        write_figure_report(
            out.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": "province_total_ebike_lifecycle",
                "figure_path": out,
                "source_scope": "Successfully processed city-level study regions",
                "province_count": int(len(ordered)),
                "city_count": int(len(sub)),
                "ranking_metric": {
                    "field": "total_ebike_lifecycle",
                    "definition": "annual-sales-weighted kappa-adjusted e-bike lifecycle emissions",
                    "display_unit": "t CO2e per year",
                    "sort_order": "descending",
                },
                "city_reduction_metric": {
                    "field": reduction_col,
                    "mode": reduction_metric,
                    "display_unit": (
                        "t CO2e per year"
                        if reduction_metric == "absolute"
                        else "percent versus e-bike"
                    ),
                },
                "province_records": province_records,
            },
        )
        print(f"Province total ebike lifecycle saved to {out}")
    else:
        plt.show()

    plt.close(fig)
def _write_regions_map_oracle_reduction_rank_report(
    save_path,
    figure_name,
    figsize,
    stats_csv_path,
    map_df,
    reduction_col,
    reduction_pct_col,
    bubble_df,
    ax_map,
    inset_ax,
    south_china_inset_ax,
    cbar_ax,
    cmap,
    norm,
    bin_edges,
    reduction_edges,
    highlighted_city_labels,
    inset_bubble_records,
    inset_bubble_area_scale,
    color_scale_style,
):
    """Write a JSON report for regions_map_oracle_reduction_rank."""
    renderer = ax_map.get_figure().canvas.get_renderer()
    tight_bbox = ax_map.get_figure().get_tightbbox(renderer)
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    payload = {
        "figure_name": figure_name,
        "figure_path": save_path,
        "source_stats_csv_path": stats_csv_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "coordinate_convention": (
            "Plotted x/y values are Matplotlib data coordinates for the stated "
            "axis. Axis bbox values are figure-fraction coordinates."
        ),
        "panels": {
            "oracle_reduction_map": {
                **_bubble_map_panel_report(
                    bubble_df,
                    reduction_col,
                    reduction_pct_col,
                    cmap,
                    norm,
                    bin_edges,
                    ax_map,
                ),
                "map_axis_saved_tight": saved_tight_axis_report(
                    ax_map.get_figure(),
                    ax_map,
                    renderer=renderer,
                    tight_bbox=tight_bbox,
                ),
                "inset_city_rank": (
                    _city_oracle_rank_report(map_df, inset_ax)
                    if inset_ax is not None
                    else None
                ),
                "colorbar": _colorbar_report(
                    cbar_ax,
                    bin_edges,
                    reduction_edges,
                    "Optimal lifecycle emission reduction (%)",
                    cmap,
                )
                if cbar_ax is not None
                else None,
                "color_scale_style": color_scale_style,
                "colorbar_axis_saved_tight": (
                    saved_tight_axis_report(
                        ax_map.get_figure(),
                        cbar_ax,
                        renderer=renderer,
                        tight_bbox=tight_bbox,
                    )
                    if cbar_ax is not None
                    else None
                ),
                "south_china_sea_inset": (
                    {
                        "axis": _axis_report(south_china_inset_ax),
                        "bubble_area_scale": _json_safe_scalar(inset_bubble_area_scale),
                        "bubbles": inset_bubble_records,
                    }
                    if south_china_inset_ax is not None
                    else None
                ),
                "highlighted_city_labels": highlighted_city_labels,
            },
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map oracle reduction rank report saved to {report_path}")


def plot_regions_map_oracle_reduction_rank(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (8.5, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    show_rank_inset: bool = True,
    figure_name: str = "regions_map_oracle_reduction_rank",
    label_leaders_from_bubble_edge: bool = False,
    draw_label_leaders: bool = True,
    draw_south_china_inset_bubbles: bool = False,
    color_scale_style: str = "colorbar",
    label_layout_path: str = None,
    label_offsets: dict = None,
    return_label_artists: bool = False,
):
    """China bubble map of Oracle reduction with top-city rank inset.

    Args:
        df_stats: all_region_stats DataFrame.
        stats_csv_path: Path to all_region_stats.csv (for map function).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        show_rank_inset: If True, show the top-city rank inset.
        figure_name: Figure name used in the JSON report.
        label_leaders_from_bubble_edge: If True, label leader lines start at bubble edge.
        draw_label_leaders: If True, draw lines between labels and city bubbles.
        draw_south_china_inset_bubbles: If True, draw scaled bubbles on the South China Sea inset.
        color_scale_style: "colorbar" for a side colorbar, or "legend" for an in-map legend.
        label_layout_path: Optional JSON file with manual city label offsets.
        label_offsets: Optional in-memory city label offsets, overriding the layout file.
        return_label_artists: If True, return figure and annotation artists without saving.
    """
    from drone_emission.visualization.regions import plot_regions_on_china_map

    map_df = read_plot_csv(stats_csv_path)
    for col in ("oracle_min_lifecycle_kg",):
        if (
            col not in map_df.columns
            and col in df_stats.columns
            and "region_key" in map_df.columns
            and "region_key" in df_stats.columns
        ):
            map_df = map_df.merge(
                df_stats[["region_key", col]],
                on="region_key",
                how="left",
            )
    try:
        reduction_col = _oracle_reduction_tonnes_column(map_df)
        reduction_pct_col = _weighted_reduction_column(map_df)
    except KeyError as exc:
        print(f"Regions map oracle reduction rank skipped — {exc}")
        return

    reduction_values = map_df[reduction_col].dropna()
    if reduction_values.empty:
        print("Regions map oracle reduction rank skipped — no valid reduction values")
        return

    map_df[reduction_pct_col] = pd.to_numeric(
        map_df[reduction_pct_col], errors="coerce"
    )
    reduction_pct_values = map_df[reduction_pct_col].dropna()
    if reduction_pct_values.empty:
        print(
            "Regions map oracle reduction rank skipped — "
            "no valid reduction-percentage values"
        )
        return

    v_max = max(float(reduction_pct_values.max(skipna=True)), 0.0)
    cmap, norm, bin_edges, reduction_edges = _build_bubble_reduction_cmap(v_max)
    bubble_df = _regions_oracle_bubble_df(
        map_df,
        reduction_col=reduction_col,
        reduction_pct_col=reduction_pct_col,
    )
    if bubble_df.empty:
        print("Regions map oracle reduction rank skipped — no valid bubble positions")
        return
    is_map_only_figure = figure_name == ORACLE_REDUCTION_MAP_ONLY_FIGURE
    bubble_area_scale = 1.5 if is_map_only_figure else 1.0
    bubble_df["_bubble_size"] *= bubble_area_scale
    bubble_draw_df = bubble_df.sort_values("_bubble_size", ascending=True)

    if is_map_only_figure:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=(1.0, 0.12),
            hspace=0.0,
        )
        ax_map = fig.add_subplot(grid[0])
        legend_ax = fig.add_subplot(grid[1])
        legend_ax.axis("off")
    else:
        fig, ax_map = plt.subplots(figsize=figsize)
        legend_ax = None
    gray_map_df = map_df.copy()
    gray_map_df["_bubble_gray"] = np.nan
    child_axes_before_map = len(ax_map.child_axes)
    plot_regions_on_china_map(
        df=gray_map_df,
        color_by="_bubble_gray",
        ax=ax_map,
        plot_title=False,
        cmap=cmap,
        norm=norm,
        draw_colorbar=False,
        draw_legend=False,
        south_china_inset_bounds=(
            (0.865, 0.02, 0.12, 0.235)
            if is_map_only_figure
            else (0.86, 0.0, 0.14, 0.28)
        ),
        south_china_inset_extent=(
            (106, 123, 3, 26)
            if is_map_only_figure
            else (107, 121, 3, 24)
        ),
    )
    south_china_inset_ax = (
        ax_map.child_axes[-1]
        if len(ax_map.child_axes) > child_axes_before_map
        else None
    )
    ax_map.scatter(
        bubble_draw_df["lng"],
        bubble_draw_df["lat"],
        s=bubble_draw_df["_bubble_size"],
        c=bubble_draw_df[reduction_pct_col],
        cmap=cmap,
        norm=norm,
        alpha=paper_alpha("city_priorities.alpha_0_82"),
        edgecolor=paper_color("city_priorities.ffffff"),
        linewidth=0.65,
        zorder=8,
    )
    inset_bubble_records = []
    inset_bubble_area_scale = None
    if label_offsets is None:
        if (
            label_layout_path is not None
            or figure_name == ORACLE_REDUCTION_MAP_ONLY_FIGURE
        ):
            label_offsets = load_oracle_reduction_label_layout(label_layout_path)
        else:
            label_offsets = {
                city: tuple(offset)
                for city, offset in ORACLE_REDUCTION_LABEL_OFFSETS.items()
            }
    highlighted_city_labels, highlighted_city_artists = _add_regions_oracle_city_highlights(
        ax_map,
        bubble_df,
        line_from_bubble_edge=label_leaders_from_bubble_edge,
        draw_label_leaders=draw_label_leaders,
        label_offsets=label_offsets,
        return_artists=return_label_artists,
    )
    unified_legend_style = {
        "loc": "lower right",
        "fontsize": 6.5,
        "title_fontsize": 6.5,
        "framealpha": paper_alpha("city_priorities.alpha_0_92"),
        "borderpad": 0.35,
        "labelspacing": 0.2,
        "handletextpad": 0.5,
    }
    size_legend_kwargs = (
        {
            **unified_legend_style,
            "loc": "lower left",
            "bbox_to_anchor": (0.015, 0.015),
            "fontsize": 6.2,
            "title_fontsize": 6.3,
            "framealpha": paper_alpha("city_priorities.alpha_0_78"),
            "borderpad": 0.3,
            "labelspacing": 0.25,
            "handletextpad": 0.45,
            "ncol": 1,
            "columnspacing": 0.8,
            "title": "Annual reduction\n(t CO2e yr$^{-1}$)",
            "quantiles": (0.25, 0.5, 0.75),
        }
        if is_map_only_figure
        else {}
    )
    size_legend = _draw_bubble_size_legend(
        ax_map,
        bubble_df[reduction_col],
        size_min=6.0 * bubble_area_scale,
        size_max=130.0 * bubble_area_scale,
        **size_legend_kwargs,
    )
    if size_legend is not None:
        if is_map_only_figure:
            size_legend.get_frame().set_edgecolor("none")
            size_legend.get_frame().set_linewidth(0.0)
        ax_map.add_artist(size_legend)
    if color_scale_style == "legend":
        color_legend_kwargs = (
            {
                **unified_legend_style,
                "loc": "center",
                "bbox_to_anchor": (0.5, 0.5),
                "fontsize": 6.2,
                "title_fontsize": 6.3,
                "borderpad": 0.0,
                "include_unstudied": False,
                "labelspacing": 0.15,
                "handletextpad": 0.35,
                "ncol": len(reduction_edges) - 1,
                "columnspacing": 0.75,
                "title": "Annual reduction (%)",
            }
            if is_map_only_figure
            else {}
        )
        color_legend = _draw_bubble_color_legend(
            legend_ax if is_map_only_figure else ax_map,
            cmap,
            bin_edges,
            reduction_edges,
            **color_legend_kwargs,
        )
        if is_map_only_figure:
            color_legend.set_frame_on(False)
        (legend_ax if is_map_only_figure else ax_map).add_artist(color_legend)
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

    inset_ax = None
    if show_rank_inset:
        inset_ax = ax_map.inset_axes([0.03, 0.04, 0.34, 0.38], zorder=10)
        _plot_city_oracle_reduction_rank(
            map_df,
            10,
            plot_title=plot_title,
            ax=inset_ax,
            invert_vertical=True,
            compact=True,
        )
        inset_ax.set_facecolor("none")
        inset_ax.patch.set_alpha(paper_alpha("city_priorities.alpha_0_0"))
        for name, spine in inset_ax.spines.items():
            spine.set_visible(name in {"left", "bottom"})
            spine.set_edgecolor(paper_color("city_priorities.666666"))
            spine.set_linewidth(0.6)

    if plot_title:
        ax_map.set_title(
            (
                "Regional Optimal Reduction Bubble Map and Top-City Rank"
                if show_rank_inset
                else "Regional Optimal Reduction Bubble Map"
            ),
            fontsize=13,
            fontweight="bold",
        )

    if is_map_only_figure:
        fig.subplots_adjust(
            left=0.01,
            right=0.995,
            top=0.995,
            bottom=0.01,
            hspace=0.0,
        )
    else:
        fig.tight_layout()
        fig.subplots_adjust(left=0.11)

    if draw_south_china_inset_bubbles and south_china_inset_ax is not None:
        inset_bubble_records, inset_bubble_area_scale = _add_regions_oracle_inset_bubbles(
            ax_map,
            south_china_inset_ax,
            bubble_draw_df,
            reduction_pct_col,
            cmap,
            norm,
        )

    fig.canvas.draw()
    cbar_ax = None
    if color_scale_style != "legend":
        map_pos = ax_map.get_position()
        cbar_ax = fig.add_axes((map_pos.x0 - 0.035, map_pos.y0, 0.014, map_pos.height))
        cbar, n_bands = _draw_discrete_reduction_colorbar(
            fig,
            cbar_ax,
            cmap,
            norm,
            bin_edges,
            reduction_edges,
            "Optimal lifecycle emission reduction (%)",
        )
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if return_label_artists:
        fig.canvas.draw()
        return {
            "fig": fig,
            "ax_map": ax_map,
            "label_artists": highlighted_city_artists,
            "label_records": highlighted_city_labels,
            "label_offsets": label_offsets,
        }

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        _write_regions_map_oracle_reduction_rank_report(
            save_path=save_path,
            figure_name=figure_name,
            figsize=figsize,
            stats_csv_path=stats_csv_path,
            map_df=map_df,
            reduction_col=reduction_col,
            reduction_pct_col=reduction_pct_col,
            bubble_df=bubble_df,
            ax_map=ax_map,
            inset_ax=inset_ax,
            south_china_inset_ax=south_china_inset_ax,
            cbar_ax=cbar_ax,
            cmap=cmap,
            norm=norm,
            bin_edges=bin_edges,
            reduction_edges=reduction_edges,
            highlighted_city_labels=highlighted_city_labels,
            inset_bubble_records=inset_bubble_records,
            inset_bubble_area_scale=inset_bubble_area_scale,
            color_scale_style=color_scale_style,
        )
        print(f"Regions map oracle reduction rank saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return None


@prepared_plot()
def plot_regions_map_oracle_reduction_rank_map_only(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (4.585, 3.43),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """China bubble map of Oracle reduction without the top-city rank inset.

    Args:
        df_stats: all_region_stats DataFrame.
        stats_csv_path: Path to all_region_stats.csv (for map function).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    plot_regions_map_oracle_reduction_rank(
        df_stats=df_stats,
        stats_csv_path=stats_csv_path,
        save_path=save_path,
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
        show_rank_inset=False,
        figure_name="regions_map_oracle_reduction_rank_map_only",
        label_leaders_from_bubble_edge=True,
        draw_label_leaders=False,
        draw_south_china_inset_bubbles=True,
        color_scale_style="legend",
    )


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


def _color_for_value(value, cmap, norm):
    """Return plotted hex color for a mapped scalar value."""
    from matplotlib.colors import to_hex

    value = _json_safe_scalar(value)
    if value is None:
        return paper_color("city_priorities.cccccc")
    return to_hex(cmap(norm(value)))


def _bin_label_for_value(value, bin_edges):
    """Return the interval label used by a discrete color scale."""
    value = _json_safe_scalar(value)
    if value is None:
        return "Unstudied"
    edges = np.asarray(bin_edges, dtype=float)
    idx = int(np.searchsorted(edges, value, side="right") - 1)
    idx = int(np.clip(idx, 0, len(edges) - 2))
    return f"[{edges[idx]:g}, {edges[idx + 1]:g})"


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
            "label": "Unstudied",
            "color": paper_color("city_priorities.cccccc"),
            "bin_index": 0,
        },
    }


def _bubble_map_panel_report(
    bubble_df,
    size_col,
    color_col,
    cmap,
    norm,
    bin_edges,
    ax,
):
    """Return row-level values used by a bubble map panel."""
    rows = []
    for _, row in bubble_df.iterrows():
        color_value = row.get(color_col)
        rows.append(
            {
                "region_key": _json_safe_scalar(row.get("region_key")),
                "region_name_cn": _json_safe_scalar(row.get("region_name_cn")),
                "region_name_en": _json_safe_scalar(row.get("region_name_en")),
                "region_level": _json_safe_scalar(row.get("region_level")),
                "province": _json_safe_scalar(row.get("province")),
                "adcode": _json_safe_scalar(row.get("adcode")),
                "lng": _json_safe_scalar(row.get("lng")),
                "lat": _json_safe_scalar(row.get("lat")),
                size_col: _json_safe_scalar(row.get(size_col)),
                color_col: _json_safe_scalar(color_value),
                "bubble_size": _json_safe_scalar(row.get("_bubble_size")),
                "bubble_color": _color_for_value(color_value, cmap, norm),
                "color_bin": _bin_label_for_value(color_value, bin_edges),
            }
        )
    return {
        "panel_letter": None,
        "axis": _axis_report(ax),
        "mark_type": "bubble",
        "size_by": size_col,
        "size_scale": "sqrt",
        "color_by": color_col,
        "rows": rows,
    }


def _add_regions_oracle_city_highlights(
    ax,
    bubble_df,
    line_from_bubble_edge=False,
    draw_label_leaders=True,
    label_offsets=None,
    return_artists=False,
):
    """Highlight and label selected major cities on the Oracle bubble map."""
    if "region_name_en" not in bubble_df.columns:
        return [], {}

    if label_offsets is None:
        label_offsets = ORACLE_REDUCTION_LABEL_OFFSETS
    records = []
    artists = {}
    name_key = bubble_df["region_name_en"].astype(str).str.casefold()

    for city in ORACLE_REDUCTION_LABEL_CITIES:
        rows = bubble_df[name_key == city.casefold()]
        if rows.empty:
            continue

        row = rows.iloc[0]
        x = float(row["lng"])
        y = float(row["lat"])
        bubble_size = float(row["_bubble_size"])
        highlight_size = bubble_size
        ax.scatter(
            [x],
            [y],
            s=highlight_size,
            marker="o",
            facecolors="none",
            edgecolors=paper_color("city_priorities.111111"),
            linewidths=0.7,
            zorder=18,
            clip_on=False,
        )
        offset = label_offsets.get(city, (8, 8))
        xy = (x, y)
        label_offset = offset
        leader_shrink_points = 0.0
        arrowprops = (
            dict(arrowstyle="-", color=paper_color("ink.0_25"), lw=0.75)
            if draw_label_leaders
            else None
        )
        if line_from_bubble_edge:
            offset_vec = np.array(offset, dtype=float)
            offset_norm = np.linalg.norm(offset_vec)
            if offset_norm > 0:
                radius_points = np.sqrt(highlight_size) / 2.0
                text_gap_points = 6.0
                label_offset_vec = (
                    offset_vec
                    + offset_vec / offset_norm * (radius_points + text_gap_points)
                )
                label_offset = tuple(label_offset_vec)
                leader_shrink_points = radius_points + 1.0
                if draw_label_leaders:
                    arrowprops = dict(
                        arrowstyle="-",
                        color=paper_color("ink.0_25"),
                        lw=0.75,
                        shrinkA=1.5,
                        shrinkB=leader_shrink_points,
                    )
        annotation = ax.annotate(
            city,
            xy,
            xytext=label_offset,
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color=paper_color("city_priorities.111111"),
            arrowprops=arrowprops,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=paper_color("ink.background"),
                edgecolor=paper_color("ink.0_55"),
                alpha=paper_alpha("city_priorities.alpha_0_9"),
            ),
            zorder=20,
            clip_on=False,
        )

        records.append(
            {
                "city": city,
                "region_key": _json_safe_scalar(row.get("region_key")),
                "lng": _json_safe_scalar(x),
                "lat": _json_safe_scalar(y),
                "bubble_size": _json_safe_scalar(bubble_size),
                "highlight_size": _json_safe_scalar(highlight_size),
                "label_offset_points": {
                    "x": _json_safe_scalar(offset[0]),
                    "y": _json_safe_scalar(offset[1]),
                },
                "effective_label_offset_points": {
                    "x": _json_safe_scalar(label_offset[0]),
                    "y": _json_safe_scalar(label_offset[1]),
                },
                "leader_shrink_points": _json_safe_scalar(leader_shrink_points),
            }
        )
        if return_artists:
            artists[city] = {
                "annotation": annotation,
                "highlight_size": highlight_size,
            }

    return records, artists


def _add_regions_oracle_inset_bubbles(
    main_ax,
    inset_ax,
    bubble_df,
    reduction_pct_col,
    cmap,
    norm,
):
    """Draw scaled Oracle bubbles on the South China Sea inset."""
    if main_ax is None or inset_ax is None or bubble_df.empty:
        return [], None

    x_min, x_max = inset_ax.get_xlim()
    y_min, y_max = inset_ax.get_ylim()
    inset_df = bubble_df[
        bubble_df["lng"].between(x_min, x_max)
        & bubble_df["lat"].between(y_min, y_max)
    ].copy()
    if inset_df.empty:
        return [], None

    fig = main_ax.get_figure()
    fig.canvas.draw()

    def _pixels_per_degree(ax):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        center_px = ax.transData.transform(center)
        x_px = ax.transData.transform((center[0] + 1.0, center[1]))
        y_px = ax.transData.transform((center[0], center[1] + 1.0))
        px_per_lng = abs(x_px[0] - center_px[0])
        px_per_lat = abs(y_px[1] - center_px[1])
        return np.sqrt(px_per_lng * px_per_lat)

    main_px_per_degree = _pixels_per_degree(main_ax)
    inset_px_per_degree = _pixels_per_degree(inset_ax)
    if not (
        np.isfinite(main_px_per_degree)
        and np.isfinite(inset_px_per_degree)
        and main_px_per_degree > 0
    ):
        area_scale = 1.0
    else:
        area_scale = (inset_px_per_degree / main_px_per_degree) ** 2

    inset_df = inset_df.sort_values("_bubble_size", ascending=True)
    inset_sizes = np.maximum(inset_df["_bubble_size"] * area_scale, 6.0)
    inset_ax.scatter(
        inset_df["lng"],
        inset_df["lat"],
        s=inset_sizes,
        c=inset_df[reduction_pct_col],
        cmap=cmap,
        norm=norm,
        alpha=paper_alpha("city_priorities.alpha_0_82"),
        edgecolor=paper_color("city_priorities.ffffff"),
        linewidth=0.35,
        zorder=8,
        clip_on=True,
    )

    records = []
    for (_, row), marker_size in zip(inset_df.iterrows(), inset_sizes):
        records.append(
            {
                "region_key": _json_safe_scalar(row.get("region_key")),
                "region_name_en": _json_safe_scalar(row.get("region_name_en")),
                "lng": _json_safe_scalar(row.get("lng")),
                "lat": _json_safe_scalar(row.get("lat")),
                "bubble_size": _json_safe_scalar(row.get("_bubble_size")),
                "inset_bubble_size": _json_safe_scalar(marker_size),
            }
        )
    return records, float(area_scale)


def _city_oracle_rank_report(
    df,
    ax,
    n=10,
    invert_vertical=True,
    compact=True,
    reduction_metric="absolute_t",
):
    """Return plotted bar data for the city Oracle-reduction inset."""
    base_cols = ["region_name_en", "total_ebike_lifecycle"]
    has_oracle = "oracle_min_lifecycle_kg" in df.columns
    fallback_cols = [col for col in WEIGHTED_REDUCTION_COLUMNS if col in df.columns]
    metric_cols = ["oracle_min_lifecycle_kg"] if has_oracle else fallback_cols
    sub = df[base_cols + metric_cols].copy()
    if reduction_metric == "percentage":
        reduction_col = _weighted_reduction_column(sub)
    else:
        reduction_col = _oracle_reduction_tonnes_column(sub)
    sub = sub.dropna(subset=[reduction_col])
    ranked = sub.sort_values(reduction_col, ascending=False).head(n)
    ranked = ranked.sort_values(reduction_col, ascending=True).reset_index(drop=True)
    if invert_vertical:
        ranked = ranked.iloc[::-1].reset_index(drop=True)

    bars = []
    for y, (_, row) in enumerate(ranked.iterrows()):
        bars.append(
            {
                "city": _json_safe_scalar(row.get("region_name_en")),
                "y_center": y,
                "height": 0.6,
                "x_left": 0.0,
                (
                    "width_reduction_pct"
                    if reduction_metric == "percentage"
                    else "width_t_co2e_per_year"
                ): _json_safe_scalar(row[reduction_col]),
                "color": paper_color("city_priorities.1f6e2d"),
                "alpha": paper_alpha("city_priorities.alpha_0_88"),
            }
        )
    return {
        "axis": _axis_report(ax),
        "reduction_col": reduction_col,
        "reduction_metric": reduction_metric,
        "compact": compact,
        "invert_vertical": invert_vertical,
        "bars": bars,
    }


def _plot_city_oracle_reduction_rank(
    df: pd.DataFrame,
    n: int,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
    invert_vertical: bool = False,
    compact: bool = False,
    reduction_metric: str = "absolute_t",
):
    """Shared horizontal bar plot for top cities by Oracle reduction."""
    base_cols = ["region_name_en", "total_ebike_lifecycle"]
    missing = [col for col in base_cols if col not in df.columns]
    has_oracle = "oracle_min_lifecycle_kg" in df.columns
    fallback_cols = [col for col in WEIGHTED_REDUCTION_COLUMNS if col in df.columns]
    if missing or (not has_oracle and not fallback_cols):
        needed = base_cols + ["oracle_min_lifecycle_kg or weighted reduction column"]
        print(f"City Oracle reduction rank skipped — missing columns {needed}")
        return

    metric_cols = ["oracle_min_lifecycle_kg"] if has_oracle else fallback_cols
    sub = df[base_cols + metric_cols].copy()
    try:
        if reduction_metric == "percentage":
            reduction_col = _weighted_reduction_column(sub)
        else:
            reduction_col = _oracle_reduction_tonnes_column(sub)
    except KeyError as exc:
        print(f"City Oracle reduction rank skipped — {exc}")
        return
    sub = sub.dropna(subset=[reduction_col])
    if sub.empty:
        print("City Oracle reduction rank skipped — no rows with e-bike + Oracle emissions")
        return

    ranked = sub.sort_values(reduction_col, ascending=False).head(n)
    ranked = ranked.sort_values(reduction_col, ascending=True).reset_index(drop=True)
    if invert_vertical:
        ranked = ranked.iloc[::-1].reset_index(drop=True)

    cities = ranked["region_name_en"].tolist()
    reductions = ranked[reduction_col].to_numpy(dtype=float)
    y = np.arange(len(cities))
    bar_color = paper_color("city_priorities.1f6e2d")

    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, reductions, height=0.6, color=bar_color, alpha=paper_alpha("city_priorities.alpha_0_88"), zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=7 if compact else 10)
    xlabel = (
        "Optimal reduction (%)"
        if reduction_metric == "percentage"
        else "E-bike - Optimal (t CO$_2$eq / yr)"
    )
    ax.set_xlabel(xlabel, fontsize=7 if compact else 12)
    if compact:
        ax.grid(False)
    else:
        ax.xaxis.grid(True, linestyle="--", alpha=paper_alpha("city_priorities.alpha_0_4"))
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7 if compact else 10, pad=1 if compact else 3)
    ax.tick_params(axis="y", pad=1 if compact else 3)

    xmax = max(float(np.nanmax(reductions)), 1.0)
    ax.set_xlim(0, xmax * (1.04 if compact else 1.18))
    if not compact:
        pad = xmax * 0.01
        for i, value in enumerate(reductions):
            if reduction_metric == "percentage":
                label = f"{value:.1f}%"
            else:
                fmt = "{:,.2f}" if value < 10 else ("{:,.1f}" if value < 100 else "{:,.0f}")
                label = fmt.format(value)
            ax.text(
                value + pad, i, label,
                va="center", ha="left", fontsize=9, color=paper_color("ink.black"), zorder=4,
            )

    if plot_title:
        title = (
            f"Top {n} Cities by Optimal Reduction Percentage"
            if reduction_metric == "percentage"
            else f"Top {n} Cities by E-bike - Optimal Lifecycle Emission"
        )
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
        print(f"City Oracle reduction rank saved to {out}")
    else:
        plt.show()

    plt.close(fig)


@prepared_plot()
def plot_city_reduction_rank_all_macro_region(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (6.86, 2.28),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Rank all cities by Oracle reduction percentage, colored by macro-region.

    Args:
        df: all_region_stats DataFrame with city rows, province, and Oracle reduction metrics.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
    """
    needed = {"region_name_en", "province"}
    missing = sorted(needed - set(df.columns))
    if missing:
        print(f"All-city reduction rank skipped — missing columns {missing}")
        return

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"].copy()

    try:
        reduction_col = _weighted_reduction_column(plot_df)
    except KeyError as exc:
        print(f"All-city reduction rank skipped — {exc}")
        return

    plot_df[reduction_col] = pd.to_numeric(plot_df[reduction_col], errors="coerce")
    plot_df["macro_region"] = plot_df["province"].map(macro_region_for_province)
    plot_df = plot_df.dropna(subset=["region_name_en", "province", reduction_col])
    plot_df = plot_df[plot_df["macro_region"].notna()].copy()
    if plot_df.empty:
        print("All-city reduction rank skipped — no city rows with reduction and macro-region")
        return

    plot_df = plot_df.sort_values(
        [reduction_col, "region_name_en"],
        ascending=[False, True],
    ).reset_index(drop=True)
    plot_df["rank"] = np.arange(1, len(plot_df) + 1)
    plot_df["macro_region_key"] = plot_df["macro_region"].map(lambda spec: spec["key"])
    label_city_by_key = {
        city.casefold(): city
        for city in ORACLE_REDUCTION_LABEL_CITIES
    }
    selected_city_mask = (
        plot_df["region_name_en"]
        .astype(str)
        .str.casefold()
        .isin(label_city_by_key.keys())
    )
    selected_cities = plot_df[selected_city_mask].copy()
    selected_cities["_label_city"] = (
        selected_cities["region_name_en"]
        .astype(str)
        .str.casefold()
        .map(label_city_by_key)
    )

    color_by_key = {spec["key"]: spec["color"] for spec in CHINA_MACRO_REGION_SPECS}
    label_by_key = {spec["key"]: spec["label"] for spec in CHINA_MACRO_REGION_SPECS}
    colors = plot_df["macro_region_key"].map(color_by_key).tolist()

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        plot_df["rank"].to_numpy(dtype=float),
        plot_df[reduction_col].to_numpy(dtype=float),
        width=0.9,
        color=colors,
        alpha=paper_alpha("city_priorities.alpha_0_9"),
        linewidth=0,
        zorder=2,
    )

    ax.set_xlabel("City rank", fontsize=7)
    ax.set_ylabel("Optimal reduction vs e-bike (pp)", fontsize=7)
    ax.yaxis.grid(True, linestyle="--", alpha=paper_alpha("city_priorities.alpha_0_35"))
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, len(plot_df) + 0.5)
    ymax = max(0.0, float(plot_df[reduction_col].max()))
    y_upper = max(10.0, float(np.ceil(ymax / 10.0) * 10.0))
    ax.set_ylim(0.0, y_upper)
    ax.set_yticks(np.arange(0.0, y_upper + 1.0, 10.0))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)

    handles = []
    for spec in CHINA_MACRO_REGION_SPECS:
        count = int((plot_df["macro_region_key"] == spec["key"]).sum())
        if count:
            handles.append(
                Patch(
                    facecolor=color_by_key[spec["key"]],
                    edgecolor="none",
                    label=f"{label_by_key[spec['key']]} ({count})",
                )
            )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.07),
        ncol=len(handles),
        frameon=False,
        fontsize=6.5,
        handlelength=0.9,
        handletextpad=0.35,
        labelspacing=0.25,
        columnspacing=0.65,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.32,
        top=0.80 if plot_title else 0.97,
    )

    if not selected_cities.empty:
        selected_indices = set(selected_cities.index)
        for idx, bar in enumerate(bars):
            if idx not in selected_indices:
                continue
            bar.set_edgecolor((0.53, 0.53, 0.53, 0.4))
            bar.set_linewidth(0.35)
            bar.set_zorder(3)

        selected_cities = selected_cities.sort_values("rank")
        label_box_style = {
            "boxstyle": "round,pad=0.18",
            "facecolor": paper_color("ink.background"),
            "edgecolor": "0.55",
            "alpha": paper_alpha("city_priorities.alpha_1_0"),
        }

        label_records = []
        for _, row in selected_cities.iterrows():
            city = row["_label_city"]
            rank = float(row["rank"])
            bar_top = float(row[reduction_col])
            label_y = max(CITY_REDUCTION_RANK_LABEL_Y[city], bar_top + 2.0)
            text = ax.text(
                rank,
                label_y,
                city,
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color=paper_color("city_priorities.111111"),
                bbox=label_box_style,
                zorder=6,
            )
            label_records.append(
                {
                    "city": city,
                    "rank": rank,
                    "bar_top": bar_top,
                    "label_y": label_y,
                    "text": text,
                }
            )

        # Keep labels centered on their bars and resolve any future data-driven
        # collisions by moving only the later-ranked label upward.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_boxes = []
        for record in label_records:
            text = record["text"]
            while True:
                patch = text.get_bbox_patch()
                bbox = patch.get_window_extent(renderer)
                axes_bbox = ax.get_window_extent(renderer)
                if bbox.y1 > axes_bbox.y1:
                    raise ValueError(
                        "City rank label layout exceeds the y-axis for "
                        f"{record['city']}; update CITY_REDUCTION_RANK_LABEL_Y"
                    )
                if not any(bbox.overlaps(other) for other in placed_boxes):
                    placed_boxes.append(bbox)
                    break
                record["label_y"] += 1.0
                text.set_y(record["label_y"])
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()

        for record in label_records:
            ax.vlines(
                record["rank"],
                record["bar_top"],
                record["label_y"],
                colors=paper_color("ink.0_25"),
                linewidth=0.6,
                zorder=4,
            )

    if plot_title:
        fig.suptitle(
            "All Cities Ranked by Optimal Reduction Percentage",
            fontsize=14,
            fontweight="bold",
        )

    if save_path:
        ext = ".pdf" if for_latex else ".png"
        out = save_path + ext
        if plot_title:
            out = _add_title_suffix(out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        record_columns = [
            "rank",
            "region_name_en",
            "province",
            "macro_region_key",
            reduction_col,
        ]

        def rank_records(rows):
            return [
                {
                    "rank": int(row["rank"]),
                    "city": row["region_name_en"],
                    "province": row["province"],
                    "macro_region": row["macro_region_key"],
                    "optimal_reduction_percentage_points": float(
                        row[reduction_col]
                    ),
                }
                for _, row in rows[record_columns].iterrows()
            ]

        labeled_rows = plot_df[selected_city_mask].sort_values("rank")
        macro_region_records = []
        for spec in CHINA_MACRO_REGION_SPECS:
            rows = plot_df[plot_df["macro_region_key"] == spec["key"]]
            if rows.empty:
                continue
            macro_region_records.append(
                {
                    "macro_region": spec["key"],
                    "label": spec["label"],
                    "city_count": int(len(rows)),
                    "optimal_reduction_percentage_points": (
                        _writing_numeric_summary(rows[reduction_col])
                    ),
                }
            )
        write_figure_report(
            out.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": "city_reduction_rank_all_macro_region",
                "figure_path": out,
                "source_scope": "Successfully processed city-level study regions with known macro-region",
                "city_count": int(len(plot_df)),
                "ranking_metric": {
                    "field": reduction_col,
                    "definition": "optimal lifecycle reduction versus the kappa-adjusted e-bike baseline",
                    "unit": "percentage points",
                    "sort_order": "descending",
                },
                "distribution": _writing_numeric_summary(plot_df[reduction_col]),
                "top_10": rank_records(plot_df.head(10)),
                "bottom_10": rank_records(plot_df.tail(10)),
                "labeled_cities": rank_records(labeled_rows),
                "macro_regions": macro_region_records,
            },
        )
        print(f"All-city reduction rank by macro-region saved to {out}")
    else:
        plt.show()

    plt.close(fig)
