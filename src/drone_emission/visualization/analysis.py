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

from visualization._utils import (
    _add_title_suffix,
    saved_tight_axis_report,
    write_figure_report,
)
from visualization.macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    macro_region_for_province,
)

# Import from constants module (no circular dependency)
from analysis.constants import (
    DRONE_SPATIAL_SCATTER_LABELED_CITIES,
    PROVINCE_NAME_EN,
    VARIABLE_LABELS,
)
from analysis.city_attribution import R2_ABSOLUTE_OUTCOME, R2_OUTCOME


WEIGHTED_REDUCTION_COLUMNS = (
    "oracle_min_reduction_vs_ebike_pct",
    "max_lifecycle_reduction_pct",
    "lifecycle_all_drone_reduction_pct",
)
WEIGHTED_DRONE_SHARE_COLUMNS = (
    "oracle_drone_order_ratio_pct",
    "optimal_drone_ratio_pct",
)
ORACLE_REDUCTION_KG_COL = "oracle_min_reduction_vs_ebike_kg"
ORACLE_REDUCTION_T_COL = "oracle_min_reduction_vs_ebike_t"
PATHWAY_SCENARIOS_CACHE_VERSION = 1
ORACLE_REDUCTION_MAP_ONLY_FIGURE = "regions_map_oracle_reduction_rank_map_only"
ORACLE_REDUCTION_LABEL_CITIES = (
    "Beijing",
    "Shanghai",
    "Guangzhou",
    "Hangzhou",
    "Shenzhen",
    "Wuhan",
    "Chongqing",
    "Chengdu",
)
ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR = "#D55E00"
PILOT_SUPER_LARGE_CITIES = (
    "Beijing",
    "Shanghai",
    "Shenzhen",
    "Chongqing",
    "Guangzhou",
    "Chengdu",
    "Tianjin",
)
PILOT_ABSOLUTE_REDUCTION_COL = "oracle_absolute_reduction_t_co2e"
PILOT_RELATIVE_REDUCTION_COL = "oracle_relative_reduction_pct"
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


def save_oracle_reduction_label_layout(offsets: dict, layout_path: str = None):
    """Save manual label offsets for the map-only Oracle reduction figure."""
    path = layout_path or ORACLE_REDUCTION_LABEL_LAYOUT_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "figure": ORACLE_REDUCTION_MAP_ONLY_FIGURE,
        "units": "offset points from each city bubble center",
        "offsets_points": {
            city: [
                float(offsets.get(city, ORACLE_REDUCTION_LABEL_OFFSETS[city])[0]),
                float(offsets.get(city, ORACLE_REDUCTION_LABEL_OFFSETS[city])[1]),
            ]
            for city in ORACLE_REDUCTION_LABEL_CITIES
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


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


def _weighted_drone_share_column(df: pd.DataFrame) -> str:
    """Return the preferred weighted drone-share metric column for city stats."""
    return _first_available_column(df, WEIGHTED_DRONE_SHARE_COLUMNS, "weighted drone share")


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


def _build_drone_share_cmap():
    """Return a 0-100 discrete white-to-blue cmap for drone share."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    drone_share_edges = np.arange(0, 101, 20, dtype=float)
    bin_edges = np.concatenate([[-20.0], drone_share_edges])
    cmap = ListedColormap([
        "#cccccc",
        "#eef5fc",
        "#d6e6f5",
        "#91bfdb",
        "#4292c6",
        "#08519c",
        "#08306b",
    ])
    norm = BoundaryNorm(bin_edges, ncolors=cmap.N, clip=True)
    return cmap, norm, bin_edges, drone_share_edges


def _build_blue_reduction_cmap(v_max: float):
    """Return reduction-percentage bins using the existing blue map style."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    step = 5 if v_max <= 25 else (10 if v_max <= 80 else 15)
    top = max(int(np.ceil(v_max / step) * step), step * 5)
    reduction_edges = np.arange(0, top + step, step, dtype=float)
    bin_edges = np.concatenate([[-step], reduction_edges])
    base_colors = [
        "#eef5fc",
        "#d6e6f5",
        "#91bfdb",
        "#4292c6",
        "#08519c",
        "#08306b",
    ]
    if len(reduction_edges) - 1 <= len(base_colors):
        value_colors = base_colors[:len(reduction_edges) - 1]
    else:
        base = plt.get_cmap("Blues")
        value_colors = [
            base(x)
            for x in np.linspace(0.15, 0.95, len(reduction_edges) - 1)
        ]
    cmap = ListedColormap(["#cccccc", *value_colors])
    norm = BoundaryNorm(bin_edges, ncolors=cmap.N, clip=True)
    return cmap, norm, bin_edges, reduction_edges


def _build_bubble_reduction_cmap(v_max: float):
    """Return a visible blue cmap for percentage-colored bubbles."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    step = 5 if v_max <= 25 else (10 if v_max <= 80 else 15)
    top = max(int(np.ceil(v_max / step) * step), step * 5)
    reduction_edges = np.arange(0, top + step, step, dtype=float)
    bin_edges = np.concatenate([[-step], reduction_edges])
    base_colors = [
        "#9BD3E6",
        "#66B7D2",
        "#2F91B8",
        "#126782",
        "#064D6B",
        "#023047",
    ]
    if len(reduction_edges) - 1 <= len(base_colors):
        value_colors = base_colors[:len(reduction_edges) - 1]
    else:
        base = plt.get_cmap("PuBuGn")
        value_colors = [
            base(x)
            for x in np.linspace(0.38, 0.95, len(reduction_edges) - 1)
        ]
    cmap = ListedColormap(["#cccccc", *value_colors])
    norm = BoundaryNorm(bin_edges, ncolors=cmap.N, clip=True)
    return cmap, norm, bin_edges, reduction_edges


def _finer_percentage_bins(reference_edges: np.ndarray, subdivisions: int = 2):
    """Return finer bins spanning the same range as a percentage color scale."""
    reference_edges = np.asarray(reference_edges, dtype=float)
    if len(reference_edges) < 2:
        return reference_edges
    base_step = float(np.diff(reference_edges).min())
    step = base_step / max(int(subdivisions), 1)
    return np.arange(reference_edges[0], reference_edges[-1] + step * 0.5, step)


def _build_absolute_reduction_log_cmap(values):
    """Discrete sequential cmap with log-spaced absolute reduction bins."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    values = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size:
        v_min = float(positive.min())
        v_max = float(positive.max())
    else:
        v_min, v_max = 1.0, 10.0
    if np.isclose(v_min, v_max):
        v_min = max(v_min / 10.0, np.finfo(float).tiny)
        v_max = v_max * 10.0

    min_decade = int(np.floor(np.log10(v_min)))
    max_decade = int(np.ceil(np.log10(v_max)))
    nice_edges = [
        mult * (10 ** decade)
        for decade in range(min_decade, max_decade + 1)
        for mult in (1, 2, 5)
    ]
    value_edges = np.array(
        [edge for edge in nice_edges if edge >= v_min and edge <= v_max],
        dtype=float,
    )
    lower_edge = 10 ** min_decade
    upper_edge = 10 ** max_decade
    value_edges = np.unique(np.concatenate([[lower_edge], value_edges, [upper_edge]]))
    if len(value_edges) > 7:
        value_edges = np.array([10 ** decade for decade in range(min_decade, max_decade + 1)], dtype=float)
    value_edges = np.concatenate([[0.0], value_edges])
    bin_edges = np.concatenate([[-v_min], value_edges])

    base = plt.get_cmap("YlGn")
    value_colors = [base(x) for x in np.linspace(0.25, 0.9, len(value_edges) - 1)]
    cmap = ListedColormap(["#cccccc", *value_colors])
    norm = BoundaryNorm(bin_edges, ncolors=cmap.N, clip=True)
    return cmap, norm, bin_edges, value_edges


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
    from visualization.regions import _batch_city_to_adcode, _ensure_all_china_city_areas

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
    framealpha=0.92,
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
                facecolor="#BDBDBD",
                edgecolor="#FFFFFF",
                linewidth=0.65,
                alpha=0.82,
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
    framealpha=0.92,
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
        handles.append(Patch(facecolor="#cccccc", edgecolor="none", label="Unstudied"))
        labels.append("Unstudied")
    value_edges = list(reduction_edges)
    for i in range(len(value_edges) - 1):
        lo = value_edges[i]
        hi = value_edges[i + 1]
        handles.append(
            Patch(
                facecolor=cmap(i + 1),
                edgecolor="0.65",
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


def _named_baseline_kappa(named: pd.DataFrame) -> float:
    """Infer the configured baseline kappa from a named-scenario table."""
    kappas = np.array(sorted(named["kappa"].dropna().unique()), dtype=float)
    if kappas.size == 0:
        return 1.0
    non_comparison = kappas[~np.isclose(kappas, 0.65)]
    if non_comparison.size:
        return float(non_comparison.max())
    return float(kappas.max())


def _baseline_kappa_for_pathway(named: pd.DataFrame, stats_df: pd.DataFrame) -> float:
    """Infer the pathway baseline kappa, preferring weighted city stats."""
    named_kappas = np.array(sorted(named["kappa"].dropna().unique()), dtype=float)
    if "ebike_kappa" in stats_df.columns:
        stats_kappas = pd.to_numeric(stats_df["ebike_kappa"], errors="coerce").dropna()
        if not stats_kappas.empty:
            stats_kappa = float(stats_kappas.median())
            if named_kappas.size and np.any(np.isclose(named_kappas, stats_kappa)):
                return stats_kappa
            fallback = _named_baseline_kappa(named)
            print(
                "plot_regions_map_pathway_scenarios: named scenarios do not "
                f"include weighted stats kappa={stats_kappa:.6g}; using "
                f"kappa={fallback:.6g}. Re-run named scenarios for the updated baseline."
            )
            return fallback
    return _named_baseline_kappa(named)


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


def _pathway_anchor_annual_kg(
    named: pd.DataFrame,
    stats_df: pd.DataFrame,
    baseline_kappa: float,
) -> pd.DataFrame:
    """Return region-level all-ebike annual lifecycle anchor in kg CO2e."""
    if {"region_key", "total_ebike_lifecycle"}.issubset(stats_df.columns):
        stats_anchor = stats_df[["region_key", "total_ebike_lifecycle"]].copy()
        stats_anchor["anchor_annual_kg"] = pd.to_numeric(
            stats_anchor["total_ebike_lifecycle"], errors="coerce"
        )
        if "ebike_kappa" in stats_df.columns:
            stats_anchor["ebike_kappa"] = pd.to_numeric(
                stats_df["ebike_kappa"], errors="coerce"
            )
            stats_anchor = stats_anchor[
                stats_anchor["ebike_kappa"].isna()
                | np.isclose(stats_anchor["ebike_kappa"], baseline_kappa)
            ]
        stats_anchor = stats_anchor[["region_key", "anchor_annual_kg"]].dropna()
        if not stats_anchor.empty:
            return stats_anchor

    if "annual_lifecycle_kg" not in named.columns:
        return pd.DataFrame(columns=["region_key", "anchor_annual_kg"])

    return named[
        np.isclose(named["kappa"], baseline_kappa)
        & np.isclose(named["gef_mult"], 1.0)
        & (named["strategy"] == "all_ebike")
    ][["region_key", "annual_lifecycle_kg"]].rename(
        columns={"annual_lifecycle_kg": "anchor_annual_kg"}
    )


def _sample_od_for_pathway_curve(
    od_df: pd.DataFrame,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, float]:
    """Return sampled OD rows and the annual-sales expansion factor."""
    required = {"road_distance_m", "distance_km", "annual_sales"}
    missing = required - set(od_df.columns)
    if missing:
        raise KeyError(f"missing OD columns {sorted(missing)}")

    valid = od_df[list(required)].copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid[
        (valid["road_distance_m"] >= 0)
        & (valid["distance_km"] > 0)
        & (valid["annual_sales"] > 0)
    ].dropna()
    if valid.empty:
        return valid, 1.0

    total_sales = float(valid["annual_sales"].sum())
    if len(valid) > max_rows:
        valid = valid.sample(n=max_rows, random_state=random_state)
    sample_sales = float(valid["annual_sales"].sum())
    scale = total_sales / sample_sales if sample_sales > 0 else 1.0
    return valid, scale


def _evaluate_pathway_optimal_curve_point(
    od_df: pd.DataFrame,
    scale: float,
    ebike_params,
    drone_params,
    province_gef_kwh: float,
    kappa: float,
    gef_mult: float,
    rider_weight: float,
    payload_kg: float,
    anchor_kappa: float = None,
    anchor_gef_mult: float = 1.0,
) -> dict:
    """Evaluate sampled annual CO2 savings and drone share for one pathway point."""
    from dataclasses import replace

    from sensitivity.vectorized_lca import (
        DroneScenario,
        EbikeScenario,
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    road_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
    air_km = od_df["distance_km"].to_numpy(dtype=float)
    annual_sales = od_df["annual_sales"].to_numpy(dtype=float)

    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)
    ebike_g = compute_ebike_per_order_g(
        road_km,
        ebike_pp,
        EbikeScenario(
            kappa=kappa,
            gef_mult=gef_mult,
            payload_kg=payload_kg,
            rider_weight=rider_weight,
        ),
    )
    drone_g = compute_drone_per_order_g(
        air_km,
        drone_pp,
        DroneScenario(gef_mult=gef_mult, payload_kg=payload_kg),
    )
    anchor_ebike_g = ebike_g
    if anchor_kappa is not None or not np.isclose(anchor_gef_mult, gef_mult):
        anchor_ebike_g = compute_ebike_per_order_g(
            road_km,
            ebike_pp,
            EbikeScenario(
                kappa=kappa if anchor_kappa is None else anchor_kappa,
                gef_mult=anchor_gef_mult,
                payload_kg=payload_kg,
                rider_weight=rider_weight,
            ),
        )

    use_drone = drone_g < ebike_g
    oracle_total_g = float((np.where(use_drone, drone_g, ebike_g) * annual_sales).sum())

    anchor_g = float((anchor_ebike_g * annual_sales).sum())
    drone_orders = float(annual_sales[use_drone].sum())
    total_orders = float(annual_sales.sum())
    return {
        "saved_kg": (anchor_g - oracle_total_g) * scale / 1000.0,
        "drone_share_pct": drone_orders / total_orders * 100.0 if total_orders > 0 else np.nan,
    }


def _evaluate_pathway_oracle_city(
    od_df: pd.DataFrame,
    ebike_params,
    drone_params,
    province_gef_kwh: float,
    kappa: float,
    gef_mult: float,
    rider_weight: float,
    payload_kg: float,
    anchor_kappa: float,
    anchor_gef_mult: float = 1.0,
) -> dict:
    """Evaluate a city-level per-OD oracle-min allocation for one pathway scenario."""
    from dataclasses import replace

    from sensitivity.vectorized_lca import (
        DroneScenario,
        EbikeScenario,
        compute_drone_per_order_g,
        compute_ebike_per_order_g,
    )

    required = {"road_distance_m", "distance_km", "annual_sales"}
    missing = required - set(od_df.columns)
    if missing:
        raise KeyError(f"missing OD columns {sorted(missing)}")

    valid = od_df[list(required)].copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid[
        (valid["road_distance_m"] >= 0)
        & (valid["distance_km"] > 0)
        & (valid["annual_sales"] > 0)
    ].dropna()
    if valid.empty:
        raise ValueError("no valid OD rows with positive annual sales")

    road_km = valid["road_distance_m"].to_numpy(dtype=float) / 1000.0
    air_km = valid["distance_km"].to_numpy(dtype=float)
    annual_sales = valid["annual_sales"].to_numpy(dtype=float)

    ebike_pp = replace(ebike_params, e_use=province_gef_kwh)
    drone_pp = replace(drone_params, e_use=province_gef_kwh)
    ebike_g = compute_ebike_per_order_g(
        road_km,
        ebike_pp,
        EbikeScenario(
            kappa=kappa,
            gef_mult=gef_mult,
            payload_kg=payload_kg,
            rider_weight=rider_weight,
        ),
    )
    drone_g = compute_drone_per_order_g(
        air_km,
        drone_pp,
        DroneScenario(gef_mult=gef_mult, payload_kg=payload_kg),
    )
    anchor_ebike_g = compute_ebike_per_order_g(
        road_km,
        ebike_pp,
        EbikeScenario(
            kappa=anchor_kappa,
            gef_mult=anchor_gef_mult,
            payload_kg=payload_kg,
            rider_weight=rider_weight,
        ),
    )

    use_drone = drone_g < ebike_g
    oracle_lifecycle_kg = float(np.sum(np.where(use_drone, drone_g, ebike_g) * annual_sales) / 1000.0)
    anchor_lifecycle_kg = float(np.sum(anchor_ebike_g * annual_sales) / 1000.0)
    scenario_ebike_lifecycle_kg = float(np.sum(ebike_g * annual_sales) / 1000.0)
    drone_orders = float(annual_sales[use_drone].sum())
    total_orders = float(annual_sales.sum())
    return {
        "annual_lifecycle_kg": oracle_lifecycle_kg,
        "anchor_annual_kg": anchor_lifecycle_kg,
        "scenario_ebike_annual_kg": scenario_ebike_lifecycle_kg,
        "drone_ratio_pct": drone_orders / total_orders * 100.0 if total_orders > 0 else np.nan,
    }


def _build_pathway_oracle_panel_rows(
    stats_df: pd.DataFrame,
    panels: list[tuple],
    data_dir: str,
    exp_config: dict | None,
    baseline_kappa: float,
) -> pd.DataFrame:
    """Build per-city oracle-min rows for the pathway scenario map panels."""
    from emission.cal_single_delivery import create_drone_params, create_ebike_params
    from utils.grid_emission_factor import fetch_factor

    if not data_dir:
        return pd.DataFrame()

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    emission_cfg = (exp_config or {}).get("emission_params", {})
    rider_weight = float(emission_cfg.get("rider_weight", 70.0))
    payload_kg = float(emission_cfg.get("payload_kg", 1.0))
    gef_path = (exp_config or {}).get("gef_path")
    if "region_level" in stats_df.columns:
        city_rows = stats_df[stats_df["region_level"] == "city"].copy()
    else:
        city_rows = stats_df.copy()
    if city_rows.empty or not {"region_key", "province"}.issubset(city_rows.columns):
        return pd.DataFrame()

    rows = []
    skipped = []
    for _, city in city_rows.iterrows():
        region_key = city["region_key"]
        province = city["province"]
        od_path = os.path.join(data_dir, "city", region_key, f"{region_key}_od_pairs.csv")
        if not os.path.exists(od_path):
            skipped.append(region_key)
            continue
        try:
            od_df = pd.read_csv(
                od_path,
                usecols=["road_distance_m", "distance_km", "annual_sales"],
            )
            province_gef = (
                fetch_factor(province, gef_path)
                if gef_path
                else float(emission_cfg.get("e_use", 0.6))
            )
            for label, _title, kappa, gef_mult, _strategy in panels:
                metrics = _evaluate_pathway_oracle_city(
                    od_df,
                    ebike_params,
                    drone_params,
                    province_gef,
                    kappa=kappa,
                    gef_mult=gef_mult,
                    rider_weight=rider_weight,
                    payload_kg=payload_kg,
                    anchor_kappa=baseline_kappa,
                    anchor_gef_mult=1.0,
                )
                rows.append(
                    {
                        "panel_label": label,
                        "region_key": region_key,
                        "province": province,
                        "kappa": kappa,
                        "strategy": "oracle_min_mixed",
                        "gef_mult": gef_mult,
                        "threshold_km": np.nan,
                        "drone_ratio_pct": metrics["drone_ratio_pct"],
                        "annual_lifecycle_kg": metrics["annual_lifecycle_kg"],
                        "anchor_annual_kg": metrics["anchor_annual_kg"],
                        "scenario_ebike_annual_kg": metrics["scenario_ebike_annual_kg"],
                        "reduction_vs_anchor_ebike_pct": (
                            (1.0 - metrics["annual_lifecycle_kg"] / metrics["anchor_annual_kg"]) * 100.0
                            if metrics["anchor_annual_kg"] > 0
                            else np.nan
                        ),
                    }
                )
        except Exception:
            skipped.append(region_key)
            continue

    if rows:
        print(
            "plot_regions_map_pathway_scenarios: built oracle-min mixed panels "
            f"from {len({row['region_key'] for row in rows})} cities "
            f"(skipped {len(set(skipped))})"
        )
    return pd.DataFrame(rows)


def _build_pathway_curve_data(
    stats_df: pd.DataFrame,
    data_dir: str,
    exp_config: dict | None,
    baseline_kappa: float,
    kappa_end: float = 0.65,
    n_points: int = 7,
    max_rows_per_city: int = 1500,
    random_state: int = 42,
) -> dict:
    """Build sampled nationwide kappa and GEF pathway curves."""
    from emission.cal_single_delivery import create_drone_params, create_ebike_params
    from utils.grid_emission_factor import fetch_factor

    if not data_dir:
        return {}

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)
    emission_cfg = (exp_config or {}).get("emission_params", {})
    rider_weight = float(emission_cfg.get("rider_weight", 70.0))
    payload_kg = float(emission_cfg.get("payload_kg", 1.0))
    gef_path = (exp_config or {}).get("gef_path")
    if "region_level" in stats_df.columns:
        city_rows = stats_df[stats_df["region_level"] == "city"].copy()
    else:
        city_rows = stats_df.copy()
    required_stats = {"region_key", "province"}
    if city_rows.empty or not required_stats.issubset(city_rows.columns):
        return {}

    sampled = []
    skipped = []
    for idx, row in city_rows.iterrows():
        region_key = row["region_key"]
        province = row["province"]
        od_path = os.path.join(data_dir, "city", region_key, f"{region_key}_od_pairs.csv")
        if not os.path.exists(od_path):
            skipped.append(region_key)
            continue
        try:
            od_df = pd.read_csv(
                od_path,
                usecols=["road_distance_m", "distance_km", "annual_sales"],
            )
            od_sample, scale = _sample_od_for_pathway_curve(
                od_df,
                max_rows=max_rows_per_city,
                random_state=random_state + int(idx),
            )
            if od_sample.empty:
                skipped.append(region_key)
                continue
            province_gef = (
                fetch_factor(province, gef_path)
                if gef_path
                else float(emission_cfg.get("e_use", 0.6))
            )
        except Exception:
            skipped.append(region_key)
            continue
        sampled.append((od_sample, scale, float(province_gef)))

    if not sampled:
        return {}

    kappa_values = np.linspace(baseline_kappa, kappa_end, n_points)
    gef_values = np.linspace(1.0, 0.5, n_points)

    def aggregate(kappa_value: float, gef_mult: float) -> dict:
        saved_kg = 0.0
        city_shares = []
        for od_sample, scale, province_gef in sampled:
            result = _evaluate_pathway_optimal_curve_point(
                od_sample,
                scale,
                ebike_params,
                drone_params,
                province_gef,
                kappa=kappa_value,
                gef_mult=gef_mult,
                rider_weight=rider_weight,
                payload_kg=payload_kg,
                anchor_kappa=baseline_kappa,
                anchor_gef_mult=1.0,
            )
            saved_kg += result["saved_kg"]
            city_shares.append(result["drone_share_pct"])
        return {
            "saved_kt": saved_kg / 1e6,
            "drone_share_pct": float(np.nanmedian(city_shares)) if city_shares else np.nan,
        }

    kappa_curve = []
    for value in kappa_values:
        point = aggregate(float(value), 1.0)
        point["x"] = float(value)
        kappa_curve.append(point)

    gef_curve = []
    for value in gef_values:
        point = aggregate(baseline_kappa, float(value))
        point["x"] = float(value)
        gef_curve.append(point)

    print(
        "plot_regions_map_pathway_scenarios: sampled "
        f"{len(sampled)} cities for pathway curves "
        f"(max {max_rows_per_city} OD rows/city; skipped {len(skipped)})"
    )
    return {
        "kappa": pd.DataFrame(kappa_curve),
        "gef": pd.DataFrame(gef_curve),
    }


def _pathway_cache_file_signature(path: str):
    """Return file metadata used to invalidate pathway figure data cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _pathway_optional_cache_file_signature(path: str):
    """Return cache signature for an optional pathway source file."""
    if not path or not os.path.exists(path):
        return None
    return _pathway_cache_file_signature(path)


def _pathway_exp_config_cache_signature(exp_config: dict | None):
    """Return relevant experiment config fields for pathway cache invalidation."""
    cfg = exp_config or {}
    return {
        "emission_params": cfg.get("emission_params", {}),
        "gef_path": cfg.get("gef_path"),
        "cache_dir": cfg.get("cache_dir"),
    }


def _pathway_city_od_cache_signatures(stats_df: pd.DataFrame, data_dir: str):
    """Return source signatures for available city OD files used by this figure."""
    if not data_dir:
        return {}
    if "region_level" in stats_df.columns:
        city_rows = stats_df[stats_df["region_level"] == "city"].copy()
    else:
        city_rows = stats_df.copy()
    if "region_key" not in city_rows.columns:
        return {}

    source_files = {}
    for region_key in city_rows["region_key"].dropna().astype(str):
        od_path = os.path.join(data_dir, "city", region_key, f"{region_key}_od_pairs.csv")
        if os.path.exists(od_path):
            source_files[region_key] = _pathway_cache_file_signature(od_path)
    return source_files


def _pathway_scenarios_cache_meta(
    named_csv_path: str,
    stats_csv_path: str,
    stats_df: pd.DataFrame,
    data_dir: str,
    exp_config: dict | None,
    panels: list[tuple],
    baseline_kappa: float,
):
    """Return metadata used to invalidate regions_map_pathway_scenarios cache."""
    return {
        "version": PATHWAY_SCENARIOS_CACHE_VERSION,
        "figure_name": "regions_map_pathway_scenarios",
        "named_file": _pathway_cache_file_signature(named_csv_path),
        "stats_file": _pathway_cache_file_signature(stats_csv_path),
        "gef_file": _pathway_optional_cache_file_signature(
            (exp_config or {}).get("gef_path")
        ),
        "data_dir": os.path.abspath(data_dir) if data_dir else None,
        "od_files": _pathway_city_od_cache_signatures(stats_df, data_dir),
        "exp_config": _pathway_exp_config_cache_signature(exp_config),
        "baseline_kappa": float(baseline_kappa),
        "panels": [
            {
                "label": label,
                "title": title,
                "kappa": float(kappa),
                "gef_mult": float(gef_mult),
                "strategy": strategy,
            }
            for label, title, kappa, gef_mult, strategy in panels
        ],
    }


def _build_pathway_prepared_data_with_cache(
    named_csv_path: str,
    stats_csv_path: str,
    stats_df: pd.DataFrame,
    data_dir: str,
    exp_config: dict | None,
    panels: list[tuple],
    baseline_kappa: float,
):
    """Return oracle panel rows and curve data with a prepared-data cache."""
    from utils.config import get_cache_dir

    if not data_dir:
        return (
            _build_pathway_oracle_panel_rows(
                stats_df,
                panels,
                data_dir=data_dir,
                exp_config=exp_config,
                baseline_kappa=baseline_kappa,
            ),
            _build_pathway_curve_data(
                stats_df,
                data_dir=data_dir,
                exp_config=exp_config,
                baseline_kappa=baseline_kappa,
            ),
        )

    cache_path = os.path.join(
        get_cache_dir("visualization", "paper"),
        "regions_map_pathway_scenarios_prepared.pkl",
    )
    cache_meta = _pathway_scenarios_cache_meta(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        stats_df=stats_df,
        data_dir=data_dir,
        exp_config=exp_config,
        panels=panels,
        baseline_kappa=baseline_kappa,
    )
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("meta") == cache_meta:
                print(
                    "regions_map_pathway_scenarios: loading prepared data cache "
                    f"{cache_path}"
                )
                return cache["oracle_rows"], cache["curve_data"]
            print("regions_map_pathway_scenarios: prepared data cache is stale; rebuilding")
        except Exception as exc:
            print(
                "regions_map_pathway_scenarios: failed to load prepared cache "
                f"({exc}); rebuilding"
            )

    oracle_rows = _build_pathway_oracle_panel_rows(
        stats_df,
        panels,
        data_dir=data_dir,
        exp_config=exp_config,
        baseline_kappa=baseline_kappa,
    )
    curve_data = _build_pathway_curve_data(
        stats_df,
        data_dir=data_dir,
        exp_config=exp_config,
        baseline_kappa=baseline_kappa,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "meta": cache_meta,
                "oracle_rows": oracle_rows,
                "curve_data": curve_data,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"regions_map_pathway_scenarios: wrote prepared data cache {cache_path}")
    return oracle_rows, curve_data


def _pathway_scenario_definitions(baseline_kappa: float) -> list[tuple]:
    """Return the standard pathway scenario definitions."""
    return [
        ("a", "Today", baseline_kappa, 1.0, "optimal_mixed"),
        ("b", "Halved GEF", baseline_kappa, 0.5, "optimal_mixed_2030"),
        ("c", "Operator bundling", 0.65, 1.0, "optimal_mixed"),
        ("d", "Halved GEF + bundling", 0.65, 0.5, "optimal_mixed_2030"),
    ]


def _prepare_pathway_scenario_data(
    named_csv_path: str,
    stats_csv_path: str,
    data_dir: str = None,
    exp_config: dict = None,
):
    """Return cache-backed pathway scenario tables used by map figures."""
    named = pd.read_csv(named_csv_path)
    base = pd.read_csv(stats_csv_path)

    baseline_kappa = _baseline_kappa_for_pathway(named, base)
    panels = _pathway_scenario_definitions(baseline_kappa)

    reduction_col = "reduction_vs_anchor_ebike_pct"
    has_annual = "annual_lifecycle_kg" in named.columns
    anchor_rows = _pathway_anchor_annual_kg(named, base, baseline_kappa)
    oracle_rows, curve_data = _build_pathway_prepared_data_with_cache(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        stats_df=base,
        data_dir=data_dir,
        exp_config=exp_config,
        panels=panels,
        baseline_kappa=baseline_kappa,
    )
    if not oracle_rows.empty:
        has_annual = True
        anchor_rows = (
            oracle_rows[["region_key", "anchor_annual_kg"]]
            .dropna()
            .drop_duplicates(subset=["region_key"])
        )
    else:
        print(
            "pathway scenario maps: oracle-min panel rows unavailable; "
            "falling back to named threshold-sweep strategies."
        )
    if has_annual and anchor_rows.empty:
        print(
            "pathway scenario maps: annual scenario totals are present "
            "but no weighted all-ebike anchor was found; falling back to named reductions."
        )

    panel_dfs = {}
    panel_named = {}
    drone_share_median = {}
    annual_saved_total_kg = {}
    all_values = []
    for label, _title, k, g, strat in panels:
        if not oracle_rows.empty:
            rows = oracle_rows[oracle_rows["panel_label"] == label].copy()
        else:
            rows = named[
                np.isclose(named["kappa"], k)
                & np.isclose(named["gef_mult"], g)
                & (named["strategy"] == strat)
            ].copy()
        rows["province_en"] = rows["province"].map(PROVINCE_NAME_EN).fillna(
            rows["province"]
        )
        panel_named[label] = rows
        share_col = (
            _scenario_drone_share_column(rows)
            if not rows.empty
            else "drone_ratio_pct"
        )
        if has_annual and not anchor_rows.empty:
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
        else:
            sub = rows[["region_key", reduction_col]].rename(
                columns={reduction_col: "max_lifecycle_reduction_pct"}
            )
        merged = base.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            sub, on="region_key", how="left"
        )
        panel_dfs[label] = merged
        all_values.extend(merged["max_lifecycle_reduction_pct"].dropna().tolist())
        drone_share_median[label] = float(rows[share_col].median()) if len(rows) else float("nan")
        if has_annual and not anchor_rows.empty:
            joined = rows[["region_key", "annual_lifecycle_kg"]].merge(
                anchor_rows, on="region_key", how="inner"
            )
            annual_saved_total_kg[label] = float(
                (joined["anchor_annual_kg"] - joined["annual_lifecycle_kg"]).sum()
            )

    return {
        "named": named,
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
            ax.barh(x, saved, height=bar_width, left=0.0, color="#1b7837", alpha=0.75)
        else:
            ax.plot(saved, x, color="#1b7837", marker="o", linewidth=1.8, markersize=4)
        ax.set_xlabel(r"CO$_2$ saved (kt/yr)", fontsize=10, color="#1b7837")
        ax.set_ylabel(axis_label, fontsize=10)
        ax.tick_params(axis="x", labelcolor="#1b7837", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.35)
        if saved_limits is not None:
            ax.set_xlim(*saved_limits)
        if saved_as_bar:
            ax.margins(x=0)
            ax.set_xlim(left=0.0)

        ax_share = ax.twiny()
        ax_share.plot(share, x, color="#2166ac", marker="s", linewidth=1.6, markersize=3.5)
        ax_share.set_xlabel("Drone share (%)", fontsize=10, color="#2166ac")
        ax_share.tick_params(axis="x", labelcolor="#2166ac", labelsize=8)
        if share_limits is not None:
            ax_share.set_xlim(*share_limits)
    else:
        if saved_as_bar:
            ax.bar(x, saved, width=bar_width, bottom=0.0, color="#1b7837", alpha=0.75)
        else:
            ax.plot(x, saved, color="#1b7837", marker="o", linewidth=1.8, markersize=4)
        ax.set_xlabel(axis_label, fontsize=10)
        ax.set_ylabel(r"CO$_2$ saved (kt/yr)", fontsize=10, color="#1b7837")
        ax.tick_params(axis="y", labelcolor="#1b7837", labelsize=8)
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
        ax_share.plot(x, share, color="#2166ac", marker="s", linewidth=1.6, markersize=3.5)
        ax_share.set_ylabel("Drone share (%)", fontsize=10, color="#2166ac")
        ax_share.tick_params(axis="y", labelcolor="#2166ac", labelsize=8)
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
    colors = ["#2166ac", "#e08214"]
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
        "co2_saved_color": "#1b7837",
        "drone_share_mark": "line",
        "drone_share_color": "#2166ac",
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
                    "color": "#2166ac",
                },
                {
                    "label": "E-bike bundling factor kappa",
                    "value": _json_safe_scalar(kappa),
                    "unit": "dimensionless",
                    "color": "#e08214",
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
    named_csv_path: str,
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
        "source_named_csv_path": named_csv_path,
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
                "gef_multiplier": "#2166ac",
                "kappa": "#e08214",
            },
        },
        "panels": scenario_reports,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map pathway scenarios report saved to {report_path}")


def _pathway_scenario_maps_panel_report(
    panel_df: pd.DataFrame,
    color_col: str,
    ax_map,
    panel_label: str,
    title: str,
    metric_title: str,
    kappa: float,
    gef_mult: float,
    annual_saved_kg: float,
    drone_share_median: float,
):
    """Return compact metadata for one scenario-map panel."""
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

    return {
        "panel_letter": panel_label,
        "title": title,
        "metric": metric_title,
        "scenario": {
            "kappa": _json_safe_scalar(kappa),
            "gef_multiplier": _json_safe_scalar(gef_mult),
            "annual_co2_saved_kt_per_year": _json_safe_scalar(annual_saved_kg / 1e6),
            "median_city_drone_share_pct": _json_safe_scalar(drone_share_median),
        },
        "map_axis": _axis_report(ax_map),
        "map_color_by": color_col,
        "top_10_cities_by_map_value": compact_city_records(ranked.head(10)),
        "bottom_10_cities_by_map_value": compact_city_records(
            ranked.tail(10).sort_values(color_col)
        ),
    }


def _write_regions_map_pathway_scenario_maps_report(
    save_path: str,
    figsize: tuple,
    named_csv_path: str,
    stats_csv_path: str,
    panels: list[tuple],
    panel_dfs: dict,
    share_dfs: dict,
    map_panel_specs: list[tuple],
    curve_data: dict,
    curve_axes: list,
    curve_twins: list,
    annual_saved_total_kg: dict,
    drone_share_median: dict,
    reduction_cmap,
    reduction_bins,
    reduction_edges,
    reduction_cbar_ax,
    share_cmap,
    share_bins,
    share_edges,
    share_cbar_ax,
):
    """Write a compact JSON report for regions_map_pathway_scenario_maps."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    scenario_by_label = {
        label: (title, kappa, gef_mult, strategy)
        for label, title, kappa, gef_mult, strategy in panels
    }
    panel_reports = {}
    for (
        ax,
        panel_letter,
        scenario_label,
        scenario_title,
        metric_title,
        color_col,
        _cmap,
        _norm,
    ) in map_panel_specs:
        title, kappa, gef_mult, _strategy = scenario_by_label[scenario_label]
        panel_df = (
            panel_dfs[scenario_label]
            if color_col == "max_lifecycle_reduction_pct"
            else share_dfs[scenario_label]
        )
        key = f"{panel_letter}_{scenario_title.lower().replace(' ', '_')}_{metric_title.lower().replace(' ', '_')}"
        panel_reports[key] = _pathway_scenario_maps_panel_report(
            panel_df=panel_df,
            color_col=color_col,
            ax_map=ax,
            panel_label=panel_letter,
            title=title,
            metric_title=metric_title,
            kappa=kappa,
            gef_mult=gef_mult,
            annual_saved_kg=annual_saved_total_kg.get(scenario_label, np.nan),
            drone_share_median=drone_share_median.get(scenario_label, np.nan),
        )

    payload = {
        "figure_name": "regions_map_pathway_scenario_maps",
        "figure_path": save_path,
        "source_named_csv_path": named_csv_path,
        "source_stats_csv_path": stats_csv_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "coordinate_convention": (
            "Axis bbox values are figure-fraction coordinates. Map colors encode "
            "scenario reduction or drone share; gray indicates unstudied regions."
        ),
        "layout": {
            "row_1": "Emission reduction maps: Halved GEF, Today, Operator bundling",
            "row_2": "Drone share maps: Halved GEF, Today, Operator bundling",
            "row_3": "Pathway curves: mirrored GEF curve and kappa curve",
        },
        "global_encodings": {
            "reduction_map_colorbar": _colorbar_report(
                reduction_cbar_ax,
                reduction_bins,
                reduction_edges,
                "Reduction vs today's all-ebike anchor (%)",
                reduction_cmap,
            ),
            "drone_share_map_colorbar": _colorbar_report(
                share_cbar_ax,
                share_bins,
                share_edges,
                "Drone share (%)",
                share_cmap,
            ),
            "curve_colors": {
                "co2_saved": "#1b7837",
                "drone_share": "#2166ac",
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
            **panel_reports,
            "g_gef_pathway_curve": {
                **_pathway_curve_report(
                    curve_axes[0],
                    curve_twins[0],
                    curve_data.get("gef"),
                    "horizontal",
                    co2_saved_mark="bar",
                    co2_saved_baseline=0.0,
                ),
                "mirrored_x_axis": True,
                "visible_y_axis": "co2_saved",
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
                "mirrored_x_axis": False,
                "visible_y_axis": "drone_share",
            },
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map pathway scenario maps report saved to {report_path}")


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
        "#D9D9D9",
        "#BDBDBD",
        "#A1A1A1",
        "#858585",
        "#696969",
        "#4D4D4D",
        "#313131",
    )
    base_cmap = LinearSegmentedColormap.from_list(
        "gef_grayscale" if grayscale else "gef",
        grayscale_colors
        if grayscale
        else ["#006837", "#31A354", "#D95F0E", "#A50F15"],
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
        ["#4D4D4D", "#4D4D4D"]
        if grayscale
        else ["#006837", "#A50F15"]
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
            slope, intercept, _, _, _ = stats.linregress(x, y)
            x_line = np.linspace(x_global_min, x_global_max, 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=line_color,
                linewidth=1.35 if compact else 2.5,
                alpha=0.42,
                zorder=15,
            )
            n = len(x)
            se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
            t_val = stats.t.ppf(0.975, n - 2)
            x_mean = np.mean(x)
            ci = t_val * se * np.sqrt(
                1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
            )
            ax.fill_between(
                x_line,
                y_line - ci,
                y_line + ci,
                alpha=0.11,
                color=line_color,
                zorder=5,
            )

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
            fontsize=7 if compact else 9,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.14" if compact else "round,pad=0.22",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )
        ax.set_xlim(x_global_min, x_global_max)
        ax.set_ylim(y_global_min, y_global_max)
        ax.set_xlabel(
            VARIABLE_LABELS.get(feature_var, feature_var),
            fontsize=7.5 if compact else 12,
        )
        if compact:
            ax.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
        ax.grid(
            True,
            alpha=0.18 if compact else 0.3,
            linestyle="--",
            linewidth=0.45 if compact else 0.8,
        )

    ax_low.set_ylabel(
        VARIABLE_LABELS.get(result_var, result_var),
        fontsize=7.5 if compact else 12,
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
            fontsize=7,
            labelpad=1.5,
        )
        if horizontal_colorbar:
            cbar.ax.xaxis.set_label_position("top")
            cbar.ax.xaxis.set_ticks_position("bottom")
        cbar.ax.tick_params(labelsize=6.8, length=2.2)
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


def plot_grouped_scatter_split_panels(
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    group_var: str,
    threshold: float,
    save_path: str = None,
    figsize: tuple = (10, 4.8),
    for_latex: bool = False,
    plot_title: bool = False,
    group_labels: tuple = None,
):
    """Plot threshold-split scatter panels with shared axes and one central colorbar.

    Args:
        df: DataFrame with region statistics.
        feature_var: X-axis variable name.
        result_var: Y-axis variable name.
        group_var: Variable to split groups on.
        threshold: Split threshold for group_var.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        group_labels: Tuple of (low_label, high_label). Auto-generated if None.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1)
    drawn = _plot_grouped_scatter_split_panels_axes(
        fig=fig,
        subplot_spec=gs[0],
        df=df,
        feature_var=feature_var,
        result_var=result_var,
        group_var=group_var,
        threshold=threshold,
        plot_title=plot_title,
        group_labels=group_labels,
    )
    if drawn is None:
        plt.close(fig)
        return

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grouped scatter split panels saved to {save_path}")
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
            reductions from the `optimal_mixed_<year>` strategy at the baseline
            kappa in the named-sensitivity CSV.
        named_csv_path: Path to all_sensitivity_named.csv. Required when
            gef_scenario != "current".
    """
    from visualization.regions import plot_regions_on_china_map

    valid_scenarios = ("current", "2030", "2060")
    if gef_scenario not in valid_scenarios:
        raise ValueError(f"gef_scenario must be one of {valid_scenarios}, got {gef_scenario!r}")

    if gef_scenario == "current":
        map_df = None
        map_color_by = _weighted_reduction_column(df_stats)
        scatter_df = df_enriched
        scatter_result_var = _weighted_reduction_column(scatter_df)
        cbar_label = VARIABLE_LABELS.get(map_color_by, map_color_by)
        y_label_override = None
        scatter_group_labels = None
    else:
        if named_csv_path is None:
            raise ValueError("named_csv_path is required when gef_scenario != 'current'")
        named = pd.read_csv(named_csv_path)
        baseline_kappa = _named_baseline_kappa(named)
        scen_strategy = f"optimal_mixed_{gef_scenario}"
        reductions = (
            named[(named["strategy"] == scen_strategy) & np.isclose(named["kappa"], baseline_kappa)]
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


def plot_regions_map_optimal_drone_share(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 9),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """China choropleth of drone share needed for the best reduction mix.

    Args:
        df: all_region_stats DataFrame with region_name_cn and a weighted drone-share metric.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    from visualization.regions import plot_regions_on_china_map

    try:
        drone_share_col = _weighted_drone_share_column(df)
    except KeyError as exc:
        print(f"Regions map optimal drone share skipped — {exc}")
        return

    needed = {"region_name_cn", drone_share_col}
    missing = sorted(needed - set(df.columns))
    if missing:
        print(f"Regions map optimal drone share skipped — missing columns {missing}")
        return

    map_df = df.copy()
    map_df[drone_share_col] = pd.to_numeric(map_df[drone_share_col], errors="coerce")
    if map_df[drone_share_col].dropna().empty:
        print("Regions map optimal drone share skipped — no valid drone-share values")
        return

    cmap, norm, bin_edges, drone_share_edges = _build_drone_share_cmap()

    fig, ax = plt.subplots(figsize=figsize)
    plot_regions_on_china_map(
        df=map_df,
        color_by=drone_share_col,
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

    if plot_title:
        fig.suptitle(
            "Drone Share Needed for Minimum Lifecycle Emission",
            fontsize=14,
            fontweight="bold",
        )

    plt.tight_layout()

    fig.canvas.draw()
    ax_pos = ax.get_position()
    cbar_ax = fig.add_axes((ax_pos.x1 + 0.005, ax_pos.y0, 0.014, ax_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_ax,
        cmap,
        norm,
        bin_edges,
        drone_share_edges,
        "Drone share at minimum lifecycle emission (%)",
    )
    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map optimal drone share saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_kappa_panels(
    named_csv_path: str,
    stats_csv_path: str,
    kappas: tuple | None = None,
    save_path: str = None,
    figsize: tuple = (16, 7),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """2x2 panel grid: China region choropleth of drone-vs-ebike reduction across kappa.

    Reduction values come from `all_sensitivity_named.csv` rows for the
    current-grid optimal mixed strategy, using same-kappa all-ebike as the
    denominator. This keeps kappa as an operator pathway scenario rather than
    a drone-technology sensitivity feature. Panels share one diverging colour
    scale so the maps are directly comparable.

    Args:
        named_csv_path: Path to all_sensitivity_named.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        kappas: Optional kappa values to render. Defaults to the modeled kappa
            values available in the named-scenario table.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show figure-level title.
    """
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Patch

    from visualization.regions import plot_regions_on_china_map

    named = pd.read_csv(named_csv_path)
    base = pd.read_csv(stats_csv_path)
    if kappas is None:
        kappas = tuple(sorted(named["kappa"].dropna().unique()))
    else:
        kappas = tuple(kappas)
    if not kappas:
        print("plot_regions_map_kappa_panels: no kappa values in named scenarios; skipping")
        return

    reduction_col = "reduction_vs_all_ebike_same_kappa_pct"
    kappa_dfs = {}
    drone_share_median = {}
    all_values = []
    for k in kappas:
        kappa_rows = named[
            np.isclose(named["kappa"], k)
            & np.isclose(named["gef_mult"], 1.0)
            & (named["strategy"] == "optimal_mixed")
        ].copy()
        sub = kappa_rows[["region_key", reduction_col]].rename(
            columns={reduction_col: "max_lifecycle_reduction_pct"}
        )
        merged = base.drop(columns=["max_lifecycle_reduction_pct"], errors="ignore").merge(
            sub, on="region_key", how="left"
        )
        kappa_dfs[k] = merged
        all_values.extend(merged["max_lifecycle_reduction_pct"].dropna().tolist())
        share_col = _scenario_drone_share_column(kappa_rows) if not kappa_rows.empty else "drone_ratio_pct"
        drone_share_median[k] = float(kappa_rows[share_col].median()) if len(kappa_rows) else float("nan")

    if not all_values:
        print("plot_regions_map_kappa_panels: no named kappa scenario values matched any region in stats — "
              "rendering empty maps. Run sensitivity --stages named first.")
        max_abs = 1.0
    else:
        max_abs = max(abs(min(all_values)), abs(max(all_values)))
    shared_norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    shared_cmap = LinearSegmentedColormap.from_list("green_diverging", [
        "#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641",
    ])

    n = len(kappas)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    panel_labels = tuple(chr(ord("a") + i) for i in range(n))

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
        ax.set_title(f"$\\kappa = {_format_kappa_value(k)}$", fontsize=13)
        ax.text(-0.02, 1.02, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right")
        ax.text(
            0.02, 0.98,
            f"Median drone share: {drone_share_median[k]:.0f}%" if np.isfinite(drone_share_median[k]) else "Median drone share: n/a",
            transform=ax.transAxes,
            fontsize=10, va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85,
                      boxstyle="round,pad=0.3"),
        )
    for ax in axes.flat[n:]:
        ax.axis("off")

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
        fig.suptitle("Lifecycle reduction across e-bike bundling scenarios", fontsize=14, fontweight="bold")

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


def plot_regions_map_pathway_scenario_maps(
    named_csv_path: str,
    stats_csv_path: str,
    data_dir: str = None,
    exp_config: dict = None,
    save_path: str = None,
    figsize: tuple = (18, 14),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Region maps plus pathway curves for selected pathway scenarios.

    Panels show emission reduction and drone share for the halved-GEF, today's
    base, and operator-bundling pathway scenarios, followed by the GEF and
    kappa pathway curves from the shared pathway helpers. Values use the same
    cache-backed scenario preparation as `plot_regions_map_pathway_scenarios`.

    Args:
        named_csv_path: Path to all_sensitivity_named.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        data_dir: Data root containing city OD-pair CSVs for oracle-min scenarios.
        exp_config: Experiment config for vehicle parameters and GEF path.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show figure-level title.
    """
    from visualization.regions import plot_regions_on_china_map

    scenario_data = _prepare_pathway_scenario_data(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        data_dir=data_dir,
        exp_config=exp_config,
    )
    base = scenario_data["base"]
    panel_dfs = scenario_data["panel_dfs"]
    panel_named = scenario_data["panel_named"]
    all_values = scenario_data["all_values"]
    panels = scenario_data["panels"]
    curve_data = scenario_data["curve_data"]
    drone_share_median = scenario_data["drone_share_median"]
    annual_saved_total_kg = scenario_data["annual_saved_total_kg"]
    baseline_kappa = scenario_data["baseline_kappa"]

    scenario_labels = ("b", "a", "c")
    share_col_out = "pathway_drone_share_pct"
    share_dfs = {}
    for label in scenario_labels:
        rows = panel_named.get(label, pd.DataFrame())
        if rows.empty:
            sub = pd.DataFrame(columns=["region_key", share_col_out])
        else:
            share_col = _scenario_drone_share_column(rows)
            sub = rows[["region_key", share_col]].rename(
                columns={share_col: share_col_out}
            )
        share_dfs[label] = base.drop(columns=[share_col_out], errors="ignore").merge(
            sub,
            on="region_key",
            how="left",
        )

    if not all_values:
        print(
            "plot_regions_map_pathway_scenario_maps: no scenario rows matched any "
            "region in stats — rendering empty maps. Run sensitivity --stages named first."
        )
        v_max = 1.0
    else:
        v_max = max(max(all_values), 0.0)
    reduction_cmap, reduction_norm, reduction_bins, reduction_edges = _build_reduction_cmap(
        v_max
    )
    share_cmap, share_norm, share_bins, share_edges = _build_drone_share_cmap()

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

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        3,
        6,
        height_ratios=[1, 1, 0.75],
        hspace=0.16,
        wspace=0.18,
    )
    panel_specs = [
        (
            fig.add_subplot(gs[0, 0:2]),
            "a",
            "b",
            "Halved GEF",
            "Emission reduction",
            "max_lifecycle_reduction_pct",
            reduction_cmap,
            reduction_norm,
        ),
        (
            fig.add_subplot(gs[0, 2:4]),
            "b",
            "a",
            "Today",
            "Emission reduction",
            "max_lifecycle_reduction_pct",
            reduction_cmap,
            reduction_norm,
        ),
        (
            fig.add_subplot(gs[0, 4:6]),
            "c",
            "c",
            "Operator bundling",
            "Emission reduction",
            "max_lifecycle_reduction_pct",
            reduction_cmap,
            reduction_norm,
        ),
        (
            fig.add_subplot(gs[1, 0:2]),
            "d",
            "b",
            "Halved GEF",
            "Drone share",
            share_col_out,
            share_cmap,
            share_norm,
        ),
        (
            fig.add_subplot(gs[1, 2:4]),
            "e",
            "a",
            "Today",
            "Drone share",
            share_col_out,
            share_cmap,
            share_norm,
        ),
        (
            fig.add_subplot(gs[1, 4:6]),
            "f",
            "c",
            "Operator bundling",
            "Drone share",
            share_col_out,
            share_cmap,
            share_norm,
        ),
    ]

    for (
        ax,
        panel_letter,
        scenario_label,
        scenario_title,
        metric_title,
        color_col,
        cmap,
        norm,
    ) in panel_specs:
        map_df = (
            panel_dfs[scenario_label]
            if color_col == "max_lifecycle_reduction_pct"
            else share_dfs[scenario_label]
        )
        plot_regions_on_china_map(
            df=map_df,
            color_by=color_col,
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
        ax.text(
            -0.02,
            1.02,
            panel_letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    curve_axes = [
        fig.add_subplot(gs[2, 0:3]),
        fig.add_subplot(gs[2, 3:6]),
    ]
    curve_saved_limits, curve_share_limits = _pathway_curve_metric_limits(curve_data)
    curve_twins = [
        _plot_pathway_curve_panel(
            curve_axes[0],
            curve_data.get("gef"),
            "g",
            "Grid emission factor multiplier",
            orientation="horizontal",
            saved_limits=curve_saved_limits,
            share_limits=curve_share_limits,
            saved_as_bar=True,
            show_share_axis=False,
            mirror_x=True,
        ),
        _plot_pathway_curve_panel(
            curve_axes[1],
            curve_data.get("kappa"),
            "h",
            r"E-bike bundling factor $\kappa$",
            orientation="horizontal",
            saved_limits=curve_saved_limits,
            share_limits=curve_share_limits,
            saved_as_bar=True,
            show_saved_axis=False,
        ),
    ]

    if plot_title:
        fig.suptitle(
            "Pathway scenario maps and transition curves",
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout(rect=(0, 0.02, 0.90, 0.98))
    fig.canvas.draw()
    for ax in curve_axes:
        pos = ax.get_position()
        width = pos.width * 0.94
        height = pos.height * 0.72
        x0 = pos.x0 + (pos.width - width) / 2
        y0 = pos.y0 + (pos.height - height) / 2
        ax.set_position([x0, y0, width, height])
    for ax, ax_twin in zip(curve_axes, curve_twins):
        if ax_twin is not None:
            ax_twin.set_position(ax.get_position())

    def _row_colorbar_position(row_axes, x0=0.915, width=0.014, height_scale=1.12):
        positions = [ax.get_position() for ax in row_axes]
        row_y0 = min(pos.y0 for pos in positions)
        row_y1 = max(pos.y1 for pos in positions)
        row_center = (row_y0 + row_y1) / 2
        height = (row_y1 - row_y0) * height_scale
        return (x0, row_center - height / 2, width, height)

    cbar_ax_reduction = fig.add_axes(
        _row_colorbar_position([spec[0] for spec in panel_specs[:3]])
    )
    cbar_reduction, n_reduction_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_ax_reduction,
        reduction_cmap,
        reduction_norm,
        reduction_bins,
        reduction_edges,
        r"Reduction vs today's all-ebike anchor (%)",
    )
    _annotate_unstudied(fig, cbar_reduction, cbar_ax_reduction, n_reduction_bands)

    cbar_ax_share = fig.add_axes(
        _row_colorbar_position([spec[0] for spec in panel_specs[3:]])
    )
    cbar_share, n_share_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_ax_share,
        share_cmap,
        share_norm,
        share_bins,
        share_edges,
        "Drone share (%)",
    )
    _annotate_unstudied(fig, cbar_share, cbar_ax_share, n_share_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map pathway scenario maps saved to {save_path}")
        _write_regions_map_pathway_scenario_maps_report(
            save_path=save_path,
            figsize=figsize,
            named_csv_path=named_csv_path,
            stats_csv_path=stats_csv_path,
            panels=panels,
            panel_dfs=panel_dfs,
            share_dfs=share_dfs,
            map_panel_specs=panel_specs,
            curve_data=curve_data,
            curve_axes=curve_axes,
            curve_twins=curve_twins,
            annual_saved_total_kg=annual_saved_total_kg,
            drone_share_median=drone_share_median,
            reduction_cmap=reduction_cmap,
            reduction_bins=reduction_bins,
            reduction_edges=reduction_edges,
            reduction_cbar_ax=cbar_ax_reduction,
            share_cmap=share_cmap,
            share_bins=share_bins,
            share_edges=share_edges,
            share_cbar_ax=cbar_ax_share,
        )
    else:
        plt.show()

    plt.close(fig)


def plot_regions_map_pathway_scenarios(
    named_csv_path: str,
    stats_csv_path: str,
    data_dir: str = None,
    exp_config: dict = None,
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

    Panels use the per-OD oracle-min mixed allocation when city OD files and
    experiment config are available. The named threshold-sweep scenarios are
    retained as a fallback for older artifacts:
        today                    baseline kappa, gef_mult=1.0
        halved GEF               baseline kappa, gef_mult=0.5
        operator bundling        kappa=0.65, gef_mult=1.0
        halved GEF + bundling    kappa=0.65, gef_mult=0.5

    Reduction values (map fill) prefer annual lifecycle totals against the
    weighted all-ebike city baseline from `all_region_stats.csv`, falling back
    to `reduction_vs_anchor_ebike_pct` when annual scenario totals are missing.
    Boxplot y-axis is drone order share in 0..100.

    Args:
        named_csv_path: Path to all_sensitivity_named.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        data_dir: Data root containing city OD-pair CSVs for oracle-min scenarios.
        exp_config: Experiment config for vehicle parameters and GEF path.
        save_path: Base output path. Scenario suffixes are added before extension.
            Shows plots if None.
        figsize: Size tuple for each standalone scenario subfigure.
        for_latex: If True, save as PDF.
        plot_title: If True, show the scenario title in each subfigure.
    """
    from visualization.regions import plot_regions_on_china_map

    scenario_data = _prepare_pathway_scenario_data(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        data_dir=data_dir,
        exp_config=exp_config,
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
        print("plot_regions_map_pathway_scenarios: no scenario rows matched any region in stats — "
              "rendering empty maps. Run sensitivity --stages named first.")
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
        ax_box.boxplot(
            data, positions=range(len(province_order)), widths=0.65,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=0.8),
            boxprops=dict(facecolor="#a6d96a", edgecolor="#1a9641", linewidth=0.45),
            whiskerprops=dict(color="#1a9641", linewidth=0.45),
            capprops=dict(color="#1a9641", linewidth=0.45),
        )
        ax_box.set_ylim(0, 102)
        ax_box.set_xlim(-0.7, len(province_order) - 0.3)
        ax_box.set_xticks(range(len(province_order)))
        ax_box.set_xticklabels(province_order, rotation=90, fontsize=5.2)
        ax_box.set_ylabel("Drone share (%)", fontsize=6.5)
        ax_box.tick_params(axis="y", labelsize=5.8, length=2)
        ax_box.grid(axis="y", linestyle=":", alpha=0.4)
        ax_box.axhline(drone_share_median[label], color="#1a9641", linestyle="--", linewidth=0.55, alpha=0.8)

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
        "Unstudied",
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
            named_csv_path=named_csv_path,
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
    reduction_col = _weighted_reduction_column(map_df)
    v_max = max(float(map_df[reduction_col].max(skipna=True)), 0.0)
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
        df=map_df, color_by=reduction_col,
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
    plot_province_total_ebike_lifecycle(
        df_stats,
        ax=ax_c,
        plot_title=plot_title,
        reduction_metric="percent",
    )

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
        VARIABLE_LABELS.get(reduction_col, reduction_col),
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
    """Plot distribution of emission reduction and drone order ratio across all cities.

    Two subplots: histogram+KDE of the preferred weighted reduction metric, and
    histogram of the preferred weighted drone-share metric. Annotated with n,
    mean, median, std.

    Args:
        df: DataFrame with region statistics (from load_region_stats).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax1 = axes[0]
    try:
        col1 = _weighted_reduction_column(df)
    except KeyError:
        col1 = None
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
    ax1.set_xlabel(VARIABLE_LABELS.get(col1, col1) if col1 else "Reduction (%)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2 = axes[1]
    try:
        col2 = _weighted_drone_share_column(df)
    except KeyError:
        col2 = None
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
    ax2.set_xlabel(VARIABLE_LABELS.get(col2, col2) if col2 else "Drone order ratio (%)", fontsize=11)
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


def plot_city_gef_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (8, 5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot per-order use-phase emission distributions across GEF ranges.

    Args:
        df: DataFrame with city-level grid emission factor and use-phase emissions.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    gef_col = None
    for candidate in ("grid_emission_factor", "gef"):
        if candidate in df.columns and df[candidate].notna().any():
            gef_col = candidate
            break

    if gef_col is None:
        print("City GEF distribution skipped - missing grid_emission_factor or gef column")
        return

    series = [
        (
            "total_ebike_use_phase_emission_kg_per_order_g",
            "total_ebike_use_phase_emission_kg",
            "E-bike",
            "#4C8CBF",
            -0.18,
        ),
        (
            "total_drone_use_phase_emission_kg_per_order_g",
            "total_drone_use_phase_emission_kg",
            "Drone",
            "#E05C2B",
            0.18,
        ),
    ]
    fixed_series = [
        (
            "total_ebike_fixed_emission_kg_per_order_g",
            "E-bike",
            "#4C8CBF",
            -0.18,
        ),
        (
            "total_drone_fixed_emission_kg_per_order_g",
            "Drone",
            "#E05C2B",
            0.18,
        ),
    ]
    available_series = [
        spec for spec in series
        if spec[1] in df.columns and df[spec[1]].notna().any()
    ]
    if not available_series:
        print("City GEF distribution skipped - missing city use-phase emission columns")
        return
    if "total_odpair_annual_sales" not in df.columns:
        print("City GEF distribution skipped - missing total_odpair_annual_sales column")
        return
    fixed_source_pairs = {
        "total_ebike_fixed_emission_kg_per_order_g": (
            "total_ebike_production_emission_kg",
            "total_ebike_end_of_life_emission_kg",
        ),
        "total_drone_fixed_emission_kg_per_order_g": (
            "total_drone_production_emission_kg",
            "total_drone_end_of_life_emission_kg",
        ),
    }
    available_fixed_series = [
        spec for spec in fixed_series
        if all(col in df.columns and df[col].notna().any() for col in fixed_source_pairs[spec[0]])
    ]
    if not available_fixed_series:
        print(
            "City GEF distribution skipped - missing production/end-of-life emission columns"
        )
        return
    source_cols = [source_col for _, source_col, _, _, _ in available_series]
    plot_cols = [plot_col for plot_col, _, _, _, _ in available_series]
    fixed_source_cols = [
        col
        for plot_col, _, _, _ in available_fixed_series
        for col in fixed_source_pairs[plot_col]
    ]
    fixed_plot_cols = [plot_col for plot_col, _, _, _ in available_fixed_series]

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"]

    needed = [gef_col, "total_odpair_annual_sales"] + source_cols + fixed_source_cols
    plot_df[gef_col] = pd.to_numeric(plot_df[gef_col], errors="coerce")
    plot_df["total_odpair_annual_sales"] = pd.to_numeric(
        plot_df["total_odpair_annual_sales"], errors="coerce"
    )
    for _, source_col, _, _, _ in available_series:
        plot_df[source_col] = pd.to_numeric(plot_df[source_col], errors="coerce")
    for source_col in fixed_source_cols:
        plot_df[source_col] = pd.to_numeric(plot_df[source_col], errors="coerce")
    plot_df = plot_df.dropna(subset=needed)
    valid_positive = (
        (plot_df[gef_col] > 0)
        & (plot_df[gef_col] <= 0.9)
        & (plot_df["total_odpair_annual_sales"] > 0)
        & (plot_df[source_cols] > 0).all(axis=1)
        & (plot_df[fixed_source_cols] >= 0).all(axis=1)
    )
    plot_df = plot_df[valid_positive]
    if plot_df.empty:
        print("City GEF distribution skipped - no valid city GEF/emission values")
        return
    for plot_col, source_col, _, _, _ in available_series:
        plot_df[plot_col] = (
            plot_df[source_col] / plot_df["total_odpair_annual_sales"] * 1000.0
        )
    for plot_col, _, _, _ in available_fixed_series:
        production_col, eol_col = fixed_source_pairs[plot_col]
        plot_df[plot_col] = (
            (plot_df[production_col] + plot_df[eol_col])
            / plot_df["total_odpair_annual_sales"]
            * 1000.0
        )

    gef = plot_df[gef_col]
    bin_width = 0.1
    bin_edges = np.linspace(0.0, 0.9, 10)
    labels = [f"{right:.1f}" for right in bin_edges[1:]]
    plot_df["gef_range"] = pd.cut(
        gef,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    bin_groups = []
    bin_positions = []
    for idx, label in enumerate(labels):
        group = plot_df[plot_df["gef_range"] == label]
        if group.empty:
            continue
        bin_groups.append(group)
        bin_positions.append(idx)

    if not bin_groups:
        print("City GEF distribution skipped - no cities in GEF ranges")
        return

    fig, (ax, ax_fixed) = plt.subplots(
        2,
        1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.0},
        sharex=True,
    )

    x_all = np.arange(len(labels), dtype=float)
    x = x_all[np.array(bin_positions, dtype=int)]
    density_x = gef.to_numpy(dtype=float) / bin_width - 1.0
    if len(density_x) > 5 and np.unique(density_x).size > 1:
        from scipy.stats import gaussian_kde

        x_violin = np.linspace(-0.5, len(labels) - 0.5, 240)
        density = gaussian_kde(density_x)(x_violin)
    else:
        x_violin = x_all
        density = np.array(
            [len(plot_df[plot_df["gef_range"] == label]) for label in labels],
            dtype=float,
        )
    if density.max() > 0:
        density = density / density.max()

    width = 0.28 if len(plot_cols) > 1 else 0.42
    legend_handles = []
    box_values = []

    for col, _, label, color, offset in available_series:
        positions = x + (offset if len(plot_cols) > 1 else 0.0)
        data = [group[col].to_numpy(dtype=float) for group in bin_groups]
        box_values.extend(np.concatenate(data).tolist())
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.45)
        for whisker in bp["whiskers"]:
            whisker.set_color(color)
        for cap in bp["caps"]:
            cap.set_color(color)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.8)
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.45, edgecolor=color)
        )

    fixed_values = []
    for col, label, color, offset in available_fixed_series:
        positions = x + (offset if len(fixed_plot_cols) > 1 else 0.0)
        data = [group[col].to_numpy(dtype=float) for group in bin_groups]
        fixed_values.extend(np.concatenate(data).tolist())
        bp = ax_fixed.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
            zorder=3,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.45)
        for whisker in bp["whiskers"]:
            whisker.set_color(color)
        for cap in bp["caps"]:
            cap.set_color(color)
        for median in bp["medians"]:
            median.set_color(color)
            median.set_linewidth(1.8)

    gef_label = VARIABLE_LABELS.get(gef_col, "Grid Emission Factor (kg CO2e/kWh)")
    ax.set_ylabel("Use phase\n(g CO2e/order)", fontsize=12)
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_yscale("log")
    box_values = np.array(box_values, dtype=float)
    fixed_values = np.array(fixed_values, dtype=float)
    all_positive = np.concatenate(
        [box_values[box_values > 0], fixed_values[fixed_values > 0]]
    )
    y_data_min = all_positive.min()
    y_data_max = all_positive.max()
    log_min = np.log10(y_data_min)
    log_max = np.log10(y_data_max)
    log_span = max(log_max - log_min, 0.25)
    y_floor_log = log_min - log_span * 0.16
    y_base_log = log_min - log_span * 0.145
    y_top_log = log_min - log_span * 0.025
    y_floor = 10 ** y_floor_log
    y_base = 10 ** y_base_log
    y_violin = 10 ** (y_base_log + density * (y_top_log - y_base_log))
    ax.fill_between(
        x_violin,
        y_base,
        y_violin,
        color="#7A7A7A",
        alpha=0.35,
        linewidth=0,
        zorder=1,
    )
    ax.plot(x_violin, y_violin, color="#555555", linewidth=0.8, zorder=2)
    ax.set_ylim(y_floor, y_data_max * 1.12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(
        legend_handles,
        [label for _, _, label, _, _ in available_series],
        loc="upper left",
        fontsize=9,
    )

    for xi, group in zip(x, bin_groups):
        ax.text(
            xi,
            0.09,
            f"n={len(group)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.35",
            clip_on=False,
            zorder=12,
        )

    ax_fixed.set_yscale("log")
    ax_fixed.set_ylim(y_data_max * 1.12, y_floor)
    ax_fixed.set_ylabel("Production + EOL\n(g CO2e/order)", fontsize=12)
    ax_fixed.set_xlabel(gef_label, fontsize=12)
    ax_fixed.xaxis.set_label_position("bottom")
    ax_fixed.xaxis.set_ticks_position("bottom")
    ax_fixed.set_xticks(x_all)
    ax_fixed.set_xticklabels(labels)
    ax_fixed.tick_params(
        axis="x",
        which="both",
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
        pad=2,
    )
    ax_fixed.set_xlim(-0.5, len(labels) - 0.5)
    ax_fixed.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax_fixed.set_axisbelow(True)

    if plot_title:
        ax.set_title(
            "City Use-phase Emission per Order by Grid Emission Factor Range",
            fontsize=14,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.96)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"City GEF distribution saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


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
    gef_col = None
    for candidate in ("grid_emission_factor", "gef"):
        if candidate in df.columns and df[candidate].notna().any():
            gef_col = candidate
            break

    if gef_col is None:
        print(
            "City GEF stacked components skipped - "
            "missing grid_emission_factor or gef column"
        )
        return None
    if "total_odpair_annual_sales" not in df.columns:
        print(
            "City GEF stacked components skipped - "
            "missing total_odpair_annual_sales column"
        )
        return None

    vehicle_specs = [
        {
            "label": "E-bike",
            "color": "#BDBDBD" if grayscale else "#4C8CBF",
            "offset": -0.18,
            "use_col": "total_ebike_use_phase_emission_kg",
            "production_col": "total_ebike_production_emission_kg",
            "eol_col": "total_ebike_end_of_life_emission_kg",
        },
        {
            "label": "Drone",
            "color": "#636363" if grayscale else "#E05C2B",
            "offset": 0.18,
            "use_col": "total_drone_use_phase_emission_kg",
            "production_col": "total_drone_production_emission_kg",
            "eol_col": "total_drone_end_of_life_emission_kg",
        },
    ]
    required_cols = ["total_odpair_annual_sales"]
    for spec in vehicle_specs:
        required_cols.extend([spec["use_col"], spec["production_col"], spec["eol_col"]])

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"City GEF stacked components skipped - missing columns {missing}")
        return None

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"]

    numeric_cols = [gef_col] + required_cols
    for col in numeric_cols:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=numeric_cols)
    valid_positive = (
        (plot_df[gef_col] > 0)
        & (plot_df[gef_col] <= 0.9)
        & (plot_df["total_odpair_annual_sales"] > 0)
    )
    for spec in vehicle_specs:
        valid_positive = (
            valid_positive
            & (plot_df[spec["use_col"]] > 0)
            & (plot_df[spec["production_col"]] >= 0)
            & (plot_df[spec["eol_col"]] >= 0)
        )
    plot_df = plot_df[valid_positive]
    if plot_df.empty:
        print(
            "City GEF stacked components skipped - "
            "no valid city GEF/emission values"
        )
        return None

    bin_width = 0.1
    bin_edges = np.linspace(0.0, 0.9, 10)
    labels = [f"{right:.1f}" for right in bin_edges[1:]]
    plot_df["gef_range"] = pd.cut(
        plot_df[gef_col],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    bin_groups = []
    bin_positions = []
    for idx, label in enumerate(labels):
        group = plot_df[plot_df["gef_range"] == label]
        if group.empty:
            continue
        bin_groups.append(group)
        bin_positions.append(idx)

    if not bin_groups:
        print("City GEF stacked components skipped - no cities in GEF ranges")
        return None

    gs = subplot_spec.subgridspec(
        2,
        1,
        height_ratios=[3.6, 1.0] if compact else [4.0, 1.0],
        hspace=0.0,
    )
    ax = fig.add_subplot(gs[0])
    ax_dist = fig.add_subplot(gs[1], sharex=ax)
    x_all = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    tick_labels = [f"{edge:.1f}" for edge in bin_edges]
    x = x_all[np.array(bin_positions, dtype=int)]
    width = bin_width * 0.28
    max_total = 0.0
    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#777777",
            alpha=0.35,
            edgecolor="#555555",
            label="Production + EOL",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#777777",
            alpha=0.9,
            edgecolor="#555555",
            label="Use phase",
        ),
    ]

    for spec in vehicle_specs:
        positions = x + spec["offset"] * bin_width
        fixed_values = []
        use_values = []
        for group in bin_groups:
            orders_sum = group["total_odpair_annual_sales"].sum()
            fixed_values.append(
                (group[spec["production_col"]].sum() + group[spec["eol_col"]].sum())
                / orders_sum
                * 1000.0
            )
            use_values.append(group[spec["use_col"]].sum() / orders_sum * 1000.0)

        fixed_values = np.array(fixed_values, dtype=float)
        use_values = np.array(use_values, dtype=float)
        totals = fixed_values + use_values
        if totals.size:
            max_total = max(max_total, float(totals.max()))

        ax.bar(
            positions,
            fixed_values,
            width=width,
            color=spec["color"],
            alpha=1.0 if grayscale else 0.38,
            edgecolor="#222222" if grayscale else "white",
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
            alpha=1.0 if grayscale else 0.88,
            edgecolor="#222222" if grayscale else "white",
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
                color="0.25",
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
        alpha=0.18 if compact else 0.3,
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
            alpha=1.0 if grayscale else 0.88,
            edgecolor="#222222" if grayscale else "white",
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
                facecolor="#BDBDBD",
                edgecolor="#222222",
                linewidth=0.45,
                hatch="////",
                label="Production + EOL",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#BDBDBD",
                edgecolor="#222222",
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

    density_x = plot_df[gef_col].to_numpy(dtype=float)
    if len(density_x) > 5 and np.unique(density_x).size > 1:
        from scipy.stats import gaussian_kde

        x_violin = np.linspace(bin_edges[0], bin_edges[-1], 240)
        density = gaussian_kde(density_x)(x_violin)
    else:
        x_violin = x_all
        density = np.array(
            [len(plot_df[plot_df["gef_range"] == label]) for label in labels],
            dtype=float,
        )
    if density.max() > 0:
        density = density / density.max()

    ax_dist.fill_between(
        x_violin,
        0,
        density,
        color="#7A7A7A",
        alpha=0.35,
        linewidth=0,
        zorder=1,
    )
    ax_dist.plot(x_violin, density, color="#555555", linewidth=0.8, zorder=2)
    ax_dist.scatter(
        density_x,
        np.zeros_like(density_x),
        marker="|",
        s=16,
        color="#333333",
        alpha=0.35,
        linewidths=0.6,
        zorder=3,
    )
    ax_dist.axhline(0, color="#555555", linewidth=0.8, zorder=2)
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


def plot_city_gef_scatter_components(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (12, 4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot GEF-binned per-order emission components as stacked bars.

    Args:
        df: DataFrame with city-level grid emission factor and component emissions.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1)
    drawn = _plot_city_gef_scatter_components_axes(
        fig=fig,
        subplot_spec=gs[0],
        df=df,
        plot_title=plot_title,
    )
    if drawn is None:
        plt.close(fig)
        return

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.17, top=0.96)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"City GEF stacked components saved to {save_path}")
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
    if variable == "max_lifecycle_reduction_pct" and variable not in df.columns:
        try:
            variable = _weighted_reduction_column(df)
        except KeyError as exc:
            print(f"City emission ranking skipped — {exc}")
            return

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
    ax.set_ylabel("Provincial-level region", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="x")

    if plot_title:
        ax.set_title(
            (
                f"{VARIABLE_LABELS.get(variable, variable)} by "
                "Provincial-Level Region"
            ),
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
    """Scatter plot of latitude vs weighted reduction metric with regression.

    Args:
        df: DataFrame with region statistics (must have min_lat/max_lat columns).
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    try:
        result_col = _weighted_reduction_column(df)
    except KeyError as exc:
        print(f"Geographic pattern plot skipped — {exc}")
        return

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
    ax1.set_xlabel("Provincial-level region", fontsize=12)
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
        fig.suptitle(
            "Delivery Orders and E-bike Lifecycle Emissions by "
            "Provincial-Level Region",
            fontsize=14,
            fontweight="bold",
        )

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
    ax1.set_xlabel("Provincial-level region", fontsize=12)
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
        fig.suptitle(
            "Delivery Orders and Lifecycle Emissions by "
            "Provincial-Level Region",
            fontsize=14,
            fontweight="bold",
        )

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
        reduction_axis_label = "City optimal reduction\n(t CO₂eq / yr)"
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
    bar_color = "#B0B0B0"
    bar_axis_color = "#4A4A4A"
    box_color = "#1F6E2D"

    _external_ax = ax is not None
    if _external_ax:
        ax1 = ax
        fig = ax1.get_figure()
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
    axis_label_size = 12 if _external_ax else 9
    x_tick_size = 9 if _external_ax else 7
    ebike_axis_label = (
        "Kappa-adjusted e-bike\nlifecycle emission\n(t CO₂eq / yr)"
        if _external_ax
        else "Annual e-bike lifecycle emissions\n(t CO₂eq yr⁻¹)"
    )

    ax1.bar(
        x,
        ebike_t,
        color=bar_color,
        alpha=0.85,
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
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    bp = ax2.boxplot(
        reduction_data,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=reduction_metric == "absolute",
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "none",
            "markeredgecolor": box_color,
            "alpha": 0.75,
        },
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
    if reduction_metric == "absolute":
        ax2.axhline(0.0, color=box_color, linewidth=0.8, alpha=0.55, zorder=1)
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
            plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=0.85),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=box_color,
                alpha=0.35,
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


def plot_regions_map_city_rank_inset(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (14, 7),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """China choropleth with top-city Oracle reduction rank.

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
    except KeyError as exc:
        print(f"Regions map with city rank inset skipped — {exc}")
        return
    valid_reductions = map_df[reduction_col].dropna()
    shared_cmap, shared_norm, bin_edges, reduction_edges = _build_absolute_reduction_log_cmap(
        valid_reductions
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    plot_regions_on_china_map(
        df=map_df, color_by=reduction_col,
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

    _plot_city_oracle_reduction_rank(
        map_df, 15,
        plot_title=plot_title,
        ax=ax_b,
        invert_vertical=True,
        compact=False,
    )

    ax_a.text(-0.03, 1.02, "a", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_b.text(-0.08, 1.02, "b", transform=ax_b.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    fig.tight_layout()

    fig.canvas.draw()
    a_pos = ax_a.get_position()
    cbar_ax = fig.add_axes((a_pos.x1 + 0.005, a_pos.y0,
                            0.014, a_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig, cbar_ax, shared_cmap, shared_norm, bin_edges, reduction_edges,
        "E-bike - Optimal (t CO₂eq / yr)",
    )
    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map with city rank inset saved to {save_path}")
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
        draw_south_china_inset_bubbles: If True, draw scaled bubbles on the South China Sea inset.
        color_scale_style: "colorbar" for a side colorbar, or "legend" for an in-map legend.
        label_layout_path: Optional JSON file with manual city label offsets.
        label_offsets: Optional in-memory city label offsets, overriding the layout file.
        return_label_artists: If True, return figure and annotation artists without saving.
    """
    from visualization.regions import plot_regions_on_china_map

    map_df = pd.read_csv(stats_csv_path)
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
    bubble_draw_df = bubble_df.sort_values("_bubble_size", ascending=True)
    is_map_only_figure = figure_name == ORACLE_REDUCTION_MAP_ONLY_FIGURE

    if is_map_only_figure:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=(1.0, 0.18),
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
        alpha=0.82,
        edgecolor="#FFFFFF",
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
        label_offsets=label_offsets,
        return_artists=return_label_artists,
    )
    unified_legend_style = {
        "loc": "lower right",
        "fontsize": 6.5,
        "title_fontsize": 6.5,
        "framealpha": 0.92,
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
            "framealpha": 0.78,
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
        inset_ax.patch.set_alpha(0.0)
        for name, spine in inset_ax.spines.items():
            spine.set_visible(name in {"left", "bottom"})
            spine.set_edgecolor("#666666")
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
            bottom=0.02,
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


def plot_regions_map_oracle_reduction_rank_map_only(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    save_path: str = None,
    figsize: tuple = (4.70, 4.14),
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
        return "#cccccc"
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
            "color": "#cccccc",
            "bin_index": 0,
        },
    }


def _map_panel_report(df, color_col, cmap, norm, bin_edges, ax, panel_letter):
    """Return row-level values used by one map panel."""
    rows = []
    for _, row in df.iterrows():
        value = row.get(color_col)
        rows.append(
            {
                "region_key": _json_safe_scalar(row.get("region_key")),
                "region_name_cn": _json_safe_scalar(row.get("region_name_cn")),
                "region_name_en": _json_safe_scalar(row.get("region_name_en")),
                "region_level": _json_safe_scalar(row.get("region_level")),
                "province": _json_safe_scalar(row.get("province")),
                color_col: _json_safe_scalar(value),
                "fill_color": _color_for_value(value, cmap, norm),
                "color_bin": _bin_label_for_value(value, bin_edges),
            }
        )
    return {
        "panel_letter": panel_letter,
        "axis": _axis_report(ax),
        "color_by": color_col,
        "rows": rows,
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
            edgecolors="#111111",
            linewidths=0.7,
            zorder=18,
            clip_on=False,
        )
        offset = label_offsets.get(city, (8, 8))
        xy = (x, y)
        label_offset = offset
        leader_shrink_points = 0.0
        arrowprops = dict(arrowstyle="-", color="0.25", lw=0.75)
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
                arrowprops = dict(
                    arrowstyle="-",
                    color="0.25",
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
            color="#111111",
            arrowprops=arrowprops,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="0.55",
                alpha=0.9,
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
        alpha=0.82,
        edgecolor="#FFFFFF",
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
                "color": "#1F6E2D",
                "alpha": 0.88,
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


def _distribution_inset_report(df, value_col, ax, bins, bar_width, label):
    """Return histogram bars and smoothed curve for a compact distribution inset."""
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        return {"axis": _axis_report(ax), "status": "skipped", "reason": "no values"}

    clipped = values.clip(lower=float(bins[0]), upper=float(bins[-1]))
    counts, edges = np.histogram(clipped, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    x_curve = np.linspace(float(centres.min()), float(centres.max()), 200)
    if len(centres) >= 4:
        from scipy.interpolate import make_interp_spline

        spline = make_interp_spline(centres, counts, k=3)
        y_curve = np.clip(spline(x_curve), 0, None)
    else:
        y_curve = np.interp(x_curve, centres, counts)

    return {
        "axis": _axis_report(ax),
        "value_col": value_col,
        "label": label,
        "bars": [
            {
                "bin_left": _json_safe_scalar(edges[i]),
                "bin_right": _json_safe_scalar(edges[i + 1]),
                "x_center": _json_safe_scalar(centres[i]),
                "width": bar_width,
                "height_city_count": int(counts[i]),
                "color": "#91bfdb",
                "edgecolor": "#08306b",
                "alpha": 0.85,
            }
            for i in range(len(counts))
        ],
        "curve_points": [
            {"x": _json_safe_scalar(x), "y": _json_safe_scalar(y)}
            for x, y in zip(x_curve, y_curve)
        ],
    }


def _drone_share_distribution_report(df, drone_share_col, ax):
    """Return histogram bars and smoothed curve for the drone-share inset."""
    return _distribution_inset_report(
        df,
        drone_share_col,
        ax,
        bins=np.arange(0, 101, 10, dtype=float),
        bar_width=8.5,
        label="Drone share (%)",
    )


def _province_lifecycle_panel_report(df, ax_main, ax_secondary):
    """Return bar and boxplot data for the province lifecycle panel."""
    needed = ["province", "total_ebike_lifecycle", "oracle_min_lifecycle_kg"]
    sub = df[needed].copy()
    sub["total_ebike_lifecycle"] = pd.to_numeric(
        sub["total_ebike_lifecycle"], errors="coerce"
    )
    sub["oracle_min_lifecycle_kg"] = pd.to_numeric(
        sub["oracle_min_lifecycle_kg"], errors="coerce"
    )
    sub["oracle_min_reduction_vs_ebike_pct"] = np.where(
        sub["total_ebike_lifecycle"] > 0,
        (
            1.0
            - sub["oracle_min_lifecycle_kg"] / sub["total_ebike_lifecycle"]
        )
        * 100.0,
        np.nan,
    )
    sub = sub.dropna()
    sub["province_en"] = sub["province"].map(PROVINCE_NAME_EN).fillna(sub["province"])
    reduction_col = "oracle_min_reduction_vs_ebike_pct"
    prov = (
        sub.groupby("province_en")
        .agg(total_ebike=("total_ebike_lifecycle", "sum"))
        .reset_index()
        .sort_values("total_ebike", ascending=False)
    )
    ordered = prov["province_en"].tolist()
    ebike_t = prov["total_ebike"].values / 1e3

    bars = []
    boxes = []
    for x, (province, height) in enumerate(zip(ordered, ebike_t)):
        values = sub.loc[sub["province_en"] == province, reduction_col].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            q1, median, q3 = np.percentile(finite, [25, 50, 75])
            iqr = q3 - q1
            low_limit = q1 - 1.5 * iqr
            high_limit = q3 + 1.5 * iqr
            whisker_low = finite[finite >= low_limit].min()
            whisker_high = finite[finite <= high_limit].max()
        else:
            q1 = median = q3 = whisker_low = whisker_high = np.nan
        bars.append(
            {
                "province": province,
                "x_center": x,
                "height_total_ebike_t_co2e_per_year": _json_safe_scalar(height),
                "color": "#B0B0B0",
                "alpha": 0.85,
            }
        )
        boxes.append(
            {
                "province": province,
                "x_center": x,
                "width": 0.5,
                "city_values_pct": _json_safe_array(finite),
                "q1": _json_safe_scalar(q1),
                "median": _json_safe_scalar(median),
                "q3": _json_safe_scalar(q3),
                "whisker_low": _json_safe_scalar(whisker_low),
                "whisker_high": _json_safe_scalar(whisker_high),
                "showfliers": False,
                "color": "#1F6E2D",
                "alpha": 0.35,
            }
        )
    return {
        "axis_left": _axis_report(ax_main),
        "axis_right": _axis_report(ax_secondary) if ax_secondary is not None else None,
        "bars": bars,
        "boxplots": boxes,
    }


def _phase_bar_panel_report(
    df,
    ax,
    ylabel,
    body_label,
    battery_label,
    body_color,
    battery_color,
    distance_context,
    body_colors_by_vehicle=None,
    battery_colors_by_vehicle=None,
):
    """Return stacked body/battery bar data for a selected-trip phase panel."""
    x = np.arange(len(df))
    body = df["body_g"].to_numpy(dtype=float)
    battery = df["battery_g"].to_numpy(dtype=float)
    totals = body + battery
    max_total = float(np.nanmax(totals)) if totals.size else 0.0
    return {
        "axis": _axis_report(ax),
        "ylabel": ylabel,
        "distance_context": distance_context,
        "bars": [
            {
                "vehicle": _json_safe_scalar(vehicle),
                "x_center": int(x_pos),
                "width": 0.56,
                "body_segment": {
                    "label": body_label,
                    "bottom": 0.0,
                    "height_g_per_order": _json_safe_scalar(body_value),
                    "top": _json_safe_scalar(body_value),
                    "color": (
                        body_colors_by_vehicle.get(vehicle, body_color)
                        if body_colors_by_vehicle
                        else body_color
                    ),
                },
                "battery_segment": {
                    "label": battery_label,
                    "bottom": _json_safe_scalar(body_value),
                    "height_g_per_order": _json_safe_scalar(battery_value),
                    "top": _json_safe_scalar(component_sum),
                    "color": (
                        battery_colors_by_vehicle.get(vehicle, battery_color)
                        if battery_colors_by_vehicle
                        else battery_color
                    ),
                },
                "component_sum_check": {
                    "component_sum_g": _json_safe_scalar(component_sum),
                    "cached_total_g": _json_safe_scalar(cached_total),
                    "difference_g": _json_safe_scalar(component_sum - cached_total),
                    "is_close": bool(np.isclose(component_sum, cached_total, rtol=1e-9, atol=1e-9)),
                },
                "total_label": {
                    "x": int(x_pos),
                    "y": _json_safe_scalar(component_sum + max_total * 0.035),
                    "text": f"{component_sum:.1f}",
                },
            }
            for x_pos, vehicle, body_value, battery_value, component_sum, cached_total
            in zip(
                x,
                df["vehicle"],
                body,
                battery,
                totals,
                df["cached_total_g"].to_numpy(dtype=float),
            )
        ],
    }


def _draw_trip_phase_barplot(
    ax,
    df,
    ylabel,
    body_label,
    battery_label,
    body_color,
    battery_color,
    distance_label,
    body_colors_by_vehicle=None,
    battery_colors_by_vehicle=None,
    legend_body_color=None,
    legend_battery_color=None,
    body_hatch=None,
    battery_hatch=None,
):
    """Draw stacked body/battery phase emissions for one selected OD trip."""
    x = np.arange(len(df))
    body = df["body_g"].to_numpy(dtype=float)
    battery = df["battery_g"].to_numpy(dtype=float)
    totals = body + battery
    vehicles = df["vehicle"].tolist()
    body_colors = [
        body_colors_by_vehicle.get(vehicle, body_color)
        if body_colors_by_vehicle
        else body_color
        for vehicle in vehicles
    ]
    battery_colors = [
        battery_colors_by_vehicle.get(vehicle, battery_color)
        if battery_colors_by_vehicle
        else battery_color
        for vehicle in vehicles
    ]
    edge_color = "#2f3437"

    ax.bar(
        x,
        body,
        width=0.56,
        color=body_colors,
        edgecolor=edge_color,
        linewidth=0.8,
        hatch=body_hatch,
        label=body_label,
    )
    ax.bar(
        x,
        battery,
        width=0.56,
        bottom=body,
        color=battery_colors,
        edgecolor=edge_color,
        linewidth=0.8,
        hatch=battery_hatch,
        label=battery_label,
    )

    max_total = max(float(np.nanmax(totals)), 1e-6)
    for i, total in enumerate(totals):
        ax.text(
            x[i],
            total + max_total * 0.035,
            f"{total:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(vehicles)
    ax.set_xlabel(distance_label)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max_total * 1.25)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles = [
        Patch(
            facecolor=legend_body_color or body_color,
            edgecolor=edge_color,
            linewidth=0.8,
            hatch=body_hatch,
            label=body_label,
        ),
        Patch(
            facecolor=legend_battery_color or battery_color,
            edgecolor=edge_color,
            linewidth=0.8,
            hatch=battery_hatch,
            label=battery_label,
        ),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        fontsize=9,
    )


def _model_phase_component_shares(distance_km, vehicle, phase):
    """Return body/battery shares for a vehicle phase at the selected trip distance."""
    from emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()
    if vehicle == "E-bike":
        p = lca.ebike_p
        prod = lca.calculate_ebike_production_emissions()
        speed = p.velocity
    elif vehicle == "Drone":
        p = lca.drone_p
        prod = lca.calculate_drone_production_emissions()
        speed = p.v_level
    else:
        raise ValueError(f"Unknown vehicle: {vehicle}")

    delivery_time_min = (2 * distance_km / speed) * 60 + p.loading_time_min
    deliveries_per_day = p.working_hours_per_day * 60 / delivery_time_min
    lifetime_deliveries = deliveries_per_day * p.working_days_per_year * p.lifespan

    if phase == "production":
        body_kg = prod["body_production_kg"] / p.lifespan
        battery_kg = prod["battery_production_kg"] / p.battery_lifespan
    elif phase == "eol":
        body_kg = prod["body_disposal_kg"] / p.lifespan
        battery_kg = prod["battery_disposal_kg"] / p.battery_lifespan
    else:
        raise ValueError(f"Unknown phase: {phase}")

    body_g = body_kg * p.lifespan / lifetime_deliveries * 1000
    battery_g = battery_kg * p.lifespan / lifetime_deliveries * 1000
    total_g = body_g + battery_g
    if total_g <= 0:
        return 0.5, 0.5
    return body_g / total_g, battery_g / total_g


def _selected_trip_phase_parts(sample, phase):
    """Return body/battery production or EOL parts for the selected OD row."""
    if sample is None:
        return None

    if phase == "production":
        drone_col = "drone_lifecycle_production_g"
        ebike_col = "ebike_lifecycle_production_g"
    elif phase == "eol":
        drone_col = "drone_lifecycle_eol_g"
        ebike_col = "ebike_lifecycle_eol_g"
    else:
        raise ValueError(f"Unknown phase: {phase}")

    required = {drone_col, ebike_col, "distance_km", "road_distance_m"}
    if not required.issubset(sample.index):
        return None

    ebike_total = pd.to_numeric(pd.Series([sample[ebike_col]]), errors="coerce").iloc[0]
    drone_total = pd.to_numeric(pd.Series([sample[drone_col]]), errors="coerce").iloc[0]
    if not np.isfinite(ebike_total) or not np.isfinite(drone_total):
        return None

    air_km = float(sample["distance_km"])
    road_km = float(sample["road_distance_m"]) / 1000.0
    ebike_distance_km = road_km
    if "ebike_kappa" in sample.index and pd.notna(sample["ebike_kappa"]):
        ebike_distance_km *= float(sample["ebike_kappa"])

    ebike_body_share, ebike_battery_share = _model_phase_component_shares(
        ebike_distance_km,
        "E-bike",
        phase,
    )
    drone_body_share, drone_battery_share = _model_phase_component_shares(
        air_km,
        "Drone",
        phase,
    )

    return pd.DataFrame(
        [
            {
                "vehicle": "E-bike",
                "body_g": float(ebike_total) * ebike_body_share,
                "battery_g": float(ebike_total) * ebike_battery_share,
                "cached_total_g": float(ebike_total),
                "distance_km": ebike_distance_km,
            },
            {
                "vehicle": "Drone",
                "body_g": float(drone_total) * drone_body_share,
                "battery_g": float(drone_total) * drone_battery_share,
                "cached_total_g": float(drone_total),
                "distance_km": air_km,
            },
        ]
    )


def _detour_concept_panel_report(ax):
    """Return geometry and label data for the detour-ratio concept panel."""
    import math

    grid_x = [1.0, 3.5, 6.0, 8.5]
    grid_y = [1.0, 3.0, 5.0]
    cruise_alt = 1.3

    def project(x, y, z=0.0):
        return x + 0.42 * y, 0.58 * y + z

    origin = (grid_x[0], grid_y[0])
    destination = (grid_x[3], grid_y[2])
    bike_route = [
        origin,
        (grid_x[1], grid_y[0]),
        (grid_x[1], grid_y[1]),
        (grid_x[2], grid_y[1]),
        (grid_x[2], grid_y[2]),
        destination,
    ]
    road_distance = sum(
        math.dist(a, b) for a, b in zip(bike_route[:-1], bike_route[1:])
    )
    straight_distance = math.dist(origin, destination)
    return {
        "axis": _axis_report(ax),
        "world_coordinates": {
            "origin": list(origin),
            "destination": list(destination),
            "bike_route": [list(point) for point in bike_route],
            "drone_route": [list(origin), list(destination)],
            "cruise_altitude": cruise_alt,
        },
        "projected_coordinates": {
            "origin": _json_safe_array(project(*origin)),
            "destination": _json_safe_array(project(*destination)),
            "bike_route": [
                _json_safe_array(project(point[0], point[1]))
                for point in bike_route
            ],
            "drone_route": [
                _json_safe_array(project(origin[0], origin[1], cruise_alt)),
                _json_safe_array(project(destination[0], destination[1], cruise_alt)),
            ],
        },
        "distances": {
            "road_distance_world_units": _json_safe_scalar(road_distance),
            "straight_distance_world_units": _json_safe_scalar(straight_distance),
            "detour_ratio": _json_safe_scalar(road_distance / straight_distance),
        },
        "labels": [
            {"text": "Merchant", "anchor": "origin", "marker": "circle"},
            {"text": "Customer", "anchor": "destination", "marker": "square"},
            {"text": "E-bike route on roads", "color": "#d95f35"},
            {"text": "Drone route on air space", "color": "#1f77b4"},
            {"text": "takeoff", "color": "#1f77b4"},
            {"text": "cruise", "color": "#1f77b4"},
            {"text": "land", "color": "#1f77b4"},
            {"text": "Straight-line distance", "color": "#1f77b4"},
            {"text": "Road distance", "color": "#d95f35"},
        ],
    }


def _selected_trip_distance_sample(selected_trip, router):
    """Return a distance-comparison sample for the selected OD row."""
    if selected_trip is None:
        return None
    required = {"from_lat", "from_lng", "to_lat", "to_lng", "distance_km", "road_distance_m"}
    if not required.issubset(selected_trip.index):
        return None

    origin = (float(selected_trip["from_lat"]), float(selected_trip["from_lng"]))
    destination = (float(selected_trip["to_lat"]), float(selected_trip["to_lng"]))
    path_result = (
        router.find_shortest_path(origin, destination)
        if router is not None
        else {"success": False, "path_coordinates": [], "message": "router_missing"}
    )
    return {
        "origin": origin,
        "destination": destination,
        "straight_distance": float(selected_trip["distance_km"]) * 1000.0,
        "road_distance": float(selected_trip["road_distance_m"]),
        "path_result": path_result,
    }


def _selected_trip_distance_comparison_report(ax, sample, city="chengdu"):
    """Return plotted data for the selected-trip distance-comparison panel."""
    report = {
        "axis": _axis_report(ax),
        "city": city,
        "panel_type": "distance_comparison",
        "target_display_aspect_width_over_height": 2.0,
        "route_styles": {
            "ebike_road_route": {
                "color": "#d95f35",
                "linewidth": 2.5,
                "alpha": 0.9,
                "label": "E-bike (road)",
            },
            "drone_straight_route": {
                "color": "#1f77b4",
                "linewidth": 2.0,
                "linestyle": "--",
                "alpha": 0.9,
                "label": "Drone (straight)",
            },
        },
        "endpoint_marker_styles": {
            "merchant": {
                "marker": "circle",
                "matplotlib_marker": "o",
                "color": "#2f6f4e",
                "size": 90,
                "edgecolor": "white",
                "linewidth": 1.5,
            },
            "customer": {
                "marker": "square",
                "matplotlib_marker": "s",
                "color": "#3f4b64",
                "size": 90,
                "edgecolor": "white",
                "linewidth": 1.5,
            },
        },
    }
    if sample is None:
        report["selected_trip"] = {"status": "missing_or_unplottable_selected_trip"}
        return report

    straight_m = float(sample["straight_distance"])
    road_m = float(sample["road_distance"])
    path_result = sample.get("path_result", {})
    report["selected_trip"] = {
        "status": "ok",
        "origin": _json_safe_array(sample["origin"]),
        "destination": _json_safe_array(sample["destination"]),
        "straight_distance_m": _json_safe_scalar(straight_m),
        "road_distance_m": _json_safe_scalar(road_m),
        "detour_ratio": _json_safe_scalar(road_m / straight_m if straight_m > 0 else np.nan),
        "path_success": bool(path_result.get("success", False)),
        "path_message": path_result.get("message"),
    }
    return report


def _find_overlay_axis(fig, reference_ax):
    """Return another axis sharing the same figure position, usually a twinx axis."""
    ref_pos = reference_ax.get_position()
    for candidate in fig.axes:
        if candidate is reference_ax:
            continue
        pos = candidate.get_position()
        if (
            abs(pos.x0 - ref_pos.x0) < 1e-6
            and abs(pos.y0 - ref_pos.y0) < 1e-6
            and abs(pos.width - ref_pos.width) < 1e-6
            and abs(pos.height - ref_pos.height) < 1e-6
        ):
            return candidate
    return None


def _write_regions_map_city_rank_and_drone_share_report(
    save_path,
    figsize,
    stats_csv_path,
    map_df,
    drone_df,
    reduction_col,
    reduction_pct_col,
    axes,
    color_scales,
    production_df,
    eol_df,
    production_distance_context,
    eol_distance_context,
    phase_body_colors,
    phase_battery_colors,
    median_trip_city,
    distance_comparison_sample,
    median_trip_report,
):
    """Write a JSON report for regions_map_city_rank_and_drone_share."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    (
        ax_a,
        ax_b,
        ax_c,
        ax_d,
        ax_e,
        ax_f,
        ax_g,
        inset_ax,
        share_dist_ax,
        cbar_a_ax,
        cbar_b_ax,
    ) = axes
    reduction_cmap, reduction_norm, reduction_bins, reduction_edges = (
        color_scales["reduction"]
    )
    reduction_pct_cmap, reduction_pct_norm, reduction_pct_bins, reduction_pct_edges = (
        color_scales["reduction_pct"]
    )
    reduction_pct_hist_bins = _finer_percentage_bins(reduction_pct_edges)

    payload = {
        "figure_name": "regions_map_city_rank_and_drone_share",
        "figure_path": save_path,
        "source_stats_csv_path": stats_csv_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "coordinate_convention": (
            "Plotted x/y values are Matplotlib data coordinates for the stated "
            "axis. Axis bbox values are figure-fraction coordinates."
        ),
        "panels": {
            "a_oracle_reduction_map": {
                **_map_panel_report(
                    map_df, reduction_col, reduction_cmap, reduction_norm,
                    reduction_bins, ax_a, "a",
                ),
                "inset_city_rank": _city_oracle_rank_report(
                    map_df,
                    inset_ax,
                ),
                "colorbar": _colorbar_report(
                    cbar_a_ax,
                    reduction_bins,
                    reduction_edges,
                    "E-bike - Optimal (t CO2eq / yr)",
                    reduction_cmap,
                ),
            },
            "b_oracle_reduction_pct_map": {
                **_map_panel_report(
                    drone_df, reduction_pct_col, reduction_pct_cmap, reduction_pct_norm,
                    reduction_pct_bins, ax_b, "b",
                ),
                "inset_reduction_pct_distribution": _distribution_inset_report(
                    drone_df,
                    reduction_pct_col,
                    share_dist_ax,
                    bins=reduction_pct_hist_bins,
                    bar_width=float(np.diff(reduction_pct_hist_bins).min()) * 0.85,
                    label="Optimal reduction vs e-bike (%)",
                ),
                "colorbar": _colorbar_report(
                    cbar_b_ax,
                    reduction_pct_bins,
                    reduction_pct_edges,
                    "Optimal reduction vs e-bike (%)",
                    reduction_pct_cmap,
                ),
            },
            "c_detour_ratio_concept": _detour_concept_panel_report(ax_c),
            f"d_{median_trip_city}_median_trip_distance_comparison": (
                _selected_trip_distance_comparison_report(
                    ax_d,
                    distance_comparison_sample,
                    city=median_trip_city,
                )
            ),
            "e_production_phase_selected_trip": _phase_bar_panel_report(
                production_df,
                ax_e,
                "Production\n(g CO2e/order)",
                "Body",
                "Battery",
                phase_body_colors.get("E-bike", "#d95f35"),
                phase_battery_colors.get("E-bike", "#f2a07f"),
                production_distance_context,
                body_colors_by_vehicle=phase_body_colors,
                battery_colors_by_vehicle=phase_battery_colors,
            ),
            f"f_{median_trip_city}_median_trip_emission_process": median_trip_report,
            "g_end_of_life_phase_selected_trip": _phase_bar_panel_report(
                eol_df,
                ax_g,
                "EOL\n(g CO2e/order)",
                "Body",
                "Battery",
                phase_body_colors.get("E-bike", "#d95f35"),
                phase_battery_colors.get("E-bike", "#f2a07f"),
                eol_distance_context,
                body_colors_by_vehicle=phase_body_colors,
                battery_colors_by_vehicle=phase_battery_colors,
            ),
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Regions map city rank and drone share report saved to {report_path}")


def _normalize_median_trip_subfigure_text(ax):
    """Apply consistent publication-scale typography to a median-trip panel."""
    ax.xaxis.label.set_fontsize(8.5)
    ax.yaxis.label.set_fontsize(8.5)
    ax.tick_params(axis="both", labelsize=8)
    if ax.title:
        ax.title.set_fontsize(9)
    for text_artist in ax.texts:
        text_artist.set_fontsize(8)
    legend = ax.get_legend()
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_fontsize(8)


def _save_median_trip_lifecycle_subfigures(
    production_df,
    eol_df,
    median_trip_od_df,
    median_trip_city_label,
    distance_label,
    phase_body_colors,
    phase_battery_colors,
    save_dir,
    for_latex=False,
    plot_title=False,
    only=None,
):
    """Save production, use-phase, and EOL median-trip subfigures."""
    from visualization.emission import _draw_median_trip_emission_process

    mm_per_inch = 25.4
    extension = ".pdf" if for_latex else ".png"
    os.makedirs(save_dir, exist_ok=True)

    phase_body_color = "#5f6368"
    phase_battery_color = "#c7c9cc"
    panel_specs = (
        (
            "median_trip_production_emissions",
            production_df,
            "Production emissions\n(g CO$_2$e/order)",
        ),
        (
            "median_trip_eol_emissions",
            eol_df,
            "End-of-life emissions\n(g CO$_2$e/order)",
        ),
    )
    for figure_name, phase_df, ylabel in panel_specs:
        if only is not None and figure_name not in only:
            continue
        fig, ax = plt.subplots(figsize=(43.5 / mm_per_inch, 50.0 / mm_per_inch))
        fig.subplots_adjust(left=0.34, right=0.97, bottom=0.23, top=0.78)
        if phase_df is None:
            phase_name = "production" if "production" in figure_name else "EOL"
            ax.text(
                0.5,
                0.5,
                f"{median_trip_city_label} median trip {phase_name} data\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#777777",
            )
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            _draw_trip_phase_barplot(
                ax,
                phase_df,
                ylabel=ylabel,
                body_label="Body",
                battery_label="Battery",
                body_color=phase_body_color,
                battery_color=phase_battery_color,
                distance_label="",
                body_colors_by_vehicle=phase_body_colors,
                battery_colors_by_vehicle=phase_battery_colors,
                legend_body_color=phase_body_color,
                legend_battery_color=phase_battery_color,
                battery_hatch="///",
            )
            legend_handles = [
                Patch(
                    facecolor=phase_body_color,
                    edgecolor="#2f3437",
                    linewidth=0.8,
                    label="Body",
                ),
                Patch(
                    facecolor=phase_battery_color,
                    edgecolor="#2f3437",
                    linewidth=0.8,
                    hatch="///",
                    label="Battery",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                borderaxespad=0,
                ncol=2,
                fontsize=7.5,
                handlelength=1.2,
                handletextpad=0.4,
                columnspacing=0.8,
            )
            if plot_title:
                title = (
                    "Production emissions"
                    if "production" in figure_name
                    else "End-of-life emissions"
                )
                ax.set_title(title, fontsize=9, y=1.30)
        _normalize_median_trip_subfigure_text(ax)
        output_path = os.path.join(save_dir, figure_name + extension)
        if plot_title:
            output_path = _add_title_suffix(output_path)
        fig.savefig(output_path, dpi=300, facecolor="white")
        phase_name = "production" if "production" in figure_name else "end_of_life"
        if phase_df is None:
            phase_facts = {
                "status": "unavailable",
                "reason": "Required selected-trip lifecycle columns were unavailable",
                "vehicle_components": [],
            }
        else:
            vehicle_components = []
            for _, row in phase_df.iterrows():
                body_g = float(row["body_g"])
                battery_g = float(row["battery_g"])
                cached_total_g = float(row["cached_total_g"])
                vehicle_components.append(
                    {
                        "vehicle": row["vehicle"],
                        "distance_km": float(row["distance_km"]),
                        "body_g_co2e_per_order": body_g,
                        "battery_g_co2e_per_order": battery_g,
                        "component_total_g_co2e_per_order": body_g + battery_g,
                        "cached_total_g_co2e_per_order": cached_total_g,
                        "component_sum_difference_g": (
                            body_g + battery_g - cached_total_g
                        ),
                    }
                )
            phase_facts = {
                "status": "ok",
                "vehicle_components": vehicle_components,
            }
        write_figure_report(
            output_path.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": figure_name,
                "figure_path": output_path,
                "source_city": median_trip_city_label,
                "sample_selection": "annual-sales-weighted median trip when annual_sales is available",
                "lifecycle_phase": phase_name,
                "unit": "g CO2e per order",
                **phase_facts,
            },
        )
        plt.close(fig)
        print(f"Median-trip lifecycle subfigure saved to {output_path}")

    if only is not None and "median_trip_use_phase_emissions" not in only:
        return

    fig, ax = plt.subplots(figsize=(87.0 / mm_per_inch, 50.0 / mm_per_inch))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.80)
    if median_trip_od_df is None:
        ax.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip emission data\nnot available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#777777",
        )
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    else:
        use_phase_result = _draw_median_trip_emission_process(
            ax,
            median_trip_od_df,
            plot_title=plot_title,
        )
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Use-phase emissions\n(g CO$_2$e/leg)")
        legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0,
            ncol=2,
            fontsize=8,
            handlelength=1.6,
            handletextpad=0.5,
            columnspacing=1.0,
        )
        for text_artist in ax.texts:
            if text_artist.get_text().endswith("median trip"):
                text_artist.remove()
    _normalize_median_trip_subfigure_text(ax)
    output_path = os.path.join(save_dir, "median_trip_use_phase_emissions" + extension)
    if plot_title:
        output_path = _add_title_suffix(output_path)
    fig.savefig(output_path, dpi=300, facecolor="white")
    if median_trip_od_df is None or use_phase_result is None:
        use_phase_facts = {
            "status": "unavailable",
            "reason": "Median-trip use-phase data were unavailable",
        }
    else:
        selected_sample = use_phase_result["sample"]
        use_phase_facts = {
            "status": "ok",
            "eligible_od_count": int(
                len(
                    median_trip_od_df.dropna(
                        subset=[
                            "distance_km",
                            "road_distance_m",
                            "drone_emission_kg",
                            "ebike_emission_kg",
                        ]
                    )
                )
            ),
            "selected_trip": {
                "source_row_index": _json_safe_scalar(
                    use_phase_result["sample_index"]
                ),
                "air_distance_km": float(use_phase_result["air_km"]),
                "road_distance_km": float(use_phase_result["road_km"]),
                "detour_ratio": float(
                    use_phase_result["road_km"] / use_phase_result["air_km"]
                ),
                "drone_delivery_leg_end_g_co2e": float(
                    use_phase_result["drone_y"][-1]
                ),
                "ebike_delivery_leg_end_g_co2e": float(
                    use_phase_result["ebike_y"][-1]
                ),
                "annual_sales": _json_safe_scalar(
                    selected_sample.get("annual_sales")
                ),
            },
        }
    write_figure_report(
        output_path.rsplit(".", 1)[0] + "_report.json",
        {
            "figure_name": "median_trip_use_phase_emissions",
            "figure_path": output_path,
            "source_city": median_trip_city_label,
            "sample_selection": "annual-sales-weighted median trip when annual_sales is available",
            "lifecycle_phase": "use",
            "unit": "g CO2e per delivery leg",
            **use_phase_facts,
        },
    )
    plt.close(fig)
    print(f"Median-trip use-phase subfigure saved to {output_path}")


def plot_regions_map_city_rank_and_drone_share(
    df_stats: "pd.DataFrame",
    stats_csv_path: str,
    median_trip_od_df: "pd.DataFrame" = None,
    median_trip_graph=None,
    median_trip_router=None,
    save_path: str = None,
    figsize: tuple = (18, 16),
    for_latex: bool = False,
    plot_title: bool = False,
    median_trip_city: str = "chengdu",
    beijing_od_df: "pd.DataFrame" = None,
    beijing_graph=None,
    beijing_router=None,
    subfigure_save_dir: str = None,
):
    """Maps for absolute and percentage Oracle reduction, with drone-share inset.

    Args:
        df_stats: all_region_stats DataFrame.
        stats_csv_path: Path to all_region_stats.csv (for map function).
        median_trip_od_df: OD pairs DataFrame for the median-trip panels.
        median_trip_graph: Road network graph for the median-trip distance panel.
        median_trip_router: Router for drawing the median-trip road path.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        median_trip_city: City key used in labels and report metadata.
        beijing_od_df: Deprecated alias for median_trip_od_df.
        beijing_graph: Deprecated alias for median_trip_graph.
        beijing_router: Deprecated alias for median_trip_router.
        subfigure_save_dir: Directory for standalone median-trip phase panels.
    """
    from visualization.concepts import _draw_detour_ratio_concept
    from visualization.emission import (
        _draw_median_trip_emission_process,
        _select_median_trip_emission_sample,
    )
    from visualization.regions import plot_regions_on_china_map

    if median_trip_od_df is None and beijing_od_df is not None:
        median_trip_od_df = beijing_od_df
        median_trip_city = "beijing"
    if median_trip_graph is None and beijing_graph is not None:
        median_trip_graph = beijing_graph
    if median_trip_router is None and beijing_router is not None:
        median_trip_router = beijing_router
    median_trip_city_label = median_trip_city.replace("_", " ").title()

    map_df = pd.read_csv(stats_csv_path)
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
        drone_df = df_stats.copy()
        reduction_pct_col = _weighted_reduction_column(drone_df)
    except KeyError as exc:
        print(f"Regions map city rank and drone share skipped — {exc}")
        return

    reduction_values = map_df[reduction_col].dropna()
    if reduction_values.empty:
        print("Regions map city rank and drone share skipped — no valid reduction values")
        return

    drone_df[reduction_pct_col] = pd.to_numeric(
        drone_df[reduction_pct_col], errors="coerce"
    )
    if drone_df[reduction_pct_col].dropna().empty:
        print(
            "Regions map city rank and drone share skipped — "
            "no valid reduction-percentage values"
        )
        return

    reduction_cmap, reduction_norm, reduction_bins, reduction_edges = (
        _build_absolute_reduction_log_cmap(reduction_values)
    )
    v_max = max(float(drone_df[reduction_pct_col].max(skipna=True)), 0.0)
    reduction_pct_cmap, reduction_pct_norm, reduction_pct_bins, reduction_pct_edges = (
        _build_blue_reduction_cmap(v_max)
    )
    reduction_pct_hist_bins = _finer_percentage_bins(reduction_pct_edges)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.0, 0.03, 0.55, 0.07, 0.44],
        hspace=0.0,
        wspace=0.25,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    mechanism_gs = gs[2, :].subgridspec(
        1,
        2,
        width_ratios=[1.45, 1.0],
        wspace=0.18,
    )
    ax_c = fig.add_subplot(mechanism_gs[0, 0])
    ax_d = fig.add_subplot(mechanism_gs[0, 1])
    ax_d.set_box_aspect(1 / 2.0)
    phase_gs = gs[4, :].subgridspec(
        1,
        3,
        width_ratios=[0.95, 3.0, 0.95],
        wspace=0.18,
    )
    ax_e = fig.add_subplot(phase_gs[0, 0])
    ax_f = fig.add_subplot(phase_gs[0, 1])
    ax_g = fig.add_subplot(phase_gs[0, 2])

    plot_regions_on_china_map(
        df=map_df,
        color_by=reduction_col,
        ax=ax_a,
        plot_title=False,
        cmap=reduction_cmap,
        norm=reduction_norm,
        draw_colorbar=False,
        draw_legend=False,
    )
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    for spine in ax_a.spines.values():
        spine.set_visible(False)

    inset_ax = ax_a.inset_axes([0.03, 0.04, 0.34, 0.38], zorder=10)
    _plot_city_oracle_reduction_rank(
        map_df,
        10,
        plot_title=plot_title,
        ax=inset_ax,
        invert_vertical=True,
        compact=True,
    )
    inset_ax.set_facecolor("none")
    inset_ax.patch.set_alpha(0.0)
    for name, spine in inset_ax.spines.items():
        spine.set_visible(name in {"left", "bottom"})
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.6)

    plot_regions_on_china_map(
        df=drone_df,
        color_by=reduction_pct_col,
        ax=ax_b,
        plot_title=False,
        cmap=reduction_pct_cmap,
        norm=reduction_pct_norm,
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

    share_dist_ax = ax_b.inset_axes([0.03, 0.04, 0.42, 0.38], zorder=10)
    _plot_distribution_inset(
        drone_df,
        reduction_pct_col,
        ax=share_dist_ax,
        bins=reduction_pct_hist_bins,
        xlabel="Optimal reduction (%)",
    )

    ax_a.text(-0.03, 1.02, "a", transform=ax_a.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_b.text(-0.03, 1.02, "b", transform=ax_b.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    selected_trip = (
        _select_median_trip_emission_sample(median_trip_od_df)
        if median_trip_od_df is not None
        else None
    )

    _draw_detour_ratio_concept(ax_c, plot_title=plot_title)
    ax_c.text(-0.03, 1.02, "c", transform=ax_c.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    distance_comparison_sample = _selected_trip_distance_sample(
        selected_trip,
        median_trip_router,
    )
    d_target_aspect = 2.0
    if distance_comparison_sample is not None and median_trip_graph is not None:
        from visualization.roadnet import _draw_distance_panel

        _draw_distance_panel(
            ax_d,
            distance_comparison_sample,
            median_trip_graph,
            show_legend=True,
            show_stats_box=False,
            target_aspect=d_target_aspect,
            rotate_air_diagonal=True,
        )
    else:
        ax_d.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip distance comparison\nnot available",
            transform=ax_d.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax_d.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax_d.spines.values():
            spine.set_visible(False)

    production_df = _selected_trip_phase_parts(selected_trip, "production")
    eol_df = _selected_trip_phase_parts(selected_trip, "eol")
    production_subfigure_df = production_df
    eol_subfigure_df = eol_df

    if selected_trip is not None:
        air_km = float(selected_trip["distance_km"])
        road_km = float(selected_trip["road_distance_m"]) / 1000.0
        ebike_km = road_km
        if "ebike_kappa" in selected_trip.index and pd.notna(selected_trip["ebike_kappa"]):
            ebike_km *= float(selected_trip["ebike_kappa"])
        distance_context = {
            "city": median_trip_city,
            "air_distance_km_for_drone": _json_safe_scalar(air_km),
            "road_distance_km_for_ebike": _json_safe_scalar(road_km),
            "ebike_emission_distance_km": _json_safe_scalar(ebike_km),
        }
        distance_label = "Median trip"
    else:
        distance_context = {"city": median_trip_city, "status": "missing_selected_trip"}
        distance_label = "Median trip"

    production_distance_context = distance_context
    eol_distance_context = distance_context
    phase_body_color = "#5f6368"
    phase_battery_color = "#c7c9cc"
    ebike_body_color = "#d95f35"
    ebike_battery_color = "#f2a07f"
    drone_body_color = "#1f77b4"
    drone_battery_color = "#9ecae1"
    phase_body_colors = {
        "E-bike": ebike_body_color,
        "Drone": drone_body_color,
    }
    phase_battery_colors = {
        "E-bike": ebike_battery_color,
        "Drone": drone_battery_color,
    }
    if production_df is None:
        ax_e.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip production data\nnot available",
            transform=ax_e.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax_e.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        production_df = pd.DataFrame(
            columns=["vehicle", "body_g", "battery_g", "cached_total_g"]
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
    if median_trip_od_df is None:
        ax_f.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip emission data\nnot available",
            transform=ax_f.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax_f.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        median_trip_report = {
            "axis": _axis_report(ax_f),
            "city": median_trip_city,
            "status": f"missing_{median_trip_city}_od",
        }
    else:
        median_trip_data = _draw_median_trip_emission_process(
            ax_f,
            median_trip_od_df,
            plot_title=plot_title,
        )
        ax_f.set_xlabel("Distance (km)", fontsize=12)
        ax_f.set_ylabel("Use-phase emission\n(g CO$_2$e/leg)", fontsize=12)
        median_trip_report = {
            "axis": _axis_report(ax_f),
            "city": median_trip_city,
            "status": "ok" if median_trip_data is not None else "no_valid_rows",
        }
        if median_trip_data is not None:
            median_trip_report.update(
                {
                    "city": median_trip_city,
                    "air_km": _json_safe_scalar(median_trip_data["air_km"]),
                    "road_km": _json_safe_scalar(median_trip_data["road_km"]),
                    "drone_curve": [
                        {"x_km": _json_safe_scalar(x), "y_g": _json_safe_scalar(y)}
                        for x, y in zip(
                            median_trip_data["drone_x"],
                            median_trip_data["drone_y"],
                        )
                    ],
                    "ebike_curve": [
                        {"x_km": _json_safe_scalar(x), "y_g": _json_safe_scalar(y)}
                        for x, y in zip(
                            median_trip_data["ebike_x"],
                            median_trip_data["ebike_y"],
                        )
                    ],
                }
            )
    if eol_df is None:
        ax_g.text(
            0.5,
            0.5,
            f"{median_trip_city_label} median trip EOL data\nnot available",
            transform=ax_g.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#777777",
        )
        ax_g.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        eol_df = pd.DataFrame(
            columns=["vehicle", "body_g", "battery_g", "cached_total_g"]
        )
    else:
        _draw_trip_phase_barplot(
            ax_g,
            eol_df,
            ylabel="EOL\n(g CO$_2$e/order)",
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
    ax_d.text(-0.03, 1.02, "d", transform=ax_d.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_e.text(-0.03, 1.02, "e", transform=ax_e.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_f.text(-0.03, 1.02, "f", transform=ax_f.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")
    ax_g.text(-0.03, 1.02, "g", transform=ax_g.transAxes,
              fontsize=16, fontweight="bold", va="bottom", ha="right")

    if subfigure_save_dir:
        _save_median_trip_lifecycle_subfigures(
            production_df=production_subfigure_df,
            eol_df=eol_subfigure_df,
            median_trip_od_df=median_trip_od_df,
            median_trip_city_label=median_trip_city_label,
            distance_label=distance_label,
            phase_body_colors=phase_body_colors,
            phase_battery_colors=phase_battery_colors,
            save_dir=subfigure_save_dir,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if plot_title:
        fig.suptitle(
            "Regional Optimal Reduction, Drone Share, and Lifecycle Mechanisms",
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()

    fig.canvas.draw()
    a_pos = ax_a.get_position()
    cbar_a_ax = fig.add_axes((a_pos.x1 + 0.005, a_pos.y0, 0.010, a_pos.height))
    cbar_a, n_bands_a = _draw_discrete_reduction_colorbar(
        fig,
        cbar_a_ax,
        reduction_cmap,
        reduction_norm,
        reduction_bins,
        reduction_edges,
        "E-bike - Optimal (t CO₂eq / yr)",
    )
    _annotate_unstudied(fig, cbar_a, cbar_a_ax, n_bands_a)

    b_pos = ax_b.get_position()
    cbar_b_ax = fig.add_axes((b_pos.x1 + 0.005, b_pos.y0, 0.010, b_pos.height))
    cbar_b, n_bands_b = _draw_discrete_reduction_colorbar(
        fig,
        cbar_b_ax,
        reduction_pct_cmap,
        reduction_pct_norm,
        reduction_pct_bins,
        reduction_pct_edges,
        "Optimal reduction vs e-bike (%)",
    )
    _annotate_unstudied(fig, cbar_b, cbar_b_ax, n_bands_b)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        _write_regions_map_city_rank_and_drone_share_report(
            save_path=save_path,
            figsize=figsize,
            stats_csv_path=stats_csv_path,
            map_df=map_df,
            drone_df=drone_df,
            reduction_col=reduction_col,
            reduction_pct_col=reduction_pct_col,
            axes=(
                ax_a,
                ax_b,
                ax_c,
                ax_d,
                ax_e,
                ax_f,
                ax_g,
                inset_ax,
                share_dist_ax,
                cbar_a_ax,
                cbar_b_ax,
            ),
            color_scales={
                "reduction": (
                    reduction_cmap,
                    reduction_norm,
                    reduction_bins,
                    reduction_edges,
                ),
                "reduction_pct": (
                    reduction_pct_cmap,
                    reduction_pct_norm,
                    reduction_pct_bins,
                    reduction_pct_edges,
                ),
            },
            production_df=production_df,
            eol_df=eol_df,
            production_distance_context=production_distance_context,
            eol_distance_context=eol_distance_context,
            phase_body_colors=phase_body_colors,
            phase_battery_colors=phase_battery_colors,
            median_trip_city=median_trip_city,
            distance_comparison_sample=distance_comparison_sample,
            median_trip_report=median_trip_report,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map city rank and drone share saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _plot_distribution_inset(
    df: pd.DataFrame,
    value_col: str,
    ax: "plt.Axes",
    bins: np.ndarray,
    xlabel: str,
):
    """Draw compact city-count distribution for a percentage metric."""
    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if values.empty:
        ax.axis("off")
        return

    clipped = values.clip(lower=float(bins[0]), upper=float(bins[-1]))
    counts, edges = np.histogram(clipped, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    bar_width = float(np.diff(edges).min()) * 0.85

    ax.bar(
        centres,
        counts,
        width=bar_width,
        color="#91bfdb",
        edgecolor="#08306b",
        linewidth=0.5,
        alpha=0.85,
        zorder=2,
    )

    x_curve = np.linspace(float(centres.min()), float(centres.max()), 200)
    if len(centres) >= 4:
        from scipy.interpolate import make_interp_spline

        spline = make_interp_spline(centres, counts, k=3)
        y_curve = np.clip(spline(x_curve), 0, None)
    else:
        y_curve = np.interp(x_curve, centres, counts)
    ax.plot(x_curve, y_curve, color="#08306b", linewidth=1.2, zorder=3)

    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_ylim(0, max(counts.max() * 1.18, 1))
    ax.set_xlabel(xlabel, fontsize=7, labelpad=1)
    ax.set_ylabel("Cities", fontsize=7, labelpad=1)
    if len(edges) > 2:
        ax.set_xticks([float(edges[0]), float(edges[len(edges) // 2]), float(edges[-1])])
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1)
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)
    for name, spine in ax.spines.items():
        spine.set_visible(name in {"left", "bottom"})
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.6)


def _plot_drone_share_distribution_inset(
    df: pd.DataFrame,
    drone_share_col: str,
    ax: "plt.Axes",
):
    """Draw compact city-count distribution by optimal drone-share bin."""
    _plot_distribution_inset(
        df,
        drone_share_col,
        ax=ax,
        bins=np.arange(0, 101, 10, dtype=float),
        xlabel="Drone share (%)",
    )


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
    bar_color = "#1F6E2D"

    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    ax.barh(y, reductions, height=0.6, color=bar_color, alpha=0.88, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=7 if compact else 10)
    xlabel = (
        "Optimal reduction (%)"
        if reduction_metric == "percentage"
        else "E-bike - Optimal (t CO₂eq / yr)"
    )
    ax.set_xlabel(xlabel, fontsize=7 if compact else 12)
    if compact:
        ax.grid(False)
    else:
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
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
                va="center", ha="left", fontsize=9, color="black", zorder=4,
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


def plot_city_oracle_reduction_rank_top10(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (3.0, 5.0),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Compact horizontal bars for the top cities by Oracle reduction.

    Args:
        df: all_region_stats DataFrame.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
    """
    _plot_city_oracle_reduction_rank(
        df,
        10,
        save_path=save_path,
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
        invert_vertical=True,
        compact=True,
    )


def _plot_city_ebike_rank(
    df: pd.DataFrame,
    mode: str,
    n: int,
    save_path: str = None,
    figsize: tuple = (10, 7),
    for_latex: bool = False,
    plot_title: bool = False,
    ax: "plt.Axes" = None,
    invert_vertical: bool = False,
    compact: bool = False,
):
    """Shared horizontal grouped-bar plot for top/bottom city ebike rankings."""
    needed = ["region_name_en", "total_ebike_lifecycle", "oracle_min_lifecycle_kg"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        print(f"City ebike rank ({mode}) skipped — missing columns {missing}")
        return
    sub = df[needed].copy()
    sub["total_ebike_lifecycle"] = pd.to_numeric(sub["total_ebike_lifecycle"], errors="coerce")
    sub["oracle_min_lifecycle_kg"] = pd.to_numeric(sub["oracle_min_lifecycle_kg"], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        print(f"City ebike rank ({mode}) skipped — no rows with ebike + oracle-min emissions")
        return

    ascending = mode == "bottom"
    ranked = sub.sort_values("total_ebike_lifecycle", ascending=ascending).head(n)
    # Largest bar at top for readability
    ranked = ranked.sort_values("total_ebike_lifecycle", ascending=True).reset_index(drop=True)
    if invert_vertical:
        ranked = ranked.iloc[::-1].reset_index(drop=True)

    cities = ranked["region_name_en"].tolist()
    # Raw stats are kg CO₂eq; convert to tonnes for display.
    ebike_t = ranked["total_ebike_lifecycle"].values / 1e3
    oracle_t = ranked["oracle_min_lifecycle_kg"].values / 1e3

    y = np.arange(len(cities))
    bar_h = 0.6
    ebike_color = "#E05C2B"
    oracle_color = "#2CA02C"

    _external_ax = ax is not None
    if _external_ax:
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots(figsize=figsize)
    # E-bike (orange) is the full baseline; oracle minimum (green) overlays the residual.
    ax.barh(y, ebike_t, height=bar_h, color=ebike_color, alpha=0.85, label="E-bike (total)", zorder=2)
    ax.barh(y, oracle_t, height=bar_h, color=oracle_color, alpha=0.95, label="Optimal", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(cities, fontsize=7 if compact else 10)
    ax.set_xlabel("Annual lifecycle emission (t CO₂eq)", fontsize=7 if compact else 12)
    if compact:
        ax.grid(False)
    else:
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=7 if compact else 10, pad=1 if compact else 3)
    ax.tick_params(axis="y", pad=1 if compact else 3)
    ax.legend(loc="center right" if compact else "lower right",
              fontsize=6 if compact else 10, frameon=not compact)

    xmax = ebike_t.max()
    pad = xmax * 0.01
    if not compact:
        for i, (e, o) in enumerate(zip(ebike_t, oracle_t)):
            reduction = e - o
            fmt = "{:,.2f}" if reduction < 10 else ("{:,.1f}" if reduction < 100 else "{:,.0f}")
            reduction_pct = reduction / e * 100 if e > 0 else 0
            ax.text(e + pad, i, f"{fmt.format(reduction)} (−{reduction_pct:.1f}%)",
                    va="center", ha="left",
                    fontsize=9, color="black", zorder=4)
    ax.set_xlim(0, xmax * (1.04 if compact else 1.22))

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
    """Horizontal grouped bars: top-10 cities by annual e-bike lifecycle emission, with oracle minimum alongside.

    Args:
        df: all_region_stats DataFrame with region_name_en, total_ebike_lifecycle, oracle_min_lifecycle_kg.
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
    """Horizontal grouped bars: bottom-10 cities by annual e-bike lifecycle emission, with oracle minimum alongside.

    Args:
        df: all_region_stats DataFrame with region_name_en, total_ebike_lifecycle, oracle_min_lifecycle_kg.
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
        alpha=0.9,
        linewidth=0,
        zorder=2,
    )

    ax.set_xlabel("City rank", fontsize=7)
    ax.set_ylabel("Optimal reduction vs e-bike (pp)", fontsize=7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
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
            "facecolor": "white",
            "edgecolor": "0.55",
            "alpha": 1.0,
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
                color="#111111",
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
                colors="0.25",
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


def plot_city_population_detour_oracle_reduction_bubble(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (2.31, 3.39),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Bubble chart of Oracle emission reduction and city population.

    Args:
        df: Enriched all_region_stats DataFrame with socioeconomic and Oracle
            emission columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required = {
        "population_10k",
        "total_ebike_lifecycle",
        "province",
    }
    if "oracle_min_lifecycle_kg" not in df.columns and not any(
        col in df.columns for col in WEIGHTED_REDUCTION_COLUMNS
    ):
        print(
            "City population Oracle bubble skipped — missing Oracle reduction columns"
        )
        return
    missing = required - set(df.columns)
    if missing:
        print(
            "City population Oracle bubble skipped — "
            f"missing columns {sorted(missing)}"
        )
        return

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"].copy()
    reduction_pct_col = _weighted_reduction_column(plot_df)
    reduction_t_col = _oracle_reduction_tonnes_column(plot_df)

    name_col = (
        "region_name_en"
        if "region_name_en" in plot_df.columns
        else ("region_key" if "region_key" in plot_df.columns else None)
    )
    needed = [
        "population_10k",
        "province",
        reduction_pct_col,
        reduction_t_col,
    ]
    if name_col:
        needed.append(name_col)
    plot_df = plot_df[needed].copy()
    for col in ("population_10k", reduction_pct_col, reduction_t_col):
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(
        subset=["population_10k", reduction_pct_col, reduction_t_col]
    )
    plot_df = plot_df[
        (plot_df["population_10k"] > 0)
        & (plot_df[reduction_pct_col] > 0)
        & (plot_df[reduction_t_col] > 0)
    ].copy()
    plot_df["_macro_region"] = plot_df["province"].map(macro_region_for_province)
    plot_df = plot_df[plot_df["_macro_region"].notna()].copy()
    if plot_df.empty:
        print(
            "City population Oracle bubble skipped — "
            "no eligible city rows with known macro-region"
        )
        return
    plot_df["_macro_region_key"] = plot_df["_macro_region"].map(lambda spec: spec["key"])
    plot_df["_macro_region_color"] = plot_df["_macro_region"].map(
        lambda spec: spec["color"]
    )

    plot_df["population_million"] = plot_df["population_10k"] / 100.0
    reduction_t = plot_df[reduction_t_col]
    population_million = plot_df["population_million"]
    size_min, size_max = 4.0, 50.0
    if np.isclose(population_million.max(), population_million.min()):
        sizes = np.full(len(plot_df), (size_min + size_max) / 2)
    else:
        scaled = np.sqrt(population_million / population_million.max())
        sizes = size_min + scaled * (size_max - size_min)
    plot_df["_bubble_size"] = sizes

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        plot_df[reduction_t_col],
        plot_df[reduction_pct_col],
        s=plot_df["_bubble_size"],
        c=plot_df["_macro_region_color"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.55,
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Annual reduction (t CO2e yr$^{-1}$)", fontsize=7)
    ax.set_ylabel("Lifecycle reduction (%)", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylim(top=50)
    ax.set_axisbelow(True)

    legend_values = np.nanquantile(population_million, [0.25, 0.5, 0.75])
    legend_values = np.unique(np.round(legend_values, 1))
    handles = []
    labels = []
    for value in legend_values:
        if value <= 0:
            continue
        marker_size = (
            size_min
            + np.sqrt(value / population_million.max()) * (size_max - size_min)
        )
        handles.append(
            ax.scatter(
                [],
                [],
                s=marker_size,
                facecolor="#BDBDBD",
                edgecolor="#FFFFFF",
                linewidth=0.55,
                alpha=0.78,
            )
        )
        labels.append(f"{value:,.1f}M")
    if handles:
        size_legend = ax.legend(
            handles,
            labels,
            title="Population",
            loc="upper left",
            frameon=True,
            framealpha=0.92,
            fontsize=6.5,
            title_fontsize=6.5,
            borderaxespad=0.3,
            borderpad=0.25,
            labelspacing=0.1,
            handletextpad=0.35,
            ncol=1,
        )

    if name_col:
        from adjustText import adjust_text

        labeled = plot_df[
            plot_df[name_col].isin(ORACLE_REDUCTION_LABEL_CITIES)
        ].copy()
        labeled["_label_order"] = pd.Categorical(
            labeled[name_col],
            categories=ORACLE_REDUCTION_LABEL_CITIES,
            ordered=True,
        )
        labeled = labeled.sort_values("_label_order")
        ax.scatter(
            labeled[reduction_t_col],
            labeled[reduction_pct_col],
            s=labeled["_bubble_size"],
            facecolors="none",
            edgecolors="#202020",
            linewidth=0.7,
            zorder=4,
        )
        label_texts = []
        for _, row in labeled.iterrows():
            label_texts.append(
                ax.text(
                    row[reduction_t_col],
                    row[reduction_pct_col],
                    str(row[name_col]),
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#303030",
                    zorder=5,
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.68,
                    },
                )
            )
        adjust_text(
            label_texts,
            x=labeled[reduction_t_col].to_numpy(dtype=float),
            y=labeled[reduction_pct_col].to_numpy(dtype=float),
            ax=ax,
            expand=(1.15, 1.25),
            force_text=(0.2, 0.3),
            force_static=(0.1, 0.2),
            ensure_inside_axes=True,
            arrowprops={
                "arrowstyle": "-",
                "color": "0.35",
                "lw": 0.5,
            },
        )

    if plot_title:
        fig.suptitle(
            "Optimal Emission Reduction and Population",
            fontsize=14,
            fontweight="bold",
        )

    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.17)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        macro_region_records = []
        for spec in CHINA_MACRO_REGION_SPECS:
            rows = plot_df[plot_df["_macro_region_key"] == spec["key"]]
            if rows.empty:
                continue
            macro_region_records.append(
                {
                    "macro_region": spec["key"],
                    "label": spec["label"],
                    "city_count": int(len(rows)),
                    "annual_reduction_t_co2e": _writing_numeric_summary(
                        rows[reduction_t_col]
                    ),
                    "lifecycle_reduction_pct": _writing_numeric_summary(
                        rows[reduction_pct_col]
                    ),
                    "population_million": _writing_numeric_summary(
                        rows["population_million"]
                    ),
                }
            )
        named_city_records = []
        if name_col:
            named_rows = plot_df[
                plot_df[name_col].isin(ORACLE_REDUCTION_LABEL_CITIES)
            ].copy()
            for _, row in named_rows.sort_values(name_col).iterrows():
                named_city_records.append(
                    {
                        "city": row[name_col],
                        "province": row["province"],
                        "macro_region": row["_macro_region_key"],
                        "population_million": float(row["population_million"]),
                        "annual_reduction_t_co2e": float(row[reduction_t_col]),
                        "lifecycle_reduction_pct": float(row[reduction_pct_col]),
                    }
                )
        write_figure_report(
            save_path.rsplit(".", 1)[0] + "_report.json",
            {
                "figure_name": "city_population_detour_oracle_reduction_bubble",
                "figure_path": save_path,
                "source_scope": "Eligible city-level study regions with positive population and optimal reductions",
                "city_count": int(len(plot_df)),
                "encodings": {
                    "x": {
                        "field": reduction_t_col,
                        "label": "Annual reduction",
                        "unit": "t CO2e per year",
                        "scale": "logarithmic",
                    },
                    "y": {
                        "field": reduction_pct_col,
                        "label": "Lifecycle reduction",
                        "unit": "percent versus e-bike",
                    },
                    "bubble_area": {
                        "field": "population_10k",
                        "display_unit": "million people",
                    },
                    "color": "China macro-region",
                },
                "overall": {
                    "annual_reduction_t_co2e": _writing_numeric_summary(
                        plot_df[reduction_t_col]
                    ),
                    "lifecycle_reduction_pct": _writing_numeric_summary(
                        plot_df[reduction_pct_col]
                    ),
                    "population_million": _writing_numeric_summary(
                        plot_df["population_million"]
                    ),
                },
                "macro_regions": macro_region_records,
                "labeled_cities": named_city_records,
            },
        )
        print(f"City population Oracle bubble saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def _prepare_city_reduction_distribution_data(df: pd.DataFrame):
    """Return positive city-level Oracle reduction metrics and audit counts."""
    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"].copy()
    source_city_count = len(plot_df)

    try:
        reduction_pct_col = _weighted_reduction_column(plot_df)
        reduction_t_col = _oracle_reduction_tonnes_column(plot_df)
    except KeyError as exc:
        print(f"City reduction distributions skipped — {exc}")
        return None

    for column in (reduction_pct_col, reduction_t_col):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    complete = plot_df[
        np.isfinite(plot_df[reduction_pct_col])
        & np.isfinite(plot_df[reduction_t_col])
    ].copy()
    eligible = complete[
        (complete[reduction_pct_col] > 0)
        & (complete[reduction_t_col] > 0)
    ].copy()
    if len(eligible) < 2:
        print(
            "City reduction distributions skipped — "
            "fewer than two cities have positive finite reductions"
        )
        return None

    audit = {
        "source_city_count": int(source_city_count),
        "complete_metric_count": int(len(complete)),
        "eligible_city_count": int(len(eligible)),
        "excluded_missing_or_nonfinite_count": int(source_city_count - len(complete)),
        "excluded_nonpositive_count": int(len(complete) - len(eligible)),
    }
    return eligible, reduction_pct_col, reduction_t_col, audit


def _city_reduction_highlight_records(plot_df, value_col):
    """Return the map-labeled cities that remain eligible for a metric."""
    if "region_name_en" not in plot_df.columns:
        return [], list(ORACLE_REDUCTION_LABEL_CITIES)

    name_key = plot_df["region_name_en"].astype(str).str.casefold()
    records = []
    missing = []
    for city in ORACLE_REDUCTION_LABEL_CITIES:
        rows = plot_df[name_key == city.casefold()]
        if rows.empty:
            missing.append(city)
            continue
        row = rows.iloc[0]
        value = float(row[value_col])
        if not np.isfinite(value):
            missing.append(city)
            continue
        records.append(
            {
                "city": city,
                "region_key": _json_safe_scalar(row.get("region_key")),
                "value": value,
            }
        )
    return records, missing


def _city_reduction_box_statistics(values, log10_space=False):
    """Return Tukey box statistics, optionally calculated in log10 space."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if log10_space:
        valid &= values > 0
    values = values[valid]
    if values.size < 2:
        raise ValueError("At least two finite values are required for a boxplot")

    calculated = np.log10(values) if log10_space else values.copy()
    q1, median, q3 = np.quantile(calculated, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    whisker_low = float(calculated[calculated >= lower_fence].min())
    whisker_high = float(calculated[calculated <= upper_fence].max())
    outlier_mask = (calculated < lower_fence) | (calculated > upper_fence)

    def display_value(value):
        return float(10 ** value) if log10_space else float(value)

    box_stats = {
        "q1": display_value(q1),
        "med": display_value(median),
        "q3": display_value(q3),
        "whislo": display_value(whisker_low),
        "whishi": display_value(whisker_high),
        "fliers": [display_value(value) for value in calculated[outlier_mask]],
    }
    report = {
        "calculation_space": "log10" if log10_space else "linear",
        "tukey_iqr_multiplier": 1.5,
        "q1": box_stats["q1"],
        "median": box_stats["med"],
        "q3": box_stats["q3"],
        "whisker_low": box_stats["whislo"],
        "whisker_high": box_stats["whishi"],
        "lower_fence": display_value(lower_fence),
        "upper_fence": display_value(upper_fence),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_values": box_stats["fliers"],
        "outliers_displayed": True,
    }
    return box_stats, report


def _city_reduction_outlier_offsets(
    outlier_values,
    value_limits,
    log10_space=False,
):
    """Return deterministic narrow jitter for overlapping Tukey outliers."""
    outlier_values = np.asarray(outlier_values, dtype=float)
    if outlier_values.size == 0:
        return np.array([], dtype=float)

    transformed = (
        np.log10(outlier_values) if log10_space else outlier_values.copy()
    )
    transformed_limits = (
        np.log10(np.asarray(value_limits, dtype=float))
        if log10_space
        else np.asarray(value_limits, dtype=float)
    )
    span = max(float(np.ptp(transformed_limits)), np.finfo(float).eps)
    collision_distance = 0.018 * span
    candidate_offsets = (0.0, -0.055, 0.055, -0.11, 0.11)
    offsets = np.zeros_like(transformed)
    placed = []

    for index in np.argsort(transformed, kind="stable"):
        y_value = float(transformed[index])
        nearby_offsets = [
            other_offset
            for other_offset, other_y in placed
            if abs(y_value - other_y) < collision_distance
        ]
        chosen = max(
            candidate_offsets,
            key=lambda candidate: (
                min(
                    (abs(candidate - other_offset) for other_offset in nearby_offsets),
                    default=np.inf,
                ),
                -candidate_offsets.index(candidate),
            ),
        )
        offsets[index] = chosen
        placed.append((float(chosen), y_value))

    return offsets


def _city_reduction_outlier_records(plot_df, value_col, box_report):
    """Return city-level records outside the reported Tukey fences."""
    lower_fence = float(box_report["lower_fence"])
    upper_fence = float(box_report["upper_fence"])
    highlighted_names = {
        city.casefold() for city in ORACLE_REDUCTION_LABEL_CITIES
    }
    records = []
    for _, row in plot_df.iterrows():
        value = float(row[value_col])
        if lower_fence <= value <= upper_fence:
            continue
        city = _json_safe_scalar(row.get("region_name_en"))
        records.append(
            {
                "city": city,
                "region_key": _json_safe_scalar(row.get("region_key")),
                "value": value,
                "direction": "lower" if value < lower_fence else "upper",
                "highlighted_map_city": str(city).casefold() in highlighted_names,
            }
        )
    return records


def _city_reduction_highlight_offsets(
    highlight_values,
    value_limits,
    log10_space=False,
):
    """Place highlighted observations symmetrically beside a vertical box."""
    highlight_values = np.asarray(highlight_values, dtype=float)
    if highlight_values.size == 0:
        return np.array([], dtype=float)

    transformed = (
        np.log10(highlight_values) if log10_space else highlight_values.copy()
    )
    transformed_limits = (
        np.log10(np.asarray(value_limits, dtype=float))
        if log10_space
        else np.asarray(value_limits, dtype=float)
    )
    span = max(float(np.ptp(transformed_limits)), np.finfo(float).eps)
    y_collision = 0.045 * span
    x_collision = 0.16
    offsets = np.zeros_like(transformed)
    placed = []

    for sequence, index in enumerate(np.argsort(transformed, kind="stable")):
        y_value = float(transformed[index])
        preferred_sign = 1.0 if sequence % 2 == 0 else -1.0
        candidates = (
            preferred_sign * 0.24,
            -preferred_sign * 0.24,
            preferred_sign * 0.38,
            -preferred_sign * 0.38,
            preferred_sign * 0.10,
            -preferred_sign * 0.10,
        )
        scored = []
        for candidate_order, candidate in enumerate(candidates):
            if not placed:
                minimum_distance = np.inf
            else:
                minimum_distance = min(
                    ((candidate - other_x) / x_collision) ** 2
                    + ((y_value - other_y) / y_collision) ** 2
                    for other_x, other_y in placed
                )
            scored.append((minimum_distance, -candidate_order, candidate))
        separated = [item for item in scored if item[0] >= 1.0]
        chosen = max(separated or scored)[-1]
        offsets[index] = chosen
        placed.append((float(chosen), y_value))

    return offsets


def _city_reduction_violin_highlight_offsets(
    highlight_values,
    value_limits,
    log10_space=False,
    density_grid=None,
    half_width=None,
    center_avoidance_interval=None,
):
    """Place violin highlights in compact groups centred on the density."""
    highlight_values = np.asarray(highlight_values, dtype=float)
    if highlight_values.size == 0:
        return np.array([], dtype=float)

    transformed = (
        np.log10(highlight_values) if log10_space else highlight_values.copy()
    )
    transformed_limits = (
        np.log10(np.asarray(value_limits, dtype=float))
        if log10_space
        else np.asarray(value_limits, dtype=float)
    )
    span = max(float(np.ptp(transformed_limits)), np.finfo(float).eps)
    collision_distance = 0.045 * span
    ordered_indices = np.argsort(transformed, kind="stable")
    groups = []
    current_group = [int(ordered_indices[0])]

    for index in ordered_indices[1:]:
        index = int(index)
        previous_index = current_group[-1]
        if transformed[index] - transformed[previous_index] < collision_distance:
            current_group.append(index)
        else:
            groups.append(current_group)
            current_group = [index]
    groups.append(current_group)

    offsets = np.zeros_like(transformed)
    for group in groups:
        if len(group) == 1:
            continue
        maximum_offset = min(0.28, 0.06 * (len(group) - 1))
        group_offsets = np.linspace(-maximum_offset, maximum_offset, len(group))
        offsets[group] = group_offsets

    maximum_offsets = None
    if density_grid is not None or half_width is not None:
        if density_grid is None or half_width is None:
            raise ValueError("density_grid and half_width must be provided together")
        density_grid = np.asarray(density_grid, dtype=float)
        half_width = np.asarray(half_width, dtype=float)
        if density_grid.shape != half_width.shape or density_grid.ndim != 1:
            raise ValueError("density_grid and half_width must be matching 1D arrays")
        local_half_width = np.interp(
            transformed,
            density_grid,
            half_width,
            left=0.0,
            right=0.0,
        )
        maximum_offsets = 0.95 * np.maximum(local_half_width, 0.0)
        offsets = np.sign(offsets) * np.minimum(np.abs(offsets), maximum_offsets)

    if center_avoidance_interval is not None:
        interval = np.asarray(center_avoidance_interval, dtype=float)
        if interval.shape != (2,) or not np.all(np.isfinite(interval)):
            raise ValueError(
                "center_avoidance_interval must contain two finite values"
            )
        lower, upper = np.sort(interval)
        avoid_indices = np.flatnonzero(
            (highlight_values >= lower) & (highlight_values <= upper)
        )
        avoid_indices = avoid_indices[
            np.argsort(transformed[avoid_indices], kind="stable")
        ]
        for sequence, index in enumerate(avoid_indices):
            magnitude = max(abs(float(offsets[index])), 0.06)
            if maximum_offsets is not None:
                magnitude = min(magnitude, float(maximum_offsets[index]))
            sign = -1.0 if sequence % 2 == 0 else 1.0
            offsets[index] = sign * magnitude

    return offsets


def _draw_city_reduction_boxplot(
    ax,
    values,
    value_limits,
    y_label,
    log10_space=False,
    highlight_values=None,
):
    """Draw a vertical Tukey boxplot with selected-city highlights."""
    box_stats, box_report = _city_reduction_box_statistics(
        values,
        log10_space=log10_space,
    )
    curve_color = "#1F6F78"
    ax.bxp(
        [box_stats],
        positions=[1.0],
        widths=0.38,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={
            "facecolor": curve_color,
            "edgecolor": curve_color,
            "alpha": 0.22,
            "linewidth": 1.25,
        },
        whiskerprops={"color": curve_color, "linewidth": 1.15},
        capprops={"color": curve_color, "linewidth": 1.15},
        medianprops={"color": curve_color, "linewidth": 2.2},
    )

    highlight_values = np.asarray(
        [] if highlight_values is None else highlight_values,
        dtype=float,
    )
    valid_highlights = np.isfinite(highlight_values)
    if log10_space:
        valid_highlights &= highlight_values > 0
    highlight_values = highlight_values[valid_highlights]

    outlier_values = np.asarray(box_stats["fliers"], dtype=float)
    if outlier_values.size and highlight_values.size:
        highlighted_outlier_mask = np.any(
            np.isclose(
                outlier_values[:, np.newaxis],
                highlight_values[np.newaxis, :],
                rtol=1e-10,
                atol=0.0,
            ),
            axis=1,
        )
    else:
        highlighted_outlier_mask = np.zeros(outlier_values.size, dtype=bool)
    neutral_outliers = outlier_values[~highlighted_outlier_mask]
    neutral_outlier_offsets = _city_reduction_outlier_offsets(
        neutral_outliers,
        value_limits,
        log10_space=log10_space,
    )
    if neutral_outliers.size:
        ax.scatter(
            1.0 + neutral_outlier_offsets,
            neutral_outliers,
            s=7,
            marker="o",
            facecolor="#B8BDC3",
            edgecolor="white",
            linewidth=0.25,
            alpha=0.82,
            zorder=3,
            clip_on=True,
        )

    highlight_offsets = _city_reduction_highlight_offsets(
        highlight_values,
        value_limits,
        log10_space=log10_space,
    )
    if highlight_values.size:
        ax.scatter(
            1.0 + highlight_offsets,
            highlight_values,
            s=18,
            marker="o",
            facecolor=ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.96,
            zorder=5,
            clip_on=True,
        )

    if log10_space:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 1, 10, 100, 1000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylim(*value_limits)
    ax.set_xlim(0.45, 1.55)
    ax.set_ylabel(y_label, fontsize=7)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=6.5, length=2.5, pad=1.5)
    ax.grid(False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    box_report["neutral_outlier_count"] = int(neutral_outliers.size)
    box_report["highlighted_outlier_count"] = int(highlighted_outlier_mask.sum())
    box_report["outlier_marker"] = {
        "shape": "circle",
        "facecolor": "#B8BDC3",
        "edgecolor": "white",
        "size_points_squared": 7,
        "layout": "deterministic narrow horizontal jitter",
    }
    return box_report, highlight_offsets


def _draw_city_reduction_violin(
    ax,
    values,
    value_limits,
    y_label,
    log10_space=False,
    highlight_values=None,
    avoid_summary_overlap=False,
):
    """Draw a vertical KDE violin with summaries and selected-city markers."""
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    valid_values = np.isfinite(values)
    if log10_space:
        valid_values &= values > 0
    values = values[valid_values]
    transformed = np.log10(values) if log10_space else values.copy()
    density_grid = np.linspace(transformed.min(), transformed.max(), 512)
    density = gaussian_kde(transformed, bw_method="scott")(density_grid)
    density[[0, -1]] = 0.0
    half_width = 0.32 * density / density.max()
    display_grid = 10 ** density_grid if log10_space else density_grid
    curve_color = "#1F6F78"

    ax.fill_betweenx(
        display_grid,
        1.0 - half_width,
        1.0 + half_width,
        color=curve_color,
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        1.0 - half_width,
        display_grid,
        color=curve_color,
        linewidth=1.25,
        zorder=2,
    )
    ax.plot(
        1.0 + half_width,
        display_grid,
        color=curve_color,
        linewidth=1.25,
        zorder=2,
    )

    box_stats, box_report = _city_reduction_box_statistics(
        values,
        log10_space=log10_space,
    )
    ax.plot(
        [1.0, 1.0],
        [box_stats["q1"], box_stats["q3"]],
        color=curve_color,
        linewidth=2.6,
        solid_capstyle="round",
        zorder=3,
    )
    ax.scatter(
        [1.0],
        [box_stats["med"]],
        s=14,
        facecolor="white",
        edgecolor=curve_color,
        linewidth=0.8,
        zorder=4,
    )

    highlight_values = np.asarray(
        [] if highlight_values is None else highlight_values,
        dtype=float,
    )
    valid_highlights = np.isfinite(highlight_values)
    if log10_space:
        valid_highlights &= highlight_values > 0
    highlight_values = highlight_values[valid_highlights]

    highlight_offsets = _city_reduction_violin_highlight_offsets(
        highlight_values,
        value_limits,
        log10_space=log10_space,
        density_grid=density_grid,
        half_width=half_width,
        center_avoidance_interval=(box_stats["q1"], box_stats["q3"])
        if avoid_summary_overlap
        else None,
    )
    if highlight_values.size:
        ax.scatter(
            1.0 + highlight_offsets,
            highlight_values,
            s=18,
            marker="o",
            facecolor=ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.96,
            zorder=5,
            clip_on=True,
        )

    if log10_space:
        ax.set_yscale("log")
        ax.set_yticks([0.1, 1, 10, 100, 1000])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_ylim(*value_limits)
    ax.set_xlim(0.45, 1.55)
    ax.set_ylabel(y_label, fontsize=7)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=6.5, length=2.5, pad=1.5)
    ax.grid(False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)

    box_report = {
        key: box_report[key]
        for key in ("calculation_space", "q1", "median", "q3")
    }
    kde_report = {
        "bandwidth": "Scott's rule",
        "estimation_space": "log10" if log10_space else "linear",
        "support": [float(display_grid.min()), float(display_grid.max())],
        "maximum_half_width": 0.32,
    }
    return box_report, kde_report, highlight_offsets


def _save_city_reduction_distribution(
    fig,
    save_path,
    for_latex,
    plot_title,
    report,
    description,
):
    """Save one city-reduction distribution and its audit report."""
    if not save_path:
        plt.show()
        return None
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    report["figure_size_inches"] = [
        float(value) for value in fig.get_size_inches()
    ]
    report["figure_path"] = save_path
    write_figure_report(save_path.rsplit(".", 1)[0] + "_report.json", report)
    print(f"{description} saved to {save_path}")
    return save_path


def plot_city_relative_lifecycle_reduction_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.65, 2.04),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the city distribution of relative lifecycle emission reduction.

    Args:
        df: City-level region statistics with Oracle lifecycle emissions.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    prepared = _prepare_city_reduction_distribution_data(df)
    if prepared is None:
        return None
    plot_df, reduction_pct_col, _, audit = prepared
    values = plot_df[reduction_pct_col].to_numpy(dtype=float)
    highlight_records, missing_highlights = _city_reduction_highlight_records(
        plot_df,
        reduction_pct_col,
    )

    fig, ax = plt.subplots(figsize=figsize)
    box_report, _ = _draw_city_reduction_boxplot(
        ax,
        values,
        value_limits=(0.0, 50.0),
        y_label="Lifecycle reduction (%)",
        highlight_values=[record["value"] for record in highlight_records],
    )
    outlier_records = _city_reduction_outlier_records(
        plot_df,
        reduction_pct_col,
        box_report,
    )
    if plot_title:
        ax.set_title(
            "City-level relative lifecycle emission reduction",
            fontsize=9,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.06, top=0.97)

    report = {
        "figure_name": "city_relative_lifecycle_reduction_distribution",
        "source_scope": "City-level study regions with positive finite Oracle reductions",
        "metric": {
            "field": reduction_pct_col,
            "definition": "Oracle lifecycle reduction versus the all-e-bike baseline",
            "unit": "percent",
        },
        "distribution": {
            "method": "Vertical Tukey boxplot",
            "calculation_space": "linear percentage",
            "y_limits": [0.0, 50.0],
            "box": "first to third quartile",
            "median": "line inside box",
            "whiskers": "most extreme observations within 1.5 IQR",
            "outliers_displayed": True,
        },
        "box_statistics": box_report,
        "outliers": {
            "criterion": "observations outside the 1.5 IQR Tukey fences",
            "count": int(len(outlier_records)),
            "neutral_marker_count": int(box_report["neutral_outlier_count"]),
            "highlighted_map_city_count": int(
                box_report["highlighted_outlier_count"]
            ),
            "records": outlier_records,
        },
        "sample_audit": audit,
        "summary": _writing_numeric_summary(values),
        "highlighted_cities": {
            "source": "Cities labeled on the Oracle reduction map",
            "requested_cities": list(ORACLE_REDUCTION_LABEL_CITIES),
            "matched_count": int(len(highlight_records)),
            "missing_cities": missing_highlights,
            "metric_field": reduction_pct_col,
            "unit": "percent",
            "marker": {
                "shape": "circle",
                "facecolor": ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
                "edgecolor": "white",
                "size_points_squared": 18,
                "layout": "deterministic symmetric horizontal swarm beside box",
                "labels_or_legend": False,
            },
            "records": highlight_records,
        },
    }
    saved_path = _save_city_reduction_distribution(
        fig,
        save_path,
        for_latex,
        plot_title,
        report,
        "City relative lifecycle reduction distribution",
    )
    plt.close(fig)
    return saved_path


def plot_city_annual_absolute_reduction_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.65, 2.04),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the city distribution of annual absolute emission reduction.

    Args:
        df: City-level region statistics with Oracle lifecycle emissions.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    prepared = _prepare_city_reduction_distribution_data(df)
    if prepared is None:
        return None
    plot_df, _, reduction_t_col, audit = prepared
    values = plot_df[reduction_t_col].to_numpy(dtype=float)
    highlight_records, missing_highlights = _city_reduction_highlight_records(
        plot_df,
        reduction_t_col,
    )

    fig, ax = plt.subplots(figsize=figsize)
    box_report, _ = _draw_city_reduction_boxplot(
        ax,
        values,
        value_limits=(0.03, 5000.0),
        y_label="Annual reduction\n(t CO$_2$e yr$^{-1}$; log scale)",
        log10_space=True,
        highlight_values=[record["value"] for record in highlight_records],
    )
    outlier_records = _city_reduction_outlier_records(
        plot_df,
        reduction_t_col,
        box_report,
    )
    if plot_title:
        ax.set_title(
            "City-level annual absolute emission reduction",
            fontsize=9,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.06, top=0.97)

    report = {
        "figure_name": "city_annual_absolute_reduction_distribution",
        "source_scope": "City-level study regions with positive finite Oracle reductions",
        "metric": {
            "field": reduction_t_col,
            "definition": "All-e-bike lifecycle emissions minus Oracle lifecycle emissions",
            "unit": "t CO2e per year",
        },
        "distribution": {
            "method": "Vertical Tukey boxplot",
            "calculation_space": "log10 tonnes CO2e per year",
            "y_scale": "logarithmic",
            "y_limits": [0.03, 5000.0],
            "box": "first to third quartile calculated in log10 space",
            "median": "line inside box",
            "whiskers": "most extreme observations within 1.5 IQR in log10 space",
            "outliers_displayed": True,
        },
        "box_statistics": box_report,
        "outliers": {
            "criterion": "observations outside the 1.5 IQR Tukey fences in log10 space",
            "count": int(len(outlier_records)),
            "neutral_marker_count": int(box_report["neutral_outlier_count"]),
            "highlighted_map_city_count": int(
                box_report["highlighted_outlier_count"]
            ),
            "records": outlier_records,
        },
        "sample_audit": audit,
        "summary": _writing_numeric_summary(values),
        "highlighted_cities": {
            "source": "Cities labeled on the Oracle reduction map",
            "requested_cities": list(ORACLE_REDUCTION_LABEL_CITIES),
            "matched_count": int(len(highlight_records)),
            "missing_cities": missing_highlights,
            "metric_field": reduction_t_col,
            "unit": "t CO2e per year",
            "marker": {
                "shape": "circle",
                "facecolor": ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
                "edgecolor": "white",
                "size_points_squared": 18,
                "layout": "deterministic symmetric horizontal swarm beside box",
                "labels_or_legend": False,
            },
            "records": highlight_records,
        },
    }
    saved_path = _save_city_reduction_distribution(
        fig,
        save_path,
        for_latex,
        plot_title,
        report,
        "City annual absolute reduction distribution",
    )
    plt.close(fig)
    return saved_path


def plot_city_relative_lifecycle_reduction_violin_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.65, 2.04),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot a vertical violin of relative lifecycle emission reduction."""
    prepared = _prepare_city_reduction_distribution_data(df)
    if prepared is None:
        return None
    plot_df, reduction_pct_col, _, audit = prepared
    values = plot_df[reduction_pct_col].to_numpy(dtype=float)
    highlight_records, missing_highlights = _city_reduction_highlight_records(
        plot_df,
        reduction_pct_col,
    )

    fig, ax = plt.subplots(figsize=figsize)
    box_report, kde_report, _ = _draw_city_reduction_violin(
        ax,
        values,
        value_limits=(0.0, 50.0),
        y_label="Lifecycle reduction (%)",
        highlight_values=[record["value"] for record in highlight_records],
        avoid_summary_overlap=True,
    )
    if plot_title:
        ax.set_title(
            "City-level relative lifecycle emission reduction",
            fontsize=9,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.06, top=0.97)

    report = {
        "figure_name": "city_relative_lifecycle_reduction_violin_distribution",
        "source_scope": "City-level study regions with positive finite Oracle reductions",
        "metric": {
            "field": reduction_pct_col,
            "definition": "Oracle lifecycle reduction versus the all-e-bike baseline",
            "unit": "percent",
        },
        "distribution": {
            "method": "Vertical violin plot with Gaussian KDE",
            "estimation_space": "linear percentage",
            "y_limits": [0.0, 50.0],
            "width_normalization": "independent maximum density",
            "summary_marks": "median point and interquartile range line",
            "kde": kde_report,
        },
        "box_statistics": box_report,
        "sample_audit": audit,
        "summary": _writing_numeric_summary(values),
        "highlighted_cities": {
            "source": "Cities labeled on the Oracle reduction map",
            "requested_cities": list(ORACLE_REDUCTION_LABEL_CITIES),
            "matched_count": int(len(highlight_records)),
            "missing_cities": missing_highlights,
            "metric_field": reduction_pct_col,
            "unit": "percent",
            "marker": {
                "shape": "circle",
                "facecolor": ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
                "edgecolor": "white",
                "size_points_squared": 18,
                "layout": "Q1-Q3 highlights alternate left and right clear of "
                "center, clipped to 95% of local violin half-width",
                "labels_or_legend": False,
            },
            "records": highlight_records,
        },
    }
    saved_path = _save_city_reduction_distribution(
        fig,
        save_path,
        for_latex,
        plot_title,
        report,
        "City relative lifecycle reduction violin distribution",
    )
    plt.close(fig)
    return saved_path


def plot_city_annual_absolute_reduction_violin_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (1.65, 2.04),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot a vertical violin of annual absolute emission reduction."""
    prepared = _prepare_city_reduction_distribution_data(df)
    if prepared is None:
        return None
    plot_df, _, reduction_t_col, audit = prepared
    values = plot_df[reduction_t_col].to_numpy(dtype=float)
    highlight_records, missing_highlights = _city_reduction_highlight_records(
        plot_df,
        reduction_t_col,
    )

    fig, ax = plt.subplots(figsize=figsize)
    box_report, kde_report, _ = _draw_city_reduction_violin(
        ax,
        values,
        value_limits=(0.03, 5000.0),
        y_label="Annual reduction\n(t CO$_2$e yr$^{-1}$; log scale)",
        log10_space=True,
        highlight_values=[record["value"] for record in highlight_records],
    )
    if plot_title:
        ax.set_title(
            "City-level annual absolute emission reduction",
            fontsize=9,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.39, right=0.96, bottom=0.06, top=0.97)

    report = {
        "figure_name": "city_annual_absolute_reduction_violin_distribution",
        "source_scope": "City-level study regions with positive finite Oracle reductions",
        "metric": {
            "field": reduction_t_col,
            "definition": "All-e-bike lifecycle emissions minus Oracle lifecycle emissions",
            "unit": "t CO2e per year",
        },
        "distribution": {
            "method": "Vertical violin plot with Gaussian KDE",
            "estimation_space": "log10 tonnes CO2e per year",
            "y_scale": "logarithmic",
            "y_limits": [0.03, 5000.0],
            "width_normalization": "independent maximum density",
            "summary_marks": "median point and interquartile range line",
            "kde": kde_report,
        },
        "box_statistics": box_report,
        "sample_audit": audit,
        "summary": _writing_numeric_summary(values),
        "highlighted_cities": {
            "source": "Cities labeled on the Oracle reduction map",
            "requested_cities": list(ORACLE_REDUCTION_LABEL_CITIES),
            "matched_count": int(len(highlight_records)),
            "missing_cities": missing_highlights,
            "metric_field": reduction_t_col,
            "unit": "t CO2e per year",
            "marker": {
                "shape": "circle",
                "facecolor": ORACLE_REDUCTION_DISTRIBUTION_HIGHLIGHT_COLOR,
                "edgecolor": "white",
                "size_points_squared": 18,
                "layout": "compact symmetric offsets clipped to 95% of local "
                "violin half-width",
                "labels_or_legend": False,
            },
            "records": highlight_records,
        },
    }
    saved_path = _save_city_reduction_distribution(
        fig,
        save_path,
        for_latex,
        plot_title,
        report,
        "City annual absolute reduction violin distribution",
    )
    plt.close(fig)
    return saved_path


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


def _find_reduction_break_even_km(
    centres: np.ndarray,
    reduction_pct: np.ndarray,
    tol_pct: float = 0.25,
    smooth_window: int = 5,
    min_run: int = 2,
):
    """Find a sustained negative-to-positive crossing in binned reductions."""
    valid = np.isfinite(centres) & np.isfinite(reduction_pct)
    x = centres[valid]
    y = reduction_pct[valid]
    if x.size < 2:
        return None

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    half_window = max(1, smooth_window) // 2
    y_smooth = np.empty_like(y, dtype=float)
    for i in range(y.size):
        start = max(0, i - half_window)
        stop = min(y.size, i + half_window + 1)
        y_smooth[i] = np.median(y[start:stop])

    signs = np.zeros(y_smooth.size, dtype=int)
    signs[y_smooth < -tol_pct] = -1
    signs[y_smooth > tol_pct] = 1
    significant = np.flatnonzero(signs)
    if significant.size < 2:
        return None

    candidates = []
    for left, right in zip(significant[:-1], significant[1:]):
        if signs[left] != -1 or signs[right] != 1:
            continue

        before = 0
        for idx in significant[significant <= left][::-1]:
            if signs[idx] != -1:
                break
            before += 1

        after = 0
        for idx in significant[significant >= right]:
            if signs[idx] != 1:
                break
            after += 1

        if before >= min_run and after >= min_run:
            gap = right - left
            candidates.append((before + after - 0.25 * gap, left, right))

    if not candidates:
        return None

    _, left, right = max(candidates, key=lambda item: item[0])
    x0, x1 = x[left], x[right]
    y0, y1 = y_smooth[left], y_smooth[right]
    if y1 == y0:
        return float((x0 + x1) / 2)
    crossing = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
    return float(np.clip(crossing, min(x0, x1), max(x0, x1)))


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


def _plot_reduction_detour_panel(
    fig,
    subplot_spec,
    binned: dict,
    city_label: str,
    bin_width_km: float = 0.1,
    min_air_km: float = 0.0,
    max_air_km: float = 5.0,
    absolute_reduction_metric: str = "total_t",
    show_detour_boxplot: bool = True,
    mark_reduction_break_even: bool = False,
    show_city_label: bool = True,
    bar_width_fraction: float = 1.0,
    show_absolute_axis: bool = True,
    show_saturation_fit: bool = False,
    bar_metric: str = "reduction_pct",
):
    """Draw one reduction-detour city panel inside an existing figure."""
    if bar_metric not in {"reduction_pct", "oracle_drone_share_pct"}:
        raise ValueError(
            "bar_metric must be 'reduction_pct' or 'oracle_drone_share_pct'"
        )

    if show_detour_boxplot:
        inner = subplot_spec.subgridspec(2, 1, hspace=0.08, height_ratios=[1, 1])
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
    else:
        ax_top = None
        ax_bot = fig.add_subplot(subplot_spec)

    centres = np.asarray(binned["centres"], dtype=float)
    reduction_pct = np.asarray(binned["reduction_pct"], dtype=float)
    reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
    if bar_metric == "oracle_drone_share_pct":
        if "oracle_drone_share_pct" not in binned:
            raise ValueError("binned data missing 'oracle_drone_share_pct'")
        bar_values = np.asarray(binned["oracle_drone_share_pct"], dtype=float)
    else:
        bar_values = reduction_pct
    detour_by_bin = binned.get("detour_by_bin")
    if detour_by_bin is None:
        detour_by_bin = [np.array([]) for _ in centres]
    max_km = float(binned["max_km"])
    x_max = min(max_air_km, max_km)
    x_min = min(min_air_km, x_max)
    x_ticks = np.arange(np.ceil(x_min), x_max + 1e-6, 1.0)
    if x_min > 0 and (not x_ticks.size or not np.isclose(x_ticks[0], x_min)):
        x_ticks = np.insert(x_ticks, 0, x_min)
    x_tick_labels = [f"{t:g}" for t in x_ticks]
    visible = (centres >= x_min) & (centres <= x_max)

    if show_detour_boxplot:
        nonempty = [(c_, d_) for c_, d_ in zip(centres, detour_by_bin) if d_.size > 0]
        if nonempty:
            box_positions = [c_ for c_, _ in nonempty]
            box_data = [d_ for _, d_ in nonempty]
            ax_top.boxplot(
                box_data,
                positions=box_positions,
                widths=bin_width_km * 0.8,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f4e79", linewidth=0.5),
                medianprops=dict(color="#1f4e79", linewidth=1.0),
                whiskerprops=dict(color="#1f4e79", linewidth=0.5),
                capprops=dict(color="#1f4e79", linewidth=0.5),
            )
        ax_top.axhline(1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax_top.set_ylabel("Detour (road / air)", fontsize=9)
        ax_top.set_xticks(x_ticks, labels=x_tick_labels)
        ax_top.set_xlim(x_min, x_max)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="both", labelsize=8)
        ax_top.grid(axis="y", alpha=0.3)
        if show_city_label:
            ax_top.set_title(city_label, fontsize=11, fontweight="bold")
    elif show_city_label:
        ax_bot.set_title(city_label, fontsize=11, fontweight="bold")

    if bar_metric == "oracle_drone_share_pct":
        bar_colors = [
            "#9ecae1" if np.isfinite(v) else "#d9d9d9"
            for v in bar_values
        ]
        bar_label = "Optimal drone share"
        bar_ylabel = "Optimal Drone Share (%)"
        bar_tick_color = "#3182bd"
    else:
        bar_colors = [
            "#2ca02c" if (np.isfinite(v) and v >= 0) else "#d62728"
            for v in bar_values
        ]
        bar_label = "(E-bike - Drone) / E-bike"
        bar_ylabel = "(E-bike - Drone) / E-bike (%)"
        bar_tick_color = "#2ca02c"
    ax_bot.bar(
        centres,
        bar_values,
        width=bin_width_km * bar_width_fraction,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.2,
        align="center",
        label=bar_label,
    )
    ax_bot.axhline(0, color="gray", linewidth=0.5)
    ax_bot.set_xlabel("Air Distance (km)", fontsize=9)
    ax_bot.set_ylabel(bar_ylabel, fontsize=9, color=bar_tick_color)
    ax_bot.set_xlim(x_min, x_max)
    ax_bot.set_xticks(x_ticks, labels=x_tick_labels)
    ax_bot.tick_params(axis="x", labelsize=8)
    ax_bot.tick_params(axis="y", labelsize=8, labelcolor=bar_tick_color)
    ax_bot.grid(axis="y", alpha=0.3)

    saturation_fit_y = None
    if show_saturation_fit and bar_metric == "reduction_pct":
        fit_mask = visible & np.isfinite(centres) & np.isfinite(reduction_pct)
        x_fit_data = centres[fit_mask]
        y_fit_data = reduction_pct[fit_mask]
        if x_fit_data.size >= 4:
            try:
                from scipy.optimize import curve_fit

                x0 = float(x_min)

                def saturation_model(x, y0, amplitude, rate):
                    return y0 + amplitude * (1.0 - np.exp(-rate * (x - x0)))

                initial = [
                    float(y_fit_data[0]),
                    float(y_fit_data[-1] - y_fit_data[0]),
                    0.8,
                ]
                params, _ = curve_fit(
                    saturation_model,
                    x_fit_data,
                    y_fit_data,
                    p0=initial,
                    bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
                    maxfev=10000,
                )
                x_fit = np.linspace(x_min, x_max, 160)
                saturation_fit_y = saturation_model(x_fit, *params)
                ax_bot.plot(
                    x_fit,
                    saturation_fit_y,
                    color="#111111",
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=4,
                )
                y0, amplitude, rate = params
                sign = "+" if amplitude >= 0 else "-"
                equation = (
                    f"y = {y0:.1f} {sign} {abs(amplitude):.1f}"
                    f"(1 - e^(-{rate:.2f}(x - {x0:.1f})))"
                )
                ax_bot.text(
                    0.03,
                    0.94,
                    equation,
                    transform=ax_bot.transAxes,
                    fontsize=7.5,
                    ha="left",
                    va="top",
                    color="#111111",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.8),
                    zorder=6,
                )
            except Exception:
                saturation_fit_y = None

    ylim_values = bar_values[visible]
    if saturation_fit_y is not None:
        ylim_values = np.concatenate([ylim_values, saturation_fit_y])
    if bar_metric == "oracle_drone_share_pct":
        ax_bot.set_ylim(0.0, 150.0)
        ax_bot.set_yticks([0, 50, 100])
    else:
        visible_pct_ylim = _padded_visible_ylim(ylim_values)
        if visible_pct_ylim is not None:
            ax_bot.set_ylim(*visible_pct_ylim)

    if mark_reduction_break_even and bar_metric == "reduction_pct":
        break_even_km = _find_reduction_break_even_km(centres, reduction_pct)
        if break_even_km is not None and break_even_km <= x_max:
            ax_bot.axvline(
                break_even_km,
                color="#111111",
                linestyle=":",
                linewidth=1.0,
                alpha=0.85,
                zorder=5,
            )
            ax_bot.annotate(
                f"{break_even_km:.1f} km",
                xy=(break_even_km, 0.96),
                xycoords=("data", "axes fraction"),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=7,
                color="#111111",
                rotation=90,
            )

    ax_abs = None
    if not show_absolute_axis:
        return ax_top, ax_bot, ax_abs

    ax_abs = ax_bot.twinx()
    ax_abs.plot(
        centres,
        reduction_abs,
        color="#555555",
        alpha=0.5,
        linewidth=1.2,
        marker="o",
        markersize=2.5,
        label=(
            "E-bike - drone (g CO2e/order)"
            if absolute_reduction_metric == "per_order_g"
            else "E-bike - drone (t CO2e)"
        ),
    )
    ax_abs.axhline(0, color="#555555", linewidth=0.5, alpha=0.35)
    abs_ylabel = (
        "(E-bike - Drone) / Orders (g CO2e/order)"
        if absolute_reduction_metric == "per_order_g"
        else "E-bike - Drone (t CO2e/yr)"
    )
    ax_abs.set_ylabel(abs_ylabel, fontsize=9, color="#555555")
    ax_abs.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    visible_abs_ylim = _padded_visible_ylim(reduction_abs[visible])
    if visible_abs_ylim is not None:
        ax_abs.set_ylim(*visible_abs_ylim)

    return ax_top, ax_bot, ax_abs


def _plot_reduction_absolute_scatter_panel(
    fig,
    subplot_spec,
    binned: dict,
    city_label: str,
    min_air_km: float = 0.0,
    max_air_km: float = 5.0,
    absolute_reduction_metric: str = "total_t",
    show_city_label: bool = True,
    show_linear_equation: bool = True,
):
    """Draw absolute reduction as scatter with a linear fit."""
    ax = fig.add_subplot(subplot_spec)
    centres = np.asarray(binned["centres"], dtype=float)
    reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
    max_km = float(binned["max_km"])
    x_max = min(max_air_km, max_km)
    x_min = min(min_air_km, x_max)
    visible = (centres >= x_min) & (centres <= x_max)
    finite = visible & np.isfinite(centres) & np.isfinite(reduction_abs)
    x = centres[finite]
    y = reduction_abs[finite]

    ax.scatter(
        x,
        y,
        color="#555555",
        alpha=0.72,
        s=14,
        linewidth=0,
        zorder=3,
    )
    y_fit = None
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_fit = np.array([x_min, x_max], dtype=float)
        y_fit = slope * x_fit + intercept
        ax.plot(
            x_fit,
            y_fit,
            color="#111111",
            linewidth=1.2,
            alpha=0.85,
            zorder=4,
        )
        if show_linear_equation:
            sign = "+" if intercept >= 0 else "-"
            equation = f"y = {slope:.1f}x {sign} {abs(intercept):.1f}"
            ax.text(
                0.05,
                0.94,
                equation,
                transform=ax.transAxes,
                fontsize=8,
                ha="left",
                va="top",
                color="#111111",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.8),
                zorder=6,
            )

    ax.axhline(0, color="#555555", linewidth=0.5, alpha=0.35)
    x_ticks = np.arange(np.ceil(x_min), x_max + 1e-6, 1.0)
    if x_min > 0 and (not x_ticks.size or not np.isclose(x_ticks[0], x_min)):
        x_ticks = np.insert(x_ticks, 0, x_min)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(x_ticks, labels=[f"{t:g}" for t in x_ticks])
    ax.set_xlabel("Air Distance (km)", fontsize=9)
    ylabel = (
        "E-bike - Drone (g CO2e/order)"
        if absolute_reduction_metric == "per_order_g"
        else "E-bike - Drone (t CO2e/yr)"
    )
    ax.set_ylabel(ylabel, fontsize=9, color="#555555")
    ax.tick_params(axis="both", labelsize=8)
    ax.tick_params(axis="y", labelcolor="#555555")
    ax.grid(axis="y", alpha=0.3)
    ylim_values = y
    if y_fit is not None:
        ylim_values = np.concatenate([ylim_values, y_fit])
    visible_abs_ylim = _padded_visible_ylim(ylim_values)
    if visible_abs_ylim is not None:
        ax.set_ylim(*visible_abs_ylim)
    if show_city_label:
        ax.set_title(city_label, fontsize=11, fontweight="bold")
    return ax


def plot_reduction_detour_grid_binned(
    binned_by_city: dict,
    city_labels: dict = None,
    save_path: str = None,
    figsize: tuple = (12, 14),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.1,
    n_cols: int = 2,
    min_air_km: float = 0.0,
    max_air_km: float = 5.0,
    absolute_reduction_metric: str = "total_t",
    show_detour_boxplot: bool = True,
    mark_reduction_break_even: bool = False,
    grid_hspace: float = 0.45,
    grid_wspace: float = 0.22,
):
    """Multi-city grid from pre-binned emission-reduction data.

    Args:
        binned_by_city: Mapping ``{city_key: binned_data}``. Each binned data
            dict needs ``centres``, ``reduction_pct``, ``reduction_abs``, and
            ``max_km``. ``detour_by_bin`` is optional unless showing boxplots.
        city_labels: Optional ``{city_key: display_name}``. Defaults to key.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show suptitle.
        bin_width_km: Width of each road-distance bin in km.
        n_cols: Number of city columns in the grid.
        min_air_km: Minimum air distance shown on the x-axis.
        max_air_km: Maximum air distance shown on the x-axis.
        absolute_reduction_metric: Metric for the secondary y-axis. Use
            ``total_t`` for annual tonnes or ``per_order_g`` for grams per
            order in each distance bin.
        show_detour_boxplot: If True, include the detour-ratio boxplot above
            each emission-reduction panel.
        mark_reduction_break_even: If True, mark the sustained negative-to-
            positive crossing in the reduction bars when one is found.
        grid_hspace: Vertical spacing between city panels.
        grid_wspace: Horizontal spacing between city panels.
    """
    keys = list(binned_by_city.keys())
    n_cities = len(keys)
    n_rows = int(np.ceil(n_cities / n_cols))
    city_labels = city_labels or {}

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(n_rows, n_cols, hspace=grid_hspace, wspace=grid_wspace)

    for i, key in enumerate(keys):
        r, c = divmod(i, n_cols)
        _plot_reduction_detour_panel(
            fig,
            outer[r, c],
            binned_by_city[key],
            city_labels.get(key, key),
            bin_width_km=bin_width_km,
            min_air_km=min_air_km,
            max_air_km=max_air_km,
            absolute_reduction_metric=absolute_reduction_metric,
            show_detour_boxplot=show_detour_boxplot,
            mark_reduction_break_even=mark_reduction_break_even,
        )

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


def plot_reduction_detour_grid(
    od_dfs: dict,
    city_labels: dict = None,
    save_path: str = None,
    figsize: tuple = (12, 14),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.1,
    n_cols: int = 2,
    min_air_km: float = 0.0,
    max_air_km: float = 5.0,
    absolute_reduction_metric: str = "total_t",
    show_detour_boxplot: bool = True,
    mark_reduction_break_even: bool = False,
    grid_hspace: float = 0.45,
    grid_wspace: float = 0.22,
):
    """Multi-city grid of emission-reduction bars per air-distance bin.

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
        min_air_km: Minimum air distance shown on the x-axis.
        max_air_km: Maximum air distance shown on the x-axis.
        absolute_reduction_metric: Metric for the secondary y-axis. Use
            ``total_t`` for annual tonnes or ``per_order_g`` for grams per
            order in each distance bin.
        show_detour_boxplot: If True, include the detour-ratio boxplot above
            each emission-reduction panel.
        mark_reduction_break_even: If True, mark the sustained negative-to-
            positive crossing in the reduction bars when one is found.
        grid_hspace: Vertical spacing between city panels.
        grid_wspace: Horizontal spacing between city panels.
    """
    binned_by_city = {}
    for key, od_df in od_dfs.items():
        (
            centres,
            reduction_pct,
            reduction_abs,
            detour_by_bin,
            max_km,
            oracle_drone_share_pct,
        ) = (
            _bin_reduction_and_detour(
                od_df, bin_width_km, absolute_reduction_metric
            )
        )
        binned_by_city[key] = {
            "centres": centres,
            "reduction_pct": reduction_pct,
            "reduction_abs": reduction_abs,
            "detour_by_bin": detour_by_bin,
            "max_km": max_km,
            "oracle_drone_share_pct": oracle_drone_share_pct,
        }

    plot_reduction_detour_grid_binned(
        binned_by_city=binned_by_city,
        city_labels=city_labels,
        save_path=save_path,
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
        bin_width_km=bin_width_km,
        n_cols=n_cols,
        min_air_km=min_air_km,
        max_air_km=max_air_km,
        absolute_reduction_metric=absolute_reduction_metric,
        show_detour_boxplot=show_detour_boxplot,
        mark_reduction_break_even=mark_reduction_break_even,
        grid_hspace=grid_hspace,
        grid_wspace=grid_wspace,
    )


def plot_kappa_vrp_saturation(
    df: pd.DataFrame,
    plateau: dict = None,
    configured_kappa: float = 1.0,
    save_path: str = None,
    figsize: tuple = (7.2, 4.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot kappa against maximum e-bike bundle size for VRP scenarios.

    Args:
        df: Capacity-summary data with scenario, capacity, kappa, and interval columns.
        plateau: Plateau-detection result for the constrained best-case curve.
        configured_kappa: Current configured baseline shown as a reference.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required = {"scenario", "capacity", "kappa"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kappa VRP summary missing columns: {sorted(missing)}")

    styles = {
        "geometric_lower_bound": {
            "label": "Geometric lower bound",
            "color": "#4C78A8",
            "marker": "o",
            "linestyle": "--",
        },
        "baseline_operational": {
            "label": "Baseline operations",
            "color": "#7A7A7A",
            "marker": "s",
            "linestyle": "-",
        },
        "constrained_best_case": {
            "label": "Constrained best case",
            "color": "#E45756",
            "marker": "D",
            "linestyle": "-",
        },
    }
    fig, ax = plt.subplots(figsize=figsize)
    for scenario, style in styles.items():
        subset = df[df["scenario"] == scenario].sort_values("capacity")
        if subset.empty:
            continue
        x = subset["capacity"].to_numpy(dtype=float)
        y = subset["kappa"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=4.5,
        )
        if {"kappa_ci_low", "kappa_ci_high"}.issubset(subset.columns):
            low = subset["kappa_ci_low"].to_numpy(dtype=float)
            high = subset["kappa_ci_high"].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(low) & np.isfinite(high)
            if finite.any():
                ax.fill_between(
                    x[finite],
                    low[finite],
                    high[finite],
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0,
                )

    ax.axhline(
        configured_kappa,
        color="#E08214",
        linewidth=1.3,
        linestyle=":",
        label=rf"Configured baseline ($\kappa={configured_kappa:.3f}$)",
    )
    if plateau and plateau.get("reached"):
        bundle_size = float(plateau["bundle_size_star"])
        kappa_value = float(plateau["future_extreme_kappa"])
        ax.scatter(
            [bundle_size],
            [kappa_value],
            s=58,
            facecolor="white",
            edgecolor="#B22222",
            linewidth=1.5,
            zorder=8,
        )
        ax.annotate(
            rf"$B^*={bundle_size:g}$, $\kappa={kappa_value:.3f}$",
            xy=(bundle_size, kappa_value),
            xytext=(8, -18),
            textcoords="offset points",
            fontsize=8.5,
            color="#8B1A1A",
        )

    capacities = pd.to_numeric(df["capacity"], errors="coerce").dropna()
    if not capacities.empty:
        ax.set_xticks(
            np.arange(int(capacities.min()), int(capacities.max()) + 1)
        )
    ax.set_xlabel("Maximum bundle size, $B$")
    ax.set_ylabel(r"Effective per-order distance factor, $\kappa$")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    if plot_title:
        ax.set_title("E-bike bundling saturation under route optimization")
    fig.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Kappa VRP saturation curve saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manuscript R2 city-attribution subfigures

R2_FIGURE_CITY_KEYS = ("chengdu", "guangzhou", "beijing", "shanghai")
R2_FIGURE_CITY_LABELS = {
    "chengdu": "Chengdu",
    "guangzhou": "Guangzhou",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
}
R2_SCREEN_LABELS = {
    "grid_emission_factor": "Grid emission factor",
    "detour_ratio": "Detour ratio",
    "avg_street_length": "Mean street length",
    "dead_end_proportion": "Dead-end proportion",
    "four_way_proportion": "Four-way intersection proportion",
    "orientation_entropy": "Orientation entropy",
    "latitude": "Latitude",
    "population_10k": "Population",
    "store_density": "Store density",
    "pop_density": "Population density",
    "gdp_per_capita": "GDP per capita",
    "building_density": "Building density",
    "building_count": "Building count",
    "road_density": "Road density",
}
R2_MATRIX_LABELS = {
    R2_OUTCOME: "Relative reduction",
    "grid_emission_factor": "Grid factor",
    "detour_ratio": "Detour ratio",
    "avg_street_length": "Street length",
    "avg_block_length": "Block length",
    "dead_end_proportion": "Dead ends",
    "four_way_proportion": "Four-way",
    "orientation_entropy": "Orientation entropy",
    "intersection_density": "Intersection density",
    "avg_road_distance_m": "Road distance",
    "avg_straight_distance_m": "Air distance",
    "road_density": "Road density",
    "roadnet_length_total": "Road length",
    "population_10k": "Population",
    "pop_density": "Population density",
    "gdp_per_capita": "GDP per capita",
    "building_density": "Building density",
    "building_count": "Building count",
    "building_volume_total": "Building volume",
    "store_density": "Store density",
    "sales_intensity": "Sales intensity",
    "region_area_km2": "Region area",
    "latitude": "Latitude",
    "longitude": "Longitude",
}


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


def plot_r2_gef_detour_suitability_landscape(
    df: pd.DataFrame,
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (2.31, 3.39),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the two-lever city suitability landscape for manuscript R2.

    Bubble area encodes annual absolute Oracle-vs-e-bike reduction on a
    square-root scale, while colour retains the relative reduction outcome.

    Args:
        df: Prepared city-attribution DataFrame.
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required = [
        "grid_emission_factor",
        "detour_ratio",
        R2_OUTCOME,
        R2_ABSOLUTE_OUTCOME,
    ]
    plot_df = df.dropna(subset=required).copy()
    numeric_required = plot_df[required].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric_required).all(axis=1)
    positive_absolute = numeric_required[R2_ABSOLUTE_OUTCOME] > 0
    plot_df = plot_df.loc[finite & positive_absolute].copy()
    if plot_df.empty:
        raise ValueError(
            "R2 suitability landscape has no finite cities with positive "
            "annual absolute reduction"
        )

    bubble_area_min = 4.0
    bubble_area_max = 50.0
    bubble_legend_values = (10.0, 100.0, 1000.0)
    absolute_reduction = plot_df[R2_ABSOLUTE_OUTCOME].to_numpy(dtype=float)
    absolute_reduction_max = float(absolute_reduction.max())

    def _bubble_area(values):
        values = np.asarray(values, dtype=float)
        scaled = np.sqrt(values / absolute_reduction_max)
        return bubble_area_min + scaled * (bubble_area_max - bubble_area_min)

    plot_df["_bubble_area"] = _bubble_area(absolute_reduction)
    interaction = analysis["interaction_model"]
    orthogonal = analysis["gef_detour_orthogonality"]

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        plot_df["grid_emission_factor"],
        plot_df["detour_ratio"],
        c=plot_df[R2_OUTCOME],
        cmap="YlGn",
        vmin=0,
        vmax=50.0,
        s=plot_df["_bubble_area"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.35,
        zorder=3,
    )
    size_handles = [
        ax.scatter(
            [], [],
            s=float(_bubble_area([value])[0]),
            facecolor="#86C98A",
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
        )
        for value in bubble_legend_values
    ]
    ax.legend(
        size_handles,
        [f"{value:,.0f}" for value in bubble_legend_values],
        title="Annual reduction\n(t CO$_2$e yr$^{-1}$)",
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fontsize=6.5,
        title_fontsize=6.5,
        borderaxespad=0.3,
        borderpad=0.25,
        labelspacing=0.1,
        handletextpad=0.35,
    )

    name_col = (
        "region_name_en" if "region_name_en" in plot_df.columns else "region_key"
    )
    plot_df["_label_city"] = plot_df[name_col].astype(str)
    if name_col == "region_key":
        plot_df["_label_city"] = plot_df["_label_city"].str.title()
    labeled = plot_df[
        plot_df["_label_city"].isin(ORACLE_REDUCTION_LABEL_CITIES)
    ].copy()
    labeled["_label_order"] = pd.Categorical(
        labeled["_label_city"],
        categories=ORACLE_REDUCTION_LABEL_CITIES,
        ordered=True,
    )
    labeled = labeled.sort_values("_label_order")

    from adjustText import adjust_text

    ax.scatter(
        labeled["grid_emission_factor"],
        labeled["detour_ratio"],
        s=labeled["_bubble_area"],
        facecolors="none",
        edgecolors="#202020",
        linewidth=0.7,
        zorder=4,
    )
    label_texts = []
    city_records = []
    for _, row in labeled.iterrows():
        label_texts.append(
            ax.text(
                row["grid_emission_factor"],
                row["detour_ratio"],
                row["_label_city"],
                ha="center",
                va="center",
                fontsize=6.5,
                color="#303030",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.68,
                },
            )
        )
        city_records.append({
            "city": row["_label_city"],
            "region_key": row["region_key"],
            "grid_emission_factor": float(row["grid_emission_factor"]),
            "detour_ratio": float(row["detour_ratio"]),
            "relative_reduction_pct": float(row[R2_OUTCOME]),
            "annual_absolute_reduction_t_co2e": float(
                row[R2_ABSOLUTE_OUTCOME]
            ),
        })
    adjust_text(
        label_texts,
        x=labeled["grid_emission_factor"].to_numpy(dtype=float),
        y=labeled["detour_ratio"].to_numpy(dtype=float),
        ax=ax,
        expand=(1.15, 1.25),
        force_text=(0.2, 0.3),
        force_static=(0.1, 0.2),
        ensure_inside_axes=True,
        arrowprops={
            "arrowstyle": "-",
            "color": "0.35",
            "lw": 0.5,
        },
    )

    ax.set_xlabel(
        "Grid emission factor (kg CO$_2$e kWh$^{-1}$)",
        fontsize=7,
    )
    ax.set_ylabel("Annual-sales-weighted detour ratio", fontsize=7)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        orientation="horizontal",
        pad=0.18,
        fraction=0.06,
        aspect=25,
        ticks=np.arange(0, 51, 10),
    )
    colorbar.set_label("Optimal lifecycle reduction (%)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6.5)
    if plot_title:
        ax.set_title("Grid intensity modifies the benefit of avoiding road detours")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.17)
    _r2_save_figure(
        fig, save_path, for_latex, plot_title,
        "r2_gef_detour_suitability_landscape",
        {
            "plotted_city_count": int(len(plot_df)),
            "interaction_model": interaction,
            "gef_detour_orthogonality": orthogonal,
            "bubble_size_encoding": {
                "metric": R2_ABSOLUTE_OUTCOME,
                "unit": "t CO2e yr^-1",
                "transform": (
                    "sqrt(value / sample maximum) mapped linearly to marker area"
                ),
                "area_range_points_squared": [
                    bubble_area_min,
                    bubble_area_max,
                ],
                "value_range": [
                    float(absolute_reduction.min()),
                    float(absolute_reduction.max()),
                ],
                "legend_values": list(bubble_legend_values),
            },
            "colour_encoding": {
                "metric": R2_OUTCOME,
                "unit": "percent",
            },
            "selected_cities": city_records,
        },
    )


def plot_r2_hierarchical_r2_steps(
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (6.6, 4.4),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot cumulative R-squared steps for the nested R2 models.

    Args:
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    models = analysis["hierarchical_models"]
    x = np.arange(len(models))
    r2_values = np.asarray([model["r2"] for model in models])
    labels = ["Grid", "+ Detour", "+ Other road form", "+ Socioeconomic"]
    colors = ["#4C78A8", "#F28E2B", "#9C9C9C", "#C7C7C7"]

    fig, ax = plt.subplots(figsize=figsize)
    ax.step(x, r2_values, where="mid", color="#1F1F1F", linewidth=1.5, zorder=3)
    for index, (model, color) in enumerate(zip(models, colors)):
        previous = 0 if index == 0 else r2_values[index - 1]
        ax.vlines(index, previous, model["r2"], color=color, linewidth=11, alpha=0.9, zorder=2)
        p_value = model["hc3_block_test"]["p_value"]
        p_label = "$P < 0.001$" if p_value < 0.001 else f"$P = {p_value:.3f}$"
        delta_ci = model["delta_r2_ci95_percentile"]
        ax.text(
            index,
            model["r2"] + 0.025,
            f"$R^2$ = {model['r2']:.3f}\n"
            f"$\\Delta R^2$ = {model['delta_r2']:.3f} "
            f"[{delta_ci[0]:.3f}, {delta_ci[1]:.3f}]\n"
            f"HC3 {p_label}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Cumulative explained variance ($R^2$)")
    ax.set_ylim(0, min(1.02, float(r2_values.max()) + 0.16))
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98, 0.04, f"Common sample: n = {models[-1]['n']}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555",
    )
    if plot_title:
        ax.set_title("Grid intensity and detour explain most cross-city variation")
    fig.tight_layout()
    _r2_save_figure(
        fig, save_path, for_latex, plot_title,
        "r2_hierarchical_r2_steps",
        {"models": models},
    )


def plot_r2_attribute_correlation_screen(
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (7.4, 6.0),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot a two-column correlation screen for R2 candidate attributes.

    Args:
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    screen = analysis["screen_correlations"]
    outcome_lookup = {item["x"]: item for item in screen["outcome"]}
    gef_lookup = {item["x"]: item for item in screen["grid_emission_factor"]}
    variables = screen["variables"]
    y = np.arange(len(variables))[::-1]
    category_colors = {
        "core": "#D95F02",
        "road": "#1B9E77",
        "scale": "#7570B3",
    }

    def category(variable):
        if variable in {"grid_emission_factor", "detour_ratio"}:
            return "core"
        if variable in {
            "avg_street_length", "dead_end_proportion", "four_way_proportion",
            "orientation_entropy", "latitude",
        }:
            return "road"
        return "scale"

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    column_specs = (
        (axes[0], outcome_lookup, "Correlation with relative reduction"),
        (axes[1], gef_lookup, "Correlation with grid emission factor"),
    )
    for ax, lookup, x_label in column_specs:
        ax.axvspan(-0.10, 0.10, color="#F2F2F2", zorder=0)
        ax.axvline(0, color="#666666", linewidth=0.7)
        for row_y, variable in zip(y, variables):
            record = lookup.get(variable)
            if record is None or record["coefficient"] is None:
                continue
            significant = record["significant_adjusted_0_05"]
            color = category_colors[category(variable)]
            ci_low, ci_high = record["ci95"]
            ax.plot([ci_low, ci_high], [row_y, row_y], color=color,
                    alpha=0.82 if significant else 0.30, linewidth=1.1)
            ax.scatter(record["coefficient"], row_y, s=34,
                       color=color, alpha=0.95 if significant else 0.24,
                       edgecolor="white", linewidth=0.35, zorder=3)
            ax.text(record["coefficient"] + (0.025 if record["coefficient"] >= 0 else -0.025),
                    row_y, f"{record['coefficient']:+.2f}",
                    ha="left" if record["coefficient"] >= 0 else "right",
                    va="center", fontsize=6.7,
                    alpha=1.0 if significant else 0.38)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(x_label)
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y, [R2_SCREEN_LABELS.get(var, var) for var in variables])
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", color=color, label=label)
        for label, color in (("Core levers", category_colors["core"]),
                             ("Road/geographic", category_colors["road"]),
                             ("Scale/density", category_colors["scale"]))
    ]
    handles.append(
        Patch(facecolor="#F2F2F2", edgecolor="none", label="Equivalence region ±0.10")
    )
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
    )
    if plot_title:
        fig.suptitle("Attribute screening separates core levers, covariates, and null factors")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    _r2_save_figure(
        fig, save_path, for_latex, plot_title,
        "r2_attribute_correlation_screen",
        {"screen_correlations": screen},
    )


def _plot_r2_correlation_matrix(
    matrix: dict,
    save_path: str,
    figsize: tuple,
    for_latex: bool,
    plot_title: bool,
    figure_name: str,
):
    """Render one full R2 correlation matrix with significance fading."""
    from matplotlib.colors import Normalize

    variables = matrix["variables"]
    coefficients = np.asarray(matrix["coefficients"], dtype=float)
    p_adjusted = np.asarray(matrix["p_adjusted"], dtype=float)
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-1, vmax=1)
    rgba = cmap(norm(coefficients))
    insignificant = (p_adjusted >= 0.05) & (~np.eye(len(variables), dtype=bool))
    rgba[..., 3] = np.where(insignificant, 0.20, 0.95)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, interpolation="nearest", aspect="equal")
    for row in range(len(variables)):
        for col in range(len(variables)):
            value = coefficients[row, col]
            if not np.isfinite(value):
                continue
            faded = bool(insignificant[row, col])
            ax.text(
                col, row, f"{value:+.2f}" if row != col else "1",
                ha="center", va="center", fontsize=5.2,
                color="#222222", alpha=0.30 if faded else 0.92,
            )
    labels = [R2_MATRIX_LABELS.get(variable, variable) for variable in variables]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=6.5)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.025)
    symbol = "$r$" if matrix["method"] == "pearson" else "$\\rho$"
    colorbar.set_label(f"Correlation coefficient ({symbol})")
    ax.text(
        0.0, 1.015,
        f"BH-FDR family: {matrix['family_size']} unique pairs; "
        "non-significant cells are faded",
        transform=ax.transAxes, fontsize=7, color="#555555", va="bottom",
    )
    if plot_title:
        ax.set_title(f"{matrix['method'].title()} correlation audit")
    fig.tight_layout()
    _r2_save_figure(
        fig, save_path, for_latex, plot_title, figure_name,
        {
            "method": matrix["method"],
            "variables": variables,
            "family_size": matrix["family_size"],
            "coefficients": matrix["coefficients"],
            "p_adjusted": matrix["p_adjusted"],
            "adjustment_method": matrix["adjustment_method"],
            "sample_sizes": matrix["sample_sizes"],
        },
    )


def plot_r2_correlation_matrix_pearson(
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (10.2, 9.5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the complete Pearson correlation audit for R2.

    Args:
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    _plot_r2_correlation_matrix(
        analysis["correlation_matrices"]["pearson"], save_path, figsize,
        for_latex, plot_title, "r2_correlation_matrix_pearson",
    )


def plot_r2_correlation_matrix_spearman(
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (10.2, 9.5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot the complete Spearman correlation audit for R2.

    Args:
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    _plot_r2_correlation_matrix(
        analysis["correlation_matrices"]["spearman"], save_path, figsize,
        for_latex, plot_title, "r2_correlation_matrix_spearman",
    )


def plot_r2_population_reduction_contrast(
    df: pd.DataFrame,
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (8.4, 3.8),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Contrast population correlations with absolute and relative reduction.

    Args:
        df: Prepared city-attribution DataFrame.
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    specs = (
        (R2_ABSOLUTE_OUTCOME, "Annual optimal reduction (t CO$_2$e yr$^{-1}$)",
         analysis["population_contrast"]["absolute_reduction"]),
        (R2_OUTCOME, "Optimal lifecycle reduction (%)",
         analysis["population_contrast"]["relative_reduction"]),
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panel_records = []
    for ax, (outcome, y_label, record) in zip(axes, specs):
        plot_df = df[["region_key", "population_10k", outcome]].dropna().copy()
        population_million = plot_df["population_10k"] / 100.0
        ax.scatter(population_million, plot_df[outcome], s=20, color="#4C78A8",
                   alpha=0.52, edgecolor="white", linewidth=0.3)
        slope, intercept = np.polyfit(population_million, plot_df[outcome], 1)
        x_line = np.linspace(population_million.min(), population_million.max(), 120)
        ax.plot(x_line, slope * x_line + intercept, color="#D95F02", linewidth=1.5)
        city_records = []
        for idx, row in plot_df[plot_df["region_key"].isin(R2_FIGURE_CITY_KEYS)].iterrows():
            x_value = float(row["population_10k"] / 100.0)
            y_value = float(row[outcome])
            ax.scatter(x_value, y_value, s=38, facecolor="white", edgecolor="#222222",
                       linewidth=0.9, zorder=4)
            city_records.append({"region_key": row["region_key"], "x_population_million": x_value,
                                 "y": y_value})
        p_label = "$P < 0.001$" if record["p_value"] < 0.001 else f"$P = {record['p_value']:.3f}$"
        ci_low, ci_high = record["ci95"]
        equivalence = record["equivalence"]
        equivalence_label = ""
        if outcome == R2_OUTCOME:
            equivalence_label = (
                "\nequivalent within ±0.10"
                if equivalence["equivalent_0_05"]
                else "\nequivalence not established"
            )
        ax.text(
            0.03, 0.96,
            f"r = {record['coefficient']:+.3f} "
            f"[{ci_low:+.2f}, {ci_high:+.2f}]\n"
            f"{p_label}, n = {record['n']}{equivalence_label}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "#BBBBBB", "alpha": 0.9},
        )
        ax.set_xlabel("Population (million)")
        ax.set_ylabel(y_label)
        ax.grid(color="#E0E0E0", linewidth=0.5, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        panel_records.append({
            "outcome": outcome,
            "correlation": record,
            "linear_fit": {"slope": float(slope), "intercept": float(intercept)},
            "selected_cities": city_records,
        })
    if plot_title:
        fig.suptitle("City size predicts the reduction pool, not relative suitability")
    fig.tight_layout()
    _r2_save_figure(
        fig, save_path, for_latex, plot_title,
        "r2_population_reduction_contrast",
        {"panels": panel_records},
    )


def plot_r2_core_coefficient_robustness(
    analysis: dict,
    save_path: str = None,
    figsize: tuple = (8.8, 3.5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Compare core R2 coefficients across inference and sensitivity methods.

    Args:
        analysis: Shared R2 analysis result dictionary.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    summary = analysis["robustness_forest"]
    methods = [record["method"] for record in summary["terms"][0]["methods"]]
    method_labels = {
        "HC3": "HC3",
        "Province cluster": "Province cluster",
        "City bootstrap": "City bootstrap",
        "Sales-weighted WLS": "Sales-weighted WLS",
        "Leave one province out": "Leave-one-province-out",
    }
    colors = {
        "HC3": "#1F4E79",
        "Province cluster": "#4C78A8",
        "City bootstrap": "#72A0C1",
        "Sales-weighted WLS": "#D95F02",
        "Leave one province out": "#777777",
    }
    fig, axes = plt.subplots(1, len(summary["terms"]), figsize=figsize, sharex=True)
    all_limits = []
    for ax, term in zip(axes, summary["terms"]):
        y = np.arange(len(methods))[::-1]
        ax.axvspan(-0.10, 0.10, color="#F2F2F2", zorder=0)
        ax.axvline(0, color="#666666", linewidth=0.7, zorder=1)
        for position, record in zip(y, term["methods"]):
            low, high = record["interval"]
            estimate = record["estimate"]
            all_limits.extend([low, high])
            ax.plot([low, high], [position, position], color=colors[record["method"]],
                    linewidth=1.5, zorder=2)
            ax.scatter(estimate, position, s=30, color=colors[record["method"]],
                       edgecolor="white", linewidth=0.35, zorder=3)
        ax.set_yticks(y, [method_labels[method] for method in methods], fontsize=7)
        ax.set_title(term["label"], fontsize=9)
        ax.grid(axis="x", color="#E0E0E0", linewidth=0.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    padding = max((max(all_limits) - min(all_limits)) * 0.08, 0.04)
    axes[0].set_xlim(min(all_limits) - padding, max(all_limits) + padding)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[2].tick_params(axis="y", labelleft=False)
    fig.supxlabel("Standardized coefficient (95% CI; leave-one-out shows range)", fontsize=8)
    if plot_title:
        fig.suptitle("Core city-attribution coefficients are audited across specifications")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _r2_save_figure(
        fig, save_path, for_latex, plot_title,
        "r2_core_coefficient_robustness",
        {"robustness_forest": summary},
    )


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
DISTANCE_THRESHOLD_BASIS_COLORS = {
    "road": "#32688E",
    "straight": "#D9822B",
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


def _save_demand_distance_figure(
    fig,
    save_path: str,
    for_latex: bool,
    plot_title: bool,
    description: str,
):
    """Save one exploratory demand-distance figure."""
    if not save_path:
        plt.show()
        plt.close(fig)
        return None
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"{description} saved to {save_path}")
    plt.close(fig)
    return save_path


def plot_demand_distance_feature_correlations(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (9.0, 7.2),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot ranked city-feature correlations for both demand distances.

    Args:
        df: Long-form correlation results from demand-distance analysis.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required = {
        "outcome",
        "outcome_label",
        "feature_label",
        "n",
        "spearman_rho",
        "ci95_lower",
        "ci95_upper",
        "significant_fdr_0_05",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Correlation plot missing columns: {sorted(missing)}")

    outcomes = ("avg_straight_distance_m", "avg_road_distance_m")
    colors = {True: "#2166AC", False: "#A6A6A6"}
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for panel, (ax, outcome) in enumerate(zip(axes, outcomes)):
        plot_df = df[df["outcome"].eq(outcome)].dropna(
            subset=["spearman_rho", "ci95_lower", "ci95_upper"]
        ).copy()
        plot_df = plot_df.sort_values("spearman_rho", ascending=True)
        y = np.arange(len(plot_df))
        point_colors = [
            colors[bool(value)]
            for value in plot_df["significant_fdr_0_05"]
        ]
        lower_error = plot_df["spearman_rho"] - plot_df["ci95_lower"]
        upper_error = plot_df["ci95_upper"] - plot_df["spearman_rho"]
        ax.errorbar(
            plot_df["spearman_rho"],
            y,
            xerr=np.vstack([lower_error, upper_error]),
            fmt="none",
            ecolor="#777777",
            elinewidth=0.8,
            capsize=1.8,
            zorder=1,
        )
        ax.scatter(
            plot_df["spearman_rho"], y, c=point_colors, s=24,
            edgecolor="white", linewidth=0.4, zorder=2,
        )
        ax.axvline(0, color="#444444", linewidth=0.7, linestyle="--", zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([
            f"{label} (n={int(n)})"
            for label, n in zip(plot_df["feature_label"], plot_df["n"])
        ], fontsize=7.2)
        outcome_label = (
            str(plot_df["outcome_label"].iloc[0])
            if not plot_df.empty else outcome
        )
        ax.set_xlabel(
            f"Spearman $\\rho$ with {outcome_label.lower()}\n"
            "(95% bootstrap CI)"
        )
        ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
        ax.text(
            0.01, 0.99, chr(ord("a") + panel), transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", fontsize=10,
        )
        if plot_title:
            ax.set_title(outcome_label)
    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="none", color=colors[True],
                       label="BH-FDR q < 0.05", markersize=5),
            plt.Line2D([], [], marker="o", linestyle="none", color=colors[False],
                       label="Not FDR-significant", markersize=5),
        ],
        loc="upper left", frameon=False, fontsize=7,
    )
    if plot_title:
        fig.suptitle("City features associated with modeled demand distance")
    fig.tight_layout()
    return _save_demand_distance_figure(
        fig, save_path, for_latex, plot_title,
        "Demand-distance feature correlation figure",
    )


def _log_linear_confidence_band(x: np.ndarray, y: np.ndarray):
    """Return log-linear mean predictions and their 95% confidence limits."""
    log_x = np.log(x)
    log_y = np.log(y)
    design = sm.add_constant(log_x, has_constant="add")
    result = sm.OLS(log_y, design).fit()
    x_line = np.geomspace(float(x.min()), float(x.max()), 160)
    prediction_design = sm.add_constant(np.log(x_line), has_constant="add")
    frame = result.get_prediction(prediction_design).summary_frame(alpha=0.05)
    return (
        x_line,
        np.exp(frame["mean"].to_numpy(dtype=float)),
        np.exp(frame["mean_ci_lower"].to_numpy(dtype=float)),
        np.exp(frame["mean_ci_upper"].to_numpy(dtype=float)),
    )


def plot_demand_distance_key_scatters(
    df: pd.DataFrame,
    correlations: pd.DataFrame,
    top_form_feature: str,
    save_path: str = None,
    figsize: tuple = (9.0, 3.0),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot population, affluence, and strongest-form demand-distance relations.

    Args:
        df: Prepared city-level region statistics.
        correlations: Long-form demand-distance correlation results.
        top_form_feature: Selected strongest urban-form feature.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    outcome = "avg_straight_distance_m"
    features = ("population_10k", "gdp_per_capita", top_form_feature)
    label_lookup = dict(
        correlations[["feature", "feature_label"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for panel, (ax, feature) in enumerate(zip(axes, features)):
        pair = df[[feature, outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        pair = pair[
            np.isfinite(pair[feature])
            & np.isfinite(pair[outcome])
            & pair[feature].gt(0)
            & pair[outcome].gt(0)
        ]
        if len(pair) < 3:
            raise ValueError(f"Insufficient positive observations for {feature}")
        x = pair[feature].to_numpy(dtype=float)
        y = pair[outcome].to_numpy(dtype=float) / 1000.0
        x_line, y_line, lower, upper = _log_linear_confidence_band(x, y)
        ax.scatter(
            x, y, s=13, color="#4C78A8", alpha=0.58,
            edgecolor="white", linewidth=0.25, zorder=2,
        )
        ax.plot(x_line, y_line, color="#B33A3A", linewidth=1.3, zorder=3)
        ax.fill_between(x_line, lower, upper, color="#B33A3A", alpha=0.14, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label_lookup.get(feature, feature))
        if panel == 0:
            ax.set_ylabel("Mean straight-line demand distance (km)")
        ax.grid(color="#E8E8E8", linewidth=0.5)
        record = correlations[
            correlations["outcome"].eq(outcome)
            & correlations["feature"].eq(feature)
        ].iloc[0]
        ax.text(
            0.97, 0.04,
            f"$\\rho$={record['spearman_rho']:+.2f}\n"
            f"q={record['q_value']:.2g}; n={int(record['n'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                  "edgecolor": "#CCCCCC", "alpha": 0.88},
        )
        ax.text(
            0.02, 0.98, chr(ord("a") + panel), transform=ax.transAxes,
            va="top", ha="left", fontweight="bold", fontsize=10,
        )
        if plot_title:
            ax.set_title(label_lookup.get(feature, feature))
    if plot_title:
        fig.suptitle("Key correlates of modeled straight-line demand distance")
    fig.tight_layout()
    return _save_demand_distance_figure(
        fig, save_path, for_latex, plot_title,
        "Demand-distance key scatter figure",
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


def _threshold_summary_records(summary_df: pd.DataFrame) -> list[dict]:
    """Return JSON-safe policy records for figure reports."""
    columns = (
        "region_key",
        "region_name_en",
        "distance_basis",
        "policy_case",
        "optimal_threshold_km",
        "annual_reduction_pct",
        "drone_order_ratio_pct",
        "n_od_pairs_optimized",
        "n_od_pairs_unreachable",
        "unreachable_annual_orders",
    )
    records = []
    for _, row in summary_df.loc[:, list(columns)].iterrows():
        records.append(
            {
                "region_key": row["region_key"],
                "city": row["region_name_en"],
                "distance_basis": row["distance_basis"],
                "policy_case": row["policy_case"],
                "optimal_threshold_km": float(row["optimal_threshold_km"]),
                "annual_reduction_pct": float(row["annual_reduction_pct"]),
                "drone_order_ratio_pct": float(row["drone_order_ratio_pct"]),
                "n_od_pairs_optimized": int(row["n_od_pairs_optimized"]),
                "n_od_pairs_unreachable": int(row["n_od_pairs_unreachable"]),
                "unreachable_annual_orders": float(
                    row["unreachable_annual_orders"]
                ),
            }
        )
    return records


def _style_distance_threshold_axis(ax):
    """Apply restrained exploratory-analysis styling."""
    ax.grid(color="#D9D9D9", linewidth=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_distance_threshold_typical_city_curves(
    curve_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10.2, 8.2),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot threshold-response curves for four representative cities.

    Args:
        curve_df: Long-form compact threshold-response curves.
        summary_df: Long-form exact optimum records for the same cities.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required_curve = {
        "region_key",
        "distance_basis",
        "threshold_km",
        "annual_reduction_pct",
        "is_exact_optimum",
    }
    required_summary = {
        "region_key",
        "region_name_en",
        "distance_basis",
        "optimal_threshold_km",
        "annual_reduction_pct",
        "drone_order_ratio_pct",
    }
    missing_curve = required_curve.difference(curve_df.columns)
    missing_summary = required_summary.difference(summary_df.columns)
    if missing_curve or missing_summary:
        raise KeyError(
            "Missing threshold figure columns: "
            f"curve={sorted(missing_curve)}, summary={sorted(missing_summary)}"
        )

    cities = list(DISTANCE_THRESHOLD_TYPICAL_CITIES)
    curve_plot = curve_df[curve_df["region_key"].isin(cities)].copy()
    summary_plot = summary_df[summary_df["region_key"].isin(cities)].copy()
    if set(curve_plot["region_key"]) != set(cities):
        missing = sorted(set(cities).difference(curve_plot["region_key"]))
        raise ValueError(f"Missing typical-city curves: {', '.join(missing)}")

    x_values = pd.to_numeric(curve_plot["threshold_km"], errors="coerce")
    y_values = pd.to_numeric(curve_plot["annual_reduction_pct"], errors="coerce")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Typical-city curve coordinates must be finite")
    x_max = float(x_values.max())
    y_min = min(0.0, float(y_values.min()))
    y_max = max(0.0, float(y_values.max()))
    x_pad = max(x_max * 0.025, 0.1)
    y_pad = max((y_max - y_min) * 0.07, 1.0)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()
    panel_letters = "abcdef"
    for panel_idx, (ax, city) in enumerate(zip(axes, cities)):
        city_curve = curve_plot[curve_plot["region_key"] == city]
        city_summary = summary_plot[summary_plot["region_key"] == city]
        for basis in ("road", "straight"):
            color = DISTANCE_THRESHOLD_BASIS_COLORS[basis]
            basis_curve = city_curve[
                city_curve["distance_basis"] == basis
            ].sort_values("threshold_km")
            exact = basis_curve[basis_curve["is_exact_optimum"]]
            if len(exact) != 1:
                raise ValueError(
                    f"{city}/{basis} must have exactly one exact optimum"
                )
            ax.plot(
                basis_curve["threshold_km"],
                basis_curve["annual_reduction_pct"],
                color=color,
                linewidth=1.7,
                alpha=0.96,
                label=basis.title(),
            )
            ax.scatter(
                exact["threshold_km"],
                exact["annual_reduction_pct"],
                s=42,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

            optimum = city_summary[city_summary["distance_basis"] == basis]
            if len(optimum) != 1:
                raise ValueError(f"{city}/{basis} summary must have one row")
            optimum = optimum.iloc[0]
            text_y = 0.89 if basis == "road" else 0.74
            ax.text(
                0.03,
                text_y,
                (
                    f"{basis.title()}: d*={optimum['optimal_threshold_km']:.2f} km\n"
                    f"{optimum['annual_reduction_pct']:.1f}% reduction; "
                    f"{optimum['drone_order_ratio_pct']:.1f}% drone"
                ),
                transform=ax.transAxes,
                fontsize=7.2,
                color=color,
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.6,
                    "alpha": 0.88,
                },
            )

        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_xlim(-x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        _style_distance_threshold_axis(ax)
        ax.text(
            0.01,
            1.02,
            f"{panel_letters[panel_idx]}  {DISTANCE_THRESHOLD_CITY_LABELS[city]}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

    fig.supxlabel("Policy threshold, d (km)", fontsize=10)
    fig.supylabel("Annual lifecycle reduction vs all e-bike (%)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    if plot_title:
        fig.suptitle(
            "City-specific carbon benefits emerge from distinct threshold responses",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
    fig.tight_layout(rect=(0.025, 0.045, 1, 0.91), h_pad=1.7, w_pad=1.1)

    return _save_distance_threshold_figure(
        fig,
        save_path,
        for_latex,
        plot_title,
        "Typical-city distance-threshold curves",
        {
            "figure_name": "distance_threshold_typical_city_curves",
            "core_conclusion": (
                "Cities exhibit distinct threshold-response shapes, optimal cutoffs, "
                "and drone adoption levels."
            ),
            "city_count": len(cities),
            "cities": cities,
            "distance_bases": ["road", "straight"],
            "shared_axes": True,
            "summary_records": _threshold_summary_records(summary_plot),
        },
    )


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


def _pilot_projected_density(od_df, point_prefix, resolution_m=500.0):
    """Return a physically scaled density raster for eligible OD endpoints."""
    from pyproj import Transformer
    from scipy.ndimage import gaussian_filter
    from utils.tools import get_epsg_from_bounds

    all_lat = np.concatenate(
        [od_df["from_lat"].to_numpy(), od_df["to_lat"].to_numpy()]
    )
    all_lng = np.concatenate(
        [od_df["from_lng"].to_numpy(), od_df["to_lng"].to_numpy()]
    )
    bounds = {
        "min_lat": float(np.nanmin(all_lat)),
        "max_lat": float(np.nanmax(all_lat)),
        "min_lng": float(np.nanmin(all_lng)),
        "max_lng": float(np.nanmax(all_lng)),
    }
    epsg = get_epsg_from_bounds(bounds)
    transformer = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{epsg}", always_xy=True
    )
    origin_x, origin_y = transformer.transform(
        od_df["from_lng"].to_numpy(dtype=float),
        od_df["from_lat"].to_numpy(dtype=float),
    )
    destination_x, destination_y = transformer.transform(
        od_df["to_lng"].to_numpy(dtype=float),
        od_df["to_lat"].to_numpy(dtype=float),
    )
    all_x = np.concatenate([origin_x, destination_x])
    all_y = np.concatenate([origin_y, destination_y])
    padding = 1000.0
    x_min = np.floor((all_x.min() - padding) / resolution_m) * resolution_m
    x_max = np.ceil((all_x.max() + padding) / resolution_m) * resolution_m
    y_min = np.floor((all_y.min() - padding) / resolution_m) * resolution_m
    y_max = np.ceil((all_y.max() + padding) / resolution_m) * resolution_m
    x_edges = np.arange(x_min, x_max + resolution_m, resolution_m)
    y_edges = np.arange(y_min, y_max + resolution_m, resolution_m)
    if point_prefix == "from":
        x, y = origin_x, origin_y
    else:
        x, y = destination_x, destination_y
    weights = od_df["annual_sales"].to_numpy(dtype=float)
    histogram, _, _ = np.histogram2d(
        x, y, bins=(x_edges, y_edges), weights=weights
    )
    smooth = gaussian_filter(histogram, sigma=1000.0 / resolution_m)
    total_weight = float(weights.sum())
    cell_area_km2 = (resolution_m / 1000.0) ** 2
    density = smooth / total_weight * 100.0 / cell_area_km2
    return {
        "density": density.T,
        "extent_km": np.array([x_min, x_max, y_min, y_max]) / 1000.0,
        "x_centers_km": (x_edges[:-1] + x_edges[1:]) / 2000.0,
        "y_centers_km": (y_edges[:-1] + y_edges[1:]) / 2000.0,
    }


def _pilot_mass_contour_levels(density, fractions=(0.8, 0.5)):
    """Return density thresholds enclosing the requested probability masses."""
    values = np.asarray(density, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 2 or values.sum() <= 0:
        return []
    descending = np.sort(values)[::-1]
    cumulative = np.cumsum(descending) / descending.sum()
    levels = []
    for fraction in fractions:
        index = min(int(np.searchsorted(cumulative, fraction)), len(descending) - 1)
        levels.append(float(descending[index]))
    return sorted(set(levels))


def plot_representative_city_drone_heatmaps(
    city_payloads,
    save_path: str = None,
    figsize: tuple = (12, 10),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot normalized eligible-order heatmaps for representative city types.

    Args:
        city_payloads: Dictionaries containing city metrics and eligible OD data.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from matplotlib.colors import Normalize

    ordered_typologies = (
        "High concentration / High volume",
        "High concentration / Low volume",
        "Low concentration / High volume",
        "Low concentration / Low volume",
    )
    payloads = sorted(
        city_payloads,
        key=lambda payload: ordered_typologies.index(
            payload["metrics"].get("representative_typology")
            or payload["metrics"]["pilot_typology"]
        ) if (
            payload["metrics"].get("representative_typology")
            or payload["metrics"]["pilot_typology"]
        ) in ordered_typologies else 99,
    )
    prepared = []
    for payload in payloads:
        destination = _pilot_projected_density(payload["od_df"], "to")
        origin = _pilot_projected_density(payload["od_df"], "from")
        prepared.append((payload, destination, origin))
    positive_values = np.concatenate(
        [item[1]["density"][item[1]["density"] > 0] for item in prepared]
    )
    vmax = float(np.nanpercentile(positive_values, 99.5))
    norm = Normalize(vmin=0.0, vmax=max(vmax, 1e-12))

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    axes_flat = axes.ravel()
    image_artist = None
    for panel_index, (ax, item) in enumerate(zip(axes_flat, prepared)):
        payload, destination, origin = item
        metrics = payload["metrics"]
        density = destination["density"]
        image_artist = ax.imshow(
            np.ma.masked_where(density <= 0, density),
            extent=destination["extent_km"],
            origin="lower",
            cmap="YlOrRd",
            norm=norm,
            interpolation="bilinear",
            zorder=2,
        )
        contour_levels = _pilot_mass_contour_levels(origin["density"])
        if contour_levels:
            ax.contour(
                origin["x_centers_km"],
                origin["y_centers_km"],
                origin["density"],
                levels=contour_levels,
                colors="#2457A7",
                linewidths=(1.0, 1.7)[: len(contour_levels)],
                zorder=4,
            )
        center_x = metrics.get("capture_3km_center_x_m", np.nan)
        center_y = metrics.get("capture_3km_center_y_m", np.nan)
        if np.isfinite(center_x) and np.isfinite(center_y):
            ax.add_patch(
                Circle(
                    (center_x / 1000.0, center_y / 1000.0),
                    radius=3.0,
                    fill=False,
                    color="#1B7837",
                    linewidth=1.8,
                    linestyle="--",
                    zorder=5,
                )
            )
        city_name = metrics.get("region_name_en") or metrics["region_key"]
        display_typology = (
            metrics.get("representative_typology") or metrics["pilot_typology"]
        )
        ax.text(
            0.02,
            0.98,
            f"{chr(97 + panel_index)}  {city_name}\n{display_typology}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2.5),
            zorder=8,
        )
        ax.text(
            0.02,
            0.02,
            f"MPCS {metrics['pilot_concentration_score_pct']:.1f}%\n"
            f"Eligible {metrics['eligible_annual_orders']:,.0f} orders yr$^{{-1}}$ "
            f"({metrics['eligible_order_share_pct']:.1f}%)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", alpha=0.86, pad=2.5),
            zorder=8,
        )
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        ax.set_aspect("equal")
        ax.grid(False)
    for ax in axes_flat[len(prepared):]:
        ax.axis("off")
    if image_artist is not None:
        colorbar = fig.colorbar(
            image_artist, ax=axes_flat.tolist(), shrink=0.82, pad=0.02
        )
        colorbar.set_label("Eligible destination orders (% of city total km$^{-2}$)")
    legend_handles = [
        Line2D([0], [0], color="#2457A7", linewidth=1.5, label="Origin density contours"),
        Line2D([0], [0], color="#1B7837", linewidth=1.8, linestyle="--", label="Optimal 3-km pilot area"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.015),
        frameon=False,
    )
    if plot_title:
        fig.suptitle(
            "Representative spatial patterns of drone-favorable orders",
            fontsize=14,
            fontweight="bold",
        )
    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Representative drone-order heatmaps saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return save_path


def plot_city_reduction_concentration_scatter(
    df: pd.DataFrame,
    scatter_save_path: str = None,
    distribution_save_path: str = None,
    scatter_figsize: tuple = (130.5 / 25.4, 100 / 25.4),
    distribution_figsize: tuple = (43.5 / 25.4, 100 / 25.4),
    for_latex: bool = False,
):
    """Plot city reduction potential and MPCS distribution separately.

    Args:
        df: Successful city concentration results with Oracle reduction metrics.
        scatter_save_path: Path for the reduction-potential scatter figure.
        distribution_save_path: Path for the MPCS distribution figure.
        scatter_figsize: Scatter figure size tuple.
        distribution_figsize: Distribution figure size tuple.
        for_latex: If True, save as PDF.
    """
    required = [
        "pilot_concentration_score_pct",
        "concentration_class",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"City reduction scatter missing columns: {missing}")
    source_df = df.copy()
    for column in (
        "pilot_concentration_score_pct",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ):
        source_df[column] = pd.to_numeric(source_df[column], errors="coerce")
    distribution_df = source_df[
        np.isfinite(source_df["pilot_concentration_score_pct"])
        & source_df["concentration_class"].isin(("Low", "Medium", "High"))
    ].copy()
    plot_df = distribution_df[
        np.isfinite(distribution_df[PILOT_ABSOLUTE_REDUCTION_COL])
        & (distribution_df[PILOT_ABSOLUTE_REDUCTION_COL] > 0)
        & np.isfinite(distribution_df[PILOT_RELATIVE_REDUCTION_COL])
    ].copy()
    if distribution_df.empty:
        raise ValueError("No finite city concentration results to plot")
    if plot_df.empty:
        raise ValueError("No finite positive city reduction results to plot")
    absolute_median = float(plot_df[PILOT_ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[PILOT_RELATIVE_REDUCTION_COL].median())
    class_colors = {"Low": "#D95F0E", "Medium": "#7570B3", "High": "#1B9E77"}

    fig, ax = plt.subplots(figsize=scatter_figsize)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)
    for concentration_class in ("Low", "Medium", "High"):
        group = plot_df[plot_df["concentration_class"].eq(concentration_class)]
        if group.empty:
            continue
        ax.scatter(
            group[PILOT_ABSOLUTE_REDUCTION_COL],
            group[PILOT_RELATIVE_REDUCTION_COL],
            s=30,
            alpha=0.72,
            color=class_colors.get(concentration_class, "#777777"),
            edgecolor="white",
            linewidth=0.35,
            label=f"{concentration_class} MPCS",
            zorder=3,
        )
    ax.axhline(relative_median, color="#666666", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(absolute_median, color="#666666", linestyle="--", linewidth=1, zorder=1)
    ax.set_xscale("log")

    name_col = "region_name_en" if "region_name_en" in plot_df else "region_key"
    labeled_city_order = tuple(
        city_name for city_name, _ in DRONE_SPATIAL_SCATTER_LABELED_CITIES
    )
    labeled = plot_df[plot_df[name_col].isin(labeled_city_order)].copy()
    labeled["_label_order"] = pd.Categorical(
        labeled[name_col], categories=labeled_city_order, ordered=True
    )
    labeled = labeled.sort_values("_label_order")
    ax.scatter(
        labeled[PILOT_ABSOLUTE_REDUCTION_COL],
        labeled[PILOT_RELATIVE_REDUCTION_COL],
        s=30,
        facecolors="none",
        edgecolors="#555555",
        linewidth=0.8,
        zorder=5,
    )
    label_texts = [
        ax.text(
            row[PILOT_ABSOLUTE_REDUCTION_COL],
            row[PILOT_RELATIVE_REDUCTION_COL],
            row[name_col],
            fontsize=7.5,
            color="#202020",
            zorder=6,
        )
        for _, row in labeled.iterrows()
    ]
    if label_texts:
        from adjustText import adjust_text

        adjust_text(
            label_texts,
            x=labeled[PILOT_ABSOLUTE_REDUCTION_COL].to_numpy(dtype=float),
            y=labeled[PILOT_RELATIVE_REDUCTION_COL].to_numpy(dtype=float),
            ax=ax,
            expand=(1.12, 1.22),
            force_text=(0.15, 0.25),
            force_static=(0.08, 0.15),
            ensure_inside_axes=True,
            arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.5},
        )

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1),
    )
    labels.append("Median values")
    ax.set_xlabel(
        "Annual absolute emission reduction (t CO$_2$e yr$^{-1}$)",
        fontsize=9,
    )
    ax.set_ylabel("Relative lifecycle emission reduction (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig_distribution, ax_distribution = plt.subplots(figsize=distribution_figsize)
    fig_distribution.subplots_adjust(left=0.32, right=0.96, bottom=0.16, top=0.98)
    concentration_values = distribution_df[
        "pilot_concentration_score_pct"
    ].to_numpy(dtype=float)
    if concentration_values.size < 2 or np.unique(concentration_values).size < 2:
        raise ValueError(
            "At least two distinct city concentration values are required for KDE"
        )
    lower_tertile, upper_tertile = np.quantile(
        concentration_values,
        (1 / 3, 2 / 3),
    )
    concentration_kde = stats.gaussian_kde(
        concentration_values,
        bw_method="scott",
    )
    concentration_support = np.linspace(0.0, 100.0, 512)
    concentration_density = concentration_kde(concentration_support)
    concentration_regions = (
        (0.0, lower_tertile, class_colors["Low"]),
        (lower_tertile, upper_tertile, class_colors["Medium"]),
        (upper_tertile, 100.0, class_colors["High"]),
    )
    for lower_bound, upper_bound, color in concentration_regions:
        region_support = concentration_support[
            (concentration_support >= lower_bound)
            & (concentration_support <= upper_bound)
        ]
        region_support = np.unique(
            np.concatenate(([lower_bound], region_support, [upper_bound]))
        )
        ax_distribution.fill_betweenx(
            region_support,
            0.0,
            concentration_kde(region_support),
            color=color,
            alpha=0.65,
            linewidth=0,
            zorder=1,
        )
    for quantile in (lower_tertile, upper_tertile):
        ax_distribution.axhline(
            quantile,
            color="#666666",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
    ax_distribution.plot(
        concentration_density,
        concentration_support,
        color="#3F3F3F",
        linewidth=1.2,
        zorder=3,
    )
    ax_distribution.set_ylim(0, 100)
    ax_distribution.set_xlim(left=0)
    ax_distribution.set_yticks([0, 25, 50, 75, 100])
    ax_distribution.set_xlabel("Probability density", fontsize=9)
    ax_distribution.set_ylabel(
        "Multi-scale Pilot Capture Score (MPCS, %)",
        fontsize=9,
    )
    ax_distribution.tick_params(axis="both", labelsize=8)
    ax_distribution.spines[["top", "right"]].set_visible(False)

    def _save_panel(panel_fig, save_path, description):
        if not save_path:
            panel_fig.show()
            return None
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        panel_fig.savefig(
            save_path,
            dpi=300,
            bbox_inches=panel_fig.bbox_inches,
            pad_inches=0.0,
        )
        print(f"{description} saved to {save_path}")
        return save_path

    scatter_path = _save_panel(
        fig,
        scatter_save_path,
        "City reduction-potential scatter",
    )
    distribution_path = _save_panel(
        fig_distribution,
        distribution_save_path,
        "City concentration distribution",
    )
    plt.close(fig)
    plt.close(fig_distribution)
    return scatter_path, distribution_path


def plot_city_reduction_concentration_marker_scatter(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (7, 5),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot city reduction metrics with concentration encoded by marker shape.

    Args:
        df: Successful city concentration results with Oracle reduction metrics.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    required = [
        "pilot_concentration_score_pct",
        "concentration_class",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"City reduction marker scatter missing columns: {missing}")

    plot_df = df.copy()
    for column in (
        "pilot_concentration_score_pct",
        PILOT_ABSOLUTE_REDUCTION_COL,
        PILOT_RELATIVE_REDUCTION_COL,
    ):
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df[
        np.isfinite(plot_df["pilot_concentration_score_pct"])
        & plot_df["concentration_class"].isin(("Low", "Medium", "High"))
        & np.isfinite(plot_df[PILOT_ABSOLUTE_REDUCTION_COL])
        & (plot_df[PILOT_ABSOLUTE_REDUCTION_COL] > 0)
        & np.isfinite(plot_df[PILOT_RELATIVE_REDUCTION_COL])
    ].copy()
    if plot_df.empty:
        raise ValueError("No finite positive city reduction results to plot")

    absolute_median = float(plot_df[PILOT_ABSOLUTE_REDUCTION_COL].median())
    relative_median = float(plot_df[PILOT_RELATIVE_REDUCTION_COL].median())
    class_styles = {
        "Low": {"marker": "o", "size": 30},
        "Medium": {"marker": "s", "size": 28},
        "High": {"marker": "^", "size": 36},
    }

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for concentration_class in ("Low", "Medium", "High"):
        group = plot_df[plot_df["concentration_class"].eq(concentration_class)]
        if group.empty:
            continue
        style = class_styles[concentration_class]
        ax.scatter(
            group[PILOT_ABSOLUTE_REDUCTION_COL],
            group[PILOT_RELATIVE_REDUCTION_COL],
            s=style["size"],
            marker=style["marker"],
            alpha=0.72,
            color="#555555",
            edgecolor="white",
            linewidth=0.35,
            label=f"{concentration_class} concentration",
            zorder=3,
        )
    ax.axhline(
        relative_median,
        color="#666666",
        linestyle="--",
        linewidth=1,
        zorder=1,
    )
    ax.axvline(
        absolute_median,
        color="#666666",
        linestyle="--",
        linewidth=1,
        zorder=1,
    )

    name_col = "region_name_en" if "region_name_en" in plot_df else "region_key"
    labeled = plot_df[plot_df[name_col].isin(PILOT_SUPER_LARGE_CITIES)].copy()
    labeled["_label_order"] = pd.Categorical(
        labeled[name_col], categories=PILOT_SUPER_LARGE_CITIES, ordered=True
    )
    labeled = labeled.sort_values("_label_order")
    label_texts = [
        ax.text(
            row[PILOT_ABSOLUTE_REDUCTION_COL],
            row[PILOT_RELATIVE_REDUCTION_COL],
            row[name_col],
            fontsize=7.5,
            color="#202020",
            zorder=6,
        )
        for _, row in labeled.iterrows()
    ]
    if label_texts:
        from adjustText import adjust_text

        adjust_text(
            label_texts,
            x=labeled[PILOT_ABSOLUTE_REDUCTION_COL].to_numpy(dtype=float),
            y=labeled[PILOT_RELATIVE_REDUCTION_COL].to_numpy(dtype=float),
            ax=ax,
            expand=(1.12, 1.22),
            force_text=(0.15, 0.25),
            force_static=(0.08, 0.15),
            ensure_inside_axes=True,
            arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.5},
        )

    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="#666666", linestyle="--", linewidth=1)
    )
    labels.append("City medians")
    ax.set_xscale("log")
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    median_y_fraction = float(
        np.clip((relative_median - y_min) / (y_max - y_min), 0.0, 1.0)
    )
    ax.axhspan(
        y_min,
        relative_median,
        facecolor="#BDBDBD",
        alpha=0.18,
        zorder=-2,
    )
    ax.axvspan(
        x_min,
        absolute_median,
        ymin=median_y_fraction,
        ymax=1.0,
        facecolor="#E6B85C",
        alpha=0.18,
        zorder=-2,
    )
    ax.axvspan(
        absolute_median,
        x_max,
        ymin=median_y_fraction,
        ymax=1.0,
        facecolor="#5AAE61",
        alpha=0.18,
        zorder=-2,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Annual absolute emission reduction (t CO$_2$e yr$^{-1}$)")
    ax.set_ylabel("Relative lifecycle emission reduction (%)")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", color="#E5E5E5", linewidth=0.55, zorder=0)
    if plot_title:
        ax.set_title(
            "City emission-reduction potential by concentration class",
            fontsize=12,
            fontweight="bold",
        )
    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"City reduction marker scatter saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return save_path
