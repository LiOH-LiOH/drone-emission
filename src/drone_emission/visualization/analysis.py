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

from visualization._utils import _add_title_suffix, saved_tight_axis_report
from visualization.macro_regions import (
    CHINA_MACRO_REGION_SPECS,
    macro_region_for_province,
)

# Import from constants module (no circular dependency)
from analysis.constants import PROVINCE_NAME_EN, VARIABLE_LABELS


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


def _bubble_sizes_from_values(values, size_min=38.0, size_max=860.0):
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


def _draw_bubble_size_legend(ax, values, size_min=38.0, size_max=860.0):
    """Draw an in-map legend for absolute reduction bubble sizes."""
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    values = values[np.isfinite(values) & (values > 0)]
    if values.empty:
        return None
    legend_values = np.nanquantile(values, [0.25, 0.5, 0.9])
    legend_values = np.unique(np.round(legend_values, 1))
    handles = []
    labels = []
    vmax = float(values.max())
    for value in legend_values:
        if value <= 0:
            continue
        marker_size = size_min + np.sqrt(value / vmax) * (size_max - size_min)
        handles.append(
            plt.scatter(
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
        title="E-bike - Oracle\n(t CO2eq / yr)",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.31),
        frameon=True,
        framealpha=0.92,
        fontsize=8,
        title_fontsize=8,
        borderpad=0.7,
        labelspacing=1.0,
        handletextpad=1.2,
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


def _summary_bar_limits(
    drone_share_median: dict,
    annual_saved_total_kg: dict,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return shared limits for map inset summary bars."""
    share_values = np.array(list(drone_share_median.values()), dtype=float)
    share_values = share_values[np.isfinite(share_values)]
    saved_values = np.array(
        [value / 1e6 for value in annual_saved_total_kg.values()],
        dtype=float,
    )
    saved_values = saved_values[np.isfinite(saved_values)]
    share_limits = (
        0.0,
        min(105.0, max(100.0, float(share_values.max()) * 1.12))
        if share_values.size
        else 100.0,
    )
    if saved_values.size:
        saved_limits = (0.0, max(float(saved_values.max()) * 1.12, 1.0))
    else:
        saved_limits = (0.0, 1.0)
    return share_limits, saved_limits


def _add_map_summary_bars(
    ax_map,
    drone_share_pct: float,
    saved_kt: float,
    share_limits: tuple[float, float],
    saved_limits: tuple[float, float],
):
    """Add two-bar inset for drone share and CO2 saved inside a map panel."""
    inset = ax_map.inset_axes([0.035, 0.08, 0.31, 0.26], zorder=10)
    inset.set_facecolor((1, 1, 1, 0.90))
    inset_saved = inset.twinx()

    inset.bar(
        [0],
        [drone_share_pct],
        width=0.55,
        color="#2166ac",
        alpha=0.85,
        zorder=3,
    )
    inset_saved.bar(
        [1],
        [saved_kt],
        width=0.55,
        color="#1b7837",
        alpha=0.85,
        zorder=3,
    )

    inset.set_xlim(-0.6, 1.6)
    inset.set_ylim(*share_limits)
    inset_saved.set_ylim(*saved_limits)
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["Share", "Saved"], fontsize=6.5)
    inset.tick_params(axis="y", labelsize=6, colors="#2166ac", length=2)
    inset_saved.tick_params(axis="y", labelsize=6, colors="#1b7837", length=2)
    inset.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.35, zorder=0)
    for spine in inset.spines.values():
        spine.set_color("#888888")
        spine.set_linewidth(0.4)
    for spine in inset_saved.spines.values():
        spine.set_visible(False)

    inset.text(
        0,
        drone_share_pct,
        f"{drone_share_pct:.0f}%",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#2166ac",
        zorder=5,
    )
    saved_label = f"{saved_kt:.0f}"
    inset_saved.text(
        1,
        saved_kt,
        saved_label,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#1b7837",
        zorder=5,
    )
    return inset, inset_saved


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
    summary_axes,
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

    summary_inset = None
    if summary_axes is not None:
        inset, inset_saved = summary_axes
        summary_inset = {
            "axis_share": _axis_report(inset),
            "axis_saved": _axis_report(inset_saved),
            "bars": [
                {
                    "label": "Drone share",
                    "value": _json_safe_scalar(drone_share_median),
                    "unit": "%",
                    "color": "#2166ac",
                },
                {
                    "label": "CO2 saved",
                    "value": _json_safe_scalar(annual_saved_kg / 1e6),
                    "unit": "kt CO2e/yr",
                    "color": "#1b7837",
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
        "summary_bar_inset": summary_inset,
    }


def _write_regions_map_pathway_scenarios_report(
    save_path: str,
    figsize: tuple,
    named_csv_path: str,
    stats_csv_path: str,
    panels: list[tuple],
    panel_dfs: dict,
    panel_named: dict,
    panel_axes: dict,
    summary_bar_axes: dict,
    curve_data: dict,
    curve_axes: list,
    curve_twins: list,
    province_order: list[str],
    annual_saved_total_kg: dict,
    drone_share_median: dict,
    shared_cmap,
    bin_edges,
    reduction_edges,
    cbar_ax,
    panel_display_labels: dict = None,
    curve_display_labels: dict = None,
):
    """Write a compact JSON report for regions_map_pathway_scenarios."""
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    scenario_reports = {}
    color_col = "max_lifecycle_reduction_pct"
    panel_display_labels = panel_display_labels or {}
    curve_display_labels = curve_display_labels or {}
    for label, title, kappa, gef_mult, _strategy in panels:
        rows = panel_named.get(label, pd.DataFrame())
        share_col = _scenario_drone_share_column(rows) if not rows.empty else "drone_ratio_pct"
        ax_map, ax_box, ax_line = panel_axes[label]
        display_label = panel_display_labels.get(label, label)
        scenario_reports[f"{display_label}_{title.lower().replace(' ', '_').replace('+', 'plus')}"] = (
            _pathway_scenario_panel_report(
                rows=rows,
                panel_df=panel_dfs[label],
                color_col=color_col,
                ax_map=ax_map,
                ax_box=ax_box,
                ax_line=ax_line,
                summary_axes=summary_bar_axes.get(label),
                share_col=share_col,
                province_order=province_order,
                panel_label=display_label,
                title=title,
                kappa=kappa,
                gef_mult=gef_mult,
                annual_saved_kg=annual_saved_total_kg.get(label, np.nan),
                drone_share_median=drone_share_median.get(label, np.nan),
            )
        )

    payload = {
        "figure_name": "regions_map_pathway_scenarios",
        "figure_path": save_path,
        "source_named_csv_path": named_csv_path,
        "source_stats_csv_path": stats_csv_path,
        "figure_size_inches": {"width": figsize[0], "height": figsize[1]},
        "coordinate_convention": (
            "Axis bbox values are figure-fraction coordinates. Map colors encode "
            "reduction vs today's all-ebike anchor; gray indicates unstudied regions."
        ),
        "global_encodings": {
            "map_colorbar": _colorbar_report(
                cbar_ax,
                bin_edges,
                reduction_edges,
                "Reduction vs today's all-ebike anchor (%)",
                shared_cmap,
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
            "scenario_summary_bar_colors": {
                "drone_share": "#2166ac",
                "co2_saved": "#1b7837",
            },
        },
        "panels": {
            **scenario_reports,
            f"{curve_display_labels.get('gef', 'a')}_gef_pathway_curve": _pathway_curve_report(
                curve_axes[0],
                curve_twins[0],
                curve_data.get("gef"),
                "horizontal",
                co2_saved_mark="bar",
                co2_saved_baseline=0.0,
            ),
            f"{curve_display_labels.get('kappa', 'd')}_kappa_pathway_curve": _pathway_curve_report(
                curve_axes[1],
                curve_twins[1],
                curve_data.get("kappa"),
                "horizontal",
                co2_saved_mark="bar",
                co2_saved_baseline=0.0,
            ),
        },
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

    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    cmap = LinearSegmentedColormap.from_list(
        "gef", ["#006837", "#31A354", "#D95F0E", "#A50F15"]
    )
    gef_vals = plot_df[group_var].values
    norm = Normalize(vmin=gef_vals.min(), vmax=gef_vals.max())

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
    line_colors = ["#006837", "#A50F15"]

    for ax, mask, line_color, label, panel_letter in zip(
        axes, masks, line_colors, group_labels, panel_letters
    ):
        panel_df = plot_df.loc[mask]
        ax.scatter(
            panel_df[feature_var].values,
            panel_df[result_var].values,
            s=24,
            alpha=0.68,
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
                linewidth=2.5,
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
            -0.10,
            1.02,
            panel_letter,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="right",
        )
        ax.text(
            0.03,
            0.96,
            label,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="0.7",
                alpha=0.9,
            ),
        )
        ax.set_xlim(x_global_min, x_global_max)
        ax.set_ylim(y_global_min, y_global_max)
        ax.set_xlabel(VARIABLE_LABELS.get(feature_var, feature_var), fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")

    ax_low.set_ylabel(VARIABLE_LABELS.get(result_var, result_var), fontsize=12)
    ax_high.set_ylabel("")
    ax_high.tick_params(axis="y", left=False, labelleft=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(VARIABLE_LABELS.get(group_var, group_var), fontsize=10)

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
    kappa pathway curves from `plot_regions_map_pathway_scenarios`. Values use
    the same cache-backed scenario preparation as
    `plot_regions_map_pathway_scenarios`.

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
    figsize: tuple = (20, 12),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """3x2 panel grid for pathway curves and scenario maps.

    The left column shows pathway curves, while each of four scenarios occupies
    a (map, boxplot) pair in the remaining cells, with the boxplot directly
    under its map. The boxplot has one box per province (drone-share distribution
    across cities within that province). Provinces are ordered by the weighted
    all-ebike annual lifecycle baseline, consistent across panels.

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
        data_dir: Data root containing city OD-pair CSVs for sampled curve panels.
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
    anchor_rows = scenario_data["anchor_rows"]
    panels = scenario_data["panels"]
    panel_dfs = scenario_data["panel_dfs"]
    panel_named = scenario_data["panel_named"]
    drone_share_median = scenario_data["drone_share_median"]
    annual_saved_total_kg = scenario_data["annual_saved_total_kg"]
    all_values = scenario_data["all_values"]
    curve_data = scenario_data["curve_data"]
    has_annual = scenario_data["has_annual"]
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
    summary_share_limits, summary_saved_limits = _summary_bar_limits(
        drone_share_median,
        annual_saved_total_kg,
    )

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[5, 2.2, 5, 2.2],
        width_ratios=[1.05, 1, 1],
        hspace=0.32,
        wspace=0.26,
    )
    pos = {"a": (0, 1), "b": (0, 2), "c": (2, 1), "d": (2, 2)}
    panel_display_labels = {"a": "b", "b": "c", "c": "e", "d": "f"}
    curve_display_labels = {"gef": "a", "kappa": "d"}
    ax_pairs = []
    panel_axes = {}
    summary_bar_axes = {}
    curve_axes = [
        fig.add_subplot(gs[0:2, 0]),
        fig.add_subplot(gs[2:4, 0]),
    ]

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
        ax_map.text(-0.02, 1.02, panel_display_labels[label], transform=ax_map.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="right")

        saved_kt = annual_saved_total_kg.get(label, np.nan) / 1e6
        if np.isfinite(drone_share_median[label]) and np.isfinite(saved_kt):
            summary_bar_axes[label] = _add_map_summary_bars(
                ax_map,
                drone_share_median[label],
                saved_kt,
                summary_share_limits,
                summary_saved_limits,
            )

        rows = panel_named[label]
        share_col = _scenario_drone_share_column(rows) if not rows.empty else "drone_ratio_pct"
        data = [rows.loc[rows["province_en"] == p, share_col].values for p in province_order]
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
        panel_axes[label] = (ax_map, ax_box, ax_line)

    curve_saved_limits, curve_share_limits = _pathway_curve_metric_limits(curve_data)
    curve_twins = [
        _plot_pathway_curve_panel(
            curve_axes[0],
            curve_data.get("gef"),
            curve_display_labels["gef"],
            "Grid emission factor multiplier",
            orientation="horizontal",
            saved_limits=curve_saved_limits,
            share_limits=curve_share_limits,
            saved_as_bar=True,
        ),
        _plot_pathway_curve_panel(
            curve_axes[1],
            curve_data.get("kappa"),
            curve_display_labels["kappa"],
            r"E-bike bundling factor $\kappa$",
            orientation="horizontal",
            saved_limits=curve_saved_limits,
            share_limits=curve_share_limits,
            saved_as_bar=True,
        ),
    ]

    cbar_ax = fig.add_axes((0.925, 0.25, 0.012, 0.42))
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
    for ax, ax_twin in zip(curve_axes, curve_twins):
        if ax_twin is not None:
            ax_twin.set_position(ax.get_position())

    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map pathway scenarios saved to {save_path}")
        _write_regions_map_pathway_scenarios_report(
            save_path=save_path,
            figsize=figsize,
            named_csv_path=named_csv_path,
            stats_csv_path=stats_csv_path,
            panels=panels,
            panel_dfs=panel_dfs,
            panel_named=panel_named,
            panel_axes=panel_axes,
            summary_bar_axes=summary_bar_axes,
            curve_data=curve_data,
            curve_axes=curve_axes,
            curve_twins=curve_twins,
            province_order=province_order,
            annual_saved_total_kg=annual_saved_total_kg,
            drone_share_median=drone_share_median,
            shared_cmap=shared_cmap,
            bin_edges=bin_edges,
            reduction_edges=reduction_edges,
            cbar_ax=cbar_ax,
            panel_display_labels=panel_display_labels,
            curve_display_labels=curve_display_labels,
        )
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
            "color": "#4C8CBF",
            "offset": -0.18,
            "use_col": "total_ebike_use_phase_emission_kg",
            "production_col": "total_ebike_production_emission_kg",
            "eol_col": "total_ebike_end_of_life_emission_kg",
        },
        {
            "label": "Drone",
            "color": "#E05C2B",
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
        height_ratios=[4.0, 1.0],
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
            alpha=0.38,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.bar(
            positions,
            use_values,
            width=width,
            bottom=fixed_values,
            color=spec["color"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.6,
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
    ax.set_ylabel("Lifecycle emission\n(g CO2e/order)", fontsize=12)
    ax.set_xticks(bin_edges)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0, max_total * 1.22)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    vehicle_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=spec["color"],
            alpha=0.88,
            edgecolor="white",
            label=spec["label"],
        )
        for spec in vehicle_specs
    ]
    legend1 = ax.legend(
        handles=vehicle_handles,
        loc="upper left",
        fontsize=9,
        title="Vehicle",
        title_fontsize=9,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=legend_handles,
        loc="center left",
        fontsize=9,
        title="Component",
        title_fontsize=9,
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
    ax_dist.set_ylabel("Cities", fontsize=10)
    ax_dist.set_xlabel(gef_label, fontsize=12)
    ax_dist.set_xticks(bin_edges)
    ax_dist.set_xticklabels(tick_labels)
    ax_dist.tick_params(axis="x", which="both", bottom=True, labelbottom=True, pad=2)
    ax_dist.set_xlim(0.05, 0.85)
    ax_dist.spines["left"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(False)

    if plot_title:
        ax.set_title(
            "City Per-order Emission Components by Grid Emission Factor",
            fontsize=14,
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
    """Province bar chart ranked by total annual e-bike emission, with Oracle reduction overlay.

    Bars (left axis) show summed annual-sales-weighted total_ebike_lifecycle
    per province, sorted descending. Boxplot (right axis) shows the per-city
    distribution of Oracle reduction versus the kappa-adjusted e-bike baseline.

    Args:
        df: all_region_stats DataFrame with province, total_ebike_lifecycle,
            and oracle_min_lifecycle_kg.
        save_path: Path without extension. Saves PNG or PDF; shows interactively if None.
        figsize: Figure size tuple.
        for_latex: Save as PDF when True.
        plot_title: Add title when True.
        ax: External axes. When provided, skips figure creation and save.
    """
    needed = ["province", "total_ebike_lifecycle", "oracle_min_lifecycle_kg"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        print(f"Province total ebike lifecycle skipped — missing columns {missing}")
        return

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
    if sub.empty:
        print("Province total ebike lifecycle skipped — no rows with required columns")
        return

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

    ax1.bar(
        x,
        ebike_t,
        color=bar_color,
        alpha=0.85,
        zorder=2,
        label="Kappa-adjusted e-bike emission",
    )
    ax1.set_xlabel("Province", fontsize=12)
    ax1.set_ylabel(
        "Kappa-adjusted e-bike lifecycle emission (t CO₂eq / yr)",
        fontsize=12,
        color=bar_axis_color,
    )
    ax1.tick_params(axis="y", labelcolor=bar_axis_color)
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
    ax2.set_ylabel(
        "City Oracle reduction vs e-bike (%)",
        fontsize=12,
        color=box_color,
    )
    ax2.tick_params(axis="y", labelcolor=box_color)
    ax2.set_xlim(ax1.get_xlim())

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=bar_color, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor=box_color, alpha=0.35, edgecolor=box_color),
    ]
    ax1.legend(
        handles,
        [
            "Kappa-adjusted e-bike emission",
            "Oracle reduction vs e-bike (per city)",
        ],
        loc="upper right",
        fontsize=10,
    )

    if plot_title:
        title = "Province e-bike emission ranking with Oracle reduction distribution"
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
        "E-bike - Oracle (t CO₂eq / yr)",
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
    figsize,
    stats_csv_path,
    map_df,
    reduction_col,
    reduction_pct_col,
    bubble_df,
    ax_map,
    inset_ax,
    cbar_ax,
    cmap,
    norm,
    bin_edges,
    reduction_edges,
):
    """Write a JSON report for regions_map_oracle_reduction_rank."""
    renderer = ax_map.get_figure().canvas.get_renderer()
    tight_bbox = ax_map.get_figure().get_tightbbox(renderer)
    report_path = save_path.rsplit(".", 1)[0] + "_report.json"
    payload = {
        "figure_name": "regions_map_oracle_reduction_rank",
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
                "inset_city_rank": _city_oracle_rank_report(
                    map_df,
                    inset_ax,
                ),
                "colorbar": _colorbar_report(
                    cbar_ax,
                    bin_edges,
                    reduction_edges,
                    "Oracle lifecycle emission reduction (%)",
                    cmap,
                ),
                "colorbar_axis_saved_tight": saved_tight_axis_report(
                    ax_map.get_figure(),
                    cbar_ax,
                    renderer=renderer,
                    tight_bbox=tight_bbox,
                ),
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
):
    """China bubble map of Oracle reduction with top-city rank inset.

    Args:
        df_stats: all_region_stats DataFrame.
        stats_csv_path: Path to all_region_stats.csv (for map function).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
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
    cmap, norm, bin_edges, reduction_edges = _build_blue_reduction_cmap(v_max)
    bubble_df = _regions_oracle_bubble_df(
        map_df,
        reduction_col=reduction_col,
        reduction_pct_col=reduction_pct_col,
    )
    if bubble_df.empty:
        print("Regions map oracle reduction rank skipped — no valid bubble positions")
        return

    fig, ax_map = plt.subplots(figsize=figsize)
    gray_map_df = map_df.copy()
    gray_map_df["_bubble_gray"] = np.nan
    plot_regions_on_china_map(
        df=gray_map_df,
        color_by="_bubble_gray",
        ax=ax_map,
        plot_title=False,
        cmap=cmap,
        norm=norm,
        draw_colorbar=False,
        draw_legend=False,
    )
    ax_map.scatter(
        bubble_df["lng"],
        bubble_df["lat"],
        s=bubble_df["_bubble_size"],
        c=bubble_df[reduction_pct_col],
        cmap=cmap,
        norm=norm,
        alpha=0.82,
        edgecolor="#FFFFFF",
        linewidth=0.65,
        zorder=8,
    )
    size_legend = _draw_bubble_size_legend(ax_map, bubble_df[reduction_col])
    if size_legend is not None:
        ax_map.add_artist(size_legend)
    ax_map.set_xlabel("")
    ax_map.set_ylabel("")
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)

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
            "Regional Oracle Reduction Bubble Map and Top-City Rank",
            fontsize=13,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.subplots_adjust(left=0.11)

    fig.canvas.draw()
    map_pos = ax_map.get_position()
    cbar_ax = fig.add_axes((map_pos.x0 - 0.035, map_pos.y0, 0.014, map_pos.height))
    cbar, n_bands = _draw_discrete_reduction_colorbar(
        fig,
        cbar_ax,
        cmap,
        norm,
        bin_edges,
        reduction_edges,
        "Oracle lifecycle emission reduction (%)",
    )
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    _annotate_unstudied(fig, cbar, cbar_ax, n_bands)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.canvas.draw()
        _write_regions_map_oracle_reduction_rank_report(
            save_path=save_path,
            figsize=figsize,
            stats_csv_path=stats_csv_path,
            map_df=map_df,
            reduction_col=reduction_col,
            reduction_pct_col=reduction_pct_col,
            bubble_df=bubble_df,
            ax_map=ax_map,
            inset_ax=inset_ax,
            cbar_ax=cbar_ax,
            cmap=cmap,
            norm=norm,
            bin_edges=bin_edges,
            reduction_edges=reduction_edges,
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Regions map oracle reduction rank saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


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
            label=body_label,
        ),
        Patch(
            facecolor=legend_battery_color or battery_color,
            edgecolor=edge_color,
            linewidth=0.8,
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
                    "E-bike - Oracle (t CO2eq / yr)",
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
                    label="Oracle reduction vs e-bike (%)",
                ),
                "colorbar": _colorbar_report(
                    cbar_b_ax,
                    reduction_pct_bins,
                    reduction_pct_edges,
                    "Oracle reduction vs e-bike (%)",
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
        xlabel="Oracle reduction (%)",
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

    if plot_title:
        fig.suptitle(
            "Regional Oracle Reduction, Drone Share, and Lifecycle Mechanisms",
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
        "E-bike - Oracle (t CO₂eq / yr)",
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
        "Oracle reduction vs e-bike (%)",
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
        "Oracle reduction (%)"
        if reduction_metric == "percentage"
        else "E-bike - Oracle (t CO₂eq / yr)"
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
            f"Top {n} Cities by Oracle Reduction Percentage"
            if reduction_metric == "percentage"
            else f"Top {n} Cities by E-bike - Oracle Lifecycle Emission"
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
    ax.barh(y, oracle_t, height=bar_h, color=oracle_color, alpha=0.95, label="Oracle minimum", zorder=3)

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


def plot_city_population_detour_oracle_reduction_bubble(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10.5, 6.8),
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
    size_min, size_max = 26.0, 760.0
    if np.isclose(population_million.max(), population_million.min()):
        sizes = np.full(len(plot_df), (size_min + size_max) / 2)
    else:
        scaled = np.sqrt(population_million / population_million.max())
        sizes = size_min + scaled * (size_max - size_min)
    plot_df["_bubble_size"] = sizes

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 0.18, 0.24),
        height_ratios=(0.18, 1.0),
        wspace=0.08,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    legend_ax = fig.add_subplot(gs[1, 2])
    fig.add_subplot(gs[0, 1]).axis("off")
    fig.add_subplot(gs[0, 2]).axis("off")
    legend_ax.axis("off")

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

    ax.set_xlabel("Annual Oracle emission reduction (t CO2e/yr)", fontsize=12)
    ax.set_ylabel("Oracle lifecycle emission reduction (%)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(True, which="major", linestyle="--", alpha=0.28, zorder=0)
    ax.set_axisbelow(True)

    x_values = plot_df[reduction_t_col].to_numpy(dtype=float)
    y_values = plot_df[reduction_pct_col].to_numpy(dtype=float)
    if x_values.size > 1 and np.nanstd(x_values) > 0:
        x_pad = (x_values.max() - x_values.min()) * 0.05
        x_grid = np.linspace(x_values.min() - x_pad, x_values.max() + x_pad, 240)
        x_density = stats.gaussian_kde(x_values)(x_grid)
        ax_top.fill_between(
            x_grid,
            x_density,
            color="#6baed6",
            alpha=0.35,
            linewidth=0,
        )
        ax_top.plot(x_grid, x_density, color="#2171b5", linewidth=1.0)
    if y_values.size > 1 and np.nanstd(y_values) > 0:
        y_pad = (y_values.max() - y_values.min()) * 0.05
        y_grid = np.linspace(y_values.min() - y_pad, y_values.max() + y_pad, 240)
        y_density = stats.gaussian_kde(y_values)(y_grid)
        ax_right.fill_betweenx(
            y_grid,
            0,
            y_density,
            color="#6baed6",
            alpha=0.35,
            linewidth=0,
        )
        ax_right.plot(y_density, y_grid, color="#2171b5", linewidth=1.0)

    ax_top.set_ylabel("Density", fontsize=9)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", labelsize=8, length=2)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_right.set_xlabel("Density", fontsize=9)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.tick_params(axis="x", labelsize=8, length=2)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    legend_values = np.nanquantile(population_million, [0.25, 0.5, 0.9])
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
            plt.scatter(
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
        size_legend = legend_ax.legend(
            handles,
            labels,
            title="Population",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.98),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
            labelspacing=0.95,
        )
        legend_ax.add_artist(size_legend)

    region_handles = []
    plotted_region_keys = set(plot_df["_macro_region_key"])
    for spec in CHINA_MACRO_REGION_SPECS:
        if spec["key"] not in plotted_region_keys:
            continue
        region_handles.append(
            Patch(
                facecolor=spec["color"],
                edgecolor="none",
                label=spec["label"],
            )
        )
    if region_handles:
        legend_ax.legend(
            handles=region_handles,
            title="Macro-region",
            loc="lower left",
            bbox_to_anchor=(0.0, 0.02),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.45,
            labelspacing=0.65,
        )

    if name_col:
        label_idx = set(plot_df.nlargest(8, reduction_t_col).index)
        label_idx.add(plot_df["population_million"].idxmax())
        label_idx.add(plot_df[reduction_pct_col].idxmax())
        labeled = plot_df.loc[sorted(label_idx)].sort_values(
            reduction_t_col,
            ascending=False,
        )
        offsets = ((5, 4), (5, -8), (-5, 4), (-5, -8))
        for i, (_, row) in enumerate(labeled.iterrows()):
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(
                str(row[name_col]),
                xy=(row[reduction_t_col], row[reduction_pct_col]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top",
                fontsize=8,
                color="#303030",
                zorder=5,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.68,
                },
            )

    if plot_title:
        fig.suptitle(
            "Oracle Emission Reduction and Population",
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
        print(f"City population Oracle bubble saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


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
        bar_label = "Oracle drone share"
        bar_ylabel = "Oracle Drone Share (%)"
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
