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


PATHWAY_SCENARIOS_CACHE_VERSION = 2
PATHWAY_CURVE_N_POINTS = 21
PATHWAY_CURVE_MAX_ROWS_PER_CITY = 1500
PATHWAY_CURVE_RANDOM_STATE = 42
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

    from drone_emission.sensitivity.vectorized_lca import (
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

    from drone_emission.sensitivity.vectorized_lca import (
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
    from drone_emission.emission.cal_single_delivery import create_drone_params, create_ebike_params
    from drone_emission.utils.grid_emission_factor import fetch_factor

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
    n_points: int = PATHWAY_CURVE_N_POINTS,
    max_rows_per_city: int = PATHWAY_CURVE_MAX_ROWS_PER_CITY,
    random_state: int = PATHWAY_CURVE_RANDOM_STATE,
) -> dict:
    """Build sampled nationwide kappa and GEF pathway curves."""
    from drone_emission.emission.cal_single_delivery import create_drone_params, create_ebike_params
    from drone_emission.utils.grid_emission_factor import fetch_factor

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
        "curve_sampling": {
            "n_points": PATHWAY_CURVE_N_POINTS,
            "max_rows_per_city": PATHWAY_CURVE_MAX_ROWS_PER_CITY,
            "random_state": PATHWAY_CURVE_RANDOM_STATE,
        },
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
    from drone_emission.utils.config import get_cache_dir

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


def _write_regions_map_pathway_scenario_maps_report(
    save_path: str,
    output_paths: dict,
    figsize: tuple,
    named_csv_path: str,
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
        "source_named_csv_path": named_csv_path,
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


def plot_regions_map_pathway_scenario_maps(
    named_csv_path: str,
    stats_csv_path: str,
    data_dir: str = None,
    exp_config: dict = None,
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
        named_csv_path: Path to all_sensitivity_named.csv.
        stats_csv_path: Path to all_region_stats.csv (provides region_name_cn lookup).
        data_dir: Data root containing city OD-pair CSVs for oracle-min scenarios.
        exp_config: Experiment config for vehicle parameters and GEF path.
        save_path: Base output path. Panel suffixes are added before extension.
            Shows plots if None.
        figsize: Size tuple for each standalone panel.
        for_latex: If True, save as PDF.
        plot_title: If True, show a title in each panel.
    """
    scenario_data = _prepare_pathway_scenario_data(
        named_csv_path=named_csv_path,
        stats_csv_path=stats_csv_path,
        data_dir=data_dir,
        exp_config=exp_config,
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
            named_csv_path=named_csv_path,
            stats_csv_path=stats_csv_path,
            curve_data=curve_data,
            curve_axes=curve_axes,
            curve_twins=curve_twins,
        )

    for fig in curve_figures:
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
    from drone_emission.visualization.regions import plot_regions_on_china_map

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
            "label": "Unstudied",
            "color": "#cccccc",
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
