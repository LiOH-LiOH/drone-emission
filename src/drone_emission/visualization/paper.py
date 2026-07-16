"""Entry point for rendering all paper figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.paper --png              # PNG (fast iteration)
    ../../.venv/bin/python -m visualization.paper                    # PDF (for LaTeX)
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap,distance_comparison

Available figures:
    tanner_impedance_fit, tanner_impedance_fit_panel_a, alpha_calibration_5km,
    detour_lifecycle_distance, lifecycle_distance_detour_boxplot,
    lifecycle_distance_route_factor, production_phase_per_km, end_of_life_phase_per_km,
    production_detour_eol_panels, detour_ratio_concept, emission_model_flowchart,
    detour_and_lifecycle, distance_comparison, distance_comparison_single_panel,
    distance_comparison_high_3x3, distance_comparison_low_3x3,
    shanghai_heatmap, shanghai_merchant_monthly_sales,
    shanghai_building_height_merchant,
    shanghai_building_footprint_heatmap,
    beijing_road_network_subfigure, beijing_road_network_candidate_grid,
    beijing_road_network_subfigure_option3,
    shanghai_road_air_distance, shanghai_road_detour,
    shanghai_air_detour, shanghai_air_road_joint, shanghai_emission_reduction,
    shanghai_emission_by_distance, shanghai_reduction_detour_by_distance,
    beijing_median_trip_emission_process,
    shanghai_city_context, chongqing_city_context, chengdu_city_context,
    guangzhou_city_context, tianjin_city_context, dali_city_context,
    kunming_city_context, dali_chengdu_tianjin_guangzhou_city_context,
    top6_reduction_detour_grid, top6_reduction_detour_grid_low_gef,
    top6_reduction_detour_grid_random,
    province_gef_china_map,
    province_orders_emission, province_orders_both_emissions,
    province_total_ebike_lifecycle,
    city_ebike_rank_top10, city_ebike_rank_bottom10,
    city_oracle_reduction_rank_top10,
    city_reduction_rank_top10, city_reduction_rank_all_macro_region,
    city_population_detour_oracle_reduction_bubble,
    city_gef_distribution,
    city_gef_scatter_components,
    city_gef_context_detour_summary,
    break_even_contribution_decomposition,
    break_even_contribution_decomposition_marching_cubes,
    break_even_contribution_decomposition_marching_cubes_de,
    contribution_decomposition,
    contribution_decomposition_city_b,
    break_even_phase_diagram,
    regions_map,
    regions_map_oracle_reduction_rank,
    regions_map_oracle_reduction_rank_map_only,
    regions_map_optimal_drone_share,
    regions_map_city_rank_and_drone_share,
    scatter_detour_ratio_vs_reduction_by_gef,
    regions_map_and_gef_scatter,
    regions_map_city_province_ranks,
    regions_map_city_rank_inset,
    figure1_framework, figure2_framework, figure3_framework, figure4_framework,
    regions_map_kappa_panels,
    drone_tech_sensitivity,
    oat_tornado_story, oat_tornado_full
"""

import argparse
from contextlib import contextmanager
import json
import os
import pickle
import sys
import time

import pandas as pd


TOP6_REDUCTION_DETOUR_FIGURES = {
    "top6_reduction_detour_grid",
    "top6_reduction_detour_grid_low_gef",
    "top6_reduction_detour_grid_random",
}
CITY_CONTEXT_FIGURES = {
    "shanghai_city_context": {
        "region_key": "shanghai",
        "display_name": "Shanghai",
        "power_region_key": "shanghai",
    },
    "chongqing_city_context": {
        "region_key": "chongqing",
        "display_name": "Chongqing",
        "power_region_key": "chongqing",
    },
    "chengdu_city_context": {
        "region_key": "chengdu",
        "display_name": "Chengdu",
        "power_region_key": "sichuan",
    },
    "beijing_city_context": {
        "region_key": "beijing",
        "display_name": "Beijing",
        "power_region_key": "beijing",
    },
    "guangzhou_city_context": {
        "region_key": "guangzhou",
        "display_name": "Guangzhou",
        "power_region_key": "guangdong",
    },
    "tianjin_city_context": {
        "region_key": "tianjin",
        "display_name": "Tianjin",
        "power_region_key": "tianjin",
    },
    "dali_city_context": {
        "region_key": "dalibaizuzizhizhou",
        "display_name": "Dali",
        "power_region_key": "yunnan",
    },
    "kunming_city_context": {
        "region_key": "kunming",
        "display_name": "Kunming",
        "power_region_key": "yunnan",
    },
}
CITY_CONTEXT_GROUP_FIGURES = {
    "dali_chengdu_tianjin_guangzhou_city_context": (
        CITY_CONTEXT_FIGURES["dali_city_context"],
        CITY_CONTEXT_FIGURES["chengdu_city_context"],
        CITY_CONTEXT_FIGURES["tianjin_city_context"],
        CITY_CONTEXT_FIGURES["guangzhou_city_context"],
    ),
    "chengdu_beijing_guangzhou_city_context": (
        CITY_CONTEXT_FIGURES["chengdu_city_context"],
        CITY_CONTEXT_FIGURES["beijing_city_context"],
        CITY_CONTEXT_FIGURES["guangzhou_city_context"],
    ),
}

CITY_GEF_CONTEXT_DETOUR_CONTEXTS = (
    CITY_CONTEXT_FIGURES["chengdu_city_context"],
    CITY_CONTEXT_FIGURES["beijing_city_context"],
    CITY_CONTEXT_FIGURES["shanghai_city_context"],
    CITY_CONTEXT_FIGURES["guangzhou_city_context"],
)
CITY_CONTEXT_MARKERS = {
    "chengdu": "o",
    "beijing": "s",
    "shanghai": "D",
    "guangzhou": "^",
}

REDUCTION_DETOUR_CACHE_VERSION = 4
BREAK_EVEN_PHASE_CACHE_VERSION = 9
CONTRIBUTION_DECOMPOSITION_FULL_CACHE_VERSION = 5
REDUCTION_DETOUR_SELECTION_METRIC = "total_ebike_lifecycle"
RETIRED_THRESHOLD_METRIC_COLUMNS = {
    "max_lifecycle_reduction_pct",
    "optimal_threshold_km",
    "optimal_drone_ratio_pct",
}
POWER_GENERATION_2022_100M_KWH = {
    "shanghai": {
        "total": 955.0,
        "Hydro": 0.0,
        "Thermal": 912.60,
        "Wind": 24.56,
        "Solar": 17.82,
    },
    "chongqing": {
        "total": 998.0,
        "Hydro": 203.11,
        "Thermal": 755.15,
        "Wind": 34.02,
        "Solar": 5.56,
    },
    "beijing": {
        "total": 467.0,
        "Hydro": 9.43,
        "Thermal": 443.57,
        "Wind": 4.95,
        "Solar": 9.14,
    },
    "sichuan": {
        "total": 4846.0,
        "Hydro": 3886.56,
        "Thermal": 794.61,
        "Wind": 134.71,
        "Solar": 30.28,
    },
    "yunnan": {
        "total": 4017.0,
        "Hydro": 3282.91,
        "Thermal": 464.53,
        "Wind": 213.81,
        "Solar": 55.40,
    },
    "guangdong": {
        "total": 6366.0,
        "Hydro": 344.41,
        "Thermal": 4469.17,
        "Wind": 268.66,
        "Solar": 134.89,
    },
    "tianjin": {
        "total": 765.0,
        "Hydro": 0.14,
        "Thermal": 715.72,
        "Wind": 25.51,
        "Solar": 23.52,
    },
}
POWER_MIX_COLORS = {
    "Hydro": "#4c78a8",
    "Thermal": "#8c564b",
    "Wind": "#59a14f",
    "Solar": "#f2cf5b",
    "Other": "#9e9e9e",
}


@contextmanager
def _profile_block(enabled, label):
    """Print elapsed time for an optional paper-rendering block."""
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[profile] {label}: {elapsed:.2f}s")
FIGURE1_FRAMEWORK_DEPENDENCIES = (
    "regions_map_oracle_reduction_rank",
    "shanghai_heatmap",
    "distance_comparison",
    "shanghai_building_height_merchant",
)
FIGURE1_FRAMEWORK_REPORT_DEPENDENCIES = (
    "regions_map_oracle_reduction_rank",
    "shanghai_heatmap",
    "shanghai_building_height_merchant",
)
FIGURE1_FRAMEWORK_SOURCE_MODULES = {
    "regions_map_oracle_reduction_rank": ("analysis.py", "regions.py", "_utils.py"),
    "shanghai_heatmap": ("sales.py", "_utils.py"),
    "distance_comparison": ("roadnet.py",),
    "shanghai_building_height_merchant": ("buildings.py", "_utils.py"),
}
FIGURE2_FRAMEWORK_DEPENDENCIES = (
    "tanner_impedance_fit_panel_a",
    "city_oracle_reduction_rank_top10",
    "regions_map_oracle_reduction_rank_map_only",
)
FIGURE2_FRAMEWORK_SOURCE_MODULES = {
    "tanner_impedance_fit_panel_a": ("calibration.py", "_utils.py"),
    "city_oracle_reduction_rank_top10": ("analysis.py", "_utils.py"),
    "regions_map_oracle_reduction_rank_map_only": ("analysis.py", "regions.py", "_utils.py"),
}
FIGURE3_FRAMEWORK_DEPENDENCIES = (
    "city_population_detour_oracle_reduction_bubble",
    "city_reduction_rank_all_macro_region",
)
FIGURE3_FRAMEWORK_SOURCE_MODULES = {
    "city_population_detour_oracle_reduction_bubble": ("analysis.py", "_utils.py"),
    "city_reduction_rank_all_macro_region": ("analysis.py", "macro_regions.py", "_utils.py"),
}
FIGURE4_FRAMEWORK_DEPENDENCIES = (
    "flow_frame.drawio",
    "tanner_impedance_fit",
    "alpha_calibration_5km",
)
FIGURE4_FRAMEWORK_SOURCE_MODULES = {
    "tanner_impedance_fit": ("calibration.py", "_utils.py"),
    "alpha_calibration_5km": (
        "calibration.py",
        "_utils.py",
        "../calibration/alpha_calibration_5km.csv",
    ),
}


def _get_ebike_kappa(exp_config):
    """Return the configured baseline e-bike distance factor."""
    emission_params = (exp_config or {}).get("emission_params", {})
    ebike_params = emission_params.get("ebike", {})
    return float(ebike_params.get("kappa", 1.0))


def _file_cache_signature(path):
    """Return file metadata used to invalidate a figure-data cache."""
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _optional_file_cache_signature(path):
    """Return cache signature for an optional file, or None when absent."""
    if not path or not os.path.exists(path):
        return None
    return _file_cache_signature(path)


def _attach_enrichment_columns(stats_df, enriched_df):
    """Return latest stats with enrichment-only columns attached."""
    if "region_key" not in stats_df.columns or "region_key" not in enriched_df.columns:
        print(
            "GEF figures: region_key missing from stats or enriched CSV; "
            "using enriched CSV as-is"
        )
        return enriched_df

    enrichment_cols = [
        col
        for col in enriched_df.columns
        if col != "region_key"
        and col not in stats_df.columns
        and col not in RETIRED_THRESHOLD_METRIC_COLUMNS
    ]
    dropped = sorted(
        col
        for col in RETIRED_THRESHOLD_METRIC_COLUMNS
        if col in enriched_df.columns and col not in stats_df.columns
    )
    if dropped:
        print(
            "GEF figures: ignored stale threshold metrics from enriched CSV: "
            f"{', '.join(dropped)}"
        )

    merged = stats_df.merge(
        enriched_df[["region_key", *enrichment_cols]],
        on="region_key",
        how="left",
    )
    missing_enrichment = (
        merged[enrichment_cols].isna().all(axis=1).sum()
        if enrichment_cols
        else 0
    )
    if missing_enrichment:
        print(f"GEF figures: {missing_enrichment} rows missing enrichment columns")
    return merged


def _prepare_reduction_detour_city_stats(city_df):
    """Return city stats rows eligible for reduction-detour grids."""
    required = {
        "region_key",
        "region_name_en",
        "province",
        "total_odpair_annual_sales",
        "total_ebike_lifecycle",
        "total_drone_lifecycle",
        "gef",
    }
    missing = sorted(required - set(city_df.columns))
    if missing:
        print(
            "top6_reduction_detour_grid skipped — all_region_stats.csv missing "
            f"columns {missing}"
        )
        return city_df.iloc[0:0].copy()

    city_df = city_df.copy()
    numeric_cols = [
        "total_odpair_annual_sales",
        "total_ebike_lifecycle",
        "total_drone_lifecycle",
        "gef",
    ]
    for col in numeric_cols:
        city_df[col] = pd.to_numeric(city_df[col], errors="coerce")

    if "ebike_kappa" in city_df.columns:
        before = len(city_df)
        city_df["ebike_kappa"] = pd.to_numeric(city_df["ebike_kappa"], errors="coerce")
        city_df = city_df[city_df["ebike_kappa"].notna()]
        skipped = before - len(city_df)
        if skipped:
            print(
                "top6_reduction_detour_grid: skipped "
                f"{skipped} city rows without weighted kappa stats"
            )
    else:
        print(
            "top6_reduction_detour_grid: all_region_stats.csv has no ebike_kappa "
            "column; run tools.recompute_kappa_city_metrics for weighted stats"
        )

    if "drone_minus_ebike_lifecycle_kg" in city_df.columns:
        city_df["drone_minus_ebike_lifecycle_kg"] = pd.to_numeric(
            city_df["drone_minus_ebike_lifecycle_kg"], errors="coerce"
        )
    else:
        city_df["drone_minus_ebike_lifecycle_kg"] = (
            city_df["total_drone_lifecycle"] - city_df["total_ebike_lifecycle"]
        )

    return city_df.dropna(subset=list(required))


def _prepare_contribution_decomposition_city_stats(city_df, gef_df, exp_config):
    """Return Part-1-scope city stats for contribution decomposition."""
    required = {
        "region_key",
        "region_name_en",
        "province",
        "total_ebike_lifecycle",
    }
    missing = sorted(required - set(city_df.columns))
    if missing:
        print(
            "break_even_contribution_decomposition skipped — "
            f"all_region_stats.csv missing columns {missing}"
        )
        return city_df.iloc[0:0].copy()

    prepared = city_df.copy()
    if "gef" in prepared.columns:
        prepared = prepared.drop(columns=["gef"])
    if gef_df is not None and not gef_df.empty:
        prepared = prepared.merge(gef_df, on="province", how="left")
    else:
        prepared["gef"] = float((exp_config or {}).get("emission_params", {}).get("e_use", 0.5600))

    prepared["total_ebike_lifecycle"] = pd.to_numeric(
        prepared["total_ebike_lifecycle"],
        errors="coerce",
    )
    prepared["gef"] = pd.to_numeric(prepared["gef"], errors="coerce")
    fallback_gef = 0.5600
    missing_gef = int(prepared["gef"].isna().sum())
    if missing_gef:
        print(
            "break_even_contribution_decomposition: filling "
            f"{missing_gef} city rows with fallback GEF={fallback_gef:.4f}"
        )
        prepared["gef"] = prepared["gef"].fillna(fallback_gef)

    prepared = prepared[
        prepared["total_ebike_lifecycle"].notna()
        & (prepared["total_ebike_lifecycle"] > 0)
        & prepared["gef"].notna()
    ].copy()
    return prepared


def _build_reduction_detour_binned_by_city(
    data_dir,
    selected,
    figure_name,
    exp_config=None,
    absolute_reduction_metric="total_t",
    use_binned_cache=False,
    bin_width_km=0.1,
):
    """Return binned reduction-detour data for selected city rows."""
    from dataclasses import asdict, replace

    import numpy as np

    from emission.cal_single_delivery import create_ebike_params, get_emission_params
    from sensitivity.vectorized_lca import EbikeScenario, compute_ebike_per_order_g
    from utils.config import get_cache_dir
    from visualization.analysis import _bin_reduction_and_detour

    if exp_config is None:
        print(f"{figure_name}: exp_config missing; cannot apply configured kappa")
        return None

    ebike_kappa = _get_ebike_kappa(exp_config)
    ebike_params = create_ebike_params(exp_config)
    emission_cfg = get_emission_params(exp_config)
    scenario = EbikeScenario(
        kappa=ebike_kappa,
        rider_weight=emission_cfg["rider_weight"],
        payload_kg=emission_cfg["payload_kg"],
    )

    selected_keys = selected["region_key"].tolist()
    gef_map = dict(zip(selected["region_key"], selected["gef"]))
    label_map = dict(zip(selected["region_key"], selected["region_name_en"]))
    labels = {key: label_map.get(key, key) for key in selected_keys}
    od_paths = {
        key: os.path.join(data_dir, "city", key, f"{key}_od_pairs.csv")
        for key in selected_keys
    }

    cache_path = None
    cache_meta = None
    available_keys = []
    source_files = {}
    for key in selected_keys:
        od_path = od_paths[key]
        if not os.path.exists(od_path):
            print(f"{figure_name}: missing {od_path} — skipping {key}")
            continue
        gef = gef_map.get(key)
        if gef is None or pd.isna(gef):
            print(f"{figure_name}: {key} missing province GEF — skipping")
            continue
        available_keys.append(key)
        source_files[key] = _file_cache_signature(od_path)

    if not available_keys:
        print(f"{figure_name}: no city OD data available")
        return None

    if use_binned_cache:
        cache_path = os.path.join(
            get_cache_dir("visualization", "paper"),
            f"{figure_name}_binned.pkl",
        )
        cache_meta = {
            "version": REDUCTION_DETOUR_CACHE_VERSION,
            "figure_name": figure_name,
            "city_keys": available_keys,
            "source_files": source_files,
            "gef": {key: float(gef_map[key]) for key in available_keys},
            "kappa": ebike_kappa,
            "ebike_params": asdict(ebike_params),
            "scenario": asdict(scenario),
            "bin_width_km": bin_width_km,
            "absolute_reduction_metric": absolute_reduction_metric,
        }
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("meta") == cache_meta:
                    print(f"{figure_name}: loading binned data cache {cache_path}")
                    return cache["binned_by_city"], labels
                print(f"{figure_name}: binned data cache is stale; rebuilding")
            except Exception as exc:
                print(
                    f"{figure_name}: failed to load binned data cache ({exc}); "
                    "rebuilding"
                )

    binned_by_city = {}
    required = {
        "road_distance_m",
        "distance_km",
        "drone_lifecycle_g",
        "ebike_lifecycle_g",
        "annual_sales",
    }
    for key in available_keys:
        od_path = od_paths[key]
        od_df = pd.read_csv(od_path)
        if not required.issubset(od_df.columns):
            print(f"{figure_name}: {key} missing columns — skipping")
            continue
        road_m = pd.to_numeric(
            od_df["road_distance_m"], errors="coerce"
        ).to_numpy()
        air_km = pd.to_numeric(od_df["distance_km"], errors="coerce").to_numpy()
        annual_sales = pd.to_numeric(
            od_df["annual_sales"], errors="coerce"
        ).to_numpy()
        valid_detour = (
            np.isfinite(road_m)
            & np.isfinite(air_km)
            & np.isfinite(annual_sales)
            & (road_m >= 0)
            & (air_km > 0)
            & (annual_sales > 0)
        )
        weighted_road_m = float(
            np.sum(road_m[valid_detour] * annual_sales[valid_detour])
        )
        weighted_air_m = float(
            np.sum(air_km[valid_detour] * 1000.0 * annual_sales[valid_detour])
        )
        city_detour_ratio = (
            weighted_road_m / weighted_air_m
            if weighted_air_m > 0
            else np.nan
        )
        gef = gef_map[key]
        ebike_pp = replace(ebike_params, e_use=float(gef))
        road_km = road_m / 1000.0
        od_df = od_df.copy()
        od_df["ebike_lifecycle_g"] = compute_ebike_per_order_g(road_km, ebike_pp, scenario)
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
            "city_detour_ratio": city_detour_ratio,
            "valid_path_count": int(valid_detour.sum()),
            "invalid_path_count": int((~valid_detour).sum()),
        }

    if use_binned_cache and binned_by_city:
        if cache_path and cache_meta:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {"meta": cache_meta, "binned_by_city": binned_by_city},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"{figure_name}: wrote binned data cache {cache_path}")
    if binned_by_city:
        print(
            f"{figure_name}: rebuilt binned data at kappa={ebike_kappa} "
            f"for {len(binned_by_city)} cities"
        )
        return binned_by_city, labels

    print(f"{figure_name}: no city OD data available")
    return None


def _render_reduction_detour_grid_variant(
    data_dir,
    out,
    for_latex,
    plot_title,
    selected,
    figure_name,
    exp_config=None,
    absolute_reduction_metric="total_t",
    show_detour_boxplot=True,
    mark_reduction_break_even=False,
    figsize=(12, 14),
    n_cols=2,
    grid_hspace=0.45,
    grid_wspace=0.22,
    use_binned_cache=False,
    bin_width_km=0.1,
    min_air_km=0.0,
):
    """Load selected city OD data and render a reduction-detour grid."""
    from visualization.analysis import plot_reduction_detour_grid_binned

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric=absolute_reduction_metric,
        use_binned_cache=use_binned_cache,
        bin_width_km=bin_width_km,
    )
    if built is None:
        print(f"{figure_name}: no city OD data available")
        return

    binned_by_city, labels = built
    plot_reduction_detour_grid_binned(
        binned_by_city=binned_by_city,
        city_labels=labels,
        save_path=os.path.join(out, f"{figure_name}.png"),
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
        bin_width_km=bin_width_km,
        n_cols=n_cols,
        min_air_km=min_air_km,
        absolute_reduction_metric=absolute_reduction_metric,
        show_detour_boxplot=show_detour_boxplot,
        mark_reduction_break_even=mark_reduction_break_even,
        grid_hspace=grid_hspace,
        grid_wspace=grid_wspace,
    )


def _select_top6_reduction_detour_variants(city_df, gef_df):
    """Return deterministic city selections for reduction-detour grid variants."""
    selection_metric = REDUCTION_DETOUR_SELECTION_METRIC
    if city_df.empty:
        return {
            "top6_reduction_detour_grid": city_df,
            "top6_reduction_detour_grid_low_gef": city_df,
            "top6_reduction_detour_grid_random": city_df,
        }
    if selection_metric not in city_df.columns:
        print(
            "top6_reduction_detour_grid skipped — all_region_stats.csv missing "
            f"{selection_metric}"
        )
        empty = city_df.iloc[0:0].copy()
        return {
            "top6_reduction_detour_grid": empty,
            "top6_reduction_detour_grid_low_gef": empty,
            "top6_reduction_detour_grid_random": empty,
        }

    city_df = city_df.copy()
    city_df["gef_rank"] = city_df["gef"].rank(method="dense")

    big = city_df.nlargest(16, selection_metric)
    top6_big_keys = set(city_df.nlargest(6, selection_metric)["region_key"])

    weighted_lifecycle_median = city_df[selection_metric].median()
    low_gef = (
        city_df[
            (city_df[selection_metric] >= weighted_lifecycle_median)
            & ~city_df["region_key"].isin(top6_big_keys)
        ]
        .sort_values(["gef", selection_metric], ascending=[True, False])
        .head(6)
    )

    big_cutoff = city_df[selection_metric].quantile(0.75)
    small_floor = city_df[selection_metric].quantile(0.25)
    low_gef_provinces = gef_df.nsmallest(8, "gef")["province"].tolist()
    non_low_gef = ~city_df["province"].isin(low_gef_provinces)
    mid_candidates = city_df[
        non_low_gef
        & (city_df[selection_metric] < big_cutoff)
        & (city_df[selection_metric] >= small_floor)
    ]
    random_mid = mid_candidates.sample(
        n=min(6, len(mid_candidates)),
        random_state=20260528,
    ).sort_values("region_key")

    return {
        "top6_reduction_detour_grid": big,
        "top6_reduction_detour_grid_low_gef": low_gef,
        "top6_reduction_detour_grid_random": random_mid,
    }


def _render_shanghai_heatmap(
    data_dir,
    out,
    for_latex,
    plot_title,
    compact=False,
):
    """Load Shanghai sales data and render the sales heatmap figure."""
    region_key = "shanghai"
    region_level = "city"
    sales_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_sales.csv")

    if not os.path.exists(sales_path):
        print(f"Shanghai heatmap skipped — missing {sales_path}")
        return

    from visualization.sales import plot_sales_heatmap

    if compact:
        lng_range = (121.25, 121.65)
        lat_range = (31.10, 31.36)
        figsize = (1.25, 1.05)
        zoom = 12
    else:
        lng_range = (121.0, 121.9)
        lat_range = (30.9, 31.55)
        figsize = None
        zoom = 14

    df = pd.read_csv(sales_path)
    plot_sales_heatmap(
        df,
        "Shanghai",
        "上海市",
        boundaries_dict=None,
        save_path=os.path.join(out, "shanghai_heatmap.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        lng_range=lng_range,
        lat_range=lat_range,
        zoom=zoom,
        compact=compact,
        figsize=figsize,
    )


def _find_first_existing_path(paths):
    """Return the first existing path from a list of candidates."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _render_shanghai_merchant_monthly_sales(
    data_dir,
    out,
    for_latex,
    plot_title,
):
    """Render merchant monthly-sales records over local building footprints."""
    region_key = "shanghai"
    region_dir = os.path.join(data_dir, "city", region_key)
    sales_path = os.path.join(region_dir, f"{region_key}_sales.csv")
    building_path = _find_first_existing_path(
        [
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.geojson"),
            os.path.join(region_dir, f"{region_key}_buildings.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings.geojson"),
        ]
    )
    if not os.path.exists(sales_path):
        print(f"Shanghai merchant monthly sales skipped - missing {sales_path}")
        return
    if building_path is None:
        print(
            "Shanghai merchant monthly sales skipped - missing building "
            f"geometry in {region_dir}"
        )
        return

    import geopandas as gpd
    from visualization.sales import plot_merchant_monthly_sales_map

    stores_df = pd.read_csv(sales_path)
    buildings_gdf = gpd.read_file(building_path)
    plot_merchant_monthly_sales_map(
        stores_df,
        buildings_gdf,
        "Shanghai",
        save_path=os.path.join(out, "shanghai_merchant_monthly_sales.png"),
        figsize=(1.25, 1.17),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _get_cached_sparse_graph(graphml_path, graph_cache=None):
    """Load a SparseGraph once per render_all call."""
    if not os.path.exists(graphml_path):
        return None
    graph_cache = graph_cache if graph_cache is not None else {}
    if graphml_path not in graph_cache:
        from roadnet.sparse_graph import SparseGraph

        graph_cache[graphml_path] = SparseGraph.from_graphml(graphml_path)
    else:
        print(f"Reusing cached SparseGraph: {graphml_path}")
    return graph_cache[graphml_path]


def _get_cached_tile_router(graphml_path, graph, router_cache=None):
    """Create a TileRouter once per render_all call."""
    if graph is None:
        return None
    router_cache = router_cache if router_cache is not None else {}
    if graphml_path not in router_cache:
        from roadnet.tile_router import TileRouter

        router_cache[graphml_path] = TileRouter(graph)
    else:
        print(f"Reusing cached TileRouter: {graphml_path}")
    return router_cache[graphml_path]


def _render_shanghai_building_height_merchant(
    data_dir,
    out,
    for_latex,
    plot_title,
    graph_cache=None,
):
    """Load Shanghai buildings and sales to render the merchant-context map."""
    region_key = "shanghai"
    region_level = "city"
    region_dir = os.path.join(data_dir, region_level, region_key)
    sales_path = os.path.join(region_dir, f"{region_key}_sales.csv")
    od_path = os.path.join(region_dir, f"{region_key}_od_pairs.csv")
    roadnet_path = os.path.join(region_dir, f"{region_key}_roadnet.graphml")
    building_path = _find_first_existing_path(
        [
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.geojson"),
            os.path.join(region_dir, f"{region_key}_buildings.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings.geojson"),
        ]
    )

    if not os.path.exists(sales_path):
        print(
            "Shanghai building height merchant map skipped - missing "
            f"{sales_path}"
        )
        return
    if building_path is None:
        print(
            "Shanghai building height merchant map skipped - missing Shanghai "
            f"building geometry in {region_dir}"
        )
        return

    import geopandas as gpd
    from visualization.buildings import plot_building_height_merchant_map

    stores_df = pd.read_csv(sales_path)
    od_df = None
    if os.path.exists(od_path):
        od_df = pd.read_csv(od_path)
    else:
        print(
            "Shanghai building height merchant map OD overlay skipped - "
            f"missing {od_path}"
        )
    road_graph = None
    if os.path.exists(roadnet_path):
        road_graph = _get_cached_sparse_graph(roadnet_path, graph_cache=graph_cache)
    else:
        print(
            "Shanghai building height merchant map roadnet background "
            f"skipped - missing {roadnet_path}"
        )
    buildings_gdf = gpd.read_file(building_path)
    plot_building_height_merchant_map(
        buildings_gdf,
        stores_df,
        "Shanghai",
        "上海市",
        od_df=od_df,
        road_graph=road_graph,
        save_path=os.path.join(out, "shanghai_building_height_merchant.png"),
        figsize=(2.56, 2.56),
        for_latex=for_latex,
        plot_title=plot_title,
        lat_span=0.007,
        compact=True,
    )


def _render_shanghai_building_footprint_heatmap(
    data_dir,
    out,
    for_latex,
    plot_title,
):
    """Load Shanghai buildings and sales to render the footprint heatmap."""
    region_key = "shanghai"
    region_level = "city"
    region_dir = os.path.join(data_dir, region_level, region_key)
    sales_path = os.path.join(region_dir, f"{region_key}_sales.csv")
    building_path = _find_first_existing_path(
        [
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings_preprocessed.geojson"),
            os.path.join(region_dir, f"{region_key}_buildings.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings.geojson"),
        ]
    )

    if not os.path.exists(sales_path):
        print(
            "Shanghai building footprint heatmap skipped - missing "
            f"{sales_path}"
        )
        return
    if building_path is None:
        print(
            "Shanghai building footprint heatmap skipped - missing Shanghai "
            f"building geometry in {region_dir}"
        )
        return

    import geopandas as gpd
    from visualization.buildings import plot_building_footprint_heatmap

    stores_df = pd.read_csv(sales_path)
    buildings_gdf = gpd.read_file(building_path)
    plot_building_footprint_heatmap(
        buildings_gdf,
        stores_df,
        "Shanghai",
        "上海市",
        save_path=os.path.join(out, "shanghai_building_footprint_heatmap.png"),
        figsize=(1.25, 1.11),
        for_latex=for_latex,
        plot_title=plot_title,
        lat_span=0.008,
        compact=True,
    )


BEIJING_ROAD_NETWORK_CENTER = (116.337, 39.914)
BEIJING_ROAD_NETWORK_FIGSIZE = (1.25, 1.11)
BEIJING_ROAD_NETWORK_LAT_SPAN = 0.018
BEIJING_ROAD_NETWORK_CANDIDATES = (
    {
        "label": "A",
        "name": "Dongsi north hutongs",
        "lng": 116.423,
        "lat": 39.942,
    },
    {
        "label": "B",
        "name": "Xisi north hutongs",
        "lng": 116.380,
        "lat": 39.942,
    },
    {"label": "C", "name": "Hepingli", "lng": 116.419, "lat": 39.960},
    {
        "label": "D",
        "name": "Yuetan-Sanlihe",
        "lng": 116.337,
        "lat": 39.914,
    },
    {"label": "E", "name": "Zhanlanlu", "lng": 116.349, "lat": 39.934},
    {
        "label": "F",
        "name": "Xuanwu-Baizhifang",
        "lng": 116.373,
        "lat": 39.897,
    },
    {
        "label": "G",
        "name": "Fangzhuang interior",
        "lng": 116.431,
        "lat": 39.864,
    },
    {"label": "H", "name": "Jinsong east", "lng": 116.480, "lat": 39.884},
    {
        "label": "I",
        "name": "Jiuxianqiao east",
        "lng": 116.505,
        "lat": 39.973,
    },
)
BEIJING_ROAD_NETWORK_RESERVES = (
    {"name": "Wanshoulu east", "lng": 116.304, "lat": 39.914},
    {"name": "Shuangyushu", "lng": 116.329, "lat": 39.968},
    {"name": "Jiugong", "lng": 116.445, "lat": 39.806},
)


def _render_beijing_road_network_subfigure(
    data_dir,
    out,
    for_latex,
    plot_title,
    graph_cache=None,
):
    """Render the fixed Beijing road network with building context."""
    region_key = "beijing"
    region_level = "city"
    region_dir = os.path.join(data_dir, region_level, region_key)
    roadnet_path = os.path.join(region_dir, f"{region_key}_roadnet.graphml")
    building_path = _find_first_existing_path(
        [
            os.path.join(
                region_dir, f"{region_key}_buildings_preprocessed.gpkg"
            ),
            os.path.join(
                region_dir, f"{region_key}_buildings_preprocessed.geojson"
            ),
            os.path.join(region_dir, f"{region_key}_buildings.gpkg"),
            os.path.join(region_dir, f"{region_key}_buildings.geojson"),
        ]
    )

    if not os.path.exists(roadnet_path):
        print(
            "Beijing road network subfigure skipped - missing "
            f"{roadnet_path}"
        )
        return

    from visualization.roadnet import plot_road_network_subfigure

    road_graph = _get_cached_sparse_graph(roadnet_path, graph_cache=graph_cache)
    buildings_gdf = None
    if building_path is None:
        print(
            "Beijing road network subfigure building background skipped - "
            f"missing Beijing building geometry in {region_dir}"
        )
    else:
        import geopandas as gpd

        buildings_gdf = gpd.read_file(building_path)
    plot_road_network_subfigure(
        road_graph,
        "Beijing",
        BEIJING_ROAD_NETWORK_CENTER[0],
        BEIJING_ROAD_NETWORK_CENTER[1],
        save_path=os.path.join(out, "beijing_road_network_subfigure.png"),
        figsize=BEIJING_ROAD_NETWORK_FIGSIZE,
        for_latex=for_latex,
        plot_title=plot_title,
        lat_span=BEIJING_ROAD_NETWORK_LAT_SPAN,
        compact=True,
        buildings_gdf=buildings_gdf,
        figure_name="beijing_road_network_subfigure",
    )


def _render_beijing_road_network_candidate_grid(
    data_dir,
    out,
    for_latex,
    plot_title,
    graph_cache=None,
):
    """Render nine fixed Beijing road-only windows for visual selection."""
    region_key = "beijing"
    region_dir = os.path.join(data_dir, "city", region_key)
    roadnet_path = os.path.join(region_dir, f"{region_key}_roadnet.graphml")
    if not os.path.exists(roadnet_path):
        print(
            "Beijing road candidate grid skipped - missing "
            f"{roadnet_path}"
        )
        return

    from visualization.roadnet import plot_road_network_candidate_grid

    road_graph = _get_cached_sparse_graph(roadnet_path, graph_cache=graph_cache)
    plot_road_network_candidate_grid(
        road_graph,
        BEIJING_ROAD_NETWORK_CANDIDATES,
        reserve_centers=BEIJING_ROAD_NETWORK_RESERVES,
        save_path=os.path.join(
            out,
            "beijing_road_network_candidate_grid.png",
        ),
        figsize=(3.9, 3.5),
        for_latex=for_latex,
        plot_title=plot_title,
        lat_span=BEIJING_ROAD_NETWORK_LAT_SPAN,
        panel_figsize=BEIJING_ROAD_NETWORK_FIGSIZE,
        min_segments=100,
        figure_name="beijing_road_network_candidate_grid",
    )


def _render_beijing_road_network_subfigure_option3(
    data_dir,
    out,
    for_latex,
    plot_title,
):
    """Render the local all-highway OSM road-type alternative."""
    region_key = "beijing"
    region_dir = os.path.join(data_dir, "city", region_key)
    cache_path = os.path.join(
        region_dir,
        "beijing_road_network_subfigure_option3_all_highways.graphml",
    )

    from roadnet._compat import load_networkx_graph
    from roadnet.osm_roadnet import fetch_all_highway_roadnet
    from utils.boundaries import graphml_covers_bbox
    from visualization.roadnet import (
        get_road_network_subfigure_frame,
        plot_road_type_subfigure,
    )

    frame = get_road_network_subfigure_frame(
        BEIJING_ROAD_NETWORK_CENTER[0],
        BEIJING_ROAD_NETWORK_CENTER[1],
        figsize=BEIJING_ROAD_NETWORK_FIGSIZE,
        lat_span=BEIJING_ROAD_NETWORK_LAT_SPAN,
    )
    if frame is None:
        return

    cache_valid = False
    if os.path.exists(cache_path):
        try:
            cache_valid = graphml_covers_bbox(cache_path, frame["bounds"])
        except (OSError, RuntimeError, ValueError) as exc:
            print(
                "Beijing road type cache rejected - could not validate "
                f"coverage: {exc}"
            )

    source_state = "cached_local_osm_graphml"
    if not cache_valid:
        source_state = "fresh_local_osm_graphml"
        try:
            fetch_all_highway_roadnet(frame["bounds"], cache_path)
            if not graphml_covers_bbox(cache_path, frame["bounds"]):
                raise RuntimeError(
                    "fetched graph does not cover the requested map frame"
                )
        except Exception as exc:
            print(
                "Beijing road type subfigure skipped - local all-highway "
                f"OSM fetch failed and no valid cache is available: {exc}"
            )
            return

    try:
        road_graph = load_networkx_graph(cache_path)
    except Exception as exc:
        print(
            "Beijing road type subfigure skipped - valid local OSM cache "
            f"could not be loaded: {exc}"
        )
        return

    try:
        plot_road_type_subfigure(
            road_graph,
            "Beijing",
            BEIJING_ROAD_NETWORK_CENTER[0],
            BEIJING_ROAD_NETWORK_CENTER[1],
            save_path=os.path.join(
                out,
                "beijing_road_network_subfigure_option3.png",
            ),
            figsize=BEIJING_ROAD_NETWORK_FIGSIZE,
            for_latex=for_latex,
            plot_title=plot_title,
            lat_span=BEIJING_ROAD_NETWORK_LAT_SPAN,
            source_state=source_state,
            figure_name="beijing_road_network_subfigure_option3",
        )
    finally:
        del road_graph


def _paper_pdf_path(out, figure_name, plot_title=False):
    """Return the expected PDF path for a paper figure."""
    path = os.path.join(out, f"{figure_name}.pdf")
    if plot_title:
        from visualization._utils import _add_title_suffix

        path = _add_title_suffix(path)
    return path


def _paper_report_path(out, figure_name, plot_title=False):
    """Return the expected JSON report path for a paper figure."""
    pdf_path = _paper_pdf_path(out, figure_name, plot_title=plot_title)
    return pdf_path.rsplit(".", 1)[0] + "_report.json"


def _load_figure1_panel_reports(out, plot_title=False):
    """Load source-figure metadata needed for Figure 1 zoom callouts."""
    reports = {}
    for name in FIGURE1_FRAMEWORK_REPORT_DEPENDENCIES:
        report_path = _paper_report_path(out, name, plot_title=plot_title)
        if not os.path.exists(report_path):
            print(f"figure1_framework: missing source report {report_path}")
            continue
        with open(report_path, "r", encoding="utf-8") as f:
            reports[name] = json.load(f)
    return reports


def _draw_figure1_overlay(
    overlay_path,
    page_size,
    canvas_size,
    panel_rects=None,
    panel_reports=None,
):
    """Draw vector labels, process box, and arrows for Figure 1."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    page_w, page_h = page_size
    canvas_w, canvas_h = canvas_size
    fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0), facecolor="none")
    panel_rects = panel_rects or {}
    panel_reports = panel_reports or {}

    def rect_from_drawio(x, y, w, h):
        return (
            x / canvas_w,
            1.0 - (y + h) / canvas_h,
            w / canvas_w,
            h / canvas_h,
        )

    fig.text(
        460 / canvas_w,
        1.0 - 260 / canvas_h,
        "Gravity\nmodel",
        ha="center",
        va="center",
        fontsize=9,
        color="#222222",
        bbox=dict(
            boxstyle="square,pad=0.22",
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.8,
        ),
    )

    def fig_point(x, y):
        return (x / canvas_w, 1.0 - y / canvas_h)

    for start, end in (
        (fig_point(460, 200), fig_point(460, 230)),
        (fig_point(460, 290), fig_point(460, 320)),
    ):
        arrow = FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#333333",
            shrinkA=0,
            shrinkB=0,
        )
        fig.add_artist(arrow)

    for label, x, y in (
        ("a", 0, 0),
        ("b", 380, 0),
        ("c", 0, 340),
        ("d", 380, 300),
    ):
        fig.text(
            x / canvas_w,
            1.0 - y / canvas_h,
            label,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    def fig_rect_from_page(rect):
        x, y, w, h = rect
        return (x / page_w, y / page_h, w / page_w, h / page_h)

    def page_rect_from_saved_fraction(parent_name, saved_rect):
        if parent_name not in panel_rects:
            return None
        px, py, pw, ph = panel_rects[parent_name]
        return (
            px + saved_rect["x0"] * pw,
            py + saved_rect["y0"] * ph,
            (saved_rect["x1"] - saved_rect["x0"]) * pw,
            (saved_rect["y1"] - saved_rect["y0"]) * ph,
        )

    def fig_rect_from_saved_fraction(parent_name, saved_rect):
        rect = page_rect_from_saved_fraction(parent_name, saved_rect)
        return fig_rect_from_page(rect) if rect is not None else None

    def map_axis_report(parent_name):
        report = panel_reports.get(parent_name) or {}
        if parent_name == "regions_map_oracle_reduction_rank":
            panel = report.get("panels", {}).get("oracle_reduction_map", {})
            return panel.get("map_axis_saved_tight") or panel.get("axis")
        return report.get("map", {}).get("axis")

    def map_axis_rect(parent_name):
        axis = map_axis_report(parent_name)
        if not axis:
            return None
        saved_rect = axis.get("bbox_saved_tight_fraction")
        if not saved_rect:
            return None
        return fig_rect_from_saved_fraction(parent_name, saved_rect)

    def data_rect_in_map(parent_name, lng_range, lat_range):
        axis = map_axis_report(parent_name)
        map_rect = map_axis_rect(parent_name)
        if not axis or not map_rect:
            return None
        xlim = axis.get("xlim_data")
        ylim = axis.get("ylim_data")
        if not xlim or not ylim:
            return None
        x0, x1 = float(xlim[0]), float(xlim[1])
        y0, y1 = float(ylim[0]), float(ylim[1])
        if x1 == x0 or y1 == y0:
            return None
        rx0 = (float(lng_range[0]) - x0) / (x1 - x0)
        rx1 = (float(lng_range[1]) - x0) / (x1 - x0)
        ry0 = (float(lat_range[0]) - y0) / (y1 - y0)
        ry1 = (float(lat_range[1]) - y0) / (y1 - y0)
        mx, my, mw, mh = map_rect
        left = mx + min(rx0, rx1) * mw
        right = mx + max(rx0, rx1) * mw
        bottom = my + min(ry0, ry1) * mh
        top = my + max(ry0, ry1) * mh
        return (left, bottom, right - left, top - bottom)

    def add_zoom_box(rect, color="#c43c2f"):
        if rect is None:
            return
        fig.add_artist(
            Rectangle(
                (rect[0], rect[1]),
                rect[2],
                rect[3],
                transform=fig.transFigure,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                zorder=200,
            )
        )

    def add_connector(start, end, color="#c43c2f"):
        fig.add_artist(
            FancyArrowPatch(
                start,
                end,
                transform=fig.transFigure,
                arrowstyle="-",
                linewidth=0.8,
                color=color,
                alpha=0.9,
                zorder=190,
                shrinkA=0,
                shrinkB=0,
            )
        )

    b_map = panel_reports.get("shanghai_heatmap", {}).get("map", {})
    b_lng = b_map.get("lng_range")
    b_lat = b_map.get("lat_range")
    a_zoom = None
    if b_lng and b_lat:
        a_zoom = data_rect_in_map(
            "regions_map_oracle_reduction_rank",
            b_lng,
            b_lat,
        )
    add_zoom_box(a_zoom)
    b_map_rect = map_axis_rect("shanghai_heatmap")
    if a_zoom and b_map_rect:
        bx, by, bw, bh = b_map_rect
        add_connector(
            (a_zoom[0] + a_zoom[2], a_zoom[1] + a_zoom[3]),
            (bx, by + bh),
        )
        add_connector((a_zoom[0] + a_zoom[2], a_zoom[1]), (bx, by))

    d_map = panel_reports.get("shanghai_building_height_merchant", {}).get("map", {})
    d_lng = d_map.get("lng_range")
    d_lat = d_map.get("lat_range")
    if d_lng and d_lat:
        b_zoom = data_rect_in_map(
            "shanghai_heatmap",
            d_lng,
            d_lat,
        )
        add_zoom_box(b_zoom)
        d_map_rect = map_axis_rect("shanghai_building_height_merchant")
        if b_zoom and d_map_rect:
            dx, dy, dw, dh = d_map_rect
            add_connector((b_zoom[0], b_zoom[1]), (dx, dy + dh))
            add_connector((b_zoom[0] + b_zoom[2], b_zoom[1]), (dx + dw, dy + dh))

    fig.savefig(overlay_path, transparent=True)
    plt.close(fig)


def _fit_pdf_rect(source_page, target_rect, return_rect=False):
    """Return a transform that fits a PDF page in target_rect without stretching."""
    from pypdf import Transformation

    target_x, target_y, target_w, target_h = target_rect
    source_w = float(source_page.mediabox.width)
    source_h = float(source_page.mediabox.height)
    scale = min(target_w / source_w, target_h / source_h)
    fitted_w = source_w * scale
    fitted_h = source_h * scale
    tx = target_x + (target_w - fitted_w) / 2.0
    ty = target_y + (target_h - fitted_h) / 2.0
    transform = Transformation().scale(scale).translate(tx, ty)
    if return_rect:
        return transform, (tx, ty, fitted_w, fitted_h)
    return transform


def _render_figure1_framework(out, plot_title):
    """Compose Figure 1 from vector PDF panels using the draw.io framework."""
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from visualization._utils import _add_title_suffix

    canvas_w, canvas_h = 561.0, 481.0
    page_w, page_h = 7.0 * 72.0, 6.0 * 72.0

    def rect_from_drawio_points(x, y, w, h):
        return (
            x / canvas_w * page_w,
            page_h - (y + h) / canvas_h * page_h,
            w / canvas_w * page_w,
            h / canvas_h * page_h,
        )

    panel_specs = (
        ("regions_map_oracle_reduction_rank", rect_from_drawio_points(0, 20, 360, 300)),
        ("shanghai_heatmap", rect_from_drawio_points(380, 20, 160, 180)),
        ("distance_comparison", rect_from_drawio_points(0, 360, 360, 100)),
        ("shanghai_building_height_merchant", rect_from_drawio_points(380, 320, 160, 160)),
    )
    source_paths = {
        name: _paper_pdf_path(out, name, plot_title=plot_title)
        for name in FIGURE1_FRAMEWORK_DEPENDENCIES
    }
    missing = [name for name, path in source_paths.items() if not os.path.exists(path)]
    if missing:
        print(
            "figure1_framework skipped - missing vector source PDFs: "
            + ", ".join(missing)
        )
        return

    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)
    placed_rects = {}
    for name, target_rect in panel_specs:
        source_page = PdfReader(source_paths[name]).pages[0]
        transform, placed_rect = _fit_pdf_rect(
            source_page,
            target_rect,
            return_rect=True,
        )
        placed_rects[name] = placed_rect
        page.merge_transformed_page(
            source_page,
            transform,
            over=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        overlay_path = tmp.name
    try:
        _draw_figure1_overlay(
            overlay_path,
            page_size=(page_w, page_h),
            canvas_size=(canvas_w, canvas_h),
            panel_rects=placed_rects,
            panel_reports=_load_figure1_panel_reports(out, plot_title=plot_title),
        )
        overlay_page = PdfReader(overlay_path).pages[0]
        page.merge_page(overlay_page, over=True)
    finally:
        try:
            os.remove(overlay_path)
        except OSError:
            pass

    save_path = os.path.join(out, "figure1_framework.pdf")
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        writer.write(f)
    print(f"Figure 1 framework saved to {save_path}")


def _draw_figure2_overlay(overlay_path, page_size, canvas_size):
    """Draw vector panel labels for Figure 2."""
    import matplotlib.pyplot as plt

    page_w, page_h = page_size
    canvas_w, canvas_h = canvas_size
    fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0), facecolor="none")

    for label, x, y in (
        ("a", 43, 0),
        ("b", 43, 90),
        ("c", 163, 0),
    ):
        fig.text(
            x / canvas_w,
            1.0 - y / canvas_h,
            label,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.savefig(overlay_path, transparent=True)
    plt.close(fig)


def _render_figure2_framework(out, plot_title):
    """Compose Figure 2 from vector PDF panels using the draw.io frame."""
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from visualization._utils import _add_title_suffix

    canvas_w, canvas_h = 525.0, 281.0
    page_w = 7.0 * 72.0
    page_h = page_w * canvas_h / canvas_w

    def rect_from_drawio_points(x, y, w, h):
        return (
            x / canvas_w * page_w,
            page_h - (y + h) / canvas_h * page_h,
            w / canvas_w * page_w,
            h / canvas_h * page_h,
        )

    panel_specs = (
        ("tanner_impedance_fit_panel_a", rect_from_drawio_points(43, 20, 100, 70)),
        ("city_oracle_reduction_rank_top10", rect_from_drawio_points(43, 110, 100, 170)),
        (
            "regions_map_oracle_reduction_rank_map_only",
            rect_from_drawio_points(163, 20, 360, 260),
        ),
    )
    source_paths = {
        name: _paper_pdf_path(out, name, plot_title=plot_title)
        for name in FIGURE2_FRAMEWORK_DEPENDENCIES
    }
    missing = [name for name, path in source_paths.items() if not os.path.exists(path)]
    if missing:
        print(
            "figure2_framework skipped - missing vector source PDFs: "
            + ", ".join(missing)
        )
        return

    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)
    for name, target_rect in panel_specs:
        source_page = PdfReader(source_paths[name]).pages[0]
        transform = _fit_pdf_rect(source_page, target_rect)
        page.merge_transformed_page(
            source_page,
            transform,
            over=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        overlay_path = tmp.name
    try:
        _draw_figure2_overlay(
            overlay_path,
            page_size=(page_w, page_h),
            canvas_size=(canvas_w, canvas_h),
        )
        overlay_page = PdfReader(overlay_path).pages[0]
        page.merge_page(overlay_page, over=True)
    finally:
        try:
            os.remove(overlay_path)
        except OSError:
            pass

    save_path = os.path.join(out, "figure2_framework.pdf")
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        writer.write(f)
    print(f"Figure 2 framework saved to {save_path}")


def _draw_figure3_overlay(overlay_path, page_size, canvas_size):
    """Draw vector panel labels for Figure 3."""
    import matplotlib.pyplot as plt

    page_w, page_h = page_size
    canvas_w, canvas_h = canvas_size
    fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0), facecolor="none")

    for label, x, y in (
        ("a", 0, 0),
        ("b", 0, 270),
    ):
        fig.text(
            x / canvas_w,
            1.0 - y / canvas_h,
            label,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.savefig(overlay_path, transparent=True)
    plt.close(fig)


def _render_figure3_framework(out, plot_title):
    """Compose Figure 3 from vector PDF panels using the draw.io frame."""
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from visualization._utils import _add_title_suffix

    canvas_w, canvas_h = 331.0, 421.0
    page_w = 7.0 * 72.0
    page_h = page_w * canvas_h / canvas_w

    def rect_from_drawio_points(x, y, w, h):
        return (
            x / canvas_w * page_w,
            page_h - (y + h) / canvas_h * page_h,
            w / canvas_w * page_w,
            h / canvas_h * page_h,
        )

    panel_specs = (
        (
            "city_population_detour_oracle_reduction_bubble",
            rect_from_drawio_points(0, 20, 330, 230),
        ),
        (
            "city_reduction_rank_all_macro_region",
            rect_from_drawio_points(0, 290, 330, 130),
        ),
    )
    source_paths = {
        name: _paper_pdf_path(out, name, plot_title=plot_title)
        for name in FIGURE3_FRAMEWORK_DEPENDENCIES
    }
    missing = [name for name, path in source_paths.items() if not os.path.exists(path)]
    if missing:
        print(
            "figure3_framework skipped - missing vector source PDFs: "
            + ", ".join(missing)
        )
        return

    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)
    for name, target_rect in panel_specs:
        source_page = PdfReader(source_paths[name]).pages[0]
        transform = _fit_pdf_rect(source_page, target_rect)
        page.merge_transformed_page(
            source_page,
            transform,
            over=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        overlay_path = tmp.name
    try:
        _draw_figure3_overlay(
            overlay_path,
            page_size=(page_w, page_h),
            canvas_size=(canvas_w, canvas_h),
        )
        overlay_page = PdfReader(overlay_path).pages[0]
        page.merge_page(overlay_page, over=True)
    finally:
        try:
            os.remove(overlay_path)
        except OSError:
            pass

    save_path = os.path.join(out, "figure3_framework.pdf")
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        writer.write(f)
    print(f"Figure 3 framework saved to {save_path}")


def _draw_figure4_overlay(overlay_path, page_size, canvas_size):
    """Draw vector panel labels for Figure 4."""
    import matplotlib.pyplot as plt

    page_w, page_h = page_size
    canvas_w, canvas_h = canvas_size
    fig = plt.figure(figsize=(page_w / 72.0, page_h / 72.0), facecolor="none")

    for label, x, y in (
        ("a", 0, 0),
        ("b", 0, 200),
        ("c", 190, 200),
    ):
        fig.text(
            x / canvas_w,
            1.0 - y / canvas_h,
            label,
            ha="left",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    fig.savefig(overlay_path, transparent=True)
    plt.close(fig)


def _render_figure4_framework(out, plot_title):
    """Compose Figure 4 from vector PDF panels using the draw.io frame."""
    import tempfile

    from pypdf import PdfReader, PdfWriter

    from visualization._utils import _add_title_suffix

    canvas_w, canvas_h = 362.0, 331.0
    page_w = 7.0 * 72.0
    page_h = page_w * canvas_h / canvas_w

    def rect_from_drawio_points(x, y, w, h):
        return (
            x / canvas_w * page_w,
            page_h - (y + h) / canvas_h * page_h,
            w / canvas_w * page_w,
            h / canvas_h * page_h,
        )

    panel_specs = (
        ("flow_frame.drawio", rect_from_drawio_points(0, 20, 360, 160)),
        ("tanner_impedance_fit", rect_from_drawio_points(0, 220, 170, 110)),
        ("alpha_calibration_5km", rect_from_drawio_points(190, 220, 170, 110)),
    )
    source_paths = {}
    for name in FIGURE4_FRAMEWORK_DEPENDENCIES:
        source_paths[name] = _paper_pdf_path(
            out,
            name,
            plot_title=plot_title and name != "flow_frame.drawio",
        )
    missing = [name for name, path in source_paths.items() if not os.path.exists(path)]
    if missing:
        print(
            "figure4_framework skipped - missing vector source PDFs: "
            + ", ".join(missing)
        )
        return

    writer = PdfWriter()
    page = writer.add_blank_page(width=page_w, height=page_h)
    for name, target_rect in panel_specs:
        source_page = PdfReader(source_paths[name]).pages[0]
        transform = _fit_pdf_rect(source_page, target_rect)
        page.merge_transformed_page(
            source_page,
            transform,
            over=True,
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        overlay_path = tmp.name
    try:
        _draw_figure4_overlay(
            overlay_path,
            page_size=(page_w, page_h),
            canvas_size=(canvas_w, canvas_h),
        )
        overlay_page = PdfReader(overlay_path).pages[0]
        page.merge_page(overlay_page, over=True)
    finally:
        try:
            os.remove(overlay_path)
        except OSError:
            pass

    save_path = os.path.join(out, "figure4_framework.pdf")
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        writer.write(f)
    print(f"Figure 4 framework saved to {save_path}")


def _plot_power_generation_mix_pie(
    ax,
    region_key,
    legend_ax=None,
    radius=1.18,
    legend_fontsize=9,
    legend_handlelength=1.2,
    legend_handleheight=0.7,
    legend_handletextpad=0.5,
    legend_labelspacing=0.5,
):
    """Draw the 2022 power generation mix pie for one region."""
    from matplotlib.patches import Patch

    generation = POWER_GENERATION_2022_100M_KWH[region_key]
    total = generation["total"]
    components = ["Hydro", "Thermal", "Wind", "Solar"]
    values = [generation[name] for name in components]
    remainder = total - sum(values)
    if remainder / total > 0.005:
        components.append("Other")
        values.append(remainder)
    percentages = [value / total * 100 for value in values]
    nonzero = [
        (name, value)
        for name, value in zip(components, values)
        if value > 0
    ]
    pie_labels = [name for name, _ in nonzero]
    pie_values = [value for _, value in nonzero]
    pie_colors = [POWER_MIX_COLORS[name] for name in pie_labels]

    ax.pie(
        pie_values,
        colors=pie_colors,
        startangle=92,
        counterclock=False,
        radius=radius,
        wedgeprops=dict(width=radius * 0.36, edgecolor="white", linewidth=2.2),
    )
    legend_handles = [
        Patch(
            facecolor=POWER_MIX_COLORS[name],
            edgecolor="none",
            label=f"{name} {pct:.1f}%",
        )
        for name, pct in zip(components, percentages)
    ]
    if legend_ax is None:
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.24),
            frameon=False,
            fontsize=legend_fontsize,
            ncol=1,
            handlelength=legend_handlelength,
            handleheight=legend_handleheight,
            handletextpad=legend_handletextpad,
            labelspacing=legend_labelspacing,
            borderaxespad=0.0,
        )
    else:
        legend_ax.legend(
            handles=legend_handles,
            loc="center left",
            frameon=False,
            fontsize=legend_fontsize,
            ncol=1,
            handlelength=legend_handlelength,
            handleheight=legend_handleheight,
            handletextpad=legend_handletextpad,
            labelspacing=legend_labelspacing,
            borderaxespad=0.0,
        )
        legend_ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_axis_off()


def _render_city_context(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
    figure_name,
    region_key,
    display_name,
    power_region_key=None,
):
    """Render a one-row city context figure."""
    power_region_key = power_region_key or region_key
    selected = city_df[city_df["region_key"] == region_key].copy()
    if selected.empty:
        print(f"{figure_name} skipped — {display_name} row missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=False,
        bin_width_km=0.1,
    )
    if built is None:
        return

    binned_by_city, labels = built
    if region_key not in binned_by_city:
        print(f"{figure_name} skipped — {display_name} binned data missing")
        return

    import matplotlib.pyplot as plt

    from visualization._utils import _add_title_suffix
    from visualization.analysis import (
        _plot_reduction_absolute_scatter_panel,
        _plot_reduction_detour_panel,
    )
    fig = plt.figure(figsize=(6.4, 11.6))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.78], hspace=0.30)
    power_grid = outer[2].subgridspec(1, 2, width_ratios=[1.0, 0.72], wspace=0.12)
    ax_power_mix = fig.add_subplot(power_grid[0])
    ax_power_legend = fig.add_subplot(power_grid[1])

    _, ax_reduction, _ = _plot_reduction_detour_panel(
        fig,
        outer[0],
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
        outer[1],
        binned_by_city[region_key],
        labels.get(region_key, display_name),
        min_air_km=0.2,
        absolute_reduction_metric="per_order_g",
        show_city_label=False,
    )

    _plot_power_generation_mix_pie(
        ax_power_mix,
        power_region_key,
        legend_ax=ax_power_legend,
    )

    for ax in (ax_reduction, ax_scatter):
        pos = ax.get_position()
        new_height = pos.height * 0.75
        ax.set_position([
            pos.x0,
            pos.y0 + (pos.height - new_height) / 2,
            pos.width,
            new_height,
        ])

    panel_label_x = ax_reduction.get_position().x0 - 0.055
    for label, ax in (
        ("a", ax_reduction),
        ("b", ax_scatter),
        ("c", ax_power_mix),
    ):
        pos = ax.get_position()
        fig.text(
            panel_label_x,
            pos.y1 + 0.010,
            label,
            fontsize=13,
            fontweight="bold",
            va="bottom",
        )

    if plot_title:
        fig.suptitle(
            f"{display_name} Emission Reduction by Distance and Power Mix",
            fontsize=14,
        )

    save_path = os.path.join(out, f"{figure_name}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{display_name} city context saved to {save_path}")


def _plot_city_context_combined_ab_panel(
    fig,
    subplot_spec,
    binned,
    city_label,
    show_saturation_fit,
    bin_width_km=0.1,
    min_air_km=0.2,
    max_air_km=5.0,
    bar_metric="reduction_pct",
):
    """Draw percent reduction and absolute reduction on one city-context panel."""
    import numpy as np

    from visualization.analysis import _plot_reduction_detour_panel

    _, ax_left, _ = _plot_reduction_detour_panel(
        fig,
        subplot_spec,
        binned,
        city_label,
        bin_width_km=bin_width_km,
        min_air_km=min_air_km,
        max_air_km=max_air_km,
        absolute_reduction_metric="per_order_g",
        show_detour_boxplot=False,
        mark_reduction_break_even=False,
        show_city_label=False,
        bar_width_fraction=0.78,
        show_absolute_axis=False,
        show_saturation_fit=show_saturation_fit,
        bar_metric=bar_metric,
    )

    centres = np.asarray(binned["centres"], dtype=float)
    reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
    max_km = float(binned["max_km"])
    x_max = min(max_air_km, max_km)
    x_min = min(min_air_km, x_max)
    visible = (centres >= x_min) & (centres <= x_max)
    finite = visible & np.isfinite(centres) & np.isfinite(reduction_abs)
    x = centres[finite]
    y = reduction_abs[finite]

    ax_right = ax_left.twinx()
    ax_right.patch.set_visible(False)
    ax_right.scatter(
        x,
        y,
        color="#555555",
        alpha=0.72,
        s=14,
        linewidth=0,
        zorder=6,
    )

    ylim_values = y
    if x.size >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_fit = np.array([x_min, x_max], dtype=float)
        y_fit = slope * x_fit + intercept
        ax_right.plot(
            x_fit,
            y_fit,
            color="#111111",
            linewidth=1.2,
            alpha=0.85,
            zorder=7,
        )
        sign = "+" if intercept >= 0 else "-"
        equation = f"y = {slope:.1f}x {sign} {abs(intercept):.1f}"
        ax_right.text(
            0.97,
            0.94,
            equation,
            transform=ax_right.transAxes,
            fontsize=7.2,
            ha="right",
            va="top",
            color="#111111",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.6),
            zorder=8,
        )
        ylim_values = np.concatenate([ylim_values, y_fit])

    ax_right.axhline(0, color="#555555", linewidth=0.5, alpha=0.35)
    finite_ylim = ylim_values[np.isfinite(ylim_values)]
    if finite_ylim.size:
        y_min = float(np.min(finite_ylim))
        y_max = float(np.max(finite_ylim))
        pad = (
            (y_max - y_min) * 0.12
            if y_max > y_min
            else max(abs(y_min) * 0.12, 1.0)
        )
        ax_right.set_ylim(y_min - pad, y_max + pad)

    ax_right.set_ylabel("E-bike - Drone (g CO2e/order)", fontsize=9, color="#555555")
    ax_right.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    return ax_left, ax_right


def _plot_city_context_group_panels(
    fig,
    subplot_spec,
    binned_by_city,
    labels,
    contexts,
    panel_start_letter="a",
    label_each_subplot=False,
    show_city_panel_anchors=False,
    bar_metric="reduction_pct",
):
    """Draw the grouped city context panels into a parent subplot spec."""
    from matplotlib.lines import Line2D

    outer = subplot_spec.subgridspec(1, len(contexts), wspace=0.22)
    reduction_axes = []
    absolute_axes = []
    panel_records = []
    panel_start_ord = ord(panel_start_letter)

    for i, context in enumerate(contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        show_saturation_fit = region_key not in {
            "dalibaizuzizhizhou",
            "chengdu",
        }
        block = outer[0, i].subgridspec(
            2,
            1,
            height_ratios=[0.78, 1.20],
            hspace=0.20,
        )

        power_grid = block[0].subgridspec(
            1,
            2,
            width_ratios=[1.0, 0.78],
            wspace=0.04,
        )
        ax_power_mix = fig.add_subplot(power_grid[0])
        ax_power_legend = fig.add_subplot(power_grid[1])
        _plot_power_generation_mix_pie(
            ax_power_mix,
            power_region_key,
            legend_ax=ax_power_legend,
            radius=1.62,
            legend_fontsize=12.0,
            legend_handlelength=1.70,
            legend_handleheight=0.95,
            legend_handletextpad=0.72,
            legend_labelspacing=0.62,
        )
        legend = ax_power_legend.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(12.0)

        ax_reduction, ax_absolute = _plot_city_context_combined_ab_panel(
            fig,
            block[1],
            binned_by_city[region_key],
            labels.get(region_key, display_name),
            show_saturation_fit,
            bin_width_km=0.2,
            bar_metric=bar_metric,
        )
        reduction_axes.append(ax_reduction)
        absolute_axes.append(ax_absolute)

        for ax in (ax_reduction, ax_absolute):
            ax.xaxis.label.set_size(8.5)
            ax.yaxis.label.set_size(8.5)
            ax.tick_params(axis="both", labelsize=7.5)
        if i != 0:
            ax_reduction.set_ylabel("")
            ax_reduction.tick_params(axis="y", left=False, labelleft=False)
        if i != len(contexts) - 1:
            ax_absolute.set_ylabel("")
            ax_absolute.tick_params(axis="y", right=False, labelright=False)
            ax_absolute.spines["right"].set_visible(False)

        if label_each_subplot:
            power_panel_letter = chr(panel_start_ord + 2 * i)
            reduction_panel_letter = chr(panel_start_ord + 2 * i + 1)
        else:
            power_panel_letter = chr(panel_start_ord + i)
            reduction_panel_letter = None

        power_pos = ax_power_mix.get_position()
        fig.text(
            power_pos.x0 - 0.010,
            power_pos.y1 + 0.035,
            power_panel_letter,
            fontsize=13,
            fontweight="bold",
            ha="right",
            va="bottom",
        )
        fig.text(
            power_pos.x0 + 0.006,
            power_pos.y1 + 0.035,
            display_name,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if reduction_panel_letter is not None:
            reduction_pos = ax_reduction.get_position()
            fig.text(
                reduction_pos.x0 - 0.010,
                reduction_pos.y1 + 0.012,
                reduction_panel_letter,
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="bottom",
            )

        panel_anchor_points = None
        if show_city_panel_anchors:
            panel_pos = outer[0, i].get_position(fig)
            panel_x_center = (panel_pos.x0 + panel_pos.x1) / 2
            panel_anchor_points = {
                "top": (panel_x_center, panel_pos.y1),
                "bottom": (panel_x_center, panel_pos.y0),
            }
            for x, y in panel_anchor_points.values():
                fig.add_artist(
                    Line2D(
                        [x],
                        [y],
                        marker="o",
                        markersize=5.2,
                        markerfacecolor="none",
                        markeredgecolor="#111111",
                        markeredgewidth=1.0,
                        linestyle="None",
                        transform=fig.transFigure,
                        zorder=95,
                        clip_on=False,
                    )
                )

        panel_records.append(
            {
                "context": context,
                "panel_letter": power_panel_letter,
                "power_panel_letter": power_panel_letter,
                "reduction_panel_letter": reduction_panel_letter,
                "panel_anchor_points": panel_anchor_points,
                "power_mix_ax": ax_power_mix,
                "power_legend_ax": ax_power_legend,
                "reduction_ax": ax_reduction,
                "absolute_ax": ax_absolute,
            }
        )

    for axes in (reduction_axes, absolute_axes):
        if axes:
            y_min = min(ax.get_ylim()[0] for ax in axes)
            y_max = max(ax.get_ylim()[1] for ax in axes)
            for ax in axes:
                ax.set_ylim(y_min, y_max)

    return panel_records


def _plot_compact_city_context_summary_panels(
    fig,
    subplot_spec,
    binned_by_city,
    labels,
    contexts,
    panel_start_letter="b",
):
    """Draw compact power-mix and distance panels for the city summary."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    middle = subplot_spec.subgridspec(
        2,
        1,
        height_ratios=[0.12, 0.88],
        hspace=0.08,
    )
    legend_ax = fig.add_subplot(middle[0])
    legend_ax.legend(
        handles=[
            Patch(facecolor=POWER_MIX_COLORS[name], edgecolor="none", label=name)
            for name in ("Hydro", "Thermal", "Wind", "Solar", "Other")
        ],
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    legend_ax.set_axis_off()

    outer = middle[1].subgridspec(1, len(contexts), wspace=0.24)
    visible_by_city = {}
    absolute_values = []
    for context in contexts:
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        x_max = min(5.0, float(binned["max_km"]))
        x_min = min(0.2, x_max)
        visible = (
            (centres >= x_min)
            & (centres <= x_max)
            & np.isfinite(centres)
            & np.isfinite(reduction_abs)
        )
        visible_by_city[region_key] = (centres, reduction_abs, visible, x_min, x_max)
        absolute_values.extend(reduction_abs[visible].tolist())
        if visible.sum() >= 2:
            slope, intercept = np.polyfit(
                centres[visible], reduction_abs[visible], deg=1
            )
            absolute_values.extend(
                [slope * x_min + intercept, slope * x_max + intercept]
            )

    finite_absolute = np.asarray(absolute_values, dtype=float)
    finite_absolute = finite_absolute[np.isfinite(finite_absolute)]
    if finite_absolute.size:
        absolute_min = min(0.0, float(finite_absolute.min()))
        absolute_max = max(0.0, float(finite_absolute.max()))
        absolute_pad = max((absolute_max - absolute_min) * 0.08, 0.5)
        shared_absolute_ylim = (
            absolute_min - absolute_pad,
            absolute_max + absolute_pad,
        )
    else:
        shared_absolute_ylim = (-1.0, 1.0)

    panel_records = []
    panel_start_ord = ord(panel_start_letter)
    for i, context in enumerate(contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        block = outer[0, i].subgridspec(
            2,
            1,
            height_ratios=[0.34, 1.0],
            hspace=0.12,
        )
        ax_power = fig.add_subplot(block[0])
        ax_distance = fig.add_subplot(block[1])
        ax_absolute = ax_distance.twinx()
        ax_absolute.patch.set_visible(False)

        generation = POWER_GENERATION_2022_100M_KWH[power_region_key]
        total = float(generation["total"])
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [float(generation[name]) for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)
        shares = [value / total * 100.0 for value in values]
        left = 0.0
        for name, share in zip(components, shares):
            ax_power.barh(
                [0],
                [share],
                left=[left],
                height=0.52,
                color=POWER_MIX_COLORS[name],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            if share >= 12.0:
                ax_power.text(
                    left + share / 2.0,
                    0,
                    f"{share:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if name in {"Hydro", "Thermal"} else "#222222",
                    fontweight="bold",
                    zorder=4,
                )
            left += share
        ax_power.set_xlim(0, 100)
        ax_power.set_ylim(-0.55, 0.55)
        ax_power.set_axis_off()

        power_letter = chr(panel_start_ord + 2 * i)
        distance_letter = chr(panel_start_ord + 2 * i + 1)
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        ax_power.text(
            0.0,
            1.18,
            power_letter,
            transform=ax_power.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax_power.plot(
            [0.18],
            [1.25],
            marker=marker,
            markersize=4.3,
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            linestyle="None",
            transform=ax_power.transAxes,
            clip_on=False,
        )
        ax_power.text(
            0.27,
            1.18,
            display_name,
            transform=ax_power.transAxes,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        binned = binned_by_city[region_key]
        centres, reduction_abs, visible, x_min, x_max = visible_by_city[region_key]
        drone_share = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        bar_visible = visible & np.isfinite(drone_share)
        ax_distance.bar(
            centres[bar_visible],
            drone_share[bar_visible],
            width=0.156,
            color="#9ecae1",
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        ax_distance.set_xlim(x_min, x_max)
        ax_distance.set_ylim(0, 105)
        ax_distance.set_xticks([0.2, 2.5, 5.0])
        ax_distance.set_yticks([0, 50, 100])
        ax_distance.tick_params(axis="both", labelsize=7, length=2.5, pad=1.5)
        ax_distance.grid(True, axis="y", alpha=0.18, linewidth=0.45)
        ax_distance.set_axisbelow(True)
        ax_distance.spines["top"].set_visible(False)
        if i != 0:
            ax_distance.tick_params(axis="y", left=False, labelleft=False)
            ax_distance.spines["left"].set_visible(False)

        ax_absolute.scatter(
            centres[visible],
            reduction_abs[visible],
            s=7,
            color="#555555",
            alpha=0.72,
            linewidth=0,
            zorder=5,
        )
        if visible.sum() >= 2:
            slope, intercept = np.polyfit(
                centres[visible], reduction_abs[visible], deg=1
            )
            x_fit = np.array([x_min, x_max], dtype=float)
            ax_absolute.plot(
                x_fit,
                slope * x_fit + intercept,
                color="#111111",
                linewidth=0.9,
                alpha=0.82,
                zorder=6,
            )
        ax_absolute.axhline(0, color="#555555", linewidth=0.4, alpha=0.3)
        ax_absolute.set_ylim(*shared_absolute_ylim)
        ax_absolute.tick_params(
            axis="y",
            labelsize=7,
            labelcolor="#555555",
            length=2.5,
            pad=1.5,
        )
        ax_absolute.spines["top"].set_visible(False)
        if i != len(contexts) - 1:
            ax_absolute.tick_params(axis="y", right=False, labelright=False)
            ax_absolute.spines["right"].set_visible(False)

        ax_distance.text(
            0.0,
            1.02,
            distance_letter,
            transform=ax_distance.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        panel_pos = outer[0, i].get_position(fig)
        panel_x_center = (panel_pos.x0 + panel_pos.x1) / 2.0
        panel_anchor_points = {
            "top": (panel_x_center, panel_pos.y1),
            "bottom": (panel_x_center, panel_pos.y0),
        }
        for x, y in panel_anchor_points.values():
            fig.add_artist(
                Line2D(
                    [x],
                    [y],
                    marker=marker,
                    markersize=3.8,
                    markerfacecolor="white",
                    markeredgecolor="#111111",
                    markeredgewidth=0.75,
                    linestyle="None",
                    transform=fig.transFigure,
                    zorder=60,
                    clip_on=False,
                )
            )

        panel_records.append(
            {
                "context": context,
                "panel_letter": power_letter,
                "power_panel_letter": power_letter,
                "reduction_panel_letter": distance_letter,
                "panel_anchor_points": panel_anchor_points,
                "power_mix_ax": ax_power,
                "power_legend_ax": legend_ax,
                "reduction_ax": ax_distance,
                "absolute_ax": ax_absolute,
                "shared_absolute_ylim": shared_absolute_ylim,
            }
        )

    middle_pos = middle[1].get_position(fig)
    fig.text(
        middle_pos.x0 - 0.055,
        middle_pos.y0 + middle_pos.height * 0.32,
        "Oracle drone share (%)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color="#3182bd",
    )
    fig.text(
        middle_pos.x1 + 0.055,
        middle_pos.y0 + middle_pos.height * 0.32,
        "E-bike - Drone (g CO$_2$e/order)",
        rotation=90,
        fontsize=7.5,
        ha="center",
        va="center",
        color="#555555",
    )
    fig.text(
        (middle_pos.x0 + middle_pos.x1) / 2.0,
        middle_pos.y0 - 0.030,
        "Air distance (km)",
        fontsize=7.5,
        ha="center",
        va="top",
    )
    return panel_records


def _report_number(value):
    """Return a JSON-safe scalar with NaN/Inf represented as None."""
    import numpy as np

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


def _report_array(values):
    """Return a JSON-safe list from an array-like object."""
    return [_report_number(value) for value in values]


def _summary_stats(values):
    """Return compact summary statistics for an array-like numeric series."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "min": None,
            "max": None,
            "whisker_low": None,
            "whisker_high": None,
        }
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    high_limit = q3 + 1.5 * iqr
    whisker_low = arr[arr >= low_limit].min()
    whisker_high = arr[arr <= high_limit].max()
    return {
        "count": int(arr.size),
        "mean": _report_number(arr.mean()),
        "median": _report_number(median),
        "q1": _report_number(q1),
        "q3": _report_number(q3),
        "min": _report_number(arr.min()),
        "max": _report_number(arr.max()),
        "whisker_low": _report_number(whisker_low),
        "whisker_high": _report_number(whisker_high),
    }


def _report_axis(ax):
    """Return figure-fraction position and data scale for an axis."""
    pos = ax.get_position()
    return {
        "bbox_figure_fraction": {
            "x0": _report_number(pos.x0),
            "y0": _report_number(pos.y0),
            "x1": _report_number(pos.x1),
            "y1": _report_number(pos.y1),
            "width": _report_number(pos.width),
            "height": _report_number(pos.height),
        },
        "xlim_data": _report_array(ax.get_xlim()),
        "ylim_data": _report_array(ax.get_ylim()),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticks_data": _report_array(ax.get_xticks()),
        "yticks_data": _report_array(ax.get_yticks()),
    }


def _city_gef_components_report(df, axes):
    """Return exact plotted data for the GEF component panel."""
    import numpy as np

    ax, ax_dist = axes
    gef_col = None
    for candidate in ("grid_emission_factor", "gef"):
        if candidate in df.columns and df[candidate].notna().any():
            gef_col = candidate
            break
    if gef_col is None or "total_odpair_annual_sales" not in df.columns:
        return {"status": "skipped", "reason": "missing GEF or sales column"}

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

    plot_df = df.copy()
    if "region_level" in plot_df.columns:
        plot_df = plot_df[plot_df["region_level"] == "city"]
    for col in [gef_col, *required_cols]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[gef_col, *required_cols])
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
    x_all = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bars = []
    bins = []
    for idx, label in enumerate(labels):
        group = plot_df[plot_df["gef_range"] == label]
        if group.empty:
            continue
        orders_sum = group["total_odpair_annual_sales"].sum()
        bin_record = {
            "label_right_edge": label,
            "bin_left": _report_number(bin_edges[idx]),
            "bin_right": _report_number(bin_edges[idx + 1]),
            "bin_center": _report_number(x_all[idx]),
            "city_count": int(len(group)),
            "orders_sum": _report_number(orders_sum),
        }
        bins.append(bin_record)
        for spec in vehicle_specs:
            fixed = (
                (group[spec["production_col"]].sum() + group[spec["eol_col"]].sum())
                / orders_sum
                * 1000.0
            )
            use = group[spec["use_col"]].sum() / orders_sum * 1000.0
            x_center = x_all[idx] + spec["offset"] * bin_width
            bars.append(
                {
                    "vehicle": spec["label"],
                    "color": spec["color"],
                    "bin_label_right_edge": label,
                    "x_center": _report_number(x_center),
                    "width": _report_number(bin_width * 0.28),
                    "production_eol_segment": {
                        "bottom": 0.0,
                        "height_g_co2e_per_order": _report_number(fixed),
                        "top": _report_number(fixed),
                        "alpha": 0.38,
                    },
                    "use_phase_segment": {
                        "bottom": _report_number(fixed),
                        "height_g_co2e_per_order": _report_number(use),
                        "top": _report_number(fixed + use),
                        "alpha": 0.88,
                    },
                    "total_label": {
                        "x": _report_number(x_center),
                        "y": _report_number(fixed + use),
                        "text": f"{fixed + use:.1f}",
                        "rotation": 90,
                    },
                    "orders_sum": _report_number(orders_sum),
                    "city_count": int(len(group)),
                }
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
    if density.size and density.max() > 0:
        density = density / density.max()

    return {
        "status": "drawn",
        "panel_letter": "a",
        "panel_label": {
            "text": "a",
            "x_figure_fraction": _report_number(ax.get_position().x0 - 0.035),
            "y_figure_fraction": _report_number(ax.get_position().y1 + 0.006),
        },
        "axes": {
            "stacked_bars": _report_axis(ax),
            "gef_density": _report_axis(ax_dist),
        },
        "gef_column": gef_col,
        "bin_width": bin_width,
        "bin_edges": _report_array(bin_edges),
        "bins": bins,
        "bars": bars,
        "density_curve": [
            {"x": _report_number(x), "y": _report_number(y)}
            for x, y in zip(x_violin, density)
        ],
        "city_rug_marks": [
            {"x": _report_number(x), "y": 0.0}
            for x in density_x
        ],
    }


def _city_context_summary_report(
    binned_by_city,
    labels,
    panel_records,
    bar_metric="reduction_pct",
):
    """Return exact plotted data for the grouped city context row."""
    import numpy as np

    if bar_metric not in {"reduction_pct", "oracle_drone_share_pct"}:
        raise ValueError(
            "bar_metric must be 'reduction_pct' or 'oracle_drone_share_pct'"
        )

    reports = []
    for record in panel_records:
        context = record["context"]
        region_key = context["region_key"]
        binned = binned_by_city[region_key]
        centres = np.asarray(binned["centres"], dtype=float)
        reduction_pct = np.asarray(binned["reduction_pct"], dtype=float)
        reduction_abs = np.asarray(binned["reduction_abs"], dtype=float)
        oracle_drone_share_pct = np.asarray(
            binned.get("oracle_drone_share_pct", np.full_like(centres, np.nan)),
            dtype=float,
        )
        if bar_metric == "oracle_drone_share_pct":
            bar_values = oracle_drone_share_pct
            bar_color = "#9ecae1"
            bar_label = "oracle_drone_share_pct"
        else:
            bar_values = reduction_pct
            bar_color = None
            bar_label = "reduction_pct"
        detour_by_bin = binned.get("detour_by_bin") or [np.array([]) for _ in centres]
        max_km = float(binned["max_km"])
        x_min = min(0.2, max_km)
        x_max = min(5.0, max_km)
        visible = (centres >= x_min) & (centres <= x_max)
        finite_abs = visible & np.isfinite(centres) & np.isfinite(reduction_abs)
        x_abs = centres[finite_abs]
        y_abs = reduction_abs[finite_abs]

        linear_fit = None
        if x_abs.size >= 2:
            slope, intercept = np.polyfit(x_abs, y_abs, deg=1)
            x_fit = np.array([x_min, x_max], dtype=float)
            linear_fit = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "line_points": [
                    {
                        "x": _report_number(x),
                        "y": _report_number(slope * x + intercept),
                    }
                    for x in x_fit
                ],
                "equation": (
                    f"y = {slope:.1f}x "
                    f"{'+' if intercept >= 0 else '-'} {abs(intercept):.1f}"
                ),
            }

        show_saturation_fit = region_key not in {"dalibaizuzizhizhou", "chengdu"}
        saturation_fit = None
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
                    y_fit = saturation_model(x_fit, *params)
                    y0, amplitude, rate = params
                    saturation_fit = {
                        "y0": _report_number(y0),
                        "amplitude": _report_number(amplitude),
                        "rate": _report_number(rate),
                        "x0": _report_number(x0),
                        "line_points": [
                            {"x": _report_number(x), "y": _report_number(y)}
                            for x, y in zip(x_fit, y_fit)
                        ],
                        "equation": (
                            f"y = {y0:.1f} "
                            f"{'+' if amplitude >= 0 else '-'} {abs(amplitude):.1f}"
                            f"(1 - e^(-{rate:.2f}(x - {x0:.1f})))"
                        ),
                    }
                except Exception as exc:
                    saturation_fit = {"status": "fit_failed", "error": str(exc)}

        generation = POWER_GENERATION_2022_100M_KWH[context.get("power_region_key", region_key)]
        total = generation["total"]
        components = ["Hydro", "Thermal", "Wind", "Solar"]
        values = [generation[name] for name in components]
        remainder = total - sum(values)
        if remainder / total > 0.005:
            components.append("Other")
            values.append(remainder)

        reports.append(
            {
                "panel_letter": record["panel_letter"],
                "panel_label": {
                    "text": record["power_panel_letter"],
                    "x_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().x0 - 0.010
                    ),
                    "y_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().y1 + 0.035
                    ),
                },
                "reduction_panel_label": (
                    {
                        "text": record["reduction_panel_letter"],
                        "x_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().x0 - 0.010
                        ),
                        "y_figure_fraction": _report_number(
                            record["reduction_ax"].get_position().y1 + 0.012
                        ),
                    }
                    if record["reduction_panel_letter"] is not None
                    else None
                ),
                "city_label_position": {
                    "text": context["display_name"],
                    "x_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().x0 + 0.006
                    ),
                    "y_figure_fraction": _report_number(
                        record["power_mix_ax"].get_position().y1 + 0.035
                    ),
                },
                "panel_anchor_points": (
                    {
                        key: {
                            "x_figure_fraction": _report_number(point[0]),
                            "y_figure_fraction": _report_number(point[1]),
                        }
                        for key, point in record["panel_anchor_points"].items()
                    }
                    if record["panel_anchor_points"] is not None
                    else None
                ),
                "region_key": region_key,
                "city_label": labels.get(region_key, context["display_name"]),
                "display_name": context["display_name"],
                "city_detour_ratio": _report_number(
                    binned.get("city_detour_ratio")
                ),
                "valid_path_count": int(binned.get("valid_path_count", 0)),
                "invalid_path_count": int(binned.get("invalid_path_count", 0)),
                "axes": {
                    "power_mix": _report_axis(record["power_mix_ax"]),
                    "power_legend": _report_axis(record["power_legend_ax"]),
                    "bar_axis": _report_axis(record["reduction_ax"]),
                    "reduction_bars": _report_axis(record["reduction_ax"]),
                    "absolute_scatter": _report_axis(record["absolute_ax"]),
                },
                "bar_metric": bar_label,
                "power_mix": [
                    {
                        "component": name,
                        "value_100m_kwh": _report_number(value),
                        "share_pct": _report_number(value / total * 100.0),
                        "color": POWER_MIX_COLORS[name],
                    }
                    for name, value in zip(components, values)
                ],
                "distance_bins": [
                    {
                        "x_center_air_km": _report_number(center),
                        "bar_width_km": 0.2 * 0.78,
                        "bar_metric": bar_label,
                        "bar_height": _report_number(bar_value),
                        "bar_bottom": 0.0,
                        "bar_top": _report_number(bar_value),
                        "bar_color": (
                            bar_color
                            if bar_color is not None
                            else (
                                "#2ca02c"
                                if (np.isfinite(pct) and pct >= 0)
                                else "#d62728"
                            )
                        ),
                        "oracle_drone_share_pct": _report_number(oracle_share),
                        "reduction_pct_height": _report_number(pct),
                        "reduction_bar_bottom": 0.0,
                        "reduction_bar_top": _report_number(pct),
                        "reduction_bar_color": (
                            "#2ca02c"
                            if (np.isfinite(pct) and pct >= 0)
                            else "#d62728"
                        ),
                        "absolute_scatter_y_g_co2e_per_order": _report_number(abs_value),
                        "visible_in_panel": bool(is_visible),
                        "detour_ratio_summary": _summary_stats(detours),
                    }
                    for (
                        center,
                        pct,
                        oracle_share,
                        bar_value,
                        abs_value,
                        detours,
                        is_visible,
                    )
                    in zip(
                        centres,
                        reduction_pct,
                        oracle_drone_share_pct,
                        bar_values,
                        reduction_abs,
                        detour_by_bin,
                        visible,
                    )
                ],
                "absolute_scatter_points": [
                    {"x_air_km": _report_number(x), "y_g_co2e_per_order": _report_number(y)}
                    for x, y in zip(x_abs, y_abs)
                ],
                "absolute_linear_fit": linear_fit,
                "reduction_saturation_fit": saturation_fit,
                "shared_absolute_ylim": (
                    _report_array(record["shared_absolute_ylim"])
                    if record.get("shared_absolute_ylim") is not None
                    else None
                ),
            }
        )
    return reports


def _add_city_gef_xaxis_markers(fig, axes, selected, contexts):
    """Add selected city province-GEF markers to the top panel x-axis."""
    _, ax_dist = axes
    marker_records = []

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = selected[selected["region_key"] == region_key]
        if city_rows.empty or "gef" not in city_rows.columns:
            continue

        gef = pd.to_numeric(city_rows["gef"], errors="coerce").dropna()
        if gef.empty:
            continue

        x = float(gef.iloc[0])
        y = 0.0
        ax_dist.scatter(
            [x],
            [y],
            s=22,
            marker=marker,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=8,
            clip_on=False,
        )
        display_point = ax_dist.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "power_region_key": context.get("power_region_key", region_key),
                "province": _report_number(city_rows.iloc[0].get("province")),
                "gef": _report_number(x),
                "marker": marker,
                "x_data": _report_number(x),
                "y_data": _report_number(y),
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


def _add_city_detour_reduction_markers(
    fig,
    axes,
    df,
    contexts,
    reduction_col,
    panel_letters=("f", "g"),
):
    """Add selected city markers to the bottom detour-reduction scatter panels."""
    ax_low, _, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    axis_by_group = {
        "low_gef": ax_low,
        "high_gef": ax_high,
    }
    panel_letter_by_group = {
        "low_gef": panel_letters[0],
        "high_gef": panel_letters[1],
    }
    marker_records = []

    needed = [feature_var, reduction_col, group_var]
    if "region_key" not in df.columns or any(col not in df.columns for col in needed):
        return marker_records

    for context in contexts:
        region_key = context["region_key"]
        marker = CITY_CONTEXT_MARKERS.get(region_key, "o")
        city_rows = df[df["region_key"] == region_key].copy()
        if city_rows.empty:
            continue

        for col in needed:
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
        city_rows = city_rows.dropna(subset=needed)
        if city_rows.empty:
            continue

        row = city_rows.iloc[0]
        x = float(row[feature_var])
        y = float(row[reduction_col])
        gef = float(row[group_var])
        axis_name = "low_gef" if gef <= threshold else "high_gef"
        ax = axis_by_group[axis_name]
        ax.scatter(
            [x],
            [y],
            s=28,
            marker=marker,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.1,
            zorder=35,
            clip_on=False,
        )
        label_offsets = {
            "chengdu": (-4, 7),
            "beijing": (-8, -11),
            "shanghai": (6, 7),
            "guangzhou": (6, -11),
        }
        offset = label_offsets.get(region_key, (5, 5))
        ax.annotate(
            context["display_name"],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="#111111",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            bbox=dict(
                boxstyle="round,pad=0.12",
                facecolor="white",
                edgecolor="0.72",
                linewidth=0.45,
                alpha=0.88,
            ),
            zorder=36,
        )
        display_point = ax.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        marker_records.append(
            {
                "region_key": region_key,
                "display_name": context["display_name"],
                "axis_name": axis_name,
                "panel_letter": panel_letter_by_group[axis_name],
                "x_detour_ratio": _report_number(x),
                "y_reduction_pct": _report_number(y),
                "grid_emission_factor": _report_number(gef),
                "marker": marker,
                "label_offset_points": {"x": offset[0], "y": offset[1]},
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    return marker_records


def _add_major_city_detour_reduction_labels(
    fig,
    axes,
    df,
    reduction_col,
    panel_letters=("j", "k"),
):
    """Label major cities in the bottom detour-reduction scatter panels."""
    ax_low, _, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    needed = ["region_name_en", feature_var, reduction_col, group_var]
    if any(col not in df.columns for col in needed):
        return []

    columns = needed + (["region_key"] if "region_key" in df.columns else [])
    plot_df = df[columns].copy()
    for col in (feature_var, reduction_col, group_var):
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature_var, reduction_col, group_var])
    if plot_df.empty:
        return []

    axis_by_group = {
        "low_gef": ax_low,
        "high_gef": ax_high,
    }
    panel_letter_by_group = {
        "low_gef": panel_letters[0],
        "high_gef": panel_letters[1],
    }
    label_records = []
    major_cities = (
        "Beijing",
        "Shanghai",
        "Guangzhou",
        "Shenzhen",
        "Chengdu",
        "Chongqing",
        "Tianjin",
        "Hangzhou",
        "Wuhan",
        "Nanjing",
        "Xian",
        "Zhengzhou",
    )
    label_offsets = {
        "Beijing": (8, 8),
        "Shanghai": (8, -12),
        "Guangzhou": (8, -12),
        "Shenzhen": (8, 8),
        "Chengdu": (-12, -14),
        "Chongqing": (-14, -14),
        "Tianjin": (8, 8),
        "Hangzhou": (8, 8),
        "Wuhan": (-12, 10),
        "Nanjing": (8, -12),
        "Xian": (-12, -14),
        "Zhengzhou": (8, 8),
    }
    fallback_offsets = (
        (8, 8), (8, -12), (-12, 10), (-12, -14),
        (14, 18), (14, -22), (-28, 18), (-28, -22),
        (24, 4), (24, -8), (-42, 4), (-42, -8),
        (4, 26), (4, -30),
    )
    pending_labels = []

    for city in major_cities:
        rows = plot_df[plot_df["region_name_en"] == city]
        if rows.empty:
            continue

        row = rows.iloc[0]
        x = float(row[feature_var])
        y = float(row[reduction_col])
        gef = float(row[group_var])
        axis_name = "low_gef" if gef <= threshold else "high_gef"
        ax = axis_by_group[axis_name]
        ax.scatter(
            [x],
            [y],
            s=48,
            marker="o",
            facecolors="white",
            edgecolors="#111111",
            linewidths=1.0,
            zorder=36,
            clip_on=False,
        )
        offset = label_offsets.get(city, (8, 8))
        annotation = ax.annotate(
            city,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#111111",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.75),
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="0.65",
                alpha=0.88,
            ),
            zorder=37,
        )
        pending_labels.append(
            {
                "annotation": annotation,
                "ax": ax,
                "offset": offset,
                "city": city,
                "axis_name": axis_name,
            }
        )
        display_point = ax.transData.transform((x, y))
        fig_point = fig.transFigure.inverted().transform(display_point)
        label_records.append(
            {
                "region_key": row.get("region_key"),
                "display_name": city,
                "axis_name": axis_name,
                "panel_letter": panel_letter_by_group[axis_name],
                "x_detour_ratio": _report_number(x),
                "y_reduction_pct": _report_number(y),
                "grid_emission_factor": _report_number(gef),
                "label_offset_points": {
                    "x": offset[0],
                    "y": offset[1],
                },
                "x_figure_fraction": _report_number(fig_point[0]),
                "y_figure_fraction": _report_number(fig_point[1]),
            }
        )

    if pending_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed_by_axis = {ax_low: [], ax_high: []}
        records_by_city = {record["display_name"]: record for record in label_records}
        for item in pending_labels:
            annotation = item["annotation"]
            ax = item["ax"]
            preferred = item["offset"]
            candidates = (preferred,) + tuple(
                offset for offset in fallback_offsets if offset != preferred
            )
            axis_bbox = ax.get_window_extent(renderer).expanded(1.02, 1.10)
            best = None
            best_score = None
            for candidate in candidates:
                annotation.set_position(candidate)
                fig.canvas.draw()
                bbox = annotation.get_window_extent(renderer).expanded(1.05, 1.16)
                overlap_count = sum(
                    bbox.overlaps(placed_bbox)
                    for placed_bbox in placed_by_axis[ax]
                )
                outside = (
                    max(axis_bbox.x0 - bbox.x0, 0.0)
                    + max(bbox.x1 - axis_bbox.x1, 0.0)
                    + max(axis_bbox.y0 - bbox.y0, 0.0)
                    + max(bbox.y1 - axis_bbox.y1, 0.0)
                )
                distance = (candidate[0] ** 2 + candidate[1] ** 2) ** 0.5
                score = overlap_count * 10000.0 + outside * 10.0 + distance
                if best_score is None or score < best_score:
                    best = (candidate, bbox)
                    best_score = score
                if overlap_count == 0 and outside == 0:
                    break

            final_offset, final_bbox = best
            annotation.set_position(final_offset)
            placed_by_axis[ax].append(final_bbox)
            record = records_by_city.get(item["city"])
            if record:
                record["label_offset_points"] = {
                    "x": final_offset[0],
                    "y": final_offset[1],
                }
                record["label_placement"] = "greedy_nonoverlap"
        fig.canvas.draw()

    return label_records


def _add_city_panel_connector_lines(
    fig,
    top_marker_records,
    city_panel_records,
    bottom_marker_records,
):
    """Connect selected city markers to their virtual city-panel anchors."""
    from matplotlib.lines import Line2D

    top_by_region = {
        record["region_key"]: record for record in top_marker_records
    }
    bottom_by_region = {
        record["region_key"]: record for record in bottom_marker_records
    }
    connector_records = []

    for panel_record in city_panel_records:
        context = panel_record["context"]
        region_key = context["region_key"]
        anchors = panel_record.get("panel_anchor_points")
        if anchors is None:
            continue

        city_connectors = {
            "region_key": region_key,
            "display_name": context["display_name"],
            "lines": [],
        }
        for line_type, marker_record, anchor_key in (
            ("top_to_city_upper_center", top_by_region.get(region_key), "top"),
            ("bottom_to_city_lower_center", bottom_by_region.get(region_key), "bottom"),
        ):
            if marker_record is None:
                continue

            start = (
                marker_record["x_figure_fraction"],
                marker_record["y_figure_fraction"],
            )
            end = anchors[anchor_key]
            gutter_y = (start[1] + end[1]) / 2.0
            route_x = [start[0], start[0], end[0], end[0]]
            route_y = [start[1], gutter_y, gutter_y, end[1]]
            fig.add_artist(
                Line2D(
                    route_x,
                    route_y,
                    color="#555555",
                    linewidth=0.45,
                    alpha=0.25,
                    linestyle="-",
                    transform=fig.transFigure,
                    zorder=2,
                    clip_on=False,
                )
            )
            city_connectors["lines"].append(
                {
                    "type": line_type,
                    "start": {
                        "x_figure_fraction": _report_number(start[0]),
                        "y_figure_fraction": _report_number(start[1]),
                    },
                    "end": {
                        "x_figure_fraction": _report_number(end[0]),
                        "y_figure_fraction": _report_number(end[1]),
                    },
                    "route": [
                        {
                            "x_figure_fraction": _report_number(x),
                            "y_figure_fraction": _report_number(y),
                        }
                        for x, y in zip(route_x, route_y)
                    ],
                }
            )

        connector_records.append(city_connectors)

    return connector_records


def _detour_reduction_scatter_report(
    df,
    reduction_col,
    axes,
    panel_letters=("f", "g"),
    compact=False,
):
    """Return exact plotted data for the bottom detour-reduction scatter panels."""
    import numpy as np
    from scipy import stats

    ax_low, cbar_ax, ax_high = axes
    feature_var = "detour_ratio"
    group_var = "grid_emission_factor"
    threshold = 0.3
    valid_mask = df[[feature_var, reduction_col, group_var]].notna().all(axis=1)
    plot_df = df.loc[valid_mask].copy()
    low_mask = plot_df[group_var] <= threshold
    masks = [low_mask, ~low_mask]
    labels = (f"Low GEF ($\\leq$ {threshold})", f"High GEF (> {threshold})")
    line_colors = ["#006837", "#A50F15"]
    axis_names = ["low_gef", "high_gef"]
    axis_objs = [ax_low, ax_high]

    x_data_min = plot_df[feature_var].min()
    x_data_max = plot_df[feature_var].max()
    y_data_min = plot_df[reduction_col].min()
    y_data_max = plot_df[reduction_col].max()
    x_pad = (x_data_max - x_data_min) * 0.03 or 0.1
    y_pad = (y_data_max - y_data_min) * 0.06 or 1.0
    x_global_min = x_data_min - x_pad
    x_global_max = x_data_max + x_pad

    panels = []
    for axis_name, ax, mask, label, color, panel_letter in zip(
        axis_names, axis_objs, masks, labels, line_colors, panel_letters
    ):
        panel_df = plot_df.loc[mask]
        points = []
        for _, row in panel_df.iterrows():
            points.append(
                {
                    "region_key": _report_number(row.get("region_key")),
                    "region_name_en": _report_number(row.get("region_name_en")),
                    "region_name": _report_number(row.get("region_name")),
                    "x_detour_ratio": _report_number(row[feature_var]),
                    "y_reduction_pct": _report_number(row[reduction_col]),
                    "grid_emission_factor": _report_number(row[group_var]),
                }
            )

        regression = None
        x = panel_df[feature_var].to_numpy(dtype=float)
        y = panel_df[reduction_col].to_numpy(dtype=float)
        if len(x) >= 3 and np.sum((x - np.mean(x)) ** 2) > 0:
            slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
            x_line = np.linspace(x_global_min, x_global_max, 100)
            y_line = slope * x_line + intercept
            n = len(x)
            se = np.sqrt(np.sum((y - (slope * x + intercept)) ** 2) / (n - 2))
            t_val = stats.t.ppf(0.975, n - 2)
            x_mean = np.mean(x)
            ci = t_val * se * np.sqrt(
                1.0 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)
            )
            regression = {
                "slope": _report_number(slope),
                "intercept": _report_number(intercept),
                "r_value": _report_number(r_value),
                "p_value": _report_number(p_value),
                "stderr": _report_number(stderr),
                "line_color": color,
                "line_points": [
                    {
                        "x": _report_number(xv),
                        "y": _report_number(yv),
                        "ci_low": _report_number(yv - civ),
                        "ci_high": _report_number(yv + civ),
                    }
                    for xv, yv, civ in zip(x_line, y_line, ci)
                ],
            }

        panels.append(
            {
                "panel_letter": panel_letter,
                "axis_name": axis_name,
                "label_text": label,
                "axes": _report_axis(ax),
                "panel_label": {
                    "text": panel_letter,
                    "x_axes_fraction": 0.0 if compact else -0.10,
                    "y_axes_fraction": 1.01 if compact else 1.02,
                },
                "group_label": {
                    "text": label,
                    "x_axes_fraction": 0.03,
                    "y_axes_fraction": 0.96,
                },
                "scatter_style": {
                    "size": 12 if compact else 24,
                    "alpha": 0.68,
                    "edgecolors": "none",
                    "zorder": 10,
                },
                "points": points,
                "regression": regression,
            }
        )

    return {
        "axes": {
            "colorbar": _report_axis(cbar_ax),
        },
        "feature_var": feature_var,
        "result_var": reduction_col,
        "group_var": group_var,
        "threshold": threshold,
        "panels": panels,
    }


def _write_city_gef_context_detour_report(
    save_path,
    figure_name,
    figure_size,
    top_report,
    city_reports,
    bottom_report,
    connector_lines,
):
    """Write JSON and Markdown reports for the city GEF context figure."""
    report_base = save_path.rsplit(".", 1)[0] + "_report"
    json_path = report_base + ".json"
    md_path = report_base + ".md"
    payload = {
        "figure_name": figure_name,
        "figure_path": save_path,
        "figure_size_inches": {
            "width": figure_size[0],
            "height": figure_size[1],
        },
        "coordinate_convention": (
            "All x/y values are in Matplotlib data coordinates for the stated axis; "
            "axis bbox values are figure-fraction coordinates."
        ),
        "panels": {
            "top_city_gef_scatter_components": top_report,
            "middle_city_context": city_reports,
            "bottom_detour_ratio_vs_reduction_by_gef": bottom_report,
        },
        "connector_lines": connector_lines,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)

    lines = [
        f"# {figure_name} Figure Report",
        "",
        f"- Figure path: `{save_path}`",
        f"- JSON data path: `{json_path}`",
        f"- Figure size: {figure_size[0]} x {figure_size[1]} inches",
        "- Coordinate convention: data coordinates for plotted elements; figure-fraction coordinates for axes.",
        f"- Connector line groups: {len(connector_lines)}",
        "",
        "## Top: city_gef_scatter_components",
        "",
        f"- GEF column: `{top_report.get('gef_column')}`",
        f"- Bin edges: {top_report.get('bin_edges')}",
        f"- Bar count: {len(top_report.get('bars', []))}",
        f"- Density points: {len(top_report.get('density_curve', []))}",
        f"- City rug marks: {len(top_report.get('city_rug_marks', []))}",
        f"- Selected city GEF markers: {len(top_report.get('selected_city_gef_markers', []))}",
        "",
        "Each bar entry in JSON records `x_center`, `width`, production+EOL segment bottom/height/top, use-phase segment bottom/height/top, and total label location.",
        "",
        "## Middle: selected_city_context",
        "",
    ]
    for city in city_reports:
        visible_bins = [
            item for item in city["distance_bins"]
            if item["visible_in_panel"]
        ]
        lines.extend(
            [
                f"### Panel {city['panel_letter']}: {city['display_name']}",
                "",
                f"- Region key: `{city['region_key']}`",
                f"- OD-derived detour ratio: {city['city_detour_ratio']}",
                f"- Valid/invalid paths: {city['valid_path_count']}/{city['invalid_path_count']}",
                f"- Visible distance-bin count: {len(visible_bins)}",
                f"- Power mix components: {', '.join(item['component'] for item in city['power_mix'])}",
                f"- Bar metric: `{city['bar_metric']}`",
                "- JSON records every distance bin with air-distance x center, "
                "plotted bar height, oracle drone share, reduction metrics, "
                "absolute scatter y value, and compact detour-ratio summary "
                "statistics.",
                "",
            ]
        )
    lines.extend(
        [
            "## Bottom: scatter_detour_ratio_vs_reduction_by_gef",
            "",
            f"- Feature variable: `{bottom_report['feature_var']}`",
            f"- Result variable: `{bottom_report['result_var']}`",
            f"- Group variable: `{bottom_report['group_var']}`",
            f"- Threshold: {bottom_report['threshold']}",
            f"- Selected city scatter markers: {len(bottom_report.get('selected_city_scatter_markers', []))}",
            f"- Major city labels: {len(bottom_report.get('major_city_labels', []))}",
        ]
    )
    for panel in bottom_report["panels"]:
        lines.extend(
            [
                f"- Panel {panel['panel_letter']} ({panel['axis_name']}): {len(panel['points'])} scatter points.",
            ]
        )
    lines.extend(
        [
            "",
            "The JSON file is the authoritative element-level record for writing: it includes every scatter point coordinate, every bar x/height/bottom/top, fitted line points, confidence intervals, axes limits, tick positions, and axes locations.",
            "",
        ]
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"City GEF context detour summary report saved to {md_path}")
    print(f"City GEF context detour summary report data saved to {json_path}")
    return md_path, json_path


def _render_city_context_group(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
    figure_name,
    contexts,
):
    """Render a grouped city context figure."""
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: i for i, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    if selected.empty:
        print(f"{figure_name} skipped — selected city rows missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=False,
        bin_width_km=0.2,
    )
    if built is None:
        return

    binned_by_city, labels = built
    available_contexts = [
        context for context in contexts
        if context["region_key"] in binned_by_city
    ]
    missing_names = [
        context["display_name"] for context in contexts
        if context["region_key"] not in binned_by_city
    ]
    if missing_names:
        print(
            f"{figure_name}: missing binned data for "
            f"{', '.join(missing_names)}"
        )
    if not available_contexts:
        print(f"{figure_name} skipped — no selected city binned data available")
        return

    import matplotlib.pyplot as plt

    from visualization._utils import _add_title_suffix

    fig = plt.figure(figsize=(24.0, 8.2))
    gs = fig.add_gridspec(1, 1)
    _plot_city_context_group_panels(
        fig=fig,
        subplot_spec=gs[0],
        binned_by_city=binned_by_city,
        labels=labels,
        contexts=available_contexts,
        panel_start_letter="a",
    )

    if plot_title:
        fig.suptitle(
            "Selected City Emission Reduction by Distance and Power Mix",
            fontsize=14,
        )

    save_path = os.path.join(out, f"{figure_name}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Dali Chengdu Tianjin Guangzhou city context saved to {save_path}")


def _render_city_gef_context_detour_summary(
    data_dir,
    out,
    for_latex,
    plot_title,
    df_enriched,
    city_df,
    exp_config,
):
    """Render GEF components, selected city contexts, and detour-reduction scatter."""
    figure_name = "city_gef_context_detour_summary"
    contexts = CITY_GEF_CONTEXT_DETOUR_CONTEXTS
    region_keys = [context["region_key"] for context in contexts]
    selected = city_df[city_df["region_key"].isin(region_keys)].copy()
    selected["_context_order"] = selected["region_key"].map(
        {key: i for i, key in enumerate(region_keys)}
    )
    selected = selected.sort_values("_context_order")
    if selected.empty:
        print(f"{figure_name} skipped — selected city rows missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=figure_name,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=True,
        bin_width_km=0.2,
    )
    if built is None:
        return

    binned_by_city, labels = built
    available_contexts = [
        context for context in contexts
        if context["region_key"] in binned_by_city
    ]
    missing_names = [
        context["display_name"] for context in contexts
        if context["region_key"] not in binned_by_city
    ]
    if missing_names:
        print(
            f"{figure_name}: missing binned data for "
            f"{', '.join(missing_names)}"
        )
    if not available_contexts:
        print(f"{figure_name} skipped — no selected city binned data available")
        return

    import matplotlib.pyplot as plt

    from visualization._utils import _add_title_suffix
    from visualization.analysis import (
        _plot_city_gef_scatter_components_axes,
        _plot_grouped_scatter_split_panels_axes,
        _weighted_reduction_column,
    )

    summary_df = df_enriched.copy()
    if "detour_ratio" not in summary_df.columns:
        summary_df["detour_ratio"] = float("nan")
    for context in available_contexts:
        region_key = context["region_key"]
        city_detour = binned_by_city[region_key].get("city_detour_ratio")
        if city_detour is None or pd.isna(city_detour):
            print(f"{figure_name}: {region_key} OD detour unavailable")
            continue
        city_mask = summary_df["region_key"] == region_key
        existing = pd.to_numeric(
            summary_df.loc[city_mask, "detour_ratio"], errors="coerce"
        ).dropna()
        if not existing.empty and abs(float(existing.iloc[0]) - city_detour) > 1e-6:
            print(
                f"{figure_name}: {region_key} OD detour {city_detour:.6f} "
                f"differs from stats {float(existing.iloc[0]):.6f}"
            )
        summary_df.loc[city_mask, "detour_ratio"] = city_detour

    reduction_col = _weighted_reduction_column(summary_df)

    figure_size = (7.2, 8.4)
    fig = plt.figure(figsize=figure_size)
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.20,
        left=0.10,
        right=0.90,
        bottom=0.055,
        top=0.985,
    )

    top_axes = _plot_city_gef_scatter_components_axes(
        fig=fig,
        subplot_spec=outer[0],
        df=summary_df,
        plot_title=plot_title,
        compact=True,
    )
    if top_axes is None:
        plt.close(fig)
        return
    ax_top, _ = top_axes
    top_pos = ax_top.get_position()
    fig.text(
        top_pos.x0,
        top_pos.y1 - 0.004,
        "a",
        fontsize=9,
        fontweight="bold",
        ha="right",
        va="bottom",
    )
    top_city_gef_markers = _add_city_gef_xaxis_markers(
        fig,
        top_axes,
        selected,
        available_contexts,
    )

    city_panel_records = _plot_compact_city_context_summary_panels(
        fig=fig,
        subplot_spec=outer[1],
        binned_by_city=binned_by_city,
        labels=labels,
        contexts=available_contexts,
        panel_start_letter="b",
    )

    bottom_start_ord = ord("b") + 2 * len(available_contexts)
    bottom_panel_letters = (
        chr(bottom_start_ord),
        chr(bottom_start_ord + 1),
    )
    bottom_axes = _plot_grouped_scatter_split_panels_axes(
        fig=fig,
        subplot_spec=outer[2],
        df=summary_df,
        feature_var="detour_ratio",
        result_var=reduction_col,
        group_var="grid_emission_factor",
        threshold=0.3,
        plot_title=False,
        panel_letters=bottom_panel_letters,
        compact=True,
        horizontal_colorbar=True,
    )
    if bottom_axes is None:
        plt.close(fig)
        return
    bottom_city_markers = _add_city_detour_reduction_markers(
        fig,
        bottom_axes,
        summary_df,
        available_contexts,
        reduction_col,
        panel_letters=bottom_panel_letters,
    )
    bottom_major_city_labels = []
    connector_lines = _add_city_panel_connector_lines(
        fig,
        top_city_gef_markers,
        city_panel_records,
        bottom_city_markers,
    )

    if plot_title:
        fig.suptitle(
            "City Grid Intensity, Context, and Detour-reduction Relationships",
            fontsize=9,
        )

    save_path = os.path.join(out, f"{figure_name}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.canvas.draw()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
    top_report = _city_gef_components_report(summary_df, top_axes)
    top_report["selected_city_gef_markers"] = top_city_gef_markers
    city_reports = _city_context_summary_report(
        binned_by_city,
        labels,
        city_panel_records,
        bar_metric="oracle_drone_share_pct",
    )
    bottom_report = _detour_reduction_scatter_report(
        summary_df,
        reduction_col,
        bottom_axes,
        panel_letters=bottom_panel_letters,
        compact=True,
    )
    bottom_report["selected_city_scatter_markers"] = bottom_city_markers
    bottom_report["major_city_labels"] = bottom_major_city_labels
    _write_city_gef_context_detour_report(
        save_path=save_path,
        figure_name=figure_name,
        figure_size=figure_size,
        top_report=top_report,
        city_reports=city_reports,
        bottom_report=bottom_report,
        connector_lines=connector_lines,
    )
    plt.close(fig)
    print(f"City GEF context detour summary saved to {save_path}")


def _render_shanghai_road_air_distance(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render the road/air distance figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Road air distance skipped — missing {od_path}")
        return

    from visualization.emission import plot_road_air_distance_distribution

    df = pd.read_csv(od_path)
    plot_road_air_distance_distribution(
        df,
        save_path=os.path.join(out, "shanghai_road_air_distance.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_road_detour(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render the road distance / detour ratio figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Road detour distribution skipped — missing {od_path}")
        return

    from visualization.emission import plot_road_detour_distribution

    df = pd.read_csv(od_path)
    plot_road_detour_distribution(
        df,
        save_path=os.path.join(out, "shanghai_road_detour.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_air_detour(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render the air distance / detour ratio figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Air detour distribution skipped — missing {od_path}")
        return

    from visualization.emission import plot_air_detour_distribution

    df = pd.read_csv(od_path)
    plot_air_detour_distribution(
        df,
        save_path=os.path.join(out, "shanghai_air_detour.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_emission_reduction(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render emission reduction vs drone ratio."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Emission reduction skipped — missing {od_path}")
        return

    from visualization.emission import plot_emission_reduction_vs_drone_ratio

    df = pd.read_csv(od_path)
    required = {"road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g", "annual_sales"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Emission reduction skipped — {od_path} missing columns {missing}")
        return

    plot_emission_reduction_vs_drone_ratio(
        df,
        save_path=os.path.join(out, "shanghai_emission_reduction.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_emission_by_distance(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render annual-emission-by-road-distance bars."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Emission by distance skipped — missing {od_path}")
        return

    from visualization.emission import plot_emission_by_road_distance_bar

    df = pd.read_csv(od_path)
    required = {"road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g", "annual_sales"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Emission by distance skipped — {od_path} missing columns {missing}")
        return

    plot_emission_by_road_distance_bar(
        df,
        save_path=os.path.join(out, "shanghai_emission_by_distance.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_reduction_detour_by_distance(data_dir, out, for_latex, plot_title):
    """Render Shanghai reduction % bars with detour-ratio boxplots per road-distance bin."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Reduction detour by distance skipped — missing {od_path}")
        return

    from visualization.emission import plot_reduction_pct_and_detour_by_distance

    df = pd.read_csv(od_path)
    required = {"road_distance_m", "distance_km", "drone_lifecycle_g",
                "ebike_lifecycle_g", "annual_sales"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Reduction detour by distance skipped — {od_path} missing columns {missing}")
        return

    plot_reduction_pct_and_detour_by_distance(
        df,
        save_path=os.path.join(out, "shanghai_reduction_detour_by_distance.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_beijing_median_trip_emission_process(data_dir, out, for_latex, plot_title):
    """Load Beijing OD pairs and render the median-trip emission process figure."""
    region_key = "beijing"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Beijing median trip emission process skipped — missing {od_path}")
        return

    from visualization.emission import plot_median_trip_emission_process

    df = pd.read_csv(od_path)
    required = {
        "distance_km",
        "road_distance_m",
        "drone_emission_kg",
        "ebike_emission_kg",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Beijing median trip emission process skipped — {od_path} missing columns {missing}")
        return

    plot_median_trip_emission_process(
        df,
        save_path=os.path.join(out, "beijing_median_trip_emission_process.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_shanghai_air_road_joint(data_dir, out, for_latex, plot_title):
    """Load Shanghai OD pairs and render the air vs road joint distribution figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")

    if not os.path.exists(od_path):
        print(f"Air road joint skipped — missing {od_path}")
        return

    from visualization.emission import plot_air_road_joint

    df = pd.read_csv(od_path)
    plot_air_road_joint(
        df,
        save_path=os.path.join(out, "shanghai_air_road_joint.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        max_air_km=5.0,
    )


def _render_contribution_decomposition(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
):
    """Render the sampled contribution-decomposition paper figure."""
    from visualization.emission import (
        plot_contribution_decomposition,
        prepare_contribution_decomposition_data,
    )

    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    named_df = None
    if os.path.exists(named_path):
        named_df = pd.read_csv(named_path)
    else:
        print(
            "contribution_decomposition: missing "
            f"{named_path}; thresholds will fall back to sampled sweeps"
        )

    prepared = prepare_contribution_decomposition_data(
        city_df=city_df,
        data_dir=data_dir,
        named_df=named_df,
        exp_config=exp_config,
    )
    if prepared is None:
        return

    plot_contribution_decomposition(
        prepared,
        save_path=os.path.join(out, "contribution_decomposition.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_contribution_decomposition_city_b(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
):
    """Render the selected-city horizontal contribution-decomposition figure."""
    from visualization.emission import (
        plot_contribution_decomposition_city_b,
        prepare_contribution_decomposition_data,
    )

    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    named_df = None
    if os.path.exists(named_path):
        named_df = pd.read_csv(named_path)
    else:
        print(
            "contribution_decomposition_city_b: missing "
            f"{named_path}; thresholds will fall back to sampled sweeps"
        )

    prepared = prepare_contribution_decomposition_data(
        city_df=city_df,
        data_dir=data_dir,
        named_df=named_df,
        exp_config=exp_config,
    )
    if prepared is None:
        return

    plot_contribution_decomposition_city_b(
        prepared,
        save_path=os.path.join(out, "contribution_decomposition_city_b.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_break_even_contribution_decomposition_marching_cubes_de(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
):
    """Render the extracted contribution panels from the marching-cubes figure."""
    from visualization.emission import (
        plot_break_even_contribution_decomposition_marching_cubes_de,
    )

    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    named_df = None
    if os.path.exists(named_path):
        named_df = pd.read_csv(named_path)
    else:
        print(
            "break_even_contribution_decomposition_marching_cubes_de: missing "
            f"{named_path}; thresholds will fall back to sampled sweeps"
        )

    prepared = _prepare_contribution_decomposition_full(
        city_df=city_df,
        data_dir=data_dir,
        named_df=named_df,
        exp_config=exp_config,
    )
    if prepared is None:
        return

    plot_break_even_contribution_decomposition_marching_cubes_de(
        prepared,
        save_path=os.path.join(
            out,
            "break_even_contribution_decomposition_marching_cubes_de.png",
        ),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _prepare_break_even_phase_diagram(
    data_dir,
    exp_config,
):
    """Prepare lifecycle break-even / phase-diagram data with cache."""
    from visualization.emission import (
        CONTRIBUTION_SAMPLE_LIMIT,
        CONTRIBUTION_SAMPLE_SEED,
        prepare_break_even_phase_diagram_data,
    )

    city_df = None
    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    if os.path.exists(stats_path):
        city_df = pd.read_csv(stats_path)
        if "region_level" in city_df.columns:
            city_df = city_df[city_df["region_level"] == "city"].copy()

        has_gef = "gef" in city_df.columns and city_df["gef"].notna().any()
        if not has_gef:
            gef_path = os.path.join(data_dir, "gef_china.csv")
            if "province" in city_df.columns and os.path.exists(gef_path):
                gef_df = pd.read_csv(gef_path, encoding="gbk")
                gef_df.columns = ["province", "gef"]
                city_df = city_df.merge(gef_df, on="province", how="left")
            else:
                print(
                    "break_even_phase_diagram: city GEF data unavailable; "
                    "panel c will show a placeholder"
                )
    else:
        print(
            f"break_even_phase_diagram: missing {stats_path}; "
            "panel c will show a placeholder"
        )

    from utils.config import get_cache_dir

    figure_name = "break_even_phase_diagram"
    gef_path = os.path.join(data_dir, "gef_china.csv")
    cache_path = os.path.join(
        get_cache_dir("visualization", "paper"),
        f"{figure_name}_prepared.pkl",
    )
    cache_meta = {
        "version": BREAK_EVEN_PHASE_CACHE_VERSION,
        "figure_name": figure_name,
        "stats_file": _optional_file_cache_signature(stats_path),
        "gef_file": _optional_file_cache_signature(gef_path),
        "kappa": _get_ebike_kappa(exp_config),
        "max_sample_rows": CONTRIBUTION_SAMPLE_LIMIT,
        "sample_seed": CONTRIBUTION_SAMPLE_SEED,
    }
    prepared = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("meta") == cache_meta:
                print(f"{figure_name}: loading prepared data cache {cache_path}")
                prepared = cache["prepared"]
            else:
                print(f"{figure_name}: prepared data cache is stale; rebuilding")
        except Exception as exc:
            print(f"{figure_name}: failed to load prepared cache ({exc}); rebuilding")

    if prepared is None:
        prepared = prepare_break_even_phase_diagram_data(
            data_dir=data_dir,
            city_df=city_df,
            exp_config=exp_config,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"meta": cache_meta, "prepared": prepared},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"{figure_name}: wrote prepared data cache {cache_path}")

    return prepared


def _render_break_even_phase_diagram(
    data_dir,
    out,
    for_latex,
    plot_title,
    exp_config,
):
    """Render the lifecycle break-even / phase-diagram paper figure."""
    from visualization.emission import plot_break_even_phase_diagram

    prepared = _prepare_break_even_phase_diagram(
        data_dir=data_dir,
        exp_config=exp_config,
    )

    plot_break_even_phase_diagram(
        prepared,
        save_path=os.path.join(out, "break_even_phase_diagram.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _prepare_contribution_decomposition_full(
    data_dir,
    city_df,
    named_df,
    exp_config,
):
    """Prepare full-data contribution decomposition with cache."""
    from visualization.emission import (
        CONTRIBUTION_SAMPLE_SEED,
        _resolve_contribution_od_path,
        prepare_contribution_decomposition_data,
    )
    from utils.config import get_cache_dir

    stats_path = os.path.join(data_dir, "all_region_stats.csv")
    gef_path = os.path.join(data_dir, "gef_china.csv")
    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    city_rows = city_df.copy()
    required_cols = {"region_key", "total_ebike_lifecycle", "gef"}
    od_files = {}
    if required_cols.issubset(city_rows.columns):
        for col in ("total_ebike_lifecycle", "gef"):
            city_rows[col] = pd.to_numeric(city_rows[col], errors="coerce")
        city_rows = city_rows[
            city_rows["total_ebike_lifecycle"].notna()
            & (city_rows["total_ebike_lifecycle"] > 0)
            & city_rows["gef"].notna()
        ].copy()
        for region_key in sorted(city_rows["region_key"].dropna().astype(str).unique()):
            od_path = _resolve_contribution_od_path(data_dir, region_key)
            if od_path:
                od_files[region_key] = _file_cache_signature(od_path)

    figure_name = "break_even_contribution_decomposition"
    cache_path = os.path.join(
        get_cache_dir("visualization", "paper"),
        f"{figure_name}_contribution_full_prepared.pkl",
    )
    cache_meta = {
        "version": CONTRIBUTION_DECOMPOSITION_FULL_CACHE_VERSION,
        "figure_name": figure_name,
        "stats_file": _optional_file_cache_signature(stats_path),
        "gef_file": _optional_file_cache_signature(gef_path),
        "named_file": _optional_file_cache_signature(named_path),
        "od_files": od_files,
        "kappa": _get_ebike_kappa(exp_config),
        "gef_fallback": 0.5600,
        "max_sample_rows": None,
        "sample_seed": CONTRIBUTION_SAMPLE_SEED,
    }
    prepared = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("meta") == cache_meta:
                print(f"{figure_name}: loading full contribution cache {cache_path}")
                prepared = cache["prepared"]
            else:
                print(f"{figure_name}: full contribution cache is stale; rebuilding")
        except Exception as exc:
            print(
                f"{figure_name}: failed to load full contribution cache "
                f"({exc}); rebuilding"
            )

    if prepared is None:
        prepared = prepare_contribution_decomposition_data(
            city_df=city_df,
            data_dir=data_dir,
            named_df=named_df,
            exp_config=exp_config,
            max_sample_rows=None,
        )
        if prepared is None:
            return None
        with open(cache_path, "wb") as f:
            pickle.dump(
                {"meta": cache_meta, "prepared": prepared},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"{figure_name}: wrote full contribution cache {cache_path}")

    return prepared


def _render_break_even_contribution_decomposition(
    data_dir,
    out,
    for_latex,
    plot_title,
    city_df,
    exp_config,
    render_base=True,
    render_marching_cubes=False,
):
    """Render the stacked break-even and contribution-decomposition figure."""
    from visualization.emission import (
        plot_break_even_contribution_decomposition,
        plot_break_even_contribution_decomposition_marching_cubes,
    )

    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
    named_df = None
    if os.path.exists(named_path):
        named_df = pd.read_csv(named_path)
    else:
        print(
            "break_even_contribution_decomposition: missing "
            f"{named_path}; thresholds will fall back to sampled sweeps"
        )

    break_even_prepared = _prepare_break_even_phase_diagram(
        data_dir=data_dir,
        exp_config=exp_config,
    )
    contribution_prepared = _prepare_contribution_decomposition_full(
        city_df=city_df,
        data_dir=data_dir,
        named_df=named_df,
        exp_config=exp_config,
    )
    if break_even_prepared is None or contribution_prepared is None:
        return

    if render_base:
        plot_break_even_contribution_decomposition(
            break_even_prepared,
            contribution_prepared,
            save_path=os.path.join(out, "break_even_contribution_decomposition.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )
    if render_marching_cubes:
        plot_break_even_contribution_decomposition_marching_cubes(
            break_even_prepared,
            contribution_prepared,
            save_path=os.path.join(
                out,
                "break_even_contribution_decomposition_marching_cubes.png",
            ),
            for_latex=for_latex,
            plot_title=plot_title,
        )


def _render_distance_comparison_grid(data_dir, out, for_latex, plot_title, extreme):
    """Load Shanghai roadnet data and render a 3x3 distance-comparison grid for one tail."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")
    graphml_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_roadnet.graphml")

    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        print(f"Distance comparison grid ({extreme}) skipped — missing data for {region_key}")
        return

    from roadnet.sparse_graph import SparseGraph
    from roadnet.tile_router import TileRouter
    from visualization.roadnet import plot_distance_comparison_grid

    od_df = pd.read_csv(od_path)
    graph = SparseGraph.from_graphml(graphml_path)
    router = TileRouter(graph)

    # Debug: restrict low-extreme to the 4.5-5 km air-distance band so we see
    # impossible detours at a consistent scale.
    extra = {"air_km_min": 4.5, "air_km_max": 5.0} if extreme == "low" else {}

    plot_distance_comparison_grid(
        od_df=od_df,
        graph=graph,
        router=router,
        save_path=os.path.join(out, f"distance_comparison_{extreme}_3x3.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        n_rows=3,
        n_cols=3,
        extreme=extreme,
        **extra,
    )


def _render_distance_comparison(
    data_dir,
    out,
    for_latex,
    plot_title,
    plot_fn,
    graph_cache=None,
    router_cache=None,
):
    """Load Shanghai roadnet data and render the distance comparison figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")
    graphml_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_roadnet.graphml")

    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        print(f"Distance comparison skipped — missing data for {region_key}")
        return

    od_df = pd.read_csv(od_path)
    graph = _get_cached_sparse_graph(graphml_path, graph_cache=graph_cache)
    router = _get_cached_tile_router(
        graphml_path,
        graph,
        router_cache=router_cache,
    )

    plot_fn(
        od_df=od_df,
        graph=graph,
        router=router,
        save_path=os.path.join(out, "distance_comparison.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        figsize=(7.0, 13.0),
        n_samples=3,
        n_cols=1,
        show_stats_box=False,
        show_basemap=False,
        target_aspect=1.6,
    )


def _render_distance_comparison_single_panel(
    data_dir,
    out,
    for_latex,
    plot_title,
    graph_cache=None,
    router_cache=None,
):
    """Render a compact median-trip distance comparison for Shanghai."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(
        data_dir,
        region_level,
        region_key,
        f"{region_key}_od_pairs.csv",
    )
    graphml_path = os.path.join(
        data_dir,
        region_level,
        region_key,
        f"{region_key}_roadnet.graphml",
    )

    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        print(
            "Single-panel distance comparison skipped — "
            f"missing data for {region_key}"
        )
        return

    from visualization.roadnet import plot_distance_comparison_single_panel

    od_df = pd.read_csv(od_path)
    graph = _get_cached_sparse_graph(graphml_path, graph_cache=graph_cache)
    router = _get_cached_tile_router(
        graphml_path,
        graph,
        router_cache=router_cache,
    )
    plot_distance_comparison_single_panel(
        od_df=od_df,
        graph=graph,
        router=router,
        save_path=os.path.join(out, "distance_comparison_single_panel.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_tanner_impedance_fit_panel_a(data_dir, out, for_latex, plot_title):
    """Render the observed Eleme delivery distance histogram."""
    order_dir = os.path.join(
        data_dir,
        "raw",
        "即时配送数据",
        "疫情期间北京饿了么",
        "eleme_round1_train_20200313",
        "order",
    )
    if not os.path.isdir(order_dir):
        print(f"tanner_impedance_fit_panel_a skipped — missing {order_dir}")
        return

    from calibration import cali_elm
    from visualization.calibration import plot_tanner_distance_histogram

    cali_elm.elm_data_dir = order_dir + os.sep
    distances = cali_elm.load_all_eleme_distances(min_orders_per_shop=50)
    plot_tanner_distance_histogram(
        distances,
        save_path=os.path.join(out, "tanner_impedance_fit_panel_a.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _render_tanner_impedance_fit(data_dir, out, for_latex, plot_title):
    """Render the Eleme distance histogram with fitted Tanner impedance."""
    order_dir = os.path.join(
        data_dir,
        "raw",
        "即时配送数据",
        "疫情期间北京饿了么",
        "eleme_round1_train_20200313",
        "order",
    )
    if not os.path.isdir(order_dir):
        print(f"tanner_impedance_fit skipped — missing {order_dir}")
        return

    from calibration import cali_elm
    from visualization.calibration import plot_tanner_impedance_fit

    cali_elm.elm_data_dir = order_dir + os.sep
    distances = cali_elm.load_all_eleme_distances(min_orders_per_shop=50)
    plot_tanner_impedance_fit(
        distances,
        save_path=os.path.join(out, "tanner_impedance_fit.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _alpha_calibration_csv_path():
    """Return the tracked α calibration sweep CSV path."""
    return os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "calibration",
            "alpha_calibration_5km.csv",
        )
    )


def _render_alpha_calibration_5km(out, for_latex, plot_title):
    """Render the paper α calibration panel from the tracked sweep CSV."""
    csv_path = _alpha_calibration_csv_path()
    if not os.path.exists(csv_path):
        print(f"alpha_calibration_5km skipped — missing {csv_path}")
        return

    from visualization.calibration import plot_alpha_calibration_5km

    df = pd.read_csv(csv_path)
    plot_alpha_calibration_5km(
        df,
        save_path=os.path.join(out, "alpha_calibration_5km.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )


def render_all(
    data_dir: str,
    figures_dir: str,
    for_latex: bool = False,
    plot_title: bool = False,
    only: set = None,
    exp_config: dict = None,
    profile: bool = False,
):
    framework_out = os.path.join(figures_dir, "paper")
    out = os.path.join(framework_out, "subfigure")
    os.makedirs(framework_out, exist_ok=True)
    os.makedirs(out, exist_ok=True)

    def _want(name):
        return only is None or name in only

    figure1_framework_requested = _want("figure1_framework")
    figure2_framework_requested = _want("figure2_framework")
    figure3_framework_requested = _want("figure3_framework")
    figure4_framework_requested = _want("figure4_framework")
    graph_cache = {}
    router_cache = {}

    def _figure1_needs_vector_source(name):
        if not figure1_framework_requested:
            return False
        source_pdf = _paper_pdf_path(framework_out, name, plot_title=plot_title)
        if not os.path.exists(source_pdf):
            return True
        cache_paths = [source_pdf]
        if name in FIGURE1_FRAMEWORK_REPORT_DEPENDENCIES:
            source_report = _paper_report_path(
                framework_out, name, plot_title=plot_title
            )
            if not os.path.exists(source_report):
                return True
            cache_paths.append(source_report)
        cache_mtime = min(os.path.getmtime(path) for path in cache_paths)
        viz_dir = os.path.dirname(os.path.abspath(__file__))
        module_names = FIGURE1_FRAMEWORK_SOURCE_MODULES.get(name, ())
        module_paths = [os.path.join(viz_dir, module) for module in module_names]
        return any(
            os.path.exists(path) and os.path.getmtime(path) > cache_mtime
            for path in module_paths
        )

    def _figure2_needs_vector_source(name):
        if not figure2_framework_requested:
            return False
        source_pdf = _paper_pdf_path(framework_out, name, plot_title=plot_title)
        if not os.path.exists(source_pdf):
            return True
        cache_mtime = os.path.getmtime(source_pdf)
        viz_dir = os.path.dirname(os.path.abspath(__file__))
        module_names = FIGURE2_FRAMEWORK_SOURCE_MODULES.get(name, ())
        module_paths = [os.path.join(viz_dir, module) for module in module_names]
        return any(
            os.path.exists(path) and os.path.getmtime(path) > cache_mtime
            for path in module_paths
        )

    def _figure3_needs_vector_source(name):
        if not figure3_framework_requested:
            return False
        source_pdf = _paper_pdf_path(framework_out, name, plot_title=plot_title)
        if not os.path.exists(source_pdf):
            return True
        cache_mtime = os.path.getmtime(source_pdf)
        viz_dir = os.path.dirname(os.path.abspath(__file__))
        module_names = FIGURE3_FRAMEWORK_SOURCE_MODULES.get(name, ())
        module_paths = [os.path.join(viz_dir, module) for module in module_names]
        return any(
            os.path.exists(path) and os.path.getmtime(path) > cache_mtime
            for path in module_paths
        )

    def _figure4_needs_vector_source(name):
        if not figure4_framework_requested:
            return False
        source_pdf = _paper_pdf_path(framework_out, name, plot_title=plot_title)
        if not os.path.exists(source_pdf):
            return True
        cache_mtime = os.path.getmtime(source_pdf)
        viz_dir = os.path.dirname(os.path.abspath(__file__))
        module_names = FIGURE4_FRAMEWORK_SOURCE_MODULES.get(name, ())
        module_paths = [os.path.join(viz_dir, module) for module in module_names]
        return any(
            os.path.exists(path) and os.path.getmtime(path) > cache_mtime
            for path in module_paths
        )

    # --- Calibration figures ---
    tanner_impedance_fit_requested = _want("tanner_impedance_fit")
    tanner_impedance_fit_pdf_needed = _figure4_needs_vector_source(
        "tanner_impedance_fit"
    )
    if tanner_impedance_fit_requested:
        _render_tanner_impedance_fit(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )
    if tanner_impedance_fit_pdf_needed:
        _render_tanner_impedance_fit(
            data_dir=data_dir,
            out=framework_out,
            for_latex=True,
            plot_title=plot_title,
        )

    tanner_impedance_fit_panel_a_requested = _want("tanner_impedance_fit_panel_a")
    tanner_impedance_fit_panel_a_pdf_needed = _figure2_needs_vector_source(
        "tanner_impedance_fit_panel_a"
    )
    if tanner_impedance_fit_panel_a_requested:
        _render_tanner_impedance_fit_panel_a(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )
    if tanner_impedance_fit_panel_a_pdf_needed:
        _render_tanner_impedance_fit_panel_a(
            data_dir=data_dir,
            out=framework_out,
            for_latex=True,
            plot_title=plot_title,
        )

    alpha_calibration_5km_requested = _want("alpha_calibration_5km")
    alpha_calibration_5km_pdf_needed = _figure4_needs_vector_source(
        "alpha_calibration_5km"
    )
    if alpha_calibration_5km_requested:
        with _profile_block(profile, "alpha_calibration_5km"):
            _render_alpha_calibration_5km(
                out=out,
                for_latex=for_latex,
                plot_title=plot_title,
            )
    if alpha_calibration_5km_pdf_needed:
        with _profile_block(profile, "alpha_calibration_5km source"):
            _render_alpha_calibration_5km(
                out=framework_out,
                for_latex=True,
                plot_title=plot_title,
            )

    # --- Concept figures (no data dependency) ---
    concept_figures = {
        "detour_ratio_concept",
        "emission_model_flowchart",
        "production_phase_per_km",
        "end_of_life_phase_per_km",
        "production_detour_eol_panels",
        "detour_and_lifecycle",
        "detour_lifecycle_distance",
    }
    if only is None or only & concept_figures:
        from visualization.concepts import (
            plot_detour_and_lifecycle,
            plot_detour_lifecycle_distance,
            plot_detour_ratio_concept,
            plot_emission_model_flowchart,
            plot_production_detour_eol_panels,
        )
        from visualization.emission import (
            plot_end_of_life_phase_per_km,
            plot_production_phase_per_km,
        )

    if _want("detour_ratio_concept"):
        plot_detour_ratio_concept(
            save_path=os.path.join(out, "detour_ratio_concept.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("emission_model_flowchart"):
        plot_emission_model_flowchart(
            save_path=os.path.join(out, "emission_model_flowchart.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("production_phase_per_km"):
        plot_production_phase_per_km(
            save_path=os.path.join(out, "production_phase_per_km.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("end_of_life_phase_per_km"):
        plot_end_of_life_phase_per_km(
            save_path=os.path.join(out, "end_of_life_phase_per_km.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("production_detour_eol_panels"):
        plot_production_detour_eol_panels(
            save_path=os.path.join(out, "production_detour_eol_panels.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("detour_and_lifecycle"):
        od_path = os.path.join(data_dir, "city", "shanghai", "shanghai_od_pairs.csv")
        od_df = pd.read_csv(od_path) if os.path.exists(od_path) else None
        if od_df is None:
            print("detour_and_lifecycle: Shanghai OD missing, rendering without overlay")
        plot_detour_and_lifecycle(
            od_df=od_df,
            save_path=os.path.join(out, "detour_and_lifecycle.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("detour_lifecycle_distance"):
        od_path = os.path.join(data_dir, "city", "shanghai", "shanghai_od_pairs.csv")
        graphml_path = os.path.join(data_dir, "city", "shanghai", "shanghai_roadnet.graphml")
        od_df = pd.read_csv(od_path) if os.path.exists(od_path) else None
        graph, router = None, None
        if os.path.exists(graphml_path) and od_df is not None:
            from roadnet.sparse_graph import SparseGraph
            from roadnet.tile_router import TileRouter
            graph = SparseGraph.from_graphml(graphml_path)
            router = TileRouter(graph)
        else:
            print("detour_lifecycle_distance: Shanghai road data missing, panel c will show placeholder")
        plot_detour_lifecycle_distance(
            od_df=od_df,
            graph=graph,
            router=router,
            save_path=os.path.join(out, "detour_lifecycle_distance.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("lifecycle_distance_detour_boxplot"):
        od_path = os.path.join(data_dir, "city", "shanghai", "shanghai_od_pairs.csv")
        if not os.path.exists(od_path):
            print(f"lifecycle_distance_detour_boxplot skipped — missing {od_path}")
        else:
            from visualization.emission import plot_lifecycle_distance_detour_boxplot

            od_df = pd.read_csv(od_path)
            required = {"distance_km", "road_distance_m"}
            if not required.issubset(od_df.columns):
                missing = required - set(od_df.columns)
                print(f"lifecycle_distance_detour_boxplot skipped — {od_path} missing columns {missing}")
            else:
                plot_lifecycle_distance_detour_boxplot(
                    od_df,
                    save_path=os.path.join(out, "lifecycle_distance_detour_boxplot.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

    if _want("lifecycle_distance_route_factor"):
        od_path = os.path.join(data_dir, "city", "shanghai", "shanghai_od_pairs.csv")
        if not os.path.exists(od_path):
            print(f"lifecycle_distance_route_factor skipped — missing {od_path}")
        else:
            od_df = pd.read_csv(od_path)
            required = {"distance_km", "road_distance_m", "annual_sales"}
            if not required.issubset(od_df.columns):
                missing = required - set(od_df.columns)
                print(f"lifecycle_distance_route_factor skipped — {od_path} missing columns {missing}")
            else:
                from visualization.emission import plot_lifecycle_distance_route_factor

                plot_lifecycle_distance_route_factor(
                    od_df,
                    save_path=os.path.join(out, "lifecycle_distance_route_factor.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

    if only is not None and _want("break_even_phase_diagram"):
        _render_break_even_phase_diagram(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            exp_config=exp_config,
        )

    # --- Shanghai figures (independent of all_region_stats) ---
    if _want("shanghai_merchant_monthly_sales"):
        _render_shanghai_merchant_monthly_sales(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    shanghai_heatmap_requested = _want("shanghai_heatmap")
    shanghai_heatmap_pdf_needed = _figure1_needs_vector_source("shanghai_heatmap")
    if shanghai_heatmap_requested:
        _render_shanghai_heatmap(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            compact=True,
        )
    if shanghai_heatmap_pdf_needed:
        _render_shanghai_heatmap(
            data_dir=data_dir,
            out=framework_out,
            for_latex=True,
            plot_title=plot_title,
            compact=False,
        )

    shanghai_building_height_merchant_requested = _want(
        "shanghai_building_height_merchant"
    )
    shanghai_building_height_merchant_pdf_needed = _figure1_needs_vector_source(
        "shanghai_building_height_merchant"
    )
    if shanghai_building_height_merchant_requested:
        _render_shanghai_building_height_merchant(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            graph_cache=graph_cache,
        )
    if shanghai_building_height_merchant_pdf_needed:
        _render_shanghai_building_height_merchant(
            data_dir=data_dir,
            out=framework_out,
            for_latex=True,
            plot_title=plot_title,
            graph_cache=graph_cache,
        )

    if _want("shanghai_building_footprint_heatmap"):
        _render_shanghai_building_footprint_heatmap(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("beijing_road_network_subfigure"):
        _render_beijing_road_network_subfigure(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            graph_cache=graph_cache,
        )

    if _want("beijing_road_network_candidate_grid"):
        _render_beijing_road_network_candidate_grid(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            graph_cache=graph_cache,
        )

    if _want("beijing_road_network_subfigure_option3"):
        _render_beijing_road_network_subfigure_option3(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_road_air_distance"):
        _render_shanghai_road_air_distance(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_road_detour"):
        _render_shanghai_road_detour(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_air_detour"):
        _render_shanghai_air_detour(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_emission_reduction"):
        _render_shanghai_emission_reduction(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_emission_by_distance"):
        _render_shanghai_emission_by_distance(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_reduction_detour_by_distance"):
        _render_shanghai_reduction_detour_by_distance(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("beijing_median_trip_emission_process"):
        _render_beijing_median_trip_emission_process(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("shanghai_air_road_joint"):
        _render_shanghai_air_road_joint(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    distance_comparison_requested = _want("distance_comparison")
    distance_comparison_pdf_needed = _figure1_needs_vector_source("distance_comparison")
    if distance_comparison_requested:
        from visualization.roadnet import plot_distance_comparison

        _render_distance_comparison(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            plot_fn=plot_distance_comparison,
            graph_cache=graph_cache,
            router_cache=router_cache,
        )
    if distance_comparison_pdf_needed:
        from visualization.roadnet import plot_distance_comparison

        _render_distance_comparison(
            data_dir=data_dir,
            out=framework_out,
            for_latex=True,
            plot_title=plot_title,
            plot_fn=plot_distance_comparison,
            graph_cache=graph_cache,
            router_cache=router_cache,
        )

    if _want("distance_comparison_single_panel"):
        _render_distance_comparison_single_panel(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            graph_cache=graph_cache,
            router_cache=router_cache,
        )

    if _want("distance_comparison_high_3x3"):
        _render_distance_comparison_grid(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            extreme="high",
        )

    if _want("distance_comparison_low_3x3"):
        _render_distance_comparison_grid(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            extreme="low",
        )

    # --- Cross-region stats figures ---
    province_gef_figures = {"province_gef_china_map"}
    if only is None or only & province_gef_figures:
        from visualization.regions import plot_province_gef_china_map

        gef_path = (exp_config or {}).get("gef_path") or os.path.join(
            data_dir,
            "gef_china.csv",
        )
        plot_province_gef_china_map(
            gef_path=gef_path,
            save_path=os.path.join(out, "province_gef_china_map.png"),
            figsize=(1.25, 0.95),
            for_latex=for_latex,
            plot_title=plot_title,
            compact=True,
            south_china_inset_bounds=(0.80, 0.04, 0.16, 0.28),
        )

    future_gef_figures = ("regions_map_and_gef_scatter_2030", "regions_map_and_gef_scatter_2060")
    gef_figures = {
        "city_gef_distribution",
        "city_gef_scatter_components",
        "city_population_detour_oracle_reduction_bubble",
        "city_gef_context_detour_summary",
        "break_even_contribution_decomposition",
        "break_even_contribution_decomposition_marching_cubes",
        "break_even_contribution_decomposition_marching_cubes_de",
        "contribution_decomposition",
        "contribution_decomposition_city_b",
        "scatter_detour_ratio_vs_reduction_by_gef",
        "regions_map_and_gef_scatter",
        *future_gef_figures,
    }
    stats_figures = {
        "province_orders_emission",
        "province_orders_both_emissions",
        "province_total_ebike_lifecycle",
        "city_ebike_rank_top10",
        "city_ebike_rank_bottom10",
        "city_oracle_reduction_rank_top10",
        "city_reduction_rank_top10",
        "city_reduction_rank_all_macro_region",
        "regions_map",
        "regions_map_oracle_reduction_rank",
        "regions_map_oracle_reduction_rank_map_only",
        "regions_map_optimal_drone_share",
        "regions_map_city_rank_and_drone_share",
        "regions_map_city_province_ranks",
        "regions_map_city_rank_inset",
        "figure1_framework",
        "figure2_framework",
        "figure3_framework",
        "regions_map_kappa_panels",
        "regions_map_pathway_scenario_maps",
        "regions_map_pathway_scenarios",
        *CITY_CONTEXT_FIGURES,
        *CITY_CONTEXT_GROUP_FIGURES,
    } | gef_figures | TOP6_REDUCTION_DETOUR_FIGURES
    analysis_figures = {
        "province_orders_emission",
        "province_orders_both_emissions",
        "province_total_ebike_lifecycle",
        "city_ebike_rank_top10",
        "city_ebike_rank_bottom10",
        "city_oracle_reduction_rank_top10",
        "city_reduction_rank_top10",
        "city_reduction_rank_all_macro_region",
        "regions_map_oracle_reduction_rank",
        "regions_map_oracle_reduction_rank_map_only",
        "regions_map_optimal_drone_share",
        "regions_map_city_rank_and_drone_share",
        "regions_map_city_province_ranks",
        "regions_map_city_rank_inset",
        "regions_map_kappa_panels",
        "regions_map_pathway_scenario_maps",
        "regions_map_pathway_scenarios",
        *gef_figures,
    }
    analysis_source_needed = (
        _figure1_needs_vector_source("regions_map_oracle_reduction_rank")
        or _figure2_needs_vector_source("city_oracle_reduction_rank_top10")
        or _figure2_needs_vector_source("regions_map_oracle_reduction_rank_map_only")
        or _figure3_needs_vector_source("city_population_detour_oracle_reduction_bubble")
        or _figure3_needs_vector_source("city_reduction_rank_all_macro_region")
    )
    if only is None or only & stats_figures:
        stats_path = os.path.join(data_dir, "all_region_stats.csv")
        if not os.path.exists(stats_path):
            print(f"Missing {stats_path}; run consolidate_stats first — skipping stats figures")
        else:
            df = pd.read_csv(stats_path)
            needs_analysis_import = (
                only is None or bool(only & analysis_figures) or analysis_source_needed
            )
            if needs_analysis_import:
                from visualization.analysis import (
                    _weighted_reduction_column,
                    plot_city_ebike_rank_bottom10,
                    plot_city_ebike_rank_top10,
                    plot_city_gef_distribution,
                    plot_city_gef_scatter_components,
                    plot_city_oracle_reduction_rank_top10,
                    plot_city_population_detour_oracle_reduction_bubble,
                    plot_city_reduction_rank_all_macro_region,
                    plot_city_reduction_rank_top10,
                    plot_grouped_scatter_split_panels,
                    plot_province_orders_both_emissions,
                    plot_province_orders_emission,
                    plot_province_total_ebike_lifecycle,
                    plot_regions_map_and_gef_scatter,
                    plot_regions_map_city_rank_and_drone_share,
                    plot_regions_map_city_province_ranks,
                    plot_regions_map_city_rank_inset,
                    plot_regions_map_kappa_panels,
                    plot_regions_map_oracle_reduction_rank,
                    plot_regions_map_oracle_reduction_rank_map_only,
                    plot_regions_map_optimal_drone_share,
                    plot_regions_map_pathway_scenario_maps,
                    plot_regions_map_pathway_scenarios,
                )

            if _want("province_orders_emission"):
                plot_province_orders_emission(
                    df,
                    save_path=os.path.join(out, "province_orders_emission"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("province_orders_both_emissions"):
                plot_province_orders_both_emissions(
                    df,
                    save_path=os.path.join(out, "province_orders_both_emissions"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("province_total_ebike_lifecycle"):
                plot_province_total_ebike_lifecycle(
                    df,
                    save_path=os.path.join(out, "province_total_ebike_lifecycle"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("city_ebike_rank_top10"):
                plot_city_ebike_rank_top10(
                    df,
                    save_path=os.path.join(out, "city_ebike_rank_top10"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("city_ebike_rank_bottom10"):
                plot_city_ebike_rank_bottom10(
                    df,
                    save_path=os.path.join(out, "city_ebike_rank_bottom10"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            city_oracle_reduction_rank_top10_requested = _want(
                "city_oracle_reduction_rank_top10"
            )
            city_oracle_reduction_rank_top10_pdf_needed = (
                _figure2_needs_vector_source("city_oracle_reduction_rank_top10")
            )
            if city_oracle_reduction_rank_top10_requested:
                plot_city_oracle_reduction_rank_top10(
                    df,
                    save_path=os.path.join(out, "city_oracle_reduction_rank_top10"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if city_oracle_reduction_rank_top10_pdf_needed:
                plot_city_oracle_reduction_rank_top10(
                    df,
                    save_path=os.path.join(
                        framework_out, "city_oracle_reduction_rank_top10"
                    ),
                    for_latex=True,
                    plot_title=plot_title,
                )

            if _want("city_reduction_rank_top10"):
                plot_city_reduction_rank_top10(
                    df,
                    save_path=os.path.join(out, "city_reduction_rank_top10"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            city_reduction_rank_all_requested = _want(
                "city_reduction_rank_all_macro_region"
            )
            city_reduction_rank_all_pdf_needed = _figure3_needs_vector_source(
                "city_reduction_rank_all_macro_region"
            )
            if city_reduction_rank_all_requested:
                plot_city_reduction_rank_all_macro_region(
                    df,
                    save_path=os.path.join(out, "city_reduction_rank_all_macro_region"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if city_reduction_rank_all_pdf_needed:
                plot_city_reduction_rank_all_macro_region(
                    df,
                    save_path=os.path.join(
                        framework_out, "city_reduction_rank_all_macro_region"
                    ),
                    for_latex=True,
                    plot_title=plot_title,
                )

            needs_reduction_detour_city_stats = (
                any(_want(figure_name) for figure_name in CITY_CONTEXT_FIGURES)
                or any(_want(figure_name) for figure_name in CITY_CONTEXT_GROUP_FIGURES)
                or _want("city_gef_context_detour_summary")
                or _want("break_even_contribution_decomposition")
                or _want("break_even_contribution_decomposition_marching_cubes")
                or _want("break_even_contribution_decomposition_marching_cubes_de")
                or (only is not None and _want("contribution_decomposition"))
                or (only is not None and _want("contribution_decomposition_city_b"))
                or _want("top6_reduction_detour_grid")
                or _want("top6_reduction_detour_grid_low_gef")
                or _want("top6_reduction_detour_grid_random")
            )
            if needs_reduction_detour_city_stats:
                raw_city_df = df[df["region_level"] == "city"].copy()
                gef_path = os.path.join(data_dir, "gef_china.csv")
                gef_df = pd.read_csv(gef_path, encoding="gbk")
                gef_df.columns = ["province", "gef"]
                contribution_city_df = _prepare_contribution_decomposition_city_stats(
                    raw_city_df,
                    gef_df,
                    exp_config,
                )
                city_df = raw_city_df.merge(gef_df, on="province", how="left")
                city_df = _prepare_reduction_detour_city_stats(city_df)

                if (
                    _want("break_even_contribution_decomposition")
                    or _want("break_even_contribution_decomposition_marching_cubes")
                ):
                    _render_break_even_contribution_decomposition(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=contribution_city_df,
                        exp_config=exp_config,
                        render_base=_want("break_even_contribution_decomposition"),
                        render_marching_cubes=_want(
                            "break_even_contribution_decomposition_marching_cubes"
                        ),
                    )

                if _want("break_even_contribution_decomposition_marching_cubes_de"):
                    _render_break_even_contribution_decomposition_marching_cubes_de(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=contribution_city_df,
                        exp_config=exp_config,
                    )

                if only is not None and _want("contribution_decomposition"):
                    _render_contribution_decomposition(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=city_df,
                        exp_config=exp_config,
                    )

                if only is not None and _want("contribution_decomposition_city_b"):
                    _render_contribution_decomposition_city_b(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=city_df,
                        exp_config=exp_config,
                    )

                for figure_name, context in CITY_CONTEXT_FIGURES.items():
                    if not _want(figure_name):
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

                for figure_name, contexts in CITY_CONTEXT_GROUP_FIGURES.items():
                    if not _want(figure_name):
                        continue
                    _render_city_context_group(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=city_df,
                        exp_config=exp_config,
                        figure_name=figure_name,
                        contexts=contexts,
                    )

            if _want("top6_reduction_detour_grid") or _want("top6_reduction_detour_grid_low_gef") or _want("top6_reduction_detour_grid_random"):
                selected_map = _select_top6_reduction_detour_variants(city_df, gef_df)
                for figure_name, selected in selected_map.items():
                    if _want(figure_name):
                        _render_reduction_detour_grid_variant(
                            data_dir=data_dir,
                            out=out,
                            for_latex=for_latex,
                            plot_title=plot_title,
                            selected=selected,
                            figure_name=figure_name,
                            exp_config=exp_config,
                            absolute_reduction_metric="per_order_g",
                            show_detour_boxplot=False,
                            mark_reduction_break_even=True,
                            figsize=(
                                (18, 14)
                                if figure_name == "top6_reduction_detour_grid"
                                else (12, 14)
                            ),
                            n_cols=(
                                4
                                if figure_name == "top6_reduction_detour_grid"
                                else 2
                            ),
                            grid_hspace=(
                                0.40
                                if figure_name == "top6_reduction_detour_grid"
                                else 0.45
                            ),
                            grid_wspace=(
                                0.50
                                if figure_name == "top6_reduction_detour_grid"
                                else 0.22
                            ),
                            use_binned_cache=figure_name == "top6_reduction_detour_grid",
                            min_air_km=(
                                0.2
                                if figure_name == "top6_reduction_detour_grid"
                                else 0.0
                            ),
                            bin_width_km=(
                                0.1
                                if figure_name == "top6_reduction_detour_grid"
                                else 0.1
                            ),
                        )

            if _want("regions_map"):
                from visualization.regions import plot_regions_on_china_map

                plot_regions_on_china_map(
                    csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            regions_map_oracle_reduction_rank_requested = _want(
                "regions_map_oracle_reduction_rank"
            )
            regions_map_oracle_reduction_rank_pdf_needed = (
                _figure1_needs_vector_source("regions_map_oracle_reduction_rank")
            )
            if regions_map_oracle_reduction_rank_requested:
                plot_regions_map_oracle_reduction_rank(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map_oracle_reduction_rank.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if regions_map_oracle_reduction_rank_pdf_needed:
                plot_regions_map_oracle_reduction_rank(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(
                        framework_out, "regions_map_oracle_reduction_rank.png"
                    ),
                    for_latex=True,
                    plot_title=plot_title,
                )

            regions_map_oracle_reduction_rank_map_only_requested = _want(
                "regions_map_oracle_reduction_rank_map_only"
            )
            regions_map_oracle_reduction_rank_map_only_pdf_needed = (
                _figure2_needs_vector_source(
                    "regions_map_oracle_reduction_rank_map_only"
                )
            )
            if regions_map_oracle_reduction_rank_map_only_requested:
                plot_regions_map_oracle_reduction_rank_map_only(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(
                        out,
                        "regions_map_oracle_reduction_rank_map_only.png",
                    ),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if regions_map_oracle_reduction_rank_map_only_pdf_needed:
                plot_regions_map_oracle_reduction_rank_map_only(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(
                        framework_out,
                        "regions_map_oracle_reduction_rank_map_only.png",
                    ),
                    for_latex=True,
                    plot_title=plot_title,
                )

            if _want("regions_map_optimal_drone_share"):
                plot_regions_map_optimal_drone_share(
                    df,
                    save_path=os.path.join(out, "regions_map_optimal_drone_share.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("regions_map_city_rank_and_drone_share"):
                beijing_od_path = os.path.join(
                    data_dir,
                    "city",
                    "beijing",
                    "beijing_od_pairs.csv",
                )
                beijing_graphml_path = os.path.join(
                    data_dir,
                    "city",
                    "beijing",
                    "beijing_roadnet.graphml",
                )
                beijing_od_df = None
                if os.path.exists(beijing_od_path):
                    beijing_od_df = pd.read_csv(beijing_od_path)
                else:
                    print(
                        "regions_map_city_rank_and_drone_share: "
                        "Beijing OD missing, Beijing median-trip panels will show "
                        f"placeholders ({beijing_od_path})"
                    )
                beijing_graph, beijing_router = None, None
                if os.path.exists(beijing_graphml_path):
                    from roadnet.sparse_graph import SparseGraph
                    from roadnet.tile_router import TileRouter

                    beijing_graph = SparseGraph.from_graphml(beijing_graphml_path)
                    beijing_router = TileRouter(beijing_graph)
                else:
                    print(
                        "regions_map_city_rank_and_drone_share: "
                        "Beijing roadnet missing, panel d will show placeholder "
                        f"({beijing_graphml_path})"
                    )
                plot_regions_map_city_rank_and_drone_share(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    median_trip_od_df=beijing_od_df,
                    median_trip_graph=beijing_graph,
                    median_trip_router=beijing_router,
                    median_trip_city="beijing",
                    save_path=os.path.join(out, "regions_map_city_rank_and_drone_share.png"),
                    subfigure_save_dir=out,
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("regions_map_city_province_ranks"):
                plot_regions_map_city_province_ranks(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map_city_province_ranks.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("regions_map_city_rank_inset"):
                plot_regions_map_city_rank_inset(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map_city_rank_inset.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            if _want("regions_map_kappa_panels"):
                named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
                if not os.path.exists(named_path):
                    print(f"regions_map_kappa_panels skipped — missing {named_path}; run sensitivity --stages named first")
                else:
                    plot_regions_map_kappa_panels(
                        named_csv_path=named_path,
                        stats_csv_path=stats_path,
                        save_path=os.path.join(out, "regions_map_kappa_panels.png"),
                        for_latex=for_latex,
                        plot_title=plot_title,
                    )

            if _want("regions_map_pathway_scenarios"):
                named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
                if not os.path.exists(named_path):
                    print(f"regions_map_pathway_scenarios skipped — missing {named_path}; run sensitivity --stages named first")
                else:
                    plot_regions_map_pathway_scenarios(
                        named_csv_path=named_path,
                        stats_csv_path=stats_path,
                        data_dir=data_dir,
                        exp_config=exp_config,
                        save_path=os.path.join(out, "regions_map_pathway_scenarios.png"),
                        for_latex=for_latex,
                        plot_title=plot_title,
                    )

            if _want("regions_map_pathway_scenario_maps"):
                named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
                if not os.path.exists(named_path):
                    print(
                        "regions_map_pathway_scenario_maps skipped — missing "
                        f"{named_path}; run sensitivity --stages named first"
                    )
                else:
                    plot_regions_map_pathway_scenario_maps(
                        named_csv_path=named_path,
                        stats_csv_path=stats_path,
                        data_dir=data_dir,
                        exp_config=exp_config,
                        save_path=os.path.join(
                            out,
                            "regions_map_pathway_scenario_maps.png",
                        ),
                        for_latex=for_latex,
                        plot_title=plot_title,
                    )

            if (
                only is None
                or only & gef_figures
                or _figure3_needs_vector_source(
                    "city_population_detour_oracle_reduction_bubble"
                )
            ):
                enriched_path = os.path.join(data_dir, "all_region_stats_enriched.csv")
                if not os.path.exists(enriched_path):
                    print(f"GEF figures skipped — missing {enriched_path}; run tools.enrich_stats first")
                else:
                    df_enriched = pd.read_csv(enriched_path)
                    df_enriched = _attach_enrichment_columns(df, df_enriched)
                    if _want("city_gef_distribution"):
                        plot_city_gef_distribution(
                            df_enriched,
                            save_path=os.path.join(out, "city_gef_distribution.png"),
                            for_latex=for_latex,
                            plot_title=plot_title,
                        )
                    if _want("city_gef_scatter_components"):
                        plot_city_gef_scatter_components(
                            df_enriched,
                            save_path=os.path.join(out, "city_gef_scatter_components.png"),
                            for_latex=for_latex,
                            plot_title=plot_title,
                        )
                    city_population_bubble_requested = _want(
                        "city_population_detour_oracle_reduction_bubble"
                    )
                    city_population_bubble_pdf_needed = _figure3_needs_vector_source(
                        "city_population_detour_oracle_reduction_bubble"
                    )
                    if city_population_bubble_requested:
                        plot_city_population_detour_oracle_reduction_bubble(
                            df_enriched,
                            save_path=os.path.join(
                                out,
                                "city_population_detour_oracle_reduction_bubble.png",
                            ),
                            for_latex=for_latex,
                            plot_title=plot_title,
                        )
                    if city_population_bubble_pdf_needed:
                        plot_city_population_detour_oracle_reduction_bubble(
                            df_enriched,
                            save_path=os.path.join(
                                framework_out,
                                "city_population_detour_oracle_reduction_bubble.png",
                            ),
                            for_latex=True,
                            plot_title=plot_title,
                        )
                    if _want("scatter_detour_ratio_vs_reduction_by_gef"):
                        reduction_col = _weighted_reduction_column(df_enriched)
                        plot_grouped_scatter_split_panels(
                            df_enriched,
                            feature_var="detour_ratio",
                            result_var=reduction_col,
                            group_var="grid_emission_factor",
                            threshold=0.3,
                            save_path=os.path.join(out, "scatter_detour_ratio_vs_reduction_by_gef.png"),
                            figsize=(16, 4),
                            for_latex=for_latex,
                            plot_title=plot_title,
                        )
                    if _want("city_gef_context_detour_summary"):
                        _render_city_gef_context_detour_summary(
                            data_dir=data_dir,
                            out=out,
                            for_latex=for_latex,
                            plot_title=plot_title,
                            df_enriched=df_enriched,
                            city_df=city_df,
                            exp_config=exp_config,
                        )
                    if _want("regions_map_and_gef_scatter"):
                        plot_regions_map_and_gef_scatter(
                            df_stats=df,
                            df_enriched=df_enriched,
                            stats_csv_path=stats_path,
                            save_path=os.path.join(out, "regions_map_and_gef_scatter.png"),
                            for_latex=for_latex,
                            plot_title=plot_title,
                        )
                    named_path = os.path.join(data_dir, "all_sensitivity_named.csv")
                    for year in ("2030", "2060"):
                        fig_name = f"regions_map_and_gef_scatter_{year}"
                        if not _want(fig_name):
                            continue
                        if not os.path.exists(named_path):
                            print(f"{fig_name} skipped — missing {named_path}; run sensitivity first")
                            continue
                        plot_regions_map_and_gef_scatter(
                            df_stats=df,
                            df_enriched=df_enriched,
                            stats_csv_path=stats_path,
                            save_path=os.path.join(out, f"{fig_name}.png"),
                            for_latex=for_latex,
                            plot_title=plot_title,
                            gef_scenario=year,
                            named_csv_path=named_path,
                        )

    if figure1_framework_requested:
        with _profile_block(profile, "figure1_framework"):
            _render_figure1_framework(
                out=framework_out,
                plot_title=plot_title,
            )

    if figure2_framework_requested:
        with _profile_block(profile, "figure2_framework"):
            _render_figure2_framework(
                out=framework_out,
                plot_title=plot_title,
            )

    if figure3_framework_requested:
        with _profile_block(profile, "figure3_framework"):
            _render_figure3_framework(
                out=framework_out,
                plot_title=plot_title,
            )

    if figure4_framework_requested:
        with _profile_block(profile, "figure4_framework"):
            _render_figure4_framework(
                out=framework_out,
                plot_title=plot_title,
            )

    # OAT tornado diagnostics (compact + SI full): independent of
    # all_region_stats.csv; only need all_sensitivity_oat.csv. Kept outside
    # the stats-gated block so --only oat_tornado_* works standalone.
    if _want("oat_tornado_story") or _want("oat_tornado_full"):
        from visualization.sensitivity import OAT_FEATURES_STORY, plot_oat_tornado
        oat_path = os.path.join(data_dir, "all_sensitivity_oat.csv")
        if not os.path.exists(oat_path):
            print(f"oat_tornado skipped — missing {oat_path}; run sensitivity first")
        else:
            oat_df = pd.read_csv(oat_path)
            if _want("oat_tornado_story"):
                plot_oat_tornado(
                    oat_df,
                    save_path=os.path.join(out, "oat_tornado_story.png"),
                    figsize=(8, 5.5),
                    for_latex=for_latex,
                    plot_title=plot_title,
                    feature_whitelist=OAT_FEATURES_STORY,
                )
            if _want("oat_tornado_full"):
                plot_oat_tornado(
                    oat_df,
                    save_path=os.path.join(out, "oat_tornado_full.png"),
                    figsize=(8, 6.5),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

    # Sobol + Monte Carlo figures (global SA, replaces Shapley in main text).
    # Independent of all_region_stats.csv; only need all_sensitivity_sobol.csv.
    sobol_figures = {
        "drone_tech_sensitivity",
        "sobol_indices", "sobol_by_mechanism", "sobol_grouped", "sobol_strip",
        "mc_uncertainty", "mc_anchor_forest", "mc_anchor_scatter",
    }
    if only is None or only & sobol_figures:
        from visualization.sensitivity import (
            plot_drone_tech_sensitivity_panels,
            plot_mc_anchor_forest, plot_mc_anchor_scatter, plot_mc_uncertainty,
            plot_sobol_by_mechanism, plot_sobol_grouped, plot_sobol_indices,
            plot_sobol_strip,
        )
        sobol_path = os.path.join(data_dir, "all_sensitivity_sobol.csv")
        if not os.path.exists(sobol_path):
            print(f"Sobol/MC figures skipped — missing {sobol_path}; run sensitivity + consolidate_sensitivity first")
        else:
            sobol_df = pd.read_csv(sobol_path)
            if _want("drone_tech_sensitivity"):
                plot_drone_tech_sensitivity_panels(
                    sobol_df,
                    save_path=os.path.join(out, "drone_tech_sensitivity.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("sobol_indices"):
                plot_sobol_indices(
                    sobol_df,
                    save_path=os.path.join(out, "sobol_indices.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("sobol_by_mechanism"):
                plot_sobol_by_mechanism(
                    sobol_df,
                    save_path=os.path.join(out, "sobol_by_mechanism.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("sobol_grouped"):
                plot_sobol_grouped(
                    sobol_df,
                    save_path=os.path.join(out, "sobol_grouped.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("sobol_strip"):
                plot_sobol_strip(
                    sobol_df,
                    save_path=os.path.join(out, "sobol_strip.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("mc_uncertainty"):
                plot_mc_uncertainty(
                    sobol_df,
                    save_path=os.path.join(out, "mc_uncertainty.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("mc_anchor_forest"):
                plot_mc_anchor_forest(
                    sobol_df,
                    save_path=os.path.join(out, "mc_anchor_forest.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )
            if _want("mc_anchor_scatter"):
                plot_mc_anchor_scatter(
                    sobol_df,
                    save_path=os.path.join(out, "mc_anchor_scatter.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )


def _bootstrap_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.normpath(os.path.join(here, ".."))
    if src not in sys.path:
        sys.path.insert(0, src)


def main():
    _bootstrap_paths()
    from utils import load_config

    parser = argparse.ArgumentParser(description="Render paper figures")
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--png", action="store_true", help="Force PNG output")
    parser.add_argument("--titled", action="store_true", help="Add titles to figures")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated figure names to render")
    parser.add_argument("--profile", action="store_true", help="Print elapsed time for selected render blocks")
    args = parser.parse_args()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    for_latex = (not args.png) and config.get("for_latex", False)
    plot_title = args.titled or config.get("plot_title", False)
    only = set(args.only.split(",")) if args.only else None

    print(
        "for_latex="
        f"{for_latex}, plot_title={plot_title}, only={only or 'all'}, "
        f"profile={args.profile}"
    )
    render_all(
        config["data_dir"],
        config["figures_dir"],
        for_latex=for_latex,
        plot_title=plot_title,
        only=only,
        exp_config=config,
        profile=args.profile,
    )



if __name__ == "__main__":
    main()
