"""Entry point for rendering all paper figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.paper --png              # PNG (fast iteration)
    ../../.venv/bin/python -m visualization.paper                    # PDF (for LaTeX)
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap,distance_comparison

Available figures:
    detour_lifecycle_distance, lifecycle_distance_detour_boxplot,
    lifecycle_distance_route_factor, production_phase_per_km, end_of_life_phase_per_km,
    production_detour_eol_panels, emission_model_flowchart,
    detour_and_lifecycle, distance_comparison,
    distance_comparison_high_3x3, distance_comparison_low_3x3,
    shanghai_heatmap, shanghai_road_air_distance, shanghai_road_detour,
    shanghai_air_detour, shanghai_air_road_joint, shanghai_emission_reduction,
    shanghai_emission_by_distance, shanghai_reduction_detour_by_distance,
    shanghai_city_context, chongqing_city_context, chengdu_city_context,
    guangzhou_city_context, tianjin_city_context, dali_city_context,
    kunming_city_context, dali_chengdu_tianjin_guangzhou_city_context,
    top6_reduction_detour_grid, top6_reduction_detour_grid_low_gef,
    top6_reduction_detour_grid_random,
    province_orders_emission, province_orders_both_emissions,
    province_total_ebike_lifecycle,
    city_ebike_rank_top10, city_ebike_rank_bottom10,
    city_reduction_rank_top10,
    city_gef_distribution,
    city_gef_scatter_components,
    regions_map,
    regions_map_optimal_drone_share,
    regions_map_city_rank_and_drone_share,
    scatter_detour_ratio_vs_reduction_by_gef,
    regions_map_and_gef_scatter,
    regions_map_city_province_ranks,
    regions_map_city_rank_inset,
    regions_map_kappa_panels,
    oat_tornado_story, oat_tornado_full
"""

import argparse
import os
import pickle
import sys

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
}

REDUCTION_DETOUR_CACHE_VERSION = 2
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
        gef = gef_map[key]
        ebike_pp = replace(ebike_params, e_use=float(gef))
        road_km = od_df["road_distance_m"].to_numpy(dtype=float) / 1000.0
        od_df = od_df.copy()
        od_df["ebike_lifecycle_g"] = compute_ebike_per_order_g(road_km, ebike_pp, scenario)
        centres, reduction_pct, reduction_abs, detour_by_bin, max_km = (
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


def _render_shanghai_heatmap(data_dir, out, for_latex, plot_title):
    """Load Shanghai sales data and render the sales heatmap figure."""
    region_key = "shanghai"
    region_level = "city"
    sales_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_sales.csv")

    if not os.path.exists(sales_path):
        print(f"Shanghai heatmap skipped — missing {sales_path}")
        return

    from visualization.sales import plot_sales_heatmap

    df = pd.read_csv(sales_path)
    plot_sales_heatmap(
        df,
        "Shanghai",
        "上海市",
        boundaries_dict=None,
        save_path=os.path.join(out, "shanghai_heatmap.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        lng_range=(121.0, 121.9),
        lat_range=(30.9, 31.55),
    )


def _plot_power_generation_mix_pie(
    ax,
    region_key,
    legend_ax=None,
    radius=1.18,
    legend_fontsize=9,
    legend_handlelength=1.2,
    legend_handletextpad=0.5,
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
            handletextpad=legend_handletextpad,
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
            handletextpad=legend_handletextpad,
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
        mark_reduction_break_even=True,
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
        mark_reduction_break_even=True,
        show_city_label=False,
        bar_width_fraction=0.78,
        show_absolute_axis=False,
        show_saturation_fit=show_saturation_fit,
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
        bin_width_km=0.1,
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

    fig = plt.figure(figsize=(24.0, 7.8))
    outer = fig.add_gridspec(1, 4, wspace=0.24)
    reduction_axes = []
    absolute_axes = []

    for i, context in enumerate(available_contexts):
        region_key = context["region_key"]
        display_name = context["display_name"]
        power_region_key = context.get("power_region_key", region_key)
        show_saturation_fit = region_key not in {
            "dalibaizuzizhizhou",
            "chengdu",
        }
        block = outer[0, i].subgridspec(
            3,
            1,
            height_ratios=[0.13, 1.45, 0.72],
            hspace=0.26,
        )
        ax_heading = fig.add_subplot(block[0])
        ax_heading.set_axis_off()
        ax_heading.text(
            -0.075,
            0.60,
            chr(ord("a") + i),
            transform=ax_heading.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="center",
        )
        ax_heading.text(
            0.00,
            0.60,
            display_name,
            transform=ax_heading.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="center",
        )

        ax_reduction, ax_absolute = _plot_city_context_combined_ab_panel(
            fig,
            block[1],
            binned_by_city[region_key],
            labels.get(region_key, display_name),
            show_saturation_fit,
        )
        reduction_axes.append(ax_reduction)
        absolute_axes.append(ax_absolute)

        power_grid = block[2].subgridspec(
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
            legend_fontsize=9.8,
            legend_handlelength=1.45,
            legend_handletextpad=0.62,
        )
        legend = ax_power_legend.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(9.8)

        for ax in (ax_reduction, ax_absolute):
            ax.xaxis.label.set_size(8.5)
            ax.yaxis.label.set_size(8.5)
            ax.tick_params(axis="both", labelsize=7.5)

    for axes in (reduction_axes, absolute_axes):
        if axes:
            y_min = min(ax.get_ylim()[0] for ax in axes)
            y_max = max(ax.get_ylim()[1] for ax in axes)
            for ax in axes:
                ax.set_ylim(y_min, y_max)

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


def _render_distance_comparison(data_dir, out, for_latex, plot_title, plot_fn):
    """Load Shanghai roadnet data and render the distance comparison figure."""
    region_key = "shanghai"
    region_level = "city"
    od_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_od_pairs.csv")
    graphml_path = os.path.join(data_dir, region_level, region_key, f"{region_key}_roadnet.graphml")

    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        print(f"Distance comparison skipped — missing data for {region_key}")
        return

    from roadnet.sparse_graph import SparseGraph
    from roadnet.tile_router import TileRouter

    od_df = pd.read_csv(od_path)
    graph = SparseGraph.from_graphml(graphml_path)
    router = TileRouter(graph)

    plot_fn(
        od_df=od_df,
        graph=graph,
        router=router,
        save_path=os.path.join(out, "distance_comparison.png"),
        for_latex=for_latex,
        plot_title=plot_title,
        n_samples=2,
    )


def render_all(data_dir: str, figures_dir: str, for_latex: bool = False, plot_title: bool = False, only: set = None, exp_config: dict = None):
    from visualization.analysis import (
        _weighted_reduction_column,
        plot_city_ebike_rank_bottom10,
        plot_city_ebike_rank_top10,
        plot_city_gef_distribution,
        plot_city_gef_scatter_components,
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
        plot_regions_map_optimal_drone_share,
        plot_regions_map_pathway_scenarios,
    )
    from visualization.concepts import (
        plot_detour_and_lifecycle,
        plot_detour_lifecycle_distance,
        plot_emission_model_flowchart,
        plot_production_detour_eol_panels,
    )
    from visualization.emission import (
        plot_end_of_life_phase_per_km,
        plot_lifecycle_distance_route_factor,
        plot_production_phase_per_km,
    )
    from visualization.regions import plot_regions_on_china_map
    from visualization.roadnet import plot_distance_comparison

    out = os.path.join(figures_dir, "paper")
    os.makedirs(out, exist_ok=True)

    def _want(name):
        return only is None or name in only

    # --- Concept figures (no data dependency) ---
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
                plot_lifecycle_distance_route_factor(
                    od_df,
                    save_path=os.path.join(out, "lifecycle_distance_route_factor.png"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

    # --- Shanghai figures (independent of all_region_stats) ---
    if _want("shanghai_heatmap"):
        _render_shanghai_heatmap(
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

    if _want("shanghai_air_road_joint"):
        _render_shanghai_air_road_joint(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
        )

    if _want("distance_comparison"):
        _render_distance_comparison(
            data_dir=data_dir,
            out=out,
            for_latex=for_latex,
            plot_title=plot_title,
            plot_fn=plot_distance_comparison,
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
    future_gef_figures = ("regions_map_and_gef_scatter_2030", "regions_map_and_gef_scatter_2060")
    gef_figures = {
        "city_gef_distribution",
        "city_gef_scatter_components",
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
        "city_reduction_rank_top10",
        "regions_map",
        "regions_map_optimal_drone_share",
        "regions_map_city_rank_and_drone_share",
        "regions_map_city_province_ranks",
        "regions_map_city_rank_inset",
        "regions_map_kappa_panels",
        "regions_map_pathway_scenarios",
        *CITY_CONTEXT_FIGURES,
        *CITY_CONTEXT_GROUP_FIGURES,
    } | gef_figures | TOP6_REDUCTION_DETOUR_FIGURES
    if only is None or only & stats_figures:
        stats_path = os.path.join(data_dir, "all_region_stats.csv")
        if not os.path.exists(stats_path):
            print(f"Missing {stats_path}; run consolidate_stats first — skipping stats figures")
        else:
            df = pd.read_csv(stats_path)

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

            if _want("city_reduction_rank_top10"):
                plot_city_reduction_rank_top10(
                    df,
                    save_path=os.path.join(out, "city_reduction_rank_top10"),
                    for_latex=for_latex,
                    plot_title=plot_title,
                )

            needs_reduction_detour_city_stats = (
                any(_want(figure_name) for figure_name in CITY_CONTEXT_FIGURES)
                or any(_want(figure_name) for figure_name in CITY_CONTEXT_GROUP_FIGURES)
                or _want("top6_reduction_detour_grid")
                or _want("top6_reduction_detour_grid_low_gef")
                or _want("top6_reduction_detour_grid_random")
            )
            if needs_reduction_detour_city_stats:
                city_df = df[df["region_level"] == "city"].copy()
                gef_path = os.path.join(data_dir, "gef_china.csv")
                gef_df = pd.read_csv(gef_path, encoding="gbk")
                gef_df.columns = ["province", "gef"]
                city_df = city_df.merge(gef_df, on="province", how="left")
                city_df = _prepare_reduction_detour_city_stats(city_df)

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
                plot_regions_on_china_map(
                    csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map.png"),
                    for_latex=for_latex,
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
                plot_regions_map_city_rank_and_drone_share(
                    df_stats=df,
                    stats_csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map_city_rank_and_drone_share.png"),
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
                oat_path = os.path.join(data_dir, "all_sensitivity_oat.csv")
                if not os.path.exists(oat_path):
                    print(f"regions_map_kappa_panels skipped — missing {oat_path}; run sensitivity first")
                else:
                    plot_regions_map_kappa_panels(
                        oat_csv_path=oat_path,
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
                        save_path=os.path.join(out, "regions_map_pathway_scenarios.png"),
                        for_latex=for_latex,
                        plot_title=plot_title,
                    )

            if only is None or only & gef_figures:
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

    # OAT tornado figures (Fig. 5a story + SI full): independent of
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
        "sobol_indices", "sobol_by_mechanism", "sobol_grouped", "sobol_strip",
        "mc_uncertainty", "mc_anchor_forest", "mc_anchor_scatter",
    }
    if only is None or only & sobol_figures:
        from visualization.sensitivity import (
            plot_mc_anchor_forest, plot_mc_anchor_scatter, plot_mc_uncertainty,
            plot_sobol_by_mechanism, plot_sobol_grouped, plot_sobol_indices,
            plot_sobol_strip,
        )
        sobol_path = os.path.join(data_dir, "all_sensitivity_sobol.csv")
        if not os.path.exists(sobol_path):
            print(f"Sobol/MC figures skipped — missing {sobol_path}; run sensitivity + consolidate_sensitivity first")
        else:
            sobol_df = pd.read_csv(sobol_path)
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
    args = parser.parse_args()

    project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    config = load_config(os.path.join(project_root, "experiment.toml"), env=args.env)
    for_latex = (not args.png) and config.get("for_latex", False)
    plot_title = args.titled or config.get("plot_title", False)
    only = set(args.only.split(",")) if args.only else None

    print(f"for_latex={for_latex}, plot_title={plot_title}, only={only or 'all'}")
    render_all(config["data_dir"], config["figures_dir"], for_latex=for_latex, plot_title=plot_title, only=only, exp_config=config)



if __name__ == "__main__":
    main()
