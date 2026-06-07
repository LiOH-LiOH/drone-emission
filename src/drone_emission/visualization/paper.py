"""Entry point for rendering all paper figures.

Run:
    cd src/drone_emission
    ../../.venv/bin/python -m visualization.paper --png              # PNG (fast iteration)
    ../../.venv/bin/python -m visualization.paper                    # PDF (for LaTeX)
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap
    ../../.venv/bin/python -m visualization.paper --only shanghai_heatmap,distance_comparison

Available figures:
    detour_lifecycle_distance, lifecycle_distance_detour_boxplot, emission_model_flowchart,
    detour_and_lifecycle, distance_comparison,
    distance_comparison_high_3x3, distance_comparison_low_3x3,
    shanghai_heatmap, shanghai_road_air_distance, shanghai_road_detour,
    shanghai_air_detour, shanghai_air_road_joint, shanghai_emission_reduction,
    shanghai_emission_by_distance, shanghai_reduction_detour_by_distance,
    shanghai_city_context,
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
SHANGHAI_CITY_CONTEXT_FIGURE = "shanghai_city_context"

REDUCTION_DETOUR_CACHE_VERSION = 2
REDUCTION_DETOUR_SELECTION_METRIC = "total_ebike_lifecycle"


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


def _render_shanghai_city_context(data_dir, out, for_latex, plot_title, city_df, exp_config):
    """Render a one-row Shanghai city context figure."""
    region_key = "shanghai"
    sales_path = os.path.join(data_dir, "city", region_key, f"{region_key}_sales.csv")
    if not os.path.exists(sales_path):
        print(f"{SHANGHAI_CITY_CONTEXT_FIGURE} skipped — missing {sales_path}")
        return

    selected = city_df[city_df["region_key"] == region_key].copy()
    if selected.empty:
        print(f"{SHANGHAI_CITY_CONTEXT_FIGURE} skipped — Shanghai row missing")
        return

    built = _build_reduction_detour_binned_by_city(
        data_dir=data_dir,
        selected=selected,
        figure_name=SHANGHAI_CITY_CONTEXT_FIGURE,
        exp_config=exp_config,
        absolute_reduction_metric="per_order_g",
        use_binned_cache=False,
        bin_width_km=0.1,
    )
    if built is None:
        return

    binned_by_city, labels = built
    if region_key not in binned_by_city:
        print(f"{SHANGHAI_CITY_CONTEXT_FIGURE} skipped — Shanghai binned data missing")
        return

    import matplotlib.pyplot as plt

    from visualization._utils import _add_title_suffix
    from visualization.analysis import _plot_reduction_detour_panel
    from visualization.sales import _plot_sales_heatmap_on_ax

    sales_df = pd.read_csv(sales_path)
    fig = plt.figure(figsize=(14.5, 5.6))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.34)
    heatmap_grid = outer[0].subgridspec(1, 2, width_ratios=[24, 1], wspace=0.05)
    ax_heatmap = fig.add_subplot(heatmap_grid[0])
    cax_heatmap = fig.add_subplot(heatmap_grid[1])

    _plot_sales_heatmap_on_ax(
        ax_heatmap,
        sales_df,
        "上海市",
        boundaries_dict=None,
        lng_range=(121.0, 121.9),
        lat_range=(30.9, 31.55),
        zoom=14,
        cax=cax_heatmap,
    )
    ax_heatmap.set_xlabel("Longitude", fontsize=10)
    ax_heatmap.set_ylabel("Latitude", fontsize=10)
    ax_heatmap.grid(True, alpha=0.3, linestyle="--", color="white")
    ax_heatmap.ticklabel_format(useOffset=False, style="plain")
    ax_heatmap.text(
        -0.10,
        1.02,
        "a",
        transform=ax_heatmap.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )

    _, ax_reduction, _ = _plot_reduction_detour_panel(
        fig,
        outer[1],
        binned_by_city[region_key],
        labels.get(region_key, "Shanghai"),
        bin_width_km=0.1,
        min_air_km=0.2,
        absolute_reduction_metric="per_order_g",
        show_detour_boxplot=False,
        mark_reduction_break_even=True,
    )
    ax_reduction.text(
        -0.12,
        1.08,
        "b",
        transform=ax_reduction.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
    )

    if plot_title:
        fig.suptitle("Shanghai Delivery Demand and Emission Reduction by Distance", fontsize=14)

    save_path = os.path.join(out, f"{SHANGHAI_CITY_CONTEXT_FIGURE}.png")
    if for_latex and not save_path.endswith(".pdf"):
        save_path = save_path.rsplit(".", 1)[0] + ".pdf"
    if plot_title:
        save_path = _add_title_suffix(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Shanghai city context saved to {save_path}")


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
        plot_grouped_scatter,
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
        plot_detour_and_lifecycle, plot_detour_lifecycle_distance, plot_emission_model_flowchart,
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
        SHANGHAI_CITY_CONTEXT_FIGURE,
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
                _want(SHANGHAI_CITY_CONTEXT_FIGURE)
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

                if _want(SHANGHAI_CITY_CONTEXT_FIGURE):
                    _render_shanghai_city_context(
                        data_dir=data_dir,
                        out=out,
                        for_latex=for_latex,
                        plot_title=plot_title,
                        city_df=city_df,
                        exp_config=exp_config,
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
                        plot_grouped_scatter(
                            df_enriched,
                            feature_var="detour_ratio",
                            result_var=reduction_col,
                            group_var="grid_emission_factor",
                            threshold=0.3,
                            save_path=os.path.join(out, "scatter_detour_ratio_vs_reduction_by_gef.png"),
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
