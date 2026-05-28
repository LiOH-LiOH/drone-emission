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
    top6_reduction_detour_grid, top6_reduction_detour_grid_low_gef,
    top6_reduction_detour_grid_random,
    province_orders_emission, province_orders_both_emissions,
    province_total_ebike_lifecycle,
    city_ebike_rank_top10, city_ebike_rank_bottom10,
    city_reduction_rank_top10,
    regions_map,
    scatter_detour_ratio_vs_reduction_by_gef,
    regions_map_and_gef_scatter,
    regions_map_city_province_ranks,
    regions_map_kappa_panels,
    oat_tornado_story, oat_tornado_full
"""

import argparse
import os
import sys

import pandas as pd


TOP6_REDUCTION_DETOUR_FIGURES = {
    "top6_reduction_detour_grid",
    "top6_reduction_detour_grid_low_gef",
    "top6_reduction_detour_grid_random",
}


def _render_reduction_detour_grid_variant(
    data_dir,
    out,
    for_latex,
    plot_title,
    selected,
    figure_name,
):
    """Load selected city OD data and render a reduction-detour grid."""
    from visualization.analysis import plot_reduction_detour_grid

    label_map = dict(zip(selected["region_key"], selected["region_name_en"]))
    od_dfs = {}
    labels = {}
    required = {
        "road_distance_m",
        "distance_km",
        "drone_lifecycle_g",
        "ebike_lifecycle_g",
        "annual_sales",
    }
    for key in selected["region_key"].tolist():
        od_path = os.path.join(data_dir, "city", key, f"{key}_od_pairs.csv")
        if not os.path.exists(od_path):
            print(f"{figure_name}: missing {od_path} — skipping {key}")
            continue
        od_df = pd.read_csv(od_path)
        if not required.issubset(od_df.columns):
            print(f"{figure_name}: {key} missing columns — skipping")
            continue
        od_dfs[key] = od_df
        labels[key] = label_map.get(key, key)

    if od_dfs:
        plot_reduction_detour_grid(
            od_dfs=od_dfs,
            city_labels=labels,
            save_path=os.path.join(out, f"{figure_name}.png"),
            for_latex=for_latex,
            plot_title=plot_title,
        )
    else:
        print(f"{figure_name}: no city OD data available")


def _select_top6_reduction_detour_variants(city_df, gef_df):
    """Return deterministic city selections for reduction-detour grid variants."""
    city_df = city_df.copy()
    city_df["gef_rank"] = city_df["gef"].rank(method="dense")

    big = city_df.nlargest(6, "total_odpair_annual_sales")

    low_gef_provinces = gef_df.nsmallest(8, "gef")["province"].tolist()
    low_gef = (
        city_df[city_df["province"].isin(low_gef_provinces)]
        .nlargest(6, "total_odpair_annual_sales")
    )

    big_cutoff = city_df["total_odpair_annual_sales"].quantile(0.75)
    small_floor = city_df["total_odpair_annual_sales"].quantile(0.25)
    non_low_gef = ~city_df["province"].isin(low_gef_provinces)
    mid_candidates = city_df[
        non_low_gef
        & (city_df["total_odpair_annual_sales"] < big_cutoff)
        & (city_df["total_odpair_annual_sales"] >= small_floor)
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
    required = {"road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g"}
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


def render_all(data_dir: str, figures_dir: str, for_latex: bool = False, plot_title: bool = False, only: set = None):
    from visualization.analysis import plot_province_orders_emission, plot_province_orders_both_emissions, plot_province_total_ebike_lifecycle, plot_city_ebike_rank_top10, plot_city_ebike_rank_bottom10, plot_city_reduction_rank_top10, plot_grouped_scatter, plot_regions_map_and_gef_scatter, plot_regions_map_city_province_ranks, plot_regions_map_kappa_panels, plot_regions_map_pathway_scenarios
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
    stats_figures = {"province_orders_emission", "province_orders_both_emissions", "province_total_ebike_lifecycle", "city_ebike_rank_top10", "city_ebike_rank_bottom10", "city_reduction_rank_top10", "regions_map", "scatter_detour_ratio_vs_reduction_by_gef", "regions_map_and_gef_scatter", "regions_map_and_gef_scatter_2030", "regions_map_and_gef_scatter_2060", "regions_map_city_province_ranks", "regions_map_kappa_panels", "regions_map_pathway_scenarios"} | TOP6_REDUCTION_DETOUR_FIGURES
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

            if _want("top6_reduction_detour_grid") or _want("top6_reduction_detour_grid_low_gef") or _want("top6_reduction_detour_grid_random"):
                city_df = df[df["region_level"] == "city"].copy()
                gef_path = os.path.join(data_dir, "gef_china.csv")
                gef_df = pd.read_csv(gef_path, encoding="gbk")
                gef_df.columns = ["province", "gef"]
                city_df = city_df.merge(gef_df, on="province", how="left")

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
                        )

            if _want("regions_map"):
                plot_regions_on_china_map(
                    csv_path=stats_path,
                    save_path=os.path.join(out, "regions_map.png"),
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

            future_gef_figures = ("regions_map_and_gef_scatter_2030", "regions_map_and_gef_scatter_2060")
            if (
                _want("scatter_detour_ratio_vs_reduction_by_gef")
                or _want("regions_map_and_gef_scatter")
                or any(_want(f) for f in future_gef_figures)
            ):
                enriched_path = os.path.join(data_dir, "all_region_stats_enriched.csv")
                if not os.path.exists(enriched_path):
                    print(f"GEF figures skipped — missing {enriched_path}; run tools.enrich_stats first")
                else:
                    df_enriched = pd.read_csv(enriched_path)
                    if _want("scatter_detour_ratio_vs_reduction_by_gef"):
                        plot_grouped_scatter(
                            df_enriched,
                            feature_var="detour_ratio",
                            result_var="max_lifecycle_reduction_pct",
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
    render_all(config["data_dir"], config["figures_dir"], for_latex=for_latex, plot_title=plot_title, only=only)



if __name__ == "__main__":
    main()
