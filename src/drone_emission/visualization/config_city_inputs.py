"""Formal config city inputs renderers used only by the manifest registry."""

from contextlib import contextmanager
import json
import os
import pickle
import time

import pandas as pd


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
    from drone_emission.visualization.sales import plot_merchant_monthly_sales_map

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
    """Load a SparseGraph once per locked render call."""
    if not os.path.exists(graphml_path):
        return None
    graph_cache = graph_cache if graph_cache is not None else {}
    if graphml_path not in graph_cache:
        from drone_emission.roadnet.sparse_graph import SparseGraph

        graph_cache[graphml_path] = SparseGraph.from_graphml(graphml_path)
    else:
        print(f"Reusing cached SparseGraph: {graphml_path}")
    return graph_cache[graphml_path]


def _get_cached_tile_router(graphml_path, graph, router_cache=None):
    """Create a TileRouter once per locked render call."""
    if graph is None:
        return None
    router_cache = router_cache if router_cache is not None else {}
    if graphml_path not in router_cache:
        from drone_emission.roadnet.tile_router import TileRouter

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
    from drone_emission.visualization.buildings import plot_building_height_merchant_map

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
    from drone_emission.visualization.buildings import plot_building_footprint_heatmap

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

    from drone_emission.visualization.roadnet import plot_road_network_subfigure

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


def _render_beijing_median_trip_phase_emissions(
    data_dir,
    out,
    for_latex,
    plot_title,
    only,
):
    """Render selected Beijing median-trip lifecycle subfigures."""
    region_key = "beijing"
    region_level = "city"
    od_path = os.path.join(
        data_dir,
        region_level,
        region_key,
        f"{region_key}_od_pairs.csv",
    )
    if not os.path.exists(od_path):
        print(f"Beijing median trip phase emissions skipped — missing {od_path}")
        return

    from drone_emission.visualization.city_input_support import (
        _save_median_trip_lifecycle_subfigures,
        _selected_trip_phase_parts,
    )
    from drone_emission.visualization.city_input_support import _select_median_trip_emission_sample

    df = pd.read_csv(od_path)
    required = {
        "distance_km",
        "road_distance_m",
        "drone_emission_kg",
        "ebike_emission_kg",
    }
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(
            "Beijing median trip phase emissions skipped — "
            f"{od_path} missing columns {missing}"
        )
        return

    selected_trip = _select_median_trip_emission_sample(df)
    production_df = (
        _selected_trip_phase_parts(selected_trip, "production")
        if "median_trip_production_emissions" in only
        else None
    )
    eol_df = (
        _selected_trip_phase_parts(selected_trip, "eol")
        if "median_trip_eol_emissions" in only
        else None
    )
    _save_median_trip_lifecycle_subfigures(
        production_df=production_df,
        eol_df=eol_df,
        median_trip_od_df=(
            df if "median_trip_use_phase_emissions" in only else None
        ),
        median_trip_city_label="Beijing",
        distance_label="",
        phase_body_colors={"E-bike": "#d95f35", "Drone": "#1f77b4"},
        phase_battery_colors={"E-bike": "#f2a07f", "Drone": "#9ecae1"},
        save_dir=out,
        for_latex=for_latex,
        plot_title=plot_title,
        only=only,
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

    from drone_emission.visualization.roadnet import plot_distance_comparison_single_panel

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
