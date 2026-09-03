"""Formal config city inputs renderers used only by the manifest registry."""

import hashlib
import json
import os

import numpy as np
import pandas as pd


def _find_first_existing_path(paths):
    """Return the first existing path from a list of candidates."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _merchant_center(stores_df, *, require_nonnegative_sales=False):
    """Return the deterministic paper merchant center in WGS84."""
    from drone_emission.utils.coordinates import gcj02_to_wgs84
    from drone_emission.visualization.buildings import (
        MERCHANT_MAP_RANDOM_SEED,
        MERCHANT_MAP_TOP_SALES_FRACTION,
    )

    required = ["lat", "lng", "month_sales"]
    valid = stores_df.dropna(subset=required).copy()
    for column in required:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=required)
    if require_nonnegative_sales:
        valid = valid[valid["month_sales"] >= 0]
    if valid.empty:
        raise ValueError("no valid merchant sales rows")
    top_n = max(
        1,
        int(np.ceil(len(valid) * MERCHANT_MAP_TOP_SALES_FRACTION)),
    )
    merchant = valid.nlargest(top_n, "month_sales").sample(
        n=1,
        random_state=MERCHANT_MAP_RANDOM_SEED,
    ).iloc[0]
    return gcj02_to_wgs84(float(merchant["lng"]), float(merchant["lat"]))


def _merchant_bbox(stores_df, *, figsize, lat_span, layout="full"):
    """Match a paper merchant panel's geographic frame before geometry I/O."""
    center_lng, center_lat = _merchant_center(
        stores_df,
        require_nonnegative_sales=(layout == "sales"),
    )
    if layout == "height":
        lng_span = 0.01
    elif layout == "sales":
        map_rect = (0.03, 0.23, 0.94, 0.74)
        physical_ratio = (
            float(figsize[0]) * map_rect[2]
            / (float(figsize[1]) * map_rect[3])
        )
        lng_span = physical_ratio * lat_span / np.cos(np.radians(center_lat))
    else:
        lng_span = (
            float(figsize[0]) / float(figsize[1])
            * lat_span
            / np.cos(np.radians(center_lat))
        )
    return (
        center_lng - lng_span / 2,
        center_lat - lat_span / 2,
        center_lng + lng_span / 2,
        center_lat + lat_span / 2,
    )


def _read_buildings_bbox(path, bbox, columns=None):
    """Read only building features intersecting a WGS84 display window."""
    import geopandas as gpd
    from pyproj import CRS, Transformer

    probe = gpd.read_file(path, rows=slice(0, 1))
    source_crs = probe.crs
    if source_crs is None:
        raise ValueError(f"building geometry has no CRS: {path}")
    source_bbox = bbox
    if CRS(source_crs) != CRS("EPSG:4326"):
        source_bbox = Transformer.from_crs(
            "EPSG:4326",
            source_crs,
            always_xy=True,
        ).transform_bounds(*bbox)
    frame = gpd.read_file(path, bbox=source_bbox, columns=columns)
    if frame.crs is not None and CRS(frame.crs) != CRS("EPSG:4326"):
        frame = frame.to_crs(epsg=4326)
    return frame


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

    from drone_emission.visualization.sales import plot_merchant_monthly_sales_map

    stores_df = pd.read_csv(sales_path)
    figsize = (1.25, 1.17)
    bbox = _merchant_bbox(
        stores_df,
        figsize=figsize,
        lat_span=0.01,
        layout="sales",
    )
    buildings_gdf = _read_buildings_bbox(building_path, bbox)
    plot_merchant_monthly_sales_map(
        stores_df,
        buildings_gdf,
        "Shanghai",
        save_path=os.path.join(out, "shanghai_merchant_monthly_sales.png"),
        figsize=figsize,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def _get_cached_sparse_graph(graphml_path, graph_cache=None):
    """Load a SparseGraph from a persistent GraphML-signature cache."""
    if not os.path.exists(graphml_path):
        return None
    graph_cache = graph_cache if graph_cache is not None else {}
    if graphml_path not in graph_cache:
        from drone_emission.roadnet.sparse_graph import SparseGraph
        from drone_emission.utils.config import get_cache_dir

        source = os.path.realpath(graphml_path)
        stat = os.stat(source)
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        cache_root = get_cache_dir("visualization", "roadnet")
        os.makedirs(cache_root, exist_ok=True)
        cache_path = os.path.join(cache_root, f"{key}.npz")
        meta_path = os.path.join(cache_root, f"{key}.json")
        signature = {
            "source": source,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        cached = None
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                if metadata == signature:
                    cached = SparseGraph.from_npz(cache_path)
                    print(f"Loading cached SparseGraph: {cache_path}")
            except (OSError, ValueError, json.JSONDecodeError):
                cached = None
        if cached is None:
            cached = SparseGraph.from_graphml(graphml_path)
            cached.to_npz(cache_path)
            temporary = f"{meta_path}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(signature, handle, indent=2)
                handle.write("\n")
            os.replace(temporary, meta_path)
        graph_cache[graphml_path] = cached
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

    from drone_emission.visualization.buildings import plot_building_height_merchant_map

    stores_df = pd.read_csv(sales_path)
    od_df = None
    if os.path.exists(od_path):
        od_df = pd.read_csv(
            od_path,
            usecols=["from_lat", "from_lng", "to_lat", "to_lng", "annual_sales"],
        )
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
    figsize = (2.56, 2.56)
    bbox = _merchant_bbox(
        stores_df,
        figsize=figsize,
        lat_span=0.007,
        layout="height",
    )
    buildings_gdf = _read_buildings_bbox(
        building_path,
        bbox,
        columns=["height"],
    )
    plot_building_height_merchant_map(
        buildings_gdf,
        stores_df,
        "Shanghai",
        "上海市",
        od_df=od_df,
        road_graph=road_graph,
        save_path=os.path.join(out, "shanghai_building_height_merchant.png"),
        figsize=figsize,
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

    from drone_emission.visualization.buildings import plot_building_footprint_heatmap

    stores_df = pd.read_csv(sales_path)
    figsize = (1.25, 1.11)
    bbox = _merchant_bbox(
        stores_df,
        figsize=figsize,
        lat_span=0.008,
        layout="footprint",
    )
    buildings_gdf = _read_buildings_bbox(
        building_path,
        bbox,
        columns=["height"],
    )
    plot_building_footprint_heatmap(
        buildings_gdf,
        stores_df,
        "Shanghai",
        "上海市",
        save_path=os.path.join(out, "shanghai_building_footprint_heatmap.png"),
        figsize=figsize,
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
        from drone_emission.visualization.roadnet import (
            get_road_network_subfigure_frame,
        )

        frame = get_road_network_subfigure_frame(
            BEIJING_ROAD_NETWORK_CENTER[0],
            BEIJING_ROAD_NETWORK_CENTER[1],
            figsize=BEIJING_ROAD_NETWORK_FIGSIZE,
            lat_span=BEIJING_ROAD_NETWORK_LAT_SPAN,
        )
        bounds = frame["bounds"]
        buildings_gdf = _read_buildings_bbox(
            building_path,
            (
                bounds["min_lng"],
                bounds["min_lat"],
                bounds["max_lng"],
                bounds["max_lat"],
            ),
        )
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


def _render_beijing_p75_trip_phase_emissions(
    data_dir,
    out,
    cache_dir,
    for_latex,
    plot_title,
    only,
):
    """Render lifecycle subfigures for the shared Beijing P75 trip."""
    region_key = "beijing"
    od_path = os.path.join(
        data_dir,
        "city",
        region_key,
        f"{region_key}_od_pairs.csv",
    )
    if not os.path.exists(od_path):
        raise FileNotFoundError(
            f"missing Beijing OD input {od_path}; run cities --region beijing"
        )

    from drone_emission.visualization.city_input_support import (
        _save_beijing_p75_trip_lifecycle_subfigures,
        _selected_p75_trip_phase_parts,
    )
    from drone_emission.visualization.representative_trip import (
        select_beijing_p75_trip,
    )

    representative = select_beijing_p75_trip(od_path, cache_dir=cache_dir)
    selected_trip = representative.row
    production_df = (
        _selected_p75_trip_phase_parts(selected_trip, "production")
        if "beijing_p75_trip_production_emissions" in only
        else None
    )
    eol_df = (
        _selected_p75_trip_phase_parts(selected_trip, "eol")
        if "beijing_p75_trip_eol_emissions" in only
        else None
    )
    _save_beijing_p75_trip_lifecycle_subfigures(
        production_df=production_df,
        eol_df=eol_df,
        selected_trip=(
            selected_trip
            if "beijing_p75_trip_use_phase_emissions" in only
            else None
        ),
        selection_report=representative.selection,
        distance_label="",
        phase_body_colors={"E-bike": "#d95f35", "Drone": "#1f77b4"},
        phase_battery_colors={"E-bike": "#f2a07f", "Drone": "#9ecae1"},
        save_dir=out,
        for_latex=for_latex,
        plot_title=plot_title,
        only=only,
    )


def _render_beijing_p75_trip_route_comparison(
    data_dir,
    out,
    cache_dir,
    for_latex,
    plot_title,
    graph_cache=None,
    router_cache=None,
):
    """Render the shared Beijing P75 trip's road and air routes."""
    region_key = "beijing"
    region_dir = os.path.join(data_dir, "city", region_key)
    od_path = os.path.join(region_dir, f"{region_key}_od_pairs.csv")
    graphml_path = os.path.join(region_dir, f"{region_key}_roadnet.graphml")
    if not os.path.exists(od_path) or not os.path.exists(graphml_path):
        raise FileNotFoundError(
            "missing Beijing OD or road-network input; run cities --region beijing"
        )

    from drone_emission.visualization.representative_trip import (
        select_beijing_p75_trip,
    )
    from drone_emission.visualization.roadnet import (
        plot_beijing_p75_trip_route_comparison,
    )

    representative = select_beijing_p75_trip(od_path, cache_dir=cache_dir)
    graph = _get_cached_sparse_graph(graphml_path, graph_cache=graph_cache)
    router = _get_cached_tile_router(
        graphml_path,
        graph,
        router_cache=router_cache,
    )
    plot_beijing_p75_trip_route_comparison(
        selected_trip=representative.row,
        selection_report=representative.selection,
        graph=graph,
        router=router,
        save_path=os.path.join(out, "beijing_p75_trip_route_comparison.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )
