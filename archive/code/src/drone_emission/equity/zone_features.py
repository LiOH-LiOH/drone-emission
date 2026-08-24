"""
Zone Feature Calculation Module

Calculates urban features for each traffic zone.
"""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .grid import TrafficZoneGrid, assign_points_to_zones


def calculate_zone_building_features(
    grid: TrafficZoneGrid, buildings_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Calculate building-related features for each zone.

    Args:
        grid: TrafficZoneGrid object
        buildings_gdf: GeoDataFrame of buildings with centroid and volume_m3 columns

    Returns:
        DataFrame with zone_id, building_count, building_volume_m3
    """
    # Ensure buildings are in WGS84
    if buildings_gdf.crs and buildings_gdf.crs.to_epsg() != 4326:
        buildings_gdf = buildings_gdf.to_crs(epsg=4326)

    # Extract building centroids
    # Use projected CRS (EPSG:3857) for accurate centroid calculation, then convert back to WGS84
    if "centroid" in buildings_gdf.columns:
        centroids = buildings_gdf["centroid"]
    else:
        buildings_projected = buildings_gdf.to_crs(epsg=3857)
        centroids_projected = buildings_projected.geometry.centroid
        centroids = gpd.GeoSeries(centroids_projected, crs="EPSG:3857").to_crs(epsg=4326)

    centroids_gs = gpd.GeoSeries(centroids)
    building_lats = centroids_gs.y.values
    building_lngs = centroids_gs.x.values

    # Assign buildings to zones
    zone_ids = assign_points_to_zones(grid, building_lats, building_lngs)

    # Get building volumes
    volumes = buildings_gdf["volume_m3"].values if "volume_m3" in buildings_gdf.columns else np.zeros(len(buildings_gdf))

    # Aggregate by zone using groupby
    tmp = pd.DataFrame({
        "zone_id": zone_ids,
        "volume_m3": volumes,
    })
    tmp = tmp[tmp["zone_id"] >= 0]

    grouped = tmp.groupby("zone_id").agg(
        building_count=("volume_m3", "size"),
        building_volume_m3=("volume_m3", "sum"),
    )

    all_zone_ids = np.arange(grid.total_zones())
    result = grouped.reindex(all_zone_ids, fill_value=0)
    result.index.name = "zone_id"
    result = result.reset_index()
    result["building_count"] = result["building_count"].astype(int)

    return result


def calculate_zone_intersection_features(
    grid: TrafficZoneGrid, roadnet_graph: nx.MultiDiGraph
) -> pd.DataFrame:
    """
    Calculate road network intersection features for each zone.

    Args:
        grid: TrafficZoneGrid object
        roadnet_graph: NetworkX graph from OSMnx (nodes have 'x' and 'y' attributes)

    Returns:
        DataFrame with zone_id, intersection_count
    """
    # Extract node coordinates using nx.get_node_attributes (handles str→float from GraphML)
    y_attrs = nx.get_node_attributes(roadnet_graph, "y")
    x_attrs = nx.get_node_attributes(roadnet_graph, "x")
    node_lats = np.array([float(v) for v in y_attrs.values()])
    node_lngs = np.array([float(v) for v in x_attrs.values()])

    # Assign nodes to zones
    zone_ids = assign_points_to_zones(grid, node_lats, node_lngs)

    # Aggregate by zone using np.bincount (fastest for simple counting with integer keys)
    valid_mask = zone_ids >= 0
    total_zones = grid.total_zones()
    counts = np.bincount(zone_ids[valid_mask], minlength=total_zones)[:total_zones]

    result = pd.DataFrame({
        "zone_id": np.arange(total_zones),
        "intersection_count": counts,
    })

    return result
