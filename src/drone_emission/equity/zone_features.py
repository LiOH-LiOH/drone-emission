"""
Zone Feature Calculation Module

Calculates urban features for each traffic zone.
"""

from typing import Dict

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

    building_lats = np.array([c.y for c in centroids])
    building_lngs = np.array([c.x for c in centroids])

    # Assign buildings to zones
    zone_ids = assign_points_to_zones(grid, building_lats, building_lngs)

    # Get building volumes
    volumes = buildings_gdf["volume_m3"].values if "volume_m3" in buildings_gdf.columns else np.zeros(len(buildings_gdf))

    # Aggregate by zone
    zone_data = []
    for zone_id in range(grid.total_zones()):
        mask = zone_ids == zone_id
        building_count = mask.sum()
        building_volume = volumes[mask].sum() if building_count > 0 else 0.0

        zone_data.append({
            "zone_id": zone_id,
            "building_count": building_count,
            "building_volume_m3": building_volume,
        })

    return pd.DataFrame(zone_data)


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
    # Extract node coordinates
    node_lats = []
    node_lngs = []
    for node in roadnet_graph.nodes():
        node_data = roadnet_graph.nodes[node]
        node_lats.append(node_data["y"])  # y is latitude
        node_lngs.append(node_data["x"])  # x is longitude

    node_lats = np.array(node_lats)
    node_lngs = np.array(node_lngs)

    # Assign nodes to zones
    zone_ids = assign_points_to_zones(grid, node_lats, node_lngs)

    # Aggregate by zone
    zone_data = []
    for zone_id in range(grid.total_zones()):
        mask = zone_ids == zone_id
        intersection_count = mask.sum()

        zone_data.append({
            "zone_id": zone_id,
            "intersection_count": intersection_count,
        })

    return pd.DataFrame(zone_data)
