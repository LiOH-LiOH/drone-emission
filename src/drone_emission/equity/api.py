"""
Equity Analysis API Module

Main entry point for zone-level equity analysis.
"""

import os

import geopandas as gpd
import networkx as nx
import pandas as pd

from visualization.equity import (
    plot_emission_vs_features_scatter,
    plot_zone_correlation_matrix,
    plot_zone_heatmap,
)

from .grid import TrafficZoneGrid, create_traffic_zone_grid
from .zone_emissions import calculate_zone_emission_aggregates
from .zone_features import (
    calculate_zone_building_features,
    calculate_zone_intersection_features,
)


def _filter_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Remove outliers using IQR method on specified columns.

    Args:
        df: DataFrame to filter
        columns: List of column names to check for outliers

    Returns:
        DataFrame with outliers removed
    """
    mask = pd.Series(True, index=df.index)
    for col in columns:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            col_mask = (df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)
            mask = mask & col_mask
    return df[mask].copy()


def analyze_equity(region_info, exp_config) -> pd.DataFrame:
    """
    Perform zone-level equity analysis.

    Analyzes the relationship between urban features (building density,
    road network complexity) and emission reduction / time saving potential
    across traffic zones.

    Args:
        region_info: RegionConfig object
        exp_config: Experiment configuration dictionary

    Returns:
        DataFrame with zone statistics
    """
    region_config = region_info.to_dict()
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]

    print(f"\n{'='*60}")
    print(f"Equity Analysis for {region_name_en}")
    print(f"{'='*60}")

    # Get paths
    data_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    os.makedirs(fig_dir, exist_ok=True)

    # Get equity parameters
    equity_params = exp_config.get("equity_params", {})
    unit_lat = equity_params.get("unit_lat", 0.01)
    unit_lng = equity_params.get("unit_lng", 0.01)
    min_deliveries = equity_params.get("min_deliveries_per_zone", 10)
    exclude_outliers = equity_params.get("exclude_outliers", True)

    # Get boundaries
    boundaries_dict = region_info.get_stat("enlarged_boundaries")
    if not boundaries_dict:
        boundaries_dict = region_config.get("boundaries_dict")
    if not boundaries_dict:
        raise ValueError(f"No boundaries found for region {region_name_en}")

    print(f"Grid parameters: unit_lat={unit_lat}, unit_lng={unit_lng}")
    print(f"Boundaries: {boundaries_dict}")

    # Create traffic zone grid
    grid = create_traffic_zone_grid(boundaries_dict, unit_lat, unit_lng)
    print(f"Created grid: {grid.n_rows} rows x {grid.n_cols} cols = {grid.total_zones()} zones")

    # Load OD pairs
    od_path = os.path.join(data_dir, f"{region_name_en.lower()}_od_pairs.csv")
    od_df = pd.read_csv(od_path)
    print(f"Loaded {len(od_df)} OD pairs")

    # Check if emission columns exist
    required_cols = ["ebike_lifecycle_g", "drone_lifecycle_g"]
    if not all(col in od_df.columns for col in required_cols):
        raise ValueError(
            f"OD pairs missing emission columns. Run emission calculation first.\n"
            f"Required: {required_cols}, Found: {od_df.columns.tolist()}"
        )

    # Check if time columns exist
    has_time_cols = "ebike_time_min" in od_df.columns and "drone_time_min" in od_df.columns
    if not has_time_cols:
        print("Warning: Time columns not found. Run emission calculation to add time data.")

    # Load buildings
    buildings_path = os.path.join(
        data_dir, f"{region_name_en.lower()}_buildings_preprocessed.geojson"
    )
    if os.path.exists(buildings_path):
        buildings_gdf = gpd.read_file(buildings_path)
        print(f"Loaded {len(buildings_gdf)} buildings")
    else:
        buildings_gdf = None
        print("Warning: Buildings file not found, skipping building features")

    # Load road network
    roadnet_path = os.path.join(data_dir, f"{region_name_en.lower()}_roadnet.graphml")
    if os.path.exists(roadnet_path):
        roadnet_graph = nx.read_graphml(roadnet_path)
        # Convert node attributes to float (they may be strings after loading)
        for node in roadnet_graph.nodes():
            roadnet_graph.nodes[node]["x"] = float(roadnet_graph.nodes[node]["x"])
            roadnet_graph.nodes[node]["y"] = float(roadnet_graph.nodes[node]["y"])
        print(f"Loaded road network: {roadnet_graph.number_of_nodes()} nodes")
    else:
        roadnet_graph = None
        print("Warning: Road network file not found, skipping intersection features")

    # Calculate zone emission and time aggregates
    print("\nCalculating zone emission and time aggregates...")
    emission_df = calculate_zone_emission_aggregates(grid, od_df)

    # Calculate zone building features
    if buildings_gdf is not None:
        print("Calculating zone building features...")
        building_df = calculate_zone_building_features(grid, buildings_gdf)
        zone_stats = emission_df.merge(building_df, on="zone_id")
    else:
        zone_stats = emission_df
        zone_stats["building_count"] = 0
        zone_stats["building_volume_m3"] = 0.0

    # Calculate zone intersection features
    if roadnet_graph is not None:
        print("Calculating zone intersection features...")
        intersection_df = calculate_zone_intersection_features(grid, roadnet_graph)
        zone_stats = zone_stats.merge(intersection_df, on="zone_id")
    else:
        zone_stats["intersection_count"] = 0

    # Add zone center coordinates
    print("Adding zone center coordinates...")
    center_lats = []
    center_lngs = []
    rows = []
    cols = []
    for zone_id in zone_stats["zone_id"]:
        row, col = grid.get_row_col(zone_id)
        center_lat, center_lng = grid.get_zone_center(row, col)
        rows.append(row)
        cols.append(col)
        center_lats.append(center_lat)
        center_lngs.append(center_lng)

    zone_stats["row"] = rows
    zone_stats["col"] = cols
    zone_stats["center_lat"] = center_lats
    zone_stats["center_lng"] = center_lngs

    # Print summary statistics
    active_zones = zone_stats[zone_stats["delivery_count"] >= min_deliveries].copy()

    # Remove outliers using IQR method
    if exclude_outliers:
        outlier_cols = ["time_saving_pct", "emission_reduction_pct"]
        active_zones_filtered = _filter_outliers_iqr(active_zones, outlier_cols)
        n_outliers = len(active_zones) - len(active_zones_filtered)
        print(f"Removed {n_outliers} outlier zones using IQR method")
        active_zones = active_zones_filtered

    print(f"\n{'='*40}")
    print("Zone Statistics Summary")
    print(f"{'='*40}")
    print(f"Total zones: {len(zone_stats)}")
    print(f"Active zones (>= {min_deliveries} deliveries): {len(active_zones)}")
    print(f"Total deliveries: {zone_stats['delivery_count'].sum()}")
    print(f"Total annual sales: {zone_stats['annual_sales'].sum():.0f}")

    # Emission reduction statistics
    print(f"\nEmission reduction across active zones:")
    print(f"  Mean: {active_zones['emission_reduction_pct'].mean():.2f}%")
    print(f"  Std: {active_zones['emission_reduction_pct'].std():.2f}%")
    print(f"  Min: {active_zones['emission_reduction_pct'].min():.2f}%")
    print(f"  Max: {active_zones['emission_reduction_pct'].max():.2f}%")

    # Time saving statistics
    if has_time_cols:
        print(f"\nTime saving across active zones (drone vs ebike):")
        print(f"  Mean: {active_zones['time_saving_pct'].mean():.2f}%")
        print(f"  Std: {active_zones['time_saving_pct'].std():.2f}%")
        print(f"  Min: {active_zones['time_saving_pct'].min():.2f}%")
        print(f"  Max: {active_zones['time_saving_pct'].max():.2f}%")
        print(f"\nAverage delivery time per OD pair:")
        print(f"  Ebike: {active_zones['avg_ebike_time_min'].mean():.2f} min")
        print(f"  Drone: {active_zones['avg_drone_time_min'].mean():.2f} min")

    # Calculate and print correlations
    feature_cols = ["building_count", "building_volume_m3", "intersection_count"]
    print(f"\nCorrelations with emission_reduction_pct (n={len(active_zones)}):")
    for col in feature_cols:
        if col in active_zones.columns and active_zones[col].std() > 0:
            corr = active_zones[col].corr(active_zones["emission_reduction_pct"])
            print(f"  {col}: {corr:.3f}")

    if has_time_cols:
        print(f"\nCorrelations with time_saving_pct (n={len(active_zones)}):")
        for col in feature_cols:
            if col in active_zones.columns and active_zones[col].std() > 0:
                corr = active_zones[col].corr(active_zones["time_saving_pct"])
                print(f"  {col}: {corr:.3f}")

    # Save zone statistics
    save_path = os.path.join(data_dir, f"{region_name_en.lower()}_zone_stats.csv")
    zone_stats.to_csv(save_path, index=False)
    print(f"\nZone statistics saved to {save_path}")

    # Generate visualizations
    for_latex = exp_config.get("for_latex", False)
    print("\nGenerating visualizations...")

    # Emission reduction heatmap
    plot_zone_heatmap(
        zone_stats,
        grid,
        "emission_reduction_pct",
        f"{region_name_en} - Emission Reduction by Zone (%)",
        cmap="RdYlGn",
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_zone_emission_reduction.pdf"),
        for_latex=for_latex,
        min_deliveries=min_deliveries,
    )

    # Time saving heatmap
    if has_time_cols:
        plot_zone_heatmap(
            zone_stats,
            grid,
            "time_saving_pct",
            f"{region_name_en} - Time Saving by Zone (%)",
            cmap="RdYlGn",
            save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_zone_time_saving.pdf"),
            for_latex=for_latex,
            min_deliveries=min_deliveries,
        )

    # Building count heatmap
    if buildings_gdf is not None:
        plot_zone_heatmap(
            zone_stats,
            grid,
            "building_count",
            f"{region_name_en} - Building Count by Zone",
            cmap="YlOrRd",
            save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_zone_building_count.pdf"),
            for_latex=for_latex,
            min_deliveries=0,
        )

    # Intersection count heatmap
    if roadnet_graph is not None:
        plot_zone_heatmap(
            zone_stats,
            grid,
            "intersection_count",
            f"{region_name_en} - Intersection Count by Zone",
            cmap="Blues",
            save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_zone_intersections.pdf"),
            for_latex=for_latex,
            min_deliveries=0,
        )

    # Scatter plots: emission reduction vs features
    if buildings_gdf is not None:
        plot_emission_vs_features_scatter(
            active_zones,
            "building_count",
            "emission_reduction_pct",
            f"{region_name_en} - Emission Reduction vs Building Count",
            save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_emission_vs_buildings.pdf"),
            for_latex=for_latex,
            min_deliveries=0,  # Already filtered
        )

    if roadnet_graph is not None:
        plot_emission_vs_features_scatter(
            active_zones,
            "intersection_count",
            "emission_reduction_pct",
            f"{region_name_en} - Emission Reduction vs Intersection Count",
            save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_emission_vs_intersections.pdf"),
            for_latex=for_latex,
            min_deliveries=0,  # Already filtered
        )

    # Scatter plots: time saving vs features
    if has_time_cols:
        if buildings_gdf is not None:
            plot_emission_vs_features_scatter(
                active_zones,
                "building_count",
                "time_saving_pct",
                f"{region_name_en} - Time Saving vs Building Count",
                save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_time_vs_buildings.pdf"),
                for_latex=for_latex,
                min_deliveries=0,  # Already filtered
            )

        if roadnet_graph is not None:
            plot_emission_vs_features_scatter(
                active_zones,
                "intersection_count",
                "time_saving_pct",
                f"{region_name_en} - Time Saving vs Intersection Count",
                save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_time_vs_intersections.pdf"),
                for_latex=for_latex,
                min_deliveries=0,  # Already filtered
            )

    # Correlation matrix (include time_saving_pct if available)
    corr_columns = [
        "delivery_count",
        "annual_sales",
        "building_count",
        "building_volume_m3",
        "intersection_count",
        "emission_reduction_pct",
    ]
    if has_time_cols:
        corr_columns.append("time_saving_pct")

    plot_zone_correlation_matrix(
        active_zones,
        corr_columns,
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_zone_correlations.pdf"),
        for_latex=for_latex,
        min_deliveries=0,  # Already filtered
    )

    print(f"\nEquity analysis complete for {region_name_en}")

    return zone_stats
