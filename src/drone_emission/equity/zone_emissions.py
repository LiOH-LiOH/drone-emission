"""
Zone Emission and Time Aggregation Module

Aggregates emission and delivery time data for each traffic zone.
"""

import numpy as np
import pandas as pd

from .grid import TrafficZoneGrid, assign_points_to_zones


def calculate_zone_emission_aggregates(
    grid: TrafficZoneGrid, od_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate emission and time aggregates for each zone based on delivery destinations.

    Args:
        grid: TrafficZoneGrid object
        od_df: OD pairs DataFrame with to_lat, to_lng, annual_sales,
               ebike_lifecycle_g, drone_lifecycle_g, ebike_time_min, drone_time_min columns

    Returns:
        DataFrame with zone_id, delivery_count, annual_sales,
        emission metrics, and time metrics per zone
    """
    # Get destination coordinates
    to_lats = od_df["to_lat"].values
    to_lngs = od_df["to_lng"].values

    # Assign destinations to zones
    zone_ids = assign_points_to_zones(grid, to_lats, to_lngs)

    # Get emission and sales data
    annual_sales = od_df["annual_sales"].values
    ebike_lifecycle = od_df["ebike_lifecycle_g"].values
    drone_lifecycle = od_df["drone_lifecycle_g"].values

    # Get time data (if available)
    has_time_cols = "ebike_time_min" in od_df.columns and "drone_time_min" in od_df.columns
    if has_time_cols:
        ebike_time = od_df["ebike_time_min"].values
        drone_time = od_df["drone_time_min"].values
    else:
        ebike_time = np.zeros(len(od_df))
        drone_time = np.zeros(len(od_df))

    # Aggregate by zone using groupby (single pass over data)
    tmp = pd.DataFrame({
        "zone_id": zone_ids,
        "annual_sales": annual_sales,
        "ebike_lifecycle_g": ebike_lifecycle,
        "drone_lifecycle_g": drone_lifecycle,
        "ebike_time_min": ebike_time,
        "drone_time_min": drone_time,
    })

    # Filter out-of-bounds points (zone_id == -1)
    tmp = tmp[tmp["zone_id"] >= 0]

    grouped = tmp.groupby("zone_id").agg(
        delivery_count=("annual_sales", "size"),
        annual_sales=("annual_sales", "sum"),
        ebike_lifecycle_g=("ebike_lifecycle_g", "sum"),
        drone_lifecycle_g=("drone_lifecycle_g", "sum"),
        ebike_time_min=("ebike_time_min", "sum"),
        drone_time_min=("drone_time_min", "sum"),
        avg_ebike_time_min=("ebike_time_min", "mean"),
        avg_drone_time_min=("drone_time_min", "mean"),
    )

    # Reindex to include all zones (empty zones get 0)
    all_zone_ids = np.arange(grid.total_zones())
    result = grouped.reindex(all_zone_ids, fill_value=0.0)
    result.index.name = "zone_id"
    result = result.reset_index()

    # Ensure delivery_count is integer
    result["delivery_count"] = result["delivery_count"].astype(int)

    # Compute derived columns vectorially
    result["emission_reduction_g"] = result["ebike_lifecycle_g"] - result["drone_lifecycle_g"]
    result["emission_reduction_pct"] = np.where(
        result["ebike_lifecycle_g"] > 0,
        result["emission_reduction_g"] / result["ebike_lifecycle_g"] * 100,
        0.0,
    )
    result["time_saving_min"] = result["ebike_time_min"] - result["drone_time_min"]
    result["time_saving_pct"] = np.where(
        result["ebike_time_min"] > 0,
        result["time_saving_min"] / result["ebike_time_min"] * 100,
        0.0,
    )

    # Reorder columns to match original output
    result = result[[
        "zone_id", "delivery_count", "annual_sales",
        "ebike_lifecycle_g", "drone_lifecycle_g",
        "emission_reduction_g", "emission_reduction_pct",
        "ebike_time_min", "drone_time_min",
        "time_saving_min", "time_saving_pct",
        "avg_ebike_time_min", "avg_drone_time_min",
    ]]

    return result
