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

    # Aggregate by zone
    zone_data = []
    for zone_id in range(grid.total_zones()):
        mask = zone_ids == zone_id
        delivery_count = mask.sum()

        if delivery_count > 0:
            zone_annual_sales = annual_sales[mask].sum()
            zone_ebike_lifecycle = ebike_lifecycle[mask].sum()
            zone_drone_lifecycle = drone_lifecycle[mask].sum()
            emission_reduction_g = zone_ebike_lifecycle - zone_drone_lifecycle
            emission_reduction_pct = (
                (emission_reduction_g / zone_ebike_lifecycle * 100)
                if zone_ebike_lifecycle > 0
                else 0.0
            )

            # Time aggregates (total delivery time in zone)
            zone_ebike_time = ebike_time[mask].sum()
            zone_drone_time = drone_time[mask].sum()
            time_saving_min = zone_ebike_time - zone_drone_time  # positive = drone faster
            time_saving_pct = (
                (time_saving_min / zone_ebike_time * 100)
                if zone_ebike_time > 0
                else 0.0
            )

            # Average delivery time per OD pair in zone
            avg_ebike_time = ebike_time[mask].mean()
            avg_drone_time = drone_time[mask].mean()
        else:
            zone_annual_sales = 0.0
            zone_ebike_lifecycle = 0.0
            zone_drone_lifecycle = 0.0
            emission_reduction_g = 0.0
            emission_reduction_pct = 0.0
            zone_ebike_time = 0.0
            zone_drone_time = 0.0
            time_saving_min = 0.0
            time_saving_pct = 0.0
            avg_ebike_time = 0.0
            avg_drone_time = 0.0

        zone_data.append({
            "zone_id": zone_id,
            "delivery_count": delivery_count,
            "annual_sales": zone_annual_sales,
            # Emission metrics
            "ebike_lifecycle_g": zone_ebike_lifecycle,
            "drone_lifecycle_g": zone_drone_lifecycle,
            "emission_reduction_g": emission_reduction_g,
            "emission_reduction_pct": emission_reduction_pct,
            # Time metrics (total)
            "ebike_time_min": zone_ebike_time,
            "drone_time_min": zone_drone_time,
            "time_saving_min": time_saving_min,
            "time_saving_pct": time_saving_pct,
            # Time metrics (average per delivery)
            "avg_ebike_time_min": avg_ebike_time,
            "avg_drone_time_min": avg_drone_time,
        })

    return pd.DataFrame(zone_data)
