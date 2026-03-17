"""Shared constants for analysis and visualization modules."""

# Human-readable labels for variables
VARIABLE_LABELS = {
    # Results
    "max_lifecycle_reduction_pct": "Max Lifecycle Reduction (%)",
    "optimal_threshold_km": "Optimal Threshold (km)",
    "optimal_drone_ratio_pct": "Optimal Drone Ratio (%)",
    # Building features
    "building_count": "Building Count",
    "building_height_mean": "Mean Building Height (m)",
    "building_height_max": "Max Building Height (m)",
    "building_volume_total": "Total Building Volume (m³)",
    "building_area_total": "Total Building Area (m²)",
    # Road network features
    "detour_ratio": "Detour Ratio",
    "avg_road_distance_m": "Avg Road Distance (m)",
    "avg_straight_distance_m": "Avg Straight Distance (m)",
    "roadnet_length_total": "Total Road Length (m)",
    # Demand features
    "odpair_count": "OD Pair Count",
    "valid_path_count": "Valid Path Count",
    # Density features
    "region_area_km2": "Region Area (km²)",
    "building_density": "Building Density (per km²)",
    "road_density": "Road Density (km/km²)",
    "building_volume_density": "Building Volume Density (m³/km²)",
    "building_area_ratio": "Building Area Ratio (%)",
}
