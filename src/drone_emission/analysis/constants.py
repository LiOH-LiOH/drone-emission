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
    "roadnet_node_count": "Intersection Count",
    "roadnet_edge_count": "Road Segment Count",
    # Derived road network features
    "intersection_density": "Intersection Density (per km²)",
    "avg_block_length": "Avg Block Length (m)",
    "road_connectivity": "Road Connectivity (edges/node)",
    "road_distance_excess": "Road Distance Excess (m)",
    # OSMnx network metrics
    "dead_end_proportion": "Dead-End Proportion",
    "four_way_proportion": "4-Way Intersection Proportion",
    "orientation_entropy": "Orientation Entropy",
    "avg_street_length": "Avg Street Length (m)",
    # Demand features
    "odpair_count": "OD Pair Count",
    "valid_path_count": "Valid Path Count",
    # Density features
    "region_area_km2": "Region Area (km²)",
    "building_density": "Building Density (per km²)",
    "road_density": "Road Density (km/km²)",
    "building_volume_density": "Building Volume Density (m³/km²)",
    "building_area_ratio": "Building Area Ratio (%)",
    # Demand/sales features
    "store_count": "Store Count",
    "total_monthly_sales": "Total Monthly Sales",
    "store_density": "Store Density (per km²)",
    "sales_intensity": "Sales Intensity (per store)",
    # Energy features
    "grid_emission_factor": "Grid Emission Factor (kg CO₂e/kWh)",
    # Geographic features
    "latitude": "Latitude (°N)",
    "longitude": "Longitude (°E)",
    # Socioeconomic features
    "population_10k": "Population (10k)",
    "gdp_100m": "GDP (100M CNY)",
    "gdp_per_capita": "GDP per Capita (CNY)",
    "pop_density": "Population Density (per km²)",
    "sales_per_capita": "Sales per Capita",
}

# Chinese province name to English
PROVINCE_NAME_EN = {
    "北京": "Beijing",
    "天津": "Tianjin",
    "河北": "Hebei",
    "山西": "Shanxi",
    "内蒙古": "Inner Mongolia",
    "辽宁": "Liaoning",
    "吉林": "Jilin",
    "黑龙江": "Heilongjiang",
    "上海": "Shanghai",
    "江苏": "Jiangsu",
    "浙江": "Zhejiang",
    "安徽": "Anhui",
    "福建": "Fujian",
    "江西": "Jiangxi",
    "山东": "Shandong",
    "河南": "Henan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "海南": "Hainan",
    "重庆": "Chongqing",
    "四川": "Sichuan",
    "贵州": "Guizhou",
    "云南": "Yunnan",
    "西藏": "Tibet",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "香港": "Hong Kong",
    "澳门": "Macau",
    "台湾": "Taiwan",
    "宁夏回族": "Ningxia",
    "广西壮族": "Guangxi",
    "新疆维吾尔": "Xinjiang",
}
