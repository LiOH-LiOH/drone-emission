"""Shared constants for analysis and visualization modules."""

# Human-readable labels for variables
VARIABLE_LABELS = {
    # Results
    "oracle_min_reduction_vs_ebike_pct": "Optimal Lifecycle Reduction (%)",
    "oracle_drone_order_ratio_pct": "Optimal Drone Order Ratio (%)",
    "lifecycle_all_drone_reduction_pct": "All-Drone Lifecycle Reduction (%)",
    "drone_minus_ebike_lifecycle_kg": "Drone - E-bike Lifecycle (kg)",
    "total_ebike_lifecycle": "Annual E-bike Lifecycle (kg)",
    "total_drone_lifecycle": "Annual Drone Lifecycle (kg)",
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

# Official English display names for China's provincial-level regions.
# Source priority for policy-sensitive names:
# - Xizang: State Council Information Office
#   https://english.scio.gov.cn/pressroom/2023-11/10/content_116807768.htm
# - Hong Kong: Hong Kong Basic Law
#   https://www.basiclaw.gov.hk/en/basiclaw/chapter1.html
# - Macao: Government of the Macao Special Administrative Region
#   https://www.gov.mo/en/content/laws/constitutional-documents/
# - Taiwan: United Nations M49 methodology
#   https://unstats.un.org/unsd/methodology/m49/
# Short names are intended for compact figure labels. Natural-geography terms
# such as "Tibetan Plateau" are separate established names and are unaffected.
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
    "西藏": "Xizang",
    "西藏自治区": "Xizang",
    "陕西": "Shaanxi",
    "甘肃": "Gansu",
    "青海": "Qinghai",
    "宁夏": "Ningxia",
    "新疆": "Xinjiang",
    "香港": "Hong Kong",
    "香港特别行政区": "Hong Kong",
    "澳门": "Macao",
    "澳门特别行政区": "Macao",
    "台湾": "Taiwan, China",
    "台湾省": "Taiwan, China",
    "宁夏回族": "Ningxia",
    "广西壮族": "Guangxi",
    "新疆维吾尔": "Xinjiang",
}

# Formal names for policy-sensitive regions. Other entries intentionally fall
# back to their accepted short display names rather than expanding every
# province, autonomous region, and municipality on space-constrained figures.
PROVINCE_NAME_EN_FORMAL = {
    "西藏": "Xizang Autonomous Region",
    "西藏自治区": "Xizang Autonomous Region",
    "香港": "Hong Kong Special Administrative Region",
    "香港特别行政区": "Hong Kong Special Administrative Region",
    "澳门": "Macao Special Administrative Region",
    "澳门特别行政区": "Macao Special Administrative Region",
    "台湾": "Taiwan Province of China",
    "台湾省": "Taiwan Province of China",
}


def province_name_en(name: str, *, formal: bool = False) -> str:
    """Return an English provincial-level-region name for display."""
    key = str(name).strip()
    if formal:
        return PROVINCE_NAME_EN_FORMAL.get(
            key,
            PROVINCE_NAME_EN.get(key, key),
        )
    return PROVINCE_NAME_EN.get(key, key)
