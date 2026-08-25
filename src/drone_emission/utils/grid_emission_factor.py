import pandas as pd

from drone_emission.paper_cohort import load_cohort_contract


def fallback_factor() -> float:
    """Return the paper-locked national-mean fallback GEF."""
    return float(
        load_cohort_contract()["grid_emission_factor_fallback"][
            "kg_co2e_per_kwh"
        ]
    )

# 省份名称映射：combined_data.csv → gef_china.csv
PROVINCE_NAME_MAP = {
    "北京市": "北京",
    "上海市": "上海",
    "天津市": "天津",
    "重庆市": "重庆",
    "内蒙古自治区": "内蒙古",
    "广西壮族自治区": "广西",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔自治区": "新疆",
    "西藏自治区": "西藏",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
}


def normalize_province_name(province_raw: str) -> str:
    """将 combined_data.csv 中的省份名转换为 gef_china.csv 格式"""
    if province_raw in PROVINCE_NAME_MAP:
        return PROVINCE_NAME_MAP[province_raw]
    # 普通省份去掉"省"后缀
    if province_raw.endswith("省"):
        return province_raw[:-1]
    return province_raw


def fetch_factor(province: str, gef_path: str) -> float:
    """从CSV文件获取省份的电网排放因子 (kg CO2e/kWh)"""
    data = pd.read_csv(gef_path, encoding="gbk")
    match = data.loc[data["城市"] == province, "因子"]
    if not match.empty:
        return float(match.values[0])
    # Fallback: prefix match (e.g. "宁夏" is prefix of "宁夏回族")
    for _, row in data.iterrows():
        if province.startswith(row["城市"]):
            return float(row["因子"])
    fallback = fallback_factor()
    print(
        f"Warning: Province '{province}' not found in {gef_path}, using "
        f"paper fallback {fallback:.4f} kgCO2e/kWh"
    )
    return fallback
