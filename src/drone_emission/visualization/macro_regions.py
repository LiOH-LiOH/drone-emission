"""Shared China macro-region definitions for paper figures."""

CHINA_MACRO_REGION_SPECS = (
    {
        "key": "north_china",
        "label": "North China",
        "full_label": "North China",
        "cn_label": "华北",
        "color": "#4E79A7",
        "provinces": ("北京", "天津", "河北", "山西", "内蒙古"),
    },
    {
        "key": "northeast_china",
        "label": "Northeast",
        "full_label": "Northeast China",
        "cn_label": "东北",
        "color": "#F28E2B",
        "provinces": ("辽宁", "吉林", "黑龙江"),
    },
    {
        "key": "east_china",
        "label": "East China",
        "full_label": "East China",
        "cn_label": "华东",
        "color": "#59A14F",
        "provinces": ("上海", "江苏", "浙江", "安徽", "福建", "江西", "山东", "台湾"),
    },
    {
        "key": "central_china",
        "label": "Central China",
        "full_label": "Central China",
        "cn_label": "华中",
        "color": "#E15759",
        "provinces": ("河南", "湖北", "湖南"),
    },
    {
        "key": "south_china",
        "label": "South China",
        "full_label": "South China",
        "cn_label": "华南",
        "color": "#76B7B2",
        "provinces": ("广东", "广西", "海南", "香港", "澳门"),
    },
    {
        "key": "southwest_china",
        "label": "Southwest",
        "full_label": "Southwest China",
        "cn_label": "西南",
        "color": "#B07AA1",
        "provinces": ("重庆", "四川", "贵州", "云南", "西藏"),
    },
    {
        "key": "northwest_china",
        "label": "Northwest",
        "full_label": "Northwest China",
        "cn_label": "西北",
        "color": "#EDC948",
        "provinces": ("陕西", "甘肃", "青海", "宁夏", "新疆"),
    },
)
PROVINCE_NAME_ALIASES = {
    "广西壮族": "广西",
    "广西壮族自治区": "广西",
    "宁夏回族": "宁夏",
    "宁夏回族自治区": "宁夏",
    "新疆维吾尔": "新疆",
    "新疆维吾尔自治区": "新疆",
    "内蒙古自治区": "内蒙古",
    "西藏自治区": "西藏",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
}
PROVINCE_TO_MACRO_REGION = {
    province: spec
    for spec in CHINA_MACRO_REGION_SPECS
    for province in spec["provinces"]
}


def normalize_province_name(province: str) -> str:
    """Normalize province-level names before assigning macro-regions."""
    name = str(province).strip()
    return PROVINCE_NAME_ALIASES.get(name, name)


def macro_region_for_province(province: str) -> dict | None:
    """Return the macro-region spec for a province name, if known."""
    return PROVINCE_TO_MACRO_REGION.get(normalize_province_name(province))
