"""Fetch open-table socioeconomic data for studied cities.

This tool keeps the project's city list authoritative and treats public web
tables as source material to be matched and audited. It is intentionally
separate from the main pipeline because the sources are external and can change
their HTML layout.

Usage:
    cd src/drone_emission
    python -m tools.fetch_socioeconomic_data --env local
"""

from __future__ import annotations

import argparse
from io import StringIO
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

from utils import load_config

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
REGIONS_JSON_PATH = os.path.join(PROJECT_ROOT, "regions.json")

GDP_PREFECTURE_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_prefecture-level_divisions_of_China_by_GDP"
)
GDP_TOP_CITY_URL = (
    "https://en.wikipedia.org/wiki/List_of_top_Chinese_cities_by_GDP"
)
GDP_MAJOR_CITY_ZH_URL = (
    "https://zh.wikipedia.org/wiki/"
    "%E4%B8%AD%E5%8D%8E%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9B%BD"
    "%E4%B8%BB%E8%A6%81%E5%9F%8E%E5%B8%82%E5%9C%B0%E5%8C%BA"
    "%E7%94%9F%E4%BA%A7%E6%80%BB%E5%80%BC%E5%88%97%E8%A1%A8"
)
URBAN_POPULATION_URL = (
    "https://zh.wikipedia.org/wiki/"
    "%E4%B8%AD%E8%8F%AF%E4%BA%BA%E6%B0%91%E5%85%B1%E5%92%8C%E5%9C%8B"
    "%E5%9F%8E%E5%B8%82%E5%9F%8E%E5%8D%80%E5%B8%B8%E4%BD%8F%E4%BA%BA"
    "%E5%8F%A3%E6%8E%92%E5%90%8D"
)
HONG_KONG_DEMOGRAPHICS_URL = "https://en.wikipedia.org/wiki/Demographics_of_Hong_Kong"
MACAU_URL = "https://en.wikipedia.org/wiki/Macau"

SOURCE_REFERENCES = {
    GDP_PREFECTURE_URL: {
        "source_id": "wiki_prefecture_gdp",
        "title": "List of prefecture-level divisions of China by GDP",
        "publisher": "Wikipedia contributors",
        "url": GDP_PREFECTURE_URL,
        "used_for": "City and prefecture-level GDP, GDP per capita, and source year.",
        "paper_note": (
            "Secondary open table compiling prefecture-level GDP and GDP per "
            "capita from local official statistical releases; use for fast "
            "screening and cite with access date."
        ),
    },
    GDP_TOP_CITY_URL: {
        "source_id": "wiki_top_city_gdp",
        "title": "List of top Chinese cities by GDP",
        "publisher": "Wikipedia contributors",
        "url": GDP_TOP_CITY_URL,
        "used_for": "Supplemental GDP values for large cities and SARs.",
        "paper_note": (
            "Secondary open table of top city GDP values; used only when the "
            "prefecture-level GDP table does not provide a matched value."
        ),
    },
    GDP_MAJOR_CITY_ZH_URL: {
        "source_id": "wiki_zh_major_city_gdp",
        "title": "中华人民共和国主要城市地区生产总值列表",
        "publisher": "Wikipedia contributors",
        "url": GDP_MAJOR_CITY_ZH_URL,
        "used_for": "Supplemental Chinese-name GDP and administrative population.",
        "paper_note": (
            "Chinese Wikipedia open table used as a Chinese-name fallback for "
            "major cities; values should be cross-checked before final citation."
        ),
    },
    URBAN_POPULATION_URL: {
        "source_id": "wiki_zh_urban_population",
        "title": "中华人民共和国城市城区常住人口排名",
        "publisher": "Wikipedia contributors",
        "url": URBAN_POPULATION_URL,
        "used_for": (
            "Urban resident population, MOHURD 2022 urban population, "
            "built-up area, and urban density fields."
        ),
        "paper_note": (
            "Secondary open table summarizing the 2020 census and 2022 MOHURD "
            "urban construction statistics; useful for urban-population "
            "context, but not complete for all project cities."
        ),
    },
    HONG_KONG_DEMOGRAPHICS_URL: {
        "source_id": "wiki_hong_kong_demographics",
        "title": "Demographics of Hong Kong",
        "publisher": "Wikipedia contributors",
        "url": HONG_KONG_DEMOGRAPHICS_URL,
        "used_for": "Hong Kong 2024 population supplement.",
        "paper_note": (
            "Used only as an open-table supplement for Hong Kong population; "
            "prefer Hong Kong Census and Statistics Department in manuscript."
        ),
    },
    MACAU_URL: {
        "source_id": "wiki_macau",
        "title": "Macau",
        "publisher": "Wikipedia contributors",
        "url": MACAU_URL,
        "used_for": "Macau 2024 population and nominal GDP supplement.",
        "paper_note": (
            "Used only as an open-table supplement for Macau; prefer official "
            "Statistics and Census Service data in manuscript."
        ),
    },
}

SAR_SUPPLEMENTS = {
    "xianggangtebiexingzhengqu": {
        "population_10k": 750.31,
        "population_source_year": 2024,
        "source_url_population": HONG_KONG_DEMOGRAPHICS_URL,
    },
    "aomentebiexingzhengqu": {
        "population_10k": 71.2651,
        "gdp_100m": 3893.85,
        "gdp_per_capita": 519276.0,
        "gdp_source_year": 2024,
        "population_source_year": 2024,
        "source_url_gdp": MACAU_URL,
        "source_url_population": MACAU_URL,
        "notes": "Macau GDP converted from nominal USD using 2024 CNY/USD rate.",
    },
}

PROVINCE_EN = {
    "安徽": "anhui",
    "北京": "beijing",
    "重庆": "chongqing",
    "福建": "fujian",
    "甘肃": "gansu",
    "广东": "guangdong",
    "广西": "guangxi",
    "贵州": "guizhou",
    "海南": "hainan",
    "河北": "hebei",
    "河南": "henan",
    "黑龙江": "heilongjiang",
    "湖北": "hubei",
    "湖南": "hunan",
    "吉林": "jilin",
    "江苏": "jiangsu",
    "江西": "jiangxi",
    "辽宁": "liaoning",
    "内蒙古": "inner mongolia",
    "宁夏": "ningxia",
    "青海": "qinghai",
    "山东": "shandong",
    "山西": "shanxi",
    "陕西": "shaanxi",
    "上海": "shanghai",
    "四川": "sichuan",
    "天津": "tianjin",
    "西藏": "tibet",
    "新疆": "xinjiang",
    "云南": "yunnan",
    "浙江": "zhejiang",
}

# Project region_name_en is pinyin-like, while Wikipedia often uses common
# English or minority-language romanizations. These aliases are used only for
# matching; unmatched rows remain visible in the audit file.
CITY_ALIASES = {
    "abazangzuqiangzuzizhizhou": "ngawa",
    "akesudiqu": "aksu",
    "alashanmeng": "alxa",
    "aleitaidiqu": "altay",
    "alidiqu": "ngari",
    "aomen": "macau",
    "bayinguoleng": "bayingolin",
    "bayinguolengmengguzizhizhou": "bayingolin",
    "bayannaoer": "bayannur",
    "boertala": "bortala",
    "boertalamengguzizhizhou": "bortala",
    "changdou": "qamdo",
    "changjihuizuzizhizhou": "changji",
    "chuxiongyizuzizhizhou": "chuxiong",
    "dalibaizuzizhizhou": "dali",
    "daxinganlingdiqu": "daxinganling",
    "dehongdaizujingpozuzizhizhou": "dehong",
    "diqingzangzuzizhizhou": "deqen",
    "eerduosi": "ordos",
    "enshitujiazumiaozuzizhizhou": "enshi",
    "ganzi": "garze",
    "ganzizangzuzizhizhou": "garze",
    "gannanzangzuzizhizhou": "gannan",
    "guoluozangzuzizhizhou": "golog",
    "haerbin": "harbin",
    "haibeizangzuzizhizhou": "haibei",
    "hainanzangzuzizhizhou": "hainan",
    "haiximengguzuzangzuzizhizhou": "haixi",
    "hetian": "hotan",
    "hetiandiqu": "hotan",
    "honghehanizuyizuzizhizhou": "honghe",
    "huhehaote": "hohhot",
    "hulunbeier": "hulunbuir",
    "huangnanzangzuzizhizhou": "huangnan",
    "kashidiqu": "kashgar",
    "kelamayi": "karamay",
    "kezileisu": "kizilsu",
    "kezileisukeerkezizizhizhou": "kizilsu",
    "lasa": "lhasa",
    "liangshanyizuzizhizhou": "liangshan",
    "linzhi": "nyingchi",
    "linxiahuizuzizhizhou": "linxia",
    "lvliang": "luliang",
    "neimenggu": "inner mongolia",
    "nujianglisuzuzizhizhou": "nujiang",
    "qiandongnanmiaozudongzuzizhizhou": "qiandongnan",
    "qiannanbuyizumiaozuzizhizhou": "qiannan",
    "qianxinanbuyizumiaozuzizhizhou": "qianxinan",
    "qiqihaer": "qiqihar",
    "rikaze": "xigaze",
    "shannan": "lhoka",
    "tacheng": "tarbagatay",
    "tachengdiqu": "tarbagatay",
    "tahe": "luohe",
    "tulufan": "turpan",
    "wenshanzhuangzumiaozuzizhizhou": "wenshan",
    "wulanchabu": "ulanqab",
    "wulumuqi": "urumqi",
    "xilinguolemeng": "xilingol",
    "xilinguoleimeng": "xilingol",
    "xiangxitujiazumiaozuzizhizhou": "xiangxi",
    "xishuangbannadaizuzizhizhou": "xishuangbanna",
    "xianggang": "hong kong",
    "xianggangtebiexingzhengqu": "hong kong",
    "xinganmeng": "hinggan",
    "yanbianchaoxianzuzizhizhou": "yanbian",
    "yilihasakezizhizhou": "ili",
    "yushuzangzuzizhizhou": "yushu",
    "zhangzhi": "changzhi",
    "zhaoyang": "chaoyang",
}

CN_REGION_SUFFIX_RE = re.compile(
    r"[市地区盟自治州回族蒙古藏族彝族苗族侗族布依族"
    r"哈萨克柯尔克孜傣族白族壮族]+$"
)


@dataclass(frozen=True)
class CityRegion:
    region_key: str
    city_name_cn: str
    city_name_en: str
    province: str


def _flatten_columns(columns) -> list[str]:
    if isinstance(columns, pd.MultiIndex):
        flat = []
        for col in columns:
            pieces = [str(x) for x in col if str(x) != "nan"]
            flat.append(" ".join(pieces))
        return flat
    return [str(c) for c in columns]


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_number(value):
    text = _clean_text(value)
    if not text:
        return pd.NA
    text = text.replace(",", "")
    text = text.replace("＞", "").replace("约", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return pd.NA
    try:
        return float(match.group(0))
    except ValueError:
        return pd.NA


def _clean_year(value):
    text = _clean_text(value)
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return int(match.group(1)) if match else pd.NA


def _norm_en(value: str) -> str:
    text = unicodedata.normalize("NFKD", _clean_text(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\(.*?\)", "", text)
    text = text.replace("'", "")
    text = text.replace("ü", "u")
    text = text.split(",")[0]
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    return CITY_ALIASES.get(text.replace(" ", ""), text)


def _norm_cn_city(value: str) -> str:
    text = _clean_text(value)
    text = text.replace("臺", "台").replace("瀋", "沈").replace("慶", "庆")
    text = text.replace("廣", "广").replace("東", "东").replace("蘇", "苏")
    text = text.replace("烏", "乌").replace("魯", "鲁").replace("齊", "齐")
    text = text.replace("爾", "尔").replace("門", "门").replace("陽", "阳")
    return CN_REGION_SUFFIX_RE.sub("", text)


def _read_tables(url: str) -> list[pd.DataFrame]:
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text), flavor="lxml")


def _source_id_for_url(url: str) -> str:
    if not url:
        return ""
    return SOURCE_REFERENCES.get(url, {}).get("source_id", "")


def _load_regions() -> list[CityRegion]:
    with open(REGIONS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    regions = []
    for key, region in sorted(data.items()):
        if region.get("region_level") != "city":
            continue
        regions.append(
            CityRegion(
                region_key=key,
                city_name_cn=region["region_name_cn"],
                city_name_en=region["region_name_en"],
                province=region.get("stats", {}).get("province", ""),
            )
        )
    return regions


def _first_col(columns: Iterable[str], patterns: Iterable[str]) -> str | None:
    lower_cols = {col: col.lower() for col in columns}
    for pattern in patterns:
        for col, low in lower_cols.items():
            if pattern.lower() in low:
                return col
    return None


def _load_prefecture_gdp() -> pd.DataFrame:
    rows = []
    for table in _read_tables(GDP_PREFECTURE_URL):
        table = table.copy()
        table.columns = _flatten_columns(table.columns)
        cols = list(table.columns)
        name_col = _first_col(cols, ["name"])
        prov_col = _first_col(cols, ["province"])
        gdp_col = _first_col(cols, ["gdp (billion cn", "gdp (billion"])
        pc_col = _first_col(cols, ["gdp per capita (cn", "gdp per capita"])
        date_col = _first_col(cols, ["date"])
        if not all([name_col, prov_col, gdp_col, pc_col]):
            continue

        for _, row in table.iterrows():
            name = _clean_text(row[name_col])
            province = _clean_text(row[prov_col])
            gdp_billion = _clean_number(row[gdp_col])
            gdp_pc = _clean_number(row[pc_col])
            if not name or pd.isna(gdp_billion):
                continue
            rows.append(
                {
                    "gdp_match_name": _norm_en(name),
                    "province_en": _norm_en(province),
                    "gdp_100m": gdp_billion * 10,
                    "gdp_per_capita": gdp_pc,
                    "gdp_source_year": _clean_year(row[date_col]) if date_col else pd.NA,
                    "source_url_gdp": GDP_PREFECTURE_URL,
                    "source_id_gdp": SOURCE_REFERENCES[GDP_PREFECTURE_URL][
                        "source_id"
                    ],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["gdp_match_name", "province_en"])


def _load_top_city_gdp() -> pd.DataFrame:
    rows = []
    for table in _read_tables(GDP_TOP_CITY_URL):
        table = table.copy()
        table.columns = _flatten_columns(table.columns)
        cols = list(table.columns)
        city_col = _first_col(cols, ["cities", "city"])
        cn_col = _first_col(cols, ["城市"])
        prov_col = _first_col(cols, ["provinces", "province"])
        gdp_cols = [
            c for c in cols if "GDP in 2024" in c and c.strip().endswith("(CNY)")
        ]
        if not city_col or not gdp_cols:
            continue
        gdp_col = gdp_cols[0]
        for _, row in table.iterrows():
            name = _clean_text(row[city_col])
            if not name:
                continue
            gdp_million = _clean_number(row[gdp_col])
            if pd.isna(gdp_million):
                continue
            city_cn = _clean_text(row[cn_col]) if cn_col else ""
            rows.append(
                {
                    "gdp_match_name": _norm_en(name),
                    "city_cn_norm": _norm_cn_city(city_cn),
                    "province_en": _norm_en(row[prov_col]) if prov_col else "",
                    "gdp_100m": gdp_million / 100,
                    "gdp_per_capita": pd.NA,
                    "gdp_source_year": 2024,
                    "source_url_gdp": GDP_TOP_CITY_URL,
                    "source_id_gdp": SOURCE_REFERENCES[GDP_TOP_CITY_URL]["source_id"],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["gdp_match_name", "province_en"])


def _load_major_city_zh_gdp() -> pd.DataFrame:
    rows = []
    for table in _read_tables(GDP_MAJOR_CITY_ZH_URL):
        table = table.copy()
        table.columns = _flatten_columns(table.columns)
        cols = list(table.columns)
        city_col = _first_col(cols, ["地级行政区", "行政区"])
        gdp_col = _first_col(cols, ["2020年GDP", "GDP"])
        pop_col = _first_col(cols, ["行政区人口", "人口"])
        pc_col = _first_col(cols, ["人均GDP"])
        if not city_col or not gdp_col:
            continue
        for _, row in table.iterrows():
            city = _clean_text(row[city_col])
            gdp_100m = _clean_number(row[gdp_col])
            if not city or pd.isna(gdp_100m):
                continue
            population = _clean_number(row[pop_col]) if pop_col else pd.NA
            rows.append(
                {
                    "city_cn_norm": _norm_cn_city(city),
                    "gdp_100m": gdp_100m,
                    "gdp_per_capita": _clean_number(row[pc_col]) if pc_col else pd.NA,
                    "population_10k_source": (
                        population / 10_000 if pd.notna(population) else pd.NA
                    ),
                    "gdp_source_year": 2020,
                    "source_url_gdp": GDP_MAJOR_CITY_ZH_URL,
                    "source_id_gdp": SOURCE_REFERENCES[GDP_MAJOR_CITY_ZH_URL][
                        "source_id"
                    ],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["city_cn_norm"])


def _load_urban_population() -> pd.DataFrame:
    rows = []
    for table in _read_tables(URBAN_POPULATION_URL):
        table = table.copy()
        table.columns = _flatten_columns(table.columns)
        cols = list(table.columns)
        city_col = _first_col(cols, ["城市 名", "城市"])
        prov_col = _first_col(cols, ["所属 省", "所屬 省", "省份"])
        pop_2020_col = _first_col(cols, ["2020年七人普", "2020"])
        pop_2022_col = _first_col(cols, ["2022年住建部", "2022"])
        built_col = _first_col(cols, ["建成区面积", "建成區面積"])
        density_col = _first_col(cols, ["人口密度"])
        if not city_col or not (pop_2020_col or pop_2022_col):
            continue
        for _, row in table.iterrows():
            city = _clean_text(row[city_col])
            if not city or city in {"城市", "名稱", "名称"}:
                continue
            rows.append(
                {
                    "city_cn_norm": _norm_cn_city(city),
                    "province": _clean_text(row[prov_col]) if prov_col else "",
                    "urban_population_2020_10k": (
                        _clean_number(row[pop_2020_col]) if pop_2020_col else pd.NA
                    ),
                    "urban_population_2022_10k": (
                        _clean_number(row[pop_2022_col]) if pop_2022_col else pd.NA
                    ),
                    "builtup_area_2022_km2": (
                        _clean_number(row[built_col]) if built_col else pd.NA
                    ),
                    "urban_density_2022_per_km2": (
                        _clean_number(row[density_col]) if density_col else pd.NA
                    ),
                    "source_url_population": URBAN_POPULATION_URL,
                    "source_id_population": SOURCE_REFERENCES[URBAN_POPULATION_URL][
                        "source_id"
                    ],
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["city_cn_norm", "province"])


def _region_match_name(region: CityRegion) -> str:
    return _norm_en(region.city_name_en)


def _lookup_gdp(
    region: CityRegion,
    gdp: pd.DataFrame,
    major_zh_gdp: pd.DataFrame,
    top_gdp: pd.DataFrame,
) -> dict:
    match_name = _region_match_name(region)
    province_en = _norm_en(PROVINCE_EN.get(region.province, region.province))
    city_cn_norm = _norm_cn_city(region.city_name_cn)

    if not gdp.empty:
        matched = gdp[
            (gdp["gdp_match_name"] == match_name)
            & ((gdp["province_en"] == province_en) | (gdp["province_en"] == ""))
        ]
        if matched.empty:
            matched = gdp[gdp["gdp_match_name"] == match_name]
        if len(matched) == 1:
            return matched.iloc[0].to_dict()
        if len(matched) > 1:
            row = matched.iloc[0].to_dict()
            row["gdp_match_warning"] = f"multiple_gdp_matches:{len(matched)}"
            return row

    if not major_zh_gdp.empty:
        matched = major_zh_gdp[major_zh_gdp["city_cn_norm"] == city_cn_norm]
        if len(matched) >= 1:
            return matched.iloc[0].to_dict()

    if not top_gdp.empty:
        matched = top_gdp[
            (top_gdp["city_cn_norm"] == city_cn_norm)
            | (top_gdp["gdp_match_name"] == match_name)
        ]
        if len(matched) >= 1:
            return matched.iloc[0].to_dict()

    return {}


def _lookup_population(region: CityRegion, urban_pop: pd.DataFrame) -> dict:
    if urban_pop.empty:
        return {}
    city_cn_norm = _norm_cn_city(region.city_name_cn)
    matched = urban_pop[urban_pop["city_cn_norm"] == city_cn_norm]
    if len(matched) == 1:
        return matched.iloc[0].to_dict()
    if len(matched) > 1:
        province_matches = matched[matched["province"].str.contains(region.province, na=False)]
        row = (province_matches if not province_matches.empty else matched).iloc[0].to_dict()
        row["population_match_warning"] = f"multiple_population_matches:{len(matched)}"
        return row
    return {}


def build_socioeconomic_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    regions = _load_regions()
    gdp = _load_prefecture_gdp()
    major_zh_gdp = _load_major_city_zh_gdp()
    top_gdp = _load_top_city_gdp()
    urban_pop = _load_urban_population()

    output_rows = []
    audit_rows = []

    for region in regions:
        gdp_row = _lookup_gdp(region, gdp, major_zh_gdp, top_gdp)
        pop_row = _lookup_population(region, urban_pop)
        supplement = SAR_SUPPLEMENTS.get(region.region_key, {})

        gdp_100m = gdp_row.get("gdp_100m", pd.NA)
        gdp_per_capita = gdp_row.get("gdp_per_capita", pd.NA)
        if pd.isna(gdp_100m) and "gdp_100m" in supplement:
            gdp_100m = supplement["gdp_100m"]
        if pd.isna(gdp_per_capita) and "gdp_per_capita" in supplement:
            gdp_per_capita = supplement["gdp_per_capita"]

        population_10k = pd.NA
        if pd.notna(gdp_100m) and pd.notna(gdp_per_capita) and gdp_per_capita:
            population_10k = (float(gdp_100m) * 100_000_000) / float(gdp_per_capita) / 10_000
        elif pd.notna(gdp_row.get("population_10k_source", pd.NA)):
            population_10k = gdp_row["population_10k_source"]
        elif pd.notna(pop_row.get("urban_population_2022_10k", pd.NA)):
            population_10k = pop_row["urban_population_2022_10k"]
        elif pd.notna(pop_row.get("urban_population_2020_10k", pd.NA)):
            population_10k = pop_row["urban_population_2020_10k"]
        elif "population_10k" in supplement:
            population_10k = supplement["population_10k"]
        if (
            pd.isna(gdp_per_capita)
            and pd.notna(gdp_100m)
            and pd.notna(population_10k)
            and population_10k
        ):
            gdp_per_capita = float(gdp_100m) * 100_000_000 / (
                float(population_10k) * 10_000
            )

        missing = []
        if not gdp_row and "gdp_100m" not in supplement:
            missing.append("missing_gdp")
        if not pop_row and "population_10k" not in supplement:
            missing.append("missing_population")
        if pd.isna(population_10k):
            missing.append("missing_legacy_population")
        warnings = [
            gdp_row.get("gdp_match_warning"),
            pop_row.get("population_match_warning"),
        ]
        warnings = [w for w in warnings if w]
        match_status = "ok" if not missing and not warnings else ";".join(missing + warnings)

        row = {
            "region_key": region.region_key,
            "city_name_cn": region.city_name_cn,
            "city_name_en": region.city_name_en,
            "province": region.province,
            "population_10k": population_10k,
            "gdp_100m": gdp_100m,
            "gdp_per_capita": gdp_per_capita,
            "gdp_source_year": gdp_row.get(
                "gdp_source_year", supplement.get("gdp_source_year", pd.NA)
            ),
            "population_source_year": (
                2022
                if pd.notna(pop_row.get("urban_population_2022_10k", pd.NA))
                else (
                    2020
                    if pd.notna(pop_row.get("urban_population_2020_10k", pd.NA))
                    else supplement.get("population_source_year", pd.NA)
                )
            ),
            "urban_population_2020_10k": pop_row.get("urban_population_2020_10k", pd.NA),
            "urban_population_2022_10k": pop_row.get("urban_population_2022_10k", pd.NA),
            "builtup_area_2022_km2": pop_row.get("builtup_area_2022_km2", pd.NA),
            "urban_density_2022_per_km2": pop_row.get("urban_density_2022_per_km2", pd.NA),
            "source_url_gdp": gdp_row.get(
                "source_url_gdp", supplement.get("source_url_gdp", "")
            ),
            "source_id_gdp": gdp_row.get(
                "source_id_gdp", _source_id_for_url(supplement.get("source_url_gdp"))
            ),
            "source_url_population": pop_row.get(
                "source_url_population",
                supplement.get("source_url_population", ""),
            ),
            "source_id_population": pop_row.get(
                "source_id_population",
                _source_id_for_url(supplement.get("source_url_population")),
            ),
            "match_status": match_status,
            "notes": supplement.get("notes", ""),
        }
        output_rows.append(row)
        if match_status != "ok":
            audit_rows.append(row)

    return pd.DataFrame(output_rows), pd.DataFrame(audit_rows)


def _source_reference_rows() -> list[dict]:
    return list(SOURCE_REFERENCES.values())


def _write_source_references(data_dir: str) -> None:
    rows = _source_reference_rows()
    csv_path = os.path.join(data_dir, "city_socioeconomic_sources.csv")
    md_path = os.path.join(data_dir, "city_socioeconomic_sources.md")

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# City Socioeconomic Data Sources\n\n")
        f.write(
            "These references document the open-table sources used by "
            "`tools.fetch_socioeconomic_data`. The generated city CSV stores "
            "`source_id_gdp` and `source_id_population` per row.\n\n"
        )
        f.write(
            "For manuscript use, prefer the official statistical releases "
            "named in the source tables when a paper-quality citation is "
            "required.\n\n"
        )
        for row in rows:
            f.write(f"## {row['source_id']}\n\n")
            f.write(f"- Title: {row['title']}\n")
            f.write(f"- Publisher: {row['publisher']}\n")
            f.write(f"- URL: {row['url']}\n")
            f.write(f"- Used for: {row['used_for']}\n")
            f.write(f"- Paper note: {row['paper_note']}\n\n")

    print(f"Wrote source references to {csv_path}")
    print(f"Wrote source references to {md_path}")


def write_outputs(data_dir: str) -> None:
    socio, audit = build_socioeconomic_table()
    out_path = os.path.join(data_dir, "city_socioeconomic.csv")
    audit_path = os.path.join(data_dir, "city_socioeconomic_audit.csv")
    socio.to_csv(out_path, index=False)
    audit.to_csv(audit_path, index=False)
    _write_source_references(data_dir)
    print(f"Wrote {len(socio)} rows to {out_path}")
    print(f"Wrote {len(audit)} audit rows to {audit_path}")
    if "match_status" in socio.columns:
        print(socio["match_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch open-table socioeconomic data for project cities"
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    write_outputs(data_dir=config["data_dir"])
