"""
Auto-discover cities from combined_data.csv and generate regions.json entries.

Usage:
    cd src/drone_emission
    python -m tools.generate_city_configs [--sales-path PATH] [--dry-run]
"""

import argparse
import json
import os
import time

import cpca
import pandas as pd
import requests
from pypinyin import lazy_pinyin, Style

from drone_emission.utils.config import get_cache_dir
from drone_emission.utils.coordinates import gcj02_to_wgs84
from drone_emission.utils import load_config

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
REGIONS_JSON_PATH = os.path.join(PROJECT_ROOT, "regions.json")
EXP_CONFIG_PATH = os.path.join(PROJECT_ROOT, "experiment.toml")

# Special pinyin mappings for cities where pypinyin gives poor results
PINYIN_OVERRIDES = {
    "重庆市": "Chongqing",
    "厦门市": "Xiamen",
    "长沙市": "Changsha",
    "长春市": "Changchun",
}


def get_adcode(city_name_cn: str):
    """Get adcode using cpca library."""
    try:
        df = cpca.transform([city_name_cn])
        if not df.empty:
            return df.iloc[0]["adcode"]
    except Exception as e:
        print(f"Error getting adcode for {city_name_cn}: {e}")
    return None


def fetch_boundaries(adcode: str):
    """Fetch GeoJSON boundaries from Aliyun DataV API with local cache."""
    cache_file = os.path.join(get_cache_dir("boundaries"), f"{adcode}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache {cache_file}: {e}")

    url = f"https://geo.datav.aliyun.com/areas_v3/bound/{adcode}_full.json"
    try:
        print(f"Downloading boundaries for adcode {adcode}...")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return data
    except Exception as e:
        print(f"Failed to fetch boundaries: {e}")
        return None


def city_to_english(city_name_cn: str) -> str:
    """Convert Chinese city name to English using pypinyin.

    Returns capitalized pinyin, e.g. "深圳市" -> "Shenzhen".
    """
    if city_name_cn in PINYIN_OVERRIDES:
        return PINYIN_OVERRIDES[city_name_cn]

    # Strip trailing 市
    name = city_name_cn.rstrip("市")
    # Get pinyin letters
    parts = lazy_pinyin(name, style=Style.NORMAL)
    # Capitalize first letter of entire name (treat as one word)
    english = "".join(parts).capitalize()
    return english


def city_to_key(city_name_cn: str) -> str:
    """Convert Chinese city name to region key (lowercase pinyin).

    e.g. "深圳市" -> "shenzhen"
    """
    name = city_name_cn.rstrip("市")
    parts = lazy_pinyin(name, style=Style.NORMAL)
    return "".join(parts).lower()


def extract_bbox_from_geojson(geojson_data: dict) -> dict | None:
    """Extract bounding box from GeoJSON, converting GCJ02 to WGS84."""
    all_lngs = []
    all_lats = []

    for feature in geojson_data.get("features", []):
        geometry = feature.get("geometry")
        if not geometry:
            continue

        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        def collect_ring(ring):
            for pt in ring:
                wgs_lng, wgs_lat = gcj02_to_wgs84(pt[0], pt[1])
                all_lngs.append(wgs_lng)
                all_lats.append(wgs_lat)

        if geom_type == "Polygon":
            for ring in coords:
                collect_ring(ring)
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    collect_ring(ring)

    if not all_lngs:
        return None

    return {
        "min_lat": min(all_lats),
        "max_lat": max(all_lats),
        "min_lng": min(all_lngs),
        "max_lng": max(all_lngs),
    }


def load_existing_regions() -> dict:
    """Load existing regions.json if it exists."""
    if os.path.exists(REGIONS_JSON_PATH):
        with open(REGIONS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_regions(data: dict):
    """Save regions.json."""
    with open(REGIONS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved regions.json with {len(data)} entries.")


def write_region_stats_file(data_dir: str, region_key: str, entry: dict):
    """Write a per-region stats file with province and any existing stats."""
    level = entry["region_level"]
    name_en = entry["region_name_en"].lower()
    stats_path = os.path.join(data_dir, level, name_en, "region_stats.json")

    # Merge with existing file if present
    existing_stats = {}
    existing_cache = {}
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_stats = existing.get("stats", {})
        existing_cache = existing.get("cache", {})

    merged_stats = {**entry.get("stats", {}), **existing_stats}
    merged_cache = {**entry.get("cache", {}), **existing_cache}

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"stats": merged_stats, "cache": merged_cache}, f, ensure_ascii=False, indent=2)


def generate_configs(sales_path: str, dry_run: bool = False, data_dir: str = None):
    """Main function: discover cities and generate region configs."""
    print(f"Reading sales data from: {sales_path}")
    df = pd.read_csv(sales_path, usecols=["province", "city"])

    # Extract unique cities with their provinces (drop rows with NaN city names)
    city_province = df.dropna(subset=["city"]).drop_duplicates(subset=["city"]).set_index("city")["province"]
    cities = sorted(city_province.index.tolist())
    print(f"Found {len(cities)} unique cities in combined_data.csv")

    # Load existing regions
    existing = load_existing_regions()
    print(f"Existing regions.json has {len(existing)} entries")

    added = 0
    skipped_existing = 0
    skipped_no_adcode = 0
    skipped_no_boundary = 0
    errors = []

    for i, city_cn in enumerate(cities, 1):
        region_key = city_to_key(city_cn)
        print(f"\n[{i}/{len(cities)}] {city_cn} -> {region_key}")

        # Skip if already exists with boundaries and stats
        if region_key in existing and existing[region_key].get("boundaries_dict"):
            print(f"  Skipping (already configured)")
            skipped_existing += 1
            continue

        # Get adcode
        adcode = get_adcode(city_cn)
        if not adcode:
            print(f"  WARNING: No adcode found, skipping")
            skipped_no_adcode += 1
            errors.append((city_cn, "no adcode"))
            continue

        # Ensure adcode is a string and truncate to city level (6 digits)
        adcode_str = str(int(adcode))[:6]

        # Skip province-level entries (adcode ending in 0000)
        if adcode_str.endswith("0000"):
            print(f"  WARNING: Province-level adcode {adcode_str}, skipping")
            skipped_no_adcode += 1
            errors.append((city_cn, f"province-level adcode {adcode_str}"))
            continue

        # Fetch boundary GeoJSON
        geojson = fetch_boundaries(adcode_str)
        if not geojson:
            print(f"  WARNING: No boundary data, skipping")
            skipped_no_boundary += 1
            errors.append((city_cn, "no boundary"))
            continue

        # Extract bounding box
        bbox = extract_bbox_from_geojson(geojson)
        if not bbox:
            print(f"  WARNING: Could not extract bbox, skipping")
            skipped_no_boundary += 1
            errors.append((city_cn, "empty bbox"))
            continue

        english_name = city_to_english(city_cn)
        province = city_province[city_cn]
        # Normalize to gef_china.csv format (e.g. 广西壮族自治区 -> 广西)
        from drone_emission.utils.grid_emission_factor import normalize_province_name
        province_short = normalize_province_name(province)

        # Build region entry, preserving existing stats/cache if present
        old_entry = existing.get(region_key, {})
        entry = {
            "region_name_cn": city_cn,
            "region_name_en": english_name,
            "region_level": "city",
            "parent_city": None,
            "boundaries_dict": bbox,
            "stats": old_entry.get("stats", {}),
            "cache": old_entry.get("cache", {}),
        }

        # Store province in stats if not already present
        if "province" not in entry["stats"]:
            entry["stats"]["province"] = province_short

        existing[region_key] = entry

        # Write per-region stats file (province stat needed by emission stage)
        if not dry_run and data_dir:
            write_region_stats_file(data_dir, region_key, entry)

        added += 1
        print(f"  Added: {english_name} ({adcode_str}) bbox={bbox}")

        # Small delay to be polite to the API
        time.sleep(0.1)

    print("\n" + "=" * 60)
    print(f"Summary:")
    print(f"  Cities found:         {len(cities)}")
    print(f"  Added/updated:        {added}")
    print(f"  Skipped (existing):   {skipped_existing}")
    print(f"  Skipped (no adcode):  {skipped_no_adcode}")
    print(f"  Skipped (no boundary):{skipped_no_boundary}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for city, reason in errors:
            print(f"  {city}: {reason}")

    if not dry_run:
        save_regions(existing)
    else:
        print("\n[DRY RUN] No changes written to disk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate region configs for all cities in combined_data.csv"
    )
    parser.add_argument(
        "--sales-path",
        type=str,
        default=None,
        help="Path to combined_data.csv (default: from experiment.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk",
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    config = load_config(EXP_CONFIG_PATH, env=args.env)
    from drone_emission.utils.config import configure_cache
    configure_cache(config)
    sales_path = args.sales_path or config.get("sales_path")
    generate_configs(sales_path, args.dry_run, data_dir=config.get("data_dir"))
