"""
Generate a template CSV for city-level socioeconomic data.

Reads regions.json to list all city regions, then outputs a CSV template with
project keys and city names pre-filled.

Usage:
    cd src/drone_emission
    python -m tools.generate_socioeconomic_template
    python -m tools.generate_socioeconomic_template --env local
"""

import argparse
import csv
import json
import os

from drone_emission.utils import load_config

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
REGIONS_JSON_PATH = os.path.join(PROJECT_ROOT, "regions.json")


def generate_template(data_dir: str):
    """Generate socioeconomic template CSV from regions.json."""
    with open(REGIONS_JSON_PATH, "r", encoding="utf-8") as f:
        regions = json.load(f)

    out_path = os.path.join(data_dir, "city_socioeconomic.csv")

    fieldnames = [
        "region_key",
        "city_name_cn",
        "city_name_en",
        "province",
        "population_10k",
        "gdp_100m",
        "gdp_per_capita",
        "gdp_source_year",
        "population_source_year",
        "urban_population_2020_10k",
        "urban_population_2022_10k",
        "builtup_area_2022_km2",
        "urban_density_2022_per_km2",
        "source_url_gdp",
        "source_id_gdp",
        "source_url_population",
        "source_id_population",
        "match_status",
        "notes",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, region in sorted(regions.items()):
            writer.writerow(
                {
                    "region_key": key,
                    "city_name_cn": region["region_name_cn"],
                    "city_name_en": region["region_name_en"],
                    "province": region.get("stats", {}).get("province", ""),
                    "population_10k": "",
                    "gdp_100m": "",
                    "gdp_per_capita": "",
                    "gdp_source_year": "",
                    "population_source_year": "",
                    "urban_population_2020_10k": "",
                    "urban_population_2022_10k": "",
                    "builtup_area_2022_km2": "",
                    "urban_density_2022_per_km2": "",
                    "source_url_gdp": "",
                    "source_id_gdp": "",
                    "source_url_population": "",
                    "source_id_population": "",
                    "match_status": "",
                    "notes": "",
                }
            )

    print(f"Template written to {out_path}")
    print(f"  {len(regions)} cities listed")
    print(f"  Columns: {', '.join(fieldnames)}")
    print()
    print("Fill in data from China Statistical Yearbook (国家统计局):")
    print("  population_10k: 常住人口 (万人)")
    print("  gdp_100m: GDP (亿元)")
    print("  gdp_per_capita: 人均GDP (元)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate socioeconomic template CSV"
    )
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)

    generate_template(data_dir=config["data_dir"])
