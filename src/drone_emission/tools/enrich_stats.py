"""
Enrich all_region_stats.csv with additional features for regression analysis.

Reads all_region_stats.csv and joins:
  - Store count and total monthly sales from combined_data.csv
  - Grid emission factor from gef_china.csv
  - Latitude/longitude centroids from boundary coordinates
  - Socioeconomic data from city_socioeconomic.csv (if available)

Outputs: all_region_stats_enriched.csv

Usage:
    cd src/drone_emission
    python -m tools.enrich_stats
    python -m tools.enrich_stats --env local
"""

import argparse
import math
import os

import pandas as pd

from utils import load_config
from utils.grid_emission_factor import PROVINCE_NAME_MAP, normalize_province_name

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _calculate_region_area_km2(min_lat, max_lat, min_lng, max_lng) -> float:
    """Calculate region area in km² from lat/lng boundaries."""
    R = 6371.0
    lat1_rad = math.radians(min_lat)
    lat2_rad = math.radians(max_lat)
    height_km = R * math.radians(max_lat - min_lat)
    avg_lat_rad = (lat1_rad + lat2_rad) / 2
    width_km = R * math.radians(max_lng - min_lng) * math.cos(avg_lat_rad)
    return height_km * width_km


def compute_sales_features(sales_path: str) -> pd.DataFrame:
    """Compute per-city store count and total monthly sales from combined_data.csv.

    Returns DataFrame indexed by city name (Chinese) with columns:
        store_count, total_monthly_sales
    """
    df = pd.read_csv(sales_path, usecols=["city", "month_sales"])
    grouped = df.groupby("city").agg(
        store_count=("month_sales", "count"),
        total_monthly_sales=("month_sales", "sum"),
    )
    return grouped


def load_gef_lookup(gef_path: str) -> dict:
    """Load grid emission factor lookup: province_short -> factor (kg CO2e/kWh)."""
    gef_df = pd.read_csv(gef_path, encoding="gbk")
    return dict(zip(gef_df["城市"], gef_df["因子"]))


def enrich(data_dir: str, sales_path: str, gef_path: str):
    """Enrich all_region_stats.csv and write enriched version."""
    csv_path = os.path.join(data_dir, "all_region_stats.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run consolidate_stats first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} regions from {csv_path}")

    # --- Geographic centroids ---
    has_bounds = df[["min_lat", "max_lat", "min_lng", "max_lng"]].notna().all(axis=1)
    df.loc[has_bounds, "latitude"] = (
        (df.loc[has_bounds, "min_lat"] + df.loc[has_bounds, "max_lat"]) / 2
    )
    df.loc[has_bounds, "longitude"] = (
        (df.loc[has_bounds, "min_lng"] + df.loc[has_bounds, "max_lng"]) / 2
    )

    # --- Region area (if not already computed) ---
    if "region_area_km2" not in df.columns:
        df.loc[has_bounds, "region_area_km2"] = df.loc[has_bounds].apply(
            lambda r: _calculate_region_area_km2(
                r["min_lat"], r["max_lat"], r["min_lng"], r["max_lng"]
            ),
            axis=1,
        )
    else:
        # Fill missing values
        missing_area = has_bounds & df["region_area_km2"].isna()
        df.loc[missing_area, "region_area_km2"] = df.loc[missing_area].apply(
            lambda r: _calculate_region_area_km2(
                r["min_lat"], r["max_lat"], r["min_lng"], r["max_lng"]
            ),
            axis=1,
        )

    # --- Sales features from combined_data.csv ---
    if os.path.exists(sales_path):
        print(f"Computing sales features from {sales_path}...")
        sales_features = compute_sales_features(sales_path)

        # Match by region_name_cn (city name in Chinese)
        # The CSV has region_name_cn like "南山区", but combined_data has city like "深圳市"
        # For district-level regions, we need to match via parent city
        # For city-level, match directly by region_name_cn
        # We'll use region_name_cn and try both direct match and with "市" suffix
        for idx, row in df.iterrows():
            city_cn = row.get("region_name_cn", "")
            if not city_cn:
                continue

            # Try direct match
            if city_cn in sales_features.index:
                df.loc[idx, "store_count"] = sales_features.loc[city_cn, "store_count"]
                df.loc[idx, "total_monthly_sales"] = sales_features.loc[
                    city_cn, "total_monthly_sales"
                ]
            # Try with "市" suffix (city-level regions may not have it)
            elif city_cn + "市" in sales_features.index:
                df.loc[idx, "store_count"] = sales_features.loc[
                    city_cn + "市", "store_count"
                ]
                df.loc[idx, "total_monthly_sales"] = sales_features.loc[
                    city_cn + "市", "total_monthly_sales"
                ]

        # Derived: store_density and sales_intensity
        valid_area = df["region_area_km2"] > 0
        valid_stores = df["store_count"].notna() & (df["store_count"] > 0)

        df.loc[valid_area & valid_stores, "store_density"] = (
            df.loc[valid_area & valid_stores, "store_count"]
            / df.loc[valid_area & valid_stores, "region_area_km2"]
        )
        df.loc[valid_stores, "sales_intensity"] = (
            df.loc[valid_stores, "total_monthly_sales"]
            / df.loc[valid_stores, "store_count"]
        )

        matched = df["store_count"].notna().sum()
        print(f"  Matched sales data for {matched}/{len(df)} regions")
    else:
        print(f"Warning: {sales_path} not found, skipping sales features")

    # --- Grid emission factor from gef_china.csv ---
    if os.path.exists(gef_path):
        print(f"Loading grid emission factors from {gef_path}...")
        gef_lookup = load_gef_lookup(gef_path)

        # Match by province column (already normalized short form in stats)
        if "province" in df.columns:
            df["grid_emission_factor"] = df["province"].map(gef_lookup)
            matched = df["grid_emission_factor"].notna().sum()
            print(f"  Matched GEF for {matched}/{len(df)} regions")
        else:
            print("  Warning: no 'province' column, skipping GEF join")
    else:
        print(f"Warning: {gef_path} not found, skipping GEF")

    # --- Socioeconomic data (optional) ---
    socio_path = os.path.join(data_dir, "city_socioeconomic.csv")
    if os.path.exists(socio_path):
        print(f"Loading socioeconomic data from {socio_path}...")
        socio = pd.read_csv(socio_path)

        # Join by Chinese city name
        socio_cols = [
            c
            for c in ["population_10k", "gdp_100m", "gdp_per_capita"]
            if c in socio.columns
        ]
        if socio_cols and "city_name_cn" in socio.columns:
            socio_lookup = socio.set_index("city_name_cn")[socio_cols]

            for idx, row in df.iterrows():
                city_cn = row.get("region_name_cn", "")
                if city_cn in socio_lookup.index:
                    for col in socio_cols:
                        val = socio_lookup.loc[city_cn, col]
                        if pd.notna(val):
                            df.loc[idx, col] = val

            # Derived features
            if "population_10k" in df.columns:
                valid_pop = df["population_10k"].notna() & (df["population_10k"] > 0)
                valid_area_pop = valid_area & valid_pop

                df.loc[valid_area_pop, "pop_density"] = (
                    df.loc[valid_area_pop, "population_10k"]
                    * 10000
                    / df.loc[valid_area_pop, "region_area_km2"]
                )

                if "total_monthly_sales" in df.columns:
                    valid_sales_pop = (
                        valid_pop & df["total_monthly_sales"].notna()
                    )
                    df.loc[valid_sales_pop, "sales_per_capita"] = (
                        df.loc[valid_sales_pop, "total_monthly_sales"]
                        / (df.loc[valid_sales_pop, "population_10k"] * 10000)
                    )

            matched = sum(
                df[col].notna().sum() > 0 for col in socio_cols if col in df.columns
            )
            print(f"  Joined {matched} socioeconomic columns")
        else:
            print("  Warning: socioeconomic CSV missing required columns")
    else:
        print(f"Note: {socio_path} not found, skipping socioeconomic data")

    # --- Write enriched CSV ---
    out_path = os.path.join(data_dir, "all_region_stats_enriched.csv")
    df.to_csv(out_path, index=False)
    print(f"\nEnriched CSV written to {out_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  New columns added: ", end="")
    new_cols = [
        c
        for c in [
            "latitude",
            "longitude",
            "store_count",
            "total_monthly_sales",
            "store_density",
            "sales_intensity",
            "grid_emission_factor",
            "population_10k",
            "gdp_100m",
            "gdp_per_capita",
            "pop_density",
            "sales_per_capita",
        ]
        if c in df.columns
    ]
    print(", ".join(new_cols))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich all_region_stats.csv")
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)

    enrich(
        data_dir=config["data_dir"],
        sales_path=config["sales_path"],
        gef_path=config["gef_path"],
    )
