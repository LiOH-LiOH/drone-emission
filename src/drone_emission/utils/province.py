import os

import pandas as pd


def get_province_from_district(district_name_cn: str, sales_path: str) -> str:
    """
    Get province name from district name using sales data CSV.

    Args:
        district_name_cn: Chinese district name (e.g., "南山区", "浦东新区")
        sales_path: Path to sales CSV file (combined_data.csv)

    Returns:
        Province name in Chinese (e.g., "广东省", "上海市")
    """
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Sales data not found at {sales_path}")

    # Read only necessary columns to save memory
    df = pd.read_csv(sales_path, usecols=["province", "district"])

    # Find province for the district
    match = df[df["district"] == district_name_cn]["province"].drop_duplicates()

    if match.empty:
        raise ValueError(f"District '{district_name_cn}' not found in sales data")

    return match.iloc[0]
