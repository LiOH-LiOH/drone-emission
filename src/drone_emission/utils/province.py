import os

import pandas as pd


def get_province_from_district(district_name_cn: str, sales_path: str) -> str:
    """
    Get province name from a region name using sales data CSV.

    Args:
        district_name_cn: Chinese region name (e.g., "南山区", "深圳市")
        sales_path: Path to sales CSV file (combined_data.csv)

    Returns:
        Province name in Chinese (e.g., "广东省", "上海市")
    """
    if not os.path.exists(sales_path):
        raise FileNotFoundError(f"Sales data not found at {sales_path}")

    # Read only necessary columns to save memory
    df = pd.read_csv(sales_path, usecols=["province", "city", "district"])

    # Try district column first
    match = df[df["district"] == district_name_cn]["province"].drop_duplicates()

    # Fall back to city column (for city-level regions like "深圳市")
    if match.empty:
        match = df[df["city"] == district_name_cn]["province"].drop_duplicates()

    if match.empty:
        raise ValueError(
            f"Region '{district_name_cn}' not found in sales data "
            f"(checked both 'district' and 'city' columns)"
        )

    return match.iloc[0]
