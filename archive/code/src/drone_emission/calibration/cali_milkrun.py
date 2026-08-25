import os
import sys
from pathlib import Path

import pandas as pd

# Add src directory to path for imports

from drone_emission.visualization.calibration import plot_milkrun_distance_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DRONE_EMISSION_DATA_ROOT", PROJECT_ROOT / "data"))
FIGURES_ROOT = Path(os.environ.get("DRONE_EMISSION_FIGURES_ROOT", PROJECT_ROOT / "figures"))
eg_path = DATA_ROOT / "raw" / "Milkrun_data" / "order level data until 20211021.csv"
fig_dir = FIGURES_ROOT / "calibration"


def load_milkrun_data(file_path):
    """Load milkrun data and prepare distance/duration fields."""
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Total orders: {len(df)}")

    # delivery_distance_value is in meters; convert to km
    df["dist_km"] = df["delivery_distance_value"] / 1000.0
    # delivery_duration_value is in seconds; convert to minutes
    df["duration_min"] = df["delivery_duration_value"] / 60.0

    valid = df["dist_km"].notna().sum()
    print(f"Orders with distance: {valid} ({valid / len(df) * 100:.1f}%)")

    # Filter: 0 < distance <= 50 km
    mask = (df["dist_km"] > 0) & (df["dist_km"] <= 50)
    df_filtered = df[mask].copy()
    print(f"After filtering (0–50 km): {len(df_filtered)}")
    print(f"Distance range: {df_filtered['dist_km'].min():.2f} – {df_filtered['dist_km'].max():.2f} km")

    return df_filtered


if __name__ == "__main__":
    df = load_milkrun_data(eg_path)
    plot_milkrun_distance_analysis(
        df, save_path=f"{fig_dir}/milkrun_distance_analysis.png"
    )
