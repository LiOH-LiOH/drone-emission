import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

eg_path = "/Users/linjingxian/Code/drone-emission/data/raw/Milkrun_data/order level data until 20211021.csv"
fig_dir = "/Users/linjingxian/Code/drone-emission/figures/calibration"


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


def plot_milkrun_distance_analysis(df, save_path=None):
    """Create a 1×3 figure: histogram, CDF, distance-vs-time scatter."""
    dist_km = df["dist_km"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Histogram ---
    ax = axes[0]
    bin_edges = np.arange(0, dist_km.max() + 1, 1)
    ax.hist(dist_km, bins=bin_edges, color="#4C72B0", edgecolor="white", alpha=0.85)

    median_d = dist_km.median()
    mean_d = dist_km.mean()
    ax.axvline(median_d, color="red", ls="--", lw=1, label=f"Median: {median_d:.1f} km")
    ax.axvline(mean_d, color="orange", ls="--", lw=1, label=f"Mean: {mean_d:.1f} km")
    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Order Count")
    ax.set_title("Distance Distribution")
    ax.legend(fontsize=8)

    # --- Panel 2: CDF ---
    ax = axes[1]
    sorted_d = np.sort(dist_km.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, color="#4C72B0", lw=1.5)

    percentiles = [25, 50, 75, 90]
    pct_values = np.percentile(sorted_d, percentiles)
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for p, v, c in zip(percentiles, pct_values, colors):
        ax.axhline(p / 100, color=c, ls=":", lw=0.8, alpha=0.6)
        ax.axvline(v, color=c, ls=":", lw=0.8, alpha=0.6)
        ax.plot(v, p / 100, "o", color=c, ms=5)
        ax.annotate(f"P{p}: {v:.1f} km", xy=(v, p / 100), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Distance CDF")
    ax.set_ylim(0, 1.05)

    # --- Panel 3: Distance vs Time scatter ---
    ax = axes[2]
    # Need both distance and duration
    valid = df[["dist_km", "duration_min"]].dropna()
    valid = valid[(valid["duration_min"] > 0) & (valid["duration_min"] <= 180)]
    d = valid["dist_km"].values
    t = valid["duration_min"].values

    ax.scatter(d, t, s=1, alpha=0.05, color="#4C72B0", rasterized=True)

    # Linear fit
    slope, intercept, r, _, _ = stats.linregress(d, t)
    x_fit = np.linspace(0, d.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5,
            label=f"Fit: {slope:.1f} min/km + {intercept:.1f} min\n"
                  f"→ avg speed ≈ {60 / slope:.0f} km/h  (R²={r**2:.2f})")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Delivery Duration (min)")
    ax.set_title("Distance vs Delivery Time")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    df = load_milkrun_data(eg_path)
    plot_milkrun_distance_analysis(
        df, save_path=f"{fig_dir}/milkrun_distance_analysis.png"
    )
