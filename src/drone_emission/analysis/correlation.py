"""Correlation analysis between region features and emission results."""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Project root directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Feature and result variable definitions
RESULT_VARIABLES = [
    "max_lifecycle_reduction_pct",
    "optimal_threshold_km",
    "optimal_drone_ratio_pct",
]

FEATURE_VARIABLES = {
    "building": [
        "building_count",
        # "building_height_mean",
        # "building_height_max",
        "building_volume_total",
        # "building_area_total",
    ],
    "roadnet": [
        "detour_ratio",
        "avg_road_distance_m",
        "avg_straight_distance_m",
        "roadnet_length_total",
    ],
    # "demand": [
    # "odpair_count",
    # "valid_path_count",
    # ],
    "density": [
        "region_area_km2",
        "building_density",
        "road_density",
        "building_volume_density",
        # "building_area_ratio",
    ],
}

# Human-readable labels for variables
VARIABLE_LABELS = {
    # Results
    "max_lifecycle_reduction_pct": "Max Lifecycle Reduction (%)",
    "optimal_threshold_km": "Optimal Threshold (km)",
    "optimal_drone_ratio_pct": "Optimal Drone Ratio (%)",
    # Building features
    "building_count": "Building Count",
    "building_height_mean": "Mean Building Height (m)",
    "building_height_max": "Max Building Height (m)",
    "building_volume_total": "Total Building Volume (m³)",
    "building_area_total": "Total Building Area (m²)",
    # Road network features
    "detour_ratio": "Detour Ratio",
    "avg_road_distance_m": "Avg Road Distance (m)",
    "avg_straight_distance_m": "Avg Straight Distance (m)",
    "roadnet_length_total": "Total Road Length (m)",
    # Demand features
    "odpair_count": "OD Pair Count",
    "valid_path_count": "Valid Path Count",
    # Density features
    "region_area_km2": "Region Area (km²)",
    "building_density": "Building Density (per km²)",
    "road_density": "Road Density (km/km²)",
    "building_volume_density": "Building Volume Density (m³/km²)",
    "building_area_ratio": "Building Area Ratio (%)",
}


def calculate_region_area_km2(boundaries_dict: dict) -> float:
    """Calculate region area in km² from lat/lng boundaries.

    Uses the haversine-based approximation for a rectangular region.

    Args:
        boundaries_dict: Dict with min_lat, max_lat, min_lng, max_lng.

    Returns:
        Area in square kilometers.
    """
    min_lat = boundaries_dict["min_lat"]
    max_lat = boundaries_dict["max_lat"]
    min_lng = boundaries_dict["min_lng"]
    max_lng = boundaries_dict["max_lng"]

    # Earth's radius in km
    R = 6371.0

    # Convert to radians
    lat1_rad = math.radians(min_lat)
    lat2_rad = math.radians(max_lat)

    # Height in km (latitude distance)
    height_km = R * math.radians(max_lat - min_lat)

    # Width in km (longitude distance at average latitude)
    avg_lat_rad = (lat1_rad + lat2_rad) / 2
    width_km = R * math.radians(max_lng - min_lng) * math.cos(avg_lat_rad)

    return height_km * width_km


def load_region_stats(region_keys: list, regions_json_path: str = None) -> pd.DataFrame:
    """Load statistics for specified regions into a DataFrame.

    Args:
        region_keys: List of region keys to load.
        regions_json_path: Path to regions.template.json. Uses default if None.

    Returns:
        DataFrame with regions as rows and stats as columns.
    """
    if regions_json_path is None:
        regions_json_path = os.path.join(_PROJECT_ROOT, "regions.template.json")

    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    rows = []
    for region_key in region_keys:
        if region_key not in regions_data:
            print(f"Warning: Region '{region_key}' not found in regions.template.json")
            continue

        region_info = regions_data[region_key]
        stats = region_info.get("stats", {})
        boundaries = region_info.get("boundaries_dict", {})

        row = {
            "region": region_key,
            "region_name": region_info.get("region_name_en", region_key),
        }
        row.update(stats)

        # Calculate region area
        if boundaries:
            area_km2 = calculate_region_area_km2(boundaries)
            row["region_area_km2"] = area_km2

            # Calculate density metrics
            if "building_count" in stats and area_km2 > 0:
                row["building_density"] = stats["building_count"] / area_km2

            if "roadnet_length_total" in stats and area_km2 > 0:
                # Convert road length from m to km for road density
                row["road_density"] = (stats["roadnet_length_total"] / 1000) / area_km2

            if "building_volume_total" in stats and area_km2 > 0:
                row["building_volume_density"] = (
                    stats["building_volume_total"] / area_km2
                )

            if "building_area_total" in stats and area_km2 > 0:
                # Building area ratio as percentage
                area_m2 = area_km2 * 1e6  # Convert km² to m²
                row["building_area_ratio"] = (
                    stats["building_area_total"] / area_m2
                ) * 100

        rows.append(row)

    df = pd.DataFrame(rows)
    df.set_index("region", inplace=True)

    return df


def get_all_feature_variables() -> list:
    """Get flat list of all feature variables."""
    all_features = []
    for features in FEATURE_VARIABLES.values():
        all_features.extend(features)
    return all_features


def compute_correlations(
    df: pd.DataFrame,
    feature_vars: list = None,
    result_vars: list = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix between features and results.

    Args:
        df: DataFrame with region statistics.
        feature_vars: List of feature variable names. Uses all if None.
        result_vars: List of result variable names. Uses defaults if None.
        method: Correlation method ('pearson' or 'spearman').

    Returns:
        DataFrame with correlation coefficients (features x results).
    """
    if feature_vars is None:
        feature_vars = get_all_feature_variables()
    if result_vars is None:
        result_vars = RESULT_VARIABLES

    # Filter to available columns
    available_features = [f for f in feature_vars if f in df.columns]
    available_results = [r for r in result_vars if r in df.columns]

    if not available_features:
        raise ValueError("No feature variables found in DataFrame")
    if not available_results:
        raise ValueError("No result variables found in DataFrame")

    # Compute correlations
    corr_data = {}
    for result_var in available_results:
        corr_data[result_var] = {}
        for feature_var in available_features:
            # Drop NaN values for this pair
            valid_mask = df[[feature_var, result_var]].notna().all(axis=1)
            x = df.loc[valid_mask, feature_var]
            y = df.loc[valid_mask, result_var]

            if len(x) < 3:
                corr_data[result_var][feature_var] = np.nan
                continue

            if method == "pearson":
                r, _ = stats.pearsonr(x, y)
            elif method == "spearman":
                r, _ = stats.spearmanr(x, y)
            else:
                raise ValueError(f"Unknown method: {method}")

            corr_data[result_var][feature_var] = r

    corr_df = pd.DataFrame(corr_data)
    corr_df.index.name = "Feature"

    return corr_df


def compute_correlation_with_pvalues(
    df: pd.DataFrame,
    feature_vars: list = None,
    result_vars: list = None,
    method: str = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute correlation matrix with p-values.

    Returns:
        Tuple of (correlation DataFrame, p-value DataFrame).
    """
    if feature_vars is None:
        feature_vars = get_all_feature_variables()
    if result_vars is None:
        result_vars = RESULT_VARIABLES

    available_features = [f for f in feature_vars if f in df.columns]
    available_results = [r for r in result_vars if r in df.columns]

    corr_data = {}
    pval_data = {}

    for result_var in available_results:
        corr_data[result_var] = {}
        pval_data[result_var] = {}

        for feature_var in available_features:
            valid_mask = df[[feature_var, result_var]].notna().all(axis=1)
            x = df.loc[valid_mask, feature_var]
            y = df.loc[valid_mask, result_var]

            if len(x) < 3:
                corr_data[result_var][feature_var] = np.nan
                pval_data[result_var][feature_var] = np.nan
                continue

            if method == "pearson":
                r, p = stats.pearsonr(x, y)
            else:
                r, p = stats.spearmanr(x, y)

            corr_data[result_var][feature_var] = r
            pval_data[result_var][feature_var] = p

    return pd.DataFrame(corr_data), pd.DataFrame(pval_data)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    title: str = "Feature-Result Correlations",
):
    """Plot correlation heatmap.

    Args:
        corr_df: Correlation DataFrame (features x results).
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, skip title and save as PDF.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get feature labels
    row_labels = [VARIABLE_LABELS.get(f, f) for f in corr_df.index]
    col_labels = [VARIABLE_LABELS.get(c, c) for c in corr_df.columns]

    # Create heatmap
    im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel(
        "Correlation Coefficient", rotation=-90, va="bottom", fontsize=11
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = corr_df.iloc[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

    if not for_latex:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_feature_scatter(
    df: pd.DataFrame,
    feature_var: str,
    result_var: str,
    save_path: str = None,
    figsize: tuple = (8, 6),
    for_latex: bool = False,
):
    """Plot scatter plot of feature vs result with regression line.

    Args:
        df: DataFrame with region statistics.
        feature_var: Feature variable name.
        result_var: Result variable name.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, skip title and save as PDF.
    """
    # Filter valid data
    valid_mask = df[[feature_var, result_var]].notna().all(axis=1)
    x = df.loc[valid_mask, feature_var].values
    y = df.loc[valid_mask, result_var].values
    labels = df.loc[valid_mask, "region_name"].values

    if len(x) < 2:
        print(f"Not enough data points for {feature_var} vs {result_var}")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(x, y, s=100, alpha=0.7, edgecolors="darkblue", linewidths=1.5, zorder=10)

    # Add region labels
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x[i], y[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    # Add regression line
    if len(x) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, "r--", alpha=0.8, linewidth=2, label=f"r = {r_value:.3f}"
        )

        # Add correlation info
        ax.legend(loc="best", fontsize=10)

    # Labels
    ax.set_xlabel(VARIABLE_LABELS.get(feature_var, feature_var), fontsize=12)
    ax.set_ylabel(VARIABLE_LABELS.get(result_var, result_var), fontsize=12)

    if not for_latex:
        title = f"{VARIABLE_LABELS.get(feature_var, feature_var)} vs {VARIABLE_LABELS.get(result_var, result_var)}"
        ax.set_title(title, fontsize=12, fontweight="bold")

    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_top_correlations_bar(
    corr_df: pd.DataFrame,
    result_var: str,
    top_n: int = 5,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
):
    """Plot bar chart of top correlated features for a result variable.

    Args:
        corr_df: Correlation DataFrame (features x results).
        result_var: Result variable to analyze.
        top_n: Number of top correlations to show.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, skip title and save as PDF.
    """
    if result_var not in corr_df.columns:
        print(f"Result variable '{result_var}' not in correlation DataFrame")
        return

    # Get correlations for this result, sorted by absolute value
    correlations = corr_df[result_var].dropna()
    sorted_corr = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )
    top_corr = sorted_corr.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar colors based on positive/negative
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_corr.values]

    # Get labels
    feature_labels = [VARIABLE_LABELS.get(f, f) for f in top_corr.index]

    # Bar plot
    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.8)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
        x_pos = val + 0.02 if val > 0 else val - 0.02
        ha = "left" if val > 0 else "right"
        ax.text(x_pos, i, f"{val:.3f}", va="center", ha=ha, fontsize=10)

    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel("Correlation Coefficient", fontsize=12)
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8)

    if not for_latex:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        ax.set_title(
            f"Top Correlations with {result_label}", fontsize=12, fontweight="bold"
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="x")
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Bar chart saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def print_correlation_summary(
    corr_df: pd.DataFrame, pval_df: pd.DataFrame = None, threshold: float = 0.5
):
    """Print summary of significant correlations.

    Args:
        corr_df: Correlation DataFrame.
        pval_df: Optional p-value DataFrame.
        threshold: Absolute correlation threshold for significance.
    """
    print("\n" + "=" * 60)
    print("CORRELATION SUMMARY")
    print("=" * 60)

    for result_var in corr_df.columns:
        result_label = VARIABLE_LABELS.get(result_var, result_var)
        print(f"\n{result_label}:")
        print("-" * 40)

        correlations = corr_df[result_var].dropna()
        sorted_corr = correlations.reindex(
            correlations.abs().sort_values(ascending=False).index
        )

        for feature_var, r in sorted_corr.items():
            feature_label = VARIABLE_LABELS.get(feature_var, feature_var)
            significance = "**" if abs(r) >= threshold else ""

            p_str = ""
            if pval_df is not None and feature_var in pval_df.index:
                p = pval_df.loc[feature_var, result_var]
                if not np.isnan(p):
                    p_str = f" (p={p:.4f})"

            print(f"  {feature_label:30s}: r = {r:+.3f}{p_str} {significance}")

    print("\n** indicates |r| >= {:.1f}".format(threshold))


def load_annual_emissions(
    region_keys: list,
    data_dir: str = None,
) -> pd.DataFrame:
    """Load annual emissions data for specified regions.

    Uses total_ebike_lifecycle and total_drone_lifecycle from regions.template.json
    if available. Otherwise falls back to reduction percentages.

    Args:
        region_keys: List of region keys to load.
        data_dir: Base data directory. Uses default if None.

    Returns:
        DataFrame with emission data for each region.
    """
    if data_dir is None:
        data_dir = os.path.join(_PROJECT_ROOT, "data")

    # Load region info for names and optimal thresholds
    regions_json_path = os.path.join(_PROJECT_ROOT, "regions.template.json")
    with open(regions_json_path, "r", encoding="utf-8") as f:
        regions_data = json.load(f)

    rows = []
    for region_key in region_keys:
        if region_key not in regions_data:
            print(f"Warning: Region '{region_key}' not found")
            continue

        region_info = regions_data[region_key]
        region_name = region_info.get("region_name_en", region_key)
        stats = region_info.get("stats", {})

        # Use total_ebike_lifecycle and total_drone_lifecycle from stats if available
        total_ebike = stats.get("total_ebike_lifecycle")
        total_drone = stats.get("total_drone_lifecycle")
        reduction_pct = stats.get("max_lifecycle_reduction_pct")

        if total_ebike is not None and total_drone is not None:
            # Convert from kg to tonnes (values are in kg)
            all_ebike_tonnes = total_ebike / 1000.0
            all_drone_tonnes = total_drone / 1000.0

            # Calculate optimal mix: use reduction percentage to derive optimal mix
            if reduction_pct is not None:
                optimal_mix_tonnes = all_ebike_tonnes * (1 - reduction_pct / 100)
            else:
                optimal_mix_tonnes = all_drone_tonnes

            rows.append(
                {
                    "region": region_key,
                    "region_name": region_name,
                    "all_ebike_tonnes": all_ebike_tonnes,
                    "all_drone_tonnes": all_drone_tonnes,
                    "optimal_mix_tonnes": optimal_mix_tonnes,
                    "reduction_pct": reduction_pct if reduction_pct else 0,
                    "has_actual_data": True,
                }
            )
        elif reduction_pct is not None:
            # Fall back to using reduction percentage with normalized values
            all_ebike_normalized = 100.0
            optimal_mix_normalized = 100.0 * (1 - reduction_pct / 100)

            rows.append(
                {
                    "region": region_key,
                    "region_name": region_name,
                    "all_ebike_tonnes": all_ebike_normalized,
                    "all_drone_tonnes": None,
                    "optimal_mix_tonnes": optimal_mix_normalized,
                    "reduction_pct": reduction_pct,
                    "has_actual_data": False,
                }
            )

    return pd.DataFrame(rows)


def plot_annual_emissions_comparison(
    region_keys: list,
    data_dir: str = None,
    save_path: str = None,
    figsize: tuple = (12, 6),
    for_latex: bool = False,
):
    """Plot grouped bar chart comparing annual emissions across regions.

    Shows three scenarios: All E-bike, All Drone, and Optimal Mix.

    Args:
        region_keys: List of region keys to analyze.
        data_dir: Base data directory.
        save_path: Path to save figure.
        figsize: Figure size tuple.
        for_latex: If True, skip title and save as PDF.
    """
    df = load_annual_emissions(region_keys, data_dir)

    if df.empty:
        print("No emission data available")
        return

    # Check if using actual data
    has_actual = (
        df["has_actual_data"].any() if "has_actual_data" in df.columns else False
    )
    has_drone_data = (
        "all_drone_tonnes" in df.columns and df["all_drone_tonnes"].notna().any()
    )

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))

    if has_drone_data and has_actual:
        # Three bars: E-bike, Drone, Optimal Mix
        width = 0.25
        ebike_color = "#e74c3c"  # Red for e-bike
        drone_color = "#3498db"  # Blue for drone
        mix_color = "#2ecc71"  # Green for optimal mix

        bars1 = ax.bar(
            x - width,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color=ebike_color,
            alpha=0.8,
        )
        bars2 = ax.bar(
            x,
            df["all_drone_tonnes"],
            width,
            label="All Drone",
            color=drone_color,
            alpha=0.8,
        )
        bars3 = ax.bar(
            x + width,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color=mix_color,
            alpha=0.8,
        )

        # Add value labels on bars
        for bar, val in zip(bars1, df["all_ebike_tonnes"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar, val in zip(bars2, df["all_drone_tonnes"]):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for bar, val, pct in zip(bars3, df["optimal_mix_tonnes"], df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}\n(-{pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ylabel = "Annual Lifecycle Emissions (tonnes CO₂e)"
    else:
        # Two bars: E-bike vs Optimal Mix (fallback)
        width = 0.35
        ebike_color = "#e74c3c"
        mix_color = "#2ecc71"

        bars1 = ax.bar(
            x - width / 2,
            df["all_ebike_tonnes"],
            width,
            label="All E-bike",
            color=ebike_color,
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            df["optimal_mix_tonnes"],
            width,
            label="Optimal Mix",
            color=mix_color,
            alpha=0.8,
        )

        # For normalized data, show reduction percentage
        for bar, pct in zip(bars2, df["reduction_pct"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"-{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ylabel = (
            "Relative Emissions (%)"
            if not has_actual
            else "Annual Emissions (tonnes CO₂e)"
        )

    ax.set_xlabel("Region", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(df["region_name"], fontsize=10)
    ax.legend(loc="upper right", fontsize=10)

    if not for_latex:
        ax.set_title(
            "Annual Food Delivery Lifecycle Emissions by Scenario",
            fontsize=14,
            fontweight="bold",
        )

    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(0, df["all_ebike_tonnes"].max() * 1.25)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Annual emissions comparison saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

    # Print summary
    print("\n=== Annual Lifecycle Emissions Summary (tonnes CO₂e) ===")
    cols = [
        "region_name",
        "all_ebike_tonnes",
        "all_drone_tonnes",
        "optimal_mix_tonnes",
        "reduction_pct",
    ]
    available_cols = [c for c in cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    return df


def run_correlation_analysis(
    region_keys: list,
    output_dir: str = None,
    for_latex: bool = False,
) -> dict:
    """Run complete correlation analysis for specified regions.

    Args:
        region_keys: List of region keys to analyze.
        output_dir: Directory to save figures. Uses default if None.
        for_latex: If True, export figures for LaTeX.

    Returns:
        Dictionary with analysis results.
    """
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "figures", "analysis")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"Loading data for regions: {region_keys}")
    df = load_region_stats(region_keys)
    print(f"Loaded {len(df)} regions with {len(df.columns)} columns")

    # Check available variables
    all_features = get_all_feature_variables()
    available_features = [f for f in all_features if f in df.columns]
    available_results = [r for r in RESULT_VARIABLES if r in df.columns]

    print(f"\nAvailable features: {len(available_features)}/{len(all_features)}")
    print(f"Available results: {len(available_results)}/{len(RESULT_VARIABLES)}")

    if not available_features or not available_results:
        print("Error: Not enough variables available for analysis")
        return {}

    # Compute correlations (both Pearson and Spearman)
    print("\nComputing Pearson correlations...")
    pearson_corr, pearson_pval = compute_correlation_with_pvalues(
        df, available_features, available_results, method="pearson"
    )

    print("Computing Spearman correlations...")
    spearman_corr, spearman_pval = compute_correlation_with_pvalues(
        df, available_features, available_results, method="spearman"
    )

    # Print summaries
    print("\n--- PEARSON CORRELATIONS ---")
    print_correlation_summary(pearson_corr, pearson_pval)

    print("\n--- SPEARMAN CORRELATIONS ---")
    print_correlation_summary(spearman_corr, spearman_pval)

    # Plot correlation heatmap
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.pdf")
    plot_correlation_heatmap(
        pearson_corr,
        save_path=heatmap_path,
        for_latex=for_latex,
        title="Pearson Correlations: Features vs Emission Results",
    )

    # Plot scatter plots for top correlations
    for result_var in available_results:
        correlations = pearson_corr[result_var].dropna()
        if correlations.empty:
            continue

        # Get top 3 correlated features
        top_features = correlations.abs().nlargest(3).index.tolist()

        for feature_var in top_features:
            scatter_path = os.path.join(
                output_dir, f"scatter_{feature_var}_vs_{result_var}.pdf"
            )
            plot_feature_scatter(
                df,
                feature_var,
                result_var,
                save_path=scatter_path,
                for_latex=for_latex,
            )

    # Plot top correlations bar charts
    for result_var in available_results:
        bar_path = os.path.join(output_dir, f"top_correlations_{result_var}.pdf")
        plot_top_correlations_bar(
            pearson_corr,
            result_var,
            top_n=min(5, len(available_features)),
            save_path=bar_path,
            for_latex=for_latex,
        )

    return {
        "df": df,
        "pearson_corr": pearson_corr,
        "pearson_pval": pearson_pval,
        "spearman_corr": spearman_corr,
        "spearman_pval": spearman_pval,
    }


if __name__ == "__main__":
    # Run analysis for study regions
    study_regions = ["chaoyang", "luohu", "gulou", "wuhou", "tianhe", "xihu"]
    run_correlation_analysis(study_regions, for_latex=True)
