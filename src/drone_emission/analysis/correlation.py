"""Correlation analysis between region features and emission results."""

import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

from visualization.analysis import (
    plot_annual_emissions_comparison,
    plot_correlation_heatmap,
    plot_feature_scatter,
    plot_top_correlations_bar,
)

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


def run_correlation_analysis(
    region_keys: list,
    output_dir: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
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
        plot_title=plot_title,
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
                plot_title=plot_title,
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
            plot_title=plot_title,
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
