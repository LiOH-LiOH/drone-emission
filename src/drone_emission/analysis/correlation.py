"""Correlation analysis between region features and emission results."""

import math
import os

import numpy as np
import pandas as pd
from scipy import stats

from analysis.constants import VARIABLE_LABELS

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


# VARIABLE_LABELS is imported from analysis.constants and re-exported


def _default_csv_path() -> str:
    """Return default path to all_region_stats.csv."""
    return os.path.join(_PROJECT_ROOT, "data", "all_region_stats.csv")


def _calculate_region_area_km2(min_lat, max_lat, min_lng, max_lng) -> float:
    """Calculate region area in km² from lat/lng boundaries."""
    R = 6371.0
    lat1_rad = math.radians(min_lat)
    lat2_rad = math.radians(max_lat)
    height_km = R * math.radians(max_lat - min_lat)
    avg_lat_rad = (lat1_rad + lat2_rad) / 2
    width_km = R * math.radians(max_lng - min_lng) * math.cos(avg_lat_rad)
    return height_km * width_km


def load_region_stats(csv_path: str = None) -> pd.DataFrame:
    """Load region statistics from consolidated CSV.

    Args:
        csv_path: Path to all_region_stats.csv. Uses default if None.

    Returns:
        DataFrame with completed regions as rows and stats as columns.
    """
    if csv_path is None:
        csv_path = _default_csv_path()

    df = pd.read_csv(csv_path)

    # Filter to completed cities only
    df = df[df["total_ebike_lifecycle"].notna()].copy()

    # Compute derived density columns from boundary coordinates
    has_bounds = (
        df[["min_lat", "max_lat", "min_lng", "max_lng"]].notna().all(axis=1)
    )
    df.loc[has_bounds, "region_area_km2"] = df.loc[has_bounds].apply(
        lambda r: _calculate_region_area_km2(
            r["min_lat"], r["max_lat"], r["min_lng"], r["max_lng"]
        ),
        axis=1,
    )

    area = df["region_area_km2"]
    valid_area = area > 0

    if "building_count" in df.columns:
        df.loc[valid_area, "building_density"] = (
            df.loc[valid_area, "building_count"] / area[valid_area]
        )

    if "roadnet_length_total" in df.columns:
        df.loc[valid_area, "road_density"] = (
            df.loc[valid_area, "roadnet_length_total"] / 1000 / area[valid_area]
        )

    if "building_volume_total" in df.columns:
        df.loc[valid_area, "building_volume_density"] = (
            df.loc[valid_area, "building_volume_total"] / area[valid_area]
        )

    # Set index and rename
    df.set_index("region_key", inplace=True)
    df.rename(columns={"region_name_en": "region_name"}, inplace=True)

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


def load_annual_emissions(csv_path: str = None) -> pd.DataFrame:
    """Load annual emissions data from consolidated CSV.

    Args:
        csv_path: Path to all_region_stats.csv. Uses default if None.

    Returns:
        DataFrame with emission data for each region.
    """
    if csv_path is None:
        csv_path = _default_csv_path()

    df = pd.read_csv(csv_path)

    # Filter to rows with emission data
    df = df[df["total_ebike_lifecycle"].notna()].copy()

    # Compute emission columns vectorized (kg -> tonnes)
    df["all_ebike_tonnes"] = df["total_ebike_lifecycle"] / 1000.0
    df["all_drone_tonnes"] = df["total_drone_lifecycle"] / 1000.0

    # Optimal mix: apply reduction percentage
    reduction = df["max_lifecycle_reduction_pct"].fillna(0)
    df["optimal_mix_tonnes"] = df["all_ebike_tonnes"] * (1 - reduction / 100)
    df["reduction_pct"] = reduction

    df["region_name"] = df["region_name_en"]

    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for key result and feature variables."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total completed cities: {len(df)}")

    key_vars = RESULT_VARIABLES + get_all_feature_variables()
    available = [v for v in key_vars if v in df.columns]

    for var in available:
        series = df[var].dropna()
        if series.empty:
            continue
        label = VARIABLE_LABELS.get(var, var)
        print(f"\n{label} (n={len(series)}):")
        print(
            f"  min={series.min():.3f}  max={series.max():.3f}  "
            f"mean={series.mean():.3f}  median={series.median():.3f}  "
            f"std={series.std():.3f}"
        )


def run_correlation_analysis(
    csv_path: str = None,
    output_dir: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Run complete correlation analysis from consolidated CSV.

    Args:
        csv_path: Path to all_region_stats.csv. Uses default if None.
        output_dir: Directory to save figures. Uses default if None.
        for_latex: If True, export figures for LaTeX.
        plot_title: If True, show titles on plots.

    Returns:
        Dictionary with analysis results.
    """
    if output_dir is None:
        output_dir = os.path.join(_PROJECT_ROOT, "figures", "analysis")

    os.makedirs(output_dir, exist_ok=True)

    # Lazy imports to avoid circular dependency
    from visualization.analysis import (
        plot_correlation_heatmap,
        plot_emission_distribution,
        plot_feature_scatter,
        plot_geographic_pattern,
        plot_province_boxplot,
        plot_top_correlations_bar,
    )

    # Load data
    print("Loading data from CSV...")
    df = load_region_stats(csv_path)
    print(f"Loaded {len(df)} completed regions with {len(df.columns)} columns")

    # Print summary statistics
    print_summary_statistics(df)

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

    # New analyses for large sample size
    # Emission distribution
    dist_path = os.path.join(output_dir, "emission_distribution.pdf")
    plot_emission_distribution(
        df,
        save_path=dist_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )

    # Province-level boxplot
    if "province" in df.columns:
        for result_var in available_results:
            province_path = os.path.join(
                output_dir, f"province_boxplot_{result_var}.pdf"
            )
            plot_province_boxplot(
                df,
                result_var,
                save_path=province_path,
                for_latex=for_latex,
                plot_title=plot_title,
            )

    # Geographic pattern (latitude vs reduction)
    geo_path = os.path.join(output_dir, "geographic_pattern.pdf")
    plot_geographic_pattern(
        df,
        save_path=geo_path,
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
    run_correlation_analysis(for_latex=True)
