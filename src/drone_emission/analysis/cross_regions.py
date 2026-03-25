"""Cross-region analysis orchestration.

Reads from all_region_stats_enriched.csv (or all_region_stats.csv) and runs
all cross-region analyses: China map, correlations, emission distributions,
province comparisons, annual emission bar charts, and OLS regression.

Usage:
    cd src/drone_emission
    python -m analysis.cross_regions
    python -m analysis.cross_regions --env local
"""

import argparse
import os
import sys

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config
from visualization.regions import plot_regions_on_china_map
from visualization.analysis import (
    plot_annual_emissions_comparison,
    plot_city_emission_ranking,
    plot_emission_distribution,
    plot_province_boxplot,
)
from analysis.correlation import (
    RESULT_VARIABLES,
    load_annual_emissions,
    load_region_stats,
    run_correlation_analysis,
)
from analysis.regression import run_regression_analysis

# Project root directory (4 levels up from this file)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def _get_csv_path(data_dir: str) -> str:
    """Return path to stats CSV, preferring enriched version if available."""
    enriched = os.path.join(data_dir, "all_region_stats_enriched.csv")
    if os.path.exists(enriched):
        return enriched
    return os.path.join(data_dir, "all_region_stats.csv")


def analyze_regions(
    csv_path: str,
    save_path: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Analyze and visualize study regions on China map, color-coded by reduction."""
    plot_regions_on_china_map(
        csv_path=csv_path,
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
        color_by="max_lifecycle_reduction_pct",
    )


def analyze_correlations(
    csv_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Analyze correlations between region features and emission results."""
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    return run_correlation_analysis(
        csv_path=csv_path,
        output_dir=output_dir,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def analyze_annual_emissions(
    csv_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot annual emissions: distribution + top/bottom bar chart."""
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Distribution plot from region stats
    df = load_region_stats(csv_path)
    dist_path = os.path.join(output_dir, "emission_distribution.pdf")
    plot_emission_distribution(
        df,
        save_path=dist_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )

    # City ranking plot
    ranking_path = os.path.join(output_dir, "city_emission_ranking.pdf")
    plot_city_emission_ranking(
        df,
        save_path=ranking_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )

    # Top/bottom bar chart
    emissions_df = load_annual_emissions(csv_path)
    bar_path = os.path.join(output_dir, "annual_emissions_comparison.pdf")
    plot_annual_emissions_comparison(
        df=emissions_df,
        save_path=bar_path,
        for_latex=for_latex,
        plot_title=plot_title,
        top_n=10,
    )


def analyze_regression(
    csv_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
) -> dict:
    """Run OLS regression analysis on enriched region features."""
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    return run_regression_analysis(
        csv_path=csv_path,
        output_dir=output_dir,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def analyze_province_comparison(
    csv_path: str,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Province-level boxplot comparison for each result variable."""
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    df = load_region_stats(csv_path)

    if "province" not in df.columns:
        print("No province column found, skipping province comparison")
        return

    for result_var in RESULT_VARIABLES:
        if result_var not in df.columns:
            continue
        province_path = os.path.join(output_dir, f"province_boxplot_{result_var}.pdf")
        plot_province_boxplot(
            df,
            result_var,
            save_path=province_path,
            for_latex=for_latex,
            plot_title=plot_title,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-region analysis")
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    args = parser.parse_args()

    exp_config_path = os.path.join(PROJECT_ROOT, "experiment.toml")
    config = load_config(exp_config_path, env=args.env)
    data_dir = config["data_dir"]
    csv_path = _get_csv_path(data_dir)

    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Visualize study regions on China map
    print("=" * 60)
    print("Plotting regions on China map...")
    print("=" * 60)
    save_path = os.path.join(output_dir, "regions_map.pdf")
    analyze_regions(csv_path, save_path=save_path, for_latex=True)

    # 2. Run correlation analysis (includes scatter, heatmap, distribution, geographic)
    print("\n" + "=" * 60)
    print("Running correlation analysis...")
    print("=" * 60)
    analyze_correlations(csv_path, for_latex=True)

    # 3. Annual emissions comparison (distribution + top/bottom bar chart)
    print("\n" + "=" * 60)
    print("Plotting annual emissions comparison...")
    print("=" * 60)
    analyze_annual_emissions(csv_path, for_latex=True)

    # 4. Province-level comparison
    print("\n" + "=" * 60)
    print("Running province comparison...")
    print("=" * 60)
    analyze_province_comparison(csv_path, for_latex=True)

    # 5. OLS regression analysis
    print("\n" + "=" * 60)
    print("Running regression analysis...")
    print("=" * 60)
    analyze_regression(csv_path, for_latex=True)
