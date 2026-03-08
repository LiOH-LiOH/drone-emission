import os
import sys

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.regions import plot_regions_on_china_map
from visualization.analysis import plot_annual_emissions_comparison
from analysis.correlation import run_correlation_analysis

regions = ["shenzhen", "guangzhou", "beijing", "shanghai"]

# Project root directory (4 levels up from this file)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def analyze_regions(
    save_path: str = None, for_latex: bool = False, plot_title: bool = False
):
    """Analyze and visualize study regions on China map."""
    plot_regions_on_china_map(
        region_keys=regions,
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )


def analyze_correlations(for_latex: bool = False, plot_title: bool = False) -> dict:
    """Analyze correlations between region features and emission results.

    Args:
        for_latex: If True, export figures for LaTeX (PDF format).
        plot_title: If True, show titles on plots.

    Returns:
        Dictionary with correlation analysis results.
    """
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    return run_correlation_analysis(
        regions, output_dir=output_dir, for_latex=for_latex, plot_title=plot_title
    )


def analyze_annual_emissions(for_latex: bool = False, plot_title: bool = False):
    """Plot annual emissions comparison across regions."""
    output_dir = os.path.join(PROJECT_ROOT, "figures", "analysis")
    save_path = os.path.join(output_dir, "annual_emissions_comparison.pdf")
    return plot_annual_emissions_comparison(
        regions,
        data_dir=os.path.join(PROJECT_ROOT, "data"),
        save_path=save_path,
        for_latex=for_latex,
        plot_title=plot_title,
    )


if __name__ == "__main__":
    # Visualize study regions
    save_path = os.path.join(PROJECT_ROOT, "figures", "analysis", "regions_map.png")
    analyze_regions(save_path=save_path, for_latex=True)

    # Run correlation analysis
    print("\n" + "=" * 60)
    print("Running correlation analysis...")
    print("=" * 60)
    analyze_correlations(for_latex=True)

    # Plot annual emissions comparison
    print("\n" + "=" * 60)
    print("Plotting annual emissions comparison...")
    print("=" * 60)
    analyze_annual_emissions(for_latex=True)
