"""
Plot China online food delivery market size growth.

Data source: iiMedia Research via Statista
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# Project root directory
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def plot_market_size_bar(
    data_path: str = None,
    save_path: str = None,
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Plot bar chart of China online food delivery market size.

    Args:
        data_path: Path to the Excel data file. If None, uses default path.
        save_path: Path to save the figure. If None, uses default path.
        for_latex: If True, generate PDF without title for LaTeX export.
    """
    # Default paths
    if data_path is None:
        data_path = os.path.join(
            _PROJECT_ROOT,
            "data/raw/statistic_id699310_online-food-delivery-service-market-size-in-china-2017-2027.xlsx",
        )

    if save_path is None:
        save_path = os.path.join(
            _PROJECT_ROOT, "figures/market/china_food_delivery_market_size"
        )

    # Read data from Excel
    df = pd.read_excel(data_path, sheet_name="Data")

    # Extract year and market size columns (skip header rows)
    data_rows = df.iloc[4:15]  # Rows with actual data
    years = data_rows.iloc[:, 1].astype(str).tolist()
    market_size = data_rows.iloc[:, 2].astype(float).tolist()

    # Separate actual data and forecast data
    actual_years = [y for y in years if not y.endswith("*")]
    forecast_years = [y.replace("*", "") for y in years if y.endswith("*")]
    actual_size = market_size[: len(actual_years)]
    forecast_size = market_size[len(actual_years) :]

    # Create figure
    plt.rcParams["font.family"] = ["Arial", "DejaVu Sans", "sans-serif"]
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars
    x_actual = range(len(actual_years))
    x_forecast = range(len(actual_years), len(years))

    bars_actual = ax.bar(
        x_actual,
        actual_size,
        color="#2E86AB",
        edgecolor="white",
        linewidth=1.2,
        label="Actual",
    )
    bars_forecast = ax.bar(
        x_forecast,
        forecast_size,
        color="#A23B72",
        edgecolor="white",
        linewidth=1.2,
        alpha=0.7,
        hatch="//",
        label="Forecast",
    )

    # Add value labels on bars
    for bar in bars_actual:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    for bar in bars_forecast:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#A23B72",
        )

    # Set x-axis labels
    all_years = actual_years + forecast_years
    ax.set_xticks(range(len(all_years)))
    ax.set_xticklabels(all_years, fontsize=11)

    # Labels and title
    ax.set_xlabel("Year", fontsize=13)
    ax.set_ylabel("Market Size (Billion Yuan)", fontsize=13)

    if plot_title:
        ax.set_title(
            "China Online Food Delivery Market Size (2017-2027)",
            fontsize=16,
            fontweight="bold",
            pad=15,
        )

    # Add legend
    ax.legend(loc="upper left", fontsize=11)

    # Add grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Set y-axis limit
    ax.set_ylim(0, max(market_size) * 1.15)

    # Add source annotation
    if plot_title:
        ax.annotate(
            "Source: iiMedia Research via Statista",
            xy=(0.99, 0.01),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
            style="italic",
        )

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if for_latex:
        final_path = save_path + ".pdf"
    else:
        final_path = save_path + ".png"
    if plot_title:
        from ._utils import _add_title_suffix
        final_path = _add_title_suffix(final_path)
    plt.savefig(final_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {final_path}")

    plt.close()


if __name__ == "__main__":
    # Generate both PNG and PDF versions
    plot_market_size_bar(for_latex=False)
    plot_market_size_bar(for_latex=True)
