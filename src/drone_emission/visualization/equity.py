"""
Equity Analysis Visualization Module

Provides visualization functions for zone-level equity analysis.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from equity.grid import TrafficZoneGrid


def plot_zone_heatmap(
    zone_stats: pd.DataFrame,
    grid: "TrafficZoneGrid",
    value_column: str,
    title: str,
    cmap: str = "YlOrRd",
    save_path: Optional[str] = None,
    for_latex: bool = False,
    min_deliveries: int = 0,
) -> Optional[str]:
    """
    Plot a choropleth heatmap of zone statistics.

    Args:
        zone_stats: DataFrame with zone_id and value columns
        grid: TrafficZoneGrid object
        value_column: Column name to visualize
        title: Plot title
        cmap: Colormap name
        save_path: Path to save figure (None to display)
        for_latex: If True, skip title for LaTeX export
        min_deliveries: Minimum deliveries to show zone (0 to show all)

    Returns:
        Save path if saved, None otherwise
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Filter zones with minimum deliveries
    if min_deliveries > 0 and "delivery_count" in zone_stats.columns:
        valid_zones = zone_stats[zone_stats["delivery_count"] >= min_deliveries]
    else:
        valid_zones = zone_stats

    # Get value range for color normalization
    values = valid_zones[value_column].values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        print(f"No valid values for {value_column}")
        plt.close()
        return None

    vmin, vmax = valid_values.min(), valid_values.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.cm.get_cmap(cmap)

    # Draw zones as rectangles
    for _, row in valid_zones.iterrows():
        zone_id = int(row["zone_id"])
        value = row[value_column]

        if np.isnan(value):
            continue

        grid_row = zone_id // grid.n_cols
        grid_col = zone_id % grid.n_cols

        # Calculate zone bounds
        zone_min_lng = grid.min_lng + grid_col * grid.unit_lng
        zone_min_lat = grid.min_lat + grid_row * grid.unit_lat

        color = cmap_obj(norm(value))
        rect = Rectangle(
            (zone_min_lng, zone_min_lat),
            grid.unit_lng,
            grid.unit_lat,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.add_patch(rect)

    # Set axis limits
    ax.set_xlim(grid.min_lng, grid.max_lng)
    ax.set_ylim(grid.min_lat, grid.max_lat)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label(value_column.replace("_", " ").title(), fontsize=12)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    if not for_latex:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_emission_vs_features_scatter(
    zone_stats: pd.DataFrame,
    feature_column: str,
    emission_column: str = "emission_reduction_pct",
    title: str = "Emission Reduction vs Urban Feature",
    save_path: Optional[str] = None,
    for_latex: bool = False,
    min_deliveries: int = 10,
) -> Optional[str]:
    """
    Plot scatter plot of emission reduction vs urban feature with regression line.

    Args:
        zone_stats: DataFrame with zone statistics
        feature_column: Column name for x-axis feature
        emission_column: Column name for y-axis (default: emission_reduction_pct)
        title: Plot title
        save_path: Path to save figure
        for_latex: If True, skip title for LaTeX export
        min_deliveries: Minimum deliveries to include zone

    Returns:
        Save path if saved, None otherwise
    """
    # Filter zones with minimum deliveries
    df = zone_stats[zone_stats["delivery_count"] >= min_deliveries].copy()

    if len(df) < 3:
        print(f"Not enough zones with >= {min_deliveries} deliveries for scatter plot")
        return None

    x = df[feature_column].values
    y = df[emission_column].values

    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < 3:
        print("Not enough valid data points for scatter plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, s=50, c="#2196F3", edgecolors="white", linewidth=0.5)

    # Linear regression
    coeffs = np.polyfit(x, y, 1)
    poly = np.poly1d(coeffs)
    x_line = np.linspace(x.min(), x.max(), 100)

    # Calculate R²
    corr = np.corrcoef(x, y)[0, 1]
    r_squared = corr ** 2

    # Plot regression line with R² in label
    ax.plot(x_line, poly(x_line), "r-", linewidth=2, label=f"Linear fit ($R^2$ = {r_squared:.3f})")

    ax.set_xlabel(feature_column.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel(emission_column.replace("_", " ").title(), fontsize=12)
    if not for_latex:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add correlation text (non-LaTeX mode only)
    if not for_latex:
        ax.text(
            0.02, 0.98,
            f"n = {len(x)}\nCorrelation: {corr:.3f}\n$R^2$: {r_squared:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_zone_correlation_matrix(
    zone_stats: pd.DataFrame,
    columns: list,
    save_path: Optional[str] = None,
    for_latex: bool = False,
    min_deliveries: int = 10,
) -> Optional[str]:
    """
    Plot correlation matrix heatmap for zone statistics.

    Args:
        zone_stats: DataFrame with zone statistics
        columns: List of column names to include in correlation
        save_path: Path to save figure
        for_latex: If True, skip title for LaTeX export
        min_deliveries: Minimum deliveries to include zone

    Returns:
        Save path if saved, None otherwise
    """
    # Filter zones with minimum deliveries
    df = zone_stats[zone_stats["delivery_count"] >= min_deliveries].copy()

    if len(df) < 3:
        print(f"Not enough zones with >= {min_deliveries} deliveries for correlation matrix")
        return None

    # Select columns and compute correlation
    valid_columns = [col for col in columns if col in df.columns]
    corr_matrix = df[valid_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation", fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(valid_columns)))
    ax.set_yticks(np.arange(len(valid_columns)))
    labels = [col.replace("_", "\n") for col in valid_columns]
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)

    # Add correlation values as text
    for i in range(len(valid_columns)):
        for j in range(len(valid_columns)):
            value = corr_matrix.iloc[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

    if not for_latex:
        ax.set_title("Zone Statistics Correlation Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None
