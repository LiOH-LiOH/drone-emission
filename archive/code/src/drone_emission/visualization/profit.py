"""Profit analysis visualization module."""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _add_title_suffix


def plot_profit_comparison(
    od_df: pd.DataFrame,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """Plot profit distribution histograms and drone profit advantage vs distance.

    Args:
        od_df: DataFrame with ebike_profit_cny, drone_profit_cny, distance_km columns.
        save_path: Save path; if None, display interactively.
        dpi: Image resolution.
        for_latex: If True, output PDF without titles.

    Returns:
        Saved figure path or None.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Profit distribution histogram
    ax1 = axes[0]
    bins = np.linspace(
        min(od_df["ebike_profit_cny"].min(), od_df["drone_profit_cny"].min()),
        max(od_df["ebike_profit_cny"].max(), od_df["drone_profit_cny"].max()),
        50,
    )
    ax1.hist(
        od_df["ebike_profit_cny"],
        bins=bins,
        alpha=0.6,
        color="#4CAF50",
        edgecolor="white",
        label="E-bike",
    )
    ax1.hist(
        od_df["drone_profit_cny"],
        bins=bins,
        alpha=0.6,
        color="#2196F3",
        edgecolor="white",
        label="Drone",
    )
    ax1.axvline(x=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.set_xlabel("Profit per Order (CNY)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    if plot_title:
        ax1.set_title("Profit Distribution by Delivery Mode", fontsize=14)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Subplot 2: Drone profit advantage vs distance
    ax2 = axes[1]
    profit_diff = od_df["drone_profit_cny"] - od_df["ebike_profit_cny"]
    sc = ax2.scatter(
        od_df["distance_km"],
        profit_diff,
        c=profit_diff,
        cmap="RdYlGn",
        s=10,
        alpha=0.5,
        edgecolors="none",
    )
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Straight-line Distance (km)", fontsize=12)
    ax2.set_ylabel("Drone Profit Advantage (CNY)", fontsize=12)
    if plot_title:
        ax2.set_title("Drone Profit Advantage vs Distance", fontsize=14)
    ax2.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax2, label="Profit Diff (CNY)")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_cost_breakdown_by_distance(
    profit_analysis,
    exp_config: Dict = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """Plot stacked bar cost breakdown at representative distances with revenue overlay.

    Args:
        profit_analysis: ProfitAnalysis instance.
        exp_config: Experiment config dict.
        save_path: Save path; if None, display interactively.
        dpi: Image resolution.
        for_latex: If True, output PDF without titles.

    Returns:
        Saved figure path or None.
    """
    from emission.cal_single_delivery import get_emission_params

    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    rep_distances = [0.5, 1, 2, 3, 5]
    x = np.arange(len(rep_distances))
    bar_width = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # E-bike cost breakdown
    ax1 = axes[0]
    ebike_components = ["vehicle_amort", "battery_amort", "labor", "energy_cost", "maintenance", "insurance"]
    ebike_labels = ["Vehicle", "Battery", "Labor", "Energy", "Maintenance", "Insurance"]
    ebike_colors = ["#388E3C", "#66BB6A", "#A5D6A7", "#C8E6C9", "#E8F5E9", "#F1F8E9"]

    ebike_data = {comp: [] for comp in ebike_components}
    revenues = []
    for d in rep_distances:
        cost = profit_analysis.calculate_ebike_cost_per_order(d, payload_kg, rider_weight)
        for comp in ebike_components:
            ebike_data[comp].append(cost[comp])
        rev = profit_analysis.calculate_revenue(d)
        revenues.append(rev["total_revenue_cny"])

    bottom = np.zeros(len(rep_distances))
    for comp, label, color in zip(ebike_components, ebike_labels, ebike_colors):
        values = np.array(ebike_data[comp])
        ax1.bar(x, values, bar_width, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += values

    ax1.plot(x, revenues, "r--o", linewidth=2, markersize=6, label="Revenue", zorder=10)
    ax1.set_xlabel("Distance (km)", fontsize=12)
    ax1.set_ylabel("Cost / Revenue (CNY)", fontsize=12)
    if plot_title:
        ax1.set_title("E-bike Cost Breakdown by Distance", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}" for d in rep_distances])
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Drone cost breakdown
    ax2 = axes[1]
    drone_components = ["vehicle_amort", "battery_amort", "utm_cost", "energy_cost", "maintenance", "insurance"]
    drone_labels = ["Vehicle", "Battery", "UTM", "Energy", "Maintenance", "Insurance"]
    drone_colors = ["#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD", "#E1F5FE"]

    drone_data = {comp: [] for comp in drone_components}
    for d in rep_distances:
        cost = profit_analysis.calculate_drone_cost_per_order(d, payload_kg)
        for comp in drone_components:
            drone_data[comp].append(cost[comp])

    bottom = np.zeros(len(rep_distances))
    for comp, label, color in zip(drone_components, drone_labels, drone_colors):
        values = np.array(drone_data[comp])
        ax2.bar(x, values, bar_width, bottom=bottom, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottom += values

    ax2.plot(x, revenues, "r--o", linewidth=2, markersize=6, label="Revenue", zorder=10)
    ax2.set_xlabel("Distance (km)", fontsize=12)
    ax2.set_ylabel("Cost / Revenue (CNY)", fontsize=12)
    if plot_title:
        ax2.set_title("Drone Cost Breakdown by Distance", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{d}" for d in rep_distances])
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_profit_vs_distance(
    od_df: pd.DataFrame,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """Plot binned median profit curves with percentile bands, highlighting crossover.

    Args:
        od_df: DataFrame with distance_km, ebike_profit_cny, drone_profit_cny columns.
        save_path: Save path; if None, display interactively.
        dpi: Image resolution.
        for_latex: If True, output PDF without titles.

    Returns:
        Saved figure path or None.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    distances = od_df["distance_km"].values
    d_min, d_max = distances.min(), distances.max()
    n_bins = 20
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for mode, color, label in [
        ("ebike_profit_cny", "#4CAF50", "E-bike"),
        ("drone_profit_cny", "#2196F3", "Drone"),
    ]:
        profits = od_df[mode].values
        medians = []
        p25 = []
        p75 = []
        for i in range(n_bins):
            mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            if mask.sum() > 3:
                medians.append(np.median(profits[mask]))
                p25.append(np.percentile(profits[mask], 25))
                p75.append(np.percentile(profits[mask], 75))
            else:
                medians.append(np.nan)
                p25.append(np.nan)
                p75.append(np.nan)

        medians = np.array(medians)
        p25 = np.array(p25)
        p75 = np.array(p75)
        valid = ~np.isnan(medians)

        ax.fill_between(
            bin_centers[valid], p25[valid], p75[valid],
            alpha=0.2, color=color,
        )
        ax.plot(
            bin_centers[valid], medians[valid],
            color=color, linewidth=2.5, label=label,
        )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Find and mark crossover point
    ebike_med = []
    drone_med = []
    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 3:
            ebike_med.append(np.median(od_df["ebike_profit_cny"].values[mask]))
            drone_med.append(np.median(od_df["drone_profit_cny"].values[mask]))
        else:
            ebike_med.append(np.nan)
            drone_med.append(np.nan)

    ebike_med = np.array(ebike_med)
    drone_med = np.array(drone_med)
    diff = drone_med - ebike_med

    for i in range(len(diff) - 1):
        if np.isnan(diff[i]) or np.isnan(diff[i + 1]):
            continue
        if diff[i] * diff[i + 1] < 0:
            x_cross = bin_centers[i] + (bin_centers[i + 1] - bin_centers[i]) * (
                -diff[i] / (diff[i + 1] - diff[i])
            )
            ax.axvline(x=x_cross, color="orange", linestyle=":", linewidth=2)
            ax.annotate(
                f"Crossover: {x_cross:.2f} km",
                xy=(x_cross, 0),
                xytext=(x_cross, ax.get_ylim()[1] * 0.3),
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="orange"),
            )
            break

    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel("Profit per Order (CNY)", fontsize=12)
    if plot_title:
        ax.set_title("Profit vs Distance (Median with 25-75% Band)", fontsize=14)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None
