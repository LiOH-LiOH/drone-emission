"""
排放分析可视化模块
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _add_title_suffix


def add_experiment_info_box(
    ax,
    experiment_settings: Dict[str, str],
    loc: str = "upper right",
    fontsize: int = 9,
):
    """
    在图表上添加实验设定信息文本框

    参数:
        ax: Matplotlib axes 对象
        experiment_settings: 实验设定字典，键为设定名称，值为设定值
                            例如: {"Distance": "3 km", "Payload": "1 kg", "Route Factor": "1.4"}
        loc: 文本框位置，可选 "upper right", "upper left", "lower right", "lower left"
        fontsize: 字体大小
    """
    if not experiment_settings:
        return

    # 构建文本内容
    lines = [f"{k}: {v}" for k, v in experiment_settings.items()]
    text = "\n".join(lines)

    # 根据位置设置坐标
    loc_map = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
    }

    x, y, ha, va = loc_map.get(loc, loc_map["upper right"])

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        zorder=100,
    )


def _draw_emission_reduction_vs_drone_ratio(ax, od_df, plot_title: bool = False, n_thresholds: int = 10):
    """Draw the sales-weighted emission reduction vs drone-order ratio curve."""
    required = ["road_distance_m", "drone_lifecycle_g", "ebike_lifecycle_g", "annual_sales"]
    valid = od_df.dropna(subset=required).copy()
    for col in required:
        valid[col] = pd.to_numeric(valid[col], errors="coerce")
    valid = valid.dropna(subset=required)
    valid = valid[
        (valid["road_distance_m"] >= 0)
        & (valid["annual_sales"] > 0)
    ]
    if valid.empty:
        ax.text(0.5, 0.5, "weighted lifecycle data\nnot available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#888888")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return

    road_km = valid["road_distance_m"].to_numpy(dtype=float) / 1000.0
    drone_lc = valid["drone_lifecycle_g"].to_numpy(dtype=float)
    ebike_lc = valid["ebike_lifecycle_g"].to_numpy(dtype=float)
    annual_sales = valid["annual_sales"].to_numpy(dtype=float)
    total_orders = annual_sales.sum()
    ebike_total = np.sum(ebike_lc * annual_sales)

    middle = np.linspace(
        max(0.3, np.percentile(road_km, 5)),
        np.percentile(road_km, 95),
        n_thresholds,
    )
    thresholds = np.concatenate([[-0.01], middle, [np.inf]])

    drone_ratios = []
    reductions = []
    for t in thresholds:
        use_drone = road_km >= t
        mixed = np.sum(np.where(use_drone, drone_lc, ebike_lc) * annual_sales)
        drone_ratios.append(annual_sales[use_drone].sum() / total_orders * 100)
        reductions.append((ebike_total - mixed) / ebike_total * 100)

    ax.plot(drone_ratios, reductions, "b-s", linewidth=2, markersize=6)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Drone Order Ratio (%)", fontsize=12)
    ax.set_ylabel("Weighted Emission Reduction (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    if plot_title:
        ax.set_title("Emission Reduction vs Drone Ratio", fontsize=14)


def plot_emission_reduction_vs_drone_ratio(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (8, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_thresholds: int = 10,
):
    """Emission reduction (%) vs drone order ratio (%).

    Sweeps road-distance thresholds: orders with road_distance >= threshold
    are served by drone, others by e-bike. Plots resulting lifecycle reduction
    against annual-sales-weighted drone order share.

    Args:
        od_df: OD pairs DataFrame with road_distance_m, drone_lifecycle_g,
               ebike_lifecycle_g, annual_sales columns.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_thresholds: Number of interior thresholds between 5th and 95th
                      percentile of road distance.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_emission_reduction_vs_drone_ratio(ax, od_df, plot_title=plot_title, n_thresholds=n_thresholds)
    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Emission reduction vs drone ratio saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_by_road_distance_bar(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.5,
):
    """Stacked histogram: annual lifecycle emission per road-distance bin.

    Each bin holds two stacked segments:
      - bottom: optimal emission (per OD, the lower of drone vs e-bike)
      - top:    reduction (e-bike origin minus optimal)
    Total bar height = e-bike origin emission for that bin.

    Per-OD totals are weighted by ``annual_sales`` and reported in
    tonnes CO2e per year.

    Args:
        od_df: OD pairs DataFrame with ``road_distance_m``,
               ``drone_lifecycle_g``, ``ebike_lifecycle_g``, ``annual_sales``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each road-distance bin in km. Bins span 0 to
                      the observed max road distance — no overflow bin.
    """
    valid = od_df[od_df["road_distance_m"] >= 0].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    drone_lc = valid["drone_lifecycle_g"].values
    ebike_lc = valid["ebike_lifecycle_g"].values
    sales = valid["annual_sales"].values

    max_km = float(np.ceil(road_km.max() / bin_width_km) * bin_width_km)
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2

    optimal_lc = np.minimum(drone_lc, ebike_lc)
    ebike_per_od = ebike_lc * sales / 1e6  # g → tonnes
    optimal_per_od = optimal_lc * sales / 1e6

    bin_idx = np.clip(np.digitize(road_km, edges) - 1, 0, n_bins - 1)
    ebike_bin = np.zeros(n_bins)
    optimal_bin = np.zeros(n_bins)
    for k in range(n_bins):
        m = bin_idx == k
        ebike_bin[k] = ebike_per_od[m].sum()
        optimal_bin[k] = optimal_per_od[m].sum()
    reduction_bin = ebike_bin - optimal_bin

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(centers, optimal_bin, width=bin_width_km, color="#2ca02c",
           edgecolor="white", linewidth=0.3, label="Optimal (drone+e-bike)",
           align="center")
    ax.bar(centers, reduction_bin, width=bin_width_km, bottom=optimal_bin,
           color="#ff7f0e", edgecolor="white", linewidth=0.3,
           label="Emission reduction", align="center")

    ax.set_xlabel("Road Distance (km)", fontsize=12)
    ax.set_ylabel("Annual Lifecycle Emission (t CO2e)", fontsize=12)
    ax.set_xlim(0, max_km)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    if plot_title:
        ax.set_title("Annual Emission by Road Distance: Optimal + Reduction", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Emission by road distance bar saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_reduction_pct_and_detour_by_distance(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.5,
):
    """Two-panel figure binned by road distance.

    Top panel: boxplot of per-OD detour ratio (road / air) per bin.
    Bottom panel: sales-weighted emission reduction (%) per bin, defined as
    ``(sum(ebike_lc * sales) - sum(min(drone_lc, ebike_lc) * sales)) / sum(ebike_lc * sales)``.

    Args:
        od_df: OD pairs DataFrame with ``road_distance_m``, ``distance_km``,
               ``drone_lifecycle_g``, ``ebike_lifecycle_g``, ``annual_sales``.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each road-distance bin in km. Bins span 0 to
                      the observed max road distance — no overflow bin.
    """
    valid = od_df[(od_df["road_distance_m"] >= 0) & (od_df["distance_km"] > 0)].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    air_km = valid["distance_km"].values
    detour = road_km / air_km
    drone_lc = valid["drone_lifecycle_g"].values
    ebike_lc = valid["ebike_lifecycle_g"].values
    sales = valid["annual_sales"].values

    max_km = float(np.ceil(road_km.max() / bin_width_km) * bin_width_km)
    edges = np.arange(0.0, max_km + bin_width_km / 2, bin_width_km)
    n_bins = len(edges) - 1
    centers = (edges[:-1] + edges[1:]) / 2

    optimal_lc = np.minimum(drone_lc, ebike_lc)
    bin_idx = np.clip(np.digitize(road_km, edges) - 1, 0, n_bins - 1)

    reduction_pct = np.full(n_bins, np.nan)
    ebike_total_t = np.zeros(n_bins)
    detour_by_bin = []
    for k in range(n_bins):
        m = bin_idx == k
        if not m.any():
            detour_by_bin.append(np.array([]))
            continue
        ebike_tot = (ebike_lc[m] * sales[m]).sum()
        optimal_tot = (optimal_lc[m] * sales[m]).sum()
        ebike_total_t[k] = ebike_tot / 1e6
        if ebike_tot > 0:
            reduction_pct[k] = (ebike_tot - optimal_tot) / ebike_tot * 100
        detour_by_bin.append(detour[m])
    x_ticks = np.arange(0, max_km + 1e-6, 1.0)
    x_tick_labels = [f"{int(t)}" for t in x_ticks]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    nonempty = [(c, d) for c, d in zip(centers, detour_by_bin) if d.size > 0]
    box_positions = [c for c, _ in nonempty]
    box_data = [d for _, d in nonempty]
    ax_top.boxplot(
        box_data, positions=box_positions, widths=bin_width_km * 0.8,
        showfliers=False, patch_artist=True,
        boxprops=dict(facecolor="#cfe2f3", edgecolor="#1f4e79", linewidth=0.6),
        medianprops=dict(color="#1f4e79", linewidth=1.2),
        whiskerprops=dict(color="#1f4e79", linewidth=0.6),
        capprops=dict(color="#1f4e79", linewidth=0.6),
    )
    ax_top.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_top.set_ylabel("Detour Ratio (road / air)", fontsize=12)
    ax_top.set_xticks(x_ticks, labels=x_tick_labels)
    ax_top.set_xlim(0, max_km)
    ax_top.grid(axis="y", alpha=0.3)

    ax_bot.bar(
        centers, reduction_pct, width=bin_width_km,
        color="#2ca02c", edgecolor="white", linewidth=0.3, align="center",
        label="Reduction %",
    )
    ax_bot.axhline(0, color="gray", linewidth=0.6)
    ax_bot.set_xlabel("Road Distance (km)", fontsize=12)
    ax_bot.set_ylabel("Emission Reduction (%)", fontsize=12, color="#2ca02c")
    ax_bot.set_xlim(0, max_km)
    ax_bot.set_xticks(x_ticks, labels=x_tick_labels)
    ax_bot.tick_params(axis="y", labelcolor="#2ca02c")
    ax_bot.grid(axis="y", alpha=0.3)

    ax_abs = ax_bot.twinx()
    ax_abs.plot(
        centers, ebike_total_t,
        color="#555555", alpha=0.5, linewidth=1.6, marker="o", markersize=4,
        label="E-bike total (t CO2e/yr)",
    )
    ax_abs.set_ylabel("E-bike total (t CO2e/yr)", fontsize=12, color="#555555")
    ax_abs.tick_params(axis="y", labelcolor="#555555")
    ax_abs.set_ylim(bottom=0)

    if plot_title:
        fig.suptitle("Detour Ratio and Emission Reduction by Road Distance", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Reduction pct and detour by distance saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_reduction_analysis(
    results_df: pd.DataFrame,
    bundle_results: dict = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制排放减少分析图

    参数:
        results_df: 包含 threshold, drone_ratio, reduction 列的 DataFrame
        bundle_results: 束配送场景结果字典 {coef: DataFrame}
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Color scheme for bundle scenarios
    bundle_colors = {
        0.9: "#2ca02c",  # green
        0.8: "#ff7f0e",  # orange
        0.7: "#9467bd",  # purple
        0.6: "#8c564b",  # brown
    }

    # 图1: 阈值 vs 减排百分比
    ax1 = axes[0]
    ax1.plot(
        results_df["threshold"],
        results_df["reduction"],
        "b-o",
        linewidth=2,
        markersize=8,
        label="No bundling",
    )

    # Plot bundle scenarios
    if bundle_results:
        for coef, df in bundle_results.items():
            color = bundle_colors.get(coef, "gray")
            ax1.plot(
                df["threshold"],
                df["reduction"],
                "-s",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=f"Bundle {coef}",
            )

    ax1.set_xlabel("Distance Threshold (km)", fontsize=12)
    ax1.set_ylabel("Emission Reduction (%)", fontsize=12)
    if plot_title:
        ax1.set_title("Emission Reduction vs Distance Threshold", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # 图2: 无人机占比 vs 减排百分比
    ax2 = axes[1]
    ax2.plot(
        results_df["drone_ratio"],
        results_df["reduction"],
        "b-s",
        linewidth=2,
        markersize=8,
        label="No bundling",
    )

    # Plot bundle scenarios
    if bundle_results:
        for coef, df in bundle_results.items():
            color = bundle_colors.get(coef, "gray")
            ax2.plot(
                df["drone_ratio"],
                df["reduction"],
                "-s",
                color=color,
                linewidth=1.5,
                markersize=6,
                label=f"Bundle {coef}",
            )

    ax2.set_xlabel("Drone Delivery Ratio (%)", fontsize=12)
    ax2.set_ylabel("Emission Reduction (%)", fontsize=12)
    if plot_title:
        ax2.set_title("Emission Reduction vs Drone Ratio", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=9)

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


def plot_road_distance_distribution(
    road_distances: "np.ndarray",
    annual_sales: "np.ndarray",
    save_path: str = None,
    dpi: int = 150,
    n_bins: int = 50,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制道路距离分布图（按年销量加权）

    参数:
        road_distances: 道路距离数组 (km)
        annual_sales: 年销量数组
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率
        n_bins: 直方图的柱数

    返回:
        保存的图像路径
    """
    # Filter out invalid distances (negative values)
    valid_mask = road_distances >= 0
    valid_distances = road_distances[valid_mask]
    valid_sales = annual_sales[valid_mask]

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # 图1: 未加权的距离分布
    ax1 = axes[0]
    ax1.hist(
        valid_distances,
        bins=n_bins,
        color="#2196F3",
        edgecolor="white",
        alpha=0.7,
    )
    ax1.set_xlabel("Road Distance (km)", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    if plot_title:
        ax1.set_title("Road Distance Distribution (Unweighted)", fontsize=14)
    ax1.grid(True, alpha=0.3, axis="y")

    # 添加统计信息
    mean_dist = valid_distances.mean()
    median_dist = np.median(valid_distances)
    ax1.axvline(
        mean_dist,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dist:.2f} km",
    )
    ax1.axvline(
        median_dist,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median_dist:.2f} km",
    )
    if plot_title:
        ax1.legend(loc="upper right")

    # 图2: 按年销量加权的距离分布
    ax2 = axes[1]
    ax2.hist(
        valid_distances,
        bins=n_bins,
        weights=valid_sales,
        color="#4CAF50",
        edgecolor="white",
        alpha=0.7,
    )
    ax2.set_xlabel("Road Distance (km)", fontsize=12)
    ax2.set_ylabel("Weighted Count (by Annual Sales)", fontsize=12)
    if plot_title:
        ax2.set_title(
            "Road Distance Distribution (Weighted by Annual Sales)", fontsize=14
        )
    ax2.grid(True, alpha=0.3, axis="y")

    # 添加加权统计信息
    weighted_mean = np.average(valid_distances, weights=valid_sales)
    # 计算加权中位数
    sorted_idx = np.argsort(valid_distances)
    sorted_distances = valid_distances[sorted_idx]
    sorted_weights = valid_sales[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    weighted_median = sorted_distances[np.searchsorted(cumsum, cumsum[-1] / 2)]

    ax2.axvline(
        weighted_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Weighted Mean: {weighted_mean:.2f} km",
    )
    ax2.axvline(
        weighted_median,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Weighted Median: {weighted_median:.2f} km",
    )
    if plot_title:
        ax2.legend(loc="upper right")

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


def plot_road_air_distance_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 8),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 10.0,
):
    """Road distance weighted distribution with air distance boxplot per bin.

    Top panel: bar chart of road distance weighted by annual sales.
    Bottom panel: boxplot of air (haversine) distance for each road distance bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show titles.
        n_bins: Number of distance bins.
        max_km: Upper bound of distance axis in km.
    """
    valid = df[df["road_distance_m"] >= 0].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    air_km = valid["distance_km"].values
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Weighted bar heights
    weights, _ = np.histogram(road_km, bins=bins, weights=sales)

    # Air distance groups per bin
    bin_idx = np.digitize(road_km, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    air_groups = [air_km[bin_idx == i] for i in range(n_bins)]

    fig, (ax_bar, ax_box) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    # --- Top: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.85,
    )
    ax_bar.set_ylabel("Weighted Count\n(Annual Sales)", fontsize=11)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}×10⁶" if x >= 1e6 else f"{x:.0f}")
    )
    if plot_title:
        ax_bar.set_title("Road Distance Distribution (Weighted by Annual Sales)", fontsize=13)

    # --- Bottom: boxplot of air distance per bin ---
    nonempty = [(i, g) for i, g in enumerate(air_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        bp = ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.7,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.8),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        _ = bp  # suppress unused warning

    ax_box.set_xlabel("Road Distance (km)", fontsize=11)
    ax_box.set_ylabel("Air Distance (km)", fontsize=11)
    ax_box.set_xlim(0, max_km)
    ax_box.grid(True, alpha=0.3, axis="y")
    if plot_title:
        ax_box.set_title("Air Distance Distribution per Road Distance Bin", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Road air distance distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_road_detour_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 10.0,
):
    """Road distance weighted bar chart with detour ratio boxplot on twin y-axis.

    Left y-axis: bar chart of road distance weighted by annual sales.
    Right y-axis: boxplot of detour ratio (road / air distance) per bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_bins: Number of distance bins.
        max_km: Upper bound of distance axis in km.
    """
    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    road_km = valid["road_distance_m"].values / 1000.0
    detour = road_km / valid["distance_km"].values
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    weights, _ = np.histogram(road_km, bins=bins, weights=sales)

    bin_idx = np.clip(np.digitize(road_km, bins) - 1, 0, n_bins - 1)
    detour_groups = [detour[bin_idx == i] for i in range(n_bins)]

    fig, ax_bar = plt.subplots(figsize=figsize)
    ax_box = ax_bar.twinx()

    # --- Left: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.7,
        zorder=2,
    )
    ax_bar.set_xlabel("Road Distance (km)", fontsize=11)
    ax_bar.set_ylabel("Weighted Count (Annual Sales)", fontsize=11)
    ax_bar.set_xlim(0, max_km)
    ax_bar.grid(True, alpha=0.3, axis="y", zorder=1)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    )

    # --- Right: boxplot of detour ratio per bin ---
    nonempty = [(i, g) for i, g in enumerate(detour_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.85),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            zorder=3,
        )

    ax_box.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)
    ax_box.set_ylabel("Detour Ratio (road / air)", fontsize=11)

    if plot_title:
        ax_bar.set_title("Road Distance Distribution & Detour Ratio", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Road detour distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_air_detour_distribution(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 5),
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 20,
    max_km: float = 8.0,
):
    """Air distance weighted bar chart with detour ratio boxplot on twin y-axis.

    Left y-axis: bar chart of air distance weighted by annual sales.
    Right y-axis: boxplot of detour ratio (road / air) per air distance bin.

    Args:
        df: DataFrame with 'road_distance_m', 'distance_km', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        n_bins: Number of distance bins.
        max_km: Upper bound of air distance axis in km.
    """
    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    air_km = valid["distance_km"].values
    detour = (valid["road_distance_m"].values / 1000.0) / air_km
    sales = valid["annual_sales"].values

    bins = np.linspace(0, max_km, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    weights, _ = np.histogram(air_km, bins=bins, weights=sales)

    bin_idx = np.clip(np.digitize(air_km, bins) - 1, 0, n_bins - 1)
    detour_groups = [detour[bin_idx == i] for i in range(n_bins)]

    fig, ax_bar = plt.subplots(figsize=figsize)
    ax_box = ax_bar.twinx()

    # --- Left: weighted bar chart ---
    ax_bar.bar(
        bin_centers, weights,
        width=bin_width * 0.85,
        color="#4CAF50", edgecolor="white", alpha=0.7,
        zorder=2,
    )
    ax_bar.set_xlabel("Air Distance (km)", fontsize=11)
    ax_bar.set_ylabel("Weighted Count (Annual Sales)", fontsize=11)
    ax_bar.set_xlim(0, max_km)
    ax_bar.grid(True, alpha=0.3, axis="y", zorder=1)
    ax_bar.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    )

    # --- Right: boxplot of detour ratio per air distance bin ---
    nonempty = [(i, g) for i, g in enumerate(detour_groups) if len(g) > 0]
    if nonempty:
        positions = [bin_centers[i] for i, _ in nonempty]
        groups = [g for _, g in nonempty]
        ax_box.boxplot(
            groups,
            positions=positions,
            widths=bin_width * 0.5,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="#E65100", linewidth=1.5),
            boxprops=dict(facecolor="#B3E5FC", alpha=0.85),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            zorder=3,
        )

    ax_box.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, zorder=2)
    ax_box.set_ylabel("Detour Ratio (road / air)", fontsize=11)

    if plot_title:
        ax_bar.set_title("Air Distance Distribution & Detour Ratio", fontsize=13)

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Air detour distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_air_road_joint(
    df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = None,
    for_latex: bool = False,
    plot_title: bool = False,
    max_road_km: float = None,
    max_air_km: float = None,
    gridsize: int = 40,
):
    """Joint distribution of road (x) and air (y) distance with sales-weighted density.

    Centre: hexbin of road (x) vs air (y) distance weighted by annual sales.
    Top marginal: road distance distribution weighted by annual sales.
    Right marginal: air distance distribution weighted by annual sales.
    Reference lines show detour ratio = 1.2×, 1.5×, 2.0× (air = road / ratio).

    Args:
        df: DataFrame with 'distance_km', 'road_distance_m', 'annual_sales' columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        max_road_km: Upper bound for road distance axis; None = 99th percentile of data.
        max_air_km: Upper bound for air distance axis; None = 99th percentile of data.
        gridsize: Hexbin grid resolution.
    """
    import matplotlib.gridspec as gridspec

    valid = df[(df["road_distance_m"] >= 0) & (df["distance_km"] > 0)].copy()
    air_km = valid["distance_km"].values
    road_km = valid["road_distance_m"].values / 1000.0
    sales = valid["annual_sales"].values

    if max_road_km is None:
        max_road_km = float(np.percentile(road_km, 99))
    if max_air_km is None:
        max_air_km = float(np.percentile(air_km, 99))

    if figsize is None:
        # Size the figure so 1 km = 1 km on both axes (equal aspect).
        # GridSpec: width_ratios=[4,1,0.15], height_ratios=[1,4]
        main_h = 5.0  # inches for the main panel
        main_w = main_h * (max_road_km / max_air_km)
        total_w = main_w * (4 + 1 + 0.15) / 4
        total_h = main_h * (4 + 1) / 4
        figsize = (total_w, total_h)

    mask = (air_km <= max_air_km) & (road_km <= max_road_km)
    air_km, road_km, sales = air_km[mask], road_km[mask], sales[mask]

    # Equal bin width for both marginals
    n_bars = 30
    road_bins = np.linspace(0, max_road_km, n_bars + 1)
    air_bins = np.linspace(0, max_air_km, n_bars + 1)
    road_centers = (road_bins[:-1] + road_bins[1:]) / 2
    air_centers = (air_bins[:-1] + air_bins[1:]) / 2
    road_bw = road_bins[1] - road_bins[0]
    air_bw = air_bins[1] - air_bins[0]
    road_weights, _ = np.histogram(road_km, bins=road_bins, weights=sales)
    air_weights, _ = np.histogram(air_km, bins=air_bins, weights=sales)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1, 4], width_ratios=[4, 1, 0.15],
        hspace=0.05, wspace=0.25,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_cbar = fig.add_subplot(gs[1, 2])
    for ax in [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]:
        ax.set_visible(False)

    # --- Centre: hexbin (x=road, y=air) ---
    hb = ax_main.hexbin(
        road_km, air_km,
        C=sales,
        reduce_C_function=np.sum,
        gridsize=gridsize,
        cmap="YlOrRd",
        mincnt=1,
        extent=[0, max_road_km, 0, max_air_km],
    )
    ax_main.set_xlabel("Road Distance (km)", fontsize=11)
    ax_main.set_ylabel("Air Distance (km)", fontsize=11)
    ax_main.set_xlim(0, max_road_km)
    ax_main.set_ylim(0, max_air_km)
    ax_main.set_aspect("equal")

    # Detour ratio reference lines: detour = road/air → y = x/ratio
    ref_ratios = [(1.2, "--", "#555555"), (1.5, "-.", "#333333"), (2.0, ":", "#111111")]
    for ratio, ls, color in ref_ratios:
        y_end = min(max_air_km, max_road_km / ratio)
        x_end = y_end * ratio
        ax_main.plot(
            [0, x_end], [0, y_end],
            linestyle=ls, color=color, linewidth=1.0, alpha=0.7,
            label=f"detour = {ratio}×",
        )
    ax_main.legend(fontsize=9, loc="upper left")

    # Colourbar in its own dedicated axes
    cb = fig.colorbar(hb, cax=ax_cbar)
    cb.set_label("Annual Sales (sum)", fontsize=9)
    cb.formatter = plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x:.0f}")
    cb.update_ticks()

    # --- Top marginal: road distance ---
    ax_top.bar(road_centers, road_weights, width=road_bw * 0.85, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax_top.set_ylabel("Sales (M)", fontsize=9)
    ax_top.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax_top.tick_params(labelbottom=False)
    ax_top.set_xlim(0, max_road_km)
    ax_top.grid(True, alpha=0.3, axis="y")

    # --- Right marginal: air distance (horizontal) ---
    ax_right.barh(air_centers, air_weights, height=air_bw * 0.85, color="#2196F3", edgecolor="white", alpha=0.8)
    ax_right.set_xlabel("Sales (M)", fontsize=9)
    ax_right.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))
    ax_right.tick_params(labelleft=False)
    ax_right.set_ylim(0, max_air_km)
    ax_right.grid(True, alpha=0.3, axis="x")

    if plot_title:
        fig.suptitle("Air vs Road Distance Joint Distribution (Shanghai)", fontsize=13)

    # After equal-aspect shrinks ax_main, snap marginals to its actual bounds.
    fig.canvas.draw()
    m = ax_main.get_position()
    r = ax_right.get_position()
    t = ax_top.get_position()
    c = ax_cbar.get_position()
    ax_right.set_position([r.x0, m.y0, r.width, m.height])
    ax_top.set_position([m.x0, t.y0, m.width, t.height])
    ax_cbar.set_position([c.x0, m.y0, c.width, m.height])

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Air road joint distribution saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_emission_comparison_bar(
    ebike_emissions: float,
    drone_emissions: float,
    mixed_emissions: float = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制不同配送方式排放对比柱状图

    参数:
        ebike_emissions: 电动自行车总排放 (kg CO2e)
        drone_emissions: 无人机总排放 (kg CO2e)
        mixed_emissions: 混合模式总排放 (kg CO2e)，可选
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["E-bike", "Drone"]
    values = [ebike_emissions, drone_emissions]
    colors = ["#4CAF50", "#2196F3"]

    if mixed_emissions is not None:
        labels.append("Mixed")
        values.append(mixed_emissions)
        colors.append("#FF9800")

    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)

    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Total Emissions (kg CO2e)", fontsize=12)
    if plot_title:
        ax.set_title("Emission Comparison by Delivery Mode", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

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


def plot_distance_distribution(
    road_distances,
    thresholds: list = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制配送距离分布直方图

    参数:
        road_distances: 道路距离数组
        thresholds: 阈值列表，用于标记垂直线
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(road_distances, bins=50, color="#2196F3", alpha=0.7, edgecolor="black")

    if thresholds:
        colors = plt.cm.Reds(range(50, 250, int(200 / len(thresholds))))
        for i, threshold in enumerate(thresholds):
            ax.axvline(
                x=threshold,
                color=colors[i],
                linestyle="--",
                linewidth=1.5,
                label=f"{threshold:.1f} km",
            )
        ax.legend(title="Thresholds", loc="upper right")

    ax.set_xlabel("Road Distance (km)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    if plot_title:
        ax.set_title("Distribution of Delivery Distances", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

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


def plot_sensitivity_analysis(
    param_values: list,
    ebike_emissions: list,
    drone_emissions: list,
    param_name: str,
    param_unit: str = "",
    experiment_settings: Dict[str, str] = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制单参数敏感性分析图

    参数:
        param_values: 参数值列表
        ebike_emissions: 对应的电动自行车排放列表 (g CO2e)
        drone_emissions: 对应的无人机排放列表 (g CO2e)
        param_name: 参数名称
        param_unit: 参数单位
        experiment_settings: 实验设定字典，用于显示在图表上
        save_path: 保存路径
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 排放值对比
    ax1 = axes[0]
    ax1.plot(
        param_values, ebike_emissions, "g-o", linewidth=2, markersize=6, label="E-bike"
    )
    ax1.plot(
        param_values, drone_emissions, "b-s", linewidth=2, markersize=6, label="Drone"
    )
    ax1.set_xlabel(
        f"{param_name} ({param_unit})" if param_unit else param_name, fontsize=12
    )
    ax1.set_ylabel("Emissions (g CO2e)", fontsize=12)
    if plot_title:
        ax1.set_title(f"Emissions vs {param_name}", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 添加实验设定信息
    if experiment_settings and plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    # 图2: 无人机相对电动车的排放变化百分比
    ax2 = axes[1]
    relative_diff = [
        (d / e - 1) * 100 for d, e in zip(drone_emissions, ebike_emissions)
    ]
    ax2.plot(param_values, relative_diff, "r-^", linewidth=2, markersize=6)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel(
        f"{param_name} ({param_unit})" if param_unit else param_name, fontsize=12
    )
    ax2.set_ylabel("Drone Emissions Relative to E-bike (%)", fontsize=12)
    if plot_title:
        ax2.set_title(f"Relative Emission Difference vs {param_name}", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 标记交叉点
    for i in range(len(relative_diff) - 1):
        if relative_diff[i] * relative_diff[i + 1] < 0:
            # 线性插值找交叉点
            x_cross = param_values[i] + (param_values[i + 1] - param_values[i]) * (
                -relative_diff[i] / (relative_diff[i + 1] - relative_diff[i])
            )
            ax2.axvline(x=x_cross, color="orange", linestyle=":", linewidth=2)
            ax2.annotate(
                f"Crossover: {x_cross:.2f}",
                xy=(x_cross, 0),
                xytext=(x_cross, np.max(relative_diff) * 0.3),
                fontsize=10,
                ha="center",
                arrowprops=dict(arrowstyle="->", color="orange"),
            )

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


def plot_distance_route_factor_heatmap(
    distances: list,
    route_factors: list,
    emission_diff_matrix,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
    ylabel: str = "Route Factor (road/straight-line)",
    xlabel: str = "Delivery Distance (km)",
) -> str:
    """
    绘制配送距离 × 路径系数的二维热力图

    参数:
        distances: 配送距离列表 (km)
        route_factors: 路径系数列表
        emission_diff_matrix: 排放差异矩阵 (无人机相对电动车的百分比变化)
                              shape: (len(route_factors), len(distances))
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建热力图
    abs_max = max(abs(emission_diff_matrix.min()), abs(emission_diff_matrix.max()))
    im = ax.imshow(
        emission_diff_matrix,
        cmap="RdYlGn_r",  # 红色=无人机排放高，绿色=无人机排放低
        aspect="auto",
        origin="lower",
        extent=[distances[0], distances[-1], route_factors[0], route_factors[-1]],
        vmin=-abs_max,
        vmax=abs_max,
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Drone Emissions Relative to E-bike (%)", fontsize=12)

    # 添加等值线 (类似等高线，显示不同排放差异水平)
    X, Y = np.meshgrid(distances, route_factors)

    # 根据数据范围动态生成等值线水平，间隔20%
    data_min = np.floor(emission_diff_matrix.min() / 20) * 20
    data_max = np.ceil(emission_diff_matrix.max() / 20) * 20
    contour_levels = np.arange(data_min, data_max + 1, 20)

    # 绘制所有等值线
    linewidths = [3 if lvl == 0 else 2 for lvl in contour_levels]
    linestyles = ["-" if lvl == 0 else "--" for lvl in contour_levels]
    contour = ax.contour(
        X,
        Y,
        emission_diff_matrix,
        levels=contour_levels,
        colors="black",
        linewidths=linewidths,
        linestyles=linestyles,
    )
    ax.clabel(contour, inline=True, fontsize=9, fmt=lambda x: f"{x:+.0f}%")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if plot_title:
        ax.set_title(
            "Emission Comparison: Drone vs E-bike\n(Negative = Drone Better)",
            fontsize=14,
        )

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax, experiment_settings, loc="upper left")

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


def plot_velocity_power_emission(
    velocities: list,
    powers: list,
    emissions: list,
    vehicle_type: str = "Drone",
    distance_info: str = "",
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制速度-功率-排放敏感性分析图

    参数:
        velocities: 速度列表 (km/h)
        powers: 对应的功率列表 (kW)
        emissions: 对应的排放列表 (g CO2e)
        vehicle_type: 车辆类型 ("Drone" 或 "E-bike")
        distance_info: 距离信息字符串，用于图表标题
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 根据车辆类型选择颜色
    if vehicle_type.lower() == "drone":
        color_power = "b-o"
        color_emission = "r-s"
        xlabel = "Drone Horizontal Velocity (km/h)"
    else:
        color_power = "g-o"
        color_emission = "m-s"
        xlabel = "E-bike Velocity (km/h)"

    # 图1: 速度 vs 功率
    ax1 = axes[0]
    ax1.plot(velocities, powers, color_power, linewidth=2, markersize=6)
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Power (kW)", fontsize=12)
    if plot_title:
        ax1.set_title(f"{vehicle_type} Power vs Velocity", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 图2: 速度 vs 排放
    ax2 = axes[1]
    ax2.plot(velocities, emissions, color_emission, linewidth=2, markersize=6)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Emissions (g CO2e)", fontsize=12)
    if plot_title:
        title = f"{vehicle_type} Emissions vs Velocity"
        if distance_info:
            title += f" ({distance_info})"
        ax2.set_title(title, fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

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


def plot_energy_consumption_vs_speed(
    velocities: list,
    energy_per_km: list,
    vehicle_type: str = "E-bike",
    empirical_range: tuple = None,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制能耗(Wh/km)与速度的关系图

    参数:
        velocities: 速度列表 (km/h)
        energy_per_km: 每公里能耗列表 (Wh/km)
        vehicle_type: 车辆类型
        empirical_range: 经验值范围元组 (min, max)，用于对比
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典

    返回:
        保存的图像路径
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 6))

    # 选择颜色
    if vehicle_type.lower() == "drone":
        color = "#2196F3"
        xlabel = "Drone Horizontal Velocity (km/h)"
    else:
        color = "#4CAF50"
        xlabel = "E-bike Velocity (km/h)"

    # 绘制模型预测曲线
    ax.plot(
        velocities,
        energy_per_km,
        "o-",
        color=color,
        linewidth=2,
        markersize=8,
        label=f"{vehicle_type} Model",
    )

    # 添加经验值范围
    if empirical_range:
        min_val, max_val = empirical_range
        ax.axhspan(
            min_val,
            max_val,
            alpha=0.2,
            color="orange",
            label=f"Empirical Range ({min_val}-{max_val} Wh/km)",
        )
        ax.axhline(
            y=(min_val + max_val) / 2,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Energy Consumption (Wh/km)", fontsize=12)
    if plot_title:
        ax.set_title(f"{vehicle_type} Energy Consumption vs Velocity", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax, experiment_settings, loc="lower right")

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


def plot_combined_velocity_power(
    drone_velocities: list,
    drone_powers: list,
    ebike_velocities: list,
    ebike_powers: list,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    将无人机和电动车的速度-功率曲线绘制在同一张图上（双X轴）

    参数:
        drone_velocities: 无人机速度列表 (km/h)
        drone_powers: 无人机功率列表 (kW)
        ebike_velocities: 电动车速度列表 (km/h)
        ebike_powers: 电动车功率列表 (kW)
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典，用于显示在图表上的信息框

    返回:
        保存的图像路径
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 电动车曲线 (底部X轴)
    color_ebike = "#4CAF50"
    line1 = ax1.plot(
        ebike_velocities,
        ebike_powers,
        "o-",
        color=color_ebike,
        linewidth=2,
        markersize=6,
        label="E-bike",
    )
    ax1.set_xlabel("E-bike Velocity (km/h)", fontsize=12, color=color_ebike)
    ax1.set_ylabel("Power (kW)", fontsize=12)
    ax1.tick_params(axis="x", labelcolor=color_ebike)
    ax1.grid(True, alpha=0.3)

    # 无人机曲线 (顶部X轴)
    ax2 = ax1.twiny()
    color_drone = "#2196F3"
    line2 = ax2.plot(
        drone_velocities,
        drone_powers,
        "s-",
        color=color_drone,
        linewidth=2,
        markersize=6,
        label="Drone",
    )
    ax2.set_xlabel("Drone Horizontal Velocity (km/h)", fontsize=12, color=color_drone)
    ax2.tick_params(axis="x", labelcolor=color_drone)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=11)

    if plot_title:
        ax1.set_title("Power vs Velocity Comparison", fontsize=14)

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax1, experiment_settings, loc="lower right")

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


def plot_velocity_power_stacked(
    drone_velocities: list,
    drone_power_breakdowns: list,
    ebike_velocities: list,
    ebike_power_breakdowns: list,
    save_path: str = None,
    dpi: int = 150,
    experiment_settings: Dict[str, str] = None,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制速度-功率对比的堆叠柱状图，显示各功率分量

    参数:
        drone_velocities: 无人机速度列表 (km/h)
        drone_power_breakdowns: 无人机功率分解列表，每个元素是包含
            P_induced, P_profile, P_drag, P_climb, P_aux 的字典
        ebike_velocities: 电动车速度列表 (km/h)
        ebike_power_breakdowns: 电动车功率分解列表，每个元素是包含
            P_rolling, P_air, P_grade, P_acc 的字典
        save_path: 保存路径
        dpi: 图像分辨率
        experiment_settings: 实验设置字典

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ========== 无人机堆叠柱状图 ==========
    ax1 = axes[0]
    x_drone = np.arange(len(drone_velocities))
    bar_width = 0.7

    # 无人机功率分量 (按显示顺序)
    drone_components = ["P_induced", "P_profile", "P_drag", "P_climb", "P_aux"]
    drone_labels = ["Induced", "Profile", "Drag", "Climb", "Auxiliary"]
    drone_colors = ["#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB", "#E3F2FD"]

    # 提取各分量数据
    drone_data = {comp: [] for comp in drone_components}
    for breakdown in drone_power_breakdowns:
        for comp in drone_components:
            drone_data[comp].append(breakdown.get(comp, 0) * 1000)  # kW -> W

    # 绘制堆叠柱状图
    bottom = np.zeros(len(drone_velocities))
    for comp, label, color in zip(drone_components, drone_labels, drone_colors):
        values = np.array(drone_data[comp])
        ax1.bar(
            x_drone,
            values,
            bar_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax1.set_xlabel("Drone Horizontal Velocity (km/h)", fontsize=12)
    ax1.set_ylabel("Power (W)", fontsize=12)
    if plot_title:
        ax1.set_title("Drone Power Breakdown vs Velocity", fontsize=14)
    ax1.set_xticks(x_drone)
    ax1.set_xticklabels([f"{v:.0f}" for v in drone_velocities])
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ========== 电动车堆叠柱状图 ==========
    ax2 = axes[1]
    x_ebike = np.arange(len(ebike_velocities))

    # 电动车功率分量 (按显示顺序)
    ebike_components = ["P_rolling", "P_air", "P_grade", "P_acc"]
    ebike_labels = ["Rolling", "Air Drag", "Grade", "Acceleration"]
    ebike_colors = ["#388E3C", "#66BB6A", "#A5D6A7", "#C8E6C9"]

    # 提取各分量数据
    ebike_data = {comp: [] for comp in ebike_components}
    for breakdown in ebike_power_breakdowns:
        for comp in ebike_components:
            ebike_data[comp].append(breakdown.get(comp, 0) * 1000)  # kW -> W

    # 绘制堆叠柱状图
    bottom = np.zeros(len(ebike_velocities))
    for comp, label, color in zip(ebike_components, ebike_labels, ebike_colors):
        values = np.array(ebike_data[comp])
        ax2.bar(
            x_ebike,
            values,
            bar_width,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    ax2.set_xlabel("E-bike Velocity (km/h)", fontsize=12)
    ax2.set_ylabel("Power (W)", fontsize=12)
    if plot_title:
        ax2.set_title("E-bike Power Breakdown vs Velocity", fontsize=14)
    ax2.set_xticks(x_ebike)
    ax2.set_xticklabels([f"{v:.0f}" for v in ebike_velocities])
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # 添加实验设置信息框
    if plot_title:
        add_experiment_info_box(ax2, experiment_settings, loc="lower right")

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


def plot_distance_detour_scatter(
    straight_distances: "np.ndarray",
    road_distances: "np.ndarray",
    annual_sales: "np.ndarray" = None,
    save_path: str = None,
    dpi: int = 150,
    for_latex: bool = False,
    plot_title: bool = False,
) -> str:
    """
    绘制直线距离与绕行比的关系图（中位数线 + 10-90%区间）
    """
    # 计算绕行比
    valid_mask = (straight_distances > 0) & (road_distances > 0)
    distances = straight_distances[valid_mask]
    detour_ratios = road_distances[valid_mask] / distances

    fig, ax = plt.subplots(figsize=(10, 6))

    # 分箱计算统计量
    n_bins = 20
    d_min, d_max = distances.min(), distances.max()
    bin_edges = np.linspace(d_min, d_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 首尾使用边界值而非中心
    bin_centers[0] = d_min
    bin_centers[-1] = d_max
    bin_medians = []
    bin_p10 = []
    bin_p90 = []

    for i in range(n_bins):
        mask = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        if mask.sum() > 5:
            bin_medians.append(np.median(detour_ratios[mask]))
            bin_p10.append(np.percentile(detour_ratios[mask], 10))
            bin_p90.append(np.percentile(detour_ratios[mask], 90))
        else:
            bin_medians.append(np.nan)
            bin_p10.append(np.nan)
            bin_p90.append(np.nan)

    bin_medians = np.array(bin_medians)
    bin_p10 = np.array(bin_p10)
    bin_p90 = np.array(bin_p90)

    # 绘制中位数线和10-90%区间
    valid_bins = ~np.isnan(bin_medians)
    ax.fill_between(
        bin_centers[valid_bins],
        bin_p10[valid_bins],
        bin_p90[valid_bins],
        alpha=0.3,
        color="#2196F3",
        label="10-90% Range",
    )
    ax.plot(bin_centers[valid_bins], bin_medians[valid_bins], "#1565C0", linewidth=2.5, label="Median")

    # 添加参考线
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, label="Direct (ratio=1)")

    ax.set_xlabel("Straight-line Distance (km)", fontsize=12)
    ax.set_ylabel("Detour Ratio (road / straight-line)", fontsize=12)
    if plot_title:
        ax.set_title("Distance vs Detour Ratio", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(d_min, d_max)

    # 设置 y 轴范围
    ax.set_ylim(0.9, np.nanmax(bin_p90) * 1.05)

    # 添加统计信息
    if plot_title:
        stats_text = (
            f"n = {len(distances):,}\n"
            f"Mean: {detour_ratios.mean():.2f}\n"
            f"Median: {np.median(detour_ratios):.2f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

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


def _save_or_show(fig, save_path, dpi, for_latex, plot_title):
    """Helper to save or show a figure."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path
    else:
        plt.show()
        return None


def plot_time_vs_distance(
    road_distances_km,
    ebike_time_min,
    drone_time_min,
    optimal_threshold_km: float,
    save_path: str = None,
    dpi: int = 300,
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 50,
):
    """Plot trip time vs trip distance for all-ebike and optimal mix scenarios."""
    road_distances_km = np.asarray(road_distances_km)
    ebike_time_min = np.asarray(ebike_time_min)
    drone_time_min = np.asarray(drone_time_min)

    mix_time = np.where(road_distances_km >= optimal_threshold_km, drone_time_min, ebike_time_min)

    bin_edges = np.linspace(road_distances_km.min(), np.percentile(road_distances_km, 99), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(road_distances_km, bin_edges) - 1, 0, n_bins - 1)

    ebike_means = np.array([ebike_time_min[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(n_bins)])
    mix_means = np.array([mix_time[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(n_bins)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, ebike_means, color="#e74c3c", linewidth=2, label="All E-bike")
    ax.plot(bin_centers, mix_means, color="#3498db", linewidth=2, label="Optimal Mix")
    ax.axvline(optimal_threshold_km, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Threshold: {optimal_threshold_km:.1f} km")

    ax.set_xlabel("Trip Distance (km)", fontsize=12)
    ax.set_ylabel("Trip Time (min)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if plot_title:
        ax.set_title("Trip Time vs Distance: All E-bike vs Optimal Mix", fontsize=14)

    plt.tight_layout()
    return _save_or_show(fig, save_path, dpi, for_latex, plot_title)


def plot_distance_vs_emission(
    road_distances_km,
    ebike_lifecycle_g,
    drone_lifecycle_g,
    optimal_threshold_km: float,
    annual_sales=None,
    save_path: str = None,
    dpi: int = 300,
    for_latex: bool = False,
    plot_title: bool = False,
    n_bins: int = 50,
):
    """Plot lifecycle emission vs trip distance for all-ebike and optimal mix scenarios.

    When ``annual_sales`` is provided, binned means are weighted by annual OD
    order counts.
    """
    road_distances_km = np.asarray(road_distances_km)
    ebike_lifecycle_g = np.asarray(ebike_lifecycle_g)
    drone_lifecycle_g = np.asarray(drone_lifecycle_g)
    weights = None if annual_sales is None else np.asarray(annual_sales, dtype=float)

    mix_emission = np.where(road_distances_km >= optimal_threshold_km, drone_lifecycle_g, ebike_lifecycle_g)

    bin_edges = np.linspace(road_distances_km.min(), np.percentile(road_distances_km, 99), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(road_distances_km, bin_edges) - 1, 0, n_bins - 1)

    def _bin_mean(values, mask):
        if not mask.any():
            return np.nan
        if weights is None:
            return values[mask].mean()
        w = weights[mask]
        total_w = w.sum()
        return np.average(values[mask], weights=w) if total_w > 0 else np.nan

    ebike_means = np.array([_bin_mean(ebike_lifecycle_g, bin_idx == i) for i in range(n_bins)])
    mix_means = np.array([_bin_mean(mix_emission, bin_idx == i) for i in range(n_bins)])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_centers, ebike_means, color="#e74c3c", linewidth=2, label="All E-bike")
    ax.plot(bin_centers, mix_means, color="#3498db", linewidth=2, label="Optimal Mix")
    ax.axvline(optimal_threshold_km, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Threshold: {optimal_threshold_km:.1f} km")

    ax.set_xlabel("Trip Distance (km)", fontsize=12)
    ax.set_ylabel("Lifecycle Emission (g CO\u2082e)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    if plot_title:
        ax.set_title("Emission vs Distance: All E-bike vs Optimal Mix", fontsize=14)

    plt.tight_layout()
    return _save_or_show(fig, save_path, dpi, for_latex, plot_title)


def _draw_lifecycle_distance_route_factor(
    ax,
    fig,
    od_df=None,
    plot_title: bool = False,
    cax=None,
    draw_heatmap: bool = True,
    contour_percentiles=None,
) -> None:
    """Draw lifecycle distance/route-factor comparison onto *ax*."""
    distances = np.linspace(0.5, 4, 30)
    detour_ratios = np.linspace(1.0, 1.8, 20)
    if contour_percentiles is None:
        contour_percentiles = (0.20, 0.40, 0.60, 0.80)

    if draw_heatmap:
        from emission.cal_single_delivery import LifeCycleEmissions

        lca = LifeCycleEmissions()
        emission_diff_matrix = np.zeros((len(detour_ratios), len(distances)))
        for i, r in enumerate(detour_ratios):
            for j, d in enumerate(distances):
                ebike_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d * r, n_orders_per_delivery=1, vehicle_type="ebike"
                )
                drone_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d, n_orders_per_delivery=1, vehicle_type="drone"
                )
                diff = (drone_result["total_per_order_g"] / ebike_result["total_per_order_g"] - 1) * 100
                emission_diff_matrix[i, j] = diff

        abs_max = max(abs(emission_diff_matrix.min()), abs(emission_diff_matrix.max()))
        im = ax.imshow(
            emission_diff_matrix,
            cmap="RdYlGn_r",
            aspect="auto",
            origin="lower",
            extent=[distances[0], distances[-1], detour_ratios[0], detour_ratios[-1]],
            vmin=-abs_max,
            vmax=abs_max,
        )

        cbar = fig.colorbar(im, cax=cax) if cax is not None else fig.colorbar(im, ax=ax)
        cbar.set_label("Drone emissions relative to e-bike (%)", fontsize=12)
    else:
        ax.set_xlim(distances[0], distances[-1])
        ax.set_ylim(detour_ratios[0], detour_ratios[-1])
        ax.grid(True, color="#d0d0d0", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Air distance (km)", fontsize=12)
    ax.set_ylabel("Detour ratio (road / air distance)", fontsize=12)
    if plot_title:
        title = "Emission comparison: drone vs e-bike" if draw_heatmap else "Shanghai order distribution"
        ax.set_title(title, fontsize=14)

    if od_df is not None:
        from scipy.ndimage import gaussian_filter

        valid = od_df.dropna(subset=["distance_km", "road_distance_m", "annual_sales"])
        valid = valid[(valid["distance_km"] >= distances[0]) & (valid["distance_km"] <= distances[-1])].copy()
        valid["detour"] = valid["road_distance_m"] / (valid["distance_km"] * 1000)
        valid = valid[(valid["detour"] >= detour_ratios[0]) & (valid["detour"] <= detour_ratios[-1])]

        if len(valid) > 10:
            n_bins = 100
            hist, xedges, yedges = np.histogram2d(
                valid["distance_km"].values,
                valid["detour"].values,
                bins=n_bins,
                range=[[distances[0], distances[-1]], [detour_ratios[0], detour_ratios[-1]]],
                weights=valid["annual_sales"].values,
            )
            Zi = gaussian_filter(hist.T, sigma=4.0)

            xi = 0.5 * (xedges[:-1] + xedges[1:])
            yi = 0.5 * (yedges[:-1] + yedges[1:])
            Xi, Yi = np.meshgrid(xi, yi)

            z_sorted = np.sort(Zi.ravel())[::-1]
            z_cumsum = np.cumsum(z_sorted) / z_sorted.sum()
            pcts = tuple(contour_percentiles)
            kde_levels = sorted([z_sorted[np.searchsorted(z_cumsum, p)] for p in pcts])

            cs = ax.contour(Xi, Yi, Zi, levels=kde_levels, colors="black",
                            linewidths=1.2 if not draw_heatmap else 1.0,
                            linestyles="solid", alpha=0.85 if not draw_heatmap else 0.7)
            fmt = {l: f"{int(round(p * 100))}%" for l, p in zip(kde_levels, pcts)}
            ax.clabel(cs, fmt=fmt, fontsize=7, inline=True)

            legend_label = (
                "80% order distribution (Shanghai)"
                if pcts == (0.80,)
                else "Order distribution (Shanghai)"
            )
            ax.plot([], [], color="black", linewidth=1.0, linestyle="solid", label=legend_label)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.7)


def plot_lifecycle_distance_route_factor(
    od_df=None,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
):
    """Shanghai order-density contour over air distance and detour ratio.

    Draws only the 80% weighted order-density contour, without the lifecycle
    background heatmap or colorbar.

    Args:
        od_df: OD pairs DataFrame with columns distance_km, road_distance_m, annual_sales.
        save_path: Path to save figure (with extension). Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _draw_lifecycle_distance_route_factor(
        ax,
        fig,
        od_df=od_df,
        plot_title=plot_title,
        draw_heatmap=False,
        contour_percentiles=(0.80,),
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Lifecycle distance route factor heatmap saved to {save_path}")
    else:
        plt.show()


def plot_lifecycle_distance_detour_boxplot(
    od_df: pd.DataFrame,
    save_path: str = None,
    figsize: tuple = (10, 6),
    for_latex: bool = False,
    plot_title: bool = False,
    bin_width_km: float = 0.25,
):
    """Heatmap of lifecycle emission difference with detour quantile mask bands.

    Args:
        od_df: OD pairs DataFrame with distance_km and road_distance_m columns.
        save_path: Path to save figure. Shows plot if None.
        figsize: Figure size tuple.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
        bin_width_km: Width of each shared air-distance bin in km.
    """
    from emission.cal_single_delivery import LifeCycleEmissions

    lca = LifeCycleEmissions()

    distance_min, distance_max = 0.2, 5.0
    heatmap_distance_min = 0.2
    detour_min, detour_max = 1.0, 1.8
    x_edges = np.arange(distance_min, distance_max + bin_width_km, bin_width_km)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    distances = np.linspace(heatmap_distance_min, distance_max, 140)
    detour_ratios = np.linspace(detour_min, detour_max, 90)

    def _emission_diff_matrix(distance_grid):
        matrix = np.zeros((len(detour_ratios), len(distance_grid)))
        for i, r in enumerate(detour_ratios):
            for j, d in enumerate(distance_grid):
                ebike_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d * r, n_orders_per_delivery=1, vehicle_type="ebike"
                )
                drone_result = lca.calculate_lifecycle_emissions_per_order_simple(
                    d, n_orders_per_delivery=1, vehicle_type="drone"
                )
                diff = (drone_result["total_per_order_g"] / ebike_result["total_per_order_g"] - 1) * 100
                matrix[i, j] = diff
        return matrix

    emission_diff_matrix = _emission_diff_matrix(distances)
    original_distances = np.linspace(heatmap_distance_min, 4, 30)
    original_emission_diff_matrix = _emission_diff_matrix(original_distances)

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    fig, ax = plt.subplots(figsize=figsize)
    cmap = LinearSegmentedColormap.from_list(
        "teal_yellow_orange",
        ["#006b6b", "#48c4b2", "#e5dfb6", "#f0a33a", "#df3b20"],
    )
    norm = TwoSlopeNorm(
        vmin=float(original_emission_diff_matrix.min()),
        vcenter=0.0,
        vmax=float(original_emission_diff_matrix.max()),
    )
    im = ax.imshow(
        emission_diff_matrix,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=[heatmap_distance_min, distance_max, detour_min, detour_max],
        norm=norm,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Drone emissions relative to e-bike (%)", fontsize=12)

    X, Y = np.meshgrid(distances, detour_ratios)
    data_min = np.floor(emission_diff_matrix.min() / 10) * 10
    negative_levels = np.arange(data_min, -9, 10)
    contour_levels = [*negative_levels, 0, 40, 80]
    contour = ax.contour(
        X,
        Y,
        emission_diff_matrix,
        levels=contour_levels,
        colors="black",
        linewidths=[1.5 if lvl == 0 else 1.0 for lvl in contour_levels],
        linestyles=["-" if lvl == 0 else "--" for lvl in contour_levels],
        alpha=0.65,
        zorder=4,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt=lambda x: f"{x:+.0f}%")

    valid = od_df.dropna(subset=["distance_km", "road_distance_m"]).copy()
    valid = valid[
        (valid["distance_km"] > distance_min)
        & (valid["distance_km"] <= distance_max)
        & (valid["road_distance_m"] >= 0)
    ]
    valid["detour"] = valid["road_distance_m"] / (valid["distance_km"] * 1000)
    valid = valid[np.isfinite(valid["detour"])]

    if len(valid) > 0:
        from scipy.ndimage import gaussian_filter1d

        mask_x = np.linspace(distance_min, distance_max, 700)
        quantile_levels = np.array([0.05, 0.10, 0.25, 0.40, 0.60, 0.75, 0.90, 0.95])
        quantile_curves = np.full((len(quantile_levels), len(x_centers)), np.nan)
        quantile_bands = [
            (None, 0, 0.66),
            (0, 1, 0.54),
            (1, 2, 0.42),
            (2, 3, 0.24),
            (3, 4, 0.00),
            (4, 5, 0.24),
            (5, 6, 0.42),
            (6, 7, 0.54),
            (7, None, 0.66),
        ]
        for x_i, (left, right) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            if right >= distance_max:
                mask = (valid["distance_km"] >= left) & (valid["distance_km"] <= right)
            else:
                mask = (valid["distance_km"] >= left) & (valid["distance_km"] < right)
            values = valid.loc[mask, "detour"].to_numpy()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            quantile_curves[:, x_i] = np.quantile(values, quantile_levels)

        observed = np.flatnonzero(np.isfinite(quantile_curves[0]))
        if observed.size > 1:
            smooth_curves = np.full((len(quantile_levels), len(mask_x)), np.nan)
            for q_i in range(len(quantile_levels)):
                curve = quantile_curves[q_i]
                curve = np.interp(x_centers, x_centers[observed], curve[observed])
                curve = gaussian_filter1d(curve, sigma=1.0, mode="nearest")
                smooth_curves[q_i] = np.interp(mask_x, x_centers, curve)

            for low_i, high_i, alpha in quantile_bands:
                if alpha <= 0:
                    continue
                y_low = np.full_like(mask_x, detour_min) if low_i is None else smooth_curves[low_i]
                y_high = np.full_like(mask_x, detour_max) if high_i is None else smooth_curves[high_i]
                ax.fill_between(
                    mask_x,
                    y_low,
                    y_high,
                    color="white",
                    alpha=alpha,
                    linewidth=0,
                    zorder=5,
                )

    ax.set_xlim(distance_min, distance_max)
    ax.set_ylim(detour_min, detour_max)
    ax.set_xlabel("Air distance (km)", fontsize=12)
    ax.set_ylabel("Detour ratio (road / air distance)", fontsize=12)
    if plot_title:
        ax.set_title("Emission comparison with observed detour distribution", fontsize=14)

    plt.tight_layout()

    if save_path:
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Lifecycle distance detour quantile mask heatmap saved to {save_path}")
    else:
        plt.show()
