"""
排放分析可视化模块
"""

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_emission_reduction_analysis(
    results_df: pd.DataFrame,
    save_path: str = None,
    dpi: int = 150,
) -> str:
    """
    绘制排放减少分析图

    参数:
        results_df: 包含 threshold, drone_ratio, reduction 列的 DataFrame
        save_path: 保存路径，如果为 None 则显示图形
        dpi: 图像分辨率

    返回:
        保存的图像路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 阈值 vs 减排百分比
    ax1 = axes[0]
    ax1.plot(
        results_df["threshold"],
        results_df["reduction"],
        "b-o",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Distance Threshold (km)", fontsize=12)
    ax1.set_ylabel("Emission Reduction (%)", fontsize=12)
    ax1.set_title("Emission Reduction vs Distance Threshold", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 图2: 无人机占比 vs 减排百分比
    ax2 = axes[1]
    ax2.plot(
        results_df["drone_ratio"],
        results_df["reduction"],
        "r-s",
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("Drone Delivery Ratio (%)", fontsize=12)
    ax2.set_ylabel("Emission Reduction (%)", fontsize=12)
    ax2.set_title("Emission Reduction vs Drone Ratio", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_emission_comparison_bar(
    ebike_emissions: float,
    drone_emissions: float,
    mixed_emissions: float = None,
    save_path: str = None,
    dpi: int = 150,
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
    ax.set_title("Emission Comparison by Delivery Mode", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    ax.set_title("Distribution of Delivery Distances", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    ax1.set_title(f"Emissions vs {param_name}", fontsize=14)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 添加实验设定信息
    if experiment_settings:
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

    fig, ax = plt.subplots(figsize=(12, 8))

    # 创建热力图
    im = ax.imshow(
        emission_diff_matrix,
        cmap="RdYlGn_r",  # 红色=无人机排放高，绿色=无人机排放低
        aspect="auto",
        origin="lower",
        extent=[distances[0], distances[-1], route_factors[0], route_factors[-1]],
    )

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Drone Emissions Relative to E-bike (%)", fontsize=12)

    # 添加等值线
    X, Y = np.meshgrid(distances, route_factors)
    contour = ax.contour(
        X, Y, emission_diff_matrix, levels=[0], colors="black", linewidths=2
    )
    ax.clabel(contour, inline=True, fontsize=10, fmt="Breakeven")

    ax.set_xlabel("Delivery Distance (km)", fontsize=12)
    ax.set_ylabel("Route Factor (road/straight-line)", fontsize=12)
    ax.set_title(
        "Emission Comparison: Drone vs E-bike\n(Negative = Drone Better)", fontsize=14
    )

    # 添加实验设置信息框
    add_experiment_info_box(ax, experiment_settings, loc="upper left")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    ax1.set_title(f"{vehicle_type} Power vs Velocity", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 图2: 速度 vs 排放
    ax2 = axes[1]
    ax2.plot(velocities, emissions, color_emission, linewidth=2, markersize=6)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel("Emissions (g CO2e)", fontsize=12)
    title = f"{vehicle_type} Emissions vs Velocity"
    if distance_info:
        title += f" ({distance_info})"
    ax2.set_title(title, fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 添加实验设置信息框
    add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

    ax1.set_title("Power vs Velocity Comparison", fontsize=14)

    # 添加实验设置信息框
    add_experiment_info_box(ax1, experiment_settings, loc="lower right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return save_path
    else:
        plt.show()
        return None
