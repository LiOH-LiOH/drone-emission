import os

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from utils.coordinates import gcj02_to_wgs84
from visualization._utils import (
    _add_title_suffix,
    configure_contextily_cache,
    finalize_plot,
    setup_figure,
)


def plot_delivery_coverage(
    shop_id, shop_loc, orders, save_path=None, for_latex=False, plot_title=False
):
    """绘制商家位置和配送范围散点图，带有地图底图"""

    # 数据偏移修正（通过POI匹配确定的精确偏移量）
    # 原始数据被偏移：经度+5.1192°，纬度-0.781°
    # 匹配率97.0%（11267/11633商家），平均误差约28.8米
    OFFSET_LNG = -5.1192
    OFFSET_LAT = 0.781

    # 提取配送点坐标并修正偏移
    deliver_lng = orders["deliver_lng"].values + OFFSET_LNG
    deliver_lat = orders["deliver_lat"].values + OFFSET_LAT

    # 修正商家坐标偏移
    shop_lng_corrected = shop_loc[0] + OFFSET_LNG
    shop_lat_corrected = shop_loc[1] + OFFSET_LAT

    # 将 GCJ02 坐标转换为 WGS84
    shop_lng_wgs, shop_lat_wgs = gcj02_to_wgs84(shop_lng_corrected, shop_lat_corrected)

    # 转换配送点坐标
    deliver_lng_wgs = []
    deliver_lat_wgs = []
    for lng, lat in zip(deliver_lng, deliver_lat):
        w_lng, w_lat = gcj02_to_wgs84(lng, lat)
        deliver_lng_wgs.append(w_lng)
        deliver_lat_wgs.append(w_lat)
    deliver_lng_wgs = np.array(deliver_lng_wgs)
    deliver_lat_wgs = np.array(deliver_lat_wgs)

    # 计算边界（包含商家位置和所有配送点）
    all_lng = np.append(deliver_lng_wgs, shop_lng_wgs)
    all_lat = np.append(deliver_lat_wgs, shop_lat_wgs)

    # 添加padding
    padding = 0.005
    boundaries_dict = {
        "min_lng": all_lng.min() - padding,
        "max_lng": all_lng.max() + padding,
        "min_lat": all_lat.min() - padding,
        "max_lat": all_lat.max() + padding,
    }

    # 设置图形
    fig, ax, aspect_ratio, _, _ = setup_figure(all_lng, all_lat, boundaries_dict)

    # 绘制配送点散点图
    sc = ax.scatter(
        deliver_lng_wgs,
        deliver_lat_wgs,
        c="blue",
        s=10,
        alpha=0.5,
        zorder=10,
        edgecolors="none",
        label=f"Delivery Points ({len(orders)})",
    )

    # 绘制商家位置（用红色大点标记）
    ax.scatter(
        shop_lng_wgs,
        shop_lat_wgs,
        c="red",
        s=200,
        marker="*",
        zorder=20,
        edgecolors="black",
        linewidth=1,
        label=f"Shop (ID: {shop_id})",
    )

    # 添加地图底图（不绘制行政边界）
    configure_contextily_cache()
    ax.set_xlim(boundaries_dict["min_lng"], boundaries_dict["max_lng"])
    ax.set_ylim(boundaries_dict["min_lat"], boundaries_dict["max_lat"])
    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=14,
        alpha=0.6,
        zorder=0,
    )

    # 添加图例
    ax.legend(loc="upper right", fontsize=10)

    # 统计信息
    stats_text = (
        f"Shop ID: {shop_id}\n"
        f"Total Orders: {len(orders)}\n"
        f"Shop Location:\n"
        f"  Lng: {shop_lng_wgs:.6f}\n"
        f"  Lat: {shop_lat_wgs:.6f}"
    )

    finalize_plot(
        fig,
        ax,
        save_path,
        f"Eleme Shop {shop_id} Delivery Coverage",
        stats_text,
        boundaries_dict,
        for_latex,
        plot_title,
    )


def plot_tanner_fit(distances, a, b, save_path=None, for_latex=False, plot_title=False):
    """Plot observed distance distribution with fitted Tanner curve.

    Panel 1: Histogram + fitted Tanner curve overlay.
    Panel 2: CDF of observed vs fitted.

    Args:
        distances: Array of delivery distances in km.
        a: Tanner exponent parameter.
        b: Tanner decay parameter.
        save_path: Path to save the figure.
    """
    bin_width = 0.1
    max_distance = 10.0
    bins = np.arange(0, max_distance + bin_width, bin_width)
    counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit C for plotting (a and b are given)
    mask = counts > 0
    x_fit = bin_centers[mask]
    y_fit = counts[mask].astype(float)

    def tanner_fixed(d, C):
        return C * (d ** a) * np.exp(-b * d)

    popt, _ = curve_fit(tanner_fixed, x_fit, y_fit, p0=[y_fit.max()])
    C_fit = popt[0]

    # R-squared
    y_pred = C_fit * (x_fit ** a) * np.exp(-b * x_fit)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - y_fit.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Histogram + fitted curve
    ax = axes[0]
    ax.bar(
        bin_centers, counts, width=bin_width * 0.9,
        color="#4C72B0", edgecolor="white", alpha=0.85,
        label="Observed",
    )
    d_smooth = np.linspace(0.01, max_distance, 500)
    y_smooth = C_fit * (d_smooth ** a) * np.exp(-b * d_smooth)
    ax.plot(d_smooth, y_smooth, "r-", lw=2, label="Tanner fit")

    d_peak = a / b
    ax.axvline(d_peak, color="orange", ls="--", lw=1.5, label=f"Peak = {d_peak:.2f} km")
    ax.axvline(distances.mean(), color="red", ls="--", lw=1, label=f"Mean = {distances.mean():.2f} km")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Order Count")
    if plot_title:
        ax.set_title("Distance Distribution with Tanner Fit")

    # Annotate parameters
    if plot_title:
        text = (
            f"$f(d) = C \\cdot d^{{{a:.3f}}} \\cdot e^{{-{b:.3f} \\cdot d}}$\n"
            f"$d_{{peak}} = a/b = {d_peak:.2f}$ km\n"
            f"$R^2 = {r_squared:.4f}$"
        )
        ax.text(
            0.97, 0.97, text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
    ax.legend(fontsize=8, loc="center right")

    # Panel 2: CDF comparison
    ax = axes[1]
    sorted_d = np.sort(distances)
    cdf_obs = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf_obs, color="#4C72B0", lw=1.5, label="Observed CDF")

    # Fitted CDF (numerical integration of Tanner PDF)
    d_cdf = np.linspace(0.01, max_distance, 1000)
    pdf_fitted = (d_cdf ** a) * np.exp(-b * d_cdf)
    cdf_fitted = np.cumsum(pdf_fitted)
    cdf_fitted = cdf_fitted / cdf_fitted[-1]
    ax.plot(d_cdf, cdf_fitted, "r--", lw=1.5, label="Tanner CDF")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Cumulative Proportion")
    if plot_title:
        ax.set_title("CDF: Observed vs Tanner Fit")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


def plot_milkrun_distance_analysis(df, save_path=None, for_latex=False, plot_title=False):
    """Create a 1×3 figure: histogram, CDF, distance-vs-time scatter."""
    dist_km = df["dist_km"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Histogram ---
    ax = axes[0]
    bin_edges = np.arange(0, dist_km.max() + 1, 1)
    ax.hist(dist_km, bins=bin_edges, color="#4C72B0", edgecolor="white", alpha=0.85)

    median_d = dist_km.median()
    mean_d = dist_km.mean()
    ax.axvline(median_d, color="red", ls="--", lw=1, label=f"Median: {median_d:.1f} km")
    ax.axvline(mean_d, color="orange", ls="--", lw=1, label=f"Mean: {mean_d:.1f} km")
    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Order Count")
    if plot_title:
        ax.set_title("Distance Distribution")
    ax.legend(fontsize=8)

    # --- Panel 2: CDF ---
    ax = axes[1]
    sorted_d = np.sort(dist_km.values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, color="#4C72B0", lw=1.5)

    percentiles = [25, 50, 75, 90]
    pct_values = np.percentile(sorted_d, percentiles)
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for p, v, c in zip(percentiles, pct_values, colors):
        ax.axhline(p / 100, color=c, ls=":", lw=0.8, alpha=0.6)
        ax.axvline(v, color=c, ls=":", lw=0.8, alpha=0.6)
        ax.plot(v, p / 100, "o", color=c, ms=5)
        ax.annotate(f"P{p}: {v:.1f} km", xy=(v, p / 100), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Cumulative Proportion")
    if plot_title:
        ax.set_title("Distance CDF")
    ax.set_ylim(0, 1.05)

    # --- Panel 3: Distance vs Time scatter ---
    ax = axes[2]
    # Need both distance and duration
    valid = df[["dist_km", "duration_min"]].dropna()
    valid = valid[(valid["duration_min"] > 0) & (valid["duration_min"] <= 180)]
    d = valid["dist_km"].values
    t = valid["duration_min"].values

    ax.scatter(d, t, s=1, alpha=0.05, color="#4C72B0", rasterized=True)

    # Linear fit
    slope, intercept, r, _, _ = stats.linregress(d, t)
    x_fit = np.linspace(0, d.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, "r-", lw=1.5,
            label=f"Fit: {slope:.1f} min/km + {intercept:.1f} min\n"
                  f"→ avg speed ≈ {60 / slope:.0f} km/h  (R²={r**2:.2f})")

    ax.set_xlabel("Delivery Distance (km)")
    ax.set_ylabel("Delivery Duration (min)")
    if plot_title:
        ax.set_title("Distance vs Delivery Time")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


def plot_alpha_calibration(
    all_alphas, all_means, target_mean_distance, best_alpha, best_mean,
    delivery_radius_km, save_path=None, for_latex=False, plot_title=False,
):
    """Plot α calibration results: mean distance vs α with target line.

    Args:
        all_alphas: Array of α values tested.
        all_means: Array of corresponding weighted mean distances.
        target_mean_distance: Target mean distance in km.
        best_alpha: Best α value found.
        best_mean: Mean distance at best α.
        delivery_radius_km: Delivery radius used (for title).
        save_path: Path to save the figure.
        for_latex: If True, save as PDF.
        plot_title: If True, show title.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(all_alphas, all_means, "o-", color="#4C72B0", lw=2, markersize=5)
    ax.axhline(
        target_mean_distance, color="red", ls="--", lw=1.5,
        label=f"Target = {target_mean_distance} km",
    )
    ax.axvline(
        best_alpha, color="orange", ls="--", lw=1.5,
        label=f"Best α = {best_alpha:.2f} (mean = {best_mean:.2f} km)",
    )
    ax.set_xlabel("α (Volume Exponent)")
    ax.set_ylabel("Weighted Mean OD Distance (km)")
    if plot_title:
        ax.set_title(f"Calibration of α (delivery radius = {delivery_radius_km} km)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if for_latex and not save_path.endswith(".pdf"):
            save_path = save_path.rsplit(".", 1)[0] + ".pdf"
        if plot_title:
            save_path = _add_title_suffix(save_path)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()
