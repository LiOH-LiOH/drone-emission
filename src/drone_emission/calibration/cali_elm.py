import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy.optimize import curve_fit

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import contextily as ctx
from utils.coordinates import gcj02_to_wgs84
from visualization._utils import configure_contextily_cache, finalize_plot, setup_figure

elm_data_dir = "/Users/linjingxian/Code/drone-emission/data/raw/即时配送数据/疫情期间北京饿了么/eleme_round1_train_20200313/order/"


def get_eg_data():
    # get file names in elm_data_dir
    file_names = os.listdir(elm_data_dir)
    file_paths = [
        os.path.join(elm_data_dir, fn) for fn in file_names if fn.endswith(".txt")
    ]

    eg_df = pd.read_csv(file_paths[0])
    # find the shod_id appearing most frequently and its location
    eg_id = eg_df["shop_id"].value_counts().idxmax()
    eg_shop_loc = (
        eg_df[eg_df["shop_id"] == eg_id][["pick_lng", "pick_lat"]].iloc[0].tolist()
    )

    # from all files, extract orders from this shop_id, only save deliver_lng, deliver_lat
    all_orders = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        shop_orders = df[df["shop_id"] == eg_id][["deliver_lng", "deliver_lat"]]
        all_orders.append(shop_orders)

    return eg_id, eg_shop_loc, pd.concat(all_orders, ignore_index=True)


def caliberate_elm_data():
    id, loc, orders = get_eg_data()
    print(f"Total orders from shop {id}: {len(orders)}")
    print(f"Shop location: {loc}")
    print(orders.head())

    min_lng, min_lat = orders[["deliver_lng", "deliver_lat"]].min().tolist()
    max_lng, max_lat = orders[["deliver_lng", "deliver_lat"]].max().tolist()
    print(f"Delivery location bounds:")
    print(f"Longitude: {min_lng} to {max_lng}")
    print(f"Latitude: {min_lat} to {max_lat}")


def plot_delivery_coverage(save_path=None):
    """绘制商家位置和配送范围散点图，带有地图底图"""
    shop_id, shop_loc, orders = get_eg_data()

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
    )


def load_all_eleme_distances(min_orders_per_shop=50):
    """Load all Eleme order files and compute delivery distances.

    Applies offset correction and GCJ02→WGS84 conversion, then computes
    Haversine distance (pick → deliver) for each order.

    Args:
        min_orders_per_shop: Minimum orders per shop to include (filters noise).

    Returns:
        np.ndarray of distances in km, filtered to (0, 10] km range.
    """
    OFFSET_LNG = -5.1192
    OFFSET_LAT = 0.781

    file_names = os.listdir(elm_data_dir)
    file_paths = [
        os.path.join(elm_data_dir, fn) for fn in file_names if fn.endswith(".txt")
    ]

    all_dfs = []
    for fp in file_paths:
        df = pd.read_csv(fp)
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"Loaded {len(combined)} total orders from {len(file_paths)} files")

    # Filter shops with enough orders
    shop_counts = combined["shop_id"].value_counts()
    valid_shops = shop_counts[shop_counts >= min_orders_per_shop].index
    combined = combined[combined["shop_id"].isin(valid_shops)]
    print(f"After filtering (>={min_orders_per_shop} orders/shop): {len(combined)} orders from {len(valid_shops)} shops")

    # Apply offset correction
    pick_lng = combined["pick_lng"].values + OFFSET_LNG
    pick_lat = combined["pick_lat"].values + OFFSET_LAT
    deliver_lng = combined["deliver_lng"].values + OFFSET_LNG
    deliver_lat = combined["deliver_lat"].values + OFFSET_LAT

    # Convert GCJ02 → WGS84
    pick_wgs = np.array([gcj02_to_wgs84(lng, lat) for lng, lat in zip(pick_lng, pick_lat)])
    deliver_wgs = np.array([gcj02_to_wgs84(lng, lat) for lng, lat in zip(deliver_lng, deliver_lat)])

    # Compute Haversine distances
    distances = np.array([
        haversine(
            (pick_wgs[i, 1], pick_wgs[i, 0]),  # (lat, lng)
            (deliver_wgs[i, 1], deliver_wgs[i, 0]),
            unit=Unit.KILOMETERS,
        )
        for i in range(len(pick_wgs))
    ])

    # Filter: 0 < distance <= 10 km
    mask = (distances > 0) & (distances <= 10)
    distances = distances[mask]
    print(f"After distance filter (0, 10] km: {len(distances)} orders")
    print(f"Mean distance: {distances.mean():.3f} km, Median: {np.median(distances):.3f} km")

    return distances


def fit_tanner_impedance(distances, bin_width=0.1, max_distance=10.0):
    """Fit Tanner impedance function N(d) = C * d^a * exp(-b*d) to distance histogram.

    Args:
        distances: Array of delivery distances in km.
        bin_width: Histogram bin width in km.
        max_distance: Maximum distance for binning.

    Returns:
        Tuple of (a, b, r_squared) where peak distance = a/b.
    """
    bins = np.arange(0, max_distance + bin_width, bin_width)
    counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Exclude zero-count bins for fitting
    mask = counts > 0
    x = bin_centers[mask]
    y = counts[mask].astype(float)

    def tanner(d, C, a, b):
        return C * (d ** a) * np.exp(-b * d)

    # Initial guess and bounds
    p0 = [y.max(), 1.5, 1.0]
    bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

    popt, _ = curve_fit(tanner, x, y, p0=p0, bounds=bounds, maxfev=10000)
    C_fit, a_fit, b_fit = popt

    # R-squared
    y_pred = tanner(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nTanner fit: N(d) = {C_fit:.1f} * d^{a_fit:.4f} * exp(-{b_fit:.4f}*d)")
    print(f"Peak distance (a/b): {a_fit / b_fit:.3f} km")
    print(f"R²: {r_squared:.4f}")

    return a_fit, b_fit, r_squared


def plot_tanner_fit(distances, a, b, save_path=None):
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
    ax.set_title("Distance Distribution with Tanner Fit")

    # Annotate parameters
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
    ax.set_title("CDF: Observed vs Tanner Fit")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()


def _sweep_alpha(alpha_values, indices_list, stores_df, buildings_gdf,
                  store_coords, tanner_a, tanner_b, min_sales):
    """Run one sweep of α values and return (alpha_values, mean_distances)."""
    from demand.api import generate_one_store_od_pairs_vectorized

    mean_distances = []
    n_alphas = len(alpha_values)
    for ai, alpha in enumerate(alpha_values):
        od_params = {
            "impedance_type": "tanner",
            "alpha": alpha,
            "tanner_a": tanner_a,
            "tanner_b": tanner_b,
            "min_sales": min_sales,
        }

        all_distances = []
        all_sales = []
        for store_idx, building_indices in enumerate(indices_list):
            if len(building_indices) == 0:
                continue
            nearby_buildings = buildings_gdf.iloc[building_indices]
            store_coord_wgs84 = store_coords[store_idx]
            month_sales = stores_df.iloc[store_idx]["month_sales"]

            od_pairs = generate_one_store_od_pairs_vectorized(
                nearby_buildings, month_sales, store_coord_wgs84, od_params
            )
            for pair in od_pairs:
                all_distances.append(pair["distance_km"])
                all_sales.append(pair["annual_sales"])

        if len(all_distances) > 0:
            weighted_mean = np.average(all_distances, weights=all_sales)
        else:
            weighted_mean = np.nan
        mean_distances.append(weighted_mean)
        print(f"  [{ai+1}/{n_alphas}] α = {alpha:.2f}  →  weighted mean distance = {weighted_mean:.4f} km  ({len(all_distances)} OD pairs)", flush=True)

    return np.array(mean_distances)


def calibrate_alpha(target_mean_distance=2.23, delivery_radius_km=None,
                    save_path=None):
    """Calibrate building volume exponent α by matching target mean delivery distance.

    Two-stage sweep: coarse (step 0.5) then fine (step 0.1) around the best.

    Args:
        target_mean_distance: Target weighted mean distance in km (default 2.23).
        delivery_radius_km: Override delivery radius (default: use experiment.toml).
        save_path: Path to save the calibration plot.

    Returns:
        The calibrated α value.
    """
    from pathlib import Path

    from demand.api import prepare_store_building_matching
    from utils import load_config
    from utils.region_config import get_registry, init_registry

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    REGIONS_CONFIG_PATH = PROJECT_ROOT / "regions.json"
    EXP_CONFIG_PATH = PROJECT_ROOT / "experiment.toml"

    registry = init_registry(str(REGIONS_CONFIG_PATH))
    config = load_config(str(EXP_CONFIG_PATH))

    if delivery_radius_km is not None:
        config["delivery_radius_km"] = delivery_radius_km
        print(f"Using delivery_radius_km = {delivery_radius_km}")

    region = registry["nanshan"]

    # Prepare store-building matching (expensive, do once)
    print("\n=== Preparing store-building matching ===")
    indices_list, stores_df, buildings_gdf, store_coords = (
        prepare_store_building_matching(region, config)
    )

    # Fixed Tanner distance impedance parameters
    tanner_a = 1.1131
    tanner_b = 1.4115
    min_sales = config.get("od_params", {}).get("min_sales", 5)

    # === Stage 1: Coarse sweep (step 0.5) ===
    coarse_alphas = np.arange(0.5, 5.5, 0.5)
    print(f"\n=== Stage 1: Coarse sweep ({len(coarse_alphas)} values, step=0.5) ===")
    coarse_means = _sweep_alpha(
        coarse_alphas, indices_list, stores_df, buildings_gdf,
        store_coords, tanner_a, tanner_b, min_sales,
    )

    # Find coarse best
    valid_mask = ~np.isnan(coarse_means)
    valid_alphas = coarse_alphas[valid_mask]
    valid_means = coarse_means[valid_mask]
    coarse_best_idx = np.argmin(np.abs(valid_means - target_mean_distance))
    coarse_best = valid_alphas[coarse_best_idx]
    print(f"\n  Coarse best: α = {coarse_best:.1f} (mean = {valid_means[coarse_best_idx]:.4f} km)")

    # === Stage 2: Fine sweep (step 0.1) around coarse best ===
    fine_lo = max(0.1, coarse_best - 0.5)
    fine_hi = coarse_best + 0.5
    fine_alphas = np.arange(fine_lo, fine_hi + 0.05, 0.1)
    # Remove duplicates already computed in coarse sweep
    fine_alphas = np.array([a for a in fine_alphas if not np.any(np.isclose(coarse_alphas, a))])

    if len(fine_alphas) > 0:
        print(f"\n=== Stage 2: Fine sweep ({len(fine_alphas)} values, step=0.1, range [{fine_lo:.1f}, {fine_hi:.1f}]) ===")
        fine_means = _sweep_alpha(
            fine_alphas, indices_list, stores_df, buildings_gdf,
            store_coords, tanner_a, tanner_b, min_sales,
        )
    else:
        fine_alphas = np.array([])
        fine_means = np.array([])

    # Combine coarse + fine results
    all_alphas = np.concatenate([coarse_alphas[valid_mask], fine_alphas[~np.isnan(fine_means)] if len(fine_means) > 0 else np.array([])])
    all_means = np.concatenate([valid_means, fine_means[~np.isnan(fine_means)] if len(fine_means) > 0 else np.array([])])
    sort_idx = np.argsort(all_alphas)
    all_alphas = all_alphas[sort_idx]
    all_means = all_means[sort_idx]

    # Find overall best
    best_idx = np.argmin(np.abs(all_means - target_mean_distance))
    best_alpha = all_alphas[best_idx]
    best_mean = all_means[best_idx]

    print(f"\n=== Calibration Result ===")
    print(f"Delivery radius: {config['delivery_radius_km']} km")
    print(f"Target mean distance: {target_mean_distance} km")
    print(f"Best α: {best_alpha:.2f}")
    print(f"Achieved mean distance: {best_mean:.4f} km")
    print(f"Error: {abs(best_mean - target_mean_distance):.4f} km")

    # Plot
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
    ax.set_title(f"Calibration of α (delivery radius = {config['delivery_radius_km']} km)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved to {save_path}")
    plt.show()

    return best_alpha


if __name__ == "__main__":
    # Project root figures directory
    FIGURES_DIR = "/Users/linjingxian/Code/drone-emission/figures"

    caliberate_elm_data()
    save_path = os.path.join(FIGURES_DIR, "calibration", "delivery_coverage.png")
    plot_delivery_coverage(save_path=save_path)

    # Tanner impedance function fitting
    print("\n=== Tanner Impedance Function Fitting ===")
    distances = load_all_eleme_distances(min_orders_per_shop=50)
    a, b, r_sq = fit_tanner_impedance(distances)
    plot_tanner_fit(
        distances, a, b,
        save_path=os.path.join(FIGURES_DIR, "calibration", "tanner_impedance_fit.png"),
    )

    # Alpha calibration
    print("\n=== Alpha Calibration ===")
    best_alpha = calibrate_alpha(
        target_mean_distance=2.23,
        save_path=os.path.join(FIGURES_DIR, "calibration", "alpha_calibration.png"),
    )
    print(f"\nCalibrated α = {best_alpha:.1f}")
    print(f"Update experiment.toml: alpha = {best_alpha}")
