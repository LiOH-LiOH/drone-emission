import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from haversine import Unit, haversine
from scipy.optimize import curve_fit

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.coordinates import gcj02_to_wgs84
from visualization.calibration import (
    plot_alpha_calibration,
    plot_delivery_coverage,
    plot_tanner_distance_histogram,
    plot_tanner_fit,
)

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


def load_all_eleme_distances(min_orders_per_shop=50, max_distance_km=10.0):
    """Load all Eleme order files and compute delivery distances.

    Applies offset correction and GCJ02→WGS84 conversion, then computes
    Haversine distance (pick → deliver) for each order.

    Args:
        min_orders_per_shop: Minimum orders per shop to include (filters noise).
            If None, include every shop.
        max_distance_km: Maximum retained distance in km. If None, retain all
            finite positive distances.

    Returns:
        np.ndarray of finite positive distances in km, optionally capped at
        max_distance_km.
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

    # Optionally filter shops with enough orders.
    shop_counts = combined["shop_id"].value_counts()
    if min_orders_per_shop is not None:
        valid_shops = shop_counts[shop_counts >= min_orders_per_shop].index
        combined = combined[combined["shop_id"].isin(valid_shops)]
        print(
            f"After filtering (>={min_orders_per_shop} orders/shop): "
            f"{len(combined)} orders from {len(valid_shops)} shops"
        )
    else:
        print(
            "Shop-order filter disabled: "
            f"{len(combined)} orders from {len(shop_counts)} shops"
        )

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

    # Filter invalid and zero-distance observations, with an optional upper cap.
    mask = np.isfinite(distances) & (distances > 0)
    if max_distance_km is not None:
        mask &= distances <= max_distance_km
    distances = distances[mask]
    if max_distance_km is None:
        distance_range = "(0, inf) km"
    else:
        distance_range = f"(0, {max_distance_km:g}] km"
    print(f"After distance filter {distance_range}: {len(distances)} orders")
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


def _sweep_alpha(alpha_values, indices_list, stores_df, buildings_gdf,
                  store_coords, tanner_a, tanner_b, min_sales,
                  store_weights=None):
    """Run one sweep of α values and return (alpha_values, mean_distances)."""
    from demand.api import generate_one_store_od_pairs_vectorized

    mean_distances = []
    n_alphas = len(alpha_values)
    if store_weights is None:
        store_weights = np.ones(len(indices_list), dtype=float)
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
            store_weight = float(store_weights[store_idx])

            od_pairs = generate_one_store_od_pairs_vectorized(
                nearby_buildings, month_sales, store_coord_wgs84, od_params
            )
            for pair in od_pairs:
                all_distances.append(pair["distance_km"])
                all_sales.append(pair["annual_sales"] * store_weight)

        if len(all_distances) > 0:
            weighted_mean = np.average(all_distances, weights=all_sales)
        else:
            weighted_mean = np.nan
        mean_distances.append(weighted_mean)
        print(f"  [{ai+1}/{n_alphas}] α = {alpha:.2f}  →  weighted mean distance = {weighted_mean:.4f} km  ({len(all_distances)} OD pairs)", flush=True)

    return np.array(mean_distances)


def _sample_alpha_calibration_inputs(
    indices_list,
    stores_df,
    store_coords,
    sample_stores=None,
    sample_buildings_per_store=None,
    random_state=0,
    sample_strategy="stratified_sales_spatial",
):
    """Subsample stores with expansion weights for calibration checks."""
    if sample_stores is not None and sample_stores <= 0:
        raise ValueError("sample_stores must be a positive integer")
    if sample_buildings_per_store is not None:
        raise ValueError(
            "sample_buildings_per_store is disabled for alpha calibration; "
            "building truncation changes per-store demand normalization"
        )

    rng = np.random.default_rng(random_state)
    n_stores = len(indices_list)
    if sample_stores is None or sample_stores >= n_stores:
        return (
            list(indices_list),
            stores_df,
            store_coords,
            np.ones(n_stores, dtype=float),
            "full",
            0,
            0,
        )
    if sample_strategy != "stratified_sales_spatial":
        raise ValueError(
            "sample_strategy must be 'stratified_sales_spatial' for sampled "
            "alpha calibration"
        )

    sample_df = stores_df[["month_sales", "lat", "lng"]].copy()
    sample_df["_store_idx"] = np.arange(n_stores)
    sample_df["_sales_rank"] = sample_df["month_sales"].rank(
        method="first", ascending=False
    )
    certainty_count = min(
        int(np.ceil(n_stores * 0.05)),
        max(1, sample_stores // 5),
    )
    certainty_idx = set(
        sample_df.nsmallest(certainty_count, "_sales_rank")["_store_idx"]
    )

    remaining = sample_df[~sample_df["_store_idx"].isin(certainty_idx)].copy()
    remaining_slots = sample_stores - len(certainty_idx)
    if remaining_slots <= 0:
        selected_idx = sorted(certainty_idx)
        sampled_indices = [indices_list[i] for i in selected_idx]
        sampled_stores = stores_df.iloc[selected_idx].reset_index(drop=True)
        sampled_store_coords = store_coords[selected_idx]
        return (
            sampled_indices,
            sampled_stores,
            sampled_store_coords,
            np.ones(len(selected_idx), dtype=float),
            sample_strategy,
            len(certainty_idx),
            0,
        )

    sales_bins = pd.qcut(
        remaining["month_sales"].rank(method="first"),
        q=min(4, len(remaining)),
        labels=False,
        duplicates="drop",
    )
    lat_bins = pd.qcut(
        remaining["lat"].rank(method="first"),
        q=min(3, len(remaining)),
        labels=False,
        duplicates="drop",
    )
    lng_bins = pd.qcut(
        remaining["lng"].rank(method="first"),
        q=min(3, len(remaining)),
        labels=False,
        duplicates="drop",
    )
    remaining["_stratum"] = (
        sales_bins.astype(str) + "_" + lat_bins.astype(str) + "_" + lng_bins.astype(str)
    )
    strata = list(remaining.groupby("_stratum", sort=False))
    strata_sizes = np.array([len(group) for _, group in strata], dtype=float)
    raw_alloc = remaining_slots * strata_sizes / strata_sizes.sum()
    alloc = np.floor(raw_alloc).astype(int)
    non_empty = strata_sizes > 0
    alloc[(alloc == 0) & non_empty & (remaining_slots >= non_empty.sum())] = 1

    while alloc.sum() > remaining_slots:
        candidates = np.where(alloc > 1)[0]
        if len(candidates) == 0:
            candidates = np.where(alloc > 0)[0]
        drop_idx = candidates[np.argmin(raw_alloc[candidates] - alloc[candidates])]
        alloc[drop_idx] -= 1
    while alloc.sum() < remaining_slots:
        candidates = np.where(alloc < strata_sizes)[0]
        add_idx = candidates[np.argmax(raw_alloc[candidates] - alloc[candidates])]
        alloc[add_idx] += 1

    selected_idx = list(certainty_idx)
    weight_by_idx = {idx: 1.0 for idx in certainty_idx}
    for (_, group), n_sample in zip(strata, alloc):
        if n_sample <= 0:
            continue
        group_indices = group["_store_idx"].to_numpy()
        sampled = rng.choice(group_indices, size=n_sample, replace=False)
        selected_idx.extend(sampled.tolist())
        expansion_weight = len(group_indices) / n_sample
        for idx in sampled:
            weight_by_idx[int(idx)] = float(expansion_weight)

    selected_idx = sorted(selected_idx)
    sampled_indices = [indices_list[i] for i in selected_idx]
    sampled_stores = stores_df.iloc[selected_idx].reset_index(drop=True)
    sampled_store_coords = store_coords[selected_idx]
    store_weights = np.array([weight_by_idx[i] for i in selected_idx], dtype=float)

    return (
        sampled_indices,
        sampled_stores,
        sampled_store_coords,
        store_weights,
        sample_strategy,
        len(certainty_idx),
        len(strata),
    )


def build_alpha_calibration_sweep(
    target_mean_distance=2.23,
    delivery_radius_km=None,
    sample_stores=None,
    sample_buildings_per_store=None,
    sample_strategy="stratified_sales_spatial",
    random_state=0,
    coarse_only=False,
):
    """Build α calibration sweep data by matching target mean delivery distance.

    Two-stage sweep: coarse (step 0.5) then fine (step 0.1) around the best.

    Args:
        target_mean_distance: Target weighted mean distance in km (default 2.23).
        delivery_radius_km: Override delivery radius (default: use experiment.toml).
        sample_stores: Optional number of stores to sample for fast checks.
        sample_buildings_per_store: Deprecated; building caps are disabled here.
        sample_strategy: Store sampling strategy for sampled checks.
        random_state: Random seed for reproducible sampling.
        coarse_only: If True, skip the fine sweep around the coarse best.

    Returns:
        Tuple of (sweep_df, best_alpha, best_mean).
    """
    from pathlib import Path

    from demand.api import prepare_store_building_matching
    from utils import load_config
    from utils.region_config import init_registry

    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    EXP_CONFIG_PATH = PROJECT_ROOT / "experiment.toml"

    region_key = "nanshan"
    region_config_candidates = (
        PROJECT_ROOT / "regions.json",
        PROJECT_ROOT / "backup_regions.json",
        PROJECT_ROOT / "regions.template.json",
    )
    registry = None
    for config_path in region_config_candidates:
        candidate_registry = init_registry(str(config_path))
        if region_key in candidate_registry:
            registry = candidate_registry
            break
    if registry is None:
        raise KeyError(
            f"Region '{region_key}' not found in active or fallback region configs"
        )
    config = load_config(str(EXP_CONFIG_PATH))

    if delivery_radius_km is not None:
        config["delivery_radius_km"] = delivery_radius_km
        print(f"Using delivery_radius_km = {delivery_radius_km}")

    region = registry[region_key]

    # Prepare store-building matching (expensive, do once)
    print("\n=== Preparing store-building matching ===")
    indices_list, stores_df, buildings_gdf, store_coords = (
        prepare_store_building_matching(region, config)
    )
    full_store_count = len(stores_df)
    full_building_count = len(buildings_gdf)
    store_weights = np.ones(full_store_count, dtype=float)
    resolved_sample_strategy = "full"
    certainty_store_count = 0
    stratum_count = 0
    if sample_stores is not None or sample_buildings_per_store is not None:
        (
            indices_list,
            stores_df,
            store_coords,
            store_weights,
            resolved_sample_strategy,
            certainty_store_count,
            stratum_count,
        ) = _sample_alpha_calibration_inputs(
            indices_list,
            stores_df,
            store_coords,
            sample_stores=sample_stores,
            sample_buildings_per_store=sample_buildings_per_store,
            random_state=random_state,
            sample_strategy=sample_strategy,
        )
        total_pairs = sum(len(idx) for idx in indices_list)
        print(
            "\n=== Alpha calibration sampled input ===\n"
            f"Stores: {len(stores_df)} of {full_store_count}\n"
            f"Store-building pairs after sampling: {total_pairs}\n"
            f"Strategy: {resolved_sample_strategy}\n"
            f"Store expansion weight range: "
            f"{store_weights.min():.2f}-{store_weights.max():.2f}"
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
        store_weights=store_weights,
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

    if coarse_only:
        print("\n=== Stage 2: Fine sweep skipped (coarse_only=True) ===")
        fine_alphas = np.array([])
        fine_means = np.array([])
    elif len(fine_alphas) > 0:
        print(f"\n=== Stage 2: Fine sweep ({len(fine_alphas)} values, step=0.1, range [{fine_lo:.1f}, {fine_hi:.1f}]) ===")
        fine_means = _sweep_alpha(
            fine_alphas, indices_list, stores_df, buildings_gdf,
            store_coords, tanner_a, tanner_b, min_sales,
            store_weights=store_weights,
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

    sweep_df = pd.DataFrame(
        {
            "alpha": all_alphas,
            "weighted_mean_od_distance_km": all_means,
        }
    )
    sweep_df["target_mean_distance_km"] = target_mean_distance
    sweep_df["is_best"] = np.isclose(sweep_df["alpha"], best_alpha)
    sweep_df["delivery_radius_km"] = config["delivery_radius_km"]
    sweep_df["region_key"] = region_key
    sweep_df["store_count"] = len(stores_df)
    sweep_df["building_count"] = len(buildings_gdf)
    sweep_df["full_store_count"] = full_store_count
    sweep_df["full_building_count"] = full_building_count
    sweep_df["sample_stores"] = sample_stores
    sweep_df["sample_buildings_per_store"] = sample_buildings_per_store
    sweep_df["sample_strategy"] = resolved_sample_strategy
    sweep_df["certainty_store_count"] = certainty_store_count
    sweep_df["stratum_count"] = stratum_count
    sweep_df["uses_expansion_weights"] = not np.allclose(store_weights, 1.0)
    sweep_df["min_store_expansion_weight"] = float(store_weights.min())
    sweep_df["max_store_expansion_weight"] = float(store_weights.max())
    sweep_df["random_state"] = random_state
    sweep_df["is_sampled"] = (
        sample_stores is not None or sample_buildings_per_store is not None
    )
    sweep_df["coarse_only"] = coarse_only
    sweep_df["tanner_beta"] = tanner_a
    sweep_df["tanner_gamma"] = tanner_b
    sweep_df["min_sales"] = min_sales

    return sweep_df, best_alpha, best_mean


def export_alpha_calibration_sweep_csv(
    save_path=None,
    target_mean_distance=2.23,
    delivery_radius_km=5.0,
    sample_stores=None,
    sample_buildings_per_store=None,
    sample_strategy="stratified_sales_spatial",
    random_state=0,
    coarse_only=False,
):
    """Export the α calibration sweep used by paper figures.

    Args:
        save_path: CSV path to write. Defaults to alpha_calibration_5km.csv
            beside this module.
        target_mean_distance: Target weighted mean distance in km.
        delivery_radius_km: Delivery radius used for the sweep.
        sample_stores: Optional number of stores to sample for fast checks.
        sample_buildings_per_store: Deprecated; building caps are disabled here.
        sample_strategy: Store sampling strategy for sampled checks.
        random_state: Random seed for reproducible sampling.
        coarse_only: If True, skip the fine sweep around the coarse best.

    Returns:
        Path to the written CSV.
    """
    from pathlib import Path

    if save_path is None:
        save_path = Path(__file__).with_name("alpha_calibration_5km.csv")
    else:
        save_path = Path(save_path)

    sweep_df, best_alpha, best_mean = build_alpha_calibration_sweep(
        target_mean_distance=target_mean_distance,
        delivery_radius_km=delivery_radius_km,
        sample_stores=sample_stores,
        sample_buildings_per_store=sample_buildings_per_store,
        sample_strategy=sample_strategy,
        random_state=random_state,
        coarse_only=coarse_only,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(save_path, index=False)
    print(
        "Alpha calibration sweep saved to "
        f"{save_path} (best α = {best_alpha:.2f}, mean = {best_mean:.4f} km)"
    )
    return save_path


def calibrate_alpha(target_mean_distance=2.23, delivery_radius_km=None,
                    save_path=None, for_latex=False, plot_title=False):
    """Calibrate building volume exponent α by matching target mean delivery distance.

    Two-stage sweep: coarse (step 0.5) then fine (step 0.1) around the best.

    Args:
        target_mean_distance: Target weighted mean distance in km (default 2.23).
        delivery_radius_km: Override delivery radius (default: use experiment.toml).
        save_path: Path to save the calibration plot.

    Returns:
        The calibrated α value.
    """
    sweep_df, best_alpha, best_mean = build_alpha_calibration_sweep(
        target_mean_distance=target_mean_distance,
        delivery_radius_km=delivery_radius_km,
    )

    plot_alpha_calibration(
        sweep_df["alpha"].to_numpy(),
        sweep_df["weighted_mean_od_distance_km"].to_numpy(),
        target_mean_distance,
        best_alpha,
        best_mean,
        float(sweep_df["delivery_radius_km"].iloc[0]),
        save_path,
        for_latex,
        plot_title,
    )

    return best_alpha


if __name__ == "__main__":
    # Project root figures directory
    FIGURES_DIR = "/Users/linjingxian/Code/drone-emission/figures"

    caliberate_elm_data()
    shop_id, shop_loc, orders = get_eg_data()
    save_path = os.path.join(FIGURES_DIR, "calibration", "delivery_coverage.png")
    plot_delivery_coverage(shop_id, shop_loc, orders, save_path=save_path)

    # Tanner impedance function fitting
    print("\n=== Tanner Impedance Function Fitting ===")
    distances = load_all_eleme_distances(min_orders_per_shop=50)
    a, b, r_sq = fit_tanner_impedance(distances)
    plot_tanner_fit(
        distances, a, b,
        save_path=os.path.join(FIGURES_DIR, "calibration", "tanner_impedance_fit.png"),
    )
    plot_tanner_distance_histogram(
        distances,
        save_path=os.path.join(
            FIGURES_DIR, "paper", "tanner_impedance_fit_panel_a.pdf"
        ),
    )

    # Alpha calibration
    print("\n=== Alpha Calibration ===")
    best_alpha = calibrate_alpha(
        target_mean_distance=2.23,
        save_path=os.path.join(FIGURES_DIR, "calibration", "alpha_calibration.png"),
    )
    print(f"\nCalibrated α = {best_alpha:.1f}")
    print(f"Update experiment.toml: alpha = {best_alpha}")
