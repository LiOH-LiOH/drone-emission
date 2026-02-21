import os

import numpy as np
import pandas as pd
from emission.cal_single_delivery import (
    create_drone_params,
    create_ebike_params,
    get_emission_params,
)
from visualization.profit import (
    plot_cost_breakdown_by_distance,
    plot_profit_comparison,
    plot_profit_vs_distance,
)

from .cal_profit import (
    ProfitAnalysis,
    create_drone_cost_params,
    create_ebike_cost_params,
    create_revenue_params,
)


def cal_profit(region_info, exp_config) -> pd.DataFrame:
    """Calculate per-order profit for all OD pairs in a region.

    Analogous to emission/api.py:cal_emission.
    """
    region_config = region_info.to_dict()
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]

    od_save_path = os.path.join(
        exp_config["data_dir"],
        region_level,
        region_name_en.lower(),
        f"{region_name_en.lower()}_od_pairs.csv",
    )
    od_df = pd.read_csv(od_save_path)
    print(f"Calculating profit for {len(od_df)} OD pairs in {region_name_en}...")

    # Load parameters
    emission_cfg = get_emission_params(exp_config)
    payload_kg = emission_cfg["payload_kg"]
    rider_weight = emission_cfg["rider_weight"]

    ebike_params = create_ebike_params(exp_config)
    drone_params = create_drone_params(exp_config)

    pa = ProfitAnalysis(
        ebike_params=ebike_params,
        drone_params=drone_params,
        revenue_params=create_revenue_params(exp_config),
        ebike_cost_params=create_ebike_cost_params(exp_config),
        drone_cost_params=create_drone_cost_params(exp_config),
    )

    # Check for existing profit columns (caching pattern)
    has_profit_cols = (
        "revenue_cny" in od_df.columns
        and "ebike_cost_cny" in od_df.columns
        and "drone_cost_cny" in od_df.columns
        and "ebike_profit_cny" in od_df.columns
        and "drone_profit_cny" in od_df.columns
    )

    if has_profit_cols:
        print("Profit columns already exist, loading from CSV...")
    else:
        print("Calculating profit for each OD pair...")
        revenue_list = []
        ebike_cost_list = []
        drone_cost_list = []
        ebike_profit_list = []
        drone_profit_list = []

        for row in od_df.itertuples():
            distance_km = row.distance_km  # Straight-line for drone & revenue
            road_distance_km = row.road_distance_m / 1000.0  # Road distance for e-bike

            # Revenue uses straight-line distance
            rev = pa.calculate_revenue(distance_km)
            revenue_cny = rev["total_revenue_cny"]

            # E-bike cost uses road distance
            ebike_cost = pa.calculate_ebike_cost_per_order(
                road_distance_km, payload_kg, rider_weight
            )
            ebike_cost_cny = ebike_cost["total_cost_cny"]

            # Drone cost uses straight-line distance
            drone_cost = pa.calculate_drone_cost_per_order(distance_km, payload_kg)
            drone_cost_cny = drone_cost["total_cost_cny"]

            revenue_list.append(revenue_cny)
            ebike_cost_list.append(ebike_cost_cny)
            drone_cost_list.append(drone_cost_cny)
            ebike_profit_list.append(revenue_cny - ebike_cost_cny)
            drone_profit_list.append(revenue_cny - drone_cost_cny)

        od_df["revenue_cny"] = revenue_list
        od_df["ebike_cost_cny"] = ebike_cost_list
        od_df["drone_cost_cny"] = drone_cost_list
        od_df["ebike_profit_cny"] = ebike_profit_list
        od_df["drone_profit_cny"] = drone_profit_list

        od_df.to_csv(od_save_path, index=False)
        print(f"Profit data saved to {od_save_path}")

    # Summary statistics
    mean_ebike_profit = od_df["ebike_profit_cny"].mean()
    mean_drone_profit = od_df["drone_profit_cny"].mean()
    print(f"\n===== Profit Summary =====")
    print(f"Mean e-bike profit: {mean_ebike_profit:.4f} CNY/order")
    print(f"Mean drone profit:  {mean_drone_profit:.4f} CNY/order")
    print(f"Mean revenue:       {od_df['revenue_cny'].mean():.4f} CNY/order")

    region_info.set_stat("mean_ebike_profit_cny", float(mean_ebike_profit))
    region_info.set_stat("mean_drone_profit_cny", float(mean_drone_profit))

    # Visualization
    fig_save_dir = os.path.join(
        exp_config["figures_dir"],
        region_level,
        region_name_en.lower(),
    )
    for_latex = exp_config.get("for_latex", False)

    # Plot 1: Profit comparison
    save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_profit_comparison.png"
    )
    fig_path = plot_profit_comparison(od_df, save_path=save_path, for_latex=for_latex)
    if fig_path:
        print(f"Saved: {fig_path}")

    # Plot 2: Cost breakdown by distance
    save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_cost_breakdown.png"
    )
    fig_path = plot_cost_breakdown_by_distance(
        pa, exp_config, save_path=save_path, for_latex=for_latex
    )
    if fig_path:
        print(f"Saved: {fig_path}")

    # Plot 3: Profit vs distance
    save_path = os.path.join(
        fig_save_dir, f"{region_name_en.lower()}_profit_vs_distance.png"
    )
    fig_path = plot_profit_vs_distance(od_df, save_path=save_path, for_latex=for_latex)
    if fig_path:
        print(f"Saved: {fig_path}")

    return od_df
