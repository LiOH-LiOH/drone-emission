import os

import pandas as pd
from utils import filter_region_data
from visualization.sales import (
    plot_sales_distribution,
    plot_sales_heatmap,
    plot_store_scatter,
)


def prepare_sales(region_info, exp_config):
    region_config = region_info.to_dict()

    region_name_cn = region_config["region_name_cn"]
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict", None)

    data_path = exp_config["sales_path"]

    save_dir = os.path.join(
        exp_config["data_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{region_name_en.lower()}_sales.csv")
    filter_region_data(
        data_path,
        region_name_cn,
        boundaries_dict=boundaries_dict,
        region_level=region_level,
        save_path=save_path,
    )

    region_data = pd.read_csv(save_path)

    # Save merchant/sales statistics
    total_merchants = len(region_data)
    total_monthly_sales = region_data["month_sales"].sum()
    mean_monthly_sales = region_data["month_sales"].mean()
    region_info.set_stat("total_merchants", int(total_merchants))
    region_info.set_stat("total_monthly_sales", float(total_monthly_sales))
    region_info.set_stat("mean_monthly_sales", round(float(mean_monthly_sales), 2))

    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    for_latex = exp_config.get("for_latex", False)
    plot_title = exp_config.get("plot_title", False)

    plot_sales_distribution(
        region_data,
        region_name_en,
        log_scale=False,
        save_path=os.path.join(
            fig_dir, f"{region_name_en.lower()}_month_sales_distribution.png"
        ),
        for_latex=for_latex,
        plot_title=plot_title,
    )
    plot_sales_distribution(
        region_data,
        region_name_en,
        log_scale=True,
        save_path=os.path.join(
            fig_dir, f"{region_name_en.lower()}_month_sales_distribution_log.png"
        ),
        for_latex=for_latex,
        plot_title=plot_title,
    )
    plot_sales_heatmap(
        region_data,
        region_name_en,
        region_name_cn,
        boundaries_dict=boundaries_dict,
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_heatmap.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )
    plot_store_scatter(
        region_data,
        region_name_en,
        region_name_cn,
        boundaries_dict=boundaries_dict,
        save_path=os.path.join(fig_dir, f"{region_name_en.lower()}_scatter.png"),
        for_latex=for_latex,
        plot_title=plot_title,
    )
    print(f"Visualization for {region_name_en} completed.\n")
