import os

from utils import read_region_data

from .visualization import (
    plot_sales_distribution,
    plot_sales_heatmap,
    plot_store_scatter,
)


def visualize_region_data(region_config, exp_config):
    region_name_cn = region_config["region_name_cn"]
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict", None)

    data_path = os.path.join(exp_config["data_dir"], "combined_data.csv")

    region_data = read_region_data(
        data_path,
        region_name_cn,
        boundaries_dict=boundaries_dict,
        region_level=region_level,
    )
    region_save_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(region_save_dir):
        os.makedirs(region_save_dir)

    plot_sales_distribution(region_data, region_name_en, save_dir=region_save_dir)
    plot_sales_heatmap(
        region_data, region_name_en, region_name_cn, save_dir=region_save_dir
    )
    plot_store_scatter(
        region_data, region_name_en, region_name_cn, save_dir=region_save_dir
    )
    print(f"Visualization for {region_name_en} completed.\n")
