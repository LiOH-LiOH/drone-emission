import os

from utils import read_city_data

from .visualization import (
    plot_sales_distribution,
    plot_sales_heatmap,
    plot_store_scatter,
)


def preprocessing(city_name_cn, city_name_en, config):
    data_path = os.path.join(config["data_dir"], "combined_data.csv")

    city_data = read_city_data(data_path, city_name_cn)
    city_save_dir = os.path.join(config["output_dir"], city_name_en.lower())
    if not os.path.exists(city_save_dir):
        os.makedirs(city_save_dir)

    plot_sales_distribution(city_data, city_name_en, save_dir=city_save_dir)
    plot_sales_heatmap(city_data, city_name_en, save_dir=city_save_dir)
    plot_store_scatter(city_data, city_name_en, save_dir=city_save_dir)
    print(f"Preprocessing for {city_name_en} completed.\n")
