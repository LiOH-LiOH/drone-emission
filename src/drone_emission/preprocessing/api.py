import os

import pandas as pd
from drone_emission.utils import filter_region_data


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
