import os

from .fetch_buildings import fetch_region_buildings


def visualize_region_buildings(region_config, exp_config):
    region_name_en = region_config["region_name_en"]
    region_level = region_config["region_level"]
    boundaries_dict = region_config.get("boundaries_dict", None)
    if not boundaries_dict:
        print(
            f"No boundaries_dict found for {region_name_en}. Skipping building fetch."
        )
        return

    bbox_4326 = (
        boundaries_dict["min_lng"],
        boundaries_dict["min_lat"],
        boundaries_dict["max_lng"],
        boundaries_dict["max_lat"],
    )

    region_save_dir = os.path.join(
        exp_config["buildings_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(region_save_dir):
        os.makedirs(region_save_dir)

    fetch_region_buildings(
        region_name=region_name_en,
        bbox_4326=bbox_4326,
        output_dir=region_save_dir,
    )
