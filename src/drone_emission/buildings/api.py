import os

from visualization.buildings import visualize_buildings

from .fetch_buildings import fetch_region_buildings


def visualize_region_buildings(region_config, exp_config):
    region_name_en = region_config["region_name_en"]
    region_name_cn = region_config["region_name_cn"]
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

    geojson_file = os.path.join(
        region_save_dir, f"{region_name_en.lower()}_buildings.geojson"
    )
    # fetch_region_buildings(
    #     region_name=region_name_en,
    #     bbox_4326=bbox_4326,
    #     save_path=geojson_file,
    # )

    # Visualization
    fig_dir = os.path.join(
        exp_config["figures_dir"], region_level, region_name_en.lower()
    )
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    save_path = os.path.join(fig_dir, f"{region_name_en.lower()}_buildings_vis.png")

    visualize_buildings(
        geojson_file,
        city_name=region_name_en,
        city_name_cn=region_name_cn,
        boundaries_dict=boundaries_dict,
        save_path=save_path,
    )
