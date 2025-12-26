import argparse

from buildings import visualize_region_buildings
from preprocessing import visualize_region_data
from utils import load_config

CITY_CONFIGS = {
    "shenzhen": {
        "region_name_cn": "深圳市",
        "region_name_en": "Shenzhen",
        "region_level": "city",
    },
    "beijing": {
        "region_name_cn": "北京市",
        "region_name_en": "Beijing",
        "region_level": "city",
    },
    "shanghai": {
        "region_name_cn": "上海市",
        "region_name_en": "Shanghai",
        "region_level": "city",
    },
    "guangzhou": {
        "region_name_cn": "广州市",
        "region_name_en": "Guangzhou",
        "region_level": "city",
    },
    "futian": {
        "region_name_cn": "福田区",
        "region_name_en": "Futian",
        "region_level": "district",
        "boundaries_dict": {
            "min_lat": 22.52,
            "max_lat": 22.56,
            "min_lng": 114.03,
            "max_lng": 114.08,
        },
    },
    "longhua": {
        "region_name_cn": "龙华区",
        "region_name_en": "Longhua",
        "region_level": "district",
    },
    "luohu": {
        "region_name_cn": "罗湖区",
        "region_name_en": "Luohu",
        "region_level": "district",
    },
}


def run_region_analysis(region_key, config):
    if region_key not in CITY_CONFIGS:
        print(f"Region '{region_key}' not found in configuration.")
        return

    print(f"Running analysis for {region_key}...")
    region_conf = CITY_CONFIGS[region_key]
    # visualize_region_data(
    #     region_config=region_conf,
    #     exp_config=config,
    # )
    visualize_region_buildings(
        region_config=region_conf,
        exp_config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="default", help="Environment name")
    args = parser.parse_args()
    config = load_config(
        "/Users/linjingxian/Code/drone-emission/experiment.toml", env=args.env
    )

    run_region_analysis("futian", config)
