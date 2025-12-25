import argparse

from preprocessing import preprocessing
from utils import load_config

CITY_CONFIGS = {
    "shenzhen": {
        "city_name_cn": "深圳市",
        "city_name_en": "Shenzhen",
        "region_level": "city",
    },
    "beijing": {
        "city_name_cn": "北京市",
        "city_name_en": "Beijing",
        "region_level": "city",
    },
    "shanghai": {
        "city_name_cn": "上海市",
        "city_name_en": "Shanghai",
        "region_level": "city",
    },
    "guangzhou": {
        "city_name_cn": "广州市",
        "city_name_en": "Guangzhou",
        "region_level": "city",
    },
    "futian": {
        "city_name_cn": "福田区",
        "city_name_en": "Futian",
        "region_level": "district",
    },
}


def run_region_analysis(region_key, config):
    if region_key not in CITY_CONFIGS:
        print(f"Region '{region_key}' not found in configuration.")
        return

    print(f"Running analysis for {region_key}...")
    region_conf = CITY_CONFIGS[region_key]
    preprocessing(
        region_name_cn=region_conf["city_name_cn"],
        region_name_en=region_conf["city_name_en"],
        region_level=region_conf["region_level"],
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="default", help="Environment name")
    args = parser.parse_args()
    config = load_config(
        "/Users/linjingxian/Code/drone-emission/experiment.toml", env=args.env
    )

    run_region_analysis("futian", config)
