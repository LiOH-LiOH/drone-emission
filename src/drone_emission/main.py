import argparse

from preprocessing import preprocessing
from utils import load_config

CITY_CONFIGS = {
    "shenzhen": {
        "city_name_cn": "深圳市",
        "city_name_en": "Shenzhen",
        "boundaries_dict": {
            "min_lat": 22.43,
            "max_lat": 22.85,
            "min_lng": 113.75,
            "max_lng": 114.65,
        },
    },
    "beijing": {
        "city_name_cn": "北京市",
        "city_name_en": "Beijing",
        "boundaries_dict": {
            "min_lat": 39.65,
            "max_lat": 40.40,
            "min_lng": 115.95,
            "max_lng": 116.90,
        },
    },
    "shanghai": {
        "city_name_cn": "上海市",
        "city_name_en": "Shanghai",
        "boundaries_dict": {
            "min_lat": 30.70,
            "max_lat": 31.55,
            "min_lng": 121.05,
            "max_lng": 121.85,
        },
    },
    "guangzhou": {
        "city_name_cn": "广州市",
        "city_name_en": "Guangzhou",
        "boundaries_dict": {
            "min_lat": 22.75,
            "max_lat": 23.60,
            "min_lng": 113.10,
            "max_lng": 113.85,
        },
    },
}


def run_city_analysis(city_key, config):
    if city_key not in CITY_CONFIGS:
        print(f"City '{city_key}' not found in configuration.")
        return

    print(f"Running analysis for {city_key}...")
    city_conf = CITY_CONFIGS[city_key]
    preprocessing(
        city_name_cn=city_conf["city_name_cn"],
        city_name_en=city_conf["city_name_en"],
        config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="default", help="Environment name")
    args = parser.parse_args()
    config = load_config(
        "/Users/linjingxian/Code/drone-emission/experiment.toml", env=args.env
    )

    run_city_analysis("guangzhou", config)
