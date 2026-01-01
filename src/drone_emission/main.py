import argparse
from pathlib import Path

from buildings import load_buildings
from demand import generate_od_pairs
from preprocessing import prepare_sales
from roadnet import load_roadnet
from utils import load_config
from utils.region_config import get_registry, init_registry

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
REGIONS_CONFIG_PATH = PROJECT_ROOT / "regions.json"
EXP_CONFIG_PATH = PROJECT_ROOT / "experiment.toml"


def run_region_analysis(region_key: str, config: dict):
    """Run region analysis"""
    registry = get_registry()

    if region_key not in registry:
        print(f"Region '{region_key}' not found in configuration.")
        return

    print(f"Running analysis for {region_key}...")
    region = registry[region_key]

    # prepare_sales(
    #     region_info=region,
    #     exp_config=config,
    # )
    # load_buildings(
    #     region_info=region,
    #     exp_config=config,
    # )
    # load_roadnet(
    #     region_info=region,
    #     exp_config=config,
    # )
    generate_od_pairs(
        region_info=region,
        exp_config=config,
    )

    # Example: Add statistics and save
    # region.set_stat("building_count", 12345)
    # registry.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="default", help="Environment name")
    parser.add_argument(
        "--region", type=str, default="nanshan", help="Region to analyze"
    )
    args = parser.parse_args()

    # Initialize region configuration
    registry = init_registry(str(REGIONS_CONFIG_PATH))

    # Example: Dynamically add new region at runtime
    # registry.register(
    #     "baoan",
    #     region_name_cn="Baoan District",
    #     region_name_en="Baoan",
    #     region_level="district",
    #     parent_city="shenzhen",
    # )

    config = load_config(str(EXP_CONFIG_PATH), env=args.env)

    run_region_analysis(args.region, config)
    registry.save()
