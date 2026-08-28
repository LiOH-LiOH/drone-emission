import traceback
import os
from pathlib import Path

from drone_emission.buildings import load_buildings
from drone_emission.demand import generate_od_pairs
from drone_emission.emission import cal_emission
from drone_emission.preprocessing import prepare_sales
from drone_emission.roadnet import load_roadnet
from drone_emission.utils.region_config import get_registry

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
REGIONS_CONFIG_PATH = PROJECT_ROOT / "regions.json"
EXP_CONFIG_PATH = PROJECT_ROOT / "experiment.toml"


def _record_stage(stage: str) -> None:
    path = os.environ.get("DRONE_EMISSION_STAGE_FILE")
    if path:
        from drone_emission.utils.atomic import atomic_open

        with atomic_open(path, "w", encoding="utf-8") as handle:
            handle.write(stage + "\n")


def run_region_analysis(region_key: str, config: dict):
    """Run region analysis"""
    registry = get_registry()

    if region_key not in registry:
        raise KeyError(f"Region '{region_key}' not found in configuration")

    print(f"Running analysis for {region_key}...")
    region = registry[region_key]

    _record_stage("sales")
    prepare_sales(
        region_info=region,
        exp_config=config,
    )
    _record_stage("buildings")
    load_buildings(
        region_info=region,
        exp_config=config,
    )
    _record_stage("demand")
    generate_od_pairs(
        region_info=region,
        exp_config=config,
    )
    _record_stage("roadnet")
    load_roadnet(
        region_info=region,
        exp_config=config,
    )
    _record_stage("emission")
    cal_emission(
        region_info=region,
        exp_config=config,
    )

    # Example: Add statistics and save
    # region.set_stat("building_count", 12345)
    # registry.save()


def run_region_from_roadnet(region_key: str, config: dict):
    """Resume one city from its existing OD pairs through final emissions."""
    registry = get_registry()
    if region_key not in registry:
        raise KeyError(f"Region '{region_key}' not found in configuration")

    print(f"Resuming analysis from roadnet for {region_key}...")
    region = registry[region_key]
    region_config = region.to_dict()
    name = region_config["region_name_en"].lower()
    od_path = (
        Path(config["data_dir"])
        / region_config["region_level"]
        / name
        / f"{name}_od_pairs.csv"
    )
    from drone_emission.roadnet.api import validate_od_checkpoint

    validate_od_checkpoint(od_path)
    _record_stage("roadnet")
    load_roadnet(region_info=region, exp_config=config)
    _record_stage("emission")
    cal_emission(region_info=region, exp_config=config)


def has_city_results(region_config, config: dict) -> bool:
    """Check the completion marker and required reusable city artifacts."""
    if "total_ebike_lifecycle" not in region_config.stats:
        return False
    name = region_config.region_name_en.lower()
    root = Path(config["data_dir"]) / region_config.region_level / name
    buildings = (root / f"{name}_buildings.gpkg").exists() or (
        root / f"{name}_buildings.geojson"
    ).exists()
    return buildings and all(
        (root / filename).exists()
        for filename in (
            f"{name}_sales.csv",
            f"{name}_od_pairs.csv",
            f"{name}_roadnet.graphml",
            "region_stats.json",
        )
    )


def run_all(config: dict, level: str = "city", skip_completed: bool = False):
    """Run analysis for all regions of the given level."""
    registry = get_registry()
    regions = registry.list_by_level(level)

    if not regions:
        print(f"No regions found with level '{level}'.")
        return

    total = len(regions)
    succeeded = 0
    skipped = 0
    failed = []

    print(f"Found {total} {level}-level regions to process.")
    if skip_completed:
        print("Skip-completed mode: regions with emission results will be skipped.")

    for i, (region_key, region_config) in enumerate(sorted(regions.items()), 1):
        print(f"\n{'='*60}")
        print(f"Processing {level} {i}/{total}: {region_key} ({region_config.region_name_cn})")
        print(f"{'='*60}")

        if skip_completed and has_city_results(region_config, config):
            print(f"Skipping {region_key} (already has emission results)")
            skipped += 1
            continue

        try:
            run_region_analysis(region_key, config)
            registry.save_region(region_key)
            succeeded += 1
        except Exception as e:
            print(f"\nERROR processing {region_key}: {e}")
            traceback.print_exc()
            failed.append(region_key)
            # Save progress even on failure
            registry.save_region(region_key)

    print(f"\n{'='*60}")
    print(f"Batch run complete:")
    print(f"  Total:     {total}")
    print(f"  Succeeded: {succeeded}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"  Failed regions: {', '.join(failed)}")
        raise RuntimeError(
            f"{len(failed)} city experiment(s) failed: {', '.join(failed)}"
        )
