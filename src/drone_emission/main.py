import traceback
from pathlib import Path

from buildings import load_buildings
from demand import generate_od_pairs
from emission import cal_emission
from preprocessing import prepare_sales
from roadnet import load_roadnet
from utils.region_config import get_registry

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

    prepare_sales(
        region_info=region,
        exp_config=config,
    )
    load_buildings(
        region_info=region,
        exp_config=config,
    )
    generate_od_pairs(
        region_info=region,
        exp_config=config,
    )
    load_roadnet(
        region_info=region,
        exp_config=config,
    )
    cal_emission(
        region_info=region,
        exp_config=config,
    )

    # Example: Add statistics and save
    # region.set_stat("building_count", 12345)
    # registry.save()


def has_emission_results(region_config) -> bool:
    """Check if a region already has emission results in stats."""
    return "total_ebike_lifecycle" in region_config.stats


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

        if skip_completed and has_emission_results(region_config):
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
