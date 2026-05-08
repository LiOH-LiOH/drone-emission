#!/bin/bash

# Step 1: Identify cities affected by the multi-tile building fix.
#
# For each completed city whose bbox crosses tile boundaries:
#   - Re-fetch buildings from all tiles (with bbox filter)
#   - Compare new building count vs old
#   - If count increased: save new buildings, clean downstream data, add to rerun list
#   - If count unchanged: skip
#
# Run this BEFORE scripts/rerun_affected_run.sh.
# Outputs affected city keys to affected_cities.txt at repo root.
#
# Usage:
#   bash scripts/rerun_affected_clean.sh [env]
#   bash scripts/rerun_affected_clean.sh remote   # (default)

set -euo pipefail
cd "$(dirname "$0")/.."

ENV=${1:-remote}

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd src/drone_emission

python -u -c "
import json, os, sys, glob

sys.path.insert(0, '.')
from buildings.load_gba import get_gba_tile_infos, load_buildings_from_gba, _resolve_gba_source
from buildings.api import _resolve_gba_data_paths, enlarge_boundaries
from utils import load_config
from utils.region_config import init_registry

config = load_config('../../experiment.toml', env='$ENV')
registry = init_registry('../../regions.json', data_dir=config.get('data_dir'))
data_dir = config['data_dir']
delivery_radius_km = config.get('delivery_radius_km', 5.0)

affected = []
unchanged = []
skipped_single = []
skipped_incomplete = []

candidates = []
for key, region in sorted(registry.list_by_level('city').items()):
    if not region.boundaries_dict:
        continue
    if 'total_ebike_lifecycle' not in region.stats:
        skipped_incomplete.append(key)
        continue
    tiles = get_gba_tile_infos(region.boundaries_dict)
    if len(tiles) <= 1:
        skipped_single.append(key)
        continue
    candidates.append((key, region, tiles))

print(f'Checking {len(candidates)} multi-tile cities...', file=sys.stderr)
print(f'(Skipping {len(skipped_single)} single-tile, {len(skipped_incomplete)} incomplete)', file=sys.stderr)
print(file=sys.stderr)

for i, (key, region, tiles) in enumerate(candidates, 1):
    old_count = region.stats.get('building_count', 0)

    # Resolve tile paths (already downloaded/converted)
    try:
        gba_paths = _resolve_gba_data_paths(region.boundaries_dict, config)
    except FileNotFoundError:
        print(f'  [{i}/{len(candidates)}] {key}: tiles not available, skipping', file=sys.stderr)
        continue

    # Compute enlarged bbox (same as pipeline)
    large_bounds = region.stats.get('enlarged_boundaries')
    if not large_bounds:
        large_bounds = enlarge_boundaries(region.boundaries_dict, delivery_radius_km)
    bbox_4326 = (large_bounds['min_lng'], large_bounds['min_lat'],
                 large_bounds['max_lng'], large_bounds['max_lat'])

    # Save path for new buildings
    city_dir = os.path.join(data_dir, 'city', key)
    gpkg_file = os.path.join(city_dir, f'{key}_buildings.gpkg')

    # Fetch buildings from all tiles
    result_path = load_buildings_from_gba(
        gba_data_paths=gba_paths,
        bbox_4326=bbox_4326,
        save_path=gpkg_file,
    )

    # Read new count from saved file
    import geopandas as gpd
    new_gdf = gpd.read_file(result_path)
    new_count = len(new_gdf)

    if new_count > old_count:
        affected.append(key)
        delta = new_count - old_count
        pct = (delta / old_count * 100) if old_count > 0 else float('inf')
        print(f'  [{i}/{len(candidates)}] {key}: {old_count} -> {new_count} (+{delta}, +{pct:.0f}%) ** AFFECTED **', file=sys.stderr)

        # Clean downstream data (buildings already saved above, roadnet unchanged)
        for pat in ['*_buildings_preprocessed.gpkg', '*_buildings_preprocessed.geojson',
                     '*_od_pairs.csv', '*_zone_stats.csv']:
            for f in glob.glob(os.path.join(city_dir, pat)):
                os.remove(f)

        # Clear stats (keep basic info, remove computed results)
        stats_path = os.path.join(city_dir, 'region_stats.json')
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                sdata = json.load(f)
            stats = sdata.get('stats', {})
            keep_keys = ('province', 'total_merchants', 'total_monthly_sales', 'mean_monthly_sales')
            for k in list(stats.keys()):
                if k not in keep_keys:
                    del stats[k]
            sdata['stats'] = stats
            sdata['cache'] = {}
            with open(stats_path, 'w') as f:
                json.dump(sdata, f, ensure_ascii=False, indent=2)
    else:
        unchanged.append(key)
        print(f'  [{i}/{len(candidates)}] {key}: {old_count} -> {new_count} (unchanged)', file=sys.stderr)

print(file=sys.stderr)
print(f'=== Summary ===', file=sys.stderr)
print(f'  Affected (buildings increased): {len(affected)}', file=sys.stderr)
print(f'  Unchanged (same count):         {len(unchanged)}', file=sys.stderr)
print(f'  Single-tile (skipped):          {len(skipped_single)}', file=sys.stderr)
print(f'  Incomplete (skipped):           {len(skipped_incomplete)}', file=sys.stderr)

# Write list to file
with open('../../affected_cities.txt', 'w') as f:
    for key in affected:
        f.write(key + '\n')

print(f'', file=sys.stderr)
print(f'Wrote {len(affected)} cities to affected_cities.txt', file=sys.stderr)
"
