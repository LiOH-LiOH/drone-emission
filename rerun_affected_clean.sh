#!/bin/bash

# Step 1: Identify cities affected by the multi-tile building fix and clean
# their stale data so the pipeline will re-fetch buildings.
#
# Run this BEFORE rerun_affected_run.sh.
# Outputs the list of affected city keys to stdout and to affected_cities.txt.
#
# Usage:
#   bash rerun_affected_clean.sh [env]
#   bash rerun_affected_clean.sh remote   # (default)

set -euo pipefail
cd "$(dirname "$0")"

ENV=${1:-remote}

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd src/drone_emission

python -u -c "
import json, os, sys, glob

sys.path.insert(0, '.')
from buildings.load_gba import get_gba_tile_infos
from utils import load_config
from utils.region_config import init_registry

config = load_config('../../experiment.toml', env='$ENV')
registry = init_registry('../../regions.json', data_dir=config.get('data_dir'))
data_dir = config['data_dir']

affected = []
skipped = []
for key, region in sorted(registry.list_by_level('city').items()):
    if not region.boundaries_dict:
        continue
    if 'total_ebike_lifecycle' not in region.stats:
        continue
    tiles = get_gba_tile_infos(region.boundaries_dict)
    if len(tiles) <= 1:
        skipped.append(key)
        continue

    affected.append(key)

    # Clean stale downstream data
    city_dir = os.path.join(data_dir, 'city', key)
    patterns = [
        '*_buildings.gpkg', '*_buildings.geojson',
        '*_buildings_preprocessed.gpkg', '*_buildings_preprocessed.geojson',
        '*_od_pairs.csv', '*_roadnet.graphml', '*_zone_stats.csv',
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(city_dir, pat)):
            os.remove(f)
            removed += 1

    # Clear stats (keep basic info, remove computed results)
    stats_path = os.path.join(city_dir, 'region_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            data = json.load(f)
        stats = data.get('stats', {})
        keep_keys = ('province', 'total_merchants', 'total_monthly_sales', 'mean_monthly_sales')
        for k in list(stats.keys()):
            if k not in keep_keys:
                del stats[k]
        data['stats'] = stats
        data['cache'] = {}
        with open(stats_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'  {key} ({len(tiles)} tiles, {removed} files cleaned)', file=sys.stderr)

print(f'', file=sys.stderr)
print(f'Affected: {len(affected)} cities (multi-tile bbox, data cleaned)', file=sys.stderr)
print(f'Unaffected: {len(skipped)} cities (single tile, skipped)', file=sys.stderr)

# Write list to file and stdout
with open('../../affected_cities.txt', 'w') as f:
    for key in affected:
        print(key)
        f.write(key + '\n')

print(f'City list written to affected_cities.txt', file=sys.stderr)
"
