#!/bin/bash

# Step 1: Identify cities whose loaded road network was fetched with a smaller
# bbox than the current regions.json + delivery_radius_km. For each such city:
#   - Delete the stale roadnet, buildings, and all derived per-region data
#   - Clear computed stats in region_stats.json (preserve raw preprocessing info)
#   - Add the city key to affected_cities.txt
#
# Run this BEFORE scripts/rerun_affected_run.sh.
#
# Usage:
#   bash scripts/rerun_affected_bbox_clean.sh [env]
#   bash scripts/rerun_affected_bbox_clean.sh remote   # (default)

set -euo pipefail
cd "$(dirname "$0")/.."

ENV=${1:-remote}

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd src/drone_emission

python -u -c "
import json, os, sys, glob

sys.path.insert(0, '.')
from utils import load_config
from utils.region_config import init_registry
from utils.boundaries import enlarge_boundaries, graphml_covers_bbox

config = load_config('../../experiment.toml', env='$ENV')
registry = init_registry('../../regions.json', data_dir=config.get('data_dir'))
data_dir = config['data_dir']
delivery_radius_km = config.get('delivery_radius_km', 5.0)

affected = []
ok = []
no_graphml = []
no_boundaries = []

# Files to remove for each affected city (all are derived from buildings/roadnet bbox)
DERIVED_PATTERNS = [
    '*_roadnet.graphml',
    '*_buildings.geojson', '*_buildings.gpkg',
    '*_buildings_preprocessed.geojson', '*_buildings_preprocessed.gpkg',
    '*_od_pairs.csv',
    '*_zone_stats.csv',
    '*_oat.csv', '*_sobol.csv', '*_shapley.csv',
    '*_named.csv',
]

# Stats keys to preserve (set by preprocessing, not derived from bbox-clipped data)
KEEP_STAT_KEYS = ('province', 'total_merchants', 'total_monthly_sales', 'mean_monthly_sales')

regions = sorted(registry.list_by_level('city').items())
print(f'Checking {len(regions)} cities for stale roadnet bbox...', file=sys.stderr)

for i, (key, region) in enumerate(regions, 1):
    if not region.boundaries_dict:
        no_boundaries.append(key)
        continue
    city_dir = os.path.join(data_dir, 'city', key)
    graphml = os.path.join(city_dir, f'{key}_roadnet.graphml')
    if not os.path.exists(graphml):
        no_graphml.append(key)
        continue

    expected = enlarge_boundaries(region.boundaries_dict, delivery_radius_km)
    try:
        covers = graphml_covers_bbox(graphml, expected)
    except Exception as e:
        print(f'  [{i:>3}/{len(regions)}] {key}: bbox check failed ({e}); marking affected', file=sys.stderr)
        covers = False

    if covers:
        ok.append(key)
        continue

    affected.append(key)
    print(f'  [{i:>3}/{len(regions)}] {key}: stale roadnet, cleaning...', file=sys.stderr)

    # Delete derived files
    for pat in DERIVED_PATTERNS:
        for f in glob.glob(os.path.join(city_dir, pat)):
            os.remove(f)

    # Clear computed stats in region_stats.json (keep raw preprocessing info)
    stats_path = os.path.join(city_dir, 'region_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            sdata = json.load(f)
        stats = sdata.get('stats', {})
        for k in list(stats.keys()):
            if k not in KEEP_STAT_KEYS:
                del stats[k]
        sdata['stats'] = stats
        sdata['cache'] = {}
        with open(stats_path, 'w') as f:
            json.dump(sdata, f, ensure_ascii=False, indent=2)

print(file=sys.stderr)
print(f'=== Summary ===', file=sys.stderr)
print(f'  Affected (stale roadnet):     {len(affected)}', file=sys.stderr)
print(f'  OK (graphml covers bbox):     {len(ok)}', file=sys.stderr)
print(f'  No graphml on disk (skipped): {len(no_graphml)}', file=sys.stderr)
print(f'  No boundaries_dict (skipped): {len(no_boundaries)}', file=sys.stderr)

with open('../../affected_cities.txt', 'w') as f:
    for key in affected:
        f.write(key + '\n')

print(f'', file=sys.stderr)
print(f'Wrote {len(affected)} cities to affected_cities.txt', file=sys.stderr)
"
