#!/bin/bash

# Step 1: Identify cities whose loaded roadnet bbox actually excludes some of
# their merchants. A city is flagged only when at least one merchant in
# <region>_sales.csv (after GCJ02 -> WGS84 conversion) falls outside the
# existing graphml's bbox. Cities whose stores all happen to cluster within
# the smaller cached bbox are NOT flagged — refetching them would not change
# any OD pair, building, or emission.
#
# For each flagged city:
#   - Delete the stale roadnet, buildings, and all derived per-region data
#   - Clear computed stats in region_stats.json (preserve raw preprocessing info)
#   - Add the city key to affected_cities.txt
#
# Run this BEFORE scripts/rerun_affected_run.sh.
#
# Usage:
#   bash scripts/rerun_affected_bbox_clean.sh [env] [min_outside_count]
#   bash scripts/rerun_affected_bbox_clean.sh remote        # min_outside_count = 1
#   bash scripts/rerun_affected_bbox_clean.sh remote 100    # skip cities with < 100 outside stores

set -euo pipefail
cd "$(dirname "$0")/.."

ENV=${1:-remote}
MIN_OUTSIDE=${2:-1}

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd src/drone_emission

MIN_OUTSIDE=$MIN_OUTSIDE python -u -c "
import json, os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, '.')
from utils import load_config
from utils.region_config import init_registry
from utils.boundaries import bbox_from_graphml
from utils.coordinates import gcj02_to_wgs84

config = load_config('../../experiment.toml', env='$ENV')
registry = init_registry('../../regions.json', data_dir=config.get('data_dir'))
data_dir = config['data_dir']
min_outside = int(os.environ['MIN_OUTSIDE'])

affected = []
clean = []
no_graphml = []
no_sales = []
no_boundaries = []
errors = []

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
print(f'Checking {len(regions)} cities for merchants outside current roadnet bbox '
      f'(min_outside={min_outside})...', file=sys.stderr)

for i, (key, region) in enumerate(regions, 1):
    if not region.boundaries_dict:
        no_boundaries.append(key)
        continue
    city_dir = os.path.join(data_dir, 'city', key)
    graphml = os.path.join(city_dir, f'{key}_roadnet.graphml')
    sales_csv = os.path.join(city_dir, f'{key}_sales.csv')
    if not os.path.exists(graphml):
        no_graphml.append(key)
        continue
    if not os.path.exists(sales_csv):
        no_sales.append(key)
        continue

    try:
        lat_min, lat_max, lng_min, lng_max = bbox_from_graphml(graphml)
        df = pd.read_csv(sales_csv)
        gcj_lat = df['lat'].to_numpy(dtype=float)
        gcj_lng = df['lng'].to_numpy(dtype=float)
        wgs = np.array([gcj02_to_wgs84(float(lng), float(lat))
                        for lat, lng in zip(gcj_lat, gcj_lng)])
        wgs_lng = wgs[:, 0]; wgs_lat = wgs[:, 1]
        out_mask = ((wgs_lat < lat_min) | (wgs_lat > lat_max) |
                    (wgs_lng < lng_min) | (wgs_lng > lng_max))
        n_out = int(out_mask.sum())
        n_total = len(df)
    except Exception as e:
        errors.append((key, str(e)))
        print(f'  [{i:>3}/{len(regions)}] {key}: error ({e})', file=sys.stderr)
        continue

    if n_out < min_outside:
        clean.append(key)
        continue

    pct = 100.0 * n_out / max(n_total, 1)
    affected.append(key)
    print(f'  [{i:>3}/{len(regions)}] {key}: {n_out:,} / {n_total:,} merchants outside '
          f'({pct:.1f}%) - cleaning...', file=sys.stderr)

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
print(f'  Affected (>= {min_outside} merchants outside bbox): {len(affected)}', file=sys.stderr)
print(f'  Clean (all merchants inside):                       {len(clean)}', file=sys.stderr)
print(f'  No graphml on disk (skipped):                       {len(no_graphml)}', file=sys.stderr)
print(f'  No sales CSV (skipped):                             {len(no_sales)}', file=sys.stderr)
print(f'  No boundaries_dict (skipped):                       {len(no_boundaries)}', file=sys.stderr)
if errors:
    print(f'  Errors:                                             {len(errors)}', file=sys.stderr)

with open('../../affected_cities.txt', 'w') as f:
    for key in affected:
        f.write(key + '\n')

print(f'', file=sys.stderr)
print(f'Wrote {len(affected)} cities to affected_cities.txt', file=sys.stderr)
"
