#!/bin/bash

# Post-processing: consolidate per-region stats and run cross-region analysis.
# Results are saved to a timestamped directory for reproducibility.
# Run this after entrypoint.sh completes.
#
# Usage: ./postprocess.sh [env]
# Examples:
#   ./postprocess.sh          # remote env (default)
#   ./postprocess.sh local    # local env

set -euo pipefail
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found" >&2
    exit 1
fi

ENV=${1:-remote}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Consolidating per-region stats (env=$ENV) ==="
cd src/drone_emission
python -u -m tools.consolidate_stats --env "$ENV"

echo ""
echo "=== Enriching region stats ==="
python -u -m tools.enrich_stats --env "$ENV"

echo ""
echo "=== Running cross-region analysis ==="
python -u -m analysis.cross_regions --env "$ENV"

echo ""
echo "=== Generating analysis report ==="
python -u -m analysis.report --env "$ENV"

# ---- Save timestamped snapshot ----
cd ../..

# Resolve data_dir from config
DATA_DIR=$(cd src/drone_emission && python -c "
from utils import load_config
config = load_config('../../experiment.toml', env='$ENV')
print(config['data_dir'])
")

SNAPSHOT_DIR="${DATA_DIR}/postprocess_${TIMESTAMP}"
mkdir -p "${SNAPSHOT_DIR}"

# Copy data outputs
for f in all_region_stats.json all_region_stats.csv all_region_stats_enriched.csv; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        cp "${DATA_DIR}/${f}" "${SNAPSHOT_DIR}/${f}"
    fi
done

# Copy figures
FIGURES_ANALYSIS="figures/analysis"
if [ -d "${FIGURES_ANALYSIS}" ]; then
    cp -r "${FIGURES_ANALYSIS}" "${SNAPSHOT_DIR}/figures_analysis"
fi

# Copy report
REPORT="docs/shared/analysis_report.md"
if [ -f "${REPORT}" ]; then
    cp "${REPORT}" "${SNAPSHOT_DIR}/analysis_report.md"
fi

echo ""
echo "=== Snapshot saved to ${SNAPSHOT_DIR} ==="
ls -lh "${SNAPSHOT_DIR}/"
echo ""
echo "Done!"
