#!/bin/bash

# Post-processing: consolidate per-region stats and run cross-region analysis.
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

echo ""
echo "Done! Figures saved to figures/analysis/"
