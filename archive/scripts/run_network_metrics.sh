#!/bin/bash

# Compute OSMnx network metrics for all regions, then re-consolidate and re-run analysis.
# Lightweight — only reads existing GraphML files, no full pipeline re-run.
#
# Usage: ./run_network_metrics.sh [parallel_jobs] [env]
# Examples:
#   ./run_network_metrics.sh          # 4 parallel, remote env (default)
#   ./run_network_metrics.sh 8        # 8 parallel, remote env
#   ./run_network_metrics.sh 1        # sequential, remote env
#   ./run_network_metrics.sh 4 local  # 4 parallel, local env

SESSION_NAME="network-metrics"

# ---- If not inside tmux, wrap ourselves in a tmux session ----
if [ -z "$TMUX" ] && [ -z "$INSIDE_ENTRYPOINT" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi

    echo "Starting tmux session '${SESSION_NAME}'..."
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_ENTRYPOINT=1 bash $(cd "$(dirname "$0")" && pwd)/run_network_metrics.sh $*; echo ''; echo 'Done. Press Enter to close.'; read"
    exec tmux attach -t "${SESSION_NAME}"
fi

# ---- Actual work (runs inside tmux) ----
cd "$(dirname "$0")"

# Setup logging
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/network_metrics_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found, using system Python"
fi

cd src/drone_emission

PARALLEL=${1:-4}
ENV=${2:-remote}
echo "Environment: $ENV, Parallel jobs: $PARALLEL"
echo "Started at: $(date)"
echo ""

# Step 1: Compute OSMnx network metrics from existing GraphML files
echo "========================================"
echo "Step 1/3: Computing network metrics..."
echo "========================================"

if [ "$PARALLEL" -gt 1 ]; then
    echo "Running $PARALLEL cities in parallel..."
    python -c "
import json, os, sys
sys.path.insert(0, '.')
from utils import load_config
from utils.region_config import RegionRegistry
config = load_config('../../experiment.toml', env='$ENV')
registry = RegionRegistry('../../regions.json', data_dir=config['data_dir'])
for key in sorted(registry.list_by_level('city')):
    region = registry[key]
    if region.stats.get('orientation_entropy') is None:
        print(key)
" | xargs -P "$PARALLEL" -I {} \
    python -u -m tools.compute_network_metrics --env "$ENV" --region {}

    echo ""
    echo "Parallel computation complete."
else
    echo "Running sequentially..."
    python -u -m tools.compute_network_metrics --env "$ENV"
fi

echo ""

# Step 2: Re-consolidate per-region stats into summary CSV/JSON
echo "========================================"
echo "Step 2/3: Consolidating stats..."
echo "========================================"
python -u -m tools.consolidate_stats --env "$ENV"

echo ""

# Step 3: Re-run cross-region analysis (correlation + regression)
echo "========================================"
echo "Step 3/3: Running cross-region analysis..."
echo "========================================"
python -u -m analysis.cross_regions --env "$ENV"

echo ""
echo "Finished at: $(date)"
echo "Log: ${LOG_FILE}"
