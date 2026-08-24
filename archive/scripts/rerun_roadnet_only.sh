#!/bin/bash

# Re-run only the roadnet -> emission -> equity stages for cities listed in
# affected_cities.txt. Used after rerun_affected_run.sh succeeded for
# buildings + demand but failed at the OSMnx fetch (Overpass timeout).
#
# Usage:
#   bash scripts/rerun_roadnet_only.sh [parallel_jobs] [env]
#   bash scripts/rerun_roadnet_only.sh           # 1 sequential, remote (default)

MEMORY_MAX="128G"
SESSION_NAME="rerun-roadnet-only"
CITY_LIST="affected_cities.txt"

# ---- If not inside tmux, wrap ourselves in a tmux session ----
if [ -z "$TMUX" ] && [ -z "$INSIDE_RERUN" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi
    echo "Starting '${SESSION_NAME}' (mem=${MEMORY_MAX})"
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_RERUN=1 bash $(cd "$(dirname "$0")" && pwd)/rerun_roadnet_only.sh $*; echo ''; echo 'Done.'"
    exec tmux attach -t "${SESSION_NAME}"
fi

cd "$(dirname "$0")/.."
PARALLEL=${1:-1}
ENV=${2:-remote}

if [ ! -f "${CITY_LIST}" ]; then
    echo "Error: ${CITY_LIST} not found."
    exit 1
fi

LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/roadnet_rerun_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

if [ -d ".venv" ]; then source .venv/bin/activate; fi
cd src/drone_emission

cat "../../${CITY_LIST}" | xargs -P "$PARALLEL" -I {} \
    bash -c 'systemd-run --user --scope -p MemoryMax='"${MEMORY_MAX}"' python -u -c "
import sys, os
sys.path.insert(0, \".\")
from utils import load_config
from utils.region_config import init_registry
from roadnet import load_roadnet
from emission import cal_emission
from equity import analyze_equity

region_key = sys.argv[1]
config = load_config(\"../../experiment.toml\", env=\"'"$ENV"'\")
registry = init_registry(\"../../regions.json\", data_dir=config.get(\"data_dir\"))
region = registry[region_key]

print(f\"=== {region_key}: roadnet stage ===\", flush=True)
load_roadnet(region_info=region, exp_config=config)
print(f\"=== {region_key}: emission stage ===\", flush=True)
cal_emission(region_info=region, exp_config=config)
print(f\"=== {region_key}: equity stage ===\", flush=True)
analyze_equity(region_info=region, exp_config=config)
print(f\"=== {region_key}: DONE ===\", flush=True)
" "$1" || echo "FAILED: $1"' _ {}

echo ""
echo "All done."
