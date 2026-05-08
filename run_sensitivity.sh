#!/bin/bash

# Standalone sensitivity-stage runner.
# Mirrors entrypoint.sh structure (tmux + memory cap + logging) but only runs
# the sensitivity stage, which depends on cal_emission's per-region OD CSV
# already being on disk.
#
# Usage:
#   ./run_sensitivity.sh                       # 4 parallel jobs, remote env
#   ./run_sensitivity.sh 8                     # 8 parallel jobs, remote env
#   ./run_sensitivity.sh 1 local               # sequential, local env
#   ./run_sensitivity.sh 4 remote --include-districts
#
# Logs go to ./logs/sensitivity_TIMESTAMP.log

MEMORY_MAX="64G"
SESSION_NAME="drone-sensitivity"

# ---- If not inside tmux, wrap ourselves in a tmux session ----
if [ -z "$TMUX" ] && [ -z "$INSIDE_SENSITIVITY" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi
    echo "Starting '${SESSION_NAME}' (mem=${MEMORY_MAX})"
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_SENSITIVITY=1 bash $(cd "$(dirname "$0")" && pwd)/run_sensitivity.sh $*; echo ''; echo 'Done. Press Enter to close.'; read"
    exec tmux attach -t "${SESSION_NAME}"
fi

# ---- Actual run (inside tmux) ----
cd "$(dirname "$0")"
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/sensitivity_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found, using system Python"
fi

cd src/drone_emission

# ---- Parse args (ignore --include-districts so it doesn't land in PARALLEL/ENV) ----
EXTRA_FLAGS=()
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --include-districts|--level=*|--region=*)
            EXTRA_FLAGS+=("$arg")
            ;;
        --level|--region)
            EXTRA_FLAGS+=("$arg")
            ;;
        *)
            POSITIONAL+=("$arg")
            ;;
    esac
done
PARALLEL=${POSITIONAL[0]:-4}
ENV=${POSITIONAL[1]:-remote}

echo "Environment: $ENV (mem=${MEMORY_MAX}, parallel=${PARALLEL})"

# ---- Build the region list ----
# Cities from regions.json + optional districts from data/district/ if --include-districts is set
INCLUDE_DISTRICTS=0
for f in "${EXTRA_FLAGS[@]}"; do
    [ "$f" = "--include-districts" ] && INCLUDE_DISTRICTS=1
done

REGION_LIST_TMP=$(mktemp)
trap 'rm -f "$REGION_LIST_TMP"' EXIT

python -c "
import json, os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))
regions = json.load(open(os.path.join(project_root, 'regions.json')))
for k in regions:
    print(k)
" > "$REGION_LIST_TMP"

if [ "$INCLUDE_DISTRICTS" = "1" ]; then
    # Read data_dir from experiment.toml for the chosen env, then list its district subdirs
    DATA_DIR=$(python -c "
from utils import load_config
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath('.')))
print(load_config(os.path.join(project_root, 'experiment.toml'), env='$ENV')['data_dir'])
")
    if [ -d "$DATA_DIR/district" ]; then
        ls -1 "$DATA_DIR/district" >> "$REGION_LIST_TMP"
    fi
fi

REGION_COUNT=$(wc -l < "$REGION_LIST_TMP" | tr -d ' ')
echo "Total regions to process: ${REGION_COUNT}"

# ---- Run ----
if [ "$PARALLEL" -gt 1 ]; then
    echo "Running in parallel ($PARALLEL jobs)..."
    cat "$REGION_LIST_TMP" \
        | xargs -P "$PARALLEL" -I {} \
        bash -c 'systemd-run --user --scope -p MemoryMax='"${MEMORY_MAX}"' python -u -m tools.run_sensitivity_all --env '"$ENV"' --region "$1" || echo "FAILED: $1"' _ {}
else
    echo "Running sequentially..."
    python -u -m tools.run_sensitivity_all --env "$ENV" "${EXTRA_FLAGS[@]}"
fi

echo "Done!"
