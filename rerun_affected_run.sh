#!/bin/bash

# Step 2: Rerun cities listed in affected_cities.txt through the full pipeline.
#
# Run rerun_affected_clean.sh first to generate affected_cities.txt.
# Wraps in tmux session with resource limits, like entrypoint.sh.
#
# Usage: ./rerun_affected_run.sh [parallel_jobs] [env]
# Examples:
#   ./rerun_affected_run.sh           # 4 parallel, remote (default)
#   ./rerun_affected_run.sh 8         # 8 parallel
#   ./rerun_affected_run.sh 1 local   # sequential, local

MEMORY_MAX="128G"
SESSION_NAME="rerun-affected"
CITY_LIST="affected_cities.txt"

# ---- If not inside tmux, wrap ourselves in a tmux session ----
if [ -z "$TMUX" ] && [ -z "$INSIDE_RERUN" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi

    echo "Starting '${SESSION_NAME}' (mem=${MEMORY_MAX})"
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_RERUN=1 bash $(cd "$(dirname "$0")" && pwd)/rerun_affected_run.sh $*; echo ''; echo 'Done. Press Enter to close.'; read"
    exec tmux attach -t "${SESSION_NAME}"
fi

# ---- Actual work (runs inside tmux) ----
cd "$(dirname "$0")"

PARALLEL=${1:-4}
ENV=${2:-remote}

if [ ! -f "${CITY_LIST}" ]; then
    echo "Error: ${CITY_LIST} not found. Run rerun_affected_clean.sh first."
    exit 1
fi

CITY_COUNT=$(wc -l < "${CITY_LIST}" | tr -d ' ')
echo "Cities to rerun: ${CITY_COUNT} (from ${CITY_LIST})"

if [ "$CITY_COUNT" -eq 0 ]; then
    echo "No cities to rerun."
    exit 0
fi

# Setup logging
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rerun_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

# Activate venv
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
fi

cd src/drone_emission

echo "Environment: $ENV (mem=${MEMORY_MAX}, parallel=${PARALLEL})"
echo "Rerunning ${CITY_COUNT} affected cities..."
echo ""

cat "../../${CITY_LIST}" | xargs -P "$PARALLEL" -I {} \
    bash -c 'systemd-run --user --scope -p MemoryMax='"${MEMORY_MAX}"' python -u main.py --region "$1" --skip-completed --env '"$ENV"' || echo "FAILED: $1"' _ {}

echo ""
echo "Done!"
