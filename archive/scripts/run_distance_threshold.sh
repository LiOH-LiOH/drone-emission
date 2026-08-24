#!/bin/bash

# Standalone all-city distance-threshold lifecycle analysis.
#
# This script is intentionally separate from the main pipeline. Each city runs
# in its own systemd scope so the worker memory limit is enforced independently.
# Beijing calibration (7,148,885 optimized rows, 2026-08-05): GNU time -v
# measured 1,020,968 KiB (0.974 GiB) for the main analysis and 1,170,688 KiB
# (1.116 GiB) with manual curve export. Adding 30% and rounding up gives a
# 2 GiB worker limit. Workers are also capped for shared-storage I/O.
#
# Usage:
#   ./run_distance_threshold.sh                 # calibrated workers, remote env
#   ./run_distance_threshold.sh 4 remote        # lower concurrency
#   ./run_distance_threshold.sh 1 local --force

set -uo pipefail

TOTAL_WORKER_BUDGET_GIB=60
TASK_MEMORY_BUDGET_GIB=64
WORKER_MEMORY_GIB=2
IO_WORKER_CAP=8
MEMORY_WORKER_CAP=$((TOTAL_WORKER_BUDGET_GIB / WORKER_MEMORY_GIB))
if [ "$MEMORY_WORKER_CAP" -lt "$IO_WORKER_CAP" ]; then
    MAX_WORKERS=$MEMORY_WORKER_CAP
else
    MAX_WORKERS=$IO_WORKER_CAP
fi
DEFAULT_WORKERS=$MAX_WORKERS

SESSION_NAME="drone-distance-threshold"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "${TMUX:-}" ] && [ -z "${INSIDE_DISTANCE_THRESHOLD:-}" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi
    echo "Starting '${SESSION_NAME}'"
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_DISTANCE_THRESHOLD=1 bash ${SCRIPT_DIR}/run_distance_threshold.sh $*; status=\$?; echo ''; echo \"Finished with status \$status. Press Enter to close.\"; read; exit \$status"
    exec tmux attach -t "${SESSION_NAME}"
fi

POSITIONAL=()
FORCE_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_FLAG="--force" ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

WORKERS=${POSITIONAL[0]:-$DEFAULT_WORKERS}
ENVIRONMENT=${POSITIONAL[1]:-remote}

if ! [[ "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "workers must be a positive integer: $WORKERS" >&2
    exit 2
fi
if [ "$WORKERS" -gt "$MAX_WORKERS" ]; then
    echo "workers=$WORKERS exceeds calibrated maximum $MAX_WORKERS" >&2
    echo "Per-worker limit=${WORKER_MEMORY_GIB} GiB; worker budget=${TOTAL_WORKER_BUDGET_GIB} GiB" >&2
    exit 2
fi

cd "$SCRIPT_DIR"
mkdir -p logs
LOG_FILE="logs/distance_threshold_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Environment: $ENVIRONMENT"
echo "Workers: $WORKERS (max=$MAX_WORKERS, I/O cap=$IO_WORKER_CAP)"
echo "Per-worker MemoryMax: ${WORKER_MEMORY_GIB}G"
echo "Worker budget: ${TOTAL_WORKER_BUDGET_GIB}G; task budget: ${TASK_MEMORY_BUDGET_GIB}G"

if [ -d .venv ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found" >&2
    exit 1
fi

cd src/drone_emission
REGION_LIST_TMP=$(mktemp)
trap 'rm -f "$REGION_LIST_TMP"' EXIT
python -m tools.run_distance_threshold_analysis \
    --env "$ENVIRONMENT" --list-regions > "$REGION_LIST_TMP"
REGION_COUNT=$(wc -l < "$REGION_LIST_TMP" | tr -d ' ')
echo "Cities to process: $REGION_COUNT"

export DISTANCE_THRESHOLD_ENV="$ENVIRONMENT"
export DISTANCE_THRESHOLD_FORCE="$FORCE_FLAG"
export DISTANCE_THRESHOLD_WORKER_MEMORY_GIB="$WORKER_MEMORY_GIB"

set +e
xargs -P "$WORKERS" -I {} bash -c '
    city="$1"
    systemd-run --user --scope \
        -p "MemoryMax=${DISTANCE_THRESHOLD_WORKER_MEMORY_GIB}G" \
        python -u -m tools.run_distance_threshold_analysis \
            --env "$DISTANCE_THRESHOLD_ENV" --region "$city" \
            $DISTANCE_THRESHOLD_FORCE
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "FAILED: $city"
    fi
    exit "$result"
' _ {} < "$REGION_LIST_TMP"
WORKER_STATUS=$?
set -e

echo "Consolidating current city summaries..."
set +e
systemd-run --user --scope -p MemoryMax=4G \
    python -u -m tools.run_distance_threshold_analysis \
        --env "$ENVIRONMENT" --consolidate
CONSOLIDATE_STATUS=$?
set -e

if [ "$WORKER_STATUS" -ne 0 ] || [ "$CONSOLIDATE_STATUS" -ne 0 ]; then
    echo "Distance-threshold analysis incomplete. Check the status CSV and log." >&2
    exit 1
fi

echo "Distance-threshold analysis complete."
