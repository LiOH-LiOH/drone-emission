#!/bin/bash

# Run fixed-threshold Monte Carlo resilience for every eligible city.
#
# Usage:
#   ./run_threshold_resilience.sh
#   ./run_threshold_resilience.sh 4 remote
#   ./run_threshold_resilience.sh 8 remote --force

set -euo pipefail

MAX_WORKERS=8
DEFAULT_WORKERS=8
MEMORY_HIGH_GIB=56
MEMORY_MAX_GIB=60
SESSION_NAME="drone-threshold-resilience"
SCOPE_NAME="drone-threshold-resilience-run"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

INSIDE=0
FORCE_FLAG=""
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --inside) INSIDE=1 ;;
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
    echo "workers=$WORKERS exceeds maximum $MAX_WORKERS" >&2
    exit 2
fi

if [ "$INSIDE" -eq 0 ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session '$SESSION_NAME' already exists. Attaching..."
        if [ -n "${TMUX:-}" ]; then
            exec tmux switch-client -t "$SESSION_NAME"
        fi
        exec tmux attach -t "$SESSION_NAME"
    fi

    INNER_ARGS=(
        systemd-run --user --scope --unit="$SCOPE_NAME"
        -p MemoryAccounting=yes
        -p "MemoryHigh=${MEMORY_HIGH_GIB}G"
        -p "MemoryMax=${MEMORY_MAX_GIB}G"
        bash "$SCRIPT_DIR/run_threshold_resilience.sh"
        --inside "$WORKERS" "$ENVIRONMENT"
    )
    if [ -n "$FORCE_FLAG" ]; then
        INNER_ARGS+=("$FORCE_FLAG")
    fi
    printf -v INNER_COMMAND '%q ' "${INNER_ARGS[@]}"
    INNER_COMMAND+="; status=\$?; echo; echo \"Finished with status \$status. Press Enter to close.\"; read -r; exit \$status"

    echo "Starting '$SESSION_NAME' with ${WORKERS} workers"
    echo "Aggregate memory: high=${MEMORY_HIGH_GIB}G, max=${MEMORY_MAX_GIB}G"
    tmux new-session -d -s "$SESSION_NAME" "$INNER_COMMAND"
    if [ -n "${TMUX:-}" ]; then
        exec tmux switch-client -t "$SESSION_NAME"
    fi
    exec tmux attach -t "$SESSION_NAME"
fi

cd "$SCRIPT_DIR"
if [ ! -d .venv ]; then
    echo "Error: .venv not found at $SCRIPT_DIR/.venv" >&2
    exit 1
fi
source .venv/bin/activate

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$SCRIPT_DIR/logs/threshold_resilience_runs/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/run.log"
ln -sfn "threshold_resilience_runs/$RUN_ID/run.log" \
    "$SCRIPT_DIR/logs/threshold_resilience_latest.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run ID: $RUN_ID"
echo "Environment: $ENVIRONMENT"
echo "Workers: $WORKERS (maximum=$MAX_WORKERS)"
echo "Aggregate memory: high=${MEMORY_HIGH_GIB}G, max=${MEMORY_MAX_GIB}G"
echo "Log: $LOG_FILE"

cd src/drone_emission
PREFLIGHT_FILE="$RUN_DIR/preflight.csv"
FINAL_STATUS_FILE="$RUN_DIR/final_status.csv"
QUEUE_FILE="$RUN_DIR/queue.txt"

INVENTORY_ARGS=(
    -m tools.run_threshold_resilience
    --env "$ENVIRONMENT"
    --save-samples
    --inventory-out "$PREFLIGHT_FILE"
)
if [ -n "$FORCE_FLAG" ]; then
    INVENTORY_ARGS+=("$FORCE_FLAG")
fi
python "${INVENTORY_ARGS[@]}"

awk -F, 'NR > 1 && ($2 == "ready" || $2 == "cached") {print $1}' \
    "$PREFLIGHT_FILE" > "$QUEUE_FILE"
RUNNABLE_COUNT=$(wc -l < "$QUEUE_FILE" | tr -d ' ')
TOTAL_COUNT=$(($(wc -l < "$PREFLIGHT_FILE") - 1))
echo "Registered cities: $TOTAL_COUNT"
echo "Runnable or cached cities: $RUNNABLE_COUNT"
echo "Preflight exclusions: $((TOTAL_COUNT - RUNNABLE_COUNT))"

export THRESHOLD_RESILIENCE_ENV="$ENVIRONMENT"
export THRESHOLD_RESILIENCE_FORCE="$FORCE_FLAG"

set +e
xargs -r -P "$WORKERS" -I {} bash -c '
    city="$1"
    python -u -m tools.run_threshold_resilience \
        --env "$THRESHOLD_RESILIENCE_ENV" \
        --region "$city" --save-samples $THRESHOLD_RESILIENCE_FORCE
    result=$?
    if [ "$result" -ne 0 ]; then
        echo "FAILED: $city"
    fi
    exit "$result"
' _ {} < "$QUEUE_FILE"
WORKER_STATUS=$?

python -m tools.run_threshold_resilience \
    --env "$ENVIRONMENT" --save-samples \
    --preflight-file "$PREFLIGHT_FILE" \
    --status-report-out "$FINAL_STATUS_FILE"
REPORT_STATUS=$?

python -m tools.run_threshold_resilience \
    --env "$ENVIRONMENT" --save-samples --consolidate
CONSOLIDATE_STATUS=$?
set -e

if [ -f "$FINAL_STATUS_FILE" ]; then
    ERROR_COUNT=$(awk -F, 'NR > 1 && $2 == "error" {count++} END {print count+0}' \
        "$FINAL_STATUS_FILE")
    OK_COUNT=$(awk -F, 'NR > 1 && ($2 == "ok" || $2 == "cached") {count++} END {print count+0}' \
        "$FINAL_STATUS_FILE")
    SKIPPED_COUNT=$((TOTAL_COUNT - OK_COUNT - ERROR_COUNT))
else
    ERROR_COUNT=1
    OK_COUNT=0
    SKIPPED_COUNT=$TOTAL_COUNT
fi

echo "Valid city results: $OK_COUNT"
echo "Skipped prerequisites: $SKIPPED_COUNT"
echo "Runtime errors: $ERROR_COUNT"
echo "Final status: $FINAL_STATUS_FILE"

if [ "$WORKER_STATUS" -ne 0 ] || [ "$REPORT_STATUS" -ne 0 ] \
    || [ "$CONSOLIDATE_STATUS" -ne 0 ] || [ "$ERROR_COUNT" -ne 0 ]; then
    echo "Threshold-resilience batch completed with errors." >&2
    exit 1
fi

echo "Threshold-resilience batch complete."
