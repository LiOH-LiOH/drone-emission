#!/bin/bash
# log_summary.sh — Summarize a pipeline run log
#
# Usage:
#   bash log_summary.sh [LOG_FILE]
#
# If LOG_FILE is omitted, uses the most recent log in ./logs/.
# Can also be run via SSH:
#   ssh hku-sto "cd ~/code/drone-emission && bash log_summary.sh"

set -euo pipefail

# ---- Resolve log file ----
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs"

if [ $# -ge 1 ]; then
    LOG="$1"
else
    LOG=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
    if [ -z "$LOG" ]; then
        echo "No log files found in ${LOG_DIR}/" >&2
        exit 1
    fi
fi

if [ ! -f "$LOG" ]; then
    echo "Log file not found: $LOG" >&2
    exit 1
fi

LOG_NAME=$(basename "$LOG")

# ---- Extract timestamp from filename ----
# Format: run_YYYYMMDD_HHMMSS.log
if [[ "$LOG_NAME" =~ run_([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})([0-9]{2}) ]]; then
    START_TIME="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]} ${BASH_REMATCH[4]}:${BASH_REMATCH[5]}:${BASH_REMATCH[6]}"
else
    START_TIME="unknown"
fi

# ---- Count regions by status ----
# "Running analysis for X..." = region started processing
STARTED=$(grep -cE '^Running analysis for ' "$LOG" 2>/dev/null || echo 0)

# "Skipping X (already has emission results)" = skipped (already done)
SKIPPED=$(grep -cE 'Skipping .* \(already has emission results\)' "$LOG" 2>/dev/null || echo 0)

# "Equity analysis complete for X" = last stage completed = region fully done
COMPLETED=$(grep -cE '^Equity analysis complete for ' "$LOG" 2>/dev/null || echo 0)

# "FAILED: X" = region failed (parallel mode marker from entrypoint.sh)
FAILED_COUNT=$(grep -cE '^FAILED: ' "$LOG" 2>/dev/null || echo 0)

# ---- Detect active workers ----
ACTIVE_WORKERS=""
ACTIVE_COUNT=0
if command -v ps &>/dev/null; then
    ACTIVE_WORKERS=$(ps -eo pid,stat,etime,args 2>/dev/null \
        | grep '[p]ython.*main.py.*--region' \
        | grep -v grep || true)
    ACTIVE_COUNT=$(echo "$ACTIVE_WORKERS" | grep -c '[^ ]' 2>/dev/null || echo 0)
fi

# ---- Detect D-state (stuck) processes ----
DSTATE_COUNT=0
if [ -n "$ACTIVE_WORKERS" ]; then
    DSTATE_COUNT=$(echo "$ACTIVE_WORKERS" | awk '$2 ~ /^D/' | wc -l | tr -d ' ')
fi

# ---- Extract failed region names ----
FAILED_LIST=$(grep -oE '^FAILED: (.+)' "$LOG" 2>/dev/null | sed 's/FAILED: //' || true)

# ---- Extract per-stage progress for active regions ----
# Find regions that started but didn't complete or fail
STARTED_REGIONS=$(grep -oE '^Running analysis for (.+)\.\.\.' "$LOG" | sed 's/Running analysis for //;s/\.\.\.//')
COMPLETED_REGIONS=$(grep -oE '^Equity analysis complete for (.+)' "$LOG" | sed 's/Equity analysis complete for //')
FAILED_REGIONS=$(grep -oE '^FAILED: (.+)' "$LOG" | sed 's/FAILED: //')

# Regions still in progress (started but not completed or failed)
IN_PROGRESS=""
while IFS= read -r region; do
    [ -z "$region" ] && continue
    region_lower=$(echo "$region" | tr '[:upper:]' '[:lower:]')
    # Check if completed (case-insensitive match — completion log uses title case)
    if echo "$COMPLETED_REGIONS" | tr '[:upper:]' '[:lower:]' | grep -qxF "$region_lower"; then
        continue
    fi
    if echo "$FAILED_REGIONS" | grep -qxF "$region"; then
        continue
    fi
    # Determine last stage reached
    last_stage="started"
    if grep -q "Saved.*${region}.*_od_pairs.csv" "$LOG" 2>/dev/null; then
        last_stage="demand"
    fi
    if grep -q "Road network saved.*${region}" "$LOG" 2>/dev/null; then
        last_stage="roadnet"
    fi
    if grep -qE "Emission (results|comparison).*${region}|emission.*saved.*${region}" "$LOG" 2>/dev/null; then
        last_stage="emission"
    fi
    # Check buildings stage
    if [ "$last_stage" = "started" ]; then
        if grep -q "buildings.*${region}\|${region}.*buildings" "$LOG" 2>/dev/null; then
            last_stage="buildings"
        fi
    fi
    IN_PROGRESS="${IN_PROGRESS}  ${region} (at: ${last_stage})\n"
done <<< "$STARTED_REGIONS"

# ---- Per-stage counts ----
BUILDINGS_DONE=$(grep -cE 'Loaded [0-9]+ buildings\.' "$LOG" 2>/dev/null || echo 0)
OD_DONE=$(grep -cE 'Saved OD pairs to' "$LOG" 2>/dev/null || echo 0)
ROADNET_DONE=$(grep -cE 'Road network saved to' "$LOG" 2>/dev/null || echo 0)

# ---- Print summary ----
echo "============================================================"
echo "  Pipeline Log Summary"
echo "============================================================"
echo "Log file:    ${LOG_NAME}"
echo "Started at:  ${START_TIME}"
echo ""
echo "--- Region Status ---"
printf "  Skipped (prior runs):  %d\n" "$SKIPPED"
printf "  Started (this run):    %d\n" "$STARTED"
printf "  Completed (this run):  %d\n" "$COMPLETED"
printf "  Failed:                %d\n" "$FAILED_COUNT"
printf "  Total processed:       %d\n" "$((SKIPPED + COMPLETED))"
echo ""
echo "--- Stage Progress (this run) ---"
printf "  Buildings loaded:      %d\n" "$BUILDINGS_DONE"
printf "  OD pairs generated:    %d\n" "$OD_DONE"
printf "  Road networks fetched: %d\n" "$ROADNET_DONE"
printf "  Equity complete:       %d\n" "$COMPLETED"
echo ""
echo "--- Active Workers ---"
printf "  Running: %d" "$ACTIVE_COUNT"
if [ "$DSTATE_COUNT" -gt 0 ]; then
    printf "  (!! %d in D-state !!)" "$DSTATE_COUNT"
fi
echo ""

if [ "$ACTIVE_COUNT" -gt 0 ]; then
    echo "$ACTIVE_WORKERS" | while IFS= read -r line; do
        [ -z "$line" ] && continue
        # Extract region name from --region flag
        region=$(echo "$line" | grep -oE '\-\-region [a-z]+' | sed 's/--region //')
        etime=$(echo "$line" | awk '{print $3}')
        stat=$(echo "$line" | awk '{print $2}')
        [ -n "$region" ] && printf "    %-40s  elapsed: %-14s  state: %s\n" "$region" "$etime" "$stat"
    done
fi

if [ -n "$IN_PROGRESS" ] && [ "$ACTIVE_COUNT" -gt 0 ]; then
    echo ""
    echo "--- In-Progress Regions (last stage reached) ---"
    printf "$IN_PROGRESS"
fi

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo ""
    echo "--- Failed Regions ---"
    echo "$FAILED_LIST" | while IFS= read -r r; do
        [ -z "$r" ] && continue
        # Try to extract error reason
        error=$(grep -A2 "ERROR processing ${r}" "$LOG" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' | head -c 120)
        if [ -n "$error" ]; then
            printf "  %-45s  %s\n" "$r" "$error"
        else
            echo "  $r"
        fi
    done
fi

echo ""
echo "============================================================"
