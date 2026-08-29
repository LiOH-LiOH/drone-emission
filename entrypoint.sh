#!/usr/bin/env bash
# Formal remote launcher: one cgroup, optional tmux, and bounded city parallelism.
set -euo pipefail

cd "$(dirname "$0")"

MEMORY_MAX_GB="${DRONE_EMISSION_MEMORY_MAX_GB:-96}"
SYSTEM_RESERVE_GB="${DRONE_EMISSION_SYSTEM_RESERVE_GB:-12}"
CITY_RESERVATION_GB="${DRONE_EMISSION_CITY_RESERVATION_GB:-21}"
ENVIRONMENT="remote"
JOBS=""
TMUX_NAME=""
ORIGINAL_ARGS=("$@")

usage() {
    echo "Usage: $0 [--env ENV] [--jobs N] [--tmux NAME] STAGE [stage arguments]" >&2
    echo "STAGE: inputs, calibrate, cities, aggregate, cohort, analyze, render, promote, city-promote, status, reproduce" >&2
}

while (( $# )); do
    case "$1" in
        --env) [[ $# -ge 2 ]] || { usage; exit 2; }; ENVIRONMENT="$2"; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || { usage; exit 2; }; JOBS="$2"; shift 2 ;;
        --tmux) [[ $# -ge 2 ]] || { usage; exit 2; }; TMUX_NAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done

[[ $# -ge 1 ]] || { usage; exit 2; }
STAGE="$1"
shift
STAGE_ARGS=("$@")
case "$STAGE" in
    inputs|calibrate|cities|aggregate|cohort|analyze|render|promote|city-promote|status|reproduce) ;;
    *) echo "Unknown stage: $STAGE" >&2; usage; exit 2 ;;
esac

for value in "$MEMORY_MAX_GB" "$SYSTEM_RESERVE_GB" "$CITY_RESERVATION_GB"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "Memory settings must be positive integer GiB values" >&2
        exit 2
    }
done
(( MEMORY_MAX_GB > SYSTEM_RESERVE_GB )) || {
    echo "Global memory cap must exceed the system reserve" >&2
    exit 2
}
CITY_BUDGET_GB=$((MEMORY_MAX_GB - SYSTEM_RESERVE_GB))
MAX_CITY_JOBS=$((CITY_BUDGET_GB / CITY_RESERVATION_GB))
(( MAX_CITY_JOBS >= 1 )) || {
    echo "City reservation does not fit within the scheduling budget" >&2
    exit 2
}
JOBS="${JOBS:-$MAX_CITY_JOBS}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
(( JOBS <= MAX_CITY_JOBS )) || {
    echo "--jobs=$JOBS exceeds the $MAX_CITY_JOBS-worker budget (${CITY_BUDGET_GB}G / ${CITY_RESERVATION_GB}G)" >&2
    exit 2
}

PYTHON="${DRONE_EMISSION_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Python environment not found: $PYTHON" >&2; exit 1; }

if [[ -n "$TMUX_NAME" && -z "${DRONE_EMISSION_IN_TMUX:-}" ]]; then
    command -v tmux >/dev/null 2>&1 || { echo "tmux is required for --tmux" >&2; exit 1; }
    if tmux has-session -t "$TMUX_NAME" 2>/dev/null; then
        exec tmux attach -t "$TMUX_NAME"
    fi
    printf -v tmux_command '%q ' "$PWD/entrypoint.sh" "${ORIGINAL_ARGS[@]}"
    tmux new-session -d -s "$TMUX_NAME" "DRONE_EMISSION_IN_TMUX=1 $tmux_command"
    exec tmux attach -t "$TMUX_NAME"
fi

if [[ -z "${DRONE_EMISSION_IN_SCOPE:-}" ]]; then
    command -v systemd-run >/dev/null 2>&1 || {
        echo "systemd-run is required; refusing to run without the process-tree cap" >&2
        exit 1
    }
    exec systemd-run --user --scope --quiet \
        -p "MemoryMax=${MEMORY_MAX_GB}G" \
        -p "MemorySwapMax=0" \
        env DRONE_EMISSION_IN_SCOPE=1 "$PWD/entrypoint.sh" "${ORIGINAL_ARGS[@]}"
fi

run_python_stage() {
    local stage="$1"
    shift
    "$PYTHON" -m drone_emission --env "$ENVIRONMENT" "$stage" "$@"
}

run_cities() {
    local args=("$@")
    local has_region=0
    local has_batch=0
    local item
    for item in ${args[@]+"${args[@]}"}; do
        [[ "$item" == "--region" || "$item" == --region=* ]] && has_region=1
        [[ "$item" == "--run-dir" || "$item" == --run-dir=* ]] && has_batch=1
    done
    if (( has_region )); then
        run_python_stage cities ${args[@]+"${args[@]}"}
        return
    fi
    if (( has_batch )); then
        run_python_stage cities --jobs "$JOBS" ${args[@]+"${args[@]}"}
        return
    fi
    "$PYTHON" -c 'import json, sys; registry=json.load(open(sys.argv[1], encoding="utf-8")); print("\n".join(key for key, value in registry.items() if value.get("region_level") == "city"))' regions.json \
        | xargs -P "$JOBS" -I '{}' \
            "$PYTHON" -m drone_emission --env "$ENVIRONMENT" cities --region '{}' ${args[@]+"${args[@]}"}
}

if [[ "$STAGE" != "reproduce" ]]; then
    if [[ "$STAGE" == "cities" ]]; then
        run_cities ${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"}
    else
        run_python_stage "$STAGE" ${STAGE_ARGS[@]+"${STAGE_ARGS[@]}"}
    fi
    exit 0
fi

city_args=()
aggregate_args=()
analyze_args=()
render_args=()
index=0
while (( index < ${#STAGE_ARGS[@]} )); do
    argument="${STAGE_ARGS[$index]}"
    case "$argument" in
        --force)
            city_args+=("$argument"); aggregate_args+=("$argument"); analyze_args+=("$argument") ;;
        --no-fetch) city_args+=("$argument") ;;
        --region|--only)
            (( index + 1 < ${#STAGE_ARGS[@]} )) || { echo "$argument requires a value" >&2; exit 2; }
            value="${STAGE_ARGS[$((index + 1))]}"
            if [[ "$argument" == "--region" ]]; then
                city_args+=("$argument" "$value")
            else
                render_args+=("$argument" "$value")
            fi
            index=$((index + 1))
            ;;
        --region=*|--only=*)
            if [[ "$argument" == --region=* ]]; then city_args+=("$argument"); else render_args+=("$argument"); fi ;;
        *) echo "Unsupported reproduce argument: $argument" >&2; exit 2 ;;
    esac
    index=$((index + 1))
done

run_python_stage inputs
run_cities ${city_args[@]+"${city_args[@]}"}
run_python_stage aggregate ${aggregate_args[@]+"${aggregate_args[@]}"}
run_python_stage analyze ${analyze_args[@]+"${analyze_args[@]}"}
run_python_stage render ${render_args[@]+"${render_args[@]}"}
