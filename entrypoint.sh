#!/bin/bash

# Run analysis for all regions (designed for remote server)
# Starts a tmux session with resource limits, auto-resumes if already running.
#
# Usage: ./entrypoint.sh [parallel_jobs] [env]
# Examples:
#   ./entrypoint.sh           # Sequential, remote env (default)
#   ./entrypoint.sh 4         # 4 parallel jobs, remote env
#   ./entrypoint.sh 1 local   # Sequential, local env

# Resource limits (edit below):
MEMORY_MAX="32G"
CPU_QUOTA="400%"   # 200% = 2 cores
SESSION_NAME="drone-emission"

# ---- If not inside tmux, wrap ourselves in a tmux session ----
if [ -z "$TMUX" ] && [ -z "$INSIDE_ENTRYPOINT" ]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        echo "Session '${SESSION_NAME}' already running. Attaching..."
        exec tmux attach -t "${SESSION_NAME}"
    fi

    echo "Starting '${SESSION_NAME}' (mem=${MEMORY_MAX}, cpu=${CPU_QUOTA})"
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_ENTRYPOINT=1 bash $(cd "$(dirname "$0")" && pwd)/entrypoint.sh $*; echo ''; echo 'Done. Press Enter to close.'; read"
    exec tmux attach -t "${SESSION_NAME}"
fi

# ---- Actual analysis (runs inside tmux) ----
cd "$(dirname "$0")"

# Setup logging — logs go to ./logs/ with timestamp
LOG_DIR="./logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found, using system Python"
fi

cd src/drone_emission

# Parse flags
ALL_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--all" ]; then
        ALL_MODE=true
    fi
done

PARALLEL=${1:-1}  # Default to 1 (sequential), pass number for parallel jobs
ENV=${2:-remote}  # Default to remote (entrypoint.sh is typically run on server)

echo "Environment: $ENV (mem=${MEMORY_MAX}, cpu=${CPU_QUOTA})"

if [ "$ALL_MODE" = true ]; then
    if [ "$PARALLEL" -gt 1 ]; then
        echo "Running ALL regions in parallel ($PARALLEL jobs)..."
        # Extract region keys from regions.json, pipe through xargs for parallel execution
        python -c "import json; [print(k) for k in json.load(open('../../regions.json'))]" \
            | xargs -P "$PARALLEL" -I {} \
            systemd-run --user --scope -p MemoryMax=${MEMORY_MAX} -p CPUQuota=${CPU_QUOTA} \
            python -u main.py --region {} --skip-completed --env "$ENV"
    else
        echo "Running ALL cities via main.py --all..."
        systemd-run --user --scope -p MemoryMax=${MEMORY_MAX} -p CPUQuota=${CPU_QUOTA} \
            python -u main.py --all --skip-completed --env "$ENV"
    fi
else
    REGIONS=(changsha qingdao xiamen)

    if [ "$PARALLEL" -gt 1 ]; then
        echo "Running $PARALLEL regions in parallel..."
        printf '%s\n' "${REGIONS[@]}" | xargs -P "$PARALLEL" -I {} \
            systemd-run --user --scope -p MemoryMax=${MEMORY_MAX} -p CPUQuota=${CPU_QUOTA} \
            python -u main.py --region {} --env "$ENV"
    else
        echo "Running regions sequentially..."
        for region in "${REGIONS[@]}"; do
            echo "=== Processing: $region ==="
            systemd-run --user --scope -p MemoryMax=${MEMORY_MAX} -p CPUQuota=${CPU_QUOTA} \
                python -u main.py --region "$region" --env "$ENV"
        done
    fi
fi

echo "Done!"
