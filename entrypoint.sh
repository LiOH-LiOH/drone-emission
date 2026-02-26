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

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating .venv..."
    source .venv/bin/activate
else
    echo "Warning: .venv not found, using system Python"
fi

cd src/drone_emission

REGIONS=(pudong gulou chaoyang wuhou tianhe luohu xihu)
# REGIONS=(gulou chaoyang luohu)

PARALLEL=${1:-1}  # Default to 1 (sequential), pass number for parallel jobs
ENV=${2:-remote}  # Default to remote (entrypoint.sh is typically run on server)

echo "Environment: $ENV (mem=${MEMORY_MAX}, cpu=${CPU_QUOTA})"

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

echo "Done!"
