#!/bin/bash
# run_remote.sh
#
# One-command launcher for remote SSH sessions.
# Starts a tmux session with resource limits, auto-resumes if already running.
#
# Usage:
#   bash run_remote.sh                    # run download_gba.sh with defaults
#   bash run_remote.sh "python train.py"  # run any command
#
# Resource limits (edit below):
MEMORY_MAX="32G"
CPU_QUOTA="400%"   # 200% = 2 cores
SESSION_NAME="gba"

# ---- What to run ----
DEFAULT_CMD="bash $(dirname "$0")/download_gba.sh"
CMD="${1:-$DEFAULT_CMD}"

# ---- Check if session already exists ----
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Session '${SESSION_NAME}' already running. Attaching..."
    exec tmux attach -t "${SESSION_NAME}"
fi

# ---- Launch new tmux session with resource limits ----
echo "Starting '${SESSION_NAME}' (mem=${MEMORY_MAX}, cpu=${CPU_QUOTA})"
tmux new-session -d -s "${SESSION_NAME}" \
    "systemd-run --user --scope \
        -p MemoryMax=${MEMORY_MAX} \
        -p CPUQuota=${CPU_QUOTA} \
        ${CMD}; \
     echo ''; echo 'Done. Press Enter to close.'; read"

exec tmux attach -t "${SESSION_NAME}"
