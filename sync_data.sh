#!/bin/bash
# sync_data.sh - Sync sensitive data files to remote server
#
# Usage:
#   bash sync_data.sh              # uses default remote host
#   bash sync_data.sh hku-cpu      # specify SSH host

set -euo pipefail

REMOTE_HOST="${1:-hku-cpu}"
REMOTE_DIR="/home/ljx/code/drone-emission"
LOCAL_DIR="$(dirname "$0")"

FILES=(
    "data/combined_data.csv"
    "data/gef_china.csv"
)

echo "Syncing data to ${REMOTE_HOST}:${REMOTE_DIR}"

# Ensure remote data dir exists
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}/data"

for file in "${FILES[@]}"; do
    src="${LOCAL_DIR}/${file}"
    if [ -f "${src}" ]; then
        echo "  ${file}"
        rsync -ahP "${src}" "${REMOTE_HOST}:${REMOTE_DIR}/${file}"
    else
        echo "  ${file} (not found locally, skipping)"
    fi
done

echo "Done."
