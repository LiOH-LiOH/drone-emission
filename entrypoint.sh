#!/bin/bash

# Run analysis for all regions (designed for remote server)
# Usage: ./entrypoint.sh [parallel_jobs] [env]
# Examples:
#   ./entrypoint.sh           # Sequential, remote env (default)
#   ./entrypoint.sh 4         # 4 parallel jobs, remote env
#   ./entrypoint.sh 1 local   # Sequential, local env

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

echo "Environment: $ENV"

if [ "$PARALLEL" -gt 1 ]; then
    echo "Running $PARALLEL regions in parallel..."
    printf '%s\n' "${REGIONS[@]}" | xargs -P "$PARALLEL" -I {} python -u main.py --region {} --env "$ENV"
else
    echo "Running regions sequentially..."
    for region in "${REGIONS[@]}"; do
        echo "=== Processing: $region ==="
        python -u main.py --region "$region" --env "$ENV"
    done
fi

echo "Done!"
