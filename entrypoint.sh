#!/bin/bash

# Run analysis for all regions
# Usage: ./entrypoint.sh [parallel_jobs]

cd "$(dirname "$0")/src/drone_emission"

REGIONS=(jianghan pudong yuzhong siming gulou)

PARALLEL=${1:-1}  # Default to 1 (sequential), pass number for parallel jobs

if [ "$PARALLEL" -gt 1 ]; then
    echo "Running $PARALLEL regions in parallel..."
    printf '%s\n' "${REGIONS[@]}" | xargs -P "$PARALLEL" -I {} python -u main.py --region {}
else
    echo "Running regions sequentially..."
    for region in "${REGIONS[@]}"; do
        echo "=== Processing: $region ==="
        python -u main.py --region "$region"
    done
fi

echo "Done!"
