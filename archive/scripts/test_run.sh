#!/bin/bash

# Test run: single city to verify the pipeline works end-to-end.
# Usage: bash scripts/test_run.sh [region] [env]
# Examples:
#   bash scripts/test_run.sh               # changsha, remote env (default)
#   bash scripts/test_run.sh beijing local # beijing, local env

cd "$(dirname "$0")/.."

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

cd src/drone_emission

REGION=${1:-changsha}
ENV=${2:-remote}

echo "=== Test run: ${REGION} (env=${ENV}) ==="
python -u main.py --region "$REGION" --env "$ENV"
echo "Done!"
