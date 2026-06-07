#!/bin/bash

# Fast lifecycle component refresh from existing OD CSVs.
#
# Usage:
#   ./refresh_lifecycle_parts.sh [env] [extra args...]
# Examples:
#   ./refresh_lifecycle_parts.sh remote --consolidate --enrich
#   ./refresh_lifecycle_parts.sh remote --region shanghai --dry-run
#   ./refresh_lifecycle_parts.sh local --level city --no-write-od

set -euo pipefail
cd "$(dirname "$0")"

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Error: .venv not found" >&2
    exit 1
fi

ENV=${1:-remote}
shift || true

cd src/drone_emission
python -u -m tools.recompute_lifecycle_parts --env "$ENV" "$@"
