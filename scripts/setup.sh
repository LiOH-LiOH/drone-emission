#!/bin/bash
# setup.sh - Create venv and install dependencies (run from repo root)
#
# Usage:
#   bash scripts/setup.sh             # uses python3
#   bash scripts/setup.sh python3.11  # specify python version

set -euo pipefail

PYTHON="${1:-python3}"
VENV_DIR=".venv"
REQ_FILE="src/drone_emission/requirements.txt"

cd "$(dirname "$0")/.."

# Copy regions config template if regions.json doesn't exist
if [ ! -f "regions.json" ]; then
    echo "Creating regions.json from template..."
    cp regions.template.json regions.json
else
    echo "regions.json already exists, skipping"
fi

echo "Python: $(${PYTHON} --version)"

if [ -d "${VENV_DIR}" ]; then
    echo "venv already exists at ${VENV_DIR}, skipping creation"
else
    echo "Creating venv..."
    ${PYTHON} -m venv "${VENV_DIR}"
fi

echo "Installing dependencies..."
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"
"${VENV_DIR}/bin/pip" install -e .

echo ""
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
