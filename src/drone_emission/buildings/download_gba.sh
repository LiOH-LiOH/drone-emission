#!/bin/bash
# download_gba.sh
#
# Download GlobalBuildingAtlas tile data from HuggingFace and produce
# complete LoD1 GeoJSON with building polygons + heights.
#
# Usage:
#   bash download_gba.sh TILE REGION [DATA_DIR]
#
# Example:
#   bash download_gba.sh e110_n25_e115_n20 asiaeast ./gba_data

set -euo pipefail

# ---- Configuration ----
TILE="${1:?Usage: download_gba.sh TILE REGION [DATA_DIR]}"
REGION="${2:?Usage: download_gba.sh TILE REGION [DATA_DIR]}"
DATA_DIR="${3:-./gba_data}"

# HuggingFace base URLs
HF_ODBL="https://huggingface.co/datasets/zhu-xlab/GBA.ODbLPolygon/resolve/main"
HF_LOD1="https://huggingface.co/datasets/zhu-xlab/GBA.LoD1/resolve/main"

# Files to download
ODBL_FILE="${REGION}/${TILE}.geojson"       # ~183 MB
POLYGON_FILE="Polygon/${REGION}/${TILE}.geojson"  # ~3.68 GB
LOD1_JSON_FILE="LoD1/${REGION}/${TILE}.json"      # ~571 MB
SCRIPT_FILE="produce_lod1.py"

# Local directory structure (matches what produce_lod1.py expects)
ODBL_DIR="${DATA_DIR}/ODbLPolygon/${REGION}"
POLYGON_DIR="${DATA_DIR}/Polygon/${REGION}"
LOD1_DIR="${DATA_DIR}/LoD1/${REGION}"
OUTPUT_DIR="${DATA_DIR}/LoD1_GeoJSON"

echo "============================================================"
echo "  GlobalBuildingAtlas Tile Download"
echo "============================================================"
echo "Tile:       ${TILE}"
echo "Region:     ${REGION}"
echo "Data dir:   ${DATA_DIR}"
echo ""
echo "Files to download:"
echo "  1. ODbLPolygon  ${ODBL_FILE}        (~183 MB)"
echo "  2. Polygon      ${POLYGON_FILE}     (~3.68 GB)"
echo "  3. LoD1 JSON    ${LOD1_JSON_FILE}   (~571 MB)"
echo "  4. Script       ${SCRIPT_FILE}"
echo "  Total: ~4.4 GB"
echo "============================================================"

# ---- Create directories ----
mkdir -p "${ODBL_DIR}" "${POLYGON_DIR}" "${LOD1_DIR}" "${OUTPUT_DIR}"

# ---- Download function with resume support ----
download() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [ -f "${dest}" ]; then
        echo "[SKIP] ${label} already exists: ${dest}"
        return 0
    fi

    echo "[DOWN] ${label}"
    echo "       ${url}"
    echo "    -> ${dest}"

    # Use curl with resume support (-C -) and progress bar
    curl -L -C - --retry 5 --retry-delay 10 \
        -o "${dest}" \
        "${url}"

    echo "[DONE] ${label}"
    echo ""
}

# ---- Step 1: Download tile files ----
echo ""
echo "---- Step 1: Downloading tile files ----"

download \
    "${HF_ODBL}/${ODBL_FILE}" \
    "${ODBL_DIR}/${TILE}.geojson" \
    "ODbLPolygon (building polygons, ODbL source)"

download \
    "${HF_LOD1}/${POLYGON_FILE}" \
    "${POLYGON_DIR}/${TILE}.geojson" \
    "Polygon (building polygons, non-ODbL source)"

download \
    "${HF_LOD1}/${LOD1_JSON_FILE}" \
    "${LOD1_DIR}/${TILE}.json" \
    "LoD1 JSON (height + variance data)"

# ---- Step 2: Download produce_lod1.py ----
echo ""
echo "---- Step 2: Downloading produce_lod1.py ----"

download \
    "${HF_LOD1}/produce_lod1.py" \
    "${DATA_DIR}/produce_lod1.py" \
    "produce_lod1.py (merge script)"

# ---- Step 3: Install dependencies ----
echo ""
echo "---- Step 3: Checking Python dependencies ----"

for pkg in ijson fiona; do
    if python -c "import ${pkg}" 2>/dev/null; then
        echo "[OK]   ${pkg}"
    else
        echo "[INST] Installing ${pkg}..."
        pip install "${pkg}"
    fi
done

# ---- Step 4: Run produce_lod1.py ----
echo ""
echo "---- Step 4: Producing LoD1 GeoJSON with heights ----"
echo "This merges ODbLPolygon + Polygon + LoD1 JSON -> complete GeoJSON"
echo ""

# Symlink only this tile into a temp directory to avoid integrity check
# conflicts with other tiles being downloaded in parallel
TMPDIR=$(mktemp -d)
mkdir -p "${TMPDIR}/odbl" "${TMPDIR}/poly" "${TMPDIR}/json"
ln -s "${ODBL_DIR}/${TILE}.geojson" "${TMPDIR}/odbl/${TILE}.geojson"
ln -s "${POLYGON_DIR}/${TILE}.geojson" "${TMPDIR}/poly/${TILE}.geojson"
ln -s "${LOD1_DIR}/${TILE}.json" "${TMPDIR}/json/${TILE}.json"

python "${DATA_DIR}/produce_lod1.py" \
    --odbl_root "${TMPDIR}/odbl" \
    --polygon_root "${TMPDIR}/poly" \
    --json_root "${TMPDIR}/json" \
    --output_root "${OUTPUT_DIR}/${REGION}"

rm -rf "${TMPDIR}"

echo ""
echo "============================================================"
echo "  Complete!"
echo "============================================================"
echo "Output: ${OUTPUT_DIR}/${REGION}/${TILE}.geojson"
echo ""
echo "This GeoJSON contains all buildings in the tile ${TILE}"
echo "with height and variance fields."
echo ""
