#!/usr/bin/env bash
# Run only on hku-sto after the implementation commit has been pulled.
set -euo pipefail

cd "$(dirname "$0")/.."

git diff --quiet && git diff --cached --quiet || {
    echo "Remote worktree has tracked modifications; refusing to test or pull over them" >&2
    exit 1
}
if pgrep -af '[p]ython.*-m drone_emission.*(cities|render|reproduce)' >/dev/null; then
    echo "Another drone-emission city/render task is already running" >&2
    pgrep -af '[p]ython.*-m drone_emission.*(cities|render|reproduce)' >&2 || true
    exit 1
fi

REMOTE_STORAGE_ROOT="${DRONE_EMISSION_REMOTE_STORAGE_ROOT:-/mnt/minor_storage/$USER/drone-emission}"
FORMAL_DATA="${DRONE_EMISSION_FORMAL_DATA_ROOT:-$REMOTE_STORAGE_ROOT/data}"
FORMAL_CACHE="${DRONE_EMISSION_FORMAL_CACHE_ROOT:-$REMOTE_STORAGE_ROOT/cache}"
FORMAL_FIGURES="${DRONE_EMISSION_FORMAL_FIGURES_ROOT:-$PWD/figures}"
FORMAL_PAPER="$PWD/docs/paper/figures"
VALIDATION_ROOT="$(mktemp -d "$REMOTE_STORAGE_ROOT/validation.XXXXXX")"
CITY_DATA="$VALIDATION_ROOT/city/data"
CITY_CACHE="$VALIDATION_ROOT/city/cache"
CITY_WORK="$VALIDATION_ROOT/city/work"
CITY_FIGURES="$VALIDATION_ROOT/city/figures"
FIGURE_DATA="$FORMAL_DATA"
FIGURE_CACHE="$VALIDATION_ROOT/figures/cache"
FIGURE_WORK="$VALIDATION_ROOT/figures/work"
FIGURE_OUTPUT="$VALIDATION_ROOT/figures/output"
REPORTS="$VALIDATION_ROOT/reports"
mkdir -p "$CITY_DATA/city/chengdu" "$CITY_CACHE" "$CITY_WORK" "$CITY_FIGURES" \
    "$FIGURE_CACHE" "$FIGURE_WORK" "$FIGURE_OUTPUT" "$REPORTS"

inventory() {
    local output="$1"
    {
        find "$FORMAL_DATA" -maxdepth 1 -type f -printf '%p\t%s\t%T@\n'
        for level in city district; do
            if [[ -d "$FORMAL_DATA/$level" ]]; then
                find "$FORMAL_DATA/$level" -maxdepth 2 -type f -printf '%p\t%s\t%T@\n'
            fi
        done
        if [[ -d "$FORMAL_CACHE/drone_spatial_concentration" ]]; then
            find "$FORMAL_CACHE/drone_spatial_concentration" -type f -printf '%p\t%s\t%T@\n'
        fi
        for root in "$FORMAL_FIGURES" "$FORMAL_PAPER"; do
            if [[ -d "$root" ]]; then
                find "$root" -type f -printf '%p\t%s\t%T@\n'
            fi
        done
    } | LC_ALL=C sort > "$output"
}

inventory "$REPORTS/formal.before.tsv"

PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m drone_emission --help \
    > "$REPORTS/cli-help.txt"
PYTHONDONTWRITEBYTECODE=1 "$PWD/.venv/bin/python" -m pytest -p no:cacheprovider -q \
    tests/test_reproduction_contract.py > "$REPORTS/contract-tests.txt"

cp --reflink=auto --preserve=mode,timestamps "$FORMAL_DATA/combined_data.csv" "$CITY_DATA/combined_data.csv"
cp --reflink=auto --preserve=mode,timestamps "$FORMAL_DATA/gef_china.csv" "$CITY_DATA/gef_china.csv"
for source in \
    "$FORMAL_DATA/city/chengdu/chengdu_buildings.gpkg" \
    "$FORMAL_DATA/city/chengdu/chengdu_buildings.geojson" \
    "$FORMAL_DATA/city/chengdu/chengdu_roadnet.graphml"; do
    if [[ -f "$source" ]]; then
        cp --reflink=auto --preserve=mode,timestamps "$source" "$CITY_DATA/city/chengdu/"
    fi
done
[[ -f "$CITY_DATA/city/chengdu/chengdu_roadnet.graphml" ]] || {
    echo "Authoritative Chengdu road network is missing" >&2
    exit 1
}
[[ -f "$CITY_DATA/city/chengdu/chengdu_buildings.gpkg" || -f "$CITY_DATA/city/chengdu/chengdu_buildings.geojson" ]] || {
    echo "Authoritative Chengdu buildings are missing" >&2
    exit 1
}

DRONE_EMISSION_DATA_ROOT="$CITY_DATA" \
DRONE_EMISSION_CACHE_ROOT="$CITY_CACHE" \
DRONE_EMISSION_WORK_ROOT="$CITY_WORK" \
DRONE_EMISSION_FIGURES_ROOT="$CITY_FIGURES" \
    ./entrypoint.sh --jobs 1 cities --region chengdu --force --no-fetch

"$PWD/.venv/bin/python" -m drone_emission.tools.compare_city_results \
    "$FORMAL_DATA/city/chengdu" "$CITY_DATA/city/chengdu" \
    --report "$REPORTS/chengdu-comparison.json"

# Render treats the formal data tree as a read-only source. The CLI first builds
# its compatibility view under FIGURE_WORK: mutable derived inputs are copied,
# while large immutable inputs are linked for reads only. Copy the one cache
# family named by the manifest so all writable cache state remains isolated.
SPATIAL_CACHE_SOURCE="$FORMAL_CACHE/drone_spatial_concentration"
if [[ ! -d "$SPATIAL_CACHE_SOURCE" ]]; then
    # Compatibility with the existing remote cache layout; the active workflow
    # writes future results to cache, but validation must reuse what is present.
    SPATIAL_CACHE_SOURCE="$FORMAL_DATA/analysis/drone_spatial_concentration"
fi
[[ -f "$SPATIAL_CACHE_SOURCE/city_drone_spatial_concentration.csv" ]] || {
    echo "Authoritative spatial-concentration result is missing" >&2
    exit 1
}
cp -a "$SPATIAL_CACHE_SOURCE" "$FIGURE_CACHE/drone_spatial_concentration"

DRONE_EMISSION_DATA_ROOT="$FIGURE_DATA" \
DRONE_EMISSION_CACHE_ROOT="$FIGURE_CACHE" \
DRONE_EMISSION_WORK_ROOT="$FIGURE_WORK" \
DRONE_EMISSION_FIGURES_ROOT="$FIGURE_OUTPUT" \
    ./entrypoint.sh render

mapfile -t PDF_PATHS < <(
    "$PWD/.venv/bin/python" -c '
import json, sys
for figure in json.load(open(sys.argv[1], encoding="utf-8"))["figures"]:
    if figure["mode"] == "direct":
        print("paper/" + figure["id"] + ".pdf")
    else:
        for name in figure["required_subfigures"]:
            print("paper/subfigure/" + name)
' src/drone_emission/paper_figures.json
)
[[ "${#PDF_PATHS[@]}" -eq 39 ]] || { echo "Expected 39 review PDFs, found ${#PDF_PATHS[@]}" >&2; exit 1; }
: > "$REPORTS/fetch-pdfs.txt"
for relative in "${PDF_PATHS[@]}"; do
    path="$FIGURE_OUTPUT/$relative"
    [[ -s "$path" ]] || { echo "Missing rendered PDF: $path" >&2; exit 1; }
    [[ "$(head -c 5 "$path")" == "%PDF-" ]] || { echo "Invalid PDF: $path" >&2; exit 1; }
    printf '%s\n' "$path" >> "$REPORTS/fetch-pdfs.txt"
done

# Promote validates mapping/PDF completeness, not visual composition. Seed its
# isolated nine manual staging slots with valid generated panels when the paper
# subrepository is not mounted; no seeded placeholder is included in the fetch list.
while IFS=$'\t' read -r figure_id panel; do
    cp --reflink=auto --preserve=mode,timestamps \
        "$FIGURE_OUTPUT/paper/subfigure/$panel" \
        "$FIGURE_OUTPUT/paper/$figure_id.pdf"
done < <(
    "$PWD/.venv/bin/python" -c '
import json, sys
for figure in json.load(open(sys.argv[1], encoding="utf-8"))["figures"]:
    if figure["mode"] == "inkscape":
        print(figure["id"] + "\t" + figure["required_subfigures"][0])
' src/drone_emission/paper_figures.json
)
DRONE_EMISSION_DATA_ROOT="$FIGURE_DATA" \
DRONE_EMISSION_CACHE_ROOT="$FIGURE_CACHE" \
DRONE_EMISSION_WORK_ROOT="$FIGURE_WORK" \
DRONE_EMISSION_FIGURES_ROOT="$FIGURE_OUTPUT" \
    ./entrypoint.sh promote --dry-run > "$REPORTS/promote-dry-run.txt"

inventory "$REPORTS/formal.after.tsv"
cmp "$REPORTS/formal.before.tsv" "$REPORTS/formal.after.tsv"
printf '%s\n' "$VALIDATION_ROOT"
