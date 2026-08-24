#!/usr/bin/env bash
# Remote wrapper only: tmux lifetime, parallelism and memory limits. No SSH.
set -euo pipefail

MEMORY_MAX="${DRONE_EMISSION_MEMORY_MAX:-64G}"
SESSION_NAME="${DRONE_EMISSION_SESSION_NAME:-drone-emission}"
PARALLEL="${1:-1}"
ENVIRONMENT="${2:-remote}"
shift $(( $# >= 2 ? 2 : $# ))
cd "$(dirname "$0")"

if [[ -z "${TMUX:-}" && -z "${INSIDE_ENTRYPOINT:-}" ]]; then
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        exec tmux attach -t "${SESSION_NAME}"
    fi
    tmux new-session -d -s "${SESSION_NAME}" \
        "INSIDE_ENTRYPOINT=1 $PWD/entrypoint.sh '$PARALLEL' '$ENVIRONMENT' $*"
    exec tmux attach -t "${SESSION_NAME}"
fi

PYTHON="${DRONE_EMISSION_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found: ${PYTHON}" >&2
    exit 1
fi

if (( PARALLEL <= 1 )); then
    exec systemd-run --user --scope -p "MemoryMax=${MEMORY_MAX}" \
        "${PYTHON}" -m drone_emission --env "${ENVIRONMENT}" cities "$@"
fi

if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required for parallel city selection" >&2
    exit 1
fi

jq -r 'to_entries[] | select(.value.region_level == "city") | .key' regions.json \
    | xargs -P "${PARALLEL}" -I '{}' \
        systemd-run --user --scope -p "MemoryMax=${MEMORY_MAX}" \
        "${PYTHON}" -m drone_emission --env "${ENVIRONMENT}" cities --region '{}' "$@"
