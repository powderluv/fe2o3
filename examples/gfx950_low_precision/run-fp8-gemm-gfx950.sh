#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || (${1:-tile} != tile && ${1:-tile} != simt) ]]; then
    printf 'usage: %s [tile|simt]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "$SCRIPT_DIR/run-gemm-gfx950.sh" fp8 "${1:-tile}"
