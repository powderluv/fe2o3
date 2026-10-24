#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
    printf "usage: %s [simt|tile]\n" "$0" >&2
    exit 2
fi
MODE=${1:-${FE2O3_GFX950_ADVANCED_MODE:-simt}}
case "$MODE" in
    simt|tile) ;;
    *) printf "unsupported execution mode: %s\n" "$MODE" >&2; exit 2 ;;
esac
export FE2O3_GFX950_ADVANCED_MODE=$MODE
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/run-gfx950.sh" kernel-stage-gradient-shard
