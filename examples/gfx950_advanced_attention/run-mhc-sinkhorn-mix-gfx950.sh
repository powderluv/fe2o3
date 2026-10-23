#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
    printf 'usage: %s [simt|tile]\n' "$0" >&2
    exit 2
fi
case ${1:-} in
    '') ;;
    simt|tile) export FE2O3_GFX950_ADVANCED_MODE=$1 ;;
    *) printf 'mode must be simt or tile\n' >&2; exit 2 ;;
esac
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/run-gfx950.sh" kernel-mhc-sinkhorn-mix
