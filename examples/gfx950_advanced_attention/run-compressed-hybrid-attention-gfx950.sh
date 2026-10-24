#!/usr/bin/env bash
set -euo pipefail
if [[ $# -gt 1 ]]; then
    printf 'usage: %s [tile|simt]\n' "$0" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    case "$1" in
        tile|simt) export FE2O3_GFX950_ADVANCED_MODE=$1 ;;
        *) printf 'usage: %s [tile|simt]\n' "$0" >&2; exit 2 ;;
    esac
fi
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/run-gfx950.sh" kernel-compressed-hybrid-attention
