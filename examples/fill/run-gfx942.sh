#!/usr/bin/env bash
set -Eeuo pipefail

# A compile entry point, not a hardware launch adapter.
if [[ ${FE2O3_EXAMPLE_COMPILE_ONLY:-1} != 1 ]]; then
    printf '%s\n' 'this runner requires FE2O3_EXAMPLE_COMPILE_ONLY=1' >&2
    exit 2
fi
FE2O3_EXAMPLE_COMPILE_ONLY=1
FE2O3_EXAMPLE_CARGO_ARGS=()
if [[ $# -gt 1 || ( $# -eq 1 && ${1:-} != structured-tile ) ]]; then
    printf 'Usage: %s [structured-tile]\n' "$0" >&2
    exit 2
fi
if (( $# == 1 )); then
    FE2O3_EXAMPLE_CARGO_ARGS=(--no-default-features --features structured-tile)
fi
FE2O3_EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FE2O3_EXAMPLE_CRATE=fe2o3_fill
FE2O3_EXAMPLE_STEM=fill
FE2O3_EXAMPLE_HOST_BIN=fe2o3-fill
FE2O3_EXAMPLE_HSACO_ENV=FE2O3_FILL_HSACO
FE2O3_EXAMPLE_LINK_DEVICE_LIBS=0
source "$FE2O3_EXAMPLE_DIR/../run-gfx942-common.sh"
fe2o3_run_gfx942
