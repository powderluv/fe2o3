#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    printf 'Usage: %s [tile|simt]\n' "$0" >&2
    exit 2
fi

FE2O3_EXAMPLE_CARGO_ARGS=()
case "${1:-tile}" in
    tile) FE2O3_EXAMPLE_STEM=moe_grouped_expert_general_v1 ;;
    simt)
        FE2O3_EXAMPLE_STEM=moe_grouped_expert_general_simt_v1
        FE2O3_EXAMPLE_CARGO_ARGS=(--no-default-features --features kernel-simt-moe-expert)
        ;;
    *)
        printf 'Usage: %s [tile|simt]\n' "$0" >&2
        exit 2
        ;;
esac

FE2O3_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FE2O3_EXAMPLE_CRATE=fe2o3_moe_grouped_expert_general_v1
FE2O3_EXAMPLE_HOST_BIN=fe2o3-moe-grouped-expert-general-v1
FE2O3_EXAMPLE_HSACO_ENV=FE2O3_MOE_EXPERT_HSACO
FE2O3_EXAMPLE_LINK_DEVICE_LIBS=0

source "$FE2O3_EXAMPLE_DIR/../run-gfx942-common.sh"
fe2o3_run_gfx942
