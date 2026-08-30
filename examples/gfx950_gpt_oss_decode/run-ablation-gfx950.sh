#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <optimized|serial-router|held-fragments|scalar-attention|pipelined-attention|interleaved-stores|materialized|materialized-router|materialized-attention|materialized-expert>\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
COMMON=$SCRIPT_DIR/../gfx950_advanced_attention/run-gfx950.sh
VARIANT=$1
case "$VARIANT" in
    optimized) FEATURE=kernel-gpt-oss-decode ;;
    serial-router) FEATURE=kernel-gpt-oss-decode-router-serial ;;
    held-fragments) FEATURE=kernel-gpt-oss-decode-held-fragments ;;
    scalar-attention) FEATURE=kernel-gpt-oss-decode-scalar-attention ;;
    pipelined-attention) FEATURE=kernel-gpt-oss-decode-pipelined-attention ;;
    interleaved-stores) FEATURE=kernel-gpt-oss-decode-interleaved-stores ;;
    materialized)
        "$0" materialized-router
        "$0" materialized-attention
        "$0" materialized-expert
        exit 0 ;;
    materialized-router) FEATURE=kernel-gpt-oss-router-component ;;
    materialized-attention) FEATURE=kernel-gpt-oss-attention-component ;;
    materialized-expert) FEATURE=kernel-gpt-oss-expert-component ;;
    *) printf 'unknown GPT-OSS ablation variant: %s\n' "$VARIANT" >&2; exit 2 ;;
esac

GPU=${ROCR_VISIBLE_DEVICES:-5}
if [[ $GPU != 5 && $GPU != 6 ]]; then
    printf 'GPT-OSS ablations require physical GPU 5 or 6, got %s\n' "$GPU" >&2
    exit 2
fi
export ROCR_VISIBLE_DEVICES=$GPU
unset HIP_VISIBLE_DEVICES
TARGET_DIR=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
if [[ $TARGET_DIR != /* ]]; then
    printf 'CARGO_TARGET_DIR must be absolute\n' >&2
    exit 2
fi
export CARGO_TARGET_DIR=$TARGET_DIR
export FE2O3_ROOT_TARGET_DIR=$TARGET_DIR
export FE2O3_REPO_ROOT=$REPO_ROOT
export FE2O3_GFX950_ADVANCED_OUTPUT_DIR=${FE2O3_GFX950_ADVANCED_OUTPUT_DIR:-$TARGET_DIR/gpt-oss-$FEATURE}
export FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=${FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID:-$VARIANT}
FE2O3_ADVANCED_SUITE=gpt_oss FE2O3_ADVANCED_SCRIPT_DIR=$SCRIPT_DIR exec "$COMMON" "$FEATURE"
