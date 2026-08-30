#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
TARGET_DIR=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
GPU=${ROCR_VISIBLE_DEVICES:-4}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
EVIDENCE_DIR=${FE2O3_GFX950_SYSTEMS_ABLATION_EVIDENCE_DIR:?set an absolute ablation evidence directory}
RECORDS=$EVIDENCE_DIR/samples.jsonl

CURRENT_OUTPUT=
cleanup() {
    if [[ -n $CURRENT_OUTPUT && -d $CURRENT_OUTPUT ]]; then
        find "$CURRENT_OUTPUT" -type d -name amdgpu-target -prune -exec rm -rf -- {} +
    fi
}
trap cleanup EXIT

: "${FE2O3_RUSTC_EXTRACTOR:?set FE2O3_RUSTC_EXTRACTOR to a worktree-bound extractor copy}"
if [[ $TARGET_DIR != /* || $EVIDENCE_DIR != /* ]]; then
    printf 'target and evidence directories must be absolute\n' >&2
    exit 2
fi
mkdir -p -- "$EVIDENCE_DIR/artifacts" "$EVIDENCE_DIR/logs"
if [[ -e $RECORDS ]]; then
    printf 'refusing to append to existing evidence: %s\n' "$RECORDS" >&2
    exit 1
fi
touch -- "$RECORDS"

DEFAULT_SPECS=(
    kernel-moe-route:canonical
    kernel-moe-expert-rank:canonical
    kernel-moe-expert-rank:expert-serial
    kernel-combine-expert-ranks:canonical
    kernel-combine-expert-ranks:combine-transposed
    kernel-speculative-transaction:canonical
    kernel-speculative-transaction:speculative-recompute-prefix
    kernel-qwen-ngram-gather:canonical
    kernel-qwen-ngram-gather:ngram-reverse-probe
    kernel-stage-gradient-shard:canonical
    kernel-stage-gradient-shard:stage-tile4
    kernel-muon-update:canonical
    kernel-muon-update:muon-broadcast16
)

if (( $# )); then
    SPECS=("$@")
else
    SPECS=("${DEFAULT_SPECS[@]}")
fi

for spec in "${SPECS[@]}"; do
    feature=${spec%%:*}
    variant=${spec#*:}
    output=$EVIDENCE_DIR/artifacts/$feature-$variant
    CURRENT_OUTPUT=$output
    log=$EVIDENCE_DIR/logs/$feature-$variant.log
    printf 'RUN %s %s on physical GPU %s\n' "$feature" "$variant" "$GPU"
    env -u HIP_VISIBLE_DEVICES \
        ROCR_VISIBLE_DEVICES=$GPU \
        CARGO_TARGET_DIR=$TARGET_DIR \
        FE2O3_ROOT_TARGET_DIR=$TARGET_DIR \
        FE2O3_REPO_ROOT=$REPO_ROOT \
        FE2O3_RUSTC_EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR \
        FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=$variant \
        FE2O3_GFX950_ADVANCED_OUTPUT_DIR=$output \
        FE2O3_GFX950_PRUNE_AMDGPU_TARGET=1 \
        FE2O3_GFX950_ADVANCED_PERF_OUTPUT=$RECORDS \
        FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=${FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID:-systems-ablation-$STAMP-gpu$GPU} \
        FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=fe2o3-rust \
        FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=$variant \
        FE2O3_GFX950_ADVANCED_PERF_PROCESS=${FE2O3_GFX950_ADVANCED_PERF_PROCESS:-0} \
        FE2O3_GFX950_ADVANCED_PERF_WARMUPS=${FE2O3_GFX950_ADVANCED_PERF_WARMUPS:-50} \
        FE2O3_GFX950_ADVANCED_PERF_BLOCKS=${FE2O3_GFX950_ADVANCED_PERF_BLOCKS:-2} \
        FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=${FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK:-50} \
        FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=${FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM:-10} \
        "$SCRIPT_DIR/run-gfx950.sh" "$feature" 2>&1 | tee "$log"
    test "$(find "$output" -type f -name '*.hsaco' | wc -l)" -eq 1
    test "$(find "$output" -type f -name '*.ll' | wc -l)" -eq 1
    test "$(find "$output" -type d -name amdgpu-target | wc -l)" -eq 0
    CURRENT_OUTPUT=
    df -BG "$TARGET_DIR" | awk 'NR == 2 { sub(/G$/, "", $4); if ($4 < 10) exit 1 }'
done

printf 'PASS all systems ablations; evidence: %s\n' "$EVIDENCE_DIR"
