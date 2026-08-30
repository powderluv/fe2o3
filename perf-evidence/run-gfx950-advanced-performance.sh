#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    printf 'usage: %s <new output directory> [advanced wrapper ...]\n' "$0" >&2
    exit 2
fi

OUTPUT_DIR=$1
shift
if [[ $OUTPUT_DIR != /* ]]; then
    printf 'output directory must be absolute\n' >&2
    exit 2
fi
mkdir -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd -P)
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
GPU=${FE2O3_PERF_PHYSICAL_GPU:-6}
if [[ ! $GPU =~ ^[0-9]+$ ]]; then
    printf 'FE2O3_PERF_PHYSICAL_GPU must be a decimal ordinal\n' >&2
    exit 2
fi
if [[ ${ROCR_VISIBLE_DEVICES:-$GPU} != "$GPU" ]]; then
    printf 'ROCR_VISIBLE_DEVICES must select only physical GPU %s\n' "$GPU" >&2
    exit 2
export HOSTNAME=${HOSTNAME:-$(hostname)}
fi
export ROCR_VISIBLE_DEVICES=$GPU
unset HIP_VISIBLE_DEVICES

if [[ $# -eq 0 ]]; then
    set -- \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-kda-decode-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-kda-prefill-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-content-sparse-attention-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-compressed-hybrid-attention-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-attnres-aggregate-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-four-branch-residual-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_attention/run-mhc-sinkhorn-mix-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-moe-route-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-moe-expert-rank-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-combine-expert-ranks-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-speculative-transaction-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-qwen-ngram-gather-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-stage-gradient-shard-gfx950.sh" \
        "$REPO_ROOT/examples/gfx950_advanced_systems/run-muon-update-gfx950.sh"
fi

CAMPAIGN_ID=${FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID:-gfx950-$(date -u +%Y%m%dT%H%M%SZ)}
export FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=$CAMPAIGN_ID
export FE2O3_GFX950_ADVANCED_PERF_OUTPUT=$OUTPUT_DIR/samples.jsonl
export FE2O3_GFX950_ADVANCED_HIP_ORDINAL=$GPU
export FE2O3_GFX950_ADVANCED_PERF_WARMUPS=${FE2O3_GFX950_ADVANCED_PERF_WARMUPS:-1000}
export FE2O3_GFX950_ADVANCED_PERF_BLOCKS=${FE2O3_GFX950_ADVANCED_PERF_BLOCKS:-30}
export FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=${FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK:-100}
export FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=${FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM:-20}
export FE2O3_GFX950_ADVANCED_PERF_PROCESS=${FE2O3_GFX950_ADVANCED_PERF_PROCESS:-0}
export FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=${FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID:-fe2o3-production-rust}
export FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=${FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID:-candidate}

AMD_SMI=${AMD_SMI:-/opt/rocm/bin/amd-smi}
if [[ ! -x $AMD_SMI ]]; then
    printf 'amd-smi is unavailable: %s\n' "$AMD_SMI" >&2
    exit 1
fi

{
    printf 'campaign_id=%s\n' "$CAMPAIGN_ID"
    printf 'host=%s\n' "$(hostname)"
    printf 'kernel=%s\n' "$(uname -a)"
    printf 'rocr_visible_devices=%s\n' "$ROCR_VISIBLE_DEVICES"
    printf 'hip_visible_devices=unset\n'
    printf 'command='
    printf '%q ' "$0" "$OUTPUT_DIR" "$@"
    printf '\n'
} > "$OUTPUT_DIR/campaign.txt"
"$AMD_SMI" static -g "$GPU" --json > "$OUTPUT_DIR/amd-smi-static.json"

index=0
for wrapper in "$@"; do
    if [[ ! -x $wrapper ]]; then
        printf 'wrapper is not executable: %s\n' "$wrapper" >&2
        exit 1
    fi
    name=$(basename -- "$wrapper" .sh)
    "$AMD_SMI" process -g "$GPU" --json > "$OUTPUT_DIR/$index-$name-process-before.json"
    "$AMD_SMI" metric -g "$GPU" -p -c -t -l -v --json > "$OUTPUT_DIR/$index-$name-metric-before.json"
    "$wrapper" > "$OUTPUT_DIR/$index-$name.log" 2>&1
    "$AMD_SMI" metric -g "$GPU" -p -c -t -l -v --json > "$OUTPUT_DIR/$index-$name-metric-after.json"
    "$AMD_SMI" process -g "$GPU" --json > "$OUTPUT_DIR/$index-$name-process-after.json"
    index=$((index + 1))
done

python3 "$REPO_ROOT/perf-evidence/analyze.py" "$OUTPUT_DIR/samples.jsonl" \
    > "$OUTPUT_DIR/summary.json"
sha256sum "$OUTPUT_DIR"/* > "$OUTPUT_DIR/SHA256SUMS"
printf 'PASS performance evidence: %s\n' "$OUTPUT_DIR"
