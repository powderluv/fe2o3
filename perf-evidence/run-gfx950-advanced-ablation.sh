#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
    printf 'usage: %s <new absolute evidence directory>\n' "$0" >&2
    exit 2
fi

EVIDENCE_DIR=$1
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
COMMON=$REPO_ROOT/examples/gfx950_advanced_attention/run-gfx950.sh
TARGET_DIR=${CARGO_TARGET_DIR:-$REPO_ROOT/target}
GPU=${FE2O3_PERF_PHYSICAL_GPU:-6}
AMD_SMI=${AMD_SMI:-/opt/rocm/bin/amd-smi}
CAMPAIGN_ID=${FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID:-advanced-ablation-$(date -u +%Y%m%dT%H%M%SZ)-gpu$GPU}

: "${FE2O3_RUSTC_EXTRACTOR:?set FE2O3_RUSTC_EXTRACTOR to an extractor built from this worktree}"
if [[ $TARGET_DIR != /* || ! $GPU =~ ^[0-9]+$ ]]; then
    printf 'CARGO_TARGET_DIR must be absolute and GPU must be a decimal ordinal\n' >&2
    exit 2
fi
if [[ -e $EVIDENCE_DIR ]]; then
    printf 'refusing to reuse evidence directory: %s\n' "$EVIDENCE_DIR" >&2
    exit 1
fi
if [[ ! -x $AMD_SMI || ! -x $COMMON ]]; then
    printf 'required benchmark tool is unavailable\n' >&2
    exit 1
fi

mkdir -p -- "$EVIDENCE_DIR/artifacts" "$EVIDENCE_DIR/logs"
touch -- "$EVIDENCE_DIR/samples.jsonl"
SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')
SOURCE_TREE=$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{tree}')
{
    printf 'campaign_id=%s\n' "$CAMPAIGN_ID"
    printf 'source_commit=%s\n' "$SOURCE_COMMIT"
    printf 'source_tree=%s\n' "$SOURCE_TREE"
    printf 'host=%s\n' "$(hostname)"
    printf 'physical_gpu=%s\n' "$GPU"
    printf 'target=gfx950:xnack-\n'
    printf 'hip_visible_devices=unset\n'
    printf 'warmups=%s\n' "${FE2O3_GFX950_ADVANCED_PERF_WARMUPS:-1000}"
    printf 'blocks=%s\n' "${FE2O3_GFX950_ADVANCED_PERF_BLOCKS:-30}"
    printf 'samples_per_block=%s\n' "${FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK:-100}"
    printf 'block_rewarm=%s\n' "${FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM:-20}"
} > "$EVIDENCE_DIR/campaign.txt"
"$AMD_SMI" static -g "$GPU" --json > "$EVIDENCE_DIR/amd-smi-static-before.json"
"$AMD_SMI" process -g "$GPU" --json > "$EVIDENCE_DIR/amd-smi-process-before.json"

run_case() {
    local suite=$1 feature=$2 variant=$3 script_dir output log
    case "$suite" in
        attention) script_dir=$REPO_ROOT/examples/gfx950_advanced_attention ;;
        systems) script_dir=$REPO_ROOT/examples/gfx950_advanced_systems ;;
        gpt_oss) script_dir=$REPO_ROOT/examples/gfx950_gpt_oss_decode ;;
        *) printf 'unknown suite: %s\n' "$suite" >&2; return 2 ;;
    esac
    output=$EVIDENCE_DIR/artifacts/$suite-$feature-$variant
    log=$EVIDENCE_DIR/logs/$suite-$feature-$variant.log
    printf 'RUN suite=%s feature=%s variant=%s gpu=%s\n' "$suite" "$feature" "$variant" "$GPU"

    env -u HIP_VISIBLE_DEVICES \
        ROCR_VISIBLE_DEVICES=$GPU \
        CARGO_TARGET_DIR=$TARGET_DIR \
        FE2O3_ROOT_TARGET_DIR=$TARGET_DIR \
        FE2O3_REPO_ROOT=$REPO_ROOT \
        FE2O3_RUSTC_EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR \
        FE2O3_ADVANCED_SUITE=$suite \
        FE2O3_ADVANCED_SCRIPT_DIR=$script_dir \
        FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT=$variant \
        FE2O3_GFX950_ADVANCED_OUTPUT_DIR=$output \
        FE2O3_GFX950_PRUNE_AMDGPU_TARGET=1 \
        FE2O3_GFX950_ADVANCED_PERF_OUTPUT=$EVIDENCE_DIR/samples.jsonl \
        FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=$CAMPAIGN_ID \
        FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=fe2o3-rust \
        FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=$variant \
        FE2O3_GFX950_ADVANCED_PERF_PROCESS=0 \
        FE2O3_GFX950_ADVANCED_PERF_WARMUPS=${FE2O3_GFX950_ADVANCED_PERF_WARMUPS:-1000} \
        FE2O3_GFX950_ADVANCED_PERF_BLOCKS=${FE2O3_GFX950_ADVANCED_PERF_BLOCKS:-30} \
        FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=${FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK:-100} \
        FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=${FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM:-20} \
        "$COMMON" "$feature" > "$log" 2>&1

    test "$(find "$output" -type f -name '*.hsaco' | wc -l)" -eq 1
    test "$(find "$output" -type f -name '*.ll' | wc -l)" -eq 1
    test "$(find "$output" -type d -name amdgpu-target | wc -l)" -eq 0
    df -BG "$TARGET_DIR" | awk 'NR == 2 { sub(/G$/, "", $4); if ($4 < 10) exit 1 }'
}

# Attention: exact same-export star-shaped ablations.
run_case attention kernel-kda-decode canonical
run_case attention kernel-kda-decode-wave-tiled-v1 kda-decode-wave-tiled
run_case attention kernel-kda-prefill canonical
run_case attention kernel-kda-prefill-channel-mask-v1 kda-prefill-channel-mask
run_case attention kernel-content-sparse-attention canonical
run_case attention kernel-content-sparse-attention-reciprocal-reuse-v1 content-sparse-reciprocal
run_case attention kernel-compressed-hybrid-attention-division-baseline-v1 compressed-hybrid-division
run_case attention kernel-compressed-hybrid-attention canonical
run_case attention kernel-attnres-aggregate canonical
run_case attention kernel-attnres-aggregate-explicit-reuse-v1 attnres-explicit
run_case attention kernel-four-branch-residual canonical
run_case attention kernel-four-branch-residual-explicit-v1 four-branch-explicit
run_case attention kernel-mhc-sinkhorn-mix canonical
run_case attention kernel-mhc-sinkhorn-mix-scalar-v1 mhc-scalar

# Systems: route alternatives are retained compiler rejections, so route has a
# canonical observation only. Every other workload has one exact alternate.
run_case systems kernel-moe-route canonical
run_case systems kernel-moe-expert-rank canonical
run_case systems kernel-moe-expert-rank expert-serial
run_case systems kernel-combine-expert-ranks canonical
run_case systems kernel-speculative-transaction canonical
run_case systems kernel-speculative-transaction speculative-recompute-prefix
run_case systems kernel-qwen-ngram-gather canonical
run_case systems kernel-qwen-ngram-gather ngram-reverse-probe
run_case systems kernel-stage-gradient-shard canonical
run_case systems kernel-muon-update canonical
run_case systems kernel-muon-update muon-broadcast16

# GPT-OSS: full-kernel one-factor variants plus exact materialized components.
run_case gpt_oss kernel-gpt-oss-decode optimized
run_case gpt_oss kernel-gpt-oss-decode-router-serial serial-router
run_case gpt_oss kernel-gpt-oss-decode-held-fragments held-fragments
run_case gpt_oss kernel-gpt-oss-decode-interleaved-stores interleaved-stores
run_case gpt_oss kernel-gpt-oss-router-component materialized-router
run_case gpt_oss kernel-gpt-oss-attention-component materialized-attention
run_case gpt_oss kernel-gpt-oss-expert-component materialized-expert

python3 "$REPO_ROOT/perf-evidence/analyze.py" "$EVIDENCE_DIR/samples.jsonl" \
    --compare canonical:kda-decode-wave-tiled \
    --compare canonical:kda-prefill-channel-mask \
    --compare canonical:content-sparse-reciprocal \
    --compare compressed-hybrid-division:canonical \
    --compare canonical:attnres-explicit \
    --compare canonical:four-branch-explicit \
    --compare canonical:mhc-scalar \
    --compare canonical:expert-serial \
    --compare canonical:speculative-recompute-prefix \
    --compare canonical:ngram-reverse-probe \
    --compare canonical:muon-broadcast16 \
    --compare optimized:serial-router \
    --compare optimized:held-fragments \
    --compare optimized:interleaved-stores \
    > "$EVIDENCE_DIR/summary.json"

"$AMD_SMI" process -g "$GPU" --json > "$EVIDENCE_DIR/amd-smi-process-after.json"
"$AMD_SMI" metric -g "$GPU" -p -c -t -l -v --json > "$EVIDENCE_DIR/amd-smi-metric-after.json"
(
    cd -- "$EVIDENCE_DIR"
    find . -type f ! -name SHA256SUMS -print0 |
        sort -z |
        xargs -0 sha256sum > SHA256SUMS
)
printf 'PASS advanced gfx950 ablation campaign: %s\n' "$EVIDENCE_DIR"
