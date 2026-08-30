#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
    printf "usage: %s <new absolute output directory>\n" "$0" >&2
    exit 2
fi
OUTPUT_DIR=$1
mkdir -- "$OUTPUT_DIR"
OUTPUT_DIR=$(cd -- "$OUTPUT_DIR" && pwd -P)
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
GPU=${FE2O3_PERF_PHYSICAL_GPU:-6}
if [[ ! $GPU =~ ^[0-9]+$ ]]; then
    printf "FE2O3_PERF_PHYSICAL_GPU must be a decimal ordinal\n" >&2
    exit 2
fi
export ROCR_VISIBLE_DEVICES=$GPU
unset HIP_VISIBLE_DEVICES
export HOSTNAME=$(hostname)
export FE2O3_GFX950_ADVANCED_HIP_ORDINAL=$GPU
export FE2O3_GFX950_ADVANCED_PERF_OUTPUT=$OUTPUT_DIR/samples.jsonl
export FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=${FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID:-gpt-oss-$(date -u +%Y%m%dT%H%M%SZ)}
export FE2O3_GFX950_ADVANCED_PERF_WARMUPS=1000
export FE2O3_GFX950_ADVANCED_PERF_BLOCKS=30
export FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=100
export FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=20
AMD_SMI=${AMD_SMI:-/opt/rocm/bin/amd-smi}
FUSED=$REPO_ROOT/examples/gfx950_gpt_oss_decode/run-gfx950.sh
UNFUSED=$REPO_ROOT/examples/gfx950_gpt_oss_decode/run-unfused-gfx950.sh
for executable in "$AMD_SMI" "$FUSED" "$UNFUSED"; do
    if [[ ! -x $executable ]]; then
        printf "required executable is unavailable: %s\n" "$executable" >&2
        exit 1
    fi
done
{
    printf "campaign_id=%s\n" "$FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID"
    printf "host=%s\n" "$HOSTNAME"
    printf "rocr_visible_devices=%s\n" "$ROCR_VISIBLE_DEVICES"
    printf "hip_visible_devices=unset\n"
    printf "gpu=%s\n" "$GPU"
    printf "protocol=5 fresh AB/BA process pairs; 1000 warmups; 30x100 samples; 20 block rewarm\n"
    printf "source_commit=%s\n" "$(git -C "$REPO_ROOT" rev-parse HEAD)"
    printf "source_tree=%s\n" "$(git -C "$REPO_ROOT" rev-parse HEAD^{tree})"
} > "$OUTPUT_DIR/campaign.txt"
"$AMD_SMI" static -g "$GPU" --json > "$OUTPUT_DIR/amd-smi-static.json"

run_one() {
    local process=$1 variant=$2 wrapper=$3 order=$4
    local prefix="$process-$order-$variant"
    export FE2O3_GFX950_ADVANCED_PERF_PROCESS=$process
    export FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=$variant
    if [[ $variant == fused-optimized ]]; then
        export FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=fe2o3-production-rust
    else
        export FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=exact-hip-stage-sequence
    fi
    "$AMD_SMI" process -g "$GPU" --json > "$OUTPUT_DIR/$prefix-process-before.json"
    "$AMD_SMI" metric -g "$GPU" -p -c -t -l -v --json > "$OUTPUT_DIR/$prefix-metric-before.json"
    "$wrapper" > "$OUTPUT_DIR/$prefix.log" 2>&1
    "$AMD_SMI" metric -g "$GPU" -p -c -t -l -v --json > "$OUTPUT_DIR/$prefix-metric-after.json"
    "$AMD_SMI" process -g "$GPU" --json > "$OUTPUT_DIR/$prefix-process-after.json"
}
for process in 0 1 2 3 4; do
    if (( process % 2 == 0 )); then
        run_one "$process" fused-optimized "$FUSED" A
        run_one "$process" exact-unfused "$UNFUSED" B
    else
        run_one "$process" exact-unfused "$UNFUSED" B
        run_one "$process" fused-optimized "$FUSED" A
    fi
done
python3 "$REPO_ROOT/perf-evidence/analyze.py" "$OUTPUT_DIR/samples.jsonl" > "$OUTPUT_DIR/series-summary.json"
python3 "$REPO_ROOT/perf-evidence/analyze-gpt-oss.py" "$OUTPUT_DIR/samples.jsonl" > "$OUTPUT_DIR/comparison.json"
(
    cd -- "$OUTPUT_DIR"
    find . -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "$OUTPUT_DIR/SHA256SUMS"
printf "PASS GPT-OSS layer-tile performance evidence: %s\n" "$OUTPUT_DIR"
