#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    printf 'usage: %s <kernel symbol> <absolute samples.jsonl>\n' "$0" >&2
    exit 2
fi
SYMBOL=$1
OUTPUT=$2
if [[ $OUTPUT != /* ]]; then
    printf 'samples path must be absolute\n' >&2
    exit 2
fi
OUTPUT_PARENT=$(dirname -- "$OUTPUT")
if [[ ! -d $OUTPUT_PARENT ]]; then
    printf 'samples parent does not exist: %s\n' "$OUTPUT_PARENT" >&2
    exit 2
fi
OUTPUT_PARENT=$(cd -- "$OUTPUT_PARENT" && pwd -P)
OUTPUT=$OUTPUT_PARENT/$(basename -- "$OUTPUT")

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
MANIFEST=$REPO_ROOT/perf-evidence/published-baseline-artifacts-v1.json
VAULT=${FE2O3_GFX950_ADVANCED_BASELINE_VAULT:-/home/harmenon/fe2o3-advanced-baseline-artifacts-20260828}
TOOLCHAIN=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
GPU=${FE2O3_PERF_PHYSICAL_GPU:-6}
export ROCR_VISIBLE_DEVICES=$GPU
unset HIP_VISIBLE_DEVICES

IFS=$'\t' read -r SUITE TEST NAMESPACE HSACO_SHA LLVM_SHA ISA_SHA SOURCE_COMMIT SOURCE_TREE < <(
export HOSTNAME=${HOSTNAME:-$(hostname)}
    python3 - "$MANIFEST" "$SYMBOL" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    manifest = json.load(source)
matches = [entry for entry in manifest["artifacts"] if entry["symbol"] == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(f"manifest has {len(matches)} entries for {sys.argv[2]}")
entry = matches[0]
values = [
    entry["suite"],
    entry["test"],
    entry["namespace"],
    entry["hsaco_sha256"],
    entry["llvm_sha256"],
    entry["isa_sha256"],
    manifest["source_commit"],
    manifest["source_tree"],
]
if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in values):
    raise SystemExit("manifest entry contains unsafe shell text")
print(*values, sep="\t")
PY
)

for field in "$NAMESPACE" "$HSACO_SHA" "$LLVM_SHA" "$ISA_SHA"; do
    [[ $field =~ ^[0-9a-f]{64}$ ]] || {
        printf 'manifest contains a malformed SHA-256 field\n' >&2
        exit 1
    }
done
for field in "$SOURCE_COMMIT" "$SOURCE_TREE"; do
    [[ $field =~ ^[0-9a-f]{40}$ ]] || {
        printf 'manifest contains a malformed Git object ID\n' >&2
        exit 1
    }
done
HSACO=$VAULT/$SUITE/$HSACO_SHA.hsaco
LLVM=$VAULT/$SUITE/$LLVM_SHA.ll
ISA=$VAULT/$SUITE/$ISA_SHA.isa
for path in "$HSACO" "$LLVM" "$ISA"; do
    if [[ ! -f $path || -L $path ]]; then
        printf 'baseline artifact is absent or unsafe: %s\n' "$path" >&2
        exit 1
    fi
    expected=$(basename -- "$path")
    expected=${expected%%.*}
    actual=$(sha256sum -- "$path" | awk '{print $1}')
    if [[ $actual != "$expected" ]]; then
        printf 'baseline artifact digest mismatch: %s\n' "$path" >&2
        exit 1
    fi
done

export FE2O3_RUN_GFX950_ADVANCED_HARDWARE=1
export FE2O3_GFX950_ADVANCED_HSACO=$HSACO
export FE2O3_GFX950_ADVANCED_SHA256=$HSACO_SHA
export FE2O3_GFX950_ADVANCED_PERF_OUTPUT=$OUTPUT
export FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID=${FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID:-gfx950-baseline-$(date -u +%Y%m%dT%H%M%SZ)}
export FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID=published-fe2o3-rust
export FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID=${FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID:-baseline}
export FE2O3_GFX950_ADVANCED_PERF_PROCESS=${FE2O3_GFX950_ADVANCED_PERF_PROCESS:-0}
export FE2O3_GFX950_ADVANCED_LLVM_SHA256=$LLVM_SHA
export FE2O3_GFX950_ADVANCED_ISA_SHA256=$ISA_SHA
export FE2O3_GFX950_ADVANCED_CRATE_BINDING=$NAMESPACE
export FE2O3_GFX950_ADVANCED_SOURCE_COMMIT=$SOURCE_COMMIT
export FE2O3_GFX950_ADVANCED_SOURCE_TREE=$SOURCE_TREE
export FE2O3_GFX950_ADVANCED_PERF_WARMUPS=${FE2O3_GFX950_ADVANCED_PERF_WARMUPS:-1000}
export FE2O3_GFX950_ADVANCED_PERF_BLOCKS=${FE2O3_GFX950_ADVANCED_PERF_BLOCKS:-30}
export FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK=${FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK:-100}
export FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM=${FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM:-20}

export PATH=/home/harmenon/.cargo/bin:$PATH
cd -- "$REPO_ROOT"
CARGO_TARGET_DIR=${FE2O3_ROOT_TARGET_DIR:-$REPO_ROOT/target} \
    rustup run "$TOOLCHAIN" cargo test --locked \
    -p fe2o3-hsa-runtime --features hardware-test-hooks \
    --test gfx950_advanced_hardware "$TEST" -- --ignored --exact --nocapture
printf 'PASS pinned baseline %s on physical GPU %s\n' "$SYMBOL" "$GPU"
