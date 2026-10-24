#!/usr/bin/env bash
set -euo pipefail
REQUESTED_COMPILE_ONLY=${FE2O3_EXAMPLE_COMPILE_ONLY:-0}
if [[ $REQUESTED_COMPILE_ONLY != 0 && $REQUESTED_COMPILE_ONLY != 1 ]]; then
    printf 'FE2O3_EXAMPLE_COMPILE_ONLY must be 0 or 1\n' >&2; exit 2
fi
FE2O3_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
FE2O3_EXAMPLE_CRATE=fe2o3_scalar_gemm_v1
FE2O3_EXAMPLE_STEM=scalar_gemm_v1
FE2O3_EXAMPLE_HOST_BIN=fe2o3-scalar-gemm-v1
FE2O3_EXAMPLE_HSACO_ENV=FE2O3_SCALAR_GEMM_HSACO
FE2O3_EXAMPLE_LINK_DEVICE_LIBS=0
FE2O3_EXAMPLE_CARGO_ARGS=(--features qualification-legacy-hip)
source "$FE2O3_EXAMPLE_DIR/../run-gfx942-common.sh"
FE2O3_EXAMPLE_COMPILE_ONLY=1 fe2o3_run_gfx942
OUTPUT=${FE2O3_OUTPUT_DIR:-$FE2O3_EXAMPLE_DIR/target/fe2o3-gfx942}
cp -- "$FE2O3_RUN_WORK_DIR/crate-binding-v1" "$OUTPUT/$FE2O3_EXAMPLE_STEM.binding"
cp -- "$FE2O3_RUN_WORK_DIR/$FE2O3_EXAMPLE_STEM.ll" "$OUTPUT/$FE2O3_EXAMPLE_STEM.ll"
git -C "$FE2O3_EXAMPLE_DIR" rev-parse HEAD 'HEAD^{tree}' > "$OUTPUT/$FE2O3_EXAMPLE_STEM.source-pins"
if [[ $REQUESTED_COMPILE_ONLY == 1 ]]; then exit 0; fi
IFS= read -r CRATE_BINDING < "$OUTPUT/$FE2O3_EXAMPLE_STEM.binding"
[[ $CRATE_BINDING =~ ^[0-9a-f]{64}$ ]] || exit 1
HSACO=$OUTPUT/$FE2O3_EXAMPLE_STEM.hsaco
ROCM_DIR=${ROCM_PATH:-/opt/rocm}
"$ROCM_DIR/llvm/bin/llvm-readobj" --file-headers --notes "$HSACO" > "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q 'EF_AMDGPU_MACH_AMDGCN_GFX942' "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q 'EF_AMDGPU_FEATURE_XNACK_OFF_V4' "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q "amdhsa.target:.*amdgcn-amd-amdhsa--gfx942:xnack-" "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q '^[[:space:]]*\.kernarg_segment_size:[[:space:]]+60$' "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q '^[[:space:]]*\.wavefront_size:[[:space:]]+64$' "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q '^[[:space:]]*\.max_flat_workgroup_size:[[:space:]]+256$' "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
rg -q "^[[:space:]]*\.name:[[:space:]]+$FE2O3_EXAMPLE_STEM$" "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes"
sha256sum "$HSACO" "$OUTPUT/$FE2O3_EXAMPLE_STEM.notes" > "$OUTPUT/$FE2O3_EXAMPLE_STEM.artifact-sha256.txt"
(
    cd -- "$FE2O3_EXAMPLE_DIR"
    env -u FE2O3_CARGO_METADATA_BUILD_OBSERVATION_V2 \
        "FE2O3_SCALAR_GEMM_HSACO=$HSACO" \
        "FE2O3_CRATE_BINDING_ID_V1=$CRATE_BINDING" \
        "CARGO_TARGET_DIR=${FE2O3_ROOT_TARGET_DIR:-$FE2O3_EXAMPLE_DIR/../../target}" \
        CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_DEV_OPT_LEVEL=0 \
        CARGO_PROFILE_DEV_DEBUG_ASSERTIONS=true CARGO_PROFILE_DEV_OVERFLOW_CHECKS=true \
        rustup run "${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}" cargo run --locked --offline \
            --features qualification-legacy-hip --bin "$FE2O3_EXAMPLE_HOST_BIN"
)
