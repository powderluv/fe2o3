#!/usr/bin/env bash
set -euo pipefail
# Build and explicitly opt into existing HSA numerical qualification.
# This path grants no production Worker V3/protected execution authority.
if [[ $# -gt 1 ]]; then printf 'usage: %s [simt|tile]\n' "$0" >&2; exit 2; fi
MODE=${1:-simt}
case "$MODE" in simt) FEATURES=gfx950-target,qualification-legacy-hip ;; tile) FEATURES=gfx950-target,structured-tile,qualification-legacy-hip ;; *) printf 'unsupported mode: %s\n' "$MODE" >&2; exit 2 ;; esac
COMPILE_ONLY=${FE2O3_EXAMPLE_COMPILE_ONLY:-0}
if [[ $COMPILE_ONLY != 0 && $COMPILE_ONLY != 1 ]]; then
    printf 'FE2O3_EXAMPLE_COMPILE_ONLY must be 0 or 1\n' >&2; exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
STEM=vecadd
CRATE=fe2o3_vecadd
TOOLCHAIN=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
CLANG=$ROCM_PATH/llvm/bin/clang
LD_LLD=$ROCM_PATH/llvm/bin/ld.lld
SHA256SUM=sha256sum
OUTPUT_ROOT=${FE2O3_OUTPUT_DIR:-$SCRIPT_DIR/target/fe2o3-gfx950-$MODE}
: "${FE2O3_RUSTC_EXTRACTOR:?set the prebuilt production extractor path}"
EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR
if [[ ! -f $EXTRACTOR || -L $EXTRACTOR || ! -x $EXTRACTOR ]]; then
    printf 'extractor must be a regular executable: %s\n' "$EXTRACTOR" >&2; exit 1
fi
EXTRACTOR_DIR=$(cd -- "$(dirname -- "$EXTRACTOR")" && pwd -P)
SYSROOT=$(rustup run "$TOOLCHAIN" rustc --print sysroot)
source "$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-closure.sh"
FE2O3_GFX950_OCML_MANIFEST=${FE2O3_GFX950_OCML_MANIFEST:-$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.4.manifest} validate_gfx950_ocml_closure
OCML_ARGS=()
# F32 addition has no OCML calls; no gfx942 library selector is linked.
mkdir -p -- "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
ATTEMPT=$(mktemp -d "$OUTPUT_ROOT/attempt.XXXXXX")
chmod 700 "$ATTEMPT"
LLVM_IR=$ATTEMPT/$STEM.ll
BINDING_PATH=$ATTEMPT/crate-binding-v1
OBJECT=$ATTEMPT/$STEM.o
HSACO=$ATTEMPT/$STEM.hsaco
# Retain every attempt, including compiler diagnostics and intermediate artifacts.
git -C "$REPO_ROOT" rev-parse HEAD 'HEAD^{tree}' > "$ATTEMPT/source-pins.txt"
git -C "$REPO_ROOT" diff --binary > "$ATTEMPT/source-diff.patch"
(
    cd -- "$SCRIPT_DIR"
    FE2O3_EXTRACT_CRATE_V1=$CRATE \
    FE2O3_EXTRACT_CRATE_BINDING_PATH_V1=$BINDING_PATH \
    FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1=$LLVM_IR \
    RUSTC_WORKSPACE_WRAPPER=$EXTRACTOR \
    CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Zalways-encode-mir -Ctarget-cpu=gfx950 -Ctarget-feature=-wavefrontsize32,+wavefrontsize64,-xnack' \
    LD_LIBRARY_PATH="$EXTRACTOR_DIR:$EXTRACTOR_DIR/deps:$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    rustup run "$TOOLCHAIN" cargo check --release --locked --offline -Zbuild-std=core \
        --target amdgcn-amd-amdhsa --target-dir "$ATTEMPT/amdgpu-target" \
        --no-default-features --features "$FEATURES" --lib
)
if [[ ! -f $BINDING_PATH || -L $BINDING_PATH ]] ||
   [[ $(stat -c '%a:%h:%F' "$BINDING_PATH") != '600:1:regular file' ]] ||
   [[ $(wc -c < "$BINDING_PATH") -ne 65 ]]; then
    printf 'compiler did not publish a secure canonical crate binding\n' >&2; exit 1
fi
IFS= read -r BINDING < "$BINDING_PATH"
[[ $BINDING =~ ^[0-9a-f]{64}$ ]] || exit 1
rg -q '"target-cpu"="gfx950"' "$LLVM_IR"
rg -q '"target-features"="-wavefrontsize32,\+wavefrontsize64,-xnack"' "$LLVM_IR"
"$CLANG" -O3 -nogpulib -x ir --target=amdgcn-amd-amdhsa -mcpu=gfx950:xnack- \
    "${OCML_ARGS[@]}" -c "$LLVM_IR" -o "$OBJECT"
"$LD_LLD" -shared "$OBJECT" -o "$HSACO"
"$ROCM_PATH/llvm/bin/llvm-readobj" --file-headers --notes "$HSACO" > "$ATTEMPT/$STEM.notes"
rg -q 'EF_AMDGPU_MACH_AMDGCN_GFX950' "$ATTEMPT/$STEM.notes"
rg -q 'EF_AMDGPU_FEATURE_XNACK_OFF_V4' "$ATTEMPT/$STEM.notes"
rg -q "amdhsa.target:.*amdgcn-amd-amdhsa--gfx950:xnack-" "$ATTEMPT/$STEM.notes"
rg -q '^[[:space:]]*\.kernarg_segment_size:[[:space:]]+48$' "$ATTEMPT/$STEM.notes"
rg -q '^[[:space:]]*\.wavefront_size:[[:space:]]+64$' "$ATTEMPT/$STEM.notes"
rg -q '^[[:space:]]*\.max_flat_workgroup_size:[[:space:]]+256$' "$ATTEMPT/$STEM.notes"
rg -q "^[[:space:]]*\.name:[[:space:]]+$STEM$" "$ATTEMPT/$STEM.notes"
"$ROCM_PATH/llvm/bin/llvm-objdump" --disassemble --mcpu=gfx950 "$HSACO" > "$ATTEMPT/$STEM.isa"
sha256sum "$LLVM_IR" "$HSACO" "$ATTEMPT/$STEM.notes" "$ATTEMPT/$STEM.isa" "$BINDING_PATH" > "$ATTEMPT/artifact-sha256.txt"
if [[ $COMPILE_ONLY == 1 ]]; then
    printf 'COMPILE PASS: %s %s reached gfx950:xnack- HSACO; hardware execution skipped\nHSACO: %s\n' "$STEM" "$MODE" "$HSACO"
    exit 0
fi
# Use the existing real HSA numerical qualification fixture. Binding-only CLI
# host tests deliberately disable HIP and are not a GPU execution route.
case "$MODE" in
    simt) SELECTOR=gfx950_vecadd_rust_cov6_matches_cpu_reference ;;
    tile) SELECTOR=gfx950_vecadd_tile_rust_cov6_matches_cpu_reference ;;
esac
(
    cd -- "$REPO_ROOT"
    env -u RUSTC -u RUSTDOC -u RUSTC_WRAPPER -u RUSTC_WORKSPACE_WRAPPER \
        -u RUSTFLAGS -u CARGO_ENCODED_RUSTFLAGS -u FE2O3_HIP_SYS_DISABLE \
        -u FE2O3_CARGO_METADATA_BUILD_OBSERVATION_V2 -u FE2O3_CRATE_BINDING_ID_V1 \
        LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        FE2O3_RUN_GFX950_ADVANCED_HARDWARE=1 \
        FE2O3_GFX950_ADVANCED_HSACO="$HSACO" \
        FE2O3_GFX950_ADVANCED_SHA256="$(sha256sum "$HSACO" | cut -d ' ' -f1)" \
        FE2O3_GFX950_ADVANCED_WORKGROUP_X=256 \
        FE2O3_GFX950_ADVANCED_LLVM_SHA256="$(sha256sum "$LLVM_IR" | cut -d ' ' -f1)" \
        FE2O3_GFX950_ADVANCED_ISA_SHA256="$(sha256sum "$ATTEMPT/$STEM.isa" | cut -d ' ' -f1)" \
        FE2O3_GFX950_ADVANCED_CRATE_BINDING="$BINDING" \
        FE2O3_GFX950_ADVANCED_SOURCE_COMMIT="$(git rev-parse HEAD)" \
        FE2O3_GFX950_ADVANCED_SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')" \
        CARGO_TARGET_DIR="${FE2O3_ROOT_TARGET_DIR:-$REPO_ROOT/target}" \
        rustup run "$TOOLCHAIN" cargo test --locked --offline -j2 \
            -p fe2o3-hsa-runtime --features hardware-qualification \
            --test gfx950_advanced_hardware "$SELECTOR" -- --ignored --exact --nocapture
)
printf 'HSACO: %s\n' "$HSACO"
