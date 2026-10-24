#!/usr/bin/env bash
set -Eeuo pipefail

# Compile the ordinary SIMT and checked-tile sources; no hardware launch path.
if [[ ${FE2O3_EXAMPLE_COMPILE_ONLY:-1} != 1 ]]; then
    printf '%s\n' 'this runner requires FE2O3_EXAMPLE_COMPILE_ONLY=1' >&2
    exit 2
fi
case ${1:-both} in
    simt) variants=(simt) ;;
    tile) variants=(tile) ;;
    both) variants=(simt tile) ;;
    *) printf 'usage: %s [simt|tile|both]\n' "$0" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    printf 'usage: %s [simt|tile|both]\n' "$0" >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
TOOLCHAIN=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
ROOT_TARGET_DIR=${FE2O3_ROOT_TARGET_DIR:-$REPO_ROOT/target}
OUTPUT_ROOT=${FE2O3_OUTPUT_DIR:-$SCRIPT_DIR/target/fe2o3-gfx950}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
RUSTUP=${RUSTUP:-rustup}
CLANG=${CLANG:-$ROCM_PATH/llvm/bin/clang}
LD_LLD=${LD_LLD:-$ROCM_PATH/llvm/bin/ld.lld}
READOBJ=${READOBJ:-$ROCM_PATH/llvm/bin/llvm-readobj}
OBJDUMP=${OBJDUMP:-$ROCM_PATH/llvm/bin/llvm-objdump}

mkdir -p -- "$ROOT_TARGET_DIR" "$OUTPUT_ROOT"
ROOT_TARGET_DIR=$(cd -- "$ROOT_TARGET_DIR" && pwd -P)
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
if [[ -n ${FE2O3_RUSTC_EXTRACTOR:-} ]]; then
    EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR
else
    # Use the checked-in normal test profile also exercised by compiler CI.
    CARGO_TARGET_DIR=$ROOT_TARGET_DIR "$RUSTUP" run "$TOOLCHAIN" cargo build \
        --locked --profile test --manifest-path "$REPO_ROOT/Cargo.toml" \
        -p rustc-codegen-fe2o3 --lib --bin fe2o3-rustc-extract
    EXTRACTOR=$ROOT_TARGET_DIR/debug/fe2o3-rustc-extract
fi
# shellcheck source=../gfx950_advanced_attention/gfx950-extractor-runtime.sh
source "$SCRIPT_DIR/../gfx950_advanced_attention/gfx950-extractor-runtime.sh"
resolve_gfx950_extractor_runtime "$EXTRACTOR"
SYSROOT=$("$RUSTUP" run "$TOOLCHAIN" rustc --print sysroot)

for variant in "${variants[@]}"; do
    ATTEMPT_DIR=$(mktemp -d "$OUTPUT_ROOT/$variant.XXXXXX")
    chmod 700 "$ATTEMPT_DIR"
    LLVM_IR=$ATTEMPT_DIR/wave64_collectives_v1.ll
    OBJECT=$ATTEMPT_DIR/wave64_collectives_v1.o
    HSACO=$ATTEMPT_DIR/wave64_collectives_v1.hsaco
    BINDING_PATH=$ATTEMPT_DIR/crate-binding-v1
    NOTES=$ATTEMPT_DIR/wave64_collectives_v1.notes
    ISA=$ATTEMPT_DIR/wave64_collectives_v1.isa
    feature_args=()
    if [[ $variant == tile ]]; then
        feature_args=(--features tile-kernel)
    fi
    (
        cd -- "$SCRIPT_DIR"
        FE2O3_EXTRACT_CRATE_V1=fe2o3_wave64_collectives_v1 \
        FE2O3_EXTRACT_CRATE_BINDING_PATH_V1=$BINDING_PATH \
        FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1=$LLVM_IR \
        RUSTC_WORKSPACE_WRAPPER=$EXTRACTOR \
        CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Zalways-encode-mir -Ctarget-cpu=gfx950 -Ctarget-feature=-wavefrontsize32,+wavefrontsize64,-xnack' \
        LD_LIBRARY_PATH="$EXTRACTOR_RUNTIME_DIR:$EXTRACTOR_DEPS_DIR:$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$RUSTUP" run "$TOOLCHAIN" cargo check --release --locked \
            -Zbuild-std=core --target amdgcn-amd-amdhsa \
            --target-dir "$ATTEMPT_DIR/amdgpu-target" \
            --no-default-features "${feature_args[@]}" --lib
    )
    if [[ ! -f $BINDING_PATH || -L $BINDING_PATH ]] ||
        [[ $(stat -c '%a:%h:%F' "$BINDING_PATH") != '600:1:regular file' ]] ||
        [[ $(wc -c < "$BINDING_PATH") -ne 65 ]]; then
        printf 'compiler did not publish a valid regular crate-binding handoff\n' >&2
        exit 1
    fi
    IFS= read -r binding < "$BINDING_PATH"
    if [[ ! $binding =~ ^[0-9a-f]{64}$ ]]; then
        printf 'compiler published a noncanonical crate-binding handoff\n' >&2
        exit 1
    fi
    if [[ ! -f $LLVM_IR || -L $LLVM_IR ]]; then
        printf 'compiler did not publish regular LLVM output\n' >&2
        exit 1
    fi
    grep -Fq 'target triple = "amdgcn-amd-amdhsa"' "$LLVM_IR"
    [[ $(grep -c '^define amdgpu_kernel ' "$LLVM_IR") -eq 1 ]]
    grep -Fq 'define amdgpu_kernel void @wave64_collectives_v1(' "$LLVM_IR"
    grep -Fq '"target-cpu"="gfx950"' "$LLVM_IR"
    grep -Fq '"target-features"="-wavefrontsize32,+wavefrontsize64,-xnack"' "$LLVM_IR"
    grep -Fq '"amdgpu-flat-work-group-size"="64,64"' "$LLVM_IR"
    "$CLANG" -O3 -nogpulib -x ir --target=amdgcn-amd-amdhsa \
        -mcpu=gfx950:xnack- -mcode-object-version=6 \
        -mllvm -amdgpu-internalize-symbols -c "$LLVM_IR" -o "$OBJECT"
    "$LD_LLD" -shared --no-undefined "$OBJECT" -o "$HSACO"
    "$READOBJ" --file-headers --notes "$HSACO" > "$NOTES"
    for marker in 'Format: elf64-amdgpu' 'Machine: EM_AMDGPU' \
        'EF_AMDGPU_MACH_AMDGCN_GFX950' 'EF_AMDGPU_FEATURE_XNACK_OFF_V4' \
        'gfx950:xnack-' '.wavefront_size: 64' '.max_flat_workgroup_size: 64'; do
        grep -Fq -- "$marker" "$NOTES"
    done
    "$OBJDUMP" --disassemble --mcpu=gfx950 "$HSACO" > "$ISA"
    grep -Fq '<wave64_collectives_v1>:' "$ISA"
    printf 'COMPILE PASS: Wave64 %s reached gfx950:xnack- HSACO; hardware execution skipped\n' "$variant"
    printf 'HSACO: %s\n' "$HSACO"
done
