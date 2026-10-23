#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    printf 'Usage: %s [tile|simt]\n' "$0" >&2
    exit 2
fi
variant=${1:-tile}
FE2O3_EXAMPLE_CARGO_ARGS=()
case "$variant" in
    tile) FE2O3_EXAMPLE_STEM=gemm_autoresearch_v1 ;;
    simt)
        FE2O3_EXAMPLE_STEM=gemm_autoresearch_simt_v1
        FE2O3_EXAMPLE_CARGO_ARGS=(--no-default-features --features kernel-simt-autoresearch)
        ;;
    *) printf 'Usage: %s [tile|simt]\n' "$0" >&2; exit 2 ;;
esac
FE2O3_EXAMPLE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FE2O3_EXAMPLE_CRATE=fe2o3_gemm_autoresearch_v1
FE2O3_EXAMPLE_HOST_BIN=fe2o3-gemm-autoresearch-v1
FE2O3_EXAMPLE_HSACO_ENV=FE2O3_AUTORESEARCH_GEMM_HSACO
export FE2O3_AUTORESEARCH_GEMM_TARGET=gfx950

# Shared qualification runner for ordinary Rust kernels lowered to gfx950.

fe2o3_run_gfx950() {
    : "${FE2O3_EXAMPLE_DIR:?missing FE2O3_EXAMPLE_DIR}"
    : "${FE2O3_EXAMPLE_CRATE:?missing FE2O3_EXAMPLE_CRATE}"
    : "${FE2O3_EXAMPLE_STEM:?missing FE2O3_EXAMPLE_STEM}"
    : "${FE2O3_EXAMPLE_HOST_BIN:?missing FE2O3_EXAMPLE_HOST_BIN}"
    : "${FE2O3_EXAMPLE_HSACO_ENV:?missing FE2O3_EXAMPLE_HSACO_ENV}"

    local compile_only=${FE2O3_EXAMPLE_COMPILE_ONLY:-0}
    if [[ "$compile_only" != 0 && "$compile_only" != 1 ]]; then
        printf 'FE2O3_EXAMPLE_COMPILE_ONLY must be 0 or 1\n' >&2
        return 2
    fi

    local -a example_cargo_args=()
    if declare -p FE2O3_EXAMPLE_CARGO_ARGS &>/dev/null; then
        example_cargo_args=("${FE2O3_EXAMPLE_CARGO_ARGS[@]}")
    fi

    local repo_root toolchain root_target output_dir rocm_dir sysroot extractor
    local llvm_ir object hsaco amd_target binding_path binding compiler_input
    repo_root=$(cd -- "$FE2O3_EXAMPLE_DIR/../.." && pwd)
    toolchain=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
    root_target=${FE2O3_ROOT_TARGET_DIR:-$repo_root/target}
    output_dir=${FE2O3_OUTPUT_DIR:-$FE2O3_EXAMPLE_DIR/target/fe2o3-gfx950}
    rocm_dir=${ROCM_PATH:-/opt/rocm}
    FE2O3_RUN_WORK_DIR=$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-gfx950.XXXXXX")
    chmod 700 "$FE2O3_RUN_WORK_DIR"
    trap 'rm -rf -- "${FE2O3_RUN_WORK_DIR:?}"' EXIT

    sysroot=$(rustup run "$toolchain" rustc --print sysroot)
    extractor=${FE2O3_RUSTC_EXTRACTOR:-$root_target/debug/fe2o3-rustc-extract}
    llvm_ir="$FE2O3_RUN_WORK_DIR/$FE2O3_EXAMPLE_STEM.ll"
    object="$FE2O3_RUN_WORK_DIR/$FE2O3_EXAMPLE_STEM.o"
    hsaco="$output_dir/$FE2O3_EXAMPLE_STEM.hsaco"
    amd_target="$FE2O3_RUN_WORK_DIR/amd-target"
    binding_path="$FE2O3_RUN_WORK_DIR/crate-binding-v1"
    if [[ -e "$binding_path" || -L "$binding_path" ]]; then
        printf 'crate-binding handoff path was not attempt-fresh\n' >&2
        return 1
    fi

    mkdir -p -- "$output_dir"
    if [[ -z ${FE2O3_RUSTC_EXTRACTOR:-} ]]; then
        CARGO_TARGET_DIR="$root_target" rustup run "$toolchain" cargo build \
            --locked --manifest-path "$repo_root/Cargo.toml" \
            -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
    fi
    if [[ ! -f "$extractor" || -L "$extractor" || ! -x "$extractor" ]]; then
        printf 'production rustc extractor is not a regular executable: %s\n' "$extractor" >&2
        return 1
    fi

    (
        cd -- "$FE2O3_EXAMPLE_DIR" || exit
        FE2O3_EXTRACT_CRATE_V1="$FE2O3_EXAMPLE_CRATE" \
        FE2O3_EXTRACT_CRATE_BINDING_PATH_V1="$binding_path" \
        FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1="$llvm_ir" \
        RUSTC_WORKSPACE_WRAPPER="$extractor" \
        CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Zalways-encode-mir -Ctarget-cpu=gfx950 -Ctarget-feature=-xnack,+wavefrontsize64,-wavefrontsize32' \
        LD_LIBRARY_PATH="$root_target/debug/deps:$sysroot/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        rustup run "$toolchain" cargo check --release --locked -Zbuild-std=core \
            --target amdgcn-amd-amdhsa --target-dir "$amd_target" --lib \
            "${example_cargo_args[@]}"
    )

    if [[ ! -f "$binding_path" || -L "$binding_path" ]]; then
        printf 'compiler did not publish a regular crate-binding handoff\n' >&2
        return 1
    fi
    if [[ $(stat -c '%a:%h:%F' "$binding_path") != '600:1:regular file' ]]; then
        printf 'compiler published an insecure crate-binding handoff\n' >&2
        return 1
    fi
    if [[ $(wc -c < "$binding_path") -ne 65 ]]; then
        printf 'compiler published a malformed crate-binding handoff length\n' >&2
        return 1
    fi
    IFS= read -r binding < "$binding_path"
    if [[ ! "$binding" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'compiler published a noncanonical crate-binding handoff\n' >&2
        return 1
    fi

    compiler_input=$llvm_ir
    "$rocm_dir/llvm/bin/clang" -O3 -nogpulib -x ir \
        --target=amdgcn-amd-amdhsa -mcpu=gfx950 -mno-xnack \
        -c "$compiler_input" -o "$object"
    "$rocm_dir/llvm/bin/ld.lld" -shared "$object" -o "$hsaco"

    "$rocm_dir/llvm/bin/llvm-objdump" -d "$hsaco" > "$output_dir/$FE2O3_EXAMPLE_STEM.isa"
    "$rocm_dir/llvm/bin/llvm-readobj" --notes "$hsaco" > "$output_dir/$FE2O3_EXAMPLE_STEM.notes"
    grep -Fq '"target-cpu"="gfx950"' "$llvm_ir"
    grep -Fq '+wavefrontsize64' "$llvm_ir"
    grep -Fq 'gfx950' "$output_dir/$FE2O3_EXAMPLE_STEM.notes"
    if [[ $variant == tile ]]; then
        grep -Fq 'llvm.amdgcn.mfma.f32.16x16x16bf16.1k' "$llvm_ir"
        grep -Fq 'v_mfma_f32_16x16x16_bf16' "$output_dir/$FE2O3_EXAMPLE_STEM.isa"
    elif grep -Eq 'v_mfma_|v_wmma_' "$output_dir/$FE2O3_EXAMPLE_STEM.isa"; then
        printf 'SIMT artifact unexpectedly contains a matrix instruction\n' >&2
        return 1
    fi
    cp -- "$llvm_ir" "$output_dir/$FE2O3_EXAMPLE_STEM.ll"
    cp -- "$binding_path" "$output_dir/$FE2O3_EXAMPLE_STEM.binding"
    cp -- "$object" "$output_dir/$FE2O3_EXAMPLE_STEM.o"
    grep -Eq '^[[:space:]]*\.kernarg_segment_size:[[:space:]]+80$' "$output_dir/$FE2O3_EXAMPLE_STEM.notes"
    grep -Eq '^[[:space:]]*\.wavefront_size:[[:space:]]+64$' "$output_dir/$FE2O3_EXAMPLE_STEM.notes"
    grep -Eq '^[[:space:]]*\.max_flat_workgroup_size:[[:space:]]+64$' "$output_dir/$FE2O3_EXAMPLE_STEM.notes"
    sha256sum "$output_dir/$FE2O3_EXAMPLE_STEM.ll" "$output_dir/$FE2O3_EXAMPLE_STEM.o" "$hsaco" "$output_dir/$FE2O3_EXAMPLE_STEM.isa" "$output_dir/$FE2O3_EXAMPLE_STEM.notes" "$output_dir/$FE2O3_EXAMPLE_STEM.binding" > "$output_dir/$FE2O3_EXAMPLE_STEM.sha256"

    if [[ "$compile_only" == 1 ]]; then
        printf 'COMPILE PASS: %s reached gfx950:xnack- HSACO; hardware execution skipped\n' \
            "$FE2O3_EXAMPLE_STEM"
        printf 'HSACO: %s\n' "$hsaco"
        return 0
    fi

    (
        cd -- "$FE2O3_EXAMPLE_DIR" || exit
        env -u FE2O3_CARGO_METADATA_BUILD_OBSERVATION_V2 \
            "$FE2O3_EXAMPLE_HSACO_ENV=$hsaco" \
            "FE2O3_CRATE_BINDING_ID_V1=$binding" \
            "CARGO_TARGET_DIR=$root_target" \
            rustup run "$toolchain" cargo run --release --locked \
                --bin "$FE2O3_EXAMPLE_HOST_BIN" "${example_cargo_args[@]}"
    )

    printf 'HSACO: %s\n' "$hsaco"
}

fe2o3_run_gfx950
