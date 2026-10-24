#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
if [[ ${FE2O3_EXAMPLE_COMPILE_ONLY:-0} != 0 ]]; then
    printf 'unfused HIP comparator is runtime-only; compile-only is unsupported\n' >&2
    exit 2
fi
GPU=${ROCR_VISIBLE_DEVICES:-}
if [[ -z $GPU || ! $GPU =~ ^([0-9]+|GPU-[[:xdigit:]]{16})$ ]]; then
    printf 'GPT-OSS runtime requires one explicit ROCR_VISIBLE_DEVICES UUID or ordinal\n' >&2
    exit 2
fi
unset HIP_VISIBLE_DEVICES
BUILD_DIR=$SCRIPT_DIR/target/unfused
BUNDLE=$BUILD_DIR/gpt_oss_unfused.bundle
HSACO=$BUILD_DIR/gpt_oss_unfused.hsaco
LLVM_IR=$BUILD_DIR/gpt_oss_unfused.ll
ISA=$BUILD_DIR/gpt_oss_unfused.isa
HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
BUNDLER=${CLANG_OFFLOAD_BUNDLER:-/opt/rocm/llvm/bin/clang-offload-bundler}
READOBJ=${LLVM_READOBJ:-/opt/rocm/llvm/bin/llvm-readobj}
OBJDUMP=${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}
GIT=${GIT:-$(command -v git)}
SHA256SUM=${SHA256SUM:-$(command -v sha256sum)}

for tool in "$HIPCC" "$BUNDLER" "$READOBJ" "$OBJDUMP" "$GIT" "$SHA256SUM"; do
    if [[ ! -x $tool ]]; then
        printf "required ROCm tool is unavailable: %s\n" "$tool" >&2
        exit 1
    fi
done

mkdir -p "$BUILD_DIR"
"$HIPCC" -O3 --genco --offload-arch=gfx950:xnack- \
    "$SCRIPT_DIR/gpt_oss_unfused.hip" -o "$BUNDLE"
"$HIPCC" -O3 --offload-arch=gfx950:xnack- --cuda-device-only \
    -S -emit-llvm "$SCRIPT_DIR/gpt_oss_unfused.hip" -o "$LLVM_IR"

mapfile -t targets < <("$BUNDLER" --list --type=o --input="$BUNDLE" | \
    grep "amdgcn.*gfx950:xnack-")
if [[ ${#targets[@]} -ne 1 ]]; then
    printf "expected exactly one gfx950:xnack- image, found %d\n" "${#targets[@]}" >&2
    exit 1
fi
"$BUNDLER" --unbundle --type=o --targets="${targets[0]}" \
    --input="$BUNDLE" --output="$HSACO"
"$OBJDUMP" --disassemble --mcpu=gfx950 "$HSACO" >"$ISA"

metadata=$BUILD_DIR/metadata.txt
"$READOBJ" --notes "$HSACO" >"$metadata"
if [[ $(grep -c "^    .name:           gpt_oss_unfused_" "$metadata") -ne 3 ]] ||
    ! grep -q "^    .name:           gpt_oss_unfused_router$" "$metadata" ||
    ! grep -q "^    .name:           gpt_oss_unfused_attention$" "$metadata" ||
    ! grep -q "^    .name:           gpt_oss_unfused_expert$" "$metadata" ||
    ! grep -q "amdgcn-amd-amdhsa--gfx950:xnack-" "$metadata"; then
    printf "unfused comparator metadata is not the reviewed three-kernel gfx950 profile\n" >&2
    exit 1
fi
for size in 24 40 48; do
    if ! grep -q "^    .kernarg_segment_size: $size$" "$metadata"; then
        printf "unfused comparator omitted kernarg size %s\n" "$size" >&2
        exit 1
    fi
done

export PATH=${CARGO_HOME:-/home/harmenon/.cargo}/bin:$PATH
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=$GPU
export FE2O3_RUN_GFX950_ADVANCED_HARDWARE=1
export FE2O3_GFX950_ADVANCED_HSACO=$HSACO
export FE2O3_GFX950_ADVANCED_SHA256=$($SHA256SUM "$HSACO" | cut -d " " -f 1)
export FE2O3_GFX950_ADVANCED_LLVM_SHA256=$($SHA256SUM "$LLVM_IR" | cut -d " " -f 1)
export FE2O3_GFX950_ADVANCED_ISA_SHA256=$($SHA256SUM "$ISA" | cut -d " " -f 1)
export FE2O3_GFX950_ADVANCED_CRATE_BINDING=$($SHA256SUM "$SCRIPT_DIR/gpt_oss_unfused.hip" | cut -d " " -f 1)
export FE2O3_GFX950_ADVANCED_SOURCE_COMMIT=$($GIT -C "$ROOT" rev-parse --verify 'HEAD^{commit}')
export FE2O3_GFX950_ADVANCED_SOURCE_TREE=$($GIT -C "$ROOT" rev-parse --verify 'HEAD^{tree}')
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/target}

TEST=gfx950_gpt_oss_unfused_hip_matches_cpu_reference
TEST_ARGS=(test --locked --manifest-path "$ROOT/Cargo.toml"
    -p fe2o3-hsa-runtime --features hardware-test-hooks
    --test gfx950_advanced_hardware "$TEST")
TEST_LIST=$(rustup run nightly-2026-04-03 cargo "${TEST_ARGS[@]}" -- --ignored --exact --list)
if [[ $(grep -Fxc -- "$TEST: test" <<< "$TEST_LIST" || true) -ne 1 \
    || $(grep -Ec -- ': test$' <<< "$TEST_LIST" || true) -ne 1 ]]; then
    printf 'hardware selector discovery requires exactly one test: %s\n%s\n' "$TEST" "$TEST_LIST" >&2
    exit 1
fi
rustup run nightly-2026-04-03 cargo "${TEST_ARGS[@]}" -- --ignored --exact --nocapture
