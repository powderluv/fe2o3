#!/usr/bin/env bash
set -euo pipefail

# Builds one ordinary Rust attention kernel through the production extractor,
# validates its exact gfx950 LLVM/HSACO/ISA profile, then runs the corresponding
# digest-pinned numerical HSA test. Host paths can be overridden for SSH runs.

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s <fp4|fp8> [tile|simt]\n' "$0" >&2
    exit 2
fi

PRECISION=$1
VARIANT=${2:-tile}
if [[ $VARIANT != tile && $VARIANT != simt ]]; then
    printf 'unsupported attention selection: %s %s\n' "$PRECISION" "$VARIANT" >&2
    exit 2
fi
case $PRECISION in
    fp4)
        FEATURE=kernel-fp4-attention
        SYMBOL=gfx950_fp4_attention_rust
        DESCRIPTOR_REGEX='gfx950_fp4_attention_rust[.]kd'
        LABEL='FP4 attention'
        SELECTORS='i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)'
        WRONG_SELECTORS='i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)'
        TRANSPOSE_LLVM=llvm.amdgcn.ds.read.tr4.b64.v2i32
        WRONG_TRANSPOSE_LLVM=llvm.amdgcn.ds.read.tr8.b64.v2i32
        TRANSPOSE_ISA=ds_read_b64_tr_b4
        WRONG_TRANSPOSE_ISA=ds_read_b64_tr_b8
        TRANSPOSE_COUNT=2
        STATIC_LDS_BYTES=4096
        FUNCTION_TARGET_BINDINGS=2
        OUTPUT_ENV=FE2O3_GFX950_FP4_ATTENTION_OUTPUT_DIR
        HSACO_ENV=FE2O3_GFX950_FP4_ATTENTION_HSACO
        SHA256_ENV=FE2O3_GFX950_FP4_ATTENTION_SHA256
        HARDWARE_TEST=gfx950_fp4_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output
        ;;
    fp8)
        FEATURE=kernel-fp8-attention
        SYMBOL=gfx950_fp8_attention_rust
        DESCRIPTOR_REGEX='gfx950_fp8_attention_rust[.]kd'
        LABEL='FP8 attention'
        SELECTORS='i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)'
        WRONG_SELECTORS='i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)'
        TRANSPOSE_LLVM=llvm.amdgcn.ds.read.tr8.b64.v2i32
        WRONG_TRANSPOSE_LLVM=llvm.amdgcn.ds.read.tr4.b64.v2i32
        TRANSPOSE_ISA=ds_read_b64_tr_b8
        WRONG_TRANSPOSE_ISA=ds_read_b64_tr_b4
        TRANSPOSE_COUNT=4
        STATIC_LDS_BYTES=8192
        FUNCTION_TARGET_BINDINGS=1
        OUTPUT_ENV=FE2O3_GFX950_FP8_ATTENTION_OUTPUT_DIR
        HSACO_ENV=FE2O3_GFX950_FP8_ATTENTION_HSACO
        SHA256_ENV=FE2O3_GFX950_FP8_ATTENTION_SHA256
        HARDWARE_TEST=gfx950_fp8_attention_rust_cov6_runs_multigrid_and_matches_every_cpu_reference_output
        ;;
    *)
        printf 'unsupported attention precision: %s (expected fp4 or fp8)\n' "$PRECISION" >&2
        exit 2
        ;;
esac

OUTPUT_SUFFIX=
if [[ $VARIANT == simt ]]; then
    FEATURE=kernel-simt-$PRECISION-attention
    SYMBOL=gfx950_${PRECISION}_attention_simt
    DESCRIPTOR_REGEX="${SYMBOL}[.]kd"
    LABEL="${PRECISION^^} SIMT attention"
    STATIC_LDS_BYTES=0
    OUTPUT_SUFFIX=-simt
    HARDWARE_TEST=gfx950_${PRECISION}_attention_simt_cov6_runs_multigrid_and_matches_every_cpu_reference_output
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=${FE2O3_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}
TOOLCHAIN=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
ROOT_TARGET_DIR=${FE2O3_ROOT_TARGET_DIR:-$REPO_ROOT/target}
OUTPUT_ROOT=${!OUTPUT_ENV:-$SCRIPT_DIR/target/fe2o3-gfx950-$PRECISION-attention$OUTPUT_SUFFIX}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
RUSTUP=${RUSTUP:-rustup}
CARGO_BIN=${CARGO:-cargo}
CLANG=${CLANG:-$ROCM_PATH/llvm/bin/clang}
LD_LLD=${LD_LLD:-$ROCM_PATH/llvm/bin/ld.lld}
OBJDUMP=${OBJDUMP:-$ROCM_PATH/llvm/bin/llvm-objdump}
READOBJ=${READOBJ:-$ROCM_PATH/llvm/bin/llvm-readobj}
SHA256SUM=${SHA256SUM:-sha256sum}
COMPILE_ONLY=${FE2O3_EXAMPLE_COMPILE_ONLY:-0}

if [[ $COMPILE_ONLY != 0 && $COMPILE_ONLY != 1 ]]; then
    printf 'FE2O3_EXAMPLE_COMPILE_ONLY must be 0 or 1\n' >&2
    exit 2
fi

for executable in "$RUSTUP" "$CLANG" "$LD_LLD" "$OBJDUMP" "$READOBJ" "$SHA256SUM"; do
    if [[ ! -x $executable ]] && ! command -v -- "$executable" >/dev/null 2>&1; then
        printf 'required executable is unavailable: %s\n' "$executable" >&2
        exit 1
    fi
done
if [[ ! -f $REPO_ROOT/Cargo.toml || ! -f $SCRIPT_DIR/Cargo.toml ]]; then
    printf 'FE2O3_REPO_ROOT does not identify this fe2o3 checkout: %s\n' "$REPO_ROOT" >&2
    exit 1
fi

# SCRIPT_DIR is resolved canonically at runtime.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/gfx950-ocml-closure.sh"
validate_gfx950_ocml_closure

mkdir -p -- "$ROOT_TARGET_DIR" "$OUTPUT_ROOT"
ROOT_TARGET_DIR=$(cd -- "$ROOT_TARGET_DIR" && pwd -P)
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
ATTEMPT_DIR=$(mktemp -d "$OUTPUT_ROOT/attempt.XXXXXX")
chmod 700 "$ATTEMPT_DIR"

LLVM_IR=$ATTEMPT_DIR/gfx950-$PRECISION-attention.ll
OBJECT=$ATTEMPT_DIR/gfx950-$PRECISION-attention.o
HSACO=$ATTEMPT_DIR/gfx950-$PRECISION-attention.hsaco
DISASSEMBLY=$ATTEMPT_DIR/gfx950-$PRECISION-attention.isa
NOTES=$ATTEMPT_DIR/gfx950-$PRECISION-attention.notes
BINDING_PATH=$ATTEMPT_DIR/crate-binding-v1
AMD_TARGET_DIR=$(mktemp -d "${TMPDIR:-/tmp}/fe2o3-gfx950-attention-amdgpu.XXXXXX")
trap 'rm -rf -- "${AMD_TARGET_DIR:?}"' EXIT
chmod 700 "$AMD_TARGET_DIR"

if [[ -n ${FE2O3_RUSTC_EXTRACTOR:-} ]]; then
    EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR
else
    EXTRACTOR=$ROOT_TARGET_DIR/debug/fe2o3-rustc-extract
    CARGO_TARGET_DIR=$ROOT_TARGET_DIR \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" build \
        --locked --manifest-path "$REPO_ROOT/Cargo.toml" \
        -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
fi
if [[ ! -f $EXTRACTOR || -L $EXTRACTOR || ! -x $EXTRACTOR ]]; then
    printf 'generic rustc extractor must be a regular executable: %s\n' "$EXTRACTOR" >&2
    exit 1
fi

# Preserve a checked caller-relative override when Cargo changes directory.
EXTRACTOR_DIR=$(cd -- "$(dirname -- "$EXTRACTOR")" && pwd -P)
EXTRACTOR=$EXTRACTOR_DIR/$(basename -- "$EXTRACTOR")

SYSROOT=$("$RUSTUP" run "$TOOLCHAIN" rustc --print sysroot)
(
    cd -- "$SCRIPT_DIR"
    FE2O3_EXTRACT_CRATE_V1=fe2o3_gfx950_low_precision \
    FE2O3_EXTRACT_CRATE_BINDING_PATH_V1=$BINDING_PATH \
    FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1=$LLVM_IR \
    RUSTC_WORKSPACE_WRAPPER=$EXTRACTOR \
    CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Zalways-encode-mir -Ctarget-cpu=gfx950 -Ctarget-feature=-wavefrontsize32,+wavefrontsize64,-xnack' \
    LD_LIBRARY_PATH="$ROOT_TARGET_DIR/debug/deps:$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" check \
        --release --locked -Zbuild-std=core \
        --target amdgcn-amd-amdhsa --target-dir "$AMD_TARGET_DIR" \
        --no-default-features --features "$FEATURE" --lib
)

if [[ ! -f $BINDING_PATH || -L $BINDING_PATH ]]; then
    printf 'compiler did not publish a regular crate-binding handoff\n' >&2
    exit 1
fi
if [[ $(stat -c '%a:%h:%F' "$BINDING_PATH") != '600:1:regular file' ]]; then
    printf 'compiler published an insecure crate-binding handoff\n' >&2
    exit 1
fi
if [[ $(wc -c < "$BINDING_PATH") -ne 65 ]]; then
    printf 'compiler published a malformed crate-binding handoff length\n' >&2
    exit 1
fi
IFS= read -r CRATE_BINDING < "$BINDING_PATH"
if [[ ! $CRATE_BINDING =~ ^[0-9a-f]{64}$ ]]; then
    printf 'compiler published a noncanonical crate-binding handoff\n' >&2
    exit 1
fi

require_llvm_line_count() {
    local needle=$1
    local expected=$2
    local description=$3
    local actual
    actual=$(awk -v needle="$needle" 'index($0, needle) { count += 1 } END { print count + 0 }' "$LLVM_IR")
    if [[ $actual -ne $expected ]]; then
        printf 'LLVM validation failed: expected %s %s, found %s\n' \
            "$expected" "$description" "$actual" >&2
        exit 1
    fi
}

require_llvm_line_count 'target triple = "amdgcn-amd-amdhsa"' 1 'AMDGPU HSA target triple'
require_llvm_line_count 'define amdgpu_kernel' 1 'kernel definition'
require_llvm_line_count "define amdgpu_kernel void @$SYMBOL(" 1 "$SYMBOL definition"
if [[ $VARIANT == tile ]]; then
require_llvm_line_count '"target-cpu"="gfx950"' "$FUNCTION_TARGET_BINDINGS" \
    'gfx950 function target bindings'
require_llvm_line_count '"target-features"="-wavefrontsize32,+wavefrontsize64,-xnack"' \
    "$FUNCTION_TARGET_BINDINGS" \
    'exact Wave64/xnack- function target binding'
require_llvm_line_count \
    'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' \
    1 "scaled $LABEL MFMA call"
require_llvm_line_count "$SELECTORS" 1 "$LABEL selectors and disabled scaling controls"
require_llvm_line_count "$WRONG_SELECTORS" 0 "wrong low-precision selectors in $LABEL"
require_llvm_line_count \
    'declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' \
    1 "scaled $LABEL MFMA declaration"
require_llvm_line_count "call <2 x i32> @$TRANSPOSE_LLVM(" "$TRANSPOSE_COUNT" \
    "$LABEL transpose calls"
require_llvm_line_count "declare <2 x i32> @$TRANSPOSE_LLVM(" 1 "$LABEL transpose declaration"
require_llvm_line_count "@$WRONG_TRANSPOSE_LLVM(" 0 "wrong-width transpose references"
require_llvm_line_count 'call float @__ocml_exp_f32(float ' 4 'OCML exp_f32 calls'
require_llvm_line_count 'declare float @__ocml_exp_f32(float)' 1 'OCML exp_f32 declaration'
require_llvm_line_count "internal addrspace(3) global [$STATIC_LDS_BYTES x i8] undef, align 64" \
    1 "exact $STATIC_LDS_BYTES-byte transpose LDS allocation"
require_llvm_line_count 'call void asm sideeffect "s_barrier", ""()' 1 \
    'workgroup publication barrier'

else
KERNEL_ATTRIBUTES=$(awk -v symbol="$SYMBOL" '
    index($0, "define amdgpu_kernel void @" symbol "(") == 1 {
        definitions += 1
        if (match($0, /#[0-9]+/)) group = substr($0, RSTART, RLENGTH)
        else bad = 1
    }
    /^attributes #[0-9]+ = / { attributes[$2] = $0; groups[$2] += 1 }
    END {
        if (bad || definitions != 1 || group == "" || groups[group] != 1) exit 1
        print attributes[group]
    }
' "$LLVM_IR") || {
    printf 'LLVM validation failed: missing or ambiguous kernel attribute group\n' >&2
    exit 1
}

require_target_attribute() {
    local attribute=$1 value=$2 expected values kernel_value
    expected="\"$attribute\"=\"$value\""
    values=$(grep -oE "\"$attribute\"=\"[^\"]*\"" "$LLVM_IR" || true)
    kernel_value=$(grep -oE "\"$attribute\"=\"[^\"]*\"" <<< "$KERNEL_ATTRIBUTES" || true)
    if [[ $kernel_value != "$expected" ]] || grep -Fvxq -- "$expected" <<< "$values"; then
        printf 'LLVM validation failed: inconsistent or missing %s binding\n' "$attribute" >&2
        exit 1
    fi
}
require_target_attribute target-cpu gfx950
require_target_attribute target-features '-wavefrontsize32,+wavefrontsize64,-xnack'

    require_llvm_line_count 'llvm.amdgcn.mfma' 0 'matrix instructions in SIMT attention'
    require_llvm_line_count 'llvm.amdgcn.ds.read.tr' 0 'transpose instructions in SIMT attention'
    require_llvm_line_count 'addrspace(3) global' 0 'static LDS in SIMT attention'
    require_llvm_line_count 'call float @__ocml_exp_f32(float ' 1 'SIMT stable-softmax exp call'
    require_llvm_line_count 'declare float @__ocml_exp_f32(float)' 1 'OCML exp declaration'
fi

# The ordered bitcode list is populated only after the helper validates every
# manifest entry and the exact clang/lld digests. Disable Clang's implicit
# device-library injection so only that explicit, pinned OCML closure is linked.
"$CLANG" -O3 -nogpulib -x ir \
    --target=amdgcn-amd-amdhsa -mcpu=gfx950:xnack- \
    -mcode-object-version=6 -mllvm -amdgpu-internalize-symbols \
    "${GFX950_OCML_CLANG_ARGS[@]}" \
    -c "$LLVM_IR" -o "$OBJECT"
"$LD_LLD" -shared --no-undefined "$OBJECT" -o "$HSACO"

"$READOBJ" --file-headers --notes "$HSACO" > "$NOTES"
for required in \
    'Format: elf64-amdgpu' \
    'Machine: EM_AMDGPU' \
    'EF_AMDGPU_MACH_AMDGCN_GFX950' \
    'EF_AMDGPU_FEATURE_XNACK_OFF_V4'; do
    if ! grep -Fq -- "$required" "$NOTES"; then
        printf 'HSACO metadata validation failed: missing %s\n' "$required" >&2
        exit 1
    fi
done

require_notes_regex_count() {
    local expression=$1
    local expected=$2
    local description=$3
    local actual
    actual=$(grep -Ec -- "$expression" "$NOTES" || true)
    if [[ $actual -ne $expected ]]; then
        printf 'HSACO metadata validation failed: expected %s %s, found %s\n' \
            "$expected" "$description" "$actual" >&2
        exit 1
    fi
}

require_notes_regex_count \
    "^[[:space:]]*amdhsa.target:[[:space:]]+'amdgcn-amd-amdhsa--gfx950:xnack-'[[:space:]]*$" \
    1 'exact gfx950:xnack- target declarations'
# Argument metadata also uses `.name`; the exact kernel and descriptor lines
# below identify the single exported kernel without conflating those fields.
require_notes_regex_count "^[[:space:]]*[.]name:[[:space:]]+${SYMBOL}[[:space:]]*$" 1 \
    'exact kernel name'
require_notes_regex_count "^[[:space:]]*[.]symbol:[[:space:]]+${DESCRIPTOR_REGEX}[[:space:]]*$" 1 \
    'exact kernel descriptor symbol'
require_notes_regex_count '^[[:space:]]*[.]kernarg_segment_size:[[:space:]]+64[[:space:]]*$' 1 \
    '64-byte explicit kernarg segment'
require_notes_regex_count '^[[:space:]]*[.]kernarg_segment_align:[[:space:]]+8[[:space:]]*$' 1 \
    '8-byte metadata kernarg alignment'
require_notes_regex_count \
    "^[[:space:]]*[.]group_segment_fixed_size:[[:space:]]+${STATIC_LDS_BYTES}[[:space:]]*$" \
    1 "exact $STATIC_LDS_BYTES-byte static LDS declaration"
require_notes_regex_count '^[[:space:]]*[.]uses_dynamic_stack:[[:space:]]+false[[:space:]]*$' 1 \
    'disabled dynamic stack declarations'
require_notes_regex_count '^[[:space:]]*[.]wavefront_size:[[:space:]]+64[[:space:]]*$' 1 \
    'Wave64 metadata declarations'
if ! awk '
    /amdhsa.version:/ { version = 1; next }
    version == 1 && /^[[:space:]]*-[[:space:]]+1[[:space:]]*$/ { version = 2; next }
    version == 2 && /^[[:space:]]*-[[:space:]]+2[[:space:]]*$/ { found = 1 }
    END { exit(found ? 0 : 1) }
' "$NOTES"; then
    printf 'HSACO metadata validation failed: missing COV6 metadata version 1.2\n' >&2
    exit 1
fi

"$OBJDUMP" --disassemble --mcpu=gfx950 "$HSACO" > "$DISASSEMBLY"
KERNEL_ISA=$(awk -v marker="<$SYMBOL>:" '
    index($0, marker) { capture = 1 }
    capture && /^[[:xdigit:]]+ <[^>]+>:/ && !index($0, marker) { exit }
    capture { print }
' "$DISASSEMBLY")
if [[ -z $KERNEL_ISA ]]; then
    printf 'ISA validation failed: %s is absent\n' "$SYMBOL" >&2
    exit 1
fi

if [[ $VARIANT == tile ]]; then
MFMA_COUNT=$(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true)
ALL_MFMA_COUNT=$(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true)
if [[ $MFMA_COUNT -ne 1 || $ALL_MFMA_COUNT -ne 1 ]]; then
    printf 'ISA validation failed: expected exactly one scaled gfx950 MFMA, found %s exact/%s total\n' \
        "$MFMA_COUNT" "$ALL_MFMA_COUNT" >&2
    exit 1
fi
TRANSPOSE_ISA_COUNT=$(grep -Fc -- "$TRANSPOSE_ISA" <<< "$KERNEL_ISA" || true)
if [[ $TRANSPOSE_ISA_COUNT -ne $TRANSPOSE_COUNT ]]; then
    printf 'ISA validation failed: expected %s %s instructions, found %s\n' \
        "$TRANSPOSE_COUNT" "$TRANSPOSE_ISA" "$TRANSPOSE_ISA_COUNT" >&2
    exit 1
fi
if grep -Fq -- "$WRONG_TRANSPOSE_ISA" <<< "$KERNEL_ISA"; then
    printf 'ISA validation failed: wrong transpose instruction is present: %s\n' \
        "$WRONG_TRANSPOSE_ISA" >&2
    exit 1
fi
if [[ $PRECISION == fp4 ]]; then
    if ! grep -Fq -- 'cbsz:4 blgp:4' <<< "$KERNEL_ISA"; then
        printf 'ISA validation failed: FP4 MFMA selectors cbsz:4 blgp:4 are absent\n' >&2
        exit 1
    fi
elif grep -Fq -- 'cbsz:4' <<< "$KERNEL_ISA"; then
    printf 'ISA validation failed: FP4 MFMA selectors are present in FP8 attention\n' >&2
    exit 1
fi

TRANSPOSE_LINE=$(awk -v needle="$TRANSPOSE_ISA" 'index($0, needle) { print NR; exit }' <<< "$KERNEL_ISA")
MFMA_LINE=$(awk 'index($0, "v_mfma_f32_16x16x128_f8f6f4") { print NR; exit }' <<< "$KERNEL_ISA")
if [[ -z $TRANSPOSE_LINE || -z $MFMA_LINE || $TRANSPOSE_LINE -ge $MFMA_LINE ]]; then
    printf 'ISA validation failed: transpose must execute before scaled MFMA\n' >&2
    exit 1
fi

# Weighted-value accumulation legitimately uses scalar f32 arithmetic. Reject
# conversion/dot fallbacks that could replace the reviewed score MFMA instead.
for forbidden in v_cvt_f32_fp4 v_cvt_f32_fp8 v_dot; do
    if grep -Fq -- "$forbidden" <<< "$KERNEL_ISA"; then
        printf 'ISA validation failed: low-precision MFMA fallback is present: %s\n' \
            "$forbidden" >&2
        exit 1
    fi
done

else

    if grep -Eq -- 'v_mfma|ds_read_b64_tr_' <<< "$KERNEL_ISA"; then
        printf 'ISA validation failed: SIMT attention contains matrix/transpose instructions\n' >&2
        exit 1
    fi
    for required in global_load_ubyte v_mul_f32 v_add_f32; do
        if ! grep -Fq -- "$required" <<< "$KERNEL_ISA"; then
            printf 'ISA validation failed: SIMT attention is missing %s\n' "$required" >&2
            exit 1
        fi
    done
fi

HSACO=$(cd -- "$(dirname -- "$HSACO")" && pwd -P)/$(basename -- "$HSACO")
if [[ $COMPILE_ONLY == 1 ]]; then
    printf 'COMPILE PASS: %s reached validated gfx950:xnack- HSACO; hardware execution skipped\n' \
        "$SYMBOL"
    printf 'HSACO: %s\n' "$HSACO"
    exit 0
fi
# Reject missing or ambiguous selectors before the single numerical invocation.
TEST_LIST=$ATTEMPT_DIR/hardware-test-list.txt
(
    cd -- "$REPO_ROOT"
    CARGO_TARGET_DIR="$ROOT_TARGET_DIR" \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" test --locked \
        -p fe2o3-hsa-runtime --features hardware-qualification \
        --test gfx950_attention_hardware "$HARDWARE_TEST" -- --ignored --exact --list
) > "$TEST_LIST"
if ! grep -Fxq -- "$HARDWARE_TEST: test" "$TEST_LIST" ||
   ! grep -Fxq -- '1 test, 0 benchmarks' "$TEST_LIST"; then
    printf 'hardware selector must discover exactly one ignored test: %s\n' "$HARDWARE_TEST" >&2
    exit 1
fi

HSACO_SHA256=$("$SHA256SUM" -- "$HSACO" | awk '{ print $1 }')
(
    cd -- "$REPO_ROOT"
    env \
        FE2O3_RUN_GFX950_ATTENTION_HARDWARE=1 \
        "$HSACO_ENV=$HSACO" \
        "$SHA256_ENV=$HSACO_SHA256" \
        CARGO_TARGET_DIR="$ROOT_TARGET_DIR" \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" test --locked \
        -p fe2o3-hsa-runtime --features hardware-qualification \
        --test gfx950_attention_hardware "$HARDWARE_TEST" \
        -- --ignored --exact --nocapture
)

printf 'PASS fe2o3 gfx950 %s production build and numerical run\n' "$LABEL"
printf 'OCML:   %s\n' "$GFX950_OCML_DEVICE_LIBRARY_DIR"
printf 'LLVM:   %s\n' "$LLVM_IR"
printf 'HSACO:  %s\n' "$HSACO"
printf 'SHA256: %s\n' "$HSACO_SHA256"
printf 'ISA:    %s\n' "$DISASSEMBLY"
