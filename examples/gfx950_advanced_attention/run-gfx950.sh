#!/usr/bin/env bash
set -euo pipefail

# Production Rust -> gfx950 HSACO -> HSA numerical verification for one advanced
# kernel. The systems dispatcher reuses this fail-closed implementation.

if [[ $# -ne 1 ]]; then
    printf 'usage: %s <kernel feature>\n' "$0" >&2
    exit 2
fi

FEATURE=$1
SUITE=${FE2O3_ADVANCED_SUITE:-attention}
COMMON_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_DIR=${FE2O3_ADVANCED_SCRIPT_DIR:-$COMMON_DIR}

SYSTEMS_ABLATION=${FE2O3_GFX950_SYSTEMS_ABLATION_VARIANT:-canonical}
EXTRA_FEATURE=
if [[ $SUITE == systems ]]; then
    case "$FEATURE:$SYSTEMS_ABLATION" in
        kernel-moe-route:canonical|kernel-moe-expert-rank:canonical|\
        kernel-combine-expert-ranks:canonical|kernel-speculative-transaction:canonical|\
        kernel-qwen-ngram-gather:canonical|kernel-stage-gradient-shard:canonical|\
        kernel-muon-update:canonical) ;;
        kernel-moe-expert-rank:expert-serial) EXTRA_FEATURE=ablation-expert-serial ;;
        kernel-combine-expert-ranks:combine-transposed)
            EXTRA_FEATURE=ablation-combine-transposed ;;
        kernel-speculative-transaction:speculative-recompute-prefix)
            EXTRA_FEATURE=ablation-speculative-recompute-prefix ;;
        kernel-qwen-ngram-gather:ngram-reverse-probe)
            EXTRA_FEATURE=ablation-ngram-reverse-probe ;;
        kernel-stage-gradient-shard:stage-tile4) EXTRA_FEATURE=ablation-stage-tile4 ;;
        kernel-muon-update:muon-broadcast16) EXTRA_FEATURE=ablation-muon-broadcast16 ;;
        *)
            printf 'unsupported systems ablation pairing: %s:%s\n' \
                "$FEATURE" "$SYSTEMS_ABLATION" >&2
            exit 2 ;;
    esac
fi
BUILD_FEATURES=$FEATURE${EXTRA_FEATURE:+,$EXTRA_FEATURE}

case "$SUITE:$FEATURE" in
    attention:kernel-kda-decode)
        SYMBOL=gfx950_kda_gdn_decode; KERNARG=96; WG=64; LDS=0; OCML=1
        TEST=gfx950_kda_gdn_decode_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-kda-decode-wave-tiled-v1)
        SYMBOL=gfx950_kda_gdn_decode; KERNARG=96; WG=64; LDS=0; OCML=1
        TEST=gfx950_kda_gdn_decode_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-kda-prefill)
        SYMBOL=gfx950_kda_gdn_prefill; KERNARG=112; WG=64; LDS=0; OCML=1
        TEST=gfx950_kda_gdn_prefill_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-kda-prefill-channel-mask-v1)
        SYMBOL=gfx950_kda_gdn_prefill; KERNARG=112; WG=64; LDS=0; OCML=1
        TEST=gfx950_kda_gdn_prefill_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-content-sparse-attention)
        SYMBOL=gfx950_content_sparse_attention; KERNARG=96; WG=64; LDS=2048; OCML=1
        TEST=gfx950_content_sparse_attention_rust_cov6_matches_cpu_reference; ISA=fp8_attention ;;
    attention:kernel-content-sparse-attention-reciprocal-reuse-v1)
        SYMBOL=gfx950_content_sparse_attention; KERNARG=96; WG=64; LDS=2048; OCML=1
        TEST=gfx950_content_sparse_attention_rust_cov6_matches_cpu_reference; ISA=fp8_attention ;;
    attention:kernel-compressed-hybrid-attention)
        SYMBOL=gfx950_compressed_hybrid_attention; KERNARG=80; WG=64; LDS=2048; OCML=1
        TEST=gfx950_compressed_hybrid_attention_rust_cov6_matches_cpu_reference; ISA=fp8_attention ;;
    attention:kernel-compressed-hybrid-attention-division-baseline-v1)
        SYMBOL=gfx950_compressed_hybrid_attention; KERNARG=80; WG=64; LDS=2048; OCML=1
        TEST=gfx950_compressed_hybrid_attention_rust_cov6_matches_cpu_reference; ISA=fp8_attention ;;
    attention:kernel-attnres-aggregate)
        SYMBOL=gfx950_attnres_aggregate; KERNARG=48; WG=64; LDS=0; OCML=1
        TEST=gfx950_attnres_aggregate_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-attnres-aggregate-explicit-reuse-v1)
        SYMBOL=gfx950_attnres_aggregate; KERNARG=48; WG=64; LDS=0; OCML=1
        TEST=gfx950_attnres_aggregate_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-four-branch-residual)
        SYMBOL=gfx950_four_branch_residual; KERNARG=64; WG=64; LDS=0; OCML=1
        TEST=gfx950_four_branch_residual_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-four-branch-residual-explicit-v1)
        SYMBOL=gfx950_four_branch_residual; KERNARG=64; WG=64; LDS=0; OCML=1
        TEST=gfx950_four_branch_residual_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-mhc-sinkhorn-mix)
        SYMBOL=gfx950_mhc_sinkhorn_mix; KERNARG=48; WG=64; LDS=0; OCML=1
        TEST=gfx950_mhc_sinkhorn_mix_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    attention:kernel-mhc-sinkhorn-mix-scalar-v1)
        SYMBOL=gfx950_mhc_sinkhorn_mix; KERNARG=48; WG=64; LDS=0; OCML=1
        TEST=gfx950_mhc_sinkhorn_mix_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-moe-route)
        SYMBOL=gfx950_moe_route_fp4_t16_e4_k2_v1; KERNARG=96; WG=256; LDS=0; OCML=1
        TEST=gfx950_moe_route_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-moe-expert-rank)
        SYMBOL=gfx950_moe_expert_rank_fp4_fp8_v1; KERNARG=88; WG=64; LDS=0; OCML=1
        TEST=gfx950_moe_expert_rank_rust_cov6_matches_cpu_reference; ISA=mixed_expert ;;
    systems:kernel-combine-expert-ranks)
        SYMBOL=gfx950_combine_expert_ranks_v1; KERNARG=48; WG=256; LDS=0; OCML=0
        TEST=gfx950_combine_expert_ranks_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-speculative-transaction)
        SYMBOL=gfx950_speculative_transaction_v1; KERNARG=144; WG=64; LDS=0; OCML=0
        TEST=gfx950_speculative_transaction_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-qwen-ngram-gather)
        SYMBOL=gfx950_qwen_ngram_gather_v1; KERNARG=96; WG=64; LDS=0; OCML=0
        TEST=gfx950_qwen_ngram_gather_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-stage-gradient-shard)
        SYMBOL=gfx950_stage_gradient_shard_v1; KERNARG=32; WG=64; LDS=0; OCML=0
        TEST=gfx950_stage_gradient_shard_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    systems:kernel-muon-update)
        SYMBOL=gfx950_muon_update_4x4_v1; KERNARG=48; WG=64; LDS=0; OCML=0
        TEST=gfx950_muon_update_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    gpt_oss:kernel-gpt-oss-decode|gpt_oss:kernel-gpt-oss-decode-router-serial|gpt_oss:kernel-gpt-oss-decode-held-fragments|gpt_oss:kernel-gpt-oss-decode-interleaved-stores)
        SYMBOL=gfx950_gpt_oss_120b_decode_megakernel_v1; KERNARG=208; WG=64; LDS=0; OCML=1
        TEST=gfx950_gpt_oss_layer_tile_rust_cov6_matches_cpu_reference; ISA=gpt_oss ;;
    gpt_oss:kernel-gpt-oss-decode-scalar-attention)
        SYMBOL=gfx950_gpt_oss_120b_decode_megakernel_v1; KERNARG=208; WG=64; LDS=0; OCML=1
        TEST=gfx950_gpt_oss_layer_tile_rust_cov6_matches_cpu_reference; ISA=gpt_oss_scalar_attention ;;
    gpt_oss:kernel-gpt-oss-decode-pipelined-attention)
        SYMBOL=gfx950_gpt_oss_120b_decode_megakernel_v1; KERNARG=208; WG=64; LDS=2048; OCML=1
        TEST=gfx950_gpt_oss_pipelined_attention_rust_cov6_matches_cpu_reference; ISA=gpt_oss ;;
    gpt_oss:kernel-gpt-oss-router-component)
        SYMBOL=gfx950_gpt_oss_120b_router_v1; KERNARG=48; WG=64; LDS=0; OCML=0
        TEST=gfx950_gpt_oss_router_component_rust_cov6_matches_cpu_reference; ISA=scalar ;;
    gpt_oss:kernel-gpt-oss-attention-component)
        SYMBOL=gfx950_gpt_oss_120b_attention_v1; KERNARG=80; WG=64; LDS=0; OCML=1
        TEST=gfx950_gpt_oss_attention_component_rust_cov6_matches_cpu_reference; ISA=gpt_oss_attention ;;
    gpt_oss:kernel-gpt-oss-expert-component)
        SYMBOL=gfx950_gpt_oss_120b_expert_v1; KERNARG=96; WG=64; LDS=0; OCML=0
        TEST=gfx950_gpt_oss_expert_component_rust_cov6_matches_cpu_reference; ISA=gpt_oss_expert ;;
    *)
        printf 'unsupported %s kernel feature: %s\n' "$SUITE" "$FEATURE" >&2
        exit 2 ;;
esac

if [[ $SUITE == attention ]]; then
    CRATE=fe2o3_gfx950_advanced_attention
elif [[ $SUITE == systems ]]; then
    CRATE=fe2o3_gfx950_advanced_systems
else
    CRATE=fe2o3_gfx950_gpt_oss_decode
fi

REPO_ROOT=${FE2O3_REPO_ROOT:-$(cd -- "$COMMON_DIR/../.." && pwd -P)}
TOOLCHAIN=${FE2O3_RUST_TOOLCHAIN:-nightly-2026-04-03}
ROOT_TARGET_DIR=${FE2O3_ROOT_TARGET_DIR:-$REPO_ROOT/target}
OUTPUT_SUFFIX=$FEATURE
if [[ $SUITE == systems && $SYSTEMS_ABLATION != canonical ]]; then
    OUTPUT_SUFFIX=$FEATURE-$SYSTEMS_ABLATION
fi
OUTPUT_ROOT=${FE2O3_GFX950_ADVANCED_OUTPUT_DIR:-$SCRIPT_DIR/target/fe2o3-$OUTPUT_SUFFIX}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
RUSTUP=${RUSTUP:-rustup}
CARGO_BIN=${CARGO:-cargo}
CLANG=${CLANG:-$ROCM_PATH/llvm/bin/clang}
LD_LLD=${LD_LLD:-$ROCM_PATH/llvm/bin/ld.lld}
OBJDUMP=${OBJDUMP:-$ROCM_PATH/llvm/bin/llvm-objdump}
READOBJ=${READOBJ:-$ROCM_PATH/llvm/bin/llvm-readobj}
SHA256SUM=${SHA256SUM:-sha256sum}
GIT=${GIT:-git}

if ! command -v -- "$RUSTUP" >/dev/null 2>&1 && [[ -x $HOME/.cargo/bin/rustup ]]; then
    RUSTUP=$HOME/.cargo/bin/rustup
fi
if ! command -v -- "$CARGO_BIN" >/dev/null 2>&1 && [[ -x $HOME/.cargo/bin/cargo ]]; then
    CARGO_BIN=$HOME/.cargo/bin/cargo
fi
for executable in "$RUSTUP" "$CARGO_BIN" "$CLANG" "$LD_LLD" "$OBJDUMP" "$READOBJ" "$SHA256SUM" "$GIT"; do
    if [[ ! -x $executable ]] && ! command -v -- "$executable" >/dev/null 2>&1; then
        printf 'required executable is unavailable: %s\n' "$executable" >&2
        exit 1
    fi
done
if [[ ! -f $REPO_ROOT/Cargo.toml || ! -f $SCRIPT_DIR/Cargo.toml ]]; then
    printf 'advanced checkout is incomplete: repo=%s suite=%s\n' "$REPO_ROOT" "$SCRIPT_DIR" >&2
    exit 1
fi

OCML_ARGS=()
OCML_HELPER=$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-closure.sh
OCML_MANIFEST=$REPO_ROOT/examples/gfx950_low_precision/gfx950-ocml-rocm-7.2.1.manifest
if [[ ! -f $OCML_HELPER || ! -f $OCML_MANIFEST ]]; then
    printf 'reviewed gfx950 ROCm closure is incomplete\n' >&2
    exit 1
fi
# The closure pins Clang/LLD for every finalization and the device libraries
# used only by kernels with an admitted exp call.
# shellcheck disable=SC1090
source "$OCML_HELPER"
FE2O3_GFX950_OCML_MANIFEST=$OCML_MANIFEST validate_gfx950_ocml_closure
if [[ $OCML -eq 1 ]]; then
    OCML_ARGS=("${GFX950_OCML_CLANG_ARGS[@]}")
fi

mkdir -p -- "$ROOT_TARGET_DIR" "$OUTPUT_ROOT"
ROOT_TARGET_DIR=$(cd -- "$ROOT_TARGET_DIR" && pwd -P)
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
ATTEMPT_DIR=$(mktemp -d "$OUTPUT_ROOT/attempt.XXXXXX")
chmod 700 "$ATTEMPT_DIR"
LLVM_IR=$ATTEMPT_DIR/$FEATURE.ll
OBJECT=$ATTEMPT_DIR/$FEATURE.o
HSACO=$ATTEMPT_DIR/$FEATURE.hsaco
NOTES=$ATTEMPT_DIR/$FEATURE.notes
DISASSEMBLY=$ATTEMPT_DIR/$FEATURE.isa
BINDING_PATH=$ATTEMPT_DIR/crate-binding-v1
AMD_TARGET_DIR=$ATTEMPT_DIR/amdgpu-target

cleanup_amdgpu_target() {
    if [[ ${FE2O3_GFX950_PRUNE_AMDGPU_TARGET:-0} == 1 ]]; then
        rm -rf -- "$AMD_TARGET_DIR"
    fi
}
trap cleanup_amdgpu_target EXIT

if [[ -n ${FE2O3_RUSTC_EXTRACTOR:-} ]]; then
    EXTRACTOR=$FE2O3_RUSTC_EXTRACTOR
else
    EXTRACTOR=$ROOT_TARGET_DIR/debug/fe2o3-rustc-extract
    CARGO_TARGET_DIR=$ROOT_TARGET_DIR "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" build \
        --locked --manifest-path "$REPO_ROOT/Cargo.toml" \
        -p rustc-codegen-fe2o3 --bin fe2o3-rustc-extract
fi
# shellcheck disable=SC1091
source "$COMMON_DIR/gfx950-extractor-runtime.sh"
resolve_gfx950_extractor_runtime "$EXTRACTOR"

SYSROOT=$("$RUSTUP" run "$TOOLCHAIN" rustc --print sysroot)
(
    cd -- "$SCRIPT_DIR"
    FE2O3_EXTRACT_CRATE_V1=$CRATE \
    FE2O3_EXTRACT_CRATE_BINDING_PATH_V1=$BINDING_PATH \
    FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1=$LLVM_IR \
    RUSTC_WORKSPACE_WRAPPER=$EXTRACTOR \
    CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS='-Zalways-encode-mir -Ctarget-cpu=gfx950 -Ctarget-feature=-wavefrontsize32,+wavefrontsize64,-xnack' \
    LD_LIBRARY_PATH="$EXTRACTOR_RUNTIME_DIR:$EXTRACTOR_DEPS_DIR:$SYSROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" check --release --locked \
        -Zbuild-std=core --target amdgcn-amd-amdhsa --target-dir "$AMD_TARGET_DIR" \
        --no-default-features --features "$BUILD_FEATURES" --lib
)

if [[ ! -f $BINDING_PATH || -L $BINDING_PATH ]] ||
    [[ $(stat -c '%a:%h:%F' "$BINDING_PATH") != '600:1:regular file' ]] ||
    [[ $(wc -c < "$BINDING_PATH") -ne 65 ]]; then
    printf 'compiler published an absent, insecure, or malformed crate-binding handoff\n' >&2
    exit 1
fi
IFS= read -r CRATE_BINDING < "$BINDING_PATH"
if [[ ! $CRATE_BINDING =~ ^[0-9a-f]{64}$ ]]; then
    printf 'compiler published a noncanonical crate-binding handoff\n' >&2
    exit 1
fi

require_count() {
    local file=$1 needle=$2 expected=$3 description=$4 actual
    actual=$(awk -v needle="$needle" 'index($0, needle) { count++ } END { print count + 0 }' "$file")
    if [[ $actual -ne $expected ]]; then
        printf 'validation failed: expected %s %s, found %s\n' "$expected" "$description" "$actual" >&2
        exit 1
    fi
}
require_regex_count() {
    local file=$1 expression=$2 expected=$3 description=$4 actual
    actual=$(grep -Ec -- "$expression" "$file" || true)
    if [[ $actual -ne $expected ]]; then
        printf 'validation failed: expected %s %s, found %s\n' "$expected" "$description" "$actual" >&2
        exit 1
    fi
}

require_count "$LLVM_IR" 'target triple = "amdgcn-amd-amdhsa"' 1 'AMDGPU HSA target triple'
require_count "$LLVM_IR" 'define amdgpu_kernel' 1 'kernel definition'
require_count "$LLVM_IR" "define amdgpu_kernel void @$SYMBOL(" 1 "$SYMBOL definition"
require_count "$LLVM_IR" '"target-cpu"="gfx950"' 1 'gfx950 function target binding'
require_count "$LLVM_IR" '"target-features"="-wavefrontsize32,+wavefrontsize64,-xnack"' 1 'Wave64/xnack- target binding'
require_count "$LLVM_IR" '@llvm.experimental.constrained.sqrt.f32' 0 'gfx950-incompatible constrained sqrt references'
if [[ $ISA == gpt_oss ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(' 4 'BF16 MFMA calls'
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 4 'FP4 MFMA calls'
    require_count "$LLVM_IR" 'i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)' 4 'FP4 selectors and disabled scaling controls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references with pretransposed KV cache'
elif [[ $ISA == gpt_oss_scalar_attention ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(' 0 'unexpected BF16 MFMA calls'
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 4 'FP4 MFMA calls'
    require_count "$LLVM_IR" 'i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)' 4 'FP4 selectors and disabled scaling controls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references'
elif [[ $ISA == gpt_oss_attention ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(' 4 'BF16 MFMA calls'
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 0 'unexpected FP4 MFMA calls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references'
elif [[ $ISA == gpt_oss_expert ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(' 0 'unexpected BF16 MFMA calls'
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 4 'FP4 MFMA calls'
    require_count "$LLVM_IR" 'i32 4, i32 4, i32 0, i32 0, i32 0, i32 0)' 4 'FP4 selectors and disabled scaling controls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references'
elif [[ $ISA == fp8_attention ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 1 'FP8 MFMA call'
    require_count "$LLVM_IR" 'i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)' 1 'FP8 selectors and disabled scaling controls'
    require_count "$LLVM_IR" 'i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)' 0 'mixed FP4/FP8 selectors in FP8 attention'
    require_count "$LLVM_IR" 'call <2 x i32> @llvm.amdgcn.ds.read.tr8.b64.v2i32(' 4 'B8 transpose calls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 5 'total transpose references (one declaration and four B8 calls)'
elif [[ $ISA == mixed_expert ]]; then
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(' 3 'mixed FP4/FP8 MFMA calls'
    require_count "$LLVM_IR" 'i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)' 3 'mixed FP4/FP8 selectors and disabled scaling controls'
    require_count "$LLVM_IR" 'i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)' 0 'FP8-by-FP8 selectors in the mixed expert kernel'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references'
else
    require_count "$LLVM_IR" 'call <4 x float> @llvm.amdgcn.mfma.scale.' 0 'unexpected scaled MFMA calls'
    require_count "$LLVM_IR" '@llvm.amdgcn.ds.read.tr' 0 'unexpected transpose references'
fi
if [[ $OCML -eq 1 ]]; then
    if ! grep -Fq -- 'call float @__ocml_exp_f32(' "$LLVM_IR"; then
        printf 'validation failed: %s has no reviewed OCML exp call\n' "$SYMBOL" >&2
        exit 1
    fi
    require_count "$LLVM_IR" 'declare float @__ocml_exp_f32(float)' 1 'reviewed OCML exp declaration'
    UNREVIEWED_OCML=$(grep -Eo -- '@__ocml_[A-Za-z0-9_.$-]+' "$LLVM_IR" | \
        sort -u | grep -Fxv -- '@__ocml_exp_f32' || true)
    if [[ -n $UNREVIEWED_OCML ]]; then
        printf 'validation failed: %s imports an unreviewed OCML function\n' "$SYMBOL" >&2
        exit 1
    fi
else
    require_count "$LLVM_IR" '@__ocml_' 0 'unexpected OCML references'
fi

"$CLANG" -O3 -nogpulib -x ir --target=amdgcn-amd-amdhsa -mcpu=gfx950:xnack- \
    -mcode-object-version=6 -mllvm -amdgpu-internalize-symbols \
    "${OCML_ARGS[@]}" -c "$LLVM_IR" -o "$OBJECT"
"$LD_LLD" -shared --no-undefined "$OBJECT" -o "$HSACO"
"$READOBJ" --file-headers --notes "$HSACO" > "$NOTES"

for required in 'Format: elf64-amdgpu' 'Machine: EM_AMDGPU' \
    'EF_AMDGPU_MACH_AMDGCN_GFX950' 'EF_AMDGPU_FEATURE_XNACK_OFF_V4'; do
    if ! grep -Fq -- "$required" "$NOTES"; then
        printf 'HSACO metadata is missing %s\n' "$required" >&2
        exit 1
    fi
done
require_regex_count "$NOTES" "^[[:space:]]*amdhsa.target:[[:space:]]+'amdgcn-amd-amdhsa--gfx950:xnack-'[[:space:]]*$" 1 'gfx950:xnack- target'
require_regex_count "$NOTES" "^[[:space:]]*[.]name:[[:space:]]+${SYMBOL}[[:space:]]*$" 1 'kernel name'
require_regex_count "$NOTES" "^[[:space:]]*[.]symbol:[[:space:]]+${SYMBOL}[.]kd[[:space:]]*$" 1 'kernel descriptor'
require_regex_count "$NOTES" "^[[:space:]]*[.]kernarg_segment_size:[[:space:]]+${KERNARG}[[:space:]]*$" 1 'kernarg size'
require_regex_count "$NOTES" '^[[:space:]]*[.]kernarg_segment_align:[[:space:]]+8[[:space:]]*$' 1 'kernarg alignment'
require_regex_count "$NOTES" "^[[:space:]]*[.]group_segment_fixed_size:[[:space:]]+${LDS}[[:space:]]*$" 1 'static LDS size'
require_regex_count "$NOTES" '^[[:space:]]*[.]wavefront_size:[[:space:]]+64[[:space:]]*$' 1 'Wave64 metadata'
if ! awk -v expected_x="$WG" '
    /^[[:space:]]*[.]reqd_workgroup_size:[[:space:]]*$/ { state=1; next }
    state == 1 { if ($1 == "-" && $2 == expected_x) state=2; else state=0; next }
    state == 2 { if ($1 == "-" && $2 == 1) state=3; else state=0; next }
    state == 3 { if ($1 == "-" && $2 == 1) matches++; state=0 }
    END { exit(matches == 1 ? 0 : 1) }
' "$NOTES"; then
    printf 'HSACO metadata is missing the exact [%s, 1, 1] required workgroup size\n' "$WG" >&2
    exit 1
fi
require_regex_count "$NOTES" "^[[:space:]]*[.]max_flat_workgroup_size:[[:space:]]+${WG}[[:space:]]*$" 1 'maximum workgroup size'
require_regex_count "$NOTES" '^[[:space:]]*[.]uses_dynamic_stack:[[:space:]]+false[[:space:]]*$' 1 'disabled dynamic stack'
if ! awk '/amdhsa.version:/ { v=1; next } v==1 && /^[[:space:]]*-[[:space:]]+1[[:space:]]*$/ { v=2; next } v==2 && /^[[:space:]]*-[[:space:]]+2[[:space:]]*$/ { ok=1 } END { exit(ok ? 0 : 1) }' "$NOTES"; then
    printf 'HSACO metadata is missing exact COV6 version 1.2\n' >&2
    exit 1
fi

"$OBJDUMP" --disassemble --mcpu=gfx950 "$HSACO" > "$DISASSEMBLY"
KERNEL_ISA=$(awk -v marker="<$SYMBOL>:" 'index($0, marker) { capture=1 } capture && /^[[:xdigit:]]+ <[^>]+>:/ && !index($0, marker) { exit } capture { print }' "$DISASSEMBLY")
if [[ -z $KERNEL_ISA ]]; then
    printf 'ISA validation failed: %s is absent\n' "$SYMBOL" >&2
    exit 1
fi
if [[ $ISA == gpt_oss ]]; then
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x16_bf16' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four BF16 MFMAs\n' >&2; exit 1; }
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four FP4 MFMAs\n' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 8 ]] || { printf 'ISA validation failed: unexpected additional MFMA\n' >&2; exit 1; }
    [[ $(grep -Fc -- 'cbsz:4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected cbsz:4 on all FP4 MFMAs\n' >&2; exit 1; }
    ! grep -Eq -- 'ds_read_b64_tr_b[[:digit:]]+' <<< "$KERNEL_ISA" || { printf 'ISA validation failed: pretransposed KV path emitted a transpose instruction\n' >&2; exit 1; }
elif [[ $ISA == gpt_oss_scalar_attention ]]; then
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x16_bf16' <<< "$KERNEL_ISA" || true) -eq 0 ]] || { printf 'ISA validation failed: scalar attention emitted BF16 MFMA
' >&2; exit 1; }
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four FP4 MFMAs
' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: unexpected additional MFMA
' >&2; exit 1; }
    [[ $(grep -Fc -- 'cbsz:4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected cbsz:4 on all FP4 MFMAs
' >&2; exit 1; }
elif [[ $ISA == gpt_oss_attention ]]; then
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x16_bf16' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four BF16 MFMAs
' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: unexpected additional MFMA
' >&2; exit 1; }
elif [[ $ISA == gpt_oss_expert ]]; then
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four FP4 MFMAs
' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: unexpected additional MFMA
' >&2; exit 1; }
    [[ $(grep -Fc -- 'cbsz:4' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected cbsz:4 on all FP4 MFMAs
' >&2; exit 1; }
elif [[ $ISA == fp8_attention ]]; then
    [[ $(grep -Fc -- 'ds_read_b64_tr_b8' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four B8 transpose instructions\n' >&2; exit 1; }
    [[ $(grep -Ec -- 'ds_read_b64_tr_b[[:digit:]]+' <<< "$KERNEL_ISA" || true) -eq 4 ]] || { printf 'ISA validation failed: expected exactly four total transpose instructions\n' >&2; exit 1; }
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true) -eq 1 ]] || { printf 'ISA validation failed: expected exactly one FP8 scaled MFMA\n' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 1 ]] || { printf 'ISA validation failed: unexpected additional MFMA\n' >&2; exit 1; }
    ! grep -Fq -- 'cbsz:4' <<< "$KERNEL_ISA" || { printf 'ISA validation failed: FP4 selector present in FP8 attention\n' >&2; exit 1; }
    TR_LINE=$(awk 'index($0,"ds_read_b64_tr_b8") { print NR; exit }' <<< "$KERNEL_ISA")
    MFMA_LINE=$(awk 'index($0,"v_mfma_f32_16x16x128_f8f6f4") { print NR; exit }' <<< "$KERNEL_ISA")
    [[ $TR_LINE -lt $MFMA_LINE ]] || { printf 'ISA validation failed: transpose does not precede MFMA\n' >&2; exit 1; }
elif [[ $ISA == mixed_expert ]]; then
    [[ $(grep -Fc -- 'v_mfma_f32_16x16x128_f8f6f4' <<< "$KERNEL_ISA" || true) -eq 3 ]] || { printf 'ISA validation failed: expected exactly three mixed MFMAs\n' >&2; exit 1; }
    [[ $(grep -c -- 'v_mfma_' <<< "$KERNEL_ISA" || true) -eq 3 ]] || { printf 'ISA validation failed: unexpected additional MFMA\n' >&2; exit 1; }
    [[ $(grep -Fc -- 'cbsz:4' <<< "$KERNEL_ISA" || true) -eq 3 ]] || { printf 'ISA validation failed: expected cbsz:4 on all three mixed MFMAs\n' >&2; exit 1; }
    ! grep -Eq -- 'ds_read_b64_tr_b[[:digit:]]+' <<< "$KERNEL_ISA" || { printf 'ISA validation failed: unexpected transpose instruction in mixed expert kernel\n' >&2; exit 1; }
else
    ! grep -Eq -- 'v_mfma_|ds_read_b64_tr_b[[:digit:]]+' <<< "$KERNEL_ISA" || { printf 'ISA validation failed: unexpected tensor/transpose instruction in scalar kernel\n' >&2; exit 1; }
fi
for forbidden in v_cvt_f32_fp4 v_cvt_f32_fp8 v_dot; do
    ! grep -Fq -- "$forbidden" <<< "$KERNEL_ISA" || { printf 'ISA validation failed: low-precision fallback is present: %s\n' "$forbidden" >&2; exit 1; }
done

HSACO=$(cd -- "$(dirname -- "$HSACO")" && pwd -P)/$(basename -- "$HSACO")
HSACO_SHA256=$("$SHA256SUM" -- "$HSACO" | awk '{ print $1 }')
LLVM_SHA256=$("$SHA256SUM" -- "$LLVM_IR" | awk '{ print $1 }')
ISA_SHA256=$("$SHA256SUM" -- "$DISASSEMBLY" | awk '{ print $1 }')
SOURCE_COMMIT=$("$GIT" -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')
SOURCE_TREE=$("$GIT" -C "$REPO_ROOT" rev-parse --verify 'HEAD^{tree}')
(
    cd -- "$REPO_ROOT"
    FE2O3_RUN_GFX950_ADVANCED_HARDWARE=1 \
    FE2O3_GFX950_ADVANCED_HSACO=$HSACO \
    FE2O3_GFX950_ADVANCED_SHA256=$HSACO_SHA256 \
    FE2O3_GFX950_ADVANCED_LLVM_SHA256=$LLVM_SHA256 \
    FE2O3_GFX950_ADVANCED_ISA_SHA256=$ISA_SHA256 \
    FE2O3_GFX950_ADVANCED_CRATE_BINDING=$CRATE_BINDING \
    FE2O3_GFX950_ADVANCED_SOURCE_COMMIT=$SOURCE_COMMIT \
    FE2O3_GFX950_ADVANCED_SOURCE_TREE=$SOURCE_TREE \
    CARGO_TARGET_DIR=$ROOT_TARGET_DIR \
        "$RUSTUP" run "$TOOLCHAIN" "$CARGO_BIN" test --locked \
        -p fe2o3-hsa-runtime --features hardware-test-hooks \
        --test gfx950_advanced_hardware "$TEST" -- --ignored --exact --nocapture
)

printf 'PASS %s production Rust gfx950 build and numerical run\n' "$SYMBOL"
printf 'Binding: %s\nLLVM:   %s\nHSACO:  %s\nSHA256: %s\nISA:    %s\n' \
    "$CRATE_BINDING" "$LLVM_IR" "$HSACO" "$HSACO_SHA256" "$DISASSEMBLY"
