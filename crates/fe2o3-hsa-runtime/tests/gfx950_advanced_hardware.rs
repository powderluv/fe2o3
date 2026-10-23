//! Non-authoritative gfx950 advanced-kernel hardware and numerical harness.
//!
//! Each ignored test consumes one caller-supplied, digest-pinned COV6 HSACO.
//! It validates the exact single-kernel ABI before using the reviewed raw HSA
//! adapter. This bypasses protected publication authority and exists only to
//! test Rust-produced artifacts on a directly visible gfx950 device.

#[cfg(feature = "hardware-test-hooks")]
use fe2o3_amd_target::FeatureState;
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_artifacts::{DigestAlgorithm, PayloadDigest};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_core::GpuContext;
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_host::{
    HsaKernelResolutionObservationV1, HsaLaunchGeometryV1, ReviewedHsaExecutableLifecycleAdapterV1,
    ReviewedHsaImplicitKernargAdapterV1,
};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_hsa_runtime::{
    ReviewedHsaExecutableV1, ReviewedHsaHardwareTestBufferV1, ReviewedHsaKernelV1,
    ReviewedHsaRuntimeAdapterV1,
};
#[cfg(feature = "hardware-test-hooks")]
use fe2o3_hsaco::{CodeObjectVersion, ExplicitValueKind};
#[cfg(feature = "hardware-test-hooks")]
use serde_json::json;
#[cfg(feature = "hardware-test-hooks")]
use sha2::{Digest, Sha256};
#[cfg(feature = "hardware-test-hooks")]
use std::io::Write;

#[cfg(feature = "hardware-test-hooks")]
const CHANNELS_V1: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const KDA_KEY_DIMENSION_V1: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const KDA_VALUE_DIMENSION_V1: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const KDA_STATE_ELEMENTS_V1: usize = KDA_KEY_DIMENSION_V1 * KDA_VALUE_DIMENSION_V1;
#[cfg(feature = "hardware-test-hooks")]
const KDA_CHUNK_TOKENS_V1: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const PREFILL_TOKENS_V1: usize = 8;
#[cfg(feature = "hardware-test-hooks")]
const ATTENTION_TOKENS_V1: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const HEAD_DIMENSION_V1: usize = 128;
#[cfg(feature = "hardware-test-hooks")]
const TOKENS_PER_BLOCK_V1: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const SELECTED_BLOCKS_V1: usize = 2;
#[cfg(feature = "hardware-test-hooks")]
const SELECTED_TOKENS_V1: usize = 3;
#[cfg(feature = "hardware-test-hooks")]
const DEEPSEEK_SPARSE_TOP_K_V1: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const MIXING_STREAMS_V1: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const SINKHORN_ITERATIONS_V1: usize = 3;
#[cfg(feature = "hardware-test-hooks")]
const MULTIGRID_WORKGROUPS_V1: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const MULTIGRID_SUBGROUP_BATCHES_V1: usize = 64;
#[cfg(feature = "hardware-test-hooks")]
const MULTIGRID_WAVE_BATCHES_V1: usize = 16;

#[cfg(feature = "hardware-test-hooks")]
const TOKENS: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const HIDDEN: usize = 128;
#[cfg(feature = "hardware-test-hooks")]
const OUTPUT: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const EXPERTS: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const ALL_EXPERTS: usize = 5;
#[cfg(feature = "hardware-test-hooks")]
const TOP_K: usize = 2;
#[cfg(feature = "hardware-test-hooks")]
const DISPATCH_CAPACITY: usize = TOKENS * TOP_K;
#[cfg(feature = "hardware-test-hooks")]
const CANDIDATES: usize = 8;
#[cfg(feature = "hardware-test-hooks")]
const DRAFT_STEPS: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const STATE_WIDTH: usize = 8;
#[cfg(feature = "hardware-test-hooks")]
const QUERIES: usize = 8;
#[cfg(feature = "hardware-test-hooks")]
const NGRAM: usize = 3;
#[cfg(feature = "hardware-test-hooks")]
const TABLE_SIZE: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const MUON_DIM: usize = 4;
#[cfg(feature = "hardware-test-hooks")]
const MUON_ELEMENTS: usize = MUON_DIM * MUON_DIM;
#[cfg(feature = "hardware-test-hooks")]
const GRADIENT_SHARDS: usize = 2;
#[cfg(feature = "hardware-test-hooks")]
const MUON_ITERATIONS: usize = 5;
#[cfg(feature = "hardware-test-hooks")]
const MUON_LEARNING_RATE: f32 = 0.05;
#[cfg(feature = "hardware-test-hooks")]
const SYSTEM_BATCHES: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const COMBINE_BATCHES: usize = 4;

#[cfg(feature = "hardware-test-hooks")]
mod dimensions {
    pub(super) use super::{
        ALL_EXPERTS, ATTENTION_TOKENS_V1, CANDIDATES, CHANNELS_V1, DEEPSEEK_SPARSE_TOP_K_V1,
        DISPATCH_CAPACITY, DRAFT_STEPS, EXPERTS, GRADIENT_SHARDS, HEAD_DIMENSION_V1, HIDDEN,
        KDA_CHUNK_TOKENS_V1, KDA_KEY_DIMENSION_V1, KDA_STATE_ELEMENTS_V1, KDA_VALUE_DIMENSION_V1,
        MIXING_STREAMS_V1, MUON_DIM, MUON_ELEMENTS, MUON_ITERATIONS, MUON_LEARNING_RATE, NGRAM,
        OUTPUT, PREFILL_TOKENS_V1, QUERIES, SELECTED_BLOCKS_V1, SELECTED_TOKENS_V1,
        SINKHORN_ITERATIONS_V1, STATE_WIDTH, SYSTEM_BATCHES, TABLE_SIZE, TOKENS,
        TOKENS_PER_BLOCK_V1, TOP_K,
    };
}

#[cfg(feature = "hardware-test-hooks")]
#[path = "../../../examples/gfx950_advanced_attention/src/reference.rs"]
mod attention_reference;
#[cfg(feature = "hardware-test-hooks")]
#[path = "../../../examples/gfx950_gpt_oss_decode/src/reference.rs"]
mod gpt_oss_reference;

#[cfg(feature = "hardware-test-hooks")]
#[path = "../../../examples/gfx950_advanced_systems/src/reference.rs"]
mod systems_reference;

#[cfg(feature = "hardware-test-hooks")]
const RUN_ENV: &str = "FE2O3_RUN_GFX950_ADVANCED_HARDWARE";
#[cfg(feature = "hardware-test-hooks")]
const HSACO_ENV: &str = "FE2O3_GFX950_ADVANCED_HSACO";
#[cfg(feature = "hardware-test-hooks")]
const WORKGROUP_X_ENV: &str = "FE2O3_GFX950_ADVANCED_WORKGROUP_X";
#[cfg(feature = "hardware-test-hooks")]
const SHA256_ENV: &str = "FE2O3_GFX950_ADVANCED_SHA256";
#[cfg(feature = "hardware-test-hooks")]
const HSA_KERNARG_ALIGNMENT: u64 = 16;
#[cfg(feature = "hardware-test-hooks")]
const PERF_OUTPUT_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_OUTPUT";
#[cfg(feature = "hardware-test-hooks")]
const PERF_WARMUPS_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_WARMUPS";
#[cfg(feature = "hardware-test-hooks")]
const PERF_BLOCKS_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_BLOCKS";
#[cfg(feature = "hardware-test-hooks")]
const PERF_SAMPLES_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_SAMPLES_PER_BLOCK";
#[cfg(feature = "hardware-test-hooks")]
const PERF_REWARM_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_BLOCK_REWARM";
#[cfg(feature = "hardware-test-hooks")]
const PERF_CAMPAIGN_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_CAMPAIGN_ID";
#[cfg(feature = "hardware-test-hooks")]
const PERF_IMPLEMENTATION_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_IMPLEMENTATION_ID";
#[cfg(feature = "hardware-test-hooks")]
const PERF_VARIANT_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_VARIANT_ID";
#[cfg(feature = "hardware-test-hooks")]
const PERF_PROCESS_ENV: &str = "FE2O3_GFX950_ADVANCED_PERF_PROCESS";
#[cfg(feature = "hardware-test-hooks")]
const PERF_LLVM_SHA256_ENV: &str = "FE2O3_GFX950_ADVANCED_LLVM_SHA256";
#[cfg(feature = "hardware-test-hooks")]
const PERF_ISA_SHA256_ENV: &str = "FE2O3_GFX950_ADVANCED_ISA_SHA256";
#[cfg(feature = "hardware-test-hooks")]
const PERF_CRATE_BINDING_ENV: &str = "FE2O3_GFX950_ADVANCED_CRATE_BINDING";
#[cfg(feature = "hardware-test-hooks")]
const PERF_SOURCE_COMMIT_ENV: &str = "FE2O3_GFX950_ADVANCED_SOURCE_COMMIT";
#[cfg(feature = "hardware-test-hooks")]
const PERF_SOURCE_TREE_ENV: &str = "FE2O3_GFX950_ADVANCED_SOURCE_TREE";
#[cfg(feature = "hardware-test-hooks")]
const DEFAULT_PERF_WARMUPS: usize = 1_000;
#[cfg(feature = "hardware-test-hooks")]
const DEFAULT_PERF_BLOCKS: usize = 30;
#[cfg(feature = "hardware-test-hooks")]
const DEFAULT_PERF_SAMPLES_PER_BLOCK: usize = 100;
#[cfg(feature = "hardware-test-hooks")]
const DEFAULT_PERF_BLOCK_REWARM: usize = 20;
#[cfg(feature = "hardware-test-hooks")]
const METADATA_KERNARG_ALIGNMENT: u64 = 8;
#[cfg(feature = "hardware-test-hooks")]
const GUARD_BYTES: usize = 16;
#[cfg(feature = "hardware-test-hooks")]
const PREFIX_BYTE: u8 = 0x95;
#[cfg(feature = "hardware-test-hooks")]
const SUFFIX_BYTE: u8 = 0x59;
#[cfg(feature = "hardware-test-hooks")]
const F32_POISON_BITS: u32 = 0x7fc0_00a5;

#[cfg(feature = "hardware-test-hooks")]
type BoxError = Box<dyn std::error::Error>;

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Debug)]
struct PerformanceConfig {
    output: std::path::PathBuf,
    warmups: usize,
    blocks: usize,
    samples_per_block: usize,
    block_rewarm: usize,
    campaign_id: String,
    implementation_id: String,
    variant_id: String,
    process: usize,
    llvm_sha256: String,
    isa_sha256: String,
    crate_binding: String,
    source_commit: String,
    source_tree: String,
}

#[cfg(feature = "hardware-test-hooks")]
impl PerformanceConfig {
    fn from_environment() -> Result<Option<Self>, BoxError> {
        let Some(output) = std::env::var_os(PERF_OUTPUT_ENV) else {
            return Ok(None);
        };
        let output = std::path::PathBuf::from(output);
        require(
            output.is_absolute(),
            format!("{PERF_OUTPUT_ENV} must be absolute"),
        )?;
        let parent = output
            .parent()
            .ok_or_else(|| format!("{PERF_OUTPUT_ENV} has no parent"))?;
        require(
            parent.is_dir() && std::fs::canonicalize(parent)? == parent,
            format!("{PERF_OUTPUT_ENV} parent must be an existing canonical directory"),
        )?;
        if output.exists() {
            let metadata = std::fs::symlink_metadata(&output)?;
            require(
                metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
                format!("{PERF_OUTPUT_ENV} must be a regular non-symlink file"),
            )?;
        }
        Ok(Some(Self {
            output,
            warmups: parse_performance_count(PERF_WARMUPS_ENV, DEFAULT_PERF_WARMUPS, true)?,
            blocks: parse_performance_count(PERF_BLOCKS_ENV, DEFAULT_PERF_BLOCKS, false)?,
            samples_per_block: parse_performance_count(
                PERF_SAMPLES_ENV,
                DEFAULT_PERF_SAMPLES_PER_BLOCK,
                false,
            )?,
            block_rewarm: parse_performance_count(
                PERF_REWARM_ENV,
                DEFAULT_PERF_BLOCK_REWARM,
                true,
            )?,
            campaign_id: performance_text(PERF_CAMPAIGN_ENV, None)?,
            implementation_id: performance_text(
                PERF_IMPLEMENTATION_ENV,
                Some("fe2o3-production-rust"),
            )?,
            variant_id: performance_text(PERF_VARIANT_ENV, Some("candidate"))?,
            process: parse_performance_count(PERF_PROCESS_ENV, 0, true)?,
            llvm_sha256: performance_hex(PERF_LLVM_SHA256_ENV, 64)?,
            isa_sha256: performance_hex(PERF_ISA_SHA256_ENV, 64)?,
            crate_binding: performance_hex(PERF_CRATE_BINDING_ENV, 64)?,
            source_commit: performance_hex(PERF_SOURCE_COMMIT_ENV, 40)?,
            source_tree: performance_hex(PERF_SOURCE_TREE_ENV, 40)?,
        }))
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn parse_performance_count(
    name: &'static str,
    default: usize,
    allow_zero: bool,
) -> Result<usize, BoxError> {
    let value = match std::env::var(name) {
        Ok(text) => text
            .parse::<usize>()
            .map_err(|_| format!("{name} must be an unsigned decimal integer"))?,
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => return Err(format!("{name} is not valid text: {error}").into()),
    };
    require(
        value <= 1_000_000 && (allow_zero || value != 0),
        format!("{name} is outside the admitted range"),
    )?;
    Ok(value)
}

#[cfg(feature = "hardware-test-hooks")]
fn performance_text(name: &'static str, default: Option<&str>) -> Result<String, BoxError> {
    let value = match std::env::var(name) {
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => default
            .ok_or_else(|| format!("{name} is required when {PERF_OUTPUT_ENV} is set"))?
            .to_owned(),
        Err(error) => return Err(format!("{name} is not valid text: {error}").into()),
    };
    require(
        !value.is_empty()
            && value.len() <= 128
            && value
                .bytes()
                .all(|byte| byte.is_ascii_graphic() && byte != b'"' && byte != b'\\'),
        format!("{name} must be 1..=128 safe printable ASCII bytes"),
    )?;
    Ok(value)
}

#[cfg(feature = "hardware-test-hooks")]
fn performance_hex(name: &'static str, length: usize) -> Result<String, BoxError> {
    let value = performance_text(name, None)?;
    require(
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        format!("{name} must be exactly {length} lowercase hex digits"),
    )?;
    Ok(value)
}

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AbiArg {
    Slice,
    Pointer,
    U32,
}

#[cfg(feature = "hardware-test-hooks")]
const SIX_SLICES: &[AbiArg] = &[AbiArg::Slice; 6];
#[cfg(feature = "hardware-test-hooks")]
const EIGHT_SLICES: &[AbiArg] = &[AbiArg::Slice; 8];
#[cfg(feature = "hardware-test-hooks")]
const NINE_KDA_SLICES: &[AbiArg] = &[AbiArg::Slice; 9];
#[cfg(feature = "hardware-test-hooks")]
const FIVE_SLICES: &[AbiArg] = &[AbiArg::Slice; 5];
#[cfg(feature = "hardware-test-hooks")]
const FOUR_SLICES: &[AbiArg] = &[AbiArg::Slice; 4];
#[cfg(feature = "hardware-test-hooks")]
const THREE_SLICES: &[AbiArg] = &[AbiArg::Slice; 3];
#[cfg(feature = "hardware-test-hooks")]
const TWO_SLICES: &[AbiArg] = &[AbiArg::Slice; 2];
#[cfg(feature = "hardware-test-hooks")]
const DEEPSEEK_SPARSE_ARGS: &[AbiArg] = &[
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::U32,
    AbiArg::U32,
    AbiArg::U32,
    AbiArg::U32,
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::Slice,
];
#[cfg(feature = "hardware-test-hooks")]
const EXPERT_RANK_ARGS: &[AbiArg] = &[
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::Slice,
    AbiArg::U32,
    AbiArg::U32,
    AbiArg::Slice,
];
#[cfg(feature = "hardware-test-hooks")]
const NINE_SLICES: &[AbiArg] = &[AbiArg::Slice; 9];
#[cfg(feature = "hardware-test-hooks")]
const THIRTEEN_SLICES: &[AbiArg] = &[AbiArg::Slice; 13];
#[cfg(feature = "hardware-test-hooks")]
const THREE_POINTERS: &[AbiArg] = &[AbiArg::Pointer; 3];
#[cfg(feature = "hardware-test-hooks")]
const FIVE_POINTERS: &[AbiArg] = &[AbiArg::Pointer; 5];
#[cfg(feature = "hardware-test-hooks")]
const SIX_POINTERS: &[AbiArg] = &[AbiArg::Pointer; 6];

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AdvancedCase {
    label: &'static str,
    export: &'static str,
    descriptor: &'static str,
    workgroup_x: u32,
    static_lds_bytes: u64,
    args: &'static [AbiArg],
}

#[cfg(feature = "hardware-test-hooks")]
impl AdvancedCase {
    fn grid_x(self) -> u32 {
        if matches!(self.args.first(), Some(AbiArg::Pointer)) {
            1
        } else {
            4
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
const KDA_DECODE: AdvancedCase = AdvancedCase {
    label: "gfx950 Kimi Delta Attention decode",
    export: "gfx950_kda_decode",
    descriptor: "gfx950_kda_decode.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: EIGHT_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const KDA_PREFILL: AdvancedCase = AdvancedCase {
    label: "gfx950 Kimi Delta Attention chunkwise prefill",
    export: "gfx950_kda_chunkwise_prefill",
    descriptor: "gfx950_kda_chunkwise_prefill.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: NINE_KDA_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const SPARSE_ATTENTION: AdvancedCase = AdvancedCase {
    label: "gfx950 content sparse attention",
    export: "gfx950_content_sparse_attention",
    descriptor: "gfx950_content_sparse_attention.kd",
    workgroup_x: 256,
    static_lds_bytes: 8192,
    args: SIX_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const DEEPSEEK_SPARSE_ATTENTION: AdvancedCase = AdvancedCase {
    label: "gfx950 DeepSeek sparse attention",
    export: "gfx950_deepseek_sparse_attention",
    descriptor: "gfx950_deepseek_sparse_attention.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: DEEPSEEK_SPARSE_ARGS,
};
#[cfg(feature = "hardware-test-hooks")]
const HYBRID_ATTENTION: AdvancedCase = AdvancedCase {
    label: "gfx950 compressed hybrid attention",
    export: "gfx950_compressed_hybrid_attention",
    descriptor: "gfx950_compressed_hybrid_attention.kd",
    workgroup_x: 256,
    static_lds_bytes: 8192,
    args: FIVE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const ATTNRES: AdvancedCase = AdvancedCase {
    label: "gfx950 AttnRes aggregation",
    export: "gfx950_attnres_aggregate",
    descriptor: "gfx950_attnres_aggregate.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const FOUR_BRANCH: AdvancedCase = AdvancedCase {
    label: "gfx950 four-branch residual",
    export: "gfx950_four_branch_residual",
    descriptor: "gfx950_four_branch_residual.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: FOUR_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const MHC: AdvancedCase = AdvancedCase {
    label: "gfx950 mHC Sinkhorn mix",
    export: "gfx950_mhc_sinkhorn_mix",
    descriptor: "gfx950_mhc_sinkhorn_mix.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B 16-item layer-tile megakernel",
    export: "gfx950_gpt_oss_120b_decode_megakernel_v1",
    descriptor: "gfx950_gpt_oss_120b_decode_megakernel_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THIRTEEN_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_PIPELINED: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B 16-item pipelined-attention megakernel",
    export: "gfx950_gpt_oss_120b_decode_megakernel_v1",
    descriptor: "gfx950_gpt_oss_120b_decode_megakernel_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 8192,
    args: THIRTEEN_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_ROUTER_COMPONENT: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B materialized Rust router",
    export: "gfx950_gpt_oss_120b_router_v1",
    descriptor: "gfx950_gpt_oss_120b_router_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_ATTENTION_COMPONENT: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B materialized Rust attention",
    export: "gfx950_gpt_oss_120b_attention_v1",
    descriptor: "gfx950_gpt_oss_120b_attention_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: FIVE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_EXPERT_COMPONENT: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B materialized Rust expert",
    export: "gfx950_gpt_oss_120b_expert_v1",
    descriptor: "gfx950_gpt_oss_120b_expert_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: SIX_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_UNFUSED_ROUTER: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B unfused router comparator",
    export: "gpt_oss_unfused_router",
    descriptor: "gpt_oss_unfused_router.kd",
    workgroup_x: 64,
    static_lds_bytes: 0,
    args: THREE_POINTERS,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_UNFUSED_ATTENTION: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B unfused attention comparator",
    export: "gpt_oss_unfused_attention",
    descriptor: "gpt_oss_unfused_attention.kd",
    workgroup_x: 64,
    static_lds_bytes: 0,
    args: FIVE_POINTERS,
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_UNFUSED_EXPERT: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS-120B unfused expert comparator",
    export: "gpt_oss_unfused_expert",
    descriptor: "gpt_oss_unfused_expert.kd",
    workgroup_x: 64,
    static_lds_bytes: 0,
    args: SIX_POINTERS,
};
#[cfg(feature = "hardware-test-hooks")]
const MOE_ROUTE: AdvancedCase = AdvancedCase {
    label: "gfx950 MoE route",
    export: "gfx950_moe_route_fp4_t16_e4_k2_v1",
    descriptor: "gfx950_moe_route_fp4_t16_e4_k2_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: SIX_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const MOE_EXPERT: AdvancedCase = AdvancedCase {
    label: "gfx950 MoE expert rank",
    export: "gfx950_moe_expert_rank_fp4_fp8_v1",
    descriptor: "gfx950_moe_expert_rank_fp4_fp8_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: EXPERT_RANK_ARGS,
};
#[cfg(feature = "hardware-test-hooks")]
const COMBINE: AdvancedCase = AdvancedCase {
    label: "gfx950 expert rank combine",
    export: "gfx950_combine_expert_ranks_v1",
    descriptor: "gfx950_combine_expert_ranks_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const SPECULATIVE: AdvancedCase = AdvancedCase {
    label: "gfx950 speculative transaction",
    export: "gfx950_speculative_transaction_v1",
    descriptor: "gfx950_speculative_transaction_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: NINE_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const NGRAM_GATHER: AdvancedCase = AdvancedCase {
    label: "gfx950 Qwen N-gram gather",
    export: "gfx950_qwen_ngram_gather_v1",
    descriptor: "gfx950_qwen_ngram_gather_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: SIX_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const STAGE_SHARD: AdvancedCase = AdvancedCase {
    label: "gfx950 gradient shard stage",
    export: "gfx950_stage_gradient_shard_v1",
    descriptor: "gfx950_stage_gradient_shard_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: TWO_SLICES,
};
#[cfg(feature = "hardware-test-hooks")]
const MUON: AdvancedCase = AdvancedCase {
    label: "gfx950 Muon update",
    export: "gfx950_muon_update_4x4_v1",
    descriptor: "gfx950_muon_update_4x4_v1.kd",
    workgroup_x: 256,
    static_lds_bytes: 0,
    args: THREE_SLICES,
};

#[cfg(feature = "hardware-test-hooks")]
fn require(condition: bool, message: impl Into<String>) -> Result<(), BoxError> {
    if condition {
        Ok(())
    } else {
        Err(message.into().into())
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_runtime_visibility_is_single(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        !value.is_empty()
            && (value.bytes().all(|byte| byte.is_ascii_digit())
                || value.strip_prefix("GPU-").is_some_and(|uuid| {
                    uuid.len() == 16 && uuid.bytes().all(|byte| byte.is_ascii_hexdigit())
                }))
    })
}

#[cfg(feature = "hardware-test-hooks")]
fn require_gpt_runtime_visibility() -> Result<(), BoxError> {
    let value = std::env::var_os("ROCR_VISIBLE_DEVICES");
    require(
        gpt_runtime_visibility_is_single(value.as_deref().and_then(|value| value.to_str())),
        "GPT-OSS runtime requires one explicit ROCR_VISIBLE_DEVICES UUID or ordinal",
    )
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gpt_runtime_visibility_requires_one_explicit_device() {
    for value in [
        None,
        Some(""),
        Some("0,1"),
        Some("-1"),
        Some(" 0"),
        Some("GPU-0123"),
        Some("GPU-0123456789abcdeg"),
    ] {
        assert!(!gpt_runtime_visibility_is_single(value), "{value:?}");
    }
    for value in ["0", "12", "GPU-0123456789abcdef", "GPU-ABCDEF0123456789"] {
        assert!(gpt_runtime_visibility_is_single(Some(value)), "{value:?}");
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn kernarg_size(args: &[AbiArg]) -> usize {
    let size = args
        .iter()
        .map(|arg| match arg {
            AbiArg::Slice => 16,
            AbiArg::Pointer => 8,
            AbiArg::U32 => 4,
        })
        .sum::<usize>();
    size.next_multiple_of(METADATA_KERNARG_ALIGNMENT as usize)
}

#[cfg(feature = "hardware-test-hooks")]
fn expected_metadata_arguments(args: &[AbiArg]) -> Vec<(u64, u64, ExplicitValueKind)> {
    let mut offset = 0_u64;
    let mut result = Vec::new();
    for arg in args {
        match arg {
            AbiArg::Slice => {
                result.push((offset, 8, ExplicitValueKind::GlobalBuffer));
                result.push((offset + 8, 8, ExplicitValueKind::ByValue));
                offset += 16;
            }
            AbiArg::Pointer => {
                result.push((offset, 8, ExplicitValueKind::GlobalBuffer));
                offset += 8;
            }
            AbiArg::U32 => {
                result.push((offset, 4, ExplicitValueKind::ByValue));
                offset += 4;
            }
        }
    }
    result
}

#[cfg(feature = "hardware-test-hooks")]
fn parse_sha256(text: &str) -> Result<[u8; 32], BoxError> {
    require(
        text.len() == 64
            && text
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        format!("{SHA256_ENV} must be exactly 64 lowercase hex digits"),
    )?;
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16)
            .map_err(|_| format!("{SHA256_ENV} is malformed"))?;
    }
    Ok(bytes)
}

#[cfg(feature = "hardware-test-hooks")]
fn read_pinned_hsaco() -> Result<(Vec<u8>, PayloadDigest), BoxError> {
    require(
        std::env::var(RUN_ENV).as_deref() == Ok("1"),
        format!("set {RUN_ENV}=1 to opt into this raw hardware test"),
    )?;
    let path = std::path::PathBuf::from(
        std::env::var_os(HSACO_ENV).ok_or_else(|| format!("{HSACO_ENV} is not set"))?,
    );
    require(path.is_absolute(), format!("{HSACO_ENV} must be absolute"))?;
    let metadata = std::fs::symlink_metadata(&path)?;
    require(
        metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
        format!("{HSACO_ENV} must name a regular non-symlink file"),
    )?;
    require(
        std::fs::canonicalize(&path)? == path,
        format!("{HSACO_ENV} must already be canonical"),
    )?;
    require(
        (1..=fe2o3_hsaco::MAX_HSACO_BYTES as u64).contains(&metadata.len()),
        format!("{HSACO_ENV} has an invalid byte length"),
    )?;
    let expected =
        parse_sha256(&std::env::var(SHA256_ENV).map_err(|_| format!("{SHA256_ENV} is not set"))?)?;
    let bytes = std::fs::read(&path)?;
    let final_metadata = std::fs::symlink_metadata(&path)?;
    require(
        bytes.len() as u64 == metadata.len()
            && final_metadata.file_type().is_file()
            && !final_metadata.file_type().is_symlink()
            && final_metadata.len() == metadata.len()
            && std::fs::canonicalize(&path)? == path,
        format!("{HSACO_ENV} changed identity while being read"),
    )?;
    let digest = DigestAlgorithm::Sha256.calculate(&bytes);
    require(
        digest.bytes().as_bytes() == &expected,
        format!("{HSACO_ENV} does not match its exact SHA-256 pin"),
    )?;
    Ok((bytes, digest))
}

#[cfg(feature = "hardware-test-hooks")]
fn inspect_profile(case: AdvancedCase, bytes: &[u8]) -> Result<(), BoxError> {
    let bound = fe2o3_hsaco::inspect_and_bind_kernel_descriptors(bytes)?;
    let inspected = bound.inspection();
    require(
        inspected.code_object_version() == CodeObjectVersion::V6,
        format!("{} must use code object V6", case.label),
    )?;
    require(
        inspected.target().processor() == "gfx950"
            && inspected.target().xnack() == Some(FeatureState::Disabled),
        format!("{} must target exact gfx950:xnack-", case.label),
    )?;
    require(
        !inspected.has_printf_metadata(),
        format!("{} must not carry printf metadata", case.label),
    )?;
    let [kernel] = inspected.kernels() else {
        return Err(format!("{} must declare exactly one kernel", case.label).into());
    };
    let expected_kernarg = kernarg_size(case.args);
    require(
        kernel.name() == case.export && kernel.symbol() == case.descriptor,
        format!("{} has a substituted kernel symbol", case.label),
    )?;
    require(
        kernel.kernarg_segment_size() == expected_kernarg as u64
            && kernel.kernarg_segment_alignment() == METADATA_KERNARG_ALIGNMENT
            && kernel.implicit_argument_offset().is_none()
            && kernel.implicit_argument_size() == 0,
        format!("{} has a substituted explicit kernarg ABI", case.label),
    )?;
    require(
        kernel.required_workgroup_size() == Some([case.workgroup_x, 1, 1])
            && kernel.max_flat_workgroup_size() == case.workgroup_x
            && kernel.wavefront_size() == 64
            && kernel.group_segment_fixed_size() == case.static_lds_bytes
            && !kernel.uses_dynamic_stack(),
        format!("{} has a substituted workgroup/LDS profile", case.label),
    )?;
    let arguments = kernel
        .explicit_arguments()
        .iter()
        .map(|argument| (argument.offset(), argument.size(), argument.value_kind()))
        .collect::<Vec<_>>();
    require(
        arguments == expected_metadata_arguments(case.args),
        format!("{} explicit argument metadata changed", case.label),
    )?;
    let [binding] = bound.bindings() else {
        return Err(format!("{} must bind exactly one descriptor", case.label).into());
    };
    let descriptor = binding.descriptor();
    require(
        binding.kernel_index() == 0
            && descriptor.kernarg_size() == expected_kernarg as u32
            && u64::from(descriptor.group_segment_fixed_size()) == case.static_lds_bytes
            && descriptor.wavefront_size() == 64
            && !descriptor.uses_dynamic_stack(),
        format!("{} descriptor disagrees with metadata", case.label),
    )
}

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone)]
enum ExpectedOutput {
    F32 {
        values: Vec<f32>,
        absolute_tolerance: f32,
        relative_tolerance: f32,
        exact_mask: Option<Vec<bool>>,
    },
    U32(Vec<u32>),
    I32(Vec<i32>),
}

#[cfg(feature = "hardware-test-hooks")]
impl ExpectedOutput {
    fn element_count(&self) -> usize {
        match self {
            Self::F32 { values, .. } => values.len(),
            Self::U32(values) => values.len(),
            Self::I32(values) => values.len(),
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[derive(Clone)]
struct PlannedBuffer {
    name: &'static str,
    initial: Vec<u8>,
    body_offset: usize,
    elements: usize,
    immutable: bool,
    expected: Option<ExpectedOutput>,
}

#[cfg(feature = "hardware-test-hooks")]
enum PlannedArg {
    Slice { buffer: usize, elements: usize },
    Pointer { buffer: usize },
    U32(u32),
}

#[cfg(feature = "hardware-test-hooks")]
struct LaunchPlan {
    label: String,
    buffers: Vec<PlannedBuffer>,
    args: Vec<PlannedArg>,
}

#[cfg(feature = "hardware-test-hooks")]
fn value_bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(values);
    // SAFETY: every input type used here is a plain integer or f32 and the
    // returned Vec owns a byte-for-byte copy of the complete initialized span.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len) }.to_vec()
}

#[cfg(feature = "hardware-test-hooks")]
fn input<T: Copy>(name: &'static str, values: &[T]) -> PlannedBuffer {
    PlannedBuffer {
        name,
        initial: value_bytes(values),
        body_offset: 0,
        elements: values.len(),
        immutable: true,
        expected: None,
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn output(name: &'static str, expected: ExpectedOutput) -> PlannedBuffer {
    let mut initial = vec![PREFIX_BYTE; GUARD_BYTES];
    match &expected {
        ExpectedOutput::F32 {
            values, exact_mask, ..
        } => {
            for (index, value) in values.iter().enumerate() {
                let poison = if exact_mask.as_ref().is_some_and(|mask| mask[index]) {
                    !value.to_bits()
                } else {
                    F32_POISON_BITS
                };
                initial.extend_from_slice(&poison.to_le_bytes());
            }
        }
        ExpectedOutput::U32(values) => {
            for value in values {
                initial.extend_from_slice(&(!*value).to_le_bytes());
            }
        }
        ExpectedOutput::I32(values) => {
            for value in values {
                initial.extend_from_slice(&(!*value).to_le_bytes());
            }
        }
    }
    initial.extend(std::iter::repeat_n(SUFFIX_BYTE, GUARD_BYTES));
    PlannedBuffer {
        name,
        initial,
        body_offset: GUARD_BYTES,
        elements: expected.element_count(),
        immutable: false,
        expected: Some(expected),
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn f32_output(name: &'static str, values: Vec<f32>, tolerance: f32) -> PlannedBuffer {
    output(
        name,
        ExpectedOutput::F32 {
            values,
            absolute_tolerance: tolerance,
            relative_tolerance: 0.0,
            exact_mask: None,
        },
    )
}

#[cfg(feature = "hardware-test-hooks")]
fn deterministic_floats(count: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let centered = ((index * 17 + salt * 11) % 19) as i32 - 9;
            scale * centered as f32 / 9.0
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn normalized_kda_rows(rows: usize, salt: usize) -> Vec<f32> {
    let mut values = deterministic_floats(rows * KDA_KEY_DIMENSION_V1, salt, 0.8);
    for row in values.chunks_mut(KDA_KEY_DIMENSION_V1) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        for value in row {
            *value /= norm;
        }
    }
    values
}

#[cfg(feature = "hardware-test-hooks")]
fn kda_state_value_major(state: &[f32]) -> Vec<f32> {
    let mut physical = vec![0.0; KDA_STATE_ELEMENTS_V1];
    for value in 0..KDA_VALUE_DIMENSION_V1 {
        for key in 0..KDA_KEY_DIMENSION_V1 {
            physical[value * KDA_KEY_DIMENSION_V1 + key] =
                state[key * KDA_VALUE_DIMENSION_V1 + value];
        }
    }
    physical
}

#[cfg(feature = "hardware-test-hooks")]
fn kda_replicated_decode_output(output: &[f32]) -> Vec<f32> {
    let mut physical = vec![0.0; KDA_STATE_ELEMENTS_V1];
    for value in 0..KDA_VALUE_DIMENSION_V1 {
        for key in 0..KDA_KEY_DIMENSION_V1 {
            physical[value * KDA_KEY_DIMENSION_V1 + key] = output[value];
        }
    }
    physical
}

#[cfg(feature = "hardware-test-hooks")]
fn kda_replicated_chunk_output(output: &[f32], chunk: usize) -> Vec<f32> {
    let mut physical = vec![0.0; KDA_STATE_ELEMENTS_V1];
    for value in 0..KDA_VALUE_DIMENSION_V1 {
        for key in 0..KDA_KEY_DIMENSION_V1 {
            let token = chunk * KDA_CHUNK_TOKENS_V1 + key / 4;
            physical[value * KDA_KEY_DIMENSION_V1 + key] =
                output[token * KDA_VALUE_DIMENSION_V1 + value];
        }
    }
    physical
}

#[cfg(feature = "hardware-test-hooks")]
fn deterministic_fp8(count: usize, salt: usize) -> Vec<u8> {
    const CODES: [u8; 5] = [0xb8, 0xb0, 0x00, 0x30, 0x38];
    (0..count)
        .map(|index| CODES[(index * 3 + salt) % CODES.len()])
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn nonuniform_fp8(count: usize, salt: usize, batch: usize) -> Vec<u8> {
    const CODES: [u8; 5] = [0xb8, 0xb0, 0x00, 0x30, 0x38];
    let mut values = deterministic_fp8(count, salt + batch);
    if count > 1 {
        values[0] = CODES[batch % CODES.len()];
        values[1] = CODES[(batch / CODES.len()) % CODES.len()];
    }
    values
}

#[cfg(feature = "hardware-test-hooks")]
fn repeated_attention_query(batch: usize) -> (Vec<u8>, Vec<u8>) {
    let logical = nonuniform_fp8(HEAD_DIMENSION_V1, 1, batch);
    let mut physical = Vec::with_capacity(ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1);
    for _ in 0..ATTENTION_TOKENS_V1 {
        physical.extend_from_slice(&logical);
    }
    (logical, physical)
}

#[cfg(feature = "hardware-test-hooks")]
fn attention_plans(case: AdvancedCase) -> Result<Vec<LaunchPlan>, BoxError> {
    let plan = match case.export {
        "gfx950_kda_decode" => {
            let mut query = Vec::new();
            let mut key = Vec::new();
            let mut value = Vec::new();
            let mut alpha = Vec::new();
            let mut beta = Vec::new();
            let mut physical_initial_state = Vec::new();
            let mut expected_state = Vec::new();
            let mut expected_output = Vec::new();
            for batch in 0..MULTIGRID_WORKGROUPS_V1 {
                let batch_query = normalized_kda_rows(1, 1 + batch * 3);
                let batch_key = normalized_kda_rows(1, 2 + batch * 3);
                let batch_value = deterministic_floats(
                    KDA_VALUE_DIMENSION_V1,
                    3 + batch,
                    0.45 + batch as f32 * 0.03,
                );
                let batch_alpha = deterministic_floats(
                    KDA_KEY_DIMENSION_V1,
                    4 + batch,
                    0.16 + batch as f32 * 0.01,
                )
                .into_iter()
                .map(|entry| 0.75 + entry)
                .collect::<Vec<_>>();
                let batch_beta = vec![0.45 + batch as f32 * 0.08];
                let batch_state = deterministic_floats(
                    KDA_STATE_ELEMENTS_V1,
                    5 + batch,
                    0.20 + batch as f32 * 0.02,
                );
                let expected = attention_reference::kda_decode_reference_v2(
                    &batch_query,
                    &batch_key,
                    &batch_value,
                    &batch_alpha,
                    &batch_beta,
                    &batch_state,
                )
                .map_err(|error| format!("KDA decode reference failed: {error:?}"))?;
                query.extend(batch_query);
                key.extend(batch_key);
                value.extend(batch_value);
                alpha.extend(batch_alpha);
                beta.extend(batch_beta);
                physical_initial_state.extend(kda_state_value_major(&batch_state));
                expected_state.extend(kda_state_value_major(&expected.state));
                expected_output.extend(kda_replicated_decode_output(&expected.output));
            }
            let lengths = [
                query.len(),
                key.len(),
                value.len(),
                alpha.len(),
                beta.len(),
                physical_initial_state.len(),
                expected_state.len(),
                expected_output.len(),
            ];
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("query", &query),
                    input("key", &key),
                    input("value", &value),
                    input("alpha", &alpha),
                    input("beta", &beta),
                    input("initial_state_value_major", &physical_initial_state),
                    f32_output("final_state_value_major", expected_state, 2.0e-5),
                    f32_output("output_replicated", expected_output, 2.0e-5),
                ],
                args: (0..8)
                    .map(|buffer| PlannedArg::Slice {
                        elements: lengths[buffer],
                        buffer,
                    })
                    .collect(),
            }
        }
        "gfx950_kda_chunkwise_prefill" => {
            let mut query = Vec::new();
            let mut key = Vec::new();
            let mut value = Vec::new();
            let mut alpha = Vec::new();
            let mut beta = Vec::new();
            let mut physical_initial_state = Vec::new();
            let mut expected_state = Vec::new();
            let mut expected_chunk0 = Vec::new();
            let mut expected_chunk1 = Vec::new();
            for batch in 0..MULTIGRID_WORKGROUPS_V1 {
                let batch_query = normalized_kda_rows(PREFILL_TOKENS_V1, 6 + batch * 3);
                let batch_key = normalized_kda_rows(PREFILL_TOKENS_V1, 7 + batch * 3);
                let batch_value = deterministic_floats(
                    PREFILL_TOKENS_V1 * KDA_VALUE_DIMENSION_V1,
                    8 + batch,
                    0.42 + batch as f32 * 0.025,
                );
                let batch_alpha = deterministic_floats(
                    PREFILL_TOKENS_V1 * KDA_KEY_DIMENSION_V1,
                    9 + batch,
                    0.15 + batch as f32 * 0.01,
                )
                .into_iter()
                .map(|entry| 0.75 + entry)
                .collect::<Vec<_>>();
                let batch_beta = (0..PREFILL_TOKENS_V1)
                    .map(|token| 0.25 + batch as f32 * 0.03 + 0.04 * token as f32)
                    .collect::<Vec<_>>();
                let batch_state = deterministic_floats(
                    KDA_STATE_ELEMENTS_V1,
                    10 + batch,
                    0.18 + batch as f32 * 0.02,
                );
                let expected = attention_reference::kda_prefill_reference_v2(
                    &batch_query,
                    &batch_key,
                    &batch_value,
                    &batch_alpha,
                    &batch_beta,
                    &batch_state,
                )
                .map_err(|error| format!("KDA prefill reference failed: {error:?}"))?;
                query.extend(batch_query);
                key.extend(batch_key);
                value.extend(batch_value);
                alpha.extend(batch_alpha);
                beta.extend(batch_beta);
                physical_initial_state.extend(kda_state_value_major(&batch_state));
                expected_state.extend(kda_state_value_major(&expected.final_state));
                expected_chunk0.extend(kda_replicated_chunk_output(&expected.output, 0));
                expected_chunk1.extend(kda_replicated_chunk_output(&expected.output, 1));
            }
            let lengths = [
                query.len(),
                key.len(),
                value.len(),
                alpha.len(),
                beta.len(),
                physical_initial_state.len(),
                expected_state.len(),
                expected_chunk0.len(),
                expected_chunk1.len(),
            ];
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("query", &query),
                    input("key", &key),
                    input("value", &value),
                    input("alpha", &alpha),
                    input("beta", &beta),
                    input("initial_state_value_major", &physical_initial_state),
                    f32_output("final_state_value_major", expected_state, 2.0e-4),
                    f32_output("output_chunk0_replicated", expected_chunk0, 2.0e-4),
                    f32_output("output_chunk1_replicated", expected_chunk1, 2.0e-4),
                ],
                args: (0..9)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: lengths[buffer],
                    })
                    .collect(),
            }
        }
        "gfx950_deepseek_sparse_attention" => {
            let indices = vec![13_u32, u32::MAX, 2, 9];
            require(
                indices.len() == DEEPSEEK_SPARSE_TOP_K_V1,
                "DeepSeek sparse fixture must provide exactly top-k indices",
            )?;
            let mut query = Vec::new();
            let mut key = Vec::new();
            let mut value = Vec::new();
            let mut expected_output = Vec::new();
            let mut expected_maximum = Vec::new();
            let mut expected_normalizer = Vec::new();
            for batch in 0..MULTIGRID_SUBGROUP_BATCHES_V1 {
                let batch_query =
                    deterministic_floats(HEAD_DIMENSION_V1, 1 + batch, 0.30 + batch as f32 * 0.002);
                let batch_key = deterministic_floats(
                    ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1,
                    2 + batch,
                    0.34 + batch as f32 * 0.002,
                );
                let batch_value = deterministic_floats(
                    ATTENTION_TOKENS_V1 * CHANNELS_V1,
                    3 + batch,
                    0.38 + batch as f32 * 0.002,
                );
                let expected = attention_reference::deepseek_sparse_attention_reference_v1(
                    &batch_query,
                    &batch_key,
                    &batch_value,
                    &indices,
                )
                .map_err(|error| format!("DeepSeek sparse reference failed: {error:?}"))?;
                query.extend(batch_query);
                key.extend(batch_key);
                value.extend(batch_value);
                expected_output.extend(expected.output);
                expected_maximum.extend(std::iter::repeat_n(expected.softmax_maximum, CHANNELS_V1));
                expected_normalizer.extend(std::iter::repeat_n(
                    expected.softmax_normalizer,
                    CHANNELS_V1,
                ));
            }
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("q", &query),
                    input("k", &key),
                    input("v", &value),
                    f32_output("output", expected_output, 5.0e-3),
                    f32_output("softmax_maximum_output", expected_maximum, 5.0e-3),
                    f32_output("softmax_normalizer_output", expected_normalizer, 5.0e-3),
                ],
                args: vec![
                    PlannedArg::Slice {
                        buffer: 0,
                        elements: query.len(),
                    },
                    PlannedArg::Slice {
                        buffer: 1,
                        elements: key.len(),
                    },
                    PlannedArg::Slice {
                        buffer: 2,
                        elements: value.len(),
                    },
                    PlannedArg::U32(indices[0]),
                    PlannedArg::U32(indices[1]),
                    PlannedArg::U32(indices[2]),
                    PlannedArg::U32(indices[3]),
                    PlannedArg::Slice {
                        buffer: 3,
                        elements: MULTIGRID_SUBGROUP_BATCHES_V1 * CHANNELS_V1,
                    },
                    PlannedArg::Slice {
                        buffer: 4,
                        elements: MULTIGRID_SUBGROUP_BATCHES_V1 * CHANNELS_V1,
                    },
                    PlannedArg::Slice {
                        buffer: 5,
                        elements: MULTIGRID_SUBGROUP_BATCHES_V1 * CHANNELS_V1,
                    },
                ],
            }
        }
        "gfx950_content_sparse_attention" | "gfx950_compressed_hybrid_attention" => {
            if [SPARSE_ATTENTION, SPARSE_ATTENTION_SIMT].contains(&case) {
                let mut physical_q = Vec::new();
                let mut key = Vec::new();
                let mut value = Vec::new();
                let mut scores = Vec::new();
                let mut expected_output = Vec::new();
                let mut expected_selected = Vec::new();
                for batch in 0..MULTIGRID_WAVE_BATCHES_V1 {
                    let (logical_batch_q, physical_batch_q) = repeated_attention_query(batch);
                    let batch_key =
                        nonuniform_fp8(ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1, 2, batch);
                    let batch_value = nonuniform_fp8(ATTENTION_TOKENS_V1 * CHANNELS_V1, 3, batch);
                    let mut batch_scores = deterministic_floats(
                        ATTENTION_TOKENS_V1,
                        4 + batch,
                        0.45 + batch as f32 * 0.01,
                    );
                    for (token, score) in batch_scores.iter_mut().enumerate() {
                        *score += ((token * 7 + batch * 3) % 17) as f32 * 0.01;
                    }
                    let expected = attention_reference::content_sparse_attention_reference_v1(
                        &logical_batch_q,
                        &batch_key,
                        &batch_value,
                        &batch_scores,
                    )
                    .map_err(|error| format!("sparse reference failed: {error:?}"))?;
                    physical_q.extend(physical_batch_q);
                    key.extend(batch_key);
                    value.extend(batch_value);
                    scores.extend(batch_scores);
                    expected_output.extend(expected.output);
                    expected_selected.extend(expected.selected);
                }
                LaunchPlan {
                    label: case.label.into(),
                    buffers: vec![
                        input("q", &physical_q),
                        input("k", &key),
                        input("v", &value),
                        input("content_scores", &scores),
                        f32_output("output", expected_output, 5.0e-3),
                        output("selected_output", ExpectedOutput::U32(expected_selected)),
                    ],
                    args: (0..6)
                        .map(|buffer| PlannedArg::Slice {
                            buffer,
                            elements: [
                                physical_q.len(),
                                key.len(),
                                value.len(),
                                scores.len(),
                                MULTIGRID_WAVE_BATCHES_V1 * CHANNELS_V1,
                                MULTIGRID_WAVE_BATCHES_V1 * SELECTED_TOKENS_V1,
                            ][buffer],
                        })
                        .collect(),
                }
            } else {
                let mut physical_q = Vec::new();
                let mut key = Vec::new();
                let mut value = Vec::new();
                let mut bias = Vec::new();
                let mut expected_output = Vec::new();
                for batch in 0..MULTIGRID_WAVE_BATCHES_V1 {
                    let (logical_batch_q, physical_batch_q) = repeated_attention_query(batch);
                    let batch_key =
                        nonuniform_fp8(ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1, 2, batch);
                    let batch_value = nonuniform_fp8(ATTENTION_TOKENS_V1 * CHANNELS_V1, 3, batch);
                    let batch_bias = deterministic_floats(
                        ATTENTION_TOKENS_V1,
                        7 + batch,
                        0.25 + batch as f32 * 0.01,
                    );
                    let expected = attention_reference::compressed_hybrid_attention_reference_v1(
                        &logical_batch_q,
                        &batch_key,
                        &batch_value,
                        &batch_bias,
                    )
                    .map_err(|error| format!("hybrid reference failed: {error:?}"))?;
                    physical_q.extend(physical_batch_q);
                    key.extend(batch_key);
                    value.extend(batch_value);
                    bias.extend(batch_bias);
                    expected_output.extend(expected);
                }
                let lengths = [
                    physical_q.len(),
                    key.len(),
                    value.len(),
                    bias.len(),
                    MULTIGRID_WAVE_BATCHES_V1 * CHANNELS_V1,
                ];
                LaunchPlan {
                    label: case.label.into(),
                    buffers: vec![
                        input("q", &physical_q),
                        input("k", &key),
                        input("v", &value),
                        input("token_bias", &bias),
                        f32_output("output", expected_output, 5.0e-3),
                    ],
                    args: (0..5)
                        .map(|buffer| PlannedArg::Slice {
                            buffer,
                            elements: lengths[buffer],
                        })
                        .collect(),
                }
            }
        }
        "gfx950_attnres_aggregate" => {
            let mut values = Vec::new();
            let mut logits = Vec::new();
            let mut expected = Vec::new();
            for batch in 0..MULTIGRID_SUBGROUP_BATCHES_V1 {
                let batch_values = deterministic_floats(
                    MIXING_STREAMS_V1 * CHANNELS_V1,
                    8 + batch,
                    0.45 + batch as f32 * 0.003,
                );
                let batch_logits = deterministic_floats(
                    MIXING_STREAMS_V1 * CHANNELS_V1,
                    9 + batch,
                    0.55 + batch as f32 * 0.004,
                );
                expected.extend(
                    attention_reference::attnres_aggregate_reference_v1(
                        &batch_values,
                        &batch_logits,
                    )
                    .map_err(|error| format!("AttnRes reference failed: {error:?}"))?,
                );
                values.extend(batch_values);
                logits.extend(batch_logits);
            }
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("depth_values", &values),
                    input("depth_logits", &logits),
                    f32_output("output", expected, 3.0e-3),
                ],
                args: (0..3)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: [
                            values.len(),
                            logits.len(),
                            MULTIGRID_SUBGROUP_BATCHES_V1 * CHANNELS_V1,
                        ][buffer],
                    })
                    .collect(),
            }
        }
        "gfx950_four_branch_residual" => {
            let mut residual = Vec::new();
            let mut branches = Vec::new();
            let mut gates = Vec::new();
            let mut expected = Vec::new();
            for batch in 0..MULTIGRID_SUBGROUP_BATCHES_V1 {
                let batch_residual =
                    deterministic_floats(CHANNELS_V1, 10 + batch, 0.30 + batch as f32 * 0.002);
                let batch_branches = deterministic_floats(
                    MIXING_STREAMS_V1 * CHANNELS_V1,
                    11 + batch,
                    0.40 + batch as f32 * 0.003,
                );
                let batch_gates = deterministic_floats(
                    MIXING_STREAMS_V1 * CHANNELS_V1,
                    12 + batch,
                    0.50 + batch as f32 * 0.004,
                );
                expected.extend(
                    attention_reference::four_branch_residual_reference_v1(
                        &batch_residual,
                        &batch_branches,
                        &batch_gates,
                    )
                    .map_err(|error| format!("residual reference failed: {error:?}"))?,
                );
                residual.extend(batch_residual);
                branches.extend(batch_branches);
                gates.extend(batch_gates);
            }
            let lengths = [
                residual.len(),
                branches.len(),
                gates.len(),
                MULTIGRID_SUBGROUP_BATCHES_V1 * CHANNELS_V1,
            ];
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("residual", &residual),
                    input("branches", &branches),
                    input("gate_logits", &gates),
                    f32_output("output", expected, 3.0e-3),
                ],
                args: (0..4)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: lengths[buffer],
                    })
                    .collect(),
            }
        }
        "gfx950_mhc_sinkhorn_mix" => {
            let mut streams = Vec::new();
            let mut logits = Vec::new();
            let mut expected = Vec::new();
            for batch in 0..MULTIGRID_WAVE_BATCHES_V1 {
                let batch_streams = deterministic_floats(
                    MIXING_STREAMS_V1 * CHANNELS_V1,
                    13 + batch,
                    0.45 + batch as f32 * 0.01,
                );
                let batch_logits = deterministic_floats(
                    MIXING_STREAMS_V1 * MIXING_STREAMS_V1,
                    14 + batch,
                    0.30 + batch as f32 * 0.01,
                );
                expected.extend(
                    attention_reference::mhc_sinkhorn_mix_reference_v1(
                        &batch_streams,
                        &batch_logits,
                    )
                    .map_err(|error| format!("mHC reference failed: {error:?}"))?,
                );
                streams.extend(batch_streams);
                logits.extend(batch_logits);
            }
            LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("streams", &streams),
                    input("mixing_logits", &logits),
                    f32_output("output", expected, 3.0e-3),
                ],
                args: (0..3)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: [
                            streams.len(),
                            logits.len(),
                            MULTIGRID_WAVE_BATCHES_V1 * MIXING_STREAMS_V1 * CHANNELS_V1,
                        ][buffer],
                    })
                    .collect(),
            }
        }
        _ => return Err(format!("unknown attention case {}", case.export).into()),
    };
    Ok(vec![plan])
}

#[cfg(feature = "hardware-test-hooks")]
fn make_activations() -> Vec<u8> {
    const CODES: [u8; 5] = [0xa, 0x9, 0x0, 0x1, 0x2];
    (0..SYSTEM_BATCHES * TOKENS * HIDDEN)
        .map(|index| {
            let batch = index / (TOKENS * HIDDEN);
            let within_batch = index % (TOKENS * HIDDEN);
            let token = within_batch / HIDDEN;
            let depth = index % HIDDEN;
            CODES[(batch * 4 + token * 3 + depth * 2 + 1) % CODES.len()]
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn make_expert_weights() -> Vec<u8> {
    const CODES: [u8; 5] = [0xb0, 0xa8, 0x00, 0x28, 0x30];
    (0..SYSTEM_BATCHES * ALL_EXPERTS * HIDDEN * OUTPUT)
        .map(|index| {
            let batch = index / (ALL_EXPERTS * HIDDEN * OUTPUT);
            let within_batch = index % (ALL_EXPERTS * HIDDEN * OUTPUT);
            let expert = within_batch / (HIDDEN * OUTPUT);
            let within = within_batch % (HIDDEN * OUTPUT);
            let depth = within / OUTPUT;
            let column = within % OUTPUT;
            CODES[(batch * 3 + expert * 2 + depth * 3 + column * 4) % CODES.len()]
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn make_router_weights() -> Vec<f32> {
    (0..SYSTEM_BATCHES * EXPERTS * HIDDEN)
        .map(|index| {
            let batch = index / (EXPERTS * HIDDEN);
            let within_batch = index % (EXPERTS * HIDDEN);
            let expert = within_batch / HIDDEN;
            let depth = index % HIDDEN;
            (((expert + 1) * (depth % 7) + depth / 13 + batch * 5) % 9) as f32 / 64.0 - 4.0 / 64.0
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn moe_fixture() -> (
    Vec<u8>,
    Vec<u8>,
    Vec<f32>,
    systems_reference::MoeRoutingReference,
) {
    let activations = make_activations();
    let expert_weights = make_expert_weights();
    let router_weights = make_router_weights();
    let routing = systems_reference::batched_moe_routing_reference(&activations, &router_weights);
    (activations, expert_weights, router_weights, routing)
}

#[cfg(feature = "hardware-test-hooks")]
fn hash_gram(gram: &[i32]) -> u64 {
    gram.iter()
        .fold(1_469_598_103_934_665_603_u64, |hash, value| {
            (hash ^ (*value as u32 as u64)).wrapping_mul(1_099_511_628_211)
        })
}

#[cfg(feature = "hardware-test-hooks")]
fn make_muon_shards() -> Vec<f32> {
    (0..SYSTEM_BATCHES * GRADIENT_SHARDS * MUON_ELEMENTS)
        .map(|index| {
            let batch = index / (GRADIENT_SHARDS * MUON_ELEMENTS);
            let within_batch = index % (GRADIENT_SHARDS * MUON_ELEMENTS);
            let shard = within_batch / MUON_ELEMENTS;
            let element = within_batch % MUON_ELEMENTS;
            0.025
                * ((shard + 1) as f32 * (((element * 5 + shard * 3 + batch * 2) % 11) as f32 - 5.0))
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn systems_plans(case: AdvancedCase) -> Result<Vec<LaunchPlan>, BoxError> {
    match case.export {
        "gfx950_moe_route_fp4_t16_e4_k2_v1" => {
            let (activations, _, router_weights, routing) = moe_fixture();
            let lengths = [
                activations.len(),
                router_weights.len(),
                SYSTEM_BATCHES * TOKENS * TOP_K,
                SYSTEM_BATCHES * TOKENS * TOP_K,
                SYSTEM_BATCHES * EXPERTS,
                SYSTEM_BATCHES * EXPERTS * DISPATCH_CAPACITY,
            ];
            Ok(vec![LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("activations", &activations),
                    input("router_weights", &router_weights),
                    output("top_experts", ExpectedOutput::U32(routing.top_experts)),
                    f32_output("top_weights", routing.top_weights, 2.0e-6),
                    output("expert_counts", ExpectedOutput::U32(routing.expert_counts)),
                    output("dispatch", ExpectedOutput::I32(routing.dispatch)),
                ],
                args: (0..6)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: lengths[buffer],
                    })
                    .collect(),
            }])
        }
        "gfx950_moe_expert_rank_fp4_fp8_v1" => {
            let (activations, weights, _, routing) = moe_fixture();
            let mut plans = Vec::new();
            for (rank, first_expert, include_shared) in [(0, 0, true), (1, 2, false)] {
                let expected = systems_reference::batched_moe_rank_reference(
                    &activations,
                    &weights,
                    &routing,
                    first_expert,
                    include_shared,
                );
                let buffers = vec![
                    input("activations", &activations),
                    input("expert_weights", &weights),
                    input("top_experts", &routing.top_experts),
                    input("top_weights", &routing.top_weights),
                    f32_output("output", expected, 3.0e-3),
                ];
                plans.push(LaunchPlan {
                    label: format!("{} rank {rank}", case.label),
                    args: vec![
                        PlannedArg::Slice {
                            buffer: 0,
                            elements: activations.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 1,
                            elements: weights.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 2,
                            elements: routing.top_experts.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 3,
                            elements: routing.top_weights.len(),
                        },
                        PlannedArg::U32(first_expert as u32),
                        PlannedArg::U32(u32::from(include_shared)),
                        PlannedArg::Slice {
                            buffer: 4,
                            elements: SYSTEM_BATCHES * TOKENS * OUTPUT,
                        },
                    ],
                    buffers,
                });
            }
            for first_expert in [3_u32, u32::MAX] {
                let canary = vec![0.25_f32; SYSTEM_BATCHES * TOKENS * OUTPUT];
                let mut preserved_output = f32_output("output", canary.clone(), 0.0);
                let canary_bytes = value_bytes(&canary);
                preserved_output.initial[GUARD_BYTES..GUARD_BYTES + canary_bytes.len()]
                    .copy_from_slice(&canary_bytes);
                let buffers = vec![
                    input("activations", &activations),
                    input("expert_weights", &weights),
                    input("top_experts", &routing.top_experts),
                    input("top_weights", &routing.top_weights),
                    preserved_output,
                ];
                plans.push(LaunchPlan {
                    label: format!("{} invalid first_expert {first_expert}", case.label),
                    args: vec![
                        PlannedArg::Slice {
                            buffer: 0,
                            elements: activations.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 1,
                            elements: weights.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 2,
                            elements: routing.top_experts.len(),
                        },
                        PlannedArg::Slice {
                            buffer: 3,
                            elements: routing.top_weights.len(),
                        },
                        PlannedArg::U32(first_expert),
                        PlannedArg::U32(1),
                        PlannedArg::Slice {
                            buffer: 4,
                            elements: SYSTEM_BATCHES * TOKENS * OUTPUT,
                        },
                    ],
                    buffers,
                });
            }
            Ok(plans)
        }
        "gfx950_combine_expert_ranks_v1" => {
            let (activations, weights, _, routing) = moe_fixture();
            let rank0 = systems_reference::batched_moe_rank_reference(
                &activations,
                &weights,
                &routing,
                0,
                true,
            );
            let rank1 = systems_reference::batched_moe_rank_reference(
                &activations,
                &weights,
                &routing,
                2,
                false,
            );
            let combine_elements = COMBINE_BATCHES * TOKENS * OUTPUT;
            let rank0 = rank0[..combine_elements].to_vec();
            let rank1 = rank1[..combine_elements].to_vec();
            let expected = rank0.iter().zip(&rank1).map(|(a, b)| a + b).collect();
            Ok(vec![LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("rank0", &rank0),
                    input("rank1", &rank1),
                    f32_output("output", expected, 3.0e-3),
                ],
                args: (0..3)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: combine_elements,
                    })
                    .collect(),
            }])
        }
        "gfx950_speculative_transaction_v1" => {
            let mut target = vec![0; SYSTEM_BATCHES * DRAFT_STEPS];
            let mut draft = vec![0; SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS];
            let mut scores = vec![0.0; SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS];
            let mut thresholds = vec![0.0; SYSTEM_BATCHES * DRAFT_STEPS];
            let mut base = vec![0.0; SYSTEM_BATCHES * STATE_WIDTH];
            let mut deltas = vec![0.0; SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS * STATE_WIDTH];
            for batch in 0..SYSTEM_BATCHES {
                let target_base = batch * DRAFT_STEPS;
                let transaction_base = batch * CANDIDATES * DRAFT_STEPS;
                let state_base = batch * STATE_WIDTH;
                let delta_base = batch * CANDIDATES * DRAFT_STEPS * STATE_WIDTH;
                for step in 0..DRAFT_STEPS {
                    target[target_base + step] = 11 + (batch * 17 + step) as i32;
                    thresholds[target_base + step] =
                        0.25 + 0.1 * step as f32 + 0.01 * (batch % 3) as f32;
                }
                for element in 0..STATE_WIDTH {
                    base[state_base + element] =
                        0.125 * (element as f32 - 3.0) + 0.03125 * batch as f32;
                }
                for candidate in 0..CANDIDATES {
                    for step in 0..DRAFT_STEPS {
                        let transaction = transaction_base + candidate * DRAFT_STEPS + step;
                        draft[transaction] = if step == (candidate + batch) % 5 {
                            90 + (batch * CANDIDATES + candidate) as i32
                        } else {
                            target[target_base + step]
                        };
                        scores[transaction] = 0.2 + 0.11 * ((batch + candidate + step) % 6) as f32;
                        for element in 0..STATE_WIDTH {
                            deltas[delta_base
                                + (candidate * DRAFT_STEPS + step) * STATE_WIDTH
                                + element] =
                                0.001 * (1 + batch + candidate + step * 2 + element) as f32;
                        }
                    }
                }
                for candidate in [4, 7] {
                    for step in 0..DRAFT_STEPS {
                        let transaction = transaction_base + candidate * DRAFT_STEPS + step;
                        draft[transaction] = target[target_base + step];
                        scores[transaction] = 0.9;
                    }
                }
            }
            let expected = systems_reference::batched_speculative_reference(
                &draft,
                &target,
                &scores,
                &thresholds,
                &base,
                &deltas,
            );
            let exact_mask = expected
                .committed
                .iter()
                .flat_map(|committed| std::iter::repeat_n(*committed == 0, STATE_WIDTH))
                .collect();
            let lengths = [
                draft.len(),
                target.len(),
                scores.len(),
                thresholds.len(),
                base.len(),
                deltas.len(),
                SYSTEM_BATCHES * CANDIDATES,
                SYSTEM_BATCHES * CANDIDATES,
                SYSTEM_BATCHES * CANDIDATES * STATE_WIDTH,
            ];
            Ok(vec![LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("draft_tokens", &draft),
                    input("target_tokens", &target),
                    input("draft_scores", &scores),
                    input("thresholds", &thresholds),
                    input("base_state", &base),
                    input("proposed_deltas", &deltas),
                    output("accepted_steps", ExpectedOutput::U32(expected.accepted)),
                    output("committed", ExpectedOutput::U32(expected.committed)),
                    output(
                        "output_state",
                        ExpectedOutput::F32 {
                            values: expected.state,
                            absolute_tolerance: 2.0e-7,
                            relative_tolerance: 0.0,
                            exact_mask: Some(exact_mask),
                        },
                    ),
                ],
                args: (0..9)
                    .map(|buffer| PlannedArg::Slice {
                        buffer,
                        elements: lengths[buffer],
                    })
                    .collect(),
            }])
        }
        "gfx950_qwen_ngram_gather_v1" => {
            let mut queries = vec![0; SYSTEM_BATCHES * QUERIES * NGRAM];
            let mut hashes = vec![0; SYSTEM_BATCHES * TABLE_SIZE];
            let mut grams = vec![-1; SYSTEM_BATCHES * TABLE_SIZE * NGRAM];
            let mut values = vec![-1; SYSTEM_BATCHES * TABLE_SIZE];
            let mut priorities = vec![-1; SYSTEM_BATCHES * TABLE_SIZE];
            for batch in 0..SYSTEM_BATCHES {
                let query_base = batch * QUERIES * NGRAM;
                let table_base = batch * TABLE_SIZE;
                let gram_base = batch * TABLE_SIZE * NGRAM;
                for query in 0..QUERIES {
                    let base = query_base + query * NGRAM;
                    queries[base] = 100 + (batch * QUERIES + query) as i32;
                    queries[base + 1] = 200 + ((batch + query) % 3) as i32;
                    queries[base + 2] = 300 + (batch * 5 + query * 2) as i32;
                }
                for query in 0..6 {
                    let base = query_base + query * NGRAM;
                    let hash = hash_gram(&queries[base..base + NGRAM]);
                    let slot = (hash as usize + 3) & (TABLE_SIZE - 1);
                    hashes[table_base + slot] = hash;
                    grams[gram_base + slot * NGRAM..gram_base + (slot + 1) * NGRAM]
                        .copy_from_slice(&queries[base..base + NGRAM]);
                    values[table_base + slot] = 1000 + (batch * QUERIES + query) as i32;
                    priorities[table_base + slot] = (batch + query) as i32 % 3;
                }
                let duplicate_hash = hash_gram(&queries[query_base..query_base + NGRAM]);
                hashes[table_base + 1] = duplicate_hash;
                grams[gram_base + NGRAM..gram_base + 2 * NGRAM]
                    .copy_from_slice(&queries[query_base..query_base + NGRAM]);
                values[table_base + 1] = 4242 + batch as i32;
                priorities[table_base + 1] = 0;
                hashes[table_base + 2] = duplicate_hash;
                grams[gram_base + 2 * NGRAM..gram_base + 3 * NGRAM].copy_from_slice(&[
                    999 + batch as i32,
                    998,
                    997,
                ]);
                values[table_base + 2] = 7777 + batch as i32;
                priorities[table_base + 2] = 99;
            }
            let expected = systems_reference::batched_ngram_reference(
                &queries,
                &hashes,
                &grams,
                &values,
                &priorities,
            );
            let lengths = [
                queries.len(),
                hashes.len(),
                grams.len(),
                values.len(),
                priorities.len(),
                SYSTEM_BATCHES * QUERIES,
            ];
            Ok(vec![
                LaunchPlan {
                    label: case.label.into(),
                    buffers: vec![
                        input("queries", &queries),
                        input("table_hashes", &hashes),
                        input("table_grams", &grams),
                        input("table_values", &values),
                        input("priorities", &priorities),
                        output("output", ExpectedOutput::I32(expected)),
                    ],
                    args: (0..6)
                        .map(|buffer| PlannedArg::Slice {
                            buffer,
                            elements: lengths[buffer],
                        })
                        .collect(),
                },
                ngram_integer_plan(case),
            ])
        }
        "gfx950_stage_gradient_shard_v1" => {
            let shards = make_stage_shards();
            Ok((0..GRADIENT_SHARDS)
                .map(|shard| {
                    let values = (0..SYSTEM_BATCHES)
                        .flat_map(|batch| {
                            let base = (batch * GRADIENT_SHARDS + shard) * MUON_ELEMENTS;
                            shards[base..base + MUON_ELEMENTS].iter().copied()
                        })
                        .collect::<Vec<_>>();
                    LaunchPlan {
                        label: format!("{} {shard}", case.label),
                        buffers: vec![
                            input("input", &values),
                            output(
                                "output",
                                ExpectedOutput::F32 {
                                    values,
                                    absolute_tolerance: 0.0,
                                    relative_tolerance: 0.0,
                                    exact_mask: Some(vec![true; SYSTEM_BATCHES * MUON_ELEMENTS]),
                                },
                            ),
                        ],
                        args: vec![
                            PlannedArg::Slice {
                                buffer: 0,
                                elements: SYSTEM_BATCHES * MUON_ELEMENTS,
                            },
                            PlannedArg::Slice {
                                buffer: 1,
                                elements: SYSTEM_BATCHES * MUON_ELEMENTS,
                            },
                        ],
                    }
                })
                .collect())
        }
        "gfx950_muon_update_4x4_v1" => {
            let shards = make_muon_shards();
            let expected = systems_reference::batched_muon_reference(&shards);
            Ok(vec![LaunchPlan {
                label: case.label.into(),
                buffers: vec![
                    input("shards", &shards),
                    f32_output("output", expected.update, 2.0e-6),
                    f32_output("output_norm", expected.norms, 2.0e-6),
                ],
                args: vec![
                    PlannedArg::Slice {
                        buffer: 0,
                        elements: shards.len(),
                    },
                    PlannedArg::Slice {
                        buffer: 1,
                        elements: SYSTEM_BATCHES * MUON_ELEMENTS,
                    },
                    PlannedArg::Slice {
                        buffer: 2,
                        elements: SYSTEM_BATCHES,
                    },
                ],
            }])
        }
        _ => Err(format!("unknown systems case {}", case.export).into()),
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn pack_mfma_blocked_output(logical_row_major: &[f32]) -> Result<Vec<f32>, BoxError> {
    require(
        logical_row_major.len() == 16 * 16,
        "MFMA blocked output requires one 16x16 tile",
    )?;
    let mut physical = vec![0.0_f32; 16 * 16];
    for component in 0..4 {
        for lane in 0..64 {
            let row = (lane / 16) * 4 + component;
            let column = lane % 16;
            physical[component * 64 + lane] = logical_row_major[row * 16 + column];
        }
    }
    Ok(physical)
}

// GPT-OSS tile variants retain every input/guard and expose row-major 16x16 outputs.
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_TILE: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS tile",
    ..GPT_OSS
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_ROUTER_TILE: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS router tile",
    ..GPT_OSS_ROUTER_COMPONENT
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_ATTENTION_TILE: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS attention tile",
    ..GPT_OSS_ATTENTION_COMPONENT
};
#[cfg(feature = "hardware-test-hooks")]
const GPT_OSS_EXPERT_TILE: AdvancedCase = AdvancedCase {
    label: "gfx950 GPT-OSS expert tile",
    ..GPT_OSS_EXPERT_COMPONENT
};

#[cfg(feature = "hardware-test-hooks")]
fn gpt_oss_plans(case: AdvancedCase) -> Result<Vec<LaunchPlan>, BoxError> {
    gpt_oss_plans_for_fixture(case, false)
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_oss_plans_for_fixture(
    case: AdvancedCase,
    finite_router_edge: bool,
) -> Result<Vec<LaunchPlan>, BoxError> {
    let inputs = if finite_router_edge {
        gpt_oss_reference::finite_router_edge_batch_inputs()
    } else {
        gpt_oss_reference::deterministic_batch_inputs()
    };
    let expected = gpt_oss_reference::reference_batch(&inputs);
    let row_major = [
        GPT_OSS_TILE,
        GPT_OSS_ROUTER_TILE,
        GPT_OSS_ATTENTION_TILE,
        GPT_OSS_EXPERT_TILE,
    ]
    .contains(&case);
    let expected_attention = expected
        .attention
        .chunks_exact(16 * 16)
        .map(|tile| {
            if row_major {
                Ok(tile.to_vec())
            } else {
                pack_mfma_blocked_output(tile)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let expected_expert = expected
        .expert
        .chunks_exact(16 * 16)
        .map(|tile| {
            if row_major {
                Ok(tile.to_vec())
            } else {
                pack_mfma_blocked_output(tile)
            }
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if finite_router_edge {
        require(
            expected.top4 == vec![[0, 1, 2, 3]; 16],
            "finite below-floor GPT-OSS router fixture lost stable ties",
        )?;
    } else {
        require(
            expected.top4.len() == 16 && expected.top4[0][0] == 127 && expected.top4[1][0] == 0,
            "deterministic GPT-OSS batch did not select alternating experts 127 and 0",
        )?;
    }
    let lengths = [
        inputs.hidden_f32.len(),
        inputs.router_f32.len(),
        inputs.query_bf16.len(),
        inputs.key_transposed_bf16.len(),
        inputs.value_f32.len(),
        inputs.sinks_f32.len(),
        inputs.expert_activation_blocks_fp4.len(),
        inputs.expert_weight_blocks_fp4.len(),
        inputs.activation_scales.len(),
        inputs.expert_weight_scales.len(),
        expected.attention.len(),
        expected.expert.len(),
        expected.packed_top4.len(),
    ];
    Ok(vec![LaunchPlan {
        label: if finite_router_edge {
            format!("{} finite below-floor router", case.label)
        } else {
            case.label.into()
        },
        buffers: vec![
            input("hidden_f32", &inputs.hidden_f32),
            input("router_f32", &inputs.router_f32),
            input("query_bf16", &inputs.query_bf16),
            input("key_transposed_bf16", &inputs.key_transposed_bf16),
            input("value_f32", &inputs.value_f32),
            input("sinks_f32", &inputs.sinks_f32),
            input(
                "expert_activation_blocks_fp4",
                &inputs.expert_activation_blocks_fp4,
            ),
            input("expert_weight_blocks_fp4", &inputs.expert_weight_blocks_fp4),
            input("activation_scales", &inputs.activation_scales),
            input("expert_weight_scales", &inputs.expert_weight_scales),
            f32_output("attention_output", expected_attention, 8.0e-3),
            f32_output("expert_output", expected_expert, 8.0e-3),
            output("packed_top4", ExpectedOutput::U32(expected.packed_top4)),
        ],
        args: (0..13)
            .map(|buffer| PlannedArg::Slice {
                buffer,
                elements: lengths[buffer],
            })
            .collect(),
    }])
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_oss_single_plan(case: AdvancedCase) -> Result<LaunchPlan, BoxError> {
    let inputs = gpt_oss_reference::deterministic_inputs();
    let expected = gpt_oss_reference::reference(&inputs);
    let expected_attention = pack_mfma_blocked_output(&expected.attention)?;
    let expected_expert = pack_mfma_blocked_output(&expected.expert)?;
    require(
        expected.top4[0] == 127,
        "deterministic GPT-OSS fixture did not select expert 127",
    )?;
    let lengths = [
        inputs.hidden_f32.len(),
        inputs.router_f32.len(),
        inputs.query_bf16.len(),
        inputs.key_transposed_bf16.len(),
        inputs.value_f32.len(),
        inputs.sinks_f32.len(),
        inputs.expert_activation_blocks_fp4.len(),
        inputs.expert_weight_blocks_fp4.len(),
        inputs.activation_scales.len(),
        inputs.expert_weight_scales.len(),
        expected.attention.len(),
        expected.expert.len(),
        1,
    ];
    Ok(LaunchPlan {
        label: case.label.into(),
        buffers: vec![
            input("hidden_f32", &inputs.hidden_f32),
            input("router_f32", &inputs.router_f32),
            input("query_bf16", &inputs.query_bf16),
            input("key_transposed_bf16", &inputs.key_transposed_bf16),
            input("value_f32", &inputs.value_f32),
            input("sinks_f32", &inputs.sinks_f32),
            input(
                "expert_activation_blocks_fp4",
                &inputs.expert_activation_blocks_fp4,
            ),
            input("expert_weight_blocks_fp4", &inputs.expert_weight_blocks_fp4),
            input("activation_scales", &inputs.activation_scales),
            input("expert_weight_scales", &inputs.expert_weight_scales),
            f32_output("attention_output", expected_attention, 8.0e-3),
            f32_output("expert_output", expected_expert, 8.0e-3),
            output(
                "packed_top4",
                ExpectedOutput::U32(vec![expected.packed_top4]),
            ),
        ],
        args: (0..13)
            .map(|buffer| PlannedArg::Slice {
                buffer,
                elements: lengths[buffer],
            })
            .collect(),
    })
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_oss_component_plans(case: AdvancedCase) -> Result<Vec<LaunchPlan>, BoxError> {
    gpt_oss_component_plans_for_fixture(case, false)
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_oss_component_plans_for_fixture(
    case: AdvancedCase,
    finite_router_edge: bool,
) -> Result<Vec<LaunchPlan>, BoxError> {
    let mut plans = gpt_oss_plans_for_fixture(
        if [
            GPT_OSS_ROUTER_TILE,
            GPT_OSS_ATTENTION_TILE,
            GPT_OSS_EXPERT_TILE,
        ]
        .contains(&case)
        {
            GPT_OSS_TILE
        } else {
            GPT_OSS
        },
        finite_router_edge,
    )?;
    let case = if case == GPT_OSS_ROUTER_TILE {
        GPT_OSS_ROUTER_COMPONENT
    } else if case == GPT_OSS_ATTENTION_TILE {
        GPT_OSS_ATTENTION_COMPONENT
    } else if case == GPT_OSS_EXPERT_TILE {
        GPT_OSS_EXPERT_COMPONENT
    } else {
        case
    };
    let full = plans.pop().ok_or("GPT-OSS launch plan is absent")?;
    require(plans.is_empty(), "GPT-OSS launch plan was duplicated")?;
    let indices: &[usize] = if case == GPT_OSS_ROUTER_COMPONENT {
        &[0, 1, 12]
    } else if case == GPT_OSS_ATTENTION_COMPONENT {
        &[2, 3, 4, 5, 10]
    } else if case == GPT_OSS_EXPERT_COMPONENT {
        &[6, 7, 8, 9, 12, 11]
    } else {
        return Err(format!("unknown GPT-OSS component {}", case.export).into());
    };
    let mut buffers = indices
        .iter()
        .map(|index| full.buffers[*index].clone())
        .collect::<Vec<_>>();
    if case == GPT_OSS_EXPERT_COMPONENT {
        let packed = buffers
            .get_mut(4)
            .ok_or("GPT-OSS packed route buffer is absent")?;
        let packed_values = match packed.expected.take() {
            Some(ExpectedOutput::U32(values)) => values,
            _ => return Err("GPT-OSS packed route reference changed kind".into()),
        };
        let packed_bytes = value_bytes(&packed_values);
        let start = packed.body_offset;
        packed.initial[start..start + packed_bytes.len()].copy_from_slice(&packed_bytes);
        packed.immutable = true;
    }
    let lengths = buffers
        .iter()
        .map(|buffer| buffer.elements)
        .collect::<Vec<_>>();
    Ok(vec![LaunchPlan {
        label: if finite_router_edge {
            format!("{} finite below-floor router", case.label)
        } else {
            case.label.into()
        },
        buffers,
        args: lengths
            .into_iter()
            .enumerate()
            .map(|(buffer, elements)| PlannedArg::Slice { buffer, elements })
            .collect(),
    }])
}

#[cfg(feature = "hardware-test-hooks")]
fn plans_for(case: AdvancedCase) -> Result<Vec<LaunchPlan>, BoxError> {
    if [GPT_OSS, GPT_OSS_PIPELINED, GPT_OSS_TILE].contains(&case) {
        let mut plans = gpt_oss_plans(case)?;
        // Add the edge fixture; never replace the original nonuniform 16-item case.
        plans.extend(gpt_oss_plans_for_fixture(case, true)?);
        return Ok(plans);
    }
    if [
        GPT_OSS_ROUTER_COMPONENT,
        GPT_OSS_ATTENTION_COMPONENT,
        GPT_OSS_EXPERT_COMPONENT,
        GPT_OSS_ROUTER_TILE,
        GPT_OSS_ATTENTION_TILE,
        GPT_OSS_EXPERT_TILE,
    ]
    .contains(&case)
    {
        let mut plans = gpt_oss_component_plans(case)?;
        if [GPT_OSS_ROUTER_COMPONENT, GPT_OSS_ROUTER_TILE].contains(&case) {
            plans.extend(gpt_oss_component_plans_for_fixture(case, true)?);
        }
        return Ok(plans);
    }
    if [
        KDA_DECODE,
        KDA_PREFILL,
        SPARSE_ATTENTION,
        SPARSE_ATTENTION_SIMT,
        DEEPSEEK_SPARSE_ATTENTION,
        HYBRID_ATTENTION,
        HYBRID_ATTENTION_SIMT,
        ATTNRES,
        FOUR_BRANCH,
        MHC,
    ]
    .contains(&case)
    {
        attention_plans(case)
    } else {
        systems_plans(case)
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "hardware-test-hooks")]
fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

#[cfg(feature = "hardware-test-hooks")]
fn explicit_kernarg(
    case: AdvancedCase,
    plan: &LaunchPlan,
    buffers: &[ReviewedHsaHardwareTestBufferV1],
) -> Result<Vec<u8>, BoxError> {
    require(
        plan.args.len() == case.args.len(),
        format!("{} launch argument count changed", plan.label),
    )?;
    let mut bytes = vec![0; kernarg_size(case.args)];
    let mut offset = 0;
    for (expected, actual) in case.args.iter().zip(&plan.args) {
        match (expected, actual) {
            (AbiArg::Slice, PlannedArg::Slice { buffer, elements }) => {
                require(
                    *elements == plan.buffers[*buffer].elements,
                    format!("{} slice element count changed", plan.buffers[*buffer].name),
                )?;
                put_u64(
                    &mut bytes,
                    offset,
                    buffers[*buffer].device_address(plan.buffers[*buffer].body_offset)?,
                );
                put_u64(&mut bytes, offset + 8, *elements as u64);
                offset += 16;
            }
            (AbiArg::Pointer, PlannedArg::Pointer { buffer }) => {
                put_u64(
                    &mut bytes,
                    offset,
                    buffers[*buffer].device_address(plan.buffers[*buffer].body_offset)?,
                );
                offset += 8;
            }
            (AbiArg::U32, PlannedArg::U32(value)) => {
                put_u32(&mut bytes, offset, *value);
                offset += 4;
            }
            _ => return Err(format!("{} kernarg kind changed", plan.label).into()),
        }
    }
    Ok(bytes)
}

#[cfg(feature = "hardware-test-hooks")]
struct RuntimeKernarg {
    pointer: std::ptr::NonNull<u8>,
    layout: std::alloc::Layout,
}

#[cfg(feature = "hardware-test-hooks")]
impl RuntimeKernarg {
    fn new(size: usize) -> Result<Self, BoxError> {
        let layout = std::alloc::Layout::from_size_align(size, HSA_KERNARG_ALIGNMENT as usize)?;
        // SAFETY: layout is valid and this owner deallocates the result once.
        let pointer = std::ptr::NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) })
            .ok_or("failed to allocate aligned advanced-kernel kernarg")?;
        Ok(Self { pointer, layout })
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is live and exactly layout.size() bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.layout.size()) }
    }
}

#[cfg(feature = "hardware-test-hooks")]
impl Drop for RuntimeKernarg {
    fn drop(&mut self) {
        // SAFETY: this owner deallocates its exact live allocation once.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

#[cfg(feature = "hardware-test-hooks")]
unsafe fn dispatch(
    case: AdvancedCase,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    resolution: &HsaKernelResolutionObservationV1,
    explicit: &[u8],
) -> Result<(), BoxError> {
    // SAFETY: preserve the existing fixed geometry and reviewed launch boundary.
    unsafe {
        dispatch_with_grid(
            case,
            adapter,
            executable,
            kernel,
            resolution,
            explicit,
            case.grid_x(),
        )
    }
}

#[cfg(feature = "hardware-test-hooks")]
unsafe fn dispatch_with_grid(
    case: AdvancedCase,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    executable: &ReviewedHsaExecutableV1,
    kernel: &ReviewedHsaKernelV1,
    resolution: &HsaKernelResolutionObservationV1,
    explicit: &[u8],
    grid_x: u32,
) -> Result<(), BoxError> {
    require(grid_x != 0, "zero workgroups must not dispatch")?;
    let size = kernarg_size(case.args);
    require(
        resolution.export_symbol() == case.export
            && resolution.kernarg_segment_size() == size as u64
            && resolution.kernarg_segment_alignment() == HSA_KERNARG_ALIGNMENT
            && resolution.group_segment_size() == case.static_lds_bytes,
        format!(
            "runtime resolution differs from the exact {} ABI",
            case.label
        ),
    )?;
    let geometry = HsaLaunchGeometryV1::new([grid_x, 1, 1], [case.workgroup_x, 1, 1], 0);
    let mut storage = RuntimeKernarg::new(size)?;
    let kernarg = storage.bytes_mut();
    kernarg.copy_from_slice(explicit);
    let original = explicit.to_vec();
    // SAFETY: inspection admitted this exact explicit-only single-kernel ABI;
    // all buffers remain live through synchronous completion.
    unsafe {
        let initialization = adapter
            .initialize_implicit_kernarg(executable, kernel, geometry, size, size, 0, kernarg)?;
        require(
            initialization.initialized()
                && initialization.explicit_byte_len() == size as u64
                && initialization.implicit_byte_offset() == size as u64
                && initialization.implicit_byte_len() == 0
                && kernarg == original,
            format!("{} implicit initialization changed the ABI", case.label),
        )?;
        let completion = adapter.launch_and_wait(executable, kernel, geometry, kernarg)?;
        require(
            completion.completed(),
            format!("{} dispatch did not complete", case.label),
        )?;
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, BoxError> {
    require(
        bytes.len() % 4 == 0,
        "f32 output byte length is not aligned",
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(feature = "hardware-test-hooks")]
fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>, BoxError> {
    require(
        bytes.len() % 4 == 0,
        "u32 output byte length is not aligned",
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(feature = "hardware-test-hooks")]
fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>, BoxError> {
    require(
        bytes.len() % 4 == 0,
        "i32 output byte length is not aligned",
    )?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

#[cfg(feature = "hardware-test-hooks")]
fn verify_buffer(plan: &PlannedBuffer, actual: &[u8]) -> Result<(), BoxError> {
    if plan.immutable {
        return require(
            actual == plan.initial,
            format!("{} input was modified", plan.name),
        );
    }
    let expected = plan
        .expected
        .as_ref()
        .ok_or_else(|| format!("{} output has no oracle", plan.name))?;
    require(
        actual.len() == plan.initial.len()
            && actual[..GUARD_BYTES]
                .iter()
                .all(|byte| *byte == PREFIX_BYTE)
            && actual[actual.len() - GUARD_BYTES..]
                .iter()
                .all(|byte| *byte == SUFFIX_BYTE),
        format!("{} output canary changed", plan.name),
    )?;
    let body = &actual[GUARD_BYTES..actual.len() - GUARD_BYTES];
    match expected {
        ExpectedOutput::F32 {
            values,
            absolute_tolerance,
            relative_tolerance,
            exact_mask,
        } => {
            let actual = decode_f32(body)?;
            let mut maximum_error = 0.0_f32;
            for (index, (actual, expected)) in actual.iter().zip(values).enumerate() {
                if exact_mask.as_ref().is_some_and(|mask| mask[index]) {
                    require(
                        actual.to_bits() == expected.to_bits(),
                        format!("{}[{index}] is not bitwise exact", plan.name),
                    )?;
                    continue;
                }
                let error = (actual - expected).abs();
                let tolerance = absolute_tolerance + relative_tolerance * expected.abs();
                require(
                    actual.is_finite() && expected.is_finite() && error <= tolerance,
                    format!(
                        "{}[{index}] mismatch: actual={actual}, expected={expected}, tolerance={tolerance}",
                        plan.name
                    ),
                )?;
                maximum_error = maximum_error.max(error);
            }
            println!(
                "PASS {} outputs={} max_absolute_error={maximum_error:.9e}",
                plan.name,
                values.len()
            );
        }
        ExpectedOutput::U32(values) => {
            require(
                decode_u32(body)? == *values,
                format!("{} exact u32 output mismatch", plan.name),
            )?;
            println!("PASS {} exact_u32_outputs={}", plan.name, values.len());
        }
        ExpectedOutput::I32(values) => {
            require(
                decode_i32(body)? == *values,
                format!("{} exact i32 output mismatch", plan.name),
            )?;
            println!("PASS {} exact_i32_outputs={}", plan.name, values.len());
        }
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn execute_plan(
    case: AdvancedCase,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    bytes: &[u8],
    digest: PayloadDigest,
    plan: LaunchPlan,
) -> Result<(), BoxError> {
    let buffers = plan
        .buffers
        .iter()
        .map(|buffer| adapter.allocate_hardware_test_buffer(&buffer.initial))
        .collect::<Result<Vec<_>, _>>()?;
    let explicit = explicit_kernarg(case, &plan, &buffers)?;
    // SAFETY: immutable digest-pinned bytes and allocations are retained until
    // synchronous dispatch and the sole consuming unload complete.
    let (executable, load) = unsafe { adapter.load_executable(bytes, digest) }?;
    let executable_identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            format!("{} load observation changed", plan.label),
        )?;
        // SAFETY: profile inspection admitted exactly this export/descriptor.
        let (kernels, resolutions) =
            unsafe { adapter.resolve_kernel_set(&executable, [case.export]) }?;
        let kernel = kernels
            .get(0)
            .ok_or_else(|| format!("runtime omitted {}", plan.label))?;
        require(
            kernels.len() == 1
                && resolutions.len() == 1
                && resolutions[0].executable_object() == executable_identity,
            format!("runtime resolved a substituted {} kernel", plan.label),
        )?;
        // SAFETY: dispatch owns the reviewed raw launch boundary.
        unsafe {
            dispatch(
                case,
                adapter,
                &executable,
                kernel,
                &resolutions[0],
                &explicit,
            )?;
        }
        for (planned, actual) in plan.buffers.iter().zip(&buffers) {
            verify_buffer(planned, &actual.read_after_synchronous_dispatch())?;
        }
        Ok(())
    })();
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == executable_identity,
        format!("{} executable was not released", plan.label),
    )?;
    execution
}

#[cfg(feature = "hardware-test-hooks")]
#[cfg(feature = "hardware-test-hooks")]
fn inspect_gpt_unfused_profile(bytes: &[u8]) -> Result<(), BoxError> {
    let bound = fe2o3_hsaco::inspect_and_bind_kernel_descriptors(bytes)?;
    let inspected = bound.inspection();
    require(
        inspected.code_object_version() == CodeObjectVersion::V6
            && inspected.target().processor() == "gfx950"
            && inspected.target().xnack() == Some(FeatureState::Disabled)
            && !inspected.has_printf_metadata(),
        "unfused GPT-OSS comparator must be printf-free gfx950:xnack- COV6",
    )?;
    let cases = [
        GPT_OSS_UNFUSED_ROUTER,
        GPT_OSS_UNFUSED_ATTENTION,
        GPT_OSS_UNFUSED_EXPERT,
    ];
    require(
        inspected.kernels().len() == cases.len(),
        "unfused GPT-OSS comparator kernel count changed",
    )?;
    for case in cases {
        let kernel = inspected
            .kernels()
            .iter()
            .find(|kernel| kernel.name() == case.export)
            .ok_or_else(|| format!("unfused comparator omitted {}", case.export))?;
        require(
            kernel.symbol() == case.descriptor
                && kernel.kernarg_segment_size() == kernarg_size(case.args) as u64
                && kernel.kernarg_segment_alignment() == METADATA_KERNARG_ALIGNMENT
                && kernel.group_segment_fixed_size() == 0
                && kernel.wavefront_size() == 64
                && kernel.max_flat_workgroup_size() == 64
                && !kernel.uses_dynamic_stack(),
            format!("{} metadata changed", case.label),
        )?;
        let arguments = kernel
            .explicit_arguments()
            .iter()
            .map(|argument| (argument.offset(), argument.size(), argument.value_kind()))
            .collect::<Vec<_>>();
        require(
            arguments == expected_metadata_arguments(case.args),
            format!("{} pointer ABI changed", case.label),
        )?;
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn pointer_kernarg(
    case: AdvancedCase,
    plan: &LaunchPlan,
    buffers: &[ReviewedHsaHardwareTestBufferV1],
    indices: &[usize],
) -> Result<Vec<u8>, BoxError> {
    require(
        indices.len() == case.args.len()
            && case
                .args
                .iter()
                .all(|argument| *argument == AbiArg::Pointer),
        format!("{} pointer launch ABI changed", case.label),
    )?;
    let mut explicit = vec![0_u8; kernarg_size(case.args)];
    for (slot, buffer) in indices.iter().enumerate() {
        put_u64(
            &mut explicit,
            slot * 8,
            buffers[*buffer].device_address(plan.buffers[*buffer].body_offset)?,
        );
    }
    Ok(explicit)
}

#[cfg(feature = "hardware-test-hooks")]
fn gpt_unfused_profile_plans() -> Result<Vec<(AdvancedCase, LaunchPlan)>, BoxError> {
    let full = gpt_oss_single_plan(GPT_OSS_UNFUSED_ROUTER)?;

    let label = "gpt-oss-120b-batch1-layer-tile-exact-unfused";
    let router = LaunchPlan {
        label: label.to_owned(),
        buffers: [0, 1, 12]
            .into_iter()
            .map(|index| full.buffers[index].clone())
            .collect(),
        args: (0..3)
            .map(|buffer| PlannedArg::Pointer { buffer })
            .collect(),
    };
    let attention = LaunchPlan {
        label: label.to_owned(),
        buffers: [2, 3, 4, 5, 10]
            .into_iter()
            .map(|index| full.buffers[index].clone())
            .collect(),
        args: (0..5)
            .map(|buffer| PlannedArg::Pointer { buffer })
            .collect(),
    };

    let mut packed = full.buffers[12].clone();
    let packed_values = match packed.expected.clone() {
        Some(ExpectedOutput::U32(values)) => values,
        _ => return Err("GPT-OSS packed top-4 reference changed kind".into()),
    };
    let packed_bytes = value_bytes(&packed_values);
    require(
        packed_bytes.len() == packed.elements * std::mem::size_of::<u32>(),
        "GPT-OSS packed top-4 reference changed size",
    )?;
    let packed_start = packed.body_offset;
    packed.initial[packed_start..packed_start + packed_bytes.len()].copy_from_slice(&packed_bytes);
    packed.immutable = true;
    packed.expected = None;
    let expert = LaunchPlan {
        label: label.to_owned(),
        buffers: vec![
            full.buffers[6].clone(),
            full.buffers[7].clone(),
            full.buffers[8].clone(),
            full.buffers[9].clone(),
            packed,
            full.buffers[11].clone(),
        ],
        args: (0..6)
            .map(|buffer| PlannedArg::Pointer { buffer })
            .collect(),
    };

    Ok(vec![
        (GPT_OSS_UNFUSED_ROUTER, router),
        (GPT_OSS_UNFUSED_ATTENTION, attention),
        (GPT_OSS_UNFUSED_EXPERT, expert),
    ])
}

#[cfg(feature = "hardware-test-hooks")]
fn run_gpt_unfused_comparator() -> Result<(), BoxError> {
    require_gpt_runtime_visibility()?;
    let (bytes, digest) = read_pinned_hsaco()?;
    inspect_gpt_unfused_profile(&bytes)?;
    let plan = gpt_oss_single_plan(GPT_OSS_UNFUSED_ROUTER)?;

    let context = GpuContext::new(0)?;
    let mut adapter = ReviewedHsaRuntimeAdapterV1::new_gfx950(context)?;
    require(
        adapter.environment().physical_device().target().processor() == "gfx950"
            && adapter.environment().physical_device().target().xnack()
                == Some(FeatureState::Disabled),
        "unfused GPT-OSS comparator requires gfx950:xnack-",
    )?;
    let buffers = plan
        .buffers
        .iter()
        .map(|buffer| adapter.allocate_hardware_test_buffer(&buffer.initial))
        .collect::<Result<Vec<_>, _>>()?;
    let cases = [
        GPT_OSS_UNFUSED_ROUTER,
        GPT_OSS_UNFUSED_ATTENTION,
        GPT_OSS_UNFUSED_EXPERT,
    ];
    let mappings: [&[usize]; 3] = [&[0, 1, 12], &[2, 3, 4, 5, 10], &[6, 7, 8, 9, 12, 11]];
    let (executable, load) = unsafe { adapter.load_executable(&bytes, digest) }?;
    let executable_identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            "unfused comparator load observation changed",
        )?;
        let exports = cases.map(|case| case.export);
        let (kernels, resolutions) = unsafe { adapter.resolve_kernel_set(&executable, exports) }?;
        require(
            kernels.len() == cases.len()
                && resolutions.len() == cases.len()
                && resolutions
                    .iter()
                    .all(|resolution| resolution.executable_object() == executable_identity),
            "unfused comparator resolved a substituted kernel set",
        )?;
        for index in 0..cases.len() {
            let explicit = pointer_kernarg(cases[index], &plan, &buffers, mappings[index])?;
            unsafe {
                dispatch(
                    cases[index],
                    &mut adapter,
                    &executable,
                    kernels
                        .get(index)
                        .ok_or("unfused comparator runtime omitted a kernel")?,
                    &resolutions[index],
                    &explicit,
                )?;
            }
        }
        for (planned, actual) in plan.buffers.iter().zip(&buffers) {
            verify_buffer(planned, &actual.read_after_synchronous_dispatch())?;
        }
        Ok(())
    })();
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == executable_identity,
        "unfused comparator executable was not released",
    )?;
    execution?;
    if let Some(config) = PerformanceConfig::from_environment()? {
        for (case, plan) in gpt_unfused_profile_plans()? {
            profile_plan(case, &mut adapter, &bytes, digest, plan, &config)?;
        }
    }
    println!("PASS gfx950 GPT-OSS exact unfused three-dispatch comparator");
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
fn lowercase_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn workload_identity(plan: &LaunchPlan) -> (String, serde_json::Value) {
    let mut digest = Sha256::new();
    let buffers = plan
        .buffers
        .iter()
        .map(|buffer| {
            digest.update(buffer.name.as_bytes());
            digest.update((buffer.initial.len() as u64).to_le_bytes());
            digest.update((buffer.body_offset as u64).to_le_bytes());
            digest.update((buffer.elements as u64).to_le_bytes());
            digest.update([u8::from(buffer.immutable)]);
            digest.update(&buffer.initial);
            let oracle = match buffer.expected.as_ref() {
                None => json!({"kind": "immutable-input"}),
                Some(ExpectedOutput::F32 {
                    values,
                    absolute_tolerance,
                    relative_tolerance,
                    exact_mask,
                }) => json!({
                    "kind": "cpu-reference-f32",
                    "elements": values.len(),
                    "absolute_tolerance": absolute_tolerance.to_string(),
                    "relative_tolerance": relative_tolerance.to_string(),
                    "bitwise_exact_elements": exact_mask
                        .as_ref()
                        .map(|mask| mask.iter().filter(|exact| **exact).count())
                        .unwrap_or(0),
                }),
                Some(ExpectedOutput::U32(values)) => {
                    json!({"kind": "cpu-reference-u32-exact", "elements": values.len()})
                }
                Some(ExpectedOutput::I32(values)) => {
                    json!({"kind": "cpu-reference-i32-exact", "elements": values.len()})
                }
            };
            json!({
                "name": buffer.name,
                "byte_len": buffer.initial.len(),
                "body_offset": buffer.body_offset,
                "elements": buffer.elements,
                "immutable": buffer.immutable,
                "initial_sha256": lowercase_hex(&Sha256::digest(&buffer.initial)),
                "oracle": oracle,
            })
        })
        .collect::<Vec<_>>();
    (lowercase_hex(&digest.finalize()), json!(buffers))
}

#[cfg(feature = "hardware-test-hooks")]
fn profile_plan(
    case: AdvancedCase,
    adapter: &mut ReviewedHsaRuntimeAdapterV1,
    bytes: &[u8],
    digest: PayloadDigest,
    plan: LaunchPlan,
    config: &PerformanceConfig,
) -> Result<(), BoxError> {
    let buffers = plan
        .buffers
        .iter()
        .map(|buffer| adapter.allocate_hardware_test_buffer(&buffer.initial))
        .collect::<Result<Vec<_>, _>>()?;
    let explicit = explicit_kernarg(case, &plan, &buffers)?;
    let (workload_sha256, buffer_identity) = workload_identity(&plan);
    let environment = {
        let observed = adapter.environment();
        json!({
            "hostname": std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_owned()),
            "process_id": std::process::id(),
            "rocr_visible_devices": std::env::var("ROCR_VISIBLE_DEVICES").ok(),
            "hip_visible_devices": std::env::var("HIP_VISIBLE_DEVICES").ok(),
            "hsa_runtime": {
                "implementation": observed.runtime().implementation(),
                "version": observed.runtime().version(),
                "image_sha256": lowercase_hex(
                    observed.runtime().image_digest().bytes().as_bytes()
                ),
            },
            "physical_device": {
                "uuid": lowercase_hex(&observed.physical_device().uuid()),
                "node_id": observed.physical_device().node_id(),
                "hip_ordinal": observed.physical_device().hip_ordinal(),
                "target": "gfx950:xnack-",
                "hsa_agent_handle": observed.agent().agent_handle().to_string(),
            },
        })
    };
    let mut output = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.output)?,
    );
    // SAFETY: immutable digest-pinned bytes and allocations are retained until
    // the profiled session and the sole consuming unload complete.
    let (executable, load) = unsafe { adapter.load_executable(bytes, digest) }?;
    let executable_identity = load.executable_object();
    let execution = (|| -> Result<(), BoxError> {
        require(
            load.finalized_digest() == digest && load.byte_len() == bytes.len() as u64,
            format!("{} profiled load observation changed", plan.label),
        )?;
        // SAFETY: profile inspection admitted exactly this export/descriptor.
        let (kernels, resolutions) =
            unsafe { adapter.resolve_kernel_set(&executable, [case.export]) }?;
        let kernel = kernels
            .get(0)
            .ok_or_else(|| format!("profiled runtime omitted {}", plan.label))?;
        let resolution = &resolutions[0];
        let size = kernarg_size(case.args);
        require(
            kernels.len() == 1
                && resolutions.len() == 1
                && resolution.executable_object() == executable_identity
                && resolution.export_symbol() == case.export
                && resolution.kernarg_segment_size() == size as u64
                && resolution.kernarg_segment_alignment() == HSA_KERNARG_ALIGNMENT
                && resolution.group_segment_size() == case.static_lds_bytes,
            format!(
                "runtime resolved a substituted profiled {} kernel",
                plan.label
            ),
        )?;
        let geometry = HsaLaunchGeometryV1::new([case.grid_x(), 1, 1], [case.workgroup_x, 1, 1], 0);
        let mut storage = RuntimeKernarg::new(size)?;
        let kernarg = storage.bytes_mut();
        kernarg.copy_from_slice(&explicit);
        let original = explicit.clone();
        // SAFETY: the same admitted ABI and live buffers used by the existing
        // correctness dispatch remain live for the complete session.
        let mut session = unsafe {
            let initialization = adapter.initialize_implicit_kernarg(
                &executable,
                kernel,
                geometry,
                size,
                size,
                0,
                kernarg,
            )?;
            require(
                initialization.initialized()
                    && initialization.explicit_byte_len() == size as u64
                    && initialization.implicit_byte_offset() == size as u64
                    && initialization.implicit_byte_len() == 0
                    && kernarg == original,
                format!("{} profiled initialization changed the ABI", case.label),
            )?;
            adapter.prepare_profiled_dispatch_session(&executable, kernel, geometry, kernarg)?
        };

        session.dispatch()?;
        for (planned, actual) in plan.buffers.iter().zip(&buffers) {
            verify_buffer(planned, &actual.read_after_synchronous_dispatch())?;
        }
        for _ in 0..config.warmups {
            session.dispatch()?;
        }

        for block in 0..config.blocks {
            for _ in 0..config.block_rewarm {
                session.dispatch()?;
            }
            let observations = (0..config.samples_per_block)
                .map(|_| session.dispatch())
                .collect::<Result<Vec<_>, _>>()?;
            for (planned, actual) in plan.buffers.iter().zip(&buffers) {
                verify_buffer(planned, &actual.read_after_synchronous_dispatch())?;
            }
            for (sample, observation) in observations.iter().enumerate() {
                let record = json!({
                    "schema": "fe2o3.gfx950.advanced-dispatch-sample.v1",
                    "campaign_id": config.campaign_id,
                    "record_id": format!(
                        "{}:{}:{}:{}:{}:{}:{}",
                        config.campaign_id, config.variant_id, config.process,
                        case.export, plan.label, block, sample
                    ),
                    "implementation": {
                        "id": config.implementation_id,
                        "variant": config.variant_id,
                    },
                    "artifact": {
                        "source_commit": config.source_commit,
                        "source_tree": config.source_tree,
                        "crate_binding_sha256": config.crate_binding,
                        "llvm_sha256": config.llvm_sha256,
                        "hsaco_sha256": std::env::var(SHA256_ENV)?,
                        "isa_sha256": config.isa_sha256,
                        "kernel_export": case.export,
                        "descriptor": case.descriptor,
                    },
                    "workload": {
                        "id": plan.label,
                        "input_sha256": workload_sha256,
                        "buffers": buffer_identity,
                        "cache_regime": if case.grid_x() == 1 {
                            "persistent-allocation-repeated-single-workgroup"
                        } else {
                            "persistent-allocation-repeated-multi-workgroup"
                        },
                    },
                    "launch": {
                        "grid": [case.grid_x(), 1, 1],
                        "workgroup": [case.workgroup_x, 1, 1],
                        "dynamic_lds_bytes": 0,
                        "static_lds_bytes": case.static_lds_bytes,
                        "kernarg_bytes": size,
                    },
                    "trial": {
                        "process": config.process,
                        "block": block,
                        "sample": sample,
                        "initial_warmups": config.warmups,
                        "block_rewarm": config.block_rewarm,
                        "samples_per_block": config.samples_per_block,
                    },
                    "timer": {
                        "source": "rocr-hsa-dispatch-timestamps",
                        "start_tick": observation.start_tick().to_string(),
                        "end_tick": observation.end_tick().to_string(),
                        "duration_ticks": (observation.end_tick() - observation.start_tick()).to_string(),
                        "frequency_hz": observation.timestamp_frequency_hz().to_string(),
                        "duration_ns": observation.duration_ns().to_string(),
                        "aql_packet_id": observation.packet_id().to_string(),
                    },
                    "correctness": {
                        "passed": true,
                        "preflight_dispatch_checked": true,
                        "post_block_checked": true,
                        "guard_canaries_checked": true,
                        "oracle": "current-repository-cpu-reference",
                    },
                    "environment": environment,
                });
                serde_json::to_writer(&mut output, &record)?;
                output.write_all(b"\n")?;
            }
        }
        output.flush()?;
        Ok(())
    })();
    let unload = unsafe { adapter.unload_executable(executable) }?;
    require(
        unload.released() && unload.executable_object() == executable_identity,
        format!("{} profiled executable was not released", plan.label),
    )?;
    execution
}

#[cfg(feature = "hardware-test-hooks")]
fn run_case(mut case: AdvancedCase) -> Result<(), BoxError> {
    if case.export.starts_with("gfx950_gpt_oss_") {
        require_gpt_runtime_visibility()?;
    }
    let plan_case = case;
    if let Some(value) = std::env::var_os(WORKGROUP_X_ENV) {
        let value = value
            .to_str()
            .ok_or_else(|| format!("{WORKGROUP_X_ENV} must be valid text"))?
            .parse::<u32>()
            .map_err(|_| format!("{WORKGROUP_X_ENV} must be an unsigned decimal integer"))?;
        require(
            value != 0 && value <= 1024,
            format!("{WORKGROUP_X_ENV} is outside 1..=1024"),
        )?;
        case.workgroup_x = value;
    }
    let (bytes, digest) = read_pinned_hsaco()?;
    inspect_profile(case, &bytes)?;
    let context = GpuContext::new(0)?;
    let mut adapter = ReviewedHsaRuntimeAdapterV1::new_gfx950(context)?;
    require(
        adapter.environment().physical_device().target().processor() == "gfx950"
            && adapter.environment().physical_device().target().xnack()
                == Some(FeatureState::Disabled),
        format!("{} requires a gfx950:xnack- physical device", case.label),
    )?;
    for plan in plans_for(plan_case)? {
        execute_plan(case, &mut adapter, &bytes, digest, plan)?;
    }
    if let Some(config) = PerformanceConfig::from_environment()? {
        for plan in plans_for(plan_case)? {
            profile_plan(case, &mut adapter, &bytes, digest, plan, &config)?;
        }
    }
    println!("PASS {} production HSA verification", case.label);
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
macro_rules! hardware_case {
    ($name:ident, $case:ident) => {
        #[test]
        #[ignore = "non-authoritative: requires a Rust-produced digest-pinned gfx950:xnack- COV6 HSACO and MI350"]
        fn $name() -> Result<(), BoxError> {
            run_case($case)
        }
    };
}

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_decode_rust_cov6_matches_cpu_reference,
    KDA_DECODE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_chunkwise_prefill_rust_cov6_matches_cpu_reference,
    KDA_PREFILL
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_content_sparse_attention_rust_cov6_matches_cpu_reference,
    SPARSE_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_deepseek_sparse_attention_rust_cov6_matches_cpu_reference,
    DEEPSEEK_SPARSE_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_compressed_hybrid_attention_rust_cov6_matches_cpu_reference,
    HYBRID_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_attnres_aggregate_rust_cov6_matches_cpu_reference,
    ATTNRES
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_four_branch_residual_rust_cov6_matches_cpu_reference,
    FOUR_BRANCH
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(gfx950_mhc_sinkhorn_mix_rust_cov6_matches_cpu_reference, MHC);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(gfx950_moe_route_rust_cov6_matches_cpu_reference, MOE_ROUTE);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_moe_expert_rank_rust_cov6_matches_cpu_reference,
    MOE_EXPERT
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_layer_tile_rust_cov6_matches_cpu_reference,
    GPT_OSS
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_pipelined_attention_rust_cov6_matches_cpu_reference,
    GPT_OSS_PIPELINED
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_router_component_rust_cov6_matches_cpu_reference,
    GPT_OSS_ROUTER_COMPONENT
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_attention_component_rust_cov6_matches_cpu_reference,
    GPT_OSS_ATTENTION_COMPONENT
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_expert_component_rust_cov6_matches_cpu_reference,
    GPT_OSS_EXPERT_COMPONENT
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_layer_tile_tile_rust_cov6_matches_cpu_reference,
    GPT_OSS_TILE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_router_component_tile_rust_cov6_matches_cpu_reference,
    GPT_OSS_ROUTER_TILE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_attention_component_tile_rust_cov6_matches_cpu_reference,
    GPT_OSS_ATTENTION_TILE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_gpt_oss_expert_component_tile_rust_cov6_matches_cpu_reference,
    GPT_OSS_EXPERT_TILE
);
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires the exact HIP gfx950:xnack- comparator HSACO and MI350"]
fn gfx950_gpt_oss_unfused_hip_matches_cpu_reference() -> Result<(), BoxError> {
    run_gpt_unfused_comparator()
}

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_combine_expert_ranks_rust_cov6_matches_cpu_reference,
    COMBINE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_speculative_transaction_rust_cov6_matches_cpu_reference,
    SPECULATIVE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_qwen_ngram_gather_rust_cov6_matches_cpu_reference,
    NGRAM_GATHER
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_stage_gradient_shard_rust_cov6_matches_cpu_reference,
    STAGE_SHARD
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(gfx950_muon_update_rust_cov6_matches_cpu_reference, MUON);

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn output_poison_cannot_match_an_unwritten_expected_element() {
    let float = output(
        "float",
        ExpectedOutput::F32 {
            values: vec![0.0, 1.0],
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            exact_mask: None,
        },
    );
    let float_body = &float.initial[GUARD_BYTES..float.initial.len() - GUARD_BYTES];
    assert!(
        decode_f32(float_body)
            .expect("float poison")
            .iter()
            .all(|value| value.is_nan())
    );
    assert!(verify_buffer(&float, &float.initial).is_err());

    let unsigned = output("unsigned", ExpectedOutput::U32(vec![0, u32::MAX]));
    let unsigned_body = &unsigned.initial[GUARD_BYTES..unsigned.initial.len() - GUARD_BYTES];
    assert_eq!(
        decode_u32(unsigned_body).expect("u32 poison"),
        [u32::MAX, 0]
    );
    assert!(verify_buffer(&unsigned, &unsigned.initial).is_err());

    let signed = output("signed", ExpectedOutput::I32(vec![-1, 0]));
    let signed_body = &signed.initial[GUARD_BYTES..signed.initial.len() - GUARD_BYTES];
    assert_eq!(decode_i32(signed_body).expect("i32 poison"), [0, -1]);
    assert!(verify_buffer(&signed, &signed.initial).is_err());
}

#[cfg(not(feature = "hardware-test-hooks"))]
#[test]
#[ignore = "requires feature hardware-test-hooks and caller-supplied gfx950 HSACO"]
fn gfx950_advanced_hardware_tests_require_explicit_hooks() {}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gpt_oss_finite_router_edge_is_an_additional_guarded_full_fixture() -> Result<(), BoxError> {
    for case in [
        GPT_OSS,
        GPT_OSS_PIPELINED,
        GPT_OSS_TILE,
        GPT_OSS_ROUTER_COMPONENT,
        GPT_OSS_ROUTER_TILE,
    ] {
        let plans = plans_for(case)?;
        assert_eq!(plans.len(), 2);
        let edge = &plans[1];
        let route = edge.buffers.last().ok_or("route output absent")?;
        let Some(ExpectedOutput::U32(values)) = &route.expected else {
            return Err("route output kind changed".into());
        };
        assert_eq!(values, &vec![(1 << 7) | (2 << 14) | (3 << 21); 1024]);
        assert_eq!(route.elements, 1024);
        let inputs = gpt_oss_reference::finite_router_edge_batch_inputs();
        let expected_inputs = [
            value_bytes(&inputs.hidden_f32),
            value_bytes(&inputs.router_f32),
            value_bytes(&inputs.query_bf16),
            value_bytes(&inputs.key_transposed_bf16),
            value_bytes(&inputs.value_f32),
            value_bytes(&inputs.sinks_f32),
            value_bytes(&inputs.expert_activation_blocks_fp4),
            value_bytes(&inputs.expert_weight_blocks_fp4),
            value_bytes(&inputs.activation_scales),
            value_bytes(&inputs.expert_weight_scales),
        ];
        let input_count = if case.args.len() == 13 { 10 } else { 2 };
        for (buffer, bytes) in edge.buffers.iter().zip(&expected_inputs).take(input_count) {
            assert!(buffer.immutable && buffer.expected.is_none());
            assert_eq!(buffer.body_offset, 0);
            assert_eq!(&buffer.initial, bytes);
        }
        for buffer in edge
            .buffers
            .iter()
            .filter(|buffer| buffer.expected.is_some())
        {
            assert_eq!(buffer.body_offset, GUARD_BYTES);
            assert_eq!(&buffer.initial[..GUARD_BYTES], &[PREFIX_BYTE; GUARD_BYTES]);
            assert_eq!(
                &buffer.initial[buffer.initial.len() - GUARD_BYTES..],
                &[SUFFIX_BYTE; GUARD_BYTES]
            );
            assert!(
                verify_buffer(buffer, &buffer.initial).is_err(),
                "unwritten poison must not satisfy edge fixture"
            );
        }
        if case.args.len() == 13 {
            for index in [10, 11] {
                let Some(ExpectedOutput::F32 { values, .. }) = &edge.buffers[index].expected else {
                    return Err("float output kind changed".into());
                };
                assert_eq!(values.len(), 4096);
                assert!(values.iter().all(|value| value.is_finite()));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gpt_oss_tile_fixture_preserves_all_values_and_guards() -> Result<(), BoxError> {
    let canonical = gpt_oss_plans(GPT_OSS)?.remove(0);
    let tiled = gpt_oss_plans(GPT_OSS_TILE)?.remove(0);
    assert_eq!(canonical.buffers.len(), 13);
    assert_eq!(tiled.buffers.len(), 13);
    for i in 0..10 {
        assert!(tiled.buffers[i].immutable);
        assert_eq!(canonical.buffers[i].initial, tiled.buffers[i].initial);
    }
    for i in [10, 11] {
        let ExpectedOutput::F32 {
            values: logical, ..
        } = tiled.buffers[i].expected.as_ref().unwrap()
        else {
            panic!("float tile expected")
        };
        let ExpectedOutput::F32 {
            values: blocked, ..
        } = canonical.buffers[i].expected.as_ref().unwrap()
        else {
            panic!("float tile expected")
        };
        assert_eq!(logical.len(), 4096);
        assert_eq!(blocked.len(), 4096);
        for item in 0..16 {
            for lane in 0..64 {
                for component in 0..4 {
                    let logical_offset =
                        item * 256 + ((lane / 16) * 4 + component) * 16 + lane % 16;
                    let packed_offset = item * 256 + component * 64 + lane;
                    assert_eq!(logical[logical_offset], blocked[packed_offset]);
                }
            }
        }
        assert_eq!(tiled.buffers[i].body_offset, GUARD_BYTES);
        assert_eq!(
            &tiled.buffers[i].initial[..GUARD_BYTES],
            &[PREFIX_BYTE; GUARD_BYTES]
        );
        let initial = &tiled.buffers[i].initial;
        assert_eq!(
            &initial[initial.len() - GUARD_BYTES..],
            &[SUFFIX_BYTE; GUARD_BYTES]
        );
    }
    assert_eq!(tiled.buffers[12].elements, 1024);
    assert_eq!(canonical.buffers[12].initial, tiled.buffers[12].initial);
    for (case, count) in [
        (GPT_OSS_ROUTER_TILE, 3),
        (GPT_OSS_ATTENTION_TILE, 5),
        (GPT_OSS_EXPERT_TILE, 6),
    ] {
        let plan = plans_for(case)?.remove(0);
        assert_eq!(plan.buffers.len(), count);
        assert_eq!(plan.args.len(), count);
        assert!(
            plan.buffers
                .iter()
                .filter(|b| b.expected.is_some())
                .all(|b| b.body_offset == GUARD_BYTES)
        );
    }
    Ok(())
}

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_decode_simt_rust_cov6_matches_cpu_reference,
    KDA_DECODE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_decode_tile_rust_cov6_matches_cpu_reference,
    KDA_DECODE
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_chunkwise_prefill_simt_rust_cov6_matches_cpu_reference,
    KDA_PREFILL
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_kda_chunkwise_prefill_tile_rust_cov6_matches_cpu_reference,
    KDA_PREFILL
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_deepseek_sparse_attention_simt_rust_cov6_matches_cpu_reference,
    DEEPSEEK_SPARSE_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_deepseek_sparse_attention_tile_rust_cov6_matches_cpu_reference,
    DEEPSEEK_SPARSE_ATTENTION
);

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gfx950_kda_c_oracle_maps_all_value_major_state_and_eight_token_words() {
    // Distinct exact f32 values identify every logical coordinate. These raw
    // oracle layouts are shared by canonical SIMT and checked tile.
    let state: Vec<f32> = (0..256)
        .map(|i| f32::from_bits(0x3f000001 + i as u32))
        .collect();
    let tokens: Vec<f32> = (0..128)
        .map(|i| f32::from_bits(0x3e800001 + i as u32))
        .collect();
    let physical = kda_state_value_major(&state);
    for batch in 0..4 {
        for value in 0..16 {
            for key in 0..16 {
                let parent_row = batch * 16 + value;
                let tile_address = parent_row * 16 + key;
                assert_eq!(tile_address, batch * 256 + value * 16 + key);
                assert_eq!(
                    physical[value * 16 + key].to_bits(),
                    state[key * 16 + value].to_bits()
                );
                for chunk in 0..2 {
                    let packed = kda_replicated_chunk_output(&tokens, chunk);
                    assert_eq!(
                        packed[value * 16 + key].to_bits(),
                        tokens[(chunk * 4 + key / 4) * 16 + value].to_bits()
                    );
                }
            }
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gfx950_kda_oracle_checks_every_f32_state_and_token_output_word() {
    // Canonical SIMT and checked tile use these same complete oracle plans.
    for (case, output_count) in [(KDA_DECODE, 2), (KDA_PREFILL, 3)] {
        let plans = plans_for(case).expect("complete KDA plans");
        assert_eq!(plans.len(), 1);
        let mut outputs = 0;
        for buffer in &plans[0].buffers {
            if buffer.immutable {
                assert!(buffer.expected.is_none());
                continue;
            }
            let Some(ExpectedOutput::F32 {
                values, exact_mask, ..
            }) = &buffer.expected
            else {
                panic!("every KDA output must retain its f32 oracle");
            };
            assert_eq!(
                values.len(),
                MULTIGRID_WORKGROUPS_V1 * KDA_STATE_ELEMENTS_V1
            );
            assert!(values.iter().all(|value| value.is_finite()));
            assert!(exact_mask.is_none());
            let bytes = value_bytes(values);
            assert_eq!(bytes.len(), values.len() * 4);
            let mut valid = buffer.initial.clone();
            valid[buffer.body_offset..buffer.body_offset + bytes.len()].copy_from_slice(&bytes);
            verify_buffer(buffer, &valid).expect("complete expected f32 bytes accepted");
            assert!(
                verify_buffer(buffer, &buffer.initial).is_err(),
                "unwritten output must fail"
            );
            for word in 0..values.len() {
                let offset = buffer.body_offset + word * 4;
                let saved: [u8; 4] = valid[offset..offset + 4].try_into().unwrap();
                valid[offset..offset + 4].copy_from_slice(&f32::NAN.to_le_bytes());
                assert!(
                    verify_buffer(buffer, &valid).is_err(),
                    "unverified output word {word}"
                );
                valid[offset..offset + 4].copy_from_slice(&saved);
            }
            outputs += 1;
        }
        assert_eq!(outputs, output_count);
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn deepseek_oracles_reject_every_corrupted_output_word() {
    for case in [DEEPSEEK_SPARSE_ATTENTION] {
        let plans = plans_for(case).expect("complete existing oracle plans");
        assert_eq!(plans.len(), 1);
        let mut output_sizes = Vec::new();
        for buffer in &plans[0].buffers {
            if buffer.immutable {
                assert!(buffer.expected.is_none());
                continue;
            }
            let expected = buffer
                .expected
                .as_ref()
                .expect("every output has an oracle");
            let (bytes, floating) = match expected {
                ExpectedOutput::F32 {
                    values, exact_mask, ..
                } => {
                    if let Some(mask) = exact_mask {
                        assert_eq!(mask.len(), values.len());
                        assert!(mask.iter().any(|&exact| exact));
                        assert!(mask.iter().any(|&exact| !exact));
                    }
                    (value_bytes(values), true)
                }
                ExpectedOutput::U32(values) => (value_bytes(values), false),
                ExpectedOutput::I32(_) => panic!("these kernels have no signed output"),
            };
            output_sizes.push(expected.element_count());
            let mut valid = buffer.initial.clone();
            valid[buffer.body_offset..buffer.body_offset + bytes.len()].copy_from_slice(&bytes);
            verify_buffer(buffer, &valid).expect("full original expected output accepted");
            assert!(verify_buffer(buffer, &buffer.initial).is_err());
            for word in 0..expected.element_count() {
                let offset = buffer.body_offset + word * 4;
                let saved: [u8; 4] = valid[offset..offset + 4].try_into().unwrap();
                let corrupt = if floating {
                    f32::NAN.to_le_bytes()
                } else {
                    (u32::from_le_bytes(saved) ^ 1).to_le_bytes()
                };
                valid[offset..offset + 4].copy_from_slice(&corrupt);
                assert!(
                    verify_buffer(buffer, &valid).is_err(),
                    "unverified {} word {word}",
                    buffer.name
                );
                valid[offset..offset + 4].copy_from_slice(&saved);
            }
        }
        output_sizes.sort_unstable();
        assert_eq!(output_sizes, [1024, 1024, 1024]);
    }
}

#[cfg(feature = "hardware-test-hooks")]
const SPARSE_ATTENTION_SIMT: AdvancedCase = AdvancedCase {
    label: "gfx950 content sparse attention SIMT",
    static_lds_bytes: 0,
    ..SPARSE_ATTENTION
};

#[cfg(feature = "hardware-test-hooks")]
const HYBRID_ATTENTION_SIMT: AdvancedCase = AdvancedCase {
    label: "gfx950 compressed hybrid attention SIMT",
    static_lds_bytes: 0,
    ..HYBRID_ATTENTION
};

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_content_sparse_attention_tile_rust_cov6_matches_cpu_reference,
    SPARSE_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_content_sparse_attention_simt_rust_cov6_matches_cpu_reference,
    SPARSE_ATTENTION_SIMT
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_compressed_hybrid_attention_tile_rust_cov6_matches_cpu_reference,
    HYBRID_ATTENTION
);
#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_compressed_hybrid_attention_simt_rust_cov6_matches_cpu_reference,
    HYBRID_ATTENTION_SIMT
);

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gfx950_attention_simt_cases_preserve_independent_oracle_payloads() {
    for (tile, simt) in [
        (SPARSE_ATTENTION, SPARSE_ATTENTION_SIMT),
        (HYBRID_ATTENTION, HYBRID_ATTENTION_SIMT),
    ] {
        assert_eq!(tile.static_lds_bytes, 8192);
        assert_eq!(simt.static_lds_bytes, 0);
        assert_eq!(tile.export, simt.export);
        assert_eq!(tile.descriptor, simt.descriptor);
        assert_eq!(tile.args, simt.args);
        assert_eq!(tile.workgroup_x, simt.workgroup_x);
        let tile_plans = plans_for(tile).expect("tile oracle plans");
        let simt_plans = plans_for(simt).expect("SIMT oracle plans");
        assert_eq!(tile_plans.len(), simt_plans.len());
        for (tile_plan, simt_plan) in tile_plans.iter().zip(&simt_plans) {
            assert_eq!(tile_plan.args.len(), simt_plan.args.len());
            assert_eq!(tile_plan.buffers.len(), simt_plan.buffers.len());
            for (tile_buffer, simt_buffer) in tile_plan.buffers.iter().zip(&simt_plan.buffers) {
                assert_eq!(tile_buffer.name, simt_buffer.name);
                assert_eq!(tile_buffer.initial, simt_buffer.initial);
                assert_eq!(tile_buffer.body_offset, simt_buffer.body_offset);
                assert_eq!(tile_buffer.elements, simt_buffer.elements);
                assert_eq!(tile_buffer.immutable, simt_buffer.immutable);
                match (&tile_buffer.expected, &simt_buffer.expected) {
                    (None, None) => {}
                    (
                        Some(ExpectedOutput::F32 {
                            values: tile_values,
                            absolute_tolerance: tile_absolute,
                            relative_tolerance: tile_relative,
                            exact_mask: tile_mask,
                        }),
                        Some(ExpectedOutput::F32 {
                            values: simt_values,
                            absolute_tolerance: simt_absolute,
                            relative_tolerance: simt_relative,
                            exact_mask: simt_mask,
                        }),
                    ) => {
                        assert_eq!(value_bytes(tile_values), value_bytes(simt_values));
                        assert_eq!(tile_absolute, simt_absolute);
                        assert_eq!(tile_relative, simt_relative);
                        assert_eq!(tile_mask, simt_mask);
                    }
                    (
                        Some(ExpectedOutput::U32(tile_values)),
                        Some(ExpectedOutput::U32(simt_values)),
                    ) => {
                        assert_eq!(tile_values, simt_values);
                    }
                    _ => panic!("tile and SIMT attention oracle output contracts differ"),
                }
            }
        }
    }
}

// Vecadd shares the reviewed runtime boundary, with its own complete F32 oracle.
#[cfg(feature = "hardware-test-hooks")]
#[path = "gfx950_advanced_hardware/vecadd.rs"]
mod vecadd_qualification;

#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a digest-pinned Rust gfx950 HSACO and independently leased MI350"]
fn gfx950_vecadd_rust_cov6_matches_cpu_reference() -> Result<(), BoxError> {
    vecadd_qualification::run()
}
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "non-authoritative: requires a digest-pinned Rust gfx950 tile HSACO and independently leased MI350"]
fn gfx950_vecadd_tile_rust_cov6_matches_cpu_reference() -> Result<(), BoxError> {
    vecadd_qualification::run()
}
#[cfg(feature = "hardware-test-hooks")]
#[test]
#[ignore = "CPU-only metadata inspection of an explicitly pinned vecadd HSACO; no GPU initialization"]
fn gfx950_vecadd_profile_cpu_inspection() -> Result<(), BoxError> {
    let (bytes, _) = read_pinned_hsaco()?;
    inspect_profile(vecadd_qualification::CASE, &bytes)
}

#[cfg(feature = "hardware-test-hooks")]
const MOE_ROUTE_TILE: AdvancedCase = MOE_ROUTE;
#[cfg(feature = "hardware-test-hooks")]
const MOE_EXPERT_SIMT: AdvancedCase = MOE_EXPERT;
#[cfg(feature = "hardware-test-hooks")]
const COMBINE_TILE: AdvancedCase = COMBINE;
#[cfg(feature = "hardware-test-hooks")]
const STAGE_SHARD_TILE: AdvancedCase = STAGE_SHARD;
#[cfg(feature = "hardware-test-hooks")]
const MUON_TILE: AdvancedCase = MUON;
#[cfg(feature = "hardware-test-hooks")]
const NGRAM_GATHER_TILE: AdvancedCase = NGRAM_GATHER;

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_moe_route_tile_rust_cov6_matches_cpu_reference,
    MOE_ROUTE_TILE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_moe_expert_rank_simt_rust_cov6_matches_cpu_reference,
    MOE_EXPERT_SIMT
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_combine_expert_ranks_tile_rust_cov6_matches_cpu_reference,
    COMBINE_TILE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_speculative_transaction_simt_rust_cov6_matches_cpu_reference,
    SPECULATIVE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_speculative_transaction_tile_rust_cov6_matches_cpu_reference,
    SPECULATIVE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_stage_gradient_shard_tile_rust_cov6_matches_cpu_reference,
    STAGE_SHARD_TILE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_muon_update_tile_rust_cov6_matches_cpu_reference,
    MUON_TILE
);

#[cfg(feature = "hardware-test-hooks")]
hardware_case!(
    gfx950_qwen_ngram_gather_tile_rust_cov6_matches_cpu_reference,
    NGRAM_GATHER_TILE
);

#[cfg(feature = "hardware-test-hooks")]
fn make_stage_shards() -> Vec<f32> {
    let special = [
        0x0000_0000_u32,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_00a5,
        0x7fc1_2345,
        0xffc5_4321,
        0x3f80_0001,
    ];
    (0..SYSTEM_BATCHES * GRADIENT_SHARDS * MUON_ELEMENTS)
        .map(|index| {
            let batch = index / (GRADIENT_SHARDS * MUON_ELEMENTS);
            let shard = index / MUON_ELEMENTS % GRADIENT_SHARDS;
            let element = index % MUON_ELEMENTS;
            let bits = if batch < 2 {
                special[(element + shard * 5 + batch * 3) % special.len()]
            } else {
                // Vary all mantissa bits across both shards and all later batches.
                0x3e00_0000 | ((index as u32).wrapping_mul(0x0045_d9f3) & 0x807f_ffff)
            };
            f32::from_bits(bits)
        })
        .collect()
}

#[cfg(feature = "hardware-test-hooks")]
fn ngram_integer_plan(case: AdvancedCase) -> LaunchPlan {
    let mut queries = vec![0; SYSTEM_BATCHES * QUERIES * NGRAM];
    let mut hashes = vec![0; SYSTEM_BATCHES * TABLE_SIZE];
    let mut grams = vec![0; SYSTEM_BATCHES * TABLE_SIZE * NGRAM];
    let mut values = vec![0; SYSTEM_BATCHES * TABLE_SIZE];
    let mut priorities = vec![i32::MIN; SYSTEM_BATCHES * TABLE_SIZE];
    for batch in 0..SYSTEM_BATCHES {
        for query in 0..QUERIES {
            let base = (batch * QUERIES + query) * NGRAM;
            queries[base..base + NGRAM].copy_from_slice(&[
                i32::MIN + (batch * QUERIES + query) as i32,
                -16_777_217 - query as i32,
                i32::MAX - batch as i32,
            ]);
        }
        // Force a wrapped probe order: slot14 is visited before slot1, but
        // absolute slot1 must win their equal-priority tie.
        let first = batch * QUERIES * NGRAM;
        while hash_gram(&queries[first..first + NGRAM]) as usize & (TABLE_SIZE - 1) != 14 {
            queries[first + 1] = queries[first + 1].wrapping_add(1);
        }
        for (query, slot, priority, value) in [
            (0, 14, 7, 16_777_217),
            (0, 1, 7, -16_777_219),
            (1, 3, -8, 3),
            (1, 8, 12, i32::MAX),
            (2, 5, i32::MIN, i32::MIN),
            (3, 4, i32::MAX, 99),
            (4, 7, 0, i32::MAX),
            (5, 9, i32::MAX - 1, -123_456_789),
        ] {
            let query_base = (batch * QUERIES + query) * NGRAM;
            let table_slot = batch * TABLE_SIZE + slot;
            hashes[table_slot] = hash_gram(&queries[query_base..query_base + NGRAM]);
            let gram_base = table_slot * NGRAM;
            grams[gram_base..gram_base + NGRAM]
                .copy_from_slice(&queries[query_base..query_base + NGRAM]);
            if query == 3 {
                // Matching hash and higher priority cannot override full-key mismatch.
                grams[gram_base + 2] ^= 1;
            }
            priorities[table_slot] = priority;
            values[table_slot] = value;
        }
    }
    let expected =
        systems_reference::batched_ngram_reference(&queries, &hashes, &grams, &values, &priorities);
    let lengths = [
        queries.len(),
        hashes.len(),
        grams.len(),
        values.len(),
        priorities.len(),
        SYSTEM_BATCHES * QUERIES,
    ];
    LaunchPlan {
        label: format!(
            "{} signed integers, collision, priority and wrapped tie",
            case.label
        ),
        buffers: vec![
            input("queries", &queries),
            input("table_hashes", &hashes),
            input("table_grams", &grams),
            input("table_values", &values),
            input("priorities", &priorities),
            output("output", ExpectedOutput::I32(expected)),
        ],
        args: lengths
            .into_iter()
            .enumerate()
            .map(|(buffer, elements)| PlannedArg::Slice { buffer, elements })
            .collect(),
    }
}

#[cfg(feature = "hardware-test-hooks")]
fn fixture_expected_bytes(plan: &PlannedBuffer) -> Vec<u8> {
    let body = match plan.expected.as_ref().expect("output reference") {
        ExpectedOutput::F32 { values, .. } => value_bytes(values),
        ExpectedOutput::U32(values) => value_bytes(values),
        ExpectedOutput::I32(values) => value_bytes(values),
    };
    let mut actual = plan.initial.clone();
    actual[GUARD_BYTES..GUARD_BYTES + body.len()].copy_from_slice(&body);
    actual
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn exact_f32_verification_preserves_payloads_zero_and_unwritten_detection() {
    let values = vec![
        f32::from_bits(F32_POISON_BITS),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x3f80_0001),
    ];
    let plan = output(
        "raw_bits",
        ExpectedOutput::F32 {
            exact_mask: Some(vec![true; values.len()]),
            values,
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
        },
    );
    let actual = fixture_expected_bytes(&plan);
    verify_buffer(&plan, &actual).expect("all exact bits, including NaNs");
    assert!(
        verify_buffer(&plan, &plan.initial).is_err(),
        "unwritten NaN poison must fail"
    );
    // A mutation at every element rejects payload changes, +0/-0 changes and
    // incomplete last-element writes without relying on floating arithmetic.
    for element in 0..plan.elements {
        let mut changed = actual.clone();
        changed[GUARD_BYTES + element * 4 + 3] ^= 0x80;
        assert!(verify_buffer(&plan, &changed).is_err(), "element {element}");
    }
    let mut changed_payload = actual.clone();
    changed_payload[GUARD_BYTES + 4] ^= 1;
    assert!(verify_buffer(&plan, &changed_payload).is_err());
    let mut changed_guard = actual.clone();
    changed_guard[0] ^= 1;
    assert!(verify_buffer(&plan, &changed_guard).is_err());
    let tolerance_plan = f32_output("finite_only", vec![f32::NAN], 0.0);
    assert!(verify_buffer(&tolerance_plan, &fixture_expected_bytes(&tolerance_plan)).is_err());
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn stage_tile_and_simt_keep_two_complete_bitwise_shard_plans() {
    assert_eq!(STAGE_SHARD_TILE, STAGE_SHARD);
    let plans = systems_plans(STAGE_SHARD_TILE).expect("stage plans");
    assert_eq!(plans.len(), GRADIENT_SHARDS);
    let mut first_input = None;
    for plan in &plans {
        assert_eq!(plan.buffers[0].elements, 256);
        let output = &plan.buffers[1];
        assert_eq!(output.elements, 256);
        let Some(ExpectedOutput::F32 {
            values,
            exact_mask: Some(mask),
            ..
        }) = &output.expected
        else {
            panic!("full exact stage reference")
        };
        assert_eq!(mask.len(), 256);
        assert!(mask.iter().all(|exact| *exact));
        assert_eq!(value_bytes(values), plan.buffers[0].initial);
        for bits in [
            0_u32,
            0x8000_0000,
            0x7fc0_00a5,
            0x7fc1_2345,
            0xffc5_4321,
            1,
            0x8000_0001,
        ] {
            assert!(values.iter().any(|value| value.to_bits() == bits));
        }
        let actual = fixture_expected_bytes(output);
        verify_buffer(output, &actual).expect("complete stage reference");
        assert!(verify_buffer(output, &output.initial).is_err());
        let mut omitted_last = actual.clone();
        let last = GUARD_BYTES + 255 * 4;
        omitted_last[last..last + 4].copy_from_slice(&output.initial[last..last + 4]);
        assert!(
            verify_buffer(output, &omitted_last).is_err(),
            "last output is mandatory"
        );
        if let Some(first) = &first_input {
            assert_ne!(first, &plan.buffers[0].initial, "two distinct shard inputs");
        } else {
            first_input = Some(plan.buffers[0].initial.clone());
        }
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn ngram_tile_and_simt_keep_full_integer_collision_priority_and_tie_plans() {
    assert_eq!(NGRAM_GATHER_TILE, NGRAM_GATHER);
    let plans = systems_plans(NGRAM_GATHER_TILE).expect("ngram plans");
    assert_eq!(
        plans.len(),
        2,
        "preserve original plan and add integer edge cases"
    );
    for plan in &plans {
        let output = &plan.buffers[5];
        assert_eq!(output.elements, SYSTEM_BATCHES * QUERIES);
        verify_buffer(output, &fixture_expected_bytes(output)).expect("complete integer output");
        assert!(verify_buffer(output, &output.initial).is_err());
    }
    let edge = &plans[1];
    let Some(ExpectedOutput::I32(values)) = &edge.buffers[5].expected else {
        panic!("integer output")
    };
    for batch in 0..SYSTEM_BATCHES {
        assert_eq!(
            &values[batch * QUERIES..(batch + 1) * QUERIES],
            &[
                -16_777_219,
                i32::MAX,
                i32::MIN,
                -1,
                i32::MAX,
                -123_456_789,
                -1,
                -1
            ]
        );
    }
    let mut corrupted = fixture_expected_bytes(&edge.buffers[5]);
    let last = GUARD_BYTES + (SYSTEM_BATCHES * QUERIES - 1) * 4;
    corrupted[last] ^= 1;
    assert!(verify_buffer(&edge.buffers[5], &corrupted).is_err());
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn muon_tile_and_simt_keep_complete_independent_arithmetic_reference() {
    assert_eq!(MUON_TILE, MUON);
    let plans = systems_plans(MUON_TILE).expect("Muon plans");
    assert_eq!(plans.len(), 1);
    let plan = &plans[0];
    assert_eq!(
        plan.buffers[0].elements,
        SYSTEM_BATCHES * GRADIENT_SHARDS * MUON_ELEMENTS
    );
    for (buffer, count) in [(&plan.buffers[1], 256), (&plan.buffers[2], 16)] {
        assert_eq!(buffer.elements, count);
        let Some(ExpectedOutput::F32 {
            values,
            absolute_tolerance,
            relative_tolerance,
            exact_mask,
        }) = &buffer.expected
        else {
            panic!("full Muon arithmetic oracle")
        };
        assert_eq!(values.len(), count);
        assert!(values.iter().all(|value| value.is_finite()));
        assert_eq!(*absolute_tolerance, 2.0e-6);
        assert_eq!(*relative_tolerance, 0.0);
        assert!(exact_mask.is_none());
        verify_buffer(buffer, &fixture_expected_bytes(buffer)).expect("full reference output");
        let mut corrupted = fixture_expected_bytes(buffer);
        let last = GUARD_BYTES + (count - 1) * 4;
        corrupted[last..last + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(verify_buffer(buffer, &corrupted).is_err());
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn speculative_oracle_rejects_every_corrupted_output_word() {
    for case in [SPECULATIVE] {
        let plans = plans_for(case).expect("complete existing oracle plans");
        assert_eq!(plans.len(), 1);
        let mut output_sizes = Vec::new();
        for buffer in &plans[0].buffers {
            if buffer.immutable {
                assert!(buffer.expected.is_none());
                continue;
            }
            let expected = buffer
                .expected
                .as_ref()
                .expect("every output has an oracle");
            let (bytes, floating) = match expected {
                ExpectedOutput::F32 {
                    values, exact_mask, ..
                } => {
                    if let Some(mask) = exact_mask {
                        assert_eq!(mask.len(), values.len());
                        assert!(mask.iter().any(|&exact| exact));
                        assert!(mask.iter().any(|&exact| !exact));
                    }
                    (value_bytes(values), true)
                }
                ExpectedOutput::U32(values) => (value_bytes(values), false),
                ExpectedOutput::I32(_) => panic!("these kernels have no signed output"),
            };
            output_sizes.push(expected.element_count());
            let mut valid = buffer.initial.clone();
            valid[buffer.body_offset..buffer.body_offset + bytes.len()].copy_from_slice(&bytes);
            verify_buffer(buffer, &valid).expect("full original expected output accepted");
            assert!(verify_buffer(buffer, &buffer.initial).is_err());
            for word in 0..expected.element_count() {
                let offset = buffer.body_offset + word * 4;
                let saved: [u8; 4] = valid[offset..offset + 4].try_into().unwrap();
                let corrupt = if floating {
                    f32::NAN.to_le_bytes()
                } else {
                    (u32::from_le_bytes(saved) ^ 1).to_le_bytes()
                };
                valid[offset..offset + 4].copy_from_slice(&corrupt);
                assert!(
                    verify_buffer(buffer, &valid).is_err(),
                    "unverified {} word {word}",
                    buffer.name
                );
                valid[offset..offset + 4].copy_from_slice(&saved);
            }
        }
        output_sizes.sort_unstable();
        assert_eq!(output_sizes, [128, 128, 1024]);
    }
}

#[cfg(feature = "hardware-test-hooks")]
#[test]
fn gfx950_moe_modes_preserve_independent_oracle_payloads() {
    for (tile, simt) in [
        (MOE_ROUTE, MOE_ROUTE_TILE),
        (MOE_EXPERT, MOE_EXPERT_SIMT),
        (COMBINE, COMBINE_TILE),
    ] {
        assert_eq!(tile.static_lds_bytes, 0);
        assert_eq!(simt.static_lds_bytes, 0);
        assert_eq!(tile.export, simt.export);
        assert_eq!(tile.descriptor, simt.descriptor);
        assert_eq!(tile.args, simt.args);
        assert_eq!(tile.workgroup_x, simt.workgroup_x);
        let tile_plans = plans_for(tile).expect("tile oracle plans");
        let simt_plans = plans_for(simt).expect("SIMT oracle plans");
        assert_eq!(tile_plans.len(), simt_plans.len());
        for (tile_plan, simt_plan) in tile_plans.iter().zip(&simt_plans) {
            assert_eq!(tile_plan.args.len(), simt_plan.args.len());
            for (left, right) in tile_plan.args.iter().zip(&simt_plan.args) {
                match (left, right) {
                    (
                        PlannedArg::Slice {
                            buffer: a,
                            elements: x,
                        },
                        PlannedArg::Slice {
                            buffer: b,
                            elements: y,
                        },
                    ) => {
                        assert_eq!(a, b);
                        assert_eq!(x, y);
                    }
                    (PlannedArg::Pointer { buffer: a }, PlannedArg::Pointer { buffer: b }) => {
                        assert_eq!(a, b)
                    }
                    (PlannedArg::U32(a), PlannedArg::U32(b)) => assert_eq!(a, b),
                    _ => panic!("MoE mode ABI argument kinds differ"),
                }
            }
            assert_eq!(tile_plan.buffers.len(), simt_plan.buffers.len());
            for (tile_buffer, simt_buffer) in tile_plan.buffers.iter().zip(&simt_plan.buffers) {
                assert_eq!(tile_buffer.name, simt_buffer.name);
                assert_eq!(tile_buffer.initial, simt_buffer.initial);
                assert_eq!(tile_buffer.body_offset, simt_buffer.body_offset);
                assert_eq!(tile_buffer.elements, simt_buffer.elements);
                assert_eq!(tile_buffer.immutable, simt_buffer.immutable);
                match (&tile_buffer.expected, &simt_buffer.expected) {
                    (None, None) => {}
                    (
                        Some(ExpectedOutput::F32 {
                            values: tile_values,
                            absolute_tolerance: tile_absolute,
                            relative_tolerance: tile_relative,
                            exact_mask: tile_mask,
                        }),
                        Some(ExpectedOutput::F32 {
                            values: simt_values,
                            absolute_tolerance: simt_absolute,
                            relative_tolerance: simt_relative,
                            exact_mask: simt_mask,
                        }),
                    ) => {
                        assert_eq!(value_bytes(tile_values), value_bytes(simt_values));
                        assert_eq!(tile_absolute, simt_absolute);
                        assert_eq!(tile_relative, simt_relative);
                        assert_eq!(tile_mask, simt_mask);
                    }
                    (
                        Some(ExpectedOutput::U32(tile_values)),
                        Some(ExpectedOutput::U32(simt_values)),
                    ) => {
                        assert_eq!(tile_values, simt_values);
                    }
                    (Some(ExpectedOutput::I32(left)), Some(ExpectedOutput::I32(right))) => {
                        assert_eq!(left, right);
                    }
                    _ => panic!("MoE mode oracle output contracts differ"),
                }
            }
        }
    }
}
