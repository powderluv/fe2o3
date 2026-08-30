#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]
#![deny(missing_docs)]

//! A GPT-OSS-120B batch-1 layer-tile megakernel for gfx950.
//!
//! This is an architecture-shaped slice, not a claim to execute the complete
//! 120B checkpoint in one workgroup. It covers one eight-query-head GQA group,
//! 16 cached tokens, 16 of 64 value channels, the full 2,880-by-128 router, and
//! one 128-by-16 MXFP4 tile from the selected top-1 expert. Attention rows
//! 8..16 and expert rows 1..16 are native-instruction padding.

#[cfg(all(
    target_arch = "amdgpu",
    not(any(
        feature = "kernel-gpt-oss-decode",
        feature = "kernel-gpt-oss-decode-router-serial",
        feature = "kernel-gpt-oss-decode-held-fragments",
        feature = "kernel-gpt-oss-decode-scalar-attention",
        feature = "kernel-gpt-oss-decode-pipelined-attention",
        feature = "kernel-gpt-oss-decode-interleaved-stores",
        feature = "kernel-gpt-oss-router-component",
        feature = "kernel-gpt-oss-attention-component",
        feature = "kernel-gpt-oss-expert-component",
    ))
))]
compile_error!("an AMDGPU build must select one GPT-OSS kernel feature");

pub mod kernel;
pub mod kernel_components;
pub mod kernel_held_fragments;
pub mod kernel_interleaved_stores;
pub mod kernel_pipelined_attention;
pub mod kernel_router_serial;
pub mod kernel_scalar_attention;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

/// Pinned upstream implementation commit used for the architecture contract.
pub const OPENAI_GPT_OSS_COMMIT: &str = "7b583341fe16729127f6d5b94a7b09ccae97e1a1";
/// Model hidden width.
pub const HIDDEN_SIZE: usize = 2880;
/// Model attention head width.
pub const HEAD_DIM: usize = 64;
/// Model query head count.
pub const QUERY_HEADS: usize = 64;
/// Model key/value head count.
pub const KV_HEADS: usize = 8;
/// Query heads sharing one key/value head.
pub const GQA_GROUP: usize = QUERY_HEADS / KV_HEADS;
/// Model routed expert count.
pub const EXPERTS: usize = 128;
/// Experts selected for each token.
pub const TOP_K: usize = 4;
/// Published sliding-attention window.
pub const SLIDING_WINDOW: usize = 128;
/// Fixed cache extent in this bounded decode profile.
pub const CONTEXT_TOKENS: usize = 16;
/// Value/output columns retained by this profile.
pub const VALUE_TILE: usize = 16;
/// Native matrix tile rows, including eight zero padding rows.
pub const MATRIX_ROWS: usize = 16;
/// Reduction depth covered by the selected MXFP4 expert tile.
pub const EXPERT_K_TILE: usize = 128;
/// Output columns covered by the selected MXFP4 expert tile.
pub const EXPERT_N_TILE: usize = 16;
/// MXFP4 block width.
pub const MXFP4_BLOCK: usize = 32;
/// Scale-separated blocks in the expert reduction tile.
pub const MXFP4_BLOCKS: usize = EXPERT_K_TILE / MXFP4_BLOCK;
/// Attention output elements, including padded rows.
pub const ATTENTION_OUTPUT_ELEMENTS: usize = MATRIX_ROWS * VALUE_TILE;
/// Expert tile output elements, including padded rows.
pub const EXPERT_OUTPUT_ELEMENTS: usize = MATRIX_ROWS * EXPERT_N_TILE;

/// Exact boundary of the runnable tutorial profile.
pub const PROFILE_BOUNDARY: &str = "batch=1; one of eight GQA groups; context=16 (valid for both sliding-window and full-attention layers); value columns 0..16 of 64; full 128-way top-4 router over hidden=2880; selected top-1 MLP1 reduction depth 0..128 and output columns 0..16 of 2880; attention rows 8..16, expert rows 1..16, and unused reduction lanes are canonical zero padding";
