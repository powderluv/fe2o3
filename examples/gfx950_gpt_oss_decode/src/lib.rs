#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]
#![deny(missing_docs)]

//! A 16-item GPT-OSS-120B layer-tile megakernel for gfx950.
//!
//! This is an architecture-shaped slice, not a claim to execute the complete
//! 120B checkpoint in one dispatch. Each Wave64 owns one independent bounded
//! item covering one eight-query-head GQA group, 16 cached tokens, 16 of 64
//! value channels, the full 2,880-by-128 router, and one 128-by-16 MXFP4 tile
//! from the selected top-1 expert. Attention rows 8..16 and expert rows 1..16
//! are native-instruction padding.

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

#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode",
    not(any(
        feature = "kernel-gpt-oss-decode-router-serial",
        feature = "kernel-gpt-oss-decode-held-fragments",
        feature = "kernel-gpt-oss-decode-scalar-attention",
        feature = "kernel-gpt-oss-decode-pipelined-attention",
        feature = "kernel-gpt-oss-decode-interleaved-stores",
    ))
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-router-component",
    feature = "kernel-gpt-oss-attention-component",
    feature = "kernel-gpt-oss-expert-component"
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel_components;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel_held_fragments;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub mod kernel_held_fragments_simt;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub use kernel_held_fragments_simt as kernel_held_fragments;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub mod kernel_held_fragments_tile;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub use kernel_held_fragments_tile as kernel_held_fragments;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel_interleaved_stores;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub mod kernel_interleaved_stores_simt;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub use kernel_interleaved_stores_simt as kernel_interleaved_stores;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub mod kernel_interleaved_stores_tile;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub use kernel_interleaved_stores_tile as kernel_interleaved_stores;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel_pipelined_attention;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub mod kernel_pipelined_attention_simt;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub use kernel_pipelined_attention_simt as kernel_pipelined_attention;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub mod kernel_pipelined_attention_tile;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub use kernel_pipelined_attention_tile as kernel_pipelined_attention;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-router-serial"
))]
#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
pub mod kernel_router_serial;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-scalar-attention"
))]
#[cfg(not(any(feature = "kernel-gpt-oss-mode-simt")))]
pub mod kernel_scalar_attention;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-scalar-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub mod kernel_scalar_attention_simt;
#[cfg(any(
    target_arch = "amdgpu",
    feature = "kernel-gpt-oss-decode-scalar-attention"
))]
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub use kernel_scalar_attention_simt as kernel_scalar_attention;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

/// Pinned upstream implementation commit used for the architecture contract.
pub const OPENAI_GPT_OSS_COMMIT: &str = "7b583341fe16729127f6d5b94a7b09ccae97e1a1";
/// Hardware wave width used by every matrix and subgroup operation.
pub const WAVE_SIZE: usize = 64;
/// Exact workgroup width: four independent Wave64 items.
pub const WORKGROUP_SIZE: usize = 256;
/// Useful Wave64 items in one workgroup.
pub const WAVES_PER_WORKGROUP: usize = WORKGROUP_SIZE / WAVE_SIZE;
/// Maximum admitted one-dimensional workgroup count.
pub const MAX_WORKGROUPS: usize = 4;
/// Independent items covered by the maximum launch.
pub const PROFILE_ITEMS: usize = WAVES_PER_WORKGROUP * MAX_WORKGROUPS;
/// Route words are replicated once per lane to retain identity-mapped ownership.
pub const PACKED_ROUTE_ELEMENTS: usize = PROFILE_ITEMS * WAVE_SIZE;
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
/// Attention elements in one item, including padded rows.
pub const ATTENTION_TILE_ELEMENTS: usize = MATRIX_ROWS * VALUE_TILE;
/// Attention output elements across the maximum launch.
pub const ATTENTION_OUTPUT_ELEMENTS: usize = PROFILE_ITEMS * ATTENTION_TILE_ELEMENTS;
/// Expert elements in one item, including padded rows.
pub const EXPERT_TILE_ELEMENTS: usize = MATRIX_ROWS * EXPERT_N_TILE;
/// Expert output elements across the maximum launch.
pub const EXPERT_OUTPUT_ELEMENTS: usize = PROFILE_ITEMS * EXPERT_TILE_ELEMENTS;

/// Exact boundary of the runnable tutorial profile.
pub const PROFILE_BOUNDARY: &str = "16 independent bounded items at grid=[4,1,1] and workgroup=[256,1,1], one item per Wave64; each item covers batch=1, one of eight GQA groups, context=16 (valid for both sliding-window and full-attention layers), value columns 0..16 of 64, a full 128-way top-4 router over hidden=2880, and selected top-1 MLP1 reduction depth 0..128 with output columns 0..16 of 2880; attention rows 8..16, expert rows 1..16, and unused reduction lanes are canonical zero padding";

#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
))]
compile_error!("GPT-OSS SIMT and tile modes are mutually exclusive");
#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    not(any(
        feature = "kernel-gpt-oss-decode-held-fragments",
        feature = "kernel-gpt-oss-decode-interleaved-stores",
        feature = "kernel-gpt-oss-decode-pipelined-attention",
        feature = "kernel-gpt-oss-decode-scalar-attention"
    ))
))]
pub mod kernel_simt;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
pub use kernel_held_fragments as kernel_simt;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
pub use kernel_interleaved_stores as kernel_simt;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
pub use kernel_pipelined_attention as kernel_simt;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-decode-scalar-attention"
))]
pub use kernel_scalar_attention as kernel_simt;
#[cfg(feature = "kernel-gpt-oss-mode-simt")]
pub mod kernel_simt_components;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-tile",
    not(any(
        feature = "kernel-gpt-oss-decode-held-fragments",
        feature = "kernel-gpt-oss-decode-interleaved-stores",
        feature = "kernel-gpt-oss-decode-pipelined-attention"
    ))
))]
pub mod kernel_tile;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-tile",
    feature = "kernel-gpt-oss-decode-held-fragments"
))]
pub use kernel_held_fragments as kernel_tile;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-tile",
    feature = "kernel-gpt-oss-decode-interleaved-stores"
))]
pub use kernel_interleaved_stores as kernel_tile;
#[cfg(all(
    feature = "kernel-gpt-oss-mode-tile",
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
pub use kernel_pipelined_attention as kernel_tile;
#[cfg(feature = "kernel-gpt-oss-mode-tile")]
pub mod kernel_tile_components;
