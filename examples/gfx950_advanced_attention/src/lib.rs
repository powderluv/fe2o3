#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Fixed-shape Rust gfx950 advanced-attention kernels.
//!
//! The attributed functions are ordinary Rust source with independent safe CPU
//! references. AMDGPU builds select exactly one production kernel root.

#[cfg(all(
    target_arch = "amdgpu",
    not(any(
        feature = "kernel-kda-decode",
        feature = "kernel-kda-prefill",
        feature = "kernel-content-sparse-attention",
        feature = "kernel-compressed-hybrid-attention",
        feature = "kernel-attnres-aggregate",
        feature = "kernel-four-branch-residual",
        feature = "kernel-mhc-sinkhorn-mix",
    ))
))]
compile_error!("an AMDGPU build must select exactly one advanced-attention kernel feature");

#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(feature = "kernel-kda-decode", feature = "kernel-kda-prefill"),
        all(
            feature = "kernel-kda-decode",
            feature = "kernel-content-sparse-attention"
        ),
        all(
            feature = "kernel-kda-decode",
            feature = "kernel-compressed-hybrid-attention"
        ),
        all(feature = "kernel-kda-decode", feature = "kernel-attnres-aggregate"),
        all(feature = "kernel-kda-decode", feature = "kernel-four-branch-residual"),
        all(feature = "kernel-kda-decode", feature = "kernel-mhc-sinkhorn-mix"),
        all(
            feature = "kernel-kda-prefill",
            feature = "kernel-content-sparse-attention"
        ),
        all(
            feature = "kernel-kda-prefill",
            feature = "kernel-compressed-hybrid-attention"
        ),
        all(feature = "kernel-kda-prefill", feature = "kernel-attnres-aggregate"),
        all(
            feature = "kernel-kda-prefill",
            feature = "kernel-four-branch-residual"
        ),
        all(feature = "kernel-kda-prefill", feature = "kernel-mhc-sinkhorn-mix"),
        all(
            feature = "kernel-content-sparse-attention",
            feature = "kernel-compressed-hybrid-attention"
        ),
        all(
            feature = "kernel-content-sparse-attention",
            feature = "kernel-attnres-aggregate"
        ),
        all(
            feature = "kernel-content-sparse-attention",
            feature = "kernel-four-branch-residual"
        ),
        all(
            feature = "kernel-content-sparse-attention",
            feature = "kernel-mhc-sinkhorn-mix"
        ),
        all(
            feature = "kernel-compressed-hybrid-attention",
            feature = "kernel-attnres-aggregate"
        ),
        all(
            feature = "kernel-compressed-hybrid-attention",
            feature = "kernel-four-branch-residual"
        ),
        all(
            feature = "kernel-compressed-hybrid-attention",
            feature = "kernel-mhc-sinkhorn-mix"
        ),
        all(
            feature = "kernel-attnres-aggregate",
            feature = "kernel-four-branch-residual"
        ),
        all(
            feature = "kernel-attnres-aggregate",
            feature = "kernel-mhc-sinkhorn-mix"
        ),
        all(
            feature = "kernel-four-branch-residual",
            feature = "kernel-mhc-sinkhorn-mix"
        ),
    )
))]
compile_error!("an AMDGPU build must not select more than one advanced-attention kernel feature");

#[cfg(all(
    target_arch = "amdgpu",
    any(
        feature = "kernel-kda-decode-wave-tiled-v1",
        feature = "kernel-content-sparse-attention-reciprocal-reuse-v1",
        feature = "kernel-compressed-hybrid-attention-division-baseline-v1",
        feature = "kernel-attnres-aggregate-explicit-reuse-v1",
        feature = "kernel-four-branch-residual-explicit-v1",
        feature = "kernel-mhc-sinkhorn-mix-scalar-v1",
    )
))]
mod ablation;

pub mod kernel;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

/// Channels written by the recurrent and residual-mixing profiles.
pub const CHANNELS_V1: usize = 16;
/// History taps consumed by each KDA/GDN recurrence step.
pub const KDA_TAPS_V1: usize = 3;
/// Tokens in the fixed prefill recurrence.
pub const PREFILL_TOKENS_V1: usize = 8;
/// Tokens in the sparse and compressed-hybrid attention fixtures.
pub const ATTENTION_TOKENS_V1: usize = 16;
/// Reduction depth of each quantized attention score.
pub const HEAD_DIMENSION_V1: usize = 128;
/// Blocks considered by indexed sparse attention.
pub const SPARSE_BLOCKS_V1: usize = 4;
/// Tokens in each sparse-attention block.
pub const TOKENS_PER_BLOCK_V1: usize = 4;
/// Blocks retained by the content selector.
pub const SELECTED_BLOCKS_V1: usize = 2;
/// Tokens retained after block and token ranking.
pub const SELECTED_TOKENS_V1: usize = 3;
/// Residual depths, branches, and streams in the bounded mixing profiles.
pub const MIXING_STREAMS_V1: usize = 4;
/// Sinkhorn row/column normalization iterations.
pub const SINKHORN_ITERATIONS_V1: usize = 3;

/// Exact workgroup dimensions declared by every teaching kernel.
pub const GFX950_ADVANCED_ATTENTION_WORKGROUP_V1: [u32; 3] = [64, 1, 1];
/// Exact grid dimensions declared by every teaching kernel.
pub const GFX950_ADVANCED_ATTENTION_GRID_V1: [u32; 3] = [1, 1, 1];
/// Whether the seven source roots use the production semantic lowering surface.
pub const GFX950_ADVANCED_ATTENTION_SOURCE_LOWERING_SUPPORTED_V1: bool = true;
/// Boundary not established by the production source-lowering and runtime suite.
pub const GFX950_ADVANCED_ATTENTION_SOURCE_BLOCKER_V1: &str = "the retained production extraction, finalization, ISA inspection, and gfx950 numerical runs do not establish formal compiler refinement, protected publication authority, performance, or full-model behavior";
