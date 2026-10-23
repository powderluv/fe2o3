#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]
#![deny(missing_docs)]

//! Ordinary attributed Rust source and independent CPU references for bounded
//! gfx950 ML systems kernels.
//!
//! The Rust source is the fe2o3 tutorial implementation. The adjacent HIP
//! program remains a separate compiler and ISA-validation companion.

#[cfg(all(
    target_arch = "amdgpu",
    not(any(
        feature = "kernel-moe-route",
        feature = "kernel-moe-expert-rank",
        feature = "kernel-combine-expert-ranks",
        feature = "kernel-speculative-transaction",
        feature = "kernel-qwen-ngram-gather",
        feature = "kernel-stage-gradient-shard",
        feature = "kernel-muon-update",
    ))
))]
compile_error!("an AMDGPU build must select exactly one gfx950 advanced-systems kernel feature");

#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(
            feature = "kernel-moe-route",
            any(
                feature = "kernel-moe-expert-rank",
                feature = "kernel-combine-expert-ranks",
                feature = "kernel-speculative-transaction",
                feature = "kernel-qwen-ngram-gather",
                feature = "kernel-stage-gradient-shard",
                feature = "kernel-muon-update"
            )
        ),
        all(
            feature = "kernel-moe-expert-rank",
            any(
                feature = "kernel-combine-expert-ranks",
                feature = "kernel-speculative-transaction",
                feature = "kernel-qwen-ngram-gather",
                feature = "kernel-stage-gradient-shard",
                feature = "kernel-muon-update"
            )
        ),
        all(
            feature = "kernel-combine-expert-ranks",
            any(
                feature = "kernel-speculative-transaction",
                feature = "kernel-qwen-ngram-gather",
                feature = "kernel-stage-gradient-shard",
                feature = "kernel-muon-update"
            )
        ),
        all(
            feature = "kernel-speculative-transaction",
            any(
                feature = "kernel-qwen-ngram-gather",
                feature = "kernel-stage-gradient-shard",
                feature = "kernel-muon-update"
            )
        ),
        all(
            feature = "kernel-qwen-ngram-gather",
            any(
                feature = "kernel-stage-gradient-shard",
                feature = "kernel-muon-update"
            )
        ),
        all(
            feature = "kernel-stage-gradient-shard",
            feature = "kernel-muon-update"
        ),
    )
))]
compile_error!(
    "an AMDGPU build must not select more than one gfx950 advanced-systems kernel feature"
);

#[cfg(all(target_arch = "amdgpu", feature = "ablation-route-owner-only"))]
compile_error!(
    "ablation-route-owner-only is rejected because lane-conditional induction does not satisfy production semantic-to-ranked projection"
);
#[cfg(all(target_arch = "amdgpu", feature = "ablation-route-unpacked"))]
compile_error!("ablation-route-unpacked is retained only in the rejected-variant registry");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-expert-serial",
    not(feature = "kernel-moe-expert-rank")
))]
compile_error!("ablation-expert-serial requires kernel-moe-expert-rank");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-combine-transposed",
    not(feature = "kernel-combine-expert-ranks")
))]
compile_error!("ablation-combine-transposed requires kernel-combine-expert-ranks");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-speculative-recompute-prefix",
    not(feature = "kernel-speculative-transaction")
))]
compile_error!("ablation-speculative-recompute-prefix requires kernel-speculative-transaction");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-ngram-reverse-probe",
    not(feature = "kernel-qwen-ngram-gather")
))]
compile_error!("ablation-ngram-reverse-probe requires kernel-qwen-ngram-gather");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-stage-tile4",
    not(feature = "kernel-stage-gradient-shard")
))]
compile_error!("ablation-stage-tile4 requires kernel-stage-gradient-shard");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "ablation-muon-broadcast16",
    not(feature = "kernel-muon-update")
))]
compile_error!("ablation-muon-broadcast16 requires kernel-muon-update");

mod dimensions;
#[cfg(feature = "kernel-moe-expert-rank-simt")]
mod kernel_simt_moe_expert_rank;
#[cfg(feature = "kernel-moe-route-tile")]
mod kernel_tile_moe_route;
#[cfg(feature = "kernel-combine-expert-ranks-tile")]
mod kernel_tile_combine_expert_ranks;
pub mod kernel;
pub use dimensions::*;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

/// The ordinary Rust kernel sources are present and host-checked.
pub const GFX950_ADVANCED_SYSTEMS_RUST_SOURCE_PRESENT_V1: bool = true;
/// Whether all seven source roots use the production semantic lowering surface.
pub const GFX950_ADVANCED_SYSTEMS_SOURCE_LOWERING_SUPPORTED: bool = true;
/// Boundary not established by the production source-lowering and runtime suite.
pub const GFX950_ADVANCED_SYSTEMS_SOURCE_BLOCKER: &str = "the retained production extraction, finalization, ISA inspection, and gfx950 numerical runs do not establish formal compiler refinement, protected publication authority, performance, distributed-runtime behavior, or full-model behavior";

#[cfg(feature = "kernel-stage-gradient-shard-tile")]
mod kernel_tile_stage_gradient_shard;

#[cfg(feature = "kernel-qwen-ngram-gather-tile")]
mod kernel_tile_qwen_ngram_gather;

#[cfg(feature = "kernel-muon-update-tile")]
mod kernel_tile_muon_update;
