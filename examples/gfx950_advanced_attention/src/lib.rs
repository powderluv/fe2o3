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
        feature = "kernel-deepseek-sparse-attention",
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
            feature = "kernel-deepseek-sparse-attention"
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
            feature = "kernel-deepseek-sparse-attention"
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
            feature = "kernel-deepseek-sparse-attention"
        ),
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
            feature = "kernel-deepseek-sparse-attention",
            feature = "kernel-compressed-hybrid-attention"
        ),
        all(
            feature = "kernel-deepseek-sparse-attention",
            feature = "kernel-attnres-aggregate"
        ),
        all(
            feature = "kernel-deepseek-sparse-attention",
            feature = "kernel-four-branch-residual"
        ),
        all(
            feature = "kernel-deepseek-sparse-attention",
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
        feature = "kernel-content-sparse-attention-reciprocal-reuse-v1",
        feature = "kernel-compressed-hybrid-attention-division-baseline-v1",
        feature = "kernel-attnres-aggregate-explicit-reuse-v1",
        feature = "kernel-four-branch-residual-explicit-v1",
        feature = "kernel-mhc-sinkhorn-mix-scalar-v1",
    )
))]
mod ablation;

#[cfg(all(
    target_arch = "amdgpu",
    any(
        feature = "kernel-kda-decode-baseline-v1",
        feature = "kernel-kda-prefill-baseline-v1"
    )
))]
mod kda_baseline;

#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(feature = "kernel-attnres-aggregate-simt", feature = "kernel-attnres-aggregate-tile"),
        all(feature = "kernel-attnres-aggregate-simt", feature = "kernel-attnres-aggregate-explicit-reuse-v1"),
        all(feature = "kernel-attnres-aggregate-tile", feature = "kernel-attnres-aggregate-explicit-reuse-v1"),
        all(feature = "kernel-four-branch-residual-simt", feature = "kernel-four-branch-residual-tile"),
        all(feature = "kernel-four-branch-residual-simt", feature = "kernel-four-branch-residual-explicit-v1"),
        all(feature = "kernel-four-branch-residual-tile", feature = "kernel-four-branch-residual-explicit-v1"),
        all(feature = "kernel-mhc-sinkhorn-mix-simt", feature = "kernel-mhc-sinkhorn-mix-tile"),
        all(feature = "kernel-mhc-sinkhorn-mix-simt", feature = "kernel-mhc-sinkhorn-mix-scalar-v1"),
        all(feature = "kernel-mhc-sinkhorn-mix-tile", feature = "kernel-mhc-sinkhorn-mix-scalar-v1")
    )
))]
compile_error!("select one SIMT, tile, or legacy ablation mode for an advanced mixing kernel");

#[cfg(all(
    target_arch = "amdgpu",
    any(
        feature = "kernel-attnres-aggregate-tile",
        feature = "kernel-four-branch-residual-tile",
        feature = "kernel-mhc-sinkhorn-mix-tile"
    )
))]
mod kernel_tile_mixing;

#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(
            feature = "kernel-kda-decode-tile",
            feature = "kernel-kda-decode-baseline-v1"
        ),
        all(
            feature = "kernel-kda-prefill-tile",
            feature = "kernel-kda-prefill-baseline-v1"
        )
    )
))]
compile_error!("a KDA tile feature must not be combined with its canonical ablation");

pub mod kernel;
#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(
            feature = "kernel-content-sparse-attention-simt",
            feature = "kernel-content-sparse-attention-reciprocal-reuse-v1"
        ),
        all(
            feature = "kernel-compressed-hybrid-attention-simt",
            feature = "kernel-compressed-hybrid-attention-division-baseline-v1"
        )
    )
))]
compile_error!("an attention SIMT feature must not be combined with its tile ablation feature");
#[cfg(all(
    target_arch = "amdgpu",
    feature = "kernel-compressed-hybrid-attention-simt"
))]
mod kernel_simt_compressed_hybrid;
#[cfg(all(
    target_arch = "amdgpu",
    feature = "kernel-content-sparse-attention-simt"
))]
mod kernel_simt_content_sparse;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

mod dimensions;
pub use dimensions::*;
