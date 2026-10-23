#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Rust-first gfx950 low-precision source examples.
//!
//! The attributed functions in [`kernel`] are the fe2o3 kernel source. The
//! neighboring HIP file is a compiler/ISA/hardware fixture, not the source of
//! these Rust kernels.

#[cfg(all(
    target_arch = "amdgpu",
    not(any(
        feature = "kernel-fp4-gemm",
        feature = "kernel-fp8-gemm",
        feature = "kernel-simt-fp8-gemm",
        feature = "kernel-simt-fp4-gemm",
        feature = "kernel-mixed-fp4-fp8-gemm",
        feature = "kernel-fp4-attention",
        feature = "kernel-fp8-attention",
        feature = "kernel-simt-fp8-attention",
        feature = "kernel-simt-fp4-attention",
    ))
))]
compile_error!("an AMDGPU build must select exactly one gfx950 kernel feature");

#[cfg(all(
    target_arch = "amdgpu",
    any(
        all(
            feature = "kernel-simt-fp4-attention",
            any(
                feature = "kernel-fp4-gemm",
                feature = "kernel-fp8-gemm",
                feature = "kernel-simt-fp8-gemm",
                feature = "kernel-simt-fp4-gemm",
                feature = "kernel-mixed-fp4-fp8-gemm",
                feature = "kernel-fp4-attention",
                feature = "kernel-fp8-attention",
                feature = "kernel-simt-fp8-attention",
            )
        ),
        all(
            feature = "kernel-simt-fp8-attention",
            any(
                feature = "kernel-fp4-gemm",
                feature = "kernel-fp8-gemm",
                feature = "kernel-simt-fp8-gemm",
                feature = "kernel-simt-fp4-gemm",
                feature = "kernel-mixed-fp4-fp8-gemm",
                feature = "kernel-fp4-attention",
                feature = "kernel-fp8-attention",
            )
        ),
        all(
            feature = "kernel-simt-fp4-gemm",
            any(
                feature = "kernel-fp4-gemm",
                feature = "kernel-fp8-gemm",
                feature = "kernel-simt-fp8-gemm",
                feature = "kernel-mixed-fp4-fp8-gemm",
                feature = "kernel-fp4-attention",
                feature = "kernel-fp8-attention",
            )
        ),
        all(
            feature = "kernel-simt-fp8-gemm",
            any(
                feature = "kernel-fp4-gemm",
                feature = "kernel-fp8-gemm",
                feature = "kernel-mixed-fp4-fp8-gemm",
                feature = "kernel-fp4-attention",
                feature = "kernel-fp8-attention",
            )
        ),
        all(feature = "kernel-fp4-gemm", feature = "kernel-fp8-gemm"),
        all(feature = "kernel-fp4-gemm", feature = "kernel-fp4-attention"),
        all(feature = "kernel-fp4-gemm", feature = "kernel-fp8-attention"),
        all(feature = "kernel-fp8-gemm", feature = "kernel-fp4-attention"),
        all(feature = "kernel-fp8-gemm", feature = "kernel-fp8-attention"),
        all(feature = "kernel-fp4-attention", feature = "kernel-fp8-attention"),
        all(feature = "kernel-mixed-fp4-fp8-gemm", feature = "kernel-fp4-gemm"),
        all(feature = "kernel-mixed-fp4-fp8-gemm", feature = "kernel-fp8-gemm"),
        all(
            feature = "kernel-mixed-fp4-fp8-gemm",
            feature = "kernel-fp4-attention"
        ),
        all(
            feature = "kernel-mixed-fp4-fp8-gemm",
            feature = "kernel-fp8-attention"
        ),
    )
))]
compile_error!("an AMDGPU build must not select more than one gfx950 kernel feature");

mod dimensions;
pub mod kernel;
#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-fp8-attention",
    feature = "kernel-simt-fp8-gemm",
    feature = "kernel-simt-fp8-attention"
))]
pub mod kernel_simt;
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-fp8-attention"))]
pub mod kernel_simt_attention;
#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-simt-fp4-gemm",
    feature = "kernel-simt-fp4-attention"
))]
pub mod kernel_simt_fp4;
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-fp4-attention"))]
pub mod kernel_simt_fp4_attention;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

/// The ordinary Rust kernel source exists and is checked by host compilation.
pub const GFX950_RUST_KERNEL_SOURCE_PRESENT_V1: bool = true;

/// Historical production lowering claim for the four original pure-format kernels.
/// This does not qualify the separately named mixed FP4-A/FP8-B CPU example.
pub const GFX950_RUST_TO_HSACO_LOWERING_SUPPORTED_V1: bool = true;

/// Exact production finalization contract used by the runnable examples.
pub const GFX950_PRODUCTION_FINALIZER_V1: &str = "ROCm 7.2.1 clang/LLD with implicit device libraries disabled and the manifest-pinned nine-file gfx950 OCML closure";

/// Remaining boundary for protected Worker V3 publication, not Rust lowering.
pub const GFX950_PROTECTED_WORKER_BUILD_BOUNDARY_V1: &str = "the reviewed gfx950 Worker V3 provider is implemented and admission-tested, but a measured native worker build still requires a matching LLVM/LLD development package";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_lowering_and_protected_boundary_are_explicit() {
        let source = include_str!("lib.rs");
        assert!(source.contains("GFX950_RUST_KERNEL_SOURCE_PRESENT_V1: bool = true"));
        assert!(source.contains("GFX950_RUST_TO_HSACO_LOWERING_SUPPORTED_V1: bool = true"));
        assert!(GFX950_PRODUCTION_FINALIZER_V1.contains("implicit device libraries disabled"));
        assert!(GFX950_PRODUCTION_FINALIZER_V1.contains("nine-file gfx950 OCML closure"));
        assert!(GFX950_PROTECTED_WORKER_BUILD_BOUNDARY_V1.contains("admission-tested"));
        assert!(GFX950_PROTECTED_WORKER_BUILD_BOUNDARY_V1.contains("LLVM/LLD development package"));
    }
}
