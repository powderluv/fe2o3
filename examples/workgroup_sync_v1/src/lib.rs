#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Source, oracle, and formal contracts for two fixed wave64 synchronization profiles.
//!
//! Both kernels are type-checked as ordinary attributed Rust and enter the
//! same production compiler, artifact, and protected runtime pipeline.

#[cfg(not(target_arch = "amdgpu"))]
pub mod contract;
#[cfg(feature = "lds-kernel")]
pub mod kernel;
#[cfg(feature = "lds-f32-kernel")]
mod kernel_f32;
#[cfg(all(
    feature = "row-affine-u32-kernel",
    feature = "row-affine-u32-tile-kernel"
))]
compile_error!("select only one row-affine kernel feature");
#[cfg(feature = "row-affine-u32-kernel")]
mod kernel_row_affine_u32;
#[cfg(feature = "row-affine-u32-tile-kernel")]
mod kernel_row_affine_u32_tile;
#[cfg(any(
    feature = "lds-scan-f32-kernel",
    feature = "lds-scan-f32-3-kernel",
    feature = "lds-scan-f32-65-kernel"
))]
mod kernel_scan_f32;
#[cfg(any(
    feature = "lds-scan-f32-exclusive-kernel",
    feature = "lds-scan-f32-exclusive-3-kernel",
    feature = "lds-scan-f32-exclusive-255-kernel"
))]
mod kernel_scan_f32_exclusive;
#[cfg(any(
    feature = "lds-scan-i32-kernel",
    feature = "lds-scan-i32-3-kernel",
    feature = "lds-scan-i32-255-kernel"
))]
mod kernel_scan_i32;
#[cfg(any(
    feature = "lds-scan-i32-inclusive-kernel",
    feature = "lds-scan-i32-inclusive-65-kernel",
    feature = "lds-scan-i32-inclusive-255-kernel"
))]
mod kernel_scan_i32_inclusive;
#[cfg(any(
    feature = "lds-scan-u32-kernel",
    feature = "lds-scan-u32-65-kernel",
    feature = "lds-scan-u32-255-kernel"
))]
mod kernel_scan_u32;
#[cfg(any(
    feature = "lds-scan-u32-exclusive-kernel",
    feature = "lds-scan-u32-exclusive-3-kernel",
    feature = "lds-scan-u32-exclusive-65-kernel"
))]
mod kernel_scan_u32_exclusive;
#[cfg(feature = "lds-u32-kernel")]
mod kernel_u32;
#[cfg(not(target_arch = "amdgpu"))]
pub mod row_affine_oracle;
#[cfg(feature = "scoped-atomic-kernel")]
pub mod scoped_atomic;
#[cfg(not(target_arch = "amdgpu"))]
pub mod vectors;

#[cfg(not(target_arch = "amdgpu"))]
pub use contract::{
    ATOMIC_ADDRESS_SPACE_V1, ATOMIC_ORDERING_V1, ATOMIC_SCOPE_V1, AtomicAddressSpaceV1,
    AtomicComparisonErrorV1, AtomicLaneV1, AtomicOrderingV1, AtomicProfileErrorV1, AtomicProfileV1,
    AtomicScopeV1, LDS_OWNER_LANE_V1, ReductionComparisonErrorV1, ReductionLaneV1,
    ReductionProfileErrorV1, WORKGROUP_LANES_V1, atomic_add_oracle_v1, canonical_atomic_lanes_v1,
    canonical_atomic_profile_v1, canonical_reduction_trace_v1, compare_atomic_output_v1,
    compare_reduction_output_v1, lds_reduction_oracle_v1,
};
#[cfg(feature = "lds-kernel")]
pub use kernel::LDS_REDUCTION_WORKGROUP_V1;
#[cfg(not(target_arch = "amdgpu"))]
pub use vectors::{AtomicVectorV1, ReductionVectorV1, atomic_vectors_v1, reduction_vectors_v1};
