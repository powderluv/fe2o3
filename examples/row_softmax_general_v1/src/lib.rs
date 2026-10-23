#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Dynamic multi-row softmax through the workload-neutral production pipeline.

#[cfg(not(feature = "structured-tile"))]
pub mod kernel;
#[cfg(feature = "structured-tile")]
pub mod kernel_tile;
#[cfg(feature = "structured-tile")]
pub use kernel_tile as kernel;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;
#[cfg(not(target_arch = "amdgpu"))]
pub mod target;

pub const ROW_SOFTMAX_SAFE_SOURCE_PRESENT_V1: bool = true;
pub const ROW_SOFTMAX_SOURCE_LOWERING_SUPPORTED_V1: bool = true;
pub const ROW_SOFTMAX_QUALIFICATION_EXECUTION_SUPPORTED_V1: bool = true;
