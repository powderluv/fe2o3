#![cfg_attr(target_arch = "amdgpu", no_std)]

#[cfg(not(target_arch = "amdgpu"))]
pub mod harness;
#[cfg(not(feature = "structured-tile"))]
pub mod kernel;
#[cfg(feature = "structured-tile")]
pub mod kernel_tile;
#[cfg(feature = "structured-tile")]
pub use kernel_tile as kernel;
#[cfg(not(target_arch = "amdgpu"))]
pub mod target;
