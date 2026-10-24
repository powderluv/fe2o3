#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! One mutable GEMM candidate and an immutable independent reference.

#[cfg(not(feature = "kernel-simt-autoresearch"))]
pub mod kernel;
#[cfg(feature = "kernel-simt-autoresearch")]
pub mod kernel_simt;
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;
