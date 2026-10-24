#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Dynamic routed expert computation through the common production pipeline.

#[cfg(not(any(feature = "kernel-tile-moe-expert", feature = "kernel-simt-moe-expert")))]
compile_error!("select exactly one MoE expert kernel feature");
#[cfg(all(feature = "kernel-tile-moe-expert", feature = "kernel-simt-moe-expert"))]
compile_error!("tile and SIMT MoE expert features are mutually exclusive");

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-tile-moe-expert"))]
pub mod kernel;
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-moe-expert"))]
pub mod kernel_simt;

#[cfg(not(feature = "kernel-simt-moe-expert"))]
pub const MOE_EXPERT_KERNEL_NAME_V1: &str = "moe_grouped_expert_general_v1";
#[cfg(feature = "kernel-simt-moe-expert")]
pub const MOE_EXPERT_KERNEL_NAME_V1: &str = "moe_grouped_expert_general_simt_v1";
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

pub const MOE_SAFE_SOURCE_PRESENT_V1: bool = true;
pub const MOE_SOURCE_LOWERING_SUPPORTED_V1: bool = true;
pub const MOE_QUALIFICATION_EXECUTION_SUPPORTED_V1: bool = true;
