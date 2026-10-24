#![forbid(unsafe_code)]
#![cfg_attr(target_arch = "amdgpu", no_std)]

//! Dynamic fused attention through the workload-neutral production pipeline.

#[cfg(not(any(
    feature = "kernel-tile-flash-attention",
    feature = "kernel-simt-flash-attention"
)))]
compile_error!("select exactly one flash attention kernel feature");
#[cfg(all(
    feature = "kernel-tile-flash-attention",
    feature = "kernel-simt-flash-attention"
))]
compile_error!("tile and SIMT flash attention features are mutually exclusive");

#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-tile-flash-attention"))]
pub mod kernel;
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-simt-flash-attention"))]
pub mod kernel_simt;

#[cfg(not(feature = "kernel-simt-flash-attention"))]
pub const FLASH_ATTENTION_KERNEL_NAME_V1: &str = "flash_attention_general_v1";
#[cfg(feature = "kernel-simt-flash-attention")]
pub const FLASH_ATTENTION_KERNEL_NAME_V1: &str = "flash_attention_general_simt_v1";
#[cfg(not(target_arch = "amdgpu"))]
pub mod reference;

pub const FLASH_ATTENTION_SAFE_SOURCE_PRESENT_V1: bool = true;
pub const FLASH_ATTENTION_SOURCE_LOWERING_SUPPORTED_V1: bool = true;
pub const FLASH_ATTENTION_QUALIFICATION_EXECUTION_SUPPORTED_V1: bool = true;
pub const FLASH_ATTENTION_PROTECTED_EXECUTION_SUPPORTED_V1: bool = false;
pub const FLASH_ATTENTION_PROTECTED_EXECUTION_BLOCKER_V1: &str = "protected Worker publication and artifact-currentness admission remain separate from the qualification runner";
