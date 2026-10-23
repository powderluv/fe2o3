#![no_std]

#[cfg(not(target_arch = "amdgpu"))]
extern crate std;

use fe2o3_device::{DisjointSlice, kernel, thread};

include!("vecadd_body.rs");

macro_rules! production_f32_add {
    ($lhs:expr, $rhs:expr) => {{ $lhs + $rhs }};
}

#[cfg(not(feature = "structured-tile"))]
#[kernel(typed)]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    vecadd_kernel_body!(thread, (), production_f32_add, a, b, c);
}

#[cfg(any(feature = "structured-tile", test))]
include!("vecadd_tile_body.rs");

#[cfg(feature = "structured-tile")]
#[kernel(typed)]
pub fn vecadd(
    a: &[f32],
    b: &[f32],
    mut c: DisjointSlice<f32, fe2o3_device::Tiled2D<fe2o3_device::Index1D, 64, 1, 64, 1>>,
) {
    vecadd_tile_body!(thread, (), production_f32_add, a, b, c);
}

#[cfg(test)]
mod host_reference_tests;

#[cfg(not(target_arch = "amdgpu"))]
pub mod target;
