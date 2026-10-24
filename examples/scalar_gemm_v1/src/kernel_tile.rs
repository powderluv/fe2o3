//! Ordered F32 GEMM with checked 1x64 output memory tiles.
//!
//! Each in-range thread owns one C element, derives its row and column, evaluates
//! the K recurrence in strict order, and stores through a disjoint capability.

use fe2o3_device::{DisjointSlice, Index1D, Tiled2D, kernel, thread};

// The compiler qualification profile authenticates this explicit recurrence and division guard.
#[allow(
    clippy::assign_op_pattern,
    clippy::collapsible_if,
    clippy::manual_checked_ops
)]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1]),
    control_flow(loop_bounds(4294967295))
)]
/// Computes row-major `C = A * B` with one output element per in-range thread.
pub fn scalar_gemm_v1(
    a: &[f32],
    b: &[f32],
    mut c: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    m: u32,
    n: u32,
    k: u32,
) {
    // Flattened grid ownership gives each participating thread one output index.
    let index = thread::index_1d();
    let p = index.get();
    let n_index = n as usize;
    let output_extent = (m as usize) * n_index;
    if n_index != 0 {
        if p < output_extent {
            let row = (p / n_index) as u32;
            let col = (p % n_index) as u32;
            // Preserve the explicit K recurrence authenticated by the compiler profile.
            let mut accumulator = 0.0_f32;
            let mut t = 0_u32;
            while t < k {
                let a_index = (row as usize) * (k as usize) + (t as usize);
                let b_index = (t as usize) * n_index + (col as usize);
                let product = a[a_index] * b[b_index];
                accumulator = accumulator + product;
                t = t + 1;
            }
            // DisjointSlice turns the ownership mapping into the store permission.
            // Contiguous 1x64 tiles cover the flattened C storage, with a masked tail.
            let Some(tile) = index.checked_tiled_2d::<64, 1, 64, 1>() else {
                return;
            };
            if let Some(output) = c.get_tiled_2d_mut(&tile, 0, 1, output_extent, output_extent) {
                *output = accumulator;
            }
        }
    }
}
