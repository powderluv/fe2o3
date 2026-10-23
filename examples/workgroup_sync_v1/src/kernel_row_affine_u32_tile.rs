//! Checked output-tile implementation of the masked wrapping-u32 row contract.
//!
//! A 1x64 output tile has one active column per logical row. Arithmetic and
//! the uniform LDS collective remain explicit SIMT; this is not MaskedTile1D.

use fe2o3_device::{
    DisjointSlice, DynamicLds, Index1D, StridedReadView2D, Tiled2D, WorkgroupCollectives,
    WorkgroupLdsScope, kernel, thread,
};

/// Transforms up to 128 logical columns and reduces each row modulo 2^32.
///
/// Each complete 64-lane workgroup owns one row. Extra groups participate but
/// do not access input or output. Invalid views, output extents and observable
/// physical launch geometry trap uniformly before collective work or writes.
/// Complete logical workgroups are a launch precondition; a partial final
/// workgroup may fail after earlier workgroups have already written output.
#[allow(clippy::too_many_arguments)]
#[kernel(
    typed,
    launch(
        required = [64, 1, 1],
        max = [64, 1, 1],
        static_shared_memory_bytes = 256
    )
)]
pub fn row_affine_sum_u32_v1(
    values: &[u32],
    offset: usize,
    rows: usize,
    columns: usize,
    row_stride: usize,
    scale: u32,
    bias: u32,
    mut output: DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>,
) {
    if columns > 128
        || output.len() < rows
        || thread::block_dim_x() != 64
        || thread::block_dim_y() != 1
        || thread::block_dim_z() != 1
        || thread::grid_dim_y() != 1
        || thread::grid_dim_z() != 1
        || (thread::grid_dim_x() as usize) < rows
    {
        fe2o3_device::trap();
    }
    let Ok(view) = StridedReadView2D::from_shared_slice(values, offset, rows, columns, row_stride)
    else {
        fe2o3_device::trap();
    };
    let row = thread::block_idx_x() as usize;
    let lane = thread::thread_idx_x() as usize;
    let first = lane * 2;
    let second = first + 1;
    let mut local = 0_u32;
    if row < rows && first < columns {
        local = view
            .load_or(row, first, 0)
            .wrapping_mul(scale)
            .wrapping_add(bias);
    }
    if row < rows && second < columns {
        local = local.wrapping_add(
            view.load_or(row, second, 0)
                .wrapping_mul(scale)
                .wrapping_add(bias),
        );
    }

    let mut scope = WorkgroupLdsScope::current();
    let scratch = DynamicLds::<u32>::exact_current::<64>(&mut scope);
    let sum = WorkgroupCollectives::current().reduce_sum_portable(scratch, local);
    if lane == 0 && row < rows {
        let Some(tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            fe2o3_device::trap();
        };
        let Some(slot) = output.get_tiled_2d_mut(&tile, 0, rows, 1, 1) else {
            fe2o3_device::trap();
        };
        *slot = sum;
    }
}
