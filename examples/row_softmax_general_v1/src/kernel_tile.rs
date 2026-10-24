//! Checked 1x64 output tiles for dynamic strided F32 softmax.
//!
//! Review it as dynamic-contract validation, wave/row ownership, one uniform
//! max reduction, one uniform sum reduction, and capability-checked row stores.

#![allow(missing_docs)]

use fe2o3_device::{
    DisjointSlice, Index1D, KernelError, KernelResult, Math, StridedReadView2D, Tiled2D, kernel,
    thread,
};

#[cfg(feature = "gfx950-target")]
use fe2o3_device::Gfx950Subgroup as Subgroup;
#[cfg(not(feature = "gfx950-target"))]
use fe2o3_device::Subgroup;

pub const ROW_SOFTMAX_WORKGROUP_V1: [u32; 3] = [64, 1, 1];
pub const ROW_SOFTMAX_MAX_COLUMNS_V1: usize = 4096;

fn accessed_extent(rows: u32, columns: u32, stride: u32) -> usize {
    if rows == 0 || columns == 0 {
        return 0;
    }
    (rows - 1) as usize * stride as usize + columns as usize
}

/// Computes independent softmax rows with dynamic dimensions and strides.
///
/// Each wave owns a checked 1x64 column tile. It reduces the complete logical
/// row before storing its column tile, including a masked final tile.
/// Wide rows repeat the full-row reduction once per output tile.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
    control_flow(loop_bounds(64, 64))
)]
pub fn row_softmax_general_v1(
    input: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    rows: u32,
    columns: u32,
    input_stride: u32,
    output_stride: u32,
) -> KernelResult {
    // Prove shape and stride extents before creating the checked input view.
    if columns == 0
        || columns as usize > ROW_SOFTMAX_MAX_COLUMNS_V1
        || input_stride < columns
        || output_stride < columns
        || rows > u32::MAX / 64
        || input.len() < accessed_extent(rows, columns, input_stride)
        || output.len() < accessed_extent(rows, columns, output_stride)
    {
        return Err(KernelError::InvalidArgument);
    }
    let Ok(input) = StridedReadView2D::from_shared_slice(
        input,
        0,
        rows as usize,
        columns as usize,
        input_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };

    // One Wave64 owns one column tile; all lanes reduce the same complete row.
    let thread_index = thread::index_1d();
    let raw = thread_index.get();
    let tiles_per_row = (columns as usize + 63) / 64;
    let wave = raw / 64;
    let row = wave / tiles_per_row;
    let tile_column = wave % tiles_per_row;
    let lane = raw % 64;
    let Some(output_stripe) = thread_index.checked_tiled_2d::<64, 1, 64, 1>() else {
        return Err(KernelError::OutOfBounds);
    };

    let subgroup = Subgroup::current();
    let math = Math::current();

    // Max subtraction keeps the exponentials finite for large logits.
    let mut local_max = f32::NEG_INFINITY;
    let mut component = 0;
    while component < 64 {
        let column = lane + component * 64;
        let value = input.load_or(row, column, f32::NEG_INFINITY);
        if value > local_max {
            local_max = value;
        }
        component += 1;
    }
    #[cfg(not(feature = "gfx950-target"))]
    let maximum = subgroup.subgroup_reduce_max_f32::<64>(local_max);
    #[cfg(feature = "gfx950-target")]
    let maximum = subgroup.reduce_max_f32::<64>(local_max);

    // The second uniform reduction forms the row's FP32 normalizer.
    let mut local_sum = 0.0_f32;
    // The output tile clips the logical column tail and preserves row padding.
    let mut component = 0;
    while component < 64 {
        let column = lane + component * 64;
        local_sum += math.exp_f32(input.load_or(row, column, f32::NEG_INFINITY) - maximum);
        component += 1;
    }
    #[cfg(not(feature = "gfx950-target"))]
    let denominator = subgroup.subgroup_reduce_sum_f32::<64>(local_sum);
    #[cfg(feature = "gfx950-target")]
    let denominator = subgroup.reduce_sum_f32::<64>(local_sum);

    let column = tile_column * 64 + lane;
    if let Some(element) = output.get_tiled_2d_mut(
        &output_stripe,
        0,
        rows as usize,
        columns as usize,
        output_stride as usize,
    ) {
        *element =
            math.exp_f32(input.load_or(row, column, f32::NEG_INFINITY) - maximum) / denominator;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessed_extent_excludes_trailing_padding() {
        assert_eq!(accessed_extent(3, 5, 8), 21);
        assert_eq!(accessed_extent(0, 5, 8), 0);
    }
}
