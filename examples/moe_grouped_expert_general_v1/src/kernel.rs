//! Safe Rust MFMA expert projection for one dynamically selected route group.
//!
//! Review it as extent validation, wave/tile ownership, checked route and matrix
//! views, a uniform K reduction, then a gated epilogue with disjoint tiled stores.

#![allow(missing_docs)]

use fe2o3_device::{
    Bf16MfmaAMatrix, Bf16MfmaBMatrix, DisjointSlice, F32AccumulatorFragment, Index1D, KernelError,
    KernelResult, Matrix, StridedReadView2D, Tiled2D, Wave64, WaveLane, kernel, thread,
};

pub const MOE_EXPERT_WORKGROUP_V1: [u32; 3] = [64, 1, 1];

// Same-width 0/1 flag and value keep constant returns free of padding.
// Callers reject every flag other than 1 before using the value.
// The complement form is exactly lhs + rhs, with each subtraction bounded.
// Retain the existing full-width arithmetic regression oracle below.
// Native validation keeps the same complement guards in its own CFG.
#[cfg(test)]
fn checked_extent_add(lhs: usize, rhs: usize) -> (usize, usize) {
    let available = usize::MAX - lhs;
    if rhs > available {
        return (0, 0);
    }
    let remaining = available - rhs;
    (1, usize::MAX - remaining)
}

fn matrix_extent(rows: u32, columns: u32, stride: u32) -> usize {
    if rows == 0 || columns == 0 {
        0
    } else {
        (rows - 1) as usize * stride as usize + columns as usize
    }
}

/// Computes one routed expert group with a gated bias epilogue.
///
/// Routing packs each expert's selected tokens into a separate padded matrix.
/// The same kernel is launched for every nonempty expert; the expert argument
/// selects a strided weight and bias matrix without changing the pipeline.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
    control_flow(loop_bounds(4294967295))
)]
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_expert_general_v1(
    routed_tokens: &[u16],
    expert_weights: &[u16],
    route_gates: &[f32],
    expert_bias: &[f32],
    mut routed_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    rows_padded: u32,
    output_columns: u32,
    reduction: u32,
    token_stride: u32,
    weight_stride: u32,
    expert_weight_stride: u32,
    bias_stride: u32,
    output_stride: u32,
    expert: u32,
    expert_count: u32,
) -> KernelResult {
    // Prove all dynamic extents and selected-expert offsets before MFMA.
    let token_extent = matrix_extent(rows_padded, reduction, token_stride);
    let weight_extent = if expert_count == 0 || reduction == 0 || output_columns == 0 {
        0
    } else {
        // Keep both checked-add guards directly on uniform kernel dimensions.
        // The remaining capacity after the first sum is MAX - that exact sum.
        let expert_offset = (expert_count - 1) as usize * expert_weight_stride as usize;
        let reduction_offset = (reduction - 1) as usize * weight_stride as usize;
        let after_experts = usize::MAX - expert_offset;
        if reduction_offset > after_experts {
            return Err(KernelError::InvalidArgument);
        }
        let after_reduction = after_experts - reduction_offset;
        if output_columns as usize > after_reduction {
            return Err(KernelError::InvalidArgument);
        }
        usize::MAX - (after_reduction - output_columns as usize)
    };
    let bias_extent = matrix_extent(expert_count, output_columns, bias_stride);
    let output_extent = matrix_extent(rows_padded, output_columns, output_stride);
    if rows_padded == 0
        || rows_padded % 16 != 0
        || output_columns == 0
        || reduction == 0
        || token_stride < reduction
        || weight_stride < output_columns
        || (expert_weight_stride as u64) < reduction as u64 * weight_stride as u64
        || bias_stride < output_columns
        || output_stride < output_columns
        || expert >= expert_count
        || routed_tokens.len() < token_extent
        || expert_weights.len() < weight_extent
        || route_gates.len() < rows_padded as usize
        || expert_bias.len() < bias_extent
        || routed_output.len() < output_extent
    {
        return Err(KernelError::InvalidArgument);
    }
    // Checked views separate route metadata from the quantized matrix operands.
    let Ok(gates) =
        StridedReadView2D::from_shared_slice(route_gates, 0, rows_padded as usize, 1, 1)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(biases) = StridedReadView2D::from_shared_slice(
        expert_bias,
        expert as usize * bias_stride as usize,
        1,
        output_columns as usize,
        bias_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };

    // One Wave64 owns one 16x16 routed-output tile.
    let thread_index = thread::index_1d();
    let raw = thread_index.get();
    let lane = raw % 64;
    let lane_column = lane % 16;
    let tiles_per_row = ((output_columns as usize - 1) / 16) + 1;
    let tile = raw / 64;
    let tile_row = tile / tiles_per_row;
    let tile_column = tile % tiles_per_row;
    let output_column = tile_column * 16 + lane_column;
    let Some(output_tile) = thread_index.checked_tiled_2d::<64, 16, 16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    let Ok(token_matrix) = Bf16MfmaAMatrix::row_major(
        routed_tokens,
        0,
        rows_padded as usize,
        reduction as usize,
        token_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let weight_base = expert as usize * expert_weight_stride as usize;
    let Ok(weight_matrix) = Bf16MfmaBMatrix::row_major(
        expert_weights,
        weight_base,
        reduction as usize,
        output_columns as usize,
        weight_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let wave_lane = WaveLane::<Wave64>::current();
    let matrix = Matrix::current();
    // All lanes traverse the same K phases and retain accumulation in FP32.
    let mut accumulator = F32AccumulatorFragment::zero(&wave_lane);
    let mut phase = 0_usize;
    while phase < reduction as usize {
        let lhs = token_matrix.load_m16k16(&wave_lane, tile_row * 16, phase);
        let rhs = weight_matrix.load_k16n16(&wave_lane, phase, tile_column * 16);
        accumulator = matrix.multiply_accumulate(lhs, rhs, accumulator);
        phase += 16;
    }

    // Fuse route gating and expert bias before capability-checked edge stores.
    let values = accumulator.into_values();
    let row_base = tile_row * 16 + (lane / 16) * 4;
    let bias = biases.load_or(0, output_column, 0.0);
    if let Some(element) = routed_output.get_tiled_2d_mut(
        &output_tile,
        0,
        rows_padded as usize,
        output_columns as usize,
        output_stride as usize,
    ) {
        *element = gates.load_or(row_base, 0, 0.0) * (values[0] + bias);
    }
    if let Some(element) = routed_output.get_tiled_2d_mut(
        &output_tile,
        1,
        rows_padded as usize,
        output_columns as usize,
        output_stride as usize,
    ) {
        *element = gates.load_or(row_base + 1, 0, 0.0) * (values[1] + bias);
    }
    if let Some(element) = routed_output.get_tiled_2d_mut(
        &output_tile,
        2,
        rows_padded as usize,
        output_columns as usize,
        output_stride as usize,
    ) {
        *element = gates.load_or(row_base + 2, 0, 0.0) * (values[2] + bias);
    }
    if let Some(element) = routed_output.get_tiled_2d_mut(
        &output_tile,
        3,
        rows_padded as usize,
        output_columns as usize,
        output_stride as usize,
    ) {
        *element = gates.load_or(row_base + 3, 0, 0.0) * (values[3] + bias);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expect_checked_value<T: From<u8> + PartialEq>((valid, value): (T, T)) -> T {
        assert!(valid == T::from(1_u8), "unexpected rejected checked pair");
        value
    }

    #[test]
    fn private_checked_pairs_have_no_padding() {
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<(usize, usize)>(), 2 * size_of::<usize>());
        assert_eq!(offset_of!((usize, usize), 0), 0);
        assert_eq!(offset_of!((usize, usize), 1), size_of::<usize>());
    }

    #[test]
    fn private_checked_pairs_preserve_zero_max_and_rejection() {
        assert_eq!(checked_extent_add(0, 0), (1, 0));
        assert_eq!(checked_extent_add(usize::MAX, 0), (1, usize::MAX));
        assert_eq!(checked_extent_add(0, usize::MAX), (1, usize::MAX));
        assert_eq!(checked_extent_add(usize::MAX, 1), (0, 0));
    }

    #[test]
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    fn checked_extent_add_matches_independent_u128_oracle() {
        let check = |lhs: usize, rhs: usize| {
            let expected = lhs as u128 + rhs as u128;
            let actual = checked_extent_add(lhs, rhs);
            if expected > usize::MAX as u128 {
                assert_eq!(actual, (0, 0));
            } else {
                assert_eq!(expect_checked_value(actual) as u128, expected);
            }
        };
        let boundaries = [
            0,
            1,
            2,
            15,
            16,
            usize::MAX / 3,
            usize::MAX / 2,
            usize::MAX / 2 + 1,
            usize::MAX - 1,
            usize::MAX,
        ];
        for lhs in boundaries {
            for rhs in boundaries {
                check(lhs, rhs);
            }
        }
        let mut state = 0xb1d9_5037_248a_6c12_u64;
        for _ in 0..1024 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let lhs = state as usize;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            check(lhs, state as usize);
        }
    }

    #[test]
    fn checked_extent_add_preserves_two_stage_boundaries() {
        assert_eq!(checked_extent_add(usize::MAX, 1), (0, 0));
        let prefix = expect_checked_value(checked_extent_add(usize::MAX - 1, 1));
        assert_eq!(prefix, usize::MAX);
        assert_eq!(
            expect_checked_value(checked_extent_add(prefix, 0)),
            usize::MAX
        );
        assert_eq!(checked_extent_add(prefix, 1), (0, 0));
        assert_eq!(expect_checked_value(checked_extent_add(0, 0)), 0);
        assert_eq!(
            expect_checked_value(checked_extent_add(0, usize::MAX)),
            usize::MAX
        );
        assert_eq!(
            expect_checked_value(checked_extent_add(usize::MAX, 0)),
            usize::MAX
        );
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn checked_extent_add_does_not_narrow_to_u32() {
        let prefix = expect_checked_value(checked_extent_add(u32::MAX as usize, 1));
        assert_eq!(prefix, 1_usize << 32);
        assert_eq!(
            expect_checked_value(checked_extent_add(prefix, prefix)),
            1_usize << 33
        );
        assert_eq!(
            expect_checked_value(checked_extent_add(
                usize::MAX - u32::MAX as usize,
                u32::MAX as usize
            )),
            usize::MAX
        );
    }

    #[test]
    fn matrix_extent_excludes_trailing_padding() {
        assert_eq!(matrix_extent(3, 5, 8), 21);
        assert_eq!(matrix_extent(0, 5, 8), 0);
    }
}
