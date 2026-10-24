//! Ordinary BF16/F32 scalar projection for the existing routed expert ABI.
#![allow(missing_docs)]

use fe2o3_device::{
    Bf16, DisjointSlice, Index1D, KernelError, KernelResult, StridedReadView2D, Tiled2D, kernel,
    thread,
};

pub const MOE_EXPERT_WORKGROUP_V1: [u32; 3] = [64, 1, 1];

// Same-width 0/1 flag and value keep constant returns free of padding.
// Callers reject every flag other than 1 before using the value.
// The complement form is exactly lhs + rhs, with each subtraction bounded.
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

/// Same routed group, selected-expert weights and gate * (projection + bias).
/// Increasing-depth F32 MACs replace MFMA, without subgroup or LDS operations.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
    control_flow(loop_bounds(4, 4294967295))
)]
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_expert_general_simt_v1(
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
    // Prove all dynamic extents and selected-expert offsets before scalar projection.
    let token_extent = matrix_extent(rows_padded, reduction, token_stride);
    let weight_extent = if expert_count == 0 || reduction == 0 || output_columns == 0 {
        0
    } else {
        let (extent_valid, extent) = checked_extent_add(
            (expert_count - 1) as usize * expert_weight_stride as usize,
            (reduction - 1) as usize * weight_stride as usize,
        );
        if extent_valid != 1 {
            return Err(KernelError::InvalidArgument);
        }
        let (extent_valid, extent) = checked_extent_add(extent, output_columns as usize);
        if extent_valid != 1 {
            return Err(KernelError::InvalidArgument);
        }
        extent
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

    // One Wave64 owns the same 16x16 output tile as the structured kernel.
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
    let Ok(tokens) = StridedReadView2D::from_shared_slice(
        routed_tokens,
        0,
        rows_padded as usize,
        reduction as usize,
        token_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let weight_base = expert as usize * expert_weight_stride as usize;
    let Ok(weights) = StridedReadView2D::from_shared_slice(
        expert_weights,
        weight_base,
        reduction as usize,
        output_columns as usize,
        weight_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let row_base = tile_row * 16 + (lane / 16) * 4;
    let mut component = 0_usize;
    while component < 4 {
        let row = row_base + component;
        let mut result = 0.0_f32;
        if row < rows_padded as usize && output_column < output_columns as usize {
            let mut projection = 0.0_f32;
            let mut depth = 0_usize;
            while depth < reduction as usize {
                let activation = Bf16::from_bits(tokens.load_or(row, depth, 0)).to_f32();
                let weight = Bf16::from_bits(weights.load_or(depth, output_column, 0)).to_f32();
                projection += activation * weight;
                depth += 1;
            }
            // Do not skip zero gates: preserve the original IEEE epilogue.
            let bias = biases.load_or(0, output_column, 0.0);
            result = gates.load_or(row, 0, 0.0) * (projection + bias);
        }
        if component == 0 {
            if let Some(element) = routed_output.get_tiled_2d_mut(
                &output_tile,
                0,
                rows_padded as usize,
                output_columns as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if component == 1 {
            if let Some(element) = routed_output.get_tiled_2d_mut(
                &output_tile,
                1,
                rows_padded as usize,
                output_columns as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if component == 2 {
            if let Some(element) = routed_output.get_tiled_2d_mut(
                &output_tile,
                2,
                rows_padded as usize,
                output_columns as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if let Some(element) = routed_output.get_tiled_2d_mut(
            &output_tile,
            3,
            rows_padded as usize,
            output_columns as usize,
            output_stride as usize,
        ) {
            *element = result;
        }
        component += 1;
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
}
