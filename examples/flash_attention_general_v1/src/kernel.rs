//! Safe Rust, memory-bounded attention with double-buffered workgroup staging.
//!
//! The kernel follows five visible phases: validate shape/stride extents, map a
//! wave to a query tile, construct checked views, advance the uniform MFMA and
//! online-softmax pipeline, then store through a tiled disjoint capability.

#![allow(missing_docs)]

use fe2o3_device::{
    Bf16MfmaAFragment, Bf16MfmaAMatrix, Bf16MfmaBFragment, Bf16MfmaBMatrix, DisjointSlice,
    F32AccumulatorFragment, Index1D, KernelError, KernelResult, Math, Matrix, StridedReadView2D,
    Subgroup, Tiled2D, Wave64, WaveLane, WorkgroupLdsScope, WorkgroupPipeline, kernel, thread,
};

pub const FLASH_ATTENTION_WORKGROUP_V1: [u32; 3] = [64, 1, 1];
pub const FLASH_ATTENTION_MAX_KEYS_V1: u32 = 4096;
pub const FLASH_ATTENTION_MAX_DEPTH_V1: u32 = 1024;
pub const FLASH_ATTENTION_MAX_VALUE_DIMENSION_V1: u32 = 16;

// Same-width 0/1 flag and value keep constant returns free of padding.
// Callers reject every flag other than 1 before using the value.
// Every u32 pair has an exact u64 product. Keep the original u32 overflow
// rejection before any extent work, without calling a cross-crate wrapper.
fn checked_output_rows(batch_heads: u32, query_rows_padded: u32) -> (u64, u64) {
    let product = batch_heads as u64 * query_rows_padded as u64;
    if product > u32::MAX as u64 {
        return (0, 0);
    }
    (1, product)
}

// The complement form is exactly lhs + rhs, with each subtraction bounded.
fn checked_extent_add(lhs: usize, rhs: usize) -> (usize, usize) {
    let available = usize::MAX - lhs;
    if rhs > available {
        return (0, 0);
    }
    let remaining = available - rhs;
    (1, usize::MAX - remaining)
}

// Keep the exact unsigned lower bound adjacent to the checked subtraction.
fn checked_head_row_offset(score_row_base: usize, head_row_base: usize) -> (usize, usize) {
    if score_row_base < head_row_base {
        return (0, 0);
    }
    (1, score_row_base - head_row_base)
}

// These callers compose a four-row group, not a general unchecked sum.
// query_tile * 16 and head * query_rows_padded are multiples of 16
// (the existing validation requires query_rows_padded % 16 == 0), while
// (lane / 16) * 4 is a multiple of 4. Their checked difference is aligned
// to 4 too. Components are literals 1..=3 or the SIMT loop's 0..4 range.
// With those exact caller invariants, OR equals addition without overflow.
fn row_group_component(base: usize, component: usize) -> usize {
    base | component
}

fn matrix_extent(rows: u32, columns: u32, stride: u32) -> usize {
    if rows == 0 || columns == 0 {
        0
    } else {
        (rows - 1) as usize * stride as usize + columns as usize
    }
}

/// Computes fused scaled dot-product attention without materializing scores.
///
/// Q and transposed K are BF16. V, the additive mask, and output are FP32.
/// Query and key storage is padded to 16 for MFMA, while `query_rows` and
/// `keys` describe the independent logical extents. Padded queries and fully
/// masked logical rows produce zero; padded keys never contribute.
#[kernel(
    typed,
    launch(
        required = [64, 1, 1],
        max = [64, 1, 1],
        static_shared_memory_bytes = 2048
    ),
    control_flow(loop_bounds(256, 64, 16))
)]
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_general_v1(
    q: &[u16],
    k_transposed: &[u16],
    v: &[f32],
    additive_mask: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    batch_heads: u32,
    query_rows: u32,
    query_rows_padded: u32,
    keys: u32,
    keys_padded: u32,
    depth: u32,
    value_dimension: u32,
    q_stride: u32,
    k_depth_stride: u32,
    k_head_stride: u32,
    v_stride: u32,
    v_head_stride: u32,
    mask_stride: u32,
    output_stride: u32,
    output_rows: u32,
    scale: f32,
) -> KernelResult {
    // Prove every dynamic extent before constructing views or entering barriers.
    let expected_output_rows = batch_heads as u64 * query_rows_padded as u64;
    if expected_output_rows > u32::MAX as u64 {
        return Err(KernelError::InvalidArgument);
    }
    let q_extent = matrix_extent(output_rows, depth, q_stride);
    let k_extent = if batch_heads == 0 || depth == 0 || keys_padded == 0 {
        0
    } else {
        let head_extent = (batch_heads - 1) as usize * k_head_stride as usize;
        let depth_extent = (depth - 1) as usize * k_depth_stride as usize;
        let after_heads = usize::MAX - head_extent;
        if depth_extent > after_heads {
            return Err(KernelError::InvalidArgument);
        }
        let after_depth = after_heads - depth_extent;
        if keys_padded as usize > after_depth {
            return Err(KernelError::InvalidArgument);
        }
        usize::MAX - (after_depth - keys_padded as usize)
    };
    let v_extent = if batch_heads == 0 || keys_padded == 0 || value_dimension == 0 {
        0
    } else {
        let head_extent = (batch_heads - 1) as usize * v_head_stride as usize;
        let key_extent = (keys_padded - 1) as usize * v_stride as usize;
        let after_heads = usize::MAX - head_extent;
        if key_extent > after_heads {
            return Err(KernelError::InvalidArgument);
        }
        let after_keys = after_heads - key_extent;
        if value_dimension as usize > after_keys {
            return Err(KernelError::InvalidArgument);
        }
        usize::MAX - (after_keys - value_dimension as usize)
    };
    let mask_extent = matrix_extent(output_rows, keys, mask_stride);
    let output_extent = matrix_extent(output_rows, value_dimension, output_stride);
    if batch_heads == 0
        || query_rows == 0
        || query_rows > query_rows_padded
        || query_rows_padded == 0
        || query_rows_padded % 16 != 0
        || keys == 0
        || keys > keys_padded
        || keys_padded == 0
        || keys_padded % 16 != 0
        || keys_padded > FLASH_ATTENTION_MAX_KEYS_V1
        || depth == 0
        || depth > FLASH_ATTENTION_MAX_DEPTH_V1
        || value_dimension == 0
        || value_dimension > FLASH_ATTENTION_MAX_VALUE_DIMENSION_V1
        || output_rows as u64 != expected_output_rows
        || q_stride < depth
        || k_depth_stride < keys_padded
        || (k_head_stride as u64) < depth as u64 * k_depth_stride as u64
        || v_stride < value_dimension
        || (v_head_stride as u64) < keys_padded as u64 * v_stride as u64
        || mask_stride < keys
        || output_stride < value_dimension
        || q.len() < q_extent
        || k_transposed.len() < k_extent
        || v.len() < v_extent
        || additive_mask.len() < mask_extent
        || output.len() < output_extent
    {
        return Err(KernelError::InvalidArgument);
    }

    // One Wave64 owns one 16-row query tile; Wave16 quarters own four rows each.
    let thread_index = thread::index_1d();
    let raw = thread_index.get();
    let lane = raw % 64;
    let lane_column = lane % 16;
    let query_tile = raw / 64;
    let tiles_per_head = query_rows_padded as usize / 16;
    let expected_query_tiles = batch_heads as usize * tiles_per_head;
    if query_tile >= expected_query_tiles {
        return Err(KernelError::InvalidArgument);
    }
    if tiles_per_head == 0 {
        return Err(KernelError::InvalidArgument);
    }
    let head = query_tile / tiles_per_head;
    let query_row_base = query_tile * 16;
    let head_row_base = head * query_rows_padded as usize;
    let score_row_base = query_tile * 16 + (lane / 16) * 4;
    // With P = 16 * tiles_per_head and tile = head * tiles_per_head + r,
    // score_row_base - head_row_base = 16 * r + 4 * (lane / 16).
    // The terms have disjoint low four bits and the result is below P.
    let score_row_in_head = (query_tile % tiles_per_head) * 16 | (lane / 16) * 4;
    let Some(output_tile) = thread_index.checked_tiled_2d::<64, 16, 16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    // Checked views name each logical layout and keep stride arithmetic out of the loop.
    // Validated batch_heads * query_rows_padded equals output_rows <= u32::MAX.
    // The query-tile bound makes head < batch_heads, so this cast is lossless.
    // Keep the check adjacent and the stride product at full usize width.
    if head_row_base > u32::MAX as usize {
        return Err(KernelError::InvalidArgument);
    }
    let mask_head_row_base = head_row_base as u32;
    let Ok(mask) = StridedReadView2D::from_shared_slice(
        additive_mask,
        mask_head_row_base as usize * mask_stride as usize,
        query_rows as usize,
        keys as usize,
        mask_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(v_view) = StridedReadView2D::from_shared_slice(
        v,
        head * v_head_stride as usize,
        keys_padded as usize,
        value_dimension as usize,
        v_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(q_matrix) = Bf16MfmaAMatrix::row_major(
        q,
        0,
        output_rows as usize,
        depth as usize,
        q_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(k_matrix) = Bf16MfmaBMatrix::row_major(
        k_transposed,
        head * k_head_stride as usize,
        depth as usize,
        keys_padded as usize,
        k_depth_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let wave_lane = WaveLane::<Wave64>::current();
    let matrix = Matrix::current();
    let subgroup = Subgroup::current();
    let math = Math::current();

    // Keep the online-softmax maximum, denominator, and numerator in FP32.
    let mut maximum0 = f32::NEG_INFINITY;
    let mut maximum1 = f32::NEG_INFINITY;
    let mut maximum2 = f32::NEG_INFINITY;
    let mut maximum3 = f32::NEG_INFINITY;
    let mut denominator0 = 0.0_f32;
    let mut denominator1 = 0.0_f32;
    let mut denominator2 = 0.0_f32;
    let mut denominator3 = 0.0_f32;
    let mut numerator0 = 0.0_f32;
    let mut numerator1 = 0.0_f32;
    let mut numerator2 = 0.0_f32;
    let mut numerator3 = 0.0_f32;
    let mut pipeline_scope = WorkgroupLdsScope::current();
    let mut q_pipeline =
        WorkgroupPipeline::<Bf16MfmaAFragment<'_>, 2, 64, 1>::current(&mut pipeline_scope);
    let mut k_pipeline =
        WorkgroupPipeline::<Bf16MfmaBFragment<'_>, 2, 64, 1>::current(&mut pipeline_scope);
    let mut key_base = 0_usize;
    // Each key tile advances the stable online (maximum, sum, numerator) state.
    while key_base < keys_padded as usize {
        let key_column = key_base + lane_column;
        let mut scores = F32AccumulatorFragment::zero(&wave_lane);
        // Double-buffer Q/K fragments so staging of phase n+1 overlaps MFMA for phase n.
        let phase_count = (depth as usize + 15) / 16;
        let lhs = q_matrix.load_m16k16(&wave_lane, query_row_base, 0);
        let rhs = k_matrix.load_k16n16(&wave_lane, 0, key_base);
        q_pipeline.stage(0);
        q_pipeline.write(0, lane, lhs);
        q_pipeline.commit(0);
        k_pipeline.stage(0);
        k_pipeline.write(0, lane, rhs);
        k_pipeline.commit(0);

        let mut phase_index = 0_usize;
        while phase_index < phase_count {
            let future_epoch = phase_index + 1;
            let next_phase = future_epoch * 16;
            let next_lhs = q_matrix.load_m16k16(&wave_lane, query_row_base, next_phase);
            let next_rhs = k_matrix.load_k16n16(&wave_lane, next_phase, key_base);

            q_pipeline.stage(future_epoch);
            q_pipeline.write(future_epoch, lane, next_lhs);
            q_pipeline.commit(future_epoch);
            k_pipeline.stage(future_epoch);
            k_pipeline.write(future_epoch, lane, next_rhs);
            k_pipeline.commit(future_epoch);

            q_pipeline.wait(phase_index);
            q_pipeline.consume(phase_index);
            let lhs = q_pipeline.read(phase_index, lane);
            k_pipeline.wait(phase_index);
            k_pipeline.consume(phase_index);
            let rhs = k_pipeline.read(phase_index, lane);
            scores = matrix.multiply_accumulate(lhs, rhs, scores);
            q_pipeline.release(phase_index);
            k_pipeline.release(phase_index);
            phase_index += 1;
        }
        q_pipeline.wait(phase_count);
        q_pipeline.discard(phase_count);
        q_pipeline.release(phase_count);
        k_pipeline.wait(phase_count);
        k_pipeline.discard(phase_count);
        k_pipeline.release(phase_count);
        let values = scores.into_values();
        let mask0 = mask.load_or(score_row_in_head, key_column, f32::NEG_INFINITY);
        let mask1 = mask.load_or(
            row_group_component(score_row_in_head, 1),
            key_column,
            f32::NEG_INFINITY,
        );
        let mask2 = mask.load_or(
            row_group_component(score_row_in_head, 2),
            key_column,
            f32::NEG_INFINITY,
        );
        let mask3 = mask.load_or(
            row_group_component(score_row_in_head, 3),
            key_column,
            f32::NEG_INFINITY,
        );
        let score0 = if mask0 > f32::NEG_INFINITY && mask0 < f32::INFINITY {
            values[0] * scale + mask0
        } else {
            f32::NEG_INFINITY
        };
        let score1 = if mask1 > f32::NEG_INFINITY && mask1 < f32::INFINITY {
            values[1] * scale + mask1
        } else {
            f32::NEG_INFINITY
        };
        let score2 = if mask2 > f32::NEG_INFINITY && mask2 < f32::INFINITY {
            values[2] * scale + mask2
        } else {
            f32::NEG_INFINITY
        };
        let score3 = if mask3 > f32::NEG_INFINITY && mask3 < f32::INFINITY {
            values[3] * scale + mask3
        } else {
            f32::NEG_INFINITY
        };
        let tile_maximum0 = subgroup.subgroup_reduce_max_f32::<16>(score0);
        let tile_maximum1 = subgroup.subgroup_reduce_max_f32::<16>(score1);
        let tile_maximum2 = subgroup.subgroup_reduce_max_f32::<16>(score2);
        let tile_maximum3 = subgroup.subgroup_reduce_max_f32::<16>(score3);
        let next_maximum0 = if tile_maximum0 > maximum0 {
            tile_maximum0
        } else {
            maximum0
        };
        let next_maximum1 = if tile_maximum1 > maximum1 {
            tile_maximum1
        } else {
            maximum1
        };
        let next_maximum2 = if tile_maximum2 > maximum2 {
            tile_maximum2
        } else {
            maximum2
        };
        let next_maximum3 = if tile_maximum3 > maximum3 {
            tile_maximum3
        } else {
            maximum3
        };
        let rescale0 = if next_maximum0 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(maximum0 - next_maximum0)
        };
        let rescale1 = if next_maximum1 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(maximum1 - next_maximum1)
        };
        let rescale2 = if next_maximum2 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(maximum2 - next_maximum2)
        };
        let rescale3 = if next_maximum3 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(maximum3 - next_maximum3)
        };
        let probability0 = if next_maximum0 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(score0 - next_maximum0)
        };
        let probability1 = if next_maximum1 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(score1 - next_maximum1)
        };
        let probability2 = if next_maximum2 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(score2 - next_maximum2)
        };
        let probability3 = if next_maximum3 == f32::NEG_INFINITY {
            0.0
        } else {
            math.exp_f32(score3 - next_maximum3)
        };
        denominator0 =
            denominator0 * rescale0 + subgroup.subgroup_reduce_sum_f32::<16>(probability0);
        denominator1 =
            denominator1 * rescale1 + subgroup.subgroup_reduce_sum_f32::<16>(probability1);
        denominator2 =
            denominator2 * rescale2 + subgroup.subgroup_reduce_sum_f32::<16>(probability2);
        denominator3 =
            denominator3 * rescale3 + subgroup.subgroup_reduce_sum_f32::<16>(probability3);
        numerator0 *= rescale0;
        numerator1 *= rescale1;
        numerator2 *= rescale2;
        numerator3 *= rescale3;

        let mut dimension = 0_usize;
        while dimension < value_dimension as usize {
            // A zero weight must not multiply NaN/Inf in a padded V row.
            let value = if key_column < keys as usize {
                v_view.load_or(key_column, dimension, 0.0)
            } else {
                0.0
            };
            let contribution0 = subgroup.subgroup_reduce_sum_f32::<16>(probability0 * value);
            let contribution1 = subgroup.subgroup_reduce_sum_f32::<16>(probability1 * value);
            let contribution2 = subgroup.subgroup_reduce_sum_f32::<16>(probability2 * value);
            let contribution3 = subgroup.subgroup_reduce_sum_f32::<16>(probability3 * value);
            if lane_column == dimension {
                numerator0 += contribution0;
                numerator1 += contribution1;
                numerator2 += contribution2;
                numerator3 += contribution3;
            }
            dimension += 1;
        }
        maximum0 = next_maximum0;
        maximum1 = next_maximum1;
        maximum2 = next_maximum2;
        maximum3 = next_maximum3;
        key_base += 16;
    }

    // Tiled ownership suppresses padded rows/columns and makes valid stores disjoint.
    if let Some(element) = output.get_tiled_2d_mut(
        &output_tile,
        0,
        output_rows as usize,
        value_dimension as usize,
        output_stride as usize,
    ) {
        *element = if denominator0 > 0.0 {
            numerator0 / denominator0
        } else {
            0.0
        };
    }
    if let Some(element) = output.get_tiled_2d_mut(
        &output_tile,
        1,
        output_rows as usize,
        value_dimension as usize,
        output_stride as usize,
    ) {
        *element = if denominator1 > 0.0 {
            numerator1 / denominator1
        } else {
            0.0
        };
    }
    if let Some(element) = output.get_tiled_2d_mut(
        &output_tile,
        2,
        output_rows as usize,
        value_dimension as usize,
        output_stride as usize,
    ) {
        *element = if denominator2 > 0.0 {
            numerator2 / denominator2
        } else {
            0.0
        };
    }
    if let Some(element) = output.get_tiled_2d_mut(
        &output_tile,
        3,
        output_rows as usize,
        value_dimension as usize,
        output_stride as usize,
    ) {
        *element = if denominator3 > 0.0 {
            numerator3 / denominator3
        } else {
            0.0
        };
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
    fn aligned_row_group_components_match_u128_at_full_width() {
        let check = |base: usize| {
            assert_eq!(base & 3, 0);
            for component in 0..4 {
                let expected = base as u128 + component as u128;
                assert!(expected <= usize::MAX as u128);
                assert_eq!(row_group_component(base, component) as u128, expected);
            }
        };
        for base in [
            0,
            4,
            16,
            (u32::MAX as usize) & !3,
            (usize::MAX / 2) & !3,
            usize::MAX - 3,
        ] {
            check(base);
        }
        let mut state = 0x391b_8f25_7c06_a4d2_u64;
        for _ in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            check((state as usize) & !3);
        }
    }

    #[test]
    fn row_group_composition_does_not_claim_general_addition() {
        for (base, component) in [(1_usize, 1_usize), (4, 4), (usize::MAX, 1)] {
            assert_ne!(
                row_group_component(base, component) as u128,
                base as u128 + component as u128
            );
        }
    }

    #[test]
    fn private_checked_pairs_have_no_padding() {
        use core::mem::{offset_of, size_of};
        assert_eq!(size_of::<(usize, usize)>(), 2 * size_of::<usize>());
        assert_eq!(offset_of!((usize, usize), 0), 0);
        assert_eq!(offset_of!((usize, usize), 1), size_of::<usize>());
        assert_eq!(size_of::<(u64, u64)>(), 2 * size_of::<u64>());
        assert_eq!(offset_of!((u64, u64), 0), 0);
        assert_eq!(offset_of!((u64, u64), 1), size_of::<u64>());
    }

    #[test]
    fn private_checked_pairs_preserve_zero_max_and_rejection() {
        assert_eq!(checked_extent_add(0, 0), (1, 0));
        assert_eq!(checked_extent_add(usize::MAX, 0), (1, usize::MAX));
        assert_eq!(checked_extent_add(0, usize::MAX), (1, usize::MAX));
        assert_eq!(checked_extent_add(usize::MAX, 1), (0, 0));
        assert_eq!(checked_output_rows(0, u32::MAX), (1, 0));
        assert_eq!(checked_output_rows(u32::MAX, 1), (1, u32::MAX as u64));
        assert_eq!(checked_output_rows(u32::MAX, 2), (0, 0));
        assert_eq!(checked_head_row_offset(0, 0), (1, 0));
        assert_eq!(checked_head_row_offset(usize::MAX, 0), (1, usize::MAX));
        assert_eq!(checked_head_row_offset(0, 1), (0, 0));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn validated_head_stride_products_match_full_width_oracle() {
        for (batch_heads, padded_rows) in [
            (1_u32, 16_u32),
            (3, 32),
            (17, 4112),
            (u32::MAX / 16, 16),
            (1, u32::MAX - 15),
            (65535, 65536),
        ] {
            let (valid, output_rows) = checked_output_rows(batch_heads, padded_rows);
            assert_eq!(valid, 1);
            let per_head = padded_rows as usize / 16;
            let tiles = batch_heads as usize * per_head;
            for tile in [0, per_head - 1, per_head, tiles - 1] {
                if tile >= tiles {
                    continue;
                }
                let head = tile / per_head;
                let head_row_base = head * padded_rows as usize;
                assert!(head < batch_heads as usize);
                assert!((head_row_base as u128) < output_rows as u128);
                assert!(head_row_base <= u32::MAX as usize);
                let mask_head_row_base = head_row_base as u32;
                assert_eq!(mask_head_row_base as u128, head_row_base as u128);
                for stride in [0_u32, 1, 16, 65536, u32::MAX] {
                    let expected = head_row_base as u128 * stride as u128;
                    assert!(expected <= usize::MAX as u128);
                    assert_eq!(
                        (mask_head_row_base as usize * stride as usize) as u128,
                        expected
                    );
                    // K/V use the head index rather than the flattened row.
                    assert_eq!(
                        (head * stride as usize) as u128,
                        head as u128 * stride as u128
                    );
                }
            }
        }
        // Preserve high product bits: bounding the row is not bounding its offset.
        let base = u32::MAX - 16;
        let product = base as usize * u32::MAX as usize;
        assert!(product > u32::MAX as usize);
        assert_eq!(product as u128, base as u128 * u32::MAX as u128);
    }

    #[test]
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    fn checked_head_row_offset_preserves_full_width_and_tile_layout() {
        for (score, head) in [
            (0, 0),
            (usize::MAX, 0),
            (usize::MAX, usize::MAX),
            (usize::MAX, usize::MAX - 1),
            (usize::MAX - 1, usize::MAX - 1),
        ] {
            assert_eq!(
                expect_checked_value(checked_head_row_offset(score, head)) as u128,
                score as u128 - head as u128
            );
        }
        for (score, head) in [(0, 1), (0, usize::MAX), (usize::MAX - 1, usize::MAX)] {
            assert_eq!(checked_head_row_offset(score, head), (0, 0));
        }
        for (batch_heads, padded_rows) in [
            (1_u32, 16_u32),
            (3, 32),
            (17, 4112),
            (u32::MAX / 16, 16),
            (1, u32::MAX - 15),
            (65535, 65536),
        ] {
            let per_head = padded_rows as usize / 16;
            let tiles = batch_heads as usize * per_head;
            for tile in [0, per_head - 1, per_head, tiles - 1] {
                if tile >= tiles {
                    continue;
                }
                for lane in [0_usize, 15, 16, 31, 32, 47, 48, 63] {
                    let score =
                        usize::try_from(tile as u128 * 16 + (lane as u128 / 16) * 4).unwrap();
                    let head =
                        usize::try_from((tile as u128 / per_head as u128) * padded_rows as u128)
                            .unwrap();
                    let expected = (tile as u128 % per_head as u128) * 16 + (lane as u128 / 16) * 4;
                    let local_row = (tile % per_head) * 16 | (lane / 16) * 4;
                    assert_eq!(local_row as u128, expected);
                    assert_eq!(
                        expect_checked_value(checked_head_row_offset(score, head)) as u128,
                        expected
                    );
                    assert!(expected < padded_rows as u128);
                    assert_eq!(expected & 3, 0);
                    for component in 0..4 {
                        let row = row_group_component(expected as usize, component);
                        assert_eq!(row as u128, expected + component as u128);
                        assert!((row as u128) < padded_rows as u128);
                    }
                }
            }
        }
    }

    #[test]
    fn checked_read_view_errors_keep_public_kernel_mapping() {
        use fe2o3_device::StridedReadView2DError;
        let data = [0.0_f32];
        let errors = [
            StridedReadView2D::from_shared_slice(&[()], 0, 1, 1, 1)
                .err()
                .unwrap(),
            StridedReadView2D::from_shared_slice(&data, 0, 1, 2, 1)
                .err()
                .unwrap(),
            StridedReadView2D::from_shared_slice(&data, usize::MAX, 1, 1, 1)
                .err()
                .unwrap(),
            StridedReadView2D::from_shared_slice(&data, 0, 1, 2, 2)
                .err()
                .unwrap(),
        ];
        assert_eq!(errors[0], StridedReadView2DError::ZeroSizedElement);
        assert_eq!(errors[1], StridedReadView2DError::InvalidStride);
        assert_eq!(errors[2], StridedReadView2DError::ExtentOverflow);
        assert_eq!(
            errors[3],
            StridedReadView2DError::OutOfBounds {
                required: 2,
                actual: 1
            }
        );
        for error in errors {
            assert_eq!(KernelError::from(error), KernelError::InvalidArgument);
        }
        assert!(StridedReadView2D::from_shared_slice(&data, 0, 1, 1, 1).is_ok());
    }

    #[test]
    fn checked_bf16_matrix_errors_keep_public_kernel_mapping() {
        use fe2o3_device::Bf16MatrixViewError;
        let data = [0_u16];
        let a_errors = [
            Bf16MfmaAMatrix::row_major(&data, 0, 1, 2, 1).err().unwrap(),
            Bf16MfmaAMatrix::row_major(&data, usize::MAX, 1, 1, 1)
                .err()
                .unwrap(),
            Bf16MfmaAMatrix::row_major(&data, 0, 1, 2, 2).err().unwrap(),
        ];
        let b_errors = [
            Bf16MfmaBMatrix::row_major(&data, 0, 1, 2, 1).err().unwrap(),
            Bf16MfmaBMatrix::row_major(&data, usize::MAX, 1, 1, 1)
                .err()
                .unwrap(),
            Bf16MfmaBMatrix::row_major(&data, 0, 1, 2, 2).err().unwrap(),
        ];
        assert_eq!(a_errors, b_errors);
        assert_eq!(a_errors[0], Bf16MatrixViewError::InvalidStride);
        assert_eq!(a_errors[1], Bf16MatrixViewError::ExtentOverflow);
        assert_eq!(
            a_errors[2],
            Bf16MatrixViewError::OutOfBounds {
                required: 2,
                actual: 1
            }
        );
        for error in a_errors {
            assert_eq!(KernelError::from(error), KernelError::InvalidArgument);
        }
        assert!(Bf16MfmaAMatrix::row_major(&data, 0, 1, 1, 1).is_ok());
        assert!(Bf16MfmaBMatrix::row_major(&data, 0, 1, 1, 1).is_ok());
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
    fn checked_output_rows_matches_independent_full_width_oracle() {
        let check = |lhs: u32, rhs: u32| {
            let expected = lhs as u128 * rhs as u128;
            let actual = checked_output_rows(lhs, rhs);
            if expected > u32::MAX as u128 {
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
            17,
            255,
            256,
            65_535,
            65_536,
            65_537,
            3,
            (1 << 31) - 1,
            1 << 31,
            u32::MAX - 1,
            u32::MAX,
        ];
        for lhs in boundaries {
            for rhs in boundaries {
                check(lhs, rhs);
            }
        }
        let mut state = 0x527a_4931_c6d8_0f25_u64;
        for _ in 0..4096 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let lhs = (state >> 32) as u32;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            check(lhs, (state >> 32) as u32);
        }
    }

    #[test]
    fn checked_output_rows_preserves_zero_and_overflow_boundaries() {
        for (lhs, rhs) in [
            (u32::MAX, 2),
            (2, 1 << 31),
            (65_536, 65_536),
            (u32::MAX, u32::MAX),
        ] {
            assert_eq!(checked_output_rows(lhs, rhs), (0, 0));
        }
        assert_eq!(
            expect_checked_value(checked_output_rows(u32::MAX, 1)),
            u32::MAX as u64
        );
        assert_eq!(expect_checked_value(checked_output_rows(0, u32::MAX)), 0);
        assert_eq!(expect_checked_value(checked_output_rows(u32::MAX, 0)), 0);
    }

    #[test]
    fn nonfinite_masks_are_excluded_before_each_tile_softmax_row() {
        let source = include_str!("kernel.rs");
        let body = source.split("#[cfg(test)]").next().unwrap();
        for component in 0..4 {
            let predicate =
                format!("mask{component} > f32::NEG_INFINITY && mask{component} < f32::INFINITY");
            assert_eq!(body.matches(predicate.as_str()).count(), 1);
            assert!(body.contains(&format!("values[{component}] * scale + mask{component}")));
        }
    }

    #[test]
    fn matrix_extent_excludes_trailing_padding() {
        assert_eq!(matrix_extent(3, 5, 8), 21);
        assert_eq!(matrix_extent(0, 5, 8), 0);
    }

    #[test]
    fn score_mfma_uses_workgroup_double_buffering() {
        let source = include_str!("kernel.rs");
        let key_loop = ["while key_base <", " keys_padded"].concat();
        let mfma = ["matrix.multiply", "_accumulate"].concat();
        assert_eq!(source.matches(&key_loop).count(), 1);
        assert_eq!(source.matches(&mfma).count(), 1);
        assert!(source.contains("WorkgroupPipeline::<Bf16MfmaAFragment<'_>, 2, 64, 1>"));
        assert!(source.contains("WorkgroupPipeline::<Bf16MfmaBFragment<'_>, 2, 64, 1>"));
        assert!(source.contains("q_pipeline.read(phase_index, lane)"));
        assert!(source.contains("k_pipeline.read(phase_index, lane)"));
        assert!(source.contains("q_pipeline.discard(phase_count)"));
    }
}
