//! Scalar BF16/F32 counterpart of the existing dynamic fused attention kernel.
#![allow(missing_docs)]

use fe2o3_device::{
    DisjointSlice, Index1D, KernelError, KernelResult, Math, StridedReadView2D, Tiled2D, kernel,
    thread,
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

/// Uses increasing-depth F32 MACs and two stable-softmax score passes.
/// No MFMA, subgroup, pipeline, or LDS operations implement this scalar path.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1]),
    control_flow(loop_bounds(4, 4096, 1024, 4096, 1024))
)]
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_general_simt_v1(
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

    // One Wave64 owns one 16-row query tile, with the identical tiled stores.
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
    let score_row_base = query_row_base + (lane / 16) * 4;
    let (score_row_in_head_valid, score_row_in_head) =
        checked_head_row_offset(score_row_base, head_row_base);
    if score_row_in_head_valid != 1 {
        return Err(KernelError::InvalidArgument);
    }
    let Some(output_tile) = thread_index.checked_tiled_2d::<64, 16, 16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    let Ok(q_view) = StridedReadView2D::from_shared_slice(
        q,
        0,
        output_rows as usize,
        depth as usize,
        q_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(k_view) = StridedReadView2D::from_shared_slice(
        k_transposed,
        head * k_head_stride as usize,
        depth as usize,
        keys_padded as usize,
        k_depth_stride as usize,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
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
    let math = Math::current();
    let mut component = 0_usize;
    while component < 4 {
        let row_in_head = row_group_component(score_row_in_head, component);
        let global_row = row_group_component(score_row_base, component);
        let mut result = 0.0_f32;
        if row_in_head < query_rows as usize && lane_column < value_dimension as usize {
            let mut maximum = f32::NEG_INFINITY;
            let mut maximum_key = 0_usize;
            while maximum_key < keys as usize {
                let mask_value = mask.load_or(row_in_head, maximum_key, f32::NEG_INFINITY);
                // Match the reference: every nonfinite logical mask is excluded.
                if mask_value > f32::NEG_INFINITY && mask_value < f32::INFINITY {
                    let mut score = 0.0_f32;
                    let mut maximum_depth = 0_usize;
                    while maximum_depth < depth as usize {
                        let query_value = fe2o3_device::Bf16::from_bits(q_view.load_or(
                            global_row,
                            maximum_depth,
                            0,
                        ))
                        .to_f32();
                        let key_value = fe2o3_device::Bf16::from_bits(k_view.load_or(
                            maximum_depth,
                            maximum_key,
                            0,
                        ))
                        .to_f32();
                        score += query_value * key_value;
                        maximum_depth += 1;
                    }
                    let scaled = score * scale + mask_value;
                    if scaled > maximum {
                        maximum = scaled;
                    }
                }
                maximum_key += 1;
            }
            let mut numerator = 0.0_f32;
            let mut denominator = 0.0_f32;
            if maximum != f32::NEG_INFINITY {
                // Distinct induction variables preserve the two source passes.
                let mut weighted_key = 0_usize;
                while weighted_key < keys as usize {
                    let mask_value = mask.load_or(row_in_head, weighted_key, f32::NEG_INFINITY);
                    let mut weighted_score = f32::NEG_INFINITY;
                    if mask_value > f32::NEG_INFINITY && mask_value < f32::INFINITY {
                        let mut score = 0.0_f32;
                        let mut weighted_depth = 0_usize;
                        while weighted_depth < depth as usize {
                            let query_value = fe2o3_device::Bf16::from_bits(q_view.load_or(
                                global_row,
                                weighted_depth,
                                0,
                            ))
                            .to_f32();
                            let key_value = fe2o3_device::Bf16::from_bits(k_view.load_or(
                                weighted_depth,
                                weighted_key,
                                0,
                            ))
                            .to_f32();
                            score += query_value * key_value;
                            weighted_depth += 1;
                        }
                        weighted_score = score * scale + mask_value;
                    }
                    let probability = math.exp_f32(weighted_score - maximum);
                    denominator += probability;
                    numerator += probability * v_view.load_or(weighted_key, lane_column, 0.0);
                    weighted_key += 1;
                }
                if denominator > 0.0 {
                    result = numerator / denominator;
                }
            }
        }
        // Padded query rows store zero; tiled ownership leaves stride padding alone.
        if component == 0 {
            if let Some(element) = output.get_tiled_2d_mut(
                &output_tile,
                0,
                output_rows as usize,
                value_dimension as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if component == 1 {
            if let Some(element) = output.get_tiled_2d_mut(
                &output_tile,
                1,
                output_rows as usize,
                value_dimension as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if component == 2 {
            if let Some(element) = output.get_tiled_2d_mut(
                &output_tile,
                2,
                output_rows as usize,
                value_dimension as usize,
                output_stride as usize,
            ) {
                *element = result;
            }
        } else if let Some(element) = output.get_tiled_2d_mut(
            &output_tile,
            3,
            output_rows as usize,
            value_dimension as usize,
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
    fn every_bf16_encoding_preserves_exact_storage_bits() {
        for bits in 0_u16..=u16::MAX {
            let expected_bits = u32::from(bits) << 16;
            assert_eq!(
                fe2o3_device::Bf16::from_bits(bits).to_f32().to_bits(),
                expected_bits,
                "{bits:#06x}"
            );
        }
    }

    #[test]
    fn validation_and_supported_maxima_match_tile() {
        assert_eq!(
            FLASH_ATTENTION_WORKGROUP_V1,
            crate::kernel::FLASH_ATTENTION_WORKGROUP_V1
        );
        assert_eq!(
            FLASH_ATTENTION_MAX_KEYS_V1,
            crate::kernel::FLASH_ATTENTION_MAX_KEYS_V1
        );
        assert_eq!(
            FLASH_ATTENTION_MAX_DEPTH_V1,
            crate::kernel::FLASH_ATTENTION_MAX_DEPTH_V1
        );
        assert_eq!(
            FLASH_ATTENTION_MAX_VALUE_DIMENSION_V1,
            crate::kernel::FLASH_ATTENTION_MAX_VALUE_DIMENSION_V1
        );
        let tile = include_str!("kernel.rs");
        let simt = include_str!("kernel_simt.rs");
        let start = ["let expected_", "output_rows ="].concat();
        let end = ["// One ", "Wave64"].concat();
        let validation = |source: &str| -> String {
            source
                .split_once(start.as_str())
                .unwrap()
                .1
                .split_once(end.as_str())
                .unwrap()
                .0
                .split_whitespace()
                .collect()
        };
        assert_eq!(validation(tile), validation(simt));
        let body = simt.split("#[cfg(test)]").next().unwrap();
        assert_eq!(body.matches("while ").count(), 5);
        assert!(body.contains("loop_bounds(4, 4096, 1024, 4096, 1024)"));
        assert!(!body.contains("Matrix::"));
        assert!(!body.contains("Subgroup"));
        assert!(!body.contains("WorkgroupPipeline"));
    }

    #[test]
    fn tiled_lane_components_cover_dynamic_padded_rows_once() {
        let mut writes = vec![0_u8; 64 * 19];
        for thread in 0..4 * 64 {
            let tile = thread / 64;
            let lane = thread % 64;
            for component in 0..4 {
                let row = tile * 16 + (lane / 16) * 4 + component;
                let column = lane % 16;
                if column < 13 {
                    writes[row * 19 + column] += 1;
                }
            }
        }
        for row in 0..64 {
            for column in 0..19 {
                assert_eq!(writes[row * 19 + column], u8::from(column < 13));
            }
        }
    }
}
