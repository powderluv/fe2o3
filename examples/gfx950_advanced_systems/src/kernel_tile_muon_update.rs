//! Structured output tiles for the existing muon-update kernel.
#![allow(missing_docs)]
use crate::{GRADIENT_SHARDS, MUON_ELEMENTS, MUON_LEARNING_RATE, SYSTEM_BATCHES};
use fe2o3_device::{
    DeviceMath, DisjointSlice, Gfx950Subgroup, Index1D, StridedReadView2D, Tiled2D, kernel, thread,
};

/// Reduces two shards and computes five Newton-Schulz Muon iterations.
#[cfg(feature = "kernel-muon-update-tile")]
#[cfg_attr(
    not(feature = "ablation-muon-broadcast16"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-muon-broadcast16",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
pub fn gfx950_muon_update_4x4_v1(
    shards: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut output_norm: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
) {
    // One Wave64 owns one 4x4 update; the first 16 lanes hold the matrix elements.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let lane = global_index & 63;
    // Validate every launch-wide shape before the norm reduction.
    if batch >= SYSTEM_BATCHES
        || shards.len() != SYSTEM_BATCHES * GRADIENT_SHARDS * MUON_ELEMENTS
        || output.len() != SYSTEM_BATCHES * MUON_ELEMENTS
        || output_norm.len() != SYSTEM_BATCHES
    {
        return;
    }
    // A checked two-row view makes the two-shard reduction explicit.
    let Ok(shards) = StridedReadView2D::from_shared_slice(
        shards,
        batch
            .wrapping_mul(GRADIENT_SHARDS)
            .wrapping_mul(MUON_ELEMENTS),
        GRADIENT_SHARDS,
        MUON_ELEMENTS,
        MUON_ELEMENTS,
    ) else {
        return;
    };
    let matrix_element = lane & (MUON_ELEMENTS - 1);
    let active = (lane < MUON_ELEMENTS) as u32 as f32;
    let mut matrix_value =
        active * (shards.load_or(0, matrix_element, 0.0) + shards.load_or(1, matrix_element, 0.0));
    let subgroup = Gfx950Subgroup::current();
    // Reduce in FP32, then normalize before the Newton-Schulz iterations.
    #[cfg(not(feature = "ablation-muon-broadcast16"))]
    let squared_norm = subgroup.reduce_sum_f32::<64>(matrix_value * matrix_value);
    #[cfg(feature = "ablation-muon-broadcast16")]
    let squared_norm = {
        let local_square = matrix_value * matrix_value;
        let mut sum = subgroup.broadcast_f32::<64>(local_square, 0);
        sum += subgroup.broadcast_f32::<64>(local_square, 1);
        sum += subgroup.broadcast_f32::<64>(local_square, 2);
        sum += subgroup.broadcast_f32::<64>(local_square, 3);
        sum += subgroup.broadcast_f32::<64>(local_square, 4);
        sum += subgroup.broadcast_f32::<64>(local_square, 5);
        sum += subgroup.broadcast_f32::<64>(local_square, 6);
        sum += subgroup.broadcast_f32::<64>(local_square, 7);
        sum += subgroup.broadcast_f32::<64>(local_square, 8);
        sum += subgroup.broadcast_f32::<64>(local_square, 9);
        sum += subgroup.broadcast_f32::<64>(local_square, 10);
        sum += subgroup.broadcast_f32::<64>(local_square, 11);
        sum += subgroup.broadcast_f32::<64>(local_square, 12);
        sum += subgroup.broadcast_f32::<64>(local_square, 13);
        sum += subgroup.broadcast_f32::<64>(local_square, 14);
        sum += subgroup.broadcast_f32::<64>(local_square, 15);
        sum
    };
    let norm = DeviceMath::current().sqrt_f32(squared_norm);
    let inverse_norm = 1.0 / (norm + 1.0e-6);
    matrix_value *= inverse_norm;
    let row = matrix_element / 4;
    let column = matrix_element.wrapping_sub(row.wrapping_mul(4));
    let row_base = row.wrapping_mul(4);
    let column_base = column.wrapping_mul(4);
    // Each uniform iteration forms X X^T and then X X^T X through wave broadcasts.
    macro_rules! muon_iteration {
        () => {{
            let mut gram = 0.0_f32;
            gram += subgroup.broadcast_f32::<64>(matrix_value, row_base as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column_base as u32 & 63);
            gram += subgroup
                .broadcast_f32::<64>(matrix_value, row_base.wrapping_add(1) as u32 & 63)
                * subgroup
                    .broadcast_f32::<64>(matrix_value, column_base.wrapping_add(1) as u32 & 63);
            gram += subgroup
                .broadcast_f32::<64>(matrix_value, row_base.wrapping_add(2) as u32 & 63)
                * subgroup
                    .broadcast_f32::<64>(matrix_value, column_base.wrapping_add(2) as u32 & 63);
            gram += subgroup
                .broadcast_f32::<64>(matrix_value, row_base.wrapping_add(3) as u32 & 63)
                * subgroup
                    .broadcast_f32::<64>(matrix_value, column_base.wrapping_add(3) as u32 & 63);
            let mut cubic = 0.0_f32;
            cubic += subgroup.broadcast_f32::<64>(gram, row_base as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, row_base.wrapping_add(1) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column.wrapping_add(4) as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, row_base.wrapping_add(2) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column.wrapping_add(8) as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, row_base.wrapping_add(3) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column.wrapping_add(12) as u32 & 63);
            matrix_value = 1.5 * matrix_value - 0.5 * cubic;
        }};
    }
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    // Matrix lanes own disjoint outputs; lane zero separately owns the reported norm.
    if lane < MUON_ELEMENTS {
        let Some(output_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(slot) = output.get_tiled_2d_mut(
            &output_tile,
            0,
            SYSTEM_BATCHES,
            MUON_ELEMENTS,
            MUON_ELEMENTS,
        ) {
            *slot = -MUON_LEARNING_RATE * matrix_value;
        }
    }
    if lane == 0 {
        let Some(norm_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(slot) = output_norm.get_tiled_2d_mut(&norm_tile, 0, SYSTEM_BATCHES, 1, 1) {
            *slot = norm;
        }
    }
}
