//! Checked 4x16 memory tiles for the existing F32 residual/mixing operations.
//!
//! These are memory-tile kernels, not matrix-multiply or reduced-precision paths.
//! Arithmetic and collective ordering are retained from their SIMT counterparts.
#![allow(missing_docs)]
use crate::{CHANNELS_V1, MIXING_STREAMS_V1};
use fe2o3_device::{
    DeviceMath, DisjointSlice, Gfx950Subgroup, Index1D, KernelError, KernelResult,
    StridedReadView2D, Tiled2D, kernel, thread,
};
const MULTIGRID_SUBGROUP_BATCHES_V1: usize = 64;
const MULTIGRID_WAVE_BATCHES_V1: usize = 16;

#[cfg(feature = "kernel-attnres-aggregate-tile")]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(4, 4))
)]
pub fn gfx950_attnres_aggregate(
    depth_values: &[f32],
    depth_logits: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 4, 16, 1>>,
) -> KernelResult {
    // Wave16 batches map one thread to one output channel.
    let batches = MULTIGRID_SUBGROUP_BATCHES_V1;
    if depth_values.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
        || depth_logits.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
        || output.len() != batches * CHANNELS_V1
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let linear = index.get();
    let batch = linear / CHANNELS_V1;
    let channel = linear % CHANNELS_V1;
    let batch_offset = batch.wrapping_mul(MIXING_STREAMS_V1 * CHANNELS_V1);
    let Ok(values) = StridedReadView2D::from_shared_slice(depth_values, batch_offset, 4, 16, 16)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(logits) = StridedReadView2D::from_shared_slice(depth_logits, batch_offset, 4, 16, 16)
    else {
        return Err(KernelError::InvalidArgument);
    };
    // Use a max-subtracted four-way softmax before the weighted depth sum.
    let math = DeviceMath::current();
    let mut maximum = logits.load_or(0, channel, f32::NEG_INFINITY);
    for depth in 1..4 {
        let logit = logits.load_or(depth, channel, f32::NEG_INFINITY);
        if logit > maximum {
            maximum = logit;
        }
    }
    let mut denominator = 0.0;
    let mut value = 0.0;
    for depth in 0..4 {
        let weight = math.exp_f32(logits.load_or(depth, channel, f32::NEG_INFINITY) - maximum);
        denominator += weight;
        value += weight * values.load_or(depth, channel, 0.0);
    }
    // The linear index is the exclusive owner of this channel.
    // Four rows by sixteen columns give every lane a distinct checked cell.
    let Some(tile) = index.checked_tiled_2d::<64, 4, 16, 1>() else {
        return Err(KernelError::InvalidArgument);
    };
    if let Some(slot) = output.get_tiled_2d_mut(&tile, 0, 64, 16, 16) {
        *slot = value / denominator;
    }
    Ok(())
}

#[cfg(feature = "kernel-four-branch-residual-tile")]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(4))
)]
pub fn gfx950_four_branch_residual(
    residual: &[f32],
    branches: &[f32],
    gate_logits: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 4, 16, 1>>,
) {
    // Wave16 batches map one thread to one residual channel.
    let batches = MULTIGRID_SUBGROUP_BATCHES_V1;
    if residual.len() != batches * CHANNELS_V1
        || branches.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
        || gate_logits.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
        || output.len() != batches * CHANNELS_V1
    {
        return;
    }
    let index = thread::index_1d();
    let linear = index.get();
    let batch = linear / CHANNELS_V1;
    let channel = linear % CHANNELS_V1;
    let batch_offset = batch.wrapping_mul(MIXING_STREAMS_V1 * CHANNELS_V1);
    let math = DeviceMath::current();
    // Accumulate each sigmoid gate explicitly in stable branch order.
    let mut value = residual[batch.wrapping_mul(CHANNELS_V1).wrapping_add(channel)];
    for branch in 0_usize..4 {
        let offset = batch_offset
            .wrapping_add(branch.wrapping_mul(CHANNELS_V1))
            .wrapping_add(channel);
        let gate = 1.0 / (1.0 + math.exp_f32(-gate_logits[offset]));
        value += 0.25 * gate * branches[offset];
    }
    // The linear index is the exclusive owner of this channel.
    // Four rows by sixteen columns give every lane a distinct checked cell.
    let Some(tile) = index.checked_tiled_2d::<64, 4, 16, 1>() else {
        return;
    };
    if let Some(slot) = output.get_tiled_2d_mut(&tile, 0, 64, 16, 16) {
        *slot = value;
    }
}

#[cfg(feature = "kernel-mhc-sinkhorn-mix-tile")]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(3))
)]
pub fn gfx950_mhc_sinkhorn_mix(
    streams: &[f32],
    mixing_logits: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 4, 16, 1>>,
) -> KernelResult {
    // One Wave64 owns one item: four rows by 16 channel lanes.
    let batches = MULTIGRID_WAVE_BATCHES_V1;
    if streams.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
        || mixing_logits.len() != batches * MIXING_STREAMS_V1 * MIXING_STREAMS_V1
        || output.len() != batches * MIXING_STREAMS_V1 * CHANNELS_V1
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let linear = index.get();
    let batch = linear / 64;
    let local = thread::thread_idx_x() as usize % 64;
    let math = DeviceMath::current();
    let subgroup = Gfx950Subgroup::current();
    let Ok(logits) = StridedReadView2D::from_shared_slice(
        mixing_logits,
        batch.wrapping_mul(MIXING_STREAMS_V1 * MIXING_STREAMS_V1),
        1,
        16,
        16,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(streams) = StridedReadView2D::from_shared_slice(
        streams,
        batch.wrapping_mul(MIXING_STREAMS_V1 * CHANNELS_V1),
        4,
        16,
        16,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    // Map each lane to one 4x4 mixing coefficient and its output channel.
    let row = local / CHANNELS_V1;
    let local_lane = local % CHANNELS_V1;
    let matrix_index = local_lane.wrapping_add(row.wrapping_mul(MIXING_STREAMS_V1))
        % (MIXING_STREAMS_V1 * MIXING_STREAMS_V1);
    let mut matrix = math.exp_f32(logits.load_or(0, matrix_index, 0.0));
    // Alternate row and column normalization for the fixed Sinkhorn depth.
    for _iteration in 0..3 {
        let row_reciprocal = 1.0 / subgroup.reduce_sum_f32::<4>(matrix);
        matrix *= row_reciprocal;

        let column = (local_lane as u32) & 3;
        let column_sum = subgroup.broadcast_f32::<16>(matrix, column)
            + subgroup.broadcast_f32::<16>(matrix, column.wrapping_add(4) & 15)
            + subgroup.broadcast_f32::<16>(matrix, column.wrapping_add(8) & 15)
            + subgroup.broadcast_f32::<16>(matrix, column.wrapping_add(12) & 15);
        matrix *= 1.0 / column_sum;
    }
    // Broadcast the normalized row and mix the four input streams.
    let weight0 = subgroup.broadcast_f32::<16>(matrix, 0);
    let weight1 = subgroup.broadcast_f32::<16>(matrix, 1);
    let weight2 = subgroup.broadcast_f32::<16>(matrix, 2);
    let weight3 = subgroup.broadcast_f32::<16>(matrix, 3);
    let value = weight0 * streams.load_or(0, local_lane, 0.0)
        + weight1 * streams.load_or(1, local_lane, 0.0)
        + weight2 * streams.load_or(2, local_lane, 0.0)
        + weight3 * streams.load_or(3, local_lane, 0.0);
    // Four rows by sixteen columns give every lane a distinct checked cell.
    let Some(tile) = index.checked_tiled_2d::<64, 4, 16, 1>() else {
        return Err(KernelError::InvalidArgument);
    };
    if let Some(slot) = output.get_tiled_2d_mut(&tile, 0, 64, 16, 16) {
        *slot = value;
    }
    Ok(())
}
