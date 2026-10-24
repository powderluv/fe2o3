//! Checked 8x8 output tiles for the fixed masked Wave64 collectives.

#![allow(missing_docs)] // The V1 kernel macro emits an undocumented helper module.

use fe2o3_device::{
    DisjointSlice, Gfx942Collectives, Index1D, SubgroupTile, Tiled2D, Wave64, WaveLane, kernel,
    thread,
};

/// Exact launch dimensions required by the Phase A source contract.
pub const WAVE64_COLLECTIVES_WORKGROUP_V1: [u32; 3] = [64, 1, 1];

/// Computes one masked sum reduction plus inclusive and exclusive scans.
///
/// The `u64` mask is logical participation data. Every physical lane follows
/// the same three collective calls. Inactive lanes contribute positive zero
/// and publish positive zero; active lanes publish the full reduction and
/// their increasing-lane-order prefixes.
#[kernel(
    typed,
    launch(required = [64, 1, 1], max = [64, 1, 1])
)]
pub fn wave64_collectives_v1(
    input: &[f32],
    active_mask: u64,
    mut reduction_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
    mut inclusive_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
    mut exclusive_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
) {
    let lane_index = thread::index_1d();
    let lane = lane_index.get();
    if lane >= 64
        || input.len() != 64
        || reduction_output.len() != 64
        || inclusive_output.len() != 64
        || exclusive_output.len() != 64
    {
        fe2o3_device::trap();
    }

    let active = active_mask & (1_u64 << lane) != 0;
    let contribution = if active { input[lane] } else { 0.0_f32 };

    let lane_snapshot = WaveLane::<Wave64>::current();
    let wave = SubgroupTile::<64>::from_wave64_snapshot(&lane_snapshot);

    let context = Gfx942Collectives::current();

    let reduction = wave.reduce_sum(&context, contribution);
    let inclusive = wave.inclusive_scan_sum(&context, contribution);
    let exclusive = wave.exclusive_scan_sum(&context, contribution);

    let published_reduction = if active { reduction } else { 0.0 };
    let published_inclusive = if active { inclusive } else { 0.0 };
    let published_exclusive = if active { exclusive } else { 0.0 };

    let Some(tile) = thread::index_1d().checked_tiled_2d::<64, 8, 8, 1>() else {
        fe2o3_device::trap();
    };
    if let Some(output) = reduction_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_reduction;
    }
    if let Some(output) = inclusive_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_inclusive;
    }
    if let Some(output) = exclusive_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_exclusive;
    }
}
