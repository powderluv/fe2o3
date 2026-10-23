//! Structured output tiles for the existing gradient staging kernel.
#![allow(missing_docs)]
use crate::{MUON_ELEMENTS, SYSTEM_BATCHES};
#[cfg(feature = "ablation-stage-tile4")]
use fe2o3_device::Gfx950Subgroup;
use fe2o3_device::{DisjointSlice, Index1D, Tiled2D, kernel, thread};

/// Copies one gradient shard into deterministic transport staging.
#[cfg(feature = "kernel-stage-gradient-shard-tile")]
#[cfg_attr(
    not(feature = "ablation-stage-tile4"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-stage-tile4",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
pub fn gfx950_stage_gradient_shard_v1(
    input: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
) {
    // One wave stages one batch; only the first 16 lanes correspond to matrix elements.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let element = global_index & 63;
    if batch >= SYSTEM_BATCHES
        || input.len() != SYSTEM_BATCHES * MUON_ELEMENTS
        || output.len() != SYSTEM_BATCHES * MUON_ELEMENTS
    {
        return;
    }
    if element >= MUON_ELEMENTS {
        return;
    }
    let input_base = batch.wrapping_mul(MUON_ELEMENTS);
    // Preserve the full F32 input bits; this tile path performs no arithmetic on them.
    #[cfg(not(feature = "ablation-stage-tile4"))]
    let value = input[input_base.wrapping_add(element)];
    #[cfg(feature = "ablation-stage-tile4")]
    let value = {
        let mut tile0 = 0.0_f32;
        let mut tile1 = 0.0_f32;
        let mut tile2 = 0.0_f32;
        let mut tile3 = 0.0_f32;
        if element < 4 {
            let tile_base = element * 4;
            tile0 = input[input_base.wrapping_add(tile_base)];
            tile1 = input[input_base.wrapping_add(tile_base).wrapping_add(1)];
            tile2 = input[input_base.wrapping_add(tile_base).wrapping_add(2)];
            tile3 = input[input_base.wrapping_add(tile_base).wrapping_add(3)];
        }
        let source = (element / 4) as u32;
        let subgroup = Gfx950Subgroup::current();
        let value0 = subgroup.broadcast_f32::<64>(tile0, source);
        let value1 = subgroup.broadcast_f32::<64>(tile1, source);
        let value2 = subgroup.broadcast_f32::<64>(tile2, source);
        let value3 = subgroup.broadcast_f32::<64>(tile3, source);
        if element & 3 == 0 {
            value0
        } else if element & 3 == 1 {
            value1
        } else if element & 3 == 2 {
            value2
        } else {
            value3
        }
    };
    // Each wave owns a 1x64 tile clipped to the full 16-column logical output.
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
        *slot = value;
    }
}
