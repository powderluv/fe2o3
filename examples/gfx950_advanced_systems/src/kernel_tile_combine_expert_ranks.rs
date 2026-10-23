//! Fixed-order expert combine with actual checked 1x256 tile write authority.
#![allow(missing_docs, unused_imports)]
#[cfg(feature="ablation-combine-transposed")]
compile_error!("expert combine tile mode has no qualified transposed ablation pairing");
use fe2o3_device::{DeviceMath, DisjointSlice, Gfx950Subgroup, Index1D, StridedReadView2D, Tiled2D, kernel, thread};
use crate::{SYSTEM_BATCHES,TOKENS,HIDDEN,EXPERTS,TOP_K,DISPATCH_CAPACITY,COMBINE_BATCHES,OUTPUT};
#[kernel(typed,launch(required=[256,1,1],max=[256,1,1],max_grid=[4,1,1]))]
pub fn gfx950_combine_expert_ranks_v1(
    rank0: &[f32],
    rank1: &[f32],
    mut output: DisjointSlice<f32, Tiled2D<Index1D, 256, 1, 256, 1>>,
) {
    // This elementwise boundary has no collectives in the production path.
    let index = thread::index_1d();
    let element = index.get();
    if rank0.len() != COMBINE_BATCHES * TOKENS * OUTPUT
        || rank1.len() != COMBINE_BATCHES * TOKENS * OUTPUT
        || output.len() != COMBINE_BATCHES * TOKENS * OUTPUT
    {
        return;
    }
    if element >= COMBINE_BATCHES * TOKENS * OUTPUT {
        return;
    }
    // Fixed rank order makes the floating-point reference and device result reproducible.
    #[cfg(not(feature = "ablation-combine-transposed"))]
    let result = rank0[element] + rank1[element];
    #[cfg(feature = "ablation-combine-transposed")]
    let result = {
        let wave_lane = element & 63;
        let source_lane = 63 - wave_lane;
        let source_element = (element & !63) + source_lane;
        let source_result = rank0[source_element] + rank1[source_element];
        Gfx950Subgroup::current().broadcast_f32::<64>(source_result, source_lane as u32)
    };
    // Row-striped ownership maps each in-range thread to exactly one result.
    let Some(output_row) = index.checked_tiled_2d::<256, 1, 256, 1>() else {
        return;
    };
    if let Some(slot) = output.get_tiled_2d_mut(
        &output_row,
        0,
        COMBINE_BATCHES,
        TOKENS * OUTPUT,
        TOKENS * OUTPUT,
    ) {
        *slot = result;
    }
}
