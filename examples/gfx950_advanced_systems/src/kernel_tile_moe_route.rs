//! Checked tile-output MoE routing; numerical routing and packed inputs are unchanged.
#![allow(missing_docs, unused_imports, unused_assignments)]
#[cfg(any(feature="ablation-route-owner-only",feature="ablation-route-unpacked"))]
compile_error!("MoE route tile mode has no qualified ablation pairing");
use fe2o3_device::{DeviceMath, DisjointSlice, Gfx950Subgroup, Index1D, StridedReadView2D, Tiled2D, kernel, thread};
use crate::{SYSTEM_BATCHES,TOKENS,HIDDEN,EXPERTS,TOP_K,DISPATCH_CAPACITY,COMBINE_BATCHES,OUTPUT};
// A Wave64 owns one clipped 1x64 top/count tile and one 2x64 dispatch tile.
// Dispatch's 32x64 parent is byte-identical to the original 16x128 parent.
#[kernel(typed, launch(required=[256,1,1],max=[256,1,1],max_grid=[4,1,1]), control_flow(loop_bounds(128,32,32,32)))]
#[allow(clippy::too_many_arguments)]
pub fn gfx950_moe_route_fp4_t16_e4_k2_v1(
    activations: &[u8],
    router_weights: &[f32],
    mut top_experts: DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut top_weights: DisjointSlice<f32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut expert_counts: DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut dispatch: DisjointSlice<i32, Tiled2D<Index1D, 64, 2, 64, 2>>,
) {
    // One Wave64 owns one batch. Every lane participates in the broadcasts below.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let wave_lane = global_index & 63;
    // Reject the complete buffer contract before any lane enters a collective.
    if batch >= SYSTEM_BATCHES
        || activations.len() != SYSTEM_BATCHES * TOKENS * HIDDEN
        || router_weights.len() != SYSTEM_BATCHES * EXPERTS * HIDDEN
        || top_experts.len() != SYSTEM_BATCHES * TOKENS * TOP_K
        || top_weights.len() != SYSTEM_BATCHES * TOKENS * TOP_K
        || expert_counts.len() != SYSTEM_BATCHES * EXPERTS
        || dispatch.len() != SYSTEM_BATCHES * EXPERTS * DISPATCH_CAPACITY
    {
        return;
    }
    // Typed views keep the batch offsets and row strides out of the dot-product loop.
    let activation_base = batch.wrapping_mul(TOKENS).wrapping_mul(HIDDEN);
    let router_base = batch.wrapping_mul(EXPERTS).wrapping_mul(HIDDEN);
    let Ok(router_weights) =
        StridedReadView2D::from_shared_slice(router_weights, router_base, EXPERTS, HIDDEN, HIDDEN)
    else {
        return;
    };
    let token = wave_lane & (TOKENS - 1);
    let Ok(activations) =
        StridedReadView2D::from_shared_slice(activations, activation_base, TOKENS, HIDDEN, HIDDEN)
    else {
        return;
    };
    // Four lanes cooperate on each token by taking every fourth reduction
    // element. This eliminates the former four-way duplicate input traffic;
    // uniform broadcasts merge the partials without a workgroup barrier.
    let mut route_logit0 = 0.0_f32;
    let mut route_logit1 = 0.0_f32;
    let mut route_logit2 = 0.0_f32;
    let mut route_logit3 = 0.0_f32;
    #[cfg(feature = "ablation-route-redundant-lanes")]
    let depth_group = 0_usize;
    #[cfg(not(feature = "ablation-route-redundant-lanes"))]
    let depth_group = wave_lane / TOKENS;
    #[cfg(feature = "ablation-route-redundant-lanes")]
    let depth_stride = 1_usize;
    #[cfg(not(feature = "ablation-route-redundant-lanes"))]
    let depth_stride = 4_usize;
    #[cfg(feature = "ablation-route-redundant-lanes")]
    let depth_steps = HIDDEN;
    #[cfg(not(feature = "ablation-route-redundant-lanes"))]
    let depth_steps = HIDDEN / 4;
    let mut step = 0_usize;
    while step < depth_steps {
        let depth = depth_group.wrapping_add(step.wrapping_mul(depth_stride));
        let bits = activations.load_or(token, depth, 0);
        let fp4_shift = ((bits & 7) as u32) * 4;
        let fp4_magnitudes = if fp4_shift < 32 { 0xc864_3210_u32 >> fp4_shift } else { 0 };
        let magnitude = (fp4_magnitudes & 15) as f32 * 0.5;
        let sign = 1.0 - 2.0 * ((bits >> 3) & 1) as f32;
        let activation = sign * magnitude;
        route_logit0 += activation * router_weights.load_or(0, depth, 0.0);
        route_logit1 += activation * router_weights.load_or(1, depth, 0.0);
        route_logit2 += activation * router_weights.load_or(2, depth, 0.0);
        route_logit3 += activation * router_weights.load_or(3, depth, 0.0);
        step += 1;
    }
    let subgroup = Gfx950Subgroup::current();
    #[cfg(not(feature = "ablation-route-redundant-lanes"))]
    {
        let partial0 = route_logit0;
        let partial1 = route_logit1;
        let partial2 = route_logit2;
        let partial3 = route_logit3;
        route_logit0 = partial0
            + subgroup.broadcast_f32::<64>(partial0, token.wrapping_add(16) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial0, token.wrapping_add(32) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial0, token.wrapping_add(48) as u32 & 63);
        route_logit1 = partial1
            + subgroup.broadcast_f32::<64>(partial1, token.wrapping_add(16) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial1, token.wrapping_add(32) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial1, token.wrapping_add(48) as u32 & 63);
        route_logit2 = partial2
            + subgroup.broadcast_f32::<64>(partial2, token.wrapping_add(16) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial2, token.wrapping_add(32) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial2, token.wrapping_add(48) as u32 & 63);
        route_logit3 = partial3
            + subgroup.broadcast_f32::<64>(partial3, token.wrapping_add(16) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial3, token.wrapping_add(32) as u32 & 63)
            + subgroup.broadcast_f32::<64>(partial3, token.wrapping_add(48) as u32 & 63);
    }
    // A branch-light ranking network gives deterministic top-2 tie handling.
    let precedes12 = (route_logit1 >= route_logit2) as u32;
    let precedes13 = (route_logit1 >= route_logit3) as u32;
    let precedes23 = (route_logit2 >= route_logit3) as u32;
    let rank1 = ((route_logit0 >= route_logit1) as u32)
        .wrapping_add(2)
        .wrapping_sub(precedes12)
        .wrapping_sub(precedes13);
    let rank2 = ((route_logit0 >= route_logit2) as u32)
        .wrapping_add(precedes12)
        .wrapping_add(1)
        .wrapping_sub(precedes23);
    let rank3 = ((route_logit0 >= route_logit3) as u32)
        .wrapping_add(precedes13)
        .wrapping_add(precedes23);
    let first_local = ((rank1 == 0) as u32)
        .wrapping_add(2_u32.wrapping_mul((rank2 == 0) as u32))
        .wrapping_add(3_u32.wrapping_mul((rank3 == 0) as u32));
    let second_local = ((rank1 == 1) as u32)
        .wrapping_add(2_u32.wrapping_mul((rank2 == 1) as u32))
        .wrapping_add(3_u32.wrapping_mul((rank3 == 1) as u32));
    let first_logit = if first_local == 0 {
        route_logit0
    } else if first_local == 1 {
        route_logit1
    } else if first_local == 2 {
        route_logit2
    } else {
        route_logit3
    };
    let second_logit = if second_local == 0 {
        route_logit0
    } else if second_local == 1 {
        route_logit1
    } else if second_local == 2 {
        route_logit2
    } else {
        route_logit3
    };
    // Normalize only the selected logits, using the max subtraction for stability.
    let maximum = if first_logit > second_logit {
        first_logit
    } else {
        second_logit
    };
    let math = DeviceMath::current();
    let first_exp = math.exp_f32(first_logit - maximum);
    let second_exp = math.exp_f32(second_logit - maximum);
    let denominator = first_exp + second_exp;
    let first_weight_local = first_exp / denominator;
    let second_weight_local = second_exp / denominator;
    // Broadcast each token's route so lanes can form counts and dispatch metadata.
    let top_source = ((wave_lane / TOP_K) & (TOKENS - 1)) as u32 & 63;
    let local_pair = first_local | (second_local << 2);
    let top_pair = subgroup.broadcast_f32::<64>(local_pair as f32, top_source) as u32;
    let top_first = top_pair & 3;
    let top_second = top_pair >> 2;
    let top_first_weight = subgroup.broadcast_f32::<64>(first_weight_local, top_source);
    let top_second_weight = subgroup.broadcast_f32::<64>(second_weight_local, top_source);
    let packed_routes = (subgroup.broadcast_f32::<64>(local_pair as f32, 0) as u64)
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 1) as u64) << 4
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 2) as u64) << 8
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 3) as u64) << 12
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 4) as u64) << 16
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 5) as u64) << 20
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 6) as u64) << 24
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 7) as u64) << 28
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 8) as u64) << 32
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 9) as u64) << 36
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 10) as u64) << 40
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 11) as u64) << 44
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 12) as u64) << 48
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 13) as u64) << 52
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 14) as u64) << 56
        | (subgroup.broadcast_f32::<64>(local_pair as f32, 15) as u64) << 60;
    // Row-striped capabilities prove that route, count, and dispatch stores do not alias.
    if wave_lane < TOKENS * TOP_K {
        let choice = wave_lane & (TOP_K - 1);
        let selected = if choice == 0 { top_first } else { top_second };
        let weight = if choice == 0 {
            top_first_weight
        } else {
            top_second_weight
        };
        let Some(top_row) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(slot) = top_experts.get_tiled_2d_mut(
            &top_row,
            0,
            SYSTEM_BATCHES,
            TOKENS * TOP_K,
            TOKENS * TOP_K,
        ) {
            *slot = selected;
        }
        if let Some(slot) = top_weights.get_tiled_2d_mut(
            &top_row,
            0,
            SYSTEM_BATCHES,
            TOKENS * TOP_K,
            TOKENS * TOP_K,
        ) {
            *slot = weight;
        }
    }
    let count_expert = wave_lane as u32;
    let mut count = 0_u32;
    let mut record = 0_usize;
    while record < TOKENS * TOP_K {
        let shift = 2 * record;
        let selected = if shift < 64 { ((packed_routes >> shift) & 3) as u32 } else { 0 };
        count = count.wrapping_add((selected == count_expert) as u32);
        record += 1;
    }
    if wave_lane < EXPERTS {
        let Some(count_row) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(slot) =
            expert_counts.get_tiled_2d_mut(&count_row, 0, SYSTEM_BATCHES, EXPERTS, EXPERTS)
        {
            *slot = count;
        }
    }
    let Some(dispatch_row) = thread::index_1d().checked_tiled_2d::<64, 2, 64, 2>() else {
        return;
    };
    let dispatch_expert0 = (wave_lane / DISPATCH_CAPACITY) as u32;
    let wanted0 =
        wave_lane.wrapping_sub((dispatch_expert0 as usize).wrapping_mul(DISPATCH_CAPACITY));
    let mut seen0 = 0_usize;
    let mut dispatched0 = -1_i32;
    let mut route0 = 0_usize;
    while route0 < TOKENS * TOP_K {
        let shift = 2 * route0;
        let selected = if shift < 64 { ((packed_routes >> shift) & 3) as u32 } else { 0 };
        let dispatch_matches = (selected == dispatch_expert0) as usize;
        let choose = ((dispatch_matches != 0) & (seen0 == wanted0)) as i32;
        dispatched0 = dispatched0.wrapping_add(
            (route0 as i32)
                .wrapping_sub(dispatched0)
                .wrapping_mul(choose),
        );
        seen0 = seen0.wrapping_add(dispatch_matches);
        route0 += 1;
    }
    if let Some(slot) = dispatch.get_tiled_2d_mut(
        &dispatch_row,
        0,
        SYSTEM_BATCHES * 2,
        64,
        64,
    ) {
        *slot = dispatched0;
    }
    let dispatch_element1 = wave_lane.wrapping_add(64);
    let dispatch_expert1 = (dispatch_element1 / DISPATCH_CAPACITY) as u32;
    let wanted1 =
        dispatch_element1.wrapping_sub((dispatch_expert1 as usize).wrapping_mul(DISPATCH_CAPACITY));
    let mut seen1 = 0_usize;
    let mut dispatched1 = -1_i32;
    let mut route1 = 0_usize;
    while route1 < TOKENS * TOP_K {
        let shift = 2 * route1;
        let selected = if shift < 64 { ((packed_routes >> shift) & 3) as u32 } else { 0 };
        let dispatch_matches = (selected == dispatch_expert1) as usize;
        let choose = ((dispatch_matches != 0) & (seen1 == wanted1)) as i32;
        dispatched1 = dispatched1.wrapping_add(
            (route1 as i32)
                .wrapping_sub(dispatched1)
                .wrapping_mul(choose),
        );
        seen1 = seen1.wrapping_add(dispatch_matches);
        route1 += 1;
    }
    if let Some(slot) = dispatch.get_tiled_2d_mut(
        &dispatch_row,
        1,
        SYSTEM_BATCHES * 2,
        64,
        64,
    ) {
        *slot = dispatched1;
    }
}
