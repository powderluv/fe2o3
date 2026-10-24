//! Complete safe Rust kernel source for the bounded systems profiles.
//!
//! Read every entry point in the same order: validate the whole launch before
//! subgroup operations, map the global thread to a batch/wave/lane owner, build
//! checked typed views, run collectives under uniform control flow, and finish
//! with a `DisjointSlice` store whose layout makes write ownership explicit.
//! Comments emphasize those invariants and the reason for non-obvious code;
//! they intentionally do not narrate ordinary Rust syntax.

#![allow(missing_docs)] // The kernel macro emits helper modules.
#![cfg_attr(target_arch = "amdgpu", allow(unused_imports))]

use fe2o3_device::{
    Blocked, DeviceMath, DisjointSlice, Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1,
    Gfx950Fp4MfmaAMatrix, Gfx950Fp8MfmaBMatrix, Gfx950Matrix, Gfx950Subgroup, Index1D,
    RowStriped2D, StridedReadView2D, Wave64, WaveLane, kernel, thread,
};

use crate::{
    ALL_EXPERTS, CANDIDATES, COMBINE_BATCHES, DISPATCH_CAPACITY, DRAFT_STEPS, EXPERTS,
    GRADIENT_SHARDS, HIDDEN, MUON_ELEMENTS, MUON_LEARNING_RATE, NGRAM, OUTPUT, QUERIES,
    STATE_WIDTH, SYSTEM_BATCHES, TABLE_SIZE, TOKENS, TOP_K,
};

/// Stable top-2 routing, weights, expert counts, and compact dispatch metadata.
#[cfg(all(any(not(target_arch = "amdgpu"), feature = "kernel-moe-route"), not(feature = "kernel-moe-route-tile")))]
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(128, 32, 32, 32))
)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn gfx950_moe_route_fp4_t16_e4_k2_v1(
    activations: &[u8],
    router_weights: &[f32],
    mut top_experts: DisjointSlice<u32, RowStriped2D<Index1D, 64, 1>>,
    mut top_weights: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
    mut expert_counts: DisjointSlice<u32, RowStriped2D<Index1D, 64, 1>>,
    mut dispatch: DisjointSlice<i32, RowStriped2D<Index1D, 64, 2>>,
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
        let Some(top_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
            return;
        };
        if let Some(slot) = top_experts.get_row_striped_2d_mut(
            &top_row,
            0,
            SYSTEM_BATCHES,
            TOKENS * TOP_K,
            TOKENS * TOP_K,
        ) {
            *slot = selected;
        }
        if let Some(slot) = top_weights.get_row_striped_2d_mut(
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
        let Some(count_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
            return;
        };
        if let Some(slot) =
            expert_counts.get_row_striped_2d_mut(&count_row, 0, SYSTEM_BATCHES, EXPERTS, EXPERTS)
        {
            *slot = count;
        }
    }
    let Some(dispatch_row) = thread::index_1d().checked_row_striped_2d::<64, 2>() else {
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
    if let Some(slot) = dispatch.get_row_striped_2d_mut(
        &dispatch_row,
        0,
        SYSTEM_BATCHES,
        EXPERTS * DISPATCH_CAPACITY,
        EXPERTS * DISPATCH_CAPACITY,
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
    if let Some(slot) = dispatch.get_row_striped_2d_mut(
        &dispatch_row,
        1,
        SYSTEM_BATCHES,
        EXPERTS * DISPATCH_CAPACITY,
        EXPERTS * DISPATCH_CAPACITY,
    ) {
        *slot = dispatched1;
    }
}

/// Computes a routed expert partition and optional shared-expert contribution.
#[cfg(all(any(not(target_arch = "amdgpu"), feature = "kernel-moe-expert-rank"), not(feature = "kernel-moe-expert-rank-simt")))]
#[cfg_attr(
    not(feature = "ablation-expert-serial"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-expert-serial",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn gfx950_moe_expert_rank_fp4_fp8_v1(
    activations: &[u8],
    expert_weights: &[u8],
    top_experts: &[u32],
    top_weights: &[f32],
    first_expert: u32,
    include_shared_expert: u32,
    mut output: DisjointSlice<f32, Blocked<Index1D, 64, 4>>,
) {
    // One Wave64 owns one batch; each lane ultimately writes four output elements.
    let thread_index = thread::index_1d();
    let global_index = thread_index.get();
    let batch = global_index / 64;
    let lane_index = global_index & 63;
    // Validate every buffer and rank selector before the first MFMA or broadcast.
    if batch >= SYSTEM_BATCHES
        || activations.len() != SYSTEM_BATCHES * TOKENS * HIDDEN
        || expert_weights.len() != SYSTEM_BATCHES * ALL_EXPERTS * HIDDEN * OUTPUT
        || top_experts.len() != SYSTEM_BATCHES * TOKENS * TOP_K
        || top_weights.len() != SYSTEM_BATCHES * TOKENS * TOP_K
        || output.len() != SYSTEM_BATCHES * TOKENS * OUTPUT
        || first_expert as usize >= EXPERTS - 1
    {
        return;
    }
    // Convert raw storage into typed MFMA views for two routed experts plus one shared expert.
    let second_expert = first_expert.wrapping_add(1);
    let lane = WaveLane::<Wave64>::current();
    let activation_base = batch.wrapping_mul(TOKENS).wrapping_mul(HIDDEN);
    let expert_batch_base = batch
        .wrapping_mul(ALL_EXPERTS)
        .wrapping_mul(HIDDEN)
        .wrapping_mul(OUTPUT);
    let route_batch_base = batch.wrapping_mul(TOKENS).wrapping_mul(TOP_K);
    let first_offset = expert_batch_base.wrapping_add(
        (first_expert as usize)
            .wrapping_mul(HIDDEN)
            .wrapping_mul(OUTPUT),
    );
    // Keep the three independent fragment lifetimes overlapping in the production path.
    #[cfg(not(feature = "ablation-expert-serial"))]
    let (first_values, second_values, shared_values) = {
        let Ok(activations_view) =
            Gfx950Fp4MfmaAMatrix::row_major(activations, activation_base, TOKENS, HIDDEN, HIDDEN)
        else {
            return;
        };
        let activations_first = activations_view.load_m16k128(&lane, 0, 0);
        let activations_second = activations_view.load_m16k128(&lane, 0, 0);
        let activations_shared = activations_view.load_m16k128(&lane, 0, 0);
        let Ok(first_weights_view) =
            Gfx950Fp8MfmaBMatrix::row_major(expert_weights, first_offset, HIDDEN, OUTPUT, OUTPUT)
        else {
            return;
        };
        let first_weights = first_weights_view.load_k128n16(&lane, 0, 0);
        let Ok(second_weights_view) = Gfx950Fp8MfmaBMatrix::row_major(
            expert_weights,
            first_offset.wrapping_add(HIDDEN * OUTPUT),
            HIDDEN,
            OUTPUT,
            OUTPUT,
        ) else {
            return;
        };
        let second_weights = second_weights_view.load_k128n16(&lane, 0, 0);
        let Ok(shared_weights_view) = Gfx950Fp8MfmaBMatrix::row_major(
            expert_weights,
            expert_batch_base.wrapping_add((ALL_EXPERTS - 1) * HIDDEN * OUTPUT),
            HIDDEN,
            OUTPUT,
            OUTPUT,
        ) else {
            return;
        };
        let shared_weights = shared_weights_view.load_k128n16(&lane, 0, 0);
        let matrix = Gfx950Matrix::current();
        (
            matrix
                .multiply_accumulate_fp4_fp8(
                    activations_first,
                    first_weights,
                    Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
                )
                .into_values(),
            matrix
                .multiply_accumulate_fp4_fp8(
                    activations_second,
                    second_weights,
                    Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
                )
                .into_values(),
            matrix
                .multiply_accumulate_fp4_fp8(
                    activations_shared,
                    shared_weights,
                    Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
                )
                .into_values(),
        )
    };
    #[cfg(feature = "ablation-expert-serial")]
    let (first_values, second_values, shared_values) = {
        let Ok(activations_view) =
            Gfx950Fp4MfmaAMatrix::row_major(activations, activation_base, TOKENS, HIDDEN, HIDDEN)
        else {
            return;
        };
        let matrix = Gfx950Matrix::current();
        let activations_first = activations_view.load_m16k128(&lane, 0, 0);
        let Ok(first_weights_view) =
            Gfx950Fp8MfmaBMatrix::row_major(expert_weights, first_offset, HIDDEN, OUTPUT, OUTPUT)
        else {
            return;
        };
        let first_weights = first_weights_view.load_k128n16(&lane, 0, 0);
        let first_values = matrix
            .multiply_accumulate_fp4_fp8(
                activations_first,
                first_weights,
                Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
            )
            .into_values();
        let activations_second = activations_view.load_m16k128(&lane, 0, 0);
        let Ok(second_weights_view) = Gfx950Fp8MfmaBMatrix::row_major(
            expert_weights,
            first_offset.wrapping_add(HIDDEN * OUTPUT),
            HIDDEN,
            OUTPUT,
            OUTPUT,
        ) else {
            return;
        };
        let second_weights = second_weights_view.load_k128n16(&lane, 0, 0);
        let second_values = matrix
            .multiply_accumulate_fp4_fp8(
                activations_second,
                second_weights,
                Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
            )
            .into_values();
        let activations_shared = activations_view.load_m16k128(&lane, 0, 0);
        let Ok(shared_weights_view) = Gfx950Fp8MfmaBMatrix::row_major(
            expert_weights,
            expert_batch_base.wrapping_add((ALL_EXPERTS - 1) * HIDDEN * OUTPUT),
            HIDDEN,
            OUTPUT,
            OUTPUT,
        ) else {
            return;
        };
        let shared_weights = shared_weights_view.load_k128n16(&lane, 0, 0);
        let shared_values = matrix
            .multiply_accumulate_fp4_fp8(
                activations_shared,
                shared_weights,
                Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
            )
            .into_values();
        (first_values, second_values, shared_values)
    };
    let subgroup = Gfx950Subgroup::current();
    let math = DeviceMath::current();
    // Redistribute MFMA accumulator fragments into token/channel output ownership.
    macro_rules! broadcast_component {
        (
            $output_component:literal,
            $first0:ident,
            $first1:ident,
            $first2:ident,
            $first3:ident,
            $second0:ident,
            $second1:ident,
            $second2:ident,
            $second3:ident,
            $shared0:ident,
            $shared1:ident,
            $shared2:ident,
            $shared3:ident
        ) => {
            let element = lane_index.wrapping_add($output_component * 64);
            let token = element / OUTPUT;
            let column = element & (OUTPUT - 1);
            let source_lane = ((token / 4).wrapping_mul(OUTPUT).wrapping_add(column) as u32) & 63;
            let $first0 = subgroup.broadcast_f32::<64>(first_values[0], source_lane);
            let $first1 = subgroup.broadcast_f32::<64>(first_values[1], source_lane);
            let $first2 = subgroup.broadcast_f32::<64>(first_values[2], source_lane);
            let $first3 = subgroup.broadcast_f32::<64>(first_values[3], source_lane);
            let $second0 = subgroup.broadcast_f32::<64>(second_values[0], source_lane);
            let $second1 = subgroup.broadcast_f32::<64>(second_values[1], source_lane);
            let $second2 = subgroup.broadcast_f32::<64>(second_values[2], source_lane);
            let $second3 = subgroup.broadcast_f32::<64>(second_values[3], source_lane);
            let $shared0 = subgroup.broadcast_f32::<64>(shared_values[0], source_lane);
            let $shared1 = subgroup.broadcast_f32::<64>(shared_values[1], source_lane);
            let $shared2 = subgroup.broadcast_f32::<64>(shared_values[2], source_lane);
            let $shared3 = subgroup.broadcast_f32::<64>(shared_values[3], source_lane);
        };
    }
    broadcast_component!(
        0, first00, first01, first02, first03, second00, second01, second02, second03, shared00,
        shared01, shared02, shared03
    );
    broadcast_component!(
        1, first10, first11, first12, first13, second10, second11, second12, second13, shared10,
        shared11, shared12, shared13
    );
    broadcast_component!(
        2, first20, first21, first22, first23, second20, second21, second22, second23, shared20,
        shared21, shared22, shared23
    );
    broadcast_component!(
        3, first30, first31, first32, first33, second30, second31, second32, second33, shared30,
        shared31, shared32, shared33
    );

    // Apply the two route gates and optional shared expert after redistribution.
    macro_rules! compute_component {
        (
            $output_component:literal,
            $first0:ident,
            $first1:ident,
            $first2:ident,
            $first3:ident,
            $second0:ident,
            $second1:ident,
            $second2:ident,
            $second3:ident,
            $shared0:ident,
            $shared1:ident,
            $shared2:ident,
            $shared3:ident
        ) => {{
            let element = lane_index.wrapping_add($output_component * 64);
            let token = element / OUTPUT;
            let accumulator_component = token & 3;
            let first = if accumulator_component == 0 {
                $first0
            } else if accumulator_component == 1 {
                $first1
            } else if accumulator_component == 2 {
                $first2
            } else {
                $first3
            };
            let second = if accumulator_component == 0 {
                $second0
            } else if accumulator_component == 1 {
                $second1
            } else if accumulator_component == 2 {
                $second2
            } else {
                $second3
            };
            let shared = if accumulator_component == 0 {
                $shared0
            } else if accumulator_component == 1 {
                $shared1
            } else if accumulator_component == 2 {
                $shared2
            } else {
                $shared3
            };
            let route_base = route_batch_base.wrapping_add(token.wrapping_mul(TOP_K));
            let route_second = route_base.wrapping_add(1);
            let selected0 = top_experts[route_base];
            let selected1 = top_experts[route_second];
            let gate0 = top_weights[route_base];
            let gate1 = top_weights[route_second];
            let mut result = 0.0_f32;
            if selected0 == first_expert {
                result += gate0 * (first / (1.0 + math.exp_f32(-first)));
            } else if selected0 == second_expert {
                result += gate0 * (second / (1.0 + math.exp_f32(-second)));
            }
            if selected1 == first_expert {
                result += gate1 * (first / (1.0 + math.exp_f32(-first)));
            } else if selected1 == second_expert {
                result += gate1 * (second / (1.0 + math.exp_f32(-second)));
            }
            if include_shared_expert != 0 {
                result += 0.25 * (shared / (1.0 + math.exp_f32(-shared)));
            }
            result
        }};
    }
    let result0 = compute_component!(
        0, first00, first01, first02, first03, second00, second01, second02, second03, shared00,
        shared01, shared02, shared03
    );
    let result1 = compute_component!(
        1, first10, first11, first12, first13, second10, second11, second12, second13, shared10,
        shared11, shared12, shared13
    );
    let result2 = compute_component!(
        2, first20, first21, first22, first23, second20, second21, second22, second23, shared20,
        shared21, shared22, shared23
    );
    let result3 = compute_component!(
        3, first30, first31, first32, first33, second30, second31, second32, second33, shared30,
        shared31, shared32, shared33
    );
    // The blocked capability assigns four unique output elements to every lane.
    let Some(output_block) = thread_index.checked_block::<64, 4>() else {
        return;
    };
    if let Some(slot) = output.get_block_mut(&output_block, 0) {
        *slot = result0;
    }
    if let Some(slot) = output.get_block_mut(&output_block, 1) {
        *slot = result1;
    }
    if let Some(slot) = output.get_block_mut(&output_block, 2) {
        *slot = result2;
    }
    if let Some(slot) = output.get_block_mut(&output_block, 3) {
        *slot = result3;
    }
}

/// Adds two expert-rank partials in fixed rank order.
#[cfg(all(any(not(target_arch = "amdgpu"), feature = "kernel-combine-expert-ranks"), not(feature = "kernel-combine-expert-ranks-tile")))]
#[cfg_attr(
    not(feature = "ablation-combine-transposed"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-combine-transposed",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
pub fn gfx950_combine_expert_ranks_v1(
    rank0: &[f32],
    rank1: &[f32],
    mut output: DisjointSlice<f32, RowStriped2D<Index1D, 256, 1>>,
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
    let Some(output_row) = index.checked_row_striped_2d::<256, 1>() else {
        return;
    };
    if let Some(slot) = output.get_row_striped_2d_mut(
        &output_row,
        0,
        COMBINE_BATCHES,
        TOKENS * OUTPUT,
        TOKENS * OUTPUT,
    ) {
        *slot = result;
    }
}

/// Commits state only when every speculative token and score is accepted.
#[cfg(all(
    any(
        not(target_arch = "amdgpu"),
        feature = "kernel-speculative-transaction"
    ),
    not(feature = "kernel-speculative-transaction-tile")
))]
#[cfg_attr(
    not(feature = "ablation-speculative-recompute-prefix"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-speculative-recompute-prefix",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn gfx950_speculative_transaction_v1(
    draft_tokens: &[i32],
    target_tokens: &[i32],
    draft_scores: &[f32],
    thresholds: &[f32],
    base_state: &[f32],
    proposed_deltas: &[f32],
    mut accepted_steps: DisjointSlice<u32, RowStriped2D<Index1D, 64, 1>>,
    mut committed: DisjointSlice<u32, RowStriped2D<Index1D, 64, 1>>,
    mut output_state: DisjointSlice<f32>,
) {
    // One Wave64 owns a batch: lanes first evaluate candidates, then own state elements.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let lane = global_index & 63;
    // Establish the full transactional buffer contract before subgroup broadcasts.
    if batch >= SYSTEM_BATCHES
        || draft_tokens.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS
        || target_tokens.len() != SYSTEM_BATCHES * DRAFT_STEPS
        || draft_scores.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS
        || thresholds.len() != SYSTEM_BATCHES * DRAFT_STEPS
        || base_state.len() != SYSTEM_BATCHES * STATE_WIDTH
        || proposed_deltas.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS * STATE_WIDTH
        || accepted_steps.len() != SYSTEM_BATCHES * CANDIDATES
        || committed.len() != SYSTEM_BATCHES * CANDIDATES
        || output_state.len() != SYSTEM_BATCHES * CANDIDATES * STATE_WIDTH
    {
        return;
    }
    // Checked views name candidate/step structure and remove repeated stride arithmetic.
    let transaction_base = batch.wrapping_mul(CANDIDATES).wrapping_mul(DRAFT_STEPS);
    let target_base = batch.wrapping_mul(DRAFT_STEPS);
    let state_base = batch.wrapping_mul(STATE_WIDTH);
    let delta_base = batch
        .wrapping_mul(CANDIDATES)
        .wrapping_mul(DRAFT_STEPS)
        .wrapping_mul(STATE_WIDTH);
    let Ok(target_tokens) = StridedReadView2D::from_shared_slice(
        target_tokens,
        target_base,
        1,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    let Ok(thresholds) =
        StridedReadView2D::from_shared_slice(thresholds, target_base, 1, DRAFT_STEPS, DRAFT_STEPS)
    else {
        return;
    };
    let Ok(draft_tokens) = StridedReadView2D::from_shared_slice(
        draft_tokens,
        transaction_base,
        CANDIDATES,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    let Ok(draft_scores) = StridedReadView2D::from_shared_slice(
        draft_scores,
        transaction_base,
        CANDIDATES,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    #[cfg(feature = "ablation-speculative-recompute-prefix")]
    macro_rules! accepted_prefix {
        ($candidate:expr) => {{
            let accepts0 = (draft_tokens.load_or($candidate, 0, 0)
                == target_tokens.load_or(0, 0, 0))
                & (draft_scores.load_or($candidate, 0, 0.0) >= thresholds.load_or(0, 0, 0.0));
            let accepts1 = accepts0
                & (draft_tokens.load_or($candidate, 1, 0) == target_tokens.load_or(0, 1, 0))
                & (draft_scores.load_or($candidate, 1, 0.0) >= thresholds.load_or(0, 1, 0.0));
            let accepts2 = accepts1
                & (draft_tokens.load_or($candidate, 2, 0) == target_tokens.load_or(0, 2, 0))
                & (draft_scores.load_or($candidate, 2, 0.0) >= thresholds.load_or(0, 2, 0.0));
            let accepts3 = accepts2
                & (draft_tokens.load_or($candidate, 3, 0) == target_tokens.load_or(0, 3, 0))
                & (draft_scores.load_or($candidate, 3, 0.0) >= thresholds.load_or(0, 3, 0.0));
            (accepts0 as usize)
                .wrapping_add(accepts1 as usize)
                .wrapping_add(accepts2 as usize)
                .wrapping_add(accepts3 as usize)
        }};
    }
    // Build a prefix: a later step can be accepted only when every earlier step was.
    let acceptance_candidate = lane & (CANDIDATES - 1);
    let accepts0 = (draft_tokens.load_or(acceptance_candidate, 0, 0)
        == target_tokens.load_or(0, 0, 0))
        & (draft_scores.load_or(acceptance_candidate, 0, 0.0) >= thresholds.load_or(0, 0, 0.0));
    let accepts1 = accepts0
        & (draft_tokens.load_or(acceptance_candidate, 1, 0) == target_tokens.load_or(0, 1, 0))
        & (draft_scores.load_or(acceptance_candidate, 1, 0.0) >= thresholds.load_or(0, 1, 0.0));
    let accepts2 = accepts1
        & (draft_tokens.load_or(acceptance_candidate, 2, 0) == target_tokens.load_or(0, 2, 0))
        & (draft_scores.load_or(acceptance_candidate, 2, 0.0) >= thresholds.load_or(0, 2, 0.0));
    let accepts3 = accepts2
        & (draft_tokens.load_or(acceptance_candidate, 3, 0) == target_tokens.load_or(0, 3, 0))
        & (draft_scores.load_or(acceptance_candidate, 3, 0.0) >= thresholds.load_or(0, 3, 0.0));
    let accepted_local = (accepts0 as usize)
        .wrapping_add(accepts1 as usize)
        .wrapping_add(accepts2 as usize)
        .wrapping_add(accepts3 as usize);
    // Re-map lanes to candidate/state pairs and broadcast one decision per candidate.
    let candidate = lane / STATE_WIDTH;
    let state_element = lane.wrapping_sub(candidate.wrapping_mul(STATE_WIDTH));
    #[cfg(not(feature = "ablation-speculative-recompute-prefix"))]
    let accepted = Gfx950Subgroup::current()
        .broadcast_f32::<64>(accepted_local as f32, candidate as u32 & 63)
        as usize;
    #[cfg(feature = "ablation-speculative-recompute-prefix")]
    let accepted = accepted_prefix!(candidate);
    // Commit status and all state deltas together; rejected candidates retain base state.
    if lane < CANDIDATES {
        let Some(status_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
            return;
        };
        if let Some(slot) = accepted_steps.get_row_striped_2d_mut(
            &status_row,
            0,
            SYSTEM_BATCHES,
            CANDIDATES,
            CANDIDATES,
        ) {
            *slot = accepted_local as u32;
        }
        if let Some(slot) =
            committed.get_row_striped_2d_mut(&status_row, 0, SYSTEM_BATCHES, CANDIDATES, CANDIDATES)
        {
            *slot = if accepted_local == DRAFT_STEPS { 1 } else { 0 };
        }
    }
    let mut value = base_state[state_base.wrapping_add(state_element)];
    if accepted == DRAFT_STEPS {
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(1)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(2)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(3)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
    }
    // The one-dimensional capability gives every lane a unique state destination.
    if let Some(slot) = output_state.get_mut(thread::index_1d()) {
        *slot = value;
    }
}

/// Probes every slot, verifies the full 3-gram, and resolves duplicate keys.
#[cfg(all(
    any(not(target_arch = "amdgpu"), feature = "kernel-qwen-ngram-gather"),
    not(feature = "kernel-qwen-ngram-gather-tile")
))]
#[cfg_attr(
    not(feature = "ablation-ngram-reverse-probe"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-ngram-reverse-probe",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
pub fn gfx950_qwen_ngram_gather_v1(
    queries: &[i32],
    table_hashes: &[u64],
    table_grams: &[i32],
    table_values: &[i32],
    priorities: &[i32],
    mut output: DisjointSlice<i32, RowStriped2D<Index1D, 64, 1>>,
) {
    // Each in-range lane owns one query; no subgroup coordination is required.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let query = global_index & 63;
    if batch >= SYSTEM_BATCHES
        || queries.len() != SYSTEM_BATCHES * QUERIES * NGRAM
        || table_hashes.len() != SYSTEM_BATCHES * TABLE_SIZE
        || table_grams.len() != SYSTEM_BATCHES * TABLE_SIZE * NGRAM
        || table_values.len() != SYSTEM_BATCHES * TABLE_SIZE
        || priorities.len() != SYSTEM_BATCHES * TABLE_SIZE
        || output.len() != SYSTEM_BATCHES * QUERIES
    {
        return;
    }
    if query >= QUERIES {
        return;
    }
    // Hash all three tokens; the full gram is still checked to reject collisions.
    let query_batch_base = batch.wrapping_mul(QUERIES).wrapping_mul(NGRAM);
    let table_batch_base = batch.wrapping_mul(TABLE_SIZE);
    let gram_batch_base = batch.wrapping_mul(TABLE_SIZE).wrapping_mul(NGRAM);
    let base = query_batch_base.wrapping_add(query.wrapping_mul(NGRAM));
    let mut hash = 1_469_598_103_934_665_603_u64;
    hash ^= queries[base] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base.wrapping_add(1)] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base.wrapping_add(2)] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    let mut best_slot = usize::MAX;
    let mut best_priority = i32::MIN;
    let mut best_value = -1_i32;
    // Probe the bounded table completely and resolve duplicate keys deterministically.
    macro_rules! probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let table_slot = table_batch_base.wrapping_add(slot);
            let gram_slot = gram_batch_base.wrapping_add(slot.wrapping_mul(NGRAM));
            let equal = (table_hashes[table_slot] == hash)
                & (table_grams[gram_slot] == queries[base])
                & (table_grams[gram_slot.wrapping_add(1)] == queries[base.wrapping_add(1)])
                & (table_grams[gram_slot.wrapping_add(2)] == queries[base.wrapping_add(2)]);
            if equal {
                let priority = priorities[table_slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_slot = slot;
                    best_priority = priority;
                    best_value = table_values[table_slot];
                }
            }
        }};
    }
    #[cfg(not(feature = "ablation-ngram-reverse-probe"))]
    macro_rules! final_probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let table_slot = table_batch_base.wrapping_add(slot);
            let gram_slot = gram_batch_base.wrapping_add(slot.wrapping_mul(NGRAM));
            let equal = (table_hashes[table_slot] == hash)
                & (table_grams[gram_slot] == queries[base])
                & (table_grams[gram_slot.wrapping_add(1)] == queries[base.wrapping_add(1)])
                & (table_grams[gram_slot.wrapping_add(2)] == queries[base.wrapping_add(2)]);
            if equal {
                let priority = priorities[table_slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_value = table_values[table_slot];
                }
            }
        }};
    }
    #[cfg(not(feature = "ablation-ngram-reverse-probe"))]
    {
        probe!(0);
        probe!(1);
        probe!(2);
        probe!(3);
        probe!(4);
        probe!(5);
        probe!(6);
        probe!(7);
        probe!(8);
        probe!(9);
        probe!(10);
        probe!(11);
        probe!(12);
        probe!(13);
        probe!(14);
        final_probe!(15);
    }
    #[cfg(feature = "ablation-ngram-reverse-probe")]
    {
        probe!(15);
        probe!(14);
        probe!(13);
        probe!(12);
        probe!(11);
        probe!(10);
        probe!(9);
        probe!(8);
        probe!(7);
        probe!(6);
        probe!(5);
        probe!(4);
        probe!(3);
        probe!(2);
        probe!(1);
        probe!(0);
    }
    // Row-striped ownership assigns exactly one result slot to the query lane.
    let Some(output_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
        return;
    };
    if let Some(slot) =
        output.get_row_striped_2d_mut(&output_row, 0, SYSTEM_BATCHES, QUERIES, QUERIES)
    {
        *slot = best_value;
    }
}

/// Copies one gradient shard into deterministic transport staging.
#[cfg(all(
    any(not(target_arch = "amdgpu"), feature = "kernel-stage-gradient-shard"),
    not(feature = "kernel-stage-gradient-shard-tile")
))]
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
    mut output: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
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
    // The production path is a direct coalesced copy; the tile path is an ablation only.
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
    // Row-striped ownership prevents multiple waves from staging the same element.
    let Some(output_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
        return;
    };
    if let Some(slot) =
        output.get_row_striped_2d_mut(&output_row, 0, SYSTEM_BATCHES, MUON_ELEMENTS, MUON_ELEMENTS)
    {
        *slot = value;
    }
}

/// Reduces two shards and computes five Newton-Schulz Muon iterations.
#[cfg(all(
    any(not(target_arch = "amdgpu"), feature = "kernel-muon-update"),
    not(feature = "kernel-muon-update-tile")
))]
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
    mut output: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
    mut output_norm: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
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
        let Some(output_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
            return;
        };
        if let Some(slot) = output.get_row_striped_2d_mut(
            &output_row,
            0,
            SYSTEM_BATCHES,
            MUON_ELEMENTS,
            MUON_ELEMENTS,
        ) {
            *slot = -MUON_LEARNING_RATE * matrix_value;
        }
    }
    if lane == 0 {
        let Some(norm_row) = thread::index_1d().checked_row_striped_2d::<64, 1>() else {
            return;
        };
        if let Some(slot) = output_norm.get_row_striped_2d_mut(&norm_row, 0, SYSTEM_BATCHES, 1, 1) {
            *slot = norm;
        }
    }
}

/// Commits state only when every speculative token and score is accepted.
#[cfg(all(
    target_arch = "amdgpu",
    feature = "kernel-speculative-transaction-tile"
))]
#[cfg_attr(
    not(feature = "ablation-speculative-recompute-prefix"),
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-speculative-recompute-prefix",
    kernel(
        typed,
        launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn gfx950_speculative_transaction_v1(
    draft_tokens: &[i32],
    target_tokens: &[i32],
    draft_scores: &[f32],
    thresholds: &[f32],
    base_state: &[f32],
    proposed_deltas: &[f32],
    mut accepted_steps: DisjointSlice<u32, fe2o3_device::Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut committed: DisjointSlice<u32, fe2o3_device::Tiled2D<Index1D, 64, 1, 64, 1>>,
    mut output_state: DisjointSlice<f32, fe2o3_device::Tiled2D<Index1D, 64, 1, 64, 1>>,
) {
    // One Wave64 owns a batch: lanes first evaluate candidates, then own state elements.
    let global_index = thread::index_1d().get();
    let batch = global_index / 64;
    let lane = global_index & 63;
    // Establish the full transactional buffer contract before subgroup broadcasts.
    if batch >= SYSTEM_BATCHES
        || draft_tokens.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS
        || target_tokens.len() != SYSTEM_BATCHES * DRAFT_STEPS
        || draft_scores.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS
        || thresholds.len() != SYSTEM_BATCHES * DRAFT_STEPS
        || base_state.len() != SYSTEM_BATCHES * STATE_WIDTH
        || proposed_deltas.len() != SYSTEM_BATCHES * CANDIDATES * DRAFT_STEPS * STATE_WIDTH
        || accepted_steps.len() != SYSTEM_BATCHES * CANDIDATES
        || committed.len() != SYSTEM_BATCHES * CANDIDATES
        || output_state.len() != SYSTEM_BATCHES * CANDIDATES * STATE_WIDTH
    {
        return;
    }
    // Checked views name candidate/step structure and remove repeated stride arithmetic.
    let transaction_base = batch.wrapping_mul(CANDIDATES).wrapping_mul(DRAFT_STEPS);
    let target_base = batch.wrapping_mul(DRAFT_STEPS);
    let state_base = batch.wrapping_mul(STATE_WIDTH);
    let delta_base = batch
        .wrapping_mul(CANDIDATES)
        .wrapping_mul(DRAFT_STEPS)
        .wrapping_mul(STATE_WIDTH);
    let Ok(target_tokens) = StridedReadView2D::from_shared_slice(
        target_tokens,
        target_base,
        1,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    let Ok(thresholds) =
        StridedReadView2D::from_shared_slice(thresholds, target_base, 1, DRAFT_STEPS, DRAFT_STEPS)
    else {
        return;
    };
    let Ok(draft_tokens) = StridedReadView2D::from_shared_slice(
        draft_tokens,
        transaction_base,
        CANDIDATES,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    let Ok(draft_scores) = StridedReadView2D::from_shared_slice(
        draft_scores,
        transaction_base,
        CANDIDATES,
        DRAFT_STEPS,
        DRAFT_STEPS,
    ) else {
        return;
    };
    #[cfg(feature = "ablation-speculative-recompute-prefix")]
    macro_rules! accepted_prefix {
        ($candidate:expr) => {{
            let accepts0 = (draft_tokens.load_or($candidate, 0, 0)
                == target_tokens.load_or(0, 0, 0))
                & (draft_scores.load_or($candidate, 0, 0.0) >= thresholds.load_or(0, 0, 0.0));
            let accepts1 = accepts0
                & (draft_tokens.load_or($candidate, 1, 0) == target_tokens.load_or(0, 1, 0))
                & (draft_scores.load_or($candidate, 1, 0.0) >= thresholds.load_or(0, 1, 0.0));
            let accepts2 = accepts1
                & (draft_tokens.load_or($candidate, 2, 0) == target_tokens.load_or(0, 2, 0))
                & (draft_scores.load_or($candidate, 2, 0.0) >= thresholds.load_or(0, 2, 0.0));
            let accepts3 = accepts2
                & (draft_tokens.load_or($candidate, 3, 0) == target_tokens.load_or(0, 3, 0))
                & (draft_scores.load_or($candidate, 3, 0.0) >= thresholds.load_or(0, 3, 0.0));
            (accepts0 as usize)
                .wrapping_add(accepts1 as usize)
                .wrapping_add(accepts2 as usize)
                .wrapping_add(accepts3 as usize)
        }};
    }
    // Build a prefix: a later step can be accepted only when every earlier step was.
    let acceptance_candidate = lane & (CANDIDATES - 1);
    let accepts0 = (draft_tokens.load_or(acceptance_candidate, 0, 0)
        == target_tokens.load_or(0, 0, 0))
        & (draft_scores.load_or(acceptance_candidate, 0, 0.0) >= thresholds.load_or(0, 0, 0.0));
    let accepts1 = accepts0
        & (draft_tokens.load_or(acceptance_candidate, 1, 0) == target_tokens.load_or(0, 1, 0))
        & (draft_scores.load_or(acceptance_candidate, 1, 0.0) >= thresholds.load_or(0, 1, 0.0));
    let accepts2 = accepts1
        & (draft_tokens.load_or(acceptance_candidate, 2, 0) == target_tokens.load_or(0, 2, 0))
        & (draft_scores.load_or(acceptance_candidate, 2, 0.0) >= thresholds.load_or(0, 2, 0.0));
    let accepts3 = accepts2
        & (draft_tokens.load_or(acceptance_candidate, 3, 0) == target_tokens.load_or(0, 3, 0))
        & (draft_scores.load_or(acceptance_candidate, 3, 0.0) >= thresholds.load_or(0, 3, 0.0));
    let accepted_local = (accepts0 as usize)
        .wrapping_add(accepts1 as usize)
        .wrapping_add(accepts2 as usize)
        .wrapping_add(accepts3 as usize);
    // Re-map lanes to candidate/state pairs and broadcast one decision per candidate.
    let candidate = lane / STATE_WIDTH;
    let state_element = lane.wrapping_sub(candidate.wrapping_mul(STATE_WIDTH));
    #[cfg(not(feature = "ablation-speculative-recompute-prefix"))]
    let accepted = Gfx950Subgroup::current()
        .broadcast_f32::<64>(accepted_local as f32, candidate as u32 & 63)
        as usize;
    #[cfg(feature = "ablation-speculative-recompute-prefix")]
    let accepted = accepted_prefix!(candidate);
    // Commit status and all state deltas together; rejected candidates retain base state.
    if lane < CANDIDATES {
        let Some(status_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
            return;
        };
        if let Some(slot) =
            accepted_steps.get_tiled_2d_mut(&status_tile, 0, SYSTEM_BATCHES, CANDIDATES, CANDIDATES)
        {
            *slot = accepted_local as u32;
        }
        if let Some(slot) =
            committed.get_tiled_2d_mut(&status_tile, 0, SYSTEM_BATCHES, CANDIDATES, CANDIDATES)
        {
            *slot = if accepted_local == DRAFT_STEPS { 1 } else { 0 };
        }
    }
    let mut value = base_state[state_base.wrapping_add(state_element)];
    if accepted == DRAFT_STEPS {
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(1)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(2)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
        value += proposed_deltas[delta_base.wrapping_add(
            candidate
                .wrapping_mul(DRAFT_STEPS)
                .wrapping_add(3)
                .wrapping_mul(STATE_WIDTH)
                .wrapping_add(state_element),
        )];
    }
    // A checked Wave64 tile owns the unchanged candidate/state destination.
    let Some(state_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
        return;
    };
    if let Some(slot) = output_state.get_tiled_2d_mut(
        &state_tile,
        0,
        SYSTEM_BATCHES,
        CANDIDATES * STATE_WIDTH,
        CANDIDATES * STATE_WIDTH,
    ) {
        *slot = value;
    }
}
