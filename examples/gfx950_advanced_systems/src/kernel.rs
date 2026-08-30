//! Complete safe Rust kernel source for the bounded systems profiles.

#![allow(missing_docs)] // The kernel macro emits helper modules.
#![cfg_attr(target_arch = "amdgpu", allow(unused_imports))]

use fe2o3_device::{
    Blocked, DeviceMath, DisjointSlice, Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1,
    Gfx950Fp4MfmaAMatrix, Gfx950Fp8MfmaBMatrix, Gfx950Matrix, Gfx950Subgroup, Index1D,
    StridedReadView2D, Wave64, WaveLane, kernel, thread,
};

use crate::{
    ALL_EXPERTS, CANDIDATES, DISPATCH_CAPACITY, DRAFT_STEPS, EXPERTS, GRADIENT_SHARDS, HIDDEN,
    MUON_ELEMENTS, MUON_LEARNING_RATE, NGRAM, OUTPUT, QUERIES, STATE_WIDTH, TABLE_SIZE, TOKENS,
    TOP_K,
};

/// Stable top-2 routing, weights, expert counts, and compact dispatch metadata.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-moe-route"))]
#[kernel(
    typed,
    namespace = "bb933fcd1e3f8124227991b6743de97b6fa108551cc44c617d9450933ad98170",
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [1, 1, 1]),
    control_flow(loop_bounds(128, 32))
)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub fn gfx950_moe_route_fp4_t16_e4_k2_v1(
    activations: &[u8],
    router_weights: &[f32],
    mut top_experts: DisjointSlice<u32>,
    mut top_weights: DisjointSlice<f32>,
    mut expert_counts: DisjointSlice<u32>,
    mut dispatch: DisjointSlice<i32>,
) {
    let lane = thread::index_1d().get();
    if activations.len() != TOKENS * HIDDEN
        || router_weights.len() != EXPERTS * HIDDEN
        || top_experts.len() != TOKENS * TOP_K
        || top_weights.len() != TOKENS * TOP_K
        || expert_counts.len() != EXPERTS
        || dispatch.len() != EXPERTS * DISPATCH_CAPACITY
    {
        return;
    }
    let Ok(router_weights) =
        StridedReadView2D::from_shared_slice(router_weights, 0, EXPERTS, HIDDEN, HIDDEN)
    else {
        return;
    };
    let wave_lane = lane & 63;
    let token = wave_lane & (TOKENS - 1);
    let Ok(activations) =
        StridedReadView2D::from_shared_slice(activations, 0, TOKENS, HIDDEN, HIDDEN)
    else {
        return;
    };
    let mut route_logit0 = 0.0_f32;
    let mut route_logit1 = 0.0_f32;
    let mut route_logit2 = 0.0_f32;
    let mut route_logit3 = 0.0_f32;
    let mut depth = 0_usize;
    while depth < HIDDEN {
        let bits = activations.load_or(token, depth, 0);
        let magnitude = ((0xc864_3210_u32 >> (((bits & 7) as u32) * 4)) & 15) as f32 * 0.5;
        let sign = 1.0 - 2.0 * ((bits >> 3) & 1) as f32;
        let activation = sign * magnitude;
        route_logit0 += activation * router_weights.load_or(0, depth, 0.0);
        route_logit1 += activation * router_weights.load_or(1, depth, 0.0);
        route_logit2 += activation * router_weights.load_or(2, depth, 0.0);
        route_logit3 += activation * router_weights.load_or(3, depth, 0.0);
        depth += 1;
    }
    let precedes12 = (route_logit1 >= route_logit2) as u32;
    let precedes13 = (route_logit1 >= route_logit3) as u32;
    let precedes23 = (route_logit2 >= route_logit3) as u32;
    let rank1 = (route_logit0 >= route_logit1) as u32 + 2 - precedes12 - precedes13;
    let rank2 = (route_logit0 >= route_logit2) as u32 + precedes12 + 1 - precedes23;
    let rank3 = (route_logit0 >= route_logit3) as u32 + precedes13 + precedes23;
    let first_local = ((rank1 == 0) as u32) + 2 * ((rank2 == 0) as u32) + 3 * ((rank3 == 0) as u32);
    let second_local =
        ((rank1 == 1) as u32) + 2 * ((rank2 == 1) as u32) + 3 * ((rank3 == 1) as u32);
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
    let subgroup = Gfx950Subgroup::current();
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
    if lane < TOKENS * TOP_K {
        let choice = lane & (TOP_K - 1);
        let selected = if choice == 0 { top_first } else { top_second };
        let weight = if choice == 0 {
            top_first_weight
        } else {
            top_second_weight
        };
        if let Some(slot) = top_experts.get_mut(thread::index_1d()) {
            *slot = selected;
        }
        if let Some(slot) = top_weights.get_mut(thread::index_1d()) {
            *slot = weight;
        }
    }
    let dispatch_expert = (lane / DISPATCH_CAPACITY) as u32;
    let count_expert = lane as u32;
    let wanted = lane - dispatch_expert as usize * DISPATCH_CAPACITY;
    let mut seen = 0_usize;
    let mut dispatched = -1_i32;
    let mut count = 0_u32;
    let mut record = 0_usize;
    while record < TOKENS * TOP_K {
        let selected = ((packed_routes >> (2 * record)) & 3) as u32;
        let dispatch_matches = (selected == dispatch_expert) as usize;
        let choose = ((dispatch_matches != 0) & (seen == wanted)) as i32;
        dispatched += (record as i32 - dispatched) * choose;
        count += (selected == count_expert) as u32;
        seen += dispatch_matches;
        record += 1;
    }
    if lane < EXPERTS {
        if let Some(slot) = expert_counts.get_mut(thread::index_1d()) {
            *slot = count;
        }
    }
    if let Some(slot) = dispatch.get_mut(thread::index_1d()) {
        *slot = dispatched;
    }
}

/// Computes a routed expert partition and optional shared-expert contribution.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-moe-expert-rank"))]
#[cfg_attr(
    not(feature = "ablation-expert-serial"),
    kernel(
        typed,
        namespace = "dad4ffb4c5c270c853b36fbb21ecc1095dcf33cf74d9585029fdce96e90d38e2",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-expert-serial",
    kernel(
        typed,
        namespace = "6de3151d7e205de375cd16a46b09c84211346b063664d0da16cd9f9b698efe2f",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
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
    let thread_index = thread::index_1d();
    let lane_index = thread_index.get();
    if activations.len() < TOKENS * HIDDEN
        || expert_weights.len() < ALL_EXPERTS * HIDDEN * OUTPUT
        || top_experts.len() < TOKENS * TOP_K
        || top_weights.len() < TOKENS * TOP_K
        || output.len() < TOKENS * OUTPUT
        || first_expert as usize + 1 >= EXPERTS
    {
        return;
    }
    let lane = WaveLane::<Wave64>::current();
    let first_offset = first_expert as usize * HIDDEN * OUTPUT;
    #[cfg(not(feature = "ablation-expert-serial"))]
    let (first_values, second_values, shared_values) = {
        let Ok(activations_view) =
            Gfx950Fp4MfmaAMatrix::row_major(activations, 0, TOKENS, HIDDEN, HIDDEN)
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
            first_offset + HIDDEN * OUTPUT,
            HIDDEN,
            OUTPUT,
            OUTPUT,
        ) else {
            return;
        };
        let second_weights = second_weights_view.load_k128n16(&lane, 0, 0);
        let Ok(shared_weights_view) = Gfx950Fp8MfmaBMatrix::row_major(
            expert_weights,
            (ALL_EXPERTS - 1) * HIDDEN * OUTPUT,
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
            Gfx950Fp4MfmaAMatrix::row_major(activations, 0, TOKENS, HIDDEN, HIDDEN)
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
            first_offset + HIDDEN * OUTPUT,
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
            (ALL_EXPERTS - 1) * HIDDEN * OUTPUT,
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
            let element = lane_index + $output_component * 64;
            let token = element / OUTPUT;
            let column = element - token * OUTPUT;
            let source_lane = (((token / 4) * OUTPUT + column) as u32) & 63;
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
            let element = lane_index + $output_component * 64;
            let token = element / OUTPUT;
            let accumulator_component = token - (token / 4) * 4;
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
            let route_base = token * TOP_K;
            let selected0 = top_experts[route_base];
            let selected1 = top_experts[route_base + 1];
            let gate0 = top_weights[route_base];
            let gate1 = top_weights[route_base + 1];
            let mut result = 0.0_f32;
            if selected0 == first_expert {
                result += gate0 * (first / (1.0 + math.exp_f32(-first)));
            } else if selected0 == first_expert + 1 {
                result += gate0 * (second / (1.0 + math.exp_f32(-second)));
            }
            if selected1 == first_expert {
                result += gate1 * (first / (1.0 + math.exp_f32(-first)));
            } else if selected1 == first_expert + 1 {
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
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-combine-expert-ranks"))]
#[cfg_attr(
    not(feature = "ablation-combine-transposed"),
    kernel(
        typed,
        namespace = "75b93b89a635855d620e2974e64c7ad6299d75329410616cdceaaabe02db89ae",
        launch(required = [256, 1, 1], max = [256, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-combine-transposed",
    kernel(
        typed,
        namespace = "8f3e0270da0acba280e5bf515bd6a4b11b5c0f615947fcf5569bc1feab92f923",
        launch(required = [256, 1, 1], max = [256, 1, 1])
    )
)]
pub fn gfx950_combine_expert_ranks_v1(
    rank0: &[f32],
    rank1: &[f32],
    mut output: DisjointSlice<f32>,
) {
    let index = thread::index_1d();
    let element = index.get();
    if rank0.len() != TOKENS * OUTPUT
        || rank1.len() != TOKENS * OUTPUT
        || output.len() != TOKENS * OUTPUT
    {
        return;
    }
    if element >= TOKENS * OUTPUT {
        return;
    }
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
    if let Some(slot) = output.get_mut(index) {
        *slot = result;
    }
}

/// Commits state only when every speculative token and score is accepted.
#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-speculative-transaction"
))]
#[cfg_attr(
    not(feature = "ablation-speculative-recompute-prefix"),
    kernel(
        typed,
        namespace = "712bf821d681a74855c892c7f02fb02b2c64fe36617092999f673a1531777f8b",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-speculative-recompute-prefix",
    kernel(
        typed,
        namespace = "bdec264337e1f6c31dec20bfe6cabbebb62ad36d413a66e8876279753c46ee26",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
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
    mut accepted_steps: DisjointSlice<u32>,
    mut committed: DisjointSlice<u32>,
    mut output_state: DisjointSlice<f32>,
) {
    let lane = thread::index_1d().get();
    if draft_tokens.len() != CANDIDATES * DRAFT_STEPS
        || target_tokens.len() != DRAFT_STEPS
        || draft_scores.len() != CANDIDATES * DRAFT_STEPS
        || thresholds.len() != DRAFT_STEPS
        || base_state.len() != STATE_WIDTH
        || proposed_deltas.len() != CANDIDATES * DRAFT_STEPS * STATE_WIDTH
        || accepted_steps.len() != CANDIDATES
        || committed.len() != CANDIDATES
        || output_state.len() != CANDIDATES * STATE_WIDTH
    {
        return;
    }
    let Ok(target_tokens) =
        StridedReadView2D::from_shared_slice(target_tokens, 0, 1, DRAFT_STEPS, DRAFT_STEPS)
    else {
        return;
    };
    let Ok(thresholds) =
        StridedReadView2D::from_shared_slice(thresholds, 0, 1, DRAFT_STEPS, DRAFT_STEPS)
    else {
        return;
    };
    let Ok(draft_tokens) =
        StridedReadView2D::from_shared_slice(draft_tokens, 0, CANDIDATES, DRAFT_STEPS, DRAFT_STEPS)
    else {
        return;
    };
    let Ok(draft_scores) =
        StridedReadView2D::from_shared_slice(draft_scores, 0, CANDIDATES, DRAFT_STEPS, DRAFT_STEPS)
    else {
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
            accepts0 as usize + accepts1 as usize + accepts2 as usize + accepts3 as usize
        }};
    }
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
    let accepted_local =
        accepts0 as usize + accepts1 as usize + accepts2 as usize + accepts3 as usize;
    let candidate = lane / STATE_WIDTH;
    let state_element = lane - candidate * STATE_WIDTH;
    #[cfg(not(feature = "ablation-speculative-recompute-prefix"))]
    let accepted = Gfx950Subgroup::current()
        .broadcast_f32::<64>(accepted_local as f32, candidate as u32 & 63)
        as usize;
    #[cfg(feature = "ablation-speculative-recompute-prefix")]
    let accepted = accepted_prefix!(candidate);
    if lane < CANDIDATES {
        if let Some(slot) = accepted_steps.get_mut(thread::index_1d()) {
            *slot = accepted_local as u32;
        }
        if let Some(slot) = committed.get_mut(thread::index_1d()) {
            *slot = if accepted_local == DRAFT_STEPS { 1 } else { 0 };
        }
    }
    let mut value = base_state[state_element];
    if accepted == DRAFT_STEPS {
        value += proposed_deltas[candidate * DRAFT_STEPS * STATE_WIDTH + state_element];
        value += proposed_deltas[(candidate * DRAFT_STEPS + 1) * STATE_WIDTH + state_element];
        value += proposed_deltas[(candidate * DRAFT_STEPS + 2) * STATE_WIDTH + state_element];
        value += proposed_deltas[(candidate * DRAFT_STEPS + 3) * STATE_WIDTH + state_element];
    }
    if let Some(slot) = output_state.get_mut(thread::index_1d()) {
        *slot = value;
    }
}

/// Probes every slot, verifies the full 3-gram, and resolves duplicate keys.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-qwen-ngram-gather"))]
#[cfg_attr(
    not(feature = "ablation-ngram-reverse-probe"),
    kernel(
        typed,
        namespace = "a9bf254981d5af7855538f611e59b2a273ed274201689cd16443b7279c327175",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-ngram-reverse-probe",
    kernel(
        typed,
        namespace = "a62bfd564c731058a0a6b9f3b1b710180c36b311241a8db5e3e4be664e5cf449",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
pub fn gfx950_qwen_ngram_gather_v1(
    queries: &[i32],
    table_hashes: &[u64],
    table_grams: &[i32],
    table_values: &[i32],
    priorities: &[i32],
    mut output: DisjointSlice<i32>,
) {
    let index = thread::index_1d();
    let query = index.get();
    if queries.len() != QUERIES * NGRAM
        || table_hashes.len() != TABLE_SIZE
        || table_grams.len() != TABLE_SIZE * NGRAM
        || table_values.len() != TABLE_SIZE
        || priorities.len() != TABLE_SIZE
        || output.len() != QUERIES
    {
        return;
    }
    if query >= QUERIES {
        return;
    }
    let base = query * NGRAM;
    let mut hash = 1_469_598_103_934_665_603_u64;
    hash ^= queries[base] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base + 1] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    hash ^= queries[base + 2] as u32 as u64;
    hash = hash.wrapping_mul(1_099_511_628_211);
    let mut best_slot = usize::MAX;
    let mut best_priority = i32::MIN;
    let mut best_value = -1_i32;
    macro_rules! probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let equal = (table_hashes[slot] == hash)
                & (table_grams[slot * NGRAM] == queries[base])
                & (table_grams[slot * NGRAM + 1] == queries[base + 1])
                & (table_grams[slot * NGRAM + 2] == queries[base + 2]);
            if equal {
                let priority = priorities[slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_slot = slot;
                    best_priority = priority;
                    best_value = table_values[slot];
                }
            }
        }};
    }
    #[cfg(not(feature = "ablation-ngram-reverse-probe"))]
    macro_rules! final_probe {
        ($probe:literal) => {{
            let slot = hash.wrapping_add($probe) as usize & (TABLE_SIZE - 1);
            let equal = (table_hashes[slot] == hash)
                & (table_grams[slot * NGRAM] == queries[base])
                & (table_grams[slot * NGRAM + 1] == queries[base + 1])
                & (table_grams[slot * NGRAM + 2] == queries[base + 2]);
            if equal {
                let priority = priorities[slot];
                if priority > best_priority || (priority == best_priority && slot < best_slot) {
                    best_value = table_values[slot];
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
    if let Some(slot) = output.get_mut(index) {
        *slot = best_value;
    }
}

/// Copies one gradient shard into deterministic transport staging.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-stage-gradient-shard"))]
#[cfg_attr(
    not(feature = "ablation-stage-tile4"),
    kernel(
        typed,
        namespace = "487472b4b767bb11afc7a2d5bb85795b2b538c040432da4c0d5755900dd4867e",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-stage-tile4",
    kernel(
        typed,
        namespace = "3acc801a14754fb8c218f9aba13cbeb53c41427b68ab12bc651da79d5574f410",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
pub fn gfx950_stage_gradient_shard_v1(input: &[f32], mut output: DisjointSlice<f32>) {
    let index = thread::index_1d();
    let element = index.get();
    if input.len() != MUON_ELEMENTS || output.len() != MUON_ELEMENTS {
        return;
    }
    if element >= MUON_ELEMENTS {
        return;
    }
    #[cfg(not(feature = "ablation-stage-tile4"))]
    let value = input[element];
    #[cfg(feature = "ablation-stage-tile4")]
    let value = {
        let mut tile0 = 0.0_f32;
        let mut tile1 = 0.0_f32;
        let mut tile2 = 0.0_f32;
        let mut tile3 = 0.0_f32;
        if element < 4 {
            let tile_base = element * 4;
            tile0 = input[tile_base];
            tile1 = input[tile_base + 1];
            tile2 = input[tile_base + 2];
            tile3 = input[tile_base + 3];
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
    if let Some(slot) = output.get_mut(index) {
        *slot = value;
    }
}

/// Reduces two shards and computes five Newton-Schulz Muon iterations.
#[cfg(any(not(target_arch = "amdgpu"), feature = "kernel-muon-update"))]
#[cfg_attr(
    not(feature = "ablation-muon-broadcast16"),
    kernel(
        typed,
        namespace = "9640ccf630920dc28c840f4d796dab11ddd9cebf804b0315b877e0c048eb7829",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
#[cfg_attr(
    feature = "ablation-muon-broadcast16",
    kernel(
        typed,
        namespace = "6de7921154ed6a9f640c8cb2ca93cfc312b36de60ead0f02bf1157c1765ee2a9",
        launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
    )
)]
pub fn gfx950_muon_update_4x4_v1(
    shards: &[f32],
    mut output: DisjointSlice<f32>,
    mut output_norm: DisjointSlice<f32>,
) {
    let lane = thread::index_1d().get();
    if shards.len() != GRADIENT_SHARDS * MUON_ELEMENTS
        || output.len() != MUON_ELEMENTS
        || output_norm.len() != 1
    {
        return;
    }
    let Ok(shards) = StridedReadView2D::from_shared_slice(
        shards,
        0,
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
    let column = matrix_element - row * 4;
    macro_rules! muon_iteration {
        () => {{
            let mut gram = 0.0_f32;
            gram += subgroup.broadcast_f32::<64>(matrix_value, (row * 4) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (column * 4) as u32 & 63);
            gram += subgroup.broadcast_f32::<64>(matrix_value, (row * 4 + 1) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (column * 4 + 1) as u32 & 63);
            gram += subgroup.broadcast_f32::<64>(matrix_value, (row * 4 + 2) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (column * 4 + 2) as u32 & 63);
            gram += subgroup.broadcast_f32::<64>(matrix_value, (row * 4 + 3) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (column * 4 + 3) as u32 & 63);
            let mut cubic = 0.0_f32;
            cubic += subgroup.broadcast_f32::<64>(gram, (row * 4) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, column as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, (row * 4 + 1) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (4 + column) as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, (row * 4 + 2) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (8 + column) as u32 & 63);
            cubic += subgroup.broadcast_f32::<64>(gram, (row * 4 + 3) as u32 & 63)
                * subgroup.broadcast_f32::<64>(matrix_value, (12 + column) as u32 & 63);
            matrix_value = 1.5 * matrix_value - 0.5 * cubic;
        }};
    }
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    muon_iteration!();
    if lane < MUON_ELEMENTS {
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = -MUON_LEARNING_RATE * matrix_value;
        }
    }
    if lane == 0 {
        if let Some(slot) = output_norm.get_mut(thread::index_1d()) {
            *slot = norm;
        }
    }
}
