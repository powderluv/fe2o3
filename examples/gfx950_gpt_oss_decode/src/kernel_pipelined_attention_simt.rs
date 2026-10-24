//! Ordinary SIMT BF16 attention over a real two-stage raw-u64 workgroup LDS pipeline.
//!
//! Historical versions were rejected for a retained scalar temporary and for
//! workgroup convergence. This source retains those checks and the pipeline
//! lifecycle; current native and numerical qualification is recorded in README.

#![allow(missing_docs)]

use fe2o3_device::{
    Blocked, DisjointSlice, Gfx950Subgroup, Index1D, KernelError, KernelResult, Math,
    StridedReadView2D, WorkgroupLdsScope, WorkgroupPipeline, kernel, thread,
};

use crate::{
    ATTENTION_OUTPUT_ELEMENTS, CONTEXT_TOKENS, EXPERT_K_TILE, EXPERT_N_TILE,
    EXPERT_OUTPUT_ELEMENTS, EXPERTS, HEAD_DIM, HIDDEN_SIZE, MATRIX_ROWS, MXFP4_BLOCKS, VALUE_TILE,
};

const ATTENTION_SCALE: f32 = 0.125;
// Empty slots are identified by u32::MAX, never by a numerical score floor.
const ROUTER_FLOOR: f32 = -1.0e30;

/// Experiments with double-buffered LDS Q/K fragments; qualification is tracked in README.
#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-gpt-oss-decode-pipelined-attention"
))]
#[kernel(
    typed,
    launch(
        required = [256, 1, 1],
        max = [256, 1, 1],
        max_grid = [4, 1, 1],
        static_shared_memory_bytes = 8192
    ),
    control_flow(loop_bounds(2880, 64, 4, 4, 4, 16, 4, 128))
)]
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
pub fn gfx950_gpt_oss_120b_decode_megakernel_v1(
    hidden_f32: &[f32],
    router_f32: &[f32],
    query_bf16: &[u16],
    key_transposed_bf16: &[u16],
    value_f32: &[f32],
    sinks_f32: &[f32],
    expert_activation_blocks_fp4: &[u8],
    expert_weight_blocks_fp4: &[u8],
    activation_scales: &[f32],
    expert_weight_scales: &[f32],
    mut attention_output: DisjointSlice<f32, Blocked<Index1D, 64, 4>>,
    mut expert_output: DisjointSlice<f32, Blocked<Index1D, 64, 4>>,
    mut packed_top4: DisjointSlice<u32>,
) -> KernelResult {
    // Validate the entire item contract before any collective or pipeline operation.
    if hidden_f32.len() < crate::PROFILE_ITEMS * HIDDEN_SIZE
        || router_f32.len() < EXPERTS * HIDDEN_SIZE
        || query_bf16.len() < crate::PROFILE_ITEMS * MATRIX_ROWS * HEAD_DIM
        || key_transposed_bf16.len() < crate::PROFILE_ITEMS * HEAD_DIM * CONTEXT_TOKENS
        || value_f32.len() < crate::PROFILE_ITEMS * CONTEXT_TOKENS * VALUE_TILE
        || sinks_f32.len() < crate::PROFILE_ITEMS * MATRIX_ROWS
        || expert_activation_blocks_fp4.len()
            < crate::PROFILE_ITEMS * MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE
        || expert_weight_blocks_fp4.len() < EXPERTS * MXFP4_BLOCKS * EXPERT_K_TILE * EXPERT_N_TILE
        || activation_scales.len() < crate::PROFILE_ITEMS * MXFP4_BLOCKS
        || expert_weight_scales.len() < EXPERTS * MXFP4_BLOCKS * EXPERT_N_TILE
        || attention_output.len() < ATTENTION_OUTPUT_ELEMENTS
        || expert_output.len() < EXPERT_OUTPUT_ELEMENTS
        || packed_top4.len() < crate::PACKED_ROUTE_ELEMENTS
    {
        return Err(KernelError::InvalidArgument);
    }

    // One Wave64 owns an item; all four workgroup waves also address the LDS pipeline.
    let index = thread::index_1d();
    let global_index = index.get();
    let lane_index = global_index % crate::WAVE_SIZE;
    let item_index = global_index / crate::WAVE_SIZE;
    let pipeline_lane = global_index % crate::WORKGROUP_SIZE;
    let subgroup = Gfx950Subgroup::current();

    // Checked views establish offsets and strides before the experimental phase.
    let Ok(hidden) = StridedReadView2D::from_shared_slice(
        hidden_f32,
        0,
        crate::PROFILE_ITEMS,
        HIDDEN_SIZE,
        HIDDEN_SIZE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(router) =
        StridedReadView2D::from_shared_slice(router_f32, 0, EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let local_expert0 = lane_index.wrapping_mul(2);
    let local_expert1 = local_expert0.wrapping_add(1);
    let mut local_logit0 = 0.0_f32;
    let mut local_logit1 = 0.0_f32;
    let mut depth = 0_usize;
    while depth < HIDDEN_SIZE {
        let activation = hidden.load_or(item_index, depth, 0.0);
        local_logit0 += activation * router.load_or(local_expert0, depth, 0.0);
        local_logit1 += activation * router.load_or(local_expert1, depth, 0.0);
        depth += 1;
    }

    // Router broadcasts remain uniform and identical to the production kernel.
    let mut best0 = ROUTER_FLOOR;
    let mut best1 = ROUTER_FLOOR;
    let mut best2 = ROUTER_FLOOR;
    let mut best3 = ROUTER_FLOOR;
    let mut id0 = u32::MAX;
    let mut id1 = u32::MAX;
    let mut id2 = u32::MAX;
    let mut id3 = u32::MAX;
    let mut source = 0_u32;
    while source < 64 {
        {
            let mut candidate_score = subgroup.broadcast_f32::<64>(local_logit0, source & 63);
            let mut candidate_id = source.wrapping_mul(2);
            let take = ((candidate_id != u32::MAX)
                & ((id0 == u32::MAX)
                    | (candidate_score > best0)
                    | ((candidate_score == best0) & (candidate_id < id0))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best0;
            let old_id = id0;
            best0 = candidate_score * choose + old_score * keep;
            id0 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id1 == u32::MAX)
                    | (candidate_score > best1)
                    | ((candidate_score == best1) & (candidate_id < id1))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best1;
            let old_id = id1;
            best1 = candidate_score * choose + old_score * keep;
            id1 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id2 == u32::MAX)
                    | (candidate_score > best2)
                    | ((candidate_score == best2) & (candidate_id < id2))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best2;
            let old_id = id2;
            best2 = candidate_score * choose + old_score * keep;
            id2 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id3 == u32::MAX)
                    | (candidate_score > best3)
                    | ((candidate_score == best3) & (candidate_id < id3))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best3;
            let old_id = id3;
            best3 = candidate_score * choose + old_score * keep;
            id3 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
        }
        {
            let mut candidate_score = subgroup.broadcast_f32::<64>(local_logit1, source & 63);
            let mut candidate_id = source.wrapping_mul(2).wrapping_add(1);
            let take = ((candidate_id != u32::MAX)
                & ((id0 == u32::MAX)
                    | (candidate_score > best0)
                    | ((candidate_score == best0) & (candidate_id < id0))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best0;
            let old_id = id0;
            best0 = candidate_score * choose + old_score * keep;
            id0 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id1 == u32::MAX)
                    | (candidate_score > best1)
                    | ((candidate_score == best1) & (candidate_id < id1))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best1;
            let old_id = id1;
            best1 = candidate_score * choose + old_score * keep;
            id1 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id2 == u32::MAX)
                    | (candidate_score > best2)
                    | ((candidate_score == best2) & (candidate_id < id2))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best2;
            let old_id = id2;
            best2 = candidate_score * choose + old_score * keep;
            id2 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id
                .wrapping_mul(take)
                .wrapping_add(candidate_id.wrapping_mul(take ^ 1));
            let take = ((candidate_id != u32::MAX)
                & ((id3 == u32::MAX)
                    | (candidate_score > best3)
                    | ((candidate_score == best3) & (candidate_id < id3))))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best3;
            let old_id = id3;
            best3 = candidate_score * choose + old_score * keep;
            id3 = candidate_id
                .wrapping_mul(take)
                .wrapping_add(old_id.wrapping_mul(take ^ 1));
        }
        source += 1;
    }
    let selected = (id0 as usize) & (EXPERTS - 1);

    // Each lane stages the same four raw BF16 values as the original MFMA fragment.
    // Whole-profile views keep every fallible constructor workgroup-uniform.
    let Ok(query) = StridedReadView2D::from_shared_slice(
        query_bf16,
        0,
        crate::PROFILE_ITEMS * MATRIX_ROWS,
        HEAD_DIM,
        HEAD_DIM,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(key) = StridedReadView2D::from_shared_slice(
        key_transposed_bf16,
        0,
        crate::PROFILE_ITEMS * HEAD_DIM,
        CONTEXT_TOKENS,
        CONTEXT_TOKENS,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let mut pipeline_scope = WorkgroupLdsScope::current();
    let mut query_pipeline = WorkgroupPipeline::<u64, 2, 256, 1>::current(&mut pipeline_scope);
    let mut key_pipeline = WorkgroupPipeline::<u64, 2, 256, 1>::current(&mut pipeline_scope);
    let query_item_row = item_index.wrapping_mul(MATRIX_ROWS);
    let key_item_row = item_index.wrapping_mul(HEAD_DIM);
    let fragment_row = lane_index % 16;
    let fragment_depth = (lane_index / 16).wrapping_mul(4);
    let query_row = query_item_row.wrapping_add(fragment_row);
    let initial_q = (query.load_or(query_row, fragment_depth, 0) as u64)
        | ((query.load_or(query_row, fragment_depth.wrapping_add(1), 0) as u64) << 16)
        | ((query.load_or(query_row, fragment_depth.wrapping_add(2), 0) as u64) << 32)
        | ((query.load_or(query_row, fragment_depth.wrapping_add(3), 0) as u64) << 48);
    let initial_k = (key.load_or(key_item_row.wrapping_add(fragment_depth), fragment_row, 0)
        as u64)
        | ((key.load_or(
            key_item_row.wrapping_add(fragment_depth).wrapping_add(1),
            fragment_row,
            0,
        ) as u64)
            << 16)
        | ((key.load_or(
            key_item_row.wrapping_add(fragment_depth).wrapping_add(2),
            fragment_row,
            0,
        ) as u64)
            << 32)
        | ((key.load_or(
            key_item_row.wrapping_add(fragment_depth).wrapping_add(3),
            fragment_row,
            0,
        ) as u64)
            << 48);
    query_pipeline.stage(0);
    query_pipeline.write(0, pipeline_lane, initial_q);
    query_pipeline.commit(0);
    key_pipeline.stage(0);
    key_pipeline.write(0, pipeline_lane, initial_k);
    key_pipeline.commit(0);
    let token_index = lane_index % CONTEXT_TOKENS;
    let output_row0 = (lane_index / CONTEXT_TOKENS).wrapping_mul(4);
    let output_row1 = output_row0.wrapping_add(1);
    let output_row2 = output_row0.wrapping_add(2);
    let output_row3 = output_row0.wrapping_add(3);
    let pipeline_wave_base = (pipeline_lane / crate::WAVE_SIZE).wrapping_mul(crate::WAVE_SIZE);
    let mut score0 = 0.0_f32;
    let mut score1 = 0.0_f32;
    let mut score2 = 0.0_f32;
    let mut score3 = 0.0_f32;
    let mut phase_index = 0_usize;
    while phase_index < 4 {
        let future_epoch = phase_index + 1;
        let next_phase = future_epoch * 16;
        let next_depth = next_phase.wrapping_add(fragment_depth);
        let next_q = (query.load_or(query_row, next_depth, 0) as u64)
            | ((query.load_or(query_row, next_depth.wrapping_add(1), 0) as u64) << 16)
            | ((query.load_or(query_row, next_depth.wrapping_add(2), 0) as u64) << 32)
            | ((query.load_or(query_row, next_depth.wrapping_add(3), 0) as u64) << 48);
        // Epoch four must not read the following item's K; all four scalars zero-fill.
        let next_key_row = if future_epoch < 4 {
            key_item_row.wrapping_add(next_phase)
        } else {
            crate::PROFILE_ITEMS * HEAD_DIM
        };
        let next_k = (key.load_or(next_key_row.wrapping_add(fragment_depth), fragment_row, 0)
            as u64)
            | ((key.load_or(
                next_key_row.wrapping_add(fragment_depth).wrapping_add(1),
                fragment_row,
                0,
            ) as u64)
                << 16)
            | ((key.load_or(
                next_key_row.wrapping_add(fragment_depth).wrapping_add(2),
                fragment_row,
                0,
            ) as u64)
                << 32)
            | ((key.load_or(
                next_key_row.wrapping_add(fragment_depth).wrapping_add(3),
                fragment_row,
                0,
            ) as u64)
                << 48);
        query_pipeline.stage(future_epoch);
        query_pipeline.write(future_epoch, pipeline_lane, next_q);
        query_pipeline.commit(future_epoch);
        key_pipeline.stage(future_epoch);
        key_pipeline.write(future_epoch, pipeline_lane, next_k);
        key_pipeline.commit(future_epoch);
        query_pipeline.wait(phase_index);
        query_pipeline.consume(phase_index);
        key_pipeline.wait(phase_index);
        key_pipeline.consume(phase_index);
        // LDS gathers reconstruct each lane's four score rows, never another wave's item.
        let mut depth_group = 0_usize;
        while depth_group < 4 {
            let read_base = pipeline_wave_base.wrapping_add(depth_group.wrapping_mul(16));
            let query_index0 = read_base.wrapping_add(output_row0);
            let query0 = if query_index0 < crate::WORKGROUP_SIZE {
                query_pipeline.read(phase_index, query_index0)
            } else {
                0_u64
            };
            let query_index1 = read_base.wrapping_add(output_row1);
            let query1 = if query_index1 < crate::WORKGROUP_SIZE {
                query_pipeline.read(phase_index, query_index1)
            } else {
                0_u64
            };
            let query_index2 = read_base.wrapping_add(output_row2);
            let query2 = if query_index2 < crate::WORKGROUP_SIZE {
                query_pipeline.read(phase_index, query_index2)
            } else {
                0_u64
            };
            let query_index3 = read_base.wrapping_add(output_row3);
            let query3 = if query_index3 < crate::WORKGROUP_SIZE {
                query_pipeline.read(phase_index, query_index3)
            } else {
                0_u64
            };
            let key_index = read_base.wrapping_add(token_index);
            let key_word = if key_index < crate::WORKGROUP_SIZE {
                key_pipeline.read(phase_index, key_index)
            } else {
                0_u64
            };
            let mut component = 0_usize;
            while component < 4 {
                let shift = component * 16;
                let key_value = f32::from_bits((((key_word >> shift) as u32) & 0xffff) << 16);
                score0 += f32::from_bits((((query0 >> shift) as u32) & 0xffff) << 16) * key_value;
                score1 += f32::from_bits((((query1 >> shift) as u32) & 0xffff) << 16) * key_value;
                score2 += f32::from_bits((((query2 >> shift) as u32) & 0xffff) << 16) * key_value;
                score3 += f32::from_bits((((query3 >> shift) as u32) & 0xffff) << 16) * key_value;
                component += 1;
            }
            depth_group += 1;
        }
        query_pipeline.release(phase_index);
        key_pipeline.release(phase_index);
        phase_index += 1;
    }
    query_pipeline.wait(phase_index);
    query_pipeline.discard(phase_index);
    query_pipeline.release(phase_index);
    key_pipeline.wait(phase_index);
    key_pipeline.discard(phase_index);
    key_pipeline.release(phase_index);

    let Ok(values) = StridedReadView2D::from_shared_slice(
        value_f32,
        item_index.wrapping_mul(CONTEXT_TOKENS * VALUE_TILE),
        CONTEXT_TOKENS,
        VALUE_TILE,
        VALUE_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(sinks) = StridedReadView2D::from_shared_slice(
        sinks_f32,
        item_index.wrapping_mul(MATRIX_ROWS),
        1,
        MATRIX_ROWS,
        MATRIX_ROWS,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let row_group = lane_index / CONTEXT_TOKENS;
    let row0 = row_group.wrapping_mul(4);
    let row1 = row0.wrapping_add(1);
    let row2 = row0.wrapping_add(2);
    let row3 = row0.wrapping_add(3);
    let sink0 = sinks.load_or(0, row0, 0.0);
    let sink1 = sinks.load_or(0, row1, 0.0);
    let sink2 = sinks.load_or(0, row2, 0.0);
    let sink3 = sinks.load_or(0, row3, 0.0);
    let reduced0 = subgroup.reduce_max_f32::<16>(score0 * ATTENTION_SCALE);
    let reduced1 = subgroup.reduce_max_f32::<16>(score1 * ATTENTION_SCALE);
    let reduced2 = subgroup.reduce_max_f32::<16>(score2 * ATTENTION_SCALE);
    let reduced3 = subgroup.reduce_max_f32::<16>(score3 * ATTENTION_SCALE);
    let choose0 = (reduced0 > sink0) as u32 as f32;
    let choose1 = (reduced1 > sink1) as u32 as f32;
    let choose2 = (reduced2 > sink2) as u32 as f32;
    let choose3 = (reduced3 > sink3) as u32 as f32;
    let max0 = reduced0 * choose0 + sink0 * (1.0 - choose0);
    let max1 = reduced1 * choose1 + sink1 * (1.0 - choose1);
    let max2 = reduced2 * choose2 + sink2 * (1.0 - choose2);
    let max3 = reduced3 * choose3 + sink3 * (1.0 - choose3);
    let math = Math::current();
    let probability0 = math.exp_f32(score0 * ATTENTION_SCALE - max0);
    let probability1 = math.exp_f32(score1 * ATTENTION_SCALE - max1);
    let probability2 = math.exp_f32(score2 * ATTENTION_SCALE - max2);
    let probability3 = math.exp_f32(score3 * ATTENTION_SCALE - max3);
    let denominator0 = subgroup.reduce_sum_f32::<16>(probability0) + math.exp_f32(sink0 - max0);
    let denominator1 = subgroup.reduce_sum_f32::<16>(probability1) + math.exp_f32(sink1 - max1);
    let denominator2 = subgroup.reduce_sum_f32::<16>(probability2) + math.exp_f32(sink2 - max2);
    let denominator3 = subgroup.reduce_sum_f32::<16>(probability3) + math.exp_f32(sink3 - max3);
    let probability0 = probability0 / denominator0;
    let probability1 = probability1 / denominator1;
    let probability2 = probability2 / denominator2;
    let probability3 = probability3 / denominator3;
    let column = lane_index % VALUE_TILE;
    let mut attention0 = 0.0_f32;
    let mut attention1 = 0.0_f32;
    let mut attention2 = 0.0_f32;
    let mut attention3 = 0.0_f32;
    let mut token = 0_usize;
    while token < CONTEXT_TOKENS {
        let value = values.load_or(token, column, 0.0);
        attention0 += subgroup.broadcast_f32::<16>(probability0, (token as u32) & 15) * value;
        attention1 += subgroup.broadcast_f32::<16>(probability1, (token as u32) & 15) * value;
        attention2 += subgroup.broadcast_f32::<16>(probability2, (token as u32) & 15) * value;
        attention3 += subgroup.broadcast_f32::<16>(probability3, (token as u32) & 15) * value;
        token += 1;
    }

    // The selected expert uses four K=128 MXFP4 blocks with per-block product scales.
    // All four real scale-separated E2M1 blocks are evaluated without matrix instructions.
    let Ok(activations) = StridedReadView2D::from_shared_slice(
        expert_activation_blocks_fp4,
        item_index.wrapping_mul(MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE),
        MXFP4_BLOCKS * MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(weights) = StridedReadView2D::from_shared_slice(
        expert_weight_blocks_fp4,
        0,
        EXPERTS * MXFP4_BLOCKS * EXPERT_K_TILE,
        EXPERT_N_TILE,
        EXPERT_N_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(activation_scale) = StridedReadView2D::from_shared_slice(
        activation_scales,
        item_index.wrapping_mul(MXFP4_BLOCKS),
        1,
        MXFP4_BLOCKS,
        MXFP4_BLOCKS,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(weight_scale) = StridedReadView2D::from_shared_slice(
        expert_weight_scales,
        0,
        EXPERTS * MXFP4_BLOCKS,
        EXPERT_N_TILE,
        EXPERT_N_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let mut expert_acc0 = 0.0_f32;
    let mut expert_acc1 = 0.0_f32;
    let mut expert_acc2 = 0.0_f32;
    let mut expert_acc3 = 0.0_f32;
    let mut block = 0_usize;
    while block < MXFP4_BLOCKS {
        let mut dot0 = 0.0_f32;
        let mut dot1 = 0.0_f32;
        let mut dot2 = 0.0_f32;
        let mut dot3 = 0.0_f32;
        let mut expert_depth = 0_usize;
        while expert_depth < EXPERT_K_TILE {
            let weight = {
                let bits = weights.load_or(
                    selected
                        .wrapping_mul(MXFP4_BLOCKS)
                        .wrapping_add(block)
                        .wrapping_mul(EXPERT_K_TILE)
                        .wrapping_add(expert_depth),
                    column,
                    0,
                );
                let exponent = (bits >> 1_u8) & 3;
                let mantissa = (bits & 1) as f32;
                let magnitude = if exponent == 0 {
                    mantissa * 0.5
                } else if exponent == 1 {
                    1.0 + mantissa * 0.5
                } else if exponent == 2 {
                    2.0 + mantissa
                } else {
                    4.0 + 2.0 * mantissa
                };
                if bits & 8 == 0 { magnitude } else { -magnitude }
            };
            let a0 = {
                let bits = activations.load_or(
                    block.wrapping_mul(MATRIX_ROWS).wrapping_add(row0),
                    expert_depth,
                    0,
                );
                let exponent = (bits >> 1_u8) & 3;
                let mantissa = (bits & 1) as f32;
                let magnitude = if exponent == 0 {
                    mantissa * 0.5
                } else if exponent == 1 {
                    1.0 + mantissa * 0.5
                } else if exponent == 2 {
                    2.0 + mantissa
                } else {
                    4.0 + 2.0 * mantissa
                };
                if bits & 8 == 0 { magnitude } else { -magnitude }
            };
            let a1 = {
                let bits = activations.load_or(
                    block.wrapping_mul(MATRIX_ROWS).wrapping_add(row1),
                    expert_depth,
                    0,
                );
                let exponent = (bits >> 1_u8) & 3;
                let mantissa = (bits & 1) as f32;
                let magnitude = if exponent == 0 {
                    mantissa * 0.5
                } else if exponent == 1 {
                    1.0 + mantissa * 0.5
                } else if exponent == 2 {
                    2.0 + mantissa
                } else {
                    4.0 + 2.0 * mantissa
                };
                if bits & 8 == 0 { magnitude } else { -magnitude }
            };
            let a2 = {
                let bits = activations.load_or(
                    block.wrapping_mul(MATRIX_ROWS).wrapping_add(row2),
                    expert_depth,
                    0,
                );
                let exponent = (bits >> 1_u8) & 3;
                let mantissa = (bits & 1) as f32;
                let magnitude = if exponent == 0 {
                    mantissa * 0.5
                } else if exponent == 1 {
                    1.0 + mantissa * 0.5
                } else if exponent == 2 {
                    2.0 + mantissa
                } else {
                    4.0 + 2.0 * mantissa
                };
                if bits & 8 == 0 { magnitude } else { -magnitude }
            };
            let a3 = {
                let bits = activations.load_or(
                    block.wrapping_mul(MATRIX_ROWS).wrapping_add(row3),
                    expert_depth,
                    0,
                );
                let exponent = (bits >> 1_u8) & 3;
                let mantissa = (bits & 1) as f32;
                let magnitude = if exponent == 0 {
                    mantissa * 0.5
                } else if exponent == 1 {
                    1.0 + mantissa * 0.5
                } else if exponent == 2 {
                    2.0 + mantissa
                } else {
                    4.0 + 2.0 * mantissa
                };
                if bits & 8 == 0 { magnitude } else { -magnitude }
            };
            dot0 += a0 * weight;
            dot1 += a1 * weight;
            dot2 += a2 * weight;
            dot3 += a3 * weight;
            expert_depth += 1;
        }
        let scale = activation_scale.load_or(0, block, 0.0)
            * weight_scale.load_or(
                selected.wrapping_mul(MXFP4_BLOCKS).wrapping_add(block),
                column,
                0.0,
            );
        expert_acc0 += dot0 * scale;
        expert_acc1 += dot1 * scale;
        expert_acc2 += dot2 * scale;
        expert_acc3 += dot3 * scale;
        block += 1;
    }

    // Original blocked publication order remains unchanged.
    let Some(output_block) = index.checked_block::<64, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    if let Some(slot) = attention_output.get_block_mut(&output_block, 0) {
        *slot = attention0;
    }
    if let Some(slot) = attention_output.get_block_mut(&output_block, 1) {
        *slot = attention1;
    }
    if let Some(slot) = attention_output.get_block_mut(&output_block, 2) {
        *slot = attention2;
    }
    if let Some(slot) = attention_output.get_block_mut(&output_block, 3) {
        *slot = attention3;
    }
    if let Some(slot) = expert_output.get_block_mut(&output_block, 0) {
        *slot = expert_acc0;
    }
    if let Some(slot) = expert_output.get_block_mut(&output_block, 1) {
        *slot = expert_acc1;
    }
    if let Some(slot) = expert_output.get_block_mut(&output_block, 2) {
        *slot = expert_acc2;
    }
    if let Some(slot) = expert_output.get_block_mut(&output_block, 3) {
        *slot = expert_acc3;
    }
    let packed = id0 | (id1 << 7) | (id2 << 14) | (id3 << 21);
    if let Some(slot) = packed_top4.get_mut(thread::index_1d()) {
        *slot = packed;
    }
    Ok(())
}
