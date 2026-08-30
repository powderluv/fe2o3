//! Exact-semantics materialized GPT-OSS-120B layer-tile component kernels.

#![allow(missing_docs)]

use fe2o3_device::{
    Bf16MfmaAMatrix, Bf16MfmaBMatrix, Blocked, DeviceMatrix, DisjointSlice, F32AccumulatorFragment,
    Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1, Gfx950Fp4MfmaAMatrix, Gfx950Fp4MfmaBMatrix,
    Gfx950Matrix, Gfx950Subgroup, Index1D, KernelError, KernelResult, Math, StridedReadView2D,
    Wave64, WaveLane, kernel, thread,
};

use crate::{
    ATTENTION_OUTPUT_ELEMENTS, CONTEXT_TOKENS, EXPERT_K_TILE, EXPERT_N_TILE,
    EXPERT_OUTPUT_ELEMENTS, EXPERTS, HEAD_DIM, HIDDEN_SIZE, MATRIX_ROWS, MXFP4_BLOCKS, VALUE_TILE,
};

const ATTENTION_SCALE: f32 = 0.125;
const ROUTER_FLOOR: f32 = -1.0e30;

#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-gpt-oss-router-component"
))]
#[kernel(
    typed,
    namespace = "8ea858c2d33346948501917b2a613d3e4844622372e15b43fb2e2d092e3f336d",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1]),
    control_flow(loop_bounds(2880, 64))
)]
pub fn gfx950_gpt_oss_120b_router_v1(
    hidden_f32: &[f32],
    router_f32: &[f32],
    mut packed_top4: DisjointSlice<u32>,
) -> KernelResult {
    if hidden_f32.len() < HIDDEN_SIZE
        || router_f32.len() < EXPERTS * HIDDEN_SIZE
        || packed_top4.len() < 1
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let lane_index = index.get();
    let subgroup = Gfx950Subgroup::current();
    let Ok(hidden) =
        StridedReadView2D::from_shared_slice(hidden_f32, 0, 1, HIDDEN_SIZE, HIDDEN_SIZE)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(router) =
        StridedReadView2D::from_shared_slice(router_f32, 0, EXPERTS, HIDDEN_SIZE, HIDDEN_SIZE)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let local_expert0 = lane_index * 2;
    let local_expert1 = local_expert0 + 1;
    let mut local_logit0 = 0.0_f32;
    let mut local_logit1 = 0.0_f32;
    let mut depth = 0_usize;
    while depth < HIDDEN_SIZE {
        let activation = hidden.load_or(0, depth, 0.0);
        local_logit0 += activation * router.load_or(local_expert0, depth, 0.0);
        local_logit1 += activation * router.load_or(local_expert1, depth, 0.0);
        depth += 1;
    }

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
            let mut candidate_score = subgroup.broadcast_f32::<64>(local_logit0, source);
            let mut candidate_id = source * 2;
            let take = ((candidate_score > best0)
                | ((candidate_score == best0) & (candidate_id < id0)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best0;
            let old_id = id0;
            best0 = candidate_score * choose + old_score * keep;
            id0 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best1)
                | ((candidate_score == best1) & (candidate_id < id1)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best1;
            let old_id = id1;
            best1 = candidate_score * choose + old_score * keep;
            id1 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best2)
                | ((candidate_score == best2) & (candidate_id < id2)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best2;
            let old_id = id2;
            best2 = candidate_score * choose + old_score * keep;
            id2 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best3)
                | ((candidate_score == best3) & (candidate_id < id3)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best3;
            let old_id = id3;
            best3 = candidate_score * choose + old_score * keep;
            id3 = candidate_id * take + old_id * (1 - take);
        }
        {
            let mut candidate_score = subgroup.broadcast_f32::<64>(local_logit1, source);
            let mut candidate_id = source * 2 + 1;
            let take = ((candidate_score > best0)
                | ((candidate_score == best0) & (candidate_id < id0)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best0;
            let old_id = id0;
            best0 = candidate_score * choose + old_score * keep;
            id0 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best1)
                | ((candidate_score == best1) & (candidate_id < id1)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best1;
            let old_id = id1;
            best1 = candidate_score * choose + old_score * keep;
            id1 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best2)
                | ((candidate_score == best2) & (candidate_id < id2)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best2;
            let old_id = id2;
            best2 = candidate_score * choose + old_score * keep;
            id2 = candidate_id * take + old_id * (1 - take);
            candidate_score = old_score * choose + candidate_score * keep;
            candidate_id = old_id * take + candidate_id * (1 - take);
            let take = ((candidate_score > best3)
                | ((candidate_score == best3) & (candidate_id < id3)))
                as u32;
            let choose = take as f32;
            let keep = 1.0 - choose;
            let old_score = best3;
            let old_id = id3;
            best3 = candidate_score * choose + old_score * keep;
            id3 = candidate_id * take + old_id * (1 - take);
        }
        source += 1;
    }
    if lane_index == 0 {
        let packed = id0 | (id1 << 7) | (id2 << 14) | (id3 << 21);
        if let Some(slot) = packed_top4.get_mut(index) {
            *slot = packed;
        }
    }
    Ok(())
}

#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-gpt-oss-attention-component"
))]
#[kernel(
    typed,
    namespace = "35a1574b42e18b024498807f2e7dd4bde031da33b5b588652fc699cdebb7bb6a",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1]),
    control_flow(loop_bounds(16))
)]
pub fn gfx950_gpt_oss_120b_attention_v1(
    query_bf16: &[u16],
    key_transposed_bf16: &[u16],
    value_f32: &[f32],
    sinks_f32: &[f32],
    mut attention_output: DisjointSlice<f32, Blocked<Index1D, 64, 4>>,
) -> KernelResult {
    if query_bf16.len() < MATRIX_ROWS * HEAD_DIM
        || key_transposed_bf16.len() < HEAD_DIM * CONTEXT_TOKENS
        || value_f32.len() < CONTEXT_TOKENS * VALUE_TILE
        || sinks_f32.len() < MATRIX_ROWS
        || attention_output.len() < ATTENTION_OUTPUT_ELEMENTS
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let lane_index = index.get();
    let lane = WaveLane::<Wave64>::current();
    let subgroup = Gfx950Subgroup::current();
    let Ok(query) = Bf16MfmaAMatrix::row_major(query_bf16, 0, MATRIX_ROWS, HEAD_DIM, HEAD_DIM)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(key) = Bf16MfmaBMatrix::row_major(
        key_transposed_bf16,
        0,
        HEAD_DIM,
        CONTEXT_TOKENS,
        CONTEXT_TOKENS,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let matrix = DeviceMatrix::current();
    let scores = F32AccumulatorFragment::zero(&lane);
    let scores = matrix.multiply_accumulate(
        query.load_m16k16(&lane, 0, 0),
        key.load_k16n16(&lane, 0, 0),
        scores,
    );
    let scores = matrix.multiply_accumulate(
        query.load_m16k16(&lane, 0, 16),
        key.load_k16n16(&lane, 16, 0),
        scores,
    );
    let scores = matrix.multiply_accumulate(
        query.load_m16k16(&lane, 0, 32),
        key.load_k16n16(&lane, 32, 0),
        scores,
    );
    let scores = matrix
        .multiply_accumulate(
            query.load_m16k16(&lane, 0, 48),
            key.load_k16n16(&lane, 48, 0),
            scores,
        )
        .into_values();

    let Ok(values) =
        StridedReadView2D::from_shared_slice(value_f32, 0, CONTEXT_TOKENS, VALUE_TILE, VALUE_TILE)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(sinks) = StridedReadView2D::from_shared_slice(sinks_f32, 0, 1, MATRIX_ROWS, MATRIX_ROWS)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let row_group = lane_index / CONTEXT_TOKENS;
    let row0 = row_group * 4;
    let row1 = row0 + 1;
    let row2 = row0 + 2;
    let row3 = row0 + 3;
    let sink0 = sinks.load_or(0, row0, 0.0);
    let sink1 = sinks.load_or(0, row1, 0.0);
    let sink2 = sinks.load_or(0, row2, 0.0);
    let sink3 = sinks.load_or(0, row3, 0.0);
    let reduced0 = subgroup.reduce_max_f32::<16>(scores[0] * ATTENTION_SCALE);
    let reduced1 = subgroup.reduce_max_f32::<16>(scores[1] * ATTENTION_SCALE);
    let reduced2 = subgroup.reduce_max_f32::<16>(scores[2] * ATTENTION_SCALE);
    let reduced3 = subgroup.reduce_max_f32::<16>(scores[3] * ATTENTION_SCALE);
    let choose0 = (reduced0 > sink0) as u32 as f32;
    let choose1 = (reduced1 > sink1) as u32 as f32;
    let choose2 = (reduced2 > sink2) as u32 as f32;
    let choose3 = (reduced3 > sink3) as u32 as f32;
    let max0 = reduced0 * choose0 + sink0 * (1.0 - choose0);
    let max1 = reduced1 * choose1 + sink1 * (1.0 - choose1);
    let max2 = reduced2 * choose2 + sink2 * (1.0 - choose2);
    let max3 = reduced3 * choose3 + sink3 * (1.0 - choose3);
    let math = Math::current();
    let probability0 = math.exp_f32(scores[0] * ATTENTION_SCALE - max0);
    let probability1 = math.exp_f32(scores[1] * ATTENTION_SCALE - max1);
    let probability2 = math.exp_f32(scores[2] * ATTENTION_SCALE - max2);
    let probability3 = math.exp_f32(scores[3] * ATTENTION_SCALE - max3);
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
        attention0 += subgroup.broadcast_f32::<16>(probability0, token as u32) * value;
        attention1 += subgroup.broadcast_f32::<16>(probability1, token as u32) * value;
        attention2 += subgroup.broadcast_f32::<16>(probability2, token as u32) * value;
        attention3 += subgroup.broadcast_f32::<16>(probability3, token as u32) * value;
        token += 1;
    }

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
    Ok(())
}

#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-gpt-oss-expert-component"
))]
#[kernel(
    typed,
    namespace = "d3675e464d34094297c8b0033dc08d278cc17baf15ed6fd5f53bd382421f8c99",
    launch(required = [64, 1, 1], max = [64, 1, 1], max_grid = [1, 1, 1])
)]
pub fn gfx950_gpt_oss_120b_expert_v1(
    expert_activation_blocks_fp4: &[u8],
    expert_weight_blocks_fp4: &[u8],
    activation_scales: &[f32],
    expert_weight_scales: &[f32],
    packed_top4: &[u32],
    mut expert_output: DisjointSlice<f32, Blocked<Index1D, 64, 4>>,
) -> KernelResult {
    if expert_activation_blocks_fp4.len() < MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE
        || expert_weight_blocks_fp4.len() < EXPERTS * MXFP4_BLOCKS * EXPERT_K_TILE * EXPERT_N_TILE
        || activation_scales.len() < MXFP4_BLOCKS
        || expert_weight_scales.len() < EXPERTS * MXFP4_BLOCKS * EXPERT_N_TILE
        || packed_top4.len() < 1
        || expert_output.len() < EXPERT_OUTPUT_ELEMENTS
    {
        return Err(KernelError::InvalidArgument);
    }
    let index = thread::index_1d();
    let lane_index = index.get();
    let lane = WaveLane::<Wave64>::current();
    let Ok(packed) = StridedReadView2D::from_shared_slice(packed_top4, 0, 1, 1, 1)
    else {
        return Err(KernelError::InvalidArgument);
    };
    let selected = (packed.load_or(0, 0, 0) as usize) & (EXPERTS - 1);
    let column = lane_index % EXPERT_N_TILE;
    let expert_reduction_base = selected * MXFP4_BLOCKS * EXPERT_K_TILE;
    let Ok(weights) = Gfx950Fp4MfmaBMatrix::row_major(
        expert_weight_blocks_fp4,
        0,
        EXPERTS * MXFP4_BLOCKS * EXPERT_K_TILE,
        EXPERT_N_TILE,
        EXPERT_N_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let Ok(activation_scale) =
        StridedReadView2D::from_shared_slice(activation_scales, 0, 1, MXFP4_BLOCKS, MXFP4_BLOCKS)
    else {
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
    let scale0 = activation_scale.load_or(0, 0, 0.0)
        * weight_scale.load_or(selected * MXFP4_BLOCKS, column, 0.0);
    let scale1 = activation_scale.load_or(0, 1, 0.0)
        * weight_scale.load_or(selected * MXFP4_BLOCKS + 1, column, 0.0);
    let scale2 = activation_scale.load_or(0, 2, 0.0)
        * weight_scale.load_or(selected * MXFP4_BLOCKS + 2, column, 0.0);
    let scale3 = activation_scale.load_or(0, 3, 0.0)
        * weight_scale.load_or(selected * MXFP4_BLOCKS + 3, column, 0.0);
    let gfx950 = Gfx950Matrix::current();

    let Ok(activation_matrix0) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        0,
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert0 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix0.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(&lane, expert_reduction_base, 0),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    let mut expert_acc0 = expert0[0] * scale0;
    let mut expert_acc1 = expert0[1] * scale0;
    let mut expert_acc2 = expert0[2] * scale0;
    let mut expert_acc3 = expert0[3] * scale0;

    let Ok(activation_matrix1) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        MATRIX_ROWS * EXPERT_K_TILE,
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert1 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix1.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(&lane, expert_reduction_base + EXPERT_K_TILE, 0),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert1[0] * scale1;
    expert_acc1 += expert1[1] * scale1;
    expert_acc2 += expert1[2] * scale1;
    expert_acc3 += expert1[3] * scale1;

    let Ok(activation_matrix2) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        2 * MATRIX_ROWS * EXPERT_K_TILE,
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert2 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix2.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(&lane, expert_reduction_base + 2 * EXPERT_K_TILE, 0),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert2[0] * scale2;
    expert_acc1 += expert2[1] * scale2;
    expert_acc2 += expert2[2] * scale2;
    expert_acc3 += expert2[3] * scale2;

    let Ok(activation_matrix3) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        3 * MATRIX_ROWS * EXPERT_K_TILE,
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert3 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix3.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(&lane, expert_reduction_base + 3 * EXPERT_K_TILE, 0),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert3[0] * scale3;
    expert_acc1 += expert3[1] * scale3;
    expert_acc2 += expert3[2] * scale3;
    expert_acc3 += expert3[3] * scale3;

    let Some(output_block) = index.checked_block::<64, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
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
    Ok(())
}
