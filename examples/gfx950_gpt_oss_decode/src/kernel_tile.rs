//! Native BF16/MXFP4 compute with checked 16x16 row-major Tiled2D output ownership.

#![allow(missing_docs)]

use fe2o3_device::{
    Bf16MfmaAMatrix, Bf16MfmaBMatrix, DeviceMatrix, DisjointSlice, F32AccumulatorFragment,
    Gfx950F32AccumulatorFragment, Gfx950Fp4E2M1, Gfx950Fp4MfmaAMatrix, Gfx950Fp4MfmaBMatrix,
    Gfx950Matrix, Gfx950Subgroup, Index1D, KernelError, KernelResult, Math, StridedReadView2D,
    Tiled2D, Wave64, WaveLane, kernel, thread,
};

use crate::{
    ATTENTION_OUTPUT_ELEMENTS, CONTEXT_TOKENS, EXPERT_K_TILE, EXPERT_N_TILE,
    EXPERT_OUTPUT_ELEMENTS, EXPERTS, HEAD_DIM, HIDDEN_SIZE, MATRIX_ROWS, MXFP4_BLOCKS, VALUE_TILE,
};

const ATTENTION_SCALE: f32 = 0.125;
// Empty slots are identified by u32::MAX, never by a numerical score floor.
const ROUTER_FLOOR: f32 = -1.0e30;

/// Fuses top-4 routing, sink attention, and selected MXFP4 expert projection.
#[cfg(any(
    not(target_arch = "amdgpu"),
    feature = "kernel-gpt-oss-decode",
    feature = "kernel-gpt-oss-decode-router-serial",
    feature = "kernel-gpt-oss-decode-scalar-attention"
))]
// The macro visits all source cfg branches; each loop keeps its exact bound.
#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(2880, 64, 368640, 64, 16))
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
    mut attention_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    mut expert_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 16, 16, 4>>,
    mut packed_top4: DisjointSlice<u32, Tiled2D<Index1D, 64, 1, 64, 1>>,
) -> KernelResult {
    // Validate the complete launch contract before any lane reaches a broadcast or MFMA.
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

    // One Wave64 owns one independent profile item; lane identity stays wave-local.
    let index = thread::index_1d();
    let global_index = index.get();
    let lane_index = global_index % crate::WAVE_SIZE;
    let item_index = global_index / crate::WAVE_SIZE;
    let lane = WaveLane::<Wave64>::current();
    let subgroup = Gfx950Subgroup::current();

    // Checked views capture offsets and strides once, outside the router dot products.
    let Ok(hidden) = StridedReadView2D::from_shared_slice(
        hidden_f32,
        item_index.wrapping_mul(HIDDEN_SIZE),
        1,
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
    #[cfg(not(feature = "kernel-gpt-oss-decode-router-serial"))]
    let (id0, id1, id2, id3) = {
        // Each lane scores two experts, covering all 128 router rows without serializing lanes.
        let local_expert0 = lane_index.wrapping_mul(2);
        let local_expert1 = local_expert0.wrapping_add(1);
        let mut local_logit0 = 0.0_f32;
        let mut local_logit1 = 0.0_f32;
        let mut depth = 0_usize;
        while depth < HIDDEN_SIZE {
            let activation = hidden.load_or(0, depth, 0.0);
            local_logit0 += activation * router.load_or(local_expert0, depth, 0.0);
            local_logit1 += activation * router.load_or(local_expert1, depth, 0.0);
            depth += 1;
        }

        // Every lane executes the same broadcasts and maintains the same deterministic top-4 list.
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
        (id0, id1, id2, id3)
    };
    #[cfg(feature = "kernel-gpt-oss-decode-router-serial")]
    let (id0, id1, id2, id3) = {
        // Lane-local scoring is the single changed variable in this control kernel.
        let mut best0 = ROUTER_FLOOR;
        let mut best1 = ROUTER_FLOOR;
        let mut best2 = ROUTER_FLOOR;
        let mut best3 = ROUTER_FLOOR;
        let mut id0 = u32::MAX;
        let mut id1 = u32::MAX;
        let mut id2 = u32::MAX;
        let mut id3 = u32::MAX;
        let mut candidate_score = 0.0_f32;
        let mut serial_index = 0_usize;
        while serial_index < EXPERTS * HIDDEN_SIZE {
            let candidate_id = (serial_index / HIDDEN_SIZE) as u32;
            let serial_depth = serial_index % HIDDEN_SIZE;
            candidate_score += hidden.load_or(0, serial_depth, 0.0)
                * router.load_or(candidate_id as usize, serial_depth, 0.0);
            if serial_depth.wrapping_add(1) == HIDDEN_SIZE {
                let mut candidate_id = candidate_id;
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
                candidate_score = 0.0;
            }
            serial_index += 1;
        }
        (id0, id1, id2, id3)
    };
    // The highest-ranked expert drives this bounded single-expert projection profile.
    let selected = (id0 as usize) & (EXPERTS - 1);

    #[cfg(not(feature = "kernel-gpt-oss-decode-scalar-attention"))]
    let (score0, score1, score2, score3) = {
        // Four BF16 MFMA operations cover K=64 while the accumulator remains in registers.
        let Ok(query) = Bf16MfmaAMatrix::row_major(
            query_bf16,
            item_index.wrapping_mul(MATRIX_ROWS * HEAD_DIM),
            MATRIX_ROWS,
            HEAD_DIM,
            HEAD_DIM,
        ) else {
            return Err(KernelError::InvalidArgument);
        };
        let Ok(key) = Bf16MfmaBMatrix::row_major(
            key_transposed_bf16,
            item_index.wrapping_mul(HEAD_DIM * CONTEXT_TOKENS),
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

        let score0 = scores[0];
        let score1 = scores[1];
        let score2 = scores[2];
        let score3 = scores[3];
        (score0, score1, score2, score3)
    };
    #[cfg(feature = "kernel-gpt-oss-decode-scalar-attention")]
    let (score0, score1, score2, score3) = {
        let Ok(query) = StridedReadView2D::from_shared_slice(
            query_bf16,
            item_index.wrapping_mul(MATRIX_ROWS * HEAD_DIM),
            MATRIX_ROWS,
            HEAD_DIM,
            HEAD_DIM,
        ) else {
            return Err(KernelError::InvalidArgument);
        };
        let Ok(key) = StridedReadView2D::from_shared_slice(
            key_transposed_bf16,
            item_index.wrapping_mul(HEAD_DIM * CONTEXT_TOKENS),
            HEAD_DIM,
            CONTEXT_TOKENS,
            CONTEXT_TOKENS,
        ) else {
            return Err(KernelError::InvalidArgument);
        };
        let token_index = lane_index % CONTEXT_TOKENS;
        let row_group = lane_index / CONTEXT_TOKENS;
        let row0 = row_group.wrapping_mul(4);
        let row1 = row0.wrapping_add(1);
        let row2 = row0.wrapping_add(2);
        let row3 = row0.wrapping_add(3);
        let mut score0 = 0.0_f32;
        let mut score1 = 0.0_f32;
        let mut score2 = 0.0_f32;
        let mut score3 = 0.0_f32;
        let mut attention_depth = 0_usize;
        while attention_depth < HEAD_DIM {
            let key_value = {
                let bits = key.load_or(attention_depth, token_index, 0);
                if (bits & 0x7f80) == 0x7f80 && (bits & 0x7f) != 0 {
                    f32::NAN
                } else {
                    fe2o3_device::Bf16::from_bits(bits).to_f32()
                }
            };
            score0 += {
                let bits = query.load_or(row0, attention_depth, 0);
                if (bits & 0x7f80) == 0x7f80 && (bits & 0x7f) != 0 {
                    f32::NAN
                } else {
                    fe2o3_device::Bf16::from_bits(bits).to_f32()
                }
            } * key_value;
            score1 += {
                let bits = query.load_or(row1, attention_depth, 0);
                if (bits & 0x7f80) == 0x7f80 && (bits & 0x7f) != 0 {
                    f32::NAN
                } else {
                    fe2o3_device::Bf16::from_bits(bits).to_f32()
                }
            } * key_value;
            score2 += {
                let bits = query.load_or(row2, attention_depth, 0);
                if (bits & 0x7f80) == 0x7f80 && (bits & 0x7f) != 0 {
                    f32::NAN
                } else {
                    fe2o3_device::Bf16::from_bits(bits).to_f32()
                }
            } * key_value;
            score3 += {
                let bits = query.load_or(row3, attention_depth, 0);
                if (bits & 0x7f80) == 0x7f80 && (bits & 0x7f) != 0 {
                    f32::NAN
                } else {
                    fe2o3_device::Bf16::from_bits(bits).to_f32()
                }
            } * key_value;
            attention_depth += 1;
        }

        (score0, score1, score2, score3)
    };
    // Typed value/sink views make the Wave16 softmax and PV ownership explicit.
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
    // Include learned sinks in a max-subtracted FP32 softmax to avoid overflow.
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
    // Each Wave16 subgroup broadcasts probabilities while lanes own value columns.
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
    let expert_reduction_base = selected
        .wrapping_mul(MXFP4_BLOCKS)
        .wrapping_mul(EXPERT_K_TILE);
    let Ok(weights) = Gfx950Fp4MfmaBMatrix::row_major(
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
    let scale0 = activation_scale.load_or(0, 0, 0.0)
        * weight_scale.load_or(selected.wrapping_mul(MXFP4_BLOCKS), column, 0.0);
    let scale1 = activation_scale.load_or(0, 1, 0.0)
        * weight_scale.load_or(
            selected.wrapping_mul(MXFP4_BLOCKS).wrapping_add(1),
            column,
            0.0,
        );
    let scale2 = activation_scale.load_or(0, 2, 0.0)
        * weight_scale.load_or(
            selected.wrapping_mul(MXFP4_BLOCKS).wrapping_add(2),
            column,
            0.0,
        );
    let scale3 = activation_scale.load_or(0, 3, 0.0)
        * weight_scale.load_or(
            selected.wrapping_mul(MXFP4_BLOCKS).wrapping_add(3),
            column,
            0.0,
        );
    let gfx950 = Gfx950Matrix::current();

    let activation_item_base = item_index.wrapping_mul(MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE);
    // Consume each fragment before loading the next block to bound register pressure.
    let Ok(activation_matrix0) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        activation_item_base,
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
        activation_item_base.wrapping_add(MATRIX_ROWS * EXPERT_K_TILE),
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert1 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix1.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(&lane, expert_reduction_base.wrapping_add(EXPERT_K_TILE), 0),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert1[0] * scale1;
    expert_acc1 += expert1[1] * scale1;
    expert_acc2 += expert1[2] * scale1;
    expert_acc3 += expert1[3] * scale1;

    let Ok(activation_matrix2) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        activation_item_base.wrapping_add(2 * MATRIX_ROWS * EXPERT_K_TILE),
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert2 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix2.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(
                &lane,
                expert_reduction_base.wrapping_add(2 * EXPERT_K_TILE),
                0,
            ),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert2[0] * scale2;
    expert_acc1 += expert2[1] * scale2;
    expert_acc2 += expert2[2] * scale2;
    expert_acc3 += expert2[3] * scale2;

    let Ok(activation_matrix3) = Gfx950Fp4MfmaAMatrix::row_major(
        expert_activation_blocks_fp4,
        activation_item_base.wrapping_add(3 * MATRIX_ROWS * EXPERT_K_TILE),
        MATRIX_ROWS,
        EXPERT_K_TILE,
        EXPERT_K_TILE,
    ) else {
        return Err(KernelError::InvalidArgument);
    };
    let expert3 = gfx950
        .multiply_accumulate_fp4(
            activation_matrix3.load_m16k128(&lane, 0, 0),
            weights.load_k128n16(
                &lane,
                expert_reduction_base.wrapping_add(3 * EXPERT_K_TILE),
                0,
            ),
            Gfx950F32AccumulatorFragment::<Gfx950Fp4E2M1>::zero(&lane),
        )
        .into_values();
    expert_acc0 += expert3[0] * scale3;
    expert_acc1 += expert3[1] * scale3;
    expert_acc2 += expert3[2] * scale3;
    expert_acc3 += expert3[3] * scale3;

    // Blocked capabilities assign four unique attention and expert elements per lane.
    let Some(output_block) = thread::index_1d().checked_tiled_2d::<64, 16, 16, 4>() else {
        return Err(KernelError::OutOfBounds);
    };
    if let Some(slot) = attention_output.get_tiled_2d_mut(
        &output_block,
        0,
        crate::PROFILE_ITEMS * MATRIX_ROWS,
        16,
        16,
    ) {
        *slot = attention0;
    }
    if let Some(slot) = attention_output.get_tiled_2d_mut(
        &output_block,
        1,
        crate::PROFILE_ITEMS * MATRIX_ROWS,
        16,
        16,
    ) {
        *slot = attention1;
    }
    if let Some(slot) = attention_output.get_tiled_2d_mut(
        &output_block,
        2,
        crate::PROFILE_ITEMS * MATRIX_ROWS,
        16,
        16,
    ) {
        *slot = attention2;
    }
    if let Some(slot) = attention_output.get_tiled_2d_mut(
        &output_block,
        3,
        crate::PROFILE_ITEMS * MATRIX_ROWS,
        16,
        16,
    ) {
        *slot = attention3;
    }
    if let Some(slot) =
        expert_output.get_tiled_2d_mut(&output_block, 0, crate::PROFILE_ITEMS * MATRIX_ROWS, 16, 16)
    {
        *slot = expert_acc0;
    }
    if let Some(slot) =
        expert_output.get_tiled_2d_mut(&output_block, 1, crate::PROFILE_ITEMS * MATRIX_ROWS, 16, 16)
    {
        *slot = expert_acc1;
    }
    if let Some(slot) =
        expert_output.get_tiled_2d_mut(&output_block, 2, crate::PROFILE_ITEMS * MATRIX_ROWS, 16, 16)
    {
        *slot = expert_acc2;
    }
    if let Some(slot) =
        expert_output.get_tiled_2d_mut(&output_block, 3, crate::PROFILE_ITEMS * MATRIX_ROWS, 16, 16)
    {
        *slot = expert_acc3;
    }
    // Every lane has the same top-4 list; the 1-D capability keeps item stores disjoint.
    let packed = id0 | (id1 << 7) | (id2 << 14) | (id3 << 21);
    let Some(route_tile) = thread::index_1d().checked_tiled_2d::<64, 1, 64, 1>() else {
        return Err(KernelError::OutOfBounds);
    };
    if let Some(slot) = packed_top4.get_tiled_2d_mut(&route_tile, 0, crate::PROFILE_ITEMS, 64, 64) {
        *slot = packed;
    }
    Ok(())
}
