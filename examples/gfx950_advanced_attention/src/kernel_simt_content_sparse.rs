//! Scalar E4M3 attention using ordinary Rust arithmetic and checked views.

#![allow(missing_docs)]

use crate::{ATTENTION_TOKENS_V1, CHANNELS_V1, HEAD_DIMENSION_V1, SELECTED_TOKENS_V1};
use fe2o3_device::{
    DeviceMath, DisjointSlice, Gfx950Subgroup, Index1D, RowStriped2D, StridedReadView2D, kernel,
    thread,
};

const MULTIGRID_WAVE_BATCHES_V1: usize = 16;
const ATTENTION_SCALE_V1: f32 = 0.088_388_346;

#[kernel(
    typed,
    launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1]),
    control_flow(loop_bounds(128))
)]
pub fn gfx950_content_sparse_attention(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    content_scores: &[f32],
    mut output: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
    mut selected_output: DisjointSlice<u32, RowStriped2D<Index1D, 64, 1>>,
) {
    // Reject malformed launch-wide storage before LDS and Wave16 collectives.
    if q.len() < MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1
        || k.len() < MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1
        || v.len() < MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * CHANNELS_V1
        || content_scores.len() < MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1
        || output.len() < MULTIGRID_WAVE_BATCHES_V1 * CHANNELS_V1
        || selected_output.len() < MULTIGRID_WAVE_BATCHES_V1 * SELECTED_TOKENS_V1
    {
        fe2o3_device::trap();
    }
    // Each global Wave64 owns one attention item; its lanes cover 16 columns.
    let index = thread::index_1d();
    let batch = index.get() / 64;
    let column = index.get() % ATTENTION_TOKENS_V1;
    // Each lane computes one token score directly from the stored E4M3 bytes.
    let Ok(query) = StridedReadView2D::from_shared_slice(
        q,
        batch.wrapping_mul(ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1),
        ATTENTION_TOKENS_V1,
        HEAD_DIMENSION_V1,
        HEAD_DIMENSION_V1,
    ) else {
        fe2o3_device::trap();
    };
    let Ok(key) = StridedReadView2D::from_shared_slice(
        k,
        batch.wrapping_mul(ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1),
        ATTENTION_TOKENS_V1,
        HEAD_DIMENSION_V1,
        HEAD_DIMENSION_V1,
    ) else {
        fe2o3_device::trap();
    };
    let mut dot = 0.0_f32;
    let mut depth = 0;
    while depth < HEAD_DIMENSION_V1 {
        let query_value = {
            let bits = query.load_or(0, depth, 0);
            let exponent = (bits >> 3_u8) & 0xf;
            let mantissa = bits & 0x7;
            let magnitude = if exponent == 0xf && mantissa == 0x7 {
                let nan_source = ((mantissa ^ 7_u8) as f32);
                nan_source / nan_source
            } else if exponent == 0 {
                ((mantissa) as f32) / 512.0
            } else {
                let exponent0 = ((exponent & 0x1) as f32);
                let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
                let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
                let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
                let scale = (1.0 + exponent0)
                    * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2)
                    * (1.0 + 255.0 * exponent3)
                    / 128.0;
                (1.0 + ((mantissa) as f32) / 8.0) * scale
            };
            if bits & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        };
        let key_value = {
            let bits = key.load_or(column, depth, 0);
            let exponent = (bits >> 3_u8) & 0xf;
            let mantissa = bits & 0x7;
            let magnitude = if exponent == 0xf && mantissa == 0x7 {
                let nan_source = ((mantissa ^ 7_u8) as f32);
                nan_source / nan_source
            } else if exponent == 0 {
                ((mantissa) as f32) / 512.0
            } else {
                let exponent0 = ((exponent & 0x1) as f32);
                let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
                let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
                let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
                let scale = (1.0 + exponent0)
                    * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2)
                    * (1.0 + 255.0 * exponent3)
                    / 128.0;
                (1.0 + ((mantissa) as f32) / 8.0) * scale
            };
            if bits & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        };
        dot += query_value * key_value;
        depth += 1;
    }
    let Ok(content) = StridedReadView2D::from_shared_slice(
        content_scores,
        batch.wrapping_mul(ATTENTION_TOKENS_V1),
        1,
        ATTENTION_TOKENS_V1,
        ATTENTION_TOKENS_V1,
    ) else {
        fe2o3_device::trap();
    };

    // Broadcast all 16 lane scores so block and token selection is deterministic.
    let subgroup = Gfx950Subgroup::current();
    let math = DeviceMath::current();
    let lane_content = content.load_or(0, column, f32::NEG_INFINITY);
    let lane_attention = dot * ATTENTION_SCALE_V1 + 0.75 * lane_content;
    let c0 = subgroup.broadcast_f32::<16>(lane_content, 0);
    let c1 = subgroup.broadcast_f32::<16>(lane_content, 1);
    let c2 = subgroup.broadcast_f32::<16>(lane_content, 2);
    let c3 = subgroup.broadcast_f32::<16>(lane_content, 3);
    let c4 = subgroup.broadcast_f32::<16>(lane_content, 4);
    let c5 = subgroup.broadcast_f32::<16>(lane_content, 5);
    let c6 = subgroup.broadcast_f32::<16>(lane_content, 6);
    let c7 = subgroup.broadcast_f32::<16>(lane_content, 7);
    let c8 = subgroup.broadcast_f32::<16>(lane_content, 8);
    let c9 = subgroup.broadcast_f32::<16>(lane_content, 9);
    let c10 = subgroup.broadcast_f32::<16>(lane_content, 10);
    let c11 = subgroup.broadcast_f32::<16>(lane_content, 11);
    let c12 = subgroup.broadcast_f32::<16>(lane_content, 12);
    let c13 = subgroup.broadcast_f32::<16>(lane_content, 13);
    let c14 = subgroup.broadcast_f32::<16>(lane_content, 14);
    let c15 = subgroup.broadcast_f32::<16>(lane_content, 15);
    let a0 = subgroup.broadcast_f32::<16>(lane_attention, 0);
    let a1 = subgroup.broadcast_f32::<16>(lane_attention, 1);
    let a2 = subgroup.broadcast_f32::<16>(lane_attention, 2);
    let a3 = subgroup.broadcast_f32::<16>(lane_attention, 3);
    let a4 = subgroup.broadcast_f32::<16>(lane_attention, 4);
    let a5 = subgroup.broadcast_f32::<16>(lane_attention, 5);
    let a6 = subgroup.broadcast_f32::<16>(lane_attention, 6);
    let a7 = subgroup.broadcast_f32::<16>(lane_attention, 7);
    let a8 = subgroup.broadcast_f32::<16>(lane_attention, 8);
    let a9 = subgroup.broadcast_f32::<16>(lane_attention, 9);
    let a10 = subgroup.broadcast_f32::<16>(lane_attention, 10);
    let a11 = subgroup.broadcast_f32::<16>(lane_attention, 11);
    let a12 = subgroup.broadcast_f32::<16>(lane_attention, 12);
    let a13 = subgroup.broadcast_f32::<16>(lane_attention, 13);
    let a14 = subgroup.broadcast_f32::<16>(lane_attention, 14);
    let a15 = subgroup.broadcast_f32::<16>(lane_attention, 15);

    let mut block0 = c0;
    if c1 > block0 {
        block0 = c1;
    }
    if c2 > block0 {
        block0 = c2;
    }
    if c3 > block0 {
        block0 = c3;
    }
    let mut block1 = c4;
    if c5 > block1 {
        block1 = c5;
    }
    if c6 > block1 {
        block1 = c6;
    }
    if c7 > block1 {
        block1 = c7;
    }
    let mut block2 = c8;
    if c9 > block2 {
        block2 = c9;
    }
    if c10 > block2 {
        block2 = c10;
    }
    if c11 > block2 {
        block2 = c11;
    }
    let mut block3 = c12;
    if c13 > block3 {
        block3 = c13;
    }
    if c14 > block3 {
        block3 = c14;
    }
    if c15 > block3 {
        block3 = c15;
    }
    let mut first_block = 0;
    let mut first_block_score = block0;
    if block1 > first_block_score {
        first_block = 1;
        first_block_score = block1;
    }
    if block2 > first_block_score {
        first_block = 2;
        first_block_score = block2;
    }
    if block3 > first_block_score {
        first_block = 3;
    }
    let mut second_block = if first_block == 0 { 1 } else { 0 };
    let mut second_block_score = if second_block == 0 { block0 } else { block1 };
    if first_block != 1 && block1 > second_block_score {
        second_block = 1;
        second_block_score = block1;
    }
    if first_block != 2 && block2 > second_block_score {
        second_block = 2;
        second_block_score = block2;
    }
    if first_block != 3 && block3 > second_block_score {
        second_block = 3;
    }

    let keep0 = first_block == 0 || second_block == 0;
    let keep1 = first_block == 1 || second_block == 1;
    let keep2 = first_block == 2 || second_block == 2;
    let keep3 = first_block == 3 || second_block == 3;
    let e0 = if keep0 { c0 } else { f32::NEG_INFINITY };
    let e1 = if keep0 { c1 } else { f32::NEG_INFINITY };
    let e2 = if keep0 { c2 } else { f32::NEG_INFINITY };
    let e3 = if keep0 { c3 } else { f32::NEG_INFINITY };
    let e4 = if keep1 { c4 } else { f32::NEG_INFINITY };
    let e5 = if keep1 { c5 } else { f32::NEG_INFINITY };
    let e6 = if keep1 { c6 } else { f32::NEG_INFINITY };
    let e7 = if keep1 { c7 } else { f32::NEG_INFINITY };
    let e8 = if keep2 { c8 } else { f32::NEG_INFINITY };
    let e9 = if keep2 { c9 } else { f32::NEG_INFINITY };
    let e10 = if keep2 { c10 } else { f32::NEG_INFINITY };
    let e11 = if keep2 { c11 } else { f32::NEG_INFINITY };
    let e12 = if keep3 { c12 } else { f32::NEG_INFINITY };
    let e13 = if keep3 { c13 } else { f32::NEG_INFINITY };
    let e14 = if keep3 { c14 } else { f32::NEG_INFINITY };
    let e15 = if keep3 { c15 } else { f32::NEG_INFINITY };

    let mut selected0 = usize::MAX;
    let mut selected0_rank = f32::NEG_INFINITY;
    let mut selected0_attention = f32::NEG_INFINITY;
    let mut selected1 = usize::MAX;
    let mut selected1_rank = f32::NEG_INFINITY;
    let mut selected1_attention = f32::NEG_INFINITY;
    let mut selected2 = usize::MAX;
    let mut selected2_rank = f32::NEG_INFINITY;
    let mut selected2_attention = f32::NEG_INFINITY;
    {
        let candidate_rank = e0;
        let candidate_attention = a0;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 0;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 0;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 0;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e1;
        let candidate_attention = a1;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 1;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 1;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 1;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e2;
        let candidate_attention = a2;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 2;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 2;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 2;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e3;
        let candidate_attention = a3;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 3;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 3;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 3;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e4;
        let candidate_attention = a4;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 4;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 4;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 4;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e5;
        let candidate_attention = a5;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 5;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 5;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 5;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e6;
        let candidate_attention = a6;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 6;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 6;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 6;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e7;
        let candidate_attention = a7;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 7;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 7;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 7;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e8;
        let candidate_attention = a8;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 8;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 8;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 8;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e9;
        let candidate_attention = a9;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 9;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 9;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 9;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e10;
        let candidate_attention = a10;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 10;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 10;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 10;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e11;
        let candidate_attention = a11;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 11;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 11;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 11;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e12;
        let candidate_attention = a12;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 12;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 12;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 12;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e13;
        let candidate_attention = a13;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 13;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 13;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 13;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e14;
        let candidate_attention = a14;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 14;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 14;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 14;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    {
        let candidate_rank = e15;
        let candidate_attention = a15;
        if candidate_rank > selected0_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = selected0;
            selected1_rank = selected0_rank;
            selected1_attention = selected0_attention;
            selected0 = 15;
            selected0_rank = candidate_rank;
            selected0_attention = candidate_attention;
        } else if candidate_rank > selected1_rank {
            selected2 = selected1;
            selected2_rank = selected1_rank;
            selected2_attention = selected1_attention;
            selected1 = 15;
            selected1_rank = candidate_rank;
            selected1_attention = candidate_attention;
        } else if candidate_rank > selected2_rank {
            selected2 = 15;
            selected2_rank = candidate_rank;
            selected2_attention = candidate_attention;
        }
    };
    // Only the first three ranks write IDs; row-striped ownership prevents races.
    let selected_index = thread::index_1d();
    let selected_rank = selected_index.get() % 64;
    if selected_rank < SELECTED_TOKENS_V1 {
        let selected = if selected_rank == 0 {
            selected0
        } else if selected_rank == 1 {
            selected1
        } else {
            selected2
        };
        let Some(selected_stripe) = selected_index.checked_row_striped_2d::<64, 1>() else {
            fe2o3_device::trap();
        };
        if let Some(slot) = selected_output.get_row_striped_2d_mut(
            &selected_stripe,
            0,
            MULTIGRID_WAVE_BATCHES_V1,
            SELECTED_TOKENS_V1,
            SELECTED_TOKENS_V1,
        ) {
            *slot = selected as u32;
        }
    }

    let Ok(value) = StridedReadView2D::from_shared_slice(
        v,
        batch.wrapping_mul(ATTENTION_TOKENS_V1 * CHANNELS_V1),
        ATTENTION_TOKENS_V1,
        CHANNELS_V1,
        CHANNELS_V1,
    ) else {
        fe2o3_device::trap();
    };
    // Normalize only the retained tokens with max-subtracted FP32 softmax.
    let mut maximum = selected0_attention;
    if selected1_attention > maximum {
        maximum = selected1_attention;
    }
    if selected2_attention > maximum {
        maximum = selected2_attention;
    }
    let weight0 = math.exp_f32(selected0_attention - maximum);
    let weight1 = math.exp_f32(selected1_attention - maximum);
    let weight2 = math.exp_f32(selected2_attention - maximum);
    let denominator = weight0 + weight1 + weight2;
    #[cfg(not(feature = "kernel-content-sparse-attention-reciprocal-reuse-v1"))]
    let result = weight0 / denominator * {
        let bits = value.load_or(selected0, column, 0);
        let exponent = (bits >> 3_u8) & 0xf;
        let mantissa = bits & 0x7;
        let magnitude = if exponent == 0xf && mantissa == 0x7 {
            let nan_source = ((mantissa ^ 7_u8) as f32);
            nan_source / nan_source
        } else if exponent == 0 {
            ((mantissa) as f32) / 512.0
        } else {
            let exponent0 = ((exponent & 0x1) as f32);
            let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
            let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
            let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
            let scale = (1.0 + exponent0)
                * (1.0 + 3.0 * exponent1)
                * (1.0 + 15.0 * exponent2)
                * (1.0 + 255.0 * exponent3)
                / 128.0;
            (1.0 + ((mantissa) as f32) / 8.0) * scale
        };
        if bits & 0x80 == 0 {
            magnitude
        } else {
            -magnitude
        }
    } + weight1 / denominator * {
        let bits = value.load_or(selected1, column, 0);
        let exponent = (bits >> 3_u8) & 0xf;
        let mantissa = bits & 0x7;
        let magnitude = if exponent == 0xf && mantissa == 0x7 {
            let nan_source = ((mantissa ^ 7_u8) as f32);
            nan_source / nan_source
        } else if exponent == 0 {
            ((mantissa) as f32) / 512.0
        } else {
            let exponent0 = ((exponent & 0x1) as f32);
            let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
            let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
            let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
            let scale = (1.0 + exponent0)
                * (1.0 + 3.0 * exponent1)
                * (1.0 + 15.0 * exponent2)
                * (1.0 + 255.0 * exponent3)
                / 128.0;
            (1.0 + ((mantissa) as f32) / 8.0) * scale
        };
        if bits & 0x80 == 0 {
            magnitude
        } else {
            -magnitude
        }
    } + weight2 / denominator * {
        let bits = value.load_or(selected2, column, 0);
        let exponent = (bits >> 3_u8) & 0xf;
        let mantissa = bits & 0x7;
        let magnitude = if exponent == 0xf && mantissa == 0x7 {
            let nan_source = ((mantissa ^ 7_u8) as f32);
            nan_source / nan_source
        } else if exponent == 0 {
            ((mantissa) as f32) / 512.0
        } else {
            let exponent0 = ((exponent & 0x1) as f32);
            let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
            let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
            let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
            let scale = (1.0 + exponent0)
                * (1.0 + 3.0 * exponent1)
                * (1.0 + 15.0 * exponent2)
                * (1.0 + 255.0 * exponent3)
                / 128.0;
            (1.0 + ((mantissa) as f32) / 8.0) * scale
        };
        if bits & 0x80 == 0 {
            magnitude
        } else {
            -magnitude
        }
    };
    #[cfg(feature = "kernel-content-sparse-attention-reciprocal-reuse-v1")]
    let result = {
        let reciprocal = 1.0 / denominator;
        (weight0 * {
            let bits = value.load_or(selected0, column, 0);
            let exponent = (bits >> 3_u8) & 0xf;
            let mantissa = bits & 0x7;
            let magnitude = if exponent == 0xf && mantissa == 0x7 {
                let nan_source = ((mantissa ^ 7_u8) as f32);
                nan_source / nan_source
            } else if exponent == 0 {
                ((mantissa) as f32) / 512.0
            } else {
                let exponent0 = ((exponent & 0x1) as f32);
                let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
                let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
                let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
                let scale = (1.0 + exponent0)
                    * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2)
                    * (1.0 + 255.0 * exponent3)
                    / 128.0;
                (1.0 + ((mantissa) as f32) / 8.0) * scale
            };
            if bits & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        } + weight1 * {
            let bits = value.load_or(selected1, column, 0);
            let exponent = (bits >> 3_u8) & 0xf;
            let mantissa = bits & 0x7;
            let magnitude = if exponent == 0xf && mantissa == 0x7 {
                let nan_source = ((mantissa ^ 7_u8) as f32);
                nan_source / nan_source
            } else if exponent == 0 {
                ((mantissa) as f32) / 512.0
            } else {
                let exponent0 = ((exponent & 0x1) as f32);
                let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
                let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
                let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
                let scale = (1.0 + exponent0)
                    * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2)
                    * (1.0 + 255.0 * exponent3)
                    / 128.0;
                (1.0 + ((mantissa) as f32) / 8.0) * scale
            };
            if bits & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        } + weight2 * {
            let bits = value.load_or(selected2, column, 0);
            let exponent = (bits >> 3_u8) & 0xf;
            let mantissa = bits & 0x7;
            let magnitude = if exponent == 0xf && mantissa == 0x7 {
                let nan_source = ((mantissa ^ 7_u8) as f32);
                nan_source / nan_source
            } else if exponent == 0 {
                ((mantissa) as f32) / 512.0
            } else {
                let exponent0 = ((exponent & 0x1) as f32);
                let exponent1 = (((exponent >> 1_u8) & 0x1) as f32);
                let exponent2 = (((exponent >> 2_u8) & 0x1) as f32);
                let exponent3 = (((exponent >> 3_u8) & 0x1) as f32);
                let scale = (1.0 + exponent0)
                    * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2)
                    * (1.0 + 255.0 * exponent3)
                    / 128.0;
                (1.0 + ((mantissa) as f32) / 8.0) * scale
            };
            if bits & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }) * reciprocal
    };
    let output_gate = 1.0 / (1.0 + math.exp_f32(-maximum * 0.01));
    // Publish the gated PV result through the proven row-striped mapping.
    let Some(output_stripe) = index.checked_row_striped_2d::<64, 1>() else {
        fe2o3_device::trap();
    };
    if let Some(slot) = output.get_row_striped_2d_mut(
        &output_stripe,
        0,
        MULTIGRID_WAVE_BATCHES_V1,
        CHANNELS_V1,
        CHANNELS_V1,
    ) {
        *slot = result * output_gate;
    }
}
