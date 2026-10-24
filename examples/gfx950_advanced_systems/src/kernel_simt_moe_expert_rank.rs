//! Ordinary scalar SIMT evaluation of the same raw OCP FP4/FP8 expert inputs.
#![allow(missing_docs)]
use crate::{ALL_EXPERTS,EXPERTS,HIDDEN,OUTPUT,SYSTEM_BATCHES,TOKENS,TOP_K};
use fe2o3_device::{Blocked,DeviceMath,DisjointSlice,Index1D,StridedReadView2D,kernel,thread};
// The kernel macro visits both cfg blocks before Rust removes the inactive one:
// one component loop, one interleaved depth loop, and three serial depth loops.
#[kernel(
    typed,
    launch(required=[256,1,1],max=[256,1,1],max_grid=[4,1,1]),
    control_flow(loop_bounds(4,128,128,128,128))
)]
#[allow(clippy::too_many_arguments)]
pub fn gfx950_moe_expert_rank_fp4_fp8_v1(
    activations: &[u8], expert_weights: &[u8], top_experts: &[u32], top_weights: &[f32],
    first_expert: u32, include_shared_expert: u32,
    mut output: DisjointSlice<f32,Blocked<Index1D,64,4>>,
) {
    let index = thread::index_1d();
    let raw = index.get();
    let batch = raw / 64;
    let lane = raw & 63;
    if batch >= SYSTEM_BATCHES || first_expert as usize >= EXPERTS - 1
        || activations.len() != SYSTEM_BATCHES*TOKENS*HIDDEN
        || expert_weights.len() != SYSTEM_BATCHES*ALL_EXPERTS*HIDDEN*OUTPUT
        || top_experts.len() != SYSTEM_BATCHES*TOKENS*TOP_K
        || top_weights.len() != SYSTEM_BATCHES*TOKENS*TOP_K
        || output.len() != SYSTEM_BATCHES*TOKENS*OUTPUT { return; }
    let second_expert = first_expert + 1;
    let activation_base = batch * TOKENS * HIDDEN;
    let weights_base = batch * ALL_EXPERTS * HIDDEN * OUTPUT;
    let routes_base = batch * TOKENS * TOP_K;
    // For the admitted launch: batch<16, component<4, lane<64, hence
    // token<16, column<16, depth<128. These retained views also check the
    // complete parent extents before any raw lowp byte is read.
    let Ok(activations) = StridedReadView2D::from_shared_slice(
        activations, activation_base, TOKENS, HIDDEN, HIDDEN,
    ) else { return; };
    let Ok(expert_weights) = StridedReadView2D::from_shared_slice(
        expert_weights, weights_base, ALL_EXPERTS*HIDDEN, OUTPUT, OUTPUT,
    ) else { return; };
    let Ok(top_experts) = StridedReadView2D::from_shared_slice(
        top_experts, routes_base, TOKENS, TOP_K, TOP_K,
    ) else { return; };
    let Ok(top_weights) = StridedReadView2D::from_shared_slice(
        top_weights, routes_base, TOKENS, TOP_K, TOP_K,
    ) else { return; };
    let math = DeviceMath::current();
    let Some(block) = index.checked_block::<64,4>() else { return; };
    // Retain the four outputs with one explicit bounded component traversal.
    let mut component = 0_usize;
    while component < 4 {
            // Disjoint bit fields encode lane + 64*component exactly for component<4.
            // The masks make the complete 0..255 coordinate explicit at this use site.
            let element = lane | ((component & 3) << 6);
            let token = element / OUTPUT;
            let column = element & (OUTPUT - 1);
            let mut first = 0.0_f32;
            let mut second = 0.0_f32;
            let mut shared = 0.0_f32;
            #[cfg(not(feature = "ablation-expert-serial"))]
            {
            let mut depth = 0_usize;
            while depth < HIDDEN {
                let a = {
                    let bits = activations.load_or(token, depth, 0);
                    let exponent = (bits >> 1_u8) & 3;
                    let mantissa = (bits & 1) as f32;
                    let magnitude = if exponent == 0 { mantissa * 0.5 }
                    else if exponent == 1 { 1.0 + mantissa * 0.5 }
                    else if exponent == 2 { 2.0 + mantissa }
                    else { 4.0 + 2.0 * mantissa };
                    if bits & 8 == 0 { magnitude } else { -magnitude }
                };
                let w0 = {
                    let bits = expert_weights.load_or((first_expert as usize).wrapping_mul(HIDDEN).wrapping_add(depth), column, 0);
                    let exponent = (bits >> 3_u8) & 15;
                    let mantissa = bits & 7;
                    let magnitude = if exponent == 15 && mantissa == 7 {
                    let nan_source = (mantissa ^ 7_u8) as f32;
                    nan_source / nan_source
                    } else if exponent == 0 { mantissa as f32 / 512.0 }
                    else {
                    let exponent0 = (exponent & 1) as f32;
                    let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                    let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                    let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                    let scale = (1.0 + exponent0) * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2) * (1.0 + 255.0 * exponent3) / 128.0;
                    (1.0 + mantissa as f32 / 8.0) * scale
                    };
                    if bits & 128 == 0 { magnitude } else { -magnitude }
                };
                let w1 = {
                    let bits = expert_weights.load_or((second_expert as usize).wrapping_mul(HIDDEN).wrapping_add(depth), column, 0);
                    let exponent = (bits >> 3_u8) & 15;
                    let mantissa = bits & 7;
                    let magnitude = if exponent == 15 && mantissa == 7 {
                    let nan_source = (mantissa ^ 7_u8) as f32;
                    nan_source / nan_source
                    } else if exponent == 0 { mantissa as f32 / 512.0 }
                    else {
                    let exponent0 = (exponent & 1) as f32;
                    let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                    let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                    let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                    let scale = (1.0 + exponent0) * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2) * (1.0 + 255.0 * exponent3) / 128.0;
                    (1.0 + mantissa as f32 / 8.0) * scale
                    };
                    if bits & 128 == 0 { magnitude } else { -magnitude }
                };
                let ws = {
                    let bits = expert_weights.load_or(((ALL_EXPERTS-1)*HIDDEN).wrapping_add(depth), column, 0);
                    let exponent = (bits >> 3_u8) & 15;
                    let mantissa = bits & 7;
                    let magnitude = if exponent == 15 && mantissa == 7 {
                    let nan_source = (mantissa ^ 7_u8) as f32;
                    nan_source / nan_source
                    } else if exponent == 0 { mantissa as f32 / 512.0 }
                    else {
                    let exponent0 = (exponent & 1) as f32;
                    let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                    let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                    let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                    let scale = (1.0 + exponent0) * (1.0 + 3.0 * exponent1)
                    * (1.0 + 15.0 * exponent2) * (1.0 + 255.0 * exponent3) / 128.0;
                    (1.0 + mantissa as f32 / 8.0) * scale
                    };
                    if bits & 128 == 0 { magnitude } else { -magnitude }
                };
                first += a * w0;
                second += a * w1;
                shared += a * ws;
                depth += 1;
            }
            }
            // Complete each expert's raw lowp dot product before loading the next.
            // All three dots remain unconditional, with the same depth and F32 sum order.
            #[cfg(feature = "ablation-expert-serial")]
            {
                {
                    let mut depth = 0_usize;
                    while depth < HIDDEN {
                        let a = {
                            let bits = activations.load_or(token, depth, 0);
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
                        let w0 = {
                            let bits = expert_weights.load_or(
                                (first_expert as usize)
                                    .wrapping_mul(HIDDEN)
                                    .wrapping_add(depth),
                                column,
                                0,
                            );
                            let exponent = (bits >> 3_u8) & 15;
                            let mantissa = bits & 7;
                            let magnitude = if exponent == 15 && mantissa == 7 {
                                let nan_source = (mantissa ^ 7_u8) as f32;
                                nan_source / nan_source
                            } else if exponent == 0 {
                                mantissa as f32 / 512.0
                            } else {
                                let exponent0 = (exponent & 1) as f32;
                                let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                                let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                                let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                                let scale = (1.0 + exponent0)
                                    * (1.0 + 3.0 * exponent1)
                                    * (1.0 + 15.0 * exponent2)
                                    * (1.0 + 255.0 * exponent3)
                                    / 128.0;
                                (1.0 + mantissa as f32 / 8.0) * scale
                            };
                            if bits & 128 == 0 {
                                magnitude
                            } else {
                                -magnitude
                            }
                        };
                        first += a * w0;
                        depth += 1;
                    }
                }
                {
                    let mut depth = 0_usize;
                    while depth < HIDDEN {
                        let a = {
                            let bits = activations.load_or(token, depth, 0);
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
                        let w1 = {
                            let bits = expert_weights.load_or(
                                (second_expert as usize)
                                    .wrapping_mul(HIDDEN)
                                    .wrapping_add(depth),
                                column,
                                0,
                            );
                            let exponent = (bits >> 3_u8) & 15;
                            let mantissa = bits & 7;
                            let magnitude = if exponent == 15 && mantissa == 7 {
                                let nan_source = (mantissa ^ 7_u8) as f32;
                                nan_source / nan_source
                            } else if exponent == 0 {
                                mantissa as f32 / 512.0
                            } else {
                                let exponent0 = (exponent & 1) as f32;
                                let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                                let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                                let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                                let scale = (1.0 + exponent0)
                                    * (1.0 + 3.0 * exponent1)
                                    * (1.0 + 15.0 * exponent2)
                                    * (1.0 + 255.0 * exponent3)
                                    / 128.0;
                                (1.0 + mantissa as f32 / 8.0) * scale
                            };
                            if bits & 128 == 0 {
                                magnitude
                            } else {
                                -magnitude
                            }
                        };
                        second += a * w1;
                        depth += 1;
                    }
                }
                {
                    let mut depth = 0_usize;
                    while depth < HIDDEN {
                        let a = {
                            let bits = activations.load_or(token, depth, 0);
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
                        let ws = {
                            let bits = expert_weights.load_or(
                                ((ALL_EXPERTS - 1) * HIDDEN).wrapping_add(depth),
                                column,
                                0,
                            );
                            let exponent = (bits >> 3_u8) & 15;
                            let mantissa = bits & 7;
                            let magnitude = if exponent == 15 && mantissa == 7 {
                                let nan_source = (mantissa ^ 7_u8) as f32;
                                nan_source / nan_source
                            } else if exponent == 0 {
                                mantissa as f32 / 512.0
                            } else {
                                let exponent0 = (exponent & 1) as f32;
                                let exponent1 = ((exponent >> 1_u8) & 1) as f32;
                                let exponent2 = ((exponent >> 2_u8) & 1) as f32;
                                let exponent3 = ((exponent >> 3_u8) & 1) as f32;
                                let scale = (1.0 + exponent0)
                                    * (1.0 + 3.0 * exponent1)
                                    * (1.0 + 15.0 * exponent2)
                                    * (1.0 + 255.0 * exponent3)
                                    / 128.0;
                                (1.0 + mantissa as f32 / 8.0) * scale
                            };
                            if bits & 128 == 0 {
                                magnitude
                            } else {
                                -magnitude
                            }
                        };
                        shared += a * ws;
                        depth += 1;
                    }
                }
            }
            let selected0 = top_experts.load_or(token, 0, 0);
            let selected1 = top_experts.load_or(token, 1, 0);
            let gate0 = top_weights.load_or(token, 0, 0.0);
            let gate1 = top_weights.load_or(token, 1, 0.0);
            let mut result = 0.0_f32;
            if selected0 == first_expert { result += gate0 * (first / (1.0+math.exp_f32(-first))); }
            else if selected0 == second_expert { result += gate0 * (second / (1.0+math.exp_f32(-second))); }
            if selected1 == first_expert { result += gate1 * (first / (1.0+math.exp_f32(-first))); }
            else if selected1 == second_expert { result += gate1 * (second / (1.0+math.exp_f32(-second))); }
            if include_shared_expert != 0 { result += 0.25 * (shared / (1.0+math.exp_f32(-shared))); }
            // Literal component calls retain the existing provider's admission contract.
            if component == 0 {
                if let Some(slot) = output.get_block_mut(&block,0) { *slot = result; }
            } else if component == 1 {
                if let Some(slot) = output.get_block_mut(&block,1) { *slot = result; }
            } else if component == 2 {
                if let Some(slot) = output.get_block_mut(&block,2) { *slot = result; }
            } else if component == 3 {
                if let Some(slot) = output.get_block_mut(&block,3) { *slot = result; }
            }
        component += 1;
    }
}
