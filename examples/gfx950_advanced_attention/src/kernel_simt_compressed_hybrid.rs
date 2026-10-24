//! Scalar E4M3 attention using ordinary Rust arithmetic and checked views.

#![allow(missing_docs)]

use crate::{ATTENTION_TOKENS_V1, CHANNELS_V1, HEAD_DIMENSION_V1};
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
pub fn gfx950_compressed_hybrid_attention(
    q: &[u8],
    k: &[u8],
    v: &[u8],
    token_bias: &[f32],
    mut output: DisjointSlice<f32, RowStriped2D<Index1D, 64, 1>>,
) {
    // Validate exact fixed shapes before LDS publication and subgroup collectives.
    if q.len() != MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1
        || k.len() != MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * HEAD_DIMENSION_V1
        || v.len() != MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1 * CHANNELS_V1
        || token_bias.len() != MULTIGRID_WAVE_BATCHES_V1 * ATTENTION_TOKENS_V1
        || output.len() != MULTIGRID_WAVE_BATCHES_V1 * CHANNELS_V1
    {
        fe2o3_device::trap();
    }
    // A global wave owns one item; lane modulo 16 selects its output channel.
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
    let Ok(value) = StridedReadView2D::from_shared_slice(
        v,
        batch.wrapping_mul(ATTENTION_TOKENS_V1 * CHANNELS_V1),
        ATTENTION_TOKENS_V1,
        CHANNELS_V1,
        CHANNELS_V1,
    ) else {
        fe2o3_device::trap();
    };
    let Ok(bias) = StridedReadView2D::from_shared_slice(
        token_bias,
        batch.wrapping_mul(ATTENTION_TOKENS_V1),
        1,
        16,
        16,
    ) else {
        fe2o3_device::trap();
    };
    let subgroup = Gfx950Subgroup::current();
    let math = DeviceMath::current();
    let score = dot * ATTENTION_SCALE_V1 + bias.load_or(0, column, 0.0);
    let score0 = subgroup.broadcast_f32::<16>(score, 0);
    let score4 = subgroup.broadcast_f32::<16>(score, 4);
    let score8 = subgroup.broadcast_f32::<16>(score, 8);
    let score12 = subgroup.broadcast_f32::<16>(score, 12);
    let score13 = subgroup.broadcast_f32::<16>(score, 13);
    let score14 = subgroup.broadcast_f32::<16>(score, 14);
    let score15 = subgroup.broadcast_f32::<16>(score, 15);

    // Normalize the exact four-token local window independently.
    let mut local_maximum = score12;
    if score13 > local_maximum {
        local_maximum = score13;
    }
    if score14 > local_maximum {
        local_maximum = score14;
    }
    if score15 > local_maximum {
        local_maximum = score15;
    }
    let local_weight0 = math.exp_f32(score12 - local_maximum);
    let local_weight1 = math.exp_f32(score13 - local_maximum);
    let local_weight2 = math.exp_f32(score14 - local_maximum);
    let local_weight3 = math.exp_f32(score15 - local_maximum);
    let local_sum = local_weight0 + local_weight1 + local_weight2 + local_weight3;
    #[cfg(feature = "kernel-compressed-hybrid-attention-division-baseline-v1")]
    let local_value = local_weight0 / local_sum * {
        let bits = value.load_or(12, column, 0);
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
    } + local_weight1 / local_sum * {
        let bits = value.load_or(13, column, 0);
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
    } + local_weight2 / local_sum * {
        let bits = value.load_or(14, column, 0);
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
    } + local_weight3 / local_sum * {
        let bits = value.load_or(15, column, 0);
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
    #[cfg(not(feature = "kernel-compressed-hybrid-attention-division-baseline-v1"))]
    let local_value = {
        let reciprocal = 1.0 / local_sum;
        (local_weight0 * {
            let bits = value.load_or(12, column, 0);
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
        } + local_weight1 * {
            let bits = value.load_or(13, column, 0);
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
        } + local_weight2 * {
            let bits = value.load_or(14, column, 0);
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
        } + local_weight3 * {
            let bits = value.load_or(15, column, 0);
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

    // Normalize three compressed global blocks independently from the local path.
    let mut global_maximum = score0;
    if score4 > global_maximum {
        global_maximum = score4;
    }
    if score8 > global_maximum {
        global_maximum = score8;
    }
    let global_weight0 = math.exp_f32(score0 - global_maximum);
    let global_weight1 = math.exp_f32(score4 - global_maximum);
    let global_weight2 = math.exp_f32(score8 - global_maximum);
    let global_sum = global_weight0 + global_weight1 + global_weight2;
    let compressed0 = ({
        let bits = value.load_or(0, column, 0);
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
    } + {
        let bits = value.load_or(1, column, 0);
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
    } + {
        let bits = value.load_or(2, column, 0);
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
    } + {
        let bits = value.load_or(3, column, 0);
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
    }) * 0.25;
    let compressed1 = ({
        let bits = value.load_or(4, column, 0);
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
    } + {
        let bits = value.load_or(5, column, 0);
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
    } + {
        let bits = value.load_or(6, column, 0);
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
    } + {
        let bits = value.load_or(7, column, 0);
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
    }) * 0.25;
    let compressed2 = ({
        let bits = value.load_or(8, column, 0);
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
    } + {
        let bits = value.load_or(9, column, 0);
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
    } + {
        let bits = value.load_or(10, column, 0);
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
    } + {
        let bits = value.load_or(11, column, 0);
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
    }) * 0.25;
    #[cfg(feature = "kernel-compressed-hybrid-attention-division-baseline-v1")]
    let global_value = global_weight0 / global_sum * compressed0
        + global_weight1 / global_sum * compressed1
        + global_weight2 / global_sum * compressed2;
    #[cfg(not(feature = "kernel-compressed-hybrid-attention-division-baseline-v1"))]
    let global_value = (global_weight0 * compressed0
        + global_weight1 * compressed1
        + global_weight2 * compressed2)
        * (1.0 / global_sum);
    // A learned gate blends the two normalized paths before one disjoint store.
    let mix = 1.0 / (1.0 + math.exp_f32(-score0 * 0.01));
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
        *slot = mix * global_value + (1.0 - mix) * local_value;
    }
}
