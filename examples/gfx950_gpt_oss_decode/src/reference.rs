//! Independent safe CPU oracle and deterministic inputs for the layer tile.

const HIDDEN_SIZE: usize = 2880;
const HEAD_DIM: usize = 64;
const GQA_GROUP: usize = 8;
const EXPERTS: usize = 128;
const TOP_K: usize = 4;
const CONTEXT_TOKENS: usize = 16;
const VALUE_TILE: usize = 16;
const MATRIX_ROWS: usize = 16;
const EXPERT_K_TILE: usize = 128;
const EXPERT_N_TILE: usize = 16;
const MXFP4_BLOCK: usize = 32;
const MXFP4_BLOCKS: usize = 4;
const ATTENTION_OUTPUT_ELEMENTS: usize = MATRIX_ROWS * VALUE_TILE;
const EXPERT_OUTPUT_ELEMENTS: usize = MATRIX_ROWS * EXPERT_N_TILE;
const WAVE_SIZE: usize = 64;
const PROFILE_ITEMS: usize = 16;

/// Inputs to the fixed GPT-OSS layer-tile profile.
#[derive(Clone, Debug)]
pub struct GptOssTileInputs {
    /// One normalized token, exactly widened from BF16 storage.
    pub hidden_f32: Vec<f32>,
    /// Expert-major router weights, exactly widened from BF16 storage.
    pub router_f32: Vec<f32>,
    /// Row-major query-head tile in BF16 bits.
    pub query_bf16: Vec<u16>,
    /// Depth-major cached key tile in BF16 bits.
    pub key_transposed_bf16: Vec<u16>,
    /// Token-major cached values, exactly widened from BF16 storage.
    pub value_f32: Vec<f32>,
    /// Learned attention sinks, exactly widened from BF16 storage.
    pub sinks_f32: Vec<f32>,
    /// Four block-isolated, storage-expanded E2M1 activation tiles.
    pub expert_activation_blocks_fp4: Vec<u8>,
    /// Expert-major, block-isolated, storage-expanded E2M1 weight tiles.
    pub expert_weight_blocks_fp4: Vec<u8>,
    /// One decoded E8M0-equivalent scale per activation block.
    pub activation_scales: Vec<f32>,
    /// Expert-, block-, and column-major decoded weight scales.
    pub expert_weight_scales: Vec<f32>,
}

/// Batched inputs for four Wave64 items per workgroup and four workgroups.
#[derive(Clone, Debug)]
pub struct GptOssBatchInputs {
    /// Item-major normalized hidden vectors.
    pub hidden_f32: Vec<f32>,
    /// Shared expert-major router weights.
    pub router_f32: Vec<f32>,
    /// Item-major row-major query tiles.
    pub query_bf16: Vec<u16>,
    /// Item-major depth-major cached key tiles.
    pub key_transposed_bf16: Vec<u16>,
    /// Item-major token-major cached value tiles.
    pub value_f32: Vec<f32>,
    /// Item-major learned attention sinks.
    pub sinks_f32: Vec<f32>,
    /// Item-major block-isolated E2M1 activation tiles.
    pub expert_activation_blocks_fp4: Vec<u8>,
    /// Shared expert-major block-isolated E2M1 weights.
    pub expert_weight_blocks_fp4: Vec<u8>,
    /// Item-major decoded activation scales.
    pub activation_scales: Vec<f32>,
    /// Shared expert-, block-, and column-major weight scales.
    pub expert_weight_scales: Vec<f32>,
}

/// Independent observation for all megakernel outputs.
#[derive(Clone, Debug)]
pub struct GptOssTileReference {
    /// Sink-softmax GQA output including canonical padding rows.
    pub attention: Vec<f32>,
    /// Selected top-1 MXFP4 expert projection tile.
    pub expert: Vec<f32>,
    /// Stable descending top-4 expert IDs.
    pub top4: [u32; TOP_K],
    /// Top-4 IDs packed into four seven-bit fields.
    pub packed_top4: u32,
}

/// Independent observation for the complete sixteen-item launch.
#[derive(Clone, Debug)]
pub struct GptOssBatchReference {
    /// Concatenated attention tiles in item order.
    pub attention: Vec<f32>,
    /// Concatenated expert tiles in item order.
    pub expert: Vec<f32>,
    /// Stable top-4 IDs for every item.
    pub top4: Vec<[u32; TOP_K]>,
    /// Per-lane replicated packed routes in global-invocation order.
    pub packed_top4: Vec<u32>,
}

/// Converts an f32 to BF16 bits with round-to-nearest, ties-to-even.
pub fn encode_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
    bits.wrapping_add(rounding_bias).wrapping_shr(16) as u16
}

/// Converts BF16 bits to f32 without relying on the device crate.
pub fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Decodes one OCP E2M1 code from its storage-expanded low nibble.
pub fn decode_fp4(bits: u8) -> f32 {
    let magnitude = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(bits & 7) as usize];
    if bits & 8 == 0 { magnitude } else { -magnitude }
}

/// Builds deterministic, finite inputs with expert 127 as the top route.
pub fn deterministic_inputs() -> GptOssTileInputs {
    let bf16_codes = [-0.75_f32, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    let fp4_codes = [0x0_u8, 0x1, 0x2, 0x3, 0x9, 0xa, 0xb, 0x4];

    let mut hidden_f32 = vec![0.0_f32; HIDDEN_SIZE];
    hidden_f32[0] = 1.0;
    for depth in 1..HIDDEN_SIZE {
        hidden_f32[depth] = decode_bf16(encode_bf16(
            bf16_codes[(depth * 5 + 3) % bf16_codes.len()] * 0.125,
        ));
    }
    let mut router_f32 = vec![0.0_f32; EXPERTS * HIDDEN_SIZE];
    for expert in 0..EXPERTS {
        router_f32[expert * HIDDEN_SIZE] =
            decode_bf16(encode_bf16((expert as f32 - 64.0) * (1.0 / 64.0)));
        for depth in 1..9 {
            let sign = if (expert + depth) % 2 == 0 { 1.0 } else { -1.0 };
            router_f32[expert * HIDDEN_SIZE + depth] =
                decode_bf16(encode_bf16(sign * (depth as f32) * (1.0 / 1024.0)));
        }
    }
    let mut query_bf16 = vec![encode_bf16(0.0); MATRIX_ROWS * HEAD_DIM];
    for row in 0..GQA_GROUP {
        for depth in 0..HEAD_DIM {
            query_bf16[row * HEAD_DIM + depth] =
                encode_bf16(bf16_codes[(row * 7 + depth * 3 + 1) % bf16_codes.len()]);
        }
    }
    let mut key_transposed_bf16 = vec![encode_bf16(0.0); HEAD_DIM * CONTEXT_TOKENS];
    for depth in 0..HEAD_DIM {
        for token in 0..CONTEXT_TOKENS {
            key_transposed_bf16[depth * CONTEXT_TOKENS + token] =
                encode_bf16(bf16_codes[(depth * 5 + token * 3 + 2) % bf16_codes.len()]);
        }
    }
    let value_f32 = (0..CONTEXT_TOKENS * VALUE_TILE)
        .map(|index| {
            let token = index / VALUE_TILE;
            let column = index % VALUE_TILE;
            decode_bf16(encode_bf16(
                bf16_codes[(token * 3 + column * 5 + 4) % bf16_codes.len()],
            ))
        })
        .collect();
    let sinks_f32 = (0..MATRIX_ROWS)
        .map(|row| {
            if row < GQA_GROUP {
                decode_bf16(encode_bf16((row as f32 - 3.5) * 0.125))
            } else {
                0.0
            }
        })
        .collect();

    let mut expert_activation_blocks_fp4 = vec![0_u8; MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE];
    for block in 0..MXFP4_BLOCKS {
        let block_base = block * MATRIX_ROWS * EXPERT_K_TILE;
        for item in 0..MXFP4_BLOCK {
            let depth = block * MXFP4_BLOCK + item;
            expert_activation_blocks_fp4[block_base + depth] =
                fp4_codes[(depth * 5 + 1) % fp4_codes.len()];
        }
    }
    let expert_stride = MXFP4_BLOCKS * EXPERT_K_TILE * EXPERT_N_TILE;
    let block_stride = EXPERT_K_TILE * EXPERT_N_TILE;
    let mut expert_weight_blocks_fp4 = vec![0_u8; EXPERTS * expert_stride];
    for expert in 0..EXPERTS {
        for block in 0..MXFP4_BLOCKS {
            let base = expert * expert_stride + block * block_stride;
            for item in 0..MXFP4_BLOCK {
                let depth = block * MXFP4_BLOCK + item;
                for column in 0..EXPERT_N_TILE {
                    expert_weight_blocks_fp4[base + depth * EXPERT_N_TILE + column] =
                        fp4_codes[(expert * 3 + depth * 5 + column * 7 + 2) % fp4_codes.len()];
                }
            }
        }
    }
    let activation_scales = vec![0.5_f32, 1.0, 2.0, 0.25];
    let expert_weight_scales = (0..EXPERTS * MXFP4_BLOCKS * EXPERT_N_TILE)
        .map(|index| {
            let block = (index / EXPERT_N_TILE) % MXFP4_BLOCKS;
            let column = index % EXPERT_N_TILE;
            [0.25_f32, 0.5, 1.0, 2.0][(block + column) % 4]
        })
        .collect();

    GptOssTileInputs {
        hidden_f32,
        router_f32,
        query_bf16,
        key_transposed_bf16,
        value_f32,
        sinks_f32,
        expert_activation_blocks_fp4,
        expert_weight_blocks_fp4,
        activation_scales,
        expert_weight_scales,
    }
}

/// Builds nonuniform deterministic inputs for every useful Wave64 item.
pub fn deterministic_batch_inputs() -> GptOssBatchInputs {
    let base = deterministic_inputs();
    let mut hidden_f32 = Vec::with_capacity(PROFILE_ITEMS * HIDDEN_SIZE);
    let mut query_bf16 = Vec::with_capacity(PROFILE_ITEMS * MATRIX_ROWS * HEAD_DIM);
    let mut key_transposed_bf16 = Vec::with_capacity(PROFILE_ITEMS * HEAD_DIM * CONTEXT_TOKENS);
    let mut value_f32 = Vec::with_capacity(PROFILE_ITEMS * CONTEXT_TOKENS * VALUE_TILE);
    let mut sinks_f32 = Vec::with_capacity(PROFILE_ITEMS * MATRIX_ROWS);
    let mut expert_activation_blocks_fp4 =
        Vec::with_capacity(PROFILE_ITEMS * MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE);
    let mut activation_scales = Vec::with_capacity(PROFILE_ITEMS * MXFP4_BLOCKS);

    for item in 0..PROFILE_ITEMS {
        let hidden_start = hidden_f32.len();
        hidden_f32.extend_from_slice(&base.hidden_f32);
        hidden_f32[hidden_start] = if item % 2 == 0 { 1.0 } else { -1.0 };

        let query_start = query_bf16.len();
        query_bf16.extend_from_slice(&base.query_bf16);
        query_bf16[query_start] = encode_bf16(-0.75 + item as f32 * 0.0625);

        key_transposed_bf16.extend_from_slice(&base.key_transposed_bf16);

        let value_start = value_f32.len();
        value_f32.extend_from_slice(&base.value_f32);
        value_f32[value_start] = decode_bf16(encode_bf16(-0.5 + item as f32 * 0.0625));

        let sink_start = sinks_f32.len();
        sinks_f32.extend_from_slice(&base.sinks_f32);
        sinks_f32[sink_start] = decode_bf16(encode_bf16(-0.5 + item as f32 * 0.125));

        let activation_start = expert_activation_blocks_fp4.len();
        expert_activation_blocks_fp4.extend_from_slice(&base.expert_activation_blocks_fp4);
        expert_activation_blocks_fp4[activation_start] = (item as u8) & 0x0f;

        activation_scales.extend_from_slice(&base.activation_scales);
        let scale_start = item * MXFP4_BLOCKS;
        activation_scales[scale_start] = [0.25, 0.5, 1.0, 2.0][item % 4];
    }

    GptOssBatchInputs {
        hidden_f32,
        router_f32: base.router_f32,
        query_bf16,
        key_transposed_bf16,
        value_f32,
        sinks_f32,
        expert_activation_blocks_fp4,
        expert_weight_blocks_fp4: base.expert_weight_blocks_fp4,
        activation_scales,
        expert_weight_scales: base.expert_weight_scales,
    }
}

/// Full 16-item numerical fixture with finite BF16-exact router logits below -1e30.
/// Attention, expert inputs, scales and output dimensions retain the ordinary fixture.
pub fn finite_router_edge_batch_inputs() -> GptOssBatchInputs {
    let mut inputs = deterministic_batch_inputs();
    inputs.hidden_f32.fill(0.0);
    for item in 0..PROFILE_ITEMS {
        inputs.hidden_f32[item * HIDDEN_SIZE] = 1.0;
    }
    inputs.router_f32.fill(0.0);
    for expert in 0..EXPERTS {
        // -(2^104): finite in F32 and exactly representable in BF16.
        inputs.router_f32[expert * HIDDEN_SIZE] = f32::from_bits(0xf380_0000);
    }
    inputs
}

fn stable_top4(inputs: &GptOssTileInputs) -> [u32; TOP_K] {
    let mut logits = vec![0.0_f32; EXPERTS];
    for expert in 0..EXPERTS {
        for depth in 0..HIDDEN_SIZE {
            logits[expert] +=
                inputs.hidden_f32[depth] * inputs.router_f32[expert * HIDDEN_SIZE + depth];
        }
    }
    let mut experts = (0..EXPERTS as u32).collect::<Vec<_>>();
    experts.sort_by(|left, right| {
        logits[*right as usize]
            .total_cmp(&logits[*left as usize])
            .then(left.cmp(right))
    });
    [experts[0], experts[1], experts[2], experts[3]]
}

/// Computes the profile without calling kernel helpers or GPU libraries.
pub fn reference(inputs: &GptOssTileInputs) -> GptOssTileReference {
    let top4 = stable_top4(inputs);
    let packed_top4 = top4[0] | (top4[1] << 7) | (top4[2] << 14) | (top4[3] << 21);

    let mut attention = vec![0.0_f32; ATTENTION_OUTPUT_ELEMENTS];
    for row in 0..MATRIX_ROWS {
        let sink = if row < GQA_GROUP {
            inputs.sinks_f32[row]
        } else {
            0.0
        };
        let mut scores = [0.0_f32; CONTEXT_TOKENS];
        for token in 0..CONTEXT_TOKENS {
            for depth in 0..HEAD_DIM {
                scores[token] += decode_bf16(inputs.query_bf16[row * HEAD_DIM + depth])
                    * decode_bf16(inputs.key_transposed_bf16[depth * CONTEXT_TOKENS + token]);
            }
            scores[token] *= 0.125;
        }
        let maximum = scores
            .iter()
            .copied()
            .fold(sink, |maximum, score| maximum.max(score));
        let mut denominator = (sink - maximum).exp();
        for score in &mut scores {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for column in 0..VALUE_TILE {
            for token in 0..CONTEXT_TOKENS {
                attention[row * VALUE_TILE + column] +=
                    scores[token] / denominator * inputs.value_f32[token * VALUE_TILE + column];
            }
        }
    }

    let selected = top4[0] as usize;
    let expert_stride = MXFP4_BLOCKS * EXPERT_K_TILE * EXPERT_N_TILE;
    let block_stride = EXPERT_K_TILE * EXPERT_N_TILE;
    let mut expert = vec![0.0_f32; EXPERT_OUTPUT_ELEMENTS];
    for column in 0..EXPERT_N_TILE {
        for block in 0..MXFP4_BLOCKS {
            let activation_base = block * MATRIX_ROWS * EXPERT_K_TILE;
            let weight_base = selected * expert_stride + block * block_stride;
            let mut dot = 0.0_f32;
            for depth in block * MXFP4_BLOCK..(block + 1) * MXFP4_BLOCK {
                dot += decode_fp4(inputs.expert_activation_blocks_fp4[activation_base + depth])
                    * decode_fp4(
                        inputs.expert_weight_blocks_fp4
                            [weight_base + depth * EXPERT_N_TILE + column],
                    );
            }
            let weight_scale = inputs.expert_weight_scales
                [(selected * MXFP4_BLOCKS + block) * EXPERT_N_TILE + column];
            expert[column] += dot * inputs.activation_scales[block] * weight_scale;
        }
    }

    GptOssTileReference {
        attention,
        expert,
        top4,
        packed_top4,
    }
}

/// Computes every independent item without device helpers or GPU libraries.
pub fn reference_batch(inputs: &GptOssBatchInputs) -> GptOssBatchReference {
    assert_eq!(inputs.hidden_f32.len(), PROFILE_ITEMS * HIDDEN_SIZE);
    assert_eq!(
        inputs.query_bf16.len(),
        PROFILE_ITEMS * MATRIX_ROWS * HEAD_DIM
    );
    assert_eq!(
        inputs.key_transposed_bf16.len(),
        PROFILE_ITEMS * HEAD_DIM * CONTEXT_TOKENS
    );
    assert_eq!(
        inputs.value_f32.len(),
        PROFILE_ITEMS * CONTEXT_TOKENS * VALUE_TILE
    );
    assert_eq!(inputs.sinks_f32.len(), PROFILE_ITEMS * MATRIX_ROWS);
    assert_eq!(
        inputs.expert_activation_blocks_fp4.len(),
        PROFILE_ITEMS * MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE
    );
    assert_eq!(inputs.activation_scales.len(), PROFILE_ITEMS * MXFP4_BLOCKS);

    let activation_elements = MXFP4_BLOCKS * MATRIX_ROWS * EXPERT_K_TILE;
    let mut attention = Vec::with_capacity(PROFILE_ITEMS * ATTENTION_OUTPUT_ELEMENTS);
    let mut expert = Vec::with_capacity(PROFILE_ITEMS * EXPERT_OUTPUT_ELEMENTS);
    let mut top4 = Vec::with_capacity(PROFILE_ITEMS);
    let mut packed_top4 = Vec::with_capacity(PROFILE_ITEMS * WAVE_SIZE);
    for item in 0..PROFILE_ITEMS {
        let tile = GptOssTileInputs {
            hidden_f32: inputs.hidden_f32[item * HIDDEN_SIZE..(item + 1) * HIDDEN_SIZE].to_vec(),
            router_f32: inputs.router_f32.clone(),
            query_bf16: inputs.query_bf16
                [item * MATRIX_ROWS * HEAD_DIM..(item + 1) * MATRIX_ROWS * HEAD_DIM]
                .to_vec(),
            key_transposed_bf16: inputs.key_transposed_bf16
                [item * HEAD_DIM * CONTEXT_TOKENS..(item + 1) * HEAD_DIM * CONTEXT_TOKENS]
                .to_vec(),
            value_f32: inputs.value_f32
                [item * CONTEXT_TOKENS * VALUE_TILE..(item + 1) * CONTEXT_TOKENS * VALUE_TILE]
                .to_vec(),
            sinks_f32: inputs.sinks_f32[item * MATRIX_ROWS..(item + 1) * MATRIX_ROWS].to_vec(),
            expert_activation_blocks_fp4: inputs.expert_activation_blocks_fp4
                [item * activation_elements..(item + 1) * activation_elements]
                .to_vec(),
            expert_weight_blocks_fp4: inputs.expert_weight_blocks_fp4.clone(),
            activation_scales: inputs.activation_scales
                [item * MXFP4_BLOCKS..(item + 1) * MXFP4_BLOCKS]
                .to_vec(),
            expert_weight_scales: inputs.expert_weight_scales.clone(),
        };
        let result = reference(&tile);
        attention.extend(result.attention);
        expert.extend(result.expert);
        top4.push(result.top4);
        packed_top4.extend(std::iter::repeat_n(result.packed_top4, WAVE_SIZE));
    }
    GptOssBatchReference {
        attention,
        expert,
        top4,
        packed_top4,
    }
}
