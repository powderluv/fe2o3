use fe2o3_gfx950_gpt_oss_decode::{
    ATTENTION_TILE_ELEMENTS, EXPERT_TILE_ELEMENTS, EXPERTS, HIDDEN_SIZE, MATRIX_ROWS,
    PACKED_ROUTE_ELEMENTS, PROFILE_ITEMS, TOP_K, VALUE_TILE, WAVE_SIZE,
    reference::{
        decode_bf16, decode_fp4, deterministic_batch_inputs, deterministic_inputs, encode_bf16,
        reference, reference_batch,
    },
};

#[test]
fn low_precision_decoders_cover_sign_rounding_and_extrema() {
    assert_eq!(decode_fp4(0x0), 0.0);
    assert_eq!(decode_fp4(0x7), 6.0);
    assert_eq!(decode_fp4(0xf), -6.0);
    for value in [-3.25_f32, -0.5, 0.0, 0.333_251_95, 2.0] {
        let decoded = decode_bf16(encode_bf16(value));
        assert!(decoded.is_finite());
        assert!((decoded - value).abs() <= value.abs().max(1.0) / 128.0);
    }
}

#[test]
fn stable_router_breaks_ties_by_lower_expert_id() {
    let mut inputs = deterministic_inputs();
    inputs.hidden_f32.fill(0.0);
    inputs.router_f32.fill(0.0);
    let result = reference(&inputs);
    assert_eq!(result.top4, [0, 1, 2, 3]);
    assert_eq!(result.packed_top4, 0 | (1 << 7) | (2 << 14) | (3 << 21));
}

#[test]
fn finite_router_logits_below_legacy_floor_keep_all_experts_eligible() {
    use fe2o3_gfx950_gpt_oss_decode::reference::finite_router_edge_batch_inputs;

    let inputs = finite_router_edge_batch_inputs();
    let score = inputs.router_f32[0];
    assert!(score.is_finite() && score < -1.0e30);
    assert_eq!(decode_bf16(encode_bf16(score)).to_bits(), score.to_bits());
    let result = reference_batch(&inputs);
    assert_eq!(result.top4, vec![[0, 1, 2, 3]; PROFILE_ITEMS]);
    let packed = (1 << 7) | (2 << 14) | (3 << 21);
    assert_eq!(result.packed_top4, vec![packed; PACKED_ROUTE_ELEMENTS]);
    assert!(result.attention.iter().all(|value| value.is_finite()));
    assert!(result.expert.iter().all(|value| value.is_finite()));
    assert_eq!(
        result.attention.len(),
        PROFILE_ITEMS * ATTENTION_TILE_ELEMENTS
    );
    assert_eq!(result.expert.len(), PROFILE_ITEMS * EXPERT_TILE_ELEMENTS);
}

#[test]
fn deterministic_profile_routes_dynamically_and_matches_packed_ids() {
    let inputs = deterministic_inputs();
    let first = reference(&inputs);
    let second = reference(&inputs);
    assert_eq!(first.top4, second.top4);
    assert_eq!(first.packed_top4, second.packed_top4);
    assert_eq!(first.attention, second.attention);
    assert_eq!(first.expert, second.expert);
    assert_eq!(first.top4[0], (EXPERTS - 1) as u32);
    assert!(first.top4.windows(2).all(|ids| ids[0] > ids[1]));
    assert!(first.attention.iter().all(|value| value.is_finite()));
    assert!(first.expert.iter().all(|value| value.is_finite()));
}

#[test]
fn attention_sink_changes_normalized_output_without_changing_router() {
    let inputs = deterministic_inputs();
    let baseline = reference(&inputs);
    let mut stronger_sink = inputs.clone();
    stronger_sink.sinks_f32[0] += 8.0;
    let changed = reference(&stronger_sink);
    assert_eq!(baseline.top4, changed.top4);
    assert_ne!(
        &baseline.attention[..VALUE_TILE],
        &changed.attention[..VALUE_TILE]
    );
    assert_eq!(
        &baseline.attention[VALUE_TILE..MATRIX_ROWS * VALUE_TILE],
        &changed.attention[VALUE_TILE..MATRIX_ROWS * VALUE_TILE]
    );
}

#[test]
fn profile_dimensions_cover_the_published_router_contract() {
    assert_eq!((HIDDEN_SIZE, EXPERTS, TOP_K), (2880, 128, 4));
}

#[test]
fn sixteen_item_reference_is_nonuniform_and_wave_disjoint() {
    let inputs = deterministic_batch_inputs();
    let result = reference_batch(&inputs);
    assert_eq!(
        result.attention.len(),
        PROFILE_ITEMS * ATTENTION_TILE_ELEMENTS
    );
    assert_eq!(result.expert.len(), PROFILE_ITEMS * EXPERT_TILE_ELEMENTS);
    assert_eq!(result.top4.len(), PROFILE_ITEMS);
    assert_eq!(result.packed_top4.len(), PACKED_ROUTE_ELEMENTS);

    assert_eq!(result.top4[0][0], (EXPERTS - 1) as u32);
    assert_eq!(result.top4[1][0], 0);
    assert_ne!(result.top4[0], result.top4[1]);
    assert_ne!(
        &result.attention[..ATTENTION_TILE_ELEMENTS],
        &result.attention[ATTENTION_TILE_ELEMENTS..2 * ATTENTION_TILE_ELEMENTS]
    );
    assert_ne!(
        &result.expert[..EXPERT_TILE_ELEMENTS],
        &result.expert[EXPERT_TILE_ELEMENTS..2 * EXPERT_TILE_ELEMENTS]
    );
    for item in 0..PROFILE_ITEMS {
        let routes = &result.packed_top4[item * WAVE_SIZE..(item + 1) * WAVE_SIZE];
        assert!(routes.iter().all(|route| *route == routes[0]));
    }
}

#[test]
fn raw_pipeline_words_preserve_all_batch_qk_values_and_depth_order() {
    use fe2o3_gfx950_gpt_oss_decode::{CONTEXT_TOKENS, HEAD_DIM, MAX_WORKGROUPS, WORKGROUP_SIZE};

    // This tests the finite raw-word address/math contract, not device barriers.
    let inputs = deterministic_batch_inputs();
    let mut scores = vec![0.0_f32; PROFILE_ITEMS * MATRIX_ROWS * CONTEXT_TOKENS];
    let mut checked = 0_usize;
    for workgroup in 0..MAX_WORKGROUPS {
        let mut query_ring = [[0_u64; WORKGROUP_SIZE]; 2];
        let mut key_ring = [[0_u64; WORKGROUP_SIZE]; 2];
        let stage = |epoch: usize,
                     query_ring: &mut [[u64; WORKGROUP_SIZE]; 2],
                     key_ring: &mut [[u64; WORKGROUP_SIZE]; 2]| {
            for producer in 0..WORKGROUP_SIZE {
                let item = workgroup * 4 + producer / WAVE_SIZE;
                let lane = producer % WAVE_SIZE;
                let row = lane % 16;
                let chunk = lane / 16 * 4;
                let key_row = if epoch < 4 {
                    item * HEAD_DIM + epoch * 16
                } else {
                    PROFILE_ITEMS * HEAD_DIM
                };
                let mut q_word = 0_u64;
                let mut k_word = 0_u64;
                for component in 0..4 {
                    let depth = epoch * 16 + chunk + component;
                    let q = if depth < HEAD_DIM {
                        inputs.query_bf16[(item * MATRIX_ROWS + row) * HEAD_DIM + depth]
                    } else {
                        0
                    };
                    let k_depth = key_row + chunk + component;
                    let k = if k_depth < PROFILE_ITEMS * HEAD_DIM {
                        inputs.key_transposed_bf16[k_depth * CONTEXT_TOKENS + row]
                    } else {
                        0
                    };
                    q_word |= (q as u64) << (component * 16);
                    k_word |= (k as u64) << (component * 16);
                }
                query_ring[epoch % 2][producer] = q_word;
                key_ring[epoch % 2][producer] = k_word;
            }
        };
        stage(0, &mut query_ring, &mut key_ring);
        for phase in 0..4 {
            stage(phase + 1, &mut query_ring, &mut key_ring);
            for consumer in 0..WORKGROUP_SIZE {
                let item = workgroup * 4 + consumer / WAVE_SIZE;
                let lane = consumer % WAVE_SIZE;
                let row0 = lane / CONTEXT_TOKENS * 4;
                let token = lane % CONTEXT_TOKENS;
                let wave_base = consumer / WAVE_SIZE * WAVE_SIZE;
                for quarter in 0..4 {
                    let read_base = wave_base + quarter * 16;
                    let k_word = key_ring[phase % 2][read_base + token];
                    for component in 0..4 {
                        let shift = component * 16;
                        let key = decode_bf16((k_word >> shift) as u16);
                        for row_component in 0..4 {
                            let row = row0 + row_component;
                            let q_word = query_ring[phase % 2][read_base + row];
                            let query = decode_bf16((q_word >> shift) as u16);
                            scores[(item * MATRIX_ROWS + row) * CONTEXT_TOKENS + token] +=
                                query * key;
                        }
                    }
                }
            }
        }
        // The final speculative epoch occupies slot zero and is never consumed.
        assert!(query_ring[0].iter().all(|word| *word == 0));
        assert!(key_ring[0].iter().all(|word| *word == 0));
    }
    for item in 0..PROFILE_ITEMS {
        for row in 0..MATRIX_ROWS {
            for token in 0..CONTEXT_TOKENS {
                let mut direct = 0.0_f32;
                for depth in 0..HEAD_DIM {
                    direct += decode_bf16(
                        inputs.query_bf16[(item * MATRIX_ROWS + row) * HEAD_DIM + depth],
                    ) * decode_bf16(
                        inputs.key_transposed_bf16
                            [(item * HEAD_DIM + depth) * CONTEXT_TOKENS + token],
                    );
                }
                assert_eq!(
                    scores[(item * MATRIX_ROWS + row) * CONTEXT_TOKENS + token].to_bits(),
                    direct.to_bits(),
                    "item={item},row={row},token={token}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 4096);
}
