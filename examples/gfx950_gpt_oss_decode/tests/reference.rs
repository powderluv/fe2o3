use fe2o3_gfx950_gpt_oss_decode::{
    EXPERTS, HIDDEN_SIZE, MATRIX_ROWS, TOP_K, VALUE_TILE,
    reference::{decode_bf16, decode_fp4, deterministic_inputs, encode_bf16, reference},
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
