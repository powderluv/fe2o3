use fe2o3_gfx950_advanced_attention::reference::{
    attnres_aggregate_reference_v1, four_branch_residual_reference_v1,
    mhc_sinkhorn_mix_reference_v1,
};
use fe2o3_gfx950_advanced_attention::{CHANNELS_V1, MIXING_STREAMS_V1};

#[test]
fn checked_tile_geometry_covers_every_existing_output_once() {
    let mut visited = vec![false; 64 * 16];
    for invocation in 0..1024 {
        let tile = invocation / 64;
        let lane = invocation % 64;
        let row = tile * 4 + lane / 16;
        let column = lane % 16;
        let offset = row * 16 + column;
        assert_eq!(offset, invocation);
        assert!(!visited[offset]);
        visited[offset] = true;
        assert_eq!(offset / 16, invocation / 16);
        assert_eq!(offset / 64, invocation / 64);
    }
    assert!(visited.into_iter().all(|x| x));
}

#[test]
fn attnres_uniform_logits_match_independent_four_term_mean_for_all_batches() {
    for batch in 0..64 {
        let values: Vec<f32> = (0..64)
            .map(|i| (i as f32 - 27.0) * 0.03125 + batch as f32 * 0.015625)
            .collect();
        let out = attnres_aggregate_reference_v1(&values, &[0.0; 64]).unwrap();
        for c in 0..CHANNELS_V1 {
            let expected = (values[c] + values[16 + c] + values[32 + c] + values[48 + c]) * 0.25;
            assert_eq!(out[c].to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn residual_zero_logits_keep_original_order_and_quarter_gate_factor() {
    for batch in 0..64 {
        let residual: Vec<f32> = (0..16)
            .map(|i| i as f32 * 0.125 - batch as f32 * 0.03125)
            .collect();
        let branches: Vec<f32> = (0..64).map(|i| (i as f32 - 31.0) * 0.0625).collect();
        let out = four_branch_residual_reference_v1(&residual, &branches, &[0.0; 64]).unwrap();
        for c in 0..16 {
            let mut expected = residual[c];
            for b in 0..4 {
                expected += 0.125 * branches[b * 16 + c];
            }
            assert_eq!(out[c].to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn sinkhorn_uniform_matrix_mixes_all_four_rows_for_every_batch() {
    for batch in 0..16 {
        let streams: Vec<f32> = (0..64)
            .map(|i| (i as f32 - 19.0) * 0.125 + batch as f32 * 0.0625)
            .collect();
        let out = mhc_sinkhorn_mix_reference_v1(&streams, &[0.0; 16]).unwrap();
        for row in 0..MIXING_STREAMS_V1 {
            for c in 0..16 {
                let expected =
                    (streams[c] + streams[16 + c] + streams[32 + c] + streams[48 + c]) * 0.25;
                assert_eq!(out[row * 16 + c].to_bits(), expected.to_bits());
            }
        }
    }
}

#[test]
fn mixing_references_refuse_wrong_lengths_and_nonfinite_inputs() {
    assert!(attnres_aggregate_reference_v1(&[0.0; 63], &[0.0; 64]).is_err());
    assert!(four_branch_residual_reference_v1(&[0.0; 16], &[0.0; 64], &[0.0; 63]).is_err());
    assert!(mhc_sinkhorn_mix_reference_v1(&[0.0; 64], &[0.0; 15]).is_err());
    let mut bad = [0.0; 64];
    bad[37] = f32::NAN;
    assert!(attnres_aggregate_reference_v1(&bad, &[0.0; 64]).is_err());
    assert!(four_branch_residual_reference_v1(&[0.0; 16], &bad, &[0.0; 64]).is_err());
    assert!(mhc_sinkhorn_mix_reference_v1(&bad, &[0.0; 16]).is_err());
}

#[test]
fn tile_roots_use_checked_memory_authority_not_a_subgroup_label() {
    let source = include_str!("../src/kernel_tile_mixing.rs");
    assert_eq!(source.matches("Tiled2D<Index1D, 64, 4, 16, 1>").count(), 3);
    assert_eq!(
        source.matches("checked_tiled_2d::<64, 4, 16, 1>()").count(),
        3
    );
    assert_eq!(
        source
            .matches("get_tiled_2d_mut(&tile, 0, 64, 16, 16)")
            .count(),
        3
    );
    assert!(!source.contains("output.get_mut(index)"));
}
