// Source-order regressions protect the ablation variable; native compilation and
// the unchanged production HSA fixture remain the numerical acceptance gates.
use fe2o3_gfx950_gpt_oss_decode::{PROFILE_ITEMS, WAVE_SIZE};

#[test]
fn held_counterparts_keep_unscaled_partials_live_until_publication() {
    let simt = include_str!("../src/kernel_held_fragments_simt.rs");
    let tile = include_str!("../src/kernel_held_fragments_tile.rs");
    let original = include_str!("../src/kernel_held_fragments.rs");
    for source in [simt, tile] {
        assert!(source.contains("while depth < HIDDEN_SIZE"));
        assert!(source.contains("while source < 64"));
        assert!(source.contains("take ^ 1"));
        let scales = source.find("let Ok(activation_scale)").unwrap();
        let stores = source.find("if let Some(slot) =").unwrap_or(source.len());
        assert!(scales < stores);
        for block in 0..4 {
            let final_partial = if source == simt {
                format!("expert{block}_3 = dot3;")
            } else {
                format!("let expert{block} = gfx950")
            };
            assert!(source.find(&final_partial).unwrap() < scales);
        }
        let mut tail = source;
        for component in 0..4 {
            for output in ["attention_output", "expert_output"] {
                let access = format!("{output}.get_");
                let start = tail.find(&access).unwrap();
                tail = &tail[start + access.len()..];
                assert!(
                    tail.starts_with(&format!("block_mut(&output_block, {component})"))
                        || tail.starts_with(&format!("tiled_2d_mut(&output_block, {component},"))
                );
            }
        }
    }
    assert!(!simt.contains("Mfma"));
    assert!(!simt.contains("multiply_accumulate"));
    assert_eq!(tile.matches("multiply_accumulate_fp4(").count(), 4);
    // Tile changes ownership, not any original arithmetic or phase ordering.
    let computation = |s: &str| {
        s[s.find("    // Validate the entire item contract").unwrap()
            ..s.find("    // Blocked capabilities prove")
                .unwrap_or_else(|| s.find("    // A 4x64 tile").unwrap())]
            .split_whitespace()
            .collect::<String>()
    };
    assert_eq!(computation(original), computation(tile));
}

#[test]
fn checked_four_by_sixty_four_geometry_preserves_every_original_output_slot() {
    let source = include_str!("../src/kernel_held_fragments_tile.rs");
    assert!(source.contains("Tiled2D<Index1D, 64, 4, 64, 4>"));
    assert!(source.contains("checked_tiled_2d::<64, 4, 64, 4>()"));
    assert_eq!(
        source.matches("crate::PROFILE_ITEMS * 4, 64, 64").count(),
        8
    );
    let mut visited = vec![false; PROFILE_ITEMS * 256];
    for item in 0..PROFILE_ITEMS {
        for lane in 0..WAVE_SIZE {
            for component in 0..4 {
                let blocked = item * 256 + component * 64 + lane;
                // DisjointTile2D geometry: one tile per row, one lane per column.
                let row = item * 4 + (lane / 64) * 4 + component;
                let column = lane % 64;
                let tiled = row * 64 + column;
                assert_eq!(tiled, blocked);
                assert!(!visited[tiled]);
                visited[tiled] = true;
            }
        }
    }
    assert!(visited.into_iter().all(|seen| seen));
}

#[cfg(feature = "kernel-gpt-oss-decode-held-fragments")]
#[test]
fn selected_held_mode_exposes_the_original_kernel_marker() {
    use fe2o3_device::KernelMarkerV1;
    use fe2o3_gfx950_gpt_oss_decode::kernel_held_fragments::
        gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker;
    assert_eq!(
        Marker::LOGICAL_NAME,
        "gfx950_gpt_oss_120b_decode_megakernel_v1"
    );
    assert_eq!(Marker::EXPORT_NAME, Marker::LOGICAL_NAME);
}
