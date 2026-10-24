//! Regression checks for the publication-order ablation and selected capabilities.
#[cfg(feature = "kernel-gpt-oss-decode-interleaved-stores")]
use fe2o3_gfx950_gpt_oss_decode as kernels;

fn publication_order(source: &str) -> Vec<(&str, usize)> {
    let mut result = Vec::new();
    for line in source.lines() {
        let line = line.trim();
        if let Some(value) = line
            .strip_prefix("*slot = ")
            .and_then(|s| s.strip_suffix(';'))
        {
            if let Some(component) = value.strip_prefix("attention") {
                result.push(("attention", component.parse().unwrap()));
            } else if let Some(component) = value.strip_prefix("expert_acc") {
                result.push(("expert", component.parse().unwrap()));
            }
        }
    }
    result
}

#[test]
fn actual_stores_preserve_paired_publication_after_all_arithmetic() {
    let original = include_str!("../src/kernel_interleaved_stores.rs");
    let simt = include_str!("../src/kernel_interleaved_stores_simt.rs");
    let tile = include_str!("../src/kernel_interleaved_stores_tile.rs");
    let expected = vec![
        ("attention", 0),
        ("expert", 0),
        ("attention", 1),
        ("expert", 1),
        ("attention", 2),
        ("expert", 2),
        ("attention", 3),
        ("expert", 3),
    ];
    for source in [original, simt, tile] {
        assert_eq!(publication_order(source), expected);
        let stores = source.find("*slot = attention0;").unwrap();
        assert!(source.find("expert_acc3 +=").unwrap() < stores);
        assert!(source.find("*slot = packed;").unwrap() > stores);
        for guard in [
            "hidden_f32.len()",
            "router_f32.len()",
            "query_bf16.len()",
            "key_transposed_bf16.len()",
            "expert_activation_blocks_fp4.len()",
            "expert_weight_blocks_fp4.len()",
            "attention_output.len()",
            "expert_output.len()",
            "packed_top4.len()",
        ] {
            assert!(source.find(guard).unwrap() < stores);
        }
        assert!(source.contains("take ^ 1"));
    }
    // This distinguishes the ablation from aliases of the canonical fused bodies.
    assert_ne!(
        publication_order(include_str!("../src/kernel_simt.rs")),
        expected
    );
    assert_ne!(
        publication_order(include_str!("../src/kernel_tile.rs")),
        expected
    );
    assert!(!simt.contains("multiply_accumulate"));
    assert!(simt.contains("while attention_depth < HEAD_DIM"));
    assert!(simt.contains("while expert_depth < EXPERT_K_TILE"));
    assert!(simt.contains("while block < MXFP4_BLOCKS"));
    assert_eq!(tile.matches("multiply_accumulate_fp4(").count(), 4);
    assert!(tile.contains("checked_tiled_2d::<64, 4, 64, 4>()"));
    assert_eq!(tile.matches(".get_tiled_2d_mut(").count(), 9);
    assert!(!tile.contains("get_block_mut"));
}

#[cfg(feature = "kernel-gpt-oss-decode-interleaved-stores")]
#[test]
fn selected_interleaved_body_exposes_the_real_export() {
    use fe2o3_device::KernelMarkerV1;
    type Marker =
        kernels::kernel_interleaved_stores::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker;
    assert_eq!(
        Marker::EXPORT_NAME,
        "gfx950_gpt_oss_120b_decode_megakernel_v1"
    );
    assert_eq!(Marker::LOGICAL_NAME, Marker::EXPORT_NAME);
}

#[test]
fn physical_tile_preserves_every_original_blocked_output_slot() {
    let source = include_str!("../src/kernel_interleaved_stores_tile.rs");
    let compact = source.split_whitespace().collect::<String>();
    assert!(compact.contains("Tiled2D<Index1D,64,4,64,4>"));
    assert_eq!(compact.matches("crate::PROFILE_ITEMS*4,64,64").count(), 8);
    let mut visited = vec![false; 4096];
    for item in 0..16 {
        for lane in 0..64 {
            for component in 0..4 {
                let blocked_index = item * 256 + component * 64 + lane;
                // One 4x64 tile per item, one full physical row per component.
                let tile_row = item * 4 + (lane / 64) * 4 + component;
                let tiled_index = tile_row * 64 + lane % 64;
                assert_eq!(tiled_index, blocked_index);
                assert!(!visited[tiled_index]);
                visited[tiled_index] = true;
            }
        }
    }
    assert!(visited.into_iter().all(|v| v));
}
