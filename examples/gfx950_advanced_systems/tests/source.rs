use fe2o3_gfx950_advanced_systems::{
    COMBINE_BATCHES, DISPATCH_CAPACITY, GFX950_ADVANCED_SYSTEMS_RUST_SOURCE_PRESENT_V1,
    GFX950_ADVANCED_SYSTEMS_SOURCE_BLOCKER, GFX950_ADVANCED_SYSTEMS_SOURCE_LOWERING_SUPPORTED,
    MUON_ELEMENTS, OUTPUT, STATE_WIDTH, SYSTEM_BATCHES, TABLE_SIZE, TOKENS, TOP_K,
};

#[test]
fn rust_source_is_primary_and_uses_production_lowering() {
    let source = include_str!("../src/kernel.rs");
    for symbol in [
        "gfx950_moe_route_fp4_t16_e4_k2_v1",
        "gfx950_moe_expert_rank_fp4_fp8_v1",
        "gfx950_combine_expert_ranks_v1",
        "gfx950_speculative_transaction_v1",
        "gfx950_qwen_ngram_gather_v1",
        "gfx950_stage_gradient_shard_v1",
        "gfx950_muon_update_4x4_v1",
    ] {
        assert!(source.contains(symbol));
    }
    assert_eq!(source.matches("pub fn gfx950_").count(), 8);
    assert_eq!(
        source
            .matches("pub fn gfx950_speculative_transaction_v1(")
            .count(),
        2
    );
    assert!(GFX950_ADVANCED_SYSTEMS_RUST_SOURCE_PRESENT_V1);
    assert!(GFX950_ADVANCED_SYSTEMS_SOURCE_LOWERING_SUPPORTED);
    assert!(GFX950_ADVANCED_SYSTEMS_SOURCE_BLOCKER.contains("formal compiler refinement"));
    assert!(GFX950_ADVANCED_SYSTEMS_SOURCE_BLOCKER.contains("protected publication"));
    let manifest = include_str!("../Cargo.toml");
    for feature in [
        "kernel-moe-route",
        "kernel-moe-expert-rank",
        "kernel-combine-expert-ranks",
        "kernel-speculative-transaction",
        "kernel-qwen-ngram-gather",
        "kernel-stage-gradient-shard",
        "kernel-muon-update",
    ] {
        assert!(manifest.contains(feature));
        assert!(source.contains(feature));
    }
    for feature in [
        "ablation-expert-serial",
        "ablation-route-redundant-lanes",
        "ablation-combine-transposed",
        "ablation-speculative-recompute-prefix",
        "ablation-ngram-reverse-probe",
        "ablation-stage-tile4",
        "ablation-muon-broadcast16",
    ] {
        assert!(manifest.contains(feature));
        assert!(source.contains(feature));
    }
    assert!(source.contains("let accepted = accepted_prefix!(candidate);"));
    let crate_root = include_str!("../src/lib.rs");
    assert!(manifest.contains("ablation-route-owner-only"));
    assert!(crate_root.contains("ablation-route-owner-only is rejected"));
    assert!(crate_root.contains("ablation-route-unpacked is retained only"));
    assert!(!source.contains("namespace ="));
    assert_eq!(source.matches("max_grid = [4, 1, 1]").count(), 15);
    assert_eq!(source.matches("launch(required = [256, 1, 1]").count(), 15);
    assert!(!source.contains("launch(required = [64, 1, 1]"));
    assert!(source.contains("let batch = global_index / 64;"));
    assert!(source.contains("RowStriped2D<Index1D, 64"));

    assert_eq!(OUTPUT, 16);
    assert!(OUTPUT.is_power_of_two());
    assert_eq!(TOP_K, 2);
    assert_eq!(STATE_WIDTH, 8);
    assert!(STATE_WIDTH.is_power_of_two());
    assert_eq!(TABLE_SIZE, 16);
    assert!(TABLE_SIZE.is_power_of_two());
    assert_eq!(MUON_ELEMENTS, 16);
    assert!(MUON_ELEMENTS.is_power_of_two());
    assert_eq!(SYSTEM_BATCHES, 16);
    assert_eq!(COMBINE_BATCHES, 4);
    assert_eq!(SYSTEM_BATCHES / 4, 4);

    assert_eq!(TOKENS * TOP_K, 32);
    assert_eq!(DISPATCH_CAPACITY, TOKENS * TOP_K);
    assert!(TOKENS * TOP_K <= i32::MAX as usize);
    let compact = source.split_whitespace().collect::<String>();
    assert!(compact.contains("(route0asi32).wrapping_sub(dispatched0).wrapping_mul(choose)"));
    assert!(source.contains("count.wrapping_add((selected == count_expert) as u32)"));
    assert!(source.contains("seen0.wrapping_add(dispatch_matches)"));
    assert!(source.contains("seen1.wrapping_add(dispatch_matches)"));
    assert!(compact.contains(".wrapping_sub(precedes12).wrapping_sub(precedes13)"));
    assert!(compact.contains(
        "dispatch_element1.wrapping_sub((dispatch_expert1asusize).wrapping_mul(DISPATCH_CAPACITY))"
    ));
    assert!(source.contains("if fp4_shift < 32 { 0xc864_3210_u32 >> fp4_shift }"));
    assert!(compact.contains("ifshift<64{((packed_routes>>shift)&3)asu32}"));
    assert!(source.contains("route0 += 1;"));
    assert!(source.contains("route1 += 1;"));
    assert!(
        source.contains("let depth = depth_group.wrapping_add(step.wrapping_mul(depth_stride));")
    );
    assert!(source.contains("let depth_stride = 4_usize;"));
    assert!(source.contains("step += 1;"));
}

#[test]
fn current_performance_audit_is_complete_and_conservative() {
    let audit: serde_json::Value =
        serde_json::from_str(include_str!("../sota-performance-audit-v1.json")).unwrap();
    assert_eq!(
        audit["schema"],
        "fe2o3.gfx950.systems-sota-performance-audit.v1"
    );
    assert_eq!(audit["device"]["physical_gpu"], 5);
    assert_eq!(audit["protocol"]["samples_per_series"], 500);
    assert_eq!(audit["results"].as_array().unwrap().len(), 8);
    assert!(
        audit["claim_boundary"]
            .as_str()
            .unwrap()
            .contains("No external state-of-the-art claim")
    );

    for result in audit["results"].as_array().unwrap() {
        assert_eq!(result["external_comparison_verdict"], "excluded-not-exact");
        assert!(result["resource_floor_ns"].as_f64().unwrap() > 0.0);
        assert!(
            result["measured_over_floor"].is_number() || result["measured_over_floor"].is_array()
        );
        assert!(result["correctness"].as_str().unwrap().contains("passed"));
    }
}

#[test]
fn a_structured_tiles_retain_complete_typed_output_contracts() {
    let manifest = include_str!("../Cargo.toml");
    let crate_root = include_str!("../src/lib.rs");
    let canonical = include_str!("../src/kernel.rs");
    let variants = [
        (
            "kernel-stage-gradient-shard",
            "gfx950_stage_gradient_shard_v1",
            include_str!("../src/kernel_tile_stage_gradient_shard.rs"),
            1,
        ),
        (
            "kernel-muon-update",
            "gfx950_muon_update_4x4_v1",
            include_str!("../src/kernel_tile_muon_update.rs"),
            2,
        ),
        (
            "kernel-qwen-ngram-gather",
            "gfx950_qwen_ngram_gather_v1",
            include_str!("../src/kernel_tile_qwen_ngram_gather.rs"),
            1,
        ),
    ];
    for (feature, symbol, tile, stores) in variants {
        assert!(manifest.contains(&format!("{feature}-tile = [\"{feature}\"]")));
        assert!(crate_root.contains(&format!("feature = \"{feature}-tile\"")));
        assert!(canonical.contains(symbol));
        assert!(canonical.contains(&format!("not(feature = \"{feature}-tile\")")));
        assert!(
            tile.contains(&format!("pub fn {symbol}(")),
            "same existing export"
        );
        assert_eq!(tile.matches("pub fn gfx950_").count(), 1);
        assert_eq!(tile.matches("DisjointSlice<").count(), stores);
        assert_eq!(
            tile.matches("Tiled2D<Index1D, 64, 1, 64, 1>").count(),
            stores
        );
        assert_eq!(
            tile.matches("checked_tiled_2d::<64, 1, 64, 1>()").count(),
            stores
        );
        assert_eq!(tile.matches("get_tiled_2d_mut(").count(), stores);
        assert!(!tile.contains("get_row_striped_2d_mut("));
        assert!(!tile.contains("unsafe"));
        assert!(tile.contains("let batch = global_index / 64;"));
        assert!(tile.contains("batch >= SYSTEM_BATCHES"));
    }
    let stage = include_str!("../src/kernel_tile_stage_gradient_shard.rs");
    assert!(stage.contains("let value = input[input_base.wrapping_add(element)];"));
    let muon = include_str!("../src/kernel_tile_muon_update.rs");
    assert_eq!(muon.matches("    muon_iteration!();").count(), 5);
    assert!(muon.contains("subgroup.reduce_sum_f32::<64>(matrix_value * matrix_value)"));
    assert!(muon.contains("*slot = -MUON_LEARNING_RATE * matrix_value;"));
    assert!(muon.contains("*slot = norm;"));
    let ngram = include_str!("../src/kernel_tile_qwen_ngram_gather.rs");
    assert!(
        ngram.contains(
            "priority > best_priority || (priority == best_priority && slot < best_slot)"
        )
    );
    assert!(ngram.contains("let mut best_value = -1_i32;"));
    assert!(ngram.contains("final_probe!(15);"));
}
