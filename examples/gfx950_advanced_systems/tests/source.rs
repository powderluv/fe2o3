use fe2o3_gfx950_advanced_systems::{
    GFX950_ADVANCED_SYSTEMS_RUST_SOURCE_PRESENT_V1, GFX950_ADVANCED_SYSTEMS_SOURCE_BLOCKER,
    GFX950_ADVANCED_SYSTEMS_SOURCE_LOWERING_SUPPORTED,
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
    assert_eq!(source.matches("pub fn gfx950_").count(), 7);
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
    assert_eq!(source.matches("namespace = \"").count(), 13);
    assert_eq!(source.matches("max_grid = [1, 1, 1]").count(), 11);
}
