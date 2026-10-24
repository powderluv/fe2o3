use fe2o3_gfx950_gpt_oss_decode::{
    CONTEXT_TOKENS, EXPERTS, HIDDEN_SIZE, MATRIX_ROWS, MAX_WORKGROUPS, OPENAI_GPT_OSS_COMMIT,
    PROFILE_BOUNDARY, PROFILE_ITEMS, WAVE_SIZE, WAVES_PER_WORKGROUP, WORKGROUP_SIZE,
};

#[cfg(not(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
)))]
#[test]
fn selected_host_features_expose_the_generated_kernel_markers() {
    use fe2o3_gfx950_gpt_oss_decode as kernels;

    fn check<M: fe2o3_device::KernelMarkerV1>(name: &str) {
        assert_eq!(M::LOGICAL_NAME, name);
        assert_eq!(M::EXPORT_NAME, name);
    }

    const DECODE: &str = "gfx950_gpt_oss_120b_decode_megakernel_v1";
    #[cfg(any(
        feature = "kernel-gpt-oss-decode",
        not(any(
            feature = "kernel-gpt-oss-decode-router-serial",
            feature = "kernel-gpt-oss-decode-held-fragments",
            feature = "kernel-gpt-oss-decode-scalar-attention",
            feature = "kernel-gpt-oss-decode-pipelined-attention",
            feature = "kernel-gpt-oss-decode-interleaved-stores",
        ))
    ))]
    check::<kernels::kernel::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(DECODE);
    #[cfg(feature = "kernel-gpt-oss-decode-router-serial")]
    check::<kernels::kernel_router_serial::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(
        DECODE,
    );
    #[cfg(feature = "kernel-gpt-oss-decode-held-fragments")]
    check::<kernels::kernel_held_fragments::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(
        DECODE,
    );
    #[cfg(feature = "kernel-gpt-oss-decode-scalar-attention")]
    check::<kernels::kernel_scalar_attention::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(
        DECODE,
    );
    #[cfg(feature = "kernel-gpt-oss-decode-pipelined-attention")]
    check::<
        kernels::kernel_pipelined_attention::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker,
    >(DECODE);
    #[cfg(feature = "kernel-gpt-oss-decode-interleaved-stores")]
    check::<kernels::kernel_interleaved_stores::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(
        DECODE,
    );
    #[cfg(feature = "kernel-gpt-oss-router-component")]
    check::<kernels::kernel_components::gfx950_gpt_oss_120b_router_v1_gpu::Marker>(
        "gfx950_gpt_oss_120b_router_v1",
    );
    #[cfg(feature = "kernel-gpt-oss-attention-component")]
    check::<kernels::kernel_components::gfx950_gpt_oss_120b_attention_v1_gpu::Marker>(
        "gfx950_gpt_oss_120b_attention_v1",
    );
    #[cfg(feature = "kernel-gpt-oss-expert-component")]
    check::<kernels::kernel_components::gfx950_gpt_oss_120b_expert_v1_gpu::Marker>(
        "gfx950_gpt_oss_120b_expert_v1",
    );
}

#[test]
fn production_source_preserves_the_fixed_layer_tile_contract() {
    let source = include_str!("../src/kernel.rs");
    assert!(source.contains("gfx950_gpt_oss_120b_decode_megakernel_v1"));
    assert!(
        source.contains("launch(required = [256, 1, 1], max = [256, 1, 1], max_grid = [4, 1, 1])")
    );
    assert!(source.contains("control_flow(loop_bounds(2880, 64, 16))"));
    assert_eq!(source.matches("multiply_accumulate(\n").count(), 4);
    assert_eq!(source.matches("multiply_accumulate_fp4(").count(), 4);
    assert!(source.contains("while source < 64"));
    assert!(source.contains("local_expert0 = lane_index.wrapping_mul(2)"));
    assert!(source.contains("local_expert1 = local_expert0.wrapping_add(1)"));
    assert!(source.contains("let selected = (id0 as usize)"));
    assert!(source.contains("let lane_index = global_index % crate::WAVE_SIZE"));
    assert!(source.contains("let item_index = global_index / crate::WAVE_SIZE"));
    assert!(source.contains("item_index.wrapping_mul(HIDDEN_SIZE)"));
    assert!(source.contains("activation_item_base"));
    assert!(!source.contains("unsafe"));
    assert_eq!(OPENAI_GPT_OSS_COMMIT.len(), 40);
    assert!(PROFILE_BOUNDARY.contains("batch=1"));
    assert!(PROFILE_BOUNDARY.contains("full 128-way top-4 router"));
    assert!(PROFILE_BOUNDARY.contains("selected top-1 MLP1"));
    assert!(PROFILE_BOUNDARY.contains("attention rows 8..16"));
    assert!(PROFILE_BOUNDARY.contains("expert rows 1..16"));
    assert_eq!(HIDDEN_SIZE, 2880);
    assert_eq!(CONTEXT_TOKENS, 16);
    assert_eq!(CONTEXT_TOKENS, MATRIX_ROWS);
    assert_eq!(EXPERTS, 128);
    assert_eq!(WAVE_SIZE, 64);
    assert_eq!(WORKGROUP_SIZE, 256);
    assert_eq!(WAVES_PER_WORKGROUP, 4);
    assert_eq!(MAX_WORKGROUPS, 4);
    assert_eq!(PROFILE_ITEMS, 16);
}

#[test]
fn ablation_sources_keep_the_production_export_and_exact_stage_shapes() {
    let variants = [
        (
            "kernel-gpt-oss-decode-router-serial",
            include_str!("../src/kernel_router_serial.rs"),
        ),
        (
            "kernel-gpt-oss-decode-held-fragments",
            include_str!("../src/kernel_held_fragments.rs"),
        ),
        (
            "kernel-gpt-oss-decode-scalar-attention",
            include_str!("../src/kernel_scalar_attention.rs"),
        ),
        (
            "kernel-gpt-oss-decode-interleaved-stores",
            include_str!("../src/kernel_interleaved_stores.rs"),
        ),
        (
            "kernel-gpt-oss-decode-pipelined-attention",
            include_str!("../src/kernel_pipelined_attention.rs"),
        ),
    ];
    for (feature, source) in variants {
        assert!(source.contains(feature));
        assert!(source.contains("gfx950_gpt_oss_120b_decode_megakernel_v1"));
        let compact = source.split_whitespace().collect::<String>();
        assert!(compact.contains("launch(required=[256,1,1]"));
        assert!(source.contains("max_grid = [4, 1, 1]"));
        assert!(source.contains("let lane_index = global_index % crate::WAVE_SIZE"));
        assert!(source.contains("let item_index = global_index / crate::WAVE_SIZE"));
        assert_eq!(source.matches("multiply_accumulate_fp4(").count(), 4);
        assert!(!source.contains("unsafe"));
    }

    let serial = include_str!("../src/kernel_router_serial.rs");
    assert!(serial.contains("while serial_index < EXPERTS * HIDDEN_SIZE"));
    let held = include_str!("../src/kernel_held_fragments.rs");
    assert_eq!(held.matches("multiply_accumulate_fp4(").count(), 4);
    let scalar = include_str!("../src/kernel_scalar_attention.rs");
    assert!(!scalar.contains("Bf16MfmaAMatrix"));
    assert!(scalar.contains("fn widen_bf16"));
    assert!(!scalar.contains("f32::from_bits"));
    let interleaved = include_str!("../src/kernel_interleaved_stores.rs");
    assert!(interleaved.contains("attention_output.get_block_mut(&output_block, 0)"));
    assert!(interleaved.contains("expert_output.get_block_mut(&output_block, 0)"));
}

#[test]
fn pipelined_attention_source_has_two_real_double_buffered_lds_pipelines() {
    let source = include_str!("../src/kernel_pipelined_attention.rs");
    assert!(source.contains("kernel-gpt-oss-decode-pipelined-attention"));
    // Each lane retains one eight-byte fragment in each of two double buffers.
    let fragment_bytes = core::mem::size_of::<fe2o3_device::Bf16MfmaAFragment<'_>>()
        + core::mem::size_of::<fe2o3_device::Bf16MfmaBFragment<'_>>();
    let shared_bytes = 2 * WORKGROUP_SIZE * fragment_bytes;
    assert_eq!(shared_bytes, 8192);
    assert!(source.contains(&format!("static_shared_memory_bytes = {shared_bytes}")));
    assert!(source.contains("WorkgroupPipeline::<Bf16MfmaAFragment<'_>, 2, 256, 1>"));
    assert!(source.contains("WorkgroupPipeline::<Bf16MfmaBFragment<'_>, 2, 256, 1>"));
    assert!(source.contains("let pipeline_lane = global_index % crate::WORKGROUP_SIZE"));
    assert!(source.contains("while phase_index < 4"));
    assert_eq!(source.matches("_pipeline.stage(").count(), 4);
    assert_eq!(source.matches("_pipeline.write(").count(), 4);
    assert_eq!(source.matches("_pipeline.commit(").count(), 4);
    assert_eq!(source.matches("_pipeline.wait(").count(), 4);
    assert_eq!(source.matches("_pipeline.consume(").count(), 2);
    assert_eq!(source.matches("_pipeline.discard(").count(), 2);
    assert_eq!(source.matches("_pipeline.release(").count(), 4);
    assert_eq!(source.matches("multiply_accumulate(").count(), 1);
}

#[test]
fn materialized_components_preserve_all_three_exact_stage_exports() {
    let source = include_str!("../src/kernel_components.rs");
    for export in [
        "gfx950_gpt_oss_120b_router_v1",
        "gfx950_gpt_oss_120b_attention_v1",
        "gfx950_gpt_oss_120b_expert_v1",
    ] {
        assert!(source.contains(export));
    }
    assert_eq!(source.matches("multiply_accumulate(\n").count(), 4);
    assert_eq!(source.matches("multiply_accumulate_fp4(").count(), 4);
    assert_eq!(source.matches("max_grid = [4, 1, 1]").count(), 3);
    assert_eq!(source.matches("launch(required = [256, 1, 1]").count(), 3);
    assert_eq!(
        source
            .matches("let item_index = global_index / crate::WAVE_SIZE")
            .count(),
        3
    );
    assert!(!source.contains("unsafe"));
}

#[test]
fn router_selector_uses_the_exact_boolean_xor_complement() {
    let sources = [
        include_str!("../src/kernel.rs"),
        include_str!("../src/kernel_components.rs"),
        include_str!("../src/kernel_held_fragments.rs"),
        include_str!("../src/kernel_interleaved_stores.rs"),
        include_str!("../src/kernel_pipelined_attention.rs"),
        include_str!("../src/kernel_router_serial.rs"),
        include_str!("../src/kernel_scalar_attention.rs"),
    ];
    for source in sources {
        let compact = source.split_whitespace().collect::<String>();
        assert!(!source.contains("1 - take"));
        assert!(source.contains("take ^ 1"));
        assert!(compact.contains("wrapping_mul(take).wrapping_add"));
    }

    for predicate in [false, true] {
        let take = predicate as u32;
        assert!(take <= 1);
        assert_eq!(take ^ 1, 1 - take);
    }
}

#[cfg(any(
    feature = "kernel-gpt-oss-mode-simt",
    feature = "kernel-gpt-oss-mode-tile"
))]
#[test]
fn explicit_mode_markers_keep_the_four_existing_exports() {
    #[cfg(feature = "kernel-gpt-oss-mode-simt")]
    use fe2o3_gfx950_gpt_oss_decode::{kernel_simt as fused, kernel_simt_components as components};
    #[cfg(feature = "kernel-gpt-oss-mode-tile")]
    use fe2o3_gfx950_gpt_oss_decode::{kernel_tile as fused, kernel_tile_components as components};
    fn check<M: fe2o3_device::KernelMarkerV1>(name: &str) {
        assert_eq!(M::LOGICAL_NAME, name);
        assert_eq!(M::EXPORT_NAME, name);
    }
    check::<fused::gfx950_gpt_oss_120b_decode_megakernel_v1_gpu::Marker>(
        "gfx950_gpt_oss_120b_decode_megakernel_v1",
    );
    check::<components::gfx950_gpt_oss_120b_router_v1_gpu::Marker>("gfx950_gpt_oss_120b_router_v1");
    check::<components::gfx950_gpt_oss_120b_attention_v1_gpu::Marker>(
        "gfx950_gpt_oss_120b_attention_v1",
    );
    check::<components::gfx950_gpt_oss_120b_expert_v1_gpu::Marker>("gfx950_gpt_oss_120b_expert_v1");
}
