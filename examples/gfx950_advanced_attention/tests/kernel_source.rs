use std::collections::BTreeSet;

use fe2o3_gfx950_advanced_attention::{
    GFX950_ADVANCED_ATTENTION_GRID_V1, GFX950_ADVANCED_ATTENTION_SOURCE_BLOCKER_V1,
    GFX950_ADVANCED_ATTENTION_SOURCE_LOWERING_SUPPORTED_V1, GFX950_ADVANCED_ATTENTION_WORKGROUP_V1,
};
use syn::{Item, Visibility};

const LIB_SOURCE: &str = include_str!("../src/lib.rs");
const CARGO_MANIFEST: &str = include_str!("../Cargo.toml");
const SOURCE: &str = include_str!("../src/kernel.rs");
const ABLATION_SOURCE: &str = include_str!("../src/ablation.rs");
const ABLATION_REGISTRY: &str = include_str!("../ablation-variants-v1.json");
const RUNNER_SOURCE: &str = include_str!("../run-gfx950.sh");

#[test]
fn source_contains_the_seven_expected_typed_kernels() {
    let file = syn::parse_file(SOURCE).expect("kernel source parses as ordinary Rust");
    let kernels: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function)
                if function.attrs.iter().any(|attribute| {
                    attribute.path().is_ident("kernel")
                        || (attribute.path().is_ident("cfg_attr")
                            && attribute
                                .meta
                                .require_list()
                                .is_ok_and(|list| list.tokens.to_string().contains("kernel")))
                }) =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect();
    let names: BTreeSet<_> = kernels
        .iter()
        .map(|function| function.sig.ident.to_string())
        .collect();
    assert_eq!(
        names,
        BTreeSet::from([
            "gfx950_attnres_aggregate".to_string(),
            "gfx950_compressed_hybrid_attention".to_string(),
            "gfx950_content_sparse_attention".to_string(),
            "gfx950_four_branch_residual".to_string(),
            "gfx950_kda_gdn_decode".to_string(),
            "gfx950_kda_gdn_prefill".to_string(),
            "gfx950_mhc_sinkhorn_mix".to_string(),
        ])
    );
    assert_eq!(kernels.len(), 7);
    for function in kernels {
        assert!(matches!(function.vis, Visibility::Public(_)));
        assert!(function.sig.unsafety.is_none());
        let attributes = function
            .attrs
            .iter()
            .filter_map(|attribute| attribute.meta.require_list().ok())
            .map(|list| list.tokens.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(attributes.contains("typed"));
        assert!(attributes.contains("namespace"));
        assert!(attributes.contains("required = [64 , 1 , 1]"));
        assert!(attributes.contains("max = [64 , 1 , 1]"));
        assert!(attributes.contains("max_grid = [1 , 1 , 1]"));
    }
}

#[test]
fn source_contains_the_four_standalone_ablation_kernels() {
    let file = syn::parse_file(ABLATION_SOURCE).expect("ablation source parses as ordinary Rust");
    let kernels: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(function)
                if function
                    .attrs
                    .iter()
                    .any(|attribute| attribute.path().is_ident("kernel")) =>
            {
                Some(function)
            }
            _ => None,
        })
        .collect();
    assert_eq!(kernels.len(), 4);
    assert_eq!(
        kernels
            .iter()
            .map(|function| function.sig.ident.to_string())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "gfx950_attnres_aggregate".to_string(),
            "gfx950_four_branch_residual".to_string(),
            "gfx950_kda_gdn_decode".to_string(),
            "gfx950_mhc_sinkhorn_mix".to_string(),
        ])
    );
    assert!(!ABLATION_SOURCE.contains("unsafe"));
    assert!(
        !ABLATION_SOURCE
            .to_ascii_lowercase()
            .contains("extern \"c\"")
    );
}

#[test]
fn source_is_safe_fixed_shape_rust_without_hip_escape_hatches() {
    let lowercase = SOURCE.to_ascii_lowercase();
    assert!(!SOURCE.contains("unsafe"));
    assert!(!SOURCE.contains("include!"));
    assert_eq!(SOURCE.matches("macro_rules!").count(), 2);
    assert!(SOURCE.contains("#[cfg(target_arch = \"amdgpu\")]\nmacro_rules! decode_fp8_e4m3_v1"));
    assert!(
        SOURCE.contains(
            "#[cfg(target_arch = \"amdgpu\")]\nmacro_rules! consider_sparse_candidate_v1"
        )
    );
    assert!(!lowercase.contains("extern \"c\""));
    assert!(!lowercase.contains("hiplaunchkernel"));
    assert!(!lowercase.contains("std::process"));
    for marker in [
        "KDA_TAPS_V1",
        "PREFILL_TOKENS_V1",
        "ATTENTION_TOKENS_V1",
        "HEAD_DIMENSION_V1",
        "SELECTED_TOKENS_V1",
        "SINKHORN_ITERATIONS_V1",
        "thread::grid_leader()",
        "get_mut_exclusive",
        "math.exp_f32(-2.0 * value)",
        "exponent == 15 && mantissa == 7.0",
        "Gfx950Fp8MfmaAMatrix::row_major",
        "stage_k_transposed",
        "read_mfma_fragment",
        "multiply_accumulate_fp8",
        "reduce_sum_f32::<16>",
        "broadcast_f32::<16>",
    ] {
        assert!(
            SOURCE.contains(marker),
            "missing fixed source marker {marker}"
        );
    }
}

#[test]
fn package_states_the_production_source_and_evidence_boundary() {
    assert_eq!(GFX950_ADVANCED_ATTENTION_WORKGROUP_V1, [64, 1, 1]);
    assert_eq!(GFX950_ADVANCED_ATTENTION_GRID_V1, [1, 1, 1]);
    assert!(GFX950_ADVANCED_ATTENTION_SOURCE_LOWERING_SUPPORTED_V1);
    assert!(
        LIB_SOURCE.contains("GFX950_ADVANCED_ATTENTION_SOURCE_LOWERING_SUPPORTED_V1: bool = true")
    );
    assert!(GFX950_ADVANCED_ATTENTION_SOURCE_BLOCKER_V1.contains("formal compiler refinement"));
    assert!(GFX950_ADVANCED_ATTENTION_SOURCE_BLOCKER_V1.contains("protected publication"));

    for feature in [
        "kernel-kda-decode",
        "kernel-kda-prefill",
        "kernel-content-sparse-attention",
        "kernel-compressed-hybrid-attention",
        "kernel-attnres-aggregate",
        "kernel-four-branch-residual",
        "kernel-mhc-sinkhorn-mix",
    ] {
        assert!(LIB_SOURCE.contains(feature));
    }

    for variant in [
        "kernel-kda-decode-wave-tiled-v1",
        "kernel-kda-prefill-channel-mask-v1",
        "kernel-content-sparse-attention-reciprocal-reuse-v1",
        "kernel-compressed-hybrid-attention-division-baseline-v1",
        "kernel-attnres-aggregate-explicit-reuse-v1",
        "kernel-four-branch-residual-explicit-v1",
        "kernel-mhc-sinkhorn-mix-scalar-v1",
    ] {
        assert!(
            CARGO_MANIFEST.contains(variant),
            "missing feature {variant}"
        );
        assert!(
            LIB_SOURCE.contains(variant) || SOURCE.contains(variant),
            "missing source gate {variant}"
        );
        assert!(
            RUNNER_SOURCE.contains(variant),
            "missing runner case {variant}"
        );
        assert!(
            ABLATION_REGISTRY.contains(variant),
            "missing registry entry {variant}"
        );
    }

    assert!(SOURCE.contains("kernel-content-sparse-attention-reciprocal-reuse-v1"));
    assert!(SOURCE.contains("kernel-compressed-hybrid-attention-division-baseline-v1"));
    for rejected in [
        "content-sparse-selected-score-scalar-v1",
        "compressed-hybrid-seven-score-scalar-v1",
        "attention-lds-double-buffer-v1",
        "kda-prefill-ping-pong-v1",
        "mixing-lds-staging-v1",
    ] {
        assert!(
            ABLATION_REGISTRY.contains(rejected),
            "missing rejected variant {rejected}"
        );
    }
}
