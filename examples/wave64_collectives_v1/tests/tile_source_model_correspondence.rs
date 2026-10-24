//! Structural/model review for the separate checked 8x8 output-tile source.
//! This is not source-to-model refinement, a Verus tile receipt, or device execution.
use fe2o3_wave64_collectives_v1::{
    OracleErrorV1, ReviewedSourceAlgorithmV2, SourceCpuOutputsV2, SourceStructureErrorV2,
    collect_reviewed_source_algorithm_v2, interpret_reviewed_source_algorithm_v2,
    wave64_collectives_oracle_v1,
};
use quote::ToTokens as _;
use sha2::{Digest as _, Sha256};
use std::{path::Path, process::Command};
const SOURCE: &str = include_str!("../src/tiled_kernel.rs");
const SIMT: &str = include_str!("../src/kernel.rs");
const EXPECTED: &str = r#"#![allow(missing_docs)]
use fe2o3_device::{
    DisjointSlice, Gfx942Collectives, Index1D, SubgroupTile, Tiled2D, Wave64, WaveLane, kernel, thread,
};
pub const WAVE64_COLLECTIVES_WORKGROUP_V1: [u32; 3] = [64, 1, 1];
#[kernel(typed, launch(required = [64, 1, 1], max = [64, 1, 1]))]
pub fn wave64_collectives_v1(
    input: &[f32], active_mask: u64,
    mut reduction_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
    mut inclusive_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
    mut exclusive_output: DisjointSlice<f32, Tiled2D<Index1D, 64, 8, 8, 1>>,
) {
    let lane_index = thread::index_1d();
    let lane = lane_index.get();
    if lane >= 64 || input.len() != 64 || reduction_output.len() != 64
        || inclusive_output.len() != 64 || exclusive_output.len() != 64 {
        fe2o3_device::trap();
    }
    let active = active_mask & (1_u64 << lane) != 0;
    let contribution = if active { input[lane] } else { 0.0_f32 };
    let lane_snapshot = WaveLane::<Wave64>::current();
    let wave = SubgroupTile::<64>::from_wave64_snapshot(&lane_snapshot);
    let context = Gfx942Collectives::current();
    let reduction = wave.reduce_sum(&context, contribution);
    let inclusive = wave.inclusive_scan_sum(&context, contribution);
    let exclusive = wave.exclusive_scan_sum(&context, contribution);
    let published_reduction = if active { reduction } else { 0.0 };
    let published_inclusive = if active { inclusive } else { 0.0 };
    let published_exclusive = if active { exclusive } else { 0.0 };
    let Some(tile) = thread::index_1d().checked_tiled_2d::<64, 8, 8, 1>() else {
        fe2o3_device::trap();
    };
    if let Some(output) = reduction_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_reduction;
    }
    if let Some(output) = inclusive_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_inclusive;
    }
    if let Some(output) = exclusive_output.get_tiled_2d_mut(&tile, 0, 8, 8, 8) {
        *output = published_exclusive;
    }
}
"#;

struct ReviewedTileAlgorithm {
    scalar: ReviewedSourceAlgorithmV2,
}
fn canonical(source: &str) -> Result<String, SourceStructureErrorV2> {
    let mut file =
        syn::parse_file(source).map_err(|_| SourceStructureErrorV2::InvalidRustSyntax)?;
    file.attrs.retain(|a| !a.path().is_ident("doc"));
    for item in &mut file.items {
        let attrs = match item {
            syn::Item::Const(x) => &mut x.attrs,
            syn::Item::Fn(x) => &mut x.attrs,
            syn::Item::Use(x) => &mut x.attrs,
            _ => continue,
        };
        attrs.retain(|a| !a.path().is_ident("doc"));
    }
    Ok(file.into_token_stream().to_string())
}
fn admit(source: &str) -> Result<ReviewedTileAlgorithm, SourceStructureErrorV2> {
    if canonical(source)? != canonical(EXPECTED).unwrap() {
        return Err(SourceStructureErrorV2::NonCanonicalSyntaxTree);
    }
    // Reuse the unchanged arithmetic abstraction only after admitting the exact
    // independently encoded tile syntax. No original formal receipt is ported.
    Ok(ReviewedTileAlgorithm {
        scalar: collect_reviewed_source_algorithm_v2(SIMT)?,
    })
}
fn interpret(
    tile: &ReviewedTileAlgorithm,
    input: &[f32],
    mask: u64,
) -> Result<SourceCpuOutputsV2, OracleErrorV1> {
    let scalar = interpret_reviewed_source_algorithm_v2(tile.scalar, input, mask)?;
    let mut result = SourceCpuOutputsV2 {
        reduction: [f32::NAN; 64],
        inclusive: [f32::NAN; 64],
        exclusive: [f32::NAN; 64],
    };
    let mut owners = [false; 64];
    for lane in 0..64 {
        // Exact Tiled2D<Index1D,64,8,8,1>, component0, rows8/cols8/stride8.
        let row = (lane / 64) * 8 + (lane % 64) / 8;
        let column = lane % 8;
        assert!(row < 8 && column < 8);
        let destination = row * 8 + column;
        assert!(!owners[destination]);
        owners[destination] = true;
        result.reduction[destination] = scalar.reduction[lane];
        result.inclusive[destination] = scalar.inclusive[lane];
        result.exclusive[destination] = scalar.exclusive[lane];
    }
    assert!(owners.iter().all(|&owner| owner));
    Ok(result)
}
fn corpus() -> [f32; 64] {
    core::array::from_fn(|lane| ((lane * 37 + 11) % 127) as f32 - 63.0)
}
fn masks() -> Vec<u64> {
    let mut masks = vec![
        0,
        u64::MAX,
        0xaaaa_aaaa_aaaa_aaaa,
        0x5555_5555_5555_5555,
        0x8000_0000_0000_0001,
    ];
    for lane in 0..64 {
        masks.push(1_u64 << lane);
        masks.push(!(1_u64 << lane));
    }
    for end in 0..=64 {
        let prefix = if end == 64 {
            u64::MAX
        } else {
            (1_u64 << end) - 1
        };
        masks.push(prefix);
        masks.push(!prefix);
    }
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    for _ in 0..4096 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        masks.push(state);
    }
    assert_eq!(masks.len(), 4359);
    masks
}
#[test]
fn tile_model_matches_full_independent_oracle_for_4359_masks_and_four_inputs() {
    let tile = admit(SOURCE).unwrap();
    for input in [
        corpus(),
        [0.0; 64],
        [-0.0; 64],
        core::array::from_fn(|lane| if lane % 2 == 0 { -1024.0 } else { 1024.0 }),
    ] {
        for mask in masks() {
            let actual = interpret(&tile, &input, mask).unwrap();
            let mut reduction = [f32::NAN; 64];
            let mut inclusive = [f32::NAN; 64];
            let mut exclusive = [f32::NAN; 64];
            wave64_collectives_oracle_v1(
                &input,
                mask,
                &mut reduction,
                &mut inclusive,
                &mut exclusive,
            )
            .unwrap();
            for lane in 0..64 {
                for (a, b) in [
                    (actual.reduction[lane], reduction[lane]),
                    (actual.inclusive[lane], inclusive[lane]),
                    (actual.exclusive[lane], exclusive[lane]),
                ] {
                    if mask & (1_u64 << lane) == 0 {
                        assert_eq!(a.to_bits(), 0);
                        assert_eq!(b.to_bits(), 0);
                    } else {
                        assert_eq!(a, b);
                    }
                }
            }
        }
    }
}
#[test]
fn tile_syntax_rejects_semantic_geometry_ownership_and_injected_mutations() {
    let mutations = [
        SOURCE.replacen("1_u64 << lane", "1_u64 << (lane ^ 1)", 1),
        SOURCE.replacen("input[lane]", "input[63 - lane]", 1),
        SOURCE.replacen("else { 0.0_f32 }", "else { -0.0_f32 }", 1),
        SOURCE.replacen("if active { exclusive }", "if active { inclusive }", 1),
        SOURCE.replacen(
            "if active { reduction } else { 0.0 }",
            "if active { reduction } else { -0.0 }",
            1,
        ),
        SOURCE.replacen(
            "wave.exclusive_scan_sum(&context, contribution)",
            "wave.exclusive_scan_sum(&context, -contribution)",
            1,
        ),
        SOURCE.replacen("input.len() != 64", "input.len() < 64", 1),
        SOURCE.replacen(
            "Tiled2D<Index1D, 64, 8, 8, 1>",
            "Tiled2D<Index1D, 64, 1, 64, 1>",
            1,
        ),
        SOURCE.replacen(
            "checked_tiled_2d::<64, 8, 8, 1>()",
            "checked_tiled_2d::<64, 1, 64, 1>()",
            1,
        ),
        SOURCE.replacen(
            "get_tiled_2d_mut(&tile, 0, 8, 8, 8)",
            "get_tiled_2d_mut(&tile, 1, 8, 8, 8)",
            1,
        ),
        SOURCE.replacen(
            "get_tiled_2d_mut(&tile, 0, 8, 8, 8)",
            "get_tiled_2d_mut(&tile, 0, 7, 8, 8)",
            1,
        ),
        SOURCE.replacen(
            "get_tiled_2d_mut(&tile, 0, 8, 8, 8)",
            "get_tiled_2d_mut(&tile, 0, 8, 7, 8)",
            1,
        ),
        SOURCE.replacen(
            "get_tiled_2d_mut(&tile, 0, 8, 8, 8)",
            "get_tiled_2d_mut(&tile, 0, 8, 8, 7)",
            1,
        ),
        SOURCE.replacen(
            "*output = published_inclusive",
            "*output = published_reduction",
            1,
        ),
        SOURCE.replacen(
            "wave.reduce_sum(&context, contribution)",
            "if active { wave.reduce_sum(&context, contribution) } else { 0.0 }",
            1,
        ),
        format!("{SOURCE}\nfn injected() {{}}"),
    ];
    for (i, mutation) in mutations.iter().enumerate() {
        assert_ne!(mutation, SOURCE, "mutation anchor{i} did not match");
        assert!(
            matches!(
                admit(mutation),
                Err(SourceStructureErrorV2::NonCanonicalSyntaxTree)
            ),
            "mutation{i} admitted"
        );
    }
    assert!(matches!(
        admit("pub fn broken("),
        Err(SourceStructureErrorV2::InvalidRustSyntax)
    ));
    assert!(admit(&format!("// comment\n{SOURCE}")).is_ok());
}
#[test]
fn committed_tree_contains_exact_tile_checker_oracle_and_feature_selection() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let head = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&repo)
        .output()
        .unwrap();
    assert!(head.status.success());
    let head = std::str::from_utf8(&head.stdout).unwrap().trim();
    let mut identity = Sha256::new();
    identity.update(b"fe2o3.wave64.output-tile.structural-review.v1\0");
    for (path, bytes) in [
        (
            "src/tiled_kernel.rs",
            include_bytes!("../src/tiled_kernel.rs").as_slice(),
        ),
        (
            "tests/tile_source_model_correspondence.rs",
            include_bytes!("tile_source_model_correspondence.rs").as_slice(),
        ),
        (
            "src/oracle.rs",
            include_bytes!("../src/oracle.rs").as_slice(),
        ),
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("src/lib.rs", include_bytes!("../src/lib.rs").as_slice()),
    ] {
        let object = format!("{head}:examples/wave64_collectives_v1/{path}");
        let actual = Command::new("git")
            .args(["show", &object])
            .current_dir(&repo)
            .output()
            .unwrap();
        assert!(actual.status.success(), "committed source missing{path}");
        assert_eq!(actual.stdout, bytes, "committed bytes differ{path}");
        identity.update(path.as_bytes());
        identity.update(Sha256::digest(bytes));
    }
    identity.update(head.as_bytes());
    assert_ne!(identity.finalize().as_slice(), [0_u8; 32]);
}
