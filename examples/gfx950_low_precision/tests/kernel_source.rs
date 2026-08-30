use std::fs;
use std::path::Path;

#[test]
fn gfx950_kernels_are_safe_attributed_rust_with_typed_operations() {
    let source = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernel.rs"))
        .expect("read Rust kernel source");
    assert_eq!(source.matches("#[kernel(").count(), 4);
    assert_eq!(source.matches("typed,").count(), 4);
    assert_eq!(source.matches("Blocked<Index1D, 16, 4>").count(), 4);
    assert_eq!(source.matches("checked_block::<16, 4>").count(), 4);
    assert_eq!(source.matches("get_block_mut").count(), 16);
    assert!(!source.contains("checked_tiled_2d"));
    assert!(!source.contains("get_tiled_2d_mut"));
    assert!(source.contains("multiply_accumulate_fp4"));
    assert!(source.contains("multiply_accumulate_fp8"));
    assert!(source.contains("Gfx950LdsTransposeTile"));
    assert!(source.contains("stage_k_transposed"));
    assert!(source.contains("read_mfma_fragment"));
    assert!(source.contains("Gfx950Subgroup::current"));
    // Each attention kernel materializes four query rows across all 16 value
    // lanes so the semantic importer sees a fixed, statically ranked graph.
    assert_eq!(source.matches("broadcast_f32::<16>").count(), 2 * 4 * 16);
    assert_eq!(source.matches("value.load_or(").count(), 2 * 16);
    assert!(!source.contains("unsafe"));
    assert!(!source.contains("asm!"));
    assert!(!source.contains("__builtin_amdgcn"));
}

#[test]
fn hip_fixture_is_not_imported_by_the_rust_package() {
    let manifest = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("read package manifest");
    assert!(!manifest.contains("build ="));
    assert!(!manifest.contains("gfx950_low_precision.hip"));
}
