use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

struct ScratchTarget {
    path: PathBuf,
}

impl ScratchTarget {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fe2o3-production-ranked-bounds-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create ranked bounds target directory");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchTarget {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn ranked_bounds_fixture_line(containing: &str) -> usize {
    include_str!("fixtures/production-ranked-bounds-device/src/lib.rs")
        .lines()
        .position(|line| line.contains(containing))
        .map(|index| index + 1)
        .unwrap_or_else(|| panic!("ranked-bounds fixture omitted {containing:?}"))
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_rust_bounds_and_production_pliron_pipeline_fail_closed() {
    let safe = run_extraction(&ScratchTarget::new(), false);
    assert!(
        safe.status.success()
            && safe
                .stderr
                .contains("all mandatory kernel checks clean true")
            && safe.stderr.contains("kernel.cond_br")
            && safe.stderr.contains("kernel.access Write")
            && !safe.stderr.contains("error[FE2O3-BOUNDS-001]"),
        "safe checked dynamic access did not pass generic PLIRON verification:\n{}",
        safe.stderr
    );

    let shifted = run_feature_extraction(&ScratchTarget::new(), "shifted");
    assert!(
        shifted.status.success()
            && shifted
                .stderr
                .contains("all mandatory kernel checks clean true")
            && shifted.stderr.contains("kernel.index_binary Add")
            && shifted.stderr.contains("kernel.cond_br")
            && shifted.stderr.contains("kernel.access Write"),
        "safe shifted disjoint access did not pass production extraction:\n{}",
        shifted.stderr,
    );

    let exclusive = run_feature_extraction(&ScratchTarget::new(), "grid_exclusive");
    assert!(
        exclusive.status.success()
            && exclusive
                .stderr
                .contains("all mandatory kernel checks clean true")
            && exclusive.stderr.contains("kernel.index_constant 7")
            && exclusive.stderr.contains("kernel.cond_br")
            && exclusive.stderr.contains("kernel.access Write"),
        "safe grid-exclusive access did not pass production extraction:\n{}",
        exclusive.stderr,
    );

    let blocked = run_feature_extraction(&ScratchTarget::new(), "blocked");
    assert!(
        blocked.status.success()
            && blocked
                .stderr
                .contains("all mandatory kernel checks clean true")
            && blocked.stderr.contains("kernel.index_binary Multiply")
            && blocked.stderr.contains("kernel.index_binary Add")
            && blocked.stderr.contains("kernel.access Write"),
        "safe blocked disjoint access did not pass production extraction:\n{}",
        blocked.stderr,
    );

    let blocked_multi_lane = run_feature_extraction(&ScratchTarget::new(), "blocked_multi_lane");
    assert!(
        blocked_multi_lane.status.success()
            && blocked_multi_lane
                .stderr
                .contains("all mandatory kernel checks clean true")
            && blocked_multi_lane
                .stderr
                .contains("kernel.index_constant 192")
            && blocked_multi_lane
                .stderr
                .contains("kernel.index_binary Add")
            && blocked_multi_lane.stderr.contains("kernel.access Write"),
        "bounded multi-lane blocked access did not pass production extraction:\n{}",
        blocked_multi_lane.stderr,
    );

    let blocked_multi_block = run_feature_extraction(&ScratchTarget::new(), "blocked_multi_block");
    assert!(
        blocked_multi_block.status.success()
            && blocked_multi_block
                .stderr
                .contains("all mandatory kernel checks clean true")
            && blocked_multi_block
                .stderr
                .contains("kernel.index_constant 16")
            && blocked_multi_block
                .stderr
                .contains("kernel.index_constant 64")
            && blocked_multi_block
                .stderr
                .contains("kernel.index_binary Divide")
            && blocked_multi_block
                .stderr
                .contains("kernel.index_binary Remainder")
            && blocked_multi_block.stderr.contains("kernel.access Write"),
        "bounded multi-block access did not pass production extraction:\n{}",
        blocked_multi_block.stderr,
    );

    let dynamic_multi_lane =
        run_feature_extraction(&ScratchTarget::new(), "blocked_multi_lane_dynamic_grid");
    assert!(
        !dynamic_multi_lane.status.success()
            && dynamic_multi_lane.stderr.contains(
                "a multi-lane blocked mapping requires an authenticated finite rank-1 launch extent"
            ),
        "dynamic-grid multi-lane blocked access did not fail closed:\n{}",
        dynamic_multi_lane.stderr,
    );

    let oob = run_extraction(&ScratchTarget::new(), true);
    assert!(
        !oob.status.success(),
        "out-of-bounds Rust kernel was accepted"
    );
    let oob_source_location = format!(
        ":{}:20",
        ranked_bounds_fixture_line("let selected = input[64];")
    );
    assert!(
        oob.stderr.contains("error[FE2O3-BOUNDS-001]")
            && oob.stderr.contains("required: 64 < 64")
            && oob.stderr.contains("Rust source")
            && oob.stderr.contains(&oob_source_location)
            && oob.stderr.contains("kernel.index_constant 64")
            && oob
                .stderr
                .contains("ranked PLIRON before rejected lowering")
            && oob
                .stderr
                .contains("lowering stopped before target IR or artifact emission"),
        "out-of-bounds diagnostic was incomplete:\n{}",
        oob.stderr,
    );
    for forbidden in ["kernel-ir-v1", "GeneralGemm", "Unknown/Unproved"] {
        assert!(
            !safe.stderr.contains(forbidden) && !oob.stderr.contains(forbidden),
            "production extraction entered forbidden path {forbidden:?}",
        );
    }
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn optimized_blocked_accessor_retains_its_checked_terminal() {
    let blocked = run_release_feature_extraction(&ScratchTarget::new(), "blocked");
    assert!(
        blocked.status.success()
            && blocked
                .stderr
                .contains("all mandatory kernel checks clean true")
            && blocked.stderr.contains("kernel.index_binary Multiply")
            && blocked.stderr.contains("kernel.index_binary Add")
            && blocked.stderr.contains("kernel.access Write"),
        "optimized blocked access lost its checked terminal:\n{}",
        blocked.stderr,
    );
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn production_barrier_cfg_preserves_order_and_fails_closed() {
    for feature in ["barrier_after_access", "barrier_before_access"] {
        let output = run_feature_extraction(&ScratchTarget::new(), feature);
        assert!(
            output.status.success()
                && output
                    .stderr
                    .contains("all mandatory kernel checks clean true")
                && output.stderr.contains("kernel.access Write")
                && output.stderr.contains("gpu.barrier"),
            "{feature} did not preserve a clean ranked CFG:\n{}",
            output.stderr,
        );
    }

    for feature in ["barrier_divergent", "barrier_early_return"] {
        let output = run_feature_extraction(&ScratchTarget::new(), feature);
        assert!(
            !output.status.success()
                && output.stderr.contains("error[FE2O3-BARRIER-001]")
                && output.stderr.contains("divergent collective barrier paths"),
            "{feature} did not fail closed as divergent:\n{}",
            output.stderr,
        );
    }

    let cyclic = run_feature_extraction(&ScratchTarget::new(), "barrier_loop");
    assert!(
        !cyclic.status.success()
            && cyclic.stderr.contains("error[FE2O3-BARRIER-002]")
            && cyclic.stderr.contains("cyclic control flow"),
        "cyclic barrier did not remain incomplete:\n{}",
        cyclic.stderr,
    );

    let helper = run_feature_extraction(&ScratchTarget::new(), "barrier_helper");
    assert!(
        !helper.status.success()
            && helper.stderr.contains(
                "a call terminator before exact callable memory-effect summaries are available"
            ),
        "helper-mediated barrier bypassed the semantic boundary:\n{}",
        helper.stderr,
    );
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_kernel_source_exports_one_verified_authority_free_simulation_bundle() {
    let target = ScratchTarget::new();
    let bundle_path = target.path().join("copy-static.fe2sim");
    let result = output(
        simulation_export_command("gfx942", &bundle_path, target.path()),
        "run production simulation-bundle extraction",
    );

    assert!(
        result.status.success(),
        "ordinary source simulation-bundle extraction failed:\n{}",
        result.stderr,
    );
    assert!(
        result
            .stderr
            .contains("sole target-neutral Kernel IR lowering")
            && result
                .stderr
                .contains("exact verified KIR V7 simulation bundle")
            && result
                .stderr
                .contains("compiler_execution_binding=extraction_only_unavailable")
            && result
                .stderr
                .contains("authenticates_compiler_execution=false")
            && result
                .stderr
                .contains("proof/artifact/compiler/hardware/load/launch authority false"),
        "simulation-bundle diagnostic overclaimed or omitted its transaction boundary:\n{}",
        result.stderr,
    );
    let bytes = std::fs::read(&bundle_path).expect("read exact simulation bundle");
    let bundle = fe2o3_kernel_ir::VerifiedSimulationBundleV1::from_canonical_bytes(bytes)
        .expect("decode compiler-produced simulation bundle");
    assert_eq!(bundle.target(), "gfx942:xnack-");
    assert_eq!(bundle.kernel_count(), 1);
    assert_eq!(
        bundle.compiler_execution_binding(),
        &fe2o3_kernel_ir::SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly
    );
    assert!(
        bundle
            .require_canonical_compiler_execution_association()
            .is_err()
    );
    assert!(
        bundle
            .source_lineage()
            .rustc_identity_inventory_receipt_bytes()
            > 0
    );
    assert!(bundle.source_lineage().rustc_preflight_plan_receipt_bytes() > 0);
    let map_bytes = bundle
        .debug_map()
        .expect("compiler extraction embeds one exact source map");
    let map = fe2o3_kernel_ir::DebugSourceMapDocumentV1::from_json_bytes(map_bytes)
        .expect("compiler source map uses the strict shared codec");
    assert_eq!(
        map.binding().bundle_subject_identity(),
        *bundle.subject_identity()
    );
    assert_eq!(
        map.binding().canonical_kir().digest(),
        *bundle.canonical_kir_v7_identity().digest()
    );
    assert_eq!(
        map.binding().canonical_kir().canonical_bytes(),
        bundle.canonical_kir_v7_identity().canonical_length()
    );
    assert_eq!(
        bundle.debug_map_identity(),
        Some(fe2o3_kernel_ir::simulation_debug_map_identity_v1(map_bytes))
    );
    let source_file = map
        .files()
        .iter()
        .find(|file| {
            file.display_path()
                .ends_with("production-ranked-bounds-device/src/lib.rs")
        })
        .expect("map retains the ordinary-source display path");
    assert_eq!(
        source_file.byte_len(),
        std::fs::metadata(workspace().join(
            "crates/rustc-codegen-fe2o3/tests/fixtures/production-ranked-bounds-device/src/lib.rs",
        ))
        .unwrap()
        .len()
    );
    assert!(!map.sites().is_empty());
    assert!(!map.eliminated().is_empty());
    assert!(map.sites().windows(2).any(|sites| {
        sites[0].site().function_ordinal() == sites[1].site().function_ordinal()
            && sites[0].site().block_ordinal() == sites[1].site().block_ordinal()
            && sites[0].site().operation_ordinal().checked_add(1)
                == Some(sites[1].site().operation_ordinal())
            && sites[0].spans() == sites[1].spans()
    }));
    assert!(!bundle.canonical_kir_v7().is_empty());
    assert!(!bundle.grants_proof_authority());
    assert!(!bundle.grants_artifact_authority());
    assert!(!bundle.grants_compiler_authority());
    assert!(!bundle.grants_hardware_authority());
    assert!(!bundle.grants_load_authority());
    assert!(!bundle.grants_launch_authority());
    assert!(!bundle.authenticates_compiler_execution());

    let simulation_request_path = target.path().join("request.json");
    std::fs::write(
        &simulation_request_path,
        serde_json::to_vec(&json!({
            "schema": "fe2o3-simulation-request-v1",
            "kernel": "barrier_before_access",
            "grid": [1, 1, 1],
            "workgroup": [64, 1, 1],
            "arguments": [{
                "kind": "buffer",
                "element": "f32",
                "access": "read_write",
                "alignment": 4,
                "bytes": format!("0x{}", "00".repeat(64 * 4)),
            }],
        }))
        .unwrap(),
    )
    .unwrap();
    let (mapped_site, breakpoint_byte) = map
        .sites()
        .iter()
        .find_map(|site| {
            site.spans()
                .iter()
                .filter(|span| span.file_identity() == source_file.identity())
                .find_map(|span| {
                    (span.byte_start()..span.byte_end())
                        .find(|byte| {
                            map.sites()
                                .iter()
                                .flat_map(|other| other.spans())
                                .filter(|other_span| {
                                    other_span.file_identity() == source_file.identity()
                                        && other_span.byte_start() < *byte + 1
                                        && *byte < other_span.byte_end()
                                })
                                .count()
                                == 1
                        })
                        .map(|byte| (site, byte))
                })
        })
        .expect("ordinary source has at least one unambiguous mapped source byte");
    let map_identity = bundle.debug_map_identity().unwrap();
    let site = mapped_site.site();
    let mut protocol_input = Vec::new();
    for request in [
        json!({
            "operation": "resolve_source",
            "schema": "fe2o3-debug-request-v1",
            "request_id": 1,
            "expected_revision": 0,
            "site": {
                "function_ordinal": site.function_ordinal(),
                "block_ordinal": site.block_ordinal(),
                "point": {
                    "kind": "operation",
                    "operation_ordinal": site.operation_ordinal(),
                },
            },
        }),
        json!({
            "operation": "set_breakpoints",
            "schema": "fe2o3-debug-request-v1",
            "request_id": 2,
            "expected_revision": 0,
            "breakpoints": [{
                "enabled": true,
                "kind": {
                    "kind": "source",
                    "source": {
                        "map_identity": hex(&map_identity),
                        "provenance": "compiler_bundle_bound",
                        "file_identity": hex(&source_file.identity()),
                        "byte_start": breakpoint_byte,
                        "byte_end": breakpoint_byte + 1,
                    },
                },
            }],
        }),
        json!({
            "operation": "continue",
            "schema": "fe2o3-debug-request-v1",
            "request_id": 3,
            "expected_revision": 1,
            "max_events": 65536,
        }),
        json!({
            "operation": "inspect_stack",
            "schema": "fe2o3-debug-request-v1",
            "request_id": 4,
            "expected_revision": 2,
            "scope": { "level": "dispatch" },
            "page": { "limit": 16 },
        }),
        json!({
            "operation": "step",
            "schema": "fe2o3-debug-request-v1",
            "request_id": 5,
            "expected_revision": 2,
            "direction": "forward",
            "granularity": "source",
            "count": 1,
        }),
    ] {
        serde_json::to_writer(&mut protocol_input, &request).unwrap();
        protocol_input.push(b'\n');
    }

    let debug_target = target.path().join("debug-cli-target");
    let build_debugger = Command::new(env!("CARGO"))
        .current_dir(workspace())
        .args([
            "build",
            "--quiet",
            "--locked",
            "-p",
            "fe2o3-debug-cli",
            "--bin",
            "fe2o3-debug",
            "--target-dir",
        ])
        .arg(&debug_target)
        .output()
        .expect("build standalone debugger for compiler-output integration");
    assert!(
        build_debugger.status.success(),
        "debugger build failed:\n{}",
        String::from_utf8_lossy(&build_debugger.stderr)
    );
    let mut debugger = Command::new(debug_target.join("debug/fe2o3-debug"))
        .args(["sim", "--bundle"])
        .arg(&bundle_path)
        .arg("--request")
        .arg(&simulation_request_path)
        .args(["--protocol", "jsonl", "--wave-width", "64"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("run debugger on compiler-produced simulation bundle");
    debugger
        .stdin
        .take()
        .unwrap()
        .write_all(&protocol_input)
        .unwrap();
    let debug_output = debugger.wait_with_output().unwrap();
    assert!(
        debug_output.status.success(),
        "debugger rejected compiler output:\n{}",
        String::from_utf8_lossy(&debug_output.stderr)
    );
    assert!(debug_output.stderr.is_empty());
    let responses = debug_output
        .stdout
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), 5);
    assert!(
        responses.iter().all(|response| response["status"] == "ok"),
        "debugger returned a typed error response: {responses:#?}",
    );
    assert_eq!(responses[0]["result"]["result"], "source");
    assert_eq!(
        responses[0]["result"]["site"]["source"]["location"]["provenance"],
        "compiler_bundle_bound"
    );
    assert_eq!(
        responses[0]["result"]["site"]["source"]["location"]["map_identity"],
        hex(&map_identity)
    );
    assert_eq!(responses[2]["result"]["stop"]["reason"], "breakpoint");
    assert_eq!(responses[3]["result"]["result"], "stack");
    assert!(
        responses[3]["result"]["frames"]
            .as_array()
            .is_some_and(|frames| !frames.is_empty())
    );
    assert_eq!(responses[4]["operation"], "step");
    assert!(responses.iter().all(|response| {
        response["session"]["simulated"] == true
            && response["session"]["hardware_observed"] == false
            && response["session"]["performance_prediction"] == false
    }));
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_kernel_source_exports_the_exact_gfx950_simulation_target() {
    let target = ScratchTarget::new();
    let bundle_path = target.path().join("gfx950.fe2sim");
    let result = output(
        simulation_export_command("gfx950", &bundle_path, target.path()),
        "run gfx950 production simulation-bundle extraction",
    );

    assert!(
        result.status.success() && result.stderr.contains("target gfx950:xnack-"),
        "gfx950 ordinary source simulation-bundle extraction failed:\n{}",
        result.stderr,
    );
    let bundle = fe2o3_kernel_ir::VerifiedSimulationBundleV1::from_canonical_bytes(
        std::fs::read(&bundle_path).expect("read gfx950 simulation bundle"),
    )
    .expect("decode compiler-produced gfx950 simulation bundle");
    assert_eq!(bundle.target(), "gfx950:xnack-");
    assert_eq!(bundle.kernel_count(), 1);
    assert!(bundle.debug_map().is_some());
    assert!(!bundle.authenticates_compiler_execution());
    assert!(!bundle.grants_hardware_authority());
    assert!(!bundle.grants_launch_authority());
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_kernel_sources_export_and_query_exact_v2_source_variables() {
    let target = ScratchTarget::new();
    let debug_target = target.path().join("debug-cli-target");
    let build_debugger = Command::new(env!("CARGO"))
        .current_dir(workspace())
        .args([
            "build",
            "--quiet",
            "--locked",
            "-p",
            "fe2o3-debug-cli",
            "--bin",
            "fe2o3-debug",
            "--target-dir",
        ])
        .arg(&debug_target)
        .output()
        .expect("build debugger for production V2 source-variable integration");
    assert!(
        build_debugger.status.success(),
        "debugger build failed:\n{}",
        String::from_utf8_lossy(&build_debugger.stderr)
    );

    let cases = [
        (
            "debug_scalar",
            "debug_scalar",
            json!([
                {"kind": "scalar", "type": "f32", "bits": "0x3f800000"},
                {
                    "kind": "buffer",
                    "element": "f32",
                    "access": "read_only",
                    "alignment": 4,
                    "bytes": format!("0x{}", "00".repeat(64 * 4)),
                },
                {
                    "kind": "buffer",
                    "element": "f32",
                    "access": "read_write",
                    "alignment": 4,
                    "bytes": format!("0x{}", "00".repeat(64 * 4)),
                },
            ]),
            &["value", "input"][..],
            &["output", "element"][..],
        ),
        (
            "shifted",
            "checked_shifted",
            json!([{
                "kind": "buffer",
                "element": "f32",
                "access": "read_write",
                "alignment": 4,
                "bytes": format!("0x{}", "00".repeat(68 * 4)),
            }]),
            &[][..],
            &["output", "index"][..],
        ),
        (
            "debug_mutated_argument",
            "debug_mutated_argument",
            json!([
                {"kind": "scalar", "type": "f32", "bits": "0x3f800000"},
                {
                    "kind": "buffer",
                    "element": "f32",
                    "access": "read_write",
                    "alignment": 4,
                    "bytes": format!("0x{}", "00".repeat(64 * 4)),
                },
            ]),
            &[][..],
            &["value", "output"][..],
        ),
    ];
    let mut exported = Vec::new();
    for (feature, kernel, arguments, parameter_names, unrepresented_names) in cases {
        let bundle_path = target.path().join(format!("{feature}-v2.fe2sim"));
        let export_target = target.path().join(format!("{feature}-export-target"));
        let result = output(
            simulation_export_command_v2("gfx942", &bundle_path, &export_target, feature),
            "run production V2 simulation-bundle extraction",
        );
        assert!(
            result.status.success()
                && result.stderr.contains("explicit simulation bundle V2")
                && result.stderr.contains("compiler-produced source variables")
                && result
                    .stderr
                    .contains("compiler_execution_binding=extraction_only_unavailable")
                && result
                    .stderr
                    .contains("authenticates_compiler_execution=false")
                && result
                    .stderr
                    .contains("proof/artifact/compiler/hardware/load/launch authority false"),
            "production V2 export failed or overclaimed authority for {feature}:\n{}",
            result.stderr,
        );
        let bundle = fe2o3_kernel_ir::VerifiedSimulationBundleV2::from_canonical_bytes(
            std::fs::read(&bundle_path).expect("read production V2 bundle"),
        )
        .expect("decode production V2 bundle");
        let map = fe2o3_kernel_ir::DebugSourceMapDocumentV2::from_canonical_json_bytes(
            bundle.debug_map(),
        )
        .expect("decode compiler-produced source-variable map");
        assert_eq!(
            map.binding().bundle_subject_identity(),
            *bundle.inner_v1().subject_identity()
        );
        assert_eq!(
            map.binding().canonical_kir().digest(),
            *bundle.inner_v1().canonical_kir_v7_identity().digest()
        );
        assert!(!bundle.authenticates_compiler_execution());
        assert!(!bundle.grants_compiler_authority());
        assert!(!bundle.grants_load_authority());
        assert!(!bundle.grants_launch_authority());
        for name in parameter_names {
            let variable = map
                .variables()
                .iter()
                .find(|variable| variable.name() == *name)
                .unwrap_or_else(|| panic!("missing exact {name} parameter in {feature} map"));
            let binding = variable
                .function_binding()
                .unwrap_or_else(|| panic!("{name} is not bound to an exact KIR parameter"));
            assert_eq!(binding.generation(), 1);
            assert!(variable.locations().is_empty());
        }
        for name in unrepresented_names {
            let unavailable = map
                .variables()
                .iter()
                .find(|variable| variable.name() == *name)
                .unwrap_or_else(|| {
                    panic!("missing exact {name} unavailable variable in {feature} map")
                });
            assert_eq!(
                unavailable.fallback(),
                fe2o3_kernel_ir::DebugSourceVariableFallbackV2::Unrepresented
            );
            assert!(unavailable.function_binding().is_none());
        }

        let request_path = target.path().join(format!("{feature}-request.json"));
        std::fs::write(
            &request_path,
            serde_json::to_vec(&json!({
                "schema": "fe2o3-simulation-request-v1",
                "kernel": kernel,
                "grid": [1, 1, 1],
                "workgroup": [64, 1, 1],
                "arguments": arguments,
            }))
            .unwrap(),
        )
        .unwrap();
        let protocol_input = concat!(
            "{\"operation\":\"step\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":1,\"expected_revision\":0,\"direction\":\"forward\",\"granularity\":\"operation\",\"count\":1}\n",
            "{\"operation\":\"inspect_source_variables\",\"schema\":\"fe2o3-debug-source-variable-request-v2\",\"request_id\":2,\"expected_revision\":1,\"scope\":{\"level\":\"dispatch\"},\"frame\":1,\"selector\":{\"selector\":\"all\"},\"page\":{\"limit\":64}}\n",
        );
        let mut debugger = Command::new(debug_target.join("debug/fe2o3-debug"))
            .args(["sim", "--bundle-v2"])
            .arg(&bundle_path)
            .arg("--request")
            .arg(&request_path)
            .args(["--protocol", "jsonl", "--wave-width", "64"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("run debugger on compiler-produced V2 bundle");
        debugger
            .stdin
            .take()
            .unwrap()
            .write_all(protocol_input.as_bytes())
            .unwrap();
        let debug_output = debugger.wait_with_output().unwrap();
        assert!(
            debug_output.status.success(),
            "debugger rejected {feature} production bundle:\n{}",
            String::from_utf8_lossy(&debug_output.stderr)
        );
        assert!(debug_output.stderr.is_empty());
        let responses = debug_output
            .stdout
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["status"], "ok");
        assert_eq!(responses[1]["status"], "ok");
        let values = responses[1]["values"]
            .as_array()
            .expect("source-variable response has values");
        for name in parameter_names {
            let value = values
                .iter()
                .find(|value| value["name"] == *name)
                .unwrap_or_else(|| panic!("debugger omitted {name} for {feature}"));
            assert_eq!(value["generation"], 1);
            assert_eq!(value["availability"]["status"], "value");
            assert_eq!(value["availability"]["value"]["status"], "captured");
            assert_eq!(
                value["availability"]["value"]["provenance"],
                "simulated_observation"
            );
            assert_ne!(
                value["availability"]["value"]["value"]["encoding"],
                "native_address"
            );
            if *name == "input" {
                assert_eq!(
                    value["availability"]["value"]["value"]["encoding"],
                    "allocation_relative_pointer"
                );
            }
        }
        for name in unrepresented_names {
            let unavailable = values
                .iter()
                .find(|value| value["name"] == *name)
                .unwrap_or_else(|| panic!("debugger omitted {name} for {feature}"));
            assert_eq!(unavailable["availability"]["status"], "value");
            assert_eq!(
                unavailable["availability"]["value"]["status"],
                "unavailable"
            );
            assert_eq!(
                unavailable["availability"]["value"]["reason"],
                "not_represented"
            );
        }
        exported.push((bundle.inner_v1().canonical_bytes().to_vec(), map));
    }

    let stale_inner =
        fe2o3_kernel_ir::VerifiedSimulationBundleV1::from_canonical_bytes(exported[0].0.clone())
            .unwrap();
    let stale_map = exported.pop().unwrap().1;
    assert!(matches!(
        fe2o3_kernel_ir::VerifiedSimulationBundleV2::new(stale_inner, stale_map),
        Err(fe2o3_kernel_ir::SimulationBundleErrorV2::DebugMapBindingMismatch)
    ));
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn v2_rejects_an_overbound_debug_name_without_inspecting_it_on_v1() {
    let target = ScratchTarget::new();
    let default_path = target.path().join("debug-long-name-default-v1.fe2sim");
    let explicit_path = target.path().join("debug-long-name-explicit-v1.fe2sim");
    let default = output(
        simulation_export_command_for_feature(
            "gfx942",
            &default_path,
            &target.path().join("default-v1-target"),
            None,
            "debug_long_name",
        ),
        "run default V1 export without V2 debug inspection",
    );
    let explicit = output(
        simulation_export_command_for_feature(
            "gfx942",
            &explicit_path,
            &target.path().join("explicit-v1-target"),
            Some(1),
            "debug_long_name",
        ),
        "run explicit V1 export without V2 debug inspection",
    );
    assert!(
        default.status.success() && explicit.status.success(),
        "V1 inspected or rejected V2 metadata:\ndefault:\n{}\nexplicit:\n{}",
        default.stderr,
        explicit.stderr,
    );
    assert_eq!(
        std::fs::read(&default_path).unwrap(),
        std::fs::read(&explicit_path).unwrap(),
        "default and explicit V1 exports must remain byte-for-byte identical",
    );

    let v2_path = target.path().join("debug-long-name-v2.fe2sim");
    let v2 = output(
        simulation_export_command_v2(
            "gfx942",
            &v2_path,
            &target.path().join("v2-target"),
            "debug_long_name",
        ),
        "run V2 export with an overbound exact debug name",
    );
    assert!(
        !v2.status.success() && v2.stderr.contains("source-variable name") && !v2_path.exists(),
        "V2 did not fail closed on the exact overbound name:\n{}",
        v2.stderr,
    );
}

struct ExtractionOutput {
    status: std::process::ExitStatus,
    stderr: String,
}

fn run_extraction(target: &ScratchTarget, oob: bool) -> ExtractionOutput {
    let mut command = base_command("check", target.path());
    command
        .env("FE2O3_EXTRACT_RANKED_MEMORY_V1", "1")
        .env(
            "RUSTC_WORKSPACE_WRAPPER",
            env!("CARGO_BIN_EXE_fe2o3-rustc-extract"),
        )
        .env(
            "FE2O3_EXTRACT_CRATE_V1",
            "fe2o3_production_ranked_bounds_fixture",
        );
    if oob {
        command.args(["--features", "oob"]);
    }
    output(command, "run AMD extraction fixture")
}

fn run_feature_extraction(target: &ScratchTarget, feature: &str) -> ExtractionOutput {
    let mut command = base_command("check", target.path());
    command
        .env("FE2O3_EXTRACT_RANKED_MEMORY_V1", "1")
        .env(
            "RUSTC_WORKSPACE_WRAPPER",
            env!("CARGO_BIN_EXE_fe2o3-rustc-extract"),
        )
        .env(
            "FE2O3_EXTRACT_CRATE_V1",
            "fe2o3_production_ranked_bounds_fixture",
        )
        .args(["--features", feature]);
    output(command, "run safe mapped AMD extraction fixture")
}

fn run_release_feature_extraction(target: &ScratchTarget, feature: &str) -> ExtractionOutput {
    let mut command = base_command("check", target.path());
    command
        .env("FE2O3_EXTRACT_RANKED_MEMORY_V1", "1")
        .env(
            "RUSTC_WORKSPACE_WRAPPER",
            env!("CARGO_BIN_EXE_fe2o3-rustc-extract"),
        )
        .env(
            "FE2O3_EXTRACT_CRATE_V1",
            "fe2o3_production_ranked_bounds_fixture",
        )
        .args(["--release", "--features", feature]);
    output(command, "run optimized safe mapped AMD extraction fixture")
}

fn simulation_export_command(target: &str, output: &Path, target_dir: &Path) -> Command {
    simulation_export_command_for_feature(target, output, target_dir, None, "barrier_before_access")
}

fn simulation_export_command_v2(
    target: &str,
    output: &Path,
    target_dir: &Path,
    feature: &str,
) -> Command {
    simulation_export_command_for_feature(target, output, target_dir, Some(2), feature)
}

fn simulation_export_command_for_feature(
    target: &str,
    output: &Path,
    target_dir: &Path,
    bundle_version: Option<u16>,
    feature: &str,
) -> Command {
    const POISONED_WRAPPER: &str = "/fe2o3-poisoned-caller-wrapper-must-not-run";

    let mut command = Command::new(env!("CARGO_BIN_EXE_fe2o3-export-sim"));
    command
        .current_dir(workspace())
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS")
        .env("RUSTC_WRAPPER", POISONED_WRAPPER)
        .env("CARGO_BUILD_RUSTC_WRAPPER", POISONED_WRAPPER)
        .env("RUSTC_WORKSPACE_WRAPPER", POISONED_WRAPPER)
        .env("CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", POISONED_WRAPPER)
        .env_remove("FE2O3_EXTRACT_CRATE_V1")
        .env_remove("FE2O3_EXTRACT_SIMULATION_BUNDLE_PATH_V1")
        .env_remove("FE2O3_EXTRACT_SIMULATION_BUNDLE_PATH_V2")
        .env_remove("FE2O3_EXTRACT_RANKED_MEMORY_V1")
        .env_remove("FE2O3_EXTRACT_AMDGPU_LLVM_PATH_V1")
        .env_remove("FE2O3_EXTRACT_GFX942_LLVM_PATH_V1")
        .env_remove("FE2O3_EXTRACT_GFX942_COMPILER_HANDOFF_PATH_V1")
        .env_remove("FE2O3_EXTRACT_CRATE_BINDING_PATH_V1")
        .arg("--crate")
        .arg("fe2o3_production_ranked_bounds_fixture")
        .arg("--output")
        .arg(output)
        .arg("--target")
        .arg(target);
    if let Some(version) = bundle_version {
        command.arg("--bundle-version").arg(version.to_string());
    }
    command.arg("--target-dir").arg(target_dir).args([
        "--",
        "--package",
        "fe2o3-production-ranked-bounds-fixture",
        "--features",
        feature,
        "--lib",
    ]);
    command
}

fn base_command(action: &str, target_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO"));
    command
        .current_dir(workspace())
        .env(
            "FE2O3_CARGO_METADATA_BUILD_OBSERVATION_V2",
            "55".repeat(32),
        )
        .env("FE2O3_CRATE_BINDING_ID_V1", "77".repeat(32))
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env(
            "CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS",
            "-Zalways-encode-mir -Ctarget-cpu=gfx942 -Ctarget-feature=-xnack,+wavefrontsize64,-wavefrontsize32",
        )
        .args([
            action,
            "--locked",
            "-Zbuild-std=core",
            "-p",
            "fe2o3-production-ranked-bounds-fixture",
            "--target",
            "amdgcn-amd-amdhsa",
            "--target-dir",
        ])
        .arg(target_dir);
    command
}

fn output(mut command: Command, label: &str) -> ExtractionOutput {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    ExtractionOutput {
        status: output.status,
        stderr: String::from_utf8(output.stderr).expect("rustc diagnostic is UTF-8"),
    }
}
