#![deny(warnings)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

struct ScratchDirectory(PathBuf);

impl ScratchDirectory {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "fe2o3-neutral-workgroup-{label}-{}-{nonce}",
            std::process::id(),
        ));
        std::fs::create_dir(&path).expect("create isolated extraction directory");
        Self(path)
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("canonical workspace")
}

include!("production_neutral_workgroup_reduce_driver_v1/target_llvm_v1.rs");
include!("production_neutral_workgroup_reduce_driver_v1/row_affine_v1.rs");

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD targets"]
fn ordinary_row_affine_source_matches_oracle_and_replay() {
    row_affine_source_matches_oracle_and_replay();
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD targets"]
fn ordinary_row_affine_tile_source_matches_oracle_and_replay() {
    row_affine_variant_matches_oracle_and_replay(
        "row-affine-u32-tile-kernel",
        "kernel_row_affine_u32_tile.rs",
    );
}

#[derive(Clone, Copy)]
struct ScanCase {
    feature: &'static str,
    kernel: &'static str,
    element: &'static str,
    extent: usize,
    inclusive: bool,
}

const SCAN_CASES: [ScanCase; 18] = [
    ScanCase {
        feature: "lds-scan-u32-kernel",
        kernel: "lds_inclusive_scan_u32_v1",
        element: "u32",
        extent: 3,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-u32-65-kernel",
        kernel: "lds_inclusive_scan_u32_65_v1",
        element: "u32",
        extent: 65,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-u32-255-kernel",
        kernel: "lds_inclusive_scan_u32_255_v1",
        element: "u32",
        extent: 255,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-u32-exclusive-3-kernel",
        kernel: "lds_exclusive_scan_u32_3_v1",
        element: "u32",
        extent: 3,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-u32-exclusive-65-kernel",
        kernel: "lds_exclusive_scan_u32_65_v1",
        element: "u32",
        extent: 65,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-u32-exclusive-kernel",
        kernel: "lds_exclusive_scan_u32_v1",
        element: "u32",
        extent: 255,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-i32-inclusive-kernel",
        kernel: "lds_inclusive_scan_i32_v1",
        element: "i32",
        extent: 3,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-i32-inclusive-65-kernel",
        kernel: "lds_inclusive_scan_i32_65_v1",
        element: "i32",
        extent: 65,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-i32-inclusive-255-kernel",
        kernel: "lds_inclusive_scan_i32_255_v1",
        element: "i32",
        extent: 255,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-i32-3-kernel",
        kernel: "lds_exclusive_scan_i32_3_v1",
        element: "i32",
        extent: 3,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-i32-kernel",
        kernel: "lds_exclusive_scan_i32_v1",
        element: "i32",
        extent: 65,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-i32-255-kernel",
        kernel: "lds_exclusive_scan_i32_255_v1",
        element: "i32",
        extent: 255,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-f32-3-kernel",
        kernel: "lds_inclusive_scan_f32_3_v1",
        element: "f32",
        extent: 3,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-f32-65-kernel",
        kernel: "lds_inclusive_scan_f32_65_v1",
        element: "f32",
        extent: 65,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-f32-kernel",
        kernel: "lds_inclusive_scan_f32_v1",
        element: "f32",
        extent: 255,
        inclusive: true,
    },
    ScanCase {
        feature: "lds-scan-f32-exclusive-3-kernel",
        kernel: "lds_exclusive_scan_f32_3_v1",
        element: "f32",
        extent: 3,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-f32-exclusive-kernel",
        kernel: "lds_exclusive_scan_f32_v1",
        element: "f32",
        extent: 65,
        inclusive: false,
    },
    ScanCase {
        feature: "lds-scan-f32-exclusive-255-kernel",
        kernel: "lds_exclusive_scan_f32_255_v1",
        element: "f32",
        extent: 255,
        inclusive: false,
    },
];

fn scan_values(case: ScanCase) -> (Vec<u8>, Vec<u8>) {
    let mut input = Vec::with_capacity(case.extent * 4);
    let mut expected = Vec::with_capacity(case.extent * 4);
    match case.element {
        "u32" => {
            let values = if case.extent == 3 {
                vec![1_u32, 2, 3]
            } else {
                vec![1_u32; case.extent]
            };
            let mut sum = 0_u32;
            for value in values {
                input.extend_from_slice(&value.to_le_bytes());
                if case.inclusive {
                    sum = sum.wrapping_add(value);
                    expected.extend_from_slice(&sum.to_le_bytes());
                } else {
                    expected.extend_from_slice(&sum.to_le_bytes());
                    sum = sum.wrapping_add(value);
                }
            }
        }
        "i32" => {
            let values = if case.extent == 3 {
                vec![-2_i32, 3, -4]
            } else {
                vec![1_i32; case.extent]
            };
            let mut sum = 0_i32;
            for value in values {
                input.extend_from_slice(&value.to_le_bytes());
                if case.inclusive {
                    sum = sum.wrapping_add(value);
                    expected.extend_from_slice(&sum.to_le_bytes());
                } else {
                    expected.extend_from_slice(&sum.to_le_bytes());
                    sum = sum.wrapping_add(value);
                }
            }
        }
        "f32" => {
            let mut sum = 0.0_f32;
            for _ in 0..case.extent {
                let value = 1.0_f32;
                input.extend_from_slice(&value.to_bits().to_le_bytes());
                if case.inclusive {
                    sum += value;
                    expected.extend_from_slice(&sum.to_bits().to_le_bytes());
                } else {
                    expected.extend_from_slice(&sum.to_bits().to_le_bytes());
                    sum += value;
                }
            }
        }
        _ => unreachable!("closed scan scalar roster"),
    }
    (input, expected)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn assert_scan_rows(case: ScanCase, actual: &[u8], expected: &[u8], path: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} {path} byte length",
        case.feature
    );
    for (row, (actual, expected)) in actual
        .chunks_exact(4)
        .zip(expected.chunks_exact(4))
        .enumerate()
    {
        assert_eq!(
            actual, expected,
            "{} {path} row {row} {}-bit output",
            case.feature, case.element
        );
    }
}

fn write_scan_request(
    scratch: &ScratchDirectory,
    label: &str,
    case: ScanCase,
    input: &[u8],
    output_bytes: usize,
) -> PathBuf {
    let path = scratch.0.join(format!("{label}.json"));
    std::fs::write(
        &path,
        serde_json::to_vec(&json!({
            "schema": "fe2o3-simulation-request-v1",
            "kernel": case.kernel,
            "grid": [case.extent, 1, 1],
            "workgroup": [case.extent, 1, 1],
            "arguments": [
                {
                    "kind": "buffer",
                    "element": case.element,
                    "access": "read_only",
                    "alignment": 4,
                    "bytes": format!("0x{}", hex(input)),
                },
                {
                    "kind": "buffer",
                    "element": case.element,
                    "access": "read_write",
                    "alignment": 4,
                    "bytes": format!("0x{}", "00".repeat(output_bytes)),
                },
            ],
        }))
        .expect("encode scan simulation request"),
    )
    .expect("write scan simulation request");
    path
}

fn export_scan_bundle(scratch: &ScratchDirectory, case: ScanCase) -> PathBuf {
    export_workgroup_bundle(scratch, case.feature, "gfx942")
}

fn export_workgroup_bundle(scratch: &ScratchDirectory, feature: &str, cpu: &str) -> PathBuf {
    const POISONED_WRAPPER: &str = "/fe2o3-poisoned-caller-wrapper-must-not-run";

    let bundle = scratch.0.join(format!("{}.fe2sim", feature));
    let output = Command::new(env!("CARGO_BIN_EXE_fe2o3-export-sim"))
        .current_dir(workspace().join("examples/workgroup_sync_v1"))
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env_remove("CARGO_TARGET_AMDGCN_AMD_AMDHSA_RUSTFLAGS")
        .env("RUSTC_WRAPPER", POISONED_WRAPPER)
        .env("CARGO_BUILD_RUSTC_WRAPPER", POISONED_WRAPPER)
        .env("RUSTC_WORKSPACE_WRAPPER", POISONED_WRAPPER)
        .env("CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", POISONED_WRAPPER)
        .env_remove("FE2O3_EXTRACT_CRATE_V1")
        .env_remove("FE2O3_EXTRACT_SIMULATION_BUNDLE_PATH_V1")
        .env_remove("FE2O3_EXTRACT_SIMULATION_BUNDLE_PATH_V5")
        .env_remove("FE2O3_EXTRACT_RANKED_MEMORY_V1")
        .arg("--crate")
        .arg("fe2o3_workgroup_sync_v1")
        .arg("--output")
        .arg(&bundle)
        .args(["--target", cpu, "--bundle-version", "5", "--target-dir"])
        .arg(scratch.0.join("export-target"))
        .args([
            "--",
            "--no-default-features",
            "--features",
            feature,
            "--lib",
        ])
        .output()
        .expect("run ordinary workgroup Bundle V5 exporter");
    let stderr = String::from_utf8(output.stderr).expect("export diagnostic is UTF-8");
    assert!(
        output.status.success()
            && stderr.contains("target-neutral production KIR V8")
            && stderr.contains("exact same-module KIR V10 simulation bundle V5")
            && stderr.contains("compiler_execution=extraction_only_unavailable")
            && stderr.contains("authority false"),
        "{} did not export under exact authority-free V5 custody:\n{stderr}",
        feature,
    );
    bundle
}

struct ScanRuntimeLayout {
    signature: [u8; 32],
    explicit_kernarg_bytes: u32,
    input_pointer_slot: u32,
    input_length_slot: u32,
    output_pointer_slot: u32,
    output_length_slot: u32,
}

fn execute_scan_through_sim_runtime(
    bundle: &fe2o3_kernel_ir::VerifiedSimulationBundleV5,
    case: ScanCase,
    layout: &ScanRuntimeLayout,
    input: &[u8],
    expected: &[u8],
) {
    use fe2o3_runtime::RuntimeBackendV1 as _;

    let mut kernarg = vec![0; layout.explicit_kernarg_bytes as usize];
    for slot in [layout.input_length_slot, layout.output_length_slot] {
        let slot = slot as usize;
        kernarg[slot..slot + 8].copy_from_slice(&(case.extent as u64).to_le_bytes());
    }
    let mut backend = fe2o3_sim_runtime::SimRuntimeBackendV1::gfx942([0x5a; 32]).unwrap();
    assert!(!backend.uses_gpu());
    assert!(!backend.evidence().hardware);
    assert!(!backend.evidence().performance_prediction);
    let stream = backend.create_stream_v1(1).unwrap();
    let input_allocation = backend
        .allocate_v1(
            1,
            fe2o3_runtime::RuntimeMemoryKindV1::HostVisible,
            input.len() as u64,
            4,
        )
        .unwrap();
    let output_allocation = backend
        .allocate_v1(
            1,
            fe2o3_runtime::RuntimeMemoryKindV1::HostVisible,
            expected.len() as u64,
            4,
        )
        .unwrap();
    backend
        .write_allocation_v1(input_allocation, 0, input)
        .unwrap();
    backend
        .write_allocation_v1(output_allocation, 0, &vec![0; expected.len()])
        .unwrap();
    let module = backend.load_module_v1(1, bundle.canonical_bytes()).unwrap();
    let kernel = backend
        .resolve_kernel_v1(module, case.kernel, layout.signature)
        .unwrap();
    let bindings = [
        fe2o3_runtime::BackendBindingV1 {
            region: fe2o3_runtime::BackendMemoryRegionV1 {
                allocation: input_allocation,
                access: fe2o3_runtime::RuntimeAccessV1::Read,
                byte_offset: 0,
                byte_len: input.len() as u64,
            },
            kernarg_byte_offset: layout.input_pointer_slot,
        },
        fe2o3_runtime::BackendBindingV1 {
            region: fe2o3_runtime::BackendMemoryRegionV1 {
                allocation: output_allocation,
                access: fe2o3_runtime::RuntimeAccessV1::ReadWrite,
                byte_offset: 0,
                byte_len: expected.len() as u64,
            },
            kernarg_byte_offset: layout.output_pointer_slot,
        },
    ];
    let submission = backend
        .submit_v1(fe2o3_runtime::BackendLaunchV1 {
            stream,
            kernel,
            explicit_kernarg: &kernarg,
            bindings: &bindings,
            dependencies: &[],
            geometry: fe2o3_runtime::RuntimeLaunchGeometryV1 {
                grid: [case.extent as u32, 1, 1],
                workgroup: [case.extent as u32, 1, 1],
                dynamic_shared_bytes: 0,
            },
            semantic_launch: fe2o3_runtime::BackendSemanticLaunchV1::Ordinary,
        })
        .unwrap();
    assert_eq!(
        backend
            .wait_v1(
                submission,
                std::time::Instant::now() + std::time::Duration::from_secs(20),
            )
            .unwrap(),
        fe2o3_runtime::BackendPollV1::Succeeded
    );
    let mut output = vec![0; expected.len()];
    backend
        .read_allocation_v1(output_allocation, 0, &mut output)
        .unwrap();
    assert_scan_rows(case, &output, expected, "runtime");
    backend.release_submission_v1(submission).unwrap();
    backend.destroy_stream_v1(stream).unwrap();
    backend.unload_module_v1(module).unwrap();
    backend.release_allocation_v1(output_allocation).unwrap();
    backend.release_allocation_v1(input_allocation).unwrap();
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_integer_scan_sources_wrap_on_overflow() {
    use fe2o3_kir_sim::{BufferBackingIdV1, SimulationScheduleRequestV1};

    const CANARY: u32 = 0xa5c3_7e19;
    let scratch = ScratchDirectory::new("integer-scan-overflow");
    let cases = SCAN_CASES
        .into_iter()
        .filter(|case| case.extent == 3 && matches!(case.element, "u32" | "i32"))
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let bundle_path = export_scan_bundle(&scratch, case);
        // Zero tails isolate overflow in speculative additions for inactive scan lanes.
        let vectors = match case.element {
            "u32" => vec![[u32::MAX, 1, 7], [u32::MAX, 0, 0]],
            "i32" => vec![
                [i32::MAX as u32, 1, 1],
                [i32::MIN as u32, (-1_i32) as u32, (-1_i32) as u32],
                [i32::MAX as u32, 0, 0],
                [i32::MIN as u32, 0, 0],
            ],
            _ => unreachable!("integer scan roster"),
        };
        for (vector, values) in vectors.into_iter().enumerate() {
            let input = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let mut sum = 0_u32;
            let mut expected = CANARY.to_le_bytes().repeat(4);
            for bits in values {
                let previous = sum;
                sum = match case.element {
                    "u32" => sum.wrapping_add(bits),
                    "i32" => (sum as i32).wrapping_add(bits as i32) as u32,
                    _ => unreachable!("integer scan roster"),
                };
                let prefix = if case.inclusive { sum } else { previous };
                expected.extend_from_slice(&prefix.to_le_bytes());
            }
            expected.extend_from_slice(&CANARY.to_le_bytes().repeat(4));
            let request_path = write_scan_request(
                &scratch,
                "integer-scan-overflow",
                case,
                &input,
                case.extent * 4,
            );
            let mut request: Value =
                serde_json::from_slice(&std::fs::read(&request_path).unwrap()).unwrap();
            // Keep the source's exact output extent while guarding its backing allocation.
            request["arguments"][1] = json!({
                "kind": "buffer_view", "backing": 1, "element": case.element,
                "access": "read_write", "alignment": 4,
                "byte_offset": 16, "elements": case.extent,
            });
            request["shared_buffers"] = json!([{
                "id": 1, "element": case.element, "access": "read_write", "alignment": 4,
                "bytes": format!("0x{}", hex(&CANARY.to_le_bytes().repeat(case.extent + 8))),
            }]);
            std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
            let admitted =
                fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(&bundle_path, &request_path)
                    .unwrap();
            for repetition in 0..2_u64 {
                for schedule in [
                    SimulationScheduleRequestV1::RecordCanonical {
                        max_decisions: 20_000,
                    },
                    SimulationScheduleRequestV1::RecordSeeded {
                        seed: 0x275_500 + vector as u64 * 4 + repetition,
                        max_decisions: 20_000,
                    },
                ] {
                    let execution = admitted
                        .input()
                        .module
                        .simulate_scheduled(
                            &admitted.input().request,
                            admitted.input().simulation_target(),
                            admitted.input().simulation_limits,
                            schedule,
                        )
                        .unwrap_or_else(|error| {
                            panic!(
                                "{} vector {vector} repetition {repetition}: {error:?}",
                                case.feature,
                            )
                        });
                    assert_eq!(execution.invocations_executed(), case.extent as u64);
                    assert_eq!(execution.workgroups_visited(), 1);
                    assert_eq!(
                        execution.buffer(0).unwrap().bytes(),
                        input,
                        "{} changed immutable scan input",
                        case.feature
                    );
                    assert_scan_rows(
                        case,
                        execution
                            .shared_buffer(BufferBackingIdV1(1))
                            .unwrap()
                            .bytes(),
                        &expected,
                        &format!("overflow vector {vector} repetition {repetition}"),
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires the pinned nightly rust-src component and AMD target"]
fn ordinary_scan_sources_export_v5_and_execute_every_cpu_observation_path() {
    use fe2o3_kernel_ir::{
        AddressSpace, LaunchExtent, OperationKind, SemanticKirComponentRepresentationV2,
        WorkgroupMemoryExtent,
    };
    use fe2o3_kir_sim::{
        PersistedSimulationScheduleDocumentV1, SimulationExecutionErrorKindV1,
        SimulationScheduleRequestV1,
    };
    use fe2o3_semantic_trace::{
        FactProvenanceV1, OpaqueIdentityV1, TraceBoundsV1, TraceCompletenessV1, WaveWidthV1,
    };

    let scratch = ScratchDirectory::new("scan-bundle-v5");
    let debug_target = scratch.0.join("debug-target");
    let debug_build = Command::new(env!("CARGO"))
        .current_dir(workspace())
        .args([
            "build",
            "--quiet",
            "--locked",
            "-p",
            "fe2o3-debug-cli",
            "--bin",
            "fe2o3-debug",
            "-p",
            "fe2o3-semantic-query",
            "--bin",
            "fe2o3-trace-query",
            "--target-dir",
        ])
        .arg(&debug_target)
        .output()
        .expect("build debugger for ordinary scan qualification");
    assert!(
        debug_build.status.success(),
        "debugger build failed:\n{}",
        String::from_utf8_lossy(&debug_build.stderr)
    );

    let mut first_schedule = None;
    let mut second_bundle = None;
    let mut second_request = None;
    for (case_index, case) in SCAN_CASES.into_iter().enumerate() {
        let bundle_path = export_scan_bundle(&scratch, case);
        let bundle = fe2o3_kernel_ir::VerifiedSimulationBundleV5::from_canonical_bytes(
            std::fs::read(&bundle_path).unwrap(),
        )
        .unwrap();
        assert_eq!(bundle.target(), "gfx942:xnack-");
        assert_eq!(bundle.production_kir_identity().version(), 8);
        assert_eq!(bundle.kernel_count(), 1);
        assert!(!bundle.authenticates_compiler_execution());
        assert!(!bundle.grants_compiler_authority());
        assert!(!bundle.grants_artifact_authority());
        assert!(!bundle.grants_hardware_authority());
        assert!(!bundle.grants_load_authority());
        assert!(!bundle.grants_launch_authority());

        let (_, kir) =
            fe2o3_kernel_ir::VerifiedCanonicalKernelIrV10::from_canonical_bytes_with_module(
                bundle.canonical_kir_v10().to_vec(),
            )
            .unwrap();
        let kernel = kir
            .kernels
            .iter()
            .find(|kernel| kernel.id.as_str() == case.kernel)
            .unwrap();
        let mut launch_extents = kernel.domain.extents();
        assert_eq!(
            launch_extents.next(),
            Some(LaunchExtent::Static(case.extent as u32))
        );
        assert_eq!(launch_extents.next(), None);
        let operations = kir
            .function(&kernel.entry)
            .unwrap()
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .collect::<Vec<_>>();
        assert!(operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::WorkgroupMemory(ref memory)
                if memory.extent == WorkgroupMemoryExtent::Static(case.extent as u32)
        )));
        assert!(operations.iter().any(|operation| matches!(
            operation.kind,
            OperationKind::GuardedLoad { access, .. }
                if access.address_space == AddressSpace::Global
        )));
        assert!(
            operations
                .iter()
                .any(|operation| matches!(operation.kind, OperationKind::WorkgroupBarrier(_)))
        );

        let source_map = fe2o3_kernel_ir::DebugSourceMapDocumentV2::from_canonical_json_bytes(
            bundle.debug_map(),
        )
        .unwrap();
        assert_eq!(
            source_map.binding().bundle_subject_identity(),
            *bundle.subject_identity()
        );
        assert!(
            source_map
                .files()
                .iter()
                .any(|file| file.display_path().contains("kernel_scan_"))
        );
        assert!(!source_map.sites().is_empty());

        let aggregate = fe2o3_kernel_ir::SemanticAggregateStorageMapV5::from_canonical_json_bytes(
            bundle.aggregate_storage_map(),
        )
        .unwrap();
        assert_eq!(
            aggregate.bundle_subject_identity(),
            bundle.subject_identity()
        );
        let [kernel_map] = aggregate.kernels() else {
            panic!(
                "{} omitted its one compiler-derived storage map",
                case.feature
            )
        };
        let [input_argument, output_argument] = kernel_map.arguments() else {
            panic!("{} changed its two source arguments", case.feature)
        };
        let [input_component] = input_argument.storage().components().unwrap() else {
            panic!("{} input is not one compiler-derived region", case.feature)
        };
        let [output_component] = output_argument.storage().components().unwrap() else {
            panic!("{} output is not one compiler-derived region", case.feature)
        };
        assert_eq!(
            input_component.representation(),
            SemanticKirComponentRepresentationV2::RegionSlice
        );
        assert_eq!(
            output_component.representation(),
            SemanticKirComponentRepresentationV2::RegionSlice
        );
        for component in [input_component, output_component] {
            assert_eq!(component.value_slot().byte_width(), 8);
            assert_eq!(component.metadata_slot().unwrap().byte_width(), 8);
        }
        let semantic = fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1::decode_current_production_canonical(
            bundle.semantic_mir(),
            fe2o3_mir_model::semantic_mir_v1::SemanticMirLimitsV1::default(),
        )
        .unwrap();
        assert_eq!(
            semantic.wire_version(),
            fe2o3_mir_model::semantic_mir_v1::SemanticMirWireVersionV1::V13
        );
        let runtime_layout = ScanRuntimeLayout {
            signature: *semantic.functions()[kernel_map.semantic_root() as usize]
                .abi()
                .identity()
                .as_bytes(),
            explicit_kernarg_bytes: kernel_map.explicit_kernarg_bytes(),
            input_pointer_slot: input_component.value_slot().byte_offset(),
            input_length_slot: input_component.metadata_slot().unwrap().byte_offset(),
            output_pointer_slot: output_component.value_slot().byte_offset(),
            output_length_slot: output_component.metadata_slot().unwrap().byte_offset(),
        };

        let (input, expected) = scan_values(case);
        let request_path = write_scan_request(
            &scratch,
            &format!("{}-request", case.feature),
            case,
            &input,
            expected.len(),
        );
        let admitted =
            fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(&bundle_path, &request_path)
                .unwrap();
        assert_eq!(
            admitted.input().simulation_bundle_subject(),
            Some(*bundle.subject_identity())
        );
        let execution = admitted
            .input()
            .module
            .simulate_scheduled(
                &admitted.input().request,
                admitted.input().simulation_target(),
                admitted.input().simulation_limits,
                SimulationScheduleRequestV1::RecordSeeded {
                    seed: 0x5ca0 + case_index as u64,
                    max_decisions: 20_000,
                },
            )
            .unwrap_or_else(|error| {
                panic!(
                    "{} extent {} direct simulation failed: {error:?}",
                    case.feature, case.extent
                )
            });
        assert_eq!(execution.invocations_executed(), case.extent as u64);
        assert_eq!(execution.workgroups_visited(), 1);
        assert_scan_rows(
            case,
            execution.buffer(1).unwrap().bytes(),
            &expected,
            "direct simulator",
        );

        let mut trace_limits = admitted.input().simulation_limits;
        trace_limits.max_events = 500_000;
        let trace = fe2o3_kir_sim_trace::simulate_with_semantic_trace_v2(
            &admitted.input().module,
            &admitted.input().request,
            admitted.input().simulation_target(),
            trace_limits,
            fe2o3_kir_sim_trace::SimulationTraceProfileV2 {
                wave_width: WaveWidthV1::Wave64,
                bounds: TraceBoundsV1::new(500_000, 32 * 1024 * 1024, 1).unwrap(),
                dispatch_occurrence: OpaqueIdentityV1::new([case_index as u8 + 1; 32]).unwrap(),
            },
        )
        .unwrap_or_else(|error| {
            panic!(
                "{} extent {} semantic trace failed: {error:?}",
                case.feature, case.extent
            )
        });
        assert!(!trace.grants_execution_authority());
        assert_eq!(
            trace.trace.header().completeness(),
            TraceCompletenessV1::Complete
        );
        assert_scan_rows(
            case,
            trace.execution.as_ref().unwrap().buffer(1).unwrap().bytes(),
            &expected,
            "Trace V2 simulator",
        );
        assert!(!trace.trace.events().is_empty());
        assert!(
            trace
                .trace
                .events()
                .iter()
                .all(|event| event.provenance() == FactProvenanceV1::Observed)
        );

        let trace_bytes = fe2o3_semantic_trace::encode_trace_v2(&trace.trace).unwrap();
        let query_trace = |arguments: &[&str]| {
            let mut child = Command::new(debug_target.join("debug/fe2o3-trace-query"))
                .args(arguments)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let written = child.stdin.take().unwrap().write_all(&trace_bytes);
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{} trace query failed: {}",
                case.feature,
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            written.expect("write complete canonical trace to query stdin");
            serde_json::from_slice::<Value>(&output.stdout).unwrap()
        };
        let summary = query_trace(&["summary"]);
        let context = &summary["context"];
        assert_eq!(context["schema"], "fe2o3-semantic-query-v2");
        assert_eq!(context["execution_kind"], "cpu_kir_simulation");
        assert_eq!(context["event_count"], trace.trace.events().len() as u64);
        assert_eq!(context["kernel_ir"]["wire_version"], 10);
        assert_eq!(context["kernel_ir"]["identity_policy"], 1);
        assert_eq!(
            context["kernel_ir"]["digest"],
            hex(bundle.canonical_kir_v10_digest())
        );
        assert_eq!(
            context["kernel_ir"]["canonical_len"],
            bundle.canonical_kir_v10_length()
        );
        assert_eq!(context["kernel_ir"]["authenticated"], false);
        assert_eq!(context["capture"]["completeness"], "complete");
        assert!(summary["summary"]["memory_accesses"].as_u64().unwrap() > 0);
        assert!(summary["summary"]["barrier_events"].as_u64().unwrap() > 0);
        if case.extent == 65 {
            let lanes = query_trace(&[
                "lanes",
                "--workgroup",
                "0,0,0",
                "--wave",
                "1",
                "--lane",
                "0",
                "--limit",
                "1",
            ]);
            assert_eq!(lanes["page"]["returned"], 1);
            let scope = &lanes["page"]["items"][0]["event"]["scope"];
            assert_eq!(scope["level"], "lane");
            assert_eq!(scope["logical_workitem"], json!([64, 0, 0]));
            assert_eq!(scope["wave"], 1);
            assert_eq!(scope["lane"], 0);
            assert_eq!(scope["active_mask"], 1);
            assert_eq!(scope["wave_width"], 64);
        }

        execute_scan_through_sim_runtime(&bundle, case, &runtime_layout, &input, &expected);

        let mut debugger = Command::new(debug_target.join("debug/fe2o3-debug"));
        let mut debugger = debugger
            .args(["sim", "--bundle-v5"])
            .arg(&bundle_path)
            .arg("--request")
            .arg(&request_path)
            .args(["--protocol", "jsonl", "--wave-width", "64"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut commands = concat!(
            "{\"operation\":\"continue\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":1,\"expected_revision\":0,\"max_events\":1000000}\n",
            "{\"operation\":\"inspect_scope\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":2,\"expected_revision\":1,\"scope\":{\"level\":\"workgroup\",\"workgroup\":[0,0,0]},\"include_children\":true,\"page\":{\"limit\":512}}\n",
            "{\"operation\":\"inspect_scope\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":3,\"expected_revision\":1,\"scope\":{\"level\":\"wave\",\"workgroup\":[0,0,0],\"wave\":0},\"include_children\":true,\"page\":{\"limit\":128}}\n",
            "{\"operation\":\"inspect_scope\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":4,\"expected_revision\":1,\"scope\":{\"level\":\"lane\",\"workgroup\":[0,0,0],\"wave\":0,\"lane\":0},\"include_children\":true,\"page\":{\"limit\":32}}\n",
        )
        .as_bytes()
        .to_vec();
        if case.extent == 65 {
            commands.extend_from_slice(concat!(
                "{\"operation\":\"inspect_scope\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":5,\"expected_revision\":1,\"scope\":{\"level\":\"wave\",\"workgroup\":[0,0,0],\"wave\":1},\"include_children\":true,\"page\":{\"limit\":128}}\n",
                "{\"operation\":\"inspect_scope\",\"schema\":\"fe2o3-debug-request-v1\",\"request_id\":6,\"expected_revision\":1,\"scope\":{\"level\":\"lane\",\"workgroup\":[0,0,0],\"wave\":1,\"lane\":0},\"include_children\":true,\"page\":{\"limit\":32}}\n",
            ).as_bytes());
        }
        debugger.stdin.take().unwrap().write_all(&commands).unwrap();
        let debugged = debugger.wait_with_output().unwrap();
        assert!(
            debugged.status.success(),
            "{} debugger failed:\n{}",
            case.feature,
            String::from_utf8_lossy(&debugged.stderr)
        );
        let responses = debugged
            .stdout
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(responses.len(), if case.extent == 65 { 6 } else { 4 });
        if case.extent == 255 {
            assert_eq!(
                responses[0]["result"]["stop"]["reason"],
                "resource_exhaustion"
            );
            assert_eq!(responses[0]["result"]["stop"]["exact"], false);
            assert_eq!(responses[0]["result"]["stop"]["outcome"], "active");
        } else {
            assert_eq!(responses[0]["result"]["stop"]["reason"], "completed");
            assert_eq!(responses[0]["result"]["stop"]["exact"], true);
        }
        assert_eq!(responses[0]["session"]["simulated"], true);
        assert_eq!(responses[0]["session"]["hardware_observed"], false);
        assert_eq!(responses[0]["session"]["performance_prediction"], false);
        for response in &responses[1..] {
            assert_eq!(response["status"], "ok");
            assert!(!response["result"]["scopes"].as_array().unwrap().is_empty());
            assert_eq!(
                response["session"]["configuration_identity"],
                responses[0]["session"]["configuration_identity"]
            );
        }
        if case.extent == 65 {
            let partial_wave = responses[4]["result"]["scopes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|scope| scope["scope"]["level"] == "wave" && scope["scope"]["wave"] == 1)
                .unwrap();
            assert_eq!(partial_wave["scope"]["active_mask"], 1);
            assert_eq!(partial_wave["scope"]["wave_width"], 64);
            assert_eq!(
                partial_wave["scope"]["interpretation"],
                "logical_visualization"
            );
            let final_lane = &responses[5]["result"]["scopes"][0]["scope"];
            assert_eq!(final_lane["level"], "lane");
            assert_eq!(final_lane["wave"], 1);
            assert_eq!(final_lane["lane"], 0);
            assert_eq!(final_lane["logical_workitem"], json!([64, 0, 0]));
            assert_eq!(final_lane["active_mask"], 1);
        }

        let schedule_path = scratch.0.join(format!("{}-schedule.json", case.feature));
        let schedule_bytes = PersistedSimulationScheduleDocumentV1::encode_record(
            admitted
                .input()
                .persisted_schedule_binding()
                .expect("existing bundle schedule binding"),
            execution.schedule_record().unwrap(),
        )
        .unwrap();
        let persisted =
            PersistedSimulationScheduleDocumentV1::from_canonical_bytes(&schedule_bytes).unwrap();
        assert_eq!(
            persisted.binding(),
            admitted
                .input()
                .persisted_schedule_binding()
                .expect("existing bundle schedule binding")
        );
        assert_eq!(persisted.record(), execution.schedule_record().unwrap());
        assert_eq!(persisted.to_canonical_bytes().unwrap(), schedule_bytes);
        let replay = admitted
            .input()
            .module
            .simulate_scheduled(
                &admitted.input().request,
                admitted.input().simulation_target(),
                admitted.input().simulation_limits,
                SimulationScheduleRequestV1::Replay(persisted.record()),
            )
            .unwrap();
        assert_eq!(replay.schedule(), execution.schedule());
        assert_eq!(
            replay.schedule_transcript_identity(),
            execution.schedule_transcript_identity()
        );
        assert_eq!(replay.schedule_coverage(), execution.schedule_coverage());
        assert_eq!(
            replay.schedule_coverage().decisions(),
            u64::try_from(persisted.record().decisions().len()).unwrap()
        );
        assert_eq!(replay.schedule_coverage().workgroups(), 1);
        assert!(replay.schedule_coverage().barrier_releases() > 0);
        assert!(replay.schedule_coverage().is_complete());
        assert_eq!(replay.invocations_executed(), case.extent as u64);
        assert_scan_rows(
            case,
            replay.buffer(1).unwrap().bytes(),
            &expected,
            "persisted seeded schedule replay",
        );
        std::fs::write(&schedule_path, &schedule_bytes).unwrap();
        if case_index == 0 {
            first_schedule = Some(schedule_path);
        } else if case_index == 1 {
            second_bundle = Some(bundle_path.clone());
            second_request = Some(request_path.clone());
        }

        let mut corrupted = bundle.canonical_bytes().to_vec();
        *corrupted.last_mut().unwrap() ^= 1;
        assert!(
            fe2o3_kernel_ir::VerifiedSimulationBundleV5::from_canonical_bytes(corrupted).is_err(),
            "{} accepted content substitution under its original identity",
            case.feature
        );

        let short_input_path = write_scan_request(
            &scratch,
            &format!("{}-short-input", case.feature),
            case,
            &input[..input.len() - 4],
            expected.len(),
        );
        let short_input =
            fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(&bundle_path, &short_input_path)
                .unwrap();
        let short_error = short_input
            .input()
            .module
            .simulate(
                &short_input.input().request,
                short_input.input().simulation_target(),
                short_input.input().simulation_limits,
            )
            .unwrap_err();
        assert!(matches!(
            short_error,
            fe2o3_kir_sim::SimulationErrorV1::Execution(ref error)
                if error.kind == SimulationExecutionErrorKindV1::ReachedUnreachable
        ));

        let wrong_geometry_path = scratch
            .0
            .join(format!("{}-wrong-geometry.json", case.feature));
        let mut wrong_geometry: Value =
            serde_json::from_slice(&std::fs::read(&request_path).unwrap()).unwrap();
        wrong_geometry["workgroup"] = json!([case.extent - 1, 1, 1]);
        std::fs::write(
            &wrong_geometry_path,
            serde_json::to_vec(&wrong_geometry).unwrap(),
        )
        .unwrap();
        let wrong_geometry =
            fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(&bundle_path, &wrong_geometry_path)
                .unwrap();
        assert!(matches!(
            wrong_geometry.input().module.simulate(
                &wrong_geometry.input().request,
                wrong_geometry.input().simulation_target(),
                wrong_geometry.input().simulation_limits,
            ),
            Err(fe2o3_kir_sim::SimulationErrorV1::Preflight(
                fe2o3_kir_sim::SimulationPreflightErrorV1::WorkgroupMismatch { .. }
            ))
        ));

        for (label, hostile_extent) in [
            ("short-grid", case.extent - 1),
            ("long-grid", case.extent + 1),
        ] {
            let hostile_grid_path = scratch.0.join(format!("{}-{label}.json", case.feature));
            let mut hostile_grid: Value =
                serde_json::from_slice(&std::fs::read(&request_path).unwrap()).unwrap();
            hostile_grid["grid"] = json!([hostile_extent, 1, 1]);
            std::fs::write(
                &hostile_grid_path,
                serde_json::to_vec(&hostile_grid).unwrap(),
            )
            .unwrap();
            let hostile_grid = fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(
                &bundle_path,
                &hostile_grid_path,
            )
            .unwrap();
            assert!(
                matches!(
                    hostile_grid.input().module.simulate(
                        &hostile_grid.input().request,
                        hostile_grid.input().simulation_target(),
                        hostile_grid.input().simulation_limits,
                    ),
                    Err(fe2o3_kir_sim::SimulationErrorV1::Preflight(
                        fe2o3_kir_sim::SimulationPreflightErrorV1::StaticLaunchMismatch {
                            axis: 0,
                            expected,
                            actual,
                        }
                    )) if expected == case.extent as u64 && actual == hostile_extent as u64
                ),
                "{} admitted its {label} extent {hostile_extent}",
                case.feature
            );
        }
    }

    let substituted = Command::new(debug_target.join("debug/fe2o3-debug"))
        .args(["sim", "--bundle-v5"])
        .arg(second_bundle.unwrap())
        .arg("--request")
        .arg(second_request.unwrap())
        .arg("--replay-schedule")
        .arg(first_schedule.unwrap())
        .args(["--protocol", "jsonl", "--wave-width", "64"])
        .output()
        .unwrap();
    assert!(!substituted.status.success());
    let error: Value = serde_json::from_slice(&substituted.stderr).unwrap();
    assert_eq!(error["stage"], "input");
    assert_eq!(error["code"], "schedule_binding_mismatch");
}
