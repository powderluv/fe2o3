#![cfg(target_os = "linux")]

#[path = "fixtures/declared_target.rs"]
mod declared_target;

use std::fmt::Write as _;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use fe2o3_kernel_ir::{
    AccessMode, AddressSpace, BasicBlock, BlockId, CastKind, Constant, DebugSourceMapDocumentV1,
    DebugSourceMapDocumentV2, DebugSourceMapFileV1, DebugSourceMapSpanV1, Function, Kernel,
    LaunchDomain, LaunchExtent, MAX_SIMULATION_BUNDLE_BYTES_V1, MemoryAccess, Module, Operation,
    OperationKind, PreparedSimulationBundleV1, PreparedSimulationBundleV6, ScalarType,
    SemanticAggregateStorageMapV6, SemanticKernelStorageV1, SemanticKernelStorageV2,
    SemanticStorageMapV6, Signature, SimulationCompilerExecutionBindingV1,
    SimulationProductionKirIdentityV1, SimulationProductionKirIdentityV6,
    SimulationSourceLineageV1, TargetCapability, Terminator, Type, ValueDef, ValueId,
    VerifiedCanonicalKernelIrV7, VerifiedCanonicalKernelIrV8, VerifiedCanonicalKernelIrV11,
    WaveOperation, WaveOperationKind, WaveWidth,
};
use fe2o3_kir_sim::{MAX_PERSISTED_SCHEDULE_BYTES_V1, MAX_SCHEDULE_DECISIONS_V1};
use sha2::{Digest, Sha256};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "fe2o3-kir-sim-command-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_fe2o3-kir-sim"))
}

fn noop_with_buffer_module() -> Module {
    let element = Type::Scalar(ScalarType::U8);
    let slice = Type::slice(element, AddressSpace::Global, AccessMode::ReadOnly);
    let mut block = BasicBlock::new(BlockId(0));
    block.terminator = Some(Terminator::Return { values: vec![] });
    let entry = Function::kernel_entry(
        "entry",
        Signature::new(vec![slice], vec![]),
        vec![ValueId(0)],
        vec![block],
    );
    let mut module = Module::new("cli-command-test");
    module.functions.push(entry);
    module.kernels.push(Kernel::new(
        "kernel",
        "entry",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    ));
    module
}

fn canonical_noop_with_buffer() -> Vec<u8> {
    VerifiedCanonicalKernelIrV7::from_module(noop_with_buffer_module())
        .unwrap()
        .into_canonical_bytes()
}

fn canonical_noop_with_buffer_variant() -> Vec<u8> {
    let mut module = noop_with_buffer_module();
    module.id = "cli-command-test-variant".into();
    VerifiedCanonicalKernelIrV7::from_module(module)
        .unwrap()
        .into_canonical_bytes()
}

fn simulation_bundle(target: &str) -> Vec<u8> {
    simulation_bundle_with_debug_map(target, None)
}

fn simulation_bundle_with_debug_map(target: &str, debug_source: Option<&[u8]>) -> Vec<u8> {
    simulation_bundle_for_module(noop_with_buffer_module(), target, debug_source)
}

fn simulation_bundle_for_module(
    module: Module,
    target: &str,
    debug_source: Option<&[u8]>,
) -> Vec<u8> {
    let production = VerifiedCanonicalKernelIrV8::from_module(module.clone()).unwrap();
    let production_identity = SimulationProductionKirIdentityV1::v8(
        *production.identity().digest(),
        production.identity().canonical_length(),
    )
    .unwrap();
    let prepared = PreparedSimulationBundleV1::new(
        SimulationCompilerExecutionBindingV1::UnavailableExtractionOnly,
        SimulationSourceLineageV1::new([1; 32], 11, [2; 32], 22).unwrap(),
        production_identity,
        target,
        VerifiedCanonicalKernelIrV7::from_module(module).unwrap(),
    )
    .unwrap();
    let bundle = if let Some(source) = debug_source {
        let source_identity: [u8; 32] = Sha256::digest(source).into();
        let document = DebugSourceMapDocumentV1::new(
            prepared.debug_source_map_binding(),
            vec![
                DebugSourceMapFileV1::new(
                    source_identity,
                    u64::try_from(source.len()).unwrap(),
                    "schedule-test.rs".to_owned(),
                )
                .unwrap(),
            ],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        prepared.finalize_with_source_map(document).unwrap()
    } else {
        prepared.finalize_without_source_map().unwrap()
    };
    bundle.into_canonical_bytes()
}

fn simulation_bundle_v6() -> fe2o3_kernel_ir::VerifiedSimulationBundleV6 {
    let scalar = Type::Scalar(ScalarType::U32);
    let read_write = Type::pointer(scalar.clone(), AddressSpace::Private, AccessMode::ReadWrite);
    let read_only = Type::pointer(scalar.clone(), AddressSpace::Private, AccessMode::ReadOnly);
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = vec![
        Operation::effect_free(
            ValueDef::new(ValueId(0), scalar.clone()),
            OperationKind::Constant(Constant::U32(7)),
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(1), read_write),
            OperationKind::Alloca {
                element: scalar.clone(),
                count: None,
                address_space: AddressSpace::Private,
                alignment: 4,
            },
        ),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(1),
                value: ValueId(0),
                access: MemoryAccess::new(AddressSpace::Private, 4),
            },
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(2), read_only.clone()),
            OperationKind::Cast {
                kind: CastKind::RestrictPointerAccess,
                value: ValueId(1),
                to: read_only,
            },
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(3), scalar),
            OperationKind::Load {
                pointer: ValueId(2),
                access: MemoryAccess::new(AddressSpace::Private, 4),
            },
        ),
    ];
    block.terminator = Some(Terminator::Return { values: vec![] });
    let entry = Function::kernel_entry(
        "bundle_v6_entry",
        Signature::new(vec![], vec![]),
        vec![],
        vec![block],
    );
    let mut module = Module::new("cli-command-bundle-v6-test");
    module.functions.push(entry);
    module.kernels.push(Kernel::new(
        "bundle_v6_kernel",
        "bundle_v6_entry",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    ));

    let canonical = VerifiedCanonicalKernelIrV11::from_module(module).unwrap();
    let production_digest = *canonical.identity().digest();
    let production_length = canonical.identity().canonical_length();
    let prepared = PreparedSimulationBundleV6::new(
        SimulationSourceLineageV1::new([0x61; 32], 101, [0x62; 32], 102).unwrap(),
        SimulationProductionKirIdentityV6::new(11, production_digest, production_length).unwrap(),
        "gfx942:xnack-",
        canonical,
    )
    .unwrap();
    let source_map = DebugSourceMapDocumentV2::new(
        prepared.debug_source_map_binding(),
        vec![DebugSourceMapFileV1::new([0x64; 32], 16, "bundle-v6.rs".into()).unwrap()],
        vec![],
        vec![DebugSourceMapSpanV1::new([0x64; 32], 1, 2, 1, 2).unwrap()],
        vec![],
        vec![],
    )
    .unwrap();
    let semantic = b"cli-command-bundle-v6-semantic-fixture".to_vec();
    let semantic_digest = <[u8; 32]>::from(Sha256::digest(&semantic));
    let storage = SemanticStorageMapV6::new(
        *prepared.subject_identity(),
        1,
        semantic_digest,
        semantic.len() as u64,
        [0x63; 32],
        *prepared.canonical_kir_v11_digest(),
        prepared.canonical_kir_v11_length(),
        vec![SemanticKernelStorageV1::new(0, 0, 0, vec![])],
        vec![],
    )
    .unwrap();
    let aggregate = SemanticAggregateStorageMapV6::new(
        *prepared.subject_identity(),
        *prepared.canonical_kir_v11_digest(),
        prepared.canonical_kir_v11_length(),
        vec![SemanticKernelStorageV2::new(0, 0, 0, 0, 1, vec![])],
    )
    .unwrap();
    prepared
        .finalize(source_map, semantic, storage, aggregate)
        .unwrap()
}

fn hex_identity(identity: &[u8; 32]) -> String {
    let mut result = String::with_capacity(64);
    for byte in identity {
        write!(&mut result, "{byte:02x}").unwrap();
    }
    result
}

fn op(result: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::effect_free(ValueDef::new(ValueId(result), ty), kind)
}

fn conflicting_store_module(entry_name: &str) -> Module {
    let scalar = Type::Scalar(ScalarType::U32);
    let pointer = Type::pointer(scalar.clone(), AddressSpace::Global, AccessMode::ReadWrite);
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = vec![
        op(1, scalar, OperationKind::Constant(Constant::U32(42))),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(0),
                value: ValueId(1),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    block.terminator = Some(Terminator::Return { values: vec![] });
    let entry = Function::kernel_entry(
        entry_name,
        Signature::new(vec![pointer], vec![]),
        vec![ValueId(0)],
        vec![block],
    );
    let mut module = Module::new("cli-command-race-test");
    module.functions.push(entry);
    module.kernels.push(Kernel::new(
        "conflict",
        entry_name,
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    ));
    module
}

fn partial_wave_module() -> Module {
    let mut block = BasicBlock::new(BlockId(0));
    block.operations.push(op(
        0,
        Type::Scalar(ScalarType::U32),
        OperationKind::Wave(WaveOperation::full(
            WaveOperationKind::LaneId,
            WaveWidth::Wave32,
        )),
    ));
    block.terminator = Some(Terminator::Return { values: vec![] });
    let capabilities = std::collections::BTreeSet::from([
        TargetCapability::Subgroups,
        TargetCapability::SubgroupSize(32),
        TargetCapability::WaveWidth(WaveWidth::Wave32),
    ]);
    let mut entry = Function::kernel_entry(
        "partial_wave_impl",
        Signature::new(vec![], vec![]),
        vec![],
        vec![block],
    );
    entry.required_capabilities = capabilities.clone();
    let mut kernel = Kernel::new(
        "partial_wave",
        "partial_wave_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.required_capabilities = capabilities.clone();
    let mut module = Module::new("cli-command-wave-test");
    module.required_capabilities = capabilities;
    module.functions.push(entry);
    module.kernels.push(kernel);
    module
}

fn write_success_fixture(directory: &TestDirectory) -> (PathBuf, PathBuf) {
    let kir = directory.path().join("kernel.kir");
    let request = directory.path().join("request.json");
    fs::write(&kir, canonical_noop_with_buffer()).unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[2,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
    )
    .unwrap();
    (kir, request)
}

#[test]
fn help_is_a_successful_input_free_command() {
    for argument in ["--help", "-h"] {
        let output = binary().arg(argument).env_clear().output().unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "usage: fe2o3-kir-sim (--kir-v7 PATH | --kir-v12 PATH | --diagnostic-kir-v16 PATH | --diagnostic-kir-v17 PATH | --diagnostic-kir-v18 PATH | --diagnostic-kir-v19 PATH | --bundle PATH | --bundle-v5 PATH | --bundle-v6 PATH) --request PATH [--output PATH] [--race-evidence] [--matrix-bf16-numerics exact-v1|finite-sequential-fma-v1] [--record-canonical-schedule PATH [--schedule-max-decisions COUNT] | --record-seeded-schedule PATH --schedule-seed U64 [--schedule-max-decisions COUNT] | --replay-schedule PATH | --explore-seeded-schedules COUNT --schedule-seed FIRST_U64 [--schedule-max-decisions COUNT] [--exploration-max-retained-decisions COUNT] | --reduce-failure [--schedule-seed U64] [--schedule-max-decisions COUNT] | --replay-failure-reduction PATH] (diagnostic V16/V17/V18/V19 do not support schedule options)\n"
        );
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn bundle_v6_executes_and_records_an_exact_replayable_schedule() {
    let directory = TestDirectory::new();
    let bundle_path = directory.path().join("kernel-v6.fe2sim");
    let request = directory.path().join("request-v6.json");
    let schedule = directory.path().join("schedule-v6.json");
    let bundle = simulation_bundle_v6();
    fs::write(&bundle_path, bundle.canonical_bytes()).unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"bundle_v6_kernel","grid":[2,1,1],"workgroup":[1,1,1],"arguments":[]}"#,
    )
    .unwrap();

    let recorded = binary()
        .arg("--bundle-v6")
        .arg(&bundle_path)
        .arg("--request")
        .arg(&request)
        .arg("--record-canonical-schedule")
        .arg(&schedule)
        .args(["--schedule-max-decisions", "8"])
        .output()
        .unwrap();
    assert!(
        recorded.status.success(),
        "{}",
        String::from_utf8_lossy(&recorded.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&recorded.stdout).unwrap();
    assert_eq!(result["status"], "ok");
    assert_eq!(result["counts"]["invocations_executed"], 2);

    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&schedule).unwrap()).unwrap();
    assert_eq!(persisted["artifact"]["kind"], "simulation_bundle_v6");
    assert_eq!(
        persisted["artifact"]["bundle_sha256"],
        hex_identity(bundle.identity().as_bytes())
    );
    assert_eq!(
        persisted["artifact"]["subject_sha256"],
        hex_identity(bundle.subject_identity())
    );
    assert_eq!(
        persisted["artifact"]["kir_sha256"],
        hex_identity(bundle.canonical_kir_v11_digest())
    );

    let replayed = binary()
        .arg("--bundle-v6")
        .arg(&bundle_path)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(
        replayed.status.success(),
        "{}",
        String::from_utf8_lossy(&replayed.stderr)
    );
    let replayed: serde_json::Value = serde_json::from_slice(&replayed.stdout).unwrap();
    assert_eq!(replayed["status"], "ok");
    assert_eq!(
        replayed["schedule"]["transcript_sha256"],
        result["schedule"]["transcript_sha256"]
    );
    assert_eq!(replayed["arguments"], result["arguments"]);
}

#[test]
fn schedule_modes_and_record_only_parameters_are_mutually_exclusive() {
    for arguments in [
        vec![
            "--record-canonical-schedule",
            "canonical.json",
            "--replay-schedule",
            "replay.json",
        ],
        vec![
            "--record-canonical-schedule",
            "canonical.json",
            "--record-seeded-schedule",
            "seeded.json",
            "--schedule-seed",
            "7",
        ],
        vec!["--record-seeded-schedule", "seeded.json"],
        vec!["--replay-schedule", "replay.json", "--schedule-seed", "7"],
        vec![
            "--replay-schedule",
            "replay.json",
            "--schedule-max-decisions",
            "8",
        ],
        vec![
            "--record-canonical-schedule",
            "canonical.json",
            "--explore-seeded-schedules",
            "2",
            "--schedule-seed",
            "7",
        ],
        vec![
            "--replay-schedule",
            "replay.json",
            "--exploration-max-retained-decisions",
            "8",
        ],
        vec!["--race-evidence", "--race-evidence"],
    ] {
        let output = binary()
            .args(["--kir-v7", "missing.kir", "--request", "missing.json"])
            .args(arguments)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["stage"], "arguments");
        assert_eq!(error["kind"], "invalid_command_line");
    }
}

#[test]
fn bundle_targets_execute_exact_embedded_kir_without_authority() {
    for target in ["gfx942:xnack-", "gfx950:xnack-"] {
        let directory = TestDirectory::new();
        let bundle_path = directory.path().join("kernel.fe2sim");
        let request = directory.path().join("request.json");
        fs::write(&bundle_path, simulation_bundle(target)).unwrap();
        fs::write(
            &request,
            br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[2,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
        )
        .unwrap();

        let admitted =
            fe2o3_kir_sim_cli::load_debug_simulation_bundle_v1(&bundle_path, &request).unwrap();
        assert_eq!(admitted.bundle().target(), target);
        assert_eq!(
            admitted.input().simulation_target(),
            fe2o3_kir_sim::SimulationTargetV1::amdgpu_from_device_target(target).unwrap()
        );
        assert_eq!(
            admitted.input().simulation_bundle_subject(),
            Some(*admitted.bundle().subject_identity())
        );
        assert_eq!(
            admitted.input().simulation_bundle_identity(),
            Some(*admitted.bundle().identity().as_bytes())
        );
        assert!(!admitted.grants_proof_authority());
        assert!(!admitted.grants_artifact_authority());
        assert!(!admitted.grants_compiler_authority());
        assert!(!admitted.grants_hardware_authority());
        assert!(!admitted.grants_load_authority());
        assert!(!admitted.grants_launch_authority());
        assert!(!admitted.authenticates_compiler_execution());

        let output = binary()
            .arg("--bundle")
            .arg(bundle_path)
            .arg("--request")
            .arg(request)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["status"], "ok");
        assert_eq!(value["authority"], "observation_only");
        assert_eq!(value["simulated"], true);
        assert_eq!(value["hardware_observed"], false);
        assert_eq!(value["performance_prediction"], false);
    }
}

#[test]
fn bundle_command_line_is_exclusive_and_hostile_bundles_fail_closed() {
    let directory = TestDirectory::new();
    let request = directory.path().join("request.json");
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[1,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
    )
    .unwrap();
    let valid = simulation_bundle("gfx942:xnack-");
    let bundle = directory.path().join("kernel.fe2sim");
    fs::write(&bundle, &valid).unwrap();
    let kir = directory.path().join("kernel.kir");
    fs::write(&kir, canonical_noop_with_buffer()).unwrap();

    let exclusive = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--bundle")
        .arg(&bundle)
        .arg("--request")
        .arg(&request)
        .output()
        .unwrap();
    assert!(!exclusive.status.success());
    let error: serde_json::Value = serde_json::from_slice(&exclusive.stderr).unwrap();
    assert_eq!(error["stage"], "arguments");
    assert_eq!(error["kind"], "invalid_command_line");

    let non_regular = binary()
        .args(["--bundle", "/dev/null"])
        .arg("--request")
        .arg(&request)
        .output()
        .unwrap();
    assert!(!non_regular.status.success());
    let error: serde_json::Value = serde_json::from_slice(&non_regular.stderr).unwrap();
    assert_eq!(error["kind"], "input_not_regular");
    assert_eq!(error["input"], "simulation_bundle");

    for (name, bytes) in [
        ("corrupted", {
            let mut bytes = valid.clone();
            bytes[0] ^= 1;
            bytes
        }),
        ("trailing", {
            let mut bytes = valid.clone();
            bytes.push(0);
            bytes
        }),
    ] {
        let path = directory.path().join(format!("{name}.fe2sim"));
        fs::write(&path, bytes).unwrap();
        let output = binary()
            .arg("--bundle")
            .arg(path)
            .arg("--request")
            .arg(&request)
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
        assert_eq!(error["stage"], "input");
        assert_eq!(error["kind"], "simulation_bundle_rejected");
        assert_eq!(error["input"], "simulation_bundle");
    }

    let oversized = directory.path().join("oversized.fe2sim");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(u64::try_from(MAX_SIMULATION_BUNDLE_BYTES_V1).unwrap() + 1)
        .unwrap();
    let output = binary()
        .arg("--bundle")
        .arg(oversized)
        .arg("--request")
        .arg(request)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["kind"], "input_too_large");
    assert_eq!(error["input"], "simulation_bundle");
}

#[test]
fn unsupported_bundle_target_and_request_kernel_mismatch_are_typed() {
    let directory = TestDirectory::new();
    let bundle = directory.path().join("kernel.fe2sim");
    let request = directory.path().join("request.json");
    fs::write(&bundle, simulation_bundle("unmapped-target-v1")).unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[1,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
    )
    .unwrap();
    let output = binary()
        .arg("--bundle")
        .arg(&bundle)
        .arg("--request")
        .arg(&request)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["kind"], "simulation_bundle_target_unsupported");

    fs::write(&bundle, simulation_bundle("gfx942:xnack-")).unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"stale_kernel","grid":[1,1,1],"workgroup":[1,1,1],"arguments":[]}"#,
    )
    .unwrap();
    let output = binary()
        .arg("--bundle")
        .arg(bundle)
        .arg("--request")
        .arg(request)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["stage"], "preflight");
    assert_eq!(error["kind"], "preflight_unknown_kernel");
}

#[test]
fn legacy_kir_flag_is_not_a_second_simulator_route() {
    let output = binary()
        .args(["--kir-v6", "kernel.kir", "--request", "request.json"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(value["stage"], "arguments");
    assert_eq!(value["kind"], "invalid_command_line");
}

#[test]
fn successful_stdout_is_complete_machine_readable_json() {
    let directory = TestDirectory::new();
    let (kir, request) = write_success_fixture(&directory);
    let output = binary()
        .arg("--kir-v7")
        .arg(kir)
        .arg("--request")
        .arg(request)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    assert_eq!(output.stdout.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(value["schema"], "fe2o3-simulation-result-v1");
    assert_eq!(value["status"], "ok");
    assert_eq!(value["authority"], "observation_only");
    assert_eq!(value["simulated"], true);
    assert_eq!(value["hardware_observed"], false);
    assert_eq!(value["hardware_validation"], false);
    assert_eq!(value["performance_prediction"], false);
    assert_eq!(
        value["target_profile"]["identity"],
        "amdgpu_64_little_endian_v1"
    );
    assert_eq!(value["target_profile"]["index_bits"], 64);
    assert_eq!(value["target_profile"]["max_workgroup_invocations"], 1024);
    assert_eq!(
        value["schedule"]["identity"],
        "workgroup_major_local_zyx_cooperative_v1"
    );
    assert_eq!(value["kir"]["sha256"].as_str().unwrap().len(), 64);
    assert!(value["kir"]["canonical_bytes"].as_u64().unwrap() > 0);
    assert_eq!(value["counts"]["invocations_executed"], 2);
    assert_eq!(value["counts"]["scheduled_slots_visited"], 2);
    assert_eq!(value["arguments"][0]["value"]["bytes"], "0x2a");
    assert!(value.get("race_assessment").is_none());
}

#[test]
fn single_run_race_evidence_is_opt_in_and_machine_readable() {
    let directory = TestDirectory::new();
    let kir = directory.path().join("race.kir");
    let request = directory.path().join("race-request.json");
    fs::write(
        &kir,
        VerifiedCanonicalKernelIrV7::from_module(conflicting_store_module("race_impl"))
            .unwrap()
            .into_canonical_bytes(),
    )
    .unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#,
    )
    .unwrap();

    let ordinary = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .output()
        .unwrap();
    assert!(ordinary.status.success());
    let ordinary: serde_json::Value = serde_json::from_slice(&ordinary.stdout).unwrap();
    assert!(ordinary.get("race_assessment").is_none());

    let evidenced = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--race-evidence")
        .output()
        .unwrap();
    assert!(
        evidenced.status.success(),
        "{}",
        String::from_utf8_lossy(&evidenced.stderr)
    );
    let evidenced: serde_json::Value = serde_json::from_slice(&evidenced.stdout).unwrap();
    assert_eq!(evidenced["race_assessment"]["status"], "races_observed");
    assert_eq!(evidenced["race_assessment"]["racing_bytes"], 4);
    assert_eq!(
        evidenced["race_assessment"]["first"]["earlier_atomic"],
        false
    );
    assert_eq!(
        evidenced["race_assessment"]["first"]["conflict"]["earlier_site"]["function_truncated"],
        false
    );
}

#[test]
fn seeded_exploration_is_bounded_agent_native_and_exactly_replayable() {
    let directory = TestDirectory::new();
    let hostile_entry = "race_impl_\"\\\n\u{0001}";
    let module = conflicting_store_module(hostile_entry);
    let kir_bytes = VerifiedCanonicalKernelIrV7::from_module(module)
        .unwrap()
        .into_canonical_bytes();
    let kir = directory.path().join("race.kir");
    let request = directory.path().join("race-request.json");
    fs::write(&kir, kir_bytes).unwrap();
    let request_bytes = br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#;
    fs::write(&request, request_bytes).unwrap();

    let explored = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "3",
            "--schedule-seed",
            "18446744073709551614",
            "--schedule-max-decisions",
            "16",
            "--exploration-max-retained-decisions",
            "16",
        ])
        .output()
        .unwrap();
    assert!(
        explored.status.success(),
        "{}",
        String::from_utf8_lossy(&explored.stderr)
    );
    assert!(explored.stderr.is_empty());
    assert_eq!(explored.stdout.last(), Some(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&explored.stdout).unwrap();
    assert_eq!(value["schema"], "fe2o3-simulation-exploration-v1");
    assert_eq!(value["authority"], "observation_only");
    assert_eq!(value["hardware_observed"], false);
    assert_eq!(value["performance_prediction"], false);
    assert_eq!(value["schedule_space_exhausted"], false);
    assert_eq!(value["exploration"]["seed_interval_wraps"], true);
    assert_eq!(value["exploration"]["attempted"], 3);
    assert_eq!(value["exploration"]["completed"], 3);
    assert_eq!(value["exploration"]["races_observed"], 3);
    assert_eq!(value["exploration"]["requested_seed_budget_consumed"], true);
    assert_eq!(value["exploration"]["witness_retention_exhausted"], false);

    let witness = &value["witnesses"]["first_race"];
    assert_eq!(witness["assessment"]["status"], "races_observed");
    assert_eq!(
        witness["assessment"]["first"]["conflict"]["earlier_site"]["function"],
        hostile_entry
    );
    let schedule = witness["replay_schedule"]["document"].as_str().unwrap();
    assert_eq!(witness["replay_schedule"]["bytes"], schedule.len());
    let mut schedule_sha256 = String::with_capacity(64);
    for byte in Sha256::digest(schedule.as_bytes()) {
        write!(&mut schedule_sha256, "{byte:02x}").unwrap();
    }
    assert_eq!(witness["replay_schedule"]["sha256"], schedule_sha256);
    let schedule_value: serde_json::Value = serde_json::from_str(schedule).unwrap();
    assert_eq!(schedule_value["schema"], "fe2o3-simulation-schedule-v1");
    assert_eq!(schedule_value["artifact"]["kind"], "canonical_kir_v7");
    assert_eq!(schedule_value["schedule"]["seed"], witness["seed"]);

    let schedule_path = directory.path().join("race-schedule.json");
    fs::write(&schedule_path, schedule).unwrap();
    let replayed = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule_path)
        .arg("--race-evidence")
        .output()
        .unwrap();
    assert!(
        replayed.status.success(),
        "{}",
        String::from_utf8_lossy(&replayed.stderr)
    );
    let replayed: serde_json::Value = serde_json::from_slice(&replayed.stdout).unwrap();
    assert_eq!(replayed["race_assessment"], witness["assessment"]);

    let stale_request = directory.path().join("stale-request.json");
    let mut stale_bytes = request_bytes.to_vec();
    stale_bytes.push(b'\n');
    fs::write(&stale_request, stale_bytes).unwrap();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(stale_request)
        .arg("--replay-schedule")
        .arg(schedule_path)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let rejected: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
    assert_eq!(rejected["kind"], "schedule_binding_mismatch");
}

#[test]
fn failure_reduction_is_agent_readable_and_cli_replayable() {
    let directory = TestDirectory::new();
    let kir = directory.path().join("race.kir");
    let request = directory.path().join("race-request.json");
    let report = directory.path().join("failure.json");
    fs::write(
        &kir,
        VerifiedCanonicalKernelIrV7::from_module(conflicting_store_module("race_impl"))
            .unwrap()
            .into_canonical_bytes(),
    )
    .unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#,
    )
    .unwrap();

    let reduced = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--reduce-failure")
        .args(["--schedule-seed", "9", "--schedule-max-decisions", "16"])
        .arg("--output")
        .arg(&report)
        .output()
        .unwrap();
    assert!(
        reduced.status.success(),
        "{}",
        String::from_utf8_lossy(&reduced.stderr)
    );
    assert!(reduced.stdout.is_empty());
    let report_bytes = fs::read(&report).unwrap();
    assert!(!report_bytes.ends_with(b"\n"));
    let report_json: serde_json::Value = serde_json::from_slice(&report_bytes).unwrap();
    assert_eq!(
        report_json["schema"],
        "fe2o3-simulation-failure-reduction-v1"
    );
    assert_eq!(report_json["fingerprint"]["class"], "data_race");
    assert_eq!(report_json["coverage"]["one_shorter_checked"], true);
    assert_eq!(report_json["grants_execution_authority"], false);
    assert_eq!(report_json["predicts_hardware_timing"], false);

    let replayed = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-failure-reduction")
        .arg(&report)
        .output()
        .unwrap();
    assert!(
        replayed.status.success(),
        "{}",
        String::from_utf8_lossy(&replayed.stderr)
    );
    assert_eq!(replayed.stdout, report_bytes);

    let hostile = directory.path().join("hostile.json");
    let mut noncanonical = fs::read(&report).unwrap();
    noncanonical.insert(0, b' ');
    fs::write(&hostile, noncanonical).unwrap();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-failure-reduction")
        .arg(&hostile)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
}

#[test]
fn exploration_witness_preserves_bundle_route_and_rejects_raw_substitution() {
    let directory = TestDirectory::new();
    let module = conflicting_store_module("bundle_race_impl");
    let raw = directory.path().join("bundle-race.kir");
    fs::write(
        &raw,
        VerifiedCanonicalKernelIrV7::from_module(module.clone())
            .unwrap()
            .into_canonical_bytes(),
    )
    .unwrap();
    let bundle = directory.path().join("bundle-race.fe2sim");
    fs::write(
        &bundle,
        simulation_bundle_for_module(module, "gfx942:xnack-", None),
    )
    .unwrap();
    let request = directory.path().join("bundle-race-request.json");
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#,
    )
    .unwrap();
    let explored = binary()
        .arg("--bundle")
        .arg(&bundle)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "1",
            "--schedule-seed",
            "7",
            "--schedule-max-decisions",
            "8",
            "--exploration-max-retained-decisions",
            "8",
        ])
        .output()
        .unwrap();
    assert!(explored.status.success());
    let explored: serde_json::Value = serde_json::from_slice(&explored.stdout).unwrap();
    assert_eq!(explored["input"]["kind"], "simulation_bundle_v1");
    let schedule = explored["witnesses"]["first_race"]["replay_schedule"]["document"]
        .as_str()
        .unwrap();
    let schedule_value: serde_json::Value = serde_json::from_str(schedule).unwrap();
    assert_eq!(schedule_value["artifact"]["kind"], "simulation_bundle_v1");
    let schedule_path = directory.path().join("bundle-witness.json");
    fs::write(&schedule_path, schedule).unwrap();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(raw)
        .arg("--request")
        .arg(request)
        .arg("--replay-schedule")
        .arg(schedule_path)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let rejected: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
    assert_eq!(rejected["kind"], "schedule_binding_mismatch");
}

#[test]
fn exploration_limits_and_dynamic_exhaustion_are_typed_bounded_results() {
    let directory = TestDirectory::new();
    let kir = directory.path().join("race.kir");
    let request = directory.path().join("race-request.json");
    fs::write(
        &kir,
        VerifiedCanonicalKernelIrV7::from_module(conflicting_store_module("race_impl"))
            .unwrap()
            .into_canonical_bytes(),
    )
    .unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#,
    )
    .unwrap();
    for extra in [
        vec!["--explore-seeded-schedules", "0", "--schedule-seed", "1"],
        vec!["--explore-seeded-schedules", "4097", "--schedule-seed", "1"],
        vec!["--explore-seeded-schedules", "1"],
        vec![
            "--explore-seeded-schedules",
            "1",
            "--schedule-seed",
            "1",
            "--exploration-max-retained-decisions",
            "65537",
        ],
    ] {
        let rejected = binary()
            .arg("--kir-v7")
            .arg(&kir)
            .arg("--request")
            .arg(&request)
            .args(extra)
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        let error: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
        assert_eq!(error["kind"], "invalid_command_line");
    }
    let above_decision_limit = (MAX_SCHEDULE_DECISIONS_V1 + 1).to_string();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "1",
            "--schedule-seed",
            "1",
            "--schedule-max-decisions",
            &above_decision_limit,
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());

    let retention_limited = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "1",
            "--schedule-seed",
            "1",
            "--schedule-max-decisions",
            "8",
            "--exploration-max-retained-decisions",
            "1",
        ])
        .output()
        .unwrap();
    assert!(retention_limited.status.success());
    let retention_limited: serde_json::Value =
        serde_json::from_slice(&retention_limited.stdout).unwrap();
    assert_eq!(
        retention_limited["exploration"]["witness_retention_exhausted"],
        true
    );
    assert!(retention_limited["witnesses"]["first_race"].is_null());

    let decision_limited = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "2",
            "--schedule-seed",
            "1",
            "--schedule-max-decisions",
            "1",
            "--exploration-max-retained-decisions",
            "2",
        ])
        .output()
        .unwrap();
    assert!(decision_limited.status.success());
    let decision_limited: serde_json::Value =
        serde_json::from_slice(&decision_limited.stdout).unwrap();
    assert_eq!(decision_limited["exploration"]["failures"], 2);
    assert_eq!(
        decision_limited["first_failure"]["kind"],
        "execution_schedule_decision_limit"
    );
    assert_eq!(
        decision_limited["exploration"]["requested_seed_budget_consumed"],
        true
    );
}

#[test]
fn wave_failures_have_exact_structured_details_in_error_and_exploration() {
    let directory = TestDirectory::new();
    let kir = directory.path().join("partial-wave.kir");
    let request = directory.path().join("partial-wave-request.json");
    fs::write(
        &kir,
        VerifiedCanonicalKernelIrV7::from_module(partial_wave_module())
            .unwrap()
            .into_canonical_bytes(),
    )
    .unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"partial_wave","grid":[1,1,1],"workgroup":[1,1,1],"arguments":[]}"#,
    )
    .unwrap();
    let failed = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let failed: serde_json::Value = serde_json::from_slice(&failed.stderr).unwrap();
    assert_eq!(failed["kind"], "execution_incomplete_wave");
    assert_eq!(failed["detail"]["kind"], "incomplete_wave");
    assert_eq!(failed["detail"]["width"], "wave32");
    assert_eq!(failed["detail"]["wave_in_workgroup"], 0);
    assert_eq!(failed["detail"]["active_mask"], "0x00000001");
    assert_eq!(failed["detail"]["required_mask"], "0xffffffff");

    let explored = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .args([
            "--explore-seeded-schedules",
            "1",
            "--schedule-seed",
            "9",
            "--schedule-max-decisions",
            "4",
            "--exploration-max-retained-decisions",
            "4",
        ])
        .output()
        .unwrap();
    assert!(explored.status.success());
    let explored: serde_json::Value = serde_json::from_slice(&explored.stdout).unwrap();
    assert_eq!(explored["exploration"]["failures"], 1);
    assert_eq!(
        explored["first_failure"]["kind"],
        "execution_incomplete_wave"
    );
    assert_eq!(
        explored["first_failure"]["detail"]["active_mask"],
        "0x00000001"
    );
    assert_eq!(explored["schedule_space_exhausted"], false);
}

#[test]
fn canonical_and_seeded_schedules_persist_and_replay_for_raw_kir() {
    let directory = TestDirectory::new();
    let (kir, request) = write_success_fixture(&directory);

    for (record_flag, seed, expected_identity) in [
        (
            "--record-canonical-schedule",
            None,
            "workgroup_major_local_zyx_cooperative_v1",
        ),
        (
            "--record-seeded-schedule",
            Some("18446744073709551615"),
            "workgroup_major_seeded_runnable_cooperative_v1",
        ),
    ] {
        let schedule = directory
            .path()
            .join(format!("{}.json", record_flag.trim_start_matches("--")));
        let mut record = binary();
        record
            .arg("--kir-v7")
            .arg(&kir)
            .arg("--request")
            .arg(&request)
            .arg(record_flag)
            .arg(&schedule)
            .args(["--schedule-max-decisions", "8"]);
        if let Some(seed) = seed {
            record.args(["--schedule-seed", seed]);
        }
        let output = record.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["schedule"]["identity"], expected_identity);
        let schedule_bytes = fs::read(&schedule).unwrap();
        assert!(!schedule_bytes.ends_with(b"\n"));
        assert_eq!(
            fs::metadata(&schedule).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let persisted: serde_json::Value = serde_json::from_slice(&schedule_bytes).unwrap();
        assert_eq!(persisted["schema"], "fe2o3-simulation-schedule-v1");
        assert_eq!(persisted["artifact"]["kind"], "canonical_kir_v7");
        assert_eq!(persisted["schedule"]["identity"], expected_identity);
        assert_eq!(persisted["coverage"]["complete"], true);
        assert_eq!(persisted["decisions"].as_array().unwrap().len(), 2);
        if seed.is_some() {
            let mut schedule_sha256 = String::with_capacity(64);
            for byte in Sha256::digest(&schedule_bytes) {
                write!(&mut schedule_sha256, "{byte:02x}").unwrap();
            }
            assert_eq!(schedule_bytes.len(), 1_346);
            assert_eq!(
                schedule_sha256,
                "a3d0a28479bf6ee12a9bb10745903a208f815213aa94771768407b662eb369cc"
            );
        }

        let replay = binary()
            .arg("--kir-v7")
            .arg(&kir)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(&schedule)
            .output()
            .unwrap();
        assert!(
            replay.status.success(),
            "{}",
            String::from_utf8_lossy(&replay.stderr)
        );
        let replayed: serde_json::Value = serde_json::from_slice(&replay.stdout).unwrap();
        assert_eq!(replayed["schedule"]["identity"], expected_identity);
        assert_eq!(
            replayed["schedule"]["transcript_sha256"],
            result["schedule"]["transcript_sha256"]
        );
        assert_eq!(replayed["arguments"], result["arguments"]);
    }
}

#[test]
fn bundle_schedule_binds_exact_bundle_subject_and_embedded_kir() {
    let directory = TestDirectory::new();
    let bundle = directory.path().join("kernel.fe2sim");
    let request = directory.path().join("request.json");
    let schedule = directory.path().join("schedule.json");
    fs::write(
        &bundle,
        simulation_bundle_with_debug_map("gfx942:xnack-", Some(b"debug-map-a")),
    )
    .unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[2,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
    )
    .unwrap();

    let recorded = binary()
        .arg("--bundle")
        .arg(&bundle)
        .arg("--request")
        .arg(&request)
        .arg("--record-canonical-schedule")
        .arg(&schedule)
        .args(["--schedule-max-decisions", "8"])
        .output()
        .unwrap();
    assert!(
        recorded.status.success(),
        "{}",
        String::from_utf8_lossy(&recorded.stderr)
    );
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&schedule).unwrap()).unwrap();
    assert_eq!(persisted["artifact"]["kind"], "simulation_bundle_v1");
    assert_eq!(
        persisted["artifact"]["bundle_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(
        persisted["artifact"]["subject_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );

    let replayed = binary()
        .arg("--bundle")
        .arg(&bundle)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(
        replayed.status.success(),
        "{}",
        String::from_utf8_lossy(&replayed.stderr)
    );

    let substituted_bundle = directory.path().join("substituted.fe2sim");
    fs::write(
        &substituted_bundle,
        simulation_bundle_with_debug_map("gfx942:xnack-", Some(b"debug-map-b")),
    )
    .unwrap();
    let substituted = binary()
        .arg("--bundle")
        .arg(substituted_bundle)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(!substituted.status.success());
    assert!(substituted.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&substituted.stderr).unwrap();
    assert_eq!(error["kind"], "schedule_binding_mismatch");

    let raw_kir = directory.path().join("kernel.kir");
    fs::write(&raw_kir, canonical_noop_with_buffer()).unwrap();
    let substituted = binary()
        .arg("--kir-v7")
        .arg(raw_kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(!substituted.status.success());
    let error: serde_json::Value = serde_json::from_slice(&substituted.stderr).unwrap();
    assert_eq!(error["kind"], "schedule_binding_mismatch");
    assert_eq!(error["input"], "semantic_schedule");
}

#[test]
fn hostile_schedule_inputs_and_stale_requests_fail_before_execution() {
    let directory = TestDirectory::new();
    let (kir, request) = write_success_fixture(&directory);
    let schedule = directory.path().join("schedule.json");
    let recorded = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--record-canonical-schedule")
        .arg(&schedule)
        .args(["--schedule-max-decisions", "8"])
        .output()
        .unwrap();
    assert!(recorded.status.success());

    let original_request = fs::read(&request).unwrap();
    let mut stale_request = original_request.clone();
    stale_request.push(b'\n');
    fs::write(&request, stale_request).unwrap();
    let stale = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(!stale.status.success());
    let error: serde_json::Value = serde_json::from_slice(&stale.stderr).unwrap();
    assert_eq!(error["kind"], "schedule_binding_mismatch");
    fs::write(&request, original_request).unwrap();

    let variant_kir = directory.path().join("variant.kir");
    fs::write(&variant_kir, canonical_noop_with_buffer_variant()).unwrap();
    let substituted = binary()
        .arg("--kir-v7")
        .arg(&variant_kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&schedule)
        .output()
        .unwrap();
    assert!(!substituted.status.success());
    let error: serde_json::Value = serde_json::from_slice(&substituted.stderr).unwrap();
    assert_eq!(error["kind"], "schedule_binding_mismatch");

    let schedule_text = String::from_utf8(fs::read(&schedule).unwrap()).unwrap();
    for (name, substituted) in [
        (
            "limits",
            schedule_text.replacen("\"max_events\":1,", "\"max_events\":2,", 1),
        ),
        (
            "target",
            schedule_text.replacen(
                "\"identity\":\"amdgpu_64_little_endian_v1\",\"index_bits\":64",
                "\"identity\":\"little_endian_index32_v1\",\"index_bits\":32",
                1,
            ),
        ),
    ] {
        let path = directory.path().join(format!("{name}-substitution.json"));
        fs::write(&path, substituted).unwrap();
        let rejected = binary()
            .arg("--kir-v7")
            .arg(&kir)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(path)
            .output()
            .unwrap();
        assert!(!rejected.status.success(), "{name}");
        let error: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
        assert_eq!(error["kind"], "schedule_binding_mismatch", "{name}");
    }

    let noncanonical = directory.path().join("noncanonical.json");
    let mut bytes = fs::read(&schedule).unwrap();
    bytes.push(b'\n');
    fs::write(&noncanonical, bytes).unwrap();
    let invalid = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--replay-schedule")
        .arg(&noncanonical)
        .output()
        .unwrap();
    assert!(!invalid.status.success());
    let error: serde_json::Value = serde_json::from_slice(&invalid.stderr).unwrap();
    assert_eq!(error["kind"], "schedule_codec_rejected");

    let linked = directory.path().join("linked.json");
    symlink(&schedule, &linked).unwrap();
    for path in [linked, PathBuf::from("/dev/null")] {
        let rejected = binary()
            .arg("--kir-v7")
            .arg(&kir)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(path)
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        let error: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
        assert!(matches!(
            error["kind"].as_str(),
            Some("input_open_failed" | "input_not_regular")
        ));
        assert_eq!(error["input"], "semantic_schedule");
    }

    let oversized = directory.path().join("oversized.json");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(u64::try_from(MAX_PERSISTED_SCHEDULE_BYTES_V1).unwrap() + 1)
        .unwrap();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(kir)
        .arg("--request")
        .arg(request)
        .arg("--replay-schedule")
        .arg(oversized)
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    let error: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
    assert_eq!(error["kind"], "input_too_large");
}

#[test]
fn schedule_recording_is_no_replace_and_emits_nothing_on_execution_failure() {
    let directory = TestDirectory::new();
    let (kir, request) = write_success_fixture(&directory);
    let existing = directory.path().join("existing.json");
    fs::write(&existing, b"retained").unwrap();
    let rejected = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--record-canonical-schedule")
        .arg(&existing)
        .args(["--schedule-max-decisions", "8"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
    assert_eq!(fs::read(&existing).unwrap(), b"retained");

    let absent = directory.path().join("absent.json");
    let failed = binary()
        .arg("--kir-v7")
        .arg(kir)
        .arg("--request")
        .arg(request)
        .arg("--record-canonical-schedule")
        .arg(&absent)
        .args(["--schedule-max-decisions", "1"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
    assert!(!absent.exists());
    let error: serde_json::Value = serde_json::from_slice(&failed.stderr).unwrap();
    assert_eq!(error["kind"], "execution_schedule_decision_limit");
}

#[test]
fn successful_output_is_private_durable_no_replace_json() {
    let directory = TestDirectory::new();
    let (kir, request) = write_success_fixture(&directory);
    let result = directory.path().join("result.json");
    let output = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--output")
        .arg(&result)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    let bytes = fs::read(&result).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["status"], "ok");
    assert_eq!(
        fs::metadata(&result).unwrap().permissions().mode() & 0o777,
        0o600
    );

    let second = binary()
        .arg("--kir-v7")
        .arg(kir)
        .arg("--request")
        .arg(request)
        .arg("--output")
        .arg(&result)
        .output()
        .unwrap();
    assert!(!second.status.success());
    let error: serde_json::Value = serde_json::from_slice(&second.stderr).unwrap();
    assert_eq!(error["stage"], "output");
    assert_eq!(error["kind"], "output_already_exists");
    assert_eq!(fs::read(result).unwrap(), bytes);
}

#[test]
fn failures_are_stable_json_with_input_identity() {
    let output = binary().output().unwrap();
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(value["stage"], "arguments");
    assert_eq!(value["kind"], "invalid_command_line");

    let output = binary()
        .args(["--kir-v7", "/dev/null", "--request", "/dev/null"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(value["stage"], "input");
    assert_eq!(value["kind"], "input_not_regular");
    assert_eq!(value["input"], "kir_v7");

    let directory = TestDirectory::new();
    let oversized = directory.path().join("oversized.kir");
    fs::File::create(&oversized)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    let output = binary()
        .arg("--kir-v7")
        .arg(&oversized)
        .args(["--request", "/dev/null"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(value["kind"], "input_too_large");
    assert_eq!(value["input"], "kir_v7");
}

#[test]
fn recorded_schedule_is_disclosed_when_later_stdout_epipe_occurs() {
    let directory = TestDirectory::new();
    let kir = directory.path().join("kernel.kir");
    let request = directory.path().join("request.json");
    let schedule = directory.path().join("schedule.json");
    fs::write(&kir, canonical_noop_with_buffer()).unwrap();
    let bytes = "00".repeat(4 * 1024 * 1024);
    fs::write(
        &request,
        format!(
            "{{\"schema\":\"fe2o3-simulation-request-v1\",\"kernel\":\"kernel\",\"grid\":[1,1,1],\"workgroup\":[1,1,1],\"arguments\":[{{\"kind\":\"buffer\",\"element\":\"u8\",\"access\":\"read_only\",\"alignment\":1,\"bytes\":\"0x{bytes}\"}}]}}"
        ),
    )
    .unwrap();

    let mut child = binary()
        .arg("--kir-v7")
        .arg(&kir)
        .arg("--request")
        .arg(&request)
        .arg("--record-canonical-schedule")
        .arg(&schedule)
        .args(["--schedule-max-decisions", "8"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let value: serde_json::Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(value["stage"], "output");
    assert_eq!(value["kind"], "output_write_failed");
    assert_eq!(value["schedule_published"], true);
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(schedule).unwrap()).unwrap();
    assert_eq!(persisted["schema"], "fe2o3-simulation-schedule-v1");
}

#[test]
fn bound_request_rejects_post_admission_substitution_without_output() {
    let directory = TestDirectory::new();
    let (_, request) = write_success_fixture(&directory);
    let identity = fe2o3_kir_sim_cli::bind_request_v1(&request).unwrap();
    fs::write(
        &request,
        br#"{"schema":"fe2o3-simulation-request-v1","kernel":"kernel","grid":[3,1,1],"workgroup":[1,1,1],"arguments":[{"kind":"buffer","element":"u8","access":"read_only","alignment":1,"bytes":"0x2a"}]}"#,
    )
    .unwrap();
    let output = directory.path().join("result.json");

    let status = fe2o3_kir_sim_cli::run_captured_kir_v7_with_bound_request(
        &canonical_noop_with_buffer(),
        &request,
        identity,
        Some(&output),
    );

    assert_ne!(status, std::process::ExitCode::SUCCESS);
    assert!(!output.exists());
}

#[test]
fn exact_bundle_profiles_survive_record_exploration_publication_and_replay_refusal() {
    for (device, tag, other) in [
        (
            "gfx942:xnack-",
            "amdgpu_gfx942_little_endian_v2",
            "amdgpu_gfx950_little_endian_v2",
        ),
        (
            "gfx950:xnack-",
            "amdgpu_gfx950_little_endian_v2",
            "amdgpu_gfx942_little_endian_v2",
        ),
    ] {
        let directory = TestDirectory::new();
        let (_, request) = write_success_fixture(&directory);
        let bundle = directory.path().join("profile.fe2sim");
        let schedule = directory.path().join("schedule.json");
        fs::write(&bundle, simulation_bundle(device)).unwrap();
        let recorded = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .arg("--record-canonical-schedule")
            .arg(&schedule)
            .output()
            .unwrap();
        assert!(
            recorded.status.success(),
            "{}",
            String::from_utf8_lossy(&recorded.stderr)
        );
        let actual: serde_json::Value = serde_json::from_slice(&recorded.stdout).unwrap();
        assert_eq!(actual["target_profile"]["identity"], tag);
        assert_eq!(actual["target_profile"]["index_bits"], 64);
        assert_eq!(actual["target_profile"]["max_workgroup_invocations"], 1024);
        assert_eq!(actual["hardware_validation"], false);
        assert_eq!(actual["performance_prediction"], false);
        let encoded = fs::read(&schedule).unwrap();
        let decoded =
            fe2o3_kir_sim::PersistedSimulationScheduleDocumentV1::from_canonical_bytes(&encoded)
                .unwrap();
        assert_eq!(decoded.binding().target().identity_tag(), tag);
        let loaded = fe2o3_kir_sim_cli::load_debug_simulation_bundle_v1(&bundle, &request).unwrap();
        assert_eq!(
            loaded.input().simulation_target(),
            decoded.binding().target()
        );
        assert_eq!(
            loaded
                .input()
                .retained_bundle_target_v1()
                .unwrap()
                .unwrap()
                .target()
                .as_str(),
            device
        );

        let published = directory.path().join("published.json");
        let replayed = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(&schedule)
            .arg("--output")
            .arg(&published)
            .output()
            .unwrap();
        assert!(
            replayed.status.success(),
            "{}",
            String::from_utf8_lossy(&replayed.stderr)
        );
        assert!(replayed.stdout.is_empty());
        let published: serde_json::Value =
            serde_json::from_slice(&fs::read(&published).unwrap()).unwrap();
        assert_eq!(published["target_profile"], actual["target_profile"]);

        let explored = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .args([
                "--explore-seeded-schedules",
                "2",
                "--schedule-seed",
                "17",
                "--schedule-max-decisions",
                "8",
                "--exploration-max-retained-decisions",
                "16",
            ])
            .output()
            .unwrap();
        assert!(
            explored.status.success(),
            "{}",
            String::from_utf8_lossy(&explored.stderr)
        );
        let explored: serde_json::Value = serde_json::from_slice(&explored.stdout).unwrap();
        assert_eq!(explored["target_profile"], actual["target_profile"]);
        assert_eq!(explored["hardware_observed"], false);
        assert_eq!(explored["exploration"]["completed"], 2);

        for (index, substitute) in [other, "amdgpu_64_little_endian_v1"]
            .into_iter()
            .enumerate()
        {
            let tampered = std::str::from_utf8(&encoded)
                .unwrap()
                .replace(tag, substitute);
            assert_ne!(tampered.as_bytes(), encoded);
            fs::write(&schedule, tampered).unwrap();
            let output = directory.path().join(format!("rejected-{index}.json"));
            let rejected = binary()
                .arg("--bundle")
                .arg(&bundle)
                .arg("--request")
                .arg(&request)
                .arg("--replay-schedule")
                .arg(&schedule)
                .arg("--output")
                .arg(&output)
                .output()
                .unwrap();
            assert!(!rejected.status.success());
            assert!(rejected.stdout.is_empty());
            assert!(!output.exists());
            assert!(!rejected.stderr.is_empty());
        }
    }
}

#[test]
fn exploration_race_witnesses_retain_each_exact_bundle_profile() {
    for (device, tag, other) in [
        (
            "gfx942:xnack-",
            "amdgpu_gfx942_little_endian_v2",
            "gfx950:xnack-",
        ),
        (
            "gfx950:xnack-",
            "amdgpu_gfx950_little_endian_v2",
            "gfx942:xnack-",
        ),
    ] {
        let directory = TestDirectory::new();
        let bundle = directory.path().join("race.fe2sim");
        let request = directory.path().join("race-request.json");
        let module = conflicting_store_module("profile_race");
        fs::write(
            &bundle,
            simulation_bundle_for_module(module.clone(), device, None),
        )
        .unwrap();
        fs::write(&request,
            br#"{"schema":"fe2o3-simulation-request-v1","kernel":"conflict","grid":[2,1,1],"workgroup":[2,1,1],"arguments":[{"kind":"buffer","element":"u32","access":"read_write","alignment":4,"bytes":"0x00000000"}]}"#).unwrap();
        let explored = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .args([
                "--explore-seeded-schedules",
                "1",
                "--schedule-seed",
                "7",
                "--schedule-max-decisions",
                "8",
                "--exploration-max-retained-decisions",
                "8",
            ])
            .output()
            .unwrap();
        assert!(
            explored.status.success(),
            "{}",
            String::from_utf8_lossy(&explored.stderr)
        );
        let explored: serde_json::Value = serde_json::from_slice(&explored.stdout).unwrap();
        assert_eq!(explored["target_profile"]["identity"], tag);
        let schedule = explored["witnesses"]["first_race"]["replay_schedule"]["document"]
            .as_str()
            .unwrap();
        let decoded = fe2o3_kir_sim::PersistedSimulationScheduleDocumentV1::from_canonical_bytes(
            schedule.as_bytes(),
        )
        .unwrap();
        assert_eq!(decoded.binding().target().identity_tag(), tag);
        let path = directory.path().join("witness.json");
        fs::write(&path, schedule).unwrap();
        let replay = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            replay.status.success(),
            "{}",
            String::from_utf8_lossy(&replay.stderr)
        );
        fs::write(&bundle, simulation_bundle_for_module(module, other, None)).unwrap();
        let rejected = binary()
            .arg("--bundle")
            .arg(&bundle)
            .arg("--request")
            .arg(&request)
            .arg("--replay-schedule")
            .arg(&path)
            .output()
            .unwrap();
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
        let error: serde_json::Value = serde_json::from_slice(&rejected.stderr).unwrap();
        assert_eq!(error["kind"], "schedule_binding_mismatch");
    }
}
