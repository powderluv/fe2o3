#[path = "../../../../examples/workgroup_sync_v1/src/row_affine_oracle.rs"]
mod row_affine_oracle;

struct RowAffineRequest {
    document: Value,
    input: Vec<u8>,
    input_initialized: Vec<bool>,
    expected_output: Vec<u8>,
    expected_initialized: Vec<bool>,
}

fn row_affine_request(case: &row_affine_oracle::RowAffineVectorV1) -> RowAffineRequest {
    const GUARD: [u8; 4] = 0xA5C3_7E19_u32.to_le_bytes();
    let config = case.config;
    let mut expected = case.output.clone();
    row_affine_oracle::row_affine_oracle_v1(&case.input, config, &mut expected).unwrap();
    let backing = |values: &[u32]| {
        let mut bytes = GUARD.to_vec();
        bytes.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        bytes.extend_from_slice(&GUARD);
        bytes
    };
    let input = backing(&case.input);
    let output = backing(&case.output);
    let mut input_initialized = vec![false; input.len()];
    for row in 0..config.rows {
        for column in 0..config.columns {
            let start = 4 * (1 + config.offset + row * config.row_stride + column);
            input_initialized[start..start + 4].fill(true);
        }
    }
    let mut output_initialized = vec![false; output.len()];
    output_initialized[..4].fill(true);
    output_initialized[output.len() - 4..].fill(true);
    let packed = |initialized: &[bool]| {
        let mut bits = vec![0_u8; initialized.len().div_ceil(8)];
        for (index, &value) in initialized.iter().enumerate() {
            if value {
                bits[index / 8] |= 1 << (index % 8);
            }
        }
        format!("0x{}", hex(&bits))
    };
    let scalar64 = |value: usize| {
        json!({
            "kind": "scalar", "type": "u64", "bits": format!("0x{value:016x}"),
        })
    };
    let scalar32 = |value: u32| {
        json!({
            "kind": "scalar", "type": "u32", "bits": format!("0x{value:08x}"),
        })
    };
    let document = json!({
        "schema": "fe2o3-simulation-request-v1",
        "kernel": "row_affine_sum_u32_v1",
        "grid": [config.workgroups * 64, 1, 1],
        "workgroup": [64, 1, 1],
        "arguments": [
            {"kind": "buffer_view", "backing": 7, "element": "u32",
             "access": "read_only", "alignment": 4, "byte_offset": 4,
             "elements": case.input.len()},
            scalar64(config.offset), scalar64(config.rows), scalar64(config.columns),
            scalar64(config.row_stride), scalar32(config.scale), scalar32(config.bias),
            {"kind": "buffer_view", "backing": 11, "element": "u32",
             "access": "read_write", "alignment": 4, "byte_offset": 4,
             "elements": case.output.len()},
        ],
        "shared_buffers": [
            {"id": 7, "element": "u32", "access": "read_only", "alignment": 4,
             "bytes": format!("0x{}", hex(&input)), "initialized": packed(&input_initialized)},
            {"id": 11, "element": "u32", "access": "read_write", "alignment": 4,
             "bytes": format!("0x{}", hex(&output)), "initialized": packed(&output_initialized)},
        ],
    });
    output_initialized[4..4 + config.rows * 4].fill(true);
    RowAffineRequest {
        document,
        input,
        input_initialized,
        expected_output: backing(&expected),
        expected_initialized: output_initialized,
    }
}

fn write_row_affine_request(scratch: &ScratchDirectory, label: &str, document: &Value) -> PathBuf {
    let path = scratch.0.join(format!("{label}.json"));
    std::fs::write(&path, serde_json::to_vec(document).unwrap()).unwrap();
    path
}

fn assert_row_affine_execution(
    execution: &fe2o3_kir_sim::SimulationExecutionV1,
    request: &RowAffineRequest,
    groups: usize,
) {
    use fe2o3_kir_sim::BufferBackingIdV1;
    let input = execution.shared_buffer(BufferBackingIdV1(7)).unwrap();
    let output = execution.shared_buffer(BufferBackingIdV1(11)).unwrap();
    assert_eq!(input.bytes(), request.input);
    assert_eq!(input.initialized(), request.input_initialized);
    assert_eq!(output.bytes(), request.expected_output);
    assert_eq!(output.initialized(), request.expected_initialized);
    assert_eq!(execution.invocations_executed(), (groups * 64) as u64);
    assert_eq!(execution.workgroups_visited(), groups as u64);
    assert!(execution.schedule_coverage().is_complete());
    assert!(execution.schedule_coverage().barrier_releases() >= groups as u64);
}

fn assert_row_affine_bundle(path: &Path, target: &str, source_file: &str) {
    use fe2o3_kernel_ir::{AccessMode, AddressSpace, ScalarType, Type};
    let bundle = fe2o3_kernel_ir::VerifiedSimulationBundleV5::from_canonical_bytes(
        std::fs::read(path).unwrap(),
    )
    .unwrap();
    assert_eq!(bundle.target(), target);
    assert_eq!(bundle.kernel_count(), 1);
    assert_eq!(bundle.production_kir_identity().version(), 8);
    assert!(!bundle.authenticates_compiler_execution());
    assert!(!bundle.grants_compiler_authority());
    assert!(!bundle.grants_artifact_authority());
    assert!(!bundle.grants_hardware_authority());
    assert!(!bundle.grants_load_authority());
    assert!(!bundle.grants_launch_authority());
    let (_, module) =
        fe2o3_kernel_ir::VerifiedCanonicalKernelIrV10::from_canonical_bytes_with_module(
            bundle.canonical_kir_v10().to_vec(),
        )
        .unwrap();
    let [kernel] = module.kernels.as_slice() else {
        panic!("expected one row-affine kernel")
    };
    assert_eq!(kernel.id.as_str(), "row_affine_sum_u32_v1");
    assert_eq!(
        module.function(&kernel.entry).unwrap().signature.parameters,
        vec![
            Type::slice(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadOnly
            ),
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::U64),
            Type::Scalar(ScalarType::U32),
            Type::Scalar(ScalarType::U32),
            Type::slice(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadWrite
            ),
        ],
    );
    let map =
        fe2o3_kernel_ir::DebugSourceMapDocumentV2::from_canonical_json_bytes(bundle.debug_map())
            .unwrap();
    assert_eq!(
        map.binding().bundle_subject_identity(),
        *bundle.subject_identity()
    );
    assert!(
        map.files()
            .iter()
            .any(|file| file.display_path().ends_with(source_file))
    );
    assert!(!map.sites().is_empty());
}

fn row_affine_source_matches_oracle_and_replay() {
    row_affine_variant_matches_oracle_and_replay(
        "row-affine-u32-kernel",
        "kernel_row_affine_u32.rs",
    );
}

fn row_affine_variant_matches_oracle_and_replay(feature: &str, source_file: &str) {
    use fe2o3_kir_sim::{PersistedSimulationScheduleDocumentV1, SimulationScheduleRequestV1};
    use fe2o3_kir_sim_cli::{load_debug_simulation_bundle_v5, load_debug_simulation_schedule_v1};

    let source = workspace()
        .join("examples/workgroup_sync_v1/src")
        .join(source_file);
    let source_before = std::fs::read(&source).unwrap();
    let cases = row_affine_oracle::row_affine_vectors_v1();
    assert_eq!(cases.len(), 86);
    for (cpu, target) in [("gfx942", "gfx942:xnack-"), ("gfx950", "gfx950:xnack-")] {
        let scratch = ScratchDirectory::new(&format!("row-affine-{cpu}"));
        let bundle = export_workgroup_bundle(&scratch, feature, cpu);
        assert_row_affine_bundle(&bundle, target, source_file);
        for (index, case) in cases.iter().enumerate() {
            let request = row_affine_request(case);
            let path =
                write_row_affine_request(&scratch, &format!("case-{index}"), &request.document);
            let admitted = load_debug_simulation_bundle_v5(&bundle, &path).unwrap();
            let input = admitted.input();
            for repetition in 0..2_u64 {
                let schedule = if repetition == 0 {
                    SimulationScheduleRequestV1::RecordCanonical {
                        max_decisions: 100_000,
                    }
                } else {
                    SimulationScheduleRequestV1::RecordSeeded {
                        seed: 0x275_AFF1 + index as u64,
                        max_decisions: 100_000,
                    }
                };
                let execution = input
                    .module
                    .simulate_scheduled(
                        &input.request,
                        input.simulation_target(),
                        input.simulation_limits,
                        schedule,
                    )
                    .unwrap_or_else(|error| {
                        panic!("{target} case{index} repetition{repetition}: {error:?}")
                    });
                assert_row_affine_execution(&execution, &request, case.config.workgroups);
                let record = PersistedSimulationScheduleDocumentV1::encode_record(
                    input.persisted_schedule_binding().unwrap(),
                    execution.schedule_record().unwrap(),
                )
                .unwrap();
                let schedule_path = scratch.0.join("recorded-schedule.json");
                std::fs::write(&schedule_path, &record).unwrap();
                let persisted = load_debug_simulation_schedule_v1(&schedule_path, input).unwrap();
                assert_eq!(persisted.to_canonical_bytes().unwrap(), record);
                let replay = input
                    .module
                    .simulate_scheduled(
                        &input.request,
                        input.simulation_target(),
                        input.simulation_limits,
                        SimulationScheduleRequestV1::Replay(persisted.record()),
                    )
                    .unwrap();
                assert_row_affine_execution(&replay, &request, case.config.workgroups);
                assert_eq!(
                    replay.schedule_transcript_identity(),
                    execution.schedule_transcript_identity()
                );
                assert_eq!(replay.schedule_coverage(), execution.schedule_coverage());
                if index == 0 && repetition == 0 {
                    let mut substituted = request.document.clone();
                    substituted["arguments"][6]["bits"] =
                        json!(format!("0x{:08x}", case.config.bias ^ 1));
                    let substituted =
                        write_row_affine_request(&scratch, "substituted-bias", &substituted);
                    let substituted =
                        load_debug_simulation_bundle_v5(&bundle, &substituted).unwrap();
                    let error =
                        load_debug_simulation_schedule_v1(&schedule_path, substituted.input())
                            .unwrap_err();
                    assert_eq!(error.code, "schedule_binding_mismatch");
                }
            }
        }
        assert_row_affine_refusals(&scratch, &bundle, &cases);
        assert_eq!(std::fs::read(&source).unwrap(), source_before);
    }
}

fn assert_row_affine_refusals(
    scratch: &ScratchDirectory,
    bundle: &Path,
    cases: &[row_affine_oracle::RowAffineVectorV1],
) {
    use fe2o3_kir_sim::{
        SimulationErrorV1, SimulationExecutionErrorKindV1, SimulationPreflightErrorV1,
    };
    let case = cases
        .iter()
        .find(|case| case.config.rows == 1 && case.config.columns == 65 && case.config.offset == 0)
        .unwrap();
    let valid = row_affine_request(case).document;
    for (label, argument, replacement) in [
        ("wide", 3, 129_u64),
        ("stride", 4, 0),
        ("offset-overflow", 1, u64::MAX),
        ("rows-outside-launch", 2, u64::MAX),
    ] {
        let mut document = valid.clone();
        document["arguments"][argument]["bits"] = json!(format!("0x{replacement:016x}"));
        assert_row_affine_trap(scratch, bundle, label, &document);
    }
    for (label, argument) in [("short-input", 0), ("short-output", 7)] {
        let mut document = valid.clone();
        document["arguments"][argument]["elements"] = json!(0);
        assert_row_affine_trap(scratch, bundle, label, &document);
    }
    let three_rows = cases
        .iter()
        .find(|case| case.config.rows == 3 && case.config.columns == 65)
        .unwrap();
    let mut undersized = row_affine_request(three_rows).document;
    undersized["grid"] = json!([64, 1, 1]);
    assert_row_affine_trap(scratch, bundle, "undersized-grid", &undersized);

    let mut extent_overflow = row_affine_request(three_rows).document;
    extent_overflow["arguments"][4]["bits"] = json!("0xffffffffffffffff");
    assert_row_affine_trap(scratch, bundle, "stride-extent-overflow", &extent_overflow);

    let mut wrong_group = valid.clone();
    wrong_group["workgroup"] = json!([32, 1, 1]);
    let path = write_row_affine_request(scratch, "wrong-workgroup", &wrong_group);
    let admitted = fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(bundle, &path).unwrap();
    let input = admitted.input();
    assert!(matches!(
        input.module.simulate(
            &input.request,
            input.simulation_target(),
            input.simulation_limits,
        ),
        Err(SimulationErrorV1::Preflight(
            SimulationPreflightErrorV1::WorkgroupMismatch { .. }
        ))
    ));

    // Physical workgroup-count checks cannot establish logical full-group coverage.
    let mut partial = valid.clone();
    partial["grid"] = json!([65, 1, 1]);
    let path = write_row_affine_request(scratch, "partial-workgroup", &partial);
    let admitted = fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(bundle, &path).unwrap();
    let input = admitted.input();
    let error = input
        .module
        .simulate(
            &input.request,
            input.simulation_target(),
            input.simulation_limits,
        )
        .expect_err("partial workgroups must not fabricate a complete reduction");
    assert!(matches!(error, SimulationErrorV1::Execution(ref error)
        if matches!(error.kind, SimulationExecutionErrorKindV1::UninitializedRead { .. })));
}

fn assert_row_affine_trap(
    scratch: &ScratchDirectory,
    bundle: &Path,
    label: &str,
    document: &Value,
) {
    let path = write_row_affine_request(scratch, label, document);
    let admitted = fe2o3_kir_sim_cli::load_debug_simulation_bundle_v5(bundle, &path).unwrap();
    let input = admitted.input();
    let error = input
        .module
        .simulate(
            &input.request,
            input.simulation_target(),
            input.simulation_limits,
        )
        .unwrap_err();
    assert!(
        matches!(error, fe2o3_kir_sim::SimulationErrorV1::Execution(ref error)
        if error.kind == fe2o3_kir_sim::SimulationExecutionErrorKindV1::ReachedUnreachable),
        "{label}: {error:?}"
    );
}
