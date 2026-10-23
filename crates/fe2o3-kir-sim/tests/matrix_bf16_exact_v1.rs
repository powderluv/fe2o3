//! Canonical inert graphs: no Rust source or GPU qualification.
use fe2o3_kernel_ir::*;
use fe2o3_kir_sim::*;
#[path = "matrix_bf16_exact/admission.rs"]
mod admission;
#[path = "matrix_bf16_exact/fixture.rs"]
mod fixture;
use fixture::*;

fn limits() -> SimulationLimitsV1 {
    SimulationLimitsV1 {
        max_canonical_bytes: 1024 * 1024,
        max_reachable_functions: 4,
        max_reachable_operations: 4096,
        max_invocations: 128,
        max_workgroups: 2,
        max_scheduled_slots: 128,
        max_steps: 2_000_000,
        max_call_depth: 4,
        max_ssa_values: 256,
        max_allocations: 8,
        max_allocation_bytes: 4096,
        max_total_bytes: 16_384,
        max_resident_bytes: 16 * 1024 * 1024,
        max_events: 100_000,
        max_memory_access_records: 8192,
    }
}
fn kind(error: SimulationErrorV1) -> SimulationExecutionErrorKindV1 {
    let SimulationErrorV1::Execution(error) = error else {
        panic!("expected execution refusal")
    };
    error.kind
}
fn dense() -> ([i64; 256], [i64; 256], [i64; 256]) {
    (
        std::array::from_fn(|n| (n % 29) as i64 - 14),
        std::array::from_fn(|n| ((n * 7 + 3) % 31) as i64 - 15),
        std::array::from_fn(|n| (n as i64 - 128) * 17),
    )
}

#[test]
fn dense_signed_domain_executes_actual_load_mfma_store_graph_and_canaries() {
    let (a, b, c) = dense();
    let request = request(&a, &b, &c, 1, 64);
    let original = request.arguments.clone();
    let run = admit(module(64))
        .simulate(&request, TARGET, limits())
        .unwrap();
    check_output(&run, &oracle(&a, &b, &c), 1);
    assert_eq!(request.arguments, original);
    assert_eq!(
        run.buffer(0).unwrap().bytes(),
        match &request.arguments[0] {
            SimulationArgumentV1::Buffer(buffer) => buffer.bytes(),
            _ => unreachable!(),
        }
    );
}

#[test]
fn every_dense_input_coordinate_has_an_independent_impulse_oracle() {
    let sim = admit(module(64));
    let identity: [i64; 256] = std::array::from_fn(|n| i64::from(n / 16 == n % 16));
    for coordinate in 0..256 {
        let mut impulse = [0; 256];
        impulse[coordinate] = if coordinate % 2 == 0 { 16 } else { -16 };
        for (a, b) in [(&impulse, &identity), (&identity, &impulse)] {
            let request = request(a, b, &[0; 256], 1, 64);
            let run = sim.simulate(&request, TARGET, limits()).unwrap();
            check_output(&run, &oracle(a, b, &[0; 256]), 1);
        }
    }
}

#[test]
fn accepted_extremes_zero_and_cancellation_are_exact() {
    let sim = admit(module(64));
    for (a, b, c) in [
        ([16; 256], [16; 256], [1 << 20; 256]),
        ([-16; 256], [16; 256], [-(1 << 20); 256]),
        ([1; 256], [1; 256], [-16; 256]),
        ([0; 256], [-16; 256], [0; 256]),
    ] {
        let run = sim
            .simulate(&request(&a, &b, &c, 1, 64), TARGET, limits())
            .unwrap();
        check_output(&run, &oracle(&a, &b, &c), 1);
    }
}

#[test]
fn two_waves_and_seeded_replay_preserve_all_components() {
    let (a, b, c) = dense();
    for workgroup in [64, 128] {
        let sim = admit(module(workgroup));
        let mut request = request(&a, &b, &c, 2, workgroup);
        // A different second wave detects accidental cross-wave operand reuse.
        let SimulationArgumentV1::Buffer(old) = &request.arguments[0] else {
            unreachable!()
        };
        let mut bytes = old.bytes().to_vec();
        for chunk in bytes[512..].chunks_exact_mut(2) {
            let original = u16::from_le_bytes([chunk[0], chunk[1]]);
            let negated = if original == 0 { 0 } else { original ^ 0x8000 };
            chunk.copy_from_slice(&negated.to_le_bytes());
        }
        request.arguments[0] = SimulationArgumentV1::Buffer(
            BufferArgumentV1::new(
                old.element(),
                old.access(),
                old.alignment(),
                bytes,
                old.initialized().to_vec(),
                TARGET,
            )
            .unwrap(),
        );
        let expected = [oracle(&a, &b, &c), oracle(&a.map(|n| -n), &b, &c)];
        let canonical = sim.simulate(&request, TARGET, limits()).unwrap();
        let seeded = sim
            .simulate_scheduled(
                &request,
                TARGET,
                limits(),
                SimulationScheduleRequestV1::RecordSeeded {
                    seed: 0xf001_1234,
                    max_decisions: 1024,
                },
            )
            .unwrap();
        let replay = sim
            .simulate_scheduled(
                &request,
                TARGET,
                limits(),
                SimulationScheduleRequestV1::Replay(seeded.schedule_record().unwrap()),
            )
            .unwrap();
        for run in [&canonical, &seeded, &replay] {
            for (wave, expected) in expected.iter().enumerate() {
                check_wave_output(run, expected, wave, 2);
            }
        }
        assert_eq!(canonical.arguments(), seeded.arguments());
        assert_eq!(seeded.arguments(), replay.arguments());
    }
}

#[derive(Default)]
struct MatrixEvents {
    operation: u32,
    arrived: usize,
    completed: usize,
    writes: usize,
}
impl SimulationEventSinkV1 for MatrixEvents {
    fn record(&mut self, event: &SimulationEventV1) -> Result<(), SimulationEventSinkErrorV1> {
        if event.site.operation == Some(self.operation)
            && matches!(event.kind, SimulationEventKindV1::OperationBegin)
        {
            self.arrived += 1;
        }
        if event.site.operation == Some(self.operation)
            && matches!(
                event.kind,
                SimulationEventKindV1::OperationEnd {
                    outcome: SimulationExecutionOutcomeV1::Completed
                }
            )
        {
            self.completed += 1;
        }
        if matches!(event.kind, SimulationEventKindV1::MemoryWrite { .. }) {
            self.writes += 1;
        }
        Ok(())
    }
}
fn matrix_position(module: &Module) -> usize {
    module.functions[0].body.as_ref().unwrap().blocks[0]
        .operations
        .iter()
        .position(|operation| matches!(operation.kind, OperationKind::Matrix(_)))
        .unwrap()
}

#[test]
fn refused_last_lane_input_never_commits_matrix_or_downstream_stores() {
    let graph = module(64);
    let position = matrix_position(&graph) as u32;
    let sim = admit(graph);
    for (argument, role, patterns) in [
        (
            0,
            MatrixInputRoleV1::A,
            vec![0x8000_u32, 1, 0x3f00, 0x4181, 0x7f80, 0x7fc0, 0x7f81],
        ),
        (
            1,
            MatrixInputRoleV1::B,
            vec![0x8000, 1, 0x3f00, 0xc181, 0xff80, 0x7fc0, 0x7f81],
        ),
        (
            2,
            MatrixInputRoleV1::Accumulator,
            vec![
                0x8000_0000,
                1,
                0x3f00_0000,
                0x4980_0001,
                0x7f80_0000,
                0x7fc0_0000,
                0x7f80_0001,
            ],
        ),
    ] {
        for bits in patterns {
            let mut request = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
            let width = if argument == 2 { 4 } else { 2 };
            alter(
                &mut request,
                argument,
                255 * width,
                &bits.to_le_bytes()[..width],
                true,
            );
            let mut events = MatrixEvents {
                operation: position,
                ..Default::default()
            };
            let refusal = sim
                .simulate_observed_with_sink(&request, TARGET, limits(), &mut events)
                .unwrap_err();
            assert_eq!(
                kind(refusal),
                SimulationExecutionErrorKindV1::UnsupportedMatrixInputDomain {
                    role,
                    lane: 63,
                    component: 3,
                }
            );
            assert_eq!(events.completed, 0);
            assert_eq!(events.writes, 0);
        }
    }
}

#[test]
fn ordinary_memory_initialization_and_bounds_checks_still_run() {
    let sim = admit(module(64));
    let mut uninitialized = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    alter(&mut uninitialized, 0, 0, &0x3f80_u16.to_le_bytes(), false);
    assert!(matches!(
        kind(sim.simulate(&uninitialized, TARGET, limits()).unwrap_err()),
        SimulationExecutionErrorKindV1::UninitializedRead { .. }
    ));
    let mut short = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    let SimulationArgumentV1::Buffer(old) = &short.arguments[1] else {
        unreachable!()
    };
    short.arguments[1] = SimulationArgumentV1::Buffer(
        BufferArgumentV1::new(
            old.element(),
            old.access(),
            old.alignment(),
            old.bytes()[..510].to_vec(),
            old.initialized()[..510].to_vec(),
            TARGET,
        )
        .unwrap(),
    );
    assert!(matches!(
        kind(sim.simulate(&short, TARGET, limits()).unwrap_err()),
        SimulationExecutionErrorKindV1::OutOfBounds { .. }
    ));
}

#[test]
fn partial_waves_are_not_zero_padded() {
    let sim = admit(module(64));
    for grid in [63, 65] {
        let mut request = request(&[1; 256], &[1; 256], &[0; 256], 2, 64);
        request.grid = GridShapeV1([grid, 1, 1]);
        assert!(matches!(
            kind(sim.simulate(&request, TARGET, limits()).unwrap_err()),
            SimulationExecutionErrorKindV1::IncompleteWave(_)
        ));
    }
}

#[test]
fn numerical_work_is_prepaid_before_any_result_completion() {
    let graph = module(64);
    let position = matrix_position(&graph) as u32;
    let sim = admit(graph);
    let request = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    // Straight-line fixture: one step per pre-MFMA op, one Matrix entry,
    // 24 operand units per lane, then the exact whole-wave resolver prepay.
    let before = 64 * (u64::from(position) + 1 + 24);
    let denied = SimulationLimitsV1 {
        max_steps: before + 66 * 64 + 21_824 - 1,
        ..limits()
    };
    let mut events = MatrixEvents {
        operation: position,
        ..Default::default()
    };
    assert!(matches!(
        kind(
            sim.simulate_observed_with_sink(&request, TARGET, denied, &mut events)
                .unwrap_err()
        ),
        SimulationExecutionErrorKindV1::StepLimit { .. }
    ));
    assert_eq!(events.arrived, 64);
    assert_eq!(events.completed, 0);
    assert_eq!(events.writes, 0);
}

fn finite_options() -> ObservationExecutionOptionsV1 {
    ObservationExecutionOptionsV1::new(SimulationDebugCaptureLimitsV1::disabled())
        .with_matrix_bf16_numerics(MatrixBf16NumericsV1::FiniteSequentialFmaV1)
}

fn finite_simulate(
    sim: &AdmittedSimulationModuleV1,
    request: &SimulationRequestV1,
    limits: SimulationLimitsV1,
    events: &mut impl SimulationEventSinkV1,
) -> Result<SimulationExecutionV1, SimulationErrorV1> {
    let mut observed = request.clone();
    observed.events = EventPolicyV1::Enabled;
    sim.simulate_debugged_with_observation_options(
        &observed,
        TARGET,
        limits,
        finite_options(),
        events,
        &mut NoopSimulationDebugSinkV1,
    )
}

#[test]
fn finite_mode_preserves_fractional_product_gradual_underflow_and_canaries() {
    let sim = admit(module(64));
    for (a, b, c, result) in [
        (0x3f81_u16, 0x3f7f_u16, 0, 0x4180_7f00), // 16 exact BF16 products.
        (1, 0x3f80, 0, 0x0010_0000),              // 16 * least BF16 subnormal, no FTZ.
        (0x8000, 0, 1, 1),                        // F32 subnormal accumulator survives every FMA.
        // Each +2^-24 is an exact halfway increment at 1.0; ties go to even.
        (0x3f80, 0x3380, 0x3f80_0000, 0x3f80_0000),
        (0x3f80, 0x3380, 0x3f80_0001, 0x3f80_0002),
        // A negative-zero product plus a negative-zero accumulator stays -0.
        (0x8000, 0x3f80, 0x8000_0000, 0x8000_0000),
    ] {
        let mut request = request(&[0; 256], &[0; 256], &[0; 256], 1, 64);
        for i in 0..256 {
            alter(&mut request, 0, i * 2, &a.to_le_bytes(), true);
            alter(&mut request, 1, i * 2, &b.to_le_bytes(), true);
            alter(&mut request, 2, i * 4, &u32::to_le_bytes(c), true);
        }
        let original = request.arguments.clone();
        let run =
            finite_simulate(&sim, &request, limits(), &mut NoopSimulationEventSinkV1).unwrap();
        assert_eq!(
            run.matrix_bf16_numerics(),
            MatrixBf16NumericsV1::FiniteSequentialFmaV1
        );
        check_output(&run, &[result; 256], 1);
        assert_eq!(request.arguments, original);
    }
}

#[test]
fn finite_mode_rejects_last_input_and_late_overflow_before_completions_or_stores() {
    let graph = module(64);
    let position = matrix_position(&graph) as u32;
    let sim = admit(graph);
    for argument in 0..3 {
        for bits in [0x7f80_0000_u32, 0xff80_0000, 0x7fc0_0000, 0x7f81_0000] {
            let mut req = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
            if argument < 2 {
                alter(
                    &mut req,
                    argument,
                    510,
                    &((bits >> 16) as u16).to_le_bytes(),
                    true,
                );
            } else {
                alter(&mut req, argument, 1020, &bits.to_le_bytes(), true);
            }
            let original = req.arguments.clone();
            let mut events = MatrixEvents {
                operation: position,
                ..Default::default()
            };
            assert!(matches!(
                kind(finite_simulate(&sim, &req, limits(), &mut events).unwrap_err()),
                SimulationExecutionErrorKindV1::UnsupportedMatrixInputDomain { .. }
            ));
            assert_eq!(events.arrived, 64);
            assert_eq!((events.completed, events.writes), (0, 0));
            assert_eq!(req.arguments, original);
        }
    }
    // Only the last output component overflows, after 255 scratch results exist.
    let mut req = request(&[0; 256], &[0; 256], &[0; 256], 1, 64);
    for depth in 0..16 {
        let a_slot = (15 + (depth / 4) * 16) * 4 + depth % 4;
        let b_slot = (15 + (depth / 4) * 16) * 4 + depth % 4;
        alter(&mut req, 0, a_slot * 2, &0x7f7f_u16.to_le_bytes(), true);
        alter(&mut req, 1, b_slot * 2, &0x4000_u16.to_le_bytes(), true);
    }
    let mut events = MatrixEvents {
        operation: position,
        ..Default::default()
    };
    let original = req.arguments.clone();
    assert!(matches!(
        kind(finite_simulate(&sim, &req, limits(), &mut events).unwrap_err()),
        SimulationExecutionErrorKindV1::NonFiniteMatrixResult {
            lane: 63,
            component: 3,
            ..
        }
    ));
    assert_eq!(events.arrived, 64);
    assert_eq!((events.completed, events.writes), (0, 0));
    assert_eq!(req.arguments, original);
}

#[test]
fn finite_precharge_exact_one_short_and_persisted_schedule_refusal() {
    let graph = module(64);
    let position = matrix_position(&graph) as u32;
    let sim = admit(graph);
    let req = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    let prepaid = 64 * (u64::from(position) + 1 + 24) + 66 * 64 + 21_824 + 4865;
    let mut events = MatrixEvents {
        operation: position,
        ..Default::default()
    };
    let denied = SimulationLimitsV1 {
        max_steps: prepaid - 1,
        ..limits()
    };
    assert!(matches!(
        kind(finite_simulate(&sim, &req, denied, &mut events).unwrap_err()),
        SimulationExecutionErrorKindV1::StepLimit { .. }
    ));
    assert_eq!(events.arrived, 64);
    assert_eq!((events.completed, events.writes), (0, 0));
    let exact = SimulationLimitsV1 {
        max_steps: prepaid,
        ..limits()
    };
    let mut events = MatrixEvents {
        operation: position,
        ..Default::default()
    };
    // The collective completes; subsequent source stores then exhaust this exact prepay.
    assert!(matches!(
        kind(finite_simulate(&sim, &req, exact, &mut events).unwrap_err()),
        SimulationExecutionErrorKindV1::StepLimit { .. }
    ));
    assert_eq!(events.arrived, 64);
    assert_eq!(events.completed, 64);
    assert!(matches!(
        sim.simulate_debugged_scheduled_with_observation_options(
            &req,
            TARGET,
            limits(),
            SimulationScheduleRequestV1::RecordCanonical {
                max_decisions: 1024
            },
            finite_options(),
            &mut NoopSimulationEventSinkV1,
            &mut NoopSimulationDebugSinkV1,
        ),
        Err(SimulationErrorV1::Preflight(
            SimulationPreflightErrorV1::InvalidLaunch(_)
        ))
    ));
}

fn branched(mismatched: bool) -> Module {
    let mut graph = module(64);
    let position = matrix_position(&graph);
    let body = graph.functions[0].body.as_mut().unwrap();
    let tail = body.blocks[0].operations.split_off(position);
    body.blocks[0].operations.extend([
        Operation::effect_free(
            ValueDef::new(ValueId(2000), Type::INDEX),
            OperationKind::Constant(Constant::Index(32)),
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(2001), Type::BOOL),
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(4),
                rhs: ValueId(2000),
            },
        ),
    ]);
    body.blocks[0].terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2001),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    let mut else_block = BasicBlock::new(BlockId(2));
    if mismatched {
        let mut other = tail[0].clone();
        for result in &mut other.results {
            result.id = ValueId(result.id.0 + 200);
        }
        else_block.operations.push(other);
    }
    then_block.operations = tail;
    then_block.terminator = Some(Terminator::Return { values: vec![] });
    else_block.terminator = Some(Terminator::Return { values: vec![] });
    body.blocks.extend([then_block, else_block]);
    let caps = graph.functions[0].derived_capabilities();
    graph.functions[0].required_capabilities = caps.clone();
    graph.kernels[0].required_capabilities = caps.clone();
    graph.required_capabilities = caps;
    graph
}
#[test]
fn divergent_or_different_actual_matrix_site_cannot_complete() {
    let request = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    for mismatched in [false, true] {
        let refusal = kind(
            admit(branched(mismatched))
                .simulate(&request, TARGET, limits())
                .unwrap_err(),
        );
        if mismatched {
            assert!(matches!(
                refusal,
                SimulationExecutionErrorKindV1::MismatchedWave(_)
            ));
        } else {
            assert!(matches!(
                refusal,
                SimulationExecutionErrorKindV1::DivergentWave(_)
            ));
        }
    }
}

#[test]
fn exact_steps_and_resident_floor_are_enforced_without_cap_increases() {
    let sim = admit(module(64));
    let request = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    let baseline = sim.simulate(&request, TARGET, limits()).unwrap();
    let peak = sim
        .preflight(&request, TARGET, limits())
        .unwrap()
        .resident_bytes();
    let exact = SimulationLimitsV1 {
        max_steps: baseline.steps_executed(),
        max_resident_bytes: peak,
        ..limits()
    };
    assert!(sim.simulate(&request, TARGET, exact).is_ok());
    let short_work = SimulationLimitsV1 {
        max_steps: exact.max_steps - 1,
        ..exact
    };
    assert!(matches!(
        kind(sim.simulate(&request, TARGET, short_work).unwrap_err()),
        SimulationExecutionErrorKindV1::StepLimit { .. }
    ));
    let short_storage = SimulationLimitsV1 {
        max_resident_bytes: peak - 1,
        ..exact
    };
    assert!(matches!(
        sim.preflight(&request, TARGET, short_storage),
        Err(SimulationPreflightErrorV1::ResourceLimit { .. })
    ));
}

#[test]
fn layoutless_matrix_is_refused_before_simulator_admission() {
    let mut graph = module(64);
    let position = matrix_position(&graph);
    let OperationKind::Matrix(matrix) =
        &mut graph.functions[0].body.as_mut().unwrap().blocks[0].operations[position].kind
    else {
        unreachable!()
    };
    matrix.tensor_layout = None;
    // Existing semantic verification rejects this before simulator admission.
    // Do not bypass that stronger invariant just to reach numerical preflight.
    let error = VerifiedCanonicalKernelIrV12::from_module(graph).unwrap_err();
    let VerifiedCanonicalKernelIrErrorV12::Verification(errors) = error else {
        panic!("expected missing-layout semantic verification refusal");
    };
    assert_eq!(errors.diagnostics().len(), 1);
    let diagnostic = &errors.diagnostics()[0];
    assert_eq!(diagnostic.code, DiagnosticCode::InvalidSemanticOperation);
    assert_eq!(diagnostic.location.operation, Some(position));
    assert_eq!(
        diagnostic.message,
        "matrix multiply requires an explicit tensor layout contract"
    );
}

#[test]
fn exact_profile_does_not_admit_a_32_bit_target_layout() {
    let sim = admit(module(64));
    let request = request(&[1; 256], &[1; 256], &[0; 256], 1, 64);
    assert!(sim.preflight(&request, TARGET, limits()).is_ok());
    assert!(
        sim.preflight(
            &request,
            SimulationTargetV1::little_endian(IndexWidthV1::Bits32),
            limits()
        )
        .is_err()
    );
}
