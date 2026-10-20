use fe2o3_kernel_ir::{
    AccessMode, AddressSpace, BasicBlock, BlockId, Constant, Function, IntrinsicOperation, Kernel,
    LaunchDomain, LaunchExtent, MemoryAccess, Module, Operation, OperationKind, ScalarType,
    Signature, TargetCapability, Terminator, Type, ValueDef, ValueId, VerifiedCanonicalKernelIrV7,
    WorkgroupMemory, WorkgroupMemoryExtent, WorkgroupSize, gfx942_xnack_minus_target_capability,
    gfx950_xnack_minus_target_capability,
};
use fe2o3_kir_sim::{
    AdmittedSimulationModuleV1, DynamicWorkgroupMemoryRequestV1, ScalarBitsV1, SimulationErrorV1,
    SimulationExecutionErrorKindV1, SimulationLimitsV1,
};
use fe2o3_runtime_model::{IdentityDigestV1, TransitionErrorV1};
use fe2o3_virtual_runtime::{
    VirtualArgumentV1, VirtualBufferAccessV1, VirtualCompletionAmbiguityV1,
    VirtualCompletionStateV1, VirtualDispatchInputBindingV1,
    VirtualDispatchInputUnavailableReasonV1, VirtualDispatchRequestV1,
    VirtualHostLifetimeCompletenessV1, VirtualHostLifetimeEvidenceLimitsV1,
    VirtualHostLifetimeEvidenceV1, VirtualHostLifetimeFindingV1, VirtualHostLifetimeOperationV1,
    VirtualRunProgressV1, VirtualRuntimeConfigV1, VirtualRuntimeErrorV1, VirtualRuntimeLimitsV1,
    VirtualRuntimeV1, VirtualTargetProfileV1,
};

fn operation(result: u32, ty: Type, kind: OperationKind) -> Operation {
    Operation::effect_free(ValueDef::new(ValueId(result), ty), kind)
}

fn admitted_fill() -> AdmittedSimulationModuleV1 {
    admitted_fill_with_capabilities([])
}

fn admitted_dynamic_fill() -> AdmittedSimulationModuleV1 {
    let mut module = admitted_fill().module().clone();
    let scalar = Type::Scalar(ScalarType::U32);
    module.functions[0].body.as_mut().unwrap().blocks[0]
        .operations
        .insert(
            0,
            operation(
                10,
                Type::pointer(
                    scalar.clone(),
                    AddressSpace::Workgroup,
                    AccessMode::ReadWrite,
                ),
                OperationKind::WorkgroupMemory(WorkgroupMemory {
                    element: scalar,
                    extent: WorkgroupMemoryExtent::Dynamic,
                    alignment: 4,
                }),
            ),
        );
    for capabilities in [
        &mut module.required_capabilities,
        &mut module.functions[0].required_capabilities,
        &mut module.kernels[0].required_capabilities,
    ] {
        capabilities.extend([
            TargetCapability::WorkgroupMemory,
            TargetCapability::DynamicWorkgroupMemory,
        ]);
    }
    let canonical = VerifiedCanonicalKernelIrV7::from_module(module).unwrap();
    AdmittedSimulationModuleV1::admit(canonical, SimulationLimitsV1::default()).unwrap()
}

fn admitted_fill_with_capabilities(
    capabilities: impl IntoIterator<Item = TargetCapability>,
) -> AdmittedSimulationModuleV1 {
    let element = Type::Scalar(ScalarType::U32);
    let pointer = Type::pointer(element.clone(), AddressSpace::Global, AccessMode::ReadWrite);
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = vec![
        operation(
            1,
            Type::INDEX,
            OperationKind::Intrinsic(IntrinsicOperation::global_id_1d()),
        ),
        operation(
            2,
            pointer.clone(),
            OperationKind::GetElementPointer {
                base: ValueId(0),
                offset: ValueId(1),
            },
        ),
        operation(3, element, OperationKind::Constant(Constant::U32(17))),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(2),
                value: ValueId(3),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    block.terminator = Some(Terminator::Return { values: vec![] });
    let entry = Function::kernel_entry(
        "fill_impl",
        Signature::new(vec![pointer], vec![]),
        vec![ValueId(0)],
        vec![block],
    );
    let mut kernel = Kernel::new(
        "fill",
        "fill_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("virtual-runtime-tests::fill");
    module.required_capabilities.extend(capabilities);
    module.functions.push(entry);
    module.kernels.push(kernel);
    let canonical = VerifiedCanonicalKernelIrV7::from_module(module).unwrap();
    AdmittedSimulationModuleV1::admit(canonical, SimulationLimitsV1::default()).unwrap()
}

fn admitted_alias_writes() -> AdmittedSimulationModuleV1 {
    let element = Type::Scalar(ScalarType::U32);
    let pointer = Type::pointer(element.clone(), AddressSpace::Global, AccessMode::ReadWrite);
    let mut block = BasicBlock::new(BlockId(0));
    block.operations = vec![
        operation(
            2,
            element.clone(),
            OperationKind::Constant(Constant::U32(17)),
        ),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(0),
                value: ValueId(2),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
        operation(3, element, OperationKind::Constant(Constant::U32(23))),
        Operation::new(
            vec![],
            OperationKind::Store {
                pointer: ValueId(1),
                value: ValueId(3),
                access: MemoryAccess::new(AddressSpace::Global, 4),
            },
        ),
    ];
    block.terminator = Some(Terminator::Return { values: vec![] });
    let entry = Function::kernel_entry(
        "alias_impl",
        Signature::new(vec![pointer.clone(), pointer], vec![]),
        vec![ValueId(0), ValueId(1)],
        vec![block],
    );
    let mut kernel = Kernel::new(
        "alias",
        "alias_impl",
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("virtual-runtime-tests::alias-writes");
    module.functions.push(entry);
    module.kernels.push(kernel);
    let canonical = VerifiedCanonicalKernelIrV7::from_module(module).unwrap();
    AdmittedSimulationModuleV1::admit(canonical, SimulationLimitsV1::default()).unwrap()
}

fn runtime(seed: u8) -> VirtualRuntimeV1 {
    VirtualRuntimeV1::new(VirtualRuntimeConfigV1 {
        runtime_identity: IdentityDigestV1::from_untrusted_bytes([seed; 32]),
        target: VirtualTargetProfileV1::Amdgpu64TargetNeutral,
        runtime_limits: VirtualRuntimeLimitsV1::default(),
        simulation_limits: SimulationLimitsV1::default(),
    })
    .unwrap()
}

fn fill_request(
    buffer: fe2o3_virtual_runtime::VirtualBufferHandleV1,
    elements: usize,
    dependencies: Vec<fe2o3_virtual_runtime::VirtualCompletionHandleV1>,
) -> VirtualDispatchRequestV1 {
    VirtualDispatchRequestV1 {
        kernel: "fill".into(),
        grid: [elements as u64, 1, 1],
        workgroup: [64, 1, 1],
        arguments: vec![VirtualArgumentV1::Buffer {
            buffer,
            element: ScalarType::U32,
            access: AccessMode::ReadWrite,
            alignment: 4,
            byte_offset: 0,
            elements,
        }],
        dependencies,
    }
}

fn alias_request(
    buffer: fe2o3_virtual_runtime::VirtualBufferHandleV1,
    first: (usize, usize),
    second: (usize, usize),
) -> VirtualDispatchRequestV1 {
    VirtualDispatchRequestV1 {
        kernel: "alias".into(),
        grid: [1, 1, 1],
        workgroup: [64, 1, 1],
        arguments: [first, second]
            .into_iter()
            .map(|(byte_offset, elements)| VirtualArgumentV1::Buffer {
                buffer,
                element: ScalarType::U32,
                access: AccessMode::ReadWrite,
                alignment: 4,
                byte_offset,
                elements,
            })
            .collect(),
        dependencies: vec![],
    }
}

#[test]
fn serial_dependencies_execute_and_copy_back_typed_bytes() {
    let mut runtime = runtime(1);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(16, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 16]).unwrap();
    let first = runtime
        .submit(queue, module, fill_request(buffer, 4, vec![]))
        .unwrap();
    let second = runtime
        .submit(queue, module, fill_request(buffer, 4, vec![first]))
        .unwrap();

    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::Completed { completion, .. } if completion == first
    ));
    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::Completed { completion, .. } if completion == second
    ));
    assert_eq!(
        runtime.completion_state(second).unwrap(),
        VirtualCompletionStateV1::Completed
    );
    let mut output = [0; 16];
    runtime.copy_to_host(buffer, 0, &mut output).unwrap();
    assert_eq!(output, [17, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0]);
    let summary = runtime.completion_summary(second).unwrap().unwrap();
    assert_eq!(summary.invocations_executed, 4);
    assert!(!runtime.grants_hardware_authority());
    assert!(!runtime.predicts_performance());
}

#[test]
fn schedule_recording_respects_the_configured_resident_budget() {
    // The preflight plan plus the full 1 << 19 decision reservation exceeds 256 MiB.
    // Keep that refusal and use 1 << 18 decisions for success at the same cap.
    for (max_schedule_decisions, max_resident_bytes, succeeds) in [
        (1 << 18, 256 << 20, true),
        (1 << 19, 256 << 20, false),
        (1 << 20, 256 << 20, false),
        (1 << 20, 512 << 20, true),
    ] {
        let mut runtime = VirtualRuntimeV1::new(VirtualRuntimeConfigV1 {
            runtime_identity: IdentityDigestV1::from_untrusted_bytes([33; 32]),
            target: VirtualTargetProfileV1::Amdgpu64TargetNeutral,
            runtime_limits: VirtualRuntimeLimitsV1 {
                max_schedule_decisions,
                ..VirtualRuntimeLimitsV1::default()
            },
            simulation_limits: SimulationLimitsV1 {
                max_allocation_bytes: 16 << 20,
                max_total_bytes: 64 << 20,
                max_resident_bytes,
                ..SimulationLimitsV1::default()
            },
        })
        .unwrap();
        let module = runtime.register_module(admitted_fill()).unwrap();
        let queue = runtime.create_queue(1).unwrap();
        let buffer = runtime
            .allocate_buffer(16, VirtualBufferAccessV1::ReadWrite)
            .unwrap();
        let initial = [0xA5; 16];
        runtime.copy_from_host(buffer, 0, &initial).unwrap();
        let submitted = runtime
            .submit(queue, module, fill_request(buffer, 4, vec![]))
            .unwrap();
        match (succeeds, runtime.run_next()) {
            (true, Ok(VirtualRunProgressV1::Completed { completion, .. })) => {
                assert_eq!(completion, submitted);
                assert_eq!(
                    runtime.completion_state(submitted).unwrap(),
                    VirtualCompletionStateV1::Completed
                );
                let summary = runtime.completion_summary(submitted).unwrap().unwrap();
                assert_eq!(summary.invocations_executed, 4);
            }
            (false, Err(VirtualRuntimeErrorV1::Simulation { completion, source })) => {
                assert_eq!(completion, submitted);
                let SimulationErrorV1::Execution(error) = *source else {
                    panic!("expected schedule execution refusal");
                };
                assert!(matches!(
                    error.kind,
                    SimulationExecutionErrorKindV1::ScheduleResidentLimit { actual, limit }
                        if actual > limit && limit == max_resident_bytes
                ));
                assert_eq!(
                    runtime.completion_state(submitted).unwrap(),
                    VirtualCompletionStateV1::AbortedSimulation
                );
                assert!(runtime.completion_summary(submitted).unwrap().is_none());
            }
            result => panic!("unexpected schedule budget result: {result:?}"),
        }
        let mut output = [0; 16];
        runtime.copy_to_host(buffer, 0, &mut output).unwrap();
        let expected = if succeeds {
            [17, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0, 17, 0, 0, 0]
        } else {
            initial
        };
        assert_eq!(output, expected);
        assert!(!runtime.grants_hardware_authority());
        assert!(!runtime.predicts_performance());
        assert!(matches!(
            runtime.run_next().unwrap(),
            VirtualRunProgressV1::Idle
        ));
        runtime.release_buffer(buffer).unwrap();
        runtime.release_module(module).unwrap();
        runtime.release_queue(queue).unwrap();
    }
}

#[test]
fn explicit_dynamic_lds_is_bound_through_virtual_dispatch_and_summary() {
    let mut virtual_runtime = runtime(31);
    let module = virtual_runtime
        .register_module(admitted_dynamic_fill())
        .unwrap();
    let queue = virtual_runtime.create_queue(8).unwrap();
    let buffer = virtual_runtime
        .allocate_buffer(16, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    virtual_runtime.copy_from_host(buffer, 0, &[0; 16]).unwrap();
    let dynamic = DynamicWorkgroupMemoryRequestV1::new(16);
    let completion = virtual_runtime
        .submit_with_dynamic_workgroup_memory(
            queue,
            module,
            fill_request(buffer, 4, vec![]),
            dynamic,
        )
        .unwrap();
    let VirtualRunProgressV1::Completed {
        completion: observed,
        summary,
    } = virtual_runtime.run_next().unwrap()
    else {
        panic!("dynamic virtual dispatch did not complete")
    };
    assert_eq!(observed, completion);
    assert_eq!(summary.dynamic_workgroup_memory, Some(dynamic));
    assert_eq!(
        virtual_runtime
            .completion_summary(completion)
            .unwrap()
            .unwrap()
            .dynamic_workgroup_memory,
        Some(dynamic)
    );

    let mut hostile = runtime(32);
    let module = hostile.register_module(admitted_fill()).unwrap();
    let queue = hostile.create_queue(8).unwrap();
    let buffer = hostile
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    hostile.copy_from_host(buffer, 0, &[0; 4]).unwrap();
    hostile
        .submit_with_dynamic_workgroup_memory(
            queue,
            module,
            fill_request(buffer, 1, vec![]),
            dynamic,
        )
        .unwrap();
    assert!(matches!(
        hostile.run_next(),
        Err(VirtualRuntimeErrorV1::Simulation { .. })
    ));
}

#[test]
fn dynamic_failure_aborts_dependents_without_fabricating_completion() {
    let mut runtime = runtime(2);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 4]).unwrap();
    let mut failing_request = fill_request(buffer, 1, vec![]);
    failing_request.grid[0] = 2;
    let failing = runtime.submit(queue, module, failing_request).unwrap();
    let dependent = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![failing]))
        .unwrap();

    assert!(matches!(
        runtime.run_next(),
        Err(VirtualRuntimeErrorV1::Simulation { completion, .. }) if completion == failing
    ));
    assert_eq!(
        runtime.completion_state(failing).unwrap(),
        VirtualCompletionStateV1::AbortedSimulation
    );
    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::AbortedDependency { completion, dependency }
            if completion == dependent && dependency == failing
    ));
}

#[test]
fn cancellation_before_publication_is_terminal_and_releases_resources() {
    let mut runtime = runtime(15);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[3; 4]).unwrap();
    let cancelled = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();

    runtime.cancel_completion(cancelled).unwrap();
    assert_eq!(
        runtime.completion_state(cancelled).unwrap(),
        VirtualCompletionStateV1::Cancelled
    );
    assert!(runtime.completion_summary(cancelled).unwrap().is_none());
    assert!(matches!(
        runtime.cancel_completion(cancelled),
        Err(VirtualRuntimeErrorV1::CompletionNotPrepared { ordinal })
            if ordinal == cancelled.ordinal()
    ));
    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::Idle
    ));
    let mut output = [0; 4];
    runtime.copy_to_host(buffer, 0, &mut output).unwrap();
    assert_eq!(output, [3; 4]);
    runtime.release_buffer(buffer).unwrap();
    runtime.release_module(module).unwrap();
    runtime.release_queue(queue).unwrap();
}

#[test]
fn cancellation_aborts_dependents_without_executing_them() {
    let mut runtime = runtime(16);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[5; 4]).unwrap();
    let cancelled = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    let dependent = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![cancelled]))
        .unwrap();

    runtime.cancel_completion(cancelled).unwrap();
    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::AbortedDependency { completion, dependency }
            if completion == dependent && dependency == cancelled
    ));
    assert_eq!(
        runtime.completion_state(dependent).unwrap(),
        VirtualCompletionStateV1::AbortedDependency
    );
    let mut output = [0; 4];
    runtime.copy_to_host(buffer, 0, &mut output).unwrap();
    assert_eq!(output, [5; 4]);
    runtime.release_buffer(buffer).unwrap();
    runtime.release_module(module).unwrap();
    runtime.release_queue(queue).unwrap();
}

#[test]
fn early_release_is_rejected_atomically_until_completion() {
    let mut runtime = runtime(3);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 4]).unwrap();
    runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();

    assert!(matches!(
        runtime.release_buffer(buffer),
        Err(VirtualRuntimeErrorV1::Model(
            TransitionErrorV1::ResourceInUse(_)
        ))
    ));
    runtime.run_next().unwrap();
    runtime.release_buffer(buffer).unwrap();
}

#[test]
fn ambiguous_completion_retains_resources_until_explicit_quiescence() {
    let mut runtime = runtime(4);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 4]).unwrap();
    let completion = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    runtime.mark_completion_ambiguous(completion).unwrap();

    assert!(matches!(
        runtime.release_buffer(buffer),
        Err(VirtualRuntimeErrorV1::Model(
            TransitionErrorV1::ResourceInUse(_)
        ))
    ));
    assert!(matches!(
        runtime.settle_ambiguous_completion(completion),
        Err(VirtualRuntimeErrorV1::QueueNotQuiescent { .. })
    ));
    runtime.quiesce_queue(queue).unwrap();
    runtime.settle_ambiguous_completion(completion).unwrap();
    assert_eq!(
        runtime.completion_state(completion).unwrap(),
        VirtualCompletionStateV1::FailedQuiescent
    );
    runtime.release_buffer(buffer).unwrap();
}

#[test]
fn timeout_observation_does_not_claim_execution_stopped() {
    let mut runtime = runtime(17);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[7; 4]).unwrap();
    let completion = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();

    runtime.observe_completion_timeout(completion).unwrap();
    assert_eq!(
        runtime.completion_state(completion).unwrap(),
        VirtualCompletionStateV1::Ambiguous
    );
    assert_eq!(
        runtime.completion_ambiguity(completion).unwrap(),
        Some(VirtualCompletionAmbiguityV1::WaitDeadlineExpired)
    );
    assert!(matches!(
        runtime.buffer_snapshot(buffer),
        Err(VirtualRuntimeErrorV1::HostAccessWhileRetained { .. })
    ));
    runtime.quiesce_queue(queue).unwrap();
    runtime.settle_ambiguous_completion(completion).unwrap();
    assert_eq!(
        runtime.completion_ambiguity(completion).unwrap(),
        Some(VirtualCompletionAmbiguityV1::WaitDeadlineExpired)
    );
    assert!(matches!(
        runtime.buffer_snapshot(buffer),
        Ok(snapshot) if snapshot.initialized == [false; 4]
    ));
}

#[test]
fn reset_atomically_replaces_the_complete_runtime_generation() {
    let mut runtime = runtime(18);
    let previous_identity = runtime.config().runtime_identity;
    let module = runtime.register_module(admitted_fill()).unwrap();
    let ambiguous_queue = runtime.create_queue(8).unwrap();
    let prepared_queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(8, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[11; 8]).unwrap();
    let ambiguous = runtime
        .submit(ambiguous_queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    let prepared = runtime
        .submit(prepared_queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    runtime.observe_completion_timeout(ambiguous).unwrap();
    let replacement_identity = IdentityDigestV1::from_untrusted_bytes([19; 32]);

    let summary = runtime.reset_generation(replacement_identity).unwrap();
    assert_eq!(summary.previous_runtime_identity, previous_identity);
    assert_eq!(summary.replacement_runtime_identity, replacement_identity);
    assert_eq!(summary.cancelled_prepared_dispatches, 1);
    assert_eq!(summary.settled_ambiguous_dispatches, 1);
    assert_eq!(summary.released_buffers, 1);
    assert_eq!(summary.released_modules, 1);
    assert_eq!(summary.released_queues, 2);
    assert_eq!(runtime.config().runtime_identity, replacement_identity);
    for (kind, foreign) in [
        ("buffer", runtime.buffer_snapshot(buffer).unwrap_err()),
        ("module", runtime.release_module(module).unwrap_err()),
        ("queue", runtime.release_queue(prepared_queue).unwrap_err()),
        (
            "completion",
            runtime.completion_state(prepared).unwrap_err(),
        ),
    ] {
        assert!(
            matches!(foreign, VirtualRuntimeErrorV1::ForeignHandle { kind: observed } if observed == kind)
        );
    }
    let fresh = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(fresh, 0, &[23; 4]).unwrap();
}

#[test]
fn reset_rejects_reused_or_zero_identity_without_mutation() {
    let mut runtime = runtime(20);
    let original_identity = runtime.config().runtime_identity;
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[29; 4]).unwrap();

    assert!(matches!(
        runtime.reset_generation(original_identity),
        Err(VirtualRuntimeErrorV1::ReusedResetIdentity)
    ));
    assert!(matches!(
        runtime.reset_generation(IdentityDigestV1::from_untrusted_bytes([0; 32])),
        Err(VirtualRuntimeErrorV1::InvalidRuntimeIdentity)
    ));
    let mut output = [0; 4];
    runtime.copy_to_host(buffer, 0, &mut output).unwrap();
    assert_eq!(output, [29; 4]);
}

#[test]
fn foreign_handles_and_uninitialized_reads_are_typed_failures() {
    let mut first = runtime(5);
    let mut second = runtime(6);
    let foreign = second
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    assert!(matches!(
        first.copy_from_host(foreign, 0, &[1]),
        Err(VirtualRuntimeErrorV1::ForeignHandle { kind: "buffer" })
    ));
    let local = first
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    let mut output = [0; 4];
    assert!(matches!(
        first.copy_to_host(local, 0, &mut output),
        Err(VirtualRuntimeErrorV1::UninitializedHostRead { offset: 0 })
    ));
    first.release_buffer(local).unwrap();
    assert!(matches!(
        first.copy_from_host(local, 0, &[1]),
        Err(VirtualRuntimeErrorV1::ReleasedHandle { kind: "buffer", .. })
    ));
}

#[test]
fn overlapping_aliases_share_one_copyback_allocation() {
    let mut runtime = runtime(7);
    let module = runtime.register_module(admitted_alias_writes()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(8, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 8]).unwrap();
    let completion = runtime
        .submit(queue, module, alias_request(buffer, (0, 1), (0, 1)))
        .unwrap();

    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::Completed { completion: observed, .. }
            if observed == completion
    ));
    let mut output = [0; 8];
    runtime.copy_to_host(buffer, 0, &mut output).unwrap();
    assert_eq!(output, [23, 0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn ambiguity_blocks_host_access_and_invalidates_writable_alias_union() {
    let mut runtime = runtime(8);
    let module = runtime.register_module(admitted_alias_writes()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(16, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[1; 16]).unwrap();
    let completion = runtime
        .submit(queue, module, alias_request(buffer, (0, 2), (4, 2)))
        .unwrap();
    let mut output = [0; 1];
    assert!(matches!(
        runtime.copy_to_host(buffer, 0, &mut output),
        Err(VirtualRuntimeErrorV1::HostAccessWhileRetained { .. })
    ));
    assert!(matches!(
        runtime.copy_from_host(buffer, 0, &[2]),
        Err(VirtualRuntimeErrorV1::HostAccessWhileRetained { .. })
    ));
    assert!(matches!(
        runtime.buffer_snapshot(buffer),
        Err(VirtualRuntimeErrorV1::HostAccessWhileRetained { .. })
    ));

    runtime.mark_completion_ambiguous(completion).unwrap();
    assert!(matches!(
        runtime.copy_to_host(buffer, 0, &mut output),
        Err(VirtualRuntimeErrorV1::HostAccessWhileRetained { .. })
    ));
    runtime.quiesce_queue(queue).unwrap();
    runtime.settle_ambiguous_completion(completion).unwrap();
    let snapshot = runtime.buffer_snapshot(buffer).unwrap();
    assert_eq!(&snapshot.initialized[..12], &[false; 12]);
    assert_eq!(&snapshot.initialized[12..], &[true; 4]);
    assert!(matches!(
        runtime.copy_to_host(buffer, 0, &mut output),
        Err(VirtualRuntimeErrorV1::UninitializedHostRead { offset: 0 })
    ));
}

#[test]
fn host_lifetime_evidence_is_canonical_and_binds_the_blocking_dispatch() {
    let mut runtime = runtime(41);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[7; 4]).unwrap();
    let completion = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    assert!(matches!(
        runtime.release_buffer(buffer),
        Err(VirtualRuntimeErrorV1::Model(
            TransitionErrorV1::ResourceInUse(_)
        ))
    ));

    let evidence = runtime
        .capture_host_lifetime_evidence_v1(
            buffer,
            VirtualHostLifetimeOperationV1::ReleaseBuffer,
            VirtualHostLifetimeEvidenceLimitsV1::new(8, 1 << 20).unwrap(),
        )
        .unwrap();
    assert_eq!(
        evidence.finding,
        VirtualHostLifetimeFindingV1::ReleaseWhileRetained
    );
    assert_eq!(evidence.retained_dispatches, 1);
    assert_eq!(
        evidence.blockers[0].completion_ordinal,
        completion.ordinal()
    );
    assert!(matches!(
        evidence.blockers[0].dispatch_input,
        VirtualDispatchInputBindingV1::Exact { .. }
    ));
    assert_eq!(
        evidence.completeness,
        VirtualHostLifetimeCompletenessV1::Complete
    );
    assert!(!evidence.grants_execution_authority());

    let canonical = evidence.to_canonical_bytes().unwrap();
    assert_eq!(
        VirtualHostLifetimeEvidenceV1::from_canonical_bytes(&canonical).unwrap(),
        evidence
    );
    let mut corrupt: serde_json::Value = serde_json::from_slice(&canonical).unwrap();
    corrupt["buffer_ordinal"] = serde_json::json!(99);
    assert!(
        VirtualHostLifetimeEvidenceV1::from_canonical_bytes(&serde_json::to_vec(&corrupt).unwrap())
            .is_err()
    );
}

#[test]
fn host_lifetime_evidence_reports_bounded_partial_inventory_and_input_identity() {
    let mut runtime = runtime(42);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[3; 4]).unwrap();
    runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();

    let evidence = runtime
        .capture_host_lifetime_evidence_v1(
            buffer,
            VirtualHostLifetimeOperationV1::SnapshotBuffer,
            VirtualHostLifetimeEvidenceLimitsV1::new(1, 0).unwrap(),
        )
        .unwrap();
    assert_eq!(evidence.retained_dispatches, 2);
    assert_eq!(evidence.blockers.len(), 1);
    assert!(matches!(
        evidence.completeness,
        VirtualHostLifetimeCompletenessV1::PartialBlockerAndInputIdentity {
            total_blockers: 2,
            retained_blockers: 1
        }
    ));
    assert!(matches!(
        evidence.blockers[0].dispatch_input,
        VirtualDispatchInputBindingV1::Unavailable {
            reason: VirtualDispatchInputUnavailableReasonV1::SnapshotByteLimit,
            ..
        }
    ));
}

#[test]
fn quiescence_rejects_sibling_prepared_work_without_stranding_it() {
    let mut runtime = runtime(9);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let buffer = runtime
        .allocate_buffer(4, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 4]).unwrap();
    let first = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();
    let second = runtime
        .submit(queue, module, fill_request(buffer, 1, vec![]))
        .unwrap();

    assert!(matches!(
        runtime.mark_completion_ambiguous(first),
        Err(VirtualRuntimeErrorV1::QueueHasPreparedDispatch {
            completion,
            ..
        }) if completion == second.ordinal()
    ));
    assert!(matches!(
        runtime.quiesce_queue(queue),
        Err(VirtualRuntimeErrorV1::QueueHasPreparedDispatch {
            completion,
            ..
        }) if completion == first.ordinal()
    ));
    assert!(matches!(
        runtime.run_next().unwrap(),
        VirtualRunProgressV1::Completed { completion, .. } if completion == first
    ));
    runtime.mark_completion_ambiguous(second).unwrap();
    runtime.quiesce_queue(queue).unwrap();
    runtime.settle_ambiguous_completion(second).unwrap();
    runtime.copy_from_host(buffer, 0, &[3; 4]).unwrap();
}

#[test]
fn malformed_views_fail_before_prepare_and_retain_nothing() {
    let mut runtime = runtime(10);
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();

    let partial = runtime
        .allocate_buffer(3, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(partial, 0, &[0; 3]).unwrap();
    assert!(matches!(
        runtime.submit(queue, module, fill_request(partial, 1, vec![])),
        Err(VirtualRuntimeErrorV1::SimulatorBuffer(_))
    ));
    runtime.release_buffer(partial).unwrap();

    let buffer = runtime
        .allocate_buffer(8, VirtualBufferAccessV1::ReadWrite)
        .unwrap();
    runtime.copy_from_host(buffer, 0, &[0; 8]).unwrap();
    let mut invalid_alignment = fill_request(buffer, 1, vec![]);
    let VirtualArgumentV1::Buffer { alignment, .. } = &mut invalid_alignment.arguments[0] else {
        unreachable!();
    };
    *alignment = 3;
    assert!(matches!(
        runtime.submit(queue, module, invalid_alignment),
        Err(VirtualRuntimeErrorV1::SimulatorBuffer(_))
    ));

    let mut out_of_bounds = fill_request(buffer, 2, vec![]);
    let VirtualArgumentV1::Buffer { byte_offset, .. } = &mut out_of_bounds.arguments[0] else {
        unreachable!();
    };
    *byte_offset = 4;
    assert!(matches!(
        runtime.submit(queue, module, out_of_bounds),
        Err(VirtualRuntimeErrorV1::InvalidBufferRange)
    ));

    let mut mixed = fill_request(buffer, 1, vec![]);
    mixed.arguments.push(VirtualArgumentV1::Buffer {
        buffer,
        element: ScalarType::U16,
        access: AccessMode::ReadWrite,
        alignment: 2,
        byte_offset: 0,
        elements: 1,
    });
    assert!(matches!(
        runtime.submit(queue, module, mixed),
        Err(VirtualRuntimeErrorV1::SimulatorBuffer(_))
    ));
    runtime.release_buffer(buffer).unwrap();
    runtime.release_module(module).unwrap();
    runtime.release_queue(queue).unwrap();
}

#[test]
fn dispatch_argument_and_retained_byte_limits_fail_closed() {
    let argument_limits = VirtualRuntimeLimitsV1 {
        max_arguments_per_dispatch: 1,
        ..VirtualRuntimeLimitsV1::default()
    };
    let mut runtime = VirtualRuntimeV1::new(VirtualRuntimeConfigV1 {
        runtime_identity: IdentityDigestV1::from_untrusted_bytes([11; 32]),
        target: VirtualTargetProfileV1::Amdgpu64TargetNeutral,
        runtime_limits: argument_limits,
        simulation_limits: SimulationLimitsV1::default(),
    })
    .unwrap();
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    let request = VirtualDispatchRequestV1 {
        kernel: "fill".into(),
        grid: [1, 1, 1],
        workgroup: [64, 1, 1],
        arguments: vec![
            VirtualArgumentV1::Scalar(ScalarBitsV1::u32(1)),
            VirtualArgumentV1::Scalar(ScalarBitsV1::u32(2)),
        ],
        dependencies: vec![],
    };
    assert!(matches!(
        runtime.submit(queue, module, request),
        Err(VirtualRuntimeErrorV1::CapacityExceeded(
            "dispatch arguments"
        ))
    ));

    let byte_limits = VirtualRuntimeLimitsV1 {
        max_retained_dispatch_bytes: 1,
        ..VirtualRuntimeLimitsV1::default()
    };
    let mut runtime = VirtualRuntimeV1::new(VirtualRuntimeConfigV1 {
        runtime_identity: IdentityDigestV1::from_untrusted_bytes([12; 32]),
        target: VirtualTargetProfileV1::Amdgpu64TargetNeutral,
        runtime_limits: byte_limits,
        simulation_limits: SimulationLimitsV1::default(),
    })
    .unwrap();
    let module = runtime.register_module(admitted_fill()).unwrap();
    let queue = runtime.create_queue(8).unwrap();
    assert!(matches!(
        runtime.submit(
            queue,
            module,
            VirtualDispatchRequestV1 {
                kernel: "fill".into(),
                grid: [1, 1, 1],
                workgroup: [64, 1, 1],
                arguments: vec![],
                dependencies: vec![],
            }
        ),
        Err(VirtualRuntimeErrorV1::CapacityExceeded(
            "retained dispatch bytes"
        ))
    ));
}

#[test]
fn aggregate_model_capacity_and_multiple_exact_targets_are_rejected() {
    let limits = VirtualRuntimeLimitsV1 {
        max_user_allocations: 4_095,
        max_modules: 1,
        max_queues: 1,
        ..VirtualRuntimeLimitsV1::default()
    };
    assert!(matches!(
        VirtualRuntimeV1::new(VirtualRuntimeConfigV1 {
            runtime_identity: IdentityDigestV1::from_untrusted_bytes([13; 32]),
            target: VirtualTargetProfileV1::Amdgpu64TargetNeutral,
            runtime_limits: limits,
            simulation_limits: SimulationLimitsV1::default(),
        }),
        Err(VirtualRuntimeErrorV1::InvalidLimit(
            "aggregate model allocations"
        ))
    ));

    let module = admitted_fill_with_capabilities([
        gfx942_xnack_minus_target_capability(),
        gfx950_xnack_minus_target_capability(),
    ]);
    let mut runtime = runtime(14);
    assert!(matches!(
        runtime.register_module(module),
        Err(VirtualRuntimeErrorV1::MultipleExactTargets { .. })
    ));
}
