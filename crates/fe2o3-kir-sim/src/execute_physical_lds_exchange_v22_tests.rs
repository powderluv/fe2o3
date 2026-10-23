//! Inert low-level controls; the actual source lane qualifies separately.
use super::*;
fn site(operation: u32) -> CompactSite {
    CompactSite {
        function: 0,
        block: BlockId(0),
        operation: Some(operation),
    }
}
fn symbolic(value: Value) -> RuntimeValue {
    RuntimeValue::PhysicalEntry(value)
}
fn with_engine(test: impl FnOnce(&mut Engine<'_, NoopSimulationEventSinkV1>)) {
    let module = crate::physical_lds_exchange_test_fixture::module();
    let allocation = Allocation {
        address_space: AddressSpace::Global,
        access: AccessMode::ReadWrite,
        alignment: 4,
        bytes: vec![0xa5; 4],
        initialized: vec![false; 4],
        workgroup_published: vec![],
        workgroup_writer: vec![],
        observation_descriptor: Vec::new(),
    };
    let mut sink = NoopSimulationEventSinkV1;
    let mut debug_sink = NoopSimulationDebugSinkV1;
    let context = physical_lds_context_v22::Context::new(&module).unwrap();
    let mut engine = Engine {
        physical_lds_context: Some(&context),
        physical_lds_state: Some(physical_lds_context_v22::State::new().unwrap()),
        module: &module,
        function_module_indices: vec![0],
        block_indices: vec![HashMap::from([(BlockId(0), 0)])],
        function_ssa_values: vec![200],
        value_layouts: &[],
        value_plan: FrameValuePlan::legacy(200),
        call_targets: vec![vec![]],
        switch_targets: vec![vec![]],
        target: SimulationTargetV1::amdgpu_64(),
        dynamic_workgroup_memory: None,
        matrix_bf16_numerics: crate::MatrixBf16NumericsV1::ExactV1,
        limits: SimulationLimitsV1::default(),
        policy: EventPolicyV1::Disabled,
        memory: Memory {
            allocations: HashMap::from([(7, allocation)]),
            argument_allocations: vec![],
            shared_allocations: HashMap::new(),
            next_allocation: 8,
            allocations_created: 1,
            live_bytes: 4,
            reuse: None,
        },
        sink: &mut sink,
        debug_capture: SimulationDebugCaptureLimitsV1::disabled(),
        debug_sink: &mut debug_sink,
        physical_debug: None,
        debug_origin_requested: false,
        debug_observation_requested: false,
        debug_frames_requested: false,
        debug_identity_failed: false,
        debug_origin: None,
        debug_records: 0,
        debug_delivery_stopped: true,
        allocation_lifecycle_requested: false,
        allocation_lifecycle_stopped: false,
        schedule_identity: SimulationScheduleIdentityV1::WorkgroupMajorLocalZyxCooperativeV1,
        schedule_decision: 0,
        steps: 0,
        events: 0,
        reserved_event_closures: 0,
        event_delivery_stopped: false,
        invocation: Some(SimulationInvocationV1 {
            global: [0, 0, 0],
            workgroup: [0, 0, 0],
            local: [0, 0, 0],
            workgroup_size: [128, 1, 1],
            workgroup_count: [1, 1, 1],
            launch_extent: [128, 1, 1],
        }),
        accesses: HashMap::new(),
        conflicting_bytes: 0,
        first_conflict: None,
        conflict_incomplete: false,
        workgroup_happens_before_epoch: 0,
        unmodeled_atomic_or_fence_happens_before: false,
        unmodeled_atomic_serialization: false,
        race_trackers: vec![],
        workgroup_allocations: vec![],
    };
    test(&mut engine);
}

#[test]
fn inline_lds_pending_value_keeps_runtime_and_binding_layout() {
    // The private runtime enum may use a smaller discriminant layout. Its
    // ceiling is the existing inline debug payload, whose legacy layout is
    // independently checked by the V21 binding regression.
    assert!(size_of::<RuntimeValue>() <= size_of::<SimulationDebugValueV1>());
    assert_eq!(size_of::<SimulationDebugBindingV1>(), 192);
    let value = Value::LdsPendingRead {
        generation: site(21),
        result: ValueId(77),
        bits: ScalarBitsV1::u32(7),
        epoch: 1,
    };
    assert_eq!(value.scalar_type(), ScalarType::U32);
    assert!(PhysicalEntryDebugSymbolicV20::from_runtime(&value).is_none());
    assert!(PhysicalGlobalCopyDebugSymbolicV21::from_runtime(&value).is_none());
    assert!(debug_value(&symbolic(value)).is_none());
}
#[test]
fn lgkm_completes_only_same_actual_lds_result_and_epoch_transactionally() {
    with_engine(|engine| {
        for local in 0..128 {
            engine.invocation.as_mut().unwrap().local[0] = local;
            engine.invocation.as_mut().unwrap().global[0] = u64::from(local);
            physical_lds_context_v22::issue_write(
                engine,
                site(16),
                local * 4,
                ScalarBitsV1::u32(local),
            )
            .unwrap();
            physical_lds_context_v22::complete_write(engine, site(17), site(16)).unwrap();
        }
        assert!(physical_lds_context_v22::validate_release(engine, site(18), 128).unwrap());
        engine.publish_workgroup();
        physical_lds_context_v22::mark_published(engine);

        let id = engine.module.functions[0].body.as_ref().unwrap().blocks[0].operations[21].results
            [0]
        .id;
        for case in 0..8 {
            let value = if case == 7 {
                RuntimeValue::Scalar(ScalarBitsV1::u32(9))
            } else {
                symbolic(Value::LdsPendingRead {
                    generation: match case {
                        1 => site(20),
                        2 => CompactSite {
                            function: 1,
                            ..site(21)
                        },
                        3 => CompactSite {
                            block: BlockId(1),
                            ..site(21)
                        },
                        _ => site(21),
                    },
                    result: if case == 4 { ValueId(999) } else { id },
                    bits: if case == 5 {
                        ScalarBitsV1::boolean(true)
                    } else {
                        ScalarBitsV1::u32(9)
                    },
                    epoch: if case == 6 { 2 } else { 1 },
                })
            };
            let mut values = RuntimeValues::from([(id, value)]);
            let before = values.clone();
            if case != 7 {
                assert!(scalar_value(engine, &values, id, &site(22)).is_err());
            }
            let result = complete_lgkm(engine, &mut values, site(22));
            assert_eq!(result.is_ok(), case == 0);
            if case == 0 {
                assert_eq!(values.len(), 1);
                assert_eq!(
                    scalar_value(engine, &values, id, &site(22)).unwrap(),
                    ScalarBitsV1::u32(9)
                );
                assert!(complete_lgkm(engine, &mut values, site(22)).is_err());
            } else {
                assert_eq!(values, before);
            }
        }
    });
}
#[test]
fn vm_wait_and_old_profile_cannot_complete_lds_or_kernarg_pending() {
    with_engine(|engine| {
        let id = engine.module.functions[0].body.as_ref().unwrap().blocks[0].operations[21].results
            [0]
        .id;
        let mut values = RuntimeValues::from([(
            id,
            symbolic(Value::LdsPendingRead {
                generation: site(21),
                result: id,
                bits: ScalarBitsV1::u32(23),
                epoch: 1,
            }),
        )]);
        let before = values.clone();
        assert!(
            physical_global_copy_pending_v21::complete_vm_for(
                engine,
                &mut values,
                site(22),
                Profile::LdsV22
            )
            .is_err()
        );
        assert!(
            physical_global_copy_pending_v21::complete_vm(engine, &mut values, site(22)).is_err()
        );
        assert!(
            physical_global_copy_pending_v21::complete_lgkm(engine, &mut values, site(22)).is_err()
        );
        assert_eq!(values, before);
    });
}
#[test]
fn write_wait_commits_but_does_not_publish_or_complete_at_barrier() {
    with_engine(|engine| {
        let frame = physical_lds_context_v22::frame_pointer(engine, site(0)).unwrap();
        assert!(
            engine.memory.allocations[&frame.allocation]
                .initialized
                .iter()
                .all(|x| !x)
        );
        physical_lds_context_v22::issue_write(engine, site(16), 0, ScalarBitsV1::u32(0x89abcdef))
            .unwrap();
        assert!(
            engine.memory.allocations[&frame.allocation]
                .initialized
                .iter()
                .all(|x| !x)
        );
        assert!(physical_lds_context_v22::barrier(engine, site(18)).is_err());
        let mut empty = RuntimeValues::new();
        assert!(
            physical_global_copy_pending_v21::complete_vm_for(
                engine,
                &mut empty,
                site(17),
                Profile::LdsV22
            )
            .is_err()
        );
        assert!(physical_lds_context_v22::complete_write(engine, site(17), site(15)).is_err());
        physical_lds_context_v22::complete_write(engine, site(17), site(16)).unwrap();
        assert!(physical_lds_context_v22::complete_write(engine, site(17), site(16)).is_err());
        assert_eq!(
            &engine.memory.allocations[&frame.allocation].bytes[..4],
            &0x89abcdefu32.to_le_bytes()
        );
        assert!(
            engine.memory.allocations[&frame.allocation]
                .workgroup_published
                .iter()
                .all(|x| !x)
        );
        engine.invocation.as_mut().unwrap().local[0] = 64;
        engine.invocation.as_mut().unwrap().global[0] = 64;
        let error = engine
            .memory
            .load(
                &frame,
                MemoryAccess::new(AddressSpace::Workgroup, 4),
                engine.target,
                engine.invocation.unwrap(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            SimulationExecutionErrorKindV1::WorkgroupUseBeforePublish { .. }
        ));
        assert!(physical_lds_context_v22::read(engine, site(21), 0).is_err());
        assert!(physical_lds_context_v22::validate_release(engine, site(18), 128).is_err());
    });
}
#[test]
fn exact_128_completed_writes_publish_once_and_no_global_happens_before_is_invented() {
    with_engine(|engine| {
        for local in 0..128 {
            engine.invocation.as_mut().unwrap().local[0] = local;
            engine.invocation.as_mut().unwrap().global[0] = u64::from(local);
            physical_lds_context_v22::issue_write(
                engine,
                site(16),
                local * 4,
                ScalarBitsV1::u32(local + 99),
            )
            .unwrap();
            physical_lds_context_v22::complete_write(engine, site(17), site(16)).unwrap();
            physical_lds_context_v22::barrier(engine, site(18)).unwrap();
        }
        assert!(physical_lds_context_v22::validate_release(engine, site(18), 64).is_err());
        assert!(physical_lds_context_v22::validate_release(engine, site(19), 128).is_err());
        assert!(physical_lds_context_v22::validate_release(engine, site(18), 128).unwrap());
        let barrier = &engine.physical_lds_context.unwrap().barrier;
        engine.publish_workgroup();
        engine.publish_global_happens_before(barrier);
        physical_lds_context_v22::mark_published(engine);
        assert_eq!(engine.workgroup_happens_before_epoch, 0);
        assert!(physical_lds_context_v22::validate_release(engine, site(18), 128).is_err());
        engine.invocation.as_mut().unwrap().local[0] = 0;
        engine.invocation.as_mut().unwrap().global[0] = 0;
        assert_eq!(
            physical_lds_context_v22::read(engine, site(21), 256).unwrap(),
            ScalarBitsV1::u32(163)
        );
        assert!(physical_lds_context_v22::read(engine, site(21), 0).is_err());
    });
}
#[test]
fn local_offsets_pending_reissue_and_non_u32_data_are_closed() {
    with_engine(|engine| {
        assert!(
            physical_lds_context_v22::issue_write(engine, site(16), 4, ScalarBitsV1::u32(1))
                .is_err()
        );
        assert!(
            physical_lds_context_v22::issue_write(engine, site(16), 0, ScalarBitsV1::boolean(true))
                .is_err()
        );
        physical_lds_context_v22::issue_write(engine, site(16), 0, ScalarBitsV1::u32(1)).unwrap();
        assert!(
            physical_lds_context_v22::issue_write(engine, site(16), 0, ScalarBitsV1::u32(2))
                .is_err()
        );
        engine.invocation.as_mut().unwrap().local[0] = 128;
        assert!(
            physical_lds_context_v22::issue_write(engine, site(16), 512, ScalarBitsV1::u32(2))
                .is_err()
        );
    });
}
