//! Inert mutation controls, not a constructor for authenticated source state.
use super::*;
use physical_entry_state_v20::{Half, Value};
use physical_global_copy_pending_v21::{complete_lgkm, complete_vm, kernarg_parameters};
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
fn pointer() -> PointerValue {
    PointerValue {
        exposed_generic: false,
        allocation: 7,
        byte_offset: 0,
        element: ScalarType::U32,
        address_space: AddressSpace::Global,
        access: AccessMode::ReadOnly,
        lower_bound: 0,
        upper_bound: 256,
        abi_argument_ordinal: 0,
    }
}
fn with_engine(test: impl FnOnce(&mut Engine<'_, NoopSimulationEventSinkV1>)) {
    let module = crate::physical_global_copy_test_fixture::module();
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
    let mut engine = Engine {
        module: &module,
        physical_lds_context: None,
        physical_lds_state: None,
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
            workgroup_size: [64, 1, 1],
            workgroup_count: [1, 1, 1],
            launch_extent: [64, 1, 1],
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
fn global_copy_kernarg_requires_same_declaration_and_both_actual_roots() {
    let params = [ValueId(0), ValueId(1)];
    let low = symbolic(Value::GlobalCopyKernarg {
        half: Half::Low,
        declaration: site(0),
        parameters: params,
    });
    let high = symbolic(Value::GlobalCopyKernarg {
        half: Half::High,
        declaration: site(0),
        parameters: params,
    });
    assert_eq!(kernarg_parameters(&low, &high), Some(params));
    assert!(kernarg_parameters(&high, &low).is_none());
    for case in 0..3 {
        let bad = symbolic(Value::GlobalCopyKernarg {
            half: Half::High,
            declaration: if case == 0 { site(1) } else { site(0) },
            parameters: if case == 1 {
                [ValueId(1), ValueId(0)]
            } else if case == 2 {
                [ValueId(0), ValueId(2)]
            } else {
                params
            },
        });
        assert!(kernarg_parameters(&low, &bad).is_none());
    }
}
#[test]
fn pending_read_is_typed_but_never_scalar_or_debug_bits() {
    let value = symbolic(Value::GlobalCopyPendingRead {
        generation: site(13),
        result: ValueId(20),
        bits: ScalarBitsV1::u32(u32::MAX),
    });
    assert_eq!(runtime_type(&value), Type::Scalar(ScalarType::U32));
    assert!(debug_value(&value).is_none());
    with_engine(|engine| {
        let values = RuntimeValues::from([(ValueId(20), value)]);
        assert!(scalar_value(engine, &values, ValueId(20), &site(14)).is_err());
    });
}
#[test]
fn vm_wait_requires_exact_actual_load_result_and_generation_then_changes_same_id() {
    with_engine(|engine| {
        let id = engine.module.functions[0].body.as_ref().unwrap().blocks[0].operations[13].results
            [0]
        .id;
        for case in 0..7 {
            let value = if case == 4 {
                RuntimeValue::Scalar(ScalarBitsV1::u32(42))
            } else {
                symbolic(Value::GlobalCopyPendingRead {
                    generation: match case {
                        1 => site(12),
                        5 => CompactSite {
                            block: BlockId(1),
                            ..site(13)
                        },
                        6 => CompactSite {
                            function: 1,
                            ..site(13)
                        },
                        _ => site(13),
                    },
                    result: if case == 2 { ValueId(999) } else { id },
                    bits: if case == 3 {
                        ScalarBitsV1::boolean(true)
                    } else {
                        ScalarBitsV1::u32(42)
                    },
                })
            };
            let mut values = RuntimeValues::from([(id, value)]);
            let before = values.clone();
            let result = complete_vm(engine, &mut values, site(14));
            assert_eq!(result.is_ok(), case == 0);
            if case == 0 {
                assert_eq!(values.len(), 1);
                assert_eq!(
                    scalar_value(engine, &values, id, &site(15)).unwrap(),
                    ScalarBitsV1::u32(42)
                );
                assert!(complete_vm(engine, &mut values, site(14)).is_err());
            } else {
                assert_eq!(values, before);
            }
        }
        let mut values = RuntimeValues::from([(
            id,
            symbolic(Value::GlobalCopyPendingRead {
                generation: site(13),
                result: id,
                bits: ScalarBitsV1::u32(7),
            }),
        )]);
        assert!(complete_vm(engine, &mut values, site(15)).is_err());
    });
}
#[test]
fn lgkm_wait_validates_all_eight_ids_before_any_readiness_change() {
    with_engine(|engine| {
        let mut values = RuntimeValues::new();
        for op in 1..=4 {
            let operation =
                &engine.module.functions[0].body.as_ref().unwrap().blocks[0].operations[op];
            for (part, half) in [Half::Low, Half::High].into_iter().enumerate() {
                let value = if !op.is_multiple_of(2) {
                    let mut p = pointer();
                    p.abi_argument_ordinal = if op == 1 { 0 } else { 1 };
                    p.allocation = if op == 1 { 7 } else { 9 };
                    p.access = if op == 1 {
                        AccessMode::ReadOnly
                    } else {
                        AccessMode::ReadWrite
                    };
                    Value::GlobalCopyPendingPointer {
                        half,
                        generation: site(op as u32),
                        pointer: p,
                    }
                } else {
                    Value::GlobalCopyPendingLength {
                        half,
                        generation: site(op as u32),
                        bits: ScalarBitsV1::u32(if part == 0 { 64 } else { 0 }),
                    }
                };
                values.insert(operation.results[part].id, symbolic(value));
            }
        }
        let pristine = values.clone();
        let id =
            engine.module.functions[0].body.as_ref().unwrap().blocks[0].operations[4].results[1].id;
        values.insert(
            id,
            symbolic(Value::GlobalCopyPendingLength {
                half: Half::High,
                generation: site(3),
                bits: ScalarBitsV1::u32(0),
            }),
        );
        let before = values.clone();
        assert!(complete_lgkm(engine, &mut values, site(5)).is_err());
        assert_eq!(values, before);
        let mut values = pristine;
        assert!(complete_lgkm(engine, &mut values, site(6)).is_err());
        complete_lgkm(engine, &mut values, site(5)).unwrap();
        assert_eq!(values.len(), 8);
        for (_, value) in values
            .legacy_iter()
            .expect("physical fixture retains Legacy values")
        {
            assert!(!matches!(
                value,
                RuntimeValue::PhysicalEntry(
                    Value::GlobalCopyPendingPointer { .. } | Value::GlobalCopyPendingLength { .. }
                )
            ));
        }
        assert!(complete_lgkm(engine, &mut values, site(5)).is_err());
    });
}
