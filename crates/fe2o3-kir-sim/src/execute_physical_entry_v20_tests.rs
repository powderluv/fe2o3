//! Internal negative controls; these inert values are not authenticated source owners.
use super::*;
use fe2o3_kernel_ir::{
    Gfx942PhysicalEntryInstructionVNext as Instruction, Gfx942PhysicalEntryOpcodeV20 as Opcode,
    Gfx942PhysicalEntrySourceSiteVNext as SourceSite, Gfx942PhysicalEntryStepVNext as Step,
};
use physical_entry_state_v20::{AddressChain, Half, Results, Value};

fn site(index: u32) -> CompactSite {
    CompactSite {
        function: 0,
        block: BlockId(0),
        operation: Some(index),
    }
}
fn pointer() -> PointerValue {
    PointerValue {
        exposed_generic: false,
        allocation: 7,
        byte_offset: 0,
        element: ScalarType::U32,
        address_space: AddressSpace::Global,
        access: AccessMode::ReadWrite,
        lower_bound: 0,
        upper_bound: 4,
        abi_argument_ordinal: 0,
    }
}
fn chain() -> AddressChain {
    AddressChain {
        pointer: pointer(),
        pointer_generation: site(1),
        displacement: 0,
        displacement_generation: site(2),
        low_add: site(3),
    }
}
fn symbolic(value: Value) -> RuntimeValue {
    RuntimeValue::PhysicalEntry(value)
}
fn mask(value: u64) -> RuntimeValue {
    RuntimeValue::Scalar(
        ScalarBitsV1::new(
            ScalarType::U64,
            u128::from(value),
            SimulationTargetV1::amdgpu_64(),
        )
        .unwrap(),
    )
}
fn operation(opcode: Opcode, d: u8, a: u8, b: u8, imm: u32, ids: &[ValueId]) -> Operation {
    let instruction = Instruction {
        opcode,
        destination: d,
        source0: a,
        source1: b,
        immediate: imm,
    };
    instruction.validate_shape().unwrap();
    let mut operands = [None; 6];
    for (slot, id) in operands.iter_mut().zip(ids) {
        *slot = Some(*id);
    }
    let step = Step {
        site: SourceSite {
            occurrence: 1,
            raw_block: 3,
            semantic_block_index: 0,
            semantic_block_identity: [1; 32],
            semantic_callable_index: 2,
        },
        native_ordinal: 0,
        instruction,
        operands,
    };
    step.validate_shape().unwrap();
    let results = instruction
        .result_registers()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, r)| ValueDef::new(ValueId(100 + i as u32), Type::Scalar(r.scalar_type())))
        .collect();
    Operation::new(results, OperationKind::Gfx942PhysicalEntryStep(step))
}
fn with_engine(test: impl FnOnce(&mut Engine<'_, NoopSimulationEventSinkV1>)) {
    let mut module = Module::new("physical-inert-unit");
    module.functions.push(Function::declaration(
        "physical-inert-unit-entry",
        fe2o3_kernel_ir::Signature::new(vec![], vec![]),
    ));
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
        block_indices: vec![HashMap::new()],
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
fn kernarg_halves_require_same_declaration_and_all_five_bindings() {
    let p = [ValueId(0), ValueId(1), ValueId(2), ValueId(3), ValueId(4)];
    let low = symbolic(Value::Kernarg {
        half: Half::Low,
        declaration: site(0),
        parameters: p,
    });
    let high = symbolic(Value::Kernarg {
        half: Half::High,
        declaration: site(0),
        parameters: p,
    });
    assert_eq!(
        physical_entry_state_v20::kernarg_parameters(&low, &high),
        Some(p)
    );
    assert!(physical_entry_state_v20::kernarg_parameters(&high, &low).is_none());
    let mut q = p;
    q.swap(1, 2);
    assert!(
        physical_entry_state_v20::kernarg_parameters(
            &low,
            &symbolic(Value::Kernarg {
                half: Half::High,
                declaration: site(0),
                parameters: q
            })
        )
        .is_none()
    );
    assert!(
        physical_entry_state_v20::kernarg_parameters(
            &low,
            &symbolic(Value::Kernarg {
                half: Half::High,
                declaration: site(9),
                parameters: p
            })
        )
        .is_none()
    );
}
#[test]
fn high_half_requires_exact_allocation_bounds_abi_and_generation() {
    let c = chain();
    let carry = symbolic(Value::CarryLow(c.clone()));
    let offset = symbolic(Value::ScaledOffset {
        half: Half::High,
        generation: c.displacement_generation,
        displacement: c.displacement,
    });
    let good = symbolic(Value::Output {
        half: Half::High,
        generation: c.pointer_generation,
        pointer: c.pointer.clone(),
    });
    assert_eq!(
        physical_entry_state_v20::high_chain(&good, &offset, &carry),
        Some(c.clone())
    );
    for index in 0..5 {
        let mut changed = c.pointer.clone();
        match index {
            0 => changed.allocation += 1,
            1 => changed.upper_bound += 4,
            2 => changed.lower_bound += 1,
            3 => changed.abi_argument_ordinal += 1,
            _ => changed.access = AccessMode::ReadOnly,
        };
        let bad = symbolic(Value::Output {
            half: Half::High,
            generation: c.pointer_generation,
            pointer: changed,
        });
        assert!(physical_entry_state_v20::high_chain(&bad, &offset, &carry).is_none());
    }
    let wrong_generation = symbolic(Value::Output {
        half: Half::High,
        generation: site(99),
        pointer: c.pointer.clone(),
    });
    assert!(physical_entry_state_v20::high_chain(&wrong_generation, &offset, &carry).is_none());
}
#[test]
fn high_half_rejects_foreign_scaled_offset_and_outgoing_carry() {
    let c = chain();
    let base = symbolic(Value::Output {
        half: Half::High,
        generation: c.pointer_generation,
        pointer: c.pointer.clone(),
    });
    let carry = symbolic(Value::CarryLow(c.clone()));
    for (half, generation, displacement) in [
        (Half::Low, c.displacement_generation, 0),
        (Half::High, site(99), 0),
        (Half::High, c.displacement_generation, 4),
    ] {
        let offset = symbolic(Value::ScaledOffset {
            half,
            generation,
            displacement,
        });
        assert!(physical_entry_state_v20::high_chain(&base, &offset, &carry).is_none());
    }
    let offset = symbolic(Value::ScaledOffset {
        half: Half::High,
        generation: c.displacement_generation,
        displacement: 0,
    });
    let outgoing = symbolic(Value::CarryHigh {
        chain: c,
        high_add: site(4),
    });
    assert!(physical_entry_state_v20::high_chain(&base, &offset, &outgoing).is_none());
    assert!(physical_entry_state_v20::high_chain(&base, &offset, &mask(0)).is_none());
}
#[test]
fn store_requires_same_exact_low_add_chain() {
    let c = chain();
    let low = symbolic(Value::AddressLow(c.clone()));
    let high = symbolic(Value::AddressHigh {
        chain: c.clone(),
        high_add: site(4),
    });
    assert_eq!(
        physical_entry_state_v20::store_chain(&low, &high),
        Some(c.clone())
    );
    let mut other = c;
    other.low_add = site(99);
    assert!(
        physical_entry_state_v20::store_chain(
            &low,
            &symbolic(Value::AddressHigh {
                chain: other,
                high_add: site(4)
            })
        )
        .is_none()
    );
    assert!(physical_entry_state_v20::store_chain(&high, &low).is_none());
}
#[test]
fn symbolic_values_never_export_as_scalar_debug_bits() {
    let c = chain();
    for value in [
        Value::Output {
            half: Half::Low,
            generation: site(1),
            pointer: pointer(),
        },
        Value::AddressLow(c.clone()),
        Value::AddressHigh {
            chain: c.clone(),
            high_add: site(4),
        },
        Value::CarryLow(c.clone()),
        Value::CarryHigh {
            chain: c,
            high_add: site(4),
        },
    ] {
        let ty = value.scalar_type();
        let runtime = symbolic(value);
        assert_eq!(runtime_type(&runtime), Type::Scalar(ty));
        assert!(debug_value(&runtime).is_none());
    }
}
#[test]
fn fixed_results_refuse_sixth_value_and_remain_inline() {
    let mut result = Results::empty();
    for _ in 0..5 {
        result.push(mask(0)).unwrap();
    }
    assert!(matches!(
        result.push(mask(0)),
        Err(SimulationExecutionErrorKindV1::InternalInvariant(
            "physical result bound"
        ))
    ));
    assert!(size_of::<Results>() <= 5 * size_of::<Option<RuntimeValue>>() + 2 * size_of::<usize>());
}
#[test]
fn zero_exec_store_skips_undefined_pointer_and_data_before_memory_validation() {
    with_engine(|engine| {
        let op = operation(
            Opcode::GlobalStoreDword,
            0,
            6,
            8,
            0,
            &[ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
        );
        let values = RuntimeValues::from([(ValueId(3), mask(0))]);
        physical_entry_v20::execute(engine, &values, &op, &site(5)).unwrap();
        assert_eq!(engine.memory.allocations[&7].bytes, vec![0xa5; 4]);
        assert_eq!(engine.memory.allocations[&7].initialized, vec![false; 4]);
        assert!(engine.accesses.is_empty());
        assert_eq!(engine.events, 0);
    });
}
#[test]
fn lane63_mask_bit_is_not_truncated_to_u32() {
    with_engine(|engine| {
        engine.invocation.as_mut().unwrap().local[0] = 63;
        let op = operation(
            Opcode::GlobalStoreDword,
            0,
            6,
            8,
            0,
            &[ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
        );
        let c = chain();
        let values = RuntimeValues::from([
            (ValueId(0), symbolic(Value::AddressLow(c.clone()))),
            (
                ValueId(1),
                symbolic(Value::AddressHigh {
                    chain: c,
                    high_add: site(4),
                }),
            ),
            (
                ValueId(2),
                RuntimeValue::Scalar(ScalarBitsV1::u32(0x12345678)),
            ),
            (ValueId(3), mask(1u64 << 63)),
        ]);
        physical_entry_v20::execute(engine, &values, &op, &site(5)).unwrap();
        assert_eq!(
            engine.memory.allocations[&7].bytes,
            0x12345678u32.to_le_bytes()
        );
        assert_eq!(engine.memory.allocations[&7].initialized, vec![true; 4]);
        assert_eq!(engine.accesses.len(), 4);
    });
}
#[test]
fn inactive_high_lane_skips_bad_pointer_even_when_low_lane_is_active() {
    with_engine(|engine| {
        engine.invocation.as_mut().unwrap().local[0] = 63;
        let op = operation(
            Opcode::GlobalStoreDword,
            0,
            6,
            8,
            0,
            &[ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
        );
        let values = RuntimeValues::from([(ValueId(3), mask(1))]);
        physical_entry_v20::execute(engine, &values, &op, &site(5)).unwrap();
        assert!(engine.accesses.is_empty());
        assert_eq!(engine.memory.allocations[&7].initialized, vec![false; 4]);
    });
}
#[test]
fn active_store_preserves_bounds_checks() {
    with_engine(|engine| {
        let op = operation(
            Opcode::GlobalStoreDword,
            0,
            6,
            8,
            0,
            &[ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
        );
        let mut c = chain();
        c.displacement = 4;
        let values = RuntimeValues::from([
            (ValueId(0), symbolic(Value::AddressLow(c.clone()))),
            (
                ValueId(1),
                symbolic(Value::AddressHigh {
                    chain: c,
                    high_add: site(4),
                }),
            ),
            (ValueId(2), RuntimeValue::Scalar(ScalarBitsV1::u32(7))),
            (ValueId(3), mask(1)),
        ]);
        assert!(physical_entry_v20::execute(engine, &values, &op, &site(5)).is_err());
        assert!(engine.accesses.is_empty());
        assert_eq!(engine.memory.allocations[&7].initialized, vec![false; 4]);
    });
}
#[test]
fn partial_exec_cannot_define_valu_without_passthrough() {
    with_engine(|engine| {
        let op = operation(
            Opcode::VectorAddU32,
            8,
            8,
            2,
            0,
            &[ValueId(0), ValueId(1), ValueId(2)],
        );
        let values = RuntimeValues::from([(ValueId(2), mask(0))]);
        let error = physical_entry_v20::execute(engine, &values, &op, &site(5))
            .err()
            .unwrap();
        assert_eq!(
            error.kind,
            SimulationExecutionErrorKindV1::InternalInvariant(
                "physical VALU requires full entry EXEC"
            )
        );
    });
}
#[test]
fn scalar_wait_and_restore_run_after_zero_exec_mask() {
    with_engine(|engine| {
        let save = operation(
            Opcode::SaveAndMaskExec,
            18,
            0,
            0,
            0,
            &[ValueId(0), ValueId(1)],
        );
        let mut values = RuntimeValues::from([(ValueId(0), mask(0)), (ValueId(1), mask(u64::MAX))]);
        let result = physical_entry_v20::execute(engine, &values, &save, &site(5)).unwrap();
        result
            .bind(engine, &mut values, &save.results, &site(5))
            .unwrap();
        assert_eq!(values.get(&ValueId(102)).unwrap(), mask(0));
        assert_eq!(
            values.get(&ValueId(103)).unwrap(),
            RuntimeValue::Scalar(ScalarBitsV1::boolean(false))
        );
        let wait = operation(Opcode::WaitVm0, 0, 0, 0, 0, &[]);
        physical_entry_v20::execute(engine, &values, &wait, &site(6)).unwrap();
        let restore = operation(
            Opcode::RestoreExec,
            0,
            18,
            0,
            0,
            &[ValueId(100), ValueId(101)],
        );
        let result = physical_entry_v20::execute(engine, &values, &restore, &site(7)).unwrap();
        let mut restored = RuntimeValues::new();
        result
            .bind(engine, &mut restored, &restore.results, &site(7))
            .unwrap();
        assert_eq!(restored.get(&ValueId(100)).unwrap(), mask(u64::MAX));
    });
}
