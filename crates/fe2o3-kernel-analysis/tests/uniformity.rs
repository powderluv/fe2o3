use fe2o3_kernel_analysis::{
    Diagnostic, UnsupportedReason, Variation, analyze_function, analyze_kernel_entry,
};
use fe2o3_kernel_ir::{
    AccessMode, AddressSpace, AmdGpuDiagnosticOperation, Axis, Barrier, BarrierSemantics,
    BasicBlock, BinaryOp, BlockId, CastKind, CheckedBinaryOperator, ComparePredicate, Constant,
    Convergence, F32MathFunction, FloatOperation, Function, FunctionId, IndexKind, IntrinsicKind,
    IntrinsicOperation, Kernel, LaunchDomain, LaunchExtent,
    MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1, MemoryAccess, MemoryOrdering, Module, Operation,
    OperationKind, ScalarType, Signature, SwitchCase, SynchronizationScope, Terminator, Type,
    ValueDef, ValueId, WaveOperation, WaveOperationKind, WaveWidth, WorkgroupBarrier,
    WorkgroupMemory, WorkgroupMemoryExtent, WorkgroupSize,
};

fn function(parameters: Vec<ValueId>, blocks: Vec<BasicBlock>) -> Function {
    Function::definition(
        "test",
        Signature::new(parameters.iter().map(|_| Type::INDEX).collect(), vec![]),
        parameters,
        blocks,
    )
}

fn constant(id: u32, value: Constant) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), value.ty()),
        OperationKind::Constant(value),
    )
}

fn intrinsic(id: u32, kind: IntrinsicKind) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::INDEX),
        OperationKind::Intrinsic(IntrinsicOperation::new(kind, Type::INDEX)),
    )
}

fn compare(id: u32, lhs: u32, rhs: u32) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::BOOL),
        OperationKind::Compare {
            predicate: ComparePredicate::NotEqual,
            lhs: ValueId(lhs),
            rhs: ValueId(rhs),
        },
    )
}

fn add(id: u32, lhs: u32, rhs: u32) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::INDEX),
        OperationKind::Binary {
            op: BinaryOp::Add,
            lhs: ValueId(lhs),
            rhs: ValueId(rhs),
        },
    )
}

fn binary(id: u32, op: BinaryOp, lhs: u32, rhs: u32) -> Operation {
    binary_with_type(id, op, lhs, rhs, Type::INDEX)
}

fn binary_with_type(id: u32, op: BinaryOp, lhs: u32, rhs: u32, ty: Type) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), ty),
        OperationKind::Binary {
            op,
            lhs: ValueId(lhs),
            rhs: ValueId(rhs),
        },
    )
}

fn cast(id: u32, kind: CastKind, value: u32, to: Type) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), to.clone()),
        OperationKind::Cast {
            kind,
            value: ValueId(value),
            to,
        },
    )
}

fn checked_index(
    value: u32,
    overflow: u32,
    operator: CheckedBinaryOperator,
    lhs: u32,
    rhs: u32,
) -> Operation {
    Operation::checked_binary(
        ValueDef::new(ValueId(value), Type::INDEX),
        ValueDef::new(ValueId(overflow), Type::BOOL),
        operator,
        ValueId(lhs),
        ValueId(rhs),
    )
}

fn compare_with(id: u32, predicate: ComparePredicate, lhs: u32, rhs: u32) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::BOOL),
        OperationKind::Compare {
            predicate,
            lhs: ValueId(lhs),
            rhs: ValueId(rhs),
        },
    )
}

fn zero_extend_to_index(id: u32, value: u32) -> Operation {
    cast(id, CastKind::ZeroExtend, value, Type::INDEX)
}

fn private_slot(id: u32) -> Operation {
    Operation::effect_free(
        ValueDef::new(
            ValueId(id),
            Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite),
        ),
        OperationKind::Alloca {
            element: Type::INDEX,
            count: None,
            address_space: AddressSpace::Private,
            alignment: 8,
        },
    )
}

fn private_store(pointer: u32, value: u32) -> Operation {
    Operation::new(
        vec![],
        OperationKind::Store {
            pointer: ValueId(pointer),
            value: ValueId(value),
            access: MemoryAccess::new(AddressSpace::Private, 8),
        },
    )
}

fn private_load(id: u32, pointer: u32) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::INDEX),
        OperationKind::Load {
            pointer: ValueId(pointer),
            access: MemoryAccess::new(AddressSpace::Private, 8),
        },
    )
}

fn analyze_as_kernel(function: &Function) -> fe2o3_kernel_analysis::AnalysisReport {
    let mut module = Module::new("uniformity_private_storage");
    module.functions.push(function.clone());
    analyze_kernel_entry(&module, function)
}

fn workgroup_barrier() -> Operation {
    Operation::new(
        vec![],
        OperationKind::Barrier(Barrier {
            execution_scope: SynchronizationScope::Workgroup,
            memory_scope: SynchronizationScope::Workgroup,
            semantics: BarrierSemantics::new(
                MemoryOrdering::AcquireRelease,
                [AddressSpace::Workgroup],
            ),
        }),
    )
}

fn explicit_workgroup_barrier() -> Operation {
    Operation::new(
        vec![],
        OperationKind::WorkgroupBarrier(WorkgroupBarrier {
            memory_scope: SynchronizationScope::Workgroup,
            semantics: BarrierSemantics::new(
                MemoryOrdering::AcquireRelease,
                [AddressSpace::Workgroup],
            ),
            convergence: Convergence::uniform(SynchronizationScope::Workgroup),
        }),
    )
}

fn wave(id: u32, kind: WaveOperationKind) -> Operation {
    Operation::effect_free(
        ValueDef::new(ValueId(id), Type::INDEX),
        OperationKind::Wave(WaveOperation::full(kind, WaveWidth::Wave64)),
    )
}

fn returning(id: u32) -> BasicBlock {
    let mut block = BasicBlock::new(BlockId(id));
    block.terminator = Some(Terminator::Return { values: vec![] });
    block
}

#[test]
fn bounded_lane_arithmetic_has_a_uniform_overflow_flag() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(64)),
        binary(2, BinaryOp::Remainder, 0, 1),
        constant(3, Constant::Index(16)),
        binary(4, BinaryOp::Divide, 2, 3),
        constant(5, Constant::Index(4)),
        checked_index(6, 7, CheckedBinaryOperator::Multiply, 4, 5),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(7),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut success = returning(2);
    success.operations.push(workgroup_barrier());

    let report = analyze_function(&function(vec![], vec![entry, returning(1), success]));

    assert_eq!(report.value(ValueId(6)), Variation::Varying);
    assert_eq!(report.value(ValueId(7)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(2)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn unconstrained_lane_arithmetic_keeps_overflow_varying() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(4)),
        checked_index(2, 3, CheckedBinaryOperator::Multiply, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(3),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut success = returning(2);
    success.operations.push(workgroup_barrier());

    let report = analyze_function(&function(vec![], vec![entry, returning(1), success]));

    assert_eq!(report.value(ValueId(2)), Variation::Varying);
    assert_eq!(report.value(ValueId(3)), Variation::Varying);
    assert!(matches!(
        report.diagnostics(),
        [Diagnostic::DivergentBarrier {
            block: BlockId(2),
            ..
        }]
    ));
}

#[test]
fn dominating_u32_guards_bound_index_multiply_and_add() {
    let parameters = vec![ValueId(0), ValueId(1), ValueId(2)];
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        zero_extend_to_index(3, 0),
        zero_extend_to_index(4, 1),
        zero_extend_to_index(5, 2),
        intrinsic(
            6,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        intrinsic(
            7,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        compare_with(8, ComparePredicate::LessThan, 6, 3),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(8),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(3),
        else_arguments: vec![],
    });
    let mut depth_guard = BasicBlock::new(BlockId(1));
    depth_guard
        .operations
        .push(compare_with(9, ComparePredicate::LessThan, 7, 5));
    depth_guard.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(9),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(3),
        else_arguments: vec![],
    });
    let mut arithmetic = returning(2);
    arithmetic.operations.extend([
        checked_index(10, 11, CheckedBinaryOperator::Multiply, 6, 4),
        checked_index(12, 13, CheckedBinaryOperator::Add, 10, 7),
    ]);
    let kernel = Function::definition(
        "guarded_index",
        Signature::new(
            vec![
                Type::Scalar(ScalarType::U32),
                Type::Scalar(ScalarType::U32),
                Type::Scalar(ScalarType::U32),
            ],
            vec![],
        ),
        parameters,
        vec![entry, depth_guard, arithmetic, returning(3)],
    );

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(10)), Variation::Varying);
    assert_eq!(report.value(ValueId(11)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(12)), Variation::Varying);
    assert_eq!(report.value(ValueId(13)), Variation::GridUniform);
}

#[test]
fn shared_branch_target_does_not_authenticate_a_range_guard() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(16)),
        compare_with(2, ComparePredicate::LessThan, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut shared_target = returning(1);
    shared_target.operations.extend([
        constant(3, Constant::Index(2)),
        checked_index(4, 5, CheckedBinaryOperator::Multiply, 0, 3),
    ]);
    let mut false_path = BasicBlock::new(BlockId(2));
    false_path.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let report = analyze_function(&function(vec![], vec![entry, shared_target, false_path]));

    assert_eq!(report.value(ValueId(5)), Variation::Varying);
}

#[test]
fn near_maximum_checked_addition_fails_closed() {
    let mut entry = returning(0);
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(1)),
        checked_index(2, 3, CheckedBinaryOperator::Add, 0, 1),
    ]);

    let report = analyze_function(&function(vec![], vec![entry]));

    assert_eq!(report.value(ValueId(3)), Variation::Varying);
}

#[test]
fn exhaustive_known_enum_switch_does_not_create_a_synthetic_divergent_exit() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare_with(2, ComparePredicate::NotEqual, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut zero = BasicBlock::new(BlockId(1));
    zero.operations.push(constant(3, Constant::I64(0)));
    zero.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(3)],
    });
    let mut one = BasicBlock::new(BlockId(2));
    one.operations.push(constant(4, Constant::I64(1)));
    one.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(4)],
    });
    let mut switch = BasicBlock::new(BlockId(3));
    switch
        .parameters
        .push(ValueDef::new(ValueId(5), Type::Scalar(ScalarType::I64)));
    switch.terminator = Some(Terminator::Switch {
        selector: ValueId(5),
        cases: vec![
            SwitchCase {
                value: 0,
                target: BlockId(4),
                arguments: vec![],
            },
            SwitchCase {
                value: 1,
                target: BlockId(5),
                arguments: vec![],
            },
        ],
        default_target: BlockId(6),
        default_arguments: vec![],
    });
    let mut case_zero = BasicBlock::new(BlockId(4));
    case_zero.terminator = Some(Terminator::Branch {
        target: BlockId(7),
        arguments: vec![],
    });
    let mut case_one = BasicBlock::new(BlockId(5));
    case_one.terminator = Some(Terminator::Branch {
        target: BlockId(7),
        arguments: vec![],
    });
    let mut merge = returning(7);
    merge.operations.push(workgroup_barrier());

    let report = analyze_function(&function(
        vec![],
        vec![
            entry,
            zero,
            one,
            switch,
            case_zero,
            case_one,
            returning(6),
            merge,
        ],
    ));

    assert_eq!(report.value(ValueId(5)), Variation::Varying);
    assert_eq!(report.block_control(BlockId(7)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn unknown_switch_selector_keeps_the_default_exit_and_fails_closed() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.push(intrinsic(
        0,
        IntrinsicKind::InvocationIndex {
            kind: IndexKind::Global,
            axis: Axis::X,
        },
    ));
    entry.terminator = Some(Terminator::Switch {
        selector: ValueId(0),
        cases: vec![
            SwitchCase {
                value: 0,
                target: BlockId(1),
                arguments: vec![],
            },
            SwitchCase {
                value: 1,
                target: BlockId(2),
                arguments: vec![],
            },
        ],
        default_target: BlockId(3),
        default_arguments: vec![],
    });
    let mut case_zero = BasicBlock::new(BlockId(1));
    case_zero.terminator = Some(Terminator::Branch {
        target: BlockId(4),
        arguments: vec![],
    });
    let mut case_one = BasicBlock::new(BlockId(2));
    case_one.terminator = Some(Terminator::Branch {
        target: BlockId(4),
        arguments: vec![],
    });
    let mut merge = returning(4);
    merge.operations.push(workgroup_barrier());

    let report = analyze_function(&function(
        vec![],
        vec![entry, case_zero, case_one, returning(3), merge],
    ));

    assert_eq!(report.block_control(BlockId(4)), Variation::Varying);
    assert!(matches!(
        report.diagnostics(),
        [Diagnostic::DivergentBarrier {
            block: BlockId(4),
            ..
        }]
    ));
}

#[test]
fn intrinsic_rules_cover_launch_hierarchy() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(0, IntrinsicKind::LaunchExtent { axis: Axis::X }),
        intrinsic(
            1,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Workgroup,
                axis: Axis::X,
            },
        ),
        intrinsic(
            2,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        intrinsic(
            4,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupSize,
                axis: Axis::X,
            },
        ),
        intrinsic(
            5,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::WorkgroupCount,
                axis: Axis::X,
            },
        ),
    ];

    let report = analyze_function(&function(vec![], vec![entry]));

    assert_eq!(report.value(ValueId(0)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(1)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(2)), Variation::Varying);
    assert_eq!(report.value(ValueId(3)), Variation::Varying);
    assert_eq!(report.value(ValueId(4)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(5)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn exact_global_index_workgroup_quotient_is_uniform_but_lane_remainder_is_not() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::U32(64)),
        cast(2, CastKind::ZeroExtend, 1, Type::INDEX),
        binary(3, BinaryOp::Divide, 0, 2),
        binary(4, BinaryOp::Remainder, 0, 2),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_workgroup_quotient");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(3)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(4)), Variation::Varying);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn exact_wg256_wave64_quotient_is_subgroup_uniform_but_lane_remainder_is_not() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::U32(64)),
        cast(2, CastKind::ZeroExtend, 1, Type::INDEX),
        binary(3, BinaryOp::Divide, 0, 2),
        binary(4, BinaryOp::Remainder, 0, 2),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(256, 1, 1));
    let mut module = Module::new("uniformity_wave64_quotient");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(3)), Variation::SubgroupUniform);
    assert_eq!(report.value(ValueId(4)), Variation::Varying);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn wave64_quotient_rule_rejects_other_divisors_axes_workgroups_and_absent_contracts() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        intrinsic(
            1,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::Y,
            },
        ),
        constant(2, Constant::Index(64)),
        constant(3, Constant::Index(32)),
        constant(4, Constant::Index(128)),
        binary(5, BinaryOp::Divide, 0, 2),
        binary(6, BinaryOp::Divide, 1, 2),
        binary(7, BinaryOp::Divide, 0, 3),
        binary(8, BinaryOp::Divide, 0, 4),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D2 {
            x: LaunchExtent::Dynamic,
            y: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(256, 1, 1));
    let mut module = Module::new("uniformity_wave64_quotient_adversarial");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(5)), Variation::SubgroupUniform);
    for value in 6..=8 {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }

    module.kernels[0].workgroup_size = Some(WorkgroupSize::new(192, 1, 1));
    assert_eq!(
        analyze_kernel_entry(&module, &function).value(ValueId(5)),
        Variation::Varying
    );
    module.kernels.clear();
    assert_eq!(
        analyze_kernel_entry(&module, &function).value(ValueId(5)),
        Variation::Varying
    );
}

#[test]
fn workgroup_quotient_uniformity_is_axis_exact_and_requires_launch_contract() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        intrinsic(
            1,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::Y,
            },
        ),
        intrinsic(
            2,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::Z,
            },
        ),
        constant(3, Constant::Index(8)),
        constant(4, Constant::Index(4)),
        constant(5, Constant::Index(2)),
        constant(6, Constant::Index(3)),
        binary(10, BinaryOp::Divide, 0, 3),
        binary(11, BinaryOp::Divide, 1, 4),
        binary(12, BinaryOp::Divide, 2, 5),
        binary(13, BinaryOp::Divide, 0, 6),
        binary(14, BinaryOp::Divide, 1, 3),
        binary(15, BinaryOp::Divide, 2, 4),
        binary(16, BinaryOp::Remainder, 0, 3),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D3 {
            x: LaunchExtent::Dynamic,
            y: LaunchExtent::Dynamic,
            z: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(8, 4, 2));
    let mut module = Module::new("uniformity_workgroup_quotient_axes");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    for value in 10..=12 {
        assert_eq!(report.value(ValueId(value)), Variation::WorkgroupUniform);
    }
    for value in 13..=16 {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }

    module.kernels.clear();
    let without_contract = analyze_kernel_entry(&module, &function);
    for value in 10..=12 {
        assert_eq!(without_contract.value(ValueId(value)), Variation::Varying);
    }
}

#[test]
fn workgroup_quotient_rejects_non_value_preserving_constant_casts() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::U64(300)),
        cast(2, CastKind::Truncate, 1, Type::Scalar(ScalarType::U8)),
        cast(3, CastKind::ZeroExtend, 2, Type::INDEX),
        binary(4, BinaryOp::Divide, 0, 3),
        constant(5, Constant::U64(256)),
        cast(6, CastKind::Truncate, 5, Type::Scalar(ScalarType::U8)),
        cast(7, CastKind::ZeroExtend, 6, Type::INDEX),
        binary(8, BinaryOp::Divide, 0, 7),
        constant(9, Constant::I32(64)),
        cast(10, CastKind::SignExtend, 9, Type::INDEX),
        binary(11, BinaryOp::Divide, 0, 10),
        constant(12, Constant::U64(64)),
        cast(13, CastKind::Bitcast, 12, Type::INDEX),
        binary(14, BinaryOp::Divide, 0, 13),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_hostile_constant_casts");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    for value in [4, 8, 11, 14] {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }
}

#[test]
fn workgroup_quotient_accepts_only_value_preserving_unsigned_cast_chains() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::U8(64)),
        cast(2, CastKind::ZeroExtend, 1, Type::Scalar(ScalarType::U16)),
        cast(3, CastKind::ZeroExtend, 2, Type::Scalar(ScalarType::U32)),
        cast(4, CastKind::ZeroExtend, 3, Type::INDEX),
        binary(5, BinaryOp::Divide, 0, 4),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_value_preserving_constant_casts");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    assert_eq!(
        analyze_kernel_entry(&module, &function).value(ValueId(5)),
        Variation::WorkgroupUniform,
    );
}

#[test]
fn workgroup_quotient_accepts_production_usize_bitcast_through_an_ssa_carrier() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        cast(1, CastKind::Bitcast, 0, Type::Scalar(ScalarType::U64)),
        constant(2, Constant::U64(64)),
        constant(3, Constant::U64(0)),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(1), ValueId(2)],
    });

    let mut carried = BasicBlock::new(BlockId(1));
    carried.parameters.extend([
        ValueDef::new(ValueId(10), Type::Scalar(ScalarType::U64)),
        ValueDef::new(ValueId(11), Type::Scalar(ScalarType::U64)),
    ]);
    carried.operations.extend([
        binary_with_type(12, BinaryOp::Divide, 10, 11, Type::Scalar(ScalarType::U64)),
        compare(13, 12, 3),
    ]);
    carried.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(13),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(3),
        else_arguments: vec![],
    });
    let mut barrier = returning(2);
    barrier.operations.push(workgroup_barrier());

    let function = function(vec![], vec![entry, carried, barrier, returning(3)]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_workgroup_quotient_usize_transport");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(10)), Variation::Varying);
    assert_eq!(report.value(ValueId(12)), Variation::WorkgroupUniform);
    assert_eq!(
        report.block_control(BlockId(2)),
        Variation::WorkgroupUniform
    );
    assert!(report.diagnostics().is_empty());
}

#[test]
fn workgroup_quotient_usize_transport_rejects_other_casts_and_malformed_types() {
    let global_x = IntrinsicKind::InvocationIndex {
        kind: IndexKind::Global,
        axis: Axis::X,
    };
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(0, global_x),
        constant(1, Constant::U64(64)),
        cast(2, CastKind::Bitcast, 0, Type::Scalar(ScalarType::U64)),
        binary_with_type(3, BinaryOp::Divide, 2, 1, Type::Scalar(ScalarType::U64)),
        cast(4, CastKind::ZeroExtend, 0, Type::Scalar(ScalarType::U64)),
        binary_with_type(5, BinaryOp::Divide, 4, 1, Type::Scalar(ScalarType::U64)),
        cast(6, CastKind::Bitcast, 0, Type::Scalar(ScalarType::I64)),
        constant(7, Constant::I64(64)),
        binary_with_type(8, BinaryOp::Divide, 6, 7, Type::Scalar(ScalarType::I64)),
        Operation::effect_free(
            ValueDef::new(ValueId(9), Type::INDEX),
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value: ValueId(0),
                to: Type::Scalar(ScalarType::U64),
            },
        ),
        constant(10, Constant::Index(64)),
        binary(11, BinaryOp::Divide, 9, 10),
        Operation::effect_free(
            ValueDef::new(ValueId(12), Type::Scalar(ScalarType::U64)),
            OperationKind::Intrinsic(IntrinsicOperation::new(global_x, Type::INDEX)),
        ),
        cast(13, CastKind::Bitcast, 12, Type::INDEX),
        binary(14, BinaryOp::Divide, 13, 10),
        Operation::effect_free(
            ValueDef::new(ValueId(15), Type::INDEX),
            OperationKind::Intrinsic(IntrinsicOperation::new(
                global_x,
                Type::Scalar(ScalarType::U64),
            )),
        ),
        binary(16, BinaryOp::Divide, 15, 10),
        Operation::new(
            vec![
                ValueDef::new(ValueId(17), Type::Scalar(ScalarType::U64)),
                ValueDef::new(ValueId(18), Type::Scalar(ScalarType::U64)),
            ],
            OperationKind::Cast {
                kind: CastKind::Bitcast,
                value: ValueId(0),
                to: Type::Scalar(ScalarType::U64),
            },
        ),
        binary_with_type(19, BinaryOp::Divide, 17, 1, Type::Scalar(ScalarType::U64)),
    ];
    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_workgroup_quotient_usize_transport_hostile");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(3)), Variation::WorkgroupUniform);
    for value in [5, 8, 11, 14, 16, 19] {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }
}

#[test]
fn workgroup_quotient_accepts_trivial_ssa_carriers_across_a_divergent_diamond() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::U8(64)),
        constant(2, Constant::Index(0)),
        compare(3, 0, 2),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(3),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(0), ValueId(1)],
        else_target: BlockId(2),
        else_arguments: vec![ValueId(0), ValueId(1)],
    });

    let mut left = BasicBlock::new(BlockId(1));
    left.parameters.extend([
        ValueDef::new(ValueId(10), Type::INDEX),
        ValueDef::new(ValueId(11), Type::Scalar(ScalarType::U8)),
    ]);
    left.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(10), ValueId(11)],
    });

    let mut right = BasicBlock::new(BlockId(2));
    right.parameters.extend([
        ValueDef::new(ValueId(12), Type::INDEX),
        ValueDef::new(ValueId(13), Type::Scalar(ScalarType::U8)),
    ]);
    right.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(12), ValueId(13)],
    });

    let mut merge = BasicBlock::new(BlockId(3));
    merge.parameters.extend([
        ValueDef::new(ValueId(20), Type::INDEX),
        ValueDef::new(ValueId(21), Type::Scalar(ScalarType::U8)),
    ]);
    merge.operations.extend([
        zero_extend_to_index(22, 21),
        binary(23, BinaryOp::Divide, 20, 22),
        compare(24, 23, 2),
    ]);
    merge.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(24),
        then_target: BlockId(4),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![],
    });
    let mut barrier = returning(4);
    barrier.operations.push(workgroup_barrier());

    let function = function(
        vec![],
        vec![entry, left, right, merge, barrier, returning(5)],
    );
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_workgroup_quotient_ssa_diamond");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(20)), Variation::Varying);
    assert_eq!(report.value(ValueId(21)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(23)), Variation::WorkgroupUniform);
    assert_eq!(
        report.block_control(BlockId(4)),
        Variation::WorkgroupUniform
    );
    assert!(report.diagnostics().is_empty());
}

#[test]
fn wave64_quotient_accepts_direct_self_loop_ssa_carriers() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(64)),
        constant(2, Constant::Bool(false)),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(0), ValueId(1)],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header.parameters.extend([
        ValueDef::new(ValueId(10), Type::INDEX),
        ValueDef::new(ValueId(11), Type::INDEX),
    ]);
    header.operations.push(binary(12, BinaryOp::Divide, 10, 11));
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(10), ValueId(11)],
        else_target: BlockId(2),
        else_arguments: vec![],
    });

    let function = function(vec![], vec![entry, header, returning(2)]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(256, 1, 1));
    let mut module = Module::new("uniformity_wave64_quotient_ssa_self_loop");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(12)), Variation::SubgroupUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn quotient_ssa_carriers_reject_different_origins_wrong_axes_and_wrong_divisors() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        intrinsic(
            1,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::Y,
            },
        ),
        constant(2, Constant::Index(64)),
        constant(3, Constant::Index(32)),
        constant(4, Constant::Index(0)),
        compare(5, 0, 4),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(5),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(0), ValueId(2), ValueId(0), ValueId(3), ValueId(1)],
        else_target: BlockId(2),
        else_arguments: vec![ValueId(1), ValueId(2), ValueId(0), ValueId(3), ValueId(1)],
    });

    let branch_parameters = || {
        vec![
            ValueDef::new(ValueId(10), Type::INDEX),
            ValueDef::new(ValueId(11), Type::INDEX),
            ValueDef::new(ValueId(12), Type::INDEX),
            ValueDef::new(ValueId(13), Type::INDEX),
            ValueDef::new(ValueId(14), Type::INDEX),
        ]
    };
    let mut left = BasicBlock::new(BlockId(1));
    left.parameters = branch_parameters();
    left.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![
            ValueId(10),
            ValueId(11),
            ValueId(12),
            ValueId(13),
            ValueId(14),
        ],
    });
    let mut right = BasicBlock::new(BlockId(2));
    right.parameters = vec![
        ValueDef::new(ValueId(15), Type::INDEX),
        ValueDef::new(ValueId(16), Type::INDEX),
        ValueDef::new(ValueId(17), Type::INDEX),
        ValueDef::new(ValueId(18), Type::INDEX),
        ValueDef::new(ValueId(19), Type::INDEX),
    ];
    right.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![
            ValueId(15),
            ValueId(16),
            ValueId(17),
            ValueId(18),
            ValueId(19),
        ],
    });

    let mut merge = BasicBlock::new(BlockId(3));
    merge.parameters = (20..=24)
        .map(|value| ValueDef::new(ValueId(value), Type::INDEX))
        .collect();
    merge.operations.extend([
        binary(25, BinaryOp::Divide, 20, 21),
        binary(26, BinaryOp::Divide, 22, 23),
        binary(27, BinaryOp::Divide, 24, 21),
        compare(28, 25, 4),
    ]);
    merge.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(28),
        then_target: BlockId(4),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![],
    });
    let mut barrier = returning(4);
    barrier.operations.push(workgroup_barrier());

    let function = function(
        vec![],
        vec![entry, left, right, merge, barrier, returning(5)],
    );
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D2 {
            x: LaunchExtent::Dynamic,
            y: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_quotient_ssa_hostile");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    for value in 25..=27 {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }
    assert_eq!(report.block_control(BlockId(4)), Variation::Varying);
    assert!(report.diagnostics().iter().any(|diagnostic| matches!(
        diagnostic,
        Diagnostic::DivergentBarrier {
            block: BlockId(4),
            ..
        }
    )));
}

#[test]
fn quotient_ssa_carriers_reject_malformed_incoming_edge_arities() {
    for (module_name, arguments) in [
        ("uniformity_quotient_ssa_short_edge", vec![ValueId(0)]),
        (
            "uniformity_quotient_ssa_extra_edge",
            vec![ValueId(0), ValueId(1), ValueId(1)],
        ),
    ] {
        let mut entry = BasicBlock::new(BlockId(0));
        entry.operations.extend([
            intrinsic(
                0,
                IntrinsicKind::InvocationIndex {
                    kind: IndexKind::Global,
                    axis: Axis::X,
                },
            ),
            constant(1, Constant::Index(64)),
        ]);
        entry.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments,
        });

        let mut carried = BasicBlock::new(BlockId(1));
        carried.parameters.extend([
            ValueDef::new(ValueId(10), Type::INDEX),
            ValueDef::new(ValueId(11), Type::INDEX),
        ]);
        carried
            .operations
            .push(binary(12, BinaryOp::Divide, 10, 11));
        carried.terminator = Some(Terminator::Return { values: vec![] });

        let function = function(vec![], vec![entry, carried]);
        let mut kernel = Kernel::new(
            "test_kernel",
            function.id.clone(),
            LaunchDomain::D1 {
                x: LaunchExtent::Dynamic,
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
        let mut module = Module::new(module_name);
        module.functions.push(function.clone());
        module.kernels.push(kernel);

        let report = analyze_kernel_entry(&module, &function);
        assert_eq!(report.value(ValueId(12)), Variation::Varying);
        assert!(report.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            Diagnostic::Unsupported {
                reason: UnsupportedReason::MalformedControlFlow,
                ..
            }
        )));
    }
}

#[test]
fn quotient_ssa_carriers_reject_index_u64_type_confusion() {
    for (module_name, carrier_type, global_value, divisor_value) in [
        (
            "uniformity_quotient_ssa_index_to_u64_edge",
            Type::Scalar(ScalarType::U64),
            ValueId(0),
            ValueId(1),
        ),
        (
            "uniformity_quotient_ssa_u64_to_index_edge",
            Type::INDEX,
            ValueId(2),
            ValueId(3),
        ),
    ] {
        let mut entry = BasicBlock::new(BlockId(0));
        entry.operations.extend([
            intrinsic(
                0,
                IntrinsicKind::InvocationIndex {
                    kind: IndexKind::Global,
                    axis: Axis::X,
                },
            ),
            constant(1, Constant::Index(64)),
            cast(2, CastKind::Bitcast, 0, Type::Scalar(ScalarType::U64)),
            constant(3, Constant::U64(64)),
        ]);
        entry.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments: vec![global_value, divisor_value],
        });

        let mut carried = BasicBlock::new(BlockId(1));
        carried.parameters.extend([
            ValueDef::new(ValueId(10), carrier_type.clone()),
            ValueDef::new(ValueId(11), carrier_type.clone()),
        ]);
        carried
            .operations
            .push(binary_with_type(12, BinaryOp::Divide, 10, 11, carrier_type));
        carried.terminator = Some(Terminator::Return { values: vec![] });

        let function = function(vec![], vec![entry, carried]);
        let mut kernel = Kernel::new(
            "test_kernel",
            function.id.clone(),
            LaunchDomain::D1 {
                x: LaunchExtent::Dynamic,
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
        let mut module = Module::new(module_name);
        module.functions.push(function.clone());
        module.kernels.push(kernel);

        let report = analyze_kernel_entry(&module, &function);
        assert_eq!(report.value(ValueId(12)), Variation::Varying);
        assert!(report.diagnostics().iter().any(|diagnostic| matches!(
            diagnostic,
            Diagnostic::Unsupported {
                reason: UnsupportedReason::MalformedControlFlow,
                ..
            }
        )));
    }
}

#[test]
fn cfg_edges_validate_function_parameter_types() {
    for (carrier_type, expected, malformed) in [
        (Type::Scalar(ScalarType::U64), Variation::GridUniform, false),
        (Type::INDEX, Variation::Varying, true),
    ] {
        let mut entry = BasicBlock::new(BlockId(0));
        entry.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(0)],
        });
        let mut carried = BasicBlock::new(BlockId(1));
        carried
            .parameters
            .push(ValueDef::new(ValueId(10), carrier_type));
        carried.terminator = Some(Terminator::Return { values: vec![] });
        let kernel = Function::definition(
            "test",
            Signature::new(vec![Type::Scalar(ScalarType::U64)], vec![]),
            vec![ValueId(0)],
            vec![entry, carried],
        );

        let report = analyze_as_kernel(&kernel);
        assert_eq!(report.value(ValueId(10)), expected);
        assert_eq!(
            report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::Unsupported {
                    reason: UnsupportedReason::MalformedControlFlow,
                    ..
                }
            )),
            malformed,
        );
    }
}

#[test]
fn quotient_constant_proof_authenticates_the_exact_cast_source_result() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        Operation::new(
            vec![
                ValueDef::new(ValueId(1), Type::Scalar(ScalarType::U8)),
                ValueDef::new(ValueId(2), Type::Scalar(ScalarType::I8)),
            ],
            OperationKind::Constant(Constant::U8(64)),
        ),
        zero_extend_to_index(3, 2),
        binary(4, BinaryOp::Divide, 0, 3),
    ]);
    entry.terminator = Some(Terminator::Return { values: vec![] });

    let function = function(vec![], vec![entry]);
    let mut kernel = Kernel::new(
        "test_kernel",
        function.id.clone(),
        LaunchDomain::D1 {
            x: LaunchExtent::Dynamic,
        },
    );
    kernel.workgroup_size = Some(WorkgroupSize::new(64, 1, 1));
    let mut module = Module::new("uniformity_quotient_exact_constant_result");
    module.functions.push(function.clone());
    module.kernels.push(kernel);

    let report = analyze_kernel_entry(&module, &function);
    assert_eq!(report.value(ValueId(4)), Variation::Varying);
}

#[test]
fn workgroup_contract_selection_rejects_conflicting_duplicate_entries() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(64)),
        binary(2, BinaryOp::Divide, 0, 1),
    ];
    let function = function(vec![], vec![entry]);
    let kernel = |name: &str, width| {
        let mut kernel = Kernel::new(
            name,
            function.id.clone(),
            LaunchDomain::D1 {
                x: LaunchExtent::Dynamic,
            },
        );
        kernel.workgroup_size = Some(WorkgroupSize::new(width, 1, 1));
        kernel
    };

    for widths in [[32, 64], [64, 32]] {
        let mut module = Module::new("uniformity_conflicting_duplicate_entries");
        module.functions.push(function.clone());
        module.kernels.push(kernel("first", widths[0]));
        module.kernels.push(kernel("second", widths[1]));
        assert_eq!(
            analyze_kernel_entry(&module, &function).value(ValueId(2)),
            Variation::Varying,
        );
    }

    let mut identical = Module::new("uniformity_identical_duplicate_entries");
    identical.functions.push(function.clone());
    identical.kernels.push(kernel("first", 64));
    identical.kernels.push(kernel("second", 64));
    assert_eq!(
        analyze_kernel_entry(&identical, &function).value(ValueId(2)),
        Variation::WorkgroupUniform,
    );
}

#[test]
fn wave_rules_preserve_uniform_inputs_and_collapse_full_wave_collectives() {
    let mut entry = returning(0);
    entry.operations = vec![
        constant(2, Constant::Bool(true)),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Workgroup,
                axis: Axis::X,
            },
        ),
        constant(4, Constant::Index(0)),
        compare(5, 3, 4),
        wave(6, WaveOperationKind::LaneId),
        wave(
            7,
            WaveOperationKind::Ballot {
                predicate: ValueId(0),
            },
        ),
        wave(
            8,
            WaveOperationKind::Any {
                predicate: ValueId(2),
            },
        ),
        wave(
            9,
            WaveOperationKind::All {
                predicate: ValueId(5),
            },
        ),
        wave(
            10,
            WaveOperationKind::ShuffleIndex {
                value: ValueId(0),
                source_lane: ValueId(4),
                tile_width: 64,
            },
        ),
        wave(
            11,
            WaveOperationKind::ShuffleIndex {
                value: ValueId(4),
                source_lane: ValueId(1),
                tile_width: 64,
            },
        ),
        wave(
            12,
            WaveOperationKind::ShuffleIndex {
                value: ValueId(0),
                source_lane: ValueId(1),
                tile_width: 64,
            },
        ),
    ];

    let report = analyze_function(&function(vec![ValueId(0), ValueId(1)], vec![entry]));

    assert_eq!(report.value(ValueId(6)), Variation::Varying);
    assert_eq!(report.value(ValueId(7)), Variation::SubgroupUniform);
    assert_eq!(report.value(ValueId(8)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(9)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(10)), Variation::SubgroupUniform);
    assert_eq!(report.value(ValueId(11)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(12)), Variation::Varying);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn arithmetic_comparison_and_memory_rules_propagate_conservatively() {
    let mut entry = returning(0);
    entry.operations = vec![
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Workgroup,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(1)),
        Operation::effect_free(
            ValueDef::new(ValueId(2), Type::INDEX),
            OperationKind::Binary {
                op: fe2o3_kernel_ir::BinaryOp::Add,
                lhs: ValueId(0),
                rhs: ValueId(1),
            },
        ),
        compare(3, 2, 1),
        Operation::effect_free(
            ValueDef::new(ValueId(4), Type::INDEX),
            OperationKind::Load {
                pointer: ValueId(1),
                access: MemoryAccess::new(AddressSpace::Global, 8),
            },
        ),
        Operation::effect_free(
            ValueDef::new(
                ValueId(5),
                Type::pointer(Type::INDEX, AddressSpace::Workgroup, AccessMode::ReadWrite),
            ),
            OperationKind::Alloca {
                element: Type::INDEX,
                count: None,
                address_space: AddressSpace::Workgroup,
                alignment: 8,
            },
        ),
        Operation::effect_free(
            ValueDef::new(
                ValueId(6),
                Type::pointer(Type::INDEX, AddressSpace::Workgroup, AccessMode::ReadWrite),
            ),
            OperationKind::Alloca {
                element: Type::INDEX,
                count: Some(ValueId(4)),
                address_space: AddressSpace::Workgroup,
                alignment: 8,
            },
        ),
    ];

    let report = analyze_function(&function(vec![], vec![entry]));

    assert_eq!(report.value(ValueId(2)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(3)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(4)), Variation::Varying);
    assert_eq!(report.value(ValueId(5)), Variation::WorkgroupUniform);
    assert_eq!(report.value(ValueId(6)), Variation::Varying);
}

#[test]
fn dominated_private_slot_round_trip_preserves_kernel_parameter_uniformity() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        private_slot(1),
        private_store(1, 0),
        private_load(2, 1),
        constant(3, Constant::Index(0)),
        compare(4, 2, 3),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    then_block.operations.push(workgroup_barrier());
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(2),
        arguments: vec![],
    });
    let kernel = function(vec![ValueId(0)], vec![entry, then_block, returning(2)]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(2)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(1)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn exact_private_slot_carriers_preserve_uniformity_across_a_divergent_diamond() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        private_slot(1),
        private_store(1, 0),
        intrinsic(
            2,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(3, Constant::Index(0)),
        compare(4, 2, 3),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(1)],
        else_target: BlockId(2),
        else_arguments: vec![ValueId(1)],
    });

    let mut left = BasicBlock::new(BlockId(1));
    left.parameters
        .push(ValueDef::new(ValueId(10), pointer_type.clone()));
    left.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(10)],
    });
    let mut right = BasicBlock::new(BlockId(2));
    right
        .parameters
        .push(ValueDef::new(ValueId(11), pointer_type.clone()));
    right.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(11)],
    });

    let mut merge = BasicBlock::new(BlockId(3));
    merge
        .parameters
        .push(ValueDef::new(ValueId(12), pointer_type));
    merge.operations.extend([
        private_load(13, 12),
        constant(14, Constant::Index(0)),
        compare(15, 13, 14),
    ]);
    merge.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(15),
        then_target: BlockId(4),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![],
    });
    let mut barrier = returning(4);
    barrier.operations.push(workgroup_barrier());
    let kernel = function(
        vec![ValueId(0)],
        vec![entry, left, right, merge, barrier, returning(5)],
    );

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(13)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(4)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn exact_private_slot_carriers_preserve_uniformity_across_a_self_loop() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        private_slot(1),
        private_store(1, 0),
        constant(2, Constant::Bool(false)),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(1)],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header
        .parameters
        .push(ValueDef::new(ValueId(10), pointer_type));
    header
        .operations
        .extend([private_load(11, 10), workgroup_barrier()]);
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(10)],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let kernel = function(vec![ValueId(0)], vec![entry, header, returning(2)]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(11)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(1)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn private_slot_carriers_require_exact_types_across_the_entire_chain() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    let mismatched_pointer_type = Type::pointer(
        Type::Scalar(ScalarType::U64),
        AddressSpace::Private,
        AccessMode::ReadWrite,
    );

    for (middle_type, expected, malformed) in [
        (pointer_type.clone(), Variation::GridUniform, false),
        (mismatched_pointer_type, Variation::Varying, true),
    ] {
        let mut entry = BasicBlock::new(BlockId(0));
        entry.operations.extend([
            constant(0, Constant::Index(7)),
            private_slot(1),
            private_store(1, 0),
        ]);
        entry.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(1)],
        });

        let mut middle = BasicBlock::new(BlockId(1));
        middle
            .parameters
            .push(ValueDef::new(ValueId(10), middle_type));
        middle.terminator = Some(Terminator::Branch {
            target: BlockId(2),
            arguments: vec![ValueId(10)],
        });

        let mut load = BasicBlock::new(BlockId(2));
        load.parameters
            .push(ValueDef::new(ValueId(20), pointer_type.clone()));
        load.operations.push(private_load(21, 20));
        load.terminator = Some(Terminator::Return { values: vec![] });

        let kernel = function(vec![], vec![entry, middle, load]);
        let report = analyze_as_kernel(&kernel);
        assert_eq!(report.value(ValueId(21)), expected);
        assert_eq!(
            report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::Unsupported {
                    reason: UnsupportedReason::MalformedControlFlow,
                    ..
                }
            )),
            malformed,
        );
    }
}

#[test]
fn mixed_private_slot_carriers_do_not_mint_uniform_loads() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        private_slot(1),
        private_slot(2),
        private_store(1, 0),
        private_store(2, 0),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(4, Constant::Index(0)),
        compare(5, 3, 4),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(5),
        then_target: BlockId(1),
        then_arguments: vec![ValueId(1)],
        else_target: BlockId(1),
        else_arguments: vec![ValueId(2)],
    });
    let mut merge = BasicBlock::new(BlockId(1));
    merge
        .parameters
        .push(ValueDef::new(ValueId(10), pointer_type));
    merge.operations.push(private_load(11, 10));
    merge.terminator = Some(Terminator::Return { values: vec![] });
    let kernel = function(vec![ValueId(0)], vec![entry, merge]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(11)), Variation::Varying);
}

#[derive(Clone, Copy, Debug)]
enum PrivateCarrierEscapeCase {
    Call,
    GetElementPointer,
    Return,
    StoreAsValue,
    MalformedTransport,
}

#[test]
fn private_slot_carriers_escape_through_every_non_transport_use() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    for case in [
        PrivateCarrierEscapeCase::Call,
        PrivateCarrierEscapeCase::GetElementPointer,
        PrivateCarrierEscapeCase::Return,
        PrivateCarrierEscapeCase::StoreAsValue,
        PrivateCarrierEscapeCase::MalformedTransport,
    ] {
        let mut entry = BasicBlock::new(BlockId(0));
        entry.operations.extend([
            constant(0, Constant::Index(7)),
            private_slot(1),
            private_store(1, 0),
        ]);
        entry.terminator = Some(Terminator::Branch {
            target: BlockId(1),
            arguments: if matches!(case, PrivateCarrierEscapeCase::MalformedTransport) {
                vec![ValueId(1), ValueId(1)]
            } else {
                vec![ValueId(1)]
            },
        });

        let mut carried = BasicBlock::new(BlockId(1));
        carried
            .parameters
            .push(ValueDef::new(ValueId(10), pointer_type.clone()));
        match case {
            PrivateCarrierEscapeCase::Call => carried.operations.push(Operation::new(
                vec![],
                OperationKind::Call {
                    callee: FunctionId::new("opaque_private_pointer_user"),
                    arguments: vec![ValueId(10)],
                },
            )),
            PrivateCarrierEscapeCase::GetElementPointer => {
                carried.operations.extend([
                    constant(21, Constant::Index(0)),
                    Operation::effect_free(
                        ValueDef::new(ValueId(22), pointer_type.clone()),
                        OperationKind::GetElementPointer {
                            base: ValueId(10),
                            offset: ValueId(21),
                        },
                    ),
                ]);
            }
            PrivateCarrierEscapeCase::StoreAsValue => {
                carried.operations.extend([
                    Operation::effect_free(
                        ValueDef::new(
                            ValueId(21),
                            Type::pointer(
                                pointer_type.clone(),
                                AddressSpace::Private,
                                AccessMode::ReadWrite,
                            ),
                        ),
                        OperationKind::Alloca {
                            element: pointer_type.clone(),
                            count: None,
                            address_space: AddressSpace::Private,
                            alignment: 8,
                        },
                    ),
                    Operation::new(
                        vec![],
                        OperationKind::Store {
                            pointer: ValueId(21),
                            value: ValueId(10),
                            access: MemoryAccess::new(AddressSpace::Private, 8),
                        },
                    ),
                ]);
            }
            PrivateCarrierEscapeCase::Return | PrivateCarrierEscapeCase::MalformedTransport => {}
        }
        carried.operations.push(private_load(20, 10));
        carried.terminator = Some(if matches!(case, PrivateCarrierEscapeCase::Return) {
            Terminator::Return {
                values: vec![ValueId(10)],
            }
        } else {
            Terminator::Return { values: vec![] }
        });
        let result_types = matches!(case, PrivateCarrierEscapeCase::Return)
            .then(|| vec![pointer_type.clone()])
            .unwrap_or_default();
        let kernel = Function::definition(
            "test",
            Signature::new(vec![], result_types),
            vec![],
            vec![entry, carried],
        );

        let report = analyze_as_kernel(&kernel);

        assert_eq!(
            report.value(ValueId(20)),
            Variation::Varying,
            "case {case:?} retained a private-slot summary after escape"
        );
        if matches!(case, PrivateCarrierEscapeCase::MalformedTransport) {
            assert!(report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::Unsupported {
                    reason: UnsupportedReason::MalformedControlFlow,
                    ..
                }
            )));
        }
    }
}

#[test]
fn private_slot_round_trip_does_not_make_lane_values_uniform() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        private_slot(0),
        intrinsic(
            1,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        private_store(0, 1),
        private_load(2, 0),
        constant(3, Constant::Index(0)),
        compare(4, 2, 3),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    then_block.operations.push(workgroup_barrier());
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(2),
        arguments: vec![],
    });
    let kernel = function(vec![], vec![entry, then_block, returning(2)]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(2)), Variation::Varying);
    assert!(matches!(
        report.diagnostics(),
        [Diagnostic::DivergentBarrier {
            block: BlockId(1),
            ..
        }]
    ));
}

#[test]
fn private_load_without_a_dominating_store_fails_closed() {
    let mut entry = returning(0);
    entry.operations.extend([
        private_slot(0),
        constant(1, Constant::Index(7)),
        private_load(2, 0),
        private_store(0, 1),
    ]);
    let kernel = function(vec![], vec![entry]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(2)), Variation::Varying);
}

#[test]
fn escaped_private_slot_load_fails_closed() {
    let pointer_type = Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite);
    let mut entry = returning(0);
    entry.operations.extend([
        private_slot(0),
        constant(1, Constant::Index(7)),
        Operation::effect_free(
            ValueDef::new(ValueId(2), pointer_type.clone()),
            OperationKind::Cast {
                kind: fe2o3_kernel_ir::CastKind::Bitcast,
                value: ValueId(0),
                to: pointer_type,
            },
        ),
        private_store(0, 1),
        private_load(3, 0),
    ]);
    let kernel = function(vec![], vec![entry]);

    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(3)), Variation::Varying);
}

#[test]
fn uniform_branch_allows_workgroup_barrier() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.push(constant(0, Constant::Bool(true)));
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(0),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    then_block.operations.push(workgroup_barrier());
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut else_block = BasicBlock::new(BlockId(2));
    else_block.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });

    let report = analyze_function(&function(
        vec![],
        vec![entry, then_block, else_block, returning(3)],
    ));

    assert_eq!(report.block_control(BlockId(1)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn thread_varying_branch_rejects_workgroup_barrier() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    then_block
        .operations
        .extend([workgroup_barrier(), explicit_workgroup_barrier()]);
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(2),
        arguments: vec![],
    });

    let report = analyze_function(&function(vec![], vec![entry, then_block, returning(2)]));

    assert_eq!(report.block_control(BlockId(1)), Variation::Varying);
    assert_eq!(
        report.diagnostics(),
        &[
            Diagnostic::DivergentBarrier {
                block: BlockId(1),
                operation_index: 0,
                execution_scope: SynchronizationScope::Workgroup,
                control: Variation::Varying,
            },
            Diagnostic::DivergentBarrier {
                block: BlockId(1),
                operation_index: 1,
                execution_scope: SynchronizationScope::Workgroup,
                control: Variation::Varying,
            },
        ]
    );
}

#[test]
fn explicit_lds_pointer_is_workgroup_uniform() {
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(
            ValueId(0),
            Type::pointer(Type::F32, AddressSpace::Workgroup, AccessMode::ReadWrite),
        ),
        OperationKind::WorkgroupMemory(WorkgroupMemory {
            element: Type::F32,
            extent: WorkgroupMemoryExtent::Static(64),
            alignment: 4,
        }),
    ));

    let report = analyze_function(&function(vec![], vec![entry]));
    assert_eq!(report.value(ValueId(0)), Variation::WorkgroupUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn nested_control_retains_outer_divergence() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(4),
        else_arguments: vec![],
    });
    let mut outer = BasicBlock::new(BlockId(1));
    outer.operations.push(constant(3, Constant::Bool(true)));
    outer.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(3),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(3),
        else_arguments: vec![],
    });
    let mut inner = BasicBlock::new(BlockId(2));
    inner.operations.push(workgroup_barrier());
    inner.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut inner_merge = BasicBlock::new(BlockId(3));
    inner_merge.terminator = Some(Terminator::Branch {
        target: BlockId(4),
        arguments: vec![],
    });

    let report = analyze_function(&function(
        vec![],
        vec![entry, outer, inner, inner_merge, returning(4)],
    ));

    assert_eq!(report.block_control(BlockId(2)), Variation::Varying);
    assert!(matches!(
        report.diagnostics(),
        [Diagnostic::DivergentBarrier {
            block: BlockId(2),
            ..
        }]
    ));
}

#[test]
fn varying_exitless_loop_fails_closed() {
    let mut header = BasicBlock::new(BlockId(0));
    header.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(0),
        else_arguments: vec![],
    });
    let mut body = BasicBlock::new(BlockId(1));
    body.operations.push(workgroup_barrier());
    body.terminator = Some(Terminator::Branch {
        target: BlockId(0),
        arguments: vec![],
    });

    let report = analyze_function(&function(vec![], vec![header, body]));

    assert_eq!(report.block_control(BlockId(1)), Variation::Varying);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: None,
        operation_index: None,
        reason: UnsupportedReason::PostdominanceUnavailable {
            blocks: vec![BlockId(0), BlockId(1)],
        },
    }));
    assert!(
        report
            .diagnostics()
            .contains(&Diagnostic::DivergentBarrier {
                block: BlockId(1),
                operation_index: 0,
                execution_scope: SynchronizationScope::Workgroup,
                control: Variation::Varying,
            })
    );
}

#[test]
fn divergent_loop_exit_does_not_manufacture_reconvergence() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(3),
        else_arguments: vec![],
    });
    let mut body = BasicBlock::new(BlockId(2));
    body.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });
    let mut exit = returning(3);
    exit.operations.push(workgroup_barrier());

    let report = analyze_function(&function(vec![], vec![entry, header, body, exit]));

    assert_eq!(report.block_control(BlockId(3)), Variation::Varying);
    assert!(matches!(
        report.diagnostics(),
        [Diagnostic::DivergentBarrier {
            block: BlockId(3),
            ..
        }]
    ));
}

#[test]
fn divergent_branch_can_reconverge_within_the_same_natural_loop() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
        constant(3, Constant::Bool(true)),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(3),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(6),
        else_arguments: vec![],
    });
    let mut body = BasicBlock::new(BlockId(2));
    body.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(3),
        then_arguments: vec![],
        else_target: BlockId(4),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(3));
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(5),
        arguments: vec![],
    });
    let mut else_block = BasicBlock::new(BlockId(4));
    else_block.terminator = Some(Terminator::Branch {
        target: BlockId(5),
        arguments: vec![],
    });
    let mut merge = BasicBlock::new(BlockId(5));
    merge.operations.push(workgroup_barrier());
    merge.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let report = analyze_function(&function(
        vec![],
        vec![
            entry,
            header,
            body,
            then_block,
            else_block,
            merge,
            returning(6),
        ],
    ));

    assert_eq!(report.block_control(BlockId(5)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn reconverged_merge_clears_control_but_not_phi_variation() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut then_block = BasicBlock::new(BlockId(1));
    then_block.operations.push(constant(3, Constant::Index(10)));
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(3)],
    });
    let mut else_block = BasicBlock::new(BlockId(2));
    else_block.operations.push(constant(4, Constant::Index(20)));
    else_block.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![ValueId(4)],
    });
    let mut merge = returning(3);
    merge
        .parameters
        .push(ValueDef::new(ValueId(5), Type::INDEX));
    merge.operations.push(workgroup_barrier());

    let report = analyze_function(&function(
        vec![],
        vec![entry, then_block, else_block, merge],
    ));

    assert_eq!(report.block_control(BlockId(3)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(5)), Variation::Varying);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn reconverged_loop_preserves_trivial_phi_uniformity() {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        constant(0, Constant::Index(0)),
        constant(1, Constant::Index(3)),
        constant(2, Constant::Index(1)),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        compare(4, 3, 0),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(0)],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header
        .parameters
        .push(ValueDef::new(ValueId(10), Type::INDEX));
    header.operations.push(compare(11, 10, 1));
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(11),
        then_target: BlockId(2),
        then_arguments: vec![ValueId(10)],
        else_target: BlockId(6),
        else_arguments: vec![],
    });

    let mut body = BasicBlock::new(BlockId(2));
    body.parameters
        .push(ValueDef::new(ValueId(12), Type::INDEX));
    body.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(3),
        then_arguments: vec![ValueId(12)],
        else_target: BlockId(4),
        else_arguments: vec![ValueId(12)],
    });

    let mut then_block = BasicBlock::new(BlockId(3));
    then_block
        .parameters
        .push(ValueDef::new(ValueId(13), Type::INDEX));
    then_block.terminator = Some(Terminator::Branch {
        target: BlockId(5),
        arguments: vec![ValueId(13)],
    });

    let mut else_block = BasicBlock::new(BlockId(4));
    else_block
        .parameters
        .push(ValueDef::new(ValueId(14), Type::INDEX));
    else_block.terminator = Some(Terminator::Branch {
        target: BlockId(5),
        arguments: vec![ValueId(14)],
    });

    let mut merge = BasicBlock::new(BlockId(5));
    merge
        .parameters
        .push(ValueDef::new(ValueId(15), Type::INDEX));
    merge
        .operations
        .extend([workgroup_barrier(), add(16, 15, 2)]);
    merge.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(16)],
    });

    let report = analyze_function(&function(
        vec![],
        vec![
            entry,
            header,
            body,
            then_block,
            else_block,
            merge,
            returning(6),
        ],
    ));

    assert_eq!(report.value(ValueId(10)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(15)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(5)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[derive(Clone, Copy, Debug)]
enum CheckedRecurrenceCase {
    Exact,
    WrongPhi,
    VaryingStep,
    UncheckedAdd,
    MismatchedFlag,
    InvertedOverflowEdge,
    DivergentTripCount,
    DivergentEntry,
}

fn checked_recurrence(case: CheckedRecurrenceCase) -> Function {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        constant(0, Constant::Index(0)),
        constant(1, Constant::Index(4)),
        constant(2, Constant::Index(1)),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        compare_with(4, ComparePredicate::NotEqual, 3, 0),
        constant(5, Constant::Bool(false)),
    ]);
    entry.terminator = Some(if matches!(case, CheckedRecurrenceCase::DivergentEntry) {
        Terminator::ConditionalBranch {
            condition: ValueId(4),
            then_target: BlockId(1),
            then_arguments: vec![ValueId(0)],
            else_target: BlockId(5),
            else_arguments: vec![ValueId(0)],
        }
    } else {
        Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(0)],
        }
    });

    let mut header = BasicBlock::new(BlockId(1));
    header
        .parameters
        .push(ValueDef::new(ValueId(10), Type::INDEX));
    if !matches!(case, CheckedRecurrenceCase::DivergentTripCount) {
        header
            .operations
            .push(compare_with(11, ComparePredicate::LessThan, 10, 1));
    }
    let condition = if matches!(case, CheckedRecurrenceCase::DivergentTripCount) {
        ValueId(4)
    } else {
        ValueId(11)
    };
    header.terminator = Some(Terminator::ConditionalBranch {
        condition,
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(5),
        else_arguments: vec![ValueId(10)],
    });

    let mut body = BasicBlock::new(BlockId(2));
    match case {
        CheckedRecurrenceCase::UncheckedAdd => body.operations.push(add(12, 10, 3)),
        CheckedRecurrenceCase::WrongPhi => {
            body.operations
                .push(checked_index(12, 13, CheckedBinaryOperator::Add, 3, 2))
        }
        CheckedRecurrenceCase::VaryingStep | CheckedRecurrenceCase::InvertedOverflowEdge => {
            body.operations
                .push(checked_index(12, 13, CheckedBinaryOperator::Add, 10, 3));
        }
        CheckedRecurrenceCase::Exact
        | CheckedRecurrenceCase::MismatchedFlag
        | CheckedRecurrenceCase::DivergentTripCount
        | CheckedRecurrenceCase::DivergentEntry => {
            body.operations
                .push(checked_index(12, 13, CheckedBinaryOperator::Add, 10, 2))
        }
    }
    let overflow = match case {
        CheckedRecurrenceCase::UncheckedAdd | CheckedRecurrenceCase::MismatchedFlag => ValueId(4),
        _ => ValueId(13),
    };
    let (then_target, else_target) = if matches!(case, CheckedRecurrenceCase::InvertedOverflowEdge)
    {
        (BlockId(3), BlockId(4))
    } else {
        (BlockId(4), BlockId(3))
    };
    body.terminator = Some(Terminator::ConditionalBranch {
        condition: overflow,
        then_target,
        then_arguments: vec![],
        else_target,
        else_arguments: vec![],
    });

    let mut latch = BasicBlock::new(BlockId(3));
    latch.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(12)],
    });

    let mut exit = BasicBlock::new(BlockId(5));
    exit.parameters
        .push(ValueDef::new(ValueId(20), Type::INDEX));
    exit.operations
        .push(compare_with(21, ComparePredicate::NotEqual, 20, 0));
    exit.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(21),
        then_target: BlockId(6),
        then_arguments: vec![],
        else_target: BlockId(7),
        else_arguments: vec![],
    });
    let mut barrier = returning(6);
    barrier.operations.push(workgroup_barrier());

    function(
        vec![],
        vec![
            entry,
            header,
            body,
            latch,
            returning(4),
            exit,
            barrier,
            returning(7),
        ],
    )
}

#[test]
fn exact_checked_add_recurrence_preserves_convergence() {
    let report = analyze_function(&checked_recurrence(CheckedRecurrenceCase::Exact));

    assert_eq!(report.value(ValueId(10)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(20)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(6)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn checked_recurrence_authority_boundaries_fail_closed() {
    for case in [
        CheckedRecurrenceCase::WrongPhi,
        CheckedRecurrenceCase::VaryingStep,
        CheckedRecurrenceCase::UncheckedAdd,
        CheckedRecurrenceCase::MismatchedFlag,
        CheckedRecurrenceCase::InvertedOverflowEdge,
        CheckedRecurrenceCase::DivergentTripCount,
        CheckedRecurrenceCase::DivergentEntry,
    ] {
        let report = analyze_function(&checked_recurrence(case));
        assert_eq!(
            report.value(ValueId(20)),
            Variation::Varying,
            "case {case:?} manufactured a uniform post-loop phi"
        );
        assert!(
            report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::DivergentBarrier {
                    block: BlockId(6),
                    ..
                }
            )),
            "case {case:?} manufactured post-loop convergence"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum RecurrenceEdgeHostile {
    NonexclusiveSuccess,
    NondominatingSuccess,
    MixedBackedges,
}

fn recurrence_edge_hostile(case: RecurrenceEdgeHostile) -> Function {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        constant(0, Constant::Index(0)),
        constant(1, Constant::Index(4)),
        constant(2, Constant::Index(1)),
        intrinsic(
            3,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        compare_with(4, ComparePredicate::NotEqual, 3, 0),
    ]);
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![ValueId(0)],
    });

    let mut header = BasicBlock::new(BlockId(1));
    header
        .parameters
        .push(ValueDef::new(ValueId(10), Type::INDEX));
    header
        .operations
        .push(compare_with(11, ComparePredicate::LessThan, 10, 1));
    header.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(11),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(8),
        else_arguments: vec![ValueId(10)],
    });

    let mut split = BasicBlock::new(BlockId(2));
    split.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(3),
        then_arguments: vec![],
        else_target: BlockId(4),
        else_arguments: vec![],
    });
    let mut checked = BasicBlock::new(BlockId(3));
    checked
        .operations
        .push(checked_index(12, 13, CheckedBinaryOperator::Add, 10, 2));
    let checked_success = match case {
        RecurrenceEdgeHostile::NondominatingSuccess => BlockId(6),
        RecurrenceEdgeHostile::NonexclusiveSuccess | RecurrenceEdgeHostile::MixedBackedges => {
            BlockId(5)
        }
    };
    checked.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(13),
        then_target: BlockId(7),
        then_arguments: vec![],
        else_target: checked_success,
        else_arguments: if matches!(case, RecurrenceEdgeHostile::NonexclusiveSuccess) {
            vec![ValueId(12)]
        } else {
            vec![]
        },
    });

    let mut alternate = BasicBlock::new(BlockId(4));
    if matches!(case, RecurrenceEdgeHostile::MixedBackedges) {
        alternate
            .operations
            .push(checked_index(14, 15, CheckedBinaryOperator::Add, 10, 3));
        alternate.terminator = Some(Terminator::ConditionalBranch {
            condition: ValueId(15),
            then_target: BlockId(7),
            then_arguments: vec![],
            else_target: BlockId(6),
            else_arguments: vec![],
        });
    } else {
        alternate.terminator = Some(Terminator::Branch {
            target: BlockId(5),
            arguments: vec![ValueId(10)],
        });
    }

    let mut latch = BasicBlock::new(BlockId(5));
    if !matches!(case, RecurrenceEdgeHostile::MixedBackedges) {
        latch
            .parameters
            .push(ValueDef::new(ValueId(20), Type::INDEX));
    }
    latch.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![if !matches!(case, RecurrenceEdgeHostile::MixedBackedges) {
            ValueId(20)
        } else {
            ValueId(12)
        }],
    });

    let mut second_latch = BasicBlock::new(BlockId(6));
    second_latch.terminator = Some(Terminator::Branch {
        target: if matches!(case, RecurrenceEdgeHostile::NondominatingSuccess) {
            BlockId(5)
        } else {
            BlockId(1)
        },
        arguments: vec![
            if matches!(case, RecurrenceEdgeHostile::NondominatingSuccess) {
                ValueId(12)
            } else {
                ValueId(14)
            },
        ],
    });

    let mut exit = BasicBlock::new(BlockId(8));
    exit.parameters
        .push(ValueDef::new(ValueId(30), Type::INDEX));
    exit.operations
        .push(compare_with(31, ComparePredicate::NotEqual, 30, 0));
    exit.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(31),
        then_target: BlockId(9),
        then_arguments: vec![],
        else_target: BlockId(10),
        else_arguments: vec![],
    });
    let mut barrier = returning(9);
    barrier.operations.push(workgroup_barrier());

    function(
        vec![],
        vec![
            entry,
            header,
            split,
            checked,
            alternate,
            latch,
            second_latch,
            returning(7),
            exit,
            barrier,
            returning(10),
        ],
    )
}

#[test]
fn checked_recurrence_rejects_edge_authority_substitutions() {
    for case in [
        RecurrenceEdgeHostile::NonexclusiveSuccess,
        RecurrenceEdgeHostile::NondominatingSuccess,
        RecurrenceEdgeHostile::MixedBackedges,
    ] {
        let report = analyze_function(&recurrence_edge_hostile(case));
        assert_eq!(
            report.value(ValueId(30)),
            Variation::Varying,
            "case {case:?} manufactured a uniform post-loop phi"
        );
        assert!(
            report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::DivergentBarrier {
                    block: BlockId(9),
                    ..
                }
            )),
            "case {case:?} manufactured post-loop convergence"
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum QuotientProofCase {
    Exact,
    MismatchedDivisibilityValue,
    RejectingDivisibilityEdge,
    NondominatingDivisibilityEdge,
    WrongDivisor,
    WrongScale,
    WrongMultiple,
    ZeroDivisor,
    MissingNumeratorProductSuccess,
    MissingQuotientProductSuccess,
    MissingExtraProductSuccess,
    MissingLhsAddSuccess,
    MissingSubtractionSuccess,
    ExhaustedResidualCapacity,
    DivergentMultiple,
    MutatedPrivateSlot,
    EscapedPrivateSlot,
}

fn quotient_residual(case: QuotientProofCase) -> Function {
    let mut entry = BasicBlock::new(BlockId(0));
    entry.operations.extend([
        intrinsic(
            33,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Global,
                axis: Axis::X,
            },
        ),
        constant(34, Constant::Index(64)),
        binary(0, BinaryOp::Remainder, 33, 34),
    ]);
    if matches!(case, QuotientProofCase::MutatedPrivateSlot) {
        entry.operations.extend([
            private_slot(40),
            private_store(40, 100),
            private_load(41, 40),
        ]);
    }
    if matches!(case, QuotientProofCase::EscapedPrivateSlot) {
        entry.operations.extend([
            constant(39, Constant::Index(0)),
            private_slot(40),
            private_store(40, 100),
            Operation::effect_free(
                ValueDef::new(
                    ValueId(41),
                    Type::pointer(Type::INDEX, AddressSpace::Private, AccessMode::ReadWrite),
                ),
                OperationKind::GetElementPointer {
                    base: ValueId(40),
                    offset: ValueId(39),
                },
            ),
            private_load(42, 40),
        ]);
    }
    entry.terminator = Some(Terminator::Branch {
        target: BlockId(1),
        arguments: vec![],
    });

    let guarded_multiple = if matches!(case, QuotientProofCase::MutatedPrivateSlot) {
        41
    } else {
        100
    };
    let mut bound = BasicBlock::new(BlockId(1));
    bound.operations.extend([
        constant(3, Constant::Index(4096)),
        compare_with(4, ComparePredicate::LessThan, guarded_multiple, 3),
    ]);
    bound.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(4),
        then_target: BlockId(2),
        then_arguments: vec![],
        else_target: BlockId(20),
        else_arguments: vec![],
    });

    let remainder_numerator = if matches!(case, QuotientProofCase::MismatchedDivisibilityValue) {
        101
    } else {
        guarded_multiple
    };
    let mut divisibility = BasicBlock::new(BlockId(2));
    divisibility.operations.extend([
        constant(5, Constant::Index(16)),
        binary(6, BinaryOp::Remainder, remainder_numerator, 5),
        constant(7, Constant::Index(0)),
        compare_with(8, ComparePredicate::Equal, 6, 7),
    ]);
    let (then_target, else_target) = match case {
        QuotientProofCase::RejectingDivisibilityEdge => (BlockId(20), BlockId(3)),
        QuotientProofCase::NondominatingDivisibilityEdge => (BlockId(21), BlockId(22)),
        _ => (BlockId(3), BlockId(20)),
    };
    divisibility.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(8),
        then_target,
        then_arguments: vec![],
        else_target,
        else_arguments: vec![],
    });

    let mut guard_then = BasicBlock::new(BlockId(21));
    guard_then.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut guard_bypass = BasicBlock::new(BlockId(22));
    guard_bypass.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });

    let mut divisor = BasicBlock::new(BlockId(3));
    let multiple = match case {
        QuotientProofCase::DivergentMultiple => 0,
        QuotientProofCase::MutatedPrivateSlot => {
            divisor
                .operations
                .extend([private_store(40, 101), private_load(42, 40)]);
            42
        }
        QuotientProofCase::EscapedPrivateSlot => 42,
        _ => 100,
    };
    let divisor_multiple = if matches!(case, QuotientProofCase::WrongDivisor) {
        101
    } else {
        multiple
    };
    let divisor_scale = if matches!(case, QuotientProofCase::WrongScale) {
        divisor.operations.push(constant(35, Constant::Index(8)));
        35
    } else {
        5
    };
    divisor.operations.extend([
        binary(9, BinaryOp::Divide, divisor_multiple, divisor_scale),
        compare_with(10, ComparePredicate::NotEqual, 9, 7),
    ]);
    divisor.terminator = Some(if matches!(case, QuotientProofCase::ZeroDivisor) {
        Terminator::Branch {
            target: BlockId(4),
            arguments: vec![],
        }
    } else {
        Terminator::ConditionalBranch {
            condition: ValueId(10),
            then_target: BlockId(4),
            then_arguments: vec![],
            else_target: BlockId(20),
            else_arguments: vec![],
        }
    });

    let mut numerator_product = BasicBlock::new(BlockId(4));
    numerator_product
        .operations
        .push(checked_index(11, 12, CheckedBinaryOperator::Multiply, 0, 5));
    numerator_product.terminator = Some(
        if matches!(case, QuotientProofCase::MissingNumeratorProductSuccess) {
            Terminator::Branch {
                target: BlockId(5),
                arguments: vec![],
            }
        } else {
            Terminator::ConditionalBranch {
                condition: ValueId(12),
                then_target: BlockId(20),
                then_arguments: vec![],
                else_target: BlockId(5),
                else_arguments: vec![],
            }
        },
    );

    let mut quotient_product = BasicBlock::new(BlockId(5));
    let rhs_multiple = if matches!(case, QuotientProofCase::WrongMultiple) {
        101
    } else {
        multiple
    };
    quotient_product.operations.extend([
        binary(13, BinaryOp::Divide, 0, 9),
        checked_index(14, 15, CheckedBinaryOperator::Multiply, 13, rhs_multiple),
    ]);
    quotient_product.terminator = Some(
        if matches!(case, QuotientProofCase::MissingQuotientProductSuccess) {
            Terminator::Branch {
                target: BlockId(6),
                arguments: vec![],
            }
        } else {
            Terminator::ConditionalBranch {
                condition: ValueId(15),
                then_target: BlockId(20),
                then_arguments: vec![],
                else_target: BlockId(6),
                else_arguments: vec![],
            }
        },
    );

    let mut intra_scale = BasicBlock::new(BlockId(6));
    let extra_factor = if matches!(case, QuotientProofCase::MissingExtraProductSuccess) {
        33
    } else {
        19
    };
    intra_scale.operations.extend([
        intrinsic(
            16,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(17, Constant::Index(64)),
        binary(18, BinaryOp::Remainder, 16, 17),
        binary(19, BinaryOp::Divide, 18, 5),
        constant(23, Constant::Index(4)),
        checked_index(24, 25, CheckedBinaryOperator::Multiply, extra_factor, 23),
    ]);
    intra_scale.terminator = Some(
        if matches!(case, QuotientProofCase::MissingExtraProductSuccess) {
            Terminator::Branch {
                target: BlockId(7),
                arguments: vec![],
            }
        } else {
            Terminator::ConditionalBranch {
                condition: ValueId(25),
                then_target: BlockId(20),
                then_arguments: vec![],
                else_target: BlockId(7),
                else_arguments: vec![],
            }
        },
    );

    let mut lhs = BasicBlock::new(BlockId(7));
    lhs.operations
        .push(checked_index(26, 27, CheckedBinaryOperator::Add, 11, 24));
    lhs.terminator = Some(if matches!(case, QuotientProofCase::MissingLhsAddSuccess) {
        Terminator::Branch {
            target: BlockId(8),
            arguments: vec![],
        }
    } else {
        Terminator::ConditionalBranch {
            condition: ValueId(27),
            then_target: BlockId(20),
            then_arguments: vec![],
            else_target: BlockId(8),
            else_arguments: vec![],
        }
    });

    let mut residual = BasicBlock::new(BlockId(8));
    residual.operations.push(checked_index(
        28,
        29,
        CheckedBinaryOperator::Subtract,
        26,
        14,
    ));
    residual.terminator = Some(
        if matches!(case, QuotientProofCase::MissingSubtractionSuccess) {
            Terminator::Branch {
                target: BlockId(9),
                arguments: vec![],
            }
        } else {
            Terminator::ConditionalBranch {
                condition: ValueId(29),
                then_target: BlockId(20),
                then_arguments: vec![],
                else_target: BlockId(9),
                else_arguments: vec![],
            }
        },
    );

    let mut offset = BasicBlock::new(BlockId(9));
    offset.operations.extend([
        constant(
            30,
            Constant::Index(
                if matches!(case, QuotientProofCase::ExhaustedResidualCapacity) {
                    4
                } else {
                    3
                },
            ),
        ),
        checked_index(31, 32, CheckedBinaryOperator::Add, 28, 30),
    ]);
    offset.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(32),
        then_target: BlockId(20),
        then_arguments: vec![],
        else_target: BlockId(10),
        else_arguments: vec![],
    });
    let mut success = returning(10);
    success.operations.push(workgroup_barrier());

    function(
        vec![ValueId(100), ValueId(101)],
        vec![
            entry,
            bound,
            divisibility,
            divisor,
            numerator_product,
            quotient_product,
            intra_scale,
            lhs,
            residual,
            offset,
            success,
            returning(20),
            guard_then,
            guard_bypass,
        ],
    )
}

#[test]
fn exact_quotient_residual_guards_authenticate_convergence() {
    let kernel = quotient_residual(QuotientProofCase::Exact);
    let report = analyze_as_kernel(&kernel);

    assert_eq!(report.value(ValueId(29)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(32)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(10)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn quotient_residual_authority_substitutions_fail_closed() {
    for case in [
        QuotientProofCase::MismatchedDivisibilityValue,
        QuotientProofCase::RejectingDivisibilityEdge,
        QuotientProofCase::NondominatingDivisibilityEdge,
        QuotientProofCase::WrongDivisor,
        QuotientProofCase::WrongScale,
        QuotientProofCase::WrongMultiple,
        QuotientProofCase::ZeroDivisor,
        QuotientProofCase::MissingNumeratorProductSuccess,
        QuotientProofCase::MissingQuotientProductSuccess,
        QuotientProofCase::MissingExtraProductSuccess,
        QuotientProofCase::MissingLhsAddSuccess,
        QuotientProofCase::MissingSubtractionSuccess,
        QuotientProofCase::ExhaustedResidualCapacity,
        QuotientProofCase::DivergentMultiple,
        QuotientProofCase::MutatedPrivateSlot,
        QuotientProofCase::EscapedPrivateSlot,
    ] {
        let kernel = quotient_residual(case);
        let report = analyze_as_kernel(&kernel);
        assert!(
            report.value(ValueId(29)) == Variation::Varying
                || report.value(ValueId(32)) == Variation::Varying,
            "case {case:?} authenticated unrelated arithmetic"
        );
        assert!(
            report.diagnostics().iter().any(|diagnostic| matches!(
                diagnostic,
                Diagnostic::DivergentBarrier {
                    block: BlockId(10),
                    ..
                }
            )),
            "case {case:?} manufactured convergence"
        );
    }
}

#[test]
fn parameters_calls_and_unknown_values_fail_closed() {
    let mut entry = returning(0);
    entry.operations = vec![
        Operation::effect_free(
            ValueDef::new(ValueId(1), Type::INDEX),
            OperationKind::Call {
                callee: FunctionId::new("helper"),
                arguments: vec![ValueId(0)],
            },
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(2), Type::INDEX),
            OperationKind::Cast {
                kind: fe2o3_kernel_ir::CastKind::Bitcast,
                value: ValueId(99),
                to: Type::INDEX,
            },
        ),
    ];

    let report = analyze_function(&function(vec![ValueId(0)], vec![entry]));

    assert_eq!(report.value(ValueId(0)), Variation::Varying);
    assert_eq!(report.value(ValueId(1)), Variation::Varying);
    assert_eq!(report.value(ValueId(2)), Variation::Varying);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary {
            callee: FunctionId::new("helper"),
        },
    }));
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: None,
        operation_index: None,
        reason: UnsupportedReason::UnknownValue { value: ValueId(99) },
    }));
}

#[test]
fn closed_float_intrinsics_and_pure_math_helpers_are_summarized() {
    let exp = FloatOperation::F32Math {
        function: F32MathFunction::Exp,
        implementation: F32MathFunction::Exp.required_implementation(),
        arguments: vec![ValueId(10)],
    };
    let mut helper_block = returning(1);
    helper_block.operations.push(exp.operation(ValueId(11)));
    helper_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(11)],
    });
    let helper = Function::definition(
        "math_helper",
        Signature::new(vec![Type::F32], vec![Type::F32]),
        vec![ValueId(10)],
        vec![helper_block],
    );

    let mut entry = returning(0);
    entry.operations = vec![
        constant(0, Constant::F32Bits(1.0_f32.to_bits())),
        Operation::effect_free(
            ValueDef::new(ValueId(1), Type::F32),
            OperationKind::Call {
                callee: helper.id.clone(),
                arguments: vec![ValueId(0)],
            },
        ),
    ];
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_math_helper");
    module.functions = vec![kernel.clone(), helper, exp.declaration()];

    let report = analyze_kernel_entry(&module, &kernel);
    assert_eq!(report.value(ValueId(1)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn nested_context_free_scalar_helpers_preserve_uniform_actuals() {
    let mut leaf_block = returning(0);
    leaf_block
        .operations
        .extend([constant(11, Constant::Index(1)), add(12, 10, 11)]);
    leaf_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(12)],
    });
    let leaf = Function::internal_helper(
        "uniform_leaf",
        Signature::new(vec![Type::INDEX], vec![Type::INDEX]),
        vec![ValueId(10)],
        vec![leaf_block],
    );

    let mut outer_block = returning(0);
    outer_block.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(21), Type::INDEX),
        OperationKind::Call {
            callee: leaf.id.clone(),
            arguments: vec![ValueId(20)],
        },
    ));
    outer_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(21)],
    });
    let outer = Function::internal_helper(
        "uniform_outer",
        Signature::new(vec![Type::INDEX], vec![Type::INDEX]),
        vec![ValueId(20)],
        vec![outer_block],
    );

    let mut entry = returning(0);
    entry.operations.extend([
        constant(0, Constant::Index(7)),
        Operation::effect_free(
            ValueDef::new(ValueId(1), Type::INDEX),
            OperationKind::Call {
                callee: outer.id.clone(),
                arguments: vec![ValueId(0)],
            },
        ),
    ]);
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_nested_scalar_helpers");
    module.functions = vec![kernel.clone(), outer, leaf];

    let report = analyze_kernel_entry(&module, &kernel);
    assert_eq!(report.value(ValueId(1)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn inert_v12_ordered_helper_is_not_a_uniform_call_summary() {
    use fe2o3_kernel_ir::{
        VerificationContractKeyV12, VerificationContractOperationV12, WorkgroupPipelineEventKindV12,
    };
    let mut helper_block = returning(0);
    helper_block
        .operations
        .push(constant(10, Constant::Index(0)));
    helper_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(10)],
    });
    let helper = Function::internal_helper(
        "ordered_helper",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![helper_block],
    );
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(1), Type::INDEX),
        OperationKind::Call {
            callee: helper.id.clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("inert_v12_uniformity");
    module.functions = vec![kernel.clone(), helper];
    assert_eq!(
        analyze_kernel_entry(&module, &kernel).value(ValueId(1)),
        Variation::GridUniform
    );
    module.functions[1].body.as_mut().unwrap().blocks[0]
        .operations
        .push(Operation::new(
            vec![],
            OperationKind::VerificationContract(
                VerificationContractOperationV12::WorkgroupPipelineEvent {
                    contract: VerificationContractKeyV12::new(0),
                    kind: WorkgroupPipelineEventKindV12::Stage,
                    storage: ValueId(10),
                    epoch: ValueId(10),
                },
            ),
        ));
    assert_eq!(
        analyze_kernel_entry(&module, &kernel).value(ValueId(1)),
        Variation::Varying
    );
}

#[test]
fn inert_v12_vector_load_stays_varying_through_layout_conversion() {
    use fe2o3_kernel_ir::{
        FixedVectorTypeV12, VectorLayoutConversionV12, VectorLayoutV12, VectorLoadOperationV12,
        VectorMemoryAccessV12,
    };
    let vector = FixedVectorTypeV12::new(ScalarType::F32, 4, VectorLayoutV12::Contiguous);
    let mut entry = returning(0);
    entry.operations = vec![
        Operation::effect_free(
            ValueDef::new(ValueId(1), Type::vector(vector)),
            OperationKind::VectorLoad(VectorLoadOperationV12::new(
                ValueId(0),
                VectorMemoryAccessV12::new(vector, MemoryAccess::new(AddressSpace::Global, 16)),
            )),
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(2), Type::vector(vector)),
            OperationKind::VectorLayoutConvert(VectorLayoutConversionV12::new(
                ValueId(1),
                vector.layout,
            )),
        ),
    ];
    let function = Function::internal_helper(
        "vector_load",
        Signature::new(
            vec![Type::pointer(
                Type::F32,
                AddressSpace::Global,
                AccessMode::ReadOnly,
            )],
            vec![],
        ),
        vec![ValueId(0)],
        vec![entry],
    );
    let report = analyze_kernel_entry(&Module::new("vector_analysis"), &function);
    assert_eq!(report.value(ValueId(1)), Variation::Varying);
    assert_eq!(report.value(ValueId(2)), Variation::Varying);
}

#[test]
fn pure_helper_with_workitem_source_is_not_a_uniform_call_summary() {
    let mut helper_block = returning(0);
    helper_block.operations.push(intrinsic(
        10,
        IntrinsicKind::InvocationIndex {
            kind: IndexKind::Local,
            axis: Axis::X,
        },
    ));
    helper_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(10)],
    });
    let helper = Function::internal_helper(
        "lane_dependent_helper",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![helper_block],
    );

    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: helper.id.clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_lane_dependent_helper");
    module.functions = vec![kernel.clone(), helper.clone()];

    let report = analyze_kernel_entry(&module, &kernel);
    assert_eq!(report.value(ValueId(0)), Variation::Varying);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary { callee: helper.id },
    }));
}

#[test]
fn execution_v15_lifecycle_operations_prevent_uniform_helper_summaries() {
    use fe2o3_kernel_ir::ExecutionOperationV15 as Execution;
    for execution in [
        Execution::ContextIssue,
        Execution::WorkgroupDerive {
            context: ValueId(10),
        },
        Execution::ScopeEnd {
            workgroup: ValueId(10),
            discarded: vec![],
        },
        Execution::MaskedTileLoadU32 {
            workgroup: ValueId(10),
            input: ValueId(10),
            base: ValueId(10),
            lanes: 3,
            elements: 2,
        },
        Execution::TileIntoFragmentU32 {
            tile: ValueId(10),
            lanes: 3,
            elements: 2,
        },
        Execution::FragmentIntoPartsU32 {
            fragment: ValueId(10),
            lanes: 3,
            elements: 2,
        },
    ] {
        let mut helper_block = returning(0);
        helper_block.operations = vec![
            constant(10, Constant::Index(1)),
            Operation::new(vec![], OperationKind::Execution(execution)),
        ];
        helper_block.terminator = Some(Terminator::Return {
            values: vec![ValueId(10)],
        });
        let helper = Function::internal_helper(
            "execution_helper",
            Signature::new(vec![], vec![Type::INDEX]),
            vec![],
            vec![helper_block],
        );
        let mut entry = returning(0);
        entry.operations.push(Operation::effect_free(
            ValueDef::new(ValueId(0), Type::INDEX),
            OperationKind::Call {
                callee: helper.id.clone(),
                arguments: vec![],
            },
        ));
        let kernel = function(vec![], vec![entry]);
        let mut module = Module::new("execution_v15_summary_refusal");
        module.functions = vec![kernel.clone(), helper.clone()];
        let report = analyze_kernel_entry(&module, &kernel);
        assert_eq!(report.value(ValueId(0)), Variation::Varying);
        assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
            block: Some(BlockId(0)),
            operation_index: Some(0),
            reason: UnsupportedReason::CallWithoutSummary { callee: helper.id },
        }));
    }
}

#[test]
fn pure_helper_with_workgroup_source_is_not_a_uniform_call_summary() {
    let mut helper_block = returning(0);
    helper_block.operations.push(intrinsic(
        10,
        IntrinsicKind::InvocationIndex {
            kind: IndexKind::Workgroup,
            axis: Axis::X,
        },
    ));
    helper_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(10)],
    });
    let helper = Function::internal_helper(
        "workgroup_dependent_helper",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![helper_block],
    );
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: helper.id.clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_workgroup_dependent_helper");
    module.functions = vec![kernel.clone(), helper.clone()];

    let report = analyze_kernel_entry(&module, &kernel);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary { callee: helper.id },
    }));
}

#[test]
fn terminating_helper_summary_requires_grid_uniform_actuals() {
    let mut dispatch = BasicBlock::new(BlockId(0));
    dispatch
        .operations
        .extend([constant(11, Constant::Index(0)), compare(12, 10, 11)]);
    dispatch.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(12),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut trap = BasicBlock::new(BlockId(1));
    trap.operations
        .push(AmdGpuDiagnosticOperation::Trap.operation(None));
    trap.terminator = Some(Terminator::Unreachable);
    let mut returned = returning(2);
    returned.terminator = Some(Terminator::Return {
        values: vec![ValueId(10)],
    });
    let helper = Function::internal_helper(
        "checked_uniform_helper",
        Signature::new(vec![Type::INDEX], vec![Type::INDEX]),
        vec![ValueId(10)],
        vec![dispatch, trap, returned],
    );

    let mut uniform_entry = returning(0);
    uniform_entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(1), Type::INDEX),
        OperationKind::Call {
            callee: helper.id.clone(),
            arguments: vec![ValueId(0)],
        },
    ));
    let uniform_kernel = function(vec![ValueId(0)], vec![uniform_entry]);
    let mut uniform_module = Module::new("uniformity_checked_helper_uniform_actual");
    uniform_module.functions = vec![uniform_kernel.clone(), helper.clone()];
    let uniform_report = analyze_kernel_entry(&uniform_module, &uniform_kernel);
    assert_eq!(uniform_report.value(ValueId(1)), Variation::GridUniform);
    assert!(uniform_report.diagnostics().is_empty());

    let mut varying_entry = returning(0);
    varying_entry.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        Operation::effect_free(
            ValueDef::new(ValueId(1), Type::INDEX),
            OperationKind::Call {
                callee: helper.id.clone(),
                arguments: vec![ValueId(0)],
            },
        ),
    ]);
    let varying_kernel = function(vec![], vec![varying_entry]);
    let mut varying_module = Module::new("uniformity_checked_helper_varying_actual");
    varying_module.functions = vec![varying_kernel.clone(), helper.clone()];
    let varying_report = analyze_kernel_entry(&varying_module, &varying_kernel);
    assert!(
        varying_report
            .diagnostics()
            .contains(&Diagnostic::Unsupported {
                block: Some(BlockId(0)),
                operation_index: Some(1),
                reason: UnsupportedReason::CallWithoutSummary {
                    callee: helper.id.clone(),
                },
            })
    );

    let mut divergent_dispatch = BasicBlock::new(BlockId(0));
    divergent_dispatch.operations.extend([
        intrinsic(
            0,
            IntrinsicKind::InvocationIndex {
                kind: IndexKind::Local,
                axis: Axis::X,
            },
        ),
        constant(1, Constant::Index(0)),
        compare(2, 0, 1),
    ]);
    divergent_dispatch.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(2),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut divergent_call = BasicBlock::new(BlockId(1));
    divergent_call.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(3), Type::INDEX),
        OperationKind::Call {
            callee: helper.id.clone(),
            arguments: vec![ValueId(1)],
        },
    ));
    divergent_call.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut bypass = BasicBlock::new(BlockId(2));
    bypass.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let control_kernel = function(
        vec![],
        vec![divergent_dispatch, divergent_call, bypass, returning(3)],
    );
    let mut control_module = Module::new("uniformity_checked_helper_divergent_control");
    control_module.functions = vec![control_kernel.clone(), helper.clone()];
    let control_report = analyze_kernel_entry(&control_module, &control_kernel);
    assert!(
        control_report
            .diagnostics()
            .contains(&Diagnostic::Unsupported {
                block: Some(BlockId(1)),
                operation_index: Some(0),
                reason: UnsupportedReason::CallWithoutSummary { callee: helper.id },
            })
    );
}

#[test]
fn helper_call_cycle_fails_closed_without_recursive_walk() {
    let mut first_block = returning(0);
    first_block.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: FunctionId::new("cycle_second"),
            arguments: vec![],
        },
    ));
    first_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(0)],
    });
    let first = Function::internal_helper(
        "cycle_first",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![first_block],
    );
    let mut second_block = returning(0);
    second_block.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: first.id.clone(),
            arguments: vec![],
        },
    ));
    second_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(0)],
    });
    let second = Function::internal_helper(
        "cycle_second",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![second_block],
    );
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: first.id.clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_helper_call_cycle");
    module.functions = vec![kernel.clone(), first.clone(), second];

    let report = analyze_kernel_entry(&module, &kernel);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary { callee: first.id },
    }));
}

#[test]
fn device_ffi_definition_is_not_admitted_as_a_uniform_helper() {
    let mut ffi_block = returning(0);
    ffi_block.operations.push(constant(10, Constant::Index(1)));
    ffi_block.terminator = Some(Terminator::Return {
        values: vec![ValueId(10)],
    });
    let ffi = Function::device_ffi_export(
        "device_ffi",
        Signature::new(vec![], vec![Type::INDEX]),
        vec![],
        vec![ffi_block],
    );
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: ffi.id.clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_device_ffi");
    module.functions = vec![kernel.clone(), ffi.clone()];

    let report = analyze_kernel_entry(&module, &kernel);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary { callee: ffi.id },
    }));
}

#[test]
fn oversized_deep_helper_chain_fails_closed_without_recursive_walk() {
    let helper_count = MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1;
    let helper_ids = (0..helper_count)
        .map(|index| FunctionId::new(format!("deep_helper_{index}")))
        .collect::<Vec<_>>();
    let mut helpers = Vec::with_capacity(helper_count);
    for (index, helper_id) in helper_ids.iter().enumerate() {
        let mut block = returning(0);
        if let Some(callee) = helper_ids.get(index + 1) {
            block.operations.push(Operation::effect_free(
                ValueDef::new(ValueId(0), Type::INDEX),
                OperationKind::Call {
                    callee: callee.clone(),
                    arguments: vec![],
                },
            ));
        } else {
            block.operations.push(constant(0, Constant::Index(0)));
        }
        block.terminator = Some(Terminator::Return {
            values: vec![ValueId(0)],
        });
        helpers.push(Function::internal_helper(
            helper_id.clone(),
            Signature::new(vec![], vec![Type::INDEX]),
            vec![],
            vec![block],
        ));
    }
    let mut entry = returning(0);
    entry.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(0), Type::INDEX),
        OperationKind::Call {
            callee: helper_ids[0].clone(),
            arguments: vec![],
        },
    ));
    let kernel = function(vec![], vec![entry]);
    let mut module = Module::new("uniformity_oversized_deep_chain");
    module.functions.push(kernel.clone());
    module.functions.extend(helpers);

    let report = analyze_kernel_entry(&module, &kernel);
    assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
        block: Some(BlockId(0)),
        operation_index: Some(0),
        reason: UnsupportedReason::CallWithoutSummary {
            callee: helper_ids[0].clone(),
        },
    }));
}

#[test]
fn known_diagnostic_call_is_accepted_but_remains_lane_local() {
    let mut entry = returning(0);
    entry
        .operations
        .push(AmdGpuDiagnosticOperation::Clock32.operation(Some(ValueId(0))));
    let kernel = function(vec![], vec![entry]);

    let report = analyze_function(&kernel);
    assert_eq!(report.value(ValueId(0)), Variation::Varying);
    assert!(report.diagnostics().is_empty());
}

#[test]
fn declarations_are_explicitly_unsupported() {
    let declaration = Function::declaration("external", Signature::new(vec![], vec![]));

    let report = analyze_function(&declaration);

    assert_eq!(
        report.diagnostics(),
        &[Diagnostic::Unsupported {
            block: None,
            operation_index: None,
            reason: UnsupportedReason::FunctionDeclaration,
        }]
    );
}
// Module-aware helper analysis supplies facts, never a helper-admission receipt.
fn module_helper_identity_v1() -> Function {
    let mut block = returning(0);
    block.terminator = Some(Terminator::Return {
        values: vec![ValueId(0)],
    });
    Function::internal_helper(
        "module_identity",
        Signature::new(vec![Type::INDEX], vec![Type::INDEX]),
        vec![ValueId(0)],
        vec![block],
    )
}

#[test]
fn module_helper_query_keeps_all_parameters_and_parameter_control_varying() {
    let mut entry = returning(0);
    entry.operations = vec![constant(10, Constant::Index(0)), compare(11, 0, 10)];
    entry.terminator = Some(Terminator::ConditionalBranch {
        condition: ValueId(11),
        then_target: BlockId(1),
        then_arguments: vec![],
        else_target: BlockId(2),
        else_arguments: vec![],
    });
    let mut varying = returning(1);
    varying.operations.push(Operation::effect_free(
        ValueDef::new(ValueId(12), Type::INDEX),
        OperationKind::Call {
            callee: FunctionId::new("module_identity"),
            arguments: vec![ValueId(1)],
        },
    ));
    varying.operations.push(workgroup_barrier());
    varying.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut other = returning(2);
    other.terminator = Some(Terminator::Branch {
        target: BlockId(3),
        arguments: vec![],
    });
    let mut helper = function(
        vec![ValueId(0), ValueId(1)],
        vec![entry, varying, other, returning(3)],
    );
    helper.required_capabilities = helper.derived_capabilities();
    let mut module = Module::new("module_helper_varying_parameters");
    module.functions = vec![helper, module_helper_identity_v1()];
    fe2o3_kernel_ir::verify_module(&module).unwrap();
    let report =
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).unwrap();
    assert_eq!(report.function(), &module.functions[0].id);
    for value in [0, 1, 11, 12] {
        assert_eq!(report.value(ValueId(value)), Variation::Varying);
    }
    assert_eq!(report.block_control(BlockId(0)), Variation::GridUniform);
    assert_eq!(report.block_control(BlockId(1)), Variation::Varying);
    assert_eq!(report.block_control(BlockId(2)), Variation::Varying);
    assert!(report.diagnostics().iter().any(|diagnostic| matches!(
        diagnostic,
        Diagnostic::DivergentBarrier {
            block: BlockId(1),
            control: Variation::Varying,
            ..
        }
    )));
    assert!(!report.diagnostics().iter().any(|diagnostic| matches!(
        diagnostic,
        Diagnostic::Unsupported {
            reason: UnsupportedReason::CallWithoutSummary { .. },
            ..
        }
    )));
}

#[test]
fn module_helper_query_summarizes_pure_calls_without_promoting_formal_parameters() {
    let mut entry = returning(0);
    entry.operations.push(constant(10, Constant::Index(7)));
    for (result, argument) in [(11, 10), (12, 0)] {
        entry.operations.push(Operation::effect_free(
            ValueDef::new(ValueId(result), Type::INDEX),
            OperationKind::Call {
                callee: FunctionId::new("module_identity"),
                arguments: vec![ValueId(argument)],
            },
        ));
    }
    let mut module = Module::new("module_helper_pure_calls");
    module.functions = vec![
        function(vec![ValueId(0)], vec![entry]),
        module_helper_identity_v1(),
    ];
    assert!(module.kernels.is_empty());
    fe2o3_kernel_ir::verify_module(&module).unwrap();
    let report =
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).unwrap();
    assert_eq!(report.value(ValueId(0)), Variation::Varying);
    assert_eq!(report.value(ValueId(11)), Variation::GridUniform);
    assert_eq!(report.value(ValueId(12)), Variation::Varying);
    assert_eq!(report.block_control(BlockId(0)), Variation::GridUniform);
    assert!(report.diagnostics().is_empty());
    assert!(
        !analyze_function(&module.functions[0])
            .diagnostics()
            .is_empty()
    );
}

#[test]
fn module_helper_query_keeps_recursive_and_unavailable_calls_unsupported() {
    for recursive in [false, true] {
        let callee = if recursive {
            "test"
        } else {
            "unavailable_helper"
        };
        let mut entry = returning(0);
        entry.operations.push(Operation::effect_free(
            ValueDef::new(ValueId(1), Type::INDEX),
            OperationKind::Call {
                callee: FunctionId::new(callee),
                arguments: vec![],
            },
        ));
        entry.terminator = Some(Terminator::Return {
            values: vec![ValueId(1)],
        });
        let helper = Function::internal_helper(
            "test",
            Signature::new(vec![], vec![Type::INDEX]),
            vec![],
            vec![entry],
        );
        let mut module = Module::new("module_helper_unsupported_calls");
        module.functions.push(helper);
        if !recursive {
            module.functions.push(Function::declaration(
                callee,
                Signature::new(vec![], vec![Type::INDEX]),
            ));
        }
        fe2o3_kernel_ir::verify_module(&module).unwrap();
        let report =
            fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).unwrap();
        assert_eq!(report.value(ValueId(1)), Variation::Varying);
        assert!(report.diagnostics().contains(&Diagnostic::Unsupported {
            block: Some(BlockId(0)),
            operation_index: Some(0),
            reason: UnsupportedReason::CallWithoutSummary {
                callee: FunctionId::new(callee)
            },
        }));
    }
}

#[test]
fn module_helper_query_refuses_foreign_duplicate_and_kernel_owners() {
    let mut module = Module::new("module_helper_owner");
    module.functions.push(module_helper_identity_v1());
    let clone = module.functions[0].clone();
    assert!(fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &clone).is_none());
    module.functions.push(clone);
    assert!(
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).is_none()
    );
    module.functions.pop();
    module.functions[0].role = fe2o3_kernel_ir::FunctionRole::KernelEntry;
    assert!(
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).is_none()
    );
    module.functions[0].role = fe2o3_kernel_ir::FunctionRole::InternalHelper;
    module.kernels.push(Kernel::new(
        "aliased_kernel",
        "module_identity",
        LaunchDomain::D1 {
            x: LaunchExtent::Static(1),
        },
    ));
    assert!(
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).is_none()
    );
}

#[test]
fn module_helper_query_exact_and_one_over_function_envelope() {
    let limit = fe2o3_kernel_ir::MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1;
    let mut module = Module::new("module_helper_function_envelope");
    module.functions = (0..limit)
        .map(|index| {
            let mut helper = module_helper_identity_v1();
            helper.id = FunctionId::new(format!("bounded_helper_{index}"));
            helper
        })
        .collect();
    fe2o3_kernel_ir::verify_module(&module).unwrap();
    let report =
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).unwrap();
    assert_eq!(report.value(ValueId(0)), Variation::Varying);
    let mut extra = module_helper_identity_v1();
    extra.id = FunctionId::new(format!("bounded_helper_{limit}"));
    module.functions.push(extra);
    assert!(
        fe2o3_kernel_analysis::analyze_module_helper_v1(&module, &module.functions[0]).is_none()
    );
}
