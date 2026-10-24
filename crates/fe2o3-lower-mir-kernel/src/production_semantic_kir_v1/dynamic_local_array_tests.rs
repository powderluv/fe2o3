use fe2o3_mir_model::semantic_mir_v1::{
    SemanticAbiArgumentV1, SemanticAbiValueAttributesV1, SemanticAggregateKindV1,
    SemanticAggregateRvalueV1,
};

#[test]
fn dynamic_local_array_emission_rejects_empty_and_overflowing_budgets_before_mutation() {
    let fixture = Fixture::new();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
    let mut lowering = fixture.lowering(&mut budget);
    let mut operations = vec![];
    fill_array(&mut lowering, &mut operations);
    let Some(SemanticValueBindingV1::Aggregate(fields)) = lowering.locals[1].clone() else {
        unreachable!();
    };
    lowering.max_operations = usize::MAX;
    lowering.emitted_operations = usize::MAX;
    let before = (
        operations.len(),
        lowering.next_value,
        lowering.emitted_operations,
    );
    for fields in [fields.as_slice(), &[]] {
        assert!(
            lowering
                .emit_local_array_selection_v1(
                    &mut operations,
                    ValueId(10_000),
                    &Type::Scalar(ScalarType::U64),
                    fields,
                    &Type::Scalar(ScalarType::U32),
                    0,
                )
                .is_err()
        );
        assert_eq!(
            (
                operations.len(),
                lowering.next_value,
                lowering.emitted_operations
            ),
            before
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum DynamicGuardCase {
    Argument,
    DefinedBefore,
    DefinedAfter,
    WrongComparison,
    WrongExtent,
    WrongIndex,
    MutatedArray,
    Bypass,
    EntryTarget,
    MissingGuard,
    EscapedIndex,
}

fn dynamic_guard_fixture(case: DynamicGuardCase) -> Fixture {
    let mut fixture = Fixture::new();
    let boolean = SemanticTypeIdV1::from_index(4);
    fixture.types.push(bool_type());
    let source = SemanticSourceProvenanceV1::unavailable();
    let plain =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let operand = |local, ty| SemanticOperandV1::Copy(plain(local, ty));
    let constant = |ty, value, size| {
        SemanticOperandV1::Constant(SemanticConstantV1::new(
            ty,
            SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, size).unwrap()),
        ))
    };
    let assign = |local, ty, kind| {
        SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                plain(local, ty),
                SemanticRvalueV1::new(ty, kind),
            )),
        )
    };
    let array = || {
        assign(
            1,
            ARRAY,
            SemanticRvalueKindV1::Aggregate(
                SemanticAggregateRvalueV1::new(
                    SemanticAggregateKindV1::Array,
                    (0..4).map(|n| constant(ELEMENT, 100 + n, 4)).collect(),
                )
                .unwrap(),
            ),
        )
    };
    let define_index = || {
        assign(
            2,
            INDEX,
            SemanticRvalueKindV1::Use(constant(INDEX, 1_u128 << 32, 8)),
        )
    };
    let mut statements = vec![array()];
    if matches!(case, DynamicGuardCase::DefinedBefore) {
        statements.push(define_index());
    }
    statements.push(assign(
        3,
        boolean,
        SemanticRvalueKindV1::Binary {
            operation: if matches!(case, DynamicGuardCase::WrongComparison) {
                SemanticBinaryOpV1::GreaterThan
            } else {
                SemanticBinaryOpV1::LessThan
            },
            left: operand(2, INDEX),
            right: constant(INDEX, 4, 8),
        },
    ));
    if matches!(case, DynamicGuardCase::DefinedAfter) {
        statements.push(define_index());
    }
    if matches!(case, DynamicGuardCase::MutatedArray) {
        statements.push(array());
    }
    if matches!(case, DynamicGuardCase::EscapedIndex) {
        statements.push(assign(
            4,
            INDEX,
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Shared,
                place: plain(2, INDEX),
            },
        ));
    }
    let edge =
        |role, target| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target));
    let terminator = if matches!(case, DynamicGuardCase::MissingGuard) {
        SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1))
    } else {
        SemanticTerminatorKindV1::Assert {
            condition: operand(3, boolean),
            expected: true,
            message: SemanticAssertMessageV1::BoundsCheck {
                length: constant(
                    INDEX,
                    if matches!(case, DynamicGuardCase::WrongExtent) {
                        3
                    } else {
                        4
                    },
                    8,
                ),
                index: operand(
                    if matches!(case, DynamicGuardCase::WrongIndex) {
                        4
                    } else {
                        2
                    },
                    INDEX,
                ),
            },
            target: edge(
                SemanticEdgeRoleV1::AssertSuccess,
                if matches!(case, DynamicGuardCase::EntryTarget) {
                    0
                } else {
                    1
                },
            ),
            unwind: SemanticUnwindActionV1::Unreachable,
        }
    };
    let block = |tag, statements, terminator| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, terminator),
        )
        .unwrap()
    };
    let mut blocks = vec![
        block(180, statements, terminator),
        block(
            181,
            vec![SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::Nop,
            )],
            SemanticTerminatorKindV1::Return,
        ),
    ];
    if matches!(case, DynamicGuardCase::Bypass) {
        blocks.push(block(
            182,
            vec![],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
        ));
    }
    let local = |tag, ty, role| {
        SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([tag; 32]),
            ty,
            role,
            source,
        )
    };
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([183; 32]),
        SemanticLayoutIdentityV1::from_sha256([184; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        1,
        vec![SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
            INDEX,
            SemanticAbiPassModeV1::Direct(SemanticAbiValueAttributesV1::plain()),
        ))],
        SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    fixture.function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([185; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([186; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([187; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([188; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([189; 32]),
        source,
        abi,
        vec![
            local(190, UNIT, SemanticLocalRoleV1::Return),
            local(191, ARRAY, SemanticLocalRoleV1::Temporary),
            local(192, INDEX, SemanticLocalRoleV1::Argument(0)),
            local(193, boolean, SemanticLocalRoleV1::Temporary),
            local(194, INDEX, SemanticLocalRoleV1::Temporary),
        ],
        BLOCK,
        blocks,
    )
    .unwrap();
    fixture
}

#[test]
fn dynamic_local_array_guards_require_definition_time_identity_and_exclusive_success() {
    for case in [
        DynamicGuardCase::Argument,
        DynamicGuardCase::DefinedBefore,
        DynamicGuardCase::DefinedAfter,
        DynamicGuardCase::WrongComparison,
        DynamicGuardCase::WrongExtent,
        DynamicGuardCase::WrongIndex,
        DynamicGuardCase::MutatedArray,
        DynamicGuardCase::Bypass,
        DynamicGuardCase::EntryTarget,
        DynamicGuardCase::MissingGuard,
        DynamicGuardCase::EscapedIndex,
    ] {
        let fixture = dynamic_guard_fixture(case);
        let mut analysis =
            FixedArrayGuardAnalysisV1::new(&fixture.types, &[], &fixture.function, 4096).unwrap();
        let expected = matches!(
            case,
            DynamicGuardCase::Argument | DynamicGuardCase::DefinedBefore
        );
        for _ in 0..2 {
            let guard = analysis
                .authorize(
                    SemanticBlockIdV1::from_index(1),
                    0,
                    SemanticLocalIdV1::from_index(1),
                    SemanticLocalIdV1::from_index(2),
                    4,
                    4096,
                )
                .unwrap();
            assert_eq!(guard, expected.then_some(0), "{case:?}");
        }
        assert!(
            analysis
                .authorize(
                    SemanticBlockIdV1::from_index(1),
                    0,
                    SemanticLocalIdV1::from_index(1),
                    SemanticLocalIdV1::from_index(2),
                    0,
                    4096,
                )
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn dynamic_local_array_analysis_has_explicit_work_and_storage_limits() {
    let fixture = dynamic_guard_fixture(DynamicGuardCase::Argument);
    assert!(matches!(
        FixedArrayGuardAnalysisV1::new(&fixture.types, &[], &fixture.function, 0,),
        Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::AnalysisWork,
            ..
        })
    ));
    let mut analysis =
        FixedArrayGuardAnalysisV1::new(&fixture.types, &[], &fixture.function, 4096).unwrap();
    let exact = analysis.work + analysis.traversal_work * 8;
    assert!(
        analysis
            .authorize(
                SemanticBlockIdV1::from_index(1),
                0,
                SemanticLocalIdV1::from_index(1),
                SemanticLocalIdV1::from_index(2),
                4,
                exact,
            )
            .unwrap()
            .is_some()
    );
    assert!(matches!(
        analysis.authorize(
            SemanticBlockIdV1::from_index(1),
            0,
            SemanticLocalIdV1::from_index(1),
            SemanticLocalIdV1::from_index(2),
            4,
            exact,
        ),
        Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::AnalysisWork,
            ..
        })
    ));
}

#[test]
fn dynamic_local_array_selection_preserves_types_and_exact_emission_budgets() {
    let fixture = Fixture::new();
    for count in [1_usize, 3, 5, 255] {
        for index_type in [
            Type::Scalar(ScalarType::U8),
            Type::Scalar(ScalarType::U16),
            Type::Scalar(ScalarType::U32),
            Type::Scalar(ScalarType::U64),
            Type::INDEX,
        ] {
            for element_type in [Type::Scalar(ScalarType::U32), Type::BOOL] {
                let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
                let mut lowering = fixture.lowering(&mut budget);
                lowering.max_operations = 4096;
                let mut operations = vec![];
                let fields = (0..count)
                    .map(|n| {
                        lowering
                            .emit(
                                &mut operations,
                                element_type.clone(),
                                OperationKind::Constant(if element_type == Type::BOOL {
                                    Constant::Bool(n % 2 == 0)
                                } else {
                                    Constant::U32(n as u32)
                                }),
                            )
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let before = (
                    operations.len(),
                    lowering.next_value,
                    lowering.emitted_operations,
                );
                let extra = 3 * (count - 1);
                if extra > 0 {
                    lowering.max_operations = before.2 + extra - 1;
                    assert!(matches!(
                        lowering.emit_local_array_selection_v1(
                            &mut operations,
                            ValueId(10_000),
                            &index_type,
                            &fields,
                            &element_type,
                            0,
                        ),
                        Err(ProductionSemanticKirErrorV1::ResourceLimit {
                            resource: ProductionSemanticKirResourceV1::Operations,
                            ..
                        })
                    ));
                    assert_eq!(
                        (
                            operations.len(),
                            lowering.next_value,
                            lowering.emitted_operations
                        ),
                        before
                    );
                }
                lowering.max_operations = before.2 + extra;
                let selected = lowering
                    .emit_local_array_selection_v1(
                        &mut operations,
                        ValueId(10_000),
                        &index_type,
                        &fields,
                        &element_type,
                        0,
                    )
                    .unwrap();
                assert_eq!(selected.value().unwrap().1, element_type);
                assert_eq!(operations.len(), before.0 + extra);
                assert_eq!(lowering.emitted_operations, before.2 + extra);
                assert!(
                    operations[before.0..]
                        .iter()
                        .all(|operation| !matches!(operation.kind, OperationKind::Cast { .. }))
                );
            }
        }
    }
}
