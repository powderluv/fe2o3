use super::*;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticCallDestinationV1, SemanticCallableIdV1, SemanticCodeObjectVersionV1,
    SemanticControlFlowEdgeV1, SemanticDeviceFfiContractIdentityV1, SemanticDeviceFfiEffectsV1,
    SemanticDeviceFfiImportContractV1, SemanticDeviceFfiPhysicalAbiIdentityV1,
    SemanticDeviceFfiSemanticIdentityV1, SemanticDeviceFfiTargetV1, SemanticLinkSymbolV1,
};

// Constructed dispatcher contexts are not admitted source owners or proof witnesses.
fn with_dispatch_callable<R>(
    replace: impl FnOnce(SemanticCallableDeclV1) -> SemanticCallableDeclV1,
    run: impl FnOnce(
        &mut SemanticFunctionLoweringV1<'_, '_>,
        &SemanticDirectCallV1,
    ) -> Result<R, ProductionSemanticKirErrorV1>,
) -> Result<R, ProductionSemanticKirErrorV1> {
    with_guarded_result_context(false, |original, call| {
        let callables = [replace(original.callables[0].clone())];
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = SemanticFunctionLoweringV1::new(
            original.types,
            &callables,
            original.function,
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: None,
            },
            Some(BlockId(2)),
            None,
            BTreeSet::new(),
            1,
            false,
            32,
            &mut lowering_budget,
        )?;
        lowering.locals = original.locals.clone();
        lowering.next_value = original.next_value;
        run(&mut lowering, call)
    })
}

fn with_dispatch_intrinsic<R>(
    operation: SemanticCompilerIntrinsicOperationV1,
    run: impl FnOnce(
        &mut SemanticFunctionLoweringV1<'_, '_>,
        &SemanticDirectCallV1,
    ) -> Result<R, ProductionSemanticKirErrorV1>,
) -> Result<R, ProductionSemanticKirErrorV1> {
    with_dispatch_callable(
        |callable| {
            let SemanticCallableDeclV1::CompilerIntrinsic {
                binding,
                operation_identity,
                ..
            } = callable
            else {
                panic!("expected the existing intrinsic fixture");
            };
            SemanticCallableDeclV1::CompilerIntrinsic {
                binding,
                operation,
                operation_identity,
            }
        },
        run,
    )
}

fn call_with(
    template: &SemanticDirectCallV1,
    arguments: Vec<SemanticOperandV1>,
    destination: Option<SemanticCallDestinationV1>,
    unwind: SemanticUnwindActionV1,
) -> SemanticDirectCallV1 {
    SemanticDirectCallV1::new_callable(template.callee(), arguments, destination, unwind).unwrap()
}

fn assert_dispatch_error(
    error: ProductionSemanticKirErrorV1,
    expected_function: u32,
    expected_block: Option<u32>,
    expected: &'static str,
) {
    match error {
        ProductionSemanticKirErrorV1::Unsupported {
            function,
            block,
            statement,
            detail,
        } => {
            assert_eq!(
                (function, block, statement),
                (expected_function, expected_block, None)
            );
            assert_eq!(detail, expected);
        }
        other => panic!("unexpected dispatcher error: {other:?}"),
    }
}

#[test]
fn cleanup_rejection_precedes_callable_lookup_and_missing_destination() {
    with_guarded_result_context(false, |lowering, _| {
        let mut operations = Vec::new();
        for (unwind, expected) in [
            (
                SemanticUnwindActionV1::Cleanup(SemanticControlFlowEdgeV1::new(
                    SemanticEdgeRoleV1::CallUnwind,
                    SemanticBlockIdV1::from_index(1),
                )),
                "trusted compiler intrinsic has a cleanup unwind edge",
            ),
            (SemanticUnwindActionV1::Unreachable, "callable is missing"),
        ] {
            let call = SemanticDirectCallV1::new_callable(
                SemanticCallableIdV1::from_index(99),
                vec![],
                None,
                unwind,
            )
            .unwrap();
            let error = lowering
                .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
                .unwrap_err();
            assert_dispatch_error(error, 0, Some(0), expected);
            assert!(operations.is_empty());
            assert_eq!(lowering.next_value, 2);
            assert_eq!(lowering.emitted_operations, 0);
            assert!(lowering.locals[3].is_none());
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn destination_is_required_before_selected_family_argument_validation() {
    with_guarded_result_context(false, |lowering, template| {
        let call = call_with(template, vec![], None, SemanticUnwindActionV1::Unreachable);
        let mut operations = Vec::new();
        assert_dispatch_error(
            lowering
                .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
                .unwrap_err(),
            0,
            Some(0),
            "compiler intrinsic call has no destination",
        );
        assert!(operations.is_empty());
        assert_eq!(lowering.next_value, 2);
        assert!(lowering.locals[3].is_none());
        Ok(())
    })
    .unwrap();
}

#[test]
fn trap_bypasses_destination_and_preserves_argument_then_destination_rejections() {
    for case in 0..3 {
        with_dispatch_intrinsic(
            SemanticCompilerIntrinsicOperationV1::Trap,
            |lowering, template| {
                let arguments = if case == 1 {
                    template.arguments().to_vec()
                } else {
                    vec![]
                };
                let destination = if case == 0 {
                    None
                } else {
                    template.destination().cloned()
                };
                let call = call_with(
                    template,
                    arguments,
                    destination,
                    SemanticUnwindActionV1::Unreachable,
                );
                let mut operations = Vec::new();
                let result =
                    lowering.lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations);
                if case == 0 {
                    assert_eq!(result.unwrap(), Terminator::Unreachable);
                    assert_eq!(
                        operations,
                        vec![AmdGpuDiagnosticOperation::Trap.operation(None)]
                    );
                    assert_eq!(lowering.emitted_operations, 1);
                } else {
                    assert_dispatch_error(
                        result.unwrap_err(),
                        0,
                        Some(0),
                        if case == 1 {
                            "compiler intrinsic argument count changed"
                        } else {
                            "trap compiler intrinsic unexpectedly has a destination"
                        },
                    );
                    assert!(operations.is_empty());
                    assert_eq!(lowering.emitted_operations, 0);
                }
                assert_eq!(lowering.next_value, 2);
                assert!(lowering.locals[3].is_none());
                Ok(())
            },
        )
        .unwrap();
    }
}

#[test]
fn retired_matrix_load_is_rejected_before_dispatch() {
    let retired = SemanticCompilerIntrinsicOperationV1::Bf16MatrixLoad {
        option_fragment: SemanticTypeIdV1::from_index(0),
        view: SemanticTypeIdV1::from_index(1),
        lane: SemanticTypeIdV1::from_index(2),
        fragment: SemanticTypeIdV1::from_index(3),
        contract: operand_contract(SemanticMfmaOperandRoleV1::A),
        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
    };
    let mut dispatched = false;
    let result = with_dispatch_intrinsic(retired, |_, _| {
        dispatched = true;
        Ok(())
    });
    assert!(!dispatched);
    assert_dispatch_error(
        result.unwrap_err(),
        0,
        None,
        "the retired Option-returning BF16 matrix load is not admitted; use Bf16MatrixLoadZeroFilledV2",
    );
}

#[test]
fn device_ffi_refusal_retains_actual_function_coordinate_before_destination_checks() {
    with_dispatch_callable(
        |callable| SemanticCallableDeclV1::DeviceFfiImport {
            binding: callable.binding().unwrap().clone(),
            contract: SemanticDeviceFfiImportContractV1::new(
                SemanticDeviceFfiContractIdentityV1::from_sha256([201; 32]),
                SemanticLinkSymbolV1::new(b"dispatcher_refusal".to_vec()).unwrap(),
                SemanticDeviceFfiTargetV1::AmdGpuGfx942XnackMinus,
                SemanticCodeObjectVersionV1::V6,
                SemanticDeviceFfiPhysicalAbiIdentityV1::from_sha256([202; 32]),
                SemanticDeviceFfiEffectsV1::none(),
                SemanticDeviceFfiSemanticIdentityV1::from_sha256([203; 32]),
            ),
        },
        |lowering, template| {
            lowering.semantic_function = SemanticFunctionIdV1::from_index(7);
            let call = call_with(template, vec![], None, SemanticUnwindActionV1::Unreachable);
            let mut operations = Vec::new();
            assert_dispatch_error(
                lowering
                    .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
                    .unwrap_err(),
                7,
                Some(0),
                "device-FFI calls remain closed in deterministic helper lowering",
            );
            assert!(operations.is_empty());
            assert_eq!(lowering.next_value, 2);
            assert!(lowering.locals[3].is_none());
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn projected_destination_is_prepared_before_family_argument_rejection() {
    with_guarded_result_context(true, |lowering, template| {
        let mut operations = Vec::new();
        install_result_slot(lowering, true, &mut operations)?;
        assert_eq!(operations.len(), 3);
        let call = call_with(
            template,
            vec![],
            template.destination().cloned(),
            SemanticUnwindActionV1::Unreachable,
        );
        assert_dispatch_error(
            lowering
                .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
                .unwrap_err(),
            0,
            Some(0),
            "compiler intrinsic argument count changed",
        );
        assert_eq!(operations.len(), 4);
        assert_saved_projected_address(&operations);
        assert_eq!(lowering.emitted_operations, 4);
        assert_eq!(lowering.next_value, 5);
        assert!(lowering.retained_local_initialized.contains(&3));
        assert!(lowering.locals[3].is_none());
        Ok(())
    })
    .unwrap();
}

fn volatile_operations() -> Vec<Operation> {
    let pointer = Type::pointer(
        Type::Scalar(ScalarType::U32),
        AddressSpace::Global,
        AccessMode::ReadOnly,
    );
    let result = |id, ty, kind| Operation::effect_free(ValueDef::new(ValueId(id), ty), kind);
    let mut access = MemoryAccess::new(AddressSpace::Global, 4);
    access.volatile = true;
    vec![
        result(
            2,
            Type::INDEX,
            OperationKind::SliceLength { slice: ValueId(0) },
        ),
        result(
            3,
            Type::Scalar(ScalarType::Bool),
            OperationKind::Compare {
                predicate: ComparePredicate::LessThan,
                lhs: ValueId(1),
                rhs: ValueId(2),
            },
        ),
        result(4, Type::INDEX, OperationKind::Constant(Constant::Index(0))),
        result(
            5,
            Type::INDEX,
            OperationKind::Select {
                condition: ValueId(3),
                true_value: ValueId(1),
                false_value: ValueId(4),
            },
        ),
        result(
            6,
            pointer.clone(),
            OperationKind::SliceData { slice: ValueId(0) },
        ),
        result(
            7,
            pointer,
            OperationKind::GetElementPointer {
                base: ValueId(6),
                offset: ValueId(5),
            },
        ),
        result(
            8,
            Type::Scalar(ScalarType::U32),
            OperationKind::Constant(Constant::U32(0)),
        ),
        result(
            9,
            Type::Scalar(ScalarType::U32),
            OperationKind::GuardedLoad {
                pointer: ValueId(7),
                predicate: ValueId(3),
                fallback: ValueId(8),
                access,
            },
        ),
    ]
}

#[test]
fn every_volatile_operation_denial_retains_the_exact_accepted_prefix() {
    let expected = volatile_operations();
    assert_eq!(expected.len(), 8);
    for cap in 0..=8 {
        with_guarded_result_context(false, |lowering, call| {
            lowering.max_operations = cap;
            let mut operations = Vec::new();
            let result =
                lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations);
            assert_eq!(operations, expected[..cap]);
            assert_eq!(lowering.emitted_operations, cap);
            if cap < 8 {
                assert!(
                    matches!(result, Err(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::Operations, actual, limit,
                }) if actual == cap + 1 && limit == cap)
                );
                // emit() consumes an SSA ID before reserve_operation() can refuse it.
                assert_eq!(lowering.next_value, 2 + cap as u32 + 1);
                assert!(lowering.locals[3].is_none());
            } else {
                assert_eq!(lowering.next_value, 10);
                assert_eq!(
                    lowering.locals[3].as_ref().unwrap().value().unwrap(),
                    (ValueId(9), Type::Scalar(ScalarType::U32)),
                );
                assert_eq!(
                    result.unwrap(),
                    Terminator::ConditionalBranch {
                        condition: ValueId(3),
                        then_target: BlockId(1),
                        then_arguments: vec![],
                        else_target: BlockId(2),
                        else_arguments: vec![],
                    }
                );
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn missing_failure_block_is_checked_after_the_result_is_committed() {
    with_guarded_result_context(false, |lowering, call| {
        lowering.assert_failure_block = None;
        let mut operations = Vec::new();
        assert_dispatch_error(
            lowering
                .lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations)
                .unwrap_err(),
            0,
            Some(0),
            "volatile load has no retained bounds-failure trap block",
        );
        assert_eq!(operations, volatile_operations());
        assert_eq!(lowering.emitted_operations, 8);
        assert_eq!(lowering.next_value, 10);
        assert_eq!(
            lowering.locals[3].as_ref().unwrap().value().unwrap(),
            (ValueId(9), Type::Scalar(ScalarType::U32)),
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn volatile_family_result_keeps_the_same_guard_across_common_destination_paths() {
    for projected in [false, true] {
        with_guarded_result_context(projected, |lowering, call| {
            let mut operations = Vec::new();
            install_result_slot(lowering, projected, &mut operations)?;
            let terminator =
                lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations)?;
            assert_guarded_result(&operations, &terminator, projected);
            if projected {
                assert_saved_projected_address(&operations);
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn guarded_result_store_type_refusal_precedes_missing_failure_block() {
    with_guarded_result_context(false, |lowering, call| {
        let mut operations = Vec::new();
        install_result_slot(lowering, false, &mut operations)?;
        let slot = lowering
            .retained_local_slots
            .get_mut(&ScopedAllocationIdentityV29::LegacyLocal(3))
            .unwrap();
        let SemanticRetainedStorageV29::ScalarArray { kernel_type, .. } = &mut slot.storage else {
            panic!("scalar fixture");
        };
        *kernel_type = Type::Scalar(ScalarType::U64);
        lowering.assert_failure_block = None;
        assert_dispatch_error(
            lowering
                .lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations)
                .unwrap_err(),
            0,
            Some(0),
            "retained-local value type differs from its private slot",
        );
        assert_eq!(operations.len(), 9);
        assert!(matches!(
            operations.last().unwrap().kind,
            OperationKind::GuardedLoad { .. }
        ));
        assert!(!lowering.retained_local_initialized.contains(&3));
        assert!(lowering.locals[3].is_none());
        Ok(())
    })
    .unwrap();
}
