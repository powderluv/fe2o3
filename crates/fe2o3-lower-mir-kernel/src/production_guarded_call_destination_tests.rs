use super::*;

// These use the existing private volatile-load context, not an admitted source
// owner. Slot installation is explicit; calls and memory operations are real.
fn with_guarded_result_context<R>(
    projected: bool,
    run: impl FnOnce(
        &mut SemanticFunctionLoweringV1<'_, '_>,
        &SemanticDirectCallV1,
    ) -> Result<R, ProductionSemanticKirErrorV1>,
) -> Result<R, ProductionSemanticKirErrorV1> {
    with_volatile_load_context_for_test(
        0,
        16,
        SemanticTypeIdV1::from_index(2),
        AddressSpace::Global,
        AccessMode::ReadOnly,
        projected,
        run,
    )
}

fn install_result_slot(
    lowering: &mut SemanticFunctionLoweringV1<'_, '_>,
    projected: bool,
    operations: &mut Vec<Operation>,
) -> Result<ValueId, ProductionSemanticKirErrorV1> {
    assert_eq!(lowering.emitted_operations, 0);
    assert_eq!(lowering.next_value, 2);
    assert!(lowering.locals[3].is_none());
    assert!(lowering.retained_local_slots.is_empty());
    let scalar = Type::Scalar(ScalarType::U32);
    let pointer_type = Type::pointer(scalar.clone(), AddressSpace::Private, AccessMode::ReadWrite);
    let output = lowering.emit_id(
        operations,
        pointer_type.clone(),
        OperationKind::Alloca {
            element: scalar.clone(),
            count: None,
            address_space: AddressSpace::Private,
            alignment: 4,
        },
    )?;
    assert_eq!(output, ValueId(2));
    let (pointer, semantic_type, kernel_type, alignment) = if projected {
        let pointer = lowering.emit_id(
            operations,
            Type::pointer(
                pointer_type.clone(),
                AddressSpace::Private,
                AccessMode::ReadWrite,
            ),
            OperationKind::Alloca {
                element: pointer_type.clone(),
                count: None,
                address_space: AddressSpace::Private,
                alignment: 8,
            },
        )?;
        (
            pointer,
            SemanticTypeIdV1::from_index(5),
            pointer_type.clone(),
            8,
        )
    } else {
        (output, SemanticTypeIdV1::from_index(1), scalar, 4)
    };
    assert_eq!(lowering.function.locals()[3].ty(), semantic_type);
    assert!(
        lowering
            .retained_local_slots
            .insert(
                ScopedAllocationIdentityV29::LegacyLocal(3),
                SemanticRetainedLocalSlotV1 {
                    pointer,
                    semantic_type,
                    storage: SemanticRetainedStorageV29::ScalarArray {
                        kernel_type,
                        alignment,
                        array: None
                    },
                }
            )
            .is_none()
    );
    if projected {
        lowering.store_retained_local_v1(
            SemanticBlockIdV1::from_index(0),
            None,
            SemanticLocalIdV1::from_index(3),
            &SemanticValueBindingV1::Value {
                id: output,
                ty: pointer_type,
            },
            SemanticVolatilityV1::NonVolatile,
            operations,
        )?;
    }
    assert_eq!(lowering.retained_local_initialized.contains(&3), projected);
    Ok(output)
}

fn assert_guarded_result(operations: &[Operation], terminator: &Terminator, projected: bool) {
    let (body, total, predicate, result, destination) = if projected {
        (4, 13, ValueId(6), ValueId(12), ValueId(4))
    } else {
        (1, 10, ValueId(4), ValueId(10), ValueId(2))
    };
    assert_eq!(operations.len(), total);
    assert!(matches!(
        operations[body].kind,
        OperationKind::SliceLength { slice: ValueId(0) }
    ));
    let load = &operations[body + 7];
    assert_eq!(
        load.results,
        [ValueDef::new(result, Type::Scalar(ScalarType::U32))]
    );
    assert!(matches!(load.kind, OperationKind::GuardedLoad {
        predicate: actual,
        access: MemoryAccess { address_space: AddressSpace::Global, alignment: 4, volatile: true },
        ..
    } if actual == predicate));
    assert_eq!(
        load.memory_effects(),
        [fe2o3_kernel_ir::MemoryEffect::Read(AddressSpace::Global)]
    );
    let store = &operations[body + 8];
    assert!(store.results.is_empty());
    assert!(matches!(store.kind, OperationKind::GuardedStore {
        pointer, predicate: actual, value,
        access: MemoryAccess { address_space: AddressSpace::Private, alignment: 4, volatile: false },
    } if pointer == destination && actual == predicate && value == result));
    assert_eq!(
        store.memory_effects(),
        [fe2o3_kernel_ir::MemoryEffect::Write(AddressSpace::Private)]
    );
    assert_eq!(
        operations
            .iter()
            .filter(|op| matches!(op.kind, OperationKind::GuardedStore { .. }))
            .count(),
        1
    );
    assert_eq!(
        operations
            .iter()
            .filter(|op| matches!(op.kind, OperationKind::Store { .. }))
            .count(),
        usize::from(projected)
    );
    assert_eq!(
        operations
            .iter()
            .filter(|op| matches!(op.kind, OperationKind::Load { .. }))
            .count(),
        usize::from(projected)
    );
    let Terminator::ConditionalBranch {
        condition,
        then_target,
        then_arguments,
        else_target,
        else_arguments,
    } = terminator
    else {
        panic!("guarded result must retain its bounds branch")
    };
    assert_eq!(*condition, predicate);
    assert_eq!(*then_target, BlockId(1));
    assert_eq!(*else_target, BlockId(2));
    assert!(then_arguments.is_empty());
    assert!(else_arguments.is_empty());
    if projected {
        assert_saved_projected_address(operations);
    }
}

fn assert_saved_projected_address(operations: &[Operation]) {
    assert!(matches!(
        operations[2].kind,
        OperationKind::Store {
            pointer: ValueId(3),
            value: ValueId(2),
            access: MemoryAccess {
                address_space: AddressSpace::Private,
                alignment: 8,
                volatile: false
            },
        }
    ));
    assert_eq!(
        operations[3].results,
        [ValueDef::new(
            ValueId(4),
            Type::pointer(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Private,
                AccessMode::ReadWrite
            )
        )]
    );
    assert!(matches!(
        operations[3].kind,
        OperationKind::Load {
            pointer: ValueId(3),
            access: MemoryAccess {
                address_space: AddressSpace::Private,
                alignment: 8,
                volatile: false
            },
        }
    ));
    // Only the one pre-intrinsic address load exists. The guarded result
    // store uses that saved SSA pointer, not the slot address or a reload.
    assert!(
        operations[4..]
            .iter()
            .all(|op| !matches!(op.kind, OperationKind::Load { .. }))
    );
}

#[test]
fn call_destination_preparation_keeps_attached_scan_budget_and_restores_ledger() {
    for projected in [false, true] {
        with_guarded_result_context(projected, |fixture, call| {
            // Zero work is available after genuine construction on the same
            // ledger; construction does not attach a replacement ledger.
            let (constructor_work, constructor_storage, constructor_peak) = {
                let mut baseline_work =
                    fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut baseline_budget =
                    ArgumentBudgetV1::new(&mut baseline_work, 16 * 1024 * 1024);
                baseline_budget.reserve_storage(37).unwrap();
                let baseline = SemanticFunctionLoweringV1::new(
                    fixture.types,
                    fixture.callables,
                    fixture.function,
                    SemanticParameterBindingsV1 {
                        declarations: &[],
                        values: &[],
                        types: &[],
                        local_bindings: None,
                    },
                    None,
                    None,
                    BTreeSet::new(),
                    1,
                    false,
                    32,
                    &mut baseline_budget,
                )?;
                drop(baseline);
                (
                    baseline_budget.work(),
                    baseline_budget.storage() - 37,
                    baseline_budget.peak_storage() - 37,
                )
            };
            eprintln!(
                "call-destination constructor prefix: projected={projected} work={constructor_work} storage={constructor_storage} peak={constructor_peak}"
            );
            let mut work =
                fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(constructor_work);
            let mut budget = ArgumentBudgetV1::new(&mut work, 37 + constructor_peak);
            budget.reserve_storage(37).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            let mut lowering = SemanticFunctionLoweringV1::new(
                fixture.types,
                fixture.callables,
                fixture.function,
                SemanticParameterBindingsV1 {
                    declarations: &[],
                    values: &[],
                    types: &[],
                    local_bindings: None,
                },
                None,
                None,
                BTreeSet::new(),
                1,
                false,
                32,
                &mut budget,
            )?;
            assert!(lowering.execution.is_none());
            assert!(lowering.scoped_memory.is_none());
            let attached = lowering.emission_work.as_deref().unwrap();
            assert_eq!(attached.emission_service_work_v1(), Some(constructor_work));
            assert_eq!(attached.storage(), 37 + constructor_storage);
            assert!(attached.work_ledger_identity_v1() == ledger);
            let next_value = lowering.next_value;
            let mut operations = Vec::new();
            let result = lowering.prepare_call_destination_v1(
                SemanticBlockIdV1::from_index(0),
                call.destination().unwrap().place(),
                &mut operations,
            );
            if projected {
                assert!(matches!(
                    result,
                    Err(
                        ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                            ArgumentResourceV1::Work(_)
                        )
                    )
                ));
            } else {
                assert!(matches!(
                    result,
                    Ok(PreparedSemanticCallDestinationV1::Unprojected)
                ));
            }
            assert!(lowering.emission_work.is_some());
            assert!(operations.is_empty());
            assert_eq!(lowering.emitted_operations, 0);
            assert_eq!(lowering.next_value, next_value);
            drop(lowering);
            assert_eq!(budget.storage(), 37 + constructor_storage);
            assert_eq!(budget.work(), constructor_work);
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn guarded_result_store_reuses_the_exact_load_and_branch_predicate() {
    for projected in [false, true] {
        with_guarded_result_context(projected, |lowering, call| {
            let mut operations = Vec::new();
            install_result_slot(lowering, projected, &mut operations)?;
            // Scalar: Alloca + eight intrinsic operations + result store.
            // Projected: two Allocas + seed store + address load + eight + one.
            let exact = if projected { 13 } else { 10 };
            lowering.max_operations = exact;
            let terminator =
                lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations)?;
            assert_guarded_result(&operations, &terminator, projected);
            assert_eq!(lowering.emitted_operations, exact);
            assert!(lowering.retained_local_initialized.contains(&3));
            if projected {
                assert!(lowering.locals[3].is_none());
            } else {
                assert_eq!(
                    lowering.locals[3].as_ref().unwrap().value().unwrap(),
                    (ValueId(10), Type::Scalar(ScalarType::U32))
                );
            }
            Ok(())
        })
        .unwrap();
    }
}

#[test]
fn one_under_operation_limit_denies_result_store_before_bookkeeping() {
    for projected in [false, true] {
        with_guarded_result_context(projected, |lowering, call| {
            let mut operations = Vec::new();
            install_result_slot(lowering, projected, &mut operations)?;
            let exact = if projected { 13 } else { 10 };
            lowering.max_operations = exact - 1;
            assert!(matches!(lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations),
                Err(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::Operations, actual, limit,
                }) if actual == exact && limit == exact - 1));
            assert_eq!(operations.len(), exact - 1);
            assert_eq!(lowering.emitted_operations, exact - 1);
            assert!(matches!(operations.last().unwrap().kind, OperationKind::GuardedLoad { .. }));
            assert!(operations.iter().all(|op| !matches!(op.kind, OperationKind::GuardedStore { .. })));
            assert_eq!(lowering.retained_local_initialized.contains(&3), projected);
            assert!(lowering.locals[3].is_none());
            Ok(())
        }).unwrap();
    }
}

#[test]
fn guarded_result_retains_the_existing_slot_type_rejection() {
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
        assert!(matches!(
            lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "retained-local value type differs from its private slot",
                ..
            })
        ));
        assert_eq!(operations.len(), 9);
        assert!(
            operations
                .iter()
                .all(|op| !matches!(op.kind, OperationKind::GuardedStore { .. }))
        );
        assert!(!lowering.retained_local_initialized.contains(&3));
        assert!(lowering.locals[3].is_none());
        Ok(())
    })
    .unwrap();
}
