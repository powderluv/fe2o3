use super::*;
use execution_binding_tests::{CONTEXT, MUT_CONTEXT, SHARED_CONTEXT, types};

const UNIT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(13);
const BLOCK: SemanticBlockIdV1 = SemanticBlockIdV1::from_index(0);
const INSTANCE: ProductionCallInstanceIdV1 = ProductionCallInstanceIdV1(7);

fn place(local: u32, ty: SemanticTypeIdV1) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap()
}

fn field(index: u32) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(
        SemanticLocalIdV1::from_index(1),
        vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Field(index), CONTEXT).unwrap()],
        CONTEXT,
    )
    .unwrap()
}

// These fixtures exercise the nominal binding layer below source-checked
// emission. They are not admitted owners or evidence of execution availability.
fn function(
    input: SemanticTypeIdV1,
    statements: Vec<SemanticStatementV1>,
) -> SemanticFunctionDeclV1 {
    function_with_reference(input, MUT_CONTEXT, statements)
}

fn function_with_reference(
    input: SemanticTypeIdV1,
    reference: SemanticTypeIdV1,
    statements: Vec<SemanticStatementV1>,
) -> SemanticFunctionDeclV1 {
    let source = SemanticSourceProvenanceV1::unavailable();
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([210; 32]),
        SemanticLayoutIdentityV1::from_sha256([211; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        1,
        vec![SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
            input,
            SemanticAbiPassModeV1::Ignore,
        ))],
        SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let locals = [
        (UNIT, SemanticLocalRoleV1::Return),
        (input, SemanticLocalRoleV1::Argument(0)),
        (reference, SemanticLocalRoleV1::Temporary),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (ty, role))| {
        SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([220 + index as u8; 32]),
            ty,
            role,
            source,
        )
    })
    .collect();
    SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([212; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([213; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([214; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([215; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([216; 32]),
        source,
        abi,
        locals,
        BLOCK,
        vec![
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([217; 32]),
                source,
                statements,
                SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

fn lowering<'source, 'service, 'work>(
    types: &'source [SemanticTypeDeclV1],
    function: &'source SemanticFunctionDeclV1,
    budget: &'service mut ArgumentBudgetV1<'work>,
) -> SemanticFunctionLoweringV1<'source, 'service> {
    SemanticFunctionLoweringV1::new(
        types,
        &[],
        function,
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
        128,
        budget,
    )
    .unwrap()
}

impl SemanticFunctionLoweringV1<'_, '_> {
    fn nominal_borrow_v29(
        &self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        ty: SemanticTypeIdV1,
        value: &SemanticRvalueKindV1,
        _operations: &mut Vec<Operation>,
    ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
        self.try_lower_execution_borrow_binding_v29(block, statement, ty, value, INSTANCE)?
            .ok_or_else(execution_availability_error_v29)
    }
}

fn context(types: &[SemanticTypeDeclV1], value: u32) -> SemanticExecutionBindingV29 {
    SemanticExecutionBindingV29::context(
        types,
        CONTEXT,
        ProductionCallOccurrenceV1 {
            caller: INSTANCE,
            block: BLOCK,
        },
        ValueId(value),
    )
    .unwrap()
}

fn borrowed(
    types: &[SemanticTypeDeclV1],
    reference: SemanticTypeIdV1,
    kind: SemanticBorrowKindV1,
) -> SemanticExecutionBorrowBindingV29 {
    SemanticExecutionBorrowBindingV29::from_source(
        types,
        SemanticExecutionBorrowSourceV29 {
            instance: ProductionCallInstanceIdV1(6),
            block: BLOCK,
            statement: 0,
            destination: &place(2, reference),
            kind,
            source: &place(1, CONTEXT),
        },
        &context(types, 100),
    )
    .unwrap()
}

fn dereferenced(local: u32) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(
        SemanticLocalIdV1::from_index(local),
        vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, CONTEXT).unwrap()],
        CONTEXT,
    )
    .unwrap()
}

fn refused<T>(result: Result<T, ProductionSemanticKirErrorV1>, expected: &str) {
    match result {
        Err(ProductionSemanticKirErrorV1::Unsupported { detail, .. }) => {
            assert_eq!(detail, expected);
        }
        Err(error) => panic!("unexpected transport error: {error:?}"),
        Ok(_) => panic!("execution transport unexpectedly accepted the operand"),
    }
}

fn missing<T>(result: Result<T, ProductionSemanticKirErrorV1>) {
    assert!(matches!(
        result,
        Err(ProductionSemanticKirErrorV1::MissingLocalDefinition { local: 1, .. })
    ));
}

#[test]
fn outer_emitter_refuses_nominal_execution_without_a_source_cursor() {
    let types = types();
    let function = function(CONTEXT, vec![]);
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(10_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &function, &mut budget);
    lowering.locals[1] = Some(SemanticValueBindingV1::Execution(context(&types, 100)));
    let mut operations = Vec::new();
    refused(
        lowering.lower_operand(
            BLOCK,
            Some(0),
            &SemanticOperandV1::Move(place(1, CONTEXT)),
            &mut operations,
        ),
        "execution availability differs from its source SSA instance",
    );
    refused(
        lowering.lower_source_operand_v29(
            BLOCK,
            Some(0),
            Some(ExecutionOperandV29::RvalueOperand(0)),
            &SemanticOperandV1::Move(place(1, CONTEXT)),
            &mut operations,
        ),
        "execution availability differs from its source SSA instance",
    );
    assert!(matches!(
        lowering.locals[1],
        Some(SemanticValueBindingV1::Execution(_))
    ));
    assert!(operations.is_empty());
}

#[test]
fn archive_check_rejects_changed_producers_borrows_and_missing_phi_bindings() {
    let types = types();
    let expected = context(&types, 100);
    let definition = SsaValueV1::BlockArgument {
        block: SsaBlockIdV1::new(0),
        variable: fe2o3_mir_model::SsaVariableIdV1::new(1),
    };
    let mut archive = SemanticSsaBindingsV1::from([(
        definition,
        SemanticValueBindingV1::Execution(expected.clone()),
    )]);
    let mut held = vec![
        None,
        Some(SemanticValueBindingV1::Execution(expected.clone())),
    ];
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(10_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 0);
    check_execution_archive_v29(&held, &archive, &place(1, CONTEXT), definition, &mut budget)
        .unwrap();
    let foreign = SemanticExecutionBindingV29::context(
        &types,
        CONTEXT,
        ProductionCallOccurrenceV1 {
            caller: ProductionCallInstanceIdV1(8),
            block: BLOCK,
        },
        ValueId(100),
    )
    .unwrap();
    held[1] = Some(SemanticValueBindingV1::Execution(foreign));
    assert!(
        check_execution_archive_v29(&held, &archive, &place(1, CONTEXT), definition, &mut budget)
            .is_err()
    );
    held[1] = Some(SemanticValueBindingV1::Execution(context(&types, 101)));
    assert!(
        check_execution_archive_v29(&held, &archive, &place(1, CONTEXT), definition, &mut budget)
            .is_err()
    );
    archive.clear();
    held[1] = Some(SemanticValueBindingV1::Execution(expected));
    assert!(
        check_execution_archive_v29(&held, &archive, &place(1, CONTEXT), definition, &mut budget)
            .is_err()
    );
    let borrowed = borrowed(&types, SHARED_CONTEXT, SemanticBorrowKindV1::Shared);
    archive.insert(
        definition,
        SemanticValueBindingV1::ExecutionBorrow(borrowed.clone()),
    );
    let mut wrong_instance = borrowed;
    wrong_instance.occurrence.instance = INSTANCE;
    held[1] = Some(SemanticValueBindingV1::ExecutionBorrow(wrong_instance));
    assert!(
        check_execution_archive_v29(
            &held,
            &archive,
            &place(1, SHARED_CONTEXT),
            definition,
            &mut budget
        )
        .is_err()
    );
}

#[test]
fn execution_whole_move_consumes_binding_and_copy_does_not() {
    let types = types();
    let function = function(CONTEXT, vec![]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &function, &mut emission_budget);
    let expected = context(&types, 100);
    lowering.locals[1] = Some(SemanticValueBindingV1::Execution(expected.clone()));
    let mut operations = Vec::new();
    refused(
        lowering.lower_operand_inner_v1(
            BLOCK,
            Some(0),
            &SemanticOperandV1::Copy(place(1, CONTEXT)),
            &mut operations,
        ),
        "owned execution roles cannot be copied",
    );
    assert!(
        matches!(&lowering.locals[1], Some(SemanticValueBindingV1::Execution(actual)) if *actual == expected)
    );
    let moved = lowering
        .lower_operand_inner_v1(
            BLOCK,
            Some(0),
            &SemanticOperandV1::Move(place(1, CONTEXT)),
            &mut operations,
        )
        .unwrap();
    assert!(matches!(moved, SemanticValueBindingV1::Execution(actual) if actual == expected));
    assert!(lowering.locals[1].is_none());
    missing(lowering.lower_operand_inner_v1(
        BLOCK,
        Some(1),
        &SemanticOperandV1::Move(place(1, CONTEXT)),
        &mut operations,
    ));
    assert!(operations.is_empty());
}

#[test]
fn execution_projected_moves_preserve_tombstones_and_reject_partial_whole_moves() {
    let mut types = types();
    let tuple = SemanticTypeIdV1::from_index(types.len() as u32);
    types.push(SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([230; 32]),
        SemanticLayoutIdentityV1::from_sha256([231; 32]),
        SemanticTypeLayoutV1::aggregate(
            Some(0),
            1,
            SemanticAggregateLayoutV1::new(vec![0, 0], vec![]).unwrap(),
        )
        .unwrap(),
        SemanticTypeShapeV1::Tuple(SemanticAggregateTypeV1::new(vec![CONTEXT, CONTEXT]).unwrap()),
    ));
    let function = function(tuple, vec![]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &function, &mut emission_budget);
    let first = context(&types, 100);
    let second = context(&types, 101);
    let aggregate = || {
        SemanticValueBindingV1::Aggregate(vec![
            SemanticValueBindingV1::Execution(first.clone()),
            SemanticValueBindingV1::Execution(second.clone()),
        ])
    };
    lowering.locals[1] = Some(aggregate());
    let mut operations = Vec::new();
    refused(
        lowering.lower_operand_inner_v1(
            BLOCK,
            Some(0),
            &SemanticOperandV1::Copy(place(1, tuple)),
            &mut operations,
        ),
        "execution-bearing aggregates cannot be copied",
    );
    let whole = lowering
        .lower_operand_inner_v1(
            BLOCK,
            Some(0),
            &SemanticOperandV1::Move(place(1, tuple)),
            &mut operations,
        )
        .unwrap();
    assert!(matches!(whole, SemanticValueBindingV1::Aggregate(ref fields) if fields.len() == 2));
    assert!(lowering.locals[1].is_none());
    lowering.locals[1] = Some(aggregate());
    let moved = lowering
        .lower_operand_inner_v1(
            BLOCK,
            Some(1),
            &SemanticOperandV1::Move(field(0)),
            &mut operations,
        )
        .unwrap();
    assert!(matches!(moved, SemanticValueBindingV1::Execution(actual) if actual == first));
    refused(
        lowering.lower_operand_inner_v1(
            BLOCK,
            Some(2),
            &SemanticOperandV1::Move(place(1, tuple)),
            &mut operations,
        ),
        "execution aggregate contains a moved value",
    );
    let moved = lowering
        .lower_operand_inner_v1(
            BLOCK,
            Some(3),
            &SemanticOperandV1::Move(field(1)),
            &mut operations,
        )
        .unwrap();
    assert!(matches!(moved, SemanticValueBindingV1::Execution(actual) if actual == second));
    assert!(
        matches!(&lowering.locals[1], Some(SemanticValueBindingV1::Aggregate(fields))
        if fields.iter().all(|field| matches!(field, SemanticValueBindingV1::MovedExecution)))
    );
    for index in 0..2 {
        refused(
            lowering.lower_operand_inner_v1(
                BLOCK,
                Some(4),
                &SemanticOperandV1::Move(field(index)),
                &mut operations,
            ),
            "execution operand has already been moved",
        );
    }
    refused(
        lowering.lower_operand_inner_v1(
            BLOCK,
            Some(5),
            &SemanticOperandV1::Move(place(1, tuple)),
            &mut operations,
        ),
        "execution aggregate contains a moved value",
    );
    assert!(operations.is_empty());
}

#[test]
fn execution_borrow_requires_the_retained_assignment_not_an_equal_clone() {
    let types = types();
    let source = SemanticSourceProvenanceV1::unavailable();
    let statement = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, MUT_CONTEXT),
            SemanticRvalueV1::new(
                MUT_CONTEXT,
                SemanticRvalueKindV1::Borrow {
                    place: place(1, CONTEXT),
                    kind: SemanticBorrowKindV1::Mutable,
                },
            ),
        )),
    );
    let empty = function(CONTEXT, vec![]);
    let function = function(CONTEXT, vec![statement]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &empty, &mut emission_budget);
    // The ordinary SSA gate rejects this borrow without an authenticated
    // consumer. Point only the hook's borrowed source declaration at the real
    // assignment; this is not a materialized or admitted source/SSA owner.
    lowering.function = &function;
    let expected = context(&types, 100);
    lowering.locals[1] = Some(SemanticValueBindingV1::Execution(expected.clone()));
    let SemanticStatementKindV1::Assign(assignment) = function.blocks()[0].statements()[0].kind()
    else {
        panic!("borrow fixture assignment")
    };
    let original = assignment.value().kind();
    let detached = (*original).clone();
    let mut operations = Vec::new();
    refused(
        lowering.nominal_borrow_v29(BLOCK, Some(0), MUT_CONTEXT, &detached, &mut operations),
        "execution borrow differs from its retained source assignment",
    );
    let borrowed = lowering
        .nominal_borrow_v29(BLOCK, Some(0), MUT_CONTEXT, original, &mut operations)
        .unwrap();
    let SemanticValueBindingV1::ExecutionBorrow(binding) = &borrowed else {
        panic!("expected nominal borrow")
    };
    assert_eq!(binding.borrowed(), &expected);
    assert_eq!(binding.source_local(), SemanticLocalIdV1::from_index(1));
    assert_eq!(
        binding.destination_local(),
        SemanticLocalIdV1::from_index(2)
    );
    assert_eq!(
        binding.occurrence(),
        SemanticExecutionBorrowOccurrenceV29 {
            instance: INSTANCE,
            block: BLOCK,
            statement: 0
        }
    );
    assert!(
        matches!(&lowering.locals[1], Some(SemanticValueBindingV1::Execution(actual)) if *actual == expected)
    );
    lowering.locals[2] = Some(borrowed);
    refused(
        lowering.lower_operand_inner_v1(
            BLOCK,
            Some(1),
            &SemanticOperandV1::Copy(place(2, MUT_CONTEXT)),
            &mut operations,
        ),
        "mutable execution borrows cannot be copied",
    );
    assert!(matches!(
        lowering
            .lower_operand_inner_v1(
                BLOCK,
                Some(1),
                &SemanticOperandV1::Move(place(2, MUT_CONTEXT)),
                &mut operations
            )
            .unwrap(),
        SemanticValueBindingV1::ExecutionBorrow(_)
    ));
    assert!(lowering.locals[2].is_none());
    assert!(operations.is_empty());
}

#[test]
fn ordinary_operands_without_a_cursor_still_require_the_exact_shared_work_budget() {
    #[derive(Debug)]
    enum Expected {
        Success,
        Work,
        Storage,
    }
    const FLOOR: usize = 37;
    let types = types();
    let scalar = |id| SemanticValueBindingV1::Value {
        id: ValueId(id),
        ty: Type::Scalar(ScalarType::U32),
    };
    // Optional array classification pays one lookup and its presence check.
    // Strict legacy lookup separately checks the original-object range and the
    // legacy key, then pays their two representation checks, even for empty maps.
    const OPTIONAL_ARRAY_PROBE: usize = 32 + 1;
    const STRICT_LEGACY_PROBE: usize = 2 * 32 + 2;
    const SLOT_LOOKUPS: usize = OPTIONAL_ARRAY_PROBE + STRICT_LEGACY_PROBE;
    for (input, binding, exact, storage) in [
        (
            SemanticTypeIdV1::from_index(2),
            scalar(77),
            6 + SLOT_LOOKUPS,
            0,
        ),
        (
            SemanticTypeIdV1::from_index(8),
            SemanticValueBindingV1::Aggregate(vec![scalar(77), scalar(78)]),
            20 + SLOT_LOOKUPS,
            2 * std::mem::size_of::<SemanticValueBindingV1>(),
        ),
    ] {
        let function = function(input, vec![]);
        let expected = binding.values().unwrap();
        let (constructor_work, constructor_storage, constructor_peak) = {
            let mut baseline_work =
                fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
            let mut baseline_budget =
                ArgumentBudgetV1::new(&mut baseline_work, 16 * 1024 * 1024);
            baseline_budget.reserve_storage(FLOOR).unwrap();
            let baseline = lowering(&types, &function, &mut baseline_budget);
            drop(baseline);
            (
                baseline_budget.work(),
                baseline_budget.storage() - FLOOR,
                baseline_budget.peak_storage() - FLOOR,
            )
        };
        eprintln!(
            "ordinary-operand constructor prefix: input={input:?} work={constructor_work} storage={constructor_storage} peak={constructor_peak}"
        );
        // Keep constructor scratch headroom occupied as an explicit test floor.
        // Constructor-owned storage is never released or replaced. The original
        // operation allowances therefore still test zero/one-under/exact storage.
        let scratch_floor = constructor_peak.checked_sub(constructor_storage).unwrap();
        let storage_floor = FLOOR + constructor_peak;
        let mut cases = vec![
            (0, storage, Expected::Work),
            (exact - 1, storage, Expected::Work),
            (exact, storage, Expected::Success),
        ];
        if storage != 0 {
            cases.extend([
                (exact, 0, Expected::Storage),
                (exact, storage - 1, Expected::Storage),
            ]);
        }
        for (limit, storage_limit, outcome) in cases {
            let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(
                constructor_work + limit,
            );
            let mut budget = ArgumentBudgetV1::new(&mut work, storage_floor + storage_limit);
            budget.reserve_storage(FLOOR).unwrap();
            let ledger = budget.work_ledger_identity_v1();
            let mut lowering = lowering(&types, &function, &mut budget);
            let attached = lowering.emission_work.as_deref_mut().unwrap();
            assert_eq!(attached.emission_service_work_v1(), Some(constructor_work));
            assert_eq!(attached.storage(), FLOOR + constructor_storage);
            assert!(attached.work_ledger_identity_v1() == ledger);
            attached.reserve_storage(scratch_floor).unwrap();
            assert_eq!(attached.storage(), storage_floor);
            lowering.locals[1] = Some(binding.clone());
            assert!(lowering.execution.is_none());
            let next_value = lowering.next_value;
            let mut operations = Vec::new();
            let result = lowering.lower_source_operand_v29(
                BLOCK,
                Some(0),
                None,
                &SemanticOperandV1::Copy(place(1, input)),
                &mut operations,
            );
            assert_eq!(
                lowering.locals[1].as_ref().unwrap().values().unwrap(),
                expected
            );
            assert_eq!(lowering.next_value, next_value);
            assert!(operations.is_empty());
            assert!(
                lowering
                    .emission_work
                    .as_deref()
                    .unwrap()
                    .work_ledger_identity_v1()
                    == ledger
            );
            drop(lowering);
            let mut failed_work = None;
            match (outcome, result) {
                (Expected::Success, Ok(copied)) => {
                    assert_eq!(copied.values().unwrap(), expected);
                    let live = match &copied {
                        SemanticValueBindingV1::Aggregate(fields) => {
                            fields.capacity() * std::mem::size_of::<SemanticValueBindingV1>()
                        }
                        _ => 0,
                    };
                    assert_eq!(live, storage);
                    assert_eq!(budget.storage(), storage_floor + live);
                    assert_eq!(budget.work(), constructor_work + exact);
                    drop(copied);
                    budget.release_storage(live).unwrap();
                }
                (
                    Expected::Work,
                    Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                        ArgumentResourceV1::Work(error),
                    )),
                ) => {
                    assert_eq!(
                        error.actual(),
                        constructor_work + if limit == 0 { 4 } else { exact },
                    );
                    assert_eq!(error.limit(), constructor_work + limit);
                    failed_work = Some(error.actual());
                    if limit == exact - 1 {
                        assert_eq!(budget.peak_storage(), storage_floor + storage);
                    }
                }
                (
                    Expected::Storage,
                    Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                        ArgumentResourceV1::Storage(_),
                    )),
                ) => {
                    assert_eq!(budget.work(), constructor_work + 16 + SLOT_LOOKUPS);
                    assert_eq!(budget.failed_storage(), Some(storage_floor + storage));
                }
                (outcome, result) => panic!("expected {outcome:?}, got {result:?}"),
            }
            assert_eq!(budget.storage(), storage_floor);
            assert!(budget.work() <= constructor_work + limit);
            assert_eq!(work.failed_work(), failed_work);
        }
    }
}

#[test]
fn emission_ledger_is_restored_after_consumer_unwind() {
    let types = types();
    let function = function(CONTEXT, vec![]);
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
    let original_ledger = budget.work_ledger_identity_v1();
    let mut lowering = lowering(&types, &function, &mut budget);
    let attached = lowering.emission_work.as_deref().unwrap();
    let original_work = attached.emission_service_work_v1().unwrap();
    let original_storage = attached.storage();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = lowering.with_emission_budget_v1::<()>(|this, meter| {
            assert!(this.emission_work.is_none());
            meter.charge_work(7)?;
            panic!("emission callback");
        });
    }));
    assert!(panic.is_err());
    assert!(
        lowering
            .emission_work
            .as_deref()
            .unwrap()
            .work_ledger_identity_v1()
            == original_ledger
    );
    lowering
        .with_emission_budget_v1(|this, meter| {
            assert!(this.emission_work.is_none());
            meter.charge_work(3)
        })
        .unwrap();
    assert!(lowering.emission_work.is_some());
    drop(lowering);
    assert_eq!(budget.work(), original_work + 10);
    assert_eq!(budget.storage(), original_storage);
}

#[test]
fn execution_storage_dead_prevents_later_operand_use() {
    let types = types();
    let statement = SemanticStatementV1::new(
        SemanticSourceProvenanceV1::unavailable(),
        SemanticStatementKindV1::StorageDead(SemanticLocalIdV1::from_index(1)),
    );
    let function = function(CONTEXT, vec![statement]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &function, &mut emission_budget);
    lowering.locals[1] = Some(SemanticValueBindingV1::Execution(context(&types, 100)));
    let mut operations = Vec::new();
    lowering
        .lower_statement(
            BLOCK,
            Some(0),
            function.blocks()[0].statements()[0].kind(),
            &mut operations,
        )
        .unwrap();
    assert!(lowering.locals[1].is_none());
    missing(lowering.lower_operand_inner_v1(
        BLOCK,
        Some(1),
        &SemanticOperandV1::Move(place(1, CONTEXT)),
        &mut operations,
    ));
    assert!(operations.is_empty());
}

#[test]
fn execution_dereference_retains_the_complete_borrow_and_cannot_move_out() {
    let types = types();
    for (reference, kind) in [
        (MUT_CONTEXT, SemanticBorrowKindV1::Mutable),
        (SHARED_CONTEXT, SemanticBorrowKindV1::Shared),
    ] {
        let function = function(reference, vec![]);
        let mut emission_work =
            fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
        let mut lowering = lowering(&types, &function, &mut emission_budget);
        let expected = borrowed(&types, reference, kind);
        lowering.locals[1] = Some(SemanticValueBindingV1::ExecutionBorrow(expected.clone()));
        let mut operations = Vec::new();
        let referent = lowering
            .resolve_place(BLOCK, Some(0), &dereferenced(1), &mut operations)
            .unwrap();
        for value in [&referent, &referent.clone()] {
            assert!(
                matches!(value, SemanticValueBindingV1::ExecutionReferent(actual) if *actual == expected)
            );
            assert!(value.values().is_err());
            assert!(!semantic_binding_can_restore_from_unique_source_v1(value));
        }
        for operand in [
            SemanticOperandV1::Copy(dereferenced(1)),
            SemanticOperandV1::Move(dereferenced(1)),
        ] {
            refused(
                lowering.lower_operand_inner_v1(BLOCK, Some(0), &operand, &mut operations),
                "execution operand requires exact logical field transport",
            );
            assert!(
                matches!(&lowering.locals[1], Some(SemanticValueBindingV1::ExecutionBorrow(actual)) if *actual == expected)
            );
        }
        assert_eq!(
            require_complete_execution_aggregate_v29(&SemanticValueBindingV1::Aggregate(vec![
                referent
            ])),
            Err("borrowed execution referents cannot become owned values")
        );
        assert!(operations.is_empty());
    }
}

#[test]
fn execution_referent_cannot_be_transported_as_an_owned_operand() {
    let types = types();
    let function = function(CONTEXT, vec![]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &function, &mut emission_budget);
    let expected = borrowed(&types, MUT_CONTEXT, SemanticBorrowKindV1::Mutable);
    lowering.locals[1] = Some(SemanticValueBindingV1::ExecutionReferent(expected.clone()));
    let mut operations = Vec::new();
    for operand in [
        SemanticOperandV1::Copy(place(1, CONTEXT)),
        SemanticOperandV1::Move(place(1, CONTEXT)),
    ] {
        refused(
            lowering.lower_operand_inner_v1(BLOCK, Some(0), &operand, &mut operations),
            "borrowed execution referents cannot become owned values",
        );
        assert!(
            matches!(&lowering.locals[1], Some(SemanticValueBindingV1::ExecutionReferent(actual)) if *actual == expected)
        );
    }
    assert!(operations.is_empty());
}

#[test]
fn execution_reborrow_hook_requires_retained_source_and_preserves_parent() {
    let types = types();
    for (parent_reference, parent_kind) in [
        (MUT_CONTEXT, SemanticBorrowKindV1::Mutable),
        (SHARED_CONTEXT, SemanticBorrowKindV1::Shared),
    ] {
        for (reference, kind) in [
            (MUT_CONTEXT, SemanticBorrowKindV1::Mutable),
            (SHARED_CONTEXT, SemanticBorrowKindV1::Shared),
        ] {
            let statement = SemanticStatementV1::new(
                SemanticSourceProvenanceV1::unavailable(),
                SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                    place(2, reference),
                    SemanticRvalueV1::new(
                        reference,
                        SemanticRvalueKindV1::Borrow {
                            place: dereferenced(1),
                            kind,
                        },
                    ),
                )),
            );
            let empty = function_with_reference(parent_reference, reference, vec![]);
            let source = function_with_reference(parent_reference, reference, vec![statement]);
            let mut emission_work =
                fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
            let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
            let mut lowering = lowering(&types, &empty, &mut emission_budget);
            // Only the retained-assignment hook is exercised, as in the borrow fixture above.
            lowering.function = &source;
            let parent = borrowed(&types, parent_reference, parent_kind);
            lowering.locals[1] = Some(SemanticValueBindingV1::ExecutionBorrow(parent.clone()));
            let SemanticStatementKindV1::Assign(assignment) =
                source.blocks()[0].statements()[0].kind()
            else {
                unreachable!()
            };
            let original = assignment.value().kind();
            let detached = original.clone();
            let mut operations = Vec::new();
            refused(
                lowering.nominal_borrow_v29(BLOCK, Some(0), reference, &detached, &mut operations),
                "execution borrow differs from its retained source assignment",
            );
            let result =
                lowering.nominal_borrow_v29(BLOCK, Some(0), reference, original, &mut operations);
            if parent_kind == SemanticBorrowKindV1::Shared && kind == SemanticBorrowKindV1::Mutable
            {
                refused(result, "execution reborrow cannot strengthen shared access");
            } else {
                let SemanticValueBindingV1::ExecutionBorrow(child) = result.unwrap() else {
                    panic!("expected reborrow");
                };
                assert_eq!(child.parent, Some(parent.occurrence()));
                assert_eq!(
                    child.occurrence(),
                    SemanticExecutionBorrowOccurrenceV29 {
                        instance: INSTANCE,
                        block: BLOCK,
                        statement: 0,
                    }
                );
                assert_ne!(child.occurrence(), parent.occurrence());
                assert_eq!(child.borrowed(), parent.borrowed());
                assert_eq!(child.reference_type(), reference);
                assert_eq!(child.kind(), kind);
                assert_eq!(child.source_local(), SemanticLocalIdV1::from_index(1));
                assert_eq!(child.destination_local(), SemanticLocalIdV1::from_index(2));
            }
            assert!(
                matches!(&lowering.locals[1], Some(SemanticValueBindingV1::ExecutionBorrow(actual)) if *actual == parent)
            );
            assert!(operations.is_empty());
        }
    }
}

#[test]
fn execution_reborrow_uses_the_actual_root_reference_type() {
    let types = types();
    let statement = SemanticStatementV1::new(
        SemanticSourceProvenanceV1::unavailable(),
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, SHARED_CONTEXT),
            SemanticRvalueV1::new(
                SHARED_CONTEXT,
                SemanticRvalueKindV1::Borrow {
                    place: dereferenced(1),
                    kind: SemanticBorrowKindV1::Shared,
                },
            ),
        )),
    );
    let empty = function_with_reference(SHARED_CONTEXT, SHARED_CONTEXT, vec![]);
    let source = function_with_reference(SHARED_CONTEXT, SHARED_CONTEXT, vec![statement]);
    let mut emission_work =
        fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, 16 * 1024 * 1024);
    let mut lowering = lowering(&types, &empty, &mut emission_budget);
    lowering.function = &source;
    let parent = borrowed(&types, MUT_CONTEXT, SemanticBorrowKindV1::Mutable);
    lowering.locals[1] = Some(SemanticValueBindingV1::ExecutionBorrow(parent.clone()));
    let SemanticStatementKindV1::Assign(assignment) = source.blocks()[0].statements()[0].kind()
    else {
        unreachable!()
    };
    let mut operations = Vec::new();
    refused(
        lowering.nominal_borrow_v29(
            BLOCK,
            Some(0),
            SHARED_CONTEXT,
            assignment.value().kind(),
            &mut operations,
        ),
        "execution borrow nominal reference type changed",
    );
    assert!(
        matches!(&lowering.locals[1], Some(SemanticValueBindingV1::ExecutionBorrow(actual)) if *actual == parent)
    );
    assert!(operations.is_empty());
}
