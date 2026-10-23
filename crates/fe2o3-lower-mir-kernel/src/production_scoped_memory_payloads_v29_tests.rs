use super::*;

#[path = "production_execution_rvalue_archive_v30_tests.rs"]
mod rvalue_archive_tests;

thread_local! {
    static PAYLOAD_COMPLETED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static PAYLOAD_MUTATION: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn scalar_fixture() -> ScopedFixture {
    ScopedFixture::Initialization(InitializationFixtureV29 {
        kill: Some(InitializationKillV29::SelfMove),
        address_read: false,
        reinitialize: true,
        ..InitializationFixtureV29::default()
    })
}

fn inspect_payloads(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut helpers = 0;
    let mut loads = 0;
    let mut constant_stores = 0;
    let mut moved_stores = 0;
    let mut promoted_stores = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        let source = instances.instance(item.instance).unwrap().declaration();
        let occurrences = instances.occurrences(item.instance).unwrap();
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
        for (ordinal, row) in anchors.rows.iter().enumerate() {
            let ScopedMemoryAnchorKindV29::Access {
                payload: Some(payload),
                ..
            } = row.kind
            else {
                continue;
            };
            let operation = &lowered
                .function
                .body
                .as_ref()
                .unwrap()
                .blocks
                .iter()
                .find(|block| block.id == row.block)
                .unwrap()
                .operations[row.position];
            check_scoped_payload_v29(source, &occurrences, row, operation, budget)?;
            match payload {
                ScopedMemoryPayloadV29::Load { result, read } => {
                    loads += 1;
                    assert_eq!(operation.results.len(), 1);
                    assert_eq!(operation.results[0].id, result);
                    assert_eq!(read.prefix, 0);
                    assert!(matches!(
                        read.occurrence,
                        ScopedMemoryOccurrenceV29::Retained { .. }
                    ));
                    let original = scoped_payload_place_v29(source, read.site, read.role).unwrap();
                    assert_eq!(original.local().index(), 2);
                }
                ScopedMemoryPayloadV29::Store {
                    value,
                    source:
                        ScopedMemoryStoreSourceV29::Operand {
                            site,
                            role,
                            ty,
                            source: capture,
                        },
                } => {
                    assert!(
                        matches!(operation.kind, OperationKind::Store { value: actual, .. } if actual == value)
                    );
                    match capture {
                        ScopedMemoryOperandSourceV29::Constant => {
                            constant_stores += 1;
                            assert!(matches!(
                                scoped_source_operand_v29(source, site, role),
                                Some(SemanticOperandV1::Constant(_))
                            ));
                        }
                        ScopedMemoryOperandSourceV29::Memory {
                            occurrence: ScopedMemoryOccurrenceV29::Retained { event },
                            access,
                        } => {
                            moved_stores += 1;
                            assert!(
                                matches!(scoped_source_operand_v29(source, site, role), Some(SemanticOperandV1::Move(place)) if place.local().index() == 2)
                            );
                            assert_eq!(
                                occurrences.events()[event].role(),
                                ExecutionEventV29::BaseUse
                            );
                            check_scoped_payload_memory_v29(
                                source,
                                anchors,
                                ordinal,
                                row,
                                value,
                                site,
                                role,
                                ty,
                                ScopedMemoryOccurrenceV29::Retained { event },
                                access,
                                budget,
                            )?;
                        }
                        ScopedMemoryOperandSourceV29::Place(
                            capture @ ScopedMemoryOccurrenceV29::Promoted { .. },
                        ) => {
                            promoted_stores += 1;
                            let original = scoped_payload_place_v29(source, site, role).unwrap();
                            check_scoped_payload_archive_v29(
                                &lowered.execution_observation.as_ref().unwrap().bindings,
                                original,
                                capture,
                                value,
                                budget,
                            )?;
                            let other = slots
                                .instances
                                .iter()
                                .find(|other| {
                                    other.function == item.function
                                        && other.instance != item.instance
                                })
                                .unwrap();
                            let foreign = emitted[other.instance.index()].as_ref().unwrap();
                            assert!(matches!(
                                check_scoped_payload_archive_v29(
                                    &foreign.execution_observation.as_ref().unwrap().bindings,
                                    original,
                                    capture,
                                    value,
                                    budget
                                ),
                                Err(ProductionSemanticKirErrorV1::Unsupported {
                                    detail: "scoped memory anchors differ from their source instance",
                                    ..
                                })
                            ));
                        }
                        _ => panic!("retained scalar source unexpectedly became promoted"),
                    }
                }
                _ => {}
            }
        }
    }
    assert_eq!(helpers, 2);
    assert!(loads >= 2 && constant_stores >= 2 && moved_stores >= 2 && promoted_stores >= 2);
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    observe(lifecycle, instances, emitted, slots, budget)
}

#[test]
fn real_repeated_scalar_instances_capture_load_results_and_pre_move_store_sources() {
    PAYLOAD_COMPLETED.set(0);
    let (result, _, _) = run(false, scalar_fixture(), inspect_payloads, LIMIT, LIMIT);
    assert!(is_stopped(&result), "{result:?}");
    assert_eq!(PAYLOAD_COMPLETED.get(), 1);
}

fn inspect_mutation(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let item = slots
        .instances
        .iter()
        .find(|item| item.function.index() == 3)
        .unwrap();
    let source = instances.instance(item.instance).unwrap().declaration();
    let occurrences = instances.occurrences(item.instance).unwrap();
    let lowered = emitted[item.instance.index()].as_ref().unwrap();
    let mutation = PAYLOAD_MUTATION.get();
    let row = lowered
        .scoped_memory_anchors
        .as_ref()
        .unwrap()
        .rows
        .iter()
        .find(|row| {
            matches!(
                row.kind,
                ScopedMemoryAnchorKindV29::Access {
                    payload: Some(ScopedMemoryPayloadV29::Load { .. }),
                    ..
                }
            ) && mutation < 4
                || matches!(
                    row.kind,
                    ScopedMemoryAnchorKindV29::Access {
                        payload: Some(ScopedMemoryPayloadV29::Store {
                            source: ScopedMemoryStoreSourceV29::Operand {
                                source: ScopedMemoryOperandSourceV29::Memory { .. },
                                ..
                            },
                            ..
                        }),
                        ..
                    }
                ) && mutation >= 4
        })
        .unwrap();
    let operation = &lowered
        .function
        .body
        .as_ref()
        .unwrap()
        .blocks
        .iter()
        .find(|block| block.id == row.block)
        .unwrap()
        .operations[row.position];
    let mut changed = *row;
    let ScopedMemoryAnchorKindV29::Access {
        payload: Some(payload),
        ..
    } = &mut changed.kind
    else {
        unreachable!();
    };
    match payload {
        ScopedMemoryPayloadV29::AtomicRmw { .. } => {
            panic!("ordinary payload mutation fixture has no atomics");
        }
        ScopedMemoryPayloadV29::IndexLoad { .. } => {
            panic!("ordinary payload mutation fixture has no projection index loads");
        }
        ScopedMemoryPayloadV29::Load { result, read } => match mutation {
            0 => *result = ValueId(u32::MAX),
            1 => read.prefix = u32::MAX,
            2 => read.ty = SemanticTypeIdV1::from_index(0),
            3 => read.occurrence = ScopedMemoryOccurrenceV29::Retained { event: usize::MAX },
            _ => unreachable!(),
        },
        ScopedMemoryPayloadV29::Store { value, source } => match mutation {
            4 => *value = ValueId(u32::MAX),
            5 => {
                let ScopedMemoryStoreSourceV29::Operand { role, .. } = source else {
                    unreachable!();
                };
                *role = ExecutionOperandV29::StoreDestination;
            }
            6 => {
                let ScopedMemoryStoreSourceV29::Operand { source, .. } = source else {
                    unreachable!();
                };
                *source = ScopedMemoryOperandSourceV29::Constant;
            }
            7 => {
                let ScopedMemoryStoreSourceV29::Operand { source, .. } = source else {
                    unreachable!();
                };
                *source =
                    ScopedMemoryOperandSourceV29::Place(ScopedMemoryOccurrenceV29::Retained {
                        event: usize::MAX,
                    });
            }
            _ => unreachable!(),
        },
    }
    let error =
        check_scoped_payload_v29(source, &occurrences, &changed, operation, budget).unwrap_err();
    assert!(
        matches!(
            error,
            ProductionSemanticKirErrorV1::Unsupported {
                detail: "scoped memory anchors differ from their source instance",
                ..
            }
        ),
        "{mutation}: {error:?}"
    );
    // The original row remains valid and the failed query cannot mutate it.
    check_scoped_payload_v29(source, &occurrences, row, operation, budget)?;
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    observe(lifecycle, instances, emitted, slots, budget)
}

#[test]
fn payload_census_rejects_result_rhs_role_prefix_type_and_occurrence_substitution() {
    for mutation in 0..8 {
        PAYLOAD_MUTATION.set(mutation);
        PAYLOAD_COMPLETED.set(0);
        let (result, _, _) = run(false, scalar_fixture(), inspect_mutation, LIMIT, LIMIT);
        assert!(is_stopped(&result), "{mutation}: {result:?}");
        assert_eq!(PAYLOAD_COMPLETED.get(), 1, "{mutation}");
    }
}

fn inspect_memory_transport(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let item = slots
        .instances
        .iter()
        .find(|row| row.function.index() == 3)
        .unwrap();
    let source = instances.instance(item.instance).unwrap().declaration();
    let lowered = emitted[item.instance.index()].as_ref().unwrap();
    let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
    let (ordinal, row) = anchors
        .rows
        .iter()
        .enumerate()
        .find(|(_, row)| {
            matches!(
                row.kind,
                ScopedMemoryAnchorKindV29::Access {
                    payload: Some(ScopedMemoryPayloadV29::Store {
                        source: ScopedMemoryStoreSourceV29::Operand {
                            source: ScopedMemoryOperandSourceV29::Memory { .. },
                            ..
                        },
                        ..
                    }),
                    ..
                }
            )
        })
        .unwrap();
    let ScopedMemoryAnchorKindV29::Access {
        payload:
            Some(ScopedMemoryPayloadV29::Store {
                value,
                source:
                    ScopedMemoryStoreSourceV29::Operand {
                        site,
                        role,
                        ty,
                        source: ScopedMemoryOperandSourceV29::Memory { occurrence, access },
                    },
            }),
        ..
    } = row.kind
    else {
        unreachable!();
    };
    check_scoped_payload_memory_v29(
        source, anchors, ordinal, row, value, site, role, ty, occurrence, access, budget,
    )?;
    for mutation in 0..9 {
        let mut changed = ScopedMemoryAnchorsV29 {
            subject: anchors.subject,
            placement: anchors.placement,
            rows: anchors.rows.clone(),
            objects: anchors.objects.clone(),
            object_components: anchors.object_components.clone(),
            zero_objects: anchors.zero_objects.clone(),
            compiler_enum: anchors.compiler_enum.clone(),
        };
        let rows = &mut changed.rows;
        let mut changed_row = *row;
        let mut changed_access = access;
        let mut changed_value = value;
        match mutation {
            0 => changed_access = ordinal,
            1 => changed_access = usize::MAX,
            2 => rows[access].position = row.position,
            3 => rows[access].block = BlockId(u32::MAX),
            4 => {
                rows[access].source = Some(ScopedMemoryFrameV29::operand(
                    site,
                    Some(ExecutionOperandV29::StoreDestination),
                ))
            }
            5 | 6 | 7 => {
                let ScopedMemoryAnchorKindV29::Access {
                    payload: Some(ScopedMemoryPayloadV29::Load { result, read }),
                    ..
                } = &mut rows[access].kind
                else {
                    unreachable!();
                };
                match mutation {
                    5 => read.prefix += 1,
                    6 => {
                        read.occurrence = ScopedMemoryOccurrenceV29::Retained { event: usize::MAX }
                    }
                    7 => *result = ValueId(u32::MAX),
                    _ => unreachable!(),
                }
            }
            8 => {
                // A different real same-typed scalar is not the Load result,
                // even when an alias/origin query could relate its computation.
                changed_value = lowered
                    .function
                    .body
                    .as_ref()
                    .unwrap()
                    .blocks
                    .iter()
                    .flat_map(|block| block.operations.iter())
                    .flat_map(|operation| operation.results.iter())
                    .find(|result| result.id != value && result.ty == Type::Scalar(ScalarType::U32))
                    .unwrap()
                    .id;
                if let ScopedMemoryAnchorKindV29::Access {
                    payload: Some(ScopedMemoryPayloadV29::Store { value, .. }),
                    ..
                } = &mut changed_row.kind
                {
                    *value = changed_value;
                }
            }
            _ => unreachable!(),
        }
        let error = check_scoped_payload_memory_v29(
            source,
            &changed,
            ordinal,
            &changed_row,
            changed_value,
            site,
            role,
            ty,
            occurrence,
            changed_access,
            budget,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                ProductionSemanticKirErrorV1::Unsupported {
                    detail: "scoped memory anchors differ from their source instance",
                    ..
                }
            ),
            "{mutation}: {error:?}"
        );
    }
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    observe(lifecycle, instances, emitted, slots, budget)
}

#[test]
fn actual_memory_transport_rejects_wrong_row_order_prefix_occurrence_and_same_typed_value() {
    PAYLOAD_COMPLETED.set(0);
    let (result, _, _) = run(
        false,
        scalar_fixture(),
        inspect_memory_transport,
        LIMIT,
        LIMIT,
    );
    assert!(is_stopped(&result), "{result:?}");
    assert_eq!(PAYLOAD_COMPLETED.get(), 1);
}

// The adapter forwards custody to the original concrete slot. Its only extra
// behavior is a test-time mutation using that same borrowed concrete budget.
struct PayloadTestBudgetV29<'budget, 'work, F> {
    original: &'budget mut ArgumentBudgetV1<'work>,
    armed: &'budget std::cell::Cell<bool>,
    grow: F,
}
impl<'work, F: FnMut(&mut ArgumentBudgetV1<'work>)> SemanticEmissionBudgetV1
    for PayloadTestBudgetV29<'_, 'work, F>
{
    fn work_ledger_identity_v1(&self) -> fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1 {
        self.original.work_ledger_identity_v1()
    }
    fn charge_work(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        self.original.charge_work(amount)?;
        if self.armed.replace(false) {
            (self.grow)(self.original);
        }
        Ok(())
    }
    fn reserve_storage(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        Ok(self.original.reserve_storage(amount)?)
    }
    fn release_storage(&mut self, amount: usize) -> Result<(), ProductionSemanticKirErrorV1> {
        Ok(self.original.release_storage(amount)?)
    }
    fn storage(&self) -> usize {
        self.original.storage()
    }
    fn emission_service_work_v1(&self) -> Option<usize> {
        Some(self.original.work())
    }
    fn prepared_input_slot_v1(&self) -> Option<usize> {
        SemanticEmissionBudgetV1::prepared_input_slot_v1(self.original)
    }
    fn permits_prepared_input_refund_v1(
        &self,
        plan: Option<&SourceReferencePlanV29<'_, '_>>,
        slot: usize,
        ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
        required: usize,
        bytes: usize,
    ) -> bool {
        SemanticEmissionBudgetV1::permits_prepared_input_refund_v1(
            self.original,
            plan,
            slot,
            ledger,
            required,
            bytes,
        )
    }
    fn source_reference_owner_v29(
        &mut self,
        plan: &SourceReferencePlanV29<'_, '_>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        SemanticEmissionBudgetV1::source_reference_owner_v29(self.original, plan)
    }
    fn source_reference_representation_v29(
        &mut self,
        plan: &SourceReferencePlanV29<'_, '_>,
        loan: usize,
    ) -> Result<SourceReferenceRepresentationV29, ProductionSemanticKirErrorV1> {
        SemanticEmissionBudgetV1::source_reference_representation_v29(self.original, plan, loan)
    }
    fn source_reference_scalar_cell_v29(
        &mut self,
        plan: &SourceReferencePlanV29<'_, '_>,
        loan: usize,
    ) -> Result<Option<(usize, SourceReferenceScalarCellV29)>, ProductionSemanticKirErrorV1> {
        SemanticEmissionBudgetV1::source_reference_scalar_cell_v29(self.original, plan, loan)
    }
}

fn inspect_payload_scopes(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let item = slots
        .instances
        .iter()
        .find(|row| row.function.index() == 3)
        .unwrap();
    let instance = item.instance;
    let row = instances.instance(instance).unwrap();
    let local = SemanticLocalIdV1::from_index(2);
    let ty = row.declaration().locals()[2].ty();
    let mut layouts =
        source_storage_v29::SourceStorageLayoutsV29::new(instances.owner(), &[ty], budget)?;
    source_storage_v29::with_source_storage_root_v29(
        &mut layouts,
        instances,
        budget,
        |plan, root, budget| {
            let references = SourceReferenceEmissionV29::new(plan, budget)?;
            let scoped = with_source_reference_availability_v29(
                instances,
                instance,
                Some(&references),
                budget,
                |cursor, budget| {
                    let placement = SemanticEmissionPlacementV1 {
                        first_block: 17,
                        first_value: 200,
                    };
                    let abi = execution_instance_plan_v29(
                        instances,
                        instance,
                        FunctionId::new("payload_scope_probe"),
                        placement,
                        budget,
                    )?;
                    let mut producer = ExecutionLifecycleProducerV29::new(
                        lifecycle, instances, instance, placement, budget,
                    )?;
                    let semantic = instances.owner().source_semantic();
                    let armed = std::cell::Cell::new(false);
                    let grown = std::cell::Cell::new(0);
                    let mut meter = PayloadTestBudgetV29 {
                        original: budget,
                        armed: &armed,
                        grow: |budget: &mut ArgumentBudgetV1<'_>| {
                            let before = budget.storage();
                            let state = root.new_state(instance, local, budget).unwrap();
                            let path = root.root_path(ty, budget).unwrap();
                            root.mutate(
                                state,
                                path,
                                source_storage_v29::SourceStorageRootMutationV29::Initialize,
                                budget,
                            )
                            .unwrap();
                            assert!(root.is_initialized(state, path, budget).unwrap());
                            grown.set(budget.storage() - before);
                            assert!(grown.get() > 0);
                        },
                    };
                    let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                        semantic.types(),
                        semantic.callables(),
                        row.declaration(),
                        row.ssa(),
                        abi.correspondence_owner,
                        abi.semantic_function,
                        BTreeMap::new(),
                        BTreeMap::new(),
                        abi.result_types.clone(),
                        SemanticParameterBindingsV1 {
                            declarations: &abi.parameter_declarations,
                            values: &abi.parameter_values,
                            types: &abi.parameter_types,
                            local_bindings: None,
                        },
                        None,
                        Some([64, 1, 1]),
                        BTreeSet::new().into(),
                        1,
                        false,
                        1024,
                        PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(1, 1024)),
                        None,
                        CallReturnBufferV1::empty(),
                        Some(&mut meter),
                        placement,
                        Some(cursor),
                        Some(&mut producer),
                        None,
)?;
                    let extra = 13;
                    // Independent lexical envelope premise, not the production helper.
                    let header = extra
                        + std::mem::size_of::<ScopedInitializationSubjectV29>()
                        + std::mem::size_of::<
                            Option<source_storage_v29::SourceStorageRootGrowthV29<'_, '_, '_>>,
                        >()
                        + std::mem::size_of::<Result<(), ProductionSemanticKirErrorV1>>()
                        + std::mem::size_of::<
                            std::thread::Result<Result<(), ProductionSemanticKirErrorV1>>,
                        >();
                    let mut completed = 0;
                    for mode in 0..3 {
                        let entry = lowering.emission_work.as_deref().unwrap().storage();
                        let before_work = lowering
                            .emission_work
                            .as_deref()
                            .unwrap()
                            .emission_service_work_v1()
                            .unwrap();
                        grown.set(0);
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            lowering.with_scoped_payload_header_v29(extra, |lowering| {
                                assert_eq!(
                                    lowering.emission_work.as_deref().unwrap().storage(),
                                    entry + header
                                );
                                assert_eq!(
                                    lowering
                                        .emission_work
                                        .as_deref()
                                        .unwrap()
                                        .emission_service_work_v1()
                                        .unwrap()
                                        - before_work,
                                    4 + 8 + 4 + 4
                                );
                                armed.set(true);
                                lowering
                                    .with_emission_budget_v1(|_, budget| budget.charge_work(1))?;
                                completed += 1;
                                match mode {
                                    0 => Ok(()),
                                    1 => Err(unsupported(
                                        73,
                                        None,
                                        None,
                                        "selected payload callback error",
                                    )),
                                    2 => std::panic::panic_any(73_u32),
                                    _ => unreachable!(),
                                }
                            })
                        }));
                        assert_eq!(
                            lowering.emission_work.as_deref().unwrap().storage(),
                            entry + grown.get()
                        );
                        match (mode, result) {
                            (0, Ok(Ok(()))) => {}
                            (
                                1,
                                Ok(Err(ProductionSemanticKirErrorV1::Unsupported {
                                    function: 73,
                                    detail: "selected payload callback error",
                                    ..
                                })),
                            ) => {}
                            (2, Err(payload)) => {
                                assert_eq!(*payload.downcast::<u32>().unwrap(), 73)
                            }
                            (_, other) => {
                                panic!("payload scope changed callback result: {other:?}")
                            }
                        }
                    }
                    assert_eq!(completed, 3);
                    // Every new frame clears a prior hint, including a frame without an
                    // operand. Errors and panics cannot leave a reusable stale Load.
                    for mode in 0..3 {
                        lowering.scoped_memory.as_mut().unwrap().last_load = Some(usize::MAX);
                        let frame = ScopedMemoryFrameV29::operand(
                            execution_site_v29(SemanticBlockIdV1::from_index(0), Some(0)),
                            None,
                        );
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            lowering.with_scoped_memory_frame_v29(frame, |lowering| {
                                assert!(
                                    lowering.scoped_memory.as_ref().unwrap().last_load.is_none()
                                );
                                completed += 1;
                                match mode {
                                    0 => Ok(()),
                                    1 => Err(scoped_memory_error_v29()),
                                    2 => std::panic::panic_any(74_u32),
                                    _ => unreachable!(),
                                }
                            })
                        }));
                        match (mode, result) {
                            (0, Ok(Ok(()))) | (1, Ok(Err(_))) => {}
                            (2, Err(payload)) => {
                                assert_eq!(*payload.downcast::<u32>().unwrap(), 74)
                            }
                            (_, other) => panic!("frame changed callback result: {other:?}"),
                        }
                        let recorder = lowering.scoped_memory.as_ref().unwrap();
                        assert!(
                            recorder.last_load.is_none()
                                && recorder.frame.is_none()
                                && recorder.read_payload.is_none()
                        );
                    }
                    assert_eq!(completed, 6);
                    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
                    Ok(())
                },
            );
            references.abort_scope(instances, budget)?;
            scoped?;
            Ok(())
        },
    )?;
    layouts.release(budget)?;
    observe(lifecycle, instances, emitted, slots, budget)
}

#[test]
fn actual_payload_scopes_preserve_live_root_growth_headers_first_error_and_raw_panic() {
    PAYLOAD_COMPLETED.set(0);
    let (result, _, _) = run(
        false,
        scalar_fixture(),
        inspect_payload_scopes,
        LIMIT,
        LIMIT,
    );
    assert!(is_stopped(&result), "{result:?}");
    assert_eq!(PAYLOAD_COMPLETED.get(), 1);
}

fn inspect_call_payloads(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let item = slots
        .instances
        .iter()
        .find(|row| row.function == CALLBACK)
        .unwrap();
    let lowered = emitted[item.instance.index()].as_ref().unwrap();
    let source = instances.instance(item.instance).unwrap().declaration();
    let occurrences = instances.occurrences(item.instance).unwrap();
    let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
    let projected = !scoped_source_call_destination_v29(
        source,
        execution_site_v29(SemanticBlockIdV1::from_index(0), None),
    )
    .unwrap()
    .projections()
    .is_empty();
    let mut checked = 0;
    let mut typed = 0;
    let mut sites = [false; 3];
    for (anchor, row) in anchors.rows.iter().enumerate() {
        if row
            .source
            .is_none_or(|frame| frame.role != Some(ScopedMemoryRoleV29::CallResult))
        {
            continue;
        }
        let block = lowered
            .function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.id == row.block)
            .unwrap();
        let operation = &block.operations[row.position];
        let (value, site, ty, is_typed) = match row.kind {
            ScopedMemoryAnchorKindV29::Access {
                payload:
                    Some(ScopedMemoryPayloadV29::Store {
                        value,
                        source: ScopedMemoryStoreSourceV29::CallResult { site, ty },
                    }),
                ..
            } => {
                assert!(
                    matches!(operation.kind, OperationKind::Store { value: actual, .. } if actual == value)
                );
                check_scoped_payload_v29(source, &occurrences, row, operation, budget)?;
                (value, site, ty, false)
            }
            ScopedMemoryAnchorKindV29::Object(_) => {
                let payload = anchors.object_payload(row, budget)?;
                let ScopedObjectRoleV29::WriteValue {
                    destination,
                    value:
                        ScopedObjectValueOriginV29::Original(ScopedMemoryStoreSourceV29::CallResult {
                            site,
                            ty,
                        }),
                } = payload.role
                else {
                    panic!("typed result write must retain its original call producer");
                };
                let ScopedObjectOperationV29::WriteValue { address, value, .. } = payload.operation
                else {
                    panic!("typed call result must be an actual WriteValue");
                };
                let place = scoped_source_call_destination_v29(source, site).unwrap();
                assert_eq!(destination.projected_type, ty);
                assert_eq!(
                    destination.source,
                    ScopedObjectSourceV29::Place {
                        site,
                        role: ExecutionOperandV29::CallDestinationAddress,
                        local: place.local(),
                        prefix: place.projections().len() as u32,
                    }
                );
                assert_eq!(operation.kind, OperationKind::Storage(payload.operation));
                assert_eq!(payload.result, None);
                assert!(operation.results.is_empty());
                assert!(matches!(operation.kind,
                    OperationKind::Storage(ScopedObjectOperationV29::WriteValue { address: actual, value: stored, .. })
                        if actual == address && stored == value));
                anchors.check_object_source(source, &occurrences, anchor, row, payload, budget)?;
                payload.check_operation(operation, budget)?;
                typed += 1;
                (value, site, ty, true)
            }
            _ => panic!("actual call result is missing its producer capture"),
        };
        assert_eq!(
            scoped_source_call_destination_v29(source, site)
                .unwrap()
                .ty(),
            ty
        );
        let (original_block, statement) = scoped_memory_site_key_v29(site);
        assert_eq!(statement, None);
        let original_block = original_block as usize;
        assert!(original_block < sites.len());
        assert!(!std::mem::replace(&mut sites[original_block], true));
        assert_eq!(is_typed, projected && matches!(original_block, 0 | 2));
        assert_eq!(
            row.source,
            Some(ScopedMemoryFrameV29 {
                site,
                role: Some(ScopedMemoryRoleV29::CallResult)
            })
        );
        let mut producers = block.operations[..row.position]
            .iter()
            .filter(|operation| operation.results.iter().any(|result| result.id == value));
        let producer = producers
            .next()
            .expect("the returned value must be defined before its destination write");
        assert!(producers.next().is_none());
        assert_eq!(producer.results.len(), 1);
        assert_eq!(producer.results[0].ty, Type::Scalar(ScalarType::U32));
        if original_block < 2 {
            assert!(matches!(producer.kind, OperationKind::Call { .. }));
        }
        checked += 1;
    }
    assert_eq!(checked, 3);
    assert_eq!(sites, [true; 3]);
    assert_eq!(typed, if projected { 2 } else { 0 });
    if projected {
        let floor = budget.storage();
        let refusal = scoped_slot_uses_v29::check_scoped_source_slot_uses_v29(
            instances, emitted, slots, 1024, budget,
        );
        assert!(
            matches!(
                refusal,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "original raw source requires consuming expanded physical admission",
                    ..
                })
            ),
            "{refusal:?}"
        );
        assert_eq!(budget.storage(), floor);
        let callbacks = slots
            .instances
            .iter()
            .filter(|row| row.function == CALLBACK)
            .count();
        super::super::call_memory_tests::typed_call_results::complete_original_source_again(
            lifecycle, false, callbacks, budget,
        )
        .expect("this same original fixture must complete consuming physical admission");
        assert_eq!(budget.storage(), floor);
        PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
        OBSERVED.set(OBSERVED.get() + 1);
        return Err(unsupported(0, None, None, STOP));
    }
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    observe(lifecycle, instances, emitted, slots, budget)
}

#[test]
fn actual_defined_and_intrinsic_call_results_capture_the_returned_scalar_before_destination() {
    for projected in [false, true] {
        PAYLOAD_COMPLETED.set(0);
        let source = ScopedFixture::CallDestinations {
            projected,
            retained_address: false,
            indexed: false,
        };
        let (result, _, _) = run(false, source, inspect_call_payloads, LIMIT, LIMIT);
        assert!(is_stopped(&result), "{projected}: {result:?}");
        assert_eq!(PAYLOAD_COMPLETED.get(), 1);
    }
}

#[test]
fn actual_scalar_payload_pipeline_retains_exact_and_one_short_limits() {
    PAYLOAD_COMPLETED.set(0);
    let (result, work, peak) = run(false, scalar_fixture(), inspect_payloads, LIMIT, LIMIT);
    assert!(is_stopped(&result), "{result:?}");
    assert_eq!(PAYLOAD_COMPLETED.get(), 1);
    PAYLOAD_COMPLETED.set(0);
    let result = run(false, scalar_fixture(), inspect_payloads, work, peak).0;
    assert!(is_stopped(&result), "{result:?}");
    assert_eq!(PAYLOAD_COMPLETED.get(), 1);
    let short_work = run(false, scalar_fixture(), inspect_payloads, work - 1, peak).0;
    assert!(
        matches!(
            short_work,
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Work(_)
                )
            )
        ),
        "{short_work:?}"
    );
    let short_storage = run(false, scalar_fixture(), inspect_payloads, work, peak - 1).0;
    assert!(
        matches!(
            short_storage,
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Storage(_)
                )
            )
        ),
        "{short_storage:?}"
    );
}

fn inspect_reference_cell_payloads(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    assert_eq!(lifecycle.owner.identity(), instances.owner().identity());
    assert!(slots.ledger == budget.work_ledger_identity_v1());
    assert!(slots.source == ExecutionCallSourceV29::from_instances(instances, budget)?);
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|row| row.function.index() == 3)
    {
        helpers += 1;
        let source = instances.instance(item.instance).unwrap().declaration();
        let occurrences = instances.occurrences(item.instance).unwrap();
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let mut entries = 0;
        let mut reads = 0;
        for row in &lowered.scoped_memory_anchors.as_ref().unwrap().rows {
            let ScopedMemoryAnchorKindV29::Access {
                payload: Some(payload),
                ..
            } = row.kind
            else {
                continue;
            };
            let operation = &lowered
                .function
                .body
                .as_ref()
                .unwrap()
                .blocks
                .iter()
                .find(|block| block.id == row.block)
                .unwrap()
                .operations[row.position];
            check_scoped_payload_v29(source, &occurrences, row, operation, budget)?;
            match payload {
                ScopedMemoryPayloadV29::Store {
                    value,
                    source: ScopedMemoryStoreSourceV29::EntryArgument { local, ty },
                } => {
                    entries += 1;
                    assert_eq!(local.index(), 1);
                    assert_eq!(source.locals()[1].ty(), ty);
                    assert!(source.locals()[1].role().is_entry_argument());
                    assert!(row.source.is_none());
                    assert!(
                        matches!(operation.kind, OperationKind::Store { value: actual, .. } if actual == value)
                    );
                }
                ScopedMemoryPayloadV29::Load { result, read } => {
                    reads += 1;
                    let original = scoped_payload_place_v29(source, read.site, read.role).unwrap();
                    // This shared borrow snapshots a retained scalar. Its later
                    // promoted dereference reuses that value without a Load.
                    assert_eq!(original.local().index(), 1);
                    assert_eq!(read.prefix, 0);
                    assert!(original.projections().is_empty());
                    assert_eq!(read.ty, source.locals()[1].ty());
                    assert_eq!(
                        read.site,
                        ExecutionSiteV29::Statement {
                            block: SsaBlockIdV1::new(0),
                            statement: 0,
                        }
                    );
                    assert_eq!(read.role, ExecutionOperandV29::RvaluePlace);
                    let SemanticStatementKindV1::Assign(assignment) =
                        source.blocks()[0].statements()[0].kind()
                    else {
                        panic!("original shared borrow assignment missing");
                    };
                    assert!(matches!(assignment.value().kind(),
                        SemanticRvalueKindV1::Borrow { kind: SemanticBorrowKindV1::Shared, place }
                        if std::ptr::eq(place, original)));
                    let ScopedMemoryOccurrenceV29::Retained { event } = read.occurrence else {
                        panic!("shared borrow must read its retained original argument");
                    };
                    let occurrence = &occurrences.events()[event];
                    assert_eq!(occurrence.site(), read.site);
                    assert_eq!(occurrence.operand(), read.role);
                    assert_eq!(occurrence.role(), ExecutionEventV29::BaseUse);
                    assert_eq!(occurrence.event().variable().get(), 1);
                    assert!(!occurrence.is_promoted());
                    assert!(occurrence.resolved().is_none());
                    assert_eq!(operation.results.len(), 1);
                    assert_eq!(operation.results[0].id, result);
                }
                _ => panic!("reference-cell payload has an unexpected source role"),
            }
        }
        assert_eq!((entries, reads), (1, 1));
        let SemanticStatementKindV1::Assign(assignment) = source.blocks()[1].statements()[0].kind()
        else {
            panic!("original promoted dereference assignment missing");
        };
        let SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(original)) =
            assignment.value().kind()
        else {
            panic!("original promoted dereference changed");
        };
        assert_eq!(original.local().index(), 2);
        assert_eq!(original.projections().len(), 1);
        assert_eq!(
            original.projections()[0].kind(),
            SemanticProjectionKindV1::Dereference
        );
        let span = lowered
            .statement_operation_spans
            .iter()
            .find(|span| span.semantic_block.index() == 1 && span.statement_ordinal == 0)
            .unwrap();
        let block = lowered
            .function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.id == span.kernel_ir_block)
            .unwrap();
        assert!(
            block.operations[span.first_operation_ordinal as usize
                ..(span.first_operation_ordinal + span.operation_count) as usize]
                .iter()
                .all(|operation| !matches!(
                    operation.kind,
                    OperationKind::Load { .. } | OperationKind::GuardedLoad { .. }
                ))
        );
    }
    assert_eq!(helpers, 2);
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    // Keep this exact candidate in production assembly. Its typed ownership
    // cannot be rechecked by the scalar-only standalone slot-use profile.
    OBSERVED.set(OBSERVED.get() + 1);
    Ok(())
}

#[test]
fn actual_reference_cell_reads_keep_full_source_prefix_and_distinct_entry_initialization() {
    PAYLOAD_COMPLETED.set(0);
    let (result, _, _, completed) = run_original_repeated_source_v29(
        super::super::retained_scalar_entry_tests::retained_reference_owner,
        inspect_reference_cell_payloads,
        LIMIT,
        LIMIT,
    );
    assert!(result.is_ok(), "{result:?}");
    assert!(
        completed,
        "the inspected reference-cell candidate must complete physical admission"
    );
    assert_eq!(PAYLOAD_COMPLETED.get(), 3);
    assert_eq!(OBSERVED.get(), 3);
}

fn inspect_promoted_reference_payloads(
    lifecycle: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    _: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    assert_eq!(lifecycle.owner.identity(), instances.owner().identity());
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|row| row.function.index() == 3)
    {
        helpers += 1;
        let source = instances.instance(item.instance).unwrap().declaration();
        assert_eq!(source.blocks().len(), 1);
        let SemanticStatementKindV1::Assign(borrow) = source.blocks()[0].statements()[0].kind()
        else {
            panic!("original same-block Shared borrow")
        };
        assert!(matches!(
            borrow.value().kind(),
            SemanticRvalueKindV1::Borrow {
                kind: SemanticBorrowKindV1::Shared,
                ..
            }
        ));
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        assert!(
            lowered
                .scoped_memory_anchors
                .as_ref()
                .unwrap()
                .rows
                .iter()
                .all(|row| !matches!(row.kind, ScopedMemoryAnchorKindV29::Access { .. }))
        );
        assert!(
            lowered
                .function
                .body
                .as_ref()
                .unwrap()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .all(|operation| !matches!(
                    operation.kind,
                    OperationKind::Alloca { .. }
                        | OperationKind::Load { .. }
                        | OperationKind::Store { .. }
                        | OperationKind::GuardedLoad { .. }
                ))
        );
    }
    assert_eq!(helpers, 2);
    PAYLOAD_COMPLETED.set(PAYLOAD_COMPLETED.get() + 1);
    OBSERVED.set(OBSERVED.get() + 1);
    Ok(())
}

#[test]
fn actual_promoted_reference_reads_need_no_memory_payloads() {
    PAYLOAD_COMPLETED.set(0);
    let (result, _, _, completed) = run_original_repeated_source_v29(
        super::super::super::fixtures::repeated_reference_owner,
        inspect_promoted_reference_payloads,
        LIMIT,
        LIMIT,
    );
    assert!(result.is_ok(), "{result:?}");
    assert!(completed);
    assert_eq!(PAYLOAD_COMPLETED.get(), 3);
    assert_eq!(OBSERVED.get(), 3);
}
