use super::*;
use crate::production_semantic_kir_v1::scoped_slot_uses_v29;

mod payload_tests {
    include!("production_scoped_memory_payloads_v29_tests.rs");
}

mod atomic_payload_tests {
    include!("production_scoped_atomic_payload_v1_tests.rs");
}

mod object_payload_tests {
    include!("production_source_object_payloads_v29_tests.rs");
    mod resources {
        include!("production_source_object_payload_resources_v29_tests.rs");
    }
}

const LIMIT: usize = 10_000_000;

fn inspect_rollback(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut work = Vec::new();
    for fail in [false, true] {
        let before = budget.work();
        let floor = budget.storage();
        with_canonical_call_scratch_v1(budget, |budget| {
            let placement = SemanticEmissionPlacementV1 {
                first_block: 17,
                first_value: 200,
            };
            let helper = instances
                .instances()
                .iter()
                .position(|row| row.function().index() == 3)
                .unwrap();
            let instance = instances.id_at(helper).unwrap();
            let row = instances.instance(instance).unwrap();
            let plan = execution_instance_plan_v29(
                instances,
                instance,
                FunctionId::new("checked_slot_probe"),
                placement,
                budget,
            )?;
            let mut producer = ExecutionLifecycleProducerV29::new(
                lifecycle_source,
                instances,
                instance,
                placement,
                budget,
            )?;
            with_execution_availability_v29(instances, instance, budget, |cursor, budget| {
                let semantic = instances.owner().source_semantic();
                let source = row.declaration();
                let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                    semantic.types(),
                    semantic.callables(),
                    source,
                    row.ssa(),
                    plan.correspondence_owner,
                    plan.semantic_function,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    plan.result_types.clone(),
                    SemanticParameterBindingsV1 {
                        declarations: &plan.parameter_declarations,
                        values: &plan.parameter_values,
                        types: &plan.parameter_types,
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
                    Some(budget),
                    placement,
                    Some(cursor),
                    Some(&mut producer),
                    None,
)?;
                let block_id = SemanticBlockIdV1::from_index(0);
                let mut block = BasicBlock::new(placement.block(0)?);
                lowering.begin_block(block_id, &mut block)?;
                for ordinal in 0..5 {
                    lowering.lower_statement(
                        block_id,
                        Some(ordinal as u32),
                        source.blocks()[0].statements()[ordinal].kind(),
                        &mut block.operations,
                    )?;
                }
                let recorder = lowering.scoped_memory.as_ref().unwrap();
                let old_rows = recorder.anchors.rows.clone();
                assert_eq!((old_rows.len(), recorder.anchors.rows.capacity()), (4, 4));
                let old_objects = recorder.anchors.objects.clone();
                let old_components = recorder.anchors.object_components.clone();
                let old_ops = block.operations.clone();
                let old_next = lowering.next_value;
                let old_count = lowering.emitted_operations;
                let old_storage = lowering.emission_work.as_deref().unwrap().storage();
                if fail {
                    lowering.max_operations = old_count + 1;
                    assert!(matches!(lowering.lower_statement(block_id, Some(5),
                        source.blocks()[0].statements()[5].kind(), &mut block.operations),
                        Err(ProductionSemanticKirErrorV1::ResourceLimit {
                            resource: ProductionSemanticKirResourceV1::Operations, actual, limit,
                        }) if actual == old_count + 2 && limit == old_count + 1));
                    assert_eq!(block.operations, old_ops);
                    assert_eq!(
                        (lowering.next_value, lowering.emitted_operations),
                        (old_next, old_count)
                    );
                    let recorder = lowering.scoped_memory.as_ref().unwrap();
                    assert_eq!(recorder.anchors.rows, old_rows);
                    assert_eq!(recorder.anchors.objects, old_objects);
                    assert_eq!(recorder.anchors.object_components, old_components);
                    assert_eq!(recorder.anchors.rows.capacity(), 8);
                    assert!(recorder.frame.is_none());
                    assert_eq!(
                        lowering.emission_work.as_deref().unwrap().storage() - old_storage,
                        4 * std::mem::size_of::<ScopedMemoryAnchorV29>()
                    );
                    assert!(
                        lowering.private_arrays.frame.is_none()
                            && lowering.private_arrays.pending.is_none()
                    );
                }
                let outer = ScopedMemoryFrameV29 {
                    site: execution_site_v29(block_id, Some(0)),
                    role: None,
                };
                let inner = ScopedMemoryFrameV29 {
                    site: execution_site_v29(block_id, Some(1)),
                    role: Some(ScopedMemoryRoleV29::Operand(
                        ExecutionOperandV29::StoreValue,
                    )),
                };
                lowering.with_scoped_memory_frame_v29(outer, |lowering| {
                    let error: Result<(), _> = lowering
                        .with_scoped_memory_frame_v29(inner, |_| Err(scoped_memory_error_v29()));
                    assert!(error.is_err());
                    assert_eq!(lowering.scoped_memory.as_ref().unwrap().frame, Some(outer));
                    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _: Result<(), _> = lowering
                            .with_scoped_memory_frame_v29(inner, |_| std::panic::panic_any(73_u32));
                    }))
                    .unwrap_err();
                    assert_eq!(*panic.downcast::<u32>().unwrap(), 73);
                    assert_eq!(lowering.scoped_memory.as_ref().unwrap().frame, Some(outer));
                    Ok(())
                })?;
                assert!(lowering.scoped_memory.as_ref().unwrap().frame.is_none());
                object_payload_tests::check_role_frames(&mut lowering)?;
                // The failed emitter is dropped, not retried with partly consumed source state.
                Ok(())
            })
        })?;
        assert_eq!(budget.storage(), floor);
        work.push(budget.work() - before);
    }
    assert!(work[1] > work[0]);
    observe(lifecycle_source, instances, emitted, slots, budget)
}

#[test]
fn checked_failure_truncates_anchors_but_retains_capacity_and_restores_frames() {
    let (result, _, _) = run(
        false,
        ScopedFixture::CheckedSlot,
        inspect_rollback,
        LIMIT,
        LIMIT,
    );
    assert!(is_stopped(&result), "{result:?}");
}

fn inspect_array_move(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        let source = instances.instance(item.instance).unwrap().declaration();
        let SemanticStatementKindV1::Assign(assignment) =
            source.blocks()[0].statements().last().unwrap().kind()
        else {
            panic!("expected move");
        };
        let SemanticRvalueKindV1::Use(SemanticOperandV1::Move(place)) = assignment.value().kind()
        else {
            panic!("expected move operand");
        };
        let projected = !place.projections().is_empty();
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
        let kills: Vec<_> = anchors
            .rows
            .iter()
            .filter(|row| matches!(row.kind, ScopedMemoryAnchorKindV29::Kill { .. }))
            .collect();
        assert_eq!(kills.len(), 1);
        let ScopedMemoryAnchorKindV29::Kill {
            event,
            local,
            cause,
        } = kills[0].kind
        else {
            unreachable!()
        };
        assert_eq!(local, 2);
        assert_eq!(
            cause,
            if projected {
                ScopedMemoryKillV29::ProjectedArrayMove
            } else {
                ScopedMemoryKillV29::Move
            }
        );
        let original = instances.occurrences(item.instance).unwrap();
        assert_eq!(
            original.events()[event].role(),
            if projected {
                ExecutionEventV29::BaseUse
            } else {
                ExecutionEventV29::MoveKill
            }
        );
        let block = lowered
            .function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.id == kills[0].block)
            .unwrap();
        assert!(matches!(
            block.operations[kills[0].position - 1].kind,
            OperationKind::Load { .. }
        ));
    }
    assert_eq!(helpers, 2);
    observe(lifecycle_source, instances, emitted, slots, budget)
}

#[test]
fn whole_and_projected_array_moves_keep_distinct_original_ssa_occurrences() {
    for projected in [false, true] {
        let (result, _, _) = run(
            false,
            ScopedFixture::InitializationArrayMove(projected),
            inspect_array_move,
            LIMIT,
            LIMIT,
        );
        assert!(is_stopped(&result), "{projected}: {result:?}");
    }
}

fn observe(
    _source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    OBSERVED.set(OBSERVED.get() + 1);
    let floor = budget.storage();
    let result = scoped_slot_uses_v29::check_scoped_source_slot_uses_v29(
        instances, emitted, slots, 1024, budget,
    );
    assert_eq!(budget.storage(), floor);
    result?;
    Err(unsupported(0, None, None, STOP))
}

fn fixture(
    kill: Option<InitializationKillV29>,
    looping: bool,
    reinitialize: bool,
) -> ScopedFixture {
    ScopedFixture::Initialization(InitializationFixtureV29 {
        kill,
        looping,
        reinitialize,
        address_read: true,
        ..InitializationFixtureV29::default()
    })
}

fn observe_original(
    _source: &ExecutionLifecycleSourceV29<'_>,
    _instances: &ExecutionInstancesV29<'_>,
    _emitted: &mut [Option<LoweredFunctionResultV1>],
    _slots: &OwnedScopedSourceSlotsV29,
    _budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    OBSERVED.set(OBSERVED.get() + 1);
    Ok(())
}

fn run_original_anchor(
    fixture: ScopedFixture,
    observer: ScopedSlotObserverV29,
    work: usize,
    storage: usize,
) -> (SourceOwnedResultV18<()>, usize, usize, bool) {
    let ScopedFixture::Initialization(config) = fixture else {
        panic!("original initialization fixture required");
    };
    run_original_repeated_source_v29(
        || super::super::fixtures::initialization_owner(config),
        observer,
        work,
        storage,
    )
}

fn assert_original_anchor_completed(result: &SourceOwnedResultV18<()>, completed: bool) {
    assert!(result.is_ok(), "{result:?}");
    assert!(
        completed,
        "the same observed candidate must complete physical admission"
    );
    assert_eq!(
        OBSERVED.get(),
        3,
        "admission, reconstruction, and consuming source replay"
    );
}

#[test]
fn source_kills_reject_alias_reads_after_zero_operation_invalidations_and_moves() {
    for looping in [false, true] {
        for kill in [
            InitializationKillV29::StorageLive,
            InitializationKillV29::StorageDead,
            InitializationKillV29::Deinitialize,
            InitializationKillV29::Move,
            InitializationKillV29::StorageDeadLive,
        ] {
            if matches!(kill, InitializationKillV29::Deinitialize) {
                let ScopedFixture::Initialization(config) = fixture(Some(kill), looping, false)
                else {
                    unreachable!();
                };
                OBSERVED.set(0);
                let error = super::super::fixtures::try_initialization_owner(config)
                    .err()
                    .expect(
                        "address of a possibly deinitialized predecessor must fail SSA admission",
                    );
                assert!(
                    matches!(error, ProductionSemanticSsaErrorV1::PartialMove {
                    function, block: 3, statement: Some(0), local: 2,
                    violation: fe2o3_pliron::SemanticPartialMoveViolationV1::MaybeMovedValueUsed,
                } if function == SemanticFunctionIdV1::from_index(3)),
                    "{looping}/{kill:?}: {error:?}"
                );
                assert_eq!(OBSERVED.get(), 0);
                continue;
            }
            let (result, _, _) = run(
                false,
                fixture(Some(kill), looping, false),
                observe,
                LIMIT,
                LIMIT,
            );
            let expected = if matches!(kill, InitializationKillV29::StorageDead) {
                if looping {
                    // The loop body's next direct Store reaches ended storage
                    // before the exit block can form its raw address.
                    "source storage object lifetime has ended"
                } else {
                    "source raw address names dead storage"
                }
            } else {
                "source reference reads an uninitialized partial holder"
            };
            assert!(
                matches!(
                    result,
                    Err(ProductionSemanticKirErrorV1::Unsupported {
                        function: 0, block: None, statement: None, detail,
                    }) if detail == expected
                ),
                "{looping}/{kill:?}: {result:?}"
            );
            assert_eq!(
                OBSERVED.get(),
                0,
                "original source validation must precede emission"
            );
        }
    }
}

#[test]
fn source_kills_reinitialized_alias_reads_complete_original_physical_admission() {
    for looping in [false, true] {
        for kill in [
            None,
            Some(InitializationKillV29::StorageLive),
            Some(InitializationKillV29::StorageDeadLive),
            Some(InitializationKillV29::Deinitialize),
            Some(InitializationKillV29::Move),
        ] {
            let (result, _, _, completed) = run_original_anchor(
                fixture(kill, looping, kill.is_some()),
                observe_original,
                LIMIT,
                LIMIT,
            );
            assert!(result.is_ok(), "{looping}/{kill:?}: {result:?}");
            assert_original_anchor_completed(&result, completed);
        }
    }
}

#[test]
fn source_kills_alias_store_cannot_restart_a_dead_original_lifetime() {
    for looping in [false, true] {
        let (result, _, _) = run(
            false,
            fixture(Some(InitializationKillV29::StorageDead), looping, true),
            observe,
            LIMIT,
            LIMIT,
        );
        assert!(
            matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    function: 0,
                    block: None,
                    statement: None,
                    detail: "source storage object lifetime has ended",
                })
            ),
            "looping={looping}: {result:?}"
        );
        assert_eq!(OBSERVED.get(), 0);
    }
}

fn inspect_self_move(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut pointers = BTreeSet::new();
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        let slot = slots.slots[item.slots.start];
        assert!(pointers.insert(slot.origin.pointer));
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
        assert_eq!(anchors.subject.instance, item.instance);
        let kills: Vec<_> = anchors
            .rows
            .iter()
            .filter(|row| matches!(row.kind, ScopedMemoryAnchorKindV29::Kill { .. }))
            .collect();
        assert_eq!(kills.len(), 1);
        let kill = kills[0];
        assert!(matches!(
            kill.kind,
            ScopedMemoryAnchorKindV29::Kill {
                local: 2,
                cause: ScopedMemoryKillV29::Move,
                ..
            }
        ));
        let block = lowered
            .function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.id == kill.block)
            .unwrap();
        let read = &block.operations[kill.position - 1];
        let write = &block.operations[kill.position];
        let OperationKind::Storage(ScopedObjectOperationV29::ReadValue {
            address: read_address,
            access: read_access,
        }) = read.kind
        else {
            panic!("original self-move must read its typed scalar before the kill");
        };
        let OperationKind::Storage(ScopedObjectOperationV29::WriteValue {
            address: write_address,
            value,
            access: write_access,
        }) = write.kind
        else {
            panic!("original self-move must write its snapshot after the kill");
        };
        assert_eq!(
            (read_address, write_address),
            (slot.origin.pointer, slot.origin.pointer)
        );
        assert_eq!(read.results.len(), 1);
        assert_eq!(value, read.results[0].id);
        assert_eq!(read.results[0].ty, Type::Scalar(ScalarType::U32));
        assert_eq!(read_access, write_access);
        assert_eq!(read_access.address_space, AddressSpace::Private);
        assert_eq!(read_access.alignment, 4);
        assert!(!read_access.volatile);
        let rows = &anchors.rows;
        let index = rows.iter().position(|row| std::ptr::eq(row, kill)).unwrap();
        assert_eq!(rows[index - 1].position + 1, kill.position);
        assert_eq!(rows[index + 1].position, kill.position);
        assert_eq!(
            rows[index - 1].source.unwrap().role,
            Some(ScopedMemoryRoleV29::Operand(
                ExecutionOperandV29::RvalueOperand(0)
            ))
        );
        assert_eq!(
            rows[index + 1].source.unwrap().role,
            Some(ScopedMemoryRoleV29::Operand(
                ExecutionOperandV29::Destination
            ))
        );
        let source = instances.instance(item.instance).unwrap().declaration();
        let occurrences = instances.occurrences(item.instance).unwrap();
        for (ordinal, operation) in [(index - 1, read), (index + 1, write)] {
            let row = &rows[ordinal];
            let ScopedMemoryAnchorKindV29::Object(object) = row.kind else {
                panic!("typed self-move operation needs its original object recipe");
            };
            let payload = &anchors.objects[object];
            assert_eq!(operation.kind, OperationKind::Storage(payload.operation));
            anchors.check_object_source(source, &occurrences, ordinal, row, payload, budget)?;
            let endpoint = match payload.role {
                ScopedObjectRoleV29::ReadValue {
                    source,
                    read: ScopedObjectReadOriginV29::Original(_),
                } => source,
                ScopedObjectRoleV29::WriteValue { destination, .. } => destination,
                _ => panic!("original self-move read or write recipe required"),
            };
            assert!(
                matches!(endpoint.object, ScopedObjectIdentityV29::Local { instance, local, .. }
                if instance == item.instance && local.index() == 2)
            );
            assert_eq!((endpoint.root_type, endpoint.projected_type), (U32, U32));
            assert_eq!((endpoint.source_path.count, endpoint.path.count), (0, 0));
        }
    }
    assert_eq!(pointers.len(), 2);
    observe_original(lifecycle_source, instances, emitted, slots, budget)
}

#[test]
fn self_move_reads_then_invalidates_then_reinitializes_each_instance() {
    for looping in [false, true] {
        let (result, _, _, completed) = run_original_anchor(
            fixture(Some(InitializationKillV29::SelfMove), looping, false),
            inspect_self_move,
            LIMIT,
            LIMIT,
        );
        assert_original_anchor_completed(&result, completed);
    }
}

fn inspect_same_gap(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let rows: Vec<_> = lowered
            .scoped_memory_anchors
            .as_ref()
            .unwrap()
            .rows
            .iter()
            .filter(|row| matches!(row.kind, ScopedMemoryAnchorKindV29::Kill { .. }))
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].block, rows[1].block);
        assert_eq!((rows[0].position, rows[1].position), (0, 0));
        assert!(matches!(
            rows[0].kind,
            ScopedMemoryAnchorKindV29::Kill {
                cause: ScopedMemoryKillV29::StorageDead,
                ..
            }
        ));
        assert!(matches!(
            rows[1].kind,
            ScopedMemoryAnchorKindV29::Kill {
                cause: ScopedMemoryKillV29::StorageLive,
                ..
            }
        ));
        assert_eq!(
            scoped_memory_site_key_v29(rows[0].source.unwrap().site),
            (2, Some(0))
        );
        assert_eq!(
            scoped_memory_site_key_v29(rows[1].source.unwrap().site),
            (2, Some(1))
        );
    }
    assert_eq!(helpers, 2);
    observe_original(lifecycle_source, instances, emitted, slots, budget)
}

#[test]
fn separate_source_kills_at_one_gap_are_not_collapsed() {
    let (result, _, _, completed) = run_original_anchor(
        fixture(Some(InitializationKillV29::StorageDeadLive), false, true),
        inspect_same_gap,
        LIMIT,
        LIMIT,
    );
    assert_original_anchor_completed(&result, completed);
}

fn inspect_alias_move(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        let source = instances.instance(item.instance).unwrap().declaration();
        let statements = source.blocks()[3].statements();
        assert_eq!(statements.len(), 4);
        let SemanticStatementKindV1::Assign(moved) = statements[1].kind() else {
            panic!("expected move assignment");
        };
        let SemanticRvalueKindV1::Use(SemanticOperandV1::Move(place)) = moved.value().kind() else {
            panic!("expected source Move through alias");
        };
        assert_eq!(place.local().index(), 5);
        assert!(
            matches!(place.projections(), [projection] if projection.kind() == SemanticProjectionKindV1::Dereference)
        );
        let SemanticStatementKindV1::Store(store) = statements[2].kind() else {
            panic!("a moved pointee must be reinitialized before another read");
        };
        assert_eq!(store.destination(), place);
        let SemanticStatementKindV1::Assign(read) = statements[3].kind() else {
            panic!("expected load assignment");
        };
        let SemanticRvalueKindV1::Load(load) = read.value().kind() else {
            panic!("expected subsequent source Load");
        };
        assert_eq!(load.source(), place);
        let lowered = emitted[item.instance.index()].as_ref().unwrap();
        let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
        let rows = &anchors.rows;
        assert!(
            rows.iter()
                .all(|row| !matches!(row.kind, ScopedMemoryAnchorKindV29::Kill { .. })),
            "a pointee Move does not end the alias or its storage activation"
        );
        let exit = item.placement.block(3)?;
        assert_eq!(rows.iter().filter(|row| row.block == exit).count(), 3);
        let block = lowered
            .function
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .find(|block| block.id == exit)
            .unwrap();
        let slots = &slots.slots[item.slots.clone()];
        assert_eq!(slots.len(), 1);
        assert!(matches!(
            slots[0].origin.identity,
            ScopedAllocationIdentityV29::OriginalObject {
                local: 2,
                generation: 0,
            }
        ));
        let mut pointer = None;
        let occurrences = instances.occurrences(item.instance).unwrap();
        for (order, (ordinal, row)) in rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.block == exit)
            .enumerate()
        {
            let ScopedMemoryAnchorKindV29::Object(object) = row.kind else {
                panic!("original typed access");
            };
            let payload = &anchors.objects[object];
            let operation = &block.operations[row.position];
            assert_eq!(operation.kind, OperationKind::Storage(payload.operation));
            let (address, access) = match (order, payload.operation) {
                (0 | 2, ScopedObjectOperationV29::ReadValue { address, access }) => {
                    (address, access)
                }
                (
                    1,
                    ScopedObjectOperationV29::WriteValue {
                        address, access, ..
                    },
                ) => (address, access),
                _ => panic!("read, reinitialize, read in original order"),
            };
            assert_eq!(*pointer.get_or_insert(address), address);
            assert_eq!(access.address_space, AddressSpace::Private);
            assert!(!access.volatile);
            anchors.check_object_source(source, &occurrences, ordinal, row, payload, budget)?;
        }
        let alias = pointer.unwrap();
        assert!(block.operations.iter().any(|operation|
            operation.results.iter().any(|result| result.id == alias)
                && matches!(operation.kind, OperationKind::Select { true_value, false_value, .. }
                    if true_value == slots[0].origin.pointer && false_value == slots[0].origin.pointer)),
            "all three accesses must retain the same original alias allocation");
    }
    assert_eq!(helpers, 2);
    observe_original(lifecycle_source, instances, emitted, slots, budget)
}

fn alias_move_configuration(looping: bool) -> InitializationFixtureV29 {
    InitializationFixtureV29 {
        looping,
        address_read: true,
        alias_move: true,
        ..InitializationFixtureV29::default()
    }
}

fn reinitialized_alias_move_owner(looping: bool) -> ProductionSemanticSsaOwnerV1 {
    let original = super::super::fixtures::initialization_owner(alias_move_configuration(looping));
    let semantic = original.source_semantic();
    let mut functions = semantic.functions().to_vec();
    let helper = &functions[3];
    let mut blocks = helper.blocks().to_vec();
    let mut statements = blocks[3].statements().to_vec();
    let SemanticStatementKindV1::Assign(moved) = statements[1].kind() else {
        unreachable!()
    };
    let SemanticRvalueKindV1::Use(SemanticOperandV1::Move(place)) = moved.value().kind() else {
        unreachable!()
    };
    let store = SemanticStatementV1::new(
        SemanticSourceProvenanceV1::unavailable(),
        SemanticStatementKindV1::Store(SemanticMemoryStoreV1::new(
            place.clone(),
            SemanticOperandV1::Constant(SemanticConstantV1::new(
                U32,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(99, 4).unwrap()),
            )),
            SemanticVolatilityV1::NonVolatile,
            None,
        )),
    );
    statements.insert(2, store);
    blocks[3] = block(143, statements, blocks[3].terminator().kind().clone());
    functions[3] = function(
        130,
        helper.role(),
        helper.abi().clone(),
        helper.locals().to_vec(),
        blocks,
    );
    super::super::fixtures::build(
        semantic.types().to_vec(),
        functions,
        semantic.callables().to_vec(),
    )
}

#[test]
fn moving_through_alias_does_not_invalidate_the_alias_or_pointee_slot() {
    for looping in [false, true] {
        let (result, _, _, completed) = run_original_repeated_source_v29(
            || reinitialized_alias_move_owner(looping),
            inspect_alias_move,
            LIMIT,
            LIMIT,
        );
        assert_original_anchor_completed(&result, completed);
    }
}

#[test]
fn reading_a_moved_pointee_without_reinitialization_is_rejected_before_emission() {
    // Pinned rustc MIR Operand::Move forbids a second load before reinitialization.
    for looping in [false, true] {
        let (result, _, _) = run(
            false,
            ScopedFixture::Initialization(alias_move_configuration(looping)),
            inspect_alias_move,
            LIMIT,
            LIMIT,
        );
        assert!(
            matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    function: 0,
                    block: None,
                    statement: None,
                    detail: "source raw pointer reads an undefined referent",
                })
            ),
            "{looping}: {result:?}"
        );
        assert_eq!(OBSERVED.get(), 0);
    }
}

thread_local! {
    static MUTATION: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

fn mutate(
    lifecycle_source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if MUTATION.get() == 11 {
        assert!(slots.instances[0].slots.is_empty());
        emitted[0].as_mut().unwrap().scoped_memory_anchors = None;
        return observe_original(lifecycle_source, instances, emitted, slots, budget);
    }
    let item = slots
        .instances
        .iter()
        .find(|item| item.function.index() == 3)
        .unwrap();
    let lowered = emitted[item.instance.index()].as_mut().unwrap();
    let anchors = lowered.scoped_memory_anchors.as_mut().unwrap();
    let kill = anchors
        .rows
        .iter()
        .position(|row| matches!(row.kind, ScopedMemoryAnchorKindV29::Kill { .. }))
        .unwrap();
    match MUTATION.get() {
        0 => {
            anchors.rows.remove(kill);
        }
        1 => {
            // Keep capacity unchanged; this mutation tests duplicate coverage, not allocation.
            let other = kill + 1;
            assert!(matches!(
                anchors.rows[other].kind,
                ScopedMemoryAnchorKindV29::Kill { .. }
            ));
            anchors.rows[other] = anchors.rows[kill];
        }
        2 => {
            anchors.rows[kill].position = usize::MAX;
        }
        3 => {
            anchors.placement.first_block += 1;
        }
        4 => {
            anchors.subject.instance = instances.root();
        }
        5 => {
            let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(LIMIT);
            let foreign = ArgumentBudgetV1::new(&mut work, LIMIT);
            anchors.subject.ledger = foreign.work_ledger_identity_v1();
        }
        6 => {
            let index = anchors
                .rows
                .iter()
                .position(|row| {
                    matches!(
                        row.kind,
                        ScopedMemoryAnchorKindV29::Access { .. }
                            | ScopedMemoryAnchorKindV29::Object(_)
                    )
                })
                .unwrap();
            anchors.rows.remove(index);
        }
        7 => {
            if let ScopedMemoryAnchorKindV29::Kill { local, .. } = &mut anchors.rows[kill].kind {
                *local = 999;
            }
        }
        8 => {
            anchors.rows[kill].source = None;
        }
        9 => {
            lowered.statement_operation_spans[0].semantic_function = ROOT;
        }
        10 => {
            lowered.terminator_operation_spans[0].correspondence_owner =
                SemanticFunctionIdV1::from_index(999);
        }
        _ => unreachable!(),
    }
    observe_original(lifecycle_source, instances, emitted, slots, budget)
}

#[test]
fn source_census_rejects_missing_duplicate_foreign_or_displaced_anchors() {
    for (kill, reinitialize) in [
        (InitializationKillV29::SelfMove, false),
        (InitializationKillV29::StorageDeadLive, true),
    ] {
        let (result, _, _, completed) = run_original_anchor(
            fixture(Some(kill), false, reinitialize),
            observe_original,
            LIMIT,
            LIMIT,
        );
        assert_original_anchor_completed(&result, completed);
    }
    for mutation in 0..12 {
        MUTATION.set(mutation);
        let (result, _, _, completed) = run_original_anchor(
            fixture(
                Some(if mutation == 1 {
                    InitializationKillV29::StorageDeadLive
                } else {
                    InitializationKillV29::SelfMove
                }),
                false,
                mutation == 1,
            ),
            mutate,
            LIMIT,
            LIMIT,
        );
        assert!(
            !completed,
            "a mutated candidate cannot complete physical admission"
        );
        let result = result.map_err(|error| match error {
            ProductionSourceOwnedViewErrorV18::Source(
                ProductionPendingScopedSourceErrorV29::Source(error),
            ) => error,
            other => panic!("the mutated candidate must reach its source rejection: {other:?}"),
        });
        let expected = match mutation {
            2 | 6 => matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    function: 0,
                    block: None,
                    statement: None,
                    detail: "typed object source payload differs from its actual operation",
                })
            ),
            3 | 4 | 11 => matches!(
                result,
                Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch)
            ),
            5 => matches!(
                result,
                Err(
                    ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                        ArgumentResourceV1::Accounting
                    )
                )
            ),
            9 | 10 => matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    function: 0,
                    block: None,
                    statement: None,
                    detail: "execution call parameters differ from their source instance",
                })
            ),
            _ => matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "scoped memory anchors differ from their source instance",
                    ..
                })
            ),
        };
        assert!(expected, "{mutation}: {result:?}");
        assert_eq!(OBSERVED.get(), 1);
    }
}

#[test]
fn composed_source_and_physical_history_obeys_exact_shared_limits() {
    let source = fixture(Some(InitializationKillV29::SelfMove), true, false);
    let (result, work, peak, completed) =
        run_original_anchor(source, observe_original, LIMIT, LIMIT);
    assert_original_anchor_completed(&result, completed);
    let (exact, exact_work, exact_peak, completed) =
        run_original_anchor(source, observe_original, work, peak);
    assert_original_anchor_completed(&exact, completed);
    assert_eq!((exact_work, exact_peak), (work, peak));
    let (short_work, _, _, work_completed) =
        run_original_anchor(source, observe_original, work - 1, peak);
    let (short_storage, _, _, storage_completed) =
        run_original_anchor(source, observe_original, work, peak - 1);
    assert!(!work_completed && !storage_completed);
    let short_work = original_repeated_source_resource_v29(short_work.unwrap_err());
    let short_storage = original_repeated_source_resource_v29(short_storage.unwrap_err());
    assert!(
        matches!(short_work, ArgumentResourceV1::Work(_)),
        "{short_work:?}"
    );
    assert!(
        matches!(short_storage, ArgumentResourceV1::Storage(_)),
        "{short_storage:?}"
    );
}
