use super::*;
use crate::production_semantic_kir_v1::scoped_slot_uses_v29;

const LIMIT: usize = 10_000_000;

include!("production_scoped_discarded_assert_source_v29_tests.rs");

fn probe_authentication(
    source: &ExecutionLifecycleSourceV29<'_>,
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
    let floor = budget.storage();
    with_canonical_call_scratch_v1(budget, |budget| {
        let row = instances.instance(item.instance).unwrap();
        let plan = execution_instance_plan_v29(
            instances,
            item.instance,
            FunctionId::new("discarded_auth_probe"),
            item.placement,
            budget,
        )?;
        let mut producer = ExecutionLifecycleProducerV29::new(
            source,
            instances,
            item.instance,
            item.placement,
            budget,
        )?;
        with_original_assertion_cursor_v29(
            source,
            instances,
            item.instance,
            budget,
            |cursor, budget| {
                let semantic = instances.owner().source_semantic();
                let function = row.declaration();
                let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                    semantic.types(),
                    semantic.callables(),
                    function,
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
                    item.placement,
                    Some(cursor),
                    Some(&mut producer),
                    None,
)?;
                let block_id = SemanticBlockIdV1::from_index(0);
                let mut block = BasicBlock::new(item.placement.block(0)?);
                lowering.begin_block(block_id, &mut block)?;
                for (ordinal, statement) in function.blocks()[0].statements().iter().enumerate() {
                    lowering.lower_statement(
                        block_id,
                        Some(ordinal as u32),
                        statement.kind(),
                        &mut block.operations,
                    )?;
                }
                let SemanticTerminatorKindV1::Assert { message, .. } =
                    function.blocks()[0].terminator().kind()
                else {
                    unreachable!();
                };
                let SemanticAssertMessageV1::BoundsCheck { length, index } = message else {
                    unreachable!()
                };
                let cloned = length.clone();
                let old_rows = lowering
                    .scoped_memory
                    .as_ref()
                    .unwrap()
                    .anchors
                    .rows
                    .clone();
                let old_frame = lowering.scoped_memory.as_ref().unwrap().frame;
                let old_operations = block.operations.clone();
                let old_counts = (lowering.next_value, lowering.emitted_operations);
                let old_initialized = lowering.retained_local_initialized.clone();
                let values = |lowering: &SemanticFunctionLoweringV1<'_, '_>| {
                    lowering
                        .locals
                        .iter()
                        .map(|value| value.as_ref().map(|value| value.value().unwrap()))
                        .collect::<Vec<_>>()
                };
                let old_values = values(&lowering);
                let old_storage = lowering.emission_work.as_deref().unwrap().storage();
                for (at, role, operand) in [
                    (block_id, ExecutionOperandV29::AssertMessage(0), &cloned),
                    (block_id, ExecutionOperandV29::AssertMessage(1), length),
                    (
                        SemanticBlockIdV1::from_index(1),
                        ExecutionOperandV29::AssertMessage(0),
                        length,
                    ),
                    (block_id, ExecutionOperandV29::RvalueOperand(0), length),
                ] {
                    assert!(matches!(
                        lowering.consume_execution_assert_operand_v29(
                            at,
                            role,
                            operand,
                            &mut block.operations
                        ),
                        Err(ProductionSemanticKirErrorV1::Unsupported {
                            function: 0,
                            block: None,
                            statement: None,
                            detail: "scoped memory anchors differ from their source instance",
                        })
                    ));
                    assert_eq!(
                        lowering.scoped_memory.as_ref().unwrap().anchors.rows,
                        old_rows
                    );
                    assert_eq!(lowering.scoped_memory.as_ref().unwrap().frame, old_frame);
                    assert_eq!(block.operations, old_operations);
                    assert_eq!(
                        (lowering.next_value, lowering.emitted_operations),
                        old_counts
                    );
                    assert_eq!(lowering.retained_local_initialized, old_initialized);
                    assert_eq!(values(&lowering), old_values);
                    assert_eq!(
                        lowering.emission_work.as_deref().unwrap().storage(),
                        old_storage
                    );
                }
                // V995 successor: the complete diagnostic tail has its own cursor
                // scope; a single operand is no longer a common-path consumer.
                lowering.consume_execution_assert_message_v29(
                    block_id,
                    message,
                    &mut block.operations,
                )?;
                assert_eq!(block.operations, old_operations);
                let moved = [length, index]
                    .iter()
                    .filter(|operand| matches!(operand, SemanticOperandV1::Move(_)))
                    .count();
                assert_eq!(
                    lowering.scoped_memory.as_ref().unwrap().anchors.rows.len(),
                    old_rows.len() + 2 + moved
                );
                assert_eq!(lowering.retained_local_initialized, old_initialized);
                assert_eq!(values(&lowering), old_values);
                assert_eq!(lowering.scoped_memory.as_ref().unwrap().frame, old_frame);
                Ok(())
            },
        )
    })?;
    assert_eq!(budget.storage(), floor);
    observe_runtime(source, instances, emitted, slots, budget)
}

#[test]
fn discarded_operand_authentication_rejects_clones_wrong_roles_and_wrong_blocks() {
    for move_message in [false, true] {
        let fixture = ScopedFixture::AssertionSlots {
            move_condition: false,
            move_message,
            reinitialize: true,
        };
        let (result, _, _) =
            run_original_assertion_v29(fixture, probe_authentication, LIMIT, LIMIT);
        assert_original_assertion_completed_v29(&result);
    }
}

fn check_assertion(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    lowered: &LoweredFunctionResultV1,
    folded: bool,
) {
    let source = instances.instance(instance).unwrap().declaration();
    let SemanticTerminatorKindV1::Assert {
        condition, message, ..
    } = source.blocks()[0].terminator().kind()
    else {
        panic!("expected assertion");
    };
    let SemanticAssertMessageV1::BoundsCheck { length, index } = message else {
        panic!("expected bounds diagnostics");
    };
    let mut expected = Vec::new();
    for (role, operand) in [
        (ExecutionOperandV29::AssertCondition, condition),
        (ExecutionOperandV29::AssertMessage(0), length),
        (ExecutionOperandV29::AssertMessage(1), index),
    ] {
        let (SemanticOperandV1::Copy(place) | SemanticOperandV1::Move(place)) = operand else {
            panic!("expected local operand");
        };
        assert_eq!(
            place.local().index(),
            match role {
                ExecutionOperandV29::AssertCondition => 4,
                ExecutionOperandV29::AssertMessage(0) => 2,
                ExecutionOperandV29::AssertMessage(1) => 3,
                _ => unreachable!(),
            }
        );
        if matches!(operand, SemanticOperandV1::Move(_)) {
            expected.push((role, place.local().index()));
        }
    }
    let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
    let span = lowered
        .terminator_operation_spans
        .iter()
        .find(|span| span.semantic_block.index() == 0)
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
    let start = span.first_operation_ordinal as usize;
    let end = start + span.operation_count as usize;
    let actual_memory: Vec<_> = block.operations[start..end]
        .iter()
        .filter(|op| {
            scoped_memory_pointer_v29(&op.kind).is_some()
                || matches!(op.kind, OperationKind::Storage(_))
        })
        .collect();
    assert_eq!(actual_memory.len(), usize::from(!folded));
    if !folded {
        let mut condition_origins = lowered
            .scoped_slot_origins
            .as_ref()
            .unwrap()
            .iter()
            .filter(|slot| slot.identity.original_local() == Some(4));
        let origin = condition_origins.next().unwrap();
        assert!(condition_origins.next().is_none());
        match (origin.source, &actual_memory[0].kind) {
            (ScopedAllocationSourceV29::Legacy, OperationKind::Load { pointer, .. }) => {
                assert_eq!(*pointer, origin.pointer);
            }
            (
                ScopedAllocationSourceV29::OriginalObject { .. },
                OperationKind::Storage(ScopedObjectOperationV29::ReadValue { address, access }),
            ) => {
                assert_eq!(*address, origin.pointer);
                assert_eq!(*access, MemoryAccess::new(AddressSpace::Private, 1));
            }
            _ => panic!(
                "condition read must match its original allocation: {:?}",
                actual_memory[0]
            ),
        }
        let [result] = actual_memory[0].results.as_slice() else {
            panic!("the original condition read must have exactly one result");
        };
        assert_eq!(result.ty, Type::BOOL);
        assert!(
            matches!(block.terminator, Some(Terminator::ConditionalBranch { condition, .. })
            if condition == result.id)
        );
    }
    let rows: Vec<_> = anchors
        .rows
        .iter()
        .filter(|row| {
            row.source.is_some_and(|frame| {
                frame.site == execution_site_v29(SemanticBlockIdV1::from_index(0), None)
            })
        })
        .collect();
    assert_eq!(rows.len(), expected.len() + 2 + usize::from(!folded));
    let mut kills = Vec::new();
    let mut failures = Vec::new();
    for row in rows {
        match row.kind {
            ScopedMemoryAnchorKindV29::Object(index) => {
                assert!(!folded);
                assert_eq!(row.position, start);
                let site = execution_site_v29(SemanticBlockIdV1::from_index(0), None);
                assert_eq!(
                    row.source,
                    Some(ScopedMemoryFrameV29::operand(
                        site,
                        Some(ExecutionOperandV29::AssertCondition)
                    ))
                );
                let payload = &anchors.objects[index];
                assert_eq!(
                    actual_memory[0].kind,
                    OperationKind::Storage(payload.operation)
                );
                assert_eq!(payload.result, Some(actual_memory[0].results[0].id));
                let ScopedObjectRoleV29::ReadValue {
                    source: endpoint,
                    read: ScopedObjectReadOriginV29::Original(read),
                } = payload.role
                else {
                    panic!("the typed condition must retain its original read recipe");
                };
                let original = source.locals()[4].ty();
                assert_eq!(
                    endpoint.source,
                    ScopedObjectSourceV29::Place {
                        site,
                        role: ExecutionOperandV29::AssertCondition,
                        local: SemanticLocalIdV1::from_index(4),
                        prefix: 0,
                    }
                );
                let ScopedObjectIdentityV29::Local {
                    instance: owner,
                    local,
                    generation,
                } = endpoint.object
                else {
                    panic!(
                        "the condition must read its original local, not a replacement reference"
                    );
                };
                assert_eq!(owner, instance);
                assert_eq!(local.index(), 4);
                let origin = lowered
                    .scoped_slot_origins
                    .as_ref()
                    .unwrap()
                    .iter()
                    .find(|origin| {
                        origin.identity
                            == (ScopedAllocationIdentityV29::OriginalObject {
                                local: 4,
                                generation,
                            })
                    })
                    .unwrap();
                let ScopedAllocationSourceV29::OriginalObject { schema, .. } = origin.source else {
                    panic!("the typed condition must retain its original schema");
                };
                assert_eq!(
                    (endpoint.root_schema, endpoint.projected_schema),
                    (schema, schema)
                );
                assert_eq!(
                    (endpoint.root_type, endpoint.projected_type),
                    (original, original)
                );
                assert_eq!((endpoint.source_path.count, endpoint.path.count), (0, 0));
                assert_eq!(
                    (read.site, read.role, read.prefix, read.ty),
                    (site, ExecutionOperandV29::AssertCondition, 0, original)
                );
                let event = match read.occurrence {
                    ScopedMemoryOccurrenceV29::Promoted { event, .. }
                    | ScopedMemoryOccurrenceV29::Retained { event } => event,
                };
                let occurrences = instances.occurrences(instance).unwrap();
                let original = &occurrences.events()[event];
                assert_eq!(original.role(), ExecutionEventV29::BaseUse);
                assert_eq!(original.site(), site);
                assert_eq!(original.operand(), ExecutionOperandV29::AssertCondition);
                assert_eq!(original.event().variable().get(), 4);
            }
            ScopedMemoryAnchorKindV29::Access { .. } => {
                assert!(!folded);
                assert_eq!(
                    row.source.unwrap().role,
                    Some(ScopedMemoryRoleV29::Operand(
                        ExecutionOperandV29::AssertCondition
                    ))
                );
                assert_eq!(row.position, start);
            }
            ScopedMemoryAnchorKindV29::Kill {
                event,
                local,
                cause,
            } => {
                assert_eq!(cause, ScopedMemoryKillV29::Move);
                assert_eq!(row.position, start + usize::from(!folded));
                let occurrences = instances.occurrences(instance).unwrap();
                let original = &occurrences.events()[event];
                assert_eq!(original.role(), ExecutionEventV29::MoveKill);
                assert_eq!(original.site(), row.source.unwrap().site);
                assert_eq!(
                    Some(ScopedMemoryRoleV29::Operand(original.operand())),
                    row.source.unwrap().role
                );
                kills.push((original.operand(), local));
            }
            ScopedMemoryAnchorKindV29::FailureRead { event, local } => {
                let occurrences = instances.occurrences(instance).unwrap();
                let original = &occurrences.events()[event];
                assert_eq!(original.role(), ExecutionEventV29::BaseUse);
                assert_eq!(original.site(), row.source.unwrap().site);
                assert_eq!(row.position, start + usize::from(!folded));
                failures.push((original.operand(), local));
            }
            ScopedMemoryAnchorKindV29::ScalarMove { .. } => {
                panic!("whole-local assertion fixture cannot issue a projected scalar move");
            }
        }
    }
    assert_eq!(kills, expected);
    assert_eq!(
        failures,
        [
            (ExecutionOperandV29::AssertMessage(0), 2),
            (ExecutionOperandV29::AssertMessage(1), 3)
        ]
    );
    assert_eq!(end, start + usize::from(!folded));
}

fn observe_runtime(
    _source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    _budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    OBSERVED.set(OBSERVED.get() + 1);
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        assert_eq!(slots.slots[item.slots.clone()].len(), 3);
        check_assertion(
            instances,
            item.instance,
            emitted[item.instance.index()].as_ref().unwrap(),
            false,
        );
    }
    assert_eq!(helpers, 2);
    Ok(())
}

#[test]
fn runtime_discarded_moves_invalidate_alias_reads_without_diagnostic_loads() {
    // V995 successor: only the common condition invalidates success state.
    for move_condition in [false, true] {
        for move_message in [false, true] {
            for reinitialize in [false, true] {
                let fixture = ScopedFixture::AssertionSlots {
                    move_condition,
                    move_message,
                    reinitialize,
                };
                let (result, _, _) =
                    run_original_assertion_v29(fixture, observe_runtime, LIMIT, LIMIT);
                if reinitialize || !move_condition {
                    assert_original_assertion_completed_v29(&result);
                } else {
                    assert!(!ASSERTION_PHYSICAL_COMPLETED_V29.get());
                    assert_eq!(
                        OBSERVED.get(),
                        0,
                        "the original source plan rejects the post-move alias read before emission"
                    );
                    assert!(
                        matches!(
                            result,
                            Err(ProductionSourceOwnedViewErrorV18::Source(
                                ProductionPendingScopedSourceErrorV29::Source(
                                    ProductionSemanticKirErrorV1::Unsupported {
                                        function: 0,
                                        block: None,
                                        statement: None,
                                        detail: "source raw pointer reads an undefined referent",
                                    }
                                )
                            ))
                        ),
                        "{fixture:?}: {result:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn original_assertion_aliases_still_refuse_nonconsuming_legacy_memory_admission() {
    fn reject_legacy(
        source: &ExecutionLifecycleSourceV29<'_>,
        instances: &ExecutionInstancesV29<'_>,
        emitted: &mut [Option<LoweredFunctionResultV1>],
        slots: &OwnedScopedSourceSlotsV29,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        observe_runtime(source, instances, emitted, slots, budget)?;
        let floor = budget.storage();
        let result = scoped_slot_uses_v29::check_scoped_source_slot_uses_v29(
            instances, emitted, slots, 1024, budget,
        );
        assert!(matches!(
            result,
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "original raw source requires consuming expanded physical admission",
                ..
            })
        ));
        assert_eq!(budget.storage(), floor);
        Err(unsupported(0, None, None, STOP))
    }
    let fixture = ScopedFixture::AssertionSlots {
        move_condition: false,
        move_message: false,
        reinitialize: false,
    };
    let (result, _, _) = run(false, fixture, reject_legacy, LIMIT, LIMIT);
    assert_eq!(OBSERVED.get(), 1, "{result:?}");
    assert!(
        is_stopped(&result),
        "the legacy refusal control must execute: {result:?}"
    );
}

fn observe_folded(
    source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        let floor = budget.storage();
        with_canonical_call_scratch_v1(budget, |budget| {
            let instance = item.instance;
            let row = instances.instance(instance).unwrap();
            let mut private = PrivateArrayLazyBudgetV1::new(1, 1024);
            let mut stage = "source cursor preparation";
            let result = with_original_assertion_frame_v29(
                source,
                instances,
                instance,
                budget,
                |cursor, signatures, budget| {
                    stage = "original instance plan";
                    let mut plan = signatures.instance_plan_v29(
                        instances,
                        instance,
                        FunctionId::new("folded_assert_probe"),
                        item.placement,
                        budget,
                    )?;
                    assert_eq!(plan.parameter_declarations.len(), 1);
                    assert_eq!(plan.parameter_declarations[0].1, 1);
                    assert!(plan.parameter_local_bindings.is_empty());
                    emission_push_v1(
                        &mut plan.parameter_local_bindings,
                        PlannedParameterLocalBindingV1::Direct {
                            local: 1,
                            value: plan.parameter_values[0],
                            ty: plan.parameter_types[0].clone(),
                        },
                        budget,
                    )?;
                    let mut producer = ExecutionLifecycleProducerV29::new(
                        source,
                        instances,
                        instance,
                        item.placement,
                        budget,
                    )?;
                    stage = "function lowering";
                    let lowered = lower_one_source_function_with_calls_v29(
                        instances.owner().source_semantic(),
                        &plan,
                        row.ssa(),
                        &BTreeMap::new(),
                        signatures,
                        None,
                        // Exercise the private folded emitter, not admission of a folding proof.
                        BTreeSet::from([0]),
                        1,
                        false,
                        1024,
                        None,
                        &mut private,
                        None,
                        budget,
                        item.placement,
                        Some(cursor),
                        None,
                        Some(&mut producer),
                    )?;
                    stage = "assertion check";
                    check_assertion(instances, instance, &lowered, true);
                    stage = "source cursor cleanup";
                    Ok(())
                },
            );
            if let Err(error) = &result {
                eprintln!(
                    "folded assertion instance {} at {stage}: {error:?}",
                    instance.index()
                );
            }
            result?;
            Ok(())
        })?;
        assert_eq!(budget.storage(), floor);
    }
    observe_runtime(source, instances, emitted, slots, budget)
}

#[test]
fn folded_conditions_and_diagnostics_preserve_same_gap_move_order_without_loads() {
    for move_condition in [false, true] {
        for move_message in [false, true] {
            let fixture = ScopedFixture::AssertionSlots {
                move_condition,
                move_message,
                reinitialize: true,
            };
            let (result, _, _) = run_original_assertion_v29(fixture, observe_folded, LIMIT, LIMIT);
            assert_original_assertion_completed_v29(&result);
        }
    }
}

#[test]
fn discarded_move_recording_keeps_exact_work_and_storage_failures() {
    let fixture = ScopedFixture::AssertionSlots {
        move_condition: true,
        move_message: true,
        reinitialize: true,
    };
    let (result, work, storage) =
        run_original_assertion_v29(fixture, observe_runtime, LIMIT, LIMIT);
    assert_original_assertion_completed_v29(&result);
    let exact = run_original_assertion_v29(fixture, observe_runtime, work, storage).0;
    assert_original_assertion_completed_v29(&exact);
    let error = run_original_assertion_v29(fixture, observe_runtime, work - 1, storage)
        .0
        .unwrap_err();
    assert!(matches!(
        assertion_source_resource_v29(error),
        ArgumentResourceV1::Work(_)
    ));
    let error = run_original_assertion_v29(fixture, observe_runtime, work, storage - 1)
        .0
        .unwrap_err();
    assert!(matches!(
        assertion_source_resource_v29(error),
        ArgumentResourceV1::Storage(_)
    ));
}

fn reject_failure_receipt_tampering(
    source: &ExecutionLifecycleSourceV29<'_>,
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
    let lowered = emitted[item.instance.index()].as_mut().unwrap();
    let original = lowered.scoped_memory_anchors.as_ref().unwrap().rows.clone();
    let witness = original
        .iter()
        .position(|row| matches!(row.kind, ScopedMemoryAnchorKindV29::FailureRead { .. }))
        .unwrap();
    let floor = budget.storage();
    with_canonical_call_scratch_v1(budget, |budget| {
        check_scoped_memory_anchors_v29(
            instances,
            item,
            lowered,
            &slots.slots[item.slots.clone()],
            budget,
        )
    })?;
    for mutation in 0..8 {
        let mut changed = original.clone();
        let row = &mut changed[witness];
        match mutation {
            0 => {
                changed.insert(witness, original[witness]);
            }
            1 => {
                changed.remove(witness);
            }
            2 => {
                let ScopedMemoryAnchorKindV29::FailureRead { local, .. } = &mut row.kind else {
                    unreachable!()
                };
                *local += 1;
            }
            3 => {
                let ScopedMemoryAnchorKindV29::FailureRead { event, .. } = &mut row.kind else {
                    unreachable!()
                };
                *event += 1;
            }
            4 => {
                row.source.as_mut().unwrap().role = Some(ScopedMemoryRoleV29::Operand(
                    ExecutionOperandV29::AssertCondition,
                ))
            }
            5 => {
                row.source.as_mut().unwrap().site =
                    execution_site_v29(SemanticBlockIdV1::from_index(1), None)
            }
            6 => {
                row.kind = ScopedMemoryAnchorKindV29::Access {
                    pointer: lowered.scoped_slot_origins.as_ref().unwrap()[0].pointer,
                    payload: None,
                }
            }
            7 => row.position = usize::MAX,
            _ => unreachable!(),
        }
        lowered.scoped_memory_anchors.as_mut().unwrap().rows = changed;
        let error = with_canonical_call_scratch_v1(budget, |budget| {
            check_scoped_memory_anchors_v29(
                instances,
                item,
                lowered,
                &slots.slots[item.slots.clone()],
                budget,
            )
        })
        .unwrap_err();
        let expected = match mutation {
            6 => "compiler enum access lacks its exact retained payload role",
            7 => "typed object source payload differs from its actual operation",
            _ => "scoped memory anchors differ from their source instance",
        };
        assert!(
            matches!(error, ProductionSemanticKirErrorV1::Unsupported { detail, .. } if detail == expected),
            "mutation {mutation}: {error:?}"
        );
        assert_eq!(budget.storage(), floor);
        lowered.scoped_memory_anchors.as_mut().unwrap().rows = original.clone();
    }
    observe_runtime(source, instances, emitted, slots, budget)
}

#[test]
fn failure_receipts_require_exact_source_census_without_physical_pointer() {
    let fixture = ScopedFixture::AssertionSlots {
        move_condition: false,
        move_message: true,
        reinitialize: false,
    };
    let (result, _, _) =
        run_original_assertion_v29(fixture, reject_failure_receipt_tampering, LIMIT, LIMIT);
    assert_original_assertion_completed_v29(&result);
}

#[test]
fn repeated_original_diagnostic_local_keeps_distinct_operand_reads_and_move_order() {
    fn observe(
        _: &ExecutionLifecycleSourceV29<'_>,
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
            let original = instances.instance(item.instance).unwrap().declaration();
            let SemanticTerminatorKindV1::Assert {
                message: SemanticAssertMessageV1::BoundsCheck { length, index },
                ..
            } = original.blocks()[0].terminator().kind()
            else {
                panic!("bounds assertion");
            };
            let SemanticOperandV1::Copy(left) = length else {
                panic!("first copy");
            };
            let (SemanticOperandV1::Copy(right) | SemanticOperandV1::Move(right)) = index else {
                panic!("second read");
            };
            assert_eq!(left.local(), right.local());
            assert!(!std::ptr::eq(left, right));
            // The owner-local lookup distinguishes original place occurrences;
            // its pointer coordinate is not transferable source authority.
            let site = SourceReferenceSiteV29 {
                instance: item.instance,
                block: SemanticBlockIdV1::from_index(0),
                statement: None,
            };
            assert_ne!(
                source_reference_access_key_v29(site, left, SourceReferenceAccessV29::Read),
                source_reference_access_key_v29(site, right, SourceReferenceAccessV29::Read)
            );
            let lowered = emitted[item.instance.index()].as_ref().unwrap();
            let anchors = lowered.scoped_memory_anchors.as_ref().unwrap();
            let occurrences = instances.occurrences(item.instance).unwrap();
            let mut roles = Vec::new();
            for row in &anchors.rows {
                if matches!(row.kind, ScopedMemoryAnchorKindV29::FailureRead { .. }) {
                    let place =
                        checked_scoped_failure_read_v29(original, &occurrences, row, budget)?;
                    assert_eq!(place.local().index(), 2);
                    roles.push(row.source.unwrap().role);
                }
            }
            assert_eq!(
                roles,
                vec![
                    Some(ScopedMemoryRoleV29::Operand(
                        ExecutionOperandV29::AssertMessage(0)
                    )),
                    Some(ScopedMemoryRoleV29::Operand(
                        ExecutionOperandV29::AssertMessage(1)
                    ))
                ]
            );
            helpers += 1;
        }
        assert_eq!(helpers, 2);
        OBSERVED.set(OBSERVED.get() + 1);
        Ok(())
    }
    struct Restore(u8);
    impl Drop for Restore {
        fn drop(&mut self) {
            ASSERTION_REPEATED_LOCAL_V29.set(self.0);
        }
    }
    let _restore = Restore(ASSERTION_REPEATED_LOCAL_V29.get());
    for mode in [1, 2] {
        ASSERTION_REPEATED_LOCAL_V29.set(mode);
        let fixture = ScopedFixture::AssertionSlots {
            move_condition: false,
            move_message: false,
            reinitialize: false,
        };
        let (result, _, _) = run_original_assertion_v29(fixture, observe, LIMIT, LIMIT);
        assert_original_assertion_completed_v29(&result);
    }
}

fn check_typed_failure_only_moves(
    _source: &ExecutionLifecycleSourceV29<'_>,
    instances: &ExecutionInstancesV29<'_>,
    emitted: &mut [Option<LoweredFunctionResultV1>],
    slots: &OwnedScopedSourceSlotsV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    OBSERVED.set(OBSERVED.get() + 1);
    let mut helpers = 0;
    for item in slots
        .instances
        .iter()
        .filter(|item| item.function.index() == 3)
    {
        helpers += 1;
        let lowered = emitted[item.instance.index()].as_mut().unwrap();
        check_assertion(instances, item.instance, lowered, false);
        for local in [2, 3] {
            assert!(slots.slots[item.slots.clone()].iter().any(|slot|
                matches!(slot.origin.identity, ScopedAllocationIdentityV29::OriginalObject { local: found, .. }
                    if found == local)));
        }
        let original = lowered.scoped_memory_anchors.as_ref().unwrap().rows.clone();
        let kills: Vec<_> = original
            .iter()
            .enumerate()
            .filter_map(|(index, row)| {
                matches!(
                    row.kind,
                    ScopedMemoryAnchorKindV29::Kill {
                        cause: ScopedMemoryKillV29::Move,
                        ..
                    }
                )
                .then_some(index)
            })
            .collect();
        assert_eq!(
            kills.len(),
            2,
            "only failure diagnostics move in this source"
        );
        let floor = budget.storage();
        let positive = with_canonical_call_scratch_v1(budget, |budget| {
            check_scoped_memory_anchors_v29(
                instances,
                item,
                lowered,
                &slots.slots[item.slots.clone()],
                budget,
            )
        });
        assert!(
            positive.is_ok(),
            "typed failure-only move census: {positive:?}"
        );
        assert_eq!(budget.storage(), floor);
        for &witness in &kills {
            for fault in 0..5 {
                let mut changed = original.clone();
                match fault {
                    0 => {
                        changed.remove(witness);
                    }
                    1 => changed.insert(witness, original[witness]),
                    2 => {
                        changed[witness].source.as_mut().unwrap().role = Some(
                            ScopedMemoryRoleV29::Operand(ExecutionOperandV29::AssertCondition),
                        )
                    }
                    3 => {
                        changed[witness].source.as_mut().unwrap().site =
                            execution_site_v29(SemanticBlockIdV1::from_index(1), None)
                    }
                    4 => {
                        let ScopedMemoryAnchorKindV29::Kill { local, .. } =
                            &mut changed[witness].kind
                        else {
                            unreachable!();
                        };
                        *local = 4;
                    }
                    _ => unreachable!(),
                }
                lowered.scoped_memory_anchors.as_mut().unwrap().rows = changed;
                let rejected = with_canonical_call_scratch_v1(budget, |budget| {
                    check_scoped_memory_anchors_v29(
                        instances,
                        item,
                        lowered,
                        &slots.slots[item.slots.clone()],
                        budget,
                    )
                });
                assert!(
                    matches!(
                        rejected,
                        Err(ProductionSemanticKirErrorV1::Unsupported {
                            detail: "scoped memory anchors differ from their source instance",
                            ..
                        })
                    ),
                    "failure-only kill {witness}, fault {fault}: {rejected:?}"
                );
                assert_eq!(budget.storage(), floor);
                lowered.scoped_memory_anchors.as_mut().unwrap().rows = original.clone();
            }
        }
    }
    assert_eq!(helpers, 2);
    Err(unsupported(0, None, None, STOP))
}

#[test]
fn typed_discarded_diagnostics_preserve_exact_failure_only_move_anchors() {
    let fixture = ScopedFixture::AssertionSlots {
        move_condition: false,
        move_message: true,
        reinitialize: false,
    };
    let (result, _, _) = run(false, fixture, check_typed_failure_only_moves, LIMIT, LIMIT);
    assert_eq!(OBSERVED.get(), 1, "{result:?}");
    assert!(is_stopped(&result), "{result:?}");
}

#[test]
fn discarded_move_classification_has_independent_exact_and_one_short_work() {
    let owner = super::super::fixtures::assertion_slots_owner(false, true, false);
    let original = &owner.source_semantic().functions()[3];
    let SemanticTerminatorKindV1::Assert {
        message:
            SemanticAssertMessageV1::BoundsCheck {
                length: SemanticOperandV1::Move(place),
                ..
            },
        ..
    } = original.blocks()[0].terminator().kind()
    else {
        panic!("original diagnostic move")
    };
    for typed in [false, true] {
        let identity = if typed {
            ScopedAllocationIdentityV29::OriginalObject {
                local: place.local().index(),
                generation: 0,
            }
        } else {
            ScopedAllocationIdentityV29::LegacyLocal(place.local().index())
        };
        let storage = if typed {
            SemanticRetainedStorageV29::Object {
                cell: 0,
                schema: fe2o3_kernel_ir::StorageLayoutIdV1(0),
                bytes: 4,
                alignment: 4,
            }
        } else {
            SemanticRetainedStorageV29::ScalarArray {
                kernel_type: Type::Scalar(ScalarType::U32),
                alignment: 4,
                array: None,
            }
        };
        // Borrowed classifier inputs are outside its allocation-free query budget.
        let slots = BTreeMap::from([(
            identity,
            SemanticRetainedLocalSlotV1 {
                pointer: ValueId(1),
                semantic_type: place.ty(),
                storage,
            },
        )]);
        // Two independently prepaid single-entry tree lookups (32 each), then 8.
        for limit in [72, 71] {
            let mut work = CanonicalKernelIrWorkBudgetV1::new(limit);
            let mut budget = ArgumentBudgetV1::new(&mut work, 0);
            let result = scoped_discarded_move_cause_v29(
                &slots,
                owner.source_semantic().types(),
                place,
                &mut budget,
            );
            if limit == 72 {
                assert_eq!(result.unwrap(), Some(ScopedMemoryKillV29::Move));
                assert_eq!(budget.work(), 72);
            } else {
                assert!(matches!(
                    result,
                    Err(
                        ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                            ArgumentResourceV1::Work(_)
                        )
                    )
                ));
                assert_eq!(budget.work(), 64, "the final eight-unit debit is atomic");
            }
            assert_eq!(budget.storage(), 0);
        }
    }
}
