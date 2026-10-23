fn original_holder_borrow(
    owner: &ProductionSemanticSsaOwnerV1,
    local: u32,
) -> (SemanticBlockIdV1, u32) {
    let mut original = None;
    for (bi, block) in owner.source_semantic().functions()[0]
        .blocks()
        .iter()
        .enumerate()
    {
        for (si, statement) in block.statements().iter().enumerate() {
            if matches!(statement.kind(), SemanticStatementKindV1::Assign(assignment)
                if assignment.destination().local().index() == local && matches!(assignment.value().kind(),
                    SemanticRvalueKindV1::Borrow { kind: SemanticBorrowKindV1::Shared, .. }))
            {
                assert!(
                    original
                        .replace((SemanticBlockIdV1::from_index(bi as u32), si as u32))
                        .is_none()
                );
            }
        }
    }
    original.expect("unique original shared borrow")
}
#[test]
fn holder_initial_live_and_terminal_dead_preserve_actual_source_and_whole_owner() {
    for mutation in [
        Mutation::HolderEntryLive,
        Mutation::HolderLocalLive,
        Mutation::HolderOrderedStorage,
        Mutation::RawShuffleHolderOrderedStorage,
    ] {
        let owner = fixture(mutation).unwrap_or_else(|error| panic!("{mutation:?}: {error}"));
        check(&owner).unwrap_or_else(|error| panic!("{mutation:?}: {error:?}"));
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        with_checked_reference_call_v1(
            &owner,
            SemanticFunctionIdV1::from_index(0),
            SemanticBlockIdV1::from_index(2),
            &mut budget,
            |view, budget| {
                assert!(std::ptr::eq(view.owner(), &owner));
                for local in [4, 5] {
                    super::super::require_closed_wave64_reference_holder_v1(
                        &owner,
                        SemanticFunctionIdV1::from_index(0),
                        local,
                        Some(original_holder_borrow(&owner, local)),
                        budget,
                    )
                    .unwrap_or_else(|error| panic!("{mutation:?}, holder {local}: {error:?}"));
                }
                Ok(())
            },
        )
        .unwrap();
        if matches!(mutation, Mutation::RawShuffleHolderOrderedStorage) {
            let whole = ProductionSemanticKirOwnerV1::try_lower(
                owner.into_source_owner().unwrap(),
                ProductionSemanticKirLimitsV1::default(),
            )
            .expect("normal raw-shuffle owner with ordered holder storage");
            whole
                .verify_equivalence()
                .expect("independent whole-owner source replay");
            assert!(
                whole
                    .module()
                    .functions
                    .iter()
                    .flat_map(|function| function.body.iter())
                    .flat_map(|body| &body.blocks)
                    .flat_map(|block| &block.operations)
                    .any(|operation| matches!(
                        operation.kind,
                        OperationKind::Wave(WaveOperation {
                            kind: WaveOperationKind::ShuffleIndex { .. },
                            ..
                        })
                    ))
            );
        }
    }
}
#[test]
fn holder_ordering_rejects_restarts_cycles_unreachable_lives_aliases_and_formal_markers() {
    for mutation in [
        Mutation::HolderLateLive,
        Mutation::HolderRestart,
        Mutation::HolderUnreachableLive,
        Mutation::HolderLiveBackedge,
        Mutation::HolderDeadBackedge,
        Mutation::HolderAliasInitialLive,
        Mutation::HolderFormalLive,
        Mutation::HolderFormalDead,
    ] {
        let owner = fixture(mutation).unwrap_or_else(|error| {
            panic!("{mutation:?} must admit and capture original SSA: {error}")
        });
        let formal = matches!(
            mutation,
            Mutation::HolderFormalLive | Mutation::HolderFormalDead
        );
        let (function, local, borrow) = if formal {
            (1, 1, None)
        } else {
            (0, 4, Some(original_holder_borrow(&owner, 4)))
        };
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        let result = super::super::require_closed_wave64_reference_holder_v1(
            &owner,
            SemanticFunctionIdV1::from_index(function),
            local,
            borrow,
            &mut budget,
        );
        assert!(
            matches!(
                &result,
                Err(ProductionSemanticKirErrorV1::Unsupported { .. }
                    | ProductionSemanticKirErrorV1::CorrespondenceMismatch)
            ),
            "{mutation:?} must fail the semantic contract, not resource accounting: {result:?}"
        );
    }
}
#[test]
fn holder_dead_before_call_is_rejected_by_original_ssa_before_reference_transport() {
    // This source must fail in the original SSA planner, before a holder
    // relation can exist. Do not weaken admission to reach the later guard.
    match fixture(Mutation::HolderDeadBeforeCall) {
        Err(error) => assert_eq!(
            error,
            "Planner { function: SemanticFunctionIdV1(0), error: UndefinedAtUse { block: SsaBlockIdV1(2), event: 7, variable: SsaVariableIdV1(4) } }"
        ),
        Ok(_) => panic!("original SSA accepted a call through a dead reference holder"),
    }
}

#[test]
fn holder_storage_query_preserves_original_receipt_and_work_denial() {
    let owner =
        fixture(Mutation::HolderOrderedStorage).expect("actual holder lifetime source owner");
    let held = owner.occurrence_storage().unwrap().retained_storage();
    let borrow = original_holder_borrow(&owner, 4);
    let mut work = Work::new(0);
    let mut budget = Budget::new(&mut work, held);
    budget.reserve_storage(held).unwrap();
    let ledger = budget.work_ledger_identity_v1();
    assert!(matches!(
        super::super::require_ordered_wave64_holder_storage_v1(
            &owner,
            SemanticFunctionIdV1::from_index(0),
            4,
            borrow,
            Some((0, 0)),
            &mut budget
        ),
        Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(Resource::Work(_)))
    ));
    assert_eq!(budget.failed_work(), Some(4));
    assert_eq!(budget.storage(), held);
    assert!(budget.work_ledger_identity_v1() == ledger);
    let mut work = Work::new(100);
    let mut budget = Budget::new(&mut work, held - 1);
    budget.reserve_storage(held - 1).unwrap();
    assert!(matches!(
        super::super::require_ordered_wave64_holder_storage_v1(
            &owner,
            SemanticFunctionIdV1::from_index(0),
            4,
            borrow,
            Some((0, 0)),
            &mut budget
        ),
        Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(Resource::Accounting))
    ));
    assert_eq!(budget.storage(), held - 1);
    assert!(budget.failed_storage().is_none());
}

#[test]
fn decreasing_u32_counter_replays_genuine_raw_shuffle_whole_owner() {
    let source =
        fixture(Mutation::ShiftCounter).expect("original loop source admission and SSA capture");
    let held = source.occurrence_storage().unwrap().retained_storage();
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, 10_000_000);
    budget.reserve_storage(held).unwrap();
    assert_eq!(
        super::super::checked_wave64_shift_counter_bound_v1(
            &source,
            SemanticFunctionIdV1::from_index(1),
            7,
            &mut budget
        )
        .unwrap(),
        Some(33)
    );
    let owner = ProductionSemanticKirOwnerV1::try_lower(
        source.into_source_owner().unwrap(),
        ProductionSemanticKirLimitsV1::default(),
    )
    .expect("public lowering of actual source loop and raw shuffle");
    owner
        .verify_equivalence()
        .expect("independent original-source whole-owner replay");
    assert!(
        owner
            .module()
            .functions
            .iter()
            .flat_map(|function| function.body.iter())
            .flat_map(|body| &body.blocks)
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                operation.kind,
                OperationKind::Wave(WaveOperation {
                    kind: WaveOperationKind::ShuffleIndex { .. },
                    ..
                })
            ))
    );
}
#[test]
fn decreasing_counter_contract_refuses_changed_source_and_aliases() {
    for mutation in [
        Mutation::ShiftCounterZero,
        Mutation::ShiftCounterOverwide,
        Mutation::ShiftCounterZeroShift,
        Mutation::ShiftCounterWideShift,
        Mutation::ShiftCounterDynamicShift,
        Mutation::ShiftCounterReassign,
        Mutation::ShiftCounterInitReentry,
        Mutation::ShiftCounterBorrow,
        Mutation::ShiftCounterRawAddress,
        Mutation::ShiftCounterSigned,
    ] {
        let source = fixture(mutation).unwrap_or_else(|error| {
            panic!("{mutation:?} must admit and capture before contract refusal: {error}")
        });
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(source.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        let result = super::super::checked_wave64_shift_counter_bound_v1(
            &source,
            SemanticFunctionIdV1::from_index(1),
            7,
            &mut budget,
        );
        assert!(
            matches!(&result, Ok(None)),
            "{mutation:?}: semantic refusal required, got {result:?}"
        );
        assert!(budget.failed_work().is_none());
        assert!(budget.failed_storage().is_none());
    }
}
#[test]
fn decreasing_counter_uses_exact_original_work_and_input_storage_limits() {
    let source = fixture(Mutation::ShiftCounter).unwrap();
    let held = source.occurrence_storage().unwrap().retained_storage();
    let query = |budget: &mut Budget<'_>| {
        super::super::checked_wave64_shift_counter_bound_v1(
            &source,
            SemanticFunctionIdV1::from_index(1),
            7,
            budget,
        )
    };
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, held);
    budget.reserve_storage(held).unwrap();
    assert_eq!(query(&mut budget).unwrap(), Some(33));
    let required = budget.work();
    assert!(required > 0);
    assert_eq!(budget.storage(), held);
    for limit in [0, required - 1] {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, held);
        budget.reserve_storage(held).unwrap();
        let result = query(&mut budget);
        assert!(matches!(
            result,
            Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(Resource::Work(_)))
        ));
        assert!(budget.failed_work().is_some());
        assert_eq!(budget.storage(), held);
    }
    let mut work = Work::new(required);
    let mut budget = Budget::new(&mut work, held);
    budget.reserve_storage(held).unwrap();
    assert_eq!(query(&mut budget).unwrap(), Some(33));
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, held - 1);
    budget.reserve_storage(held - 1).unwrap();
    assert!(matches!(
        query(&mut budget),
        Err(ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(Resource::Accounting))
    ));
    assert_eq!(budget.storage(), held - 1);
}
