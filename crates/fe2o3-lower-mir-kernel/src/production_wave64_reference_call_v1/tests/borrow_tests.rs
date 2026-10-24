#[test]
fn root_borrow_accessor_retains_exact_source_initializer_and_lane_issuer() {
    let owner = fixture(Mutation::None).unwrap();
    let root = SemanticFunctionIdV1::from_index(0);
    let block = SemanticBlockIdV1::from_index(2);
    let mut work = Work::new(10_000_000);
    let mut budget = Budget::new(&mut work, 10_000_000);
    budget
        .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
        .unwrap();
    let producer = result_value(&owner, root, SemanticBlockIdV1::from_index(0), &mut budget)
        .unwrap()
        .unwrap();
    with_checked_reference_borrow_v1(&owner, root, block, 1, &mut budget, |view, _| {
        assert!(std::ptr::eq(view.owner(), &owner));
        assert_eq!(view.function(), root);
        assert_eq!(view.block(), block);
        assert_eq!(view.statement(), 1);
        assert_eq!(view.reference_type(), SemanticTypeIdV1::from_index(6));
        assert_eq!(view.snapshot_type(), SemanticTypeIdV1::from_index(5));
        assert_eq!(view.lane_field(), 0);
        assert_eq!(view.lane_producer(), (root, producer));
        assert_eq!(view.constructor(), (block, 0));
        let SemanticStatementKindV1::Assign(original) =
            owner.source_semantic().functions()[0].blocks()[2].statements()[1].kind()
        else {
            unreachable!()
        };
        assert!(std::ptr::eq(view.source_rvalue(), original.value()));
        assert!(matches!(
            view.initializer_rvalue().kind(),
            SemanticRvalueKindV1::Aggregate(_)
        ));
        Ok(())
    })
    .unwrap();
    for (function, statement) in [
        (root, 0),
        (root, 2),
        (SemanticFunctionIdV1::from_index(1), 0),
    ] {
        assert!(matches!(
            with_checked_reference_borrow_v1(
                &owner,
                function,
                block,
                statement,
                &mut budget,
                |_, _| Ok(())
            ),
            Err(ReferenceErrorV1::Unavailable(_))
        ));
    }
}

#[test]
fn retained_source_exposure_and_lifetime_mutations_reach_checker_and_refuse() {
    for mutation in [
        Mutation::StorageDeadSnapshot,
        Mutation::ProjectedExposure,
        Mutation::MutableExposure,
        Mutation::ContextCallOverwrite,
        Mutation::StorageLiveSnapshot,
        Mutation::StorageLiveContext,
        Mutation::DeadLiveSnapshot,
        Mutation::DeadLiveContext,
    ] {
        let owner = fixture(mutation)
            .unwrap_or_else(|e| panic!("{mutation:?} must reach source checker: {e}"));
        assert!(
            matches!(check(&owner), Err(ReferenceErrorV1::Unavailable(_))),
            "{mutation:?} must fail closed retained source relation"
        );
    }
}
#[test]
fn root_context_borrow_retains_actual_issuer_call_without_fabricated_ssa() {
    for mutation in [Mutation::None, Mutation::InitialStorageLive] {
        let owner = fixture(mutation).expect("actual inert context borrow source admission");
        let root = SemanticFunctionIdV1::from_index(0);
        let block = SemanticBlockIdV1::from_index(2);
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        with_checked_context_borrow_v1(&owner, root, block, 2, &mut budget, |view, budget| {
            assert!(std::ptr::eq(view.owner(), &owner));
            assert_eq!(
                (view.function(), view.block(), view.statement()),
                (root, block, 2)
            );
            assert_eq!(view.reference_type(), SemanticTypeIdV1::from_index(7));
            assert_eq!(view.context_type(), SemanticTypeIdV1::from_index(4));
            assert_eq!(
                view.issuer(),
                (
                    root,
                    SemanticBlockIdV1::from_index(1),
                    SemanticLocalIdV1::from_index(2)
                )
            );
            assert!(std::ptr::eq(
                view.source_call(),
                call_at(&owner, root, SemanticBlockIdV1::from_index(1)).unwrap()
            ));
            let SemanticStatementKindV1::Assign(original) =
                owner.source_semantic().functions()[0].blocks()[2].statements()[2].kind()
            else {
                unreachable!()
            };
            assert!(std::ptr::eq(view.source_rvalue(), original.value()));
            assert!(
                result_value(&owner, root, SemanticBlockIdV1::from_index(1), budget)?.is_none()
            );
            Ok(())
        })
        .unwrap();
        for (function, statement) in [
            (root, 0),
            (root, 1),
            (SemanticFunctionIdV1::from_index(1), 0),
        ] {
            assert!(matches!(
                with_checked_context_borrow_v1(
                    &owner,
                    function,
                    block,
                    statement,
                    &mut budget,
                    |_, _| Ok(())
                ),
                Err(ReferenceErrorV1::Unavailable(_))
            ));
        }
    }
}
#[test]
fn context_borrow_rejects_unissued_overwritten_or_restarted_original_storage() {
    for mutation in [
        Mutation::UnissuedContext,
        Mutation::StorageLiveContext,
        Mutation::DeadLiveContext,
        Mutation::ContextCallOverwrite,
    ] {
        let owner = fixture(mutation)
            .unwrap_or_else(|e| panic!("{mutation:?} must admit before context relation: {e}"));
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        // UnissuedContext inserts one assignment before the original borrow.
        let statement = if matches!(mutation, Mutation::UnissuedContext) {
            3
        } else {
            2
        };
        let entered = std::cell::Cell::new(false);
        assert!(matches!(
            with_checked_context_borrow_v1(
                &owner,
                SemanticFunctionIdV1::from_index(0),
                SemanticBlockIdV1::from_index(2),
                statement,
                &mut budget,
                |_, _| {
                    entered.set(true);
                    Ok(())
                }
            ),
            Err(ReferenceErrorV1::Unavailable(_))
        ));
        assert!(!entered.get());
    }
}
#[test]
fn context_borrow_resource_denial_precedes_callback_and_stays_charged() {
    let owner = fixture(Mutation::None).expect("context resource fixture source admission");
    let retained = owner.occurrence_storage().unwrap().retained_storage();
    for work_cap in [0, 10_000_000] {
        let mut work = Work::new(work_cap);
        let mut budget = Budget::new(
            &mut work,
            retained + SCRATCH_V1 - usize::from(work_cap != 0),
        );
        budget.reserve_storage(retained).unwrap();
        let entered = std::cell::Cell::new(false);
        assert!(matches!(
            with_checked_context_borrow_v1(
                &owner,
                SemanticFunctionIdV1::from_index(0),
                SemanticBlockIdV1::from_index(2),
                2,
                &mut budget,
                |_, _| {
                    entered.set(true);
                    Ok(())
                }
            ),
            Err(ReferenceErrorV1::Resource(_))
        ));
        assert!(!entered.get());
        assert_eq!(budget.storage(), retained);
        assert!(budget.failed_work().is_some() || budget.failed_storage().is_some());
    }
}
// Actual production holder/use guard under the same freshly borrowed call
// relation. Ordinary original SSA ancestry deliberately still passes.
#[test]
fn original_reference_holders_and_formals_pass_actual_closed_source_guard() {
    let owner = fixture(Mutation::None).expect("original source fixture admission");
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
            for (function, local, borrow) in [
                (0, 4, Some((SemanticBlockIdV1::from_index(2), 1))),
                (0, 5, Some((SemanticBlockIdV1::from_index(2), 2))),
                (1, 1, None),
                (1, 2, None),
            ] {
                super::super::require_closed_wave64_reference_holder_v1(
                    &owner,
                    SemanticFunctionIdV1::from_index(function),
                    local,
                    borrow,
                    budget,
                )
                .expect("original admitted reference holder/formal must stay closed");
            }
            Ok(())
        },
    )
    .unwrap();
}
#[test]
fn source_holder_reassignment_and_raw_aliases_fail_actual_guard_after_admission() {
    for (mutation, function, local, borrow) in [
        (
            Mutation::ReassignSnapshotReference,
            0,
            4,
            Some((SemanticBlockIdV1::from_index(2), 1)),
        ),
        (
            Mutation::ReassignContextReference,
            0,
            5,
            Some((SemanticBlockIdV1::from_index(2), 2)),
        ),
        (Mutation::OverwriteSnapshotFormal, 1, 1, None),
        (Mutation::OverwriteContextFormal, 1, 2, None),
        (
            Mutation::RawSnapshotReference,
            0,
            4,
            Some((SemanticBlockIdV1::from_index(2), 1)),
        ),
        (
            Mutation::RawContextReference,
            0,
            5,
            Some((SemanticBlockIdV1::from_index(2), 2)),
        ),
        (Mutation::RawSnapshotFormal, 1, 1, None),
        (Mutation::RawContextFormal, 1, 2, None),
    ] {
        let owner = fixture(mutation).unwrap_or_else(|e| panic!("{mutation:?} must admit: {e}"));
        check(&owner).expect("same SSA origins alone do not close holder/alias uses");
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
                assert!(
                    super::super::require_closed_wave64_reference_holder_v1(
                        &owner,
                        SemanticFunctionIdV1::from_index(function),
                        local,
                        borrow,
                        budget
                    )
                    .is_err(),
                    "{mutation:?} escaped actual closed holder/use guard"
                );
                Ok(())
            },
        )
        .unwrap();
    }
}
