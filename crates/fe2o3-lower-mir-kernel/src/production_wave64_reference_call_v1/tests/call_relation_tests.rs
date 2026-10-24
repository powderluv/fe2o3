#[test]
fn original_initializer_and_actual_ssa_issuers_join_to_borrowed_call() {
    for mutation in [Mutation::None, Mutation::Rank, Mutation::InitialStorageLive] {
        let owner = fixture(mutation).expect("synthetic source admission");
        let borrowed = Site::Statement {
            block: fe2o3_mir_model::SsaBlockIdV1::new(2),
            statement: 1,
        };
        let occurrences = rows(&owner, SemanticFunctionIdV1::from_index(0)).unwrap();
        assert!(!occurrences.elisions().contains(&borrowed));
        let mut found = 0;
        for event in occurrences.events() {
            if event.site() == borrowed
                && event.operand() == OperandRole::RvaluePlace
                && event.role() == EventRole::BaseUse
            {
                assert!(
                    matches!(event.event(), fe2o3_mir_model::SsaEventV1::Use(variable) if variable.get() == 3)
                );
                assert!(event.is_reachable());
                assert!(!event.is_promoted());
                assert!(event.resolved().is_none());
                found += 1;
            }
        }
        assert_eq!(
            found, 1,
            "normal retained Borrow has exact syntax, not an SSA use or elision"
        );
        check(&owner).unwrap();
    }
}
#[test]
fn source_origin_mutations_cannot_mint_reference_transport() {
    for mutation in [
        Mutation::UnissuedLane,
        Mutation::UnissuedContext,
        Mutation::WrongLaneField,
        Mutation::ReassignSnapshot,
    ] {
        let owner = fixture(mutation)
            .unwrap_or_else(|error| panic!("{mutation:?} must reach relation: {error}"));
        assert!(
            matches!(check(&owner), Err(ReferenceErrorV1::Unavailable(_))),
            "{mutation:?} must fail source relation"
        );
    }
}
#[test]
fn raw_reference_type_mutation_is_rejected_by_original_source_schema() {
    let error = match fixture(Mutation::RawBorrow) {
        Ok(_) => panic!("raw address falsely typed as reference admitted"),
        Err(error) => error,
    };
    assert!(
        error.starts_with("InvalidTypeOperation { operation: Borrow,"),
        "{error}"
    );
}
#[test]
fn one_short_scratch_and_zero_work_deny_before_callback_and_keep_sticky_failure() {
    let owner = fixture(Mutation::None).unwrap();
    let retained = owner.occurrence_storage().unwrap().retained_storage();
    for work_cap in [0, 10_000_000] {
        let mut work = Work::new(work_cap);
        let mut budget = Budget::new(
            &mut work,
            retained + SCRATCH_V1 - usize::from(work_cap != 0),
        );
        budget.reserve_storage(retained).unwrap();
        let entered = std::cell::Cell::new(false);
        let result = with_checked_reference_call_v1(
            &owner,
            SemanticFunctionIdV1::from_index(0),
            SemanticBlockIdV1::from_index(2),
            &mut budget,
            |_, _| {
                entered.set(true);
                Ok(())
            },
        );
        assert!(matches!(result, Err(ReferenceErrorV1::Resource(_))));
        assert!(!entered.get());
        assert_eq!(budget.storage(), retained);
        assert!(budget.failed_work().is_some() || budget.failed_storage().is_some());
    }
}
#[test]
fn callback_denial_is_sticky_and_callback_allocations_are_not_refunded() {
    let owner = fixture(Mutation::None).unwrap();
    let retained = owner.occurrence_storage().unwrap().retained_storage();
    for deny in [false, true] {
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget.reserve_storage(retained).unwrap();
        let identity = budget.work_ledger_identity_v1();
        let result = with_checked_reference_call_v1(
            &owner,
            SemanticFunctionIdV1::from_index(0),
            SemanticBlockIdV1::from_index(2),
            &mut budget,
            |view, budget| {
                assert!(std::ptr::eq(view.owner(), &owner));
                budget.reserve_storage(17)?;
                if deny {
                    let _ = budget.charge_work(10_000_001);
                }
                Ok(())
            },
        );
        if deny {
            assert!(matches!(
                result,
                Err(ReferenceErrorV1::Resource(Resource::Accounting))
            ));
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(budget.storage(), retained + 17);
        assert!(budget.work_ledger_identity_v1() == identity);
    }
}
#[test]
fn helper_census_keeps_all_incoming_calls_under_same_borrowed_authority() {
    for mutation in [
        Mutation::None,
        Mutation::Rank,
        Mutation::SameIncoming,
        Mutation::ForeignIncoming,
        Mutation::ForeignIncomingLane,
        Mutation::DistinctIssuedLane,
    ] {
        let owner = fixture(mutation).expect("synthetic all-incoming source admission");
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        budget
            .reserve_storage(owner.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        check(&owner).expect("first source call is genuine");
        let entered = std::cell::Cell::new(false);
        let result = with_checked_reference_helper_v1(
            &owner,
            SemanticFunctionIdV1::from_index(1),
            &mut budget,
            |view, _| {
                entered.set(true);
                assert_eq!(
                    view.incoming_count(),
                    if matches!(mutation, Mutation::SameIncoming) {
                        2
                    } else {
                        1
                    }
                );
                assert!(std::ptr::eq(view.representative().owner(), &owner));
                assert_eq!(
                    view.representative().helper(),
                    SemanticFunctionIdV1::from_index(1)
                );
                Ok(())
            },
        );
        if matches!(
            mutation,
            Mutation::ForeignIncoming
                | Mutation::ForeignIncomingLane
                | Mutation::DistinctIssuedLane
        ) {
            assert!(matches!(result, Err(ReferenceErrorV1::Unavailable(_))));
            assert!(!entered.get());
        } else {
            result.unwrap();
            assert!(entered.get());
        }
    }
}

#[test]
fn saved_coordinates_do_not_join_distinct_live_source_owners() {
    let left = fixture(Mutation::None).expect("left source admission");
    let right = fixture(Mutation::None).expect("right source admission");
    let saved_helper = SemanticFunctionIdV1::from_index(1);
    let mut left_work = Work::new(10_000_000);
    let mut left_budget = Budget::new(&mut left_work, 10_000_000);
    left_budget
        .reserve_storage(left.occurrence_storage().unwrap().retained_storage())
        .unwrap();
    with_checked_reference_helper_v1(&left, saved_helper, &mut left_budget, |left_view, _| {
        let mut right_work = Work::new(10_000_000);
        let mut right_budget = Budget::new(&mut right_work, 10_000_000);
        right_budget
            .reserve_storage(right.occurrence_storage().unwrap().retained_storage())
            .unwrap();
        with_checked_reference_helper_v1(
            &right,
            saved_helper,
            &mut right_budget,
            |right_view, _| {
                assert!(
                    !left_view
                        .representative()
                        .same_source_origins(right_view.representative())
                );
                assert!(!std::ptr::eq(
                    left_view.representative().source_call(),
                    right_view.representative().source_call()
                ));
                Ok(())
            },
        )
    })
    .unwrap();
}
