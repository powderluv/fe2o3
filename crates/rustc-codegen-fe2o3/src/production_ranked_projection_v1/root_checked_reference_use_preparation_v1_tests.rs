// Component controls only. Raw fixtures do not construct the private actual-input
// loan, a completed block stream, a recipe, source custody or ready authority.
mod checked_reference_use_controls {
    use super::super::bf16_nominal_preparation_resources_v1::PreparationResourcesV1;
    use super::super::root_checked_reference_use_preparation_v1::*;
    use super::super::root_checked_references_v1::origin_for_actual_borrowed_v1;
    use super::*;
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    use std::panic::{AssertUnwindSafe, catch_unwind};
    const LIMIT: usize = 16 * 1024 * 1024;
    const FLOOR: usize = 37;

    fn site(statement: Option<usize>) -> ProjectedSemanticAccessSiteV1 {
        ProjectedSemanticAccessSiteV1 {
            block: 0,
            statement,
        }
    }
    fn function(
        statements: Vec<SemanticStatementV1>,
        terminator: SemanticTerminatorKindV1,
    ) -> SemanticFunctionDeclV1 {
        projection_function_with_locals(
            vec![block(51, statements, terminator)],
            (0u8..8)
                .map(|id| {
                    local(
                        id + 50,
                        SCALAR_TYPE,
                        if id == 0 {
                            SemanticLocalRoleV1::Return
                        } else {
                            SemanticLocalRoleV1::Temporary
                        },
                    )
                })
                .collect(),
        )
    }
    fn deref(local: u32) -> SemanticPlaceV1 {
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(local),
            vec![
                SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, SCALAR_TYPE)
                    .unwrap(),
            ],
            SCALAR_TYPE,
        )
        .unwrap()
    }
    fn guarded() -> GuardedRankedAccessV1 {
        GuardedRankedAccessV1 {
            view: ProductionRankedValueIdV1::new(0),
            indices: vec![ProductionRankedValueV1::Argument(0)],
            comparisons: vec![(
                ProductionRankedValueV1::Argument(0),
                ProductionRankedValueV1::Argument(1),
            )],
            checked_success: None,
            access: AccessKindAttr::Read,
            memory_space: MemorySpaceAttr::Global,
            source: SemanticSourceProvenanceV1::unavailable(),
            semantic_site: None,
            output_extent: None,
        }
    }
    fn operation(id: u32) -> ProductionRankedOperationV1 {
        ProductionRankedOperationV1::IndexConstant {
            result: ProductionRankedValueIdV1::new(id),
            value: 7,
        }
    }
    fn access_source() -> ProjectedAccessSourceV1 {
        ProjectedAccessSourceV1 {
            block: 0,
            operation: 0,
            access: AccessKindAttr::Read,
            memory_space: MemorySpaceAttr::Global,
            source: SemanticSourceProvenanceV1::unavailable(),
            semantic_site: None,
            output_extent: None,
        }
    }
    fn owned() -> (SemanticFunctionDeclV1, CheckedReferencesV1) {
        let (function, producers) = option_dominance_chain(1);
        let option_dominance = SemanticOptionDominanceV1::analyze(&function, &producers).unwrap();
        let enum_payload_dominance =
            SemanticEnumPayloadDominanceV1::analyze(&function, &projection_types()).unwrap();
        let mut origins = vec![None; function.locals().len()];
        origins[1] = Some(CheckedReferenceOriginV1 {
            source: CheckedReferenceSourceV1::GuardedAccess(0),
            availability: Some(CapabilityAvailabilityV1::Option(
                option_dominance
                    .availability(SemanticLocalIdV1::from_index(1))
                    .unwrap(),
            )),
        });
        (
            function,
            CheckedReferencesV1 {
                origins,
                option_dominance,
                enum_payload_dominance,
            },
        )
    }
    // Frozen original decision, not the new borrowed adapter/selector.
    fn old_decision(
        place: &SemanticPlaceV1,
        block_index: usize,
        references: &CheckedReferencesV1,
    ) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
        let Some(origin) = checked_reference_origin_for_place(place, &references.origins) else {
            return Ok(None);
        };
        if !origin.availability.is_none_or(|availability| {
            capability_availability_allows(
                &references.option_dominance,
                &references.enum_payload_dominance,
                availability,
                SemanticBlockIdV1::from_index(block_index as u32),
            )
        }) {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a checked reference is dereferenced outside its authenticated payload region",
            ));
        }
        Ok(Some(origin.source))
    }
    fn paid_decision(
        function: &SemanticFunctionDeclV1,
        table: &SemanticFunctionDeclV1,
        place: &SemanticPlaceV1,
        block: usize,
        references: &CheckedReferencesV1,
    ) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let result = origin_for_actual_borrowed_v1(
            function,
            table,
            place,
            block,
            &references.origins,
            &references.option_dominance,
            &references.enum_payload_dominance,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        );
        assert_eq!(owned, 0);
        assert_eq!(budget.storage(), 0);
        result
    }

    #[test]
    fn actual_use_selector_borrows_exact_assignment_operands_in_ordinary_order() {
        let f = function(
            vec![statement(SemanticStatementKindV1::Assign(
                SemanticAssignmentV1::new(
                    deref(2),
                    SemanticRvalueV1::new(
                        SCALAR_TYPE,
                        SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(deref(1))),
                    ),
                ),
            ))],
            SemanticTerminatorKindV1::Return,
        );
        let SemanticStatementKindV1::Assign(a) = f.blocks()[0].statements()[0].kind() else {
            panic!("fixture");
        };
        let SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(rhs)) = a.value().kind() else {
            panic!("fixture");
        };
        for (ordinal, expected, kind) in [
            (0, rhs, AccessKindAttr::Read),
            (1, a.destination(), AccessKindAttr::Write),
        ] {
            let use_site = select_source_use_v1(
                &f,
                site(Some(0)),
                ordinal,
                &mut PreparationResourcesV1::unmetered(),
            )
            .unwrap();
            assert!(std::ptr::eq(use_site.place, expected));
            assert_eq!(use_site.site, site(Some(0)));
            assert_eq!(use_site.ordinal, ordinal);
            assert_eq!(use_site.access, kind);
            assert_eq!(use_site.source, f.blocks()[0].statements()[0].source());
            assert!(use_site.atomic.is_none());
        }
        assert!(
            select_source_use_v1(
                &f,
                site(Some(0)),
                2,
                &mut PreparationResourcesV1::unmetered()
            )
            .is_err()
        );
    }

    #[test]
    fn actual_use_selector_compare_exchange_keeps_destination_first_then_address_operands() {
        let f = function(
            vec![statement(SemanticStatementKindV1::AtomicCompareExchange(
                SemanticAtomicCompareExchangeV1::new(
                    typed_place(4, SCALAR_TYPE),
                    deref(1),
                    typed_operand(2, SCALAR_TYPE),
                    typed_operand(3, SCALAR_TYPE),
                    atomic_access(),
                    SemanticAtomicOrderingV1::Relaxed,
                    false,
                ),
            ))],
            SemanticTerminatorKindV1::Return,
        );
        let expected = [
            (4, AccessKindAttr::Write, false),
            (1, AccessKindAttr::AtomicReadModifyWrite, true),
            (2, AccessKindAttr::Read, false),
            (3, AccessKindAttr::Read, false),
        ];
        for (ordinal, (local, access, atomic)) in expected.into_iter().enumerate() {
            let item = select_source_use_v1(
                &f,
                site(Some(0)),
                ordinal,
                &mut PreparationResourcesV1::unmetered(),
            )
            .unwrap();
            assert!(item.place.local() == SemanticLocalIdV1::from_index(local));
            assert_eq!(item.access, access);
            assert_eq!(item.atomic.is_some(), atomic);
        }
        assert!(
            select_source_use_v1(
                &f,
                site(Some(0)),
                4,
                &mut PreparationResourcesV1::unmetered()
            )
            .is_err()
        );
    }

    #[test]
    fn actual_use_selector_store_and_rmw_keep_value_before_address() {
        let f = function(
            vec![
                statement(SemanticStatementKindV1::Store(SemanticMemoryStoreV1::new(
                    deref(1),
                    typed_operand(2, SCALAR_TYPE),
                    SemanticVolatilityV1::NonVolatile,
                    Some(atomic_access()),
                ))),
                statement(SemanticStatementKindV1::AtomicRmw(
                    SemanticAtomicRmwV1::new(
                        typed_place(4, SCALAR_TYPE),
                        deref(1),
                        typed_operand(2, SCALAR_TYPE),
                        SemanticAtomicRmwOpV1::Exchange,
                        atomic_access(),
                    ),
                )),
            ],
            SemanticTerminatorKindV1::Return,
        );
        for (statement, expected) in [
            (
                0,
                vec![(2, AccessKindAttr::Read), (1, AccessKindAttr::AtomicWrite)],
            ),
            (
                1,
                vec![
                    (2, AccessKindAttr::Read),
                    (1, AccessKindAttr::AtomicReadModifyWrite),
                    (4, AccessKindAttr::Write),
                ],
            ),
        ] {
            for (ordinal, (local, access)) in expected.into_iter().enumerate() {
                let item = select_source_use_v1(
                    &f,
                    site(Some(statement)),
                    ordinal,
                    &mut PreparationResourcesV1::unmetered(),
                )
                .unwrap();
                assert!(item.place.local() == SemanticLocalIdV1::from_index(local));
                assert_eq!(item.access, access);
            }
        }
    }

    #[test]
    fn actual_use_selector_absence_does_not_manufacture_address_or_bounds_proof() {
        let f = function(
            vec![
                typed_assignment(
                    2,
                    POINTER_TYPE,
                    SemanticRvalueKindV1::Borrow {
                        kind: SemanticBorrowKindV1::Shared,
                        place: deref(1),
                    },
                ),
                statement(SemanticStatementKindV1::Nop),
                statement(SemanticStatementKindV1::StorageLive(
                    SemanticLocalIdV1::from_index(999),
                )),
            ],
            SemanticTerminatorKindV1::Return,
        );
        let item = select_source_use_v1(
            &f,
            site(Some(0)),
            0,
            &mut PreparationResourcesV1::unmetered(),
        )
        .unwrap();
        assert!(item.place.local() == SemanticLocalIdV1::from_index(2));
        assert_eq!(item.access, AccessKindAttr::Write);
        for (where_, ordinal) in [
            (site(Some(0)), 1),
            (site(Some(1)), 0),
            (site(Some(2)), 0),
            (site(Some(99)), 0),
            (site(None), 0),
            (
                ProjectedSemanticAccessSiteV1 {
                    block: 99,
                    statement: None,
                },
                0,
            ),
        ] {
            assert!(
                select_source_use_v1(
                    &f,
                    where_,
                    ordinal,
                    &mut PreparationResourcesV1::unmetered()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn actual_use_selector_calls_and_drop_never_bypass_pending_effect_router() {
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![typed_operand(1, SCALAR_TYPE)],
            None,
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let tail = SemanticDirectTailCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![typed_operand(1, SCALAR_TYPE)],
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        for terminator in [
            SemanticTerminatorKindV1::Call(call),
            SemanticTerminatorKindV1::TailCall(tail),
            SemanticTerminatorKindV1::Drop {
                place: deref(1),
                drop_glue: SemanticFunctionIdV1::from_index(0),
                target: cfg_edge(SemanticEdgeRoleV1::DropReturn, 0),
                unwind: SemanticUnwindActionV1::Unreachable,
            },
        ] {
            let f = function(vec![], terminator);
            assert!(matches!(
                select_source_use_v1(&f, site(None), 0, &mut PreparationResourcesV1::unmetered()),
                Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "actual source use at a call or drop requires full access/effect routing"
                ))
            ));
        }
    }

    #[test]
    fn actual_use_selector_charges_all_candidates_and_exact_work_boundary() {
        let f = function(
            vec![statement(SemanticStatementKindV1::Assume(typed_operand(
                1,
                SCALAR_TYPE,
            )))],
            SemanticTerminatorKindV1::Return,
        );
        // 64 source lookup + 8 operand visit + 32 actual place.
        for cap in [103, 104] {
            let mut work = Work::new(cap);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let result = select_source_use_v1(
                &f,
                site(Some(0)),
                0,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert_eq!(result.is_ok(), cap == 104);
            assert_eq!(owned, 0);
            assert_eq!(budget.storage(), 0);
            if cap == 103 {
                assert!(budget.failed_work().is_some());
                assert!(
                    select_source_use_v1(
                        &f,
                        site(Some(0)),
                        0,
                        &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn actual_borrowed_decision_matches_frozen_old_availability_at_every_block() {
        let (f, references) = owned();
        let p = deref(1);
        let pointer = references.origins.as_ptr();
        let capacity = references.origins.capacity();
        for block in 0..f.blocks().len() {
            assert_eq!(
                format!("{:?}", paid_decision(&f, &f, &p, block, &references)),
                format!("{:?}", old_decision(&p, block, &references))
            );
        }
        assert_eq!(
            paid_decision(&f, &f, &p, 2, &references).unwrap(),
            Some(CheckedReferenceSourceV1::GuardedAccess(0))
        );
        assert!(paid_decision(&f, &f, &p, 3, &references).is_err());
        assert_eq!(references.origins.as_ptr(), pointer);
        assert_eq!(references.origins.capacity(), capacity);
    }

    #[test]
    fn actual_borrowed_decision_rejects_foreign_source_table_block_and_legacy_ledger() {
        let (f, mut references) = owned();
        let foreign = f.clone();
        let p = deref(1);
        assert!(paid_decision(&f, &foreign, &p, 2, &references).is_err());
        assert!(paid_decision(&f, &f, &p, usize::MAX, &references).is_err());
        assert!(
            origin_for_actual_borrowed_v1(
                &f,
                &f,
                &p,
                2,
                &references.origins,
                &references.option_dominance,
                &references.enum_payload_dominance,
                &mut PreparationResourcesV1::unmetered()
            )
            .is_err()
        );
        references.origins.pop();
        assert!(paid_decision(&f, &f, &p, 2, &references).is_err());
    }

    #[test]
    fn actual_borrowed_decision_keeps_transparent_and_absent_origin_rules() {
        let (f, mut references) = owned();
        for tail in [
            SemanticProjectionKindV1::Field(0),
            SemanticProjectionKindV1::Downcast(0),
            SemanticProjectionKindV1::OpaqueCast,
            SemanticProjectionKindV1::Subtype,
        ] {
            let p = SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(1),
                vec![
                    SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, SCALAR_TYPE)
                        .unwrap(),
                    SemanticProjectionV1::new(tail, SCALAR_TYPE).unwrap(),
                ],
                SCALAR_TYPE,
            )
            .unwrap();
            assert_eq!(
                paid_decision(&f, &f, &p, 2, &references).unwrap(),
                Some(CheckedReferenceSourceV1::GuardedAccess(0))
            );
        }
        for p in [typed_place(1, SCALAR_TYPE), deref(0), deref(999)] {
            assert_eq!(paid_decision(&f, &f, &p, 3, &references).unwrap(), None);
        }
        references.origins[1] = Some(CheckedReferenceOriginV1 {
            source: CheckedReferenceSourceV1::ProjectedSharedBorrow,
            availability: None,
        });
        assert_eq!(
            paid_decision(&f, &f, &deref(1), 3, &references).unwrap(),
            Some(CheckedReferenceSourceV1::ProjectedSharedBorrow)
        );
    }

    #[test]
    fn ordinary_checked_site_adapter_matches_frozen_branch_outcomes_and_mutations() {
        let guards = [guarded()];
        for origin in [
            CheckedReferenceSourceV1::GuardedAccess(0),
            CheckedReferenceSourceV1::GuardedAccess(1),
            CheckedReferenceSourceV1::ProjectedSharedBorrow,
        ] {
            for access in [
                AccessKindAttr::Read,
                AccessKindAttr::Write,
                AccessKindAttr::AtomicRead,
            ] {
                for atomic in [None, Some(atomic_access())] {
                    let mut old = Vec::new();
                    let mut new = Vec::new();
                    let operations = [operation(0)];
                    let a = frozen_append(
                        origin,
                        access,
                        atomic,
                        guards[0].source,
                        &guards,
                        &mut old,
                        &operations,
                    );
                    let b = append_checked_reference_site_v1(
                        origin,
                        access,
                        atomic,
                        guards[0].source,
                        &guards,
                        &mut new,
                        &operations,
                        &mut PreparationResourcesV1::unmetered(),
                    );
                    assert_eq!(format!("{a:?}"), format!("{b:?}"));
                    assert_eq!(old, new);
                }
            }
        }
    }

    #[test]
    fn checked_sites_use_actual_operation_length_and_bind_repeated_real_site() {
        let guards = [guarded()];
        let mut sites = Vec::new();
        let mut operations = Vec::new();
        for next in [None, Some(operation(0)), None, Some(operation(1))] {
            if let Some(next) = next {
                operations.push(next);
            }
            append_checked_reference_site_v1(
                CheckedReferenceSourceV1::GuardedAccess(0),
                AccessKindAttr::Read,
                None,
                guards[0].source,
                &guards,
                &mut sites,
                &operations,
                &mut PreparationResourcesV1::unmetered(),
            )
            .unwrap();
        }
        assert_eq!(
            sites
                .iter()
                .map(|s| s.insertion_operation)
                .collect::<Vec<_>>(),
            vec![0, 1, 1, 2]
        );
        bind_projected_access_site_with_resources_v1(
            &mut [],
            &mut sites,
            site(Some(0)),
            &mut PreparationResourcesV1::unmetered(),
        )
        .unwrap();
        assert!(
            sites
                .iter()
                .all(|s| s.access.semantic_site == Some(site(Some(0))))
        );
        assert!(guards[0].semantic_site.is_none());
    }

    #[test]
    fn paid_checked_site_retains_complete_nested_storage_until_caller_drop() {
        let guards = [guarded()];
        let mut sites = Vec::new();
        let mut expected = Vec::new();
        let ops = [operation(0)];
        frozen_append(
            CheckedReferenceSourceV1::GuardedAccess(0),
            AccessKindAttr::Write,
            None,
            guards[0].source,
            &guards,
            &mut expected,
            &ops,
        )
        .unwrap();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        append_checked_reference_site_v1(
            CheckedReferenceSourceV1::GuardedAccess(0),
            AccessKindAttr::Write,
            None,
            guards[0].source,
            &guards,
            &mut sites,
            &ops,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        assert_eq!(sites, expected);
        assert_eq!(
            owned,
            size_of::<GuardedAccessSiteV1>()
                + size_of::<ProductionRankedValueV1>()
                + size_of::<(ProductionRankedValueV1, ProductionRankedValueV1)>()
        );
        assert_eq!(budget.storage(), FLOOR + owned);
        drop(sites);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), FLOOR);
    }

    #[test]
    fn paid_checked_site_storage_denials_retain_each_previously_admitted_owner() {
        let guards = [guarded()];
        let slot = size_of::<GuardedAccessSiteV1>();
        let index = size_of::<ProductionRankedValueV1>();
        let pair = size_of::<(ProductionRankedValueV1, ProductionRankedValueV1)>();
        for (cap, slots, indices, comparisons, credits) in [
            (slot - 1, 0, 0, 0, 0),
            (slot + index - 1, 1, 0, 0, slot),
            (slot + index + pair - 1, 1, 1, 0, slot + index),
            (slot + index + pair, 1, 1, 1, slot + index + pair),
        ] {
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, FLOOR + cap);
            budget.reserve_storage(FLOOR).unwrap();
            let mut owned = 0;
            let mut sites = Vec::new();
            let result = append_checked_reference_site_v1(
                CheckedReferenceSourceV1::GuardedAccess(0),
                AccessKindAttr::Read,
                None,
                guards[0].source,
                &guards,
                &mut sites,
                &[],
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert_eq!(result.is_ok(), comparisons == 1);
            assert_eq!(sites.len(), slots);
            if slots == 1 {
                assert_eq!(sites[0].access.indices.len(), indices);
                assert_eq!(sites[0].access.comparisons.len(), comparisons);
            }
            assert_eq!(owned, credits);
            assert_eq!(budget.storage(), FLOOR + owned);
            drop(result);
            drop(sites);
            budget.release_storage(owned).unwrap();
            assert_eq!(budget.storage(), FLOOR);
        }
    }

    #[test]
    fn paid_checked_site_work_denials_keep_partial_slot_and_sticky_ledger() {
        let guards = [guarded()];
        for (cap, slots, indices) in [(31, 0, 0), (33, 1, 0), (34, 1, 1)] {
            let mut work = Work::new(cap);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let mut sites = Vec::new();
            assert!(
                append_checked_reference_site_v1(
                    CheckedReferenceSourceV1::GuardedAccess(0),
                    AccessKindAttr::Read,
                    None,
                    guards[0].source,
                    &guards,
                    &mut sites,
                    &[],
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                )
                .is_err()
            );
            assert_eq!(sites.len(), slots);
            if slots == 1 {
                assert_eq!(sites[0].access.indices.len(), indices);
            }
            let before = (sites.len(), owned, budget.work());
            assert!(
                append_checked_reference_site_v1(
                    CheckedReferenceSourceV1::GuardedAccess(0),
                    AccessKindAttr::Read,
                    None,
                    guards[0].source,
                    &guards,
                    &mut sites,
                    &[],
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                )
                .is_err()
            );
            assert_eq!(before, (sites.len(), owned, budget.work()));
            drop(sites);
            budget.release_storage(owned).unwrap();
            assert_eq!(budget.storage(), 0);
        }
    }

    #[test]
    fn paid_checked_site_arithmetic_and_atomic_mismatch_precede_new_storage() {
        let guards = [guarded()];
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = usize::MAX;
        let mut sites = Vec::new();
        assert!(
            append_checked_reference_site_v1(
                CheckedReferenceSourceV1::GuardedAccess(0),
                AccessKindAttr::Read,
                None,
                guards[0].source,
                &guards,
                &mut sites,
                &[],
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert!(sites.is_empty());
        assert_eq!(owned, usize::MAX);
        assert_eq!(budget.storage(), 0);
        // Deliberately corrupt inert counter above owns no storage and is not refunded.
        owned = 0;
        for (access, atomic) in [
            (AccessKindAttr::AtomicRead, None),
            (AccessKindAttr::Read, Some(atomic_access())),
        ] {
            assert!(
                append_checked_reference_site_v1(
                    CheckedReferenceSourceV1::GuardedAccess(0),
                    access,
                    atomic,
                    guards[0].source,
                    &guards,
                    &mut sites,
                    &[],
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                )
                .is_err()
            );
            assert!(sites.is_empty());
            assert_eq!(owned, 0);
        }
    }

    #[test]
    fn paid_checked_site_callback_error_and_panic_keep_owned_storage_through_inspection() {
        let guards = [guarded()];
        for panic in [false, true] {
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            let mut owned = 0;
            let mut sites = Vec::new();
            let result = catch_unwind(AssertUnwindSafe(
                || -> Result<(), ProductionRankedProjectionErrorV1> {
                    append_checked_reference_site_v1(
                        CheckedReferenceSourceV1::GuardedAccess(0),
                        AccessKindAttr::Read,
                        None,
                        guards[0].source,
                        &guards,
                        &mut sites,
                        &[],
                        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
                    )?;
                    if panic {
                        panic!("site consumer");
                    }
                    Err(ProductionRankedProjectionErrorV1::Incomplete(
                        "site consumer",
                    ))
                },
            ));
            assert_eq!(result.is_err(), panic);
            assert_eq!(sites.len(), 1);
            assert!(owned > 0);
            assert_eq!(budget.storage(), FLOOR + owned);
            assert!(sites[0].access.semantic_site.is_none());
            drop(result);
            drop(sites);
            budget.release_storage(owned).unwrap();
            assert_eq!(budget.storage(), FLOOR);
        }
    }

    #[test]
    fn ordinary_binding_adapter_preserves_duplicate_partial_mutation_and_order() {
        for duplicate in 0..3 {
            let mut a = vec![access_source(), access_source()];
            let mut ga = vec![GuardedAccessSiteV1 {
                insertion_operation: 0,
                access: guarded(),
            }];
            if duplicate == 1 {
                a[1].semantic_site = Some(site(None));
            }
            if duplicate == 2 {
                ga[0].access.semantic_site = Some(site(None));
            }
            let mut b = a.clone();
            let mut gb = ga.clone();
            let old = frozen_bind(&mut a, &mut ga, site(Some(0)));
            let new = bind_projected_access_site_with_resources_v1(
                &mut b,
                &mut gb,
                site(Some(0)),
                &mut PreparationResourcesV1::unmetered(),
            );
            assert_eq!(format!("{old:?}"), format!("{new:?}"));
            assert_eq!(a, b);
            assert_eq!(ga, gb);
            assert_eq!(old.is_ok(), duplicate == 0);
        }
    }

    #[test]
    fn paid_binding_work_boundary_retains_prefix_mutations_and_never_refunds() {
        for cap in [15, 16, 31, 32] {
            let mut work = Work::new(cap);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let mut sources = [access_source()];
            let mut guards = [GuardedAccessSiteV1 {
                insertion_operation: 0,
                access: guarded(),
            }];
            let result = bind_projected_access_site_with_resources_v1(
                &mut sources,
                &mut guards,
                site(Some(0)),
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert_eq!(result.is_ok(), cap == 32);
            assert_eq!(sources[0].semantic_site.is_some(), cap >= 16);
            assert_eq!(guards[0].access.semantic_site.is_some(), cap == 32);
            assert_eq!(owned, 0);
            assert_eq!(budget.storage(), 0);
        }
    }

    // Exact old branch/body follow. They do not invoke the extracted producer.
    #[allow(clippy::too_many_arguments, clippy::needless_return)]
    fn frozen_append(
        origin: CheckedReferenceSourceV1,
        access: AccessKindAttr,
        atomic: Option<SemanticAtomicAccessV1>,
        source: SemanticSourceProvenanceV1,
        guarded_accesses: &[GuardedRankedAccessV1],
        guarded_sites: &mut Vec<GuardedAccessSiteV1>,
        operations: &[ProductionRankedOperationV1],
    ) -> Result<(), ProductionRankedProjectionErrorV1> {
        match origin {
            CheckedReferenceSourceV1::GuardedAccess(origin) => {
                if atomic.is_some() {
                    return Err(ProductionRankedProjectionErrorV1::Incomplete(
                        "an atomic access through a checked disjoint reference before exact atomic capability projection",
                    ));
                }
                let mut guarded = guarded_accesses.get(origin).cloned().ok_or(
                    ProductionRankedProjectionErrorV1::Unsupported(
                        "a checked disjoint reference whose access origin is out of range",
                    ),
                )?;
                guarded.access = access;
                guarded.source = source;
                guarded_sites.try_reserve(1).map_err(|_| {
                    ProductionRankedProjectionErrorV1::Unsupported(
                        "checked disjoint access-site storage cannot be reserved",
                    )
                })?;
                guarded_sites.push(GuardedAccessSiteV1 {
                    insertion_operation: operations.len(),
                    access: guarded,
                });
                return Ok(());
            }
            CheckedReferenceSourceV1::ProjectedSharedBorrow
                if atomic.is_none() && access == AccessKindAttr::Read =>
            {
                return Ok(());
            }
            CheckedReferenceSourceV1::ProjectedSharedBorrow => {
                return Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "a projected shared reference used for a non-read memory effect",
                ));
            }
        }
    }

    fn frozen_bind(
        sources: &mut [ProjectedAccessSourceV1],
        guarded_sites: &mut [GuardedAccessSiteV1],
        site: ProjectedSemanticAccessSiteV1,
    ) -> Result<(), ProductionRankedProjectionErrorV1> {
        for source in sources {
            if source.semantic_site.replace(site).is_some() {
                return Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "a projected access was attributed to multiple semantic sites",
                ));
            }
        }
        for guarded in guarded_sites {
            if guarded.access.semantic_site.replace(site).is_some() {
                return Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "a guarded access was attributed to multiple semantic sites",
                ));
            }
        }
        Ok(())
    }
}
