// Synthetic component controls: none constructs an actual-input loan or recipe.
mod borrowed_local_contract_controls {
    use super::super::root_local_contracts_v1::*;
    use super::*;
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    const LIMIT: usize = 16 * 1024 * 1024;
    struct Fixture {
        function: SemanticFunctionDeclV1,
        references: CheckedReferencesV1,
        counts: Vec<u8>,
        assignments: Vec<Option<ScalarAssignmentSiteV1>>,
        escaped: Vec<bool>,
        allocations: Vec<Option<AllocationContractV1>>,
        provenance: Vec<Option<LocalAllocationProvenanceV1>>,
    }
    impl Fixture {
        fn new() -> Self {
            let (function, producers) = option_dominance_chain(1);
            let option_dominance =
                SemanticOptionDominanceV1::analyze(&function, &producers).unwrap();
            let enum_payload_dominance =
                SemanticEnumPayloadDominanceV1::analyze(&function, &projection_types()).unwrap();
            let n = function.locals().len();
            let mut origins = vec![None; n];
            origins[1] = Some(CheckedReferenceOriginV1 {
                source: CheckedReferenceSourceV1::GuardedAccess(0),
                availability: Some(CapabilityAvailabilityV1::Option(
                    option_dominance
                        .availability(SemanticLocalIdV1::from_index(1))
                        .unwrap(),
                )),
            });
            let mut allocations = vec![None; n];
            allocations[1] = Some(AllocationContractV1 {
                allocation_origin: 7,
                noalias_class: 3,
                writable: true,
                singleton_object: false,
            });
            let mut provenance = vec![None; n];
            provenance[1] = Some(LocalAllocationProvenanceV1::Argument(0));
            Self {
                function,
                references: CheckedReferencesV1 {
                    origins,
                    option_dominance,
                    enum_payload_dominance,
                },
                counts: vec![1; n],
                assignments: vec![
                    Some(ScalarAssignmentSiteV1 {
                        block: 0,
                        statement: 0
                    });
                    n
                ],
                escaped: vec![false; n],
                allocations,
                provenance,
            }
        }
        fn parts(&self) -> SourceLocalContractPartsV1<'_> {
            SourceLocalContractPartsV1 {
                function: &self.function,
                table_function: &self.function,
                counts: &self.counts,
                assignments: &self.assignments,
                address_escaped: &self.escaped,
                allocations: &self.allocations,
                allocation_provenance: &self.provenance,
                origins: &self.references.origins,
                option_dominance: &self.references.option_dominance,
                enum_payload_dominance: &self.references.enum_payload_dominance,
            }
        }
        fn legacy(&self) -> ProjectionLocalContractsV1 {
            ProjectionLocalContractsV1 {
                immutable_locals: (0..self.function.locals().len())
                    .map(|local| old_immutable(self, local))
                    .collect(),
                checked_references: CheckedReferencesV1 {
                    origins: self.references.origins.clone(),
                    option_dominance: SemanticOptionDominanceV1::analyze(
                        &self.function,
                        &option_dominance_chain(1).1,
                    )
                    .unwrap(),
                    enum_payload_dominance: SemanticEnumPayloadDominanceV1::analyze(
                        &self.function,
                        &projection_types(),
                    )
                    .unwrap(),
                },
                allocations: self.allocations.clone(),
                allocation_provenance: self.provenance.clone(),
            }
        }
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
    // Frozen ordinary predicate and availability body, never the new producer.
    fn old_immutable(f: &Fixture, local: usize) -> bool {
        !f.function.locals()[local].role().is_entry_argument()
            && f.counts[local] == 1
            && f.assignments[local].is_some()
            && !f.escaped[local]
    }
    fn old_origin(
        p: &SemanticPlaceV1,
        block: usize,
        r: &CheckedReferencesV1,
    ) -> Result<Option<CheckedReferenceSourceV1>, ProductionRankedProjectionErrorV1> {
        let Some(origin) = checked_reference_origin_for_place(p, &r.origins) else {
            return Ok(None);
        };
        if !origin.availability.is_none_or(|availability| {
            capability_availability_allows(
                &r.option_dominance,
                &r.enum_payload_dominance,
                availability,
                SemanticBlockIdV1::from_index(block as u32),
            )
        }) {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a checked reference is dereferenced outside its authenticated payload region",
            ));
        }
        Ok(Some(origin.source))
    }
    fn decision_fields(
        d: LocalContractDecisionV1,
    ) -> (
        Option<CheckedReferenceSourceV1>,
        bool,
        Option<AllocationContractV1>,
        Option<LocalAllocationProvenanceV1>,
    ) {
        (d.origin, d.immutable, d.allocation, d.allocation_provenance)
    }

    #[test]
    fn borrowed_contract_immutable_predicate_matches_original_all_terms() {
        let mut f = Fixture::new();
        for count in [0, 1, 2, u8::MAX] {
            for assigned in [false, true] {
                for escaped in [false, true] {
                    f.counts.fill(count);
                    f.assignments
                        .fill(assigned.then_some(ScalarAssignmentSiteV1 {
                            block: 0,
                            statement: 0,
                        }));
                    f.escaped.fill(escaped);
                    for local in 0..f.function.locals().len() {
                        assert_eq!(
                            immutable_candidate_v1(
                                &f.function,
                                local,
                                &f.counts,
                                &f.assignments,
                                &f.escaped
                            ),
                            old_immutable(&f, local)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn borrowed_contract_immutable_short_circuit_preserves_entry_role_and_counts() {
        let function = projection_function_with_locals(
            vec![block(51, vec![], SemanticTerminatorKindV1::Return)],
            vec![
                local(50, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(51, SCALAR_TYPE, SemanticLocalRoleV1::Argument(0)),
            ],
        );
        // Later malformed slices are not touched when the original earlier term refuses.
        assert!(!immutable_candidate_v1(&function, 1, &[], &[], &[]));
        assert!(!immutable_candidate_v1(&function, 0, &[0], &[], &[]));
        assert!(!immutable_candidate_v1(&function, 0, &[1], &[None], &[]));
    }

    #[test]
    fn borrowed_contract_legacy_getters_preserve_every_row_and_missing_values() {
        let f = Fixture::new();
        let old = f.legacy();
        let view = BorrowedLocalContractsV1::legacy(&old);
        for local in 0..=f.function.locals().len() {
            assert_eq!(
                view.immutable(local),
                old.immutable_locals.get(local) == Some(&true)
            );
            assert_eq!(
                view.allocation(local),
                old.allocations.get(local).copied().flatten()
            );
            assert_eq!(
                view.allocation_provenance(local),
                old.allocation_provenance.get(local).copied().flatten()
            );
        }
        assert!(!view.immutable(usize::MAX));
        assert!(view.allocation(usize::MAX).is_none());
        assert!(view.allocation_provenance(usize::MAX).is_none());
    }

    #[test]
    fn borrowed_contract_both_views_preserve_original_availability_every_block() {
        let f = Fixture::new();
        let old = f.legacy();
        let legacy = BorrowedLocalContractsV1::legacy(&old);
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let data = BorrowedLocalContractsV1::source_data(
            f.parts(),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        let p = deref(1);
        for block in 0..f.function.blocks().len() {
            let oracle = format!("{:?}", old_origin(&p, block, &f.references));
            assert_eq!(format!("{:?}", legacy.origin(&p, block)), oracle);
            assert_eq!(format!("{:?}", data.origin(&p, block)), oracle);
        }
        assert_eq!(owned, 0);
        assert_eq!(budget.storage(), 0);
        assert_eq!(budget.work(), 128);
    }

    #[test]
    fn borrowed_contract_source_uses_exact_rows_without_allocating_replacement_tables() {
        let f = Fixture::new();
        let old = f.legacy();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let data = BorrowedLocalContractsV1::source_data(
            f.parts(),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        for local in 0..=f.function.locals().len() {
            assert_eq!(
                data.immutable(local),
                old.immutable_locals.get(local) == Some(&true)
            );
            assert_eq!(
                data.allocation(local),
                old.allocations.get(local).copied().flatten()
            );
            assert_eq!(
                data.allocation_provenance(local),
                old.allocation_provenance.get(local).copied().flatten()
            );
        }
        assert_eq!(owned, 0);
        assert_eq!(budget.storage(), 0);
    }

    #[test]
    fn borrowed_contract_source_refuses_each_short_or_long_table() {
        for field in 0..6 {
            for long in [false, true] {
                let mut f = Fixture::new();
                match field {
                    0 => {
                        if long {
                            f.counts.push(1);
                        } else {
                            f.counts.pop();
                        }
                    }
                    1 => {
                        if long {
                            f.assignments.push(None);
                        } else {
                            f.assignments.pop();
                        }
                    }
                    2 => {
                        if long {
                            f.escaped.push(false);
                        } else {
                            f.escaped.pop();
                        }
                    }
                    3 => {
                        if long {
                            f.allocations.push(None);
                        } else {
                            f.allocations.pop();
                        }
                    }
                    4 => {
                        if long {
                            f.provenance.push(None);
                        } else {
                            f.provenance.pop();
                        }
                    }
                    _ => {
                        if long {
                            f.references.origins.push(None);
                        } else {
                            f.references.origins.pop();
                        }
                    }
                }
                let mut work = Work::new(LIMIT);
                let mut budget = Budget::new(&mut work, LIMIT);
                let mut owned = 0;
                assert!(matches!(
                    BorrowedLocalContractsV1::source_data(
                        f.parts(),
                        &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                    ),
                    Err(ProductionRankedProjectionErrorV1::Incomplete(
                        "borrowed local-contract table cardinality differs"
                    ))
                ));
                assert_eq!(budget.work(), 128);
                assert_eq!(owned, 0);
            }
        }
    }

    #[test]
    fn borrowed_contract_source_refuses_equal_but_foreign_function() {
        let f = Fixture::new();
        let foreign = f.function.clone();
        let mut parts = f.parts();
        parts.table_function = &foreign;
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        assert!(
            BorrowedLocalContractsV1::source_data(
                parts,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert_eq!(budget.work(), 128);
        assert_eq!(owned, 0);
    }

    #[test]
    fn borrowed_contract_source_refuses_legacy_before_any_data_authority() {
        let f = Fixture::new();
        assert!(
            BorrowedLocalContractsV1::source_data(
                f.parts(),
                &mut PreparationResourcesV1::unmetered()
            )
            .is_err()
        );
    }

    #[test]
    fn borrowed_contract_source_exact_and_one_short_work_bound() {
        for cap in [127, 128] {
            let f = Fixture::new();
            let mut work = Work::new(cap);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let result = BorrowedLocalContractsV1::source_data(
                f.parts(),
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert_eq!(result.is_ok(), cap == 128);
            assert_eq!(owned, 0);
            assert_eq!(budget.storage(), 0);
            assert_eq!(budget.failed_work().is_some(), cap == 127);
        }
    }

    #[test]
    fn borrowed_contract_sticky_denial_prevents_constructor_and_query() {
        for storage in [false, true] {
            let f = Fixture::new();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let (data, ledger) = {
                let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
                (
                    BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap(),
                    resources.original_ledger_v1().unwrap(),
                )
            };
            if storage {
                assert!(budget.reserve_storage(LIMIT + 1).is_err());
            } else {
                assert!(budget.charge_work(LIMIT + 1).is_err());
            }
            let before = budget.work();
            assert!(
                BorrowedLocalContractsV1::source_data(
                    f.parts(),
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                )
                .is_err()
            );
            assert!(
                data.query_on_ledger(
                    &f.function,
                    &deref(1),
                    2,
                    ledger,
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned)
                )
                .is_err()
            );
            assert_eq!(budget.work(), before);
            assert_eq!(owned, 0);
        }
    }

    #[test]
    fn borrowed_contract_paid_query_matches_frozen_decisions_and_getter_order() {
        let f = Fixture::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
        let data = BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap();
        let ledger = resources.original_ledger_v1().unwrap();
        let p = deref(1);
        let got = data
            .query_on_ledger(&f.function, &p, 2, ledger, &mut resources)
            .unwrap();
        assert_eq!(
            decision_fields(got),
            (
                old_origin(&p, 2, &f.references).unwrap(),
                old_immutable(&f, 1),
                f.allocations[1],
                f.provenance[1]
            )
        );
        assert!(matches!(
            data.query_on_ledger(&f.function, &p, 3, ledger, &mut resources),
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a checked reference is dereferenced outside its authenticated payload region"
            ))
        ));
        assert_eq!(budget.work(), 128 + 2 * 81);
        assert_eq!(owned, 0);
    }

    #[test]
    fn borrowed_contract_paid_query_exact_and_one_short_work_bound() {
        for cap in [208, 209] {
            let f = Fixture::new();
            let mut work = Work::new(cap);
            let mut budget = Budget::new(&mut work, LIMIT);
            let mut owned = 0;
            let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
            let data = BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap();
            let ledger = resources.original_ledger_v1().unwrap();
            assert_eq!(
                data.query_on_ledger(&f.function, &deref(1), 2, ledger, &mut resources)
                    .is_ok(),
                cap == 209
            );
            assert_eq!(budget.failed_work().is_some(), cap == 208);
            assert_eq!(owned, 0);
        }
    }

    #[test]
    fn borrowed_contract_paid_query_refuses_foreign_ledger_on_same_function() {
        let f = Fixture::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let (data, ledger) = {
            let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
            (
                BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap(),
                resources.original_ledger_v1().unwrap(),
            )
        };
        let mut other_work = Work::new(LIMIT);
        let mut other = Budget::new(&mut other_work, LIMIT);
        let mut other_owned = 0;
        assert!(
            data.query_on_ledger(
                &f.function,
                &deref(1),
                2,
                ledger,
                &mut PreparationResourcesV1::new(&mut other, &mut other_owned)
            )
            .is_err()
        );
        assert_eq!(other.work(), 0);
        assert_eq!(other_owned, 0);
        assert!(
            data.query_on_ledger(
                &f.function,
                &deref(1),
                2,
                ledger,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_ok()
        );
    }

    #[test]
    fn borrowed_contract_paid_query_refuses_replaced_ledger_at_same_slot() {
        let f = Fixture::new();
        let mut original = Work::new(LIMIT);
        let mut foreign = Work::new(LIMIT);
        let mut budget = Budget::new(&mut original, LIMIT);
        let mut owned = 0;
        let (data, ledger) = {
            let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
            (
                BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap(),
                resources.original_ledger_v1().unwrap(),
            )
        };
        budget = Budget::new(&mut foreign, LIMIT);
        assert!(
            data.query_on_ledger(
                &f.function,
                &deref(1),
                2,
                ledger,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert_eq!(budget.work(), 0);
        assert_eq!(owned, 0);
    }

    #[test]
    fn borrowed_contract_paid_query_refuses_foreign_function_and_source_coordinates() {
        let f = Fixture::new();
        let foreign = f.function.clone();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
        let data = BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap();
        let ledger = resources.original_ledger_v1().unwrap();
        assert!(
            data.query_on_ledger(&foreign, &deref(1), 2, ledger, &mut resources)
                .is_err()
        );
        for (local, block) in [
            (u32::MAX, 2),
            (1, usize::MAX),
            (1, f.function.blocks().len()),
        ] {
            assert!(
                data.query_on_ledger(&f.function, &deref(local), block, ledger, &mut resources)
                    .is_err()
            );
        }
    }

    #[test]
    fn borrowed_contract_paid_query_refuses_legacy_view_even_on_original_ledger() {
        let f = Fixture::new();
        let old = f.legacy();
        let data = BorrowedLocalContractsV1::legacy(&old);
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
        let ledger = resources.original_ledger_v1().unwrap();
        assert!(
            data.query_on_ledger(&f.function, &deref(1), 2, ledger, &mut resources)
                .is_err()
        );
        assert_eq!(budget.work(), 81);
        assert_eq!(owned, 0);
    }

    #[test]
    fn borrowed_contract_none_origin_does_not_invent_fallback_allocation() {
        let mut f = Fixture::new();
        f.references.origins[1] = None;
        f.allocations[1] = None;
        f.provenance[1] = None;
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
        let data = BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap();
        let ledger = resources.original_ledger_v1().unwrap();
        let got = data
            .query_on_ledger(&f.function, &deref(1), 2, ledger, &mut resources)
            .unwrap();
        assert!(got.origin.is_none());
        assert!(got.allocation.is_none());
        assert!(got.allocation_provenance.is_none());
    }

    #[test]
    fn borrowed_contract_shared_and_unavailable_origins_keep_existing_semantics() {
        let mut f = Fixture::new();
        f.references.origins[1] = Some(CheckedReferenceOriginV1 {
            source: CheckedReferenceSourceV1::ProjectedSharedBorrow,
            availability: None,
        });
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let data = BorrowedLocalContractsV1::source_data(
            f.parts(),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        for block in 0..f.function.blocks().len() {
            assert_eq!(
                data.origin(&deref(1), block).unwrap(),
                Some(CheckedReferenceSourceV1::ProjectedSharedBorrow)
            );
            assert!(
                data.origin(&typed_place(1, SCALAR_TYPE), block)
                    .unwrap()
                    .is_none()
            );
        }
    }

    #[test]
    fn borrowed_contract_frame_is_source_sized_and_checked_before_storage() {
        let base = local_contract_frame_v1::<(), ()>(31).unwrap();
        assert_eq!(
            local_contract_frame_v1::<(), [u8; 4096]>(31).unwrap() - base,
            8192
        );
        assert_eq!(local_contract_frame_v1::<(), ()>(32).unwrap() - base, 2);
        assert!(local_contract_frame_v1::<(), ()>(usize::MAX).is_err());
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, base - 1);
        let mut owned = 0;
        assert!(
            PreparationResourcesV1::new(&mut budget, &mut owned)
                .reserve_storage(base)
                .is_err()
        );
        assert_eq!(owned, 0);
        assert_eq!(budget.storage(), 0);
        assert!(budget.failed_storage().is_some());
    }

    #[test]
    fn borrowed_contract_error_and_unwind_drop_borrows_without_refund_or_replacement() {
        for panic in [false, true] {
            let f = Fixture::new();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(37).unwrap();
            let mut owned = 0;
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
                let data =
                    BorrowedLocalContractsV1::source_data(f.parts(), &mut resources).unwrap();
                let ledger = resources.original_ledger_v1().unwrap();
                if panic {
                    panic!("borrowed DATA component unwind");
                }
                data.query_on_ledger(&f.function, &deref(1), 3, ledger, &mut resources)
                    .map(|_| ())
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert_eq!(owned, 0);
            assert_eq!(budget.storage(), 37);
        }
    }
}
