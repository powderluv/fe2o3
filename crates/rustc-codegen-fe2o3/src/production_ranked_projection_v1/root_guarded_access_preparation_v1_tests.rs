// Inert shared-emission controls. These raw fixtures grant no source custody.
mod root_guarded_access_preparation_controls {
    use super::super::bf16_nominal_preparation_resources_v1::PreparationResourcesV1;
    use super::super::root_guarded_access_preparation_v1::*;
    use super::*;
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    use std::panic::{AssertUnwindSafe, catch_unwind};
    type TestR<T> = std::result::Result<T, ProductionRankedProjectionErrorV1>;
    const FLOOR: usize = 65536;
    const LIMIT: usize = 1024 * 1024;

    #[derive(Debug)]
    struct State {
        allocations: Vec<Option<AllocationContractV1>>,
        provenance: Vec<Option<LocalAllocationProvenanceV1>>,
        views: Vec<Option<ProjectedViewV1>>,
        accesses: Vec<GuardedRankedAccessV1>,
        predicates: Vec<Option<GuardPredicateV1>>,
        direct: Vec<Option<GuardedRankedAccessV1>>,
        operations: Vec<ProductionRankedOperationV1>,
        next: u32,
        text: String,
    }
    impl State {
        fn new() -> Self {
            let mut allocations = vec![None; 5];
            allocations[1] = Some(AllocationContractV1 {
                allocation_origin: 1,
                noalias_class: 2,
                writable: true,
                singleton_object: false,
            });
            let mut provenance = vec![None; 5];
            provenance[1] = Some(LocalAllocationProvenanceV1::Argument(0));
            Self {
                allocations,
                provenance,
                views: vec![None; 5],
                accesses: Vec::new(),
                predicates: vec![None; 5],
                direct: vec![None; 2],
                operations: vec![ProductionRankedOperationV1::IndexConstant {
                    result: ProductionRankedValueIdV1::new(6),
                    value: 99,
                }],
                next: 7,
                text: String::new(),
            }
        }
        fn capacities(&self) -> Vec<usize> {
            let mut result = vec![
                self.operations.capacity(),
                self.accesses.capacity(),
                self.text.capacity(),
            ];
            for v in &self.views {
                result.extend(
                    v.as_ref()
                        .map(|v| vec![v.shape.capacity(), v.dynamic_extents.capacity()])
                        .unwrap_or_default(),
                );
            }
            for a in &self.accesses {
                result.extend([a.indices.capacity(), a.comparisons.capacity()]);
            }
            for p in &self.predicates {
                if let Some(p) = p {
                    result.push(p.comparisons.capacity());
                }
            }
            result
        }
    }
    fn call(destination: Option<u32>) -> SemanticDirectCallV1 {
        SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![typed_operand(1, SCALAR_TYPE), typed_operand(2, SCALAR_TYPE)],
            destination.map(|local| {
                SemanticCallDestinationV1::new(
                    typed_place(local, SCALAR_TYPE),
                    cfg_edge(SemanticEdgeRoleV1::CallReturn, 1),
                )
            }),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap()
    }
    fn emit(
        call: &SemanticDirectCallV1,
        state: &mut State,
        old: bool,
        precondition: Option<(ProductionRankedValueV1, ProductionRankedValueV1)>,
        success: Option<ProductionRankedValueV1>,
        direct: bool,
    ) -> TestR<()> {
        let apply = if old {
            frozen_common
        } else {
            append_mutable_access_legacy_v1
        };
        apply(
            &projection_types(),
            call,
            0,
            SemanticSourceProvenanceV1::unavailable(),
            SCALAR_TYPE,
            ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3)),
            precondition,
            success,
            direct,
            &state.allocations,
            &state.provenance,
            &mut state.views,
            &mut state.accesses,
            &mut state.predicates,
            &mut state.direct,
            &mut state.operations,
            &mut state.next,
            &mut state.text,
        )
    }
    fn parity(call: &SemanticDirectCallV1, initialize: impl Fn(&mut State)) -> State {
        let mut old = State::new();
        let mut new = State::new();
        initialize(&mut old);
        initialize(&mut new);
        let a = emit(call, &mut old, true, None, None, false).map_err(|e| format!("{e:?}"));
        let b = emit(call, &mut new, false, None, None, false).map_err(|e| format!("{e:?}"));
        assert_eq!(a, b);
        assert_eq!(format!("{old:?}"), format!("{new:?}"));
        assert_eq!(old.capacities(), new.capacities());
        new
    }
    #[test]
    fn ordinary_access_tail_matches_frozen_first_emission_and_cache_reuse() {
        let mut old = State::new();
        let mut new = State::new();
        for destination in [3, 4] {
            emit(&call(Some(destination)), &mut old, true, None, None, false).unwrap();
            emit(&call(Some(destination)), &mut new, false, None, None, false).unwrap();
            assert_eq!(format!("{old:?}"), format!("{new:?}"));
            assert_eq!(old.capacities(), new.capacities());
        }
        assert_eq!(new.next, 8);
        assert_eq!(new.operations.len(), 2);
        assert_eq!(new.accesses.len(), 2);
        assert_eq!(new.accesses[0].view, new.accesses[1].view);
    }
    #[test]
    fn ordinary_access_tail_preserves_receiver_and_readonly_errors() {
        parity(&call(Some(3)), |s| s.allocations[1] = None);
        parity(&call(Some(3)), |s| {
            s.allocations[1].as_mut().unwrap().writable = false
        });
    }
    #[test]
    fn ordinary_invalid_origin_fails_after_operation_id_and_diagnostic() {
        let s = parity(&call(Some(3)), |s| {
            s.allocations[1].as_mut().unwrap().allocation_origin = 90
        });
        assert_eq!(s.next, 8);
        assert_eq!(s.operations.len(), 2);
        assert!(s.text.contains("%7 = kernel.ranked_view"));
        assert!(s.views.iter().all(Option::is_none));
        assert!(s.accesses.is_empty());
    }
    #[test]
    fn ordinary_destination_refusals_preserve_already_installed_view() {
        for destination in [None, Some(90)] {
            let s = parity(&call(destination), |_| {});
            assert_eq!(s.next, 8);
            assert!(s.views[1].is_some());
            assert!(s.accesses.is_empty());
        }
        let s = parity(&call(Some(3)), |s| {
            s.predicates[3] = Some(GuardPredicateV1 {
                comparisons: vec![],
            });
        });
        assert!(s.views[1].is_some());
        assert!(s.accesses.is_empty());
        assert!(s.predicates[3].as_ref().unwrap().comparisons.is_empty());
    }
    #[test]
    fn ordinary_cache_conflict_and_ssa_overflow_preserve_frozen_refusal_order() {
        parity(&call(Some(3)), |s| {
            emit(&call(Some(4)), s, true, None, None, false).unwrap();
            s.views[1].as_mut().unwrap().noalias_class ^= 1;
        });
        let s = parity(&call(Some(3)), |s| s.next = u32::MAX);
        assert_eq!(s.next, u32::MAX);
        assert_eq!(s.operations.len(), 1);
        assert!(s.text.is_empty());
    }
    #[test]
    fn ordinary_precondition_and_checked_success_preserve_extent_and_comparison_order() {
        let index = ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3));
        let precondition = (index, ProductionRankedValueV1::Argument(2));
        for (pre, success) in [
            (None, None),
            (Some(precondition), None),
            (None, Some(ProductionRankedValueV1::Argument(4))),
        ] {
            let mut old = State::new();
            let mut new = State::new();
            emit(&call(Some(3)), &mut old, true, pre, success, false).unwrap();
            emit(&call(Some(3)), &mut new, false, pre, success, false).unwrap();
            assert_eq!(format!("{old:?}"), format!("{new:?}"));
            assert_eq!(old.capacities(), new.capacities());
            assert_eq!(
                new.accesses[0].output_extent.is_some(),
                pre.is_none() && success.is_none()
            );
            assert_eq!(
                new.accesses[0].comparisons.last(),
                Some(&(index, ProductionRankedValueV1::Argument(0)))
            );
            if let Some(pre) = pre {
                assert_eq!(new.accesses[0].comparisons[0], pre);
            }
        }
    }
    #[test]
    fn ordinary_direct_write_keeps_replacement_then_duplicate_refusal() {
        let mut old = State::new();
        let mut new = State::new();
        for _ in 0..2 {
            let a =
                emit(&call(None), &mut old, true, None, None, true).map_err(|e| format!("{e:?}"));
            let b =
                emit(&call(None), &mut new, false, None, None, true).map_err(|e| format!("{e:?}"));
            assert_eq!(a, b);
            assert_eq!(format!("{old:?}"), format!("{new:?}"));
        }
        assert!(new.direct[0].is_some());
        assert!(new.accesses.is_empty());
    }

    struct Paid {
        state: State,
        scratch: AccessScratchV1,
        result: std::result::Result<(), String>,
        owned: usize,
        work: usize,
        storage: usize,
        failed_work: bool,
        failed_storage: bool,
    }
    fn paid(work_limit: usize, storage: usize, destination: Option<u32>) -> Paid {
        // This is inert component data with an explicit preexisting fixture floor,
        // not an actual source factory, owner loan or alternative-ledger positive.
        let mut state = State::new();
        let mut scratch = AccessScratchV1::empty();
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, FLOOR + storage);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        let result = append_mutable_access_v1(
            &projection_types(),
            &call(destination),
            0,
            SemanticSourceProvenanceV1::unavailable(),
            SCALAR_TYPE,
            ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3)),
            None,
            None,
            false,
            &state.allocations,
            &state.provenance,
            &mut state.views,
            &mut state.accesses,
            &mut state.predicates,
            &mut state.direct,
            &mut state.operations,
            &mut state.next,
            None,
            &mut scratch,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .map_err(|e| format!("{e:?}"));
        assert_eq!(budget.storage(), FLOOR + owned);
        Paid {
            state,
            scratch,
            result,
            owned,
            work: budget.work(),
            storage: budget.storage(),
            failed_work: budget.failed_work().is_some(),
            failed_storage: budget.failed_storage().is_some(),
        }
    }
    #[test]
    fn paid_tail_matches_ordinary_payload_in_nonzero_existing_namespace() {
        let got = paid(LIMIT, LIMIT, Some(3));
        assert!(got.result.is_ok());
        let mut expected = State::new();
        emit(&call(Some(3)), &mut expected, true, None, None, false).unwrap();
        expected.text.clear();
        assert_eq!(format!("{:?}", got.state), format!("{expected:?}"));
        assert_eq!(got.storage, FLOOR + got.owned);
        assert!(got.owned > 0);
    }
    #[test]
    fn paid_tail_exact_and_every_short_work_boundary_retains_all_accepted_credits() {
        let full = paid(LIMIT, LIMIT, Some(3));
        assert!(full.result.is_ok());
        assert!(paid(full.work, full.owned, Some(3)).result.is_ok());
        for limit in 0..full.work {
            let short = paid(limit, LIMIT, Some(3));
            assert!(short.result.is_err() && short.failed_work);
            assert_eq!(short.storage, FLOOR + short.owned);
            assert!(!short.failed_storage);
        }
    }
    #[test]
    fn paid_tail_every_short_storage_boundary_keeps_nested_partial_owners() {
        let full = paid(LIMIT, LIMIT, Some(3));
        assert!(full.result.is_ok());
        let mut partial_operation = false;
        let mut cached = false;
        let mut predicate_before_append = false;
        for limit in 0..full.owned {
            let short = paid(LIMIT, limit, Some(3));
            assert!(short.result.is_err() && short.failed_storage);
            assert_eq!(short.storage, FLOOR + short.owned);
            partial_operation |= short.scratch.operation.is_some();
            cached |= short.state.views[1].is_some();
            predicate_before_append |= short.state.predicates[3].is_some()
                && short.state.accesses.is_empty()
                && short.scratch.access.is_some();
        }
        assert!(partial_operation && cached && predicate_before_append);
    }
    #[test]
    fn paid_semantic_refusal_retains_access_scratch_and_emitted_view() {
        let got = paid(LIMIT, LIMIT, None);
        assert!(got.result.is_err());
        assert!(!got.failed_work && !got.failed_storage);
        assert!(got.scratch.access.is_some());
        assert!(got.state.views[1].is_some());
        assert_eq!(got.state.operations.len(), 2);
        assert_eq!(got.state.next, 8);
        assert!(got.state.text.is_empty());
    }
    #[test]
    fn paid_postpreparation_error_and_panic_drop_before_refund_on_original_ledger() {
        for panic_after in [false, true] {
            let mut state = State::new();
            let mut scratch = AccessScratchV1::empty();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            let identity = budget.work_ledger_identity_v1();
            let mut owned = 0;
            let outcome = catch_unwind(AssertUnwindSafe(|| -> TestR<()> {
                append_mutable_access_v1(
                    &projection_types(),
                    &call(Some(3)),
                    0,
                    SemanticSourceProvenanceV1::unavailable(),
                    SCALAR_TYPE,
                    ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3)),
                    None,
                    None,
                    false,
                    &state.allocations,
                    &state.provenance,
                    &mut state.views,
                    &mut state.accesses,
                    &mut state.predicates,
                    &mut state.direct,
                    &mut state.operations,
                    &mut state.next,
                    None,
                    &mut scratch,
                    &mut PreparationResourcesV1::new(&mut budget, &mut owned),
                )?;
                if panic_after {
                    panic!("guarded callback panic");
                }
                Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "guarded callback error",
                ))
            }));
            assert_eq!(outcome.is_err(), panic_after);
            assert_eq!(state.accesses.len(), 1);
            assert_eq!(budget.storage(), FLOOR + owned);
            drop(outcome);
            drop(state);
            drop(scratch);
            assert!(budget.work_ledger_identity_v1() == identity);
            budget.release_storage(owned).unwrap();
            assert_eq!(budget.storage(), FLOOR);
        }
    }

    fn stage_fixture() -> (SemanticFunctionDeclV1, Vec<SemanticCallableDeclV1>) {
        (
            projection_function_with_locals(
                vec![
                    block(70, vec![], SemanticTerminatorKindV1::Call(call(Some(3)))),
                    block(71, vec![], SemanticTerminatorKindV1::Return),
                ],
                (0..5)
                    .map(|i| {
                        local(
                            80 + i,
                            SCALAR_TYPE,
                            if i == 0 {
                                SemanticLocalRoleV1::Return
                            } else {
                                SemanticLocalRoleV1::Temporary
                            },
                        )
                    })
                    .collect(),
            ),
            vec![compiler_intrinsic_callable(
                SemanticCompilerIntrinsicOperationV1::DisjointSliceGetMut {
                    disjoint_slice: SCALAR_TYPE,
                    index_witness: SCALAR_TYPE,
                    element: SCALAR_TYPE,
                    raw_index: SCALAR_TYPE,
                },
            )],
        )
    }
    fn stage(
        rows: &mut RootGuardedAccessStorageV1,
        state: &mut State,
        resources: &mut PreparationResourcesV1<'_, '_>,
    ) -> TestR<()> {
        let (function, calls) = stage_fixture();
        let options = SemanticOptionDominanceV1::analyze(&function, &[]).unwrap();
        let types = projection_types();
        let enums = SemanticEnumPayloadDominanceV1::analyze(&function, &types).unwrap();
        let mut indices = vec![None; 5];
        indices[2] = Some(ProjectedDisjointIndexV1 {
            value: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3)),
            mapping: SemanticDisjointIndexSpaceV1::Index1d,
            precondition: None,
            availability: None,
        });
        prepare_root_guarded_accesses_v1(
            &types,
            &calls,
            &function,
            &indices,
            &options,
            &enums,
            &state.allocations,
            &state.provenance,
            &mut state.predicates,
            rows,
            &mut state.operations,
            &mut state.next,
            resources,
        )
    }
    #[test]
    fn paid_stage_refuses_occupied_payloads_even_without_started_metadata() {
        for mode in 0..9 {
            let mut rows = RootGuardedAccessStorageV1::empty();
            match mode {
                0 => rows.views.push(None),
                1 => rows.views.reserve_exact(1),
                2 => rows.accesses.reserve_exact(1),
                3 => rows.scratch.comparisons.reserve_exact(1),
                4 => {
                    rows.scratch.operation = Some(ProductionRankedOperationV1::IndexConstant {
                        result: ProductionRankedValueIdV1::new(99),
                        value: 1,
                    })
                }
                5 => {
                    rows.scratch.predicate = Some(GuardPredicateV1 {
                        comparisons: vec![],
                    })
                }
                6 => rows.accesses = paid(LIMIT, LIMIT, Some(3)).state.accesses,
                7 => rows.frame_credits = 1,
                _ => {
                    let got = paid(LIMIT, LIMIT, None);
                    rows.scratch.access = got.scratch.access;
                }
            }
            let before_views = rows.views.capacity();
            let before_accesses = rows.accesses.capacity();
            let before_scratch = rows.scratch.comparisons.capacity();
            let mut state = State::new();
            let mut work = Work::new(LIMIT);
            let mut budget = Budget::new(&mut work, LIMIT);
            budget.reserve_storage(FLOOR).unwrap();
            let mut owned = 0;
            let result = stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "actual guarded access preparation cannot be replaced or retried"
                ))
            ));
            assert!(!rows.started && !rows.completed && rows.ledger.is_none());
            assert_eq!(
                (
                    rows.views.capacity(),
                    rows.accesses.capacity(),
                    rows.scratch.comparisons.capacity()
                ),
                (before_views, before_accesses, before_scratch)
            );
            assert_eq!(owned, 0);
            assert_eq!(state.operations.len(), 1);
            assert_eq!(state.next, 7);
        }
    }
    #[test]
    fn paid_stage_requires_original_ledger_and_refuses_retry_and_foreign_work_identity() {
        let mut rows = RootGuardedAccessStorageV1::empty();
        let mut state = State::new();
        assert!(
            stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::unmetered()
            )
            .is_err()
        );
        assert!(!rows.started);
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        stage(
            &mut rows,
            &mut state,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        assert!(rows.completed());
        assert_eq!(rows.accesses.len(), 1);
        let first_owned = owned;
        assert!(
            stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert_eq!(owned, first_owned);
        let mut other_work = Work::new(LIMIT);
        let mut other = Budget::new(&mut other_work, LIMIT);
        other.reserve_storage(FLOOR).unwrap();
        let mut other_owned = 0;
        assert!(
            stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut other, &mut other_owned)
            )
            .is_err()
        );
        assert_eq!(other_owned, 0);
        assert_eq!(rows.accesses.len(), 1);
        drop(rows);
        drop(state);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), FLOOR);
    }

    fn frozen_identity(
        call: &SemanticDirectCallV1,
        index_values: &[Option<ProjectedDisjointIndexV1>],
        option_dominance: &SemanticOptionDominanceV1,
        enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
        block_index: usize,
    ) -> TestR<ProjectedDisjointIndexV1> {
        // FROZEN_IDENTITY_BODY_START
        let projected = projected_disjoint_operand_v1(
            call,
            1,
            index_values,
            option_dominance,
            enum_payload_dominance,
            block_index,
        )?;
        if projected.mapping != SemanticDisjointIndexSpaceV1::Index1d {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "identity accessor received a non-identity mapping",
            ));
        }
        // FROZEN_IDENTITY_BODY_END
        Ok(projected)
    }
    #[test]
    fn ordinary_identity_normalization_matches_frozen_mapping_missing_and_range_cases() {
        let (function, _) = stage_fixture();
        let types = projection_types();
        let options = SemanticOptionDominanceV1::analyze(&function, &[]).unwrap();
        let enums = SemanticEnumPayloadDominanceV1::analyze(&function, &types).unwrap();
        let capability = ProjectedDisjointIndexV1 {
            value: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(3)),
            mapping: SemanticDisjointIndexSpaceV1::Index1d,
            precondition: None,
            availability: None,
        };
        for mode in 0..4 {
            let mut indices = vec![None; 5];
            if mode != 0 {
                indices[2] = Some(capability);
            }
            if mode == 2 {
                indices[2].as_mut().unwrap().mapping =
                    SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset: 1 };
            }
            if mode == 3 {
                indices.truncate(2);
            }
            let call = call(Some(3));
            let old =
                frozen_identity(&call, &indices, &options, &enums, 0).map_err(|e| format!("{e:?}"));
            let new = identity_operand_v1(&call, &indices, &options, &enums, 0)
                .map_err(|e| format!("{e:?}"));
            assert_eq!(old, new);
        }
    }
    #[test]
    fn ordinary_operation_cap_refuses_before_ssa_and_nested_payload_changes() {
        let mut old = State::new();
        let mut new = State::new();
        let operation = old.operations[0].clone();
        old.operations
            .resize(MAX_PROJECTED_OPERATIONS_V1, operation.clone());
        new.operations
            .resize(MAX_PROJECTED_OPERATIONS_V1, operation);
        let a =
            emit(&call(Some(3)), &mut old, true, None, None, false).map_err(|e| format!("{e:?}"));
        let b =
            emit(&call(Some(3)), &mut new, false, None, None, false).map_err(|e| format!("{e:?}"));
        assert_eq!(a, b);
        assert!(a.is_err());
        assert_eq!(old.next, 7);
        assert_eq!(new.next, 7);
        assert_eq!(old.operations, new.operations);
        assert_eq!(old.operations.capacity(), new.operations.capacity());
        assert!(new.text.is_empty());
        assert!(new.views.iter().all(Option::is_none));
    }
    #[test]
    fn paid_access_frame_reports_only_accepted_original_ledger_debit() {
        let mut rows = RootGuardedAccessStorageV1::empty();
        let mut state = State::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, FLOOR);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        assert!(
            stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert_eq!(rows.frame_credits, 0);
        assert!(!rows.started);
        assert_eq!(owned, 0);
        let mut rows = RootGuardedAccessStorageV1::empty();
        let mut state = State::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        stage(
            &mut rows,
            &mut state,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        assert_eq!(
            rows.frame_credits,
            4096 + std::mem::size_of::<RootGuardedAccessStorageV1>()
        );
        assert!(owned > rows.frame_credits);
        drop(rows);
        drop(state);
        budget.release_storage(owned).unwrap();
    }

    #[test]
    fn paid_stage_exact_and_every_short_work_storage_boundary_keeps_partial_owner_until_refund() {
        fn probe(work_limit: usize, storage_limit: usize, should_complete: bool) -> (usize, usize) {
            let mut rows = RootGuardedAccessStorageV1::empty();
            let mut state = State::new();
            let mut work = Work::new(work_limit);
            let mut budget = Budget::new(&mut work, FLOOR + storage_limit);
            budget.reserve_storage(FLOOR).unwrap();
            let mut owned = 0;
            let result = stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            );
            assert_eq!(result.is_ok(), should_complete);
            assert_eq!(rows.completed(), should_complete);
            assert_eq!(budget.storage(), FLOOR + owned);
            if !should_complete {
                assert!(budget.failed_work().is_some() || budget.failed_storage().is_some());
            }
            let used = budget.work();
            let retained = owned;
            // Partial operations/cache/access/predicate/scratch remain physically
            // in these owners until AFTER the accounting observation above.
            drop(result);
            drop(rows);
            drop(state);
            budget.release_storage(owned).unwrap();
            assert_eq!(budget.storage(), FLOOR);
            (used, retained)
        }
        let (work, storage) = probe(LIMIT, LIMIT, true);
        probe(work, storage, true);
        for short in 0..work {
            probe(short, storage, false);
        }
        for short in 0..storage {
            probe(work, short, false);
        }
    }

    #[test]
    fn paid_guard_stage_retains_exact_source_call_association_with_same_access_row() {
        let mut rows = RootGuardedAccessStorageV1::empty();
        let mut state = State::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        budget.reserve_storage(FLOOR).unwrap();
        let mut owned = 0;
        stage(
            &mut rows,
            &mut state,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        assert_eq!(
            rows.source_calls,
            vec![RootGuardedSourceCallV1 {
                source_call_ordinal: 0,
                block: 0,
                callee: SemanticCallableIdV1::from_index(0),
                destination: SemanticLocalIdV1::from_index(3),
                guarded_access: 0,
            }]
        );
        assert_eq!(rows.accesses.len(), 1);
        assert!(rows.accesses[0].semantic_site.is_none());
        assert_eq!(budget.storage(), FLOOR + owned);
        drop(rows);
        drop(state);
        budget.release_storage(owned).unwrap();
        assert_eq!(budget.storage(), FLOOR);
    }
    #[test]
    fn paid_guard_stage_refuses_unaccounted_association_capacity_without_replacing_it() {
        let mut rows = RootGuardedAccessStorageV1::empty();
        rows.source_calls.reserve_exact(1); // Preexisting inert hostile fixture, not production.
        let before = rows.source_calls.capacity();
        let mut state = State::new();
        let mut work = Work::new(LIMIT);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        assert!(
            stage(
                &mut rows,
                &mut state,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned)
            )
            .is_err()
        );
        assert_eq!(rows.source_calls.capacity(), before);
        assert!(!rows.started && rows.ledger.is_none() && !rows.completed());
        assert_eq!(owned, 0);
    }

    // FROZEN_ORACLE_START
    #[allow(clippy::too_many_arguments, clippy::redundant_field_names)]
    fn frozen_common(
        types: &[SemanticTypeDeclV1],
        call: &SemanticDirectCallV1,
        block_index: usize,
        source: SemanticSourceProvenanceV1,
        element: SemanticTypeIdV1,
        index: ProductionRankedValueV1,
        precondition: Option<(ProductionRankedValueV1, ProductionRankedValueV1)>,
        checked_success: Option<ProductionRankedValueV1>,
        direct_write: bool,
        local_allocations: &[Option<AllocationContractV1>],
        allocation_provenance: &[Option<LocalAllocationProvenanceV1>],
        views_by_origin: &mut [Option<ProjectedViewV1>],
        guarded_accesses: &mut Vec<GuardedRankedAccessV1>,
        option_predicates: &mut [Option<GuardPredicateV1>],
        direct_write_effects: &mut [Option<GuardedRankedAccessV1>],
        operations: &mut Vec<ProductionRankedOperationV1>,
        next_value: &mut u32,
        ranked_ir: &mut String,
    ) -> TestR<()> {
        let receiver = call
            .arguments()
            .first()
            .and_then(simple_operand_local)
            .ok_or(ProductionRankedProjectionErrorV1::Incomplete(
                "a checked disjoint receiver without one exact local",
            ))?
            .index() as usize;
        let allocation_contract = local_allocations.get(receiver).copied().flatten().ok_or(
            ProductionRankedProjectionErrorV1::Incomplete(
                "a checked disjoint receiver without one authenticated kernel-argument origin",
            ),
        )?;
        if !allocation_contract.writable {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "a checked mutable access is rooted in a read-only Rust allocation",
            ));
        }
        let origin_index = allocation_contract.allocation_origin as usize;
        let element_width = type_width(types, element)?;
        let view = match views_by_origin
            .get(origin_index)
            .and_then(|view| view.as_ref())
        {
            Some(view)
                if view.element_width == element_width
                    && view.writable
                    && view.shape == [DYNAMIC_EXTENT]
                    && view.dynamic_extents == [ProductionRankedValueV1::Argument(0)]
                    && view.memory_space == MemorySpaceAttr::Global
                    && view.allocation_origin == allocation_contract.allocation_origin
                    && view.noalias_class == allocation_contract.noalias_class =>
            {
                view.result
            }
            Some(_) => {
                return Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "one allocation origin was projected with conflicting element widths",
                ));
            }
            None => {
                reserve_operation(operations)?;
                let view = next_value_id(next_value)?;
                operations.push(ProductionRankedOperationV1::ViewInSpace {
                    result: view,
                    element_width,
                    writable: true,
                    shape: vec![DYNAMIC_EXTENT],
                    dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
                    memory_space: MemorySpaceAttr::Global,
                    allocation_origin: allocation_contract.allocation_origin,
                    noalias_class: allocation_contract.noalias_class,
                });
                push_ranked_ir(
                    ranked_ir,
                    &format!(
                        "  %{} = kernel.ranked_view <{}, true, [dynamic], Global>(%arg0)\n",
                        view.get(),
                        element_width,
                    ),
                )?;
                let slot = views_by_origin.get_mut(origin_index).ok_or(
                    ProductionRankedProjectionErrorV1::Unsupported(
                        "a kernel argument origin outside the semantic local table",
                    ),
                )?;
                *slot = Some(ProjectedViewV1 {
                    result: view,
                    element_width,
                    writable: true,
                    shape: vec![DYNAMIC_EXTENT],
                    dynamic_extents: vec![ProductionRankedValueV1::Argument(0)],
                    memory_space: MemorySpaceAttr::Global,
                    allocation_origin: allocation_contract.allocation_origin,
                    noalias_class: allocation_contract.noalias_class,
                });
                view
            }
        };
        let mut comparisons = Vec::with_capacity(2);
        if let Some(precondition) = precondition {
            comparisons.push(precondition);
        }
        comparisons.push((index, ProductionRankedValueV1::Argument(0)));
        // Allocation provenance excludes offset-only allocation contracts. It
        // proposes a source identity, not whole-slice equality: the latter is
        // independently rederived from the exact canonical store and guard.
        let output_extent = match allocation_provenance.get(receiver).copied().flatten() {
            Some(LocalAllocationProvenanceV1::Argument(argument))
                if checked_success.is_none() && precondition.is_none() =>
            {
                Some(ProductionRankedOutputExtentSourceV1::new(
                    argument,
                    ProductionRankedValueV1::Local(view),
                    ProductionRankedValueV1::Argument(0),
                    index,
                ))
            }
            _ => None,
        };
        let access = GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view,
            indices: vec![index],
            checked_success,
            comparisons,
            access: AccessKindAttr::Write,
            memory_space: MemorySpaceAttr::Global,
            source: source,
            semantic_site: None,
            output_extent,
        };
        if direct_write {
            let slot = direct_write_effects.get_mut(block_index).ok_or(
                ProductionRankedProjectionErrorV1::Unsupported(
                    "a write-only access block outside the semantic CFG",
                ),
            )?;
            if slot.replace(access).is_some() {
                return Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "multiple write-only effects occupy one semantic block",
                ));
            }
            return Ok(());
        }
        let destination = simple_call_destination(call)?.index() as usize;
        let predicate = option_predicates.get_mut(destination).ok_or(
            ProductionRankedProjectionErrorV1::Unsupported(
                "a checked disjoint destination outside the semantic local table",
            ),
        )?;
        if predicate.is_some() {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "multiple checked predicates for one semantic local",
            ));
        }
        *predicate = Some(GuardPredicateV1::for_access(&access));
        guarded_accesses.push(access);
        Ok(())
    }

    // FROZEN_ORACLE_END
}
