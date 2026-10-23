#[derive(Clone, Copy)]
enum FallbackGuardMutationV1 {
    None,
    Row,
    ColumnExtent,
    SkipGuard,
    Trap,
    ForwardedValue,
    AccessIndex,
    ExtraRead,
}

fn fallback_guard_recipe_v1(n: u32, mutation: FallbackGuardMutationV1) -> ProductionRankedKernelV1 {
    use FallbackGuardMutationV1 as M;
    use fe2o3_pliron::ProductionRankedTerminatorV1 as T;
    let arg = ProductionRankedValueV1::Argument;
    let view = ProductionRankedValueIdV1::new(0);
    let constant = ProductionRankedValueIdV1::new(1);
    let args = |b| {
        (0..n)
            .map(|argument| ProductionRankedValueV1::BlockArgument { block: b, argument })
            .collect::<Vec<_>>()
    };
    let branch = |b, target| {
        if n == 0 {
            T::Branch { target }
        } else {
            T::BranchArgs {
                arguments: args(b),
                target,
            }
        }
    };
    let guard = |b, lhs, rhs, next, failure| {
        if n == 0 {
            T::IndexLessThan {
                lhs,
                rhs,
                true_block: next,
                false_block: failure,
            }
        } else {
            T::IndexLessThanArgs {
                lhs,
                rhs,
                true_arguments: args(b),
                false_arguments: if b == 2 && matches!(mutation, M::ForwardedValue) {
                    vec![ProductionRankedValueV1::Local(constant); n as usize]
                } else {
                    args(b)
                },
                true_block: next,
                false_block: failure,
            }
        }
    };
    let read = ProductionRankedOperationV1::Access {
        kind: AccessKindAttr::Read,
        view: ProductionRankedValueV1::Local(view),
        indices: vec![
            arg(if matches!(mutation, M::AccessIndex) {
                3
            } else {
                2
            }),
            arg(3),
        ],
    };
    ProductionRankedKernelV1::new(
        "total_fallback_guard",
        4,
        vec![
            ProductionRankedBlockV1::new(
                vec![
                    ProductionRankedOperationV1::ViewInSpace {
                        result: view,
                        element_width: 32,
                        writable: false,
                        shape: vec![0, 0],
                        dynamic_extents: vec![arg(0), arg(1)],
                        memory_space: dialect_kernel::MemorySpaceAttr::Global,
                        allocation_origin: 1,
                        noalias_class: 1,
                    },
                    ProductionRankedOperationV1::IndexConstant {
                        result: constant,
                        value: 1,
                    },
                ],
                if n == 0 {
                    T::Branch { target: 1 }
                } else {
                    T::BranchArgs {
                        arguments: vec![ProductionRankedValueV1::Local(constant); n as usize],
                        target: 1,
                    }
                },
            ),
            ProductionRankedBlockV1::with_index_arguments(
                n,
                vec![],
                guard(
                    1,
                    arg(if matches!(mutation, M::Row) { 3 } else { 2 }),
                    arg(0),
                    2,
                    if matches!(mutation, M::SkipGuard) {
                        5
                    } else {
                        4
                    },
                ),
            ),
            ProductionRankedBlockV1::with_index_arguments(
                n,
                vec![],
                guard(
                    2,
                    arg(3),
                    arg(if matches!(mutation, M::ColumnExtent) {
                        0
                    } else {
                        1
                    }),
                    3,
                    4,
                ),
            ),
            ProductionRankedBlockV1::with_index_arguments(
                n,
                if matches!(mutation, M::ExtraRead) {
                    vec![read.clone(), read]
                } else {
                    vec![read]
                },
                branch(3, 5),
            ),
            ProductionRankedBlockV1::with_index_arguments(
                n,
                vec![],
                if matches!(mutation, M::Trap) {
                    T::Trap
                } else {
                    branch(4, 5)
                },
            ),
            ProductionRankedBlockV1::with_index_arguments(n, vec![], T::Return),
        ],
    )
    .expect("guard mutation remains typed ordinary ranked CFG")
}

#[test]
fn guarded_effect_flow_requires_exact_ranked_predicates_and_live_forwarding() {
    for n in [0, 1, 2] {
        let recipe = fallback_guard_recipe_v1(n, FallbackGuardMutationV1::None);
        let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 100_000 };
        assert_eq!(
            ranked_total_fallback_cfg_v1(&recipe, (3, 0), &mut budget),
            Some(true)
        );
    }
}

#[test]
fn guarded_effect_flow_rejects_wrong_ranked_guard_bypass_trap_and_payload() {
    use FallbackGuardMutationV1 as M;
    for mutation in [
        M::Row,
        M::ColumnExtent,
        M::SkipGuard,
        M::Trap,
        M::ForwardedValue,
        M::AccessIndex,
        M::ExtraRead,
    ] {
        let recipe = fallback_guard_recipe_v1(2, mutation);
        let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 100_000 };
        assert_eq!(
            ranked_total_fallback_cfg_v1(&recipe, (3, 0), &mut budget),
            Some(false)
        );
    }
}
fn optional_effect_test_site(block: u32, statement: Option<u32>) -> SemanticAccessSiteV1 {
    SemanticAccessSiteV1 {
        block,
        statement,
        ordinal: 0,
    }
}

fn optional_effect_test_flow(
    events: &[(u32, Vec<(u64, SemanticAccessSiteV1)>)],
    successors: &[(u32, Vec<u32>)],
    optional: &[(u32, u64)],
) -> NormalizedEffectFlowV1 {
    let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 100_000 };
    let (events, successors) = expand_optional_effects_v1(
        &events.iter().cloned().collect(),
        &successors.iter().cloned().collect(),
        &optional.iter().copied().collect(),
        &mut budget,
    )
    .unwrap();
    effect_flow_signature_v1(0, &events, &successors, &mut budget).unwrap()
}

#[test]
fn guarded_effect_flow_matches_explicit_total_fallback() {
    let read = optional_effect_test_site(37, None);
    let store = optional_effect_test_site(61, Some(1));
    let actual = optional_effect_test_flow(
        &[(0, vec![(0, read), (1, store)])],
        &[(0, vec![])],
        &[(0, 0)],
    );
    // Two guard comparisons share the no-read fallback, then reconverge.
    let ranked = optional_effect_test_flow(
        &[(2, vec![(0, read)]), (4, vec![(0, store)])],
        &[
            (0, vec![1, 3]),
            (1, vec![2, 3]),
            (2, vec![4]),
            (3, vec![4]),
            (4, vec![]),
        ],
        &[],
    );
    assert_eq!(actual, ranked);
    assert_eq!(actual.entry_effects, BTreeSet::from([read, store]));
    assert_eq!(actual.next_effects, BTreeSet::from([(read, store)]));
}

#[test]
fn guarded_effect_flow_does_not_skip_ordinary_or_trapping_access() {
    let read = optional_effect_test_site(37, None);
    let store = optional_effect_test_site(61, Some(1));
    let conditional = optional_effect_test_flow(
        &[(0, vec![(0, read), (1, store)])],
        &[(0, vec![])],
        &[(0, 0)],
    );
    let mandatory =
        optional_effect_test_flow(&[(0, vec![(0, read), (1, store)])], &[(0, vec![])], &[]);
    let trap = optional_effect_test_flow(
        &[(1, vec![(0, read)]), (3, vec![(0, store)])],
        &[(0, vec![1, 2]), (1, vec![3]), (2, vec![]), (3, vec![])],
        &[],
    );
    assert_ne!(conditional, mandatory);
    assert_eq!(mandatory, trap);
}

#[test]
fn guarded_effect_flow_rejects_missing_reordered_and_foreign_effects() {
    let read = optional_effect_test_site(37, None);
    let store = optional_effect_test_site(61, Some(1));
    let foreign = optional_effect_test_site(61, Some(2));
    let original = optional_effect_test_flow(
        &[(0, vec![(0, read), (1, store)])],
        &[(0, vec![])],
        &[(0, 0)],
    );
    for events in [
        vec![(0, vec![(1, store)])],
        vec![(0, vec![(0, store), (1, read)])],
        vec![(0, vec![(0, read), (1, foreign)])],
    ] {
        assert_ne!(
            original,
            optional_effect_test_flow(&events, &[(0, vec![])], &[])
        );
    }
}

#[test]
fn guarded_effect_flow_preserves_loop_fallback_and_order() {
    let read = optional_effect_test_site(37, None);
    let store = optional_effect_test_site(61, Some(1));
    let original = optional_effect_test_flow(
        &[(1, vec![(0, read), (1, store)])],
        &[(0, vec![1, 2]), (1, vec![0]), (2, vec![])],
        &[(1, 0)],
    );
    let ranked = optional_effect_test_flow(
        &[(2, vec![(0, read)]), (4, vec![(0, store)])],
        &[
            (0, vec![1, 5]),
            (1, vec![2, 3]),
            (2, vec![4]),
            (3, vec![4]),
            (4, vec![0]),
            (5, vec![]),
        ],
        &[],
    );
    assert_eq!(original, ranked);
    assert!(original.next_effects.contains(&(store, store)));
    assert!(original.next_effects.contains(&(store, read)));
    let mut changed = ranked;
    changed.next_effects.remove(&(store, store));
    assert_ne!(original, changed);
}

#[test]
fn guarded_effect_flow_handles_adjacent_predicated_accesses() {
    let first = optional_effect_test_site(1, None);
    let second = optional_effect_test_site(2, None);
    let store = optional_effect_test_site(3, Some(0));
    let flow = optional_effect_test_flow(
        &[(0, vec![(0, first), (1, second), (2, store)])],
        &[(0, vec![])],
        &[(0, 0), (0, 1)],
    );
    assert_eq!(flow.entry_effects, BTreeSet::from([first, second, store]));
    assert_eq!(
        flow.next_effects,
        BTreeSet::from([(first, second), (first, store), (second, store)])
    );
}

#[test]
fn guarded_effect_flow_expansion_has_an_exact_finite_work_boundary() {
    let read = optional_effect_test_site(37, None);
    let events = BTreeMap::from([(0, vec![(0, read)])]);
    let successors = BTreeMap::from([(0, vec![])]);
    let optional = BTreeSet::from([(0, 0)]);
    let mut full = UnsupportedIndexCorrelationBudgetV1 { remaining: 1000 };
    expand_optional_effects_v1(&events, &successors, &optional, &mut full).unwrap();
    let used = 1000 - full.remaining;
    assert!(used > 0);
    let mut exact = UnsupportedIndexCorrelationBudgetV1 { remaining: used };
    assert!(expand_optional_effects_v1(&events, &successors, &optional, &mut exact).is_some());
    let mut short = UnsupportedIndexCorrelationBudgetV1 {
        remaining: used - 1,
    };
    assert!(expand_optional_effects_v1(&events, &successors, &optional, &mut short).is_none());
}
