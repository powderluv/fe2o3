use super::*;
use crate::production_ranked_projection_v1::canonical_assertion_facts_v1::ProjectedAssertionConditionV1;

// Synthetic facts isolate census mechanics; the source driver exercises the
// actual canonical owner, emitted access and final ranked replay.
#[derive(Default)]
struct RecordingFacts {
    seen: Vec<(ProjectedSemanticAccessSiteV1, u32, u32)>,
}

impl ProjectedAssertionFactsV1 for RecordingFacts {
    fn private_array_initializer_count(
        &mut self,
        _: usize,
        _: usize,
    ) -> Result<Option<u64>, ProductionRankedProjectionErrorV1> {
        panic!("slice census fixtures must not query an initializer")
    }

    fn charge_private_array_work(
        &mut self,
        _: usize,
    ) -> Result<(), ProductionRankedProjectionErrorV1> {
        panic!("slice census fixtures must not charge private-array work")
    }

    fn is_materialized_block(
        &mut self,
        _: usize,
    ) -> Result<bool, ProductionRankedProjectionErrorV1> {
        unreachable!()
    }
    fn condition(
        &mut self,
        _: usize,
        _: bool,
        _: SemanticBlockIdV1,
    ) -> Result<ProjectedAssertionConditionV1, ProductionRankedProjectionErrorV1> {
        unreachable!()
    }
    fn slice_access(
        &mut self,
        site: ProjectedSemanticAccessSiteV1,
        ordinal: u32,
        assertion: u32,
    ) -> Result<ProjectedSliceInputV1, ProductionRankedProjectionErrorV1> {
        self.seen.push((site, ordinal, assertion));
        Ok(ProjectedSliceInputV1 {
            source_argument: 0,
            direct_local: None,
            element_width: 32,
        })
    }
}

fn site(statement: Option<usize>) -> ProjectedSemanticAccessSiteV1 {
    ProjectedSemanticAccessSiteV1 {
        block: 3,
        statement,
    }
}

fn check() -> ProjectedBoundsCheckV1 {
    ProjectedBoundsCheckV1 {
        assertion_block: Some(2),
        access_block: 3,
        extent_source: ProjectedBoundsExtentSourceV1::CanonicalSlice,
        index_local: SemanticLocalIdV1::from_index(1),
        index: ProductionRankedValueV1::Argument(0),
        extent: ProductionRankedValueV1::Argument(1),
        must_authorize_access: true,
    }
}

fn source(operation: usize, memory_space: MemorySpaceAttr) -> ProjectedAccessSourceV1 {
    ProjectedAccessSourceV1 {
        block: 0,
        operation,
        memory_space,
        access: AccessKindAttr::Read,
        source: SemanticSourceProvenanceV1::unavailable(),
        output_extent: None,
        semantic_site: Some(site(Some(7))),
    }
}

fn access(view: u32, kind: AccessKindAttr) -> ProductionRankedOperationV1 {
    ProductionRankedOperationV1::Access {
        kind,
        view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(view)),
        indices: vec![],
    }
}

fn guarded(memory_space: MemorySpaceAttr) -> GuardedAccessSiteV1 {
    GuardedAccessSiteV1 {
        insertion_operation: 0,
        access: GuardedRankedAccessV1 {
            failure: GuardedAccessFailureV1::Trap,
            view: ProductionRankedValueIdV1::new(10),
            indices: vec![],
            checked_success: None,
            comparisons: vec![],
            access: AccessKindAttr::Read,
            memory_space,
            source: SemanticSourceProvenanceV1::unavailable(),
            output_extent: None,
            semantic_site: Some(site(Some(7))),
        },
    }
}

#[test]
fn source_cursor_counts_retained_private_effects_and_delayed_reads_once() {
    let mut facts = RecordingFacts::default();
    let mut views = ProjectedViewsV1::new(0, Some(&mut facts));
    let operations = vec![
        access(10, AccessKindAttr::Read),
        ProductionRankedOperationV1::AtomicAccess {
            kind: AccessKindAttr::AtomicReadModifyWrite,
            ordering: AtomicOrderingAttr::Relaxed,
            scope: AtomicScopeAttr::Agent,
            view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(10)),
            indices: vec![],
        },
        access(10, AccessKindAttr::Read),
    ];
    let sources = vec![
        source(0, MemorySpaceAttr::Private),
        source(1, MemorySpaceAttr::Private),
        source(2, MemorySpaceAttr::Global),
    ];
    let mut delayed = vec![
        guarded(MemorySpaceAttr::Private),
        guarded(MemorySpaceAttr::Global),
    ];
    views.begin_site(site(Some(7)), 0, 0);
    assert_eq!(
        views
            .slice_input(check(), &sources, &delayed, &operations)
            .unwrap()
            .1,
        3
    );
    delayed.push(guarded(MemorySpaceAttr::Global));
    assert_eq!(
        views
            .slice_input(check(), &sources, &delayed, &operations)
            .unwrap()
            .1,
        4
    );
    views.begin_site(site(None), sources.len(), delayed.len());
    assert_eq!(
        views
            .slice_input(check(), &sources, &delayed, &operations)
            .unwrap()
            .1,
        0
    );
    drop(views);
    assert_eq!(
        facts.seen,
        vec![
            (site(Some(7)), 3, 2),
            (site(Some(7)), 4, 2),
            (site(None), 0, 2)
        ]
    );
}

fn queries(ordinal: u32, view: u32) -> ProjectedSliceQueriesV1 {
    ProjectedSliceQueriesV1(vec![QueriedSliceV1 {
        site: site(Some(7)),
        ordinal,
        view: ProductionRankedValueIdV1::new(view),
    }])
}

#[test]
fn final_census_retains_exact_ordinary_read_occurrence() {
    let blocks = vec![ProductionRankedBlockV1::new(
        vec![access(10, AccessKindAttr::Read)],
        ProductionRankedTerminatorV1::Return,
    )];
    let sources = vec![ProductionRankedAccessSourceV1::new(3, Some(7), 2, 0, 0)];
    queries(2, 10).validate(&blocks, &sources).unwrap();
    for (ordinal, view, expected) in [
        (
            1,
            10,
            "queried slice disappeared from the final source access census",
        ),
        (
            2,
            11,
            "queried slice differs from the final source access census",
        ),
    ] {
        assert!(matches!(queries(ordinal, view).validate(&blocks, &sources),
            Err(ProductionRankedProjectionErrorV1::Unsupported(actual)) if actual == expected));
    }
    for (kind, block, operation) in [
        (AccessKindAttr::Write, 0, 0),
        (AccessKindAttr::Read, 1, 0),
        (AccessKindAttr::Read, 0, 1),
    ] {
        let blocks = vec![ProductionRankedBlockV1::new(
            vec![access(10, kind)],
            ProductionRankedTerminatorV1::Return,
        )];
        let sources = vec![ProductionRankedAccessSourceV1::new(
            3,
            Some(7),
            2,
            block,
            operation,
        )];
        assert!(matches!(
            queries(2, 10).validate(&blocks, &sources),
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "queried slice differs from the final source access census"
            ))
        ));
    }
    assert!(matches!(
        queries(2, 10).validate(&blocks, &[]),
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "queried slice disappeared from the final source access census"
        ))
    ));
    assert!(matches!(
        queries(2, 10).validate(&blocks, &[sources[0], sources[0]]),
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "duplicate final slice occurrence"
        ))
    ));
    let mut duplicate = queries(2, 10);
    duplicate.0.extend(queries(2, 10).0);
    assert!(matches!(
        duplicate.validate(&blocks, &sources),
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "duplicate queried slice occurrence"
        ))
    ));
}

#[test]
fn query_census_limit_and_missing_assertion_fail_before_admission() {
    let mut facts = RecordingFacts::default();
    let mut views = ProjectedViewsV1::new(0, Some(&mut facts));
    views.begin_site(site(Some(7)), 0, 0);
    let mut absent = check();
    absent.assertion_block = None;
    assert!(matches!(
        views.slice_input(absent, &[], &[], &[]),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "projected slice requires an emitted Rust bounds assertion"
        ))
    ));
    for ordinal in 0..MAX_RANKED_BOUNDS_OPERATIONS {
        views
            .retain_query(ordinal as u32, ProductionRankedValueIdV1::new(0))
            .unwrap();
    }
    assert!(matches!(
        views.retain_query(
            MAX_RANKED_BOUNDS_OPERATIONS as u32,
            ProductionRankedValueIdV1::new(0)
        ),
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "slice query count exceeds ranked operation limit"
        ))
    ));
    drop(views);
    assert!(facts.seen.is_empty());
}

#[test]
fn final_census_sorts_distinct_sites_without_consuming_unrelated_rows() {
    let mut queries = queries(2, 10);
    queries.0.insert(
        0,
        QueriedSliceV1 {
            site: site(None),
            ordinal: 0,
            view: ProductionRankedValueIdV1::new(11),
        },
    );
    let blocks = vec![ProductionRankedBlockV1::new(
        vec![
            access(10, AccessKindAttr::Read),
            access(11, AccessKindAttr::Read),
        ],
        ProductionRankedTerminatorV1::Return,
    )];
    queries
        .validate(
            &blocks,
            &[
                ProductionRankedAccessSourceV1::new(3, Some(7), 0, 0, 0),
                ProductionRankedAccessSourceV1::new(3, Some(7), 2, 0, 0),
                ProductionRankedAccessSourceV1::new(3, Some(8), 0, 0, 1),
                ProductionRankedAccessSourceV1::new(3, None, 0, 0, 1),
            ],
        )
        .unwrap();
}

#[test]
fn private_initializer_forwarders_require_the_existing_facts_borrow() {
    let mut views = ProjectedViewsV1::new(1, None);
    assert!(matches!(
        views.charge_private_array_work(8),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "private array projection requires canonical facts"
        ))
    ));
    assert!(matches!(
        views.private_array_initializer_count(3, 7),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "private array projection requires canonical facts"
        ))
    ));
    assert!(views.get_mut(0).unwrap().is_none());
    views.finish().validate(&[], &[]).unwrap();
}

#[derive(Debug, PartialEq)]
enum AdapterEvent {
    Charge(usize),
    Initializer(usize, usize),
    Slice(ProjectedSemanticAccessSiteV1, u32, u32),
}

// These are inert delegation results over a real work ledger, not a fabricated
// canonical owner or a substitute for the legacy child's genuine attachment.
struct PrivateAdapterFacts<'a, 'w> {
    budget: &'a mut fe2o3_kernel_ir::CanonicalKernelIrVerificationResourceBudgetV1<'w>,
    events: Vec<AdapterEvent>,
    count: Result<Option<u64>, &'static str>,
}

impl ProjectedAssertionFactsV1 for PrivateAdapterFacts<'_, '_> {
    fn charge_private_array_work(
        &mut self,
        amount: usize,
    ) -> Result<(), ProductionRankedProjectionErrorV1> {
        self.events.push(AdapterEvent::Charge(amount));
        self.budget
            .charge_work(amount)
            .map_err(super::super::ranked_projection_source_v1::resource)
    }

    fn private_array_initializer_count(
        &mut self,
        block: usize,
        statement: usize,
    ) -> Result<Option<u64>, ProductionRankedProjectionErrorV1> {
        self.events
            .push(AdapterEvent::Initializer(block, statement));
        self.count
            .map_err(ProductionRankedProjectionErrorV1::Incomplete)
    }

    fn slice_access(
        &mut self,
        site: ProjectedSemanticAccessSiteV1,
        ordinal: u32,
        assertion: u32,
    ) -> Result<ProjectedSliceInputV1, ProductionRankedProjectionErrorV1> {
        self.events
            .push(AdapterEvent::Slice(site, ordinal, assertion));
        Ok(ProjectedSliceInputV1 {
            source_argument: 0,
            direct_local: None,
            element_width: 32,
        })
    }

    fn is_materialized_block(
        &mut self,
        _: usize,
    ) -> Result<bool, ProductionRankedProjectionErrorV1> {
        panic!("adapter component must not query CFG coverage")
    }

    fn condition(
        &mut self,
        _: usize,
        _: bool,
        _: SemanticBlockIdV1,
    ) -> Result<ProjectedAssertionConditionV1, ProductionRankedProjectionErrorV1> {
        panic!("adapter component must not query an assertion")
    }
}

#[test]
fn private_initializer_forwarders_share_facts_and_preserve_later_slice_cursor() {
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    for count in [Some(2), None] {
        let mut work = Work::new(8);
        let mut budget = Budget::new(&mut work, 17);
        budget.reserve_storage(17).unwrap();
        let mut facts = PrivateAdapterFacts {
            budget: &mut budget,
            events: Vec::new(),
            count: Ok(count),
        };
        let mut views = ProjectedViewsV1::new(0, Some(&mut facts));
        views.begin_site(site(Some(7)), 0, 0);
        views.charge_private_array_work(8).unwrap();
        assert_eq!(views.private_array_initializer_count(3, 7).unwrap(), count);
        let operations = vec![
            access(9, AccessKindAttr::Write),
            access(9, AccessKindAttr::Write),
            access(10, AccessKindAttr::Read),
        ];
        let sources = vec![
            source(0, MemorySpaceAttr::Private),
            source(1, MemorySpaceAttr::Private),
        ];
        views.begin_site(site(Some(8)), sources.len(), 0);
        let (_, ordinal) = views
            .slice_input(check(), &sources, &[], &operations)
            .unwrap();
        assert_eq!(ordinal, 0);
        views
            .retain_query(ordinal, ProductionRankedValueIdV1::new(10))
            .unwrap();
        let queries = views.finish();
        let blocks = vec![ProductionRankedBlockV1::new(
            operations,
            ProductionRankedTerminatorV1::Return,
        )];
        queries
            .validate(
                &blocks,
                &[
                    ProductionRankedAccessSourceV1::new(3, Some(7), 0, 0, 0),
                    ProductionRankedAccessSourceV1::new(3, Some(7), 1, 0, 1),
                    ProductionRankedAccessSourceV1::new(3, Some(8), 0, 0, 2),
                ],
            )
            .unwrap();
        assert_eq!(
            facts.events,
            [
                AdapterEvent::Charge(8),
                AdapterEvent::Initializer(3, 7),
                AdapterEvent::Slice(site(Some(8)), 0, 2)
            ]
        );
        assert_eq!(
            (
                facts.budget.work(),
                facts.budget.storage(),
                facts.budget.peak_storage()
            ),
            (8, 17, 17)
        );
    }
}

#[test]
fn private_initializer_forwarders_preserve_denial_and_query_error() {
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrVerificationResourceErrorV1 as Resource,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    for limit in [7, 8] {
        let mut work = Work::new(limit);
        let mut budget = Budget::new(&mut work, 17);
        budget.reserve_storage(17).unwrap();
        let mut facts = PrivateAdapterFacts {
            budget: &mut budget,
            events: Vec::new(),
            count: Err("component initializer rejection"),
        };
        let mut views = ProjectedViewsV1::new(0, Some(&mut facts));
        let result = views
            .charge_private_array_work(8)
            .and_then(|()| views.private_array_initializer_count(3, 7));
        views.finish().validate(&[], &[]).unwrap();
        if limit == 7 {
            assert!(
                matches!(result, Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                super::super::canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Work(error))))
                if error.actual() == 8 && error.limit() == 7)
            );
            assert_eq!(facts.events, [AdapterEvent::Charge(8)]);
            assert_eq!(facts.budget.work(), 0);
        } else {
            assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::Incomplete(
                    "component initializer rejection"
                ))
            ));
            assert_eq!(
                facts.events,
                [AdapterEvent::Charge(8), AdapterEvent::Initializer(3, 7)]
            );
            assert_eq!(facts.budget.work(), 8);
        }
        assert_eq!(
            (facts.budget.storage(), facts.budget.peak_storage()),
            (17, 17)
        );
    }
}
