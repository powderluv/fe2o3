//! Inert component controls. None construct an authenticated source owner.
use super::*;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget, CanonicalKernelIrWorkBudgetV1 as Work,
};
use fe2o3_mir_model::semantic_mir_v1::*;
const LIMIT: usize = 16 * 1024 * 1024;
const FLOOR: usize = 37;
fn p(local: u32) -> SemanticPlaceV1 {
    SemanticPlaceV1::new(
        SemanticLocalIdV1::from_index(local),
        vec![],
        SemanticTypeIdV1::from_index(0),
    )
    .unwrap()
}
fn function(
    statements: Vec<SemanticStatementV1>,
    terminator: SemanticTerminatorKindV1,
) -> SemanticFunctionDeclV1 {
    let source = SemanticSourceProvenanceV1::unavailable();
    let unit = SemanticTypeIdV1::from_index(0);
    let abi = SemanticFunctionAbiV1::new(
        SemanticAbiIdentityV1::from_sha256([13; 32]),
        SemanticLayoutIdentityV1::from_sha256([14; 32]),
        SemanticCanonAbiV1::Rust,
        false,
        false,
        Vec::new(),
        SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([17; 32]),
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256([18; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([19; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([20; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([21; 32]),
        source,
        abi,
        (0..5u8)
            .map(|id| {
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([22 + id; 32]),
                    unit,
                    if id == 0 {
                        SemanticLocalRoleV1::Return
                    } else {
                        SemanticLocalRoleV1::Temporary
                    },
                    source,
                )
            })
            .collect(),
        SemanticBlockIdV1::from_index(0),
        vec![
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([25; 32]),
                source,
                statements,
                SemanticTerminatorV1::new(source, terminator),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}
fn statement(kind: SemanticStatementKindV1) -> SemanticStatementV1 {
    SemanticStatementV1::new(SemanticSourceProvenanceV1::unavailable(), kind)
}
fn site(statement: Option<usize>) -> ProjectedSemanticAccessSiteV1 {
    ProjectedSemanticAccessSiteV1 {
        block: 0,
        statement,
    }
}
fn guarded() -> GuardedRankedAccessV1 {
    GuardedRankedAccessV1 {
        failure: GuardedAccessFailureV1::Trap,
        view: ProductionRankedValueIdV1::new(0),
        indices: vec![ProductionRankedValueV1::Argument(0)],
        comparisons: vec![(
            ProductionRankedValueV1::Argument(0),
            ProductionRankedValueV1::Argument(1),
        )],
        checked_success: None,
        access: AccessKindAttr::Write,
        memory_space: MemorySpaceAttr::Global,
        source: SemanticSourceProvenanceV1::unavailable(),
        semantic_site: Some(site(Some(2))),
        output_extent: None,
    }
}
fn empty_rows(n: usize) -> Vec<ProjectedSemanticBlockV1> {
    (0..n)
        .map(|_| ProjectedSemanticBlockV1 { items: vec![] })
        .collect()
}
fn constant(id: u32) -> ProductionRankedOperationV1 {
    ProductionRankedOperationV1::IndexConstant {
        result: ProductionRankedValueIdV1::new(id),
        value: 7,
    }
}
fn run(
    rows: &[ProjectedSemanticBlockV1],
    prefix: &[ProductionRankedOperationV1],
    terms: &[ProjectedCfgTerminatorV1],
    work: usize,
    storage: usize,
) -> (Result<()>, emission::Emitted, usize, usize, bool, bool) {
    let mut emitted = emission::Emitted::empty();
    let mut work = Work::new(work);
    let mut budget = Budget::new(&mut work, storage);
    budget.reserve_storage(FLOOR).unwrap();
    let mut owned = 0;
    let result = emission::emit(
        rows,
        prefix,
        terms,
        0,
        &mut emitted,
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
    );
    assert_eq!(budget.storage(), FLOOR + owned);
    (
        result,
        emitted,
        budget.work(),
        budget.peak_storage(),
        budget.failed_work().is_some(),
        budget.failed_storage().is_some(),
    )
}
#[test]
fn source_visitor_assignment_keeps_original_rhs_then_destination_and_pointers() {
    let f = function(
        vec![statement(SemanticStatementKindV1::Assign(
            SemanticAssignmentV1::new(
                p(2),
                SemanticRvalueV1::new(
                    SemanticTypeIdV1::from_index(0),
                    SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(p(1))),
                ),
            ),
        ))],
        SemanticTerminatorKindV1::Return,
    );
    let SemanticStatementKindV1::Assign(a) = f.blocks()[0].statements()[0].kind() else {
        panic!("fixture")
    };
    let SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(rhs)) = a.value().kind() else {
        panic!("fixture")
    };
    let expected = [
        (rhs, AccessKindAttr::Read),
        (a.destination(), AccessKindAttr::Write),
    ];
    let mut cursor = 0;
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let count = source_uses::visit(
        &f,
        site(Some(0)),
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        |event, _| {
            let source_uses::Use::Value(use_) = event else {
                panic!("unexpected address")
            };
            assert!(std::ptr::eq(use_.place, expected[cursor].0));
            assert_eq!(use_.access, expected[cursor].1);
            assert_eq!(use_.ordinal, cursor);
            assert_eq!(use_.site, site(Some(0)));
            cursor += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((count, cursor, owned), (2, 2, 0));
}
#[test]
fn source_visitor_does_not_omit_address_formation() {
    let f = function(
        vec![statement(SemanticStatementKindV1::Assign(
            SemanticAssignmentV1::new(
                p(2),
                SemanticRvalueV1::new(
                    SemanticTypeIdV1::from_index(0),
                    SemanticRvalueKindV1::Borrow {
                        kind: SemanticBorrowKindV1::Shared,
                        place: p(1),
                    },
                ),
            ),
        ))],
        SemanticTerminatorKindV1::Return,
    );
    let mut order = 0;
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    source_uses::visit(
        &f,
        site(Some(0)),
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        |event, _| {
            match event {
                source_uses::Use::Address(place) => {
                    assert_eq!(order, 0);
                    assert_eq!(place.local().index(), 1);
                }
                source_uses::Use::Value(use_) => {
                    assert_eq!(order, 1);
                    assert_eq!(use_.place.local().index(), 2);
                    assert_eq!(use_.ordinal, 0);
                }
            }
            order += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(order, 2);
}
#[test]
fn source_visitor_call_keeps_arguments_then_destination() {
    let f = function(
        vec![],
        SemanticTerminatorKindV1::Call(
            SemanticDirectCallV1::new_callable(
                SemanticCallableIdV1::from_index(0),
                vec![SemanticOperandV1::Copy(p(1)), SemanticOperandV1::Move(p(2))],
                Some(SemanticCallDestinationV1::new(
                    p(3),
                    SemanticControlFlowEdgeV1::new(
                        SemanticEdgeRoleV1::CallReturn,
                        SemanticBlockIdV1::from_index(0),
                    ),
                )),
                SemanticUnwindActionV1::Unreachable,
            )
            .unwrap(),
        ),
    );
    let mut cursor = 0;
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    source_uses::visit(
        &f,
        site(None),
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        |event, _| {
            let source_uses::Use::Value(use_) = event else {
                panic!("address")
            };
            assert_eq!(use_.place.local().index(), cursor + 1);
            assert_eq!(
                use_.access,
                if cursor == 2 {
                    AccessKindAttr::Write
                } else {
                    AccessKindAttr::Read
                }
            );
            cursor += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(cursor, 3);
}
#[test]
fn source_visitor_first_callback_refusal_stops_later_uses() {
    let f = function(
        vec![statement(SemanticStatementKindV1::Assign(
            SemanticAssignmentV1::new(
                p(2),
                SemanticRvalueV1::new(
                    SemanticTypeIdV1::from_index(0),
                    SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(p(1))),
                ),
            ),
        ))],
        SemanticTerminatorKindV1::Return,
    );
    let mut seen = 0;
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let result = source_uses::visit(
        &f,
        site(Some(0)),
        &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        |_, _| {
            seen += 1;
            Err(Error::Incomplete("controlled first-use refusal"))
        },
    );
    assert!(matches!(
        result,
        Err(Error::Incomplete("controlled first-use refusal"))
    ));
    assert_eq!(seen, 1);
}
#[test]
fn source_visitor_exact_work_and_unmetered_refusal_precede_callback() {
    let f = function(
        vec![statement(SemanticStatementKindV1::Assume(
            SemanticOperandV1::Copy(p(1)),
        ))],
        SemanticTerminatorKindV1::Return,
    );
    assert!(
        source_uses::visit(
            &f,
            site(Some(0)),
            &mut PreparationResourcesV1::unmetered(),
            |_, _| panic!("unmetered callback")
        )
        .is_err()
    );
    for cap in [103, 104] {
        let mut seen = 0;
        let mut work = Work::new(cap);
        let mut budget = Budget::new(&mut work, LIMIT);
        let mut owned = 0;
        let result = source_uses::visit(
            &f,
            site(Some(0)),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
            |_, _| {
                seen += 1;
                Ok(())
            },
        );
        assert_eq!(result.is_ok(), cap == 104);
        assert_eq!(seen, usize::from(cap == 104));
        assert_eq!(owned, 0);
    }
}
#[test]
fn source_visitor_drop_and_missing_site_refuse() {
    let f = function(
        vec![],
        SemanticTerminatorKindV1::Drop {
            place: p(1),
            drop_glue: SemanticFunctionIdV1::from_index(0),
            target: SemanticControlFlowEdgeV1::new(
                SemanticEdgeRoleV1::DropReturn,
                SemanticBlockIdV1::from_index(0),
            ),
            unwind: SemanticUnwindActionV1::Unreachable,
        },
    );
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    for where_ in [
        site(None),
        site(Some(0)),
        ProjectedSemanticAccessSiteV1 {
            block: 9,
            statement: None,
        },
    ] {
        assert!(
            source_uses::visit(
                &f,
                where_,
                &mut PreparationResourcesV1::new(&mut budget, &mut owned),
                |_, _| panic!("refused use")
            )
            .is_err()
        );
    }
}
#[test]
fn concrete_blocks_have_real_prefix_and_original_branch_target() {
    let rows = empty_rows(2);
    let (result, out, _, _, _, _) = run(
        &rows,
        &[constant(0)],
        &[
            ProjectedCfgTerminatorV1::Branch(1),
            ProjectedCfgTerminatorV1::Return,
        ],
        LIMIT,
        LIMIT,
    );
    result.unwrap();
    assert!(out.complete);
    assert_eq!(out.blocks.len(), 3);
    assert_eq!(out.base[..2], [Some(1), Some(2)]);
    assert_eq!(out.blocks[0].operations(), &[constant(0)]);
    assert_eq!(
        out.blocks[0].terminator(),
        &ProductionRankedTerminatorV1::Branch { target: 1 }
    );
    assert_eq!(
        out.blocks[1].terminator(),
        &ProductionRankedTerminatorV1::Branch { target: 2 }
    );
    assert_eq!(
        out.blocks[2].terminator(),
        &ProductionRankedTerminatorV1::Return
    );
}
#[test]
fn concrete_guard_has_independent_success_failure_continuation_and_site() {
    let mut rows = empty_rows(1);
    rows[0].items.push(ProjectedBlockItemV1::Guarded(guarded()));
    let (result, out, _, _, _, _) = run(
        &rows,
        &[],
        &[ProjectedCfgTerminatorV1::Return],
        LIMIT,
        LIMIT,
    );
    result.unwrap();
    assert_eq!(out.blocks.len(), 5);
    assert_eq!(out.sources.len(), 1);
    assert_eq!(
        out.blocks[1].terminator(),
        &ProductionRankedTerminatorV1::IndexLessThan {
            lhs: ProductionRankedValueV1::Argument(0),
            rhs: ProductionRankedValueV1::Argument(1),
            true_block: 2,
            false_block: 3
        }
    );
    assert_eq!(
        out.blocks[2].terminator(),
        &ProductionRankedTerminatorV1::Branch { target: 4 }
    );
    assert_eq!(
        out.blocks[3].terminator(),
        &ProductionRankedTerminatorV1::Trap
    );
    assert_eq!(
        out.blocks[4].terminator(),
        &ProductionRankedTerminatorV1::Return
    );
    assert_eq!((out.sources[0].block, out.sources[0].operation), (2, 0));
    assert_eq!(out.sources[0].semantic_site, Some(site(Some(2))));
    assert_eq!(
        out.blocks[2].operations(),
        &[ProductionRankedOperationV1::Access {
            kind: AccessKindAttr::Write,
            view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0)),
            indices: vec![ProductionRankedValueV1::Argument(0)]
        }]
    );
}
#[test]
fn concrete_multisplit_preserves_every_original_successor_in_order() {
    let terms = [
        ProjectedCfgTerminatorV1::AnalysisMultiSplit {
            blocks: vec![1, 2, 3],
        },
        ProjectedCfgTerminatorV1::Return,
        ProjectedCfgTerminatorV1::Return,
        ProjectedCfgTerminatorV1::Trap,
    ];
    let (result, out, _, _, _, _) = run(&empty_rows(4), &[], &terms, LIMIT, LIMIT);
    result.unwrap();
    assert_eq!(out.base[..4], [Some(1), Some(3), Some(4), Some(5)]);
    assert_eq!(
        out.blocks[1].terminator(),
        &ProductionRankedTerminatorV1::AnalysisSplit {
            control_dependencies: vec![],
            first_block: 3,
            second_block: 2
        }
    );
    assert_eq!(
        out.blocks[2].terminator(),
        &ProductionRankedTerminatorV1::AnalysisSplit {
            control_dependencies: vec![],
            first_block: 4,
            second_block: 5
        }
    );
}
#[test]
fn concrete_cfg_prunes_only_unreachable_rows_and_refuses_reachable_absence() {
    let rows = empty_rows(2);
    let a = run(
        &rows,
        &[],
        &[
            ProjectedCfgTerminatorV1::Return,
            ProjectedCfgTerminatorV1::AbsentMaterialized,
        ],
        LIMIT,
        LIMIT,
    );
    a.0.unwrap();
    assert_eq!(a.1.base[..2], [Some(1), None]);
    assert_eq!(a.1.blocks.len(), 2);
    let b = run(
        &rows,
        &[],
        &[
            ProjectedCfgTerminatorV1::Branch(1),
            ProjectedCfgTerminatorV1::AbsentMaterialized,
        ],
        LIMIT,
        LIMIT,
    );
    assert!(b.0.is_err());
    assert!(!b.1.complete);
    assert!(b.1.blocks.is_empty());
}
#[test]
fn concrete_cfg_refuses_invalid_edges_census_and_new_operation_family() {
    assert!(
        run(
            &empty_rows(1),
            &[],
            &[ProjectedCfgTerminatorV1::Branch(99)],
            LIMIT,
            LIMIT
        )
        .0
        .is_err()
    );
    assert!(
        run(
            &empty_rows(2),
            &[],
            &[ProjectedCfgTerminatorV1::Return],
            LIMIT,
            LIMIT
        )
        .0
        .is_err()
    );
    assert!(
        run(
            &empty_rows(33),
            &[],
            &(0..33)
                .map(|_| ProjectedCfgTerminatorV1::Return)
                .collect::<Vec<_>>(),
            LIMIT,
            LIMIT
        )
        .0
        .is_err()
    );
    let unsupported = ProductionRankedOperationV1::SemanticConstant {
        result: ProductionRankedValueIdV1::new(0),
        value: 0,
    };
    let result = run(
        &empty_rows(1),
        &[unsupported],
        &[ProjectedCfgTerminatorV1::Return],
        LIMIT,
        LIMIT,
    );
    assert!(result.0.is_err());
    assert!(!result.1.complete);
}
#[test]
fn concrete_cfg_exact_and_one_short_work_storage_preserve_partial_output() {
    let mut rows = empty_rows(1);
    rows[0].items.push(ProjectedBlockItemV1::Guarded(guarded()));
    let terms = [ProjectedCfgTerminatorV1::Return];
    let initial = run(&rows, &[], &terms, LIMIT, LIMIT);
    initial.0.unwrap();
    let work = initial.2;
    let peak = initial.3;
    assert!(run(&rows, &[], &terms, work, peak).0.is_ok());
    let short = run(&rows, &[], &terms, work - 1, peak);
    assert!(short.0.is_err());
    assert!(short.4);
    assert!(!short.1.complete);
    let short = run(&rows, &[], &terms, work, peak - 1);
    assert!(short.0.is_err());
    assert!(short.5);
    assert!(!short.1.complete);
    assert!(
        !short.1.blocks.is_empty(),
        "partial blocks remain in outer owner"
    );
}
#[test]
fn concrete_cfg_retained_nested_backings_are_independent_original_copies() {
    let mut shape = Vec::with_capacity(9);
    shape.push(4);
    let op = ProductionRankedOperationV1::ViewInSpace {
        result: ProductionRankedValueIdV1::new(0),
        element_width: 4,
        writable: true,
        shape,
        dynamic_extents: vec![],
        memory_space: MemorySpaceAttr::Private,
        allocation_origin: 77,
        noalias_class: 77,
    };
    let (result, out, _, _, _, _) = run(
        &empty_rows(1),
        &[op.clone()],
        &[ProjectedCfgTerminatorV1::Return],
        LIMIT,
        LIMIT,
    );
    result.unwrap();
    let ProductionRankedOperationV1::ViewInSpace { shape: copy, .. } =
        &out.blocks[0].operations()[0]
    else {
        panic!("view")
    };
    let ProductionRankedOperationV1::ViewInSpace {
        shape: original, ..
    } = &op
    else {
        panic!("view")
    };
    assert_eq!(copy, original);
    assert_ne!(copy.as_ptr(), original.as_ptr());
    assert_eq!(copy.capacity(), copy.len());
}
#[test]
fn private_components_share_prefix_namespace_and_retain_read_write_sites() {
    let place = p(1);
    let mut stream = BlockStream::empty();
    stream.rows = empty_rows(1);
    stream.private_views.resize(5, None);
    let mut prefix = RootEntryPrefixV1::empty();
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    for access in [AccessKindAttr::Read, AccessKindAttr::Write] {
        let occurrence = root_checked_reference_use_preparation_v1::SourceUseOccurrenceV1 {
            place: &place,
            site: site(Some(0)),
            ordinal: 0,
            access,
            atomic: None,
            requirement: PlaceAccessRequirementV1::IfMemory,
            source: SemanticSourceProvenanceV1::unavailable(),
        };
        emit_private(
            &mut stream,
            &mut prefix,
            occurrence,
            (1, 4, 2, 4),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
    }
    assert_eq!(prefix.next_value, 3);
    assert_eq!(prefix.entry_operations.len(), 3);
    assert!(
        matches!(&prefix.entry_operations[0],ProductionRankedOperationV1::ViewInSpace{
        shape,memory_space:MemorySpaceAttr::Private,..} if shape==&[4])
    );
    assert_eq!(stream.rows[0].items.len(), 2);
    for (i, access) in [AccessKindAttr::Read, AccessKindAttr::Write]
        .into_iter()
        .enumerate()
    {
        let ProjectedBlockItemV1::Effect {
            operation,
            source: Some(source),
        } = &stream.rows[0].items[i]
        else {
            panic!("effect")
        };
        assert_eq!(source.access, access);
        assert_eq!(source.semantic_site, Some(site(Some(0))));
        assert_eq!(
            operation,
            &ProductionRankedOperationV1::Access {
                kind: access,
                view: ProductionRankedValueV1::Local(ProductionRankedValueIdV1::new(0)),
                indices: vec![ProductionRankedValueV1::Local(
                    ProductionRankedValueIdV1::new((i + 1) as u32)
                )]
            }
        );
    }
}
#[test]
fn ledger_mismatch_denial_and_unmetered_cannot_authorize_stream() {
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let mut resources = PreparationResourcesV1::new(&mut budget, &mut owned);
    let ledger = resources.original_ledger_v1().unwrap();
    check(&resources, ledger).unwrap();
    assert!(check(&resources, (ledger.0 ^ 1, ledger.1)).is_err());
    assert!(check(&PreparationResourcesV1::unmetered(), ledger).is_err());
    assert!(resources.work(usize::MAX).is_err());
    assert!(check(&resources, ledger).is_err());
}
#[test]
fn stream_header_bound_includes_fixed_emission_layout_and_callback_result() {
    assert!(
        stream_frame::<[u8; 1024], fn()>().unwrap() >= 8192 + size_of::<BlockStream>() + 4 * 1024
    );
}

#[test]
fn concrete_guard_rejects_empty_comparisons_and_checked_success_substitution() {
    for mode in 0..2 {
        let mut g = guarded();
        if mode == 0 {
            g.comparisons.clear()
        } else {
            g.checked_success = Some(ProductionRankedValueV1::Argument(9))
        }
        let mut rows = empty_rows(1);
        rows[0].items.push(ProjectedBlockItemV1::Guarded(g));
        let result = run(
            &rows,
            &[],
            &[ProjectedCfgTerminatorV1::Return],
            LIMIT,
            LIMIT,
        );
        assert!(result.0.is_err());
        assert!(!result.1.complete);
        assert!(result.1.blocks.is_empty());
    }
}
#[test]
fn private_view_conflict_refuses_before_another_operation() {
    let place = p(1);
    let mut stream = BlockStream::empty();
    stream.rows = empty_rows(1);
    stream.private_views.resize(2, None);
    stream.private_views[1] = Some(PrivateView {
        value: ProductionRankedValueIdV1::new(3),
        extent: 4,
        element_width: 4,
    });
    let mut prefix = RootEntryPrefixV1::empty();
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    let occurrence = root_checked_reference_use_preparation_v1::SourceUseOccurrenceV1 {
        place: &place,
        site: site(Some(0)),
        ordinal: 0,
        access: AccessKindAttr::Read,
        atomic: None,
        requirement: PlaceAccessRequirementV1::IfMemory,
        source: SemanticSourceProvenanceV1::unavailable(),
    };
    assert!(
        emit_private(
            &mut stream,
            &mut prefix,
            occurrence,
            (1, 5, 2, 4),
            &mut PreparationResourcesV1::new(&mut budget, &mut owned)
        )
        .is_err()
    );
    assert!(prefix.entry_operations.is_empty());
    assert!(stream.rows[0].items.is_empty());
}

#[test]
fn global_read_keeps_direct_allocation_and_each_original_terminator_source() {
    let mut stream = BlockStream::empty();
    stream.rows = empty_rows(2);
    let row = ProjectedCapabilityTerminatorEffectsV1 {
        global_read: Some(AllocationContractV1 {
            allocation_origin: 71,
            noalias_class: 72,
            writable: false,
            singleton_object: false,
        }),
        ..Default::default()
    };
    let mut work = Work::new(LIMIT);
    let mut budget = Budget::new(&mut work, LIMIT);
    let mut owned = 0;
    for block in 0..2 {
        let origin = SemanticSourceOriginV1::new(
            SemanticSourceFileIdentityV1::from_sha256([31 + block as u8; 32]),
            10,
            20,
            2,
            1,
            2,
            11,
        )
        .unwrap();
        let source = SemanticSourceProvenanceV1::new(Some(origin), Some(origin));
        append_effects(
            &mut stream,
            block,
            &row,
            source,
            &mut PreparationResourcesV1::new(&mut budget, &mut owned),
        )
        .unwrap();
        let [
            ProjectedBlockItemV1::Effect {
                operation,
                source: Some(actual),
            },
        ] = stream.rows[block].items.as_slice()
        else {
            panic!("one original read")
        };
        assert_eq!(
            operation,
            &ProductionRankedOperationV1::AllocationEffect {
                kind: AccessKindAttr::Read,
                memory_space: MemorySpaceAttr::Global,
                allocation_origin: 71,
                noalias_class: 72,
            }
        );
        assert_eq!(actual.source, source);
        assert_eq!(
            actual.semantic_site,
            Some(ProjectedSemanticAccessSiteV1 {
                block,
                statement: None
            })
        );
        assert_eq!(actual.access, AccessKindAttr::Read);
        assert_eq!(actual.memory_space, MemorySpaceAttr::Global);
        assert_eq!(actual.output_extent, None);
    }
}
