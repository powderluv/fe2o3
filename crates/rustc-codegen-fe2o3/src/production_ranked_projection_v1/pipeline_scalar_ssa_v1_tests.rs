use canonical_assertion_facts_v1::ProjectedAssertionConditionV1;
use fe2o3_kernel_ir::{
    CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
    CanonicalKernelIrVerificationResourceErrorV1 as Resource,
    CanonicalKernelIrWorkBudgetV1 as Work, CanonicalKernelIrWorkLedgerIdentityV1 as Ledger,
};
use pipeline_scalar_ssa_v1::{Index, Origin, Source, with_index};
use std::cell::Cell;

mod live_out_tests {
    use super::*;
    include!("pipeline_scalar_live_out_v1_tests.rs");
}

type R<T> = Result<T, ProductionRankedProjectionErrorV1>;
const WORD: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(0);
const UNIT: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(1);
const PTR: SemanticTypeIdV1 = SemanticTypeIdV1::from_index(2);
const FUNCTION: SemanticFunctionIdV1 = SemanticFunctionIdV1::from_index(0);
const FLOOR: usize = 19;

struct Facts<'a, 'w>(&'a mut Budget<'w>);
impl ProjectedAssertionFactsV1 for Facts<'_, '_> {
    fn charge_private_array_work(&mut self, amount: usize) -> R<()> {
        self.0
            .charge_work(amount)
            .map_err(ranked_projection_source_v1::resource)
    }
    fn helper_value_ledger_v1(&self) -> R<(usize, Ledger)> {
        Ok((
            self.0 as *const _ as usize,
            self.0.work_ledger_identity_v1(),
        ))
    }
    fn scalar_private_storage_v1(&self) -> R<usize> {
        Ok(self.0.storage())
    }
    fn reserve_scalar_private_storage_v1(&mut self, amount: usize) -> R<()> {
        self.0
            .reserve_storage(amount)
            .map_err(ranked_projection_source_v1::resource)
    }
    fn release_scalar_private_storage_v1(&mut self, amount: usize) -> R<()> {
        self.0
            .release_storage(amount)
            .map_err(ranked_projection_source_v1::resource)
    }
    fn private_array_initializer_count(&mut self, _: usize, _: usize) -> R<Option<u64>> {
        Ok(None)
    }
    fn is_materialized_block(&mut self, _: usize) -> R<bool> {
        Ok(true)
    }
    fn condition(
        &mut self,
        _: usize,
        _: bool,
        _: SemanticBlockIdV1,
    ) -> R<ProjectedAssertionConditionV1> {
        Ok(ProjectedAssertionConditionV1::Dynamic)
    }
}

fn copy(local: u32) -> SemanticOperandV1 {
    SemanticOperandV1::Copy(
        SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], WORD).unwrap(),
    )
}
fn assign(destination: u32, value: SemanticOperandV1) -> SemanticStatementV1 {
    typed_assignment(destination, WORD, SemanticRvalueKindV1::Use(value))
}
fn literal(value: u64) -> SemanticOperandV1 {
    SemanticOperandV1::Constant(SemanticConstantV1::new(
        WORD,
        SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value.into(), 8).unwrap()),
    ))
}
fn site(block: usize, statement: usize) -> ScalarAssignmentSiteV1 {
    ScalarAssignmentSiteV1 { block, statement }
}
fn goto(target: u32) -> SemanticTerminatorKindV1 {
    SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, target))
}
fn sequential() -> Vec<SemanticBasicBlockV1> {
    vec![
        block(
            60,
            vec![assign(2, copy(1)), assign(2, copy(2)), assign(3, copy(2))],
            goto(1),
        ),
        block(61, vec![assign(2, literal(7)), assign(3, copy(2))], goto(2)),
        block(
            62,
            vec![assign(3, copy(2))],
            SemanticTerminatorKindV1::Return,
        ),
    ]
}

fn make_uncaptured_owner(
    blocks: Vec<SemanticBasicBlockV1>,
) -> Result<ProductionSemanticSsaOwnerV1, fe2o3_pliron::ProductionSemanticSsaErrorV1> {
    make_uncaptured_owner_with_guard(blocks, false)
}

fn make_uncaptured_owner_with_guard(
    blocks: Vec<SemanticBasicBlockV1>,
    guarded: bool,
) -> Result<ProductionSemanticSsaOwnerV1, fe2o3_pliron::ProductionSemanticSsaErrorV1> {
    let scalar = SemanticBackendScalarV1::initialized(
        SemanticBackendPrimitiveV1::integer(false, 64, 8),
        SemanticScalarValidityRangeV1::new(0, u64::MAX.into()),
    );
    let pointer = SemanticBackendScalarV1::initialized(
        SemanticBackendPrimitiveV1::pointer(0, 8, 8),
        SemanticScalarValidityRangeV1::new(0, u64::MAX.into()),
    );
    let mut types = vec![
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(1)),
            SemanticLayoutIdentityV1::from_sha256(bytes(1)),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(8),
                8,
                SemanticBackendReprV1::scalar(scalar),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 64,
            }),
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(2)),
            SemanticLayoutIdentityV1::from_sha256(bytes(2)),
            SemanticTypeLayoutV1::with_exact_rustc_layout(
                0,
                1,
                SemanticFieldsShapeV1::arbitrary(vec![], vec![]).unwrap(),
                SemanticRustcVariantsV1::Single { index: 0 },
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                1,
                0,
                SemanticTypeLayoutDetailsV1::None,
            )
            .unwrap(),
            SemanticTypeShapeV1::Unit,
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(3)),
            SemanticLayoutIdentityV1::from_sha256(bytes(3)),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(8),
                8,
                SemanticBackendReprV1::scalar(pointer),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Pointer(
                SemanticPointerTypeV1::new(
                    WORD,
                    SemanticMutabilityV1::Mutable,
                    0,
                    64,
                    SemanticPointerMetadataV1::None,
                )
                .unwrap(),
            ),
        ),
    ];
    if guarded {
        types.push(SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(4)),
            SemanticLayoutIdentityV1::from_sha256(bytes(4)),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(1),
                1,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::integer(false, 8, 1),
                    SemanticScalarValidityRangeV1::new(0, 1),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Bool),
        ));
    }
    let dimensions = SemanticWorkgroupDimensionsV1::new([64, 1, 1]).unwrap();
    let source_contract = SemanticKernelSourceContractV1::new(
        Some(SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None).unwrap()),
        None,
        None,
    )
    .unwrap();
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256(bytes(10)),
        SemanticLayoutIdentityV1::from_sha256(bytes(250)),
        SemanticCanonAbiV1::GpuKernel,
        SemanticExternAbiV1::GpuKernel,
        false,
        false,
        1,
        vec![
            fe2o3_mir_model::semantic_mir_v1::SemanticAbiArgumentV1::source(
                SemanticAbiValueV1::new(
                    WORD,
                    SemanticAbiPassModeV1::Direct(
                        SemanticAbiValueAttributesV1::new(
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiRegularAttributesV1::new(
                                false, None, false, false, false, true,
                            ),
                            fe2o3_mir_model::semantic_mir_v1::SemanticAbiExtensionV1::None,
                            0,
                            None,
                        )
                        .unwrap(),
                    ),
                ),
            ),
        ],
        SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256(bytes(11)),
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256(bytes(12)),
        SemanticMonomorphizationIdentityV1::from_sha256(bytes(13)),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(14)),
        SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(15)),
        SemanticSourceProvenanceV1::unavailable(),
        abi,
        vec![
            local(20, UNIT, SemanticLocalRoleV1::Return),
            local(21, WORD, SemanticLocalRoleV1::Argument(0)),
            local(22, WORD, SemanticLocalRoleV1::Temporary),
            local(
                23,
                if guarded {
                    SemanticTypeIdV1::from_index(3)
                } else {
                    WORD
                },
                SemanticLocalRoleV1::Temporary,
            ),
            local(24, PTR, SemanticLocalRoleV1::Temporary),
        ],
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap()
    .with_kernel_entry(SemanticKernelEntryV1::new(
        SemanticLinkSymbolV1::new(b"pipeline_actual_use".to_vec()).unwrap(),
        SemanticKernelBindingIdentityV1::from_sha256(bytes(90)),
        source_contract,
    ));
    let admitted = InertSemanticMirRequestV1::new(
        SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256(bytes(250))),
        types,
        vec![],
        vec![],
        vec![],
        vec![function],
        vec![FUNCTION],
    )
    .unwrap()
    .admit_exact_v3(SemanticMirLimitsV1::default())
    .unwrap();
    let mir = ProductionSemanticMirOwnerV1::try_new(
        admitted,
        fe2o3_pliron::ProductionSemanticMirLimitsV1::default(),
    )
    .unwrap();
    ProductionSemanticSsaOwnerV1::try_new(
        mir,
        fe2o3_pliron::ProductionSemanticSsaLimitsV1::default(),
    )
}

fn make_owner(blocks: Vec<SemanticBasicBlockV1>, capture: bool) -> ProductionSemanticSsaOwnerV1 {
    let mut owner = make_uncaptured_owner(blocks).unwrap();
    if capture {
        let mut work = Work::new(10_000_000);
        let mut budget = Budget::new(&mut work, 10_000_000);
        owner
            .try_capture_occurrences_with_budget_v1(&mut budget)
            .unwrap();
        assert_eq!(budget.storage(), 0);
    }
    owner
}

fn source(owner: &ProductionSemanticSsaOwnerV1) -> Source<'_> {
    Source {
        owner,
        function: FUNCTION,
    }
}
fn function(owner: &ProductionSemanticSsaOwnerV1) -> &SemanticFunctionDeclV1 {
    &owner.source_semantic().functions()[0]
}
fn incomplete<T>(result: R<T>, expected: &'static str) {
    assert!(
        matches!(result, Err(ProductionRankedProjectionErrorV1::Incomplete(reason)) if reason == expected)
    );
}
fn project(
    index: &Index<'_>,
    facts: &mut dyn ProjectedAssertionFactsV1,
    owner: &ProductionSemanticSsaOwnerV1,
    local: u32,
    use_site: ScalarAssignmentSiteV1,
) -> R<(ProjectedPipelineScalarV1, Vec<ProductionRankedOperationV1>)> {
    let function = function(owner);
    let types = owner.source_semantic().types();
    let indices = vec![None; function.locals().len()];
    let mut arguments = vec![None; function.locals().len()];
    let mut next_argument = 0;
    let mut operations = Vec::new();
    let mut next_value = 0;
    let value = PipelineScalarProjectorV1 {
        types,
        function,
        index_values: &indices,
        runtime_index_arguments: &mut arguments,
        next_runtime_argument: &mut next_argument,
        uniform_inductions: &[],
        entry_operations: &mut operations,
        next_value: &mut next_value,
        assertion_proofs: SemanticAssertProofsV1::new(types, function)?,
        scalar_ssa: Some((index, facts)),
        work: 0,
    }
    .project_operand(&copy(local), use_site, &mut HashSet::new())?;
    Ok((value, operations))
}

#[test]
fn actual_use_ssa_projects_sequential_same_local_and_cross_block_definitions() {
    let owner = make_owner(sequential(), true);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    budget.reserve_storage(FLOOR).unwrap();
    with_index(
        source(&owner),
        function(&owner),
        &mut Facts(&mut budget),
        |index, facts| {
            for (use_site, definition) in [
                (site(0, 1), site(0, 0)),
                (site(0, 2), site(0, 1)),
                (site(1, 1), site(1, 0)),
                (site(2, 0), site(1, 0)),
            ] {
                assert_eq!(
                    index.resolve(function(&owner), 2, use_site, facts)?,
                    Origin::Assignment {
                        local: 2,
                        site: definition
                    }
                );
            }
            let (first, _) = project(index, facts, &owner, 2, site(0, 2))?;
            assert!(matches!(
                first,
                ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Argument(0))
            ));
            let (last, operations) = project(index, facts, &owner, 2, site(2, 0))?;
            assert!(matches!(
                last,
                ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Local(_))
            ));
            assert!(matches!(
                operations.as_slice(),
                [ProductionRankedOperationV1::IndexConstant { value: 7, .. }]
            ));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(budget.storage(), FLOOR);
    owner.verify_replay().unwrap();
}

#[test]
fn scoped_pipeline_index_reborrows_facts_across_short_lived_projector_buffers() {
    let owner = make_owner(sequential(), true);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    budget.reserve_storage(FLOOR).unwrap();
    let mut facts = Facts(&mut budget);
    let identity = facts.helper_value_ledger_v1().unwrap();
    for _ in 0..2 {
        let completed = Cell::new(false);
        pipeline_scalar_ssa_v1::with_pipeline_index(
            Some(source(&owner)),
            owner.source_semantic().types(),
            function(&owner),
            Some(&mut facts),
            |scoped| {
                let (index, facts) = scoped.expect("authentic captured owner");
                // Each call owns shorter-lived mutable output buffers, while
                // the scope and concrete facts object remain independently live.
                let (first, first_operations) = project(index, facts, &owner, 2, site(0, 2))?;
                assert!(matches!(
                    first,
                    ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Argument(0))
                ));
                assert!(first_operations.is_empty());
                let (last, last_operations) = project(index, facts, &owner, 2, site(2, 0))?;
                assert!(matches!(
                    last,
                    ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Local(_))
                ));
                assert!(matches!(
                    last_operations.as_slice(),
                    [ProductionRankedOperationV1::IndexConstant { value: 7, .. }]
                ));
                assert!(facts.helper_value_ledger_v1()? == identity);
                completed.set(true);
                Ok(())
            },
        )
        .unwrap();
        assert!(completed.get());
        assert!(facts.helper_value_ledger_v1().unwrap() == identity);
        assert_eq!(facts.scalar_private_storage_v1().unwrap(), FLOOR);
    }
    owner.verify_replay().unwrap();
}

#[test]
fn actual_use_ssa_keeps_branch_and_backedge_phis_explicit() {
    for backedge in [false, true] {
        let switch = SemanticTerminatorKindV1::SwitchInt {
            discriminant: copy(1),
            targets: SemanticSwitchTargetsV1::new(
                vec![SemanticSwitchTargetV1::new(
                    0,
                    cfg_edge(SemanticEdgeRoleV1::SwitchValue, 1),
                )],
                cfg_edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
            )
            .unwrap(),
        };
        let blocks = if backedge {
            vec![
                block(60, vec![assign(2, literal(1))], goto(1)),
                block(61, vec![assign(3, copy(2))], switch),
                block(62, vec![assign(2, literal(2))], goto(1)),
            ]
        } else {
            vec![
                block(60, vec![], switch),
                block(61, vec![assign(2, literal(1))], goto(3)),
                block(62, vec![assign(2, literal(2))], goto(3)),
                block(
                    63,
                    vec![assign(3, copy(2))],
                    SemanticTerminatorKindV1::Return,
                ),
            ]
        };
        let owner = make_owner(blocks, true);
        let mut work = Work::new(100_000);
        let mut budget = Budget::new(&mut work, 100_000);
        let completed = Cell::new(false);
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |index, facts| {
                let use_site = site(if backedge { 1 } else { 3 }, 0);
                incomplete(
                    index.resolve(function(&owner), 2, use_site, facts),
                    "a pipeline scalar requires an unsupported SSA block argument",
                );
                completed.set(true);
                Ok(())
            },
        );
        incomplete(
            result,
            "a pipeline scalar requires an unsupported SSA block argument",
        );
        assert!(completed.get());
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn actual_use_ssa_refuses_missing_capture_foreign_owner_function_and_wrong_site() {
    let owner = make_owner(sequential(), true);
    let foreign = make_owner(sequential(), true);
    let missing = make_owner(sequential(), false);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    incomplete(
        with_index(
            source(&owner),
            function(&foreign),
            &mut Facts(&mut budget),
            |_, _| -> R<()> { panic!("foreign function entered") },
        ),
        "a pipeline scalar SSA index substituted its source function",
    );
    incomplete(
        with_index(
            source(&missing),
            function(&missing),
            &mut Facts(&mut budget),
            |_, _| -> R<()> { panic!("uncaptured owner entered") },
        ),
        "a pipeline scalar has no captured source SSA occurrences",
    );
    for wrong_function in [false, true] {
        let completed = Cell::new(false);
        let expected = if wrong_function {
            "a pipeline scalar SSA query substituted its source function"
        } else {
            "a pipeline scalar has no captured use at its exact source site"
        };
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |index, facts| {
                incomplete(
                    index.resolve(
                        if wrong_function {
                            function(&foreign)
                        } else {
                            function(&owner)
                        },
                        2,
                        if wrong_function {
                            site(0, 2)
                        } else {
                            site(9, 0)
                        },
                        facts,
                    ),
                    expected,
                );
                completed.set(true);
                Ok(())
            },
        );
        incomplete(result, expected);
        assert!(completed.get());
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn actual_use_ssa_keeps_address_escaped_and_unreachable_uses_unadmitted() {
    for escaped in [false, true] {
        let blocks = if escaped {
            vec![block(
                60,
                vec![
                    assign(2, literal(1)),
                    typed_assignment(
                        4,
                        PTR,
                        SemanticRvalueKindV1::AddressOf {
                            mutability: SemanticMutabilityV1::Mutable,
                            place: raw_operand_place(&copy(2)).unwrap().clone(),
                        },
                    ),
                    assign(3, copy(2)),
                ],
                SemanticTerminatorKindV1::Return,
            )]
        } else {
            vec![
                block(60, vec![], SemanticTerminatorKindV1::Return),
                block(
                    61,
                    vec![
                        statement(SemanticStatementKindV1::StorageDead(
                            SemanticLocalIdV1::from_index(2),
                        )),
                        assign(3, copy(2)),
                    ],
                    SemanticTerminatorKindV1::Return,
                ),
            ]
        };
        let owner = make_owner(blocks, true);
        let mut work = Work::new(100_000);
        let mut budget = Budget::new(&mut work, 100_000);
        let completed = Cell::new(false);
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |index, facts| {
                incomplete(
                    index.resolve(
                        function(&owner),
                        2,
                        if escaped { site(0, 2) } else { site(1, 1) },
                        facts,
                    ),
                    "a pipeline scalar use is killed, unreachable, or retained in memory",
                );
                completed.set(true);
                Ok(())
            },
        );
        incomplete(
            result,
            "a pipeline scalar use is killed, unreachable, or retained in memory",
        );
        assert!(completed.get());
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn actual_use_ssa_undefined_and_killed_values_cannot_acquire_a_source_owner() {
    for killed in [false, true] {
        let mut statements = Vec::new();
        if killed {
            statements.push(assign(2, literal(1)));
            statements.push(statement(SemanticStatementKindV1::StorageDead(
                SemanticLocalIdV1::from_index(2),
            )));
        }
        statements.push(assign(3, copy(2)));
        let result = make_uncaptured_owner(vec![block(
            60,
            statements,
            SemanticTerminatorKindV1::Return,
        )]);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("undefined or killed source value acquired an SSA owner"),
        };
        // A straight-line entry has no incoming-edge transport to validate.
        // The original use fails at event zero, or after Define and Kill.
        assert!(
            matches!(&error, fe2o3_pliron::ProductionSemanticSsaErrorV1::Planner {
                function: FUNCTION,
                error: fe2o3_mir_model::SsaPlannerErrorV1::UndefinedAtUse { block, event, variable }
            } if *block == fe2o3_mir_model::SsaBlockIdV1::new(0)
                && *event == if killed { 2 } else { 0 }
                && variable.get() == 2),
            "unexpected source SSA refusal (killed={killed}): {error:?}",
        );
    }
}

#[test]
fn actual_use_ssa_cumulative_work_and_live_storage_are_exact_and_one_short() {
    let owner = make_owner(sequential(), true);
    // Six source assignments and five reads yield eleven captured events;
    // one argument plus six assignment definitions. Each repeated query uses
    // three binary-search comparisons, two same-site rows and the next site.
    let construction_work = 3 + 6 + 7 + 3 + 4 + 11 * 4;
    let query_work = 4 + 3 + 3 * 2 + 1;
    let exact_work = construction_work + 2 * query_work;
    let run = |work_limit, storage_limit| {
        let mut work = Work::new(work_limit);
        let mut budget = Budget::new(&mut work, storage_limit);
        budget.reserve_storage(FLOOR).unwrap();
        fn query_twice<'source, 'facts>(
            owner: &'source ProductionSemanticSsaOwnerV1,
        ) -> impl FnOnce(&Index<'source>, &mut (dyn ProjectedAssertionFactsV1 + 'facts)) -> R<()>
        {
            move |index, facts| {
                for _ in 0..2 {
                    assert_eq!(
                        index.resolve(function(owner), 2, site(0, 1), facts)?,
                        Origin::Assignment {
                            local: 2,
                            site: site(0, 0)
                        }
                    );
                }
                Ok(())
            }
        }
        let consume = query_twice(&owner);
        fn headers<C>(_: &C) -> usize {
            use std::mem::size_of;
            [
                size_of::<Option<C>>(),
                size_of::<Option<Index<'_>>>(),
                size_of::<R<Index<'_>>>(),
                size_of::<std::thread::Result<R<()>>>(),
                size_of::<R<()>>(),
                size_of::<R<()>>(),
                size_of::<Option<pipeline_scalar_ssa_v1::Failure>>(),
                size_of::<(usize, Ledger)>(),
                size_of::<Box<dyn std::any::Any + Send>>(),
            ]
            .into_iter()
            .sum()
        }
        let expected_peak = FLOOR
            + headers(&consume)
            + 7 * std::mem::size_of::<Option<Origin>>()
            + 4 * std::mem::size_of::<usize>();
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            consume,
        );
        let used = budget.work();
        let peak = budget.peak_storage();
        if result.is_ok() {
            assert_eq!(peak, expected_peak);
        }
        assert_eq!(budget.storage(), FLOOR);
        (result, used, peak)
    };
    let (result, used, peak) = run(exact_work, 100_000);
    result.unwrap();
    assert_eq!(used, exact_work);
    run(exact_work, peak).0.unwrap();
    assert!(matches!(
        run(exact_work - 1, peak).0,
        Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
            canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Work(_))
        ))
    ));
    assert!(matches!(
        run(exact_work, peak - 1).0,
        Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
            canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Storage(_))
        ))
    ));
}

#[test]
fn actual_use_ssa_foreign_ledger_and_callback_unwind_restore_owned_storage() {
    let owner = make_owner(sequential(), true);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    budget.reserve_storage(FLOOR).unwrap();
    let mut foreign_work = Work::new(100_000);
    let mut foreign_budget = Budget::new(&mut foreign_work, 100_000);
    foreign_budget.reserve_storage(50_000).unwrap();
    let completed = Cell::new(false);
    let error = with_index(
        source(&owner),
        function(&owner),
        &mut Facts(&mut budget),
        |index, facts| -> R<()> {
            let error = index
                .resolve(
                    function(&owner),
                    2,
                    site(0, 1),
                    &mut Facts(&mut foreign_budget),
                )
                .unwrap_err();
            assert!(matches!(
                error,
                ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Accounting
                    )
                )
            ));
            assert!(matches!(
                index.resolve(function(&owner), 2, site(0, 1), facts),
                Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Accounting
                    )
                ))
            ));
            completed.set(true);
            Ok(())
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        ProductionRankedProjectionErrorV1::CanonicalAssertions(
            canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Accounting)
        )
    ));
    assert!(completed.get());
    assert_eq!(budget.storage(), FLOOR);
    assert_eq!(foreign_budget.storage(), 50_000);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |_, _| -> R<()> { panic!("pipeline callback panic") },
        );
    }));
    assert!(panic.is_err());
    assert_eq!(budget.storage(), FLOOR);
}

#[derive(Debug)]
struct PanicDrop(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl Drop for PanicDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        panic!("deliberate discarded-value destructor panic");
    }
}

#[test]
fn actual_use_ssa_sticky_work_refusal_survives_refund_retry_and_hostile_drops() {
    for panic_payload in [false, true] {
        let owner = make_owner(sequential(), true);
        // Construction costs67; the first query costs14. The second is one
        // work unit short, so its final definition lookup must not execute.
        let mut work = Work::new(94);
        let mut budget = Budget::new(&mut work, 100_000);
        budget.reserve_storage(FLOOR).unwrap();
        let completed = Cell::new(false);
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |index, facts| -> R<PanicDrop> {
                index.resolve(function(&owner), 2, site(0, 1), facts)?;
                let error = index
                    .resolve(function(&owner), 2, site(0, 1), facts)
                    .unwrap_err();
                assert!(matches!(
                    error,
                    ProductionRankedProjectionErrorV1::CanonicalAssertions(
                        canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                            Resource::Work(_)
                        )
                    )
                ));
                facts.release_scalar_private_storage_v1(1)?;
                assert!(matches!(
                    index.resolve(function(&owner), 2, site(0, 1), facts),
                    Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                        canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                            Resource::Work(_)
                        )
                    ))
                ));
                facts.reserve_scalar_private_storage_v1(1)?;
                completed.set(true);
                let value = PanicDrop(dropped.clone());
                if panic_payload {
                    std::panic::panic_any(value);
                }
                Ok(value)
            },
        );
        assert!(matches!(
            result,
            Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Work(
                    _
                ))
            ))
        ));
        assert!(completed.get());
        assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(budget.storage(), FLOOR);
    }
}

#[test]
fn actual_use_ssa_rejected_entry_drains_pending_capture_without_masking_refusal() {
    let owner = make_owner(sequential(), true);
    let foreign = make_owner(sequential(), true);
    for mode in 0..3 {
        let mut work = Work::new(if mode == 0 { 0 } else { 100_000 });
        let mut budget = Budget::new(&mut work, if mode == 1 { FLOOR } else { 100_000 });
        budget.reserve_storage(FLOOR).unwrap();
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let captured = PanicDrop(dropped.clone());
        let result = with_index(
            source(&owner),
            function(if mode == 2 { &foreign } else { &owner }),
            &mut Facts(&mut budget),
            move |_, _| -> R<()> {
                drop(captured);
                panic!("refused construction invoked its continuation");
            },
        );
        match mode {
            0 => assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Work(_)
                    )
                ))
            )),
            1 => assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Storage(_)
                    )
                ))
            )),
            _ => incomplete(
                result,
                "a pipeline scalar SSA index substituted its source function",
            ),
        }
        assert_eq!(dropped.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(budget.storage(), FLOOR);
    }
}

#[test]
fn actual_use_ssa_helper_identity_failure_is_sticky_after_restoration() {
    struct Flaky<'a, 'b, 'w> {
        inner: Facts<'b, 'w>,
        fail: &'a Cell<bool>,
    }
    impl ProjectedAssertionFactsV1 for Flaky<'_, '_, '_> {
        fn charge_private_array_work(&mut self, n: usize) -> R<()> {
            self.inner.charge_private_array_work(n)
        }
        fn helper_value_ledger_v1(&self) -> R<(usize, Ledger)> {
            if self.fail.get() {
                Err(ranked_projection_source_v1::resource(Resource::Arithmetic))
            } else {
                self.inner.helper_value_ledger_v1()
            }
        }
        fn scalar_private_storage_v1(&self) -> R<usize> {
            self.inner.scalar_private_storage_v1()
        }
        fn reserve_scalar_private_storage_v1(&mut self, n: usize) -> R<()> {
            self.inner.reserve_scalar_private_storage_v1(n)
        }
        fn release_scalar_private_storage_v1(&mut self, n: usize) -> R<()> {
            self.inner.release_scalar_private_storage_v1(n)
        }
        fn private_array_initializer_count(&mut self, _: usize, _: usize) -> R<Option<u64>> {
            Ok(None)
        }
        fn is_materialized_block(&mut self, _: usize) -> R<bool> {
            Ok(true)
        }
        fn condition(
            &mut self,
            _: usize,
            _: bool,
            _: SemanticBlockIdV1,
        ) -> R<ProjectedAssertionConditionV1> {
            Ok(ProjectedAssertionConditionV1::Dynamic)
        }
    }
    let owner = make_owner(sequential(), true);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    budget.reserve_storage(FLOOR).unwrap();
    let fail = Cell::new(false);
    let completed = Cell::new(false);
    let result = with_index(
        source(&owner),
        function(&owner),
        &mut Flaky {
            inner: Facts(&mut budget),
            fail: &fail,
        },
        |index, facts| {
            fail.set(true);
            assert!(matches!(
                index.resolve(function(&owner), 2, site(0, 1), facts),
                Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Arithmetic
                    )
                ))
            ));
            fail.set(false);
            assert!(matches!(
                index.resolve(function(&owner), 2, site(0, 1), facts),
                Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
                    canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(
                        Resource::Arithmetic
                    )
                ))
            ));
            completed.set(true);
            Ok(())
        },
    );
    assert!(matches!(
        result,
        Err(ProductionRankedProjectionErrorV1::CanonicalAssertions(
            canonical_assertion_facts_v1::CanonicalAssertionErrorV1::Resource(Resource::Arithmetic)
        ))
    ));
    assert!(completed.get());
    assert_eq!(budget.storage(), FLOOR);
}

#[test]
fn actual_use_ssa_pipeline_free_functions_do_not_construct_an_index_or_touch_a_ledger() {
    let owner = make_owner(sequential(), false);
    let function = function(&owner);
    let indices = vec![None; function.locals().len()];
    let mut arguments = vec![None; function.locals().len()];
    let mut next_argument = 0;
    let mut operations = Vec::new();
    let mut next_value = 0;
    let effects = project_workgroup_pipeline_effects_v1(
        &[],
        owner.source_semantic().types(),
        function,
        &indices,
        &mut arguments,
        &mut next_argument,
        &[],
        &mut operations,
        &mut next_value,
        Some(source(&owner)),
        None,
    )
    .unwrap();
    assert_eq!(effects.len(), 3);
    assert!(effects.iter().all(Option::is_none));
    assert!(operations.is_empty());
    assert_eq!(next_value, 0);
}

#[test]
fn actual_use_ssa_pipeline_entry_rejects_same_bytes_foreign_types_and_missing_ledger() {
    let owner = make_owner(sequential(), true);
    let foreign_types = owner.source_semantic().types().to_vec();
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    incomplete(
        pipeline_scalar_ssa_v1::with_pipeline_index(
            Some(source(&owner)),
            &foreign_types,
            function(&owner),
            Some(&mut Facts(&mut budget)),
            |_| -> R<()> { panic!("foreign types entered scalar projector") },
        ),
        "a pipeline scalar SSA projection substituted its source types",
    );
    assert_eq!(budget.storage(), 0);
    incomplete(
        pipeline_scalar_ssa_v1::with_pipeline_index(
            Some(source(&owner)),
            owner.source_semantic().types(),
            function(&owner),
            None,
            |_| -> R<()> { panic!("missing ledger entered scalar projector") },
        ),
        "a pipeline scalar SSA projection has no source ledger",
    );
}

#[test]
fn actual_use_ssa_same_local_chain_keeps_exact_and_over_limit_node_refusals() {
    for nodes in [
        MAX_PIPELINE_SCALAR_NODES_V1,
        MAX_PIPELINE_SCALAR_NODES_V1 + 1,
    ] {
        let aliases = nodes - 1;
        let mut statements = vec![assign(2, copy(1))];
        for _ in 1..aliases {
            statements.push(assign(2, copy(2)));
        }
        statements.push(assign(3, copy(2)));
        let owner = make_owner(
            vec![block(60, statements, SemanticTerminatorKindV1::Return)],
            true,
        );
        let mut work = Work::new(1_000_000);
        let mut budget = Budget::new(&mut work, 100_000);
        let result = with_index(
            source(&owner),
            function(&owner),
            &mut Facts(&mut budget),
            |index, facts| project(index, facts, &owner, 2, site(0, aliases)),
        );
        if nodes == MAX_PIPELINE_SCALAR_NODES_V1 {
            assert!(matches!(
                result.unwrap().0,
                ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Argument(0))
            ));
        } else {
            assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "pipeline scalar expression exceeds its node limit"
                ))
            ));
        }
        assert_eq!(budget.storage(), 0);
    }
}

#[test]
fn actual_use_ssa_cycle_keys_are_distinct_source_sites_not_local_numbers() {
    let owner = make_owner(sequential(), true);
    let mut work = Work::new(100_000);
    let mut budget = Budget::new(&mut work, 100_000);
    with_index(
        source(&owner),
        function(&owner),
        &mut Facts(&mut budget),
        |index, facts| {
            for (site, expected) in [
                (site(0, 0), 0),
                (site(0, 1), 1),
                (site(0, 2), 2),
                (site(1, 0), 3),
                (site(1, 1), 4),
                (site(2, 0), 5),
            ] {
                assert_eq!(index.visit_key(function(&owner), site, facts)?, expected);
            }
            Ok(())
        },
    )
    .unwrap();
    let completed = Cell::new(false);
    let result = with_index(
        source(&owner),
        function(&owner),
        &mut Facts(&mut budget),
        |index, facts| {
            incomplete(
                index.visit_key(function(&owner), site(0, 3), facts),
                "a pipeline scalar cycle key has no assignment statement",
            );
            completed.set(true);
            Ok(())
        },
    );
    incomplete(
        result,
        "a pipeline scalar cycle key has no assignment statement",
    );
    assert!(completed.get());
    assert_eq!(budget.storage(), 0);
}

include!("pipeline_scalar_ssa_captured_guard_tests.rs");
