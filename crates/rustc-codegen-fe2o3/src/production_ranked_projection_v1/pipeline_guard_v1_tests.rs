// Component fixtures exercise the real source-guard association and ranked CFG
// bounds checker. Full publication/epoch/owner checks remain in pipeline tests.
#[derive(Clone, Copy, Debug)]
enum Case {
    Exact,
    WrongIndex,
    FalseEdge,
    Sibling,
    Bypass,
    ChangedValue,
    Bound255,
    Bound257,
    Lane256,
    OtherView,
    WrongType,
    ReverseOrder,
    Revisit,
}

fn fixture(case: Case) -> SemanticFunctionDeclV1 {
    let calculate = typed_assignment(
        2,
        U64_TYPE,
        SemanticRvalueKindV1::Binary {
            operation: SemanticBinaryOpV1::Remainder,
            left: typed_operand(1, U64_TYPE),
            right: typed_constant(U64_TYPE, 512, 8),
        },
    );
    let compare = typed_assignment(
        3,
        BOOL_TYPE,
        SemanticRvalueKindV1::Binary {
            operation: SemanticBinaryOpV1::LessThan,
            left: typed_operand(2, U64_TYPE),
            right: typed_constant(
                if matches!(case, Case::WrongType) {
                    SCALAR_TYPE
                } else {
                    U64_TYPE
                },
                match case {
                    Case::Bound255 => 255,
                    Case::Bound257 => 257,
                    _ => 256,
                },
                if matches!(case, Case::WrongType) {
                    4
                } else {
                    8
                },
            ),
        },
    );
    let mut guard = vec![calculate, compare];
    if matches!(case, Case::ChangedValue) {
        guard.push(typed_assignment(
            2,
            U64_TYPE,
            SemanticRvalueKindV1::Use(typed_constant(U64_TYPE, 256, 8)),
        ));
    }
    let mut blocks = vec![
        block(
            220,
            vec![],
            if matches!(case, Case::Bypass | Case::Sibling) {
                zero_switch(4, BOOL_TYPE, 1, 2)
            } else {
                SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 1))
            },
        ),
        block(
            221,
            guard,
            if matches!(case, Case::FalseEdge) {
                zero_switch(3, BOOL_TYPE, 2, 3)
            } else if matches!(case, Case::Sibling) {
                zero_switch(3, BOOL_TYPE, 3, 3)
            } else {
                zero_switch(3, BOOL_TYPE, 3, 2)
            },
        ),
        block(
            222,
            vec![],
            if matches!(case, Case::Revisit) {
                zero_switch(4, BOOL_TYPE, 2, 3)
            } else {
                SemanticTerminatorKindV1::Return
            },
        ),
        block(223, vec![], SemanticTerminatorKindV1::Return),
    ];
    if matches!(case, Case::ReverseOrder) {
        // Swap guard/read numeric IDs and rebuild the actual source edges.
        let first = block(
            220,
            vec![],
            SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, 2)),
        );
        let read = block(222, vec![], SemanticTerminatorKindV1::Return);
        let original_guard = &blocks[1];
        let guard = block(
            221,
            original_guard.statements().to_vec(),
            zero_switch(3, BOOL_TYPE, 3, 1),
        );
        blocks = vec![
            first,
            read,
            guard,
            block(223, vec![], SemanticTerminatorKindV1::Return),
        ];
    }
    projection_function_with_locals(
        blocks,
        vec![
            local(220, U64_TYPE, SemanticLocalRoleV1::Return),
            local(221, U64_TYPE, SemanticLocalRoleV1::Argument(0)),
            local(222, U64_TYPE, SemanticLocalRoleV1::Temporary),
            local(223, BOOL_TYPE, SemanticLocalRoleV1::Temporary),
            local(224, BOOL_TYPE, SemanticLocalRoleV1::Argument(1)),
        ],
    )
}

fn projected(
    case: Case,
) -> Result<(Vec<ProductionRankedBlockV1>, bool), ProductionRankedProjectionErrorV1> {
    let types = assertion_proof_types();
    let function = fixture(case);
    let indices = vec![None; function.locals().len()];
    let mut arguments = vec![None; function.locals().len()];
    let mut next_argument = 0;
    let mut entry = Vec::new();
    let mut next_value = 0;
    let mut effects = vec![None; function.blocks().len()];
    let predicates = vec![None; function.locals().len()];
    let operand = match case {
        Case::WrongIndex => typed_operand(1, U64_TYPE),
        Case::Lane256 => typed_constant(U64_TYPE, 256, 8),
        _ => typed_operand(2, U64_TYPE),
    };
    let index = {
        let mut projector = PipelineScalarProjectorV1 {
            types: &types,
            function: &function,
            index_values: &indices,
            runtime_index_arguments: &mut arguments,
            next_runtime_argument: &mut next_argument,
            uniform_inductions: &[],
            entry_operations: &mut entry,
            next_value: &mut next_value,
            assertion_proofs: SemanticAssertProofsV1::new(&types, &function).unwrap(),
            scalar_ssa: None,
            work: 0,
        };
        project_pipeline_guarded_index_v1(
            &mut projector,
            if matches!(case, Case::ReverseOrder) {
                1
            } else {
                2
            },
            &operand,
            &mut effects,
        )?
    };
    let associated = effects
        .iter()
        .any(|v| matches!(v, Some(ProjectedPipelineEffectV1::Guard { .. })));
    if associated {
        assert!(matches!(index, ProjectedPipelineScalarV1::GuardedIndex));
    }
    let view = next_value_id(&mut next_value).unwrap();
    let modulus = next_value_id(&mut next_value).unwrap();
    let epoch = next_value_id(&mut next_value).unwrap();
    entry.extend([
        ProductionRankedOperationV1::ViewInSpace {
            result: view,
            element_width: 64,
            writable: false,
            shape: vec![
                2,
                if matches!(case, Case::OtherView) {
                    255
                } else {
                    256
                },
            ],
            dynamic_extents: vec![],
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: 1,
            noalias_class: 1,
        },
        ProductionRankedOperationV1::IndexConstant {
            result: modulus,
            value: 2,
        },
        ProductionRankedOperationV1::IndexConstant {
            result: epoch,
            value: 0,
        },
    ]);
    effects[if matches!(case, Case::ReverseOrder) {
        1
    } else {
        2
    }] = Some(ProjectedPipelineEffectV1::Access {
        view: ProductionRankedValueV1::Local(view),
        epoch: ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Local(epoch)),
        index,
        slot_result: None,
        modulus: ProductionRankedValueV1::Local(modulus),
        kind: AccessKindAttr::Read,
    });
    let (blocks, sources, _) = build_ranked_cfg(
        &types,
        &function,
        &[],
        &predicates,
        &vec![None; function.blocks().len()],
        &[],
        entry,
        effects
            .into_iter()
            .map(|e| ProjectedSemanticBlockV1 {
                items: e.into_iter().map(ProjectedBlockItemV1::Pipeline).collect(),
            })
            .collect(),
    )?;
    assert_eq!(sources.len(), 1);
    Ok((blocks, associated))
}

fn check(case: Case, expected_association: bool, clean: bool) {
    let (blocks, associated) = match projected(case) {
        Ok(value) => value,
        Err(error) if matches!(case, Case::Sibling | Case::Bypass | Case::ChangedValue) => {
            assert!(!clean && !expected_association);
            let expected = if matches!(case, Case::ChangedValue) {
                "a pipeline scalar temporary has multiple definitions"
            } else {
                "a pipeline scalar definition does not dominate its exact use"
            };
            assert!(
                matches!(error, ProductionRankedProjectionErrorV1::Incomplete(message) if message == expected),
                "{case:?}: {error}"
            );
            return;
        }
        Err(error) => panic!("{case:?}: {error}"),
    };
    assert_eq!(associated, expected_association, "{case:?}");
    let kernel = ProductionRankedKernelV1::new("source_guard_component", 1, blocks).unwrap();
    let construction =
        ProductionConstructionV1::ranked_kernel("source_guard_component_module", kernel).unwrap();
    let result =
        compile_ranked_kernel_for_lowering_v1(construction, ProductionSessionLimitsV1::default());
    if clean {
        assert!(result.unwrap().bounds_report().is_clean(), "{case:?}");
    } else {
        let error = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("{case:?} unexpectedly passed"),
        };
        let expected = if matches!(case, Case::Lane256) {
            "FE2O3-BOUNDS-001"
        } else {
            "FE2O3-BOUNDS-002"
        };
        assert!(
            error.contains(expected) && error.contains("Read"),
            "{case:?}: {error}"
        );
    }
}

#[test]
fn authentic_computed_guard_and_read_share_one_ssa_value() {
    check(Case::Exact, true, true);
}
#[test]
fn wrong_index_and_lane_256_do_not_borrow_guard() {
    check(Case::WrongIndex, false, false);
    check(Case::Lane256, false, false);
}
#[test]
fn false_edge_sibling_and_bypass_do_not_borrow_guard() {
    for case in [Case::FalseEdge, Case::Sibling, Case::Bypass] {
        check(case, false, false);
    }
}
#[test]
fn mutation_after_comparison_does_not_borrow_guard() {
    check(Case::ChangedValue, false, false);
}
#[test]
fn source_bound_255_is_retained_and_257_is_insufficient() {
    check(Case::Bound255, true, true);
    check(Case::Bound257, true, false);
}
#[test]
fn a_guard_for_256_cannot_authorize_a_different_view_of_255() {
    check(Case::OtherView, true, false);
}

#[test]
fn read_revisit_without_recrossing_guard_is_refused() {
    check(Case::Revisit, false, false);
}

// Exercise this new guard association with the existing strict assertion ledger.
fn resource_probe(work_limit: usize, storage_limit: usize) -> (bool, usize, usize, bool, bool) {
    use fe2o3_kernel_ir::{
        CanonicalKernelIrVerificationResourceBudgetV1 as Budget,
        CanonicalKernelIrWorkBudgetV1 as Work,
    };
    let types = assertion_proof_types();
    let function = fixture(Case::Exact);
    let graph = projected_loop_cfg_graph_v1(&function).unwrap();
    let inventory = assertion_definition_inventory(&function).unwrap();
    let indices = vec![None; function.locals().len()];
    let mut arguments = vec![None; function.locals().len()];
    let mut next_argument = 0;
    let mut operations = Vec::new();
    let mut next_value = 0;
    let mut effects = vec![None; function.blocks().len()];
    let mut work = Work::new(work_limit);
    let mut budget = Budget::new(&mut work, storage_limit);
    const FLOOR: usize = 83;
    budget.reserve_storage(FLOOR).unwrap();
    let mut owned = 0;
    let result = {
        let mut preparation = PreparationResourcesV1::new(&mut budget, &mut owned);
        (|| {
            let resources = AssertionResourcesV1::strict(&mut preparation)?;
            let proof = SemanticAssertProofsV1::new_borrowed_assertion_tables_v1(
                &types,
                &function,
                &graph,
                &inventory.counts,
                &inventory.blocks,
                &inventory.address_escaped,
                &inventory.assignments,
                resources,
            )?;
            let mut projector = PipelineScalarProjectorV1 {
                types: &types,
                function: &function,
                index_values: &indices,
                runtime_index_arguments: &mut arguments,
                next_runtime_argument: &mut next_argument,
                uniform_inductions: &[],
                entry_operations: &mut operations,
                next_value: &mut next_value,
                assertion_proofs: proof,
                scalar_ssa: None,
                work: 0,
            };
            project_pipeline_guarded_index_v1(
                &mut projector,
                2,
                &typed_operand(2, U64_TYPE),
                &mut effects,
            )
        })()
    };
    let out = (
        result.is_ok(),
        budget.work(),
        budget.peak_storage(),
        budget.failed_work().is_some(),
        budget.failed_storage().is_some(),
    );
    if result.is_ok() {
        assert!(matches!(
            effects[1],
            Some(ProjectedPipelineEffectV1::Guard { .. })
        ));
    }
    drop(result);
    drop(effects);
    drop(operations);
    assert_eq!(budget.storage(), FLOOR + owned);
    budget.release_storage(owned).unwrap();
    assert_eq!(budget.storage(), FLOOR);
    out
}

#[test]
fn guard_association_exact_and_one_under_resource_boundaries_fail_closed() {
    const LIMIT: usize = 256 * 1024 * 1024;
    let measured = resource_probe(LIMIT, LIMIT);
    assert!(measured.0 && measured.1 > 0 && measured.2 > 83);
    let exact = resource_probe(measured.1, measured.2);
    assert_eq!(exact, measured);
    let short_work = resource_probe(measured.1 - 1, measured.2);
    assert!(!short_work.0 && short_work.3);
    let short_storage = resource_probe(measured.1, measured.2 - 1);
    assert!(!short_storage.0 && short_storage.4);
    eprintln!(
        "guard association exact work={} peak={} floor=83; one-under work/storage rejected",
        measured.1, measured.2
    );
}

#[test]
fn reverse_numbered_guard_transports_exact_index_to_earlier_read_block() {
    check(Case::ReverseOrder, true, true);
}
#[test]
fn differently_typed_bound_does_not_authorize_read() {
    check(Case::WrongType, false, false);
}

#[test]
fn reverse_guard_keeps_live_induction_prefix_and_false_edge_unchanged() {
    let types = assertion_proof_types();
    let goto = |target| SemanticTerminatorKindV1::Goto(cfg_edge(SemanticEdgeRoleV1::Goto, target));
    let function = projection_function_with_locals(
        vec![
            block(
                230,
                vec![typed_assignment(
                    1,
                    SCALAR_TYPE,
                    SemanticRvalueKindV1::Use(typed_constant(SCALAR_TYPE, 0, 4)),
                )],
                goto(1),
            ),
            block(
                231,
                vec![typed_assignment(
                    2,
                    BOOL_TYPE,
                    SemanticRvalueKindV1::Binary {
                        operation: SemanticBinaryOpV1::LessThan,
                        left: typed_operand(1, SCALAR_TYPE),
                        right: typed_constant(SCALAR_TYPE, 4, 4),
                    },
                )],
                zero_switch(2, BOOL_TYPE, 5, 3),
            ),
            block(232, vec![], goto(4)),
            block(
                233,
                vec![
                    typed_assignment(
                        4,
                        U64_TYPE,
                        SemanticRvalueKindV1::Binary {
                            operation: SemanticBinaryOpV1::Remainder,
                            left: typed_operand(3, U64_TYPE),
                            right: typed_constant(U64_TYPE, 512, 8),
                        },
                    ),
                    typed_assignment(
                        5,
                        BOOL_TYPE,
                        SemanticRvalueKindV1::Binary {
                            operation: SemanticBinaryOpV1::LessThan,
                            left: typed_operand(4, U64_TYPE),
                            right: typed_constant(U64_TYPE, 256, 8),
                        },
                    ),
                ],
                zero_switch(5, BOOL_TYPE, 4, 2),
            ),
            block(
                234,
                vec![typed_assignment(
                    1,
                    SCALAR_TYPE,
                    SemanticRvalueKindV1::Binary {
                        operation: SemanticBinaryOpV1::Add,
                        left: typed_operand(1, SCALAR_TYPE),
                        right: typed_constant(SCALAR_TYPE, 1, 4),
                    },
                )],
                goto(1),
            ),
            block(235, vec![], SemanticTerminatorKindV1::Return),
        ],
        vec![
            local(230, U64_TYPE, SemanticLocalRoleV1::Return),
            local(231, SCALAR_TYPE, SemanticLocalRoleV1::Temporary),
            local(232, BOOL_TYPE, SemanticLocalRoleV1::Temporary),
            local(233, U64_TYPE, SemanticLocalRoleV1::Argument(0)),
            local(234, U64_TYPE, SemanticLocalRoleV1::Temporary),
            local(235, BOOL_TYPE, SemanticLocalRoleV1::Temporary),
        ],
    );
    let (inductions, mut entry, mut next_argument) =
        project_test_inductions_with_types(&types, &function).unwrap();
    assert_eq!(inductions.len(), 1);
    let mut next_value = entry
        .iter()
        .filter_map(ranked_operation_last_result_v1)
        .map(ProductionRankedValueIdV1::get)
        .max()
        .map_or(0, |value| value + 1);
    let indices = vec![None; function.locals().len()];
    let mut arguments = vec![None; function.locals().len()];
    arguments[3] = Some(0);
    let mut effects = vec![None; function.blocks().len()];
    let index = {
        let mut projector = PipelineScalarProjectorV1 {
            types: &types,
            function: &function,
            index_values: &indices,
            runtime_index_arguments: &mut arguments,
            next_runtime_argument: &mut next_argument,
            uniform_inductions: &inductions,
            entry_operations: &mut entry,
            next_value: &mut next_value,
            assertion_proofs: SemanticAssertProofsV1::new(&types, &function).unwrap(),
            scalar_ssa: None,
            work: 0,
        };
        project_pipeline_guarded_index_v1(
            &mut projector,
            2,
            &typed_operand(4, U64_TYPE),
            &mut effects,
        )
        .unwrap()
    };
    assert!(matches!(index, ProjectedPipelineScalarV1::GuardedIndex));
    let view = next_value_id(&mut next_value).unwrap();
    let modulus = next_value_id(&mut next_value).unwrap();
    let epoch = next_value_id(&mut next_value).unwrap();
    entry.extend([
        ProductionRankedOperationV1::ViewInSpace {
            result: view,
            element_width: 64,
            writable: false,
            shape: vec![2, 256],
            dynamic_extents: vec![],
            memory_space: MemorySpaceAttr::Global,
            allocation_origin: 1,
            noalias_class: 1,
        },
        ProductionRankedOperationV1::IndexConstant {
            result: modulus,
            value: 2,
        },
        ProductionRankedOperationV1::IndexConstant {
            result: epoch,
            value: 0,
        },
    ]);
    effects[2] = Some(ProjectedPipelineEffectV1::Access {
        view: ProductionRankedValueV1::Local(view),
        epoch: ProjectedPipelineScalarV1::Value(ProductionRankedValueV1::Local(epoch)),
        index,
        slot_result: None,
        modulus: ProductionRankedValueV1::Local(modulus),
        kind: AccessKindAttr::Read,
    });
    let (blocks, _, _) = build_ranked_cfg(
        &types,
        &function,
        &[],
        &vec![None; function.locals().len()],
        &vec![None; function.blocks().len()],
        &inductions,
        entry,
        effects
            .into_iter()
            .map(|e| ProjectedSemanticBlockV1 {
                items: e.into_iter().map(ProjectedBlockItemV1::Pipeline).collect(),
            })
            .collect(),
    )
    .unwrap();
    let mut found = false;
    for block in &blocks {
        if let ProductionRankedTerminatorV1::IndexLessThanArgs {
            lhs,
            true_arguments,
            false_arguments,
            true_block,
            ..
        } = block.terminator()
            && true_arguments.len() == 2
        {
            assert_eq!(false_arguments.len(), 1);
            assert_eq!(true_arguments[0], false_arguments[0]);
            assert_eq!(true_arguments[1], *lhs);
            assert_eq!(blocks[*true_block as usize].index_argument_count(), 2);
            assert!(blocks[*true_block as usize].operations().iter().any(|op| matches!(op,
                ProductionRankedOperationV1::Access { indices, .. } if indices[1] == (ProductionRankedValueV1::BlockArgument {block: *true_block, argument: 1}))));
            found = true;
        }
    }
    assert!(found);
    let kernel =
        ProductionRankedKernelV1::new("pipeline_guard_live_prefix", next_argument, blocks).unwrap();
    let construction =
        ProductionConstructionV1::ranked_kernel("pipeline_guard_live_prefix_module", kernel)
            .unwrap();
    assert!(
        compile_ranked_kernel_for_lowering_v1(construction, ProductionSessionLimitsV1::default())
            .unwrap()
            .bounds_report()
            .is_clean()
    );
}
