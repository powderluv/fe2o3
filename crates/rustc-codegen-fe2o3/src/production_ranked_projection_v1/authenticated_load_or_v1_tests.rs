//! Authenticated load_or projection and fallback controls.

use super::*;

fn authenticated_load_or_access() -> (Vec<ProductionRankedOperationV1>, GuardedRankedAccessV1) {
    let function = strided_read_call_function(None);
    let mut state = HashMap::from([(
        0,
        ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::ReadView(
            strided_read_view(),
        )),
    )]);
    let effects = transfer_capability_terminator_v1(
        &[strided_read_callable()],
        &function,
        0,
        &mut state,
        &[None; 5],
        &[None; 5],
        &[],
        &HashMap::new(),
        true,
    )
    .unwrap();
    let mut operations = Vec::new();
    let projected = project_strided_read_effects_v1(
        &projection_types(),
        &function,
        &[effects.read_view],
        &[None; 5],
        &mut [None; 5],
        &mut 1,
        &mut operations,
        &mut 0,
    )
    .unwrap();
    (operations, projected[0].clone().unwrap())
}

#[test]
fn authenticated_load_or_failure_reconverges_without_a_read_or_trap() {
    let (operations, access) = authenticated_load_or_access();
    assert_eq!(access.failure, GuardedAccessFailureV1::ReturnFallback);
    assert_eq!(access.indices.len(), 2);
    assert_eq!(access.comparisons[0].0, access.indices[0]);
    assert_eq!(access.comparisons[1].0, access.indices[1]);
    let expected = access.clone();
    let (blocks, sources, _) = single_guarded_cfg(operations, access);
    assert_eq!(blocks.len(), 6);
    for (ordinal, (lhs, rhs)) in expected.comparisons.iter().enumerate() {
        assert!(matches!(
            blocks[1 + ordinal].terminator(),
            ProductionRankedTerminatorV1::IndexLessThan {
                lhs: actual_lhs, rhs: actual_rhs, false_block: 4, ..
            } if actual_lhs == lhs && actual_rhs == rhs
        ));
    }
    assert!(matches!(
        blocks[3].operations(),
        [ProductionRankedOperationV1::Access { kind: AccessKindAttr::Read, view, indices }]
            if *view == ProductionRankedValueV1::Local(expected.view)
                && *indices == expected.indices
    ));
    assert!(blocks[4].operations().is_empty());
    assert!(matches!(
        blocks[4].terminator(),
        ProductionRankedTerminatorV1::Branch { target: 5 }
    ));
    assert!(
        !blocks
            .iter()
            .any(|block| matches!(block.terminator(), ProductionRankedTerminatorV1::Trap))
    );
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].block, 3);
    assert_eq!(sources[0].source, expected.source);
    assert_eq!(sources[0].semantic_site, expected.semantic_site);
}

#[test]
fn load_or_authentication_does_not_accept_foreign_view_callee_or_projected_index() {
    let function = strided_read_call_function(None);
    for origin in [
        ProjectedCapabilityOriginV1::ReadViewResult(strided_read_view()),
        ProjectedCapabilityOriginV1::ReadView(ProjectedReadViewV1 {
            element: ARRAY_TYPE,
            ..strided_read_view()
        }),
    ] {
        let mut state = HashMap::from([(0, ProjectedCapabilityValueV1::Known(origin))]);
        assert!(
            transfer_capability_terminator_v1(
                &[strided_read_callable()],
                &function,
                0,
                &mut state,
                &[None; 5],
                &[None; 5],
                &[],
                &HashMap::new(),
                true,
            )
            .is_err()
        );
    }
    let mut state = HashMap::from([(
        0,
        ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::ReadView(
            strided_read_view(),
        )),
    )]);
    assert!(
        transfer_capability_terminator_v1(
            &[zero_filled_tensor_load_callable()],
            &function,
            0,
            &mut state,
            &[None; 5],
            &[None; 5],
            &[],
            &HashMap::new(),
            true,
        )
        .is_err()
    );
    for position in [1, 2] {
        let state = HashMap::from([(
            0,
            ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::ReadView(
                strided_read_view(),
            )),
        )]);
        let mut arguments = vec![
            tensor_operand(0),
            tensor_operand(1),
            tensor_operand(2),
            tensor_operand(3),
        ];
        let direct_call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            arguments.clone(),
            None,
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        assert_eq!(
            authenticate_strided_read_v1(&direct_call, &state, SCALAR_TYPE, &[None; 5]),
            Some(ProjectedReadViewAccessV1 {
                view: strided_read_view(),
                row: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(1)),
                column: ProjectedReadValueV1::Local(SemanticLocalIdV1::from_index(2)),
            })
        );
        arguments[position] = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(position as u32),
                vec![
                    SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), SCALAR_TYPE)
                        .unwrap(),
                ],
                SCALAR_TYPE,
            )
            .unwrap(),
        );
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            arguments,
            None,
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        assert!(authenticate_strided_read_v1(&call, &state, SCALAR_TYPE, &[None; 5]).is_none());
    }
}

#[test]
fn load_or_fallback_preserves_live_induction_on_every_failure_edge() {
    let function = multi_block_induction_function(
        InductionCfgShape::Chain,
        SemanticLocalRoleV1::Argument(0),
        1,
    );
    let (inductions, mut entry_operations, _) = project_test_inductions(&function).unwrap();
    let (read_operations, mut access) = authenticated_load_or_access();
    // This fixture's read identities are independent of the loop induction.
    // Move their SSA results out of the induction prefix's namespace.
    let offset = entry_operations
        .iter()
        .filter_map(ranked_operation_last_result_v1)
        .map(ProductionRankedValueIdV1::get)
        .max()
        .map_or(0, |v| v + 1);
    // Use stable runtime arguments instead of transplanting another SSA graph.
    access.view = ProductionRankedValueIdV1::new(offset);
    access.indices = vec![
        ProductionRankedValueV1::Argument(0),
        ProductionRankedValueV1::Argument(0),
    ];
    access.comparisons = vec![
        (
            ProductionRankedValueV1::Argument(0),
            ProductionRankedValueV1::Argument(1),
        ),
        (
            ProductionRankedValueV1::Argument(0),
            ProductionRankedValueV1::Argument(1),
        ),
    ];
    drop(read_operations);
    entry_operations.push(ProductionRankedOperationV1::ViewInSpace {
        result: access.view,
        element_width: 32,
        writable: false,
        shape: vec![DYNAMIC_EXTENT, DYNAMIC_EXTENT],
        dynamic_extents: vec![ProductionRankedValueV1::Argument(1); 2],
        memory_space: MemorySpaceAttr::Global,
        allocation_origin: 3,
        noalias_class: 1,
    });
    let mut projected = (0..function.blocks().len())
        .map(|_| ProjectedSemanticBlockV1 { items: vec![] })
        .collect::<Vec<_>>();
    projected[2]
        .items
        .push(ProjectedBlockItemV1::Guarded(access));
    let (blocks, _, _) = build_ranked_cfg(
        &projection_types(),
        &function,
        &[],
        &vec![None; function.locals().len()],
        &vec![None; function.blocks().len()],
        &inductions,
        entry_operations,
        projected,
    )
    .unwrap();
    let mut observed = 0;
    for (ordinal, block) in blocks.iter().enumerate() {
        if let ProductionRankedTerminatorV1::IndexLessThanArgs {
            true_arguments,
            false_arguments,
            false_block,
            ..
        } = block.terminator()
        {
            if false_arguments.is_empty() {
                continue;
            }
            observed += 1;
            let expected = vec![ProductionRankedValueV1::BlockArgument {
                block: ordinal as u32,
                argument: 0,
            }];
            assert_eq!(*true_arguments, expected);
            assert_eq!(*false_arguments, expected);
            let failure = &blocks[*false_block as usize];
            assert!(failure.operations().is_empty());
            assert!(matches!(failure.terminator(),
                ProductionRankedTerminatorV1::BranchArgs { arguments, .. }
                    if *arguments == vec![ProductionRankedValueV1::BlockArgument { block: *false_block, argument: 0 }]
            ));
        }
    }
    assert_eq!(observed, 2);
    assert!(
        !blocks
            .iter()
            .any(|block| matches!(block.terminator(), ProductionRankedTerminatorV1::Trap))
    );
}

#[test]
fn fallback_predicates_forward_both_nested_inductions_without_aliasing_predecessors() {
    let mut blocks = Vec::new();
    let predicate = GuardPredicateV1 {
        comparisons: vec![
            (
                ProductionRankedValueV1::Argument(0),
                ProductionRankedValueV1::Argument(1),
            ),
            (
                ProductionRankedValueV1::Argument(2),
                ProductionRankedValueV1::Argument(3),
            ),
        ],
    };
    append_predicate_blocks_with_index_arguments(
        &mut blocks,
        0,
        vec![],
        &predicate,
        2,
        3,
        2,
        GuardedAccessFailureV1::ReturnFallback,
    )
    .unwrap();
    assert_eq!(blocks.len(), 2);
    for (ordinal, block) in blocks.iter().enumerate() {
        let expected = (0..2)
            .map(|argument| ProductionRankedValueV1::BlockArgument {
                block: ordinal as u32,
                argument,
            })
            .collect::<Vec<_>>();
        assert!(matches!(block.terminator(),
            ProductionRankedTerminatorV1::IndexLessThanArgs { true_arguments, false_arguments, .. }
                if *true_arguments == expected && *false_arguments == expected
        ));
    }
}

#[test]
fn load_or_fallback_rejects_changed_read_predicates_or_access_kind() {
    let (operations, original) = authenticated_load_or_access();
    let function = projection_function(vec![block(29, vec![], SemanticTerminatorKindV1::Return)]);
    for mutation in 0..4 {
        let mut access = original.clone();
        match mutation {
            0 => access.indices[0] = ProductionRankedValueV1::Argument(99),
            1 => access.comparisons[1].0 = ProductionRankedValueV1::Argument(99),
            2 => access.access = AccessKindAttr::Write,
            3 => access.checked_success = Some(ProductionRankedValueV1::Argument(0)),
            _ => unreachable!(),
        }
        let result = build_ranked_cfg(
            &projection_types(),
            &function,
            &[],
            &vec![None; function.locals().len()],
            &vec![None; function.blocks().len()],
            &[],
            operations.clone(),
            vec![ProjectedSemanticBlockV1 {
                items: vec![ProjectedBlockItemV1::Guarded(access)],
            }],
        );
        assert!(matches!(
            result,
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "a total read fallback without its exact rank-two read predicates"
            ))
        ));
    }
}
