use super::*;
use crate::reference_effect_v1::reference_signature_preimage_v1::{
    ReferenceCarrierV1, ReferencePointeeV1, ReferenceRegionV1, ReferenceReturnShapeV1,
    ReferenceSignatureInputV1,
};
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticExternAbiV1, SemanticFunctionSafetyV1, SemanticMirLimitsV1, SemanticMirResourceV1,
    SemanticMutabilityV1,
};

fn constant(bits: u128) -> ReferenceEffectExpressionV1 {
    ReferenceEffectExpressionV1::Constant(ReferenceConstantV1::Scalar {
        scalar: ReferenceScalarTypeV1::U32,
        bits,
    })
}

fn operand(bits: u128) -> ReferenceOperandV1 {
    ReferenceOperandV1::Constant(ReferenceConstantV1::Scalar {
        scalar: ReferenceScalarTypeV1::U32,
        bits,
    })
}

fn signature(scalar: ReferenceScalarTypeV1) -> ReferenceLogicalSignaturePreimageV1 {
    ReferenceLogicalSignaturePreimageV1::new(
        vec![
            ReferenceSignatureInputV1::Scalar(scalar),
            ReferenceSignatureInputV1::NominalOutput {
                carrier: ReferenceCarrierV1::DisjointSlice,
                element: ReferenceScalarTypeV1::U32,
            },
        ]
        .into_boxed_slice(),
        vec![
            ReferenceSignatureInputV1::Scalar(scalar),
            ReferenceSignatureInputV1::Reference {
                region: ReferenceRegionV1::Erased,
                mutability: SemanticMutabilityV1::Mutable,
                pointee: ReferencePointeeV1::Slice(ReferenceScalarTypeV1::U32),
            },
        ]
        .into_boxed_slice(),
        ReferenceReturnShapeV1::Unit,
        SemanticExternAbiV1::Rust,
        SemanticFunctionSafetyV1::Safe,
        false,
    )
    .unwrap()
}

fn identity() -> ReferenceFunctionIdentityV1 {
    ReferenceFunctionIdentityV1 {
        def_path_hash: [1; 16],
        function_sha256: [2; 32],
        item_definition_sha256: [3; 32],
        monomorphization_sha256: [4; 32],
        generic_type_arguments_sha256: [5; 32],
        const_generic_arguments_sha256: [6; 32],
        rustc_mir_body_sha256: [7; 32],
    }
}

// Inert graph data exercising all owned payloads, not authenticated extraction.
fn binding() -> AuthenticatedReferenceEffectBindingV1 {
    let write = ReferenceOutputWriteV1 {
        argument: 1,
        block: 1,
        statement: 0,
        coordinate: ReferenceOutputCoordinateV1::Dynamic(
            ReferenceEffectExpressionV1::KernelScalarArgument { argument: 0 },
        ),
        guard: ReferencePathPredicateV1 {
            clauses: vec![ReferenceGuardClauseV1 {
                atoms: vec![
                    ReferenceGuardAtomV1::SwitchValueSet {
                        discriminant: constant(1),
                        values: vec![0, 1].into_boxed_slice(),
                        inside_set: true,
                    },
                    ReferenceGuardAtomV1::Assert {
                        condition: constant(1),
                        expected: true,
                    },
                ]
                .into_boxed_slice(),
            }]
            .into_boxed_slice(),
        },
        rhs: constant(9),
        value: ReferenceValueV1::Use(ReferenceOperandV1::Copy(ReferencePlaceV1 {
            local: 3,
            projection: vec![ReferencePlaceProjectionV1::Field(0)].into_boxed_slice(),
        })),
    };
    AuthenticatedReferenceEffectBindingV1 {
        origin: ReferenceBindingOriginV1::SourceRegistration("fixture::registration".into()),
        logical_kernel_name: "fixture".into(),
        kernel: identity(),
        reference: identity(),
        signature_preimage: signature(ReferenceScalarTypeV1::U32),
        effect_ir_sha256: [8; 32],
        effect_ir: ReferenceEffectIrV1 {
            argument_count: 2,
            local_count: 4,
            relations: vec![
                ReferenceArgumentRelationV1::ScalarInput {
                    argument: 0,
                    scalar: ReferenceScalarTypeV1::U32,
                },
                ReferenceArgumentRelationV1::DisjointOutputSlice {
                    argument: 1,
                    element: ReferenceScalarTypeV1::U32,
                },
            ]
            .into_boxed_slice(),
            blocks: vec![
                ReferenceBlockV1 {
                    block: 0,
                    assignments: vec![ReferenceAssignmentV1 {
                        statement: 0,
                        destination: ReferencePlaceV1 {
                            local: 3,
                            projection: Box::default(),
                        },
                        value: ReferenceValueV1::SafeHelperCall {
                            helper: identity(),
                            parameters: vec![ReferenceScalarTypeV1::U32].into_boxed_slice(),
                            result: ReferenceScalarTypeV1::U32,
                            arguments: vec![operand(4)].into_boxed_slice(),
                            summary: Box::new(ReferenceEffectExpressionV1::Binary {
                                operation: ReferenceBinaryOpV1::Add,
                                lhs: Box::new(ReferenceEffectExpressionV1::KernelScalarArgument {
                                    argument: 0,
                                }),
                                rhs: Box::new(constant(1)),
                                checked: false,
                            }),
                        },
                    }]
                    .into_boxed_slice(),
                    terminator: ReferenceTerminatorV1::Switch {
                        discriminant: operand(1),
                        values: vec![(0, 1), (1, 1)].into_boxed_slice(),
                        otherwise: 1,
                    },
                },
                ReferenceBlockV1 {
                    block: 1,
                    assignments: Box::default(),
                    terminator: ReferenceTerminatorV1::Assert {
                        condition: operand(1),
                        expected: true,
                        success: 2,
                        bounds_check: Some(ReferenceBoundsCheckV1 {
                            index: operand(0),
                            length: operand(8),
                        }),
                    },
                },
                ReferenceBlockV1 {
                    block: 2,
                    assignments: Box::default(),
                    terminator: ReferenceTerminatorV1::Return,
                },
            ]
            .into_boxed_slice(),
            loop_summaries: vec![ReferenceLoopSummaryV2 {
                header: 0,
                latch: 1,
                exit: 2,
                exact_iterations: Some(1),
                maximum_iterations: 1,
                carried_locals: vec![3].into_boxed_slice(),
                initial_state_sha256: [9; 32],
                transition_sha256: [10; 32],
                variant_sha256: [11; 32],
            }]
            .into_boxed_slice(),
            observable_output_effects: vec![write.clone()].into_boxed_slice(),
        },
        observable_output_writes: vec![write].into_boxed_slice(),
    }
}

fn measure(
    lhs: &AuthenticatedReferenceEffectBindingV1,
    rhs: &AuthenticatedReferenceEffectBindingV1,
    inherited: usize,
) -> (bool, u64) {
    let mut work = SourceClosureWorkV1::default();
    work.charge(inherited).unwrap();
    let equal = equivalent_bindings_v1(lhs, rhs, &mut work).unwrap();
    (equal, work.validation_work_for_test())
}

fn leave_work(work: &mut SourceClosureWorkV1, remaining: u64) {
    let limit = SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::ValidationWork);
    let used = work.validation_work_for_test();
    work.charge(usize::try_from(limit - used - remaining).unwrap())
        .unwrap();
}

fn assert_changed_with_unchanged_identities(
    lhs: &AuthenticatedReferenceEffectBindingV1,
    rhs: &AuthenticatedReferenceEffectBindingV1,
) {
    assert_eq!(lhs.kernel, rhs.kernel);
    assert_eq!(lhs.reference, rhs.reference);
    assert_eq!(lhs.effect_ir_sha256, rhs.effect_ir_sha256);
    assert!(!measure(lhs, rhs, 17).0);
}

#[test]
fn full_comparison_preserves_inherited_work_and_has_exact_one_short_boundaries() {
    let lhs = binding();
    let rhs = lhs.clone();
    let (equal, cost) = measure(&lhs, &rhs, 0);
    assert!(equal);
    assert_eq!(measure(&lhs, &rhs, 17), (true, cost + 17));
    let limit = SemanticMirLimitsV1::default().limit(SemanticMirResourceV1::ValidationWork);
    for remaining in [cost, cost - 1] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(17).unwrap();
        leave_work(&mut work, remaining);
        let result = equivalent_bindings_v1(&lhs, &rhs, &mut work);
        if remaining == cost {
            assert!(result.unwrap());
            assert_eq!(work.validation_work_for_test(), limit);
        } else {
            assert!(result.unwrap_err().to_string().contains("ValidationWork"));
            // The final equality admission fails, after both censuses complete.
            // SourceClosureWork retains attempted work even on refusal.
            assert_eq!(work.validation_work_for_test(), limit + 1);
        }
    }
}

#[test]
fn repeated_comparisons_accumulate_without_resetting_the_caller() {
    let value = binding();
    let cost = measure(&value, &value, 0).1;
    let mut work = SourceClosureWorkV1::default();
    work.charge(17).unwrap();
    for repetitions in 1..=2 {
        assert!(equivalent_bindings_v1(&value, &value, &mut work).unwrap());
        assert_eq!(work.validation_work_for_test(), 17 + repetitions * cost);
    }
}

#[test]
fn changed_helper_summary_or_arguments_are_not_hidden_by_equal_hashes() {
    let original = binding();
    for change_summary in [false, true] {
        let mut changed = original.clone();
        let ReferenceValueV1::SafeHelperCall {
            summary, arguments, ..
        } = &mut changed.effect_ir.blocks[0].assignments[0].value
        else {
            panic!("fixture helper missing");
        };
        if change_summary {
            **summary = constant(99);
        } else {
            arguments[0] = operand(99);
        }
        assert_changed_with_unchanged_identities(&original, &changed);
    }
}

#[test]
fn changed_guard_payload_and_loop_carried_state_are_compared() {
    let original = binding();
    let mut guard = original.clone();
    let ReferenceGuardAtomV1::SwitchValueSet { values, .. } =
        &mut guard.effect_ir.observable_output_effects[0].guard.clauses[0].atoms[0]
    else {
        panic!("fixture guard missing");
    };
    values[1] = 77;
    assert_changed_with_unchanged_identities(&original, &guard);
    let mut loop_state = original.clone();
    loop_state.effect_ir.loop_summaries[0].carried_locals[0] = 2;
    assert_changed_with_unchanged_identities(&original, &loop_state);
}

#[test]
fn both_separately_owned_output_lists_are_compared() {
    let original = binding();
    for outer_list in [false, true] {
        let mut changed = original.clone();
        let writes = if outer_list {
            &mut changed.observable_output_writes
        } else {
            &mut changed.effect_ir.observable_output_effects
        };
        writes[0].rhs = constant(99);
        assert_changed_with_unchanged_identities(&original, &changed);
    }
}

#[test]
fn signature_arrays_and_nested_bounds_operands_are_compared() {
    let original = binding();
    let mut changed = original.clone();
    changed.signature_preimage = signature(ReferenceScalarTypeV1::U64);
    assert_changed_with_unchanged_identities(&original, &changed);
    let mut changed = original.clone();
    let ReferenceTerminatorV1::Assert {
        bounds_check: Some(bounds),
        ..
    } = &mut changed.effect_ir.blocks[1].terminator
    else {
        panic!("fixture bounds check missing");
    };
    bounds.length = operand(99);
    assert_changed_with_unchanged_identities(&original, &changed);
}

#[test]
fn string_and_signature_array_payloads_receive_full_byte_debits() {
    let original = binding();
    let baseline = measure(&original, &original, 0).1;
    let mut longer_name = original.clone();
    longer_name
        .origin
        .source_registration_mut_v1()
        .unwrap()
        .push_str(&"x".repeat(257));
    assert_eq!(measure(&longer_name, &longer_name, 0).1, baseline + 4 * 257);
    let mut more_inputs = original.clone();
    let mut kernel = more_inputs.signature_preimage.kernel_inputs().to_vec();
    let mut reference = more_inputs.signature_preimage.reference_inputs().to_vec();
    kernel.push(ReferenceSignatureInputV1::Scalar(
        ReferenceScalarTypeV1::U32,
    ));
    reference.push(ReferenceSignatureInputV1::Scalar(
        ReferenceScalarTypeV1::U32,
    ));
    more_inputs.signature_preimage = ReferenceLogicalSignaturePreimageV1::new(
        kernel.into_boxed_slice(),
        reference.into_boxed_slice(),
        ReferenceReturnShapeV1::Unit,
        SemanticExternAbiV1::Rust,
        SemanticFunctionSafetyV1::Safe,
        false,
    )
    .unwrap();
    assert_eq!(
        measure(&more_inputs, &more_inputs, 0).1,
        baseline + 8 * size_of::<ReferenceSignatureInputV1>() as u64,
    );
}

fn unary_chain(depth: usize) -> ReferenceEffectExpressionV1 {
    let mut expression = constant(1);
    for _ in 0..depth {
        expression = ReferenceEffectExpressionV1::Unary {
            operation: ReferenceUnaryOpV1::Not,
            operand: Box::new(expression),
        };
    }
    expression
}

#[test]
fn exact_expression_depth_is_admitted_but_either_overdeep_operand_refuses_before_eq() {
    let depth = fe2o3_pliron::MAX_PRODUCTION_SEMANTIC_EXPRESSION_DEPTH_V2;
    let mut exact = binding();
    exact.observable_output_writes[0].rhs = unary_chain(depth);
    assert!(measure(&exact, &exact, 17).0);
    let ordinary = binding();
    let mut overdeep = binding();
    overdeep
        .origin
        .source_registration_mut_v1()
        .unwrap()
        .push_str("_different_before_any_expression");
    overdeep.observable_output_writes[0].rhs = unary_chain(depth + 1);
    for (lhs, rhs) in [(&ordinary, &overdeep), (&overdeep, &ordinary)] {
        let mut work = SourceClosureWorkV1::default();
        work.charge(17).unwrap();
        let error = equivalent_bindings_v1(lhs, rhs, &mut work).unwrap_err();
        assert!(error.to_string().contains("resolution levels"));
        assert!(work.validation_work_for_test() > 17);
    }
}

fn balanced_expression(depth: usize) -> ReferenceEffectExpressionV1 {
    if depth == 0 {
        return constant(1);
    }
    ReferenceEffectExpressionV1::Binary {
        operation: ReferenceBinaryOpV1::BitOr,
        lhs: Box::new(balanced_expression(depth - 1)),
        rhs: Box::new(balanced_expression(depth - 1)),
        checked: false,
    }
}

#[test]
fn expression_node_limit_applies_before_derived_equality() {
    assert_eq!(MAX_REFERENCE_EXPRESSION_NODES_V1, 8_192);
    let mut exact = binding();
    exact.observable_output_writes[0].rhs = ReferenceEffectExpressionV1::Unary {
        operation: ReferenceUnaryOpV1::Not,
        operand: Box::new(balanced_expression(12)),
    };
    assert!(measure(&exact, &exact, 17).0);
    let mut over = binding();
    over.observable_output_writes[0].rhs = ReferenceEffectExpressionV1::Binary {
        operation: ReferenceBinaryOpV1::BitOr,
        lhs: Box::new(balanced_expression(12)),
        rhs: Box::new(constant(1)),
        checked: false,
    };
    let mut work = SourceClosureWorkV1::default();
    let error = equivalent_bindings_v1(&exact, &over, &mut work).unwrap_err();
    assert!(error.to_string().contains("8192 nodes"));
}
