// Focused inert semantic fixtures for the production pipeline payload collector.
// Lifecycle, publication, convergence and indexed effects remain separate checks.
fn payload_fixture(
    elements: &[SemanticTypeIdV1],
    values: &[SemanticOperandV1],
    read_element: Option<SemanticTypeIdV1>,
) -> (SemanticFunctionDeclV1, Vec<SemanticCallableDeclV1>) {
    let mut callables = Vec::new();
    let mut blocks = Vec::new();
    for (index, (&element, value)) in elements.iter().zip(values).enumerate() {
        callables.push(compiler_intrinsic_callable(
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineWrite {
                pipeline: ARRAY_TYPE,
                element,
            },
        ));
        blocks.push(block(
            180 + index as u8,
            vec![],
            SemanticTerminatorKindV1::Call(
                SemanticDirectCallV1::new_callable(
                    SemanticCallableIdV1::from_index(index as u32),
                    vec![
                        typed_operand(1, ARRAY_TYPE),
                        typed_constant(U64_TYPE, 0, 8),
                        typed_constant(U64_TYPE, 255, 8),
                        value.clone(),
                    ],
                    Some(SemanticCallDestinationV1::new(
                        typed_place(0, SCALAR_TYPE),
                        cfg_edge(SemanticEdgeRoleV1::CallReturn, index as u32 + 1),
                    )),
                    SemanticUnwindActionV1::Unreachable,
                )
                .unwrap(),
            ),
        ));
    }
    if let Some(element) = read_element {
        let index = callables.len() as u32;
        callables.push(compiler_intrinsic_callable(
            SemanticCompilerIntrinsicOperationV1::WorkgroupPipelineRead {
                pipeline: ARRAY_TYPE,
                element,
            },
        ));
        blocks.push(block(
            190,
            vec![],
            SemanticTerminatorKindV1::Call(
                SemanticDirectCallV1::new_callable(
                    SemanticCallableIdV1::from_index(index),
                    vec![
                        typed_operand(1, ARRAY_TYPE),
                        typed_constant(U64_TYPE, 0, 8),
                        typed_constant(U64_TYPE, 0, 8),
                    ],
                    Some(SemanticCallDestinationV1::new(
                        typed_place(3, element),
                        cfg_edge(SemanticEdgeRoleV1::CallReturn, index + 1),
                    )),
                    SemanticUnwindActionV1::Unreachable,
                )
                .unwrap(),
            ),
        ));
    }
    blocks.push(block(191, vec![], SemanticTerminatorKindV1::Return));
    (
        projection_function_with_locals(
            blocks,
            vec![
                local(170, SCALAR_TYPE, SemanticLocalRoleV1::Return),
                local(171, ARRAY_TYPE, SemanticLocalRoleV1::Temporary),
                local(172, U64_TYPE, SemanticLocalRoleV1::Temporary),
                local(
                    173,
                    read_element.unwrap_or(U64_TYPE),
                    SemanticLocalRoleV1::Temporary,
                ),
            ],
        ),
        callables,
    )
}

fn collect_payload_fixture(
    function: &SemanticFunctionDeclV1,
    callables: &[SemanticCallableDeclV1],
    owners: &[Option<usize>],
    state: ProjectedCapabilityStateV1,
) -> Result<HashMap<usize, ProjectedMfmaOperandV1>, ProductionRankedProjectionErrorV1> {
    let types = assertion_proof_types();
    let dominance = SemanticEnumPayloadDominanceV1::analyze(function, &types).unwrap();
    let entries = vec![Some(state); function.blocks().len()];
    collect_workgroup_pipeline_payloads_v1(
        callables, &types, function, &dominance, owners, &entries, &mut 0,
    )
}

#[test]
fn workgroup_pipeline_payload_accepts_raw_u64_values_without_tensor_authority() {
    let (function, callables) = payload_fixture(
        &[U64_TYPE, U64_TYPE],
        &[
            typed_constant(U64_TYPE, 0xffff_8000_0000_ffff, 8),
            typed_operand(2, U64_TYPE),
        ],
        Some(U64_TYPE),
    );
    let payloads = collect_payload_fixture(
        &function,
        &callables,
        &[None, Some(1), None, None],
        HashMap::new(),
    )
    .unwrap();
    // Raw data never becomes tensor issuance authority.
    assert!(payloads.is_empty());
    let mut state = HashMap::new();
    transfer_capability_terminator_v1(
        &callables,
        &function,
        2,
        &mut state,
        &[None; 4],
        &[None; 4],
        &[None, Some(1), None, None],
        &payloads,
        true,
    )
    .unwrap();
    assert_eq!(state.get(&3), Some(&ProjectedCapabilityValueV1::Invalid));
}

#[test]
fn workgroup_pipeline_payload_rejects_wrong_write_type_and_mixed_same_width_types() {
    let owners = [None, Some(1), None, None];
    let (function, callables) =
        payload_fixture(&[U64_TYPE], &[typed_constant(F64_TYPE, 0, 8)], None);
    assert!(matches!(
        collect_payload_fixture(&function, &callables, &owners, HashMap::new()),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "a workgroup pipeline write payload changed semantic type"
        ))
    ));
    for read in [None, Some(F64_TYPE)] {
        let (elements, values) = if read.is_some() {
            (vec![U64_TYPE], vec![typed_constant(U64_TYPE, 0, 8)])
        } else {
            (
                vec![U64_TYPE, F64_TYPE],
                vec![
                    typed_constant(U64_TYPE, 0, 8),
                    typed_constant(F64_TYPE, 0, 8),
                ],
            )
        };
        let (function, callables) = payload_fixture(&elements, &values, read);
        assert!(matches!(
            collect_payload_fixture(&function, &callables, &owners, HashMap::new()),
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "one workgroup pipeline is used with inconsistent payload types"
            ))
        ));
    }
}

#[test]
fn workgroup_pipeline_payload_rejects_missing_or_changed_pipeline_owner() {
    let (function, callables) =
        payload_fixture(&[U64_TYPE], &[typed_constant(U64_TYPE, 0, 8)], None);
    assert!(matches!(
        collect_payload_fixture(&function, &callables, &[None; 4], HashMap::new()),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "a workgroup pipeline access lacks one compiler-owned origin"
        ))
    ));
    assert!(matches!(
        collect_payload_fixture(
            &function,
            &callables,
            &[None, Some(2), None, None],
            HashMap::new()
        ),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "a workgroup pipeline access changed its compiler-owned pipeline type"
        ))
    ));
}

fn pipeline_fragment_operand() -> ProjectedMfmaOperandV1 {
    ProjectedMfmaOperandV1 {
        contract: mfma_operand_contract(SemanticMfmaOperandRoleV1::A),
        storage_layout: SemanticMfmaStorageLayoutV1::RowMajor,
        lane_root: 20,
        allocation: tensor_test_allocation(),
    }
}

#[test]
fn workgroup_pipeline_payload_preserves_fragment_origin_and_rejects_mixed_lane_origins() {
    let (function, callables) = payload_fixture(
        &[ARRAY_TYPE, ARRAY_TYPE],
        &[typed_operand(2, ARRAY_TYPE), typed_operand(3, ARRAY_TYPE)],
        None,
    );
    let payload = pipeline_fragment_operand();
    let origin =
        |payload| ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(payload));
    let owners = [None, Some(1), None, None];
    let state = HashMap::from([(2, origin(payload)), (3, origin(payload))]);
    assert_eq!(
        collect_payload_fixture(&function, &callables, &owners, state).unwrap(),
        HashMap::from([(1, payload)]),
    );
    let state = HashMap::from([
        (2, origin(payload)),
        (
            3,
            origin(ProjectedMfmaOperandV1 {
                lane_root: 21,
                ..payload
            }),
        ),
    ]);
    assert!(matches!(
        collect_payload_fixture(&function, &callables, &owners, state),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "one workgroup pipeline receives incompatible typed payload layouts"
        ))
    ));
}

#[test]
fn workgroup_pipeline_payload_does_not_accept_aggregate_or_forge_scalar_capabilities() {
    let (function, callables) =
        payload_fixture(&[ARRAY_TYPE], &[typed_operand(2, ARRAY_TYPE)], None);
    let owners = [None, Some(1), None, None];
    assert!(matches!(
        collect_payload_fixture(&function, &callables, &owners, HashMap::new()),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "a workgroup pipeline write lacks one authenticated typed payload"
        ))
    ));
    let (function, callables) = payload_fixture(&[U64_TYPE], &[typed_operand(2, U64_TYPE)], None);
    let state = HashMap::from([(
        2,
        ProjectedCapabilityValueV1::Known(ProjectedCapabilityOriginV1::Operand(
            pipeline_fragment_operand(),
        )),
    )]);
    assert!(matches!(
        collect_payload_fixture(&function, &callables, &owners, state),
        Err(ProductionRankedProjectionErrorV1::Incomplete(
            "a scalar workgroup pipeline payload carries an incompatible capability origin"
        ))
    ));
}

#[test]
fn workgroup_pipeline_payload_scalar_classification_excludes_reference_and_validity_types() {
    let types = optional_selector_types();
    for element in [U64_TYPE, U8_TYPE, U16_TYPE, I32_TYPE, F32_TYPE, F64_TYPE] {
        assert!(plain_workgroup_pipeline_scalar_v1(&types, element));
    }
    for element in [
        POINTER_TYPE,
        ARRAY_TYPE,
        BOOL_TYPE,
        CHAR_TYPE,
        U128_TYPE,
        VALIDITY_U32_TYPE,
    ] {
        assert!(!plain_workgroup_pipeline_scalar_v1(&types, element));
    }
}

#[test]
fn workgroup_pipeline_payload_final_filter_preserves_exact_work_limit() {
    let (function, callables) = payload_fixture(
        &[U64_TYPE],
        &[typed_constant(U64_TYPE, u64::MAX as u128, 8)],
        Some(U64_TYPE),
    );
    let types = assertion_proof_types();
    let dominance = SemanticEnumPayloadDominanceV1::analyze(&function, &types).unwrap();
    let entries = vec![Some(HashMap::new()); function.blocks().len()];
    let owners = [None, Some(1), None, None];
    // One write, one read and one final owner-filter visit.
    let mut work = MAX_PROJECTED_CAPABILITY_DATAFLOW_WORK_V1 - 3;
    assert!(
        collect_workgroup_pipeline_payloads_v1(
            &callables, &types, &function, &dominance, &owners, &entries, &mut work,
        )
        .unwrap()
        .is_empty()
    );
    assert_eq!(work, MAX_PROJECTED_CAPABILITY_DATAFLOW_WORK_V1);
    let mut work = MAX_PROJECTED_CAPABILITY_DATAFLOW_WORK_V1 - 2;
    assert!(matches!(
        collect_workgroup_pipeline_payloads_v1(
            &callables, &types, &function, &dominance, &owners, &entries, &mut work,
        ),
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "capability dataflow exceeds the charged projection limit"
        ))
    ));
}
