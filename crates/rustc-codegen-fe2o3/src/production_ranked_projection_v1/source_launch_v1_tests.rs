// Source-launch adapter and row-use regressions on the real production path.

fn source_launch_test_semantic_v1(
    first_identity: u8,
    first_binding: u8,
) -> AdmittedInertSemanticMirV1 {
    source_launch_test_semantic_with_access_v1(first_identity, first_binding, false)
}

fn source_launch_test_semantic_with_access_v1(
    first_identity: u8,
    first_binding: u8,
    ranked_access: bool,
) -> AdmittedInertSemanticMirV1 {
    let unit = SemanticTypeIdV1::from_index(0);
    let scalar = SemanticTypeIdV1::from_index(1);
    let array = SemanticTypeIdV1::from_index(2);
    let mut types = vec![neutral_semantic_types_v1().into_iter().next().unwrap()];
    if ranked_access {
        types.push(neutral_semantic_types_v1().remove(NEUTRAL_ELEMENT_TYPE.index() as usize));
        types.push(SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256(bytes(236)),
            SemanticLayoutIdentityV1::from_sha256(bytes(236)),
            SemanticTypeLayoutV1::with_exact_rustc_layout(
                32,
                4,
                SemanticFieldsShapeV1::array(4, 8),
                SemanticRustcVariantsV1::Single { index: 0 },
                SemanticBackendReprV1::memory(true),
                None,
                false,
                None,
                4,
                0,
                SemanticTypeLayoutDetailsV1::None,
            )
            .unwrap(),
            SemanticTypeShapeV1::Array {
                element: scalar,
                length: 8,
            },
        ));
    }
    let mut functions = Vec::new();
    for (ordinal, identity, symbol, binding) in [
        (0_u8, first_identity, "source_launch_first", first_binding),
        (1_u8, 80_u8, "source_launch_second", 0x7a_u8),
    ] {
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256(bytes(100 + ordinal)),
            SemanticLayoutIdentityV1::from_sha256(bytes(250)),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let dimensions = SemanticWorkgroupDimensionsV1::new([64, 1, 1]).unwrap();
        let contract = SemanticKernelSourceContractV1::new(
            Some(
                SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None)
                    .unwrap(),
            ),
            None,
            None,
        )
        .unwrap();
        let mut locals = vec![local(160 + ordinal, unit, SemanticLocalRoleV1::Return)];
        let mut statements = Vec::new();
        if ranked_access {
            locals.extend([
                local(170 + ordinal, scalar, SemanticLocalRoleV1::Temporary),
                local(180 + ordinal, array, SemanticLocalRoleV1::Temporary),
            ]);
            // An ordinary private indexed write makes each original root rankable.
            statements.push(typed_assignment(
                1,
                scalar,
                SemanticRvalueKindV1::Use(typed_constant(scalar, 0, 4)),
            ));
            statements.push(statement(SemanticStatementKindV1::Assign(
                SemanticAssignmentV1::new(
                    SemanticPlaceV1::new(
                        SemanticLocalIdV1::from_index(2),
                        vec![
                            SemanticProjectionV1::new(
                                SemanticProjectionKindV1::Index(SemanticLocalIdV1::from_index(1)),
                                scalar,
                            )
                            .unwrap(),
                        ],
                        scalar,
                    )
                    .unwrap(),
                    SemanticRvalueV1::new(
                        scalar,
                        SemanticRvalueKindV1::Use(typed_constant(scalar, 7, 4)),
                    ),
                ),
            )));
        }
        let blocks = vec![block(
            90 + ordinal,
            statements,
            SemanticTerminatorKindV1::Return,
        )];
        functions.push(
            SemanticFunctionDeclV1::new(
                SemanticFunctionIdentityV1::from_sha256(bytes(identity)),
                SemanticFunctionRoleV1::KernelRoot,
                SemanticItemDefinitionIdentityV1::from_sha256(bytes(120 + ordinal)),
                SemanticMonomorphizationIdentityV1::from_sha256(bytes(130 + ordinal)),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256(bytes(140 + ordinal)),
                SemanticConstGenericArgumentsIdentityV1::from_sha256(bytes(150 + ordinal)),
                SemanticSourceProvenanceV1::unavailable(),
                abi,
                locals,
                SemanticBlockIdV1::from_index(0),
                blocks,
            )
            .unwrap()
            .with_kernel_entry(SemanticKernelEntryV1::new(
                SemanticLinkSymbolV1::new(symbol.as_bytes().to_vec()).unwrap(),
                SemanticKernelBindingIdentityV1::from_sha256(bytes(binding)),
                contract,
            )),
        );
    }
    InertSemanticMirRequestV1::new(
        SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256(bytes(250))),
        types,
        vec![],
        vec![],
        vec![],
        functions,
        vec![
            SemanticFunctionIdV1::from_index(0),
            SemanticFunctionIdV1::from_index(1),
        ],
    )
    .unwrap()
    .admit_current_production(SemanticMirLimitsV1::default())
    .unwrap()
}

fn source_launch_test_inputs_v1() -> [ProductionRankedRootInputV1; 2] {
    [
        ranked_root_input_1d("logical_first", 0xa1, 64),
        ranked_root_input_1d("logical_second", 0x7a, 64),
    ]
    .map(|mut root| {
        root.source_launch = LaunchContract::new(
            1,
            BlockSize::Exact(fe2o3_artifacts::Dimensions::new(64, 1, 1).unwrap()),
            fe2o3_artifacts::Dimensions::new(3, 1, 1).unwrap(),
            0,
            0,
        )
        .unwrap();
        root
    })
}

fn source_launch_materialized_two_root_fixture_v1() -> (
    fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    [ProductionRankedRootInputV1; 2],
) {
    let semantic = source_launch_test_semantic_with_access_v1(70, 0xa1, true);
    let owner = ProductionSemanticMirOwnerV1::try_new(
        semantic,
        fe2o3_pliron::ProductionSemanticMirLimitsV1::default(),
    )
    .unwrap();
    let ssa = ProductionSemanticSsaOwnerV1::try_new(
        owner,
        fe2o3_pliron::ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap();
    let inputs = source_launch_test_inputs_v1();
    let materialized = materialize_ranked_fixture_v1(ssa, &inputs).unwrap();
    (materialized, inputs)
}

#[test]
fn source_launch_materialized_two_roots_reject_reordering_and_duplicate_labels() {
    let (materialized, inputs) = source_launch_materialized_two_root_fixture_v1();
    let source_rows = materialized.source_launch().roots().to_vec();
    let identity = *materialized.executable().canonical().identity();
    let baseline = project_and_verify_ranked_materialized_semantic_mir_v1(
        materialized,
        &inputs,
        &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
    )
    .unwrap();
    assert_eq!(baseline.root_count(), 2);
    assert!(baseline.all_kernel_checks_are_clean());
    assert_eq!(
        baseline.materialized.executable().canonical().identity(),
        &identity
    );
    for (root, source) in baseline.roots.iter().zip(&source_rows) {
        assert_eq!(root.semantic_root, source.selected_root());
        assert_eq!(root.kernel_binding, source.kernel_binding());
        // Keep each root's indexed-write row with its authenticated roster.
        // Final private-array attachment is exercised by the assertion tests.
        assert_eq!(root.access_sources.len(), 1);
        let row = root.access_sources[0];
        assert_eq!(
            (
                row.semantic_block(),
                row.semantic_statement(),
                row.semantic_access_ordinal(),
            ),
            (0, Some(1), 0)
        );
        assert!(matches!(
            &root.verification.ordinary().expect("ordinary test root").kernel().blocks()[row.ranked_block() as usize].operations()
                [row.ranked_operation() as usize],
            ProductionRankedOperationV1::Access {
                kind: AccessKindAttr::Write,
                indices,
                ..
            } if indices.len() == 1
        ));
    }

    for (reorder, expected) in [
        (
            true,
            "a projected ranked root with a substituted kernel binding",
        ),
        (false, "duplicate typed logical roots in the ranked roster"),
    ] {
        let (materialized, mut inputs) = source_launch_materialized_two_root_fixture_v1();
        let retained_rows = materialized.source_launch().roots().to_vec();
        if reorder {
            inputs.swap(0, 1);
            assert_ne!(inputs[0].kernel_binding, retained_rows[0].kernel_binding());
        } else {
            inputs[1].logical_name = inputs[0].logical_name.clone();
            assert_eq!(inputs[0].kernel_binding, retained_rows[0].kernel_binding());
            assert_eq!(inputs[1].kernel_binding, retained_rows[1].kernel_binding());
        }
        assert_eq!(materialized.source_launch().roots(), retained_rows);
        let result = project_and_verify_ranked_materialized_semantic_mir_v1(
            materialized,
            &inputs,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
        );
        assert!(
            matches!(result, Err(ProductionRankedProjectionErrorV1::Unsupported(actual)) if actual == expected),
            "wrong post-materialization refusal for reorder={reorder}"
        );
    }
}

fn source_launch_test_ssa_owner_v1() -> ProductionSemanticSsaOwnerV1 {
    let semantic = source_launch_test_semantic_v1(70, 0xa1);
    let owner = ProductionSemanticMirOwnerV1::try_new(
        semantic,
        fe2o3_pliron::ProductionSemanticMirLimitsV1::default(),
    )
    .unwrap();
    ProductionSemanticSsaOwnerV1::try_new(
        owner,
        fe2o3_pliron::ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap()
}

#[test]
fn source_launch_production_any_and_at_most_are_exactly_incomplete() {
    for block_size in [
        BlockSize::Any,
        BlockSize::AtMost(fe2o3_artifacts::Dimensions::new(64, 1, 1).unwrap()),
    ] {
        let mut inputs = source_launch_test_inputs_v1();
        inputs[0].source_launch = LaunchContract::new(
            1,
            block_size,
            fe2o3_artifacts::Dimensions::new(3, 1, 1).unwrap(),
            0,
            0,
        )
        .unwrap();
        assert_eq!(inputs[0].source_launch.block_size(), block_size);
        assert_eq!(
            source_launch_input_v1(&inputs[0].source_launch),
            ProductionSourceLaunchInputV1::new(1, None, [3, 1, 1])
        );
        let result = project_ranked_fixture_v1(
            source_launch_test_ssa_owner_v1(),
            &inputs,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
        );
        assert!(matches!(
            result,
            Err(ProductionRankedProjectionErrorV1::Incomplete(
                "concurrency verification requires an exact authenticated LaunchContract workgroup"
            ))
        ));
    }
}

#[test]
fn source_launch_production_row_guard_rejects_substituted_grid_identity_and_root() {
    let program = neutral_ranked_program_v1();
    let semantic = program.materialized.semantic_ssa().source_semantic();
    let mut inputs = [ranked_root_input_1d("neutral_generated_hostile", 247, 64)];
    let [finite_launch, _] = source_launch_test_inputs_v1();
    inputs[0].source_launch = finite_launch.source_launch;
    let detached = inputs
        .iter()
        .map(source_launch_root_input_v1)
        .collect::<Vec<_>>();
    let roster = ProductionSourceLaunchRosterV1::try_new(semantic, &detached).unwrap();
    let effects = derive_defined_callable_empty_effect_summaries_v1(
        semantic.types(),
        semantic.functions(),
        semantic.callables(),
    )
    .unwrap();
    let selected = semantic
        .select_kernel_body_for_root_v1(semantic.roots()[0])
        .unwrap();
    let references = crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default();
    let project = |launch: &LaunchContract, row: ProductionSourceLaunchRootV1| {
        let input = ProductionRankedRootInputV1::new(
            &inputs[0].logical_name,
            inputs[0].kernel_binding,
            launch,
        );
        with_budgeted_component_facts_v1(|facts| {
            project_and_verify_ranked_root_v1(
                semantic,
                &effects,
                selected,
                &input,
                row,
                &references,
                facts,
            )
        })
    };

    let exact = project(&inputs[0].source_launch, roster.roots()[0]).unwrap();
    assert_eq!(exact.semantic_root, semantic.roots()[0]);
    assert_eq!(
        exact.semantic_root_identity,
        semantic.functions()[0].identity()
    );
    assert_eq!(exact.kernel_binding, bytes(247));
    assert_eq!(exact.source_rank, 1);
    let layouts = exact
        .verification
        .ordinary()
        .expect("ordinary test root")
        .kernel()
        .blocks()
        .iter()
        .flat_map(|block| {
            block.operations().iter().filter(|operation| {
                matches!(
                    operation,
                    ProductionRankedOperationV1::ExecutionLayout { .. }
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        layouts,
        vec![&ProductionRankedOperationV1::ExecutionLayout {
            grid_identity: u64::from_le_bytes([247; 8]),
            global_extents: [192, 1, 1],
            workgroup_extents: [64, 1, 1],
            subgroup_size: 64,
            full_physical_workgroups: true,
        }]
    );

    let changed_grid = LaunchContract::new(
        1,
        inputs[0].source_launch.block_size(),
        fe2o3_artifacts::Dimensions::new(4, 1, 1).unwrap(),
        0,
        0,
    )
    .unwrap();
    assert_eq!(changed_grid.rank(), inputs[0].source_launch.rank());
    assert_eq!(
        changed_grid.block_size(),
        inputs[0].source_launch.block_size()
    );
    assert_ne!(changed_grid.max_grid(), inputs[0].source_launch.max_grid());

    let foreign_semantic = source_launch_test_semantic_v1(71, 247);
    let foreign_inputs = [
        ProductionSourceLaunchRootInputV1::new(
            "foreign_first",
            bytes(247),
            source_launch_input_v1(&inputs[0].source_launch),
        ),
        ProductionSourceLaunchRootInputV1::new(
            "foreign_second",
            bytes(0x7a),
            source_launch_input_v1(&inputs[0].source_launch),
        ),
    ];
    let foreign_roster =
        ProductionSourceLaunchRosterV1::try_new(&foreign_semantic, &foreign_inputs).unwrap();
    let exact_row = roster.roots()[0];
    let foreign_row = foreign_roster.roots()[0];
    assert_ne!(roster.semantic_sha256(), foreign_roster.semantic_sha256());
    assert_eq!(foreign_row.selected_root(), exact_row.selected_root());
    assert_eq!(foreign_row.kernel_binding(), exact_row.kernel_binding());
    assert_eq!(foreign_row.source_launch(), exact_row.source_launch());
    assert_eq!(foreign_row.layout(), exact_row.layout());
    assert_ne!(
        foreign_row.semantic_root_identity(),
        exact_row.semantic_root_identity()
    );

    for (launch, row) in [
        (&changed_grid, exact_row),
        (&inputs[0].source_launch, foreign_row),
        (&inputs[0].source_launch, foreign_roster.roots()[1]),
    ] {
        let result = project(launch, row);
        assert!(matches!(
            result,
            Err(ProductionRankedProjectionErrorV1::Unsupported(
                "source launch roster root changed before ranked projection"
            ))
        ));
    }
    assert_eq!(roster.roots()[0], exact_row);
    let retry = project(&inputs[0].source_launch, exact_row).unwrap();
    assert_eq!(retry.ranked_ir, exact.ranked_ir);
}

#[test]
fn source_launch_production_accepts_empty_roots_and_rejects_later_geometry() {
    let inputs = source_launch_test_inputs_v1();
    let individual = project_ranked_fixture_v1(
        source_launch_test_ssa_owner_v1(),
        &inputs,
        &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
    );
    let individual = individual.unwrap();
    assert_eq!(individual.roots.len(), 2);
    assert_eq!(
        individual
            .materialized
            .empty_effect_helpers()
            .iter()
            .count(),
        0
    );
    for (ordinal, root) in individual.roots.iter().enumerate() {
        assert!(root.all_kernel_checks_are_clean());
        assert_eq!(
            root.semantic_root,
            SemanticFunctionIdV1::from_index(ordinal as u32)
        );
        assert_eq!(root.kernel_binding, inputs[ordinal].kernel_binding);
        assert_eq!(root.source_rank, 1);
        assert!(root.access_sources.is_empty());
        assert!(root.executable_effect_sources.is_empty());
        let operations = root
            .verification
            .ordinary()
            .expect("ordinary test root")
            .kernel()
            .blocks()
            .iter()
            .flat_map(|block| block.operations())
            .collect::<Vec<_>>();
        assert_eq!(
            operations
                .iter()
                .filter(|operation| matches!(
                    operation,
                    ProductionRankedOperationV1::ExecutionLayout { .. }
                ))
                .count(),
            1,
        );
        assert!(operations.iter().any(|operation| matches!(operation,
            ProductionRankedOperationV1::ExecutionLayout {
                grid_identity,
                global_extents: [192, 1, 1],
                workgroup_extents: [64, 1, 1],
                subgroup_size: 64,
                full_physical_workgroups: true,
            } if *grid_identity == u64::from_le_bytes(inputs[ordinal].kernel_binding[..8].try_into().unwrap())
        )));
        assert!(!operations.iter().any(|operation| matches!(
            operation,
            ProductionRankedOperationV1::InvocationIndex { .. }
                | ProductionRankedOperationV1::Access { .. }
                | ProductionRankedOperationV1::AtomicAccess { .. }
                | ProductionRankedOperationV1::AllocationEffect { .. }
        )));
    }

    let mut conflicting = source_launch_test_inputs_v1();
    conflicting[1].source_launch = LaunchContract::new(
        1,
        BlockSize::Exact(fe2o3_artifacts::Dimensions::new(128, 1, 1).unwrap()),
        fe2o3_artifacts::Dimensions::new(3, 1, 1).unwrap(),
        0,
        0,
    )
    .unwrap();
    assert_eq!(conflicting[0].source_launch, inputs[0].source_launch);
    let combined = project_ranked_fixture_v1(
        source_launch_test_ssa_owner_v1(),
        &conflicting,
        &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
    );
    assert!(matches!(
        combined,
        Err(ProductionRankedProjectionErrorV1::Unsupported(
            "authenticated LaunchContract workgroup disagrees with semantic source workgroup"
        ))
    ));
}

fn source_launch_materialized_neutral_fixture_v1() -> (
    fe2o3_lower_mir_kernel::ProductionPreRankedKirOwnerV1,
    Vec<ProductionRankedRootInputV1>,
) {
    let mut inputs = vec![ranked_root_input_1d("neutral_generated_hostile", 247, 64)];
    inputs[0].source_launch = LaunchContract::new(
        1,
        BlockSize::Exact(fe2o3_artifacts::Dimensions::new(64, 1, 1).unwrap()),
        fe2o3_artifacts::Dimensions::new(3, 1, 1).unwrap(),
        0,
        0,
    )
    .unwrap();
    let source = neutral_ranked_source_for_operation_v1(
        SemanticCompilerIntrinsicOperationV1::NeutralWorkgroupReduceSum {
            context: NEUTRAL_CONTEXT_TYPE,
            dynamic_lds: NEUTRAL_DYNAMIC_LDS_TYPE,
            element_storage: NEUTRAL_ELEMENT_TYPE,
            element: NEUTRAL_ELEMENT_TYPE,
        },
        64,
    );
    let materialized = materialize_ranked_fixture_v1(source, &inputs).unwrap();
    (materialized, inputs)
}

#[test]
fn source_launch_materialized_baseline_and_unique_diagnostic_rename_preserve_execution() {
    for rename in [false, true] {
        let (materialized, mut inputs) = source_launch_materialized_neutral_fixture_v1();
        let identity = *materialized.executable().canonical().identity();
        let functions = materialized.executable().module().functions.as_ptr();
        let source_rows = materialized.source_launch().roots().to_vec();
        if rename {
            // Logical labels are not export symbols or executable authority.
            inputs[0].logical_name = "renamed_diagnostic_label".to_owned();
        }
        let program = project_and_verify_ranked_materialized_semantic_mir_v1(
            materialized,
            &inputs,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
        )
        .unwrap();
        assert_eq!(program.root_count(), 1);
        assert!(program.all_kernel_checks_are_clean());
        assert!(!program.grants_artifact_or_launch_authority());
        assert_eq!(program.roots[0].logical_name, inputs[0].logical_name);
        assert_eq!(
            program.roots[0].kernel_binding,
            source_rows[0].kernel_binding()
        );
        assert_eq!(
            program.roots[0].semantic_root_identity,
            source_rows[0].semantic_root_identity()
        );
        assert_eq!(program.materialized.source_launch().roots(), source_rows);
        assert_eq!(
            program.materialized.executable().canonical().identity(),
            &identity
        );
        assert_eq!(
            program
                .materialized
                .executable()
                .module()
                .functions
                .as_ptr(),
            functions
        );
    }
}

#[test]
fn source_launch_materialized_entry_rejects_changed_geometry_binding_and_roster() {
    for (case, expected) in [
        (
            "grid",
            "source launch roster root changed before ranked projection",
        ),
        (
            "workgroup",
            "source launch roster root changed before ranked projection",
        ),
        (
            "binding",
            "a projected ranked root with a substituted kernel binding",
        ),
        ("missing", "an incomplete typed/semantic ranked root roster"),
        ("extra", "an incomplete typed/semantic ranked root roster"),
    ] {
        let (materialized, mut inputs) = source_launch_materialized_neutral_fixture_v1();
        let source_row = materialized.source_launch().roots()[0];
        match case {
            "grid" | "workgroup" => {
                inputs[0].source_launch = LaunchContract::new(
                    1,
                    BlockSize::Exact(
                        fe2o3_artifacts::Dimensions::new(
                            if case == "workgroup" { 128 } else { 64 },
                            1,
                            1,
                        )
                        .unwrap(),
                    ),
                    fe2o3_artifacts::Dimensions::new(if case == "grid" { 4 } else { 3 }, 1, 1)
                        .unwrap(),
                    0,
                    0,
                )
                .unwrap();
                assert_ne!(
                    source_launch_input_v1(&inputs[0].source_launch),
                    source_row.source_launch()
                );
            }
            "binding" => {
                inputs[0].kernel_binding[0] ^= 1;
                assert_ne!(inputs[0].kernel_binding, source_row.kernel_binding());
            }
            "missing" => inputs.clear(),
            "extra" => inputs.push(ranked_root_input_1d("extra_root", 248, 64)),
            _ => unreachable!(),
        }
        assert_eq!(materialized.source_launch().roots(), &[source_row]);
        let result = project_and_verify_ranked_materialized_semantic_mir_v1(
            materialized,
            &inputs,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
        );
        assert!(
            matches!(result, Err(ProductionRankedProjectionErrorV1::Unsupported(actual)) if actual == expected),
            "wrong post-materialization refusal for {case}"
        );
    }
}

// The geometry consumed by the pipeline domain proof comes through this same
// source-owned materialization boundary, rather than a launch_extent hint.
#[test]
fn full_writer_geometry_256_is_source_owned_before_ranked_consumption() {
    for changed in [false, true] {
        let mut inputs = vec![ranked_root_input_1d("neutral_generated_hostile", 247, 256)];
        inputs[0].source_launch = LaunchContract::new(
            1,
            BlockSize::Exact(fe2o3_artifacts::Dimensions::new(256, 1, 1).unwrap()),
            fe2o3_artifacts::Dimensions::new(4, 1, 1).unwrap(),
            0,
            0,
        )
        .unwrap();
        let source = neutral_ranked_source_for_operation_v1(
            SemanticCompilerIntrinsicOperationV1::NeutralWorkgroupReduceSum {
                context: NEUTRAL_CONTEXT_TYPE,
                dynamic_lds: NEUTRAL_DYNAMIC_LDS_TYPE,
                element_storage: NEUTRAL_ELEMENT_TYPE,
                element: NEUTRAL_ELEMENT_TYPE,
            },
            256,
        );
        let materialized = materialize_ranked_fixture_v1(source, &inputs).unwrap();
        let original = materialized.source_launch().roots().to_vec();
        if changed {
            inputs[0].source_launch = LaunchContract::new(
                1,
                BlockSize::Exact(fe2o3_artifacts::Dimensions::new(128, 1, 1).unwrap()),
                fe2o3_artifacts::Dimensions::new(4, 1, 1).unwrap(),
                0,
                0,
            )
            .unwrap();
        }
        let result = project_and_verify_ranked_materialized_semantic_mir_v1(
            materialized,
            &inputs,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1::default(),
        );
        if changed {
            assert!(matches!(
                result,
                Err(ProductionRankedProjectionErrorV1::Unsupported(
                    "source launch roster root changed before ranked projection"
                ))
            ));
        } else {
            let program = result.unwrap();
            assert!(program.all_kernel_checks_are_clean());
            assert_eq!(program.materialized.source_launch().roots(), original);
            assert!(!program.grants_artifact_or_launch_authority());
        }
    }
}
