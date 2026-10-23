use super::*;

fn admit_functions(
    types: Vec<SemanticTypeDeclV1>,
    functions: Vec<SemanticFunctionDeclV1>,
    roots: Vec<u32>,
    launches: &[crate::ProductionSourceLaunchRootInputV1],
) -> ProductionPreRankedKirOwnerV1 {
    try_admit_functions(types, functions, roots, launches).unwrap()
}

fn try_admit_functions(
    types: Vec<SemanticTypeDeclV1>,
    functions: Vec<SemanticFunctionDeclV1>,
    roots: Vec<u32>,
    launches: &[crate::ProductionSourceLaunchRootInputV1],
) -> Result<ProductionPreRankedKirOwnerV1, ProductionPreRankedKirErrorV1> {
    let admitted = InertSemanticMirRequestV1::new(
        SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256([250; 32])),
        types,
        vec![],
        vec![],
        vec![],
        functions,
        roots
            .into_iter()
            .map(SemanticFunctionIdV1::from_index)
            .collect(),
    )
    .unwrap()
    .admit_current_production(SemanticMirLimitsV1::default())
    .unwrap();
    let source =
        ProductionSemanticMirOwnerV1::try_new(admitted, ProductionSemanticMirLimitsV1::default())
            .unwrap();
    let ssa =
        ProductionSemanticSsaOwnerV1::try_new(source, ProductionSemanticSsaLimitsV1::default())
            .unwrap();
    let launch =
        crate::ProductionSourceLaunchRosterV1::try_new(ssa.source_semantic(), launches).unwrap();
    let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
    let mut budget = CanonicalKernelIrVerificationResourceBudgetV1::new(&mut work, STORAGE);
    ProductionPreRankedKirOwnerV1::try_materialize_with_budget(
        ssa,
        launch,
        ProductionSemanticKirLimitsV1::default(),
        &mut budget,
    )
}

fn no_argument_abi(tag: u8, root: bool) -> SemanticFunctionAbiV1 {
    SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([tag; 32]),
        SemanticLayoutIdentityV1::from_sha256([250; 32]),
        if root {
            SemanticCanonAbiV1::GpuKernel
        } else {
            SemanticCanonAbiV1::Rust
        },
        if root {
            SemanticExternAbiV1::GpuKernel
        } else {
            SemanticExternAbiV1::Rust
        },
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(UNIT, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap()
    .with_source_argument_ownership(vec![])
    .unwrap()
}

fn multiroot_array_owner(
    retained_in_helper: bool,
) -> Result<ProductionPreRankedKirOwnerV1, ProductionPreRankedKirErrorV1> {
    let seed = array_owner(ArrayCase::Write { sparse: false });
    let semantic = seed.semantic_ssa().source_semantic();
    let original = &semantic.functions()[0];
    let mut functions = Vec::new();
    let mut launches = Vec::new();
    for ordinal in 0..4u32 {
        let root = ordinal == 1 || ordinal == 3;
        let tag = 150 + ordinal as u8;
        let blocks = if ordinal == 0 {
            if retained_in_helper {
                original.blocks().to_vec()
            } else {
                vec![block(
                    tag,
                    vec![original.blocks()[0].statements()[0].clone()],
                    SemanticTerminatorKindV1::Return,
                )]
            }
        } else {
            let callee = if ordinal == 1 { 2 } else { 0 };
            vec![
                block(
                    tag,
                    if root && !retained_in_helper {
                        original.blocks()[0].statements().to_vec()
                    } else {
                        vec![]
                    },
                    SemanticTerminatorKindV1::Call(
                        SemanticDirectCallV1::new(
                            SemanticFunctionIdV1::from_index(callee),
                            vec![],
                            Some(SemanticCallDestinationV1::new(
                                place(0, UNIT),
                                edge(SemanticEdgeRoleV1::CallReturn, 1),
                            )),
                            SemanticUnwindActionV1::Unreachable,
                        )
                        .unwrap(),
                    ),
                ),
                block(tag + 10, vec![], SemanticTerminatorKindV1::Return),
            ]
        };
        let mut function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([tag; 32]),
            if root {
                SemanticFunctionRoleV1::KernelRoot
            } else {
                SemanticFunctionRoleV1::InternalHelper
            },
            SemanticItemDefinitionIdentityV1::from_sha256([tag; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([tag; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([tag; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([tag; 32]),
            original.source(),
            no_argument_abi(tag, root),
            original.locals().to_vec(),
            SemanticBlockIdV1::from_index(0),
            blocks,
        )
        .unwrap();
        if root {
            let dimensions = SemanticWorkgroupDimensionsV1::new([64, 1, 1]).unwrap();
            let symbol = format!("private_array_root_{ordinal}");
            function = function.with_kernel_entry(SemanticKernelEntryV1::new(
                SemanticLinkSymbolV1::new(symbol.into_bytes()).unwrap(),
                SemanticKernelBindingIdentityV1::from_sha256([tag; 32]),
                SemanticKernelSourceContractV1::new(
                    Some(
                        SemanticKernelLaunchBoundsV1::new(Some(dimensions), Some(dimensions), None)
                            .unwrap(),
                    ),
                    None,
                    None,
                )
                .unwrap(),
            ));
            launches.push(crate::ProductionSourceLaunchRootInputV1::new(
                if ordinal == 1 {
                    "array_one"
                } else {
                    "array_three"
                },
                [tag; 32],
                crate::ProductionSourceLaunchInputV1::new(1, Some([64, 1, 1]), [1, 1, 1]),
            ));
        }
        functions.push(function);
    }
    try_admit_functions(semantic.types().to_vec(), functions, vec![1, 3], &launches)
}

#[test]
fn private_array_multiroot_retained_helper_preserves_the_existing_purity_rejection() {
    assert!(matches!(
        multiroot_array_owner(true),
        Err(ProductionPreRankedKirErrorV1::Lowering(
            ProductionSemanticKirErrorV1::HelperEffectsUnavailable { function: 0, .. }
        ))
    ));
}

#[test]
fn private_array_multiroot_root_storage_preserves_axes_and_pure_helper_dedup() {
    let owner = multiroot_array_owner(false).unwrap();
    let rows = &owner.correspondence.private_arrays;
    // Each array-bearing root emits7 array operations plus1 Call. The pure
    // helper0 emits1 constant, and helper2 emits1 Call. Per-root totals10/9
    // merge to18 operations, while the recorded array-instance subset is16.
    // Array-bearing helper dedup remains unsupported, as checked above.
    assert_eq!(
        (
            rows.recorded_instance_operations,
            rows.instances.len(),
            rows.slots.len(),
            rows.effects.len()
        ),
        (16, 2, 2, 2)
    );
    let module = owner.executable().module();
    let physical_operations: usize = module
        .functions
        .iter()
        .filter_map(|f| f.body.as_ref())
        .flat_map(|b| &b.blocks)
        .map(|b| b.operations.len())
        .sum();
    assert_eq!(physical_operations, 18);
    assert_eq!(
        rows.instances
            .iter()
            .map(|i| (i.owner.index(), i.function.index()))
            .collect::<Vec<_>>(),
        vec![(1, 1), (3, 3)]
    );
    assert_eq!(
        (
            rows.instances[0].module_function_ordinal,
            rows.instances[1].module_function_ordinal
        ),
        (0, 1)
    );
    assert_ne!(
        rows.instances[0].lowered_function_ordinal,
        rows.instances[1].lowered_function_ordinal
    );
    for instance in &rows.instances {
        let entry = &owner.correspondence.lowered_functions[instance.lowered_function_ordinal];
        assert_eq!(
            (entry.correspondence_owner, entry.semantic_function),
            (instance.owner, instance.function)
        );
        let slot = &rows.slots[instance.slot_start];
        let effect = &rows.effects[instance.effect_start];
        let body = module.functions[instance.module_function_ordinal]
            .body
            .as_ref()
            .unwrap();
        let mut work = PrivateArrayRecorderBudgetV1::new(1, 10_000).unwrap();
        assert!(matches!(
            private_array_exact_relation_v1(
                owner.semantic_ssa().source_semantic().types(),
                &owner.semantic_ssa().source_semantic().functions()
                    [instance.function.index() as usize],
                body,
                instance.owner,
                instance.function,
                slot,
                effect,
                10_000,
                &mut work
            ),
            Ok(0)
        ));
    }
    let helper_instances = owner
        .correspondence
        .lowered_functions
        .iter()
        .filter(|entry| entry.semantic_function.index() == 0)
        .collect::<Vec<_>>();
    assert_eq!(helper_instances.len(), 2);
    assert_eq!(
        helper_instances
            .iter()
            .map(|entry| entry.correspondence_owner.index())
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(
        helper_instances[0].kernel_ir_function,
        helper_instances[1].kernel_ir_function
    );
    let physical_helpers = module
        .functions
        .iter()
        .enumerate()
        .filter(|(_, function)| function.id == helper_instances[0].kernel_ir_function)
        .collect::<Vec<_>>();
    assert_eq!(physical_helpers.len(), 1);
    assert!(physical_helpers[0].0 >= 2);
    assert_eq!(
        physical_helpers[0]
            .1
            .body
            .as_ref()
            .unwrap()
            .blocks
            .iter()
            .map(|block| block.operations.len())
            .sum::<usize>(),
        1
    );
    assert_eq!(
        owner.correspondence.lowered_functions[0]
            .correspondence_owner
            .index(),
        1
    );
    assert_eq!(
        owner.correspondence.lowered_functions[1]
            .correspondence_owner
            .index(),
        3
    );
    owner.semantic_ssa().verify_replay().unwrap();
}

fn actual_lowering<'source, 'service, 'work>(
    owner: &'source ProductionPreRankedKirOwnerV1,
    limit: usize,
    budget: &'service mut ArgumentBudgetV1<'work>,
) -> SemanticFunctionLoweringV1<'source, 'service> {
    let source = owner.semantic_ssa().source_semantic();
    SemanticFunctionLoweringV1::new(
        source.types(),
        source.callables(),
        &source.functions()[0],
        SemanticParameterBindingsV1 {
            declarations: &[],
            values: &[],
            types: &[],
            local_bindings: None,
        },
        None,
        Some([64, 1, 1]),
        BTreeSet::new(),
        1,
        false,
        limit,
        budget,
    )
    .unwrap()
}

fn recorder_work(lowering: &SemanticFunctionLoweringV1<'_, '_>) -> usize {
    let PrivateArrayRecorderWorkV1::Owned(lazy) = &lowering.private_arrays.work else {
        panic!("test constructor owns one meter");
    };
    lazy.active.as_ref().unwrap().work.work()
}

#[test]
fn private_array_actual_prologue_reserves_slot_before_count_and_alloca() {
    let owner = array_owner(ArrayCase::Write { sparse: false });
    // This four-local/two-assignment source needs only13 inherited retained-
    // initialization work units. Both old constructor gates remain active.
    for (limit, succeeds, accepted) in [(87, false, 86), (155, true, 155)] {
        let mut emission_work = CanonicalKernelIrWorkBudgetV1::new(WORK);
        let mut emission_budget = ArgumentBudgetV1::new(&mut emission_work, STORAGE);
        let mut lowering = actual_lowering(&owner, limit, &mut emission_budget);
        assert_eq!(recorder_work(&lowering), 0);
        let original_value = lowering.next_value;
        let mut target = BasicBlock::new(BlockId(0));
        let result = lowering.begin_block(SemanticBlockIdV1::from_index(0), &mut target);
        if succeeds {
            assert!(result.is_ok());
            assert_eq!(target.operations.len(), 2);
            assert!(matches!(
                target.operations[0].kind,
                OperationKind::Constant(Constant::Index(8))
            ));
            assert!(matches!(
                target.operations[1].kind,
                OperationKind::Alloca { count: Some(_), .. }
            ));
            assert_eq!(
                (
                    lowering.private_arrays.slots.rows.len(),
                    lowering.private_arrays.definitions.rows.len()
                ),
                (1, 1)
            );
        } else {
            assert!(matches!(
                result,
                Err(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::AnalysisWork,
                    actual: 88,
                    limit: 87,
                })
            ));
            assert!(target.operations.is_empty());
            assert_eq!(
                (lowering.emitted_operations, lowering.next_value),
                (0, original_value)
            );
            assert_eq!(
                (
                    lowering.private_arrays.slots.rows.len(),
                    lowering.private_arrays.slots.logical_capacity,
                    lowering.private_arrays.slots.admitted_end,
                    lowering.private_arrays.definitions.rows.len()
                ),
                (0, 0, 0, 0)
            );
            assert!(matches!(
                lowering.private_arrays.work.charge_private_array_work(0),
                Err(ProductionSemanticKirErrorV1::ResourceLimit {
                    actual: 88,
                    limit: 87,
                    ..
                })
            ));
        }
        assert_eq!(recorder_work(&lowering), accepted);
    }
}

fn checked_owner() -> ProductionPreRankedKirOwnerV1 {
    let seed = array_owner(ArrayCase::Write { sparse: false });
    let semantic = seed.semantic_ssa().source_semantic();
    let source = &semantic.functions()[0];
    let boolean = SemanticTypeIdV1::from_index(3);
    let pair = SemanticTypeIdV1::from_index(4);
    let mut declarations = semantic.types().to_vec();
    let boolean_declaration = types()[1].clone();
    declarations.push(
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([220; 32]),
            boolean_declaration.layout_identity(),
            boolean_declaration.layout().clone(),
            boolean_declaration.shape().clone(),
        )
        .with_rustc_abi_properties(boolean_declaration.abi_properties())
        .with_rust_type_kind(boolean_declaration.rust_type_kind()),
    );
    declarations.push(SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([230; 32]),
        SemanticLayoutIdentityV1::from_sha256([230; 32]),
        SemanticTypeLayoutV1::aggregate(
            Some(8),
            4,
            SemanticAggregateLayoutV1::new(vec![0, 4], vec![SemanticPaddingV1::new(5, 3).unwrap()])
                .unwrap(),
        )
        .unwrap(),
        SemanticTypeShapeV1::Tuple(
            SemanticAggregateTypeV1::new(vec![ARRAY_SCALAR, boolean]).unwrap(),
        ),
    ));
    let mut locals = source.locals().to_vec();
    locals.push(SemanticLocalDeclV1::new(
        SemanticLocalIdentityV1::from_sha256([231; 32]),
        pair,
        SemanticLocalRoleV1::Temporary,
        source.source(),
    ));
    let mut statements = source.blocks()[0].statements().to_vec();
    statements.push(assignment(
        4,
        pair,
        SemanticRvalueKindV1::CheckedBinary(SemanticCheckedBinaryRvalueV1::new(
            SemanticCheckedBinaryOpV1::Add,
            constant(ARRAY_SCALAR, 7, 4),
            constant(ARRAY_SCALAR, 9, 4),
        )),
    ));
    let function = SemanticFunctionDeclV1::new(
        source.identity(),
        source.role(),
        source.item_definition_identity(),
        source.monomorphization_identity(),
        source.generic_type_arguments_identity(),
        source.const_generic_arguments_identity(),
        source.source(),
        source.abi().clone(),
        locals,
        source.entry(),
        vec![block(211, statements, SemanticTerminatorKindV1::Return)],
    )
    .unwrap()
    .with_kernel_entry(source.kernel_entry().unwrap().clone());
    admit_functions(
        declarations,
        vec![function],
        vec![0],
        &[crate::ProductionSourceLaunchRootInputV1::new(
            "array_relation",
            [202; 32],
            crate::ProductionSourceLaunchInputV1::new(1, Some([64, 1, 1]), [1, 1, 1]),
        )],
    )
}

#[test]
fn private_array_actual_checked_failure_rolls_back_rows_and_reused_value_ids_not_work() {
    let owner = checked_owner();
    let source = &owner.semantic_ssa().source_semantic().functions()[0];
    let mut work = CanonicalKernelIrWorkBudgetV1::new(WORK);
    let mut budget = ArgumentBudgetV1::new(&mut work, STORAGE);
    let mut lowering = actual_lowering(&owner, 10_000, &mut budget);
    let mut block = BasicBlock::new(BlockId(0));
    lowering
        .begin_block(SemanticBlockIdV1::from_index(0), &mut block)
        .unwrap();
    for ordinal in 0..2 {
        lowering
            .lower_statement(
                SemanticBlockIdV1::from_index(0),
                Some(ordinal as u32),
                source.blocks()[0].statements()[ordinal].kind(),
                &mut block.operations,
            )
            .unwrap();
    }
    assert_eq!(block.operations.len(), 7);
    let old_operations = block.operations.clone();
    let old_value = lowering.next_value;
    let old_definitions = lowering.private_arrays.definitions.rows.clone();
    let old_capacity = lowering.private_arrays.definitions.logical_capacity;
    let old_slots = lowering.private_arrays.slots.rows.clone();
    let old_effects = lowering.private_arrays.effects.rows.clone();
    let old_cursor = lowering.private_arrays.cursor;
    assert_eq!(old_cursor, 1);
    let old_work = recorder_work(&lowering);
    // Inject only an existing operation-limit failure in this private emitter
    // probe. Source admission and the recorder's shared work meter are unchanged.
    lowering.max_operations = 8;
    let checked = source.blocks()[0].statements()[2].kind();
    assert!(matches!(
        lowering.lower_statement(
            SemanticBlockIdV1::from_index(0),
            Some(2),
            checked,
            &mut block.operations
        ),
        Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Operations,
            actual: 9,
            limit: 8,
        })
    ));
    assert_eq!(block.operations, old_operations);
    assert_eq!(
        (lowering.next_value, lowering.emitted_operations),
        (old_value, 7)
    );
    assert_eq!(lowering.private_arrays.definitions.rows, old_definitions);
    assert_eq!(lowering.private_arrays.slots.rows, old_slots);
    assert_eq!(lowering.private_arrays.effects.rows, old_effects);
    assert!(lowering.private_arrays.definitions.logical_capacity > old_capacity);
    assert_eq!(
        lowering.private_arrays.definitions.admitted_end,
        old_definitions.len()
    );
    assert_eq!(
        (
            lowering.private_arrays.expected.rows.len(),
            lowering.private_arrays.expected.admitted_end,
            lowering.private_arrays.cursor
        ),
        (0, 0, old_cursor)
    );
    assert!(lowering.private_arrays.frame.is_none() && lowering.private_arrays.pending.is_none());
    assert!(recorder_work(&lowering) > old_work);
    let failed_work = recorder_work(&lowering);
    lowering.max_operations = 10_000;
    lowering
        .lower_statement(
            SemanticBlockIdV1::from_index(0),
            Some(2),
            checked,
            &mut block.operations,
        )
        .unwrap();
    let appended = &lowering.private_arrays.definitions.rows[old_definitions.len()..];
    assert_eq!(appended.len(), 2);
    assert_eq!(
        (
            appended[0].value,
            appended[0].constant,
            appended[1].constant
        ),
        (ValueId(old_value), 7, 9)
    );
    assert!(appended[0].value < appended[1].value);
    assert_eq!(block.operations.len(), 10);
    assert!(recorder_work(&lowering) > failed_work);
}

fn inert_reason_allocation() -> fe2o3_kernel_ir::FormalAllocationIdentity {
    // This independently analyzed parameter supplies only the opaque field
    // required to construct an inert diagnostic. It is not this array's origin.
    let mut block = BasicBlock::new(BlockId(0));
    block.terminator = Some(Terminator::Return { values: vec![] });
    let mut module = Module::new("inert_reason_parameter");
    module.functions.push(Function::kernel_entry(
        "inert_reason_parameter",
        Signature::new(
            vec![Type::pointer(
                Type::Scalar(ScalarType::U32),
                AddressSpace::Global,
                AccessMode::ReadOnly,
            )],
            vec![],
        ),
        vec![ValueId(0)],
        vec![block],
    ));
    module.kernels.push(Kernel::new(
        "inert_reason_parameter",
        "inert_reason_parameter",
        LaunchDomain::D1 {
            x: LaunchExtent::Static(64),
        },
    ));
    verify_module(&module).unwrap();
    fe2o3_kernel_ir::derive_kernel_memory_obligations_for_launch(
        &module,
        &module.kernels[0].id,
        fe2o3_kernel_ir::ExplicitLaunchExtent::Exact {
            rank: 1,
            extents: [64, 1, 1],
        },
        fe2o3_kernel_ir::FormalIndexWidth::Bits64,
    )
    .unwrap()
    .obligations()
    .allocations()[0]
        .identity()
}

#[test]
fn private_array_second_consumer_checks_an_explicitly_inert_unsupported_index_diagnostic() {
    use fe2o3_pliron::{
        ProductionConstructionV1, ProductionRankedBlockV1, ProductionRankedKernelV1,
        ProductionRankedTerminatorV1, ProductionSessionLimitsV1,
        compile_ranked_kernel_for_lowering_v1,
    };
    let owner = array_owner(ArrayCase::Write { sparse: false });
    let effect = owner.correspondence.private_arrays.effects[0];
    let layout = owner.source_launch().roots()[0].layout();
    let view = ProductionRankedValueIdV1::new(0);
    let index = ProductionRankedValueIdV1::new(1);
    let origin = (1u64 << 63) + 2;
    let kernel = ProductionRankedKernelV1::new(
        "private_array_relation",
        0,
        vec![ProductionRankedBlockV1::new(
            vec![
                ProductionRankedOperationV1::ExecutionLayout {
                    grid_identity: layout.grid_identity(),
                    global_extents: layout.global_extents(),
                    workgroup_extents: layout.workgroup_extents(),
                    subgroup_size: layout.subgroup_size(),
                    full_physical_workgroups: layout.full_physical_workgroups(),
                },
                ProductionRankedOperationV1::ViewInSpace {
                    result: view,
                    element_width: 32,
                    writable: true,
                    shape: vec![8],
                    dynamic_extents: vec![],
                    memory_space: dialect_kernel::MemorySpaceAttr::Private,
                    allocation_origin: origin,
                    noalias_class: origin,
                },
                ProductionRankedOperationV1::IndexConstant {
                    result: index,
                    value: 0,
                },
                ProductionRankedOperationV1::Access {
                    kind: dialect_kernel::AccessKindAttr::Write,
                    view: ProductionRankedValueV1::Local(view),
                    indices: vec![ProductionRankedValueV1::Local(index)],
                },
            ],
            ProductionRankedTerminatorV1::Return,
        )],
    )
    .unwrap();
    let lowering = compile_ranked_kernel_for_lowering_v1(
        ProductionConstructionV1::ranked_kernel("private_array_relation", kernel).unwrap(),
        ProductionSessionLimitsV1::default(),
    )
    .unwrap();
    assert!(lowering.all_mandatory_reports_are_clean());
    // This lowerer-private component uses the ordinary structured ranked
    // construction API, not the codegen common projector covered separately.
    let root = ProductionRankedSemanticProjectionRootV1::new(
        ARRAY_ROOT,
        1,
        lowering,
        "private array structured correlation component".to_owned(),
        vec![ProductionRankedAccessSourceV1::new(0, Some(1), 0, 0, 3)],
        vec![],
    );
    let receipt =
        ProductionMaterializedRankedModuleReceiptV1::from_unvalidated_projection_roster_candidate(
            owner,
            vec![root],
        )
        .unwrap();
    let mut attached =
        ProductionSemanticKirOwnerV1::try_attach_materialized_ranked_checks(receipt).unwrap();
    let kernel = attached.module().kernels[0].id.clone();
    let analysis = fe2o3_kernel_ir::derive_kernel_memory_obligations_for_launch(
        attached.module(),
        &kernel,
        fe2o3_kernel_ir::ExplicitLaunchExtent::Exact {
            rank: 1,
            extents: [64, 1, 1],
        },
        fe2o3_kernel_ir::FormalIndexWidth::Bits64,
    )
    .unwrap();
    // Ordinary private accesses are intentionally excluded by the existing
    // extractor. Do not misrepresent the supplied diagnostic as its output.
    assert!(!analysis.incomplete_reasons().iter().any(|reason| matches!(
        reason,
        FormalMemoryIncompleteReason::UnsupportedIndexExpression { .. }
    )));
    let reason = FormalMemoryIncompleteReason::UnsupportedIndexExpression {
        location: FunctionOperationLocation::new(
            effect.gep_location.block,
            effect.gep_location.operation,
        ),
        index: effect.offset,
        allocation: inert_reason_allocation(),
    };
    attached
        .retained_generic_checks_discharge_unsupported_indices(
            kernel.as_str(),
            std::slice::from_ref(&reason),
        )
        .unwrap();
    attached.correspondence.private_arrays.slots[0].length = 9;
    assert_eq!(
        attached.retained_generic_checks_discharge_unsupported_indices(kernel.as_str(), &[reason]),
        Err(ProductionMemoryDischargeFailureV1::Access {
            location: FunctionOperationLocation::new(
                effect.memory_location.block,
                effect.memory_location.operation
            ),
            detail: "private array access differs from its exact source, allocation, or index",
        })
    );
}
