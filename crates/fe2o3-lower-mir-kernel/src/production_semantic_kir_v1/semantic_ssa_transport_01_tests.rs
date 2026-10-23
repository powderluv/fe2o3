include!("shared_primitive_source_v29_tests.rs");

include!("../production_invocation_transport_v1_tests.rs");

#[test]
fn direct_parameter_transport_preserves_only_its_authenticated_carrier() {
    let semantic_type = SemanticTypeIdV1::from_index(0);
    let slice = Type::slice(
        Type::Scalar(ScalarType::F32),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let transport = SemanticPromotedTransportV1::DirectParameter { parameter_local: 3 };
    let direct_parameters = BTreeMap::from([(3, slice.clone())]);

    assert_eq!(
        transport
            .transport_types(&[], semantic_type, &direct_parameters)
            .unwrap(),
        vec![slice.clone()]
    );
    assert_eq!(
        transport
            .transport_values(
                &SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: slice.clone(),
                },
                std::slice::from_ref(&slice),
            )
            .unwrap(),
        vec![(ValueId(7), slice.clone())]
    );
    assert!(
        transport
            .transport_values(
                &SemanticValueBindingV1::Aggregate(vec![SemanticValueBindingV1::Value {
                    id: ValueId(7),
                    ty: slice.clone(),
                }]),
                std::slice::from_ref(&slice),
            )
            .is_err()
    );

    let parameter = ValueDef::new(ValueId(11), slice.clone());
    assert!(matches!(
        transport
            .binding_from_transport(
                &[],
                semantic_type,
                std::slice::from_ref(&parameter),
                std::slice::from_ref(&slice),
            )
            .unwrap(),
        SemanticValueBindingV1::Value { id: ValueId(11), ty } if ty == slice
    ));
    assert!(
        transport
            .binding_from_transport(
                &[],
                semantic_type,
                &[ValueDef::new(ValueId(12), Type::Scalar(ScalarType::U64))],
                std::slice::from_ref(&slice),
            )
            .is_err()
    );
}
fn test_option_availability_v1() -> SemanticOptionAvailabilityV1 {
    let unit = SemanticTypeIdV1::from_index(0);
    let option = SemanticTypeIdV1::from_index(1);
    let discriminant = SemanticTypeIdV1::from_index(2);
    let source = SemanticSourceProvenanceV1::unavailable();
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let edge =
        |role, target| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target));
    let block = |tag, statements, terminator| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, terminator),
        )
        .unwrap()
    };
    let call = SemanticDirectCallV1::new_callable(
        SemanticCallableIdV1::from_index(0),
        vec![],
        Some(SemanticCallDestinationV1::new(
            place(1, option),
            edge(SemanticEdgeRoleV1::CallReturn, 1),
        )),
        SemanticUnwindActionV1::Unreachable,
    )
    .unwrap();
    let bind_discriminant = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, discriminant),
            SemanticRvalueV1::new(
                discriminant,
                SemanticRvalueKindV1::Discriminant(place(1, option)),
            ),
        )),
    );
    let blocks = vec![
        block(220, vec![], SemanticTerminatorKindV1::Call(call)),
        block(
            221,
            vec![bind_discriminant],
            SemanticTerminatorKindV1::SwitchInt {
                discriminant: SemanticOperandV1::Copy(place(2, discriminant)),
                targets: SemanticSwitchTargetsV1::new(
                    vec![SemanticSwitchTargetV1::new(
                        0,
                        edge(SemanticEdgeRoleV1::SwitchValue, 2),
                    )],
                    edge(SemanticEdgeRoleV1::SwitchOtherwise, 3),
                )
                .unwrap(),
            },
        ),
        block(222, vec![], SemanticTerminatorKindV1::Return),
        block(223, vec![], SemanticTerminatorKindV1::Return),
    ];
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([224; 32]),
        SemanticLayoutIdentityV1::from_sha256([225; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let local = |tag, ty, role| {
        SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([tag; 32]),
            ty,
            role,
            source,
        )
    };
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([226; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([227; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([228; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([229; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([230; 32]),
        source,
        abi,
        vec![
            local(231, unit, SemanticLocalRoleV1::Return),
            local(232, option, SemanticLocalRoleV1::Temporary),
            local(233, discriminant, SemanticLocalRoleV1::Temporary),
        ],
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap();
    let option_local = SemanticLocalIdV1::from_index(1);
    SemanticOptionDominanceV1::analyze(
        &function,
        &[fe2o3_mir_model::SemanticOptionProducerV1::new(
            option_local,
            SemanticBlockIdV1::from_index(1),
        )],
    )
    .unwrap()
    .availability(option_local)
    .unwrap()
}

#[test]
fn optional_capability_transport_round_trips_metadata_without_strengthening() {
    let availability = test_option_availability_v1();
    let semantic_type = SemanticTypeIdV1::from_index(0);
    let tiled = SemanticDisjointIndexSpaceV1::Tiled2dIndex1d {
        lanes_per_tile: 64,
        tile_rows: 16,
        tile_columns: 16,
        elements_per_lane: 4,
    };
    let component = SemanticPromotedBindingV1::OptionComponentWitness {
        index_space: tiled,
        availability,
    };
    let component_values = [
        ValueDef::new(ValueId(20), Type::BOOL),
        ValueDef::new(ValueId(21), Type::INDEX),
    ];
    assert_eq!(
        component.transport_types(&[], semantic_type).unwrap(),
        vec![Type::BOOL, Type::INDEX]
    );
    assert!(matches!(
        component
            .binding_from_transport(&[], semantic_type, &component_values)
            .unwrap(),
        SemanticValueBindingV1::OptionComponentWitness {
            present: ValueId(20),
            raw: ValueId(21),
            index_space,
            availability: actual,
        } if index_space == tiled && actual == availability
    ));
    assert!(
        component
            .transport_values(&SemanticValueBindingV1::OptionComponentWitness {
                present: ValueId(20),
                raw: ValueId(21),
                index_space: SemanticDisjointIndexSpaceV1::Index1d,
                availability,
            },)
            .is_err()
    );

    for disjoint in [false, true] {
        let index = SemanticPromotedBindingV1::OptionIndexWitness {
            index_space: SemanticDisjointIndexSpaceV1::ShiftedIndex1d { offset: 4 },
            disjoint,
            availability,
        };
        assert!(matches!(
            index
                .binding_from_transport(&[], semantic_type, &component_values)
                .unwrap(),
            SemanticValueBindingV1::OptionIndexWitness {
                disjoint: actual,
                availability: actual_availability,
                ..
            } if actual == disjoint && actual_availability == availability
        ));
    }

    let leader = SemanticPromotedBindingV1::OptionGridLeader { availability };
    assert!(matches!(
        leader
            .binding_from_transport(
                &[],
                semantic_type,
                &[ValueDef::new(ValueId(22), Type::BOOL)],
            )
            .unwrap(),
        SemanticValueBindingV1::OptionGridLeader {
            present: ValueId(22),
            availability: actual,
        } if actual == availability
    ));

    let pointer = SemanticPromotedBindingV1::OptionPointer {
        element: ScalarType::F32,
        address_space: AddressSpace::Global,
        access: AccessMode::ReadWrite,
        availability,
    };
    let pointer_type = Type::pointer(
        Type::Scalar(ScalarType::F32),
        AddressSpace::Global,
        AccessMode::ReadWrite,
    );
    let pointer_values = [
        ValueDef::new(ValueId(23), Type::BOOL),
        ValueDef::new(ValueId(24), pointer_type.clone()),
    ];
    let pointer_binding = pointer
        .binding_from_transport(&[], semantic_type, &pointer_values)
        .unwrap();
    assert_eq!(
        pointer.transport_values(&pointer_binding).unwrap(),
        vec![
            (ValueId(23), Type::BOOL),
            (ValueId(24), pointer_type.clone()),
        ]
    );
    assert!(
        SemanticPromotedBindingV1::Ordinary
            .transport_values(&pointer_binding)
            .is_err()
    );
}

#[test]
fn promoted_transparent_borrow_uses_the_referent_aggregate_transport() {
    let unit = SemanticTypeIdV1::from_index(0);
    let u64_ty = SemanticTypeIdV1::from_index(1);
    let aggregate_ty = SemanticTypeIdV1::from_index(2);
    let reference_ty = SemanticTypeIdV1::from_index(3);
    let source = SemanticSourceProvenanceV1::unavailable();
    let types = vec![
        unit_type(),
        u64_type(),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([201; 32]),
            SemanticLayoutIdentityV1::from_sha256([202; 32]),
            SemanticTypeLayoutV1::aggregate(
                Some(16),
                8,
                SemanticAggregateLayoutV1::new(vec![0, 8], vec![]).unwrap(),
            )
            .unwrap(),
            SemanticTypeShapeV1::Aggregate(
                SemanticAggregateTypeV1::new(vec![u64_ty, u64_ty]).unwrap(),
            ),
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([203; 32]),
            SemanticLayoutIdentityV1::from_sha256([204; 32]),
            SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
            SemanticTypeShapeV1::Pointer(
                fe2o3_mir_model::semantic_mir_v1::SemanticPointerTypeV1::new_with_kind(
                    aggregate_ty,
                    SemanticPointerKindV1::Reference,
                    SemanticMutabilityV1::Immutable,
                    0,
                    64,
                    SemanticPointerMetadataV1::None,
                )
                .unwrap(),
            ),
        ),
    ];
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let alias = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(3, aggregate_ty),
            SemanticRvalueV1::new(
                aggregate_ty,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(1, aggregate_ty))),
            ),
        )),
    );
    let borrow = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, reference_ty),
            SemanticRvalueV1::new(
                reference_ty,
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared,
                    place: place(3, aggregate_ty),
                },
            ),
        )),
    );
    let block = SemanticBasicBlockV1::new(
        SemanticBlockIdentityV1::from_sha256([205; 32]),
        source,
        vec![alias, borrow],
        SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
    )
    .unwrap();
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([206; 32]),
        SemanticLayoutIdentityV1::from_sha256([207; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let local = |tag, ty, role| {
        SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([tag; 32]),
            ty,
            role,
            source,
        )
    };
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([208; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([209; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([210; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([211; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([212; 32]),
        source,
        abi,
        vec![
            local(213, unit, SemanticLocalRoleV1::Return),
            local(214, aggregate_ty, SemanticLocalRoleV1::Temporary),
            local(215, reference_ty, SemanticLocalRoleV1::Temporary),
            local(216, aggregate_ty, SemanticLocalRoleV1::Temporary),
        ],
        SemanticBlockIdV1::from_index(0),
        vec![block],
    )
    .unwrap();

    let option_dominance = SemanticOptionDominanceV1::analyze(&function, &[]).unwrap();
    let promoted = BTreeSet::from([1, 2, 3]);
    let mut capability_origins = SemanticCapabilityOriginResolverV1::new(
        &types,
        &[],
        &function,
        &option_dominance,
        &promoted,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    let (transport_type, binding) = promoted_transport_descriptor_v1(
        &types,
        &function,
        2,
        &BTreeMap::new(),
        &promoted,
        &mut capability_origins,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(transport_type, aggregate_ty);
    assert_eq!(
        binding,
        SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary)
    );
    assert_eq!(
        binding
            .transport_types(&types, transport_type, &BTreeMap::new())
            .unwrap(),
        vec![Type::Scalar(ScalarType::U64), Type::Scalar(ScalarType::U64),],
    );
    let expected = [Type::Scalar(ScalarType::U64), Type::Scalar(ScalarType::U64)];
    assert!(matches!(
        binding
            .binding_from_transport(
                &types,
                transport_type,
                &[
                    ValueDef::new(ValueId(40), Type::Scalar(ScalarType::U64)),
                    ValueDef::new(ValueId(41), Type::Scalar(ScalarType::U64)),
                ],
                &expected,
            )
            .unwrap(),
        SemanticValueBindingV1::Aggregate(fields) if fields.len() == 2
    ));

    let promoted = BTreeSet::from([2]);
    let mut capability_origins = SemanticCapabilityOriginResolverV1::new(
        &types,
        &[],
        &function,
        &option_dominance,
        &promoted,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    let (transport_type, binding) = promoted_transport_descriptor_v1(
        &types,
        &function,
        2,
        &BTreeMap::new(),
        &promoted,
        &mut capability_origins,
        &BTreeMap::new(),
    )
    .unwrap();
    assert_eq!(transport_type, reference_ty);
    assert_eq!(
        binding,
        SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary)
    );
}

#[test]
fn entry_seeds_only_the_certified_implicit_workgroup_lds_scope() {
    let scope_ty = SemanticTypeIdV1::from_index(0);
    let same_shape_ty = SemanticTypeIdV1::from_index(1);
    let reference_ty = SemanticTypeIdV1::from_index(2);
    let dynamic_lds_ty = SemanticTypeIdV1::from_index(3);
    let storage_ty = SemanticTypeIdV1::from_index(4);
    let source = SemanticSourceProvenanceV1::unavailable();
    let zst = |tag| {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag.wrapping_add(1); 32]),
            SemanticTypeLayoutV1::aggregate(
                Some(0),
                1,
                SemanticAggregateLayoutV1::new(vec![], vec![]).unwrap(),
            )
            .unwrap(),
            SemanticTypeShapeV1::Aggregate(SemanticAggregateTypeV1::new(vec![]).unwrap()),
        )
    };
    let types = vec![
        zst(150),
        zst(152),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([154; 32]),
            SemanticLayoutIdentityV1::from_sha256([155; 32]),
            SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
            SemanticTypeShapeV1::Opaque,
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([156; 32]),
            SemanticLayoutIdentityV1::from_sha256([157; 32]),
            SemanticTypeLayoutV1::new(Some(0), 1).unwrap(),
            SemanticTypeShapeV1::Opaque,
        ),
        integer_type(158, false, 32),
    ];
    let direct = |ty| {
        SemanticAbiValueV1::new(
            ty,
            SemanticAbiPassModeV1::Direct(
                fe2o3_mir_model::semantic_mir_v1::SemanticAbiValueAttributesV1::plain(),
            ),
        )
    };
    let intrinsic_abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([160; 32]),
        SemanticLayoutIdentityV1::from_sha256([161; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        1,
        vec![SemanticAbiArgumentV1::source(direct(reference_ty))],
        direct(dynamic_lds_ty),
    )
    .unwrap();
    let callable = SemanticCallableDeclV1::CompilerIntrinsic {
        binding: SemanticNonBodyCallableBindingV1::new(
            SemanticFunctionIdentityV1::from_sha256([162; 32]),
            SemanticItemDefinitionIdentityV1::from_sha256([163; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([164; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([165; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([166; 32]),
            source,
            intrinsic_abi,
        ),
        operation: SemanticCompilerIntrinsicOperationV1::DynamicLdsExactCurrent {
            scope: scope_ty,
            dynamic_lds: dynamic_lds_ty,
            element_storage: storage_ty,
            elements: 1,
        },
        operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([167; 32]),
    };
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let borrow = SemanticStatementV1::new(
        source,
        SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
            place(2, reference_ty),
            SemanticRvalueV1::new(
                reference_ty,
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Mutable,
                    place: place(1, scope_ty),
                },
            ),
        )),
    );
    let return_edge = SemanticControlFlowEdgeV1::new(
        SemanticEdgeRoleV1::CallReturn,
        SemanticBlockIdV1::from_index(1),
    );
    let call = SemanticDirectCallV1::new_callable(
        SemanticCallableIdV1::from_index(0),
        vec![SemanticOperandV1::Copy(place(2, reference_ty))],
        Some(SemanticCallDestinationV1::new(
            place(4, dynamic_lds_ty),
            return_edge,
        )),
        SemanticUnwindActionV1::Unreachable,
    )
    .unwrap();
    let blocks = vec![
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([168; 32]),
            source,
            vec![borrow],
            SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Call(call)),
        )
        .unwrap(),
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([169; 32]),
            source,
            vec![],
            SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
        )
        .unwrap(),
    ];
    let function_abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([170; 32]),
        SemanticLayoutIdentityV1::from_sha256([171; 32]),
        SemanticCanonAbiV1::GpuKernel,
        SemanticExternAbiV1::GpuKernel,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(same_shape_ty, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let local = |tag, ty, role| {
        SemanticLocalDeclV1::new(
            SemanticLocalIdentityV1::from_sha256([tag; 32]),
            ty,
            role,
            source,
        )
    };
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([172; 32]),
        SemanticFunctionRoleV1::KernelRoot,
        SemanticItemDefinitionIdentityV1::from_sha256([173; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([174; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([175; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([176; 32]),
        source,
        function_abi,
        vec![
            local(177, same_shape_ty, SemanticLocalRoleV1::Return),
            local(178, scope_ty, SemanticLocalRoleV1::Temporary),
            local(179, reference_ty, SemanticLocalRoleV1::Temporary),
            local(180, same_shape_ty, SemanticLocalRoleV1::Temporary),
            local(181, dynamic_lds_ty, SemanticLocalRoleV1::Temporary),
        ],
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap();
    let callables = [callable];
    let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
    let mut lowering = SemanticFunctionLoweringV1::new(
        &types,
        &callables,
        &function,
        SemanticParameterBindingsV1 {
            declarations: &[],
            values: &[],
            types: &[],
            local_bindings: None,
        },
        None,
        None,
        BTreeSet::new(),
        1,
        false,
        16,
        &mut lowering_budget,
    )
    .unwrap();

    assert_eq!(
        lowering.control_flow_ssa.implicit_entry_locals,
        BTreeSet::from([1])
    );
    assert!(lowering.locals[1].is_none());
    assert!(lowering.locals[3].is_none());
    let mut entry = BasicBlock::new(BlockId(0));
    lowering
        .begin_block(SemanticBlockIdV1::from_index(0), &mut entry)
        .unwrap();
    assert!(matches!(
        lowering.locals[1],
        Some(SemanticValueBindingV1::WorkgroupLdsScope)
    ));
    assert!(lowering.locals[3].is_none());

    let aggregate = SemanticRvalueKindV1::Aggregate(
        fe2o3_mir_model::semantic_mir_v1::SemanticAggregateRvalueV1::new(
            SemanticAggregateKindV1::Aggregate,
            vec![],
        )
        .unwrap(),
    );
    let mut operations = Vec::new();
    for ty in [scope_ty, same_shape_ty] {
        assert!(matches!(
            lowering
                .lower_rvalue(
                    SemanticBlockIdV1::from_index(0),
                    Some(0),
                    ty,
                    &aggregate,
                    &mut operations,
                )
                .unwrap(),
            SemanticValueBindingV1::Aggregate(fields) if fields.is_empty()
        ));
    }
    assert!(operations.is_empty());
}

fn allocation_provenance_fixture_v1(
    entry_terminator: Terminator,
    merge_terminator: Terminator,
) -> Function {
    let slice = Type::slice(
        Type::Scalar(ScalarType::F32),
        AddressSpace::Global,
        AccessMode::ReadOnly,
    );
    let mut entry = BasicBlock::new(BlockId(0));
    entry.terminator = Some(entry_terminator);
    let mut merge = BasicBlock::new(BlockId(1));
    merge
        .parameters
        .push(ValueDef::new(ValueId(2), slice.clone()));
    merge.terminator = Some(merge_terminator);
    Function::kernel_entry(
        "ssa_allocation_provenance",
        Signature::new(vec![slice.clone(), slice], vec![]),
        vec![ValueId(0), ValueId(1)],
        vec![entry, merge],
    )
}

fn allocation_origin_v1(function: &Function, value: ValueId) -> Option<u32> {
    let body = function.body.as_ref().unwrap();
    let mut index_budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 256 };
    let index = build_kir_correlation_index(body, 16, &mut index_budget).unwrap();
    let mut provenance_budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 256 };
    external_allocation_parameter_v1(
        function,
        &index,
        value,
        &mut BTreeSet::new(),
        &mut provenance_budget,
    )
}

#[test]
fn allocation_provenance_traverses_ssa_edges_and_loop_backedges() {
    let direct = allocation_provenance_fixture_v1(
        Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(0)],
        },
        Terminator::Return { values: vec![] },
    );
    assert_eq!(allocation_origin_v1(&direct, ValueId(2)), Some(0));

    let loop_carried = allocation_provenance_fixture_v1(
        Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(0)],
        },
        Terminator::Branch {
            target: BlockId(1),
            arguments: vec![ValueId(2)],
        },
    );
    assert_eq!(allocation_origin_v1(&loop_carried, ValueId(2)), Some(0));
}

#[test]
fn allocation_provenance_rejects_conflicting_parallel_ssa_edges() {
    let conflicting = allocation_provenance_fixture_v1(
        Terminator::ConditionalBranch {
            condition: ValueId(3),
            then_target: BlockId(1),
            then_arguments: vec![ValueId(0)],
            else_target: BlockId(1),
            else_arguments: vec![ValueId(1)],
        },
        Terminator::Return { values: vec![] },
    );
    assert_eq!(allocation_origin_v1(&conflicting, ValueId(2)), None);
}

fn exact_enum_ssa_fixture_v1(
    parallel_variant_edges: bool,
) -> (
    Vec<SemanticTypeDeclV1>,
    SemanticFunctionDeclV1,
    SemanticControlFlowSsaPlanV1,
) {
    let unit = SemanticTypeIdV1::from_index(0);
    let u32_ty = SemanticTypeIdV1::from_index(1);
    let enum_ty = SemanticTypeIdV1::from_index(2);
    let source = SemanticSourceProvenanceV1::unavailable();
    let types = vec![
        unit_type(),
        unsigned_scalar_type(61, 32),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([62; 32]),
            SemanticLayoutIdentityV1::from_sha256([63; 32]),
            SemanticTypeLayoutV1::new(Some(8), 4).unwrap(),
            SemanticTypeShapeV1::Enum {
                discriminant: u32_ty,
                variants: vec![
                    SemanticEnumVariantV1::new(
                        0,
                        SemanticAggregateTypeV1::new(vec![u32_ty]).unwrap(),
                    ),
                    SemanticEnumVariantV1::new(
                        1,
                        SemanticAggregateTypeV1::new(vec![u32_ty]).unwrap(),
                    ),
                ]
                .into_boxed_slice(),
            },
        ),
    ];
    let place =
        |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
    let field = |variant| {
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            vec![
                SemanticProjectionV1::new(SemanticProjectionKindV1::Downcast(variant), enum_ty)
                    .unwrap(),
                SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), u32_ty).unwrap(),
            ],
            u32_ty,
        )
        .unwrap()
    };
    let scalar = |value| {
        SemanticOperandV1::Constant(SemanticConstantV1::new(
            u32_ty,
            SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, 4).unwrap()),
        ))
    };
    let assign = |destination, ty, value| {
        SemanticStatementV1::new(
            source,
            SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                destination,
                SemanticRvalueV1::new(ty, value),
            )),
        )
    };
    let enum_variant = |variant, payload| {
        SemanticRvalueKindV1::aggregate(
            SemanticAggregateKindV1::EnumVariant(variant),
            vec![scalar(payload)],
        )
        .unwrap()
    };
    let edge =
        |role, target| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target));
    let block = |tag, statements, terminator| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, terminator),
        )
        .unwrap()
    };
    let initial_switch = SemanticTerminatorKindV1::SwitchInt {
        discriminant: scalar(1),
        targets: SemanticSwitchTargetsV1::new(
            vec![SemanticSwitchTargetV1::new(
                1,
                edge(SemanticEdgeRoleV1::SwitchValue, 1),
            )],
            edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
        )
        .unwrap(),
    };
    let variant_one_target = if parallel_variant_edges { 5 } else { 6 };
    let variant_switch = SemanticTerminatorKindV1::SwitchInt {
        discriminant: SemanticOperandV1::Copy(place(3, u32_ty)),
        targets: SemanticSwitchTargetsV1::new(
            vec![
                SemanticSwitchTargetV1::new(0, edge(SemanticEdgeRoleV1::SwitchValue, 5)),
                SemanticSwitchTargetV1::new(
                    1,
                    edge(SemanticEdgeRoleV1::SwitchValue, variant_one_target),
                ),
            ],
            edge(SemanticEdgeRoleV1::SwitchOtherwise, 7),
        )
        .unwrap(),
    };
    let blocks = vec![
        block(64, vec![], initial_switch),
        block(
            65,
            vec![assign(place(1, enum_ty), enum_ty, enum_variant(0, 11))],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 3)),
        ),
        block(
            66,
            vec![assign(place(1, enum_ty), enum_ty, enum_variant(1, 13))],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 3)),
        ),
        block(
            67,
            vec![assign(
                place(2, u32_ty),
                u32_ty,
                SemanticRvalueKindV1::Discriminant(place(1, enum_ty)),
            )],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 4)),
        ),
        block(
            68,
            vec![assign(
                place(3, u32_ty),
                u32_ty,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(2, u32_ty))),
            )],
            variant_switch,
        ),
        block(
            69,
            vec![assign(
                place(4, u32_ty),
                u32_ty,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(field(0))),
            )],
            SemanticTerminatorKindV1::Return,
        ),
        block(
            70,
            vec![assign(
                place(4, u32_ty),
                u32_ty,
                SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(field(1))),
            )],
            SemanticTerminatorKindV1::Return,
        ),
        block(71, vec![], SemanticTerminatorKindV1::Unreachable),
    ];
    let abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([72; 32]),
        SemanticLayoutIdentityV1::from_sha256([73; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::Rust,
        false,
        false,
        0,
        vec![],
        SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
    )
    .unwrap();
    let locals = [unit, enum_ty, u32_ty, u32_ty, u32_ty]
        .into_iter()
        .enumerate()
        .map(|(local, ty)| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([74 + local as u8; 32]),
                ty,
                if local == 0 {
                    SemanticLocalRoleV1::Return
                } else {
                    SemanticLocalRoleV1::Temporary
                },
                source,
            )
        })
        .collect();
    let function = SemanticFunctionDeclV1::new(
        SemanticFunctionIdentityV1::from_sha256([79; 32]),
        SemanticFunctionRoleV1::InternalHelper,
        SemanticItemDefinitionIdentityV1::from_sha256([80; 32]),
        SemanticMonomorphizationIdentityV1::from_sha256([81; 32]),
        SemanticGenericTypeArgumentsIdentityV1::from_sha256([82; 32]),
        SemanticConstGenericArgumentsIdentityV1::from_sha256([83; 32]),
        source,
        abi,
        locals,
        SemanticBlockIdV1::from_index(0),
        blocks,
    )
    .unwrap();
    let semantic_function = SemanticFunctionIdV1::from_index(0);
    let semantic_ssa = plan_semantic_function_ssa_with_module_v1(
        semantic_function,
        &function,
        &types,
        &[],
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap();
    let option_dominance = SemanticOptionDominanceV1::analyze(&function, &[]).unwrap();
    let plan = SemanticControlFlowSsaPlanV1::analyze(
        SemanticSsaTransportInputV1 {
            types: &types,
            callables: &[],
            function: &function,
            semantic_function,
        },
        &semantic_ssa,
        &option_dominance,
        &BTreeMap::new(),
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    (types, function, plan)
}
