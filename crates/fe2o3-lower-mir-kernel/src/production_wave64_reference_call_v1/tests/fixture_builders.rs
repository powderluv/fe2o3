fn fixture_types(shift_counter: bool) -> Vec<SemanticTypeDeclV1> {
    let id = SemanticTypeIdV1::from_index;
    let unit = id(0);
    let word = id(1);
    let context = id(4);
    let snapshot = id(5);
    let aggregate = |tag, fields: Vec<SemanticTypeIdV1>, bytes, align| {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag; 32]),
            SemanticTypeLayoutV1::aggregate_with_backend_repr(
                Some(bytes),
                align,
                if tag == 4 {
                    SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                        SemanticBackendPrimitiveV1::integer(false, 32, 4),
                        SemanticScalarValidityRangeV1::new(0, u32::MAX.into()),
                    ))
                } else {
                    SemanticBackendReprV1::memory(true)
                },
                false,
                SemanticAggregateLayoutV1::new(vec![0; fields.len()], vec![]).unwrap(),
            )
            .unwrap(),
            SemanticTypeShapeV1::Aggregate(SemanticAggregateTypeV1::new(fields).unwrap()),
        )
    };
    let pointer = |tag, pointee, mutable| {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(8),
                8,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::pointer(0, 8, 8),
                    SemanticScalarValidityRangeV1::new(1, u64::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Pointer(
                SemanticPointerTypeV1::new_with_kind(
                    pointee,
                    SemanticPointerKindV1::Reference,
                    if mutable {
                        SemanticMutabilityV1::Mutable
                    } else {
                        SemanticMutabilityV1::Immutable
                    },
                    0,
                    64,
                    SemanticPointerMetadataV1::None,
                )
                .unwrap(),
            ),
        )
        .with_rustc_abi_properties(
            SemanticTypeAbiPropertiesV1::new(false, false).with_scalar_pointee_info(
                Some(
                    SemanticAbiPointeeInfoV1::new(
                        if mutable {
                            SemanticAbiPointeeKindV1::MutableReference { unpin: true }
                        } else {
                            SemanticAbiPointeeKindV1::SharedReference { frozen: true }
                        },
                        if pointee == context { 0 } else { 4 },
                        if pointee == context { 1 } else { 4 },
                    )
                    .unwrap(),
                ),
                None,
            ),
        )
    };
    let raw_pointer = |tag, pointee| {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(8),
                8,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::pointer(0, 8, 8),
                    SemanticScalarValidityRangeV1::new(0, u64::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Pointer(
                SemanticPointerTypeV1::new_with_kind(
                    pointee,
                    SemanticPointerKindV1::Raw,
                    SemanticMutabilityV1::Mutable,
                    0,
                    64,
                    SemanticPointerMetadataV1::None,
                )
                .unwrap(),
            ),
        )
        .with_rustc_abi_properties(
            SemanticTypeAbiPropertiesV1::new(false, false).with_scalar_pointee_info(
                Some(SemanticAbiPointeeInfoV1::new(SemanticAbiPointeeKindV1::Raw, 0, 1).unwrap()),
                None,
            ),
        )
    };
    let mut types = vec![
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([1; 32]),
            SemanticLayoutIdentityV1::from_sha256([1; 32]),
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
            SemanticTypeIdentityV1::from_sha256([2; 32]),
            SemanticLayoutIdentityV1::from_sha256([2; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(4),
                4,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::integer(false, 32, 4),
                    SemanticScalarValidityRangeV1::new(0, u32::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: false,
                bits: 32,
            }),
        ),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([3; 32]),
            SemanticLayoutIdentityV1::from_sha256([3; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(4),
                4,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::float(32, 4),
                    SemanticScalarValidityRangeV1::new(0, u32::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Float { bits: 32 }),
        ),
        aggregate(4, vec![word], 4, 4),
        aggregate(5, vec![], 0, 1),
        aggregate(6, vec![word, unit], 4, 4),
        pointer(7, snapshot, false),
        pointer(8, context, false),
        pointer(9, word, false),
        pointer(10, snapshot, true),
        raw_pointer(11, snapshot),
        raw_pointer(12, context),
    ];
    if shift_counter {
        types.push(SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([14; 32]),
            SemanticLayoutIdentityV1::from_sha256([14; 32]),
            SemanticTypeLayoutV1::new_with_backend_repr(
                Some(4),
                4,
                SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                    SemanticBackendPrimitiveV1::integer(true, 32, 4),
                    SemanticScalarValidityRangeV1::new(0, u32::MAX.into()),
                )),
                false,
            )
            .unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
                signed: true,
                bits: 32,
            }),
        ));
        types.push(raw_pointer(15, word));
    }
    types
}

fn fixture_abi(
    tag: u8,
    inputs: Vec<SemanticTypeIdV1>,
    output: SemanticTypeIdV1,
    ownership: Vec<SemanticSourceArgumentOwnershipV1>,
    root: bool,
) -> SemanticFunctionAbiV1 {
    let id = SemanticTypeIdV1::from_index;
    let unit = id(0);
    let context = id(4);
    let snapshot_ref = id(6);
    let context_ref = id(7);
    let direct = |ty| {
        let shared_reference = ty == snapshot_ref || ty == context_ref;
        let attributes = SemanticAbiValueAttributesV1::new(
            SemanticAbiRegularAttributesV1::new(
                shared_reference,
                shared_reference.then_some(SemanticAbiPointerCaptureV1::CapturesReadOnly),
                shared_reference,
                shared_reference,
                false,
                true,
            ),
            SemanticAbiExtensionV1::None,
            if ty == snapshot_ref { 4 } else { 0 },
            (ty == snapshot_ref).then_some(4),
        )
        .unwrap();
        SemanticAbiValueV1::new(ty, SemanticAbiPassModeV1::Direct(attributes))
    };
    let abi = |tag,
               inputs: Vec<SemanticTypeIdV1>,
               output,
               ownership: Vec<SemanticSourceArgumentOwnershipV1>,
               root| {
        SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag; 32]),
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
            inputs.len() as u32,
            inputs
                .into_iter()
                .map(|ty| SemanticAbiArgumentV1::source(direct(ty)))
                .collect(),
            if output == unit || output == context {
                SemanticAbiValueV1::new(output, SemanticAbiPassModeV1::Ignore)
            } else {
                direct(output)
            },
        )
        .unwrap()
        .with_source_argument_ownership(ownership)
        .unwrap()
    };
    abi(tag, inputs, output, ownership, root)
}

fn fixture_callables(raw: bool) -> Vec<SemanticCallableDeclV1> {
    let source = SemanticSourceProvenanceV1::unavailable();
    let id = SemanticTypeIdV1::from_index;
    let word = id(1);
    let float = id(2);
    let lane = id(3);
    let context = id(4);
    let context_ref = id(7);
    let abi = fixture_abi;
    let intrinsic = |tag, operation, output| SemanticCallableDeclV1::CompilerIntrinsic {
        binding: SemanticNonBodyCallableBindingV1::new(
            SemanticFunctionIdentityV1::from_sha256([tag; 32]),
            SemanticItemDefinitionIdentityV1::from_sha256([tag; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([tag; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([tag; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([tag; 32]),
            source,
            abi(tag, vec![], output, vec![], false),
        ),
        operation,
        operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([tag; 32]),
    };
    let mut callables = vec![
        // Defined callables occupy the canonical function-index prefix.
        SemanticCallableDeclV1::Defined {
            function: SemanticFunctionIdV1::from_index(0),
        },
        SemanticCallableDeclV1::Defined {
            function: SemanticFunctionIdV1::from_index(1),
        },
        intrinsic(
            20,
            SemanticCompilerIntrinsicOperationV1::WaveLaneCurrent {
                lane,
                wave_width: 64,
            },
            lane,
        ),
        intrinsic(
            21,
            SemanticCompilerIntrinsicOperationV1::CollectiveContextCurrent { context },
            context,
        ),
    ];
    if raw {
        // This is an actual declared source intrinsic with its real typed
        // argument/return contract, not a factory activation predicate.
        callables.push(SemanticCallableDeclV1::CompilerIntrinsic {
            binding: SemanticNonBodyCallableBindingV1::new(
                SemanticFunctionIdentityV1::from_sha256([22; 32]),
                SemanticItemDefinitionIdentityV1::from_sha256([22; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([22; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([22; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([22; 32]),
                source,
                abi(
                    22,
                    vec![context_ref, float, word],
                    float,
                    vec![
                        SemanticSourceArgumentOwnershipV1::SharedBorrow,
                        SemanticSourceArgumentOwnershipV1::ByValue,
                        SemanticSourceArgumentOwnershipV1::ByValue,
                    ],
                    false,
                ),
            ),
            operation: SemanticCompilerIntrinsicOperationV1::Gfx942Wave64ShuffleIndex {
                context,
                element: float,
            },
            operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([22; 32]),
        });
    }
    callables
}

fn fixture_function(
    tag: u8,
    root: bool,
    types: Vec<(SemanticTypeIdV1, SemanticLocalRoleV1)>,
    abi: SemanticFunctionAbiV1,
    blocks: Vec<SemanticBasicBlockV1>,
) -> SemanticFunctionDeclV1 {
    let source = SemanticSourceProvenanceV1::unavailable();
    let function = |tag, root, types: Vec<(SemanticTypeIdV1, SemanticLocalRoleV1)>, abi, blocks| {
        SemanticFunctionDeclV1::new(
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
            source,
            abi,
            types
                .into_iter()
                .enumerate()
                .map(|(index, (ty, role))| {
                    SemanticLocalDeclV1::new(
                        SemanticLocalIdentityV1::from_sha256([tag + index as u8 + 1; 32]),
                        ty,
                        role,
                        source,
                    )
                })
                .collect(),
            SemanticBlockIdV1::from_index(0),
            blocks,
        )
        .unwrap()
    };
    function(tag, root, types, abi, blocks)
}
