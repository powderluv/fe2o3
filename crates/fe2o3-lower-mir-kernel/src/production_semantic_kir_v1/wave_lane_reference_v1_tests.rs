use super::*;
use fe2o3_mir_model::semantic_mir_v1::SemanticPointerTypeV1;

#[test]
fn wave_lane_reference_projection_requires_the_exact_issued_reference_shape() {
    // Predicate-only descriptors and bindings are not admitted source owners.
    let lane_type = SemanticTypeIdV1::from_index(0);
    let reference_type = SemanticTypeIdV1::from_index(1);
    let declaration = |tag, pointee, kind, mutability, address_space, width, metadata| {
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([tag; 32]),
            SemanticLayoutIdentityV1::from_sha256([tag; 32]),
            SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
            SemanticTypeShapeV1::Pointer(
                SemanticPointerTypeV1::new_with_kind(
                    pointee,
                    kind,
                    mutability,
                    address_space,
                    width,
                    metadata,
                )
                .unwrap(),
            ),
        )
    };
    let scalar = SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([240; 32]),
        SemanticLayoutIdentityV1::from_sha256([240; 32]),
        SemanticTypeLayoutV1::new(Some(4), 4).unwrap(),
        SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Integer {
            signed: false,
            bits: 32,
        }),
    );
    let shared = declaration(
        241,
        lane_type,
        SemanticPointerKindV1::Reference,
        SemanticMutabilityV1::Immutable,
        0,
        64,
        SemanticPointerMetadataV1::None,
    );
    let types = vec![scalar, shared];
    let issued = BTreeMap::from([(
        lane_type,
        SemanticPromotedBindingV1::WaveLane { wave_width: 64 },
    )]);
    let projection =
        SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, lane_type).unwrap();
    let binding = SemanticValueBindingV1::WaveLane {
        value: ValueId(42),
        wave: SemanticCurrentWaveV1::new(64),
    };
    let matches = |types: &[SemanticTypeDeclV1],
                   issued: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
                   projection: &SemanticProjectionV1,
                   binding: &SemanticValueBindingV1| {
        wave_lane_reference_dereference_matches_v1(
            types,
            issued,
            reference_type,
            projection,
            binding,
        )
    };
    assert!(matches(&types, &issued, &projection, &binding));
    let mut mutable = types.clone();
    mutable[1] = declaration(
        242,
        lane_type,
        SemanticPointerKindV1::Reference,
        SemanticMutabilityV1::Mutable,
        0,
        64,
        SemanticPointerMetadataV1::None,
    );
    assert!(matches(&mutable, &issued, &projection, &binding));

    for invalid in [
        declaration(
            243,
            lane_type,
            SemanticPointerKindV1::Raw,
            SemanticMutabilityV1::Immutable,
            0,
            64,
            SemanticPointerMetadataV1::None,
        ),
        declaration(
            244,
            lane_type,
            SemanticPointerKindV1::Reference,
            SemanticMutabilityV1::Immutable,
            1,
            64,
            SemanticPointerMetadataV1::None,
        ),
        declaration(
            245,
            lane_type,
            SemanticPointerKindV1::Reference,
            SemanticMutabilityV1::Immutable,
            0,
            32,
            SemanticPointerMetadataV1::None,
        ),
        declaration(
            246,
            lane_type,
            SemanticPointerKindV1::Reference,
            SemanticMutabilityV1::Immutable,
            0,
            64,
            SemanticPointerMetadataV1::SliceLength,
        ),
        declaration(
            247,
            SemanticTypeIdV1::from_index(9),
            SemanticPointerKindV1::Reference,
            SemanticMutabilityV1::Immutable,
            0,
            64,
            SemanticPointerMetadataV1::None,
        ),
        types[0].clone(),
    ] {
        let mut wrong = types.clone();
        wrong[1] = invalid;
        assert!(!matches(&wrong, &issued, &projection, &binding));
    }
    assert!(!matches(&types[..1], &issued, &projection, &binding));
    let foreign_type = SemanticTypeIdV1::from_index(2);
    let mut foreign = types.clone();
    foreign.push(types[0].clone());
    foreign[1] = declaration(
        248,
        foreign_type,
        SemanticPointerKindV1::Reference,
        SemanticMutabilityV1::Immutable,
        0,
        64,
        SemanticPointerMetadataV1::None,
    );
    let foreign_projection =
        SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, foreign_type).unwrap();
    assert!(!matches(&foreign, &issued, &foreign_projection, &binding));
    let wrong_result = SemanticProjectionV1::new(
        SemanticProjectionKindV1::Dereference,
        SemanticTypeIdV1::from_index(9),
    )
    .unwrap();
    assert!(!matches(&types, &issued, &wrong_result, &binding));
    let field = SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), lane_type).unwrap();
    assert!(!matches(&types, &issued, &field, &binding));
    assert!(!matches(&types, &BTreeMap::new(), &projection, &binding));
    let wrong_class = BTreeMap::from([(lane_type, SemanticPromotedBindingV1::Ordinary)]);
    assert!(!matches(&types, &wrong_class, &projection, &binding));
    let wrong_width = BTreeMap::from([(
        lane_type,
        SemanticPromotedBindingV1::WaveLane { wave_width: 32 },
    )]);
    assert!(!matches(&types, &wrong_width, &projection, &binding));
    let lane32 = SemanticValueBindingV1::WaveLane {
        value: ValueId(42),
        wave: SemanticCurrentWaveV1::new(32),
    };
    assert!(!matches(&types, &issued, &projection, &lane32));
    assert!(!matches(&types, &wrong_width, &projection, &lane32));
    let plain = SemanticValueBindingV1::Value {
        id: ValueId(42),
        ty: Type::Scalar(ScalarType::U32),
    };
    assert!(!matches(&types, &issued, &projection, &plain));
    assert!(
        matches!(binding, SemanticValueBindingV1::WaveLane { value: ValueId(42), wave } if wave.width == 64)
    );
}

#[test]
fn wave_lane_reference_rvalue_preserves_borrow_but_rejects_a_raw_result() {
    // Private descriptors/map/binding setup; no admitted owner or actual issuance claim.
    let unit = SemanticTypeIdV1::from_index(0);
    let lane = SemanticTypeIdV1::from_index(1);
    let reference = SemanticTypeIdV1::from_index(2);
    let raw = SemanticTypeIdV1::from_index(3);
    let source = SemanticSourceProvenanceV1::unavailable();
    for mutability in [
        SemanticMutabilityV1::Immutable,
        SemanticMutabilityV1::Mutable,
    ] {
        let pointer = |tag, kind| {
            SemanticTypeDeclV1::new(
                SemanticTypeIdentityV1::from_sha256([tag; 32]),
                SemanticLayoutIdentityV1::from_sha256([tag; 32]),
                SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
                SemanticTypeShapeV1::Pointer(
                    SemanticPointerTypeV1::new_with_kind(
                        lane,
                        kind,
                        mutability,
                        0,
                        64,
                        SemanticPointerMetadataV1::None,
                    )
                    .unwrap(),
                ),
            )
        };
        let types = vec![
            unit_type(),
            unsigned_scalar_type(231, 32),
            pointer(232, SemanticPointerKindV1::Reference),
            pointer(233, SemanticPointerKindV1::Raw),
        ];
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([234; 32]),
            SemanticLayoutIdentityV1::from_sha256([235; 32]),
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
            SemanticFunctionIdentityV1::from_sha256([236; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([237; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([238; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([239; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([240; 32]),
            source,
            abi,
            vec![
                local(241, unit, SemanticLocalRoleV1::Return),
                local(242, reference, SemanticLocalRoleV1::Temporary),
            ],
            SemanticBlockIdV1::from_index(0),
            vec![
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([243; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = SemanticFunctionLoweringV1::new(
            &types,
            &[],
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
        lowering
            .control_flow_ssa
            .compiler_issued_bindings
            .insert(lane, SemanticPromotedBindingV1::WaveLane { wave_width: 64 });
        lowering.locals[1] = Some(SemanticValueBindingV1::WaveLane {
            value: ValueId(42),
            wave: SemanticCurrentWaveV1::new(64),
        });
        let next_value = lowering.next_value;
        let place = SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, lane).unwrap()],
            lane,
        )
        .unwrap();
        let borrow = SemanticRvalueKindV1::Borrow {
            kind: match mutability {
                SemanticMutabilityV1::Immutable => SemanticBorrowKindV1::Shared,
                SemanticMutabilityV1::Mutable => SemanticBorrowKindV1::Mutable,
            },
            place: place.clone(),
        };
        let mut operations = Vec::new();
        let result = lowering
            .lower_rvalue(
                SemanticBlockIdV1::from_index(0),
                Some(0),
                reference,
                &borrow,
                &mut operations,
            )
            .unwrap();
        assert!(
            matches!(result, SemanticValueBindingV1::WaveLane { value: ValueId(42), wave } if wave.width == 64)
        );
        let address = SemanticRvalueKindV1::AddressOf { mutability, place };
        assert!(matches!(
            lowering.lower_rvalue(
                SemanticBlockIdV1::from_index(0),
                Some(1),
                raw,
                &address,
                &mut operations
            ),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                function: 0,
                block: Some(0),
                statement: Some(1),
                detail: "wave lane capability cannot form a raw address",
            })
        ));
        assert!(operations.is_empty());
        assert_eq!(lowering.next_value, next_value);
        assert!(
            matches!(lowering.locals[1], Some(SemanticValueBindingV1::WaveLane { value: ValueId(42), wave }) if wave.width == 64)
        );
    }
}
