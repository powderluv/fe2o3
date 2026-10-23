use super::*;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticAggregateLayoutV1, SemanticAggregateTypeV1, SemanticPointerTypeV1,
};

fn scope_pointer(
    pointee: SemanticTypeIdV1,
    kind: SemanticPointerKindV1,
    mutability: SemanticMutabilityV1,
    space: u32,
    width: u16,
    metadata: SemanticPointerMetadataV1,
) -> SemanticTypeDeclV1 {
    SemanticTypeDeclV1::new(
        SemanticTypeIdentityV1::from_sha256([232; 32]),
        SemanticLayoutIdentityV1::from_sha256([233; 32]),
        SemanticTypeLayoutV1::new(Some(8), 8).unwrap(),
        SemanticTypeShapeV1::Pointer(
            SemanticPointerTypeV1::new_with_kind(pointee, kind, mutability, space, width, metadata)
                .unwrap(),
        ),
    )
}

fn scope_component_types(mutability: SemanticMutabilityV1) -> Vec<SemanticTypeDeclV1> {
    let scope = SemanticTypeIdV1::from_index(1);
    vec![
        unit_type(),
        SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([230; 32]),
            SemanticLayoutIdentityV1::from_sha256([231; 32]),
            SemanticTypeLayoutV1::aggregate(
                Some(0),
                1,
                SemanticAggregateLayoutV1::new(vec![], vec![]).unwrap(),
            )
            .unwrap(),
            SemanticTypeShapeV1::Aggregate(SemanticAggregateTypeV1::new(vec![]).unwrap()),
        ),
        scope_pointer(
            scope,
            SemanticPointerKindV1::Reference,
            mutability,
            0,
            64,
            SemanticPointerMetadataV1::None,
        ),
        scope_pointer(
            scope,
            SemanticPointerKindV1::Raw,
            mutability,
            0,
            64,
            SemanticPointerMetadataV1::None,
        ),
    ]
}

#[test]
fn issued_scope_reference_requires_exact_nominal_authentication() {
    // Private descriptors/map/binding setup, not an admitted owner or actual issuance.
    let scope = SemanticTypeIdV1::from_index(1);
    let reference = SemanticTypeIdV1::from_index(2);
    let projection =
        SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, scope).unwrap();
    let issued = BTreeMap::from([(scope, SemanticPromotedBindingV1::WorkgroupLdsScope)]);
    let binding = SemanticValueBindingV1::WorkgroupLdsScope;
    for mutability in [
        SemanticMutabilityV1::Immutable,
        SemanticMutabilityV1::Mutable,
    ] {
        let types = scope_component_types(mutability);
        let matches = |types: &[SemanticTypeDeclV1],
                       registry: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
                       projection: &SemanticProjectionV1,
                       binding: &SemanticValueBindingV1| {
            issued_capability_reference_dereference_matches_v1(
                types, registry, reference, projection, binding,
            )
        };
        assert!(matches(&types, &issued, &projection, &binding));
        assert!(!matches(&types, &BTreeMap::new(), &projection, &binding));
        for class in [
            SemanticPromotedBindingV1::Ordinary,
            SemanticPromotedBindingV1::MathContext,
            SemanticPromotedBindingV1::WaveLane { wave_width: 64 },
        ] {
            assert!(!matches(
                &types,
                &BTreeMap::from([(scope, class)]),
                &projection,
                &binding,
            ));
        }
        for other in [
            SemanticValueBindingV1::MathContext,
            SemanticValueBindingV1::Aggregate(vec![]),
            SemanticValueBindingV1::Value {
                id: ValueId(42),
                ty: Type::INDEX,
            },
        ] {
            assert!(!matches(&types, &issued, &projection, &other));
        }
        for invalid in [
            types[3].clone(),
            scope_pointer(
                scope,
                SemanticPointerKindV1::Reference,
                mutability,
                1,
                64,
                SemanticPointerMetadataV1::None,
            ),
            scope_pointer(
                scope,
                SemanticPointerKindV1::Reference,
                mutability,
                0,
                32,
                SemanticPointerMetadataV1::None,
            ),
            scope_pointer(
                scope,
                SemanticPointerKindV1::Reference,
                mutability,
                0,
                64,
                SemanticPointerMetadataV1::SliceLength,
            ),
            types[0].clone(),
        ] {
            let mut wrong = types.clone();
            wrong[2] = invalid;
            assert!(!matches(&wrong, &issued, &projection, &binding));
        }
        assert!(!matches(&types[..2], &issued, &projection, &binding));
        for invalid in [
            SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), scope).unwrap(),
            SemanticProjectionV1::new(
                SemanticProjectionKindV1::Dereference,
                SemanticTypeIdV1::from_index(0),
            )
            .unwrap(),
        ] {
            assert!(!matches(&types, &issued, &invalid, &binding));
        }
        let foreign = SemanticTypeIdV1::from_index(4);
        let mut same_shape = types.clone();
        same_shape.push(types[1].clone());
        same_shape[2] = scope_pointer(
            foreign,
            SemanticPointerKindV1::Reference,
            mutability,
            0,
            64,
            SemanticPointerMetadataV1::None,
        );
        let foreign_projection =
            SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, foreign).unwrap();
        assert!(!matches(
            &same_shape,
            &issued,
            &foreign_projection,
            &binding
        ));
    }
}

#[test]
fn issued_scope_reference_rvalue_preserves_borrow_and_rejects_raw_result() {
    // Existing private lowerer setup with a seeded registry and binding, not source admission.
    let unit = SemanticTypeIdV1::from_index(0);
    let scope = SemanticTypeIdV1::from_index(1);
    let reference = SemanticTypeIdV1::from_index(2);
    let raw = SemanticTypeIdV1::from_index(3);
    let source = SemanticSourceProvenanceV1::unavailable();
    for mutability in [
        SemanticMutabilityV1::Immutable,
        SemanticMutabilityV1::Mutable,
    ] {
        let types = scope_component_types(mutability);
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
            .insert(scope, SemanticPromotedBindingV1::WorkgroupLdsScope);
        lowering.locals[1] = Some(SemanticValueBindingV1::WorkgroupLdsScope);
        let next_value = lowering.next_value;
        let place = SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Dereference, scope).unwrap()],
            scope,
        )
        .unwrap();
        let mut operations = Vec::new();
        let borrow = SemanticRvalueKindV1::Borrow {
            kind: match mutability {
                SemanticMutabilityV1::Immutable => SemanticBorrowKindV1::Shared,
                SemanticMutabilityV1::Mutable => SemanticBorrowKindV1::Mutable,
            },
            place: place.clone(),
        };
        assert!(matches!(
            lowering
                .lower_rvalue(
                    SemanticBlockIdV1::from_index(0),
                    Some(0),
                    reference,
                    &borrow,
                    &mut operations,
                )
                .unwrap(),
            SemanticValueBindingV1::WorkgroupLdsScope
        ));
        let address = SemanticRvalueKindV1::AddressOf { mutability, place };
        assert!(matches!(
            lowering.lower_rvalue(
                SemanticBlockIdV1::from_index(0),
                Some(1),
                raw,
                &address,
                &mut operations,
            ),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                function: 0,
                block: Some(0),
                statement: Some(1),
                detail: "workgroup LDS scope capability cannot form a raw address",
            })
        ));
        assert!(operations.is_empty());
        assert_eq!(lowering.next_value, next_value);
        assert!(matches!(
            lowering.locals[1],
            Some(SemanticValueBindingV1::WorkgroupLdsScope)
        ));
    }
}
