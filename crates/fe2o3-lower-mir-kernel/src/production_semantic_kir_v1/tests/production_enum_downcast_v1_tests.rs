mod enum_downcast_observation_tests_v1 {
    use super::*;

    fn function_for_type(
        template: &SemanticFunctionDeclV1,
        ty: SemanticTypeIdV1,
    ) -> SemanticFunctionDeclV1 {
        let source = SemanticSourceProvenanceV1::unavailable();
        SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([151; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([152; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([153; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([154; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([155; 32]),
            source,
            template.abi().clone(),
            vec![
                template.locals()[0].clone(),
                SemanticLocalDeclV1::new(
                    SemanticLocalIdentityV1::from_sha256([156; 32]),
                    ty,
                    SemanticLocalRoleV1::Temporary,
                    source,
                ),
            ],
            SemanticBlockIdV1::from_index(0),
            vec![
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([157; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn lowering<'a, 'service, 'work>(
        types: &'a [SemanticTypeDeclV1],
        function: &'a SemanticFunctionDeclV1,
        budget: &'service mut ArgumentBudgetV1<'work>,
    ) -> SemanticFunctionLoweringV1<'a, 'service> {
        SemanticFunctionLoweringV1::new(
            types,
            &[],
            function,
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
            32,
            budget,
        )
        .unwrap()
    }

    fn projection(prefix: Vec<SemanticProjectionV1>) -> SemanticPlaceV1 {
        let mut projections = prefix;
        projections.extend([
            SemanticProjectionV1::new(
                SemanticProjectionKindV1::Downcast(0),
                SemanticTypeIdV1::from_index(2),
            )
            .unwrap(),
            SemanticProjectionV1::new(
                SemanticProjectionKindV1::Field(0),
                SemanticTypeIdV1::from_index(1),
            )
            .unwrap(),
        ]);
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            projections,
            SemanticTypeIdV1::from_index(1),
        )
        .unwrap()
    }

    #[test]
    fn source_constructor_remains_known_but_payload_presence_does_not_authenticate() {
        let (types, template, _) = call_produced_enum_source_fixture_v1(false);
        let ty = SemanticTypeIdV1::from_index(2);
        let function = function_for_type(&template, ty);
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = lowering(&types, &function, &mut lowering_budget);
        let mut operations = Vec::new();
        let constructor = SemanticRvalueKindV1::aggregate(
            SemanticAggregateKindV1::EnumVariant(0),
            vec![SemanticOperandV1::Constant(SemanticConstantV1::new(
                SemanticTypeIdV1::from_index(1),
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(17, 4).unwrap()),
            ))],
        )
        .unwrap();
        let known = lowering
            .lower_rvalue(
                SemanticBlockIdV1::from_index(0),
                Some(0),
                ty,
                &constructor,
                &mut operations,
            )
            .unwrap();
        lowering.locals[1] = Some(known.clone());
        let count = operations.len();
        let next = lowering.next_value;
        assert!(matches!(
            lowering.resolve_place(
                SemanticBlockIdV1::from_index(0),
                Some(1),
                &projection(vec![]),
                &mut operations,
            ),
            Ok(SemanticValueBindingV1::Value { .. })
        ));

        // Ordinary component data, not a fabricated compiler capability.
        let mut unknown = known;
        let SemanticValueBindingV1::Enum { variant, .. } = &mut unknown else {
            panic!("constructor must retain its exact variant");
        };
        *variant = None;
        for prefix in [
            vec![],
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::OpaqueCast, ty).unwrap()],
        ] {
            lowering.locals[1] = Some(unknown.clone());
            assert!(matches!(
                lowering.resolve_place(
                    SemanticBlockIdV1::from_index(0),
                    Some(2),
                    &projection(prefix),
                    &mut operations,
                ),
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "enum downcast lacks an authenticated variant",
                    ..
                })
            ));
            assert!(matches!(
                lowering.locals[1],
                Some(SemanticValueBindingV1::Enum { variant: None, .. })
            ));
            assert_eq!(operations.len(), count);
            assert_eq!(lowering.next_value, next);
        }
    }

    #[test]
    fn a_known_outer_variant_does_not_authenticate_an_unknown_nested_enum() {
        let (mut types, template, _) = call_produced_enum_source_fixture_v1(false);
        let inner = SemanticTypeIdV1::from_index(2);
        let outer = SemanticTypeIdV1::from_index(3);
        types.push(SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([158; 32]),
            SemanticLayoutIdentityV1::from_sha256([159; 32]),
            SemanticTypeLayoutV1::new(Some(12), 4).unwrap(),
            SemanticTypeShapeV1::Enum {
                discriminant: SemanticTypeIdV1::from_index(1),
                variants: vec![SemanticEnumVariantV1::new(
                    0,
                    SemanticAggregateTypeV1::new(vec![inner]).unwrap(),
                )]
                .into_boxed_slice(),
            },
        ));
        let function = function_for_type(&template, outer);
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = lowering(&types, &function, &mut lowering_budget);
        let nested = SemanticValueBindingV1::Enum {
            discriminant: ValueId(40),
            discriminant_ty: Type::Scalar(ScalarType::U32),
            semantic_type: inner,
            variant: None,
            payloads: BTreeMap::from([(
                0,
                vec![SemanticValueBindingV1::Value {
                    id: ValueId(41),
                    ty: Type::Scalar(ScalarType::U32),
                }],
            )]),
        };
        lowering.locals[1] = Some(SemanticValueBindingV1::Enum {
            discriminant: ValueId(42),
            discriminant_ty: Type::Scalar(ScalarType::U32),
            semantic_type: outer,
            variant: Some(0),
            payloads: BTreeMap::from([(0, vec![nested])]),
        });
        let place = projection(vec![
            SemanticProjectionV1::new(SemanticProjectionKindV1::Downcast(0), outer).unwrap(),
            SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), inner).unwrap(),
        ]);
        let mut operations = Vec::new();
        let next = lowering.next_value;
        assert!(matches!(
            lowering.resolve_place(
                SemanticBlockIdV1::from_index(0),
                Some(0),
                &place,
                &mut operations,
            ),
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "enum downcast lacks an authenticated variant",
                ..
            })
        ));
        assert!(operations.is_empty());
        assert_eq!(lowering.next_value, next);
    }
}
