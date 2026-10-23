    use super::*;
    use dialect_kernel::AccessKindAttr;
    use fe2o3_mir_model::semantic_mir_v1::{InertSemanticMirRequestV1, SemanticMirLimitsV1};
    use fe2o3_mir_model::semantic_mir_v1::{
        SemanticAbiArgumentV1, SemanticAbiIdentityV1, SemanticAbiPassModeV1, SemanticAbiValueV1,
        SemanticAggregateLayoutV1, SemanticAggregateTypeV1, SemanticAssignmentV1,
        SemanticBackendReprV1, SemanticBackendScalarV1, SemanticBasicBlockV1,
        SemanticBlockIdentityV1, SemanticCallDestinationV1, SemanticCallableIdV1,
        SemanticCanonAbiV1, SemanticCompilerIntrinsicIdentityV1,
        SemanticConstGenericArgumentsIdentityV1, SemanticConstantV1, SemanticControlFlowEdgeV1,
        SemanticDirectCallV1, SemanticEdgeRoleV1, SemanticExternAbiV1, SemanticFieldsShapeV1,
        SemanticFunctionAbiV1, SemanticFunctionIdentityV1, SemanticFunctionRoleV1,
        SemanticGenericTypeArgumentsIdentityV1, SemanticItemDefinitionIdentityV1,
        SemanticKernelBindingIdentityV1, SemanticKernelEntryV1, SemanticKernelLaunchBoundsV1,
        SemanticKernelSourceContractV1, SemanticLayoutIdentityV1, SemanticLinkSymbolV1,
        SemanticLocalDeclV1, SemanticLocalIdV1, SemanticLocalIdentityV1,
        SemanticMfmaAccumulatorDistributionV1, SemanticMfmaOperandRoleV1,
        SemanticMfmaRegisterDistributionV1, SemanticMonomorphizationIdentityV1,
        SemanticNonBodyCallableBindingV1, SemanticProjectionV1, SemanticRustcVariantsV1,
        SemanticRvalueV1, SemanticScalarValidityRangeV1, SemanticSourceProvenanceV1,
        SemanticStatementV1, SemanticSwitchTargetV1, SemanticSwitchTargetsV1,
        SemanticTargetDataLayoutV1, SemanticTerminatorV1, SemanticTypeIdentityV1,
        SemanticTypeLayoutDetailsV1, SemanticTypeLayoutV1, SemanticUnwindActionV1,
        SemanticWorkgroupDimensionsV1,
    };
    use fe2o3_pliron::{
        ProductionConstructionV1, ProductionRankedBlockV1, ProductionRankedKernelV1,
        ProductionRankedTerminatorV1, ProductionRankedValueIdV1, ProductionSemanticMirLimitsV1,
        ProductionSessionLimitsV1, compile_ranked_kernel_for_gfx942_lowering_v1,
        compile_ranked_kernel_for_lowering_v1,
    };

    // Private materializer fixture only. It supplies already-classified bindings;
    // it does not mint a production provider, source receipt or convergence proof.
    fn raw_shuffle_materializer_fixture(
        scalar: ScalarType,
        source_scalar: ScalarType,
        source_bound: Option<u128>,
        source_constant: Option<u32>,
        context: SemanticValueBindingV1,
        value_scalar: ScalarType,
        matching_destination: bool,
    ) -> Result<Vec<Operation>, ProductionSemanticKirErrorV1> {
        let unit = SemanticTypeIdV1::from_index(0);
        let element = SemanticTypeIdV1::from_index(1);
        let source_type = SemanticTypeIdV1::from_index(2);
        let source = SemanticSourceProvenanceV1::unavailable();
        let scalar_decl = |tag: u8, scalar: ScalarType| {
            let bits = scalar.bit_width().unwrap();
            SemanticTypeDeclV1::new(
                SemanticTypeIdentityV1::from_sha256([tag; 32]),
                SemanticLayoutIdentityV1::from_sha256([tag + 1; 32]),
                SemanticTypeLayoutV1::new(Some(u64::from(bits / 8)), u64::from(bits / 8)).unwrap(),
                SemanticTypeShapeV1::Scalar(
                    if scalar == ScalarType::F32 || scalar == ScalarType::F64 {
                        SemanticScalarTypeV1::Float { bits }
                    } else {
                        SemanticScalarTypeV1::Integer {
                            signed: scalar.is_signed_integer(),
                            bits,
                        }
                    },
                ),
            )
        };
        let types = [
            unit_type(),
            scalar_decl(101, scalar),
            scalar_decl(103, source_scalar),
        ];
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([105; 32]),
            SemanticLayoutIdentityV1::from_sha256([106; 32]),
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
            SemanticFunctionIdentityV1::from_sha256([107; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([108; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([109; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([110; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([111; 32]),
            source,
            abi,
            vec![
                local(112, unit, SemanticLocalRoleV1::Return),
                local(113, unit, SemanticLocalRoleV1::Temporary),
                local(114, element, SemanticLocalRoleV1::Temporary),
                local(115, source_type, SemanticLocalRoleV1::Temporary),
                local(
                    116,
                    if matching_destination { element } else { unit },
                    SemanticLocalRoleV1::Temporary,
                ),
            ],
            SemanticBlockIdV1::from_index(0),
            vec![
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([117; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        // The current transport planner needs a real shared ledger even when
        // this private fixture has no nominal references or source owner.
        let semantic_function = SemanticFunctionIdV1::from_index(0);
        let semantic_ssa = plan_semantic_function_ssa_with_module_v1(
            semantic_function,
            &function,
            &types,
            &[],
            ProductionSemanticSsaLimitsV1::default(),
        )
        .map_err(ProductionSemanticKirErrorV1::SemanticSsa)?;
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut budget = ArgumentBudgetV1::new(&mut work, 1_000_000);
        let call_returns =
            CallReturnBufferV1::for_function(&function, &[], &BTreeMap::new(), 0, &mut budget)?;
        let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
            &types,
            &[],
            &function,
            &semantic_ssa,
            semantic_function,
            semantic_function,
            BTreeMap::new(),
            BTreeMap::new(),
            Vec::new(),
            SemanticParameterBindingsV1 {
                declarations: &[],
                values: &[],
                types: &[],
                local_bindings: None,
            },
            None,
            Some([64, 1, 1]),
            BTreeSet::new().into(),
            1,
            false,
            64,
            PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(1, 64)),
            None,
            call_returns,
            Some(&mut budget),
            SemanticEmissionPlacementV1::default(),
            None,
            None,
            None,
        )?;
        lowering.next_value = 50;
        lowering.locals[1] = Some(context);
        lowering.locals[2] = Some(SemanticValueBindingV1::Value {
            id: ValueId(40),
            ty: Type::Scalar(value_scalar),
        });
        let mut operations = Vec::new();
        let lane = if let Some(lane) = source_constant {
            let lane = lowering.emit_id(
                &mut operations,
                Type::Scalar(ScalarType::U32),
                OperationKind::Constant(Constant::U32(lane)),
            )?;
            operations.clear();
            lane
        } else {
            if let Some(bound) = source_bound {
                lowering
                    .emitted_unsigned_exclusive_bounds
                    .insert(ValueId(41), bound);
            }
            ValueId(41)
        };
        lowering.locals[3] = Some(SemanticValueBindingV1::Value {
            id: lane,
            ty: Type::Scalar(source_scalar),
        });
        let place =
            |index, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(index), vec![], ty).unwrap();
        let call = SemanticDirectCallV1::new_callable(
            SemanticCallableIdV1::from_index(0),
            vec![
                SemanticOperandV1::Copy(place(1, unit)),
                SemanticOperandV1::Copy(place(2, element)),
                SemanticOperandV1::Copy(place(3, source_type)),
            ],
            Some(SemanticCallDestinationV1::new(
                place(4, if matching_destination { element } else { unit }),
                SemanticControlFlowEdgeV1::new(
                    SemanticEdgeRoleV1::CallReturn,
                    SemanticBlockIdV1::from_index(0),
                ),
            )),
            SemanticUnwindActionV1::Unreachable,
        )
        .unwrap();
        let result = lowering.lower_intrinsic_gfx942_wave64_shuffle_index_v1(
            SemanticBlockIdV1::from_index(0),
            &call,
            element,
            &mut operations,
        )?;
        let (_, result_ty) = result.value().unwrap();
        assert_eq!(result_ty, Type::Scalar(scalar));
        Ok(operations)
    }

    #[test]
    fn raw_wave64_materializer_preserves_exact_bit_transport_and_lane_operand() {
        for scalar in [ScalarType::U32, ScalarType::I32, ScalarType::F32] {
            let operations = raw_shuffle_materializer_fixture(
                scalar,
                ScalarType::U32,
                Some(64),
                None,
                SemanticValueBindingV1::CollectiveContext,
                scalar,
                true,
            )
            .unwrap();
            let shuffle_index = usize::from(scalar != ScalarType::U32);
            assert_eq!(
                operations.len(),
                if scalar == ScalarType::U32 { 1 } else { 3 }
            );
            let OperationKind::Wave(wave) = &operations[shuffle_index].kind else {
                panic!("missing shuffle")
            };
            assert_eq!(wave.width, WaveWidth::Wave64);
            assert_eq!(wave.active_lanes, 64);
            assert_eq!(
                wave.convergence,
                Convergence::uniform(SynchronizationScope::Subgroup)
            );
            let WaveOperationKind::ShuffleIndex {
                value,
                source_lane,
                tile_width,
            } = wave.kind
            else {
                panic!("wrong wave operation")
            };
            assert_eq!(source_lane, ValueId(41));
            assert_eq!(tile_width, 64);
            assert_eq!(
                operations[shuffle_index].results[0].ty,
                Type::Scalar(ScalarType::U32)
            );
            if scalar == ScalarType::U32 {
                assert_eq!(value, ValueId(40));
            } else {
                assert!(matches!(&operations[0].kind, OperationKind::Cast {
                        kind: CastKind::Bitcast, value: ValueId(40), to,
                    } if *to == Type::Scalar(ScalarType::U32)));
                assert_eq!(value, operations[0].results[0].id);
                assert!(matches!(&operations[2].kind, OperationKind::Cast {
                        kind: CastKind::Bitcast, value, to,
                    } if *value == operations[1].results[0].id && *to == Type::Scalar(scalar)));
            }
        }
    }

    #[test]
    fn raw_wave64_materializer_refuses_unissued_context_and_invalid_lane_or_types() {
        for context in [
            SemanticValueBindingV1::Unit,
            SemanticValueBindingV1::MathContext,
        ] {
            let error = raw_shuffle_materializer_fixture(
                ScalarType::F32,
                ScalarType::U32,
                Some(64),
                None,
                context,
                ScalarType::F32,
                true,
            )
            .unwrap_err();
            assert!(
                matches!(&error, ProductionSemanticKirErrorV1::Unsupported {
                    detail: "Wave64 shuffle lacks compiler-issued collective context", ..
                }),
                "expected an actual context refusal, got {error:?}",
            );
        }
        for (source_scalar, bound, constant, scalar, value_scalar, destination) in [
            (
                ScalarType::U32,
                Some(65),
                None,
                ScalarType::F32,
                ScalarType::F32,
                true,
            ),
            (
                ScalarType::U32,
                None,
                Some(64),
                ScalarType::F32,
                ScalarType::F32,
                true,
            ),
            (
                ScalarType::U32,
                None,
                None,
                ScalarType::F32,
                ScalarType::F32,
                true,
            ),
            (
                ScalarType::I32,
                Some(64),
                None,
                ScalarType::F32,
                ScalarType::F32,
                true,
            ),
            (
                ScalarType::U32,
                Some(64),
                None,
                ScalarType::F64,
                ScalarType::F64,
                true,
            ),
            (
                ScalarType::U32,
                Some(64),
                None,
                ScalarType::F32,
                ScalarType::U32,
                true,
            ),
            (
                ScalarType::U32,
                Some(64),
                None,
                ScalarType::F32,
                ScalarType::F32,
                false,
            ),
        ] {
            let error = raw_shuffle_materializer_fixture(
                scalar,
                source_scalar,
                bound,
                constant,
                SemanticValueBindingV1::CollectiveContext,
                value_scalar,
                destination,
            )
            .unwrap_err();
            assert!(
                matches!(&error, ProductionSemanticKirErrorV1::Unsupported {
                    detail: "Wave64 shuffle requires exact u32/i32/f32 and an authenticated u32 lane below 64", ..
                }),
                "expected an actual lane/type refusal, got {error:?}",
            );
        }
    }

    #[test]
    fn raw_wave64_lane_range_rules_are_sound_against_enumeration() {
        let ty = Type::Scalar(ScalarType::U32);
        let lhs = ValueId(0);
        let rhs = ValueId(1);
        // All narrow, asymmetric intervals, including the exact Wave64 boundary.
        for left_bound in 1..=65_u128 {
            for right_bound in 1..=65_u128 {
                let bounds = BTreeMap::from([(lhs, left_bound), (rhs, right_bound)]);
                for op in [BinaryOp::BitXor, BinaryOp::BitAnd] {
                    let bound = wave64_unsigned_bound_v1(
                        &ty,
                        &OperationKind::Binary { op, lhs, rhs },
                        &bounds,
                        &BTreeMap::new(),
                    )
                    .unwrap();
                    for left in 0..left_bound {
                        for right in 0..right_bound {
                            let actual = if op == BinaryOp::BitXor {
                                left ^ right
                            } else {
                                left & right
                            };
                            assert!(actual < bound, "{left} {op:?} {right} >= {bound}");
                        }
                    }
                }
            }
        }
        let bounds = BTreeMap::from([(lhs, 64)]);
        for divisor in [1_u64, 3, 8, 64, 65, u64::from(u32::MAX)] {
            let constants = BTreeMap::from([(rhs, divisor)]);
            let bound = wave64_unsigned_bound_v1(
                &ty,
                &OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs,
                    rhs,
                },
                &bounds,
                &constants,
            )
            .unwrap();
            for lane in 0..64_u64 {
                assert!(u128::from(lane % divisor) < bound);
            }
        }
        assert!(
            wave64_unsigned_bound_v1(
                &ty,
                &OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs,
                    rhs
                },
                &bounds,
                &BTreeMap::from([(rhs, 0)]),
            )
            .is_none()
        );
        assert!(
            wave64_unsigned_bound_v1(
                &ty,
                &OperationKind::Binary {
                    op: BinaryOp::BitXor,
                    lhs,
                    rhs
                },
                &bounds,
                &BTreeMap::new(),
            )
            .is_none()
        );
        assert_eq!(
            wave64_unsigned_bound_v1(
                &Type::Scalar(ScalarType::U64),
                &OperationKind::Cast {
                    kind: CastKind::ZeroExtend,
                    value: lhs,
                    to: Type::Scalar(ScalarType::U64),
                },
                &bounds,
                &BTreeMap::new(),
            ),
            Some(64)
        );
        assert!(
            wave64_unsigned_bound_v1(
                &Type::Scalar(ScalarType::U8),
                &OperationKind::Cast {
                    kind: CastKind::Truncate,
                    value: lhs,
                    to: Type::Scalar(ScalarType::U8),
                },
                &BTreeMap::from([(lhs, 257)]),
                &BTreeMap::new(),
            )
            .is_none()
        );
        assert!(
            wave64_unsigned_bound_v1(
                &Type::Scalar(ScalarType::I32),
                &OperationKind::Constant(Constant::I32(-1)),
                &BTreeMap::new(),
                &BTreeMap::new(),
            )
            .is_none()
        );
    }

    #[test]
    fn workgroup_reduction_accepts_only_the_closed_scalar_and_geometry_contract() {
        assert_eq!(truncate_unsigned_constant_v1(64, &Type::INDEX), Some(64),);
        assert_eq!(
            truncate_unsigned_constant_v1(64, &Type::Scalar(ScalarType::U32)),
            Some(64),
        );
        assert_eq!(
            truncate_unsigned_constant_v1(u64::from(u32::MAX) + 2, &Type::Scalar(ScalarType::U32),),
            Some(1),
        );
        assert_eq!(
            truncate_unsigned_constant_v1(64, &Type::Scalar(ScalarType::I32)),
            None,
        );
        for scalar in [ScalarType::U32, ScalarType::I32, ScalarType::F32] {
            assert!(workgroup_reduction_scalar_supported_v1(&Type::Scalar(
                scalar
            )));
        }
        for scalar in [
            ScalarType::U8,
            ScalarType::U64,
            ScalarType::I64,
            ScalarType::F16,
            ScalarType::F64,
        ] {
            assert!(!workgroup_reduction_scalar_supported_v1(&Type::Scalar(
                scalar
            )));
        }

        for width in [1, 2, 4, 64, 128, 256] {
            assert_eq!(
                validate_workgroup_reduction_geometry_v1(
                    Some([width, 1, 1]),
                    Some(width),
                    Some(width),
                ),
                Ok(width),
            );
            assert_eq!(
                validate_workgroup_reduction_geometry_v1(Some([width, 1, 1]), Some(width), None,),
                Ok(width),
            );
        }
        for geometry in [
            None,
            Some([0, 1, 1]),
            Some([3, 1, 1]),
            Some([257, 1, 1]),
            Some([64, 2, 1]),
            Some([64, 1, 2]),
        ] {
            assert!(
                validate_workgroup_reduction_geometry_v1(geometry, Some(64), Some(64)).is_err()
            );
        }
        for compiler_slots in [None, Some(0), Some(63), Some(65), Some(256)] {
            assert!(
                validate_workgroup_reduction_geometry_v1(
                    Some([64, 1, 1]),
                    compiler_slots,
                    Some(64),
                )
                .is_err()
            );
        }
        for scratch_slots in [Some(0), Some(63), Some(65), Some(256)] {
            assert!(
                validate_workgroup_reduction_geometry_v1(
                    Some([64, 1, 1]),
                    Some(64),
                    scratch_slots,
                )
                .is_err()
            );
        }
    }

    #[test]
    fn subgroup_broadcast_accepts_local_u32_mask_below_wave64_width() {
        let unknown = ValueId(0);
        let mask = ValueId(1);
        let masked = ValueId(2);
        let constant = Operation::effect_free(
            ValueDef::new(mask, Type::Scalar(ScalarType::U32)),
            OperationKind::Constant(Constant::U32(63)),
        );
        for kind in [
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs: unknown,
                rhs: mask,
            },
            OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs: mask,
                rhs: unknown,
            },
        ] {
            let operations = [
                constant.clone(),
                Operation::effect_free(ValueDef::new(masked, Type::Scalar(ScalarType::U32)), kind),
            ];
            assert!(subgroup_broadcast_source_is_statically_bounded(
                &operations,
                masked,
                64,
            ));
        }
        assert!(subgroup_broadcast_source_is_statically_bounded(
            &[constant],
            mask,
            64,
        ));
    }

    #[test]
    fn subgroup_broadcast_rejects_mask_at_width_and_missing_constant_mask() {
        let unknown = ValueId(0);
        let mask = ValueId(1);
        let masked = ValueId(2);
        let mask64 = Operation::effect_free(
            ValueDef::new(mask, Type::Scalar(ScalarType::U32)),
            OperationKind::Constant(Constant::U32(64)),
        );
        let bitand = |rhs| {
            Operation::effect_free(
                ValueDef::new(masked, Type::Scalar(ScalarType::U32)),
                OperationKind::Binary {
                    op: BinaryOp::BitAnd,
                    lhs: unknown,
                    rhs,
                },
            )
        };
        assert!(!subgroup_broadcast_source_is_statically_bounded(
            &[mask64, bitand(mask)],
            masked,
            64,
        ));
        assert!(!subgroup_broadcast_source_is_statically_bounded(
            &[bitand(ValueId(3))],
            masked,
            64,
        ));
    }

    #[derive(Clone, Copy)]
    struct AuthenticatedInductionFixtureV1 {
        bits: u16,
        bound: u128,
        step: u128,
        extra_write: bool,
        bypass_guard: bool,
    }

    impl Default for AuthenticatedInductionFixtureV1 {
        fn default() -> Self {
            Self {
                bits: 64,
                bound: 64,
                step: 1,
                extra_write: false,
                bypass_guard: false,
            }
        }
    }

    fn authenticated_induction_fixture_v1(
        options: AuthenticatedInductionFixtureV1,
    ) -> (Vec<SemanticTypeDeclV1>, SemanticFunctionDeclV1) {
        let unit = SemanticTypeIdV1::from_index(0);
        let induction_ty = SemanticTypeIdV1::from_index(1);
        let bool_ty = SemanticTypeIdV1::from_index(2);
        let u32_ty = SemanticTypeIdV1::from_index(3);
        let source = SemanticSourceProvenanceV1::unavailable();
        let size = u8::try_from(options.bits / 8).unwrap();
        let place = |local, ty| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], ty).unwrap();
        let operand = |local, ty| SemanticOperandV1::Copy(place(local, ty));
        let constant = |ty, value| {
            SemanticOperandV1::Constant(SemanticConstantV1::new(
                ty,
                SemanticConstantValueV1::Scalar(SemanticScalarValueV1::new(value, size).unwrap()),
            ))
        };
        let assign = |local, ty, value| {
            SemanticStatementV1::new(
                source,
                SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                    place(local, ty),
                    SemanticRvalueV1::new(ty, value),
                )),
            )
        };
        let edge = |role, target| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(target));
        let block = |tag, statements, terminator| {
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([tag; 32]),
                source,
                statements,
                SemanticTerminatorV1::new(source, terminator),
            )
            .unwrap()
        };
        let entry = block(
            210,
            vec![assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Use(constant(induction_ty, 0)),
            )],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
        );
        let header = block(
            211,
            vec![assign(
                2,
                bool_ty,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::LessThan,
                    left: operand(1, induction_ty),
                    right: constant(induction_ty, options.bound),
                },
            )],
            SemanticTerminatorKindV1::SwitchInt {
                discriminant: operand(2, bool_ty),
                targets: SemanticSwitchTargetsV1::new(
                    vec![SemanticSwitchTargetV1::new(
                        0,
                        edge(SemanticEdgeRoleV1::SwitchValue, 4),
                    )],
                    edge(SemanticEdgeRoleV1::SwitchOtherwise, 2),
                )
                .unwrap(),
            },
        );
        let mut body_statements = vec![assign(
            3,
            u32_ty,
            SemanticRvalueKindV1::Cast {
                kind: SemanticCastKindV1::Integer,
                operand: operand(1, induction_ty),
            },
        )];
        if options.extra_write {
            body_statements.push(assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Use(constant(induction_ty, 0)),
            ));
        }
        let body = block(
            212,
            body_statements,
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 3)),
        );
        let latch = block(
            213,
            vec![assign(
                1,
                induction_ty,
                SemanticRvalueKindV1::Binary {
                    operation: SemanticBinaryOpV1::Add,
                    left: operand(1, induction_ty),
                    right: constant(induction_ty, options.step),
                },
            )],
            SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 1)),
        );
        let exit = block(
            214,
            vec![],
            if options.bypass_guard {
                SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto, 2))
            } else {
                SemanticTerminatorKindV1::Return
            },
        );
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([215; 32]),
            SemanticLayoutIdentityV1::from_sha256([216; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let locals: Vec<SemanticLocalDeclV1> = [
            (unit, SemanticLocalRoleV1::Return),
            (induction_ty, SemanticLocalRoleV1::Temporary),
            (bool_ty, SemanticLocalRoleV1::Temporary),
            (u32_ty, SemanticLocalRoleV1::Temporary),
        ]
        .into_iter()
        .enumerate()
        .map(|(local, (ty, role))| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([217 + local as u8; 32]),
                ty,
                role,
                source,
            )
        })
        .collect();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([221; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([222; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([223; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([224; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([225; 32]),
            source,
            abi,
            locals,
            SemanticBlockIdV1::from_index(0),
            vec![entry, header, body, latch, exit],
        )
        .unwrap();
        (
            vec![
                unit_type(),
                unsigned_scalar_type(226, options.bits),
                bool_type(),
                unsigned_scalar_type(228, 32),
            ],
            function,
        )
    }

    #[test]
    fn ranked_canonical_induction_bound_survives_exact_u64_to_u32_cast() {
        let (types, function) =
            authenticated_induction_fixture_v1(AuthenticatedInductionFixtureV1::default());
        let bounds = authenticated_loop_induction_bounds_v1(&types, &function).unwrap();
        assert_eq!(bounds.get(&(2, 1)), Some(&64));
        let alias = SemanticOperandV1::Copy(
            SemanticPlaceV1::new(
                SemanticLocalIdV1::from_index(3),
                vec![],
                SemanticTypeIdV1::from_index(3),
            )
            .unwrap(),
        );
        assert_eq!(
            authenticated_unsigned_operand_exclusive_bound_v1(
                &types,
                &function,
                &bounds,
                SemanticBlockIdV1::from_index(2),
                &alias,
            ),
            Some(64),
        );
    }

    #[test]
    fn authenticated_induction_bound_rejects_width_overrun_and_hostile_loops() {
        let (types, function) =
            authenticated_induction_fixture_v1(AuthenticatedInductionFixtureV1 {
                bound: 65,
                ..AuthenticatedInductionFixtureV1::default()
            });
        let bounds = authenticated_loop_induction_bounds_v1(&types, &function).unwrap();
        assert_eq!(bounds.get(&(2, 1)), Some(&65));
        assert!(!authenticated_subgroup_broadcast_source_is_bounded(
            *bounds.get(&(2, 1)).unwrap(),
            64,
        ));
        assert!(!authenticated_subgroup_broadcast_source_is_bounded(0, 0));

        for options in [
            AuthenticatedInductionFixtureV1 {
                extra_write: true,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                step: 0,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                bits: 8,
                bound: 255,
                step: 2,
                ..AuthenticatedInductionFixtureV1::default()
            },
            AuthenticatedInductionFixtureV1 {
                bypass_guard: true,
                ..AuthenticatedInductionFixtureV1::default()
            },
        ] {
            let (types, function) = authenticated_induction_fixture_v1(options);
            assert!(
                authenticated_loop_induction_bounds_v1(&types, &function)
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn inactive_launch_axes_use_canonical_identity_and_extent_constants() {
        for rank in 1..=3 {
            for (axis, axis_rank) in [(Axis::X, 1), (Axis::Y, 2), (Axis::Z, 3)] {
                for kind in [IndexKind::Global, IndexKind::Workgroup, IndexKind::Local] {
                    assert_eq!(
                        inactive_launch_axis_value_v1(rank, kind, axis),
                        (axis_rank > rank).then_some(0),
                    );
                }
                for kind in [IndexKind::WorkgroupSize, IndexKind::WorkgroupCount] {
                    assert_eq!(
                        inactive_launch_axis_value_v1(rank, kind, axis),
                        (axis_rank > rank).then_some(1),
                    );
                }
            }
        }
    }

    #[test]
    fn semantic_casts_use_the_shared_bounded_kernel_ir_index_paths() {
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U32),
                &Type::INDEX,
            ),
            Some([Some((CastKind::ZeroExtend, ScalarType::Index)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U64),
                &Type::INDEX,
            ),
            Some([Some((CastKind::Bitcast, ScalarType::Index)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::U64),
            ),
            Some([Some((CastKind::Bitcast, ScalarType::U64)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::I32),
                &Type::INDEX,
            ),
            Some([
                Some((CastKind::SignExtend, ScalarType::U64)),
                Some((CastKind::Bitcast, ScalarType::Index)),
            ])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::U32),
            ),
            Some([
                Some((CastKind::Bitcast, ScalarType::U64)),
                Some((CastKind::Truncate, ScalarType::U32)),
            ])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::INDEX,
                &Type::Scalar(ScalarType::F64),
            ),
            None
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Float,
                &Type::Scalar(ScalarType::U64),
                &Type::INDEX,
            ),
            None
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Integer,
                &Type::Scalar(ScalarType::U32),
                &Type::Scalar(ScalarType::U64),
            ),
            Some([Some((CastKind::ZeroExtend, ScalarType::U64)), None])
        );
        assert_eq!(
            lower_cast_path(
                SemanticCastKindV1::Transmute,
                &Type::Scalar(ScalarType::U32),
                &Type::Scalar(ScalarType::F32),
            ),
            Some([Some((CastKind::Bitcast, ScalarType::F32)), None])
        );
        for (from, to) in [
            (Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::U64)),
            (Type::Scalar(ScalarType::U32), Type::Unit),
        ] {
            assert_eq!(
                lower_cast_path(SemanticCastKindV1::Transmute, &from, &to),
                None
            );
        }
    }
