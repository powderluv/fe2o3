mod saturating_integer_v1_tests {
    use super::*;
    use SemanticSaturatingIntegerOpV1::{Add, Subtract};
    use fe2o3_mir_model::semantic_mir_v1::*;

    const SCALARS: [ScalarType; 8] = [
        ScalarType::I8,
        ScalarType::U8,
        ScalarType::I16,
        ScalarType::U16,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::I64,
        ScalarType::U64,
    ];

    fn emit(
        scalar: ScalarType,
        op: SemanticSaturatingIntegerOpV1,
        limit: usize,
    ) -> Result<Vec<Operation>, ProductionSemanticKirErrorV1> {
        emit_with_source_bound(scalar, op, limit, None).map(|(operations, _)| operations)
    }

    fn emit_with_source_bound(
        scalar: ScalarType,
        op: SemanticSaturatingIntegerOpV1,
        limit: usize,
        source_bound: Option<u128>,
    ) -> Result<(Vec<Operation>, Option<u128>), ProductionSemanticKirErrorV1> {
        let unit = SemanticTypeIdV1::from_index(0);
        let source = SemanticSourceProvenanceV1::unavailable();
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([220; 32]),
            SemanticLayoutIdentityV1::from_sha256([221; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([222; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([223; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([224; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([225; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([226; 32]),
            source,
            abi,
            vec![SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([227; 32]),
                unit,
                SemanticLocalRoleV1::Return,
                source,
            )],
            SemanticBlockIdV1::from_index(0),
            vec![
                SemanticBasicBlockV1::new(
                    SemanticBlockIdentityV1::from_sha256([228; 32]),
                    source,
                    vec![],
                    SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let types = [unit_type()];
        // Keep the fixture constructor and materialization on one finite ledger.
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
        let call_returns = CallReturnBufferV1::for_function(
            &function,
            &[],
            &BTreeMap::new(),
            0,
            &mut budget,
        )?;
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
            None,
            BTreeSet::new().into(),
            1,
            false,
            limit,
            PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(1, limit)),
            None,
            call_returns,
            Some(&mut budget),
            SemanticEmissionPlacementV1::default(),
            None,
            None,
            None,
        )?;
        lowering.next_value = 2;
        if let Some(bound) = source_bound {
            lowering
                .emitted_unsigned_exclusive_bounds
                .insert(ValueId(0), bound);
        }
        let mut operations = Vec::new();
        let result = lowering.emit_saturating_integer_v1(
            &mut operations,
            Type::Scalar(scalar),
            op,
            ValueId(0),
            ValueId(1),
        )?;
        let (value, _) = result.value().unwrap();
        let result_bound = lowering
            .emitted_unsigned_exclusive_bounds
            .get(&value)
            .copied();
        Ok((operations, result_bound))
    }

    #[test]
    fn wave64_lane_bound_attaches_only_to_final_unsigned_saturating_subtraction() {
        let (_, bound) = emit_with_source_bound(ScalarType::U32, Subtract, 11, Some(64)).unwrap();
        assert_eq!(bound, Some(64));
        for (scalar, operation, source_bound) in [
            (ScalarType::U32, Add, Some(64)),
            (ScalarType::I32, Subtract, Some(64)),
            (ScalarType::U32, Subtract, None),
        ] {
            let (_, bound) = emit_with_source_bound(scalar, operation, 11, source_bound).unwrap();
            assert_eq!(bound, None);
        }
    }

    fn signed(value: u128, bits: u16) -> i128 {
        let sign = 1_u128 << (bits - 1);
        if value & sign == 0 {
            value as i128
        } else {
            value as i128 - (1_i128 << bits)
        }
    }

    #[test]
    fn saturation_real_call_dispatch_preserves_two_operands_and_continuation() {
        let source = SemanticSourceProvenanceV1::unavailable();
        let unit = SemanticTypeIdV1::from_index(0);
        let scalar_ty = SemanticTypeIdV1::from_index(1);
        for scalar in SCALARS {
            for operation in [Add, Subtract] {
                let types = [
                    unit_type(),
                    integer_type(230, scalar.is_signed_integer(), scalar.bit_width().unwrap()),
                ];
                let scalar_value = || {
                    SemanticAbiValueV1::new(
                        scalar_ty,
                        SemanticAbiPassModeV1::Direct(SemanticAbiValueAttributesV1::plain()),
                    )
                };
                let abi = |result| {
                    SemanticFunctionAbiV1::from_rustc(
                        SemanticAbiIdentityV1::from_sha256([231; 32]),
                        SemanticLayoutIdentityV1::from_sha256([232; 32]),
                        SemanticCanonAbiV1::Rust,
                        SemanticExternAbiV1::Rust,
                        false,
                        false,
                        2,
                        vec![
                            SemanticAbiArgumentV1::source(scalar_value()),
                            SemanticAbiArgumentV1::source(scalar_value()),
                        ],
                        result,
                    )
                    .unwrap()
                };
                let callables = [SemanticCallableDeclV1::CompilerIntrinsic {
                    binding: SemanticNonBodyCallableBindingV1::new(
                        SemanticFunctionIdentityV1::from_sha256([233; 32]),
                        SemanticItemDefinitionIdentityV1::from_sha256([234; 32]),
                        SemanticMonomorphizationIdentityV1::from_sha256([235; 32]),
                        SemanticGenericTypeArgumentsIdentityV1::from_sha256([236; 32]),
                        SemanticConstGenericArgumentsIdentityV1::from_sha256([237; 32]),
                        source,
                        abi(scalar_value()),
                    ),
                    operation: SemanticCompilerIntrinsicOperationV1::SaturatingInteger(operation),
                    operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([238; 32]),
                }];
                let place = |index| {
                    SemanticPlaceV1::new(SemanticLocalIdV1::from_index(index), vec![], scalar_ty)
                        .unwrap()
                };
                let call = SemanticDirectCallV1::new_callable(
                    SemanticCallableIdV1::from_index(0),
                    vec![
                        SemanticOperandV1::Copy(place(1)),
                        SemanticOperandV1::Move(place(2)),
                    ],
                    Some(SemanticCallDestinationV1::new(
                        place(3),
                        SemanticControlFlowEdgeV1::new(
                            SemanticEdgeRoleV1::CallReturn,
                            SemanticBlockIdV1::from_index(1),
                        ),
                    )),
                    SemanticUnwindActionV1::Unreachable,
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
                let block = |tag, terminator| {
                    SemanticBasicBlockV1::new(
                        SemanticBlockIdentityV1::from_sha256([tag; 32]),
                        source,
                        vec![],
                        SemanticTerminatorV1::new(source, terminator),
                    )
                    .unwrap()
                };
                let function = SemanticFunctionDeclV1::new(
                    SemanticFunctionIdentityV1::from_sha256([239; 32]),
                    SemanticFunctionRoleV1::InternalHelper,
                    SemanticItemDefinitionIdentityV1::from_sha256([240; 32]),
                    SemanticMonomorphizationIdentityV1::from_sha256([241; 32]),
                    SemanticGenericTypeArgumentsIdentityV1::from_sha256([242; 32]),
                    SemanticConstGenericArgumentsIdentityV1::from_sha256([243; 32]),
                    source,
                    abi(SemanticAbiValueV1::new(unit, SemanticAbiPassModeV1::Ignore)),
                    vec![
                        local(244, unit, SemanticLocalRoleV1::Return),
                        local(245, scalar_ty, SemanticLocalRoleV1::Argument(0)),
                        local(246, scalar_ty, SemanticLocalRoleV1::Argument(1)),
                        local(247, scalar_ty, SemanticLocalRoleV1::Temporary),
                    ],
                    SemanticBlockIdV1::from_index(0),
                    vec![
                        block(248, SemanticTerminatorKindV1::Call(call.clone())),
                        block(249, SemanticTerminatorKindV1::Return),
                    ],
                )
                .unwrap();
                // Keep the fixture constructor and materialization on one finite ledger.
                let semantic_function = SemanticFunctionIdV1::from_index(0);
                let semantic_ssa = plan_semantic_function_ssa_with_module_v1(
                    semantic_function,
                    &function,
                    &types,
                    &callables,
                    ProductionSemanticSsaLimitsV1::default(),
                )
                .unwrap();
                let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut budget = ArgumentBudgetV1::new(&mut work, 1_000_000);
                let call_returns = CallReturnBufferV1::for_function(
                    &function,
                    &callables,
                    &BTreeMap::new(),
                    0,
                    &mut budget,
                )
                .unwrap();
                let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
                    &types,
                    &callables,
                    &function,
                    &semantic_ssa,
                    semantic_function,
                    semantic_function,
                    BTreeMap::new(),
                    BTreeMap::new(),
                    Vec::new(),
                    SemanticParameterBindingsV1 {
                        declarations: &[(0, 1, scalar_ty), (1, 2, scalar_ty)],
                        values: &[ValueId(0), ValueId(1)],
                        types: &[Type::Scalar(scalar), Type::Scalar(scalar)],
                        local_bindings: None,
                    },
                    None,
                    None,
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
                )
                .unwrap();
                let mut operations = Vec::new();
                let terminator = lowering
                    .lower_call(SemanticBlockIdV1::from_index(0), &call, &mut operations)
                    .unwrap();
                assert!(matches!(
                    terminator,
                    Terminator::Branch {
                        target: BlockId(1),
                        ..
                    }
                ));
                assert_eq!(
                    operations.len(),
                    if scalar.is_signed_integer() { 11 } else { 4 }
                );
                assert!(matches!(
                    operations[0].kind,
                    OperationKind::Binary {
                        op: BinaryOp::Checked(_),
                        lhs: ValueId(0),
                        rhs: ValueId(1)
                    }
                ));
            }
        }
    }

    fn reference(
        scalar: ScalarType,
        op: SemanticSaturatingIntegerOpV1,
        lhs: u128,
        rhs: u128,
    ) -> u128 {
        let bits = scalar.bit_width().unwrap();
        let mask = (1_u128 << bits) - 1;
        if scalar.is_signed_integer() {
            let (lhs, rhs) = (signed(lhs, bits), signed(rhs, bits));
            let wide = match op {
                Add => lhs + rhs,
                Subtract => lhs - rhs,
            };
            wide.clamp(-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1) as u128 & mask
        } else {
            match op {
                Add => (lhs + rhs).min(mask),
                Subtract => lhs.saturating_sub(rhs),
            }
        }
    }

    fn evaluate(operations: &[Operation], scalar: ScalarType, lhs: u128, rhs: u128) -> u128 {
        let bits = scalar.bit_width().unwrap();
        let mask = (1_u128 << bits) - 1;
        let mut values = BTreeMap::from([(ValueId(0), lhs), (ValueId(1), rhs)]);
        for operation in operations {
            let result = match &operation.kind {
                OperationKind::Constant(constant) => {
                    u128::from(normalize_kir_constant_v1(constant).unwrap().1)
                }
                OperationKind::Binary { op, lhs, rhs } => {
                    let (a, b) = (values[lhs], values[rhs]);
                    match op {
                        BinaryOp::Checked(CheckedBinaryOperator::Add) => a.wrapping_add(b) & mask,
                        BinaryOp::Checked(CheckedBinaryOperator::Subtract) => {
                            a.wrapping_sub(b) & mask
                        }
                        BinaryOp::BitXor => a ^ b,
                        BinaryOp::BitAnd => a & b,
                        _ => panic!("unexpected saturation arithmetic {op:?}"),
                    }
                }
                OperationKind::Compare {
                    predicate: ComparePredicate::LessThan,
                    lhs,
                    rhs,
                } => u128::from(if scalar.is_signed_integer() {
                    signed(values[lhs], bits) < signed(values[rhs], bits)
                } else {
                    values[lhs] < values[rhs]
                }),
                OperationKind::Select {
                    condition,
                    true_value,
                    false_value,
                } => {
                    values[if values[condition] != 0 {
                        true_value
                    } else {
                        false_value
                    }]
                }
                other => panic!("unexpected saturation operation {other:?}"),
            };
            if matches!(
                operation.kind,
                OperationKind::Binary {
                    op: BinaryOp::Checked(_),
                    ..
                }
            ) {
                assert_eq!(operation.results.len(), 2);
                assert_eq!(operation.results[1].ty, Type::BOOL);
                // Deliberately don't assign overflow: any accidental use fails.
            } else {
                assert_eq!(operation.results.len(), 1);
            }
            values.insert(operation.results[0].id, result);
        }
        values[&operations.last().unwrap().results[0].id]
    }

    #[test]
    fn exact_checked_saturation_exhaustively_matches_eight_bit_rust() {
        for scalar in [ScalarType::U8, ScalarType::I8] {
            for op in [Add, Subtract] {
                let operations = emit(scalar, op, 11).unwrap();
                for lhs in 0..=255_u128 {
                    for rhs in 0..=255_u128 {
                        let rust = match (scalar, op) {
                            (ScalarType::U8, Add) => (lhs as u8).saturating_add(rhs as u8),
                            (ScalarType::U8, Subtract) => (lhs as u8).saturating_sub(rhs as u8),
                            (ScalarType::I8, Add) => (lhs as i8).saturating_add(rhs as i8) as u8,
                            (ScalarType::I8, Subtract) => {
                                (lhs as i8).saturating_sub(rhs as i8) as u8
                            }
                            _ => unreachable!(),
                        };
                        assert_eq!(
                            evaluate(&operations, scalar, lhs, rhs),
                            u128::from(rust),
                            "{scalar:?} {op:?} {lhs} {rhs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn saturation_has_exact_four_or_eleven_operation_limits_and_wide_boundaries() {
        for scalar in SCALARS {
            let bits = scalar.bit_width().unwrap();
            let sign = 1_u128 << (bits - 1);
            let mask = (1_u128 << bits) - 1;
            for op in [Add, Subtract] {
                let count = if scalar.is_signed_integer() { 11 } else { 4 };
                let operations = emit(scalar, op, count).unwrap();
                assert_eq!(operations.len(), count);
                let mut block = BasicBlock::new(BlockId(0));
                block.operations = operations.clone();
                block.terminator = Some(Terminator::Return {
                    values: vec![operations.last().unwrap().results[0].id],
                });
                let mut module = Module::new("saturation-component");
                module.functions.push(Function::internal_helper(
                    "saturation",
                    Signature::new(vec![Type::Scalar(scalar); 2], vec![Type::Scalar(scalar)]),
                    vec![ValueId(0), ValueId(1)],
                    vec![block],
                ));
                verify_module(&module).unwrap();
                assert!(
                    matches!(emit(scalar, op, count - 1), Err(ProductionSemanticKirErrorV1::ResourceLimit {
                    resource: ProductionSemanticKirResourceV1::Operations, actual, limit,
                }) if actual == count && limit == count - 1)
                );
                for lhs in [0, 1, 2, sign - 1, sign, sign + 1, mask - 1, mask] {
                    for rhs in [0, 1, 2, sign - 1, sign, sign + 1, mask - 1, mask] {
                        assert_eq!(
                            evaluate(&operations, scalar, lhs, rhs),
                            reference(scalar, op, lhs, rhs)
                        );
                    }
                }
            }
        }
        for scalar in [
            ScalarType::Bool,
            ScalarType::Index,
            ScalarType::I128,
            ScalarType::U128,
            ScalarType::F32,
            ScalarType::F64,
        ] {
            for op in [Add, Subtract] {
                assert!(matches!(
                    emit(scalar, op, 11),
                    Err(ProductionSemanticKirErrorV1::Unsupported { .. })
                ));
            }
        }
    }

    fn normalize(
        operations: Vec<Operation>,
        scalar: ScalarType,
        value: ValueId,
    ) -> Option<NormalizedScalarExpressionV1> {
        let mut block = BasicBlock::new(BlockId(0));
        block.operations = operations;
        block.terminator = Some(Terminator::Return {
            values: vec![value],
        });
        let function = Function::internal_helper(
            "saturating",
            Signature::new(vec![Type::Scalar(scalar); 2], vec![Type::Scalar(scalar)]),
            vec![ValueId(0), ValueId(1)],
            vec![block],
        );
        let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining: 4096 };
        let body = function.body.as_ref().unwrap();
        let index = build_kir_correlation_index(body, 11, &mut budget)?;
        let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut resources = ArgumentBudgetV1::new(&mut work, 16 * 1024 * 1024);
        native_helper_value_expansion_v1::with_no_helpers_for_test_v1(&mut resources, |expansion| {
            normalize_kir_expression_v1(
                &function,
                &index,
                &BTreeMap::new(),
                value,
                0,
                &mut BTreeSet::new(),
                &mut budget,
                expansion,
            )
        })
    }

    fn expected(
        scalar: ScalarType,
        op: SemanticSaturatingIntegerOpV1,
    ) -> NormalizedScalarExpressionV1 {
        use NormalizedScalarExpressionV1 as E;
        let ty = ProductionSemanticScalarTypeV2::Integer {
            signed: scalar.is_signed_integer(),
            bits: scalar.bit_width().unwrap(),
        };
        let a = E::Symbol {
            symbol: PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2,
            scalar: ty,
        };
        let b = E::Symbol {
            symbol: PRODUCTION_KERNEL_SCALAR_SYMBOL_BASE_V2 + 1,
            scalar: ty,
        };
        let constant = |bits| E::Constant { scalar: ty, bits };
        let binary = |operation, lhs, rhs| E::Binary {
            operation,
            scalar: ty,
            overflow: ProductionOverflowContractV2::Wrapping,
            lhs: NormalizedScalarNodeV18::legacy(lhs),
            rhs: NormalizedScalarNodeV18::legacy(rhs),
        };
        let less = |lhs, rhs| E::Compare {
            operation: ProductionSemanticComparisonV2::LessThan,
            operand_scalar: ty,
            lhs: NormalizedScalarNodeV18::legacy(lhs),
            rhs: NormalizedScalarNodeV18::legacy(rhs),
        };
        let select = |condition, when_true, when_false| E::Select {
            scalar: ty,
            condition: NormalizedScalarNodeV18::legacy(condition),
            when_true: NormalizedScalarNodeV18::legacy(when_true),
            when_false: NormalizedScalarNodeV18::legacy(when_false),
        };
        let raw = binary(
            match op {
                Add => ProductionSemanticBinaryOpV2::Add,
                Subtract => ProductionSemanticBinaryOpV2::Subtract,
            },
            a.clone(),
            b.clone(),
        );
        if scalar.is_signed_integer() {
            let first = binary(ProductionSemanticBinaryOpV2::BitXor, a.clone(), raw.clone());
            let second = match op {
                Add => binary(ProductionSemanticBinaryOpV2::BitXor, b, raw.clone()),
                Subtract => binary(ProductionSemanticBinaryOpV2::BitXor, a.clone(), b),
            };
            let condition = less(
                binary(ProductionSemanticBinaryOpV2::BitAnd, first, second),
                constant(0),
            );
            let sign = 1_u64 << (scalar.bit_width().unwrap() - 1);
            select(
                condition,
                select(less(a, constant(0)), constant(sign), constant(sign - 1)),
                raw,
            )
        } else {
            match op {
                Add => select(
                    less(raw.clone(), a),
                    constant(((1_u128 << scalar.bit_width().unwrap()) - 1) as u64),
                    raw,
                ),
                Subtract => select(less(a, b), constant(0), raw),
            }
        }
    }

    #[test]
    fn saturation_correspondence_matches_exact_predicate_select_and_checked_result_zero() {
        for scalar in SCALARS {
            for op in [Add, Subtract] {
                let operations = emit(scalar, op, 11).unwrap();
                let result = operations.last().unwrap().results[0].id;
                let wanted = expected(scalar, op);
                let actual = normalize(operations.clone(), scalar, result).unwrap();
                let nodes = match (scalar.is_signed_integer(), op) {
                    (false, Subtract) => 8,
                    (false, Add) => 10,
                    (true, Subtract) => 21,
                    (true, Add) => 23,
                };
                for (remaining, expected_result) in [(nodes, Some(true)), (nodes - 1, None)] {
                    let mut budget = UnsupportedIndexCorrelationBudgetV1 { remaining };
                    assert_eq!(
                        scalar_value_expressions_correspond_v1(&wanted, &actual, 0, &mut budget),
                        expected_result
                    );
                    assert_eq!(budget.remaining, 0);
                }
                let overflow = operations[0].results[1].id;
                let overflow = normalize(operations.clone(), scalar, overflow)
                    .expect("checked overflow has its own Boolean normalization");
                assert!(matches!(
                    &overflow,
                    NormalizedScalarExpressionV1::Compare { .. }
                        | NormalizedScalarExpressionV1::Select {
                            scalar: ProductionSemanticScalarTypeV2::Bool,
                            ..
                        }
                ));
                assert_eq!(
                    scalar_value_expressions_correspond_v1(
                        &wanted,
                        &overflow,
                        0,
                        &mut UnsupportedIndexCorrelationBudgetV1 { remaining: 4096 }
                    ),
                    Some(false),
                    "overflow result cannot replace the saturating integer payload"
                );
                for mutation in 0..5 {
                    let mut changed = operations.clone();
                    match mutation {
                        0 => {
                            if let OperationKind::Binary { op, .. } = &mut changed[0].kind {
                                *op = BinaryOp::Checked(CheckedBinaryOperator::Multiply);
                            }
                        }
                        1 => {
                            if let OperationKind::Binary { lhs, rhs, .. } = &mut changed[0].kind {
                                std::mem::swap(lhs, rhs);
                            }
                        }
                        2 => {
                            for operation in &mut changed {
                                if let OperationKind::Compare { predicate, .. } =
                                    &mut operation.kind
                                {
                                    *predicate = ComparePredicate::GreaterThanOrEqual;
                                    break;
                                }
                            }
                        }
                        3 => {
                            if let OperationKind::Select {
                                true_value,
                                false_value,
                                ..
                            } = &mut changed.last_mut().unwrap().kind
                            {
                                std::mem::swap(true_value, false_value);
                            }
                        }
                        4 => {
                            if let OperationKind::Binary { op, .. } = &mut changed[0].kind {
                                *op = BinaryOp::Add;
                            }
                        }
                        _ => unreachable!(),
                    }
                    if let Some(actual) = normalize(changed, scalar, result) {
                        assert_eq!(
                            scalar_value_expressions_correspond_v1(
                                &wanted,
                                &actual,
                                0,
                                &mut UnsupportedIndexCorrelationBudgetV1 { remaining: 4096 }
                            ),
                            Some(false),
                            "{scalar:?} {op:?} mutation {mutation}"
                        );
                    }
                }
            }
        }
    }
}
