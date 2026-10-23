mod wrapping_arithmetic_v1_tests {
    use super::*;
    use fe2o3_kir_sim::{
        AdmittedSimulationModuleV1, BufferArgumentV1, ScalarBitsV1, SimulationArgumentV1,
        SimulationErrorV1, SimulationExecutionErrorKindV1, SimulationLimitsV1, SimulationRequestV1,
        SimulationTargetV1,
    };
    use fe2o3_mir_model::semantic_mir_v1::{
        SemanticAbiValueAttributesV1, SemanticCheckedBinaryRvalueV1,
    };

    const SCALARS: [ScalarType; 11] = [
        ScalarType::I8,
        ScalarType::U8,
        ScalarType::I16,
        ScalarType::U16,
        ScalarType::I32,
        ScalarType::U32,
        ScalarType::I64,
        ScalarType::U64,
        ScalarType::I128,
        ScalarType::U128,
        ScalarType::Index,
    ];
    const OPERATIONS: [(
        SemanticBinaryOpV1,
        SemanticCheckedBinaryOpV1,
        CheckedBinaryOperator,
        BinaryOp,
    ); 3] = [
        (
            SemanticBinaryOpV1::Add,
            SemanticCheckedBinaryOpV1::Add,
            CheckedBinaryOperator::Add,
            BinaryOp::Add,
        ),
        (
            SemanticBinaryOpV1::Subtract,
            SemanticCheckedBinaryOpV1::Subtract,
            CheckedBinaryOperator::Subtract,
            BinaryOp::Subtract,
        ),
        (
            SemanticBinaryOpV1::Multiply,
            SemanticCheckedBinaryOpV1::Multiply,
            CheckedBinaryOperator::Multiply,
            BinaryOp::Multiply,
        ),
    ];
    const BLOCK: SemanticBlockIdV1 = SemanticBlockIdV1::from_index(0);

    fn width(scalar: ScalarType) -> u32 {
        scalar.bit_width().map_or(64, u32::from)
    }

    fn bit_mask(scalar: ScalarType) -> u128 {
        let width = width(scalar);
        if width == 128 {
            u128::MAX
        } else {
            (1_u128 << width) - 1
        }
    }

    fn pairs(scalar: ScalarType) -> [(u128, u128); 10] {
        let mask = bit_mask(scalar);
        let sign = 1_u128 << (width(scalar) - 1);
        [
            (0, 0),
            (5, 7),
            (mask, 1),
            (0, 1),
            (mask, 2),
            (sign, mask),
            (sign, 1),
            (sign, 2),
            (sign - 1, 1),
            (sign - 1, 2),
        ]
    }

    fn oracle(
        scalar: ScalarType,
        operation: SemanticBinaryOpV1,
        lhs: u128,
        rhs: u128,
    ) -> (u128, bool) {
        macro_rules! native {
            ($ty:ty) => {{
                let lhs = lhs as $ty;
                let rhs = rhs as $ty;
                let (value, overflow) = match operation {
                    SemanticBinaryOpV1::Add => lhs.overflowing_add(rhs),
                    SemanticBinaryOpV1::Subtract => lhs.overflowing_sub(rhs),
                    SemanticBinaryOpV1::Multiply => lhs.overflowing_mul(rhs),
                    _ => unreachable!("closed arithmetic test operation"),
                };
                (value as u128 & bit_mask(scalar), overflow)
            }};
        }
        match scalar {
            ScalarType::I8 => native!(i8),
            ScalarType::U8 => native!(u8),
            ScalarType::I16 => native!(i16),
            ScalarType::U16 => native!(u16),
            ScalarType::I32 => native!(i32),
            ScalarType::U32 => native!(u32),
            ScalarType::I64 => native!(i64),
            ScalarType::U64 | ScalarType::Index => native!(u64),
            ScalarType::I128 => native!(i128),
            ScalarType::U128 => native!(u128),
            _ => unreachable!("closed arithmetic test scalar"),
        }
    }

    struct Fixture {
        scalar: ScalarType,
        types: Vec<SemanticTypeDeclV1>,
        function: SemanticFunctionDeclV1,
    }

    impl Fixture {
        fn new(scalar: ScalarType) -> Self {
            let scalar_type = SemanticTypeIdV1::from_index(1);
            let boolean = SemanticTypeIdV1::from_index(2);
            let bytes = u64::from(width(scalar) / 8);
            let types = vec![
                unit_type(),
                integer_type(71, scalar.is_signed_integer(), width(scalar) as u16),
                bool_type(),
                SemanticTypeDeclV1::new(
                    SemanticTypeIdentityV1::from_sha256([73; 32]),
                    SemanticLayoutIdentityV1::from_sha256([74; 32]),
                    SemanticTypeLayoutV1::aggregate(
                        Some(2 * bytes),
                        bytes,
                        SemanticAggregateLayoutV1::new(vec![0, bytes], vec![]).unwrap(),
                    )
                    .unwrap(),
                    SemanticTypeShapeV1::Tuple(
                        SemanticAggregateTypeV1::new(vec![scalar_type, boolean]).unwrap(),
                    ),
                ),
            ];
            let source = SemanticSourceProvenanceV1::unavailable();
            let argument = SemanticAbiArgumentV1::source(SemanticAbiValueV1::new(
                scalar_type,
                SemanticAbiPassModeV1::Direct(SemanticAbiValueAttributesV1::plain()),
            ));
            let abi = SemanticFunctionAbiV1::from_rustc(
                SemanticAbiIdentityV1::from_sha256([75; 32]),
                SemanticLayoutIdentityV1::from_sha256([76; 32]),
                SemanticCanonAbiV1::Rust,
                SemanticExternAbiV1::Rust,
                false,
                false,
                2,
                vec![argument.clone(), argument],
                SemanticAbiValueV1::new(
                    SemanticTypeIdV1::from_index(0),
                    SemanticAbiPassModeV1::Ignore,
                ),
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
                SemanticFunctionIdentityV1::from_sha256([77; 32]),
                SemanticFunctionRoleV1::InternalHelper,
                SemanticItemDefinitionIdentityV1::from_sha256([78; 32]),
                SemanticMonomorphizationIdentityV1::from_sha256([79; 32]),
                SemanticGenericTypeArgumentsIdentityV1::from_sha256([80; 32]),
                SemanticConstGenericArgumentsIdentityV1::from_sha256([81; 32]),
                source,
                abi,
                vec![
                    local(
                        82,
                        SemanticTypeIdV1::from_index(0),
                        SemanticLocalRoleV1::Return,
                    ),
                    local(83, scalar_type, SemanticLocalRoleV1::Argument(0)),
                    local(84, scalar_type, SemanticLocalRoleV1::Argument(1)),
                ],
                BLOCK,
                vec![
                    SemanticBasicBlockV1::new(
                        SemanticBlockIdentityV1::from_sha256([85; 32]),
                        source,
                        vec![],
                        SemanticTerminatorV1::new(source, SemanticTerminatorKindV1::Return),
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            Self {
                scalar,
                types,
                function,
            }
        }

        fn lowering<'a, 'service, 'work>(
            &'a self,
            budget: &'service mut ArgumentBudgetV1<'work>,
        ) -> SemanticFunctionLoweringV1<'a, 'service> {
            self.lowering_with_callables(&[], budget)
        }

        fn lowering_with_callables<'a, 'service, 'work>(
            &'a self,
            callables: &'a [SemanticCallableDeclV1],
            budget: &'service mut ArgumentBudgetV1<'work>,
        ) -> SemanticFunctionLoweringV1<'a, 'service> {
            let scalar_type = SemanticTypeIdV1::from_index(1);
            let mut lowering = SemanticFunctionLoweringV1::new(
                &self.types,
                callables,
                &self.function,
                SemanticParameterBindingsV1 {
                    declarations: &[(0, 1, scalar_type), (1, 2, scalar_type)],
                    values: &[ValueId(0), ValueId(1)],
                    types: &[Type::Scalar(self.scalar), Type::Scalar(self.scalar)],
                    local_bindings: None,
                },
                None,
                None,
                BTreeSet::new(),
                1,
                false,
                64,
                budget,
            )
            .unwrap();
            lowering.next_value = 4;
            lowering
        }

        fn launch_call<T>(
            &self,
            kind: IndexKind,
            axis: SemanticAxisV1,
            rank: u8,
            extra_argument: bool,
            run: impl FnOnce(&mut SemanticFunctionLoweringV1<'_, '_>, &SemanticDirectCallV1) -> T,
        ) -> T {
            let scalar = SemanticTypeIdV1::from_index(1);
            let source = SemanticSourceProvenanceV1::unavailable();
            let abi = SemanticFunctionAbiV1::from_rustc(
                SemanticAbiIdentityV1::from_sha256([91; 32]),
                SemanticLayoutIdentityV1::from_sha256([92; 32]),
                SemanticCanonAbiV1::Rust,
                SemanticExternAbiV1::Rust,
                false,
                false,
                0,
                vec![],
                SemanticAbiValueV1::new(
                    scalar,
                    SemanticAbiPassModeV1::Direct(SemanticAbiValueAttributesV1::plain()),
                ),
            )
            .unwrap();
            let operation = match kind {
                IndexKind::Local => SemanticCompilerIntrinsicOperationV1::ThreadIndex(axis),
                IndexKind::Workgroup => SemanticCompilerIntrinsicOperationV1::WorkgroupIndex(axis),
                IndexKind::WorkgroupSize => {
                    SemanticCompilerIntrinsicOperationV1::WorkgroupDimension(axis)
                }
                IndexKind::WorkgroupCount => {
                    SemanticCompilerIntrinsicOperationV1::GridDimension(axis)
                }
                IndexKind::Global => unreachable!("global index has no source launch terminal"),
            };
            let callables = [SemanticCallableDeclV1::CompilerIntrinsic {
                binding: SemanticNonBodyCallableBindingV1::new(
                    SemanticFunctionIdentityV1::from_sha256([93; 32]),
                    SemanticItemDefinitionIdentityV1::from_sha256([94; 32]),
                    SemanticMonomorphizationIdentityV1::from_sha256([95; 32]),
                    SemanticGenericTypeArgumentsIdentityV1::from_sha256([96; 32]),
                    SemanticConstGenericArgumentsIdentityV1::from_sha256([97; 32]),
                    source,
                    abi,
                ),
                operation,
                operation_identity: SemanticCompilerIntrinsicIdentityV1::from_sha256([98; 32]),
            }];
            let place = |local| {
                SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], scalar).unwrap()
            };
            let arguments = if extra_argument {
                vec![SemanticOperandV1::Copy(place(2))]
            } else {
                vec![]
            };
            let call = SemanticDirectCallV1::new_callable(
                SemanticCallableIdV1::from_index(0),
                arguments,
                Some(SemanticCallDestinationV1::new(
                    place(1),
                    SemanticControlFlowEdgeV1::new(
                        SemanticEdgeRoleV1::CallReturn,
                        SemanticBlockIdV1::from_index(1),
                    ),
                )),
                SemanticUnwindActionV1::Unreachable,
            )
            .unwrap();
            let function = SemanticFunctionDeclV1::new(
                self.function.identity(),
                self.function.role(),
                self.function.item_definition_identity(),
                self.function.monomorphization_identity(),
                self.function.generic_type_arguments_identity(),
                self.function.const_generic_arguments_identity(),
                source,
                self.function.abi().clone(),
                self.function.locals().to_vec(),
                BLOCK,
                vec![
                    SemanticBasicBlockV1::new(
                        SemanticBlockIdentityV1::from_sha256([99; 32]),
                        source,
                        vec![],
                        SemanticTerminatorV1::new(
                            source,
                            SemanticTerminatorKindV1::Call(call.clone()),
                        ),
                    )
                    .unwrap(),
                    self.function.blocks()[0].clone(),
                ],
            )
            .unwrap();
            let fixture = Self {
                scalar: self.scalar,
                types: self.types.clone(),
                function,
            };
            let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
            let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
            let mut lowering = fixture.lowering_with_callables(&callables, &mut lowering_budget);
            lowering.launch_rank = rank;
            run(&mut lowering, &call)
        }

        fn lower(
            &self,
            lowering: &mut SemanticFunctionLoweringV1<'_, '_>,
            operation: SemanticBinaryOpV1,
            checked: bool,
            operations: &mut Vec<Operation>,
        ) -> Result<SemanticValueBindingV1, ProductionSemanticKirErrorV1> {
            let operand = |local| {
                SemanticOperandV1::Copy(
                    SemanticPlaceV1::new(
                        SemanticLocalIdV1::from_index(local),
                        vec![],
                        SemanticTypeIdV1::from_index(1),
                    )
                    .unwrap(),
                )
            };
            let value = if checked {
                let checked_operation = OPERATIONS.iter().find(|row| row.0 == operation).unwrap().1;
                SemanticRvalueKindV1::CheckedBinary(SemanticCheckedBinaryRvalueV1::new(
                    checked_operation,
                    operand(1),
                    operand(2),
                ))
            } else {
                SemanticRvalueKindV1::Binary {
                    operation,
                    left: operand(1),
                    right: operand(2),
                }
            };
            lowering.lower_rvalue(
                BLOCK,
                Some(0),
                SemanticTypeIdV1::from_index(if checked { 3 } else { 1 }),
                &value,
                operations,
            )
        }

        fn module(&self, operation: SemanticBinaryOpV1, checked: bool) -> Module {
            let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
            let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
            let mut lowering = self.lowering(&mut lowering_budget);
            let mut operations = Vec::new();
            let result = self
                .lower(&mut lowering, operation, checked, &mut operations)
                .unwrap();
            assert_eq!(
                operations.len(),
                1,
                "dynamic arithmetic needs one operation"
            );
            let expected_operation = OPERATIONS.iter().find(|row| row.0 == operation).unwrap().2;
            assert!(matches!(
                operations[0].kind,
                OperationKind::Binary { op: BinaryOp::Checked(actual), lhs: ValueId(0), rhs: ValueId(1) }
                if actual == expected_operation
            ));
            assert_eq!(operations[0].results.len(), 2);
            assert_eq!(operations[0].results[0].ty, Type::Scalar(self.scalar));
            assert_eq!(operations[0].results[1].ty, Type::BOOL);
            let values = if checked {
                assert!(matches!(result, SemanticValueBindingV1::Aggregate(_)));
                let values = result.values().unwrap();
                assert_eq!(values.len(), 2);
                values
            } else {
                vec![
                    result
                        .value()
                        .expect("ordinary arithmetic returns only its modular value"),
                ]
            };
            assert_eq!(values[0].0, operations[0].results[0].id);
            for (index, (value, ty)) in values.into_iter().enumerate() {
                let alignment = if ty == Type::BOOL {
                    1
                } else {
                    width(self.scalar) / 8
                };
                operations.push(Operation::new(
                    vec![],
                    OperationKind::Store {
                        pointer: ValueId(2 + index as u32),
                        value,
                        access: MemoryAccess::new(AddressSpace::Global, alignment),
                    },
                ));
            }
            let mut block = BasicBlock::new(BlockId(0));
            block.operations = operations;
            block.terminator = Some(Terminator::Return { values: vec![] });
            let function = Function::kernel_entry(
                "arithmetic_impl",
                Signature::new(
                    vec![
                        Type::Scalar(self.scalar),
                        Type::Scalar(self.scalar),
                        Type::pointer(
                            Type::Scalar(self.scalar),
                            AddressSpace::Global,
                            AccessMode::ReadWrite,
                        ),
                        Type::pointer(Type::BOOL, AddressSpace::Global, AccessMode::ReadWrite),
                    ],
                    vec![],
                ),
                vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
                vec![block],
            );
            let mut module = Module::new("ordinary-mir-wrapping-test");
            module.functions.push(function);
            module.kernels.push(Kernel::new(
                "arithmetic",
                "arithmetic_impl",
                LaunchDomain::D1 {
                    x: LaunchExtent::Dynamic,
                },
            ));
            verify_module(&module).unwrap();
            module
        }
    }

    fn admit(module: Module) -> AdmittedSimulationModuleV1 {
        let canonical = VerifiedCanonicalKernelIrV11::from_module(module).unwrap();
        AdmittedSimulationModuleV1::admit_v11(canonical, SimulationLimitsV1::default()).unwrap()
    }

    fn request(scalar: ScalarType, lhs: u128, rhs: u128) -> SimulationRequestV1 {
        let target = SimulationTargetV1::amdgpu_64();
        let bits = |scalar, value| ScalarBitsV1::new(scalar, value, target).unwrap();
        SimulationRequestV1::new(
            "arithmetic",
            [1, 1, 1],
            [1, 1, 1],
            vec![
                SimulationArgumentV1::Scalar(bits(scalar, lhs)),
                SimulationArgumentV1::Scalar(bits(scalar, rhs)),
                SimulationArgumentV1::Buffer(
                    BufferArgumentV1::from_scalars(
                        AccessMode::ReadWrite,
                        width(scalar) / 8,
                        &[bits(scalar, 0)],
                        target,
                    )
                    .unwrap(),
                ),
                SimulationArgumentV1::Buffer(
                    BufferArgumentV1::from_scalars(
                        AccessMode::ReadWrite,
                        1,
                        &[bits(ScalarType::Bool, 1)],
                        target,
                    )
                    .unwrap(),
                ),
            ],
        )
    }

    fn read_bits(bytes: &[u8]) -> u128 {
        let mut padded = [0_u8; 16];
        padded[..bytes.len()].copy_from_slice(bytes);
        u128::from_le_bytes(padded)
    }

    #[test]
    fn ordinary_mir_add_sub_mul_match_native_modular_results_at_every_integer_width() {
        for scalar in SCALARS {
            let fixture = Fixture::new(scalar);
            for (operation, _, _, _) in OPERATIONS {
                let admitted = admit(fixture.module(operation, false));
                for (lhs, rhs) in pairs(scalar) {
                    let result = admitted
                        .simulate(
                            &request(scalar, lhs, rhs),
                            SimulationTargetV1::amdgpu_64(),
                            SimulationLimitsV1::default(),
                        )
                        .unwrap();
                    assert_eq!(
                        read_bits(result.buffer(2).unwrap().bytes()),
                        oracle(scalar, operation, lhs, rhs).0,
                        "{scalar:?} {operation:?}: {lhs:#x}, {rhs:#x}",
                    );
                    assert_eq!(
                        result.buffer(3).unwrap().bytes(),
                        &[1],
                        "ordinary result must not expose an overflow flag"
                    );
                }
            }
        }
    }

    #[test]
    fn explicit_checked_mir_retains_the_result_and_exact_overflow_flag() {
        for scalar in SCALARS
            .into_iter()
            .filter(|&scalar| scalar != ScalarType::Index)
        {
            let fixture = Fixture::new(scalar);
            for (operation, _, _, _) in OPERATIONS {
                let admitted = admit(fixture.module(operation, true));
                for (lhs, rhs) in pairs(scalar) {
                    let result = admitted
                        .simulate(
                            &request(scalar, lhs, rhs),
                            SimulationTargetV1::amdgpu_64(),
                            SimulationLimitsV1::default(),
                        )
                        .unwrap();
                    let expected = oracle(scalar, operation, lhs, rhs);
                    assert_eq!(read_bits(result.buffer(2).unwrap().bytes()), expected.0);
                    assert_eq!(result.buffer(3).unwrap().bytes(), &[u8::from(expected.1)]);
                }
            }
        }
    }

    #[test]
    fn unchanged_plain_kir_arithmetic_still_refuses_overflow() {
        for scalar in SCALARS {
            let fixture = Fixture::new(scalar);
            for (operation, _, _, plain) in OPERATIONS {
                let mut module = fixture.module(operation, false);
                let arithmetic =
                    &mut module.functions[0].body.as_mut().unwrap().blocks[0].operations[0];
                let OperationKind::Binary { op, .. } = &mut arithmetic.kind else {
                    unreachable!()
                };
                *op = plain;
                arithmetic.results.truncate(1);
                verify_module(&module).unwrap();
                let admitted = admit(module);
                let (lhs, rhs) = pairs(scalar)
                    .into_iter()
                    .find(|&(lhs, rhs)| oracle(scalar, operation, lhs, rhs).1)
                    .expect("boundary corpus contains overflow for each integer operation");
                let error = admitted
                    .simulate(
                        &request(scalar, lhs, rhs),
                        SimulationTargetV1::amdgpu_64(),
                        SimulationLimitsV1::default(),
                    )
                    .unwrap_err();
                assert!(
                    matches!(
                        error, SimulationErrorV1::Execution(error)
                        if matches!(error.kind, SimulationExecutionErrorKindV1::UndefinedIntegerOperation(_))
                    ),
                    "{scalar:?} {operation:?} must preserve plain-KIR refusal"
                );
            }
        }
    }

    #[test]
    fn arithmetic_emission_obeys_exact_and_cumulative_operation_budgets() {
        for scalar in SCALARS {
            let fixture = Fixture::new(scalar);
            for (operation, _, _, _) in OPERATIONS {
                let mut exact_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut exact_budget = ArgumentBudgetV1::new(&mut exact_work, 16 * 1024 * 1024);
                let mut exact = fixture.lowering(&mut exact_budget);
                exact.max_operations = 1;
                let mut operations = Vec::new();
                fixture
                    .lower(&mut exact, operation, false, &mut operations)
                    .unwrap();
                assert_eq!(exact.emitted_operations, 1);
                assert_eq!(operations.len(), 1);

                let mut empty_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut empty_budget = ArgumentBudgetV1::new(&mut empty_work, 16 * 1024 * 1024);
                let mut empty = fixture.lowering(&mut empty_budget);
                empty.max_operations = 0;
                assert!(matches!(
                    fixture.lower(&mut empty, operation, false, &mut Vec::new()),
                    Err(ProductionSemanticKirErrorV1::ResourceLimit {
                        resource: ProductionSemanticKirResourceV1::Operations,
                        actual: 1,
                        limit: 0,
                    })
                ));
                assert_eq!(empty.emitted_operations, 0);

                let mut cumulative_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                let mut cumulative_budget = ArgumentBudgetV1::new(&mut cumulative_work, 16 * 1024 * 1024);
                let mut cumulative = fixture.lowering(&mut cumulative_budget);
                let mut prefix = Vec::new();
                cumulative
                    .emit_id(
                        &mut prefix,
                        Type::BOOL,
                        OperationKind::Constant(Constant::Bool(false)),
                    )
                    .unwrap();
                cumulative.max_operations = 1;
                assert!(matches!(
                    fixture.lower(&mut cumulative, operation, false, &mut prefix),
                    Err(ProductionSemanticKirErrorV1::ResourceLimit {
                        resource: ProductionSemanticKirResourceV1::Operations,
                        actual: 2,
                        limit: 1,
                    })
                ));
                assert_eq!(prefix.len(), 1);
                assert_eq!(cumulative.emitted_operations, 1);
            }
        }
    }

    #[test]
    fn ordinary_arithmetic_rejects_mismatched_dynamic_operand_types_before_emission() {
        let fixture = Fixture::new(ScalarType::U32);
        for (operation, _, _, _) in OPERATIONS {
            let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
            let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
            let mut lowering = fixture.lowering(&mut lowering_budget);
            lowering.locals[2] = Some(SemanticValueBindingV1::Value {
                id: ValueId(1),
                ty: Type::Scalar(ScalarType::I32),
            });
            let mut operations = Vec::new();
            assert!(matches!(
                fixture.lower(&mut lowering, operation, false, &mut operations),
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "semantic binary operand types differ",
                    ..
                })
            ));
            assert!(operations.is_empty());
            assert_eq!(lowering.emitted_operations, 0);
        }
    }

    include!("source_launch_indices_v1_tests.rs");
    include!("row_striped_checked_arithmetic_v1_tests.rs");
}
