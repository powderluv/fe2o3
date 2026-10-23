mod source_launch_u32_tests {
    use super::*;

    const KINDS: [IndexKind; 4] = [
        IndexKind::Local,
        IndexKind::Workgroup,
        IndexKind::WorkgroupSize,
        IndexKind::WorkgroupCount,
    ];
    const AXES: [(SemanticAxisV1, Axis); 3] = [
        (SemanticAxisV1::X, Axis::X),
        (SemanticAxisV1::Y, Axis::Y),
        (SemanticAxisV1::Z, Axis::Z),
    ];

    #[test]
    fn every_source_launch_call_returns_u32_without_narrowing_internal_indices() {
        let fixture = Fixture::new(ScalarType::U32);
        for rank in 1..=3 {
            for kind in KINDS {
                for (ordinal, (source_axis, axis)) in AXES.into_iter().enumerate() {
                    fixture.launch_call(kind, source_axis, rank, false, |lowering, call| {
                            let mut operations = Vec::new();
                            let terminator = lowering.lower_call(BLOCK, call, &mut operations).unwrap();
                            assert!(matches!(terminator, Terminator::Branch { target: BlockId(1), .. }));
                            assert_eq!(operations.len(), 3);
                            assert_eq!(operations[0].results.len(), 1);
                            assert_eq!(operations[0].results[0].ty, Type::INDEX);
                            if ordinal < usize::from(rank) {
                                assert!(matches!(
                                    &operations[0].kind,
                                    OperationKind::Intrinsic(intrinsic)
                                        if intrinsic.kind == (IntrinsicKind::InvocationIndex { kind, axis })
                                ));
                            } else {
                                let expected = match kind {
                                    IndexKind::Local | IndexKind::Workgroup => 0,
                                    _ => 1,
                                };
                                assert_eq!(operations[0].kind, OperationKind::Constant(Constant::Index(expected)));
                            }
                            assert_eq!(operations[1].results.len(), 1);
                            assert_eq!(operations[1].results[0].ty, Type::Scalar(ScalarType::U64));
                            assert_eq!(operations[1].kind, OperationKind::Cast {
                                kind: CastKind::Bitcast,
                                value: operations[0].results[0].id,
                                to: Type::Scalar(ScalarType::U64),
                            });
                            assert_eq!(operations[2].results.len(), 1);
                            assert_eq!(operations[2].results[0].ty, Type::Scalar(ScalarType::U32));
                            assert_eq!(operations[2].kind, OperationKind::Cast {
                                kind: CastKind::Truncate,
                                value: operations[1].results[0].id,
                                to: Type::Scalar(ScalarType::U32),
                            });
                            assert_eq!(lowering.locals[1].as_ref().unwrap().value().unwrap(),
                                (operations[2].results[0].id, Type::Scalar(ScalarType::U32)));

                            let mut internal_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
                            let mut internal_budget = ArgumentBudgetV1::new(&mut internal_work, 16 * 1024 * 1024);
                            let mut internal = fixture.lowering(&mut internal_budget);
                            internal.launch_rank = rank;
                            let mut internal_operations = Vec::new();
                            assert_eq!(internal.emit_launch_index_v1(&mut internal_operations, kind, axis)
                                .unwrap().value().unwrap().1, Type::INDEX);
                            assert_eq!(internal_operations.len(), 1);
                            assert_eq!(internal_operations[0].kind, operations[0].kind);
                        });
                }
            }
        }
    }

    #[test]
    fn source_launch_calls_reject_arguments_and_non_u32_destinations_before_emission() {
        let mut char_fixture = Fixture::new(ScalarType::U32);
        char_fixture.types[1] = SemanticTypeDeclV1::new(
            SemanticTypeIdentityV1::from_sha256([101; 32]),
            SemanticLayoutIdentityV1::from_sha256([102; 32]),
            SemanticTypeLayoutV1::new(Some(4), 4).unwrap(),
            SemanticTypeShapeV1::Scalar(SemanticScalarTypeV1::Char),
        );
        let mut cases = SCALARS
            .into_iter()
            .filter(|scalar| *scalar != ScalarType::U32)
            .map(|scalar| (Fixture::new(scalar), false))
            .collect::<Vec<_>>();
        cases.push((char_fixture, false));
        cases.push((Fixture::new(ScalarType::U32), true));
        for (fixture, extra_argument) in cases {
            for rank in 1..=3 {
                for kind in KINDS {
                    for (axis, _) in AXES {
                        fixture.launch_call(kind, axis, rank, extra_argument, |lowering, call| {
                            let mut operations = Vec::new();
                            assert!(matches!(
                                lowering.lower_call(BLOCK, call, &mut operations),
                                Err(ProductionSemanticKirErrorV1::Unsupported { .. })
                            ));
                            assert!(operations.is_empty());
                            assert_eq!(lowering.emitted_operations, 0);
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn source_launch_cast_obeys_exact_zero_and_one_short_operation_budgets() {
        let fixture = Fixture::new(ScalarType::U32);
        for rank in 1..=3 {
            for kind in KINDS {
                for (axis, _) in AXES {
                    for limit in 0..=3 {
                        fixture.launch_call(kind, axis, rank, false, |lowering, call| {
                            lowering.max_operations = limit;
                            let mut operations = Vec::new();
                            let result = lowering.lower_call(BLOCK, call, &mut operations);
                            if limit == 3 {
                                result.unwrap();
                            } else {
                                assert!(matches!(result,
                                        Err(ProductionSemanticKirErrorV1::ResourceLimit {
                                            resource: ProductionSemanticKirResourceV1::Operations,
                                            actual, limit: actual_limit,
                                        }) if actual == limit + 1 && actual_limit == limit));
                            }
                            assert_eq!(operations.len(), limit);
                            assert_eq!(lowering.emitted_operations, limit);
                        });
                    }
                }
            }
        }
    }

    fn observed_launch_module(
        fixture: &Fixture,
        kind: IndexKind,
        axis: SemanticAxisV1,
        rank: u8,
        grid: [u64; 3],
    ) -> Module {
        observed_launch_module_with(fixture, kind, axis, rank, grid, |_, _, value| {
            (value, Type::Scalar(ScalarType::U32), None)
        })
    }

    fn observed_launch_module_with(
        fixture: &Fixture,
        kind: IndexKind,
        axis: SemanticAxisV1,
        rank: u8,
        grid: [u64; 3],
        transform: impl FnOnce(
            &mut SemanticFunctionLoweringV1<'_, '_>,
            &mut Vec<Operation>,
            ValueId,
        ) -> (ValueId, Type, Option<ValueId>),
    ) -> Module {
        fixture.launch_call(kind, axis, rank, false, |lowering, call| {
            let mut call_operations = Vec::new();
            let terminator = lowering
                .lower_call(BLOCK, call, &mut call_operations)
                .unwrap();
            let value = lowering.locals[1].as_ref().unwrap().value().unwrap().0;
            let mut operations = Vec::new();
            let (value, value_type, flag) = transform(lowering, &mut operations, value);
            // Generic output instrumentation keeps each invocation disjoint.
            let mut offset = lowering
                .emit_id(
                    &mut operations,
                    Type::INDEX,
                    OperationKind::Constant(Constant::Index(0)),
                )
                .unwrap();
            for (ordinal, (_, axis)) in AXES.into_iter().enumerate().rev() {
                let extent = lowering
                    .emit_id(
                        &mut operations,
                        Type::INDEX,
                        OperationKind::Constant(Constant::Index(grid[ordinal])),
                    )
                    .unwrap();
                let coordinate = lowering
                    .emit_launch_index_v1(&mut operations, IndexKind::Global, axis)
                    .unwrap()
                    .value()
                    .unwrap()
                    .0;
                offset = lowering
                    .emit_id(
                        &mut operations,
                        Type::INDEX,
                        OperationKind::Binary {
                            op: BinaryOp::Multiply,
                            lhs: offset,
                            rhs: extent,
                        },
                    )
                    .unwrap();
                offset = lowering
                    .emit_id(
                        &mut operations,
                        Type::INDEX,
                        OperationKind::Binary {
                            op: BinaryOp::Add,
                            lhs: offset,
                            rhs: coordinate,
                        },
                    )
                    .unwrap();
            }
            let one = lowering
                .emit_id(
                    &mut operations,
                    Type::INDEX,
                    OperationKind::Constant(Constant::Index(1)),
                )
                .unwrap();
            offset = lowering
                .emit_id(
                    &mut operations,
                    Type::INDEX,
                    OperationKind::Binary {
                        op: BinaryOp::Add,
                        lhs: offset,
                        rhs: one,
                    },
                )
                .unwrap();
            let output_type = Type::pointer(
                value_type.clone(),
                AddressSpace::Global,
                AccessMode::ReadWrite,
            );
            let pointer = lowering
                .emit_id(
                    &mut operations,
                    output_type.clone(),
                    OperationKind::GetElementPointer {
                        base: ValueId(2),
                        offset,
                    },
                )
                .unwrap();
            operations.push(Operation::new(
                vec![],
                OperationKind::Store {
                    pointer,
                    value,
                    access: MemoryAccess::new(
                        AddressSpace::Global,
                        width(value_type.as_scalar().unwrap()) / 8,
                    ),
                },
            ));
            if let Some(flag) = flag {
                let flag_pointer = lowering
                    .emit_id(
                        &mut operations,
                        Type::pointer(Type::BOOL, AddressSpace::Global, AccessMode::ReadWrite),
                        OperationKind::GetElementPointer {
                            base: ValueId(3),
                            offset,
                        },
                    )
                    .unwrap();
                operations.push(Operation::new(
                    vec![],
                    OperationKind::Store {
                        pointer: flag_pointer,
                        value: flag,
                        access: MemoryAccess::new(AddressSpace::Global, 1),
                    },
                ));
            }
            let mut entry = BasicBlock::new(BlockId(0));
            entry.operations = call_operations;
            entry.terminator = Some(terminator);
            let mut observer = BasicBlock::new(BlockId(1));
            observer.operations = operations;
            observer.terminator = Some(Terminator::Return { values: vec![] });
            let function = Function::kernel_entry(
                "source_launch_impl",
                Signature::new(
                    vec![
                        Type::Scalar(ScalarType::U32),
                        Type::Scalar(ScalarType::U32),
                        output_type,
                        Type::pointer(Type::BOOL, AddressSpace::Global, AccessMode::ReadWrite),
                    ],
                    vec![],
                ),
                vec![ValueId(0), ValueId(1), ValueId(2), ValueId(3)],
                vec![entry, observer],
            );
            let domain = match rank {
                1 => LaunchDomain::D1 {
                    x: LaunchExtent::Dynamic,
                },
                2 => LaunchDomain::D2 {
                    x: LaunchExtent::Dynamic,
                    y: LaunchExtent::Dynamic,
                },
                3 => LaunchDomain::D3 {
                    x: LaunchExtent::Dynamic,
                    y: LaunchExtent::Dynamic,
                    z: LaunchExtent::Dynamic,
                },
                _ => unreachable!(),
            };
            let mut module = Module::new("source-launch-u32-test");
            module.functions.push(function);
            module
                .kernels
                .push(Kernel::new("source_launch", "source_launch_impl", domain));
            verify_module(&module).unwrap();
            module
        })
    }

    #[test]
    fn source_launch_checked_add_shift_and_widen_compose_with_exact_flags() {
        let mut fixture = Fixture::new(ScalarType::U32);
        fixture.types.push(integer_type(103, false, 64));
        let module = observed_launch_module_with(
            &fixture,
            IndexKind::Local,
            SemanticAxisV1::X,
            1,
            [3, 1, 1],
            |lowering, operations, launch_value| {
                let u32_type = SemanticTypeIdV1::from_index(1);
                let operand = |local| {
                    SemanticOperandV1::Copy(
                        SemanticPlaceV1::new(
                            SemanticLocalIdV1::from_index(local),
                            vec![],
                            u32_type,
                        )
                        .unwrap(),
                    )
                };
                let checked = lowering
                    .lower_rvalue(
                        BLOCK,
                        None,
                        SemanticTypeIdV1::from_index(3),
                        &SemanticRvalueKindV1::CheckedBinary(SemanticCheckedBinaryRvalueV1::new(
                            SemanticCheckedBinaryOpV1::Add,
                            operand(1),
                            SemanticOperandV1::Constant(SemanticConstantV1::new(
                                u32_type,
                                SemanticConstantValueV1::Scalar(
                                    SemanticScalarValueV1::new(u128::from(u32::MAX), 4).unwrap(),
                                ),
                            )),
                        )),
                        operations,
                    )
                    .unwrap()
                    .values()
                    .unwrap();
                assert_eq!(checked.len(), 2);
                assert_eq!(checked[0].1, Type::Scalar(ScalarType::U32));
                assert_eq!(checked[1].1, Type::BOOL);
                assert!(matches!(operations.last().unwrap().kind,
                        OperationKind::Binary { op: BinaryOp::Checked(CheckedBinaryOperator::Add), lhs, .. }
                            if lhs == launch_value));
                lowering.locals[1] = Some(SemanticValueBindingV1::Value {
                    id: checked[0].0,
                    ty: checked[0].1.clone(),
                });
                // This is normalized semantic MIR with a dynamic count, not source `<< 32` admission.
                let shifted = lowering
                    .lower_rvalue(
                        BLOCK,
                        None,
                        u32_type,
                        &SemanticRvalueKindV1::Binary {
                            operation: SemanticBinaryOpV1::ShiftLeft,
                            left: operand(1),
                            right: operand(2),
                        },
                        operations,
                    )
                    .unwrap();
                assert_eq!(
                    operations[2].kind,
                    OperationKind::Constant(Constant::U32(31))
                );
                assert_eq!(
                    operations[3].kind,
                    OperationKind::Binary {
                        op: BinaryOp::BitAnd,
                        lhs: ValueId(1),
                        rhs: operations[2].results[0].id,
                    }
                );
                assert_eq!(
                    operations[4].kind,
                    OperationKind::Binary {
                        op: BinaryOp::ShiftLeft,
                        lhs: checked[0].0,
                        rhs: operations[3].results[0].id,
                    }
                );
                let shifted_value = shifted.value().unwrap().0;
                lowering.locals[1] = Some(shifted);
                let widened = lowering
                    .lower_rvalue(
                        BLOCK,
                        None,
                        SemanticTypeIdV1::from_index(4),
                        &SemanticRvalueKindV1::Cast {
                            kind: SemanticCastKindV1::Integer,
                            operand: operand(1),
                        },
                        operations,
                    )
                    .unwrap()
                    .value()
                    .unwrap();
                assert_eq!(operations.len(), 6);
                assert_eq!(
                    operations[5].kind,
                    OperationKind::Cast {
                        kind: CastKind::ZeroExtend,
                        value: shifted_value,
                        to: Type::Scalar(ScalarType::U64),
                    }
                );
                (widened.0, widened.1, Some(checked[1].0))
            },
        );
        let target = SimulationTargetV1::amdgpu_64();
        let canary = u128::from(u64::MAX - 7);
        let mut request = request(ScalarType::U32, 0, 32);
        request.kernel = "source_launch".into();
        request.grid = fe2o3_kir_sim::GridShapeV1([3, 1, 1]);
        request.workgroup = fe2o3_kir_sim::WorkgroupShapeV1([3, 1, 1]);
        for (argument, scalar, initial, alignment) in
            [(2, ScalarType::U64, canary, 8), (3, ScalarType::Bool, 1, 1)]
        {
            request.arguments[argument] = SimulationArgumentV1::Buffer(
                BufferArgumentV1::from_scalars(
                    AccessMode::ReadWrite,
                    alignment,
                    &[ScalarBitsV1::new(scalar, initial, target).unwrap(); 5],
                    target,
                )
                .unwrap(),
            );
        }
        let result = admit(module)
            .simulate(&request, target, SimulationLimitsV1::default())
            .unwrap();
        assert_eq!(
            result
                .buffer(2)
                .unwrap()
                .bytes()
                .chunks_exact(8)
                .map(read_bits)
                .collect::<Vec<_>>(),
            [canary, u128::from(u32::MAX), 0, 1, canary]
        );
        assert_eq!(result.buffer(3).unwrap().bytes(), &[1, 0, 1, 1, 1]);
    }

    #[test]
    fn source_launch_calls_simulate_exact_coordinates_with_partial_workgroups_and_canaries() {
        let fixture = Fixture::new(ScalarType::U32);
        let target = SimulationTargetV1::amdgpu_64();
        let canary = 0xa5a5_5a5a_u128;
        for rank in 1..=3 {
            let grid = [
                5,
                if rank >= 2 { 3 } else { 1 },
                if rank == 3 { 3 } else { 1 },
            ];
            let workgroup = [
                3,
                if rank >= 2 { 2 } else { 1 },
                if rank == 3 { 2 } else { 1 },
            ];
            let count = grid.into_iter().product::<u64>() as usize;
            for kind in KINDS {
                for (ordinal, (axis, _)) in AXES.into_iter().enumerate() {
                    let admitted = admit(observed_launch_module(&fixture, kind, axis, rank, grid));
                    let mut request = request(ScalarType::U32, 0, 0);
                    request.kernel = "source_launch".into();
                    request.grid = fe2o3_kir_sim::GridShapeV1(grid);
                    request.workgroup = fe2o3_kir_sim::WorkgroupShapeV1(workgroup);
                    request.arguments[2] = SimulationArgumentV1::Buffer(
                        BufferArgumentV1::from_scalars(
                            AccessMode::ReadWrite,
                            4,
                            &vec![
                                ScalarBitsV1::new(ScalarType::U32, canary, target).unwrap();
                                count + 2
                            ],
                            target,
                        )
                        .unwrap(),
                    );
                    let result = admitted
                        .simulate(&request, target, SimulationLimitsV1::default())
                        .unwrap();
                    let values = result
                        .buffer(2)
                        .unwrap()
                        .bytes()
                        .chunks_exact(4)
                        .map(read_bits)
                        .collect::<Vec<_>>();
                    assert_eq!(values.len(), count + 2);
                    assert_eq!(values[0], canary);
                    assert_eq!(values[count + 1], canary);
                    assert_eq!(result.buffer(3).unwrap().bytes(), &[1]);
                    for z in 0..grid[2] {
                        for y in 0..grid[1] {
                            for x in 0..grid[0] {
                                let coordinate = [x, y, z][ordinal];
                                let extent = u64::from(workgroup[ordinal]);
                                let expected = match kind {
                                    IndexKind::Local => coordinate % extent,
                                    IndexKind::Workgroup => coordinate / extent,
                                    IndexKind::WorkgroupSize => extent,
                                    IndexKind::WorkgroupCount => grid[ordinal].div_ceil(extent),
                                    IndexKind::Global => unreachable!(),
                                };
                                let index = ((z * grid[1] + y) * grid[0] + x) as usize + 1;
                                assert_eq!(
                                    values[index],
                                    u128::from(expected),
                                    "rank {rank}, {kind:?}, {axis:?}, [{x}, {y}, {z}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
