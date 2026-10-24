mod row_striped_checked_arithmetic_v1_tests {
    use super::*;

    const INPUTS: usize = 5;
    const OUTPUTS: usize = 6;
    const PARAMS: usize = INPUTS + OUTPUTS;
    const EMITTED_OPERATIONS: usize = 2 + 2 + 4 * 2 + 4 + 7;

    fn emit(
        lanes: u64,
        elements: u64,
        limit: usize,
    ) -> (
        Result<(ValueId, ValueId), ProductionSemanticKirErrorV1>,
        Vec<Operation>,
    ) {
        let fixture = Fixture::new(ScalarType::Index);
        let mut lowering_work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
        let mut lowering_budget = ArgumentBudgetV1::new(&mut lowering_work, 16 * 1024 * 1024);
        let mut lowering = fixture.lowering(&mut lowering_budget);
        lowering.next_value = PARAMS as u32;
        lowering.max_operations = limit;
        let mut operations = Vec::new();
        let result = lowering.lower_row_striped_2d_component_index(
            BLOCK,
            &mut operations,
            ValueId(0),
            ValueId(1),
            ValueId(2),
            ValueId(3),
            ValueId(4),
            lanes,
            elements,
        );
        assert_eq!(lowering.emitted_operations, operations.len());
        (result, operations)
    }

    fn module(lanes: u64, elements: u64) -> Module {
        let (result, mut operations) = emit(lanes, elements, EMITTED_OPERATIONS);
        let (index, present) = result.unwrap();
        assert_eq!(operations.len(), EMITTED_OPERATIONS);
        assert_eq!(
            operations.iter().map(|op| op.results.len()).sum::<usize>(),
            27
        );
        let mut overflows = Vec::new();
        let mut no_overflows = Vec::new();
        let expected = [
            CheckedBinaryOperator::Multiply,
            CheckedBinaryOperator::Add,
            CheckedBinaryOperator::Multiply,
            CheckedBinaryOperator::Add,
        ];
        for (position, operation) in operations.iter().enumerate() {
            if let OperationKind::Binary { op, .. } = operation.kind {
                assert!(!matches!(op, BinaryOp::Add | BinaryOp::Multiply));
                if let BinaryOp::Checked(operator) = op {
                    assert_eq!(operator, expected[overflows.len()]);
                    assert_eq!(operation.results.len(), 2);
                    assert_eq!(operation.results[0].ty, Type::INDEX);
                    assert_eq!(operation.results[1].ty, Type::BOOL);
                    let overflow = operation.results[1].id;
                    assert!(matches!(operations[position + 1].kind,
                        OperationKind::Unary { op: UnaryOp::Not, operand } if operand == overflow));
                    overflows.push(overflow);
                    no_overflows.push(operations[position + 1].results[0].id);
                }
            }
        }
        assert_eq!(overflows.len(), 4);
        fn conjunction_leaves(value: ValueId, operations: &[Operation], leaves: &mut Vec<ValueId>) {
            let operation = operations
                .iter()
                .find(|operation| operation.results.iter().any(|result| result.id == value))
                .unwrap();
            if let OperationKind::Binary {
                op: BinaryOp::BitAnd,
                lhs,
                rhs,
            } = operation.kind
            {
                conjunction_leaves(lhs, operations, leaves);
                conjunction_leaves(rhs, operations, leaves);
            } else {
                leaves.push(value);
            }
        }
        let mut leaves = Vec::new();
        conjunction_leaves(present, &operations, &mut leaves);
        assert_eq!(&leaves[..4], no_overflows.as_slice());
        assert_eq!(leaves.len(), 8);
        for value in &leaves[4..] {
            let operation = operations
                .iter()
                .find(|operation| operation.results[0].id == *value)
                .unwrap();
            assert!(matches!(operation.kind, OperationKind::Compare { .. }));
        }
        let outputs = [
            index,
            present,
            overflows[0],
            overflows[1],
            overflows[2],
            overflows[3],
        ];
        for (ordinal, value) in outputs.into_iter().enumerate() {
            operations.push(Operation::new(
                vec![],
                OperationKind::Store {
                    pointer: ValueId((INPUTS + ordinal) as u32),
                    value,
                    access: MemoryAccess::new(
                        AddressSpace::Global,
                        if ordinal == 0 { 8 } else { 1 },
                    ),
                },
            ));
        }
        let mut block = BasicBlock::new(BlockId(0));
        block.operations = operations;
        block.terminator = Some(Terminator::Return { values: vec![] });
        let mut parameters = vec![Type::INDEX; INPUTS];
        parameters.extend((0..OUTPUTS).map(|i| {
            Type::pointer(
                if i == 0 { Type::INDEX } else { Type::BOOL },
                AddressSpace::Global,
                AccessMode::ReadWrite,
            )
        }));
        let mut module = Module::new("row-striped-checked-arithmetic");
        module.functions.push(Function::kernel_entry(
            "row_striped_impl",
            Signature::new(parameters, vec![]),
            (0..PARAMS as u32).map(ValueId).collect(),
            vec![block],
        ));
        module.kernels.push(Kernel::new(
            "row_striped",
            "row_striped_impl",
            LaunchDomain::D1 {
                x: LaunchExtent::Dynamic,
            },
        ));
        verify_module(&module).unwrap();
        module
    }

    fn execute(
        admitted: &AdmittedSimulationModuleV1,
        input: [u64; INPUTS],
    ) -> (u64, bool, [bool; 4]) {
        let target = SimulationTargetV1::amdgpu_64();
        let bits = |scalar, value| ScalarBitsV1::new(scalar, value, target).unwrap();
        let mut arguments = input
            .into_iter()
            .map(|value| SimulationArgumentV1::Scalar(bits(ScalarType::Index, u128::from(value))))
            .collect::<Vec<_>>();
        arguments.extend((0..OUTPUTS).map(|i| {
            SimulationArgumentV1::Buffer(
                BufferArgumentV1::from_scalars(
                    AccessMode::ReadWrite,
                    if i == 0 { 8 } else { 1 },
                    &[bits(
                        if i == 0 {
                            ScalarType::Index
                        } else {
                            ScalarType::Bool
                        },
                        0,
                    )],
                    target,
                )
                .unwrap(),
            )
        }));
        let result = admitted
            .simulate(
                &SimulationRequestV1::new("row_striped", [1, 1, 1], [1, 1, 1], arguments),
                target,
                SimulationLimitsV1::default(),
            )
            .unwrap();
        let boolean = |ordinal| {
            let bytes = result.buffer(ordinal).unwrap().bytes();
            assert!(bytes == [0] || bytes == [1]);
            bytes[0] == 1
        };
        (
            read_bits(result.buffer(5).unwrap().bytes()) as u64,
            boolean(6),
            [boolean(7), boolean(8), boolean(9), boolean(10)],
        )
    }

    fn source_index(
        lanes: u64,
        elements: u64,
        [raw, component, rows, columns, stride]: [u64; INPUTS],
    ) -> Option<u64> {
        if component >= elements || stride < columns {
            return None;
        }
        let row = raw.checked_div(lanes)?;
        let lane = raw.checked_rem(lanes)?;
        let column = component.checked_mul(lanes)?.checked_add(lane)?;
        if row >= rows || column >= columns {
            return None;
        }
        row.checked_mul(stride)?.checked_add(column)
    }

    #[test]
    fn row_striped_checked_arithmetic_is_total_at_each_independent_overflow_boundary() {
        let cases = [
            (2, 1, [0, u64::MAX, 1, 1, 1], [true, false, false, false]),
            // This component is outside E; its later false predicate must not
            // excuse partial arithmetic evaluated before that predicate.
            (
                3,
                1,
                [1, u64::MAX / 3, 1, 3, 3],
                [false, true, false, false],
            ),
            (
                1,
                1,
                [u64::MAX / 2 + 1, 0, u64::MAX, 1, 2],
                [false, false, true, false],
            ),
            (2, 1, [3, 0, 2, 2, u64::MAX], [false, false, false, true]),
        ];
        for (lanes, elements, input, expected) in cases {
            let actual = execute(&admit(module(lanes, elements)), input);
            assert_eq!(actual.2, expected, "{lanes}/{elements}: {input:?}");
            assert!(!actual.1);
            assert_eq!(source_index(lanes, elements, input), None);
        }
        for (lanes, input) in [
            (2, [0, u64::MAX / 2, 1, 1, 1]),
            (3, [0, u64::MAX / 3, 1, 3, 3]),
            (1, [u64::MAX / 2, 0, u64::MAX, 1, 2]),
            (2, [2, 0, 2, 1, u64::MAX]),
        ] {
            let (index, present, overflow) = execute(&admit(module(lanes, 1)), input);
            assert_eq!(overflow, [false; 4], "last non-overflowing input {input:?}");
            assert_eq!(present.then_some(index), source_index(lanes, 1, input));
        }
    }

    #[test]
    fn row_striped_checked_arithmetic_preserves_success_zero_dimensions_and_partial_tails() {
        for (lanes, elements) in [(1, 1), (3, 2), (4, 3), (64, 4)] {
            let admitted = admit(module(lanes, elements));
            for input in [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 2, 1],
                [0, elements, 2, lanes * elements, lanes * elements],
                [lanes * 2, 0, 2, lanes * elements, lanes * elements],
                [
                    lanes * 2 - 1,
                    elements - 1,
                    2,
                    lanes * elements - 1,
                    lanes * elements,
                ],
                [
                    lanes * 2 - 1,
                    elements - 1,
                    2,
                    lanes * elements,
                    lanes * elements,
                ],
                [0, 0, 1, 1, 1],
            ] {
                let (index, present, overflow) = execute(&admitted, input);
                assert_eq!(
                    present.then_some(index),
                    source_index(lanes, elements, input),
                    "{lanes}/{elements}: {input:?}"
                );
                assert_eq!(overflow, [false; 4]);
            }
        }
        let admitted = admit(module(4, 3));
        for raw in 0..12 {
            for component in 0..4 {
                let input = [raw, component, 2, 9, 12];
                let (index, present, overflow) = execute(&admitted, input);
                assert_eq!(present.then_some(index), source_index(4, 3, input));
                assert_eq!(overflow, [false; 4]);
            }
        }
        // MAX itself is representable here, but cannot be a valid element of a
        // slice with a u64 extent. The separate extent guard remains required.
        let actual = execute(&admit(module(2, 1)), [2, 0, 2, 1, u64::MAX]);
        assert_eq!(actual, (u64::MAX, true, [false; 4]));
        assert!(!(actual.1 && actual.0 < u64::MAX));
    }

    #[test]
    fn row_striped_checked_arithmetic_keeps_exact_operation_limits_and_pair_arity() {
        let _ = module(4, 3);
        let (result, operations) = emit(4, 3, EMITTED_OPERATIONS - 1);
        assert!(
            matches!(result, Err(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Operations, actual, limit,
        }) if actual == EMITTED_OPERATIONS && limit == EMITTED_OPERATIONS - 1)
        );
        assert_eq!(operations.len(), EMITTED_OPERATIONS - 1);
        for (lanes, elements) in [(0, 1), (1, 0), (u64::MAX, 2)] {
            let (result, operations) = emit(lanes, elements, 0);
            assert!(matches!(
                result,
                Err(ProductionSemanticKirErrorV1::Unsupported {
                    detail: "row-striped-2d geometry is malformed",
                    ..
                })
            ));
            assert!(operations.is_empty());
        }
    }
}
