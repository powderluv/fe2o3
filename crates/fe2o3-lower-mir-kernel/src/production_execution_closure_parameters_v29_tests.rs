use super::*;

const SCALAR: Type = Type::Scalar(ScalarType::U32);

// Retained RustCall shape, not an authenticated provider or a source admission test.
fn closure_owner(packed: bool) -> ProductionSemanticSsaOwnerV1 {
    let nominal = nominal_owner(Input::Workgroup, false);
    let workgroup = nominal.source_semantic().functions()[1].locals()[1].ty();
    let mut types = nominal.source_semantic().types().to_vec();
    let scalar = match types[U32.index() as usize].layout().backend_repr() {
        SemanticBackendReprV1::Scalar(scalar) => *scalar,
        _ => panic!("fixture requires a scalar u32"),
    };
    let pair = aggregate(
        &mut types,
        vec![U32, U32],
        vec![0, 4],
        8,
        4,
        SemanticBackendReprV1::scalar_pair(scalar, scalar),
        None,
    );
    let capture = aggregate(
        &mut types,
        vec![pair],
        vec![0],
        8,
        4,
        SemanticBackendReprV1::scalar_pair(scalar, scalar),
        None,
    );
    let tuple = declaration(
        &mut types,
        SemanticTypeLayoutV1::aggregate_with_backend_repr(
            Some(16),
            8,
            *nominal.source_semantic().types()[workgroup.index() as usize]
                .layout()
                .backend_repr(),
            false,
            SemanticAggregateLayoutV1::new(vec![0], vec![]).unwrap(),
        )
        .unwrap(),
        SemanticTypeShapeV1::Tuple(SemanticAggregateTypeV1::new(vec![workgroup]).unwrap()),
        None,
    );
    let root_abi = SemanticFunctionAbiV1::from_rustc(
        SemanticAbiIdentityV1::from_sha256([203; 32]),
        SemanticLayoutIdentityV1::from_sha256([250; 32]),
        SemanticCanonAbiV1::GpuKernel,
        SemanticExternAbiV1::GpuKernel,
        false,
        false,
        2,
        vec![
            SemanticAbiArgumentV1::source(value_abi(&types, capture)),
            SemanticAbiArgumentV1::source(value_abi(&types, tuple)),
        ],
        ignored(UNIT),
    )
    .unwrap()
    .with_source_argument_ownership(vec![SemanticSourceArgumentOwnershipV1::ByValue; 2])
    .unwrap();
    let helper_abi = SemanticFunctionAbiV1::from_rustc_with_source_signature(
        SemanticAbiIdentityV1::from_sha256([205; 32]),
        SemanticLayoutIdentityV1::from_sha256([250; 32]),
        SemanticCanonAbiV1::Rust,
        SemanticExternAbiV1::RustCall,
        false,
        false,
        1,
        vec![capture, tuple],
        U32,
        vec![
            SemanticAbiArgumentV1::source(value_abi(&types, capture)),
            SemanticAbiArgumentV1::rust_call_tuple_field(0, value_abi(&types, workgroup)),
        ],
        value_abi(&types, U32),
    )
    .unwrap()
    .with_source_argument_ownership(vec![SemanticSourceArgumentOwnershipV1::ByValue; 2])
    .unwrap();
    let root = function(
        202,
        SemanticFunctionRoleV1::KernelRoot,
        root_abi,
        vec![
            local(210, UNIT, SemanticLocalRoleV1::Return),
            local(211, capture, SemanticLocalRoleV1::Argument(0)),
            local(212, tuple, SemanticLocalRoleV1::Argument(1)),
            local(213, U32, SemanticLocalRoleV1::Temporary),
            local(214, CONTEXT, SemanticLocalRoleV1::Temporary),
        ],
        vec![
            block(
                220,
                vec![],
                SemanticTerminatorKindV1::Call(
                    SemanticDirectCallV1::new_callable(
                        SemanticCallableIdV1::from_index(1),
                        vec![
                            SemanticOperandV1::Move(place(1, capture)),
                            SemanticOperandV1::Move(place(2, tuple)),
                        ],
                        Some(SemanticCallDestinationV1::new(
                            place(3, U32),
                            SemanticControlFlowEdgeV1::new(
                                SemanticEdgeRoleV1::CallReturn,
                                SemanticBlockIdV1::from_index(1),
                            ),
                        )),
                        SemanticUnwindActionV1::Unreachable,
                    )
                    .unwrap(),
                ),
            ),
            block(221, vec![], SemanticTerminatorKindV1::Return),
        ],
    )
    .with_kernel_entry(SemanticKernelEntryV1::new(
        SemanticLinkSymbolV1::new(b"execution_closure_fixture".to_vec()).unwrap(),
        SemanticKernelBindingIdentityV1::from_sha256([204; 32]),
        SemanticKernelSourceContractV1::new(None, None, None).unwrap(),
    ));
    let projected_workgroup = if packed {
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(2),
            vec![SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), workgroup).unwrap()],
            workgroup,
        )
        .unwrap()
    } else {
        place(2, workgroup)
    };
    let field = |index| {
        SemanticPlaceV1::new(
            SemanticLocalIdV1::from_index(1),
            vec![
                SemanticProjectionV1::new(SemanticProjectionKindV1::Field(0), pair).unwrap(),
                SemanticProjectionV1::new(SemanticProjectionKindV1::Field(index), U32).unwrap(),
            ],
            U32,
        )
        .unwrap()
    };
    let helper = function(
        206,
        SemanticFunctionRoleV1::InternalHelper,
        helper_abi,
        vec![
            local(230, U32, SemanticLocalRoleV1::Return),
            local(231, capture, SemanticLocalRoleV1::Argument(0)),
            local(
                232,
                if packed { tuple } else { workgroup },
                if packed {
                    SemanticLocalRoleV1::Argument(1)
                } else {
                    SemanticLocalRoleV1::RustCallTupleField {
                        argument: 1,
                        field: 0,
                    }
                },
            ),
            local(233, workgroup, SemanticLocalRoleV1::Temporary),
        ],
        vec![block(
            240,
            vec![
                assign(
                    place(3, workgroup),
                    SemanticRvalueKindV1::Use(SemanticOperandV1::Move(projected_workgroup)),
                ),
                assign(
                    place(0, U32),
                    SemanticRvalueKindV1::Binary {
                        operation: SemanticBinaryOpV1::BitXor,
                        left: SemanticOperandV1::Copy(field(0)),
                        right: SemanticOperandV1::Copy(field(1)),
                    },
                ),
            ],
            SemanticTerminatorKindV1::Return,
        )],
    );
    let admitted = InertSemanticMirRequestV1::new_with_callables(
        SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256([250; 32])),
        types,
        vec![],
        vec![],
        vec![],
        vec![root, helper],
        vec![
            SemanticCallableDeclV1::defined(ROOT),
            SemanticCallableDeclV1::defined(HELPER),
        ],
        vec![ROOT],
    )
    .unwrap()
    .admit_exact_v29(SemanticMirLimitsV1::default())
    .unwrap();
    ProductionSemanticSsaOwnerV1::try_new(
        ProductionSemanticMirOwnerV1::try_new(admitted, ProductionSemanticMirLimitsV1::default())
            .unwrap(),
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap()
}

fn run_closure(
    packed: bool,
    context_value: u32,
    work_limit: usize,
    storage_limit: usize,
) -> (
    Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1>,
    usize,
    usize,
) {
    let mut owner = closure_owner(packed);
    let mut work = CanonicalKernelIrWorkBudgetV1::new(work_limit);
    let mut budget = ArgumentBudgetV1::new(&mut work, storage_limit);
    let result = (|| {
        budget.reserve_storage(FLOOR)?;
        let captured = owner
            .try_capture_occurrences_with_budget_v1(&mut budget)
            .map_err(|error| match error {
                fe2o3_pliron::ProductionSemanticSsaOccurrenceErrorV1::Resource(error) => {
                    error.into()
                }
                _ => execution_call_error_v29(),
            })?;
        budget.reserve_storage(captured.retained_storage())?;
        with_production_call_instances_v1(&owner, ROOT, &mut budget, |instances, budget| {
            Ok::<_, production_call_instances_v1::ProductionCallInstanceErrorV1>(
                with_execution_call_scope_v29(budget, |scope, budget| {
                    let child = instances.calls(instances.root()).unwrap()[0]
                        .child()
                        .unwrap();
                    let plan = execution_instance_plan_v29(
                        instances,
                        child,
                        FunctionId::new("closure_child"),
                        SemanticEmissionPlacementV1 {
                            first_block: 17,
                            first_value: 300,
                        },
                        budget,
                    )?;
                    assert_eq!(plan.parameter_types, [SCALAR; 2]);
                    assert_eq!(plan.parameter_values, [ValueId(300), ValueId(301)]);
                    assert_eq!(plan.result_types, [SCALAR]);
                    assert_eq!(plan.call_arguments.len(), 2);
                    for (index, projection) in plan.call_arguments.iter().enumerate() {
                        assert_eq!(projection.source_argument, 0);
                        assert_eq!(projection.tuple_field, None);
                        assert_eq!(projection.component, Some(index));
                    }
                    let prepared = with_execution_availability_v29(
                        instances,
                        instances.root(),
                        budget,
                        |mut cursor, budget| {
                            let semantic = owner.source_semantic();
                            let capture = semantic.functions()[0].locals()[1].ty();
                            let workgroup = semantic.functions()[1].locals()[3].ty();
                            let producer = ProductionCallOccurrenceV1 {
                                caller: instances.root(),
                                block: SemanticBlockIdV1::from_index(0),
                            };
                            let context = SemanticExecutionBindingV29::context(
                                semantic.types(),
                                CONTEXT,
                                producer,
                                ValueId(context_value),
                            )
                            .unwrap();
                            let role = SemanticExecutionBindingV29::workgroup(
                                semantic.types(),
                                workgroup,
                                producer,
                                ValueId(91),
                                &context,
                            )
                            .unwrap();
                            cursor.entry_seeds = vec![(
                                2,
                                SemanticValueBindingV1::Aggregate(vec![
                                    SemanticValueBindingV1::Execution(role),
                                ]),
                            )];
                            let locals = [PlannedParameterLocalBindingV1::Flattened {
                                local: 1,
                                semantic_type: capture,
                                values: vec![
                                    ValueDef::new(ValueId(20), Type::Scalar(ScalarType::U32)),
                                    ValueDef::new(ValueId(21), Type::Scalar(ScalarType::U32)),
                                ],
                            }];
                            let mut parent = SemanticFunctionLoweringV1::new_interprocedural(
                                semantic.types(),
                                semantic.callables(),
                                &semantic.functions()[0],
                                owner.plan_for_function(ROOT).unwrap(),
                                ROOT,
                                ROOT,
                                BTreeMap::new(),
                                BTreeMap::new(),
                                vec![],
                                SemanticParameterBindingsV1 {
                                    declarations: &[],
                                    values: &[ValueId(20), ValueId(21)],
                                    types: &[SCALAR; 2],
                                    local_bindings: Some(&locals),
                                },
                                None,
                                Some([64, 1, 1]),
                                BTreeSet::new().into(),
                                1,
                                false,
                                1024,
                                PrivateArrayRecorderWorkV1::Owned(PrivateArrayLazyBudgetV1::new(
                                    1, 1024,
                                )),
                                None,
                                CallReturnBufferV1::empty(),
                                Some(budget),
                                SemanticEmissionPlacementV1 {
                                    first_block: 0,
                                    first_value: 200,
                                },
                                Some(cursor),
                                None,
                                None,
)?;
                            let mut block = BasicBlock::new(BlockId(0));
                            parent.begin_block(SemanticBlockIdV1::from_index(0), &mut block)?;
                            let prepared = parent.prepare_defined_call_arguments_v1(
                                SemanticBlockIdV1::from_index(0),
                                instances.incoming(child).unwrap().source(),
                                HELPER,
                                DefinedCallArgumentSignatureV1 {
                                    projection: DefinedCallProjectionV29::Execution(scope),
                                    semantic_types: semantic.functions()[1]
                                        .abi()
                                        .source_input_types(),
                                    projections: &plan.call_arguments,
                                    parameter_types: plan.parameter_types.clone(),
                                },
                                &mut block.operations,
                            )?;
                            assert!(block.operations.is_empty());
                            assert_eq!(prepared.arguments, [ValueId(20), ValueId(21)]);
                            assert!(parent.locals[2].is_none());
                            Ok(prepared)
                        },
                    )?;
                    let (_, parameters) = prepare_execution_parameters_v29(
                        instances, child, prepared, &plan, budget,
                    )?;
                    with_execution_availability_v29(instances, child, budget, |cursor, budget| {
                        let cursor = cursor.with_call_parameters_v29(parameters)?;
                        let mut private = PrivateArrayLazyBudgetV1::new(1, 1024);
                        lower_one_semantic_function_v1(
                            owner.source_semantic(),
                            &plan,
                            owner.plan_for_function(HELPER).unwrap(),
                            &BTreeMap::new(),
                            &BTreeMap::new(),
                            None,
                            BTreeSet::new(),
                            1,
                            false,
                            1024,
                            None,
                            &mut private,
                            None,
                            budget,
                            SemanticEmissionPlacementV1 {
                                first_block: 17,
                                first_value: 300,
                            },
                            Some(cursor),
                        )
                    })
                }),
            )
        })
        .map_err(|error| match error {
            production_call_instances_v1::ProductionCallInstanceErrorV1::Resource(error) => {
                error.into()
            }
            _ => execution_call_error_v29(),
        })?
    })();
    let peak = budget.peak_storage();
    (result, work.work(), peak)
}

#[test]
fn closure_capture_scalars_and_workgroup_tuple_reach_shared_emission() {
    for packed in [false, true] {
        let result = run_closure(packed, 90, 10_000_000, 10_000_000).0.unwrap();
        assert_eq!(result.function.signature.parameters, [SCALAR; 2]);
        assert_eq!(result.function.signature.results, [SCALAR]);
        assert_eq!(
            result.function.body.as_ref().unwrap().parameters,
            [ValueId(300), ValueId(301)]
        );
        let observed = result.execution_observation.unwrap();
        let role = match &observed.locals[3] {
            Some(SemanticValueBindingV1::Execution(role)) => role,
            _ => panic!("callee lost its workgroup"),
        };
        let fixture = closure_owner(packed);
        let semantic = fixture.source_semantic();
        let producer = ProductionCallOccurrenceV1 {
            caller: ProductionCallInstanceIdV1(0),
            block: SemanticBlockIdV1::from_index(0),
        };
        let context =
            SemanticExecutionBindingV29::context(semantic.types(), CONTEXT, producer, ValueId(90))
                .unwrap();
        let expected = SemanticExecutionBindingV29::workgroup(
            semantic.types(),
            semantic.functions()[1].locals()[3].ty(),
            producer,
            ValueId(91),
            &context,
        )
        .unwrap();
        assert_eq!(role, &expected);
        assert_eq!(
            observed.locals[1].as_ref().unwrap().values().unwrap(),
            [(ValueId(300), SCALAR), (ValueId(301), SCALAR)]
        );
        if packed {
            assert!(matches!(&observed.locals[2],
                Some(SemanticValueBindingV1::Aggregate(fields))
                    if matches!(fields[0], SemanticValueBindingV1::MovedExecution)));
        } else {
            assert!(matches!(
                observed.locals[2],
                None | Some(SemanticValueBindingV1::MovedExecution)
            ));
        }
        let blocks = &result.function.body.as_ref().unwrap().blocks;
        let operations: Vec<_> = blocks.iter().flat_map(|block| &block.operations).collect();
        assert_eq!(operations.len(), 1);
        assert_eq!(
            operations[0].kind,
            OperationKind::Binary {
                op: BinaryOp::BitXor,
                lhs: ValueId(300),
                rhs: ValueId(301),
            }
        );
        assert_eq!(operations[0].results.len(), 1);
        assert_eq!(operations[0].results[0].ty, SCALAR);
        assert_eq!(
            blocks[0].terminator,
            Some(Terminator::Return {
                values: vec![operations[0].results[0].id],
            })
        );
    }
}

#[test]
fn closure_capture_work_and_storage_are_bounded() {
    for packed in [false, true] {
        let (result, work, storage) = run_closure(packed, 90, 10_000_000, 10_000_000);
        result.unwrap();
        assert!(storage > FLOOR);
        run_closure(packed, 90, work, storage).0.unwrap();
        assert!(run_closure(packed, 90, work - 1, storage).0.is_err());
        assert!(matches!(
            run_closure(packed, 90, work, storage - 1).0,
            Err(
                ProductionSemanticKirErrorV1::ArgumentCorrespondenceResource(
                    ArgumentResourceV1::Storage(_)
                )
            )
        ));
    }
}

#[test]
fn closure_capture_parameters_cannot_alias_the_parent_context() {
    for packed in [false, true] {
        assert!(matches!(
            run_closure(packed, 300, 10_000_000, 10_000_000).0,
            Err(ProductionSemanticKirErrorV1::Unsupported {
                detail: "execution call parameters differ from their source instance",
                ..
            })
        ));
    }
}
