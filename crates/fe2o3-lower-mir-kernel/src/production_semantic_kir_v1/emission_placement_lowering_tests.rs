use super::*;

fn lower_placed_root(
    owner: ProductionSemanticMirOwnerV1,
    placement: SemanticEmissionPlacementV1,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    let ssa =
        ProductionSemanticSsaOwnerV1::try_new(owner, ProductionSemanticSsaLimitsV1::default())
            .unwrap();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut arguments = ArgumentBudgetV1::new(&mut work, 1_000_000);
    lower_placed_function(
        &ssa,
        ssa.source_semantic().roots()[0],
        placement,
        &mut arguments,
    )
}

pub(in super::super) fn lower_placed_function(
    ssa: &ProductionSemanticSsaOwnerV1,
    selected: SemanticFunctionIdV1,
    placement: SemanticEmissionPlacementV1,
    arguments: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    lower_placed_function_with_availability_v29(ssa, selected, placement, arguments, None)
}

pub(in super::super) fn lower_placed_function_with_availability_v29(
    ssa: &ProductionSemanticSsaOwnerV1,
    selected: SemanticFunctionIdV1,
    placement: SemanticEmissionPlacementV1,
    arguments: &mut ArgumentBudgetV1<'_>,
    execution: Option<ExecutionAvailabilityV29<'_>>,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    let semantic = ssa.source_semantic();
    let root = semantic.roots()[0];
    let mut ids = BTreeMap::new();
    let mut signatures = BTreeMap::new();
    for (index, function) in semantic.functions().iter().enumerate() {
        let id = SemanticFunctionIdV1::from_index(index as u32);
        if id == root {
            continue;
        }
        ids.insert(id, helper_function_id_v1(id, function));
        signatures.insert(
            id,
            LoweredFunctionSignatureV1 {
                bf16_nominal: false,
                wave64_nominal: false,
                parameter_semantic_types: vec![],
                call_arguments: vec![],
                parameter_types: vec![],
                result_types: vec![],
                result_semantic_type: function.abi().source_output_type(),
            },
        );
    }
    let plan = LoweredFunctionPlanV1 {
        correspondence_owner: root,
        semantic_function: selected,
        kernel_ir_function: if selected == root {
            FunctionId::new("placed_root")
        } else {
            ids[&selected].clone()
        },
        role: if selected == root {
            SemanticKirFunctionRoleV1::KernelEntry
        } else {
            SemanticKirFunctionRoleV1::InternalHelper
        },
        parameter_declarations: vec![],
        parameter_types: vec![],
        parameter_values: vec![],
        call_arguments: vec![],
        parameter_local_bindings: vec![],
        parameter_component_bindings: vec![],
        ignored_parameter_bindings: vec![],
        result_types: vec![],
    };
    let mut private = PrivateArrayLazyBudgetV1::new(1, 256);
    lower_one_semantic_function_v1(
        semantic,
        &plan,
        ssa.plan_for_function(selected).unwrap(),
        &ids,
        &signatures,
        Some([64, 1, 1]),
        BTreeSet::new(),
        1,
        false,
        256,
        None,
        &mut private,
        None,
        arguments,
        placement,
        execution,
    )
}

#[test]
fn captured_source_instances_drive_existing_lowering_and_call_expansion() {
    use production_call_instances_v1::{
        ProductionCallInstanceErrorV1, with_production_call_instances_v1,
    };
    let mut ssa = ProductionSemanticSsaOwnerV1::try_new(
        helper_closure_semantic_owner(),
        ProductionSemanticSsaLimitsV1::default(),
    )
    .unwrap();
    let mut work = fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1::new(1_000_000);
    let mut budget = ArgumentBudgetV1::new(&mut work, 1_000_000);
    let capture = ssa
        .try_capture_occurrences_with_budget_v1(&mut budget)
        .unwrap();
    budget.reserve_storage(capture.retained_storage()).unwrap();
    let root = ssa.source_semantic().roots()[0];
    with_production_call_instances_v1(&ssa, root, &mut budget, |plan, budget| {
        assert!(std::ptr::eq(plan.owner(), &ssa));
        assert_eq!(plan.instances().len(), 2);
        assert_eq!(plan.id_at(0), Some(plan.root()));
        assert!(plan.id_at(2).is_none());
        let call = &plan.calls(plan.root()).unwrap()[0];
        let child = call.child().unwrap();
        let child_row = plan.instance(child).unwrap();
        assert_eq!(child.index(), 1);
        assert!(std::ptr::eq(
            child_row.ssa(),
            ssa.plan_for_function(child_row.function()).unwrap()
        ));
        assert!(
            matches!(call.callable(), SemanticCallableDeclV1::Defined { function }
            if *function == child_row.function())
        );
        let caller =
            with_execution_availability_v29(plan, plan.root(), budget, |cursor, budget| {
                lower_placed_function_with_availability_v29(
                    &ssa,
                    root,
                    SemanticEmissionPlacementV1::default(),
                    budget,
                    Some(cursor),
                )
            })
            .unwrap();
        let callee = with_execution_availability_v29(plan, child, budget, |cursor, budget| {
            lower_placed_function_with_availability_v29(
                &ssa,
                child_row.function(),
                SemanticEmissionPlacementV1 {
                    first_block: 17,
                    first_value: 100,
                },
                budget,
                Some(cursor),
            )
        })
        .unwrap();
        let archive_storage = super::super::instance_correspondence_tests::scalar_archive_storage_v1(
            &caller,
            plan,
            plan.root(),
            budget,
        ) + super::super::instance_correspondence_tests::scalar_archive_storage_v1(
            &callee,
            plan,
            child,
            budget,
        );
        let scratch_storage = super::super::instance_correspondence_tests::scalar_lowering_scratch_storage_v29(&caller)
            + super::super::instance_correspondence_tests::scalar_lowering_scratch_storage_v29(&callee);
        with_production_instance_correspondence_v1(plan, budget, |mapping, budget| {
            mapping.append_lowered(plan.root(), &caller, budget)?;
            mapping.append_lowered(child, &callee, budget)?;
            drop(caller.execution_observation);
            drop(callee.execution_observation);
            budget.release_storage(archive_storage)?;
            let call_storage = CallReturnBufferV1::bytes(
                caller.call_returns.sites.rows.len() + callee.call_returns.sites.rows.len(),
                caller.call_returns.components.rows.len()
                    + callee.call_returns.components.rows.len(),
            )
            .unwrap();
            drop(caller.call_returns);
            drop(callee.call_returns);
            budget.release_storage(call_storage)?;
            let caller_initialization = caller.scoped_initialization.unwrap();
            let callee_initialization = callee.scoped_initialization.unwrap();
            let initialization_storage =
                caller_initialization.retained_storage + callee_initialization.retained_storage;
            drop((caller_initialization, callee_initialization));
            budget.release_storage(initialization_storage)?;
            let caller_memory = caller.scoped_memory_anchors.unwrap();
            let callee_memory = callee.scoped_memory_anchors.unwrap();
            let memory_storage = caller_memory.retained_storage().unwrap()
                + callee_memory.retained_storage().unwrap();
            drop((caller_memory, callee_memory));
            budget.release_storage(memory_storage)?;
            let expanded = mapping
                .splice(
                    call,
                    caller.function,
                    callee.function,
                    BlockId(18),
                    BlockId(19),
                    budget,
                )
                .unwrap();
            mapping.check_coordinates(plan.root(), &expanded.caller, budget)?;
            assert!(!mapping.spans().is_empty());
            assert!(!mapping.controls().is_empty());
            assert_eq!(expanded.split.returns, plan.returns(child).unwrap().count());
            assert!(expanded.callee_required_capabilities.is_empty());
            let retained = expanded.additional_storage_bytes;
            let mut module = Module::new("source_instances");
            module.functions.push(expanded.caller);
            module.kernels.push(Kernel::new(
                "placed_root",
                "placed_root",
                LaunchDomain::D1 {
                    x: LaunchExtent::Static(64),
                },
            ));
            assert!(
                !module.functions[0]
                    .body
                    .as_ref()
                    .unwrap()
                    .blocks
                    .iter()
                    .flat_map(|block| &block.operations)
                    .any(|operation| matches!(operation.kind, OperationKind::Call { .. }))
            );
            verify_module(&module).unwrap();
            drop(module);
            budget.release_storage(retained)?;
            budget.release_storage(scratch_storage)?;
            Ok::<_, InstanceCorrespondenceErrorV1>(())
        })
        .unwrap();
        Ok::<_, ProductionCallInstanceErrorV1>(())
    })
    .unwrap();
    assert_eq!(budget.storage(), capture.retained_storage());
}

#[test]
fn placed_lowering_keeps_source_coordinates_and_relocates_call_continuations() {
    let lowered = lower_placed_root(
        helper_closure_semantic_owner(),
        SemanticEmissionPlacementV1 {
            first_block: 17,
            first_value: 100,
        },
    )
    .unwrap();
    let body = lowered.function.body.as_ref().unwrap();
    assert_eq!(
        body.blocks.iter().map(|block| block.id).collect::<Vec<_>>(),
        [BlockId(17), BlockId(18)]
    );
    assert!(matches!(
        body.blocks[0].terminator,
        Some(Terminator::Branch {
            target: BlockId(18),
            ..
        })
    ));
    assert_eq!(lowered.blocks[0].semantic_block.index(), 0);
    assert_eq!(lowered.blocks[0].kernel_ir_block, BlockId(17));
    assert_eq!(lowered.blocks[1].semantic_block.index(), 1);
    assert_eq!(lowered.blocks[1].kernel_ir_block, BlockId(18));
    assert_eq!(
        lowered.terminator_operation_spans[0].kernel_ir_block,
        BlockId(17)
    );
    assert!(matches!(
        body.blocks[0].operations[0].kind,
        OperationKind::Call { .. }
    ));
}

#[test]
fn placed_scalar_emission_uses_fresh_values_and_remains_valid_kernel_ir() {
    let lowered = lower_placed_root(
        scalar_transmute_semantic_owner(),
        SemanticEmissionPlacementV1 {
            first_block: 42,
            first_value: 100,
        },
    )
    .unwrap();
    let body = lowered.function.body.as_ref().unwrap();
    assert_eq!(body.blocks[0].id, BlockId(42));
    let definitions = body.blocks[0]
        .operations
        .iter()
        .flat_map(|op| &op.results)
        .map(|value| value.id)
        .collect::<Vec<_>>();
    assert_eq!(definitions, [ValueId(100), ValueId(101)]);
    assert!(matches!(
        body.blocks[0].operations[1].kind,
        OperationKind::Cast {
            value: ValueId(100),
            ..
        }
    ));
    assert_eq!(
        lowered.statement_operation_spans[0].kernel_ir_block,
        BlockId(42)
    );
    let mut module = Module::new("placed");
    module.functions.push(lowered.function);
    module.kernels.push(Kernel::new(
        "placed_root",
        "placed_root",
        LaunchDomain::D1 {
            x: LaunchExtent::Static(64),
        },
    ));
    verify_module(&module).unwrap();
}

#[test]
fn placed_emission_refuses_block_or_value_identity_exhaustion() {
    assert!(
        lower_placed_root(
            helper_closure_semantic_owner(),
            SemanticEmissionPlacementV1 {
                first_block: u32::MAX,
                first_value: 0
            },
        )
        .is_err()
    );
    assert!(
        lower_placed_root(
            scalar_transmute_semantic_owner(),
            SemanticEmissionPlacementV1 {
                first_block: 0,
                first_value: u32::MAX
            },
        )
        .is_err()
    );
}

#[test]
fn placed_volatile_call_preserves_mapped_success_and_failure_edges() {
    with_volatile_load_context_for_test(
        0,
        16,
        SemanticTypeIdV1::from_index(2),
        AddressSpace::Global,
        AccessMode::ReadOnly,
        false,
        |lowering, call| {
            lowering.emission_placement = SemanticEmissionPlacementV1 {
                first_block: 17,
                first_value: 100,
            };
            lowering.next_value = 100;
            lowering.assert_failure_block = Some(BlockId(19));
            let mut operations = Vec::new();
            let result =
                lowering.lower_call(SemanticBlockIdV1::from_index(0), call, &mut operations)?;
            assert!(matches!(
                result,
                Terminator::ConditionalBranch {
                    then_target: BlockId(18),
                    else_target: BlockId(19),
                    ..
                }
            ));
            assert!(
                operations
                    .iter()
                    .flat_map(|op| &op.results)
                    .all(|value| value.id.0 >= 100)
            );
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn placed_goto_keeps_source_edge_selection() {
    with_volatile_load_context_for_test(
        0,
        16,
        SemanticTypeIdV1::from_index(2),
        AddressSpace::Global,
        AccessMode::ReadOnly,
        false,
        |lowering, _| {
            lowering.emission_placement = SemanticEmissionPlacementV1 {
                first_block: 17,
                first_value: 100,
            };
            let edge =
                |role| SemanticControlFlowEdgeV1::new(role, SemanticBlockIdV1::from_index(1));
            let mut operations = Vec::new();
            let result = lowering.lower_terminator(
                SemanticBlockIdV1::from_index(0),
                &SemanticTerminatorKindV1::Goto(edge(SemanticEdgeRoleV1::Goto)),
                &mut operations,
            )?;
            assert!(matches!(
                result,
                Terminator::Branch {
                    target: BlockId(18),
                    ..
                }
            ));
            Ok(())
        },
    )
    .unwrap();
}

#[test]
fn placed_source_switches_preserve_nonzero_entry_and_loop_phi_transport() {
    for integer_switch in [false, true] {
        let (types, original) =
            authenticated_induction_fixture_v1(AuthenticatedInductionFixtureV1::default());
        let types = types
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                let layout = match ty.shape() {
                    SemanticTypeShapeV1::Scalar(scalar) => {
                        let (size, bits, maximum) = match scalar {
                            SemanticScalarTypeV1::Bool => (1, 8, 1),
                            SemanticScalarTypeV1::Integer {
                                signed: false,
                                bits: 32,
                            } => (4, 32, u128::from(u32::MAX)),
                            SemanticScalarTypeV1::Integer {
                                signed: false,
                                bits: 64,
                            } => (8, 64, u128::from(u64::MAX)),
                            _ => panic!("unexpected induction-fixture scalar"),
                        };
                        SemanticTypeLayoutV1::new_with_backend_repr(
                            Some(size),
                            size,
                            SemanticBackendReprV1::scalar(SemanticBackendScalarV1::initialized(
                                SemanticBackendPrimitiveV1::integer(false, bits, size),
                                SemanticScalarValidityRangeV1::new(0, maximum),
                            )),
                            false,
                        )
                        .unwrap()
                    }
                    SemanticTypeShapeV1::Unit => ty.layout().clone(),
                    _ => panic!("unexpected induction-fixture type"),
                };
                SemanticTypeDeclV1::new(
                    SemanticTypeIdentityV1::from_sha256([10 + index as u8; 32]),
                    ty.layout_identity(),
                    layout,
                    ty.shape().clone(),
                )
                .with_rustc_abi_properties(ty.abi_properties())
                .with_rust_type_kind(ty.rust_type_kind())
            })
            .collect();
        let mut blocks = original.blocks().to_vec();
        if integer_switch {
            let header = &blocks[1];
            blocks[1] = SemanticBasicBlockV1::new(
                header.identity(),
                header.source(),
                header.statements().to_vec(),
                SemanticTerminatorV1::new(
                    header.source(),
                    SemanticTerminatorKindV1::SwitchInt {
                        discriminant: SemanticOperandV1::Copy(
                            SemanticPlaceV1::new(
                                SemanticLocalIdV1::from_index(1),
                                vec![],
                                SemanticTypeIdV1::from_index(1),
                            )
                            .unwrap(),
                        ),
                        targets: SemanticSwitchTargetsV1::new(
                            vec![SemanticSwitchTargetV1::new(
                                64,
                                SemanticControlFlowEdgeV1::new(
                                    SemanticEdgeRoleV1::SwitchValue,
                                    SemanticBlockIdV1::from_index(4),
                                ),
                            )],
                            SemanticControlFlowEdgeV1::new(
                                SemanticEdgeRoleV1::SwitchOtherwise,
                                SemanticBlockIdV1::from_index(2),
                            ),
                        )
                        .unwrap(),
                    },
                ),
            )
            .unwrap();
        }
        blocks.push(
            SemanticBasicBlockV1::new(
                SemanticBlockIdentityV1::from_sha256([230; 32]),
                original.source(),
                vec![],
                SemanticTerminatorV1::new(
                    original.source(),
                    SemanticTerminatorKindV1::Goto(SemanticControlFlowEdgeV1::new(
                        SemanticEdgeRoleV1::Goto,
                        SemanticBlockIdV1::from_index(0),
                    )),
                ),
            )
            .unwrap(),
        );
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([233; 32]),
            SemanticLayoutIdentityV1::from_sha256([232; 32]),
            SemanticCanonAbiV1::GpuKernel,
            SemanticExternAbiV1::GpuKernel,
            false,
            false,
            0,
            vec![],
            SemanticAbiValueV1::new(
                SemanticTypeIdV1::from_index(0),
                SemanticAbiPassModeV1::Ignore,
            ),
        )
        .unwrap();
        let function = SemanticFunctionDeclV1::new(
            original.identity(),
            SemanticFunctionRoleV1::KernelRoot,
            original.item_definition_identity(),
            original.monomorphization_identity(),
            original.generic_type_arguments_identity(),
            original.const_generic_arguments_identity(),
            original.source(),
            abi,
            original.locals().to_vec(),
            SemanticBlockIdV1::from_index(5),
            blocks,
        )
        .unwrap()
        .with_kernel_entry(SemanticKernelEntryV1::new(
            SemanticLinkSymbolV1::new(b"placed_loop".to_vec()).unwrap(),
            SemanticKernelBindingIdentityV1::from_sha256([231; 32]),
            SemanticKernelSourceContractV1::new(None, None, None).unwrap(),
        ));
        let admitted = InertSemanticMirRequestV1::new(
            SemanticTargetDataLayoutV1::gfx942(SemanticLayoutIdentityV1::from_sha256([232; 32])),
            types,
            vec![],
            vec![],
            vec![],
            vec![function],
            vec![SemanticFunctionIdV1::from_index(0)],
        )
        .unwrap()
        .admit_current_production(SemanticMirLimitsV1::default())
        .unwrap();
        let owner = ProductionSemanticMirOwnerV1::try_new(
            admitted,
            ProductionSemanticMirLimitsV1::default(),
        )
        .unwrap();
        let lowered = lower_placed_root(
            owner,
            SemanticEmissionPlacementV1 {
                first_block: 17,
                first_value: 100,
            },
        )
        .unwrap();
        let body = lowered.function.body.as_ref().unwrap();
        assert_eq!(body.blocks.len(), 6);
        assert_eq!(body.blocks[0].id, BlockId(22));
        let block = |id| {
            body.blocks
                .iter()
                .find(|block| block.id == BlockId(id))
                .unwrap()
        };
        assert!(matches!(
            block(22).terminator,
            Some(Terminator::Branch {
                target: BlockId(17),
                ..
            })
        ));
        assert_eq!(block(18).parameters.len(), 1);
        assert_eq!(block(18).parameters[0].ty, Type::Scalar(ScalarType::U64));
        if integer_switch {
            assert!(matches!(&block(18).terminator,
                Some(Terminator::Switch { selector, cases, default_target: BlockId(19), .. })
                if *selector == block(18).parameters[0].id && cases.len() == 1
                    && cases[0].value == 64 && cases[0].target == BlockId(21)));
        } else {
            assert!(matches!(
                block(18).terminator,
                Some(Terminator::ConditionalBranch {
                    then_target: BlockId(19),
                    else_target: BlockId(21),
                    ..
                })
            ));
        }
        for predecessor in [17, 20] {
            let Some(Terminator::Branch { target, arguments }) = &block(predecessor).terminator
            else {
                panic!("initialization and latch must branch to the relocated header");
            };
            assert_eq!(*target, BlockId(18));
            assert_eq!(arguments.len(), 1);
            let definition = block(predecessor)
                .operations
                .iter()
                .find(|operation| {
                    operation
                        .results
                        .iter()
                        .any(|value| value.id == arguments[0])
                })
                .unwrap();
            if predecessor == 17 {
                assert_eq!(definition.kind, OperationKind::Constant(Constant::U64(0)));
            } else {
                assert!(matches!(
                    definition.kind,
                    OperationKind::Binary {
                        op: BinaryOp::Checked(CheckedBinaryOperator::Add),
                        ..
                    }
                ));
                assert_eq!(arguments[0], definition.results[0].id);
            }
        }
        let values = body
            .blocks
            .iter()
            .flat_map(|block| {
                block.parameters.iter().chain(
                    block
                        .operations
                        .iter()
                        .flat_map(|operation| &operation.results),
                )
            })
            .map(|value| value.id)
            .collect::<Vec<_>>();
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| value.0 >= 100));
        assert_eq!(
            values.iter().copied().collect::<BTreeSet<_>>().len(),
            values.len()
        );
        assert_eq!(lowered.blocks.len(), 6);
        for source in 0..6 {
            assert!(
                lowered
                    .blocks
                    .iter()
                    .any(|binding| binding.semantic_block.index() == source
                        && binding.kernel_ir_block == BlockId(17 + source))
            );
            assert!(
                lowered
                    .terminator_operation_spans
                    .iter()
                    .any(|span| span.semantic_block.index() == source
                        && span.kernel_ir_block == BlockId(17 + source))
            );
        }
        let mut module = Module::new("placed_loop");
        module.functions.push(lowered.function);
        module.kernels.push(Kernel::new(
            "placed_root",
            "placed_root",
            LaunchDomain::D1 {
                x: LaunchExtent::Static(64),
            },
        ));
        verify_module(&module).unwrap();
    }
}
