use super::*;

#[path = "production_instance_assert_origins_v1_tests.rs"]
mod instance_assert_tests;
#[path = "production_pending_scope_root_v29_tests.rs"]
mod pending_tests;
use fe2o3_mir_model::semantic_mir_v1::*;

fn scalar_calls_owner(nested: bool) -> ProductionSemanticMirOwnerV1 {
    let original = statement_order_owner();
    let semantic = original.semantic();
    let root = &semantic.functions()[0];
    let source = root.source();
    let scalar = SemanticTypeIdV1::from_index(1);
    let place =
        |local| SemanticPlaceV1::new(SemanticLocalIdV1::from_index(local), vec![], scalar).unwrap();
    let call = |callee, argument, destination, target| {
        SemanticTerminatorKindV1::Call(
            SemanticDirectCallV1::new(
                SemanticFunctionIdV1::from_index(callee),
                vec![SemanticOperandV1::Copy(place(argument))],
                Some(SemanticCallDestinationV1::new(
                    place(destination),
                    SemanticControlFlowEdgeV1::new(
                        SemanticEdgeRoleV1::CallReturn,
                        SemanticBlockIdV1::from_index(target),
                    ),
                )),
                SemanticUnwindActionV1::Unreachable,
            )
            .unwrap(),
        )
    };
    let block = |tag, statements, terminator| {
        SemanticBasicBlockV1::new(
            SemanticBlockIdentityV1::from_sha256([tag; 32]),
            source,
            statements,
            SemanticTerminatorV1::new(source, terminator),
        )
        .unwrap()
    };
    let root = SemanticFunctionDeclV1::new(
        root.identity(),
        root.role(),
        root.item_definition_identity(),
        root.monomorphization_identity(),
        root.generic_type_arguments_identity(),
        root.const_generic_arguments_identity(),
        source,
        root.abi().clone(),
        root.locals().to_vec(),
        root.entry(),
        vec![
            block(
                208,
                root.blocks()[0].statements().to_vec(),
                call(1, 1, 2, 1),
            ),
            block(209, vec![], call(1, 2, 1, 2)),
            block(210, vec![], SemanticTerminatorKindV1::Return),
        ],
    )
    .unwrap()
    .with_kernel_entry(root.kernel_entry().unwrap().clone());
    let direct = || {
        SemanticAbiValueV1::new(
            scalar,
            SemanticAbiPassModeV1::Direct(
                SemanticAbiValueAttributesV1::new(
                    SemanticAbiRegularAttributesV1::new(false, None, false, false, false, true),
                    SemanticAbiExtensionV1::None,
                    0,
                    None,
                )
                .unwrap(),
            ),
        )
    };
    let helper = |tag: u8, calls_leaf: bool| {
        let abi = SemanticFunctionAbiV1::from_rustc(
            SemanticAbiIdentityV1::from_sha256([tag + 5; 32]),
            SemanticLayoutIdentityV1::from_sha256([250; 32]),
            SemanticCanonAbiV1::Rust,
            SemanticExternAbiV1::Rust,
            false,
            false,
            1,
            vec![SemanticAbiArgumentV1::source(direct())],
            direct(),
        )
        .unwrap()
        .with_source_argument_ownership(vec![SemanticSourceArgumentOwnershipV1::ByValue])
        .unwrap();
        let local = |offset, role| {
            SemanticLocalDeclV1::new(
                SemanticLocalIdentityV1::from_sha256([tag + offset; 32]),
                scalar,
                role,
                source,
            )
        };
        let blocks = if calls_leaf {
            vec![
                block(tag + 8, vec![], call(2, 1, 0, 1)),
                block(tag + 9, vec![], SemanticTerminatorKindV1::Return),
            ]
        } else {
            vec![block(
                tag + 8,
                vec![SemanticStatementV1::new(
                    source,
                    SemanticStatementKindV1::Assign(SemanticAssignmentV1::new(
                        place(0),
                        SemanticRvalueV1::new(
                            scalar,
                            SemanticRvalueKindV1::Use(SemanticOperandV1::Copy(place(1))),
                        ),
                    )),
                )],
                SemanticTerminatorKindV1::Return,
            )]
        };
        SemanticFunctionDeclV1::new(
            SemanticFunctionIdentityV1::from_sha256([tag; 32]),
            SemanticFunctionRoleV1::InternalHelper,
            SemanticItemDefinitionIdentityV1::from_sha256([tag + 1; 32]),
            SemanticMonomorphizationIdentityV1::from_sha256([tag + 2; 32]),
            SemanticGenericTypeArgumentsIdentityV1::from_sha256([tag + 3; 32]),
            SemanticConstGenericArgumentsIdentityV1::from_sha256([tag + 4; 32]),
            source,
            abi,
            vec![
                local(6, SemanticLocalRoleV1::Return),
                local(7, SemanticLocalRoleV1::Argument(0)),
            ],
            SemanticBlockIdV1::from_index(0),
            blocks,
        )
        .unwrap()
    };
    let mut functions = vec![root, helper(211, nested)];
    if nested {
        functions.push(helper(231, false));
    }
    let admitted = InertSemanticMirRequestV1::new(
        semantic.target(),
        semantic.types().to_vec(),
        vec![],
        vec![],
        vec![],
        functions,
        semantic.roots().to_vec(),
    )
    .unwrap()
    .admit(SemanticMirLimitsV1::default())
    .unwrap();
    ProductionSemanticMirOwnerV1::try_new(admitted, ProductionSemanticMirLimitsV1::default())
        .unwrap()
}

fn lower_scalar_instances(
    instances: &ProductionCallInstancePlanV1<'_>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Vec<LoweredFunctionResultV1> {
    let ssa = instances.owner();
    let semantic = ssa.source_semantic();
    let root = semantic.roots()[0];
    let mut templates = BTreeMap::new();
    let mut ids = BTreeMap::new();
    let mut signatures = BTreeMap::new();
    let mut closure = ReachableClosureBudgetV1::new(256);
    for (index, declaration) in semantic.functions().iter().enumerate().skip(1) {
        let id = SemanticFunctionIdV1::from_index(index as u32);
        let name = helper_function_id_v1(id, declaration);
        let plan =
            direct_scalar_helper_plan_v1(semantic, root, id, name.clone(), 256, &mut closure)
                .unwrap();
        assert_eq!(plan.parameter_types, vec![Type::Scalar(ScalarType::U32)]);
        assert_eq!(plan.result_types, plan.parameter_types);
        signatures.insert(
            id,
            LoweredFunctionSignatureV1 {
                bf16_nominal: false,
                wave64_nominal: false,
                parameter_semantic_types: declaration.abi().source_input_types().to_vec(),
                call_arguments: plan.call_arguments.clone(),
                parameter_types: plan.parameter_types.clone(),
                result_types: plan.result_types.clone(),
                result_semantic_type: declaration.abi().source_output_type(),
            },
        );
        templates.insert(id, plan);
        ids.insert(id, name);
    }
    instances
        .instances()
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let instance = instances.id_at(index).unwrap();
            let placement = SemanticEmissionPlacementV1 {
                first_block: index as u32 * 16,
                first_value: index as u32 * 128,
            };
            let plan = if index == 0 {
                LoweredFunctionPlanV1 {
                    correspondence_owner: root,
                    semantic_function: root,
                    kernel_ir_function: FunctionId::new("scalar_instance_root"),
                    role: SemanticKirFunctionRoleV1::KernelEntry,
                    parameter_declarations: vec![],
                    parameter_types: vec![],
                    parameter_values: vec![],
                    call_arguments: vec![],
                    parameter_local_bindings: vec![],
                    parameter_component_bindings: vec![],
                    ignored_parameter_bindings: vec![],
                    result_types: vec![],
                }
            } else {
                let mut plan = templates[&row.function()].clone();
                assert_eq!(plan.parameter_values.len(), 1);
                assert!(plan.parameter_component_bindings.is_empty());
                assert!(plan.ignored_parameter_bindings.is_empty());
                let parameter = ValueId(placement.first_value);
                plan.parameter_values[0] = parameter;
                let [PlannedParameterLocalBindingV1::Direct { local, value, ty }] =
                    plan.parameter_local_bindings.as_mut_slice()
                else {
                    panic!("scalar helper requires one direct source parameter");
                };
                assert_eq!(*local, 1);
                assert_eq!(*ty, Type::Scalar(ScalarType::U32));
                *value = parameter;
                plan
            };
            let mut private = PrivateArrayLazyBudgetV1::new(1, 256);
            let lowered =
                with_execution_availability_v29(instances, instance, budget, |cursor, budget| {
                    lower_one_semantic_function_v1(
                        semantic,
                        &plan,
                        ssa.plan_for_function(row.function()).unwrap(),
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
                        budget,
                        placement,
                        Some(cursor),
                    )
                })
                .unwrap();
            assert_eq!(lowered.source_call_instance, Some(instance));
            assert!(lowered.next_value < (index as u32 + 1) * 128);
            lowered
        })
        .collect()
}

fn with_scalar_expansion(
    nested: bool,
    test: impl FnOnce(
        &mut ProductionInstanceCorrespondenceV1<'_, '_>,
        &mut Function,
        &mut ArgumentBudgetV1<'_>,
    ),
) {
    with_plan_owner(scalar_calls_owner(nested), |instances, budget| {
        let floor = budget.storage();
        let lowered = lower_scalar_instances(instances, budget);
        let call_storage: usize = lowered
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.call_returns.requested_bytes().unwrap()
                    + scalar_lowering_scratch_storage_v29(row)
                    + row.scoped_initialization.as_ref().unwrap().retained_storage
                    + row
                        .scoped_memory_anchors
                        .as_ref()
                        .unwrap()
                        .retained_storage()
                        .unwrap()
                    + scalar_archive_storage_v1(
                        row,
                        instances,
                        instances.id_at(index).unwrap(),
                        budget,
                    )
            })
            .sum();
        with_production_instance_correspondence_v1(instances, budget, |map, budget| {
            let mut functions: Vec<_> = lowered
                .iter()
                .map(|row| Some(row.function.clone()))
                .collect();
            for (index, lowered) in lowered.iter().enumerate() {
                map.append_lowered(instances.id_at(index).unwrap(), lowered, budget)?;
            }
            let mut splice_storage = 0;
            // Children follow parents in the captured plan. Expand bottom-up
            // so each repeated call retains its own nested return coordinates.
            for child in (1..functions.len()).rev() {
                let instance = instances.id_at(child).unwrap();
                let call = instances.incoming(instance).unwrap();
                let caller = call.occurrence().caller.index();
                let split = 128 + child as u32 * 2;
                let expanded = map.splice(
                    call,
                    functions[caller].take().unwrap(),
                    functions[child].take().unwrap(),
                    BlockId(split),
                    BlockId(split + 1),
                    budget,
                )?;
                assert!(expanded.callee_required_capabilities.is_empty());
                splice_storage += expanded.additional_storage_bytes;
                functions[caller] = Some(expanded.caller);
            }
            let mut root = functions[0].take().unwrap();
            test(map, &mut root, budget);
            drop(root);
            budget.release_storage(splice_storage)?;
            Ok::<_, InstanceCorrespondenceErrorV1>(())
        })
        .unwrap();
        drop(lowered);
        budget.release_storage(call_storage).unwrap();
        assert_eq!(budget.storage(), floor);
    });
}

#[test]
fn scalar_repeated_and_nested_instances_transfer_nonempty_coordinates() {
    for nested in [false, true] {
        with_scalar_expansion(nested, |map, function, budget| {
            let count = if nested { 5 } else { 3 };
            assert_eq!(map.seeds.rows.len(), count);
            assert_eq!(map.components.rows.len(), count - 1);
            assert!(!map.values.rows.is_empty());
            let before = snapshot(map);
            let capacities = (map.components.capacity, map.values.capacity);
            let owned = map.take_owned_coordinates_v1(function, budget).unwrap();
            assert_eq!(
                (owned.components.capacity, owned.values.capacity),
                capacities
            );
            assert_eq!(owned.components.rows.as_ptr() as usize, before[13]);
            assert_eq!(owned.values.rows.as_ptr() as usize, before[15]);
            assert_eq!(owned.sources.rows.len(), count);
            assert_eq!(owned.anchors.rows.len(), count - 1);
            assert_eq!(owned.returns.rows.len(), count);
            let [first, second] = [&owned.sources.rows[1], &owned.sources.rows[2]];
            assert_eq!(first.function, second.function);
            assert_eq!(first.identity, second.identity);
            assert_ne!(first.instance, second.instance);
            assert_ne!(first.incoming, second.incoming);
            owned.check_source_plan(map.plan, budget).unwrap();
            let storage = owned.retained_storage();
            drop(owned);
            budget.release_storage(storage).unwrap();
        });
    }
}

#[test]
fn scalar_preheader_and_continuation_parameter_substitutions_are_refused() {
    for nested in [false, true] {
        for preheader in [false, true] {
            with_scalar_expansion(nested, |map, function, budget| {
                let call = map.plan.calls(map.plan.root()).unwrap()[0].occurrence();
                let control = map.controls.rows.iter().find(|control| {
                    if preheader {
                        matches!(control.origin, InstanceControlOriginV1::ParameterPreheader { call: actual, .. } if actual == call)
                    } else {
                        control.instance == call.caller
                            && control.semantic_block == Some(call.block)
                            && control.origin == InstanceControlOriginV1::Retained
                    }
                }).unwrap();
                let block = function
                    .body
                    .as_mut()
                    .unwrap()
                    .blocks
                    .iter_mut()
                    .find(|block| block.id == control.physical_block)
                    .unwrap();
                assert_eq!(block.parameters.len(), 1);
                assert_eq!(block.parameters[0].ty, Type::Scalar(ScalarType::U32));
                block.parameters[0].id = ValueId(1_000_000);
                let before = snapshot(map);
                let storage = budget.storage();
                assert!(matches!(
                    map.take_owned_coordinates_v1(function, budget),
                    Err(InstanceCorrespondenceErrorV1::Control)
                ));
                assert_eq!(snapshot(map), before);
                assert_eq!(budget.storage(), storage);
            });
        }
    }
}
