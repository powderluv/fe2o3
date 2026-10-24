// Physical signatures for expanded instances. Nominal bindings arrive only via
// the scoped call seed; this plan is not producer or borrow authority.
include!("production_execution_owner_parameter_v29.rs");

fn execution_reference_abi_scalar_v29(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
) -> Result<SemanticBackendScalarV1, ProductionSemanticKirErrorV1> {
    fe2o3_pliron::source_argument_v1::execution_reference_abi_scalar_v29(types, ty)
        .map_err(Into::into)
}

fn execution_reference_parameter_v29(
    types: &[SemanticTypeDeclV1],
    mapped: fe2o3_mir_model::SemanticAdjustedArgumentV1<'_>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<bool, ProductionSemanticKirErrorV1> {
    budget.charge_work(16)?;
    let argument = mapped.abi();
    if execution_cfg_nominal_kind_v29(types, argument.ty())? != Some(true) {
        return Ok(false);
    }
    let declaration = &types[argument.ty().index() as usize];
    let SemanticTypeShapeV1::Pointer(pointer) = declaration.shape() else {
        return Err(execution_call_error_v29());
    };
    let ownership = if mapped.tuple_field().is_some() {
        SemanticSourceArgumentOwnershipV1::ByValue
    } else {
        match pointer.mutability() {
            SemanticMutabilityV1::Immutable => SemanticSourceArgumentOwnershipV1::SharedBorrow,
            SemanticMutabilityV1::Mutable => SemanticSourceArgumentOwnershipV1::UniqueBorrow,
        }
    };
    execution_reference_abi_scalar_v29(types, argument.ty())?;
    if mapped.source_ownership() != ownership
        || !matches!(argument.mode(), SemanticAbiPassModeV1::Direct(_))
        || argument.value().adjusted().is_some()
        || argument.value().pointee_override().is_some()
    {
        return Err(execution_call_error_v29());
    }
    Ok(true)
}

fn execution_parameter_is_nominal_v29(
    types: &[SemanticTypeDeclV1],
    mut ty: SemanticTypeIdV1,
    path: &[SemanticKirParameterProjectionV1],
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<bool, ProductionSemanticKirErrorV1> {
    for projection in path {
        budget.charge_work(2)?;
        if execution_cfg_nominal_kind_v29(types, ty)?.is_some() {
            return Ok(true);
        }
        ty = match (types[ty.index() as usize].shape(), projection) {
            (
                SemanticTypeShapeV1::Tuple(fields) | SemanticTypeShapeV1::Aggregate(fields),
                SemanticKirParameterProjectionV1::Field(field),
            ) => *fields
                .fields()
                .get(*field as usize)
                .ok_or_else(execution_call_error_v29)?,
            (
                SemanticTypeShapeV1::Array { element, length },
                SemanticKirParameterProjectionV1::ArrayIndex(index),
            ) if u64::from(*index) < *length => *element,
            _ => return Err(execution_call_error_v29()),
        };
    }
    budget.charge_work(1)?;
    Ok(execution_cfg_nominal_kind_v29(types, ty)?.is_some())
}

fn check_execution_instance_abi_v29(
    semantic: &AdmittedInertSemanticMirV1,
    function_id: SemanticFunctionIdV1,
    references: Option<(&SourceReferencePlanV29<'_, '_>, ProductionCallInstanceIdV1)>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let function = &semantic.functions()[function_id.index() as usize];
    check_argument_function_abi_v1(
        function,
        function_id,
        SemanticKirFunctionRoleV1::InternalHelper,
    )?;
    let abi = function.abi();
    let sources = abi.source_input_types().len();
    let adjusted = abi.adjusted_arguments().len();
    let floor = budget.storage();
    let result = (|| {
        budget.charge_work(argument_sum_v1(&[
            argument_product_v1(function.locals().len(), 2)?,
            sources,
            adjusted,
        ])?)?;
        budget.reserve_storage(argument_sum_v1(&[
            argument_product_v1(sources, std::mem::size_of::<Option<SemanticLocalIdV1>>())?,
            argument_product_v1(adjusted, std::mem::size_of::<SemanticLocalIdV1>())?,
        ])?)?;
        let arguments =
            semantic
                .logical_arguments_v1(function_id)
                .map_err(|error| match error {
                    fe2o3_mir_model::SemanticLogicalArgumentErrorV1::AllocationFailure => {
                        ProductionSemanticKirErrorV1::from(ArgumentResourceV1::Allocation)
                    }
                    fe2o3_mir_model::SemanticLogicalArgumentErrorV1::UnknownFunction => {
                        execution_call_error_v29()
                    }
                })?;
        for argument in arguments.source_arguments() {
            budget.charge_work(1)?;
            if matches!(
                argument.binding(),
                fe2o3_mir_model::SemanticSourceArgumentBindingV1::ExpandedTuple([])
            ) && argument.source_ownership() != SemanticSourceArgumentOwnershipV1::ByValue
            {
                return Err(execution_call_error_v29());
            }
        }
        for mapped in arguments.adjusted_arguments() {
            if let Some((references, instance)) = references
                && check_source_reference_parameter_v29(references, instance, mapped, budget)?
            {
                continue;
            }
            if execution_reference_parameter_v29(semantic.types(), mapped, budget)? {
                continue;
            }
            let shape_floor = budget.storage();
            if let Some(direct) = execution_direct_parameter_v29(
                semantic,
                function_id,
                mapped.source_argument(),
                mapped.tuple_field(),
                mapped.abi().ty(),
                budget,
            )? {
                if mapped.ordinal() != mapped.source_argument()
                    || mapped.local_field().is_some()
                    || function.locals()[mapped.local().index() as usize].role()
                        != SemanticLocalRoleV1::Argument(mapped.source_argument())
                    || !std::ptr::eq(
                        mapped.abi(),
                        &abi.adjusted_arguments()[mapped.ordinal() as usize],
                    )
                {
                    return Err(execution_call_error_v29());
                }
                drop(direct);
                budget.release_storage(budget.storage() - shape_floor)?;
                continue;
            }
            prepay_argument_shape_v1(semantic, mapped.abi().ty(), budget)?;
            let shape_builder = if references.is_some() {
                source_helper_parameter_shape_with_policy_v18
            } else {
                helper_parameter_shape_with_policy_v1
            };
            let (_, shape) = shape_builder(
                semantic.types(),
                function,
                function_id,
                mapped,
                ParameterLeafPolicyV1::ExecutionAbiWords,
            )?;
            let physical = if references.is_some() {
                source_execution_cfg_types_v29(semantic.types(), mapped.abi().ty(), budget)?
            } else {
                execution_cfg_types_v29(semantic.types(), mapped.abi().ty(), budget)?
            };
            let mut ordinary = physical.iter();
            for (path, _, expected) in &shape {
                if !execution_parameter_is_nominal_v29(
                    semantic.types(),
                    mapped.abi().ty(),
                    path,
                    budget,
                )? && ordinary.next() != Some(expected)
                {
                    return Err(execution_call_error_v29());
                }
            }
            if ordinary.next().is_some() {
                return Err(execution_call_error_v29());
            }
            drop((shape, physical));
            budget.release_storage(budget.storage() - shape_floor)?;
        }
        Ok(())
    })();
    // This checker returns no allocations; preserve the caller's retained floor.
    budget.release_storage(budget.storage() - floor)?;
    result
}

fn execution_instance_plan_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    kernel_ir_function: FunctionId,
    placement: SemanticEmissionPlacementV1,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
    execution_instance_plan_with_references_v29(
        instances,
        instance,
        kernel_ir_function,
        placement,
        None,
        budget,
    )
}

fn execution_instance_plan_with_references_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    kernel_ir_function: FunctionId,
    placement: SemanticEmissionPlacementV1,
    references: Option<&SourceReferencePlanV29<'_, '_>>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
    if let Some(references) = references {
        references.check_owner(instances, budget)?;
    }
    let floor = budget.storage();
    if let Some(references) = references {
        source_reference_owned_prepay_v29::<LoweredFunctionPlanV1>(references, budget)?;
    }
    let result = build_execution_instance_plan_v29(
        instances,
        instance,
        kernel_ir_function,
        placement,
        references,
        budget,
    )
    .inspect_err(|error| {
        if let Some(references) = references {
            source_reference_record_failure_v29(references, error);
        }
    });
    if result.is_err() {
        budget.release_storage(budget.storage() - floor)?;
    }
    result
}

fn build_execution_instance_plan_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    kernel_ir_function: FunctionId,
    placement: SemanticEmissionPlacementV1,
    references: Option<&SourceReferencePlanV29<'_, '_>>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
    let layout = execution_function_layout_v29(instances, instance, references, budget)?;
    execution_instance_plan_from_layout_v29(
        instances,
        instance,
        kernel_ir_function,
        placement,
        layout,
        budget,
    )
}

fn execution_instance_plan_from_layout_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    kernel_ir_function: FunctionId,
    placement: SemanticEmissionPlacementV1,
    layout: ExecutionFunctionLayoutV29,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
    let row = instances
        .instance(instance)
        .ok_or_else(execution_call_error_v29)?;
    budget.charge_work(layout.parameter_types.len())?;
    let mut parameter_values = emission_vec_v1(layout.parameter_types.len(), budget)?;
    let mut next_value = placement.first_value;
    for _ in &layout.parameter_types {
        let id = ValueId(next_value);
        next_value = next_value
            .checked_add(1)
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        parameter_values.push(id);
    }
    Ok(LoweredFunctionPlanV1 {
        correspondence_owner: instances
            .instance(instances.root())
            .ok_or_else(execution_call_error_v29)?
            .function(),
        semantic_function: row.function(),
        kernel_ir_function,
        role: SemanticKirFunctionRoleV1::InternalHelper,
        parameter_declarations: layout.parameter_declarations,
        parameter_types: layout.parameter_types,
        parameter_values,
        call_arguments: layout.call_arguments,
        parameter_local_bindings: Vec::new(),
        parameter_component_bindings: Vec::new(),
        ignored_parameter_bindings: Vec::new(),
        result_types: layout.result_types,
    })
}

struct ExecutionFunctionLayoutV29 {
    parameter_declarations: Vec<(u32, usize, SemanticTypeIdV1)>,
    parameter_types: Vec<Type>,
    call_arguments: Vec<HelperCallArgumentV1>,
    result_types: Vec<Type>,
}

fn execution_function_layout_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    references: Option<&SourceReferencePlanV29<'_, '_>>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<ExecutionFunctionLayoutV29, ProductionSemanticKirErrorV1> {
    if let Some(references) = references {
        references.check_owner(instances, budget)?;
    }
    let floor = budget.storage();
    if let Some(references) = references {
        source_reference_owned_prepay_v29::<ExecutionFunctionLayoutV29>(references, budget)?;
    }
    let result = build_execution_function_layout_v29(instances, instance, references, budget)
        .inspect_err(|error| {
            if let Some(references) = references {
                source_reference_record_failure_v29(references, error);
            }
        });
    if result.is_err() {
        budget.release_storage(budget.storage() - floor)?;
    }
    result
}

fn execution_function_signature_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionSignatureV1, ProductionSemanticKirErrorV1> {
    execution_function_signature_with_references_v29(instances, instance, None, budget)
}

fn execution_function_signature_with_references_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    references: Option<&SourceReferencePlanV29<'_, '_>>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<LoweredFunctionSignatureV1, ProductionSemanticKirErrorV1> {
    if let Some(references) = references {
        references.check_owner(instances, budget)?;
    }
    let floor = budget.storage();
    if let Some(references) = references {
        source_reference_owned_prepay_v29::<LoweredFunctionSignatureV1>(references, budget)?;
    }
    let result = (|| {
        let layout = execution_function_layout_v29(instances, instance, references, budget)?;
        let function = instances
            .instance(instance)
            .ok_or_else(execution_call_error_v29)?
            .declaration();
        let inputs = function.abi().source_input_types();
        budget.charge_work(inputs.len())?;
        let mut parameter_semantic_types = emission_vec_v1(inputs.len(), budget)?;
        // Physical erasure never changes the source operand roster. In particular,
        // RustCall retains its whole source tuple and ignored closure arguments.
        parameter_semantic_types.extend_from_slice(inputs);
        let declaration_bytes = argument_product_v1(
            layout.parameter_declarations.capacity(),
            std::mem::size_of::<(u32, usize, SemanticTypeIdV1)>(),
        )?;
        drop(layout.parameter_declarations);
        budget.release_storage(declaration_bytes)?;
        Ok(LoweredFunctionSignatureV1 {
            bf16_nominal: false,
                wave64_nominal: false,
            parameter_semantic_types,
            call_arguments: layout.call_arguments,
            parameter_types: layout.parameter_types,
            result_types: layout.result_types,
            result_semantic_type: function.abi().source_output_type(),
        })
    })()
    .inspect_err(|error| {
        if let Some(references) = references {
            source_reference_record_failure_v29(references, error);
        }
    });
    if result.is_err() {
        budget.release_storage(budget.storage() - floor)?;
    }
    result
}

fn build_execution_function_layout_v29(
    instances: &ExecutionInstancesV29<'_>,
    instance: ProductionCallInstanceIdV1,
    references: Option<&SourceReferencePlanV29<'_, '_>>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<ExecutionFunctionLayoutV29, ProductionSemanticKirErrorV1> {
    if let Some(references) = references {
        references.check_owner(instances, budget)?;
    }
    budget.charge_work(8)?;
    let row = instances
        .instance(instance)
        .ok_or_else(execution_call_error_v29)?;
    let incoming = instances
        .incoming(instance)
        .ok_or_else(execution_call_error_v29)?;
    if incoming.child() != Some(instance) {
        return Err(execution_call_error_v29());
    }
    let semantic = instances.owner().source_semantic();
    let function = row.declaration();
    check_execution_instance_abi_v29(
        semantic,
        row.function(),
        references.map(|plan| (plan, instance)),
        budget,
    )?;
    let mut parameter_declarations = Vec::new();
    let mut parameter_types = Vec::new();
    let mut call_arguments = Vec::new();
    // Entry-local order is also the scoped constructor's reconstruction order.
    // A packed RustCall tuple stays whole; expanded fields retain their selector.
    for (local, declaration) in function.locals().iter().enumerate() {
        budget.charge_work(1)?;
        if !declaration.role().is_entry_argument() {
            continue;
        }
        let selector = instances
            .parameter_source(
                instance,
                SemanticLocalIdV1::from_index(
                    u32::try_from(local).map_err(|_| ArgumentResourceV1::Arithmetic)?,
                ),
                budget,
            )
            .map_err(|error| match error {
                production_call_instances_v1::ProductionCallInstanceErrorV1::Resource(error) => {
                    error.into()
                }
                _ => execution_call_error_v29(),
            })?;
        emission_push_v1(
            &mut parameter_declarations,
            (selector.source_argument, local, selector.ty),
            budget,
        )?;
        if let Some(direct) = execution_direct_parameter_v29(
            semantic,
            row.function(),
            selector.source_argument,
            selector.tuple_field,
            selector.ty,
            budget,
        )? {
            emission_push_v1(&mut parameter_types, direct, budget)?;
            emission_push_v1(
                &mut call_arguments,
                HelperCallArgumentV1 {
                    source_argument: selector.source_argument,
                    tuple_field: None,
                    component: None,
                },
                budget,
            )?;
            continue;
        }
        let physical = if let Some(references) = references {
            let node = source_reference_entry_node_v29(
                references,
                instance,
                SemanticLocalIdV1::from_index(
                    u32::try_from(local).map_err(|_| ArgumentResourceV1::Arithmetic)?,
                ),
                None,
                budget,
            )?
            .ok_or_else(execution_call_error_v29)?;
            if references.nodes[node].ty != selector.ty {
                return Err(execution_call_error_v29());
            }
            source_reference_node_types_v29(references, node, budget)?
        } else {
            execution_cfg_types_v29(semantic.types(), selector.ty, budget)?
        };
        let backing = argument_product_v1(physical.capacity(), std::mem::size_of::<Type>())?;
        for (component, ty) in physical.into_iter().enumerate() {
            emission_push_v1(&mut parameter_types, ty, budget)?;
            emission_push_v1(
                &mut call_arguments,
                HelperCallArgumentV1 {
                    source_argument: selector.source_argument,
                    tuple_field: selector.tuple_field,
                    component: Some(component),
                },
                budget,
            )?;
        }
        budget.release_storage(backing)?;
    }
    let nominal_result = execution_cfg_nominal_count_v29(
        semantic.types(),
        function.abi().source_output_type(),
        budget,
    )? != 0;
    if references.is_none() && nominal_result {
        return Err(execution_call_error_v29());
    }
    if let Some(references) = references
        && let Some(result_types) = source_reference_return_types_v29(references, instance, budget)?
    {
        return Ok(ExecutionFunctionLayoutV29 {
            parameter_declarations,
            parameter_types,
            call_arguments,
            result_types,
        });
    }
    if nominal_result {
        if let Some(references) = references
            && instances.instance_reachable(instance) == Some(true)
            && instances.instance_may_return(instance) == Some(false)
        {
            let result_types = source_reference_no_normal_result_types_v29(
                instances, references, instance, budget,
            )?;
            return Ok(ExecutionFunctionLayoutV29 {
                parameter_declarations,
                parameter_types,
                call_arguments,
                result_types,
            });
        }
        return Err(execution_call_error_v29());
    }
    budget.charge_work(function.locals().len())?;
    let floor = budget.storage();
    prepay_typed_shape_v1(
        semantic.types(),
        function.abi().source_output_type(),
        0,
        budget,
    )?;
    let scratch = budget.storage() - floor;
    let result_types = (|| {
        let shape = helper_result_components_v1(semantic.types(), function, row.function())?;
        let mut result = emission_vec_v1(shape.components.len(), budget)?;
        for (_, _, ty, _, _) in &shape.components {
            result.push(execution_cfg_clone_type_v29(ty, budget)?);
        }
        Ok::<_, ProductionSemanticKirErrorV1>(result)
    })();
    budget.release_storage(scratch)?;
    Ok(ExecutionFunctionLayoutV29 {
        parameter_declarations,
        parameter_types,
        call_arguments,
        result_types: result_types?,
    })
}
