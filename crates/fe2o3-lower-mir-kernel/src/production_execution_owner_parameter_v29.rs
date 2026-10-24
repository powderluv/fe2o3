// Whole owned arguments keep their authenticated entry representation across
// a scoped helper boundary. This is neither structural expansion nor ownership
// authority: source ABI custody and callee reconstruction remain mandatory.
fn execution_direct_parameter_v29(
    semantic: &AdmittedInertSemanticMirV1,
    function_id: SemanticFunctionIdV1,
    source_argument: u32,
    tuple_field: Option<u32>,
    ty: SemanticTypeIdV1,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<Option<Type>, ProductionSemanticKirErrorV1> {
    let floor = budget.storage();
    let result = (|| {
        budget.charge_work(8)?;
        let function = semantic
            .functions()
            .get(function_id.index() as usize)
            .ok_or_else(execution_call_error_v29)?;
        let abi = function.abi();
        let index = source_argument as usize;
        if tuple_field.is_some()
            || abi.source_argument_ownership().get(index)
                != Some(&SemanticSourceArgumentOwnershipV1::ExclusiveOwner)
        {
            return Ok(None);
        }
        let adjusted = abi
            .adjusted_arguments()
            .get(index)
            .ok_or_else(execution_call_error_v29)?;
        if abi.source_input_types().get(index) != Some(&ty)
            || adjusted.role() != SemanticAbiArgumentRoleV1::Source
            || adjusted.ty() != ty
            || adjusted.value().adjusted().is_some()
            || (matches!(adjusted.mode(), SemanticAbiPassModeV1::Pair { .. })
                && adjusted.value().pointee_override().is_some())
        {
            return Err(execution_call_error_v29());
        }
        prepay_argument_shape_v1(semantic, ty, budget)?;
        if execution_cfg_nominal_count_v29(semantic.types(), ty, budget)? != 0 {
            return Err(execution_call_error_v29());
        }
        let scratch = budget.storage() - floor;
        let authenticated = authenticated_disjoint_slice_parameter(
            semantic.types(),
            semantic.callables(),
            function,
            source_argument,
            ty,
        )
        .or_else(|| {
            authenticated_global_mut_pointer_parameter(
                semantic.types(),
                function,
                source_argument,
                ty,
            )
        })
        .ok_or_else(execution_call_error_v29)?;
        let retained = execution_cfg_clone_type_v29(&authenticated, budget)?;
        drop(authenticated);
        budget.release_storage(scratch)?;
        Ok(Some(retained))
    })();
    if result.is_err() {
        budget.release_storage(budget.storage() - floor)?;
    }
    result
}

fn execution_is_direct_owner_type_v29(ty: &Type) -> bool {
    match ty {
        Type::Pointer(pointer) => {
            pointer.address_space == AddressSpace::Global
                && matches!(
                    pointer.access,
                    AccessMode::ReadWrite | AccessMode::WriteOnly
                )
        }
        Type::Slice(slice) => {
            slice.address_space == AddressSpace::Global
                && matches!(slice.access, AccessMode::ReadWrite | AccessMode::WriteOnly)
        }
        _ => false,
    }
}

fn clone_execution_function_signature_v29(
    signature: &LoweredFunctionSignatureV1,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<LoweredFunctionSignatureV1, ProductionSemanticKirErrorV1> {
    budget.charge_work(argument_sum_v1(&[
        signature.parameter_semantic_types.len(),
        signature.call_arguments.len(),
    ])?)?;
    let mut parameter_semantic_types =
        emission_vec_v1(signature.parameter_semantic_types.len(), budget)?;
    parameter_semantic_types.extend_from_slice(&signature.parameter_semantic_types);
    let mut call_arguments = emission_vec_v1(signature.call_arguments.len(), budget)?;
    call_arguments.extend_from_slice(&signature.call_arguments);
    let mut parameter_types = emission_vec_v1(signature.parameter_types.len(), budget)?;
    for ty in &signature.parameter_types {
        parameter_types.push(execution_cfg_clone_type_v29(ty, budget)?);
    }
    let mut result_types = emission_vec_v1(signature.result_types.len(), budget)?;
    for ty in &signature.result_types {
        result_types.push(execution_cfg_clone_type_v29(ty, budget)?);
    }
    Ok(LoweredFunctionSignatureV1 {
        bf16_nominal: signature.bf16_nominal,
        wave64_nominal: signature.wave64_nominal,
        parameter_semantic_types,
        call_arguments,
        parameter_types,
        result_types,
        result_semantic_type: signature.result_semantic_type,
    })
}

// Borrow the private signature; do not allocate another representation roster.
fn execution_direct_projection_v29<'a>(
    source_argument: u32,
    projections: &[HelperCallArgumentV1],
    parameter_types: &'a [Type],
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Option<(usize, &'a Type)>, ProductionSemanticKirErrorV1> {
    budget.charge_work(projections.len())?;
    if projections.len() != parameter_types.len() {
        return Err(execution_call_error_v29());
    }
    let mut count = 0usize;
    let mut direct = None;
    for (index, (projection, ty)) in projections.iter().zip(parameter_types).enumerate() {
        if projection.source_argument != source_argument {
            continue;
        }
        count += 1;
        if projection.component.is_none() && execution_is_direct_owner_type_v29(ty) {
            if projection.tuple_field.is_some() {
                return Err(execution_call_error_v29());
            }
            direct = Some((index, ty));
        }
    }
    if direct.is_some() && count != 1 {
        return Err(execution_call_error_v29());
    }
    Ok(direct)
}

fn execution_call_argument_shape_v29(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    source_argument: u32,
    binding: &SemanticValueBindingV1,
    projections: &[HelperCallArgumentV1],
    parameter_types: &[Type],
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    execution_call_argument_shape_with_representation_v29(
        types,
        ty,
        source_argument,
        binding,
        projections,
        parameter_types,
        ExecutionCfgRepresentationV29::LegacyAbi,
        budget,
    )
}

fn execution_call_argument_shape_with_representation_v29(
    types: &[SemanticTypeDeclV1],
    ty: SemanticTypeIdV1,
    source_argument: u32,
    binding: &SemanticValueBindingV1,
    projections: &[HelperCallArgumentV1],
    parameter_types: &[Type],
    representation: ExecutionCfgRepresentationV29,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if let Some((_, expected)) =
        execution_direct_projection_v29(source_argument, projections, parameter_types, budget)?
    {
        if execution_cfg_nominal_count_v29(types, ty, budget)? != 0
            || !matches!(binding, SemanticValueBindingV1::Value { ty, .. } if ty == expected)
        {
            return Err(execution_call_error_v29());
        }
    } else {
        execution_call_shape_with_representation_v29(types, ty, binding, representation, budget)?;
    }
    Ok(())
}
