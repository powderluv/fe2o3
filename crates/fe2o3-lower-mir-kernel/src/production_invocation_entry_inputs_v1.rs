// Descriptive coordinates captured by the actual parameter installation. This
// map is retained inside the invocation relation, never a detached ABI permit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InvocationInputRowV1 {
    local: u32,
    ty: SemanticTypeIdV1,
    source_argument: u32,
    tuple_field: Option<u32>,
    first_parameter: usize,
    parameter_count: usize,
    reference_call_transport: Option<ReferenceCallTransportV26>,
}

#[allow(clippy::too_many_arguments)]
fn invocation_append_input_v1(
    output: &mut Vec<InvocationInputRowV1>,
    function: &SemanticFunctionDeclV1,
    local: usize,
    source_argument: u32,
    tuple_field: Option<u32>,
    first: usize,
    end: usize,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    budget.charge_work(3)?;
    let declaration = function
        .locals()
        .get(local)
        .ok_or_else(invocation_entry_error_v1)?;
    let expected = match declaration.role() {
        SemanticLocalRoleV1::Argument(argument) => (argument, None),
        SemanticLocalRoleV1::RustCallTupleField { argument, field } => (argument, Some(field)),
        _ => return Err(invocation_entry_error_v1()),
    };
    if expected != (source_argument, tuple_field) {
        return Err(invocation_entry_error_v1());
    }
    emission_push_v1(
        output,
        InvocationInputRowV1 {
            local: u32::try_from(local).map_err(|_| ArgumentResourceV1::Arithmetic)?,
            ty: declaration.ty(),
            source_argument,
            tuple_field,
            first_parameter: first,
            parameter_count: end
                .checked_sub(first)
                .ok_or(ArgumentResourceV1::Arithmetic)?,
            reference_call_transport: None,
        },
        budget,
    )
}

fn invocation_check_inputs_v1(
    function: &SemanticFunctionDeclV1,
    inputs: &[InvocationInputRowV1],
    parameters: usize,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<(), ProductionSemanticKirErrorV1> {
    let floor = budget.storage();
    let result = (|| {
        budget.reserve_storage(std::mem::size_of::<Vec<bool>>())?;
        let mut seen = emission_vec_v1(function.locals().len(), budget)?;
        budget.charge_work(function.locals().len())?;
        seen.resize(function.locals().len(), false);
        let mut next = 0;
        for row in inputs {
            budget.charge_work(5)?;
            let declaration = function
                .locals()
                .get(row.local as usize)
                .ok_or_else(invocation_entry_error_v1)?;
            let selector = match declaration.role() {
                SemanticLocalRoleV1::Argument(argument) => (argument, None),
                SemanticLocalRoleV1::RustCallTupleField { argument, field } => {
                    (argument, Some(field))
                }
                _ => return Err(invocation_entry_error_v1()),
            };
            if seen[row.local as usize]
                || declaration.ty() != row.ty
                || selector != (row.source_argument, row.tuple_field)
                || row.first_parameter != next
            {
                return Err(invocation_entry_error_v1());
            }
            seen[row.local as usize] = true;
            next = argument_sum_v1(&[next, row.parameter_count])?;
            if next > parameters {
                return Err(invocation_entry_error_v1());
            }
        }
        budget.charge_work(function.locals().len())?;
        if next != parameters
            || function
                .locals()
                .iter()
                .zip(&seen)
                .any(|(local, seen)| local.role().is_entry_argument() != *seen)
        {
            return Err(invocation_entry_error_v1());
        }
        Ok(())
    })();
    budget.release_storage(
        budget
            .storage()
            .checked_sub(floor)
            .ok_or(ArgumentResourceV1::Accounting)?,
    )?;
    result
}

fn invocation_root_inputs_v1(
    function: &SemanticFunctionDeclV1,
    parameters: &SemanticParameterBindingsV1<'_>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Vec<InvocationInputRowV1>, ProductionSemanticKirErrorV1> {
    let mut output = emission_vec_v1(parameters.declarations.len(), budget)?;
    if parameters.values.len() != parameters.types.len()
        || parameters
            .local_bindings
            .is_some_and(|bindings| bindings.len() != parameters.declarations.len())
    {
        return Err(invocation_entry_error_v1());
    }
    let mut next = 0;
    for (ordinal, &(argument, local, ty)) in parameters.declarations.iter().enumerate() {
        budget.charge_work(4)?;
        let declaration = function
            .locals()
            .get(local)
            .ok_or_else(invocation_entry_error_v1)?;
        if declaration.ty() != ty {
            return Err(invocation_entry_error_v1());
        }
        let first = next;
        if let Some(bindings) = parameters.local_bindings {
            match &bindings[ordinal] {
                PlannedParameterLocalBindingV1::Bf16Nominal { .. }
                | PlannedParameterLocalBindingV1::Wave64Nominal { .. } => {
                    return Err(invocation_entry_error_v1());
                }
                PlannedParameterLocalBindingV1::Direct {
                    local: bound,
                    value,
                    ty,
                } => {
                    if *bound != local
                        || parameters.values.get(next) != Some(value)
                        || !invocation_equal_types_v1(
                            parameters
                                .types
                                .get(next)
                                .ok_or_else(invocation_entry_error_v1)?,
                            ty,
                            budget,
                        )?
                    {
                        return Err(invocation_entry_error_v1());
                    }
                    next = argument_sum_v1(&[next, 1])?;
                }
                PlannedParameterLocalBindingV1::Flattened {
                    local: bound,
                    semantic_type,
                    values,
                } => {
                    if *bound != local || *semantic_type != declaration.ty() {
                        return Err(invocation_entry_error_v1());
                    }
                    for value in values {
                        budget.charge_work(2)?;
                        if parameters.values.get(next) != Some(&value.id)
                            || !invocation_equal_types_v1(
                                parameters
                                    .types
                                    .get(next)
                                    .ok_or_else(invocation_entry_error_v1)?,
                                &value.ty,
                                budget,
                            )?
                        {
                            return Err(invocation_entry_error_v1());
                        }
                        next = argument_sum_v1(&[next, 1])?;
                    }
                }
            }
        } else {
            // This is exactly the constructor's positional installation path.
            if parameters.values.get(next).is_none() || parameters.types.get(next).is_none() {
                return Err(invocation_entry_error_v1());
            }
            next = argument_sum_v1(&[next, 1])?;
        }
        let field = match declaration.role() {
            SemanticLocalRoleV1::RustCallTupleField { field, .. } => Some(field),
            _ => None,
        };
        invocation_append_input_v1(
            &mut output,
            function,
            local,
            argument,
            field,
            first,
            next,
            budget,
        )?;
    }
    invocation_check_inputs_v1(function, &output, parameters.values.len(), budget)?;
    Ok(output)
}
