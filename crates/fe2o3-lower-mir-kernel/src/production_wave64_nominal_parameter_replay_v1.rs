// Checked nominal reference entry transport. The original source arguments and
// FnABI remain complete. No reference is relabelled as a structural tuple field.
// All emitted Call/result/destination/continuation checks remain in the
// unchanged validate_call_correspondence_v1 path immediately after this check.
fn validate_parameter_correspondence_with_wave64_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    instance: &SemanticKirFunctionCorrespondenceV1,
    target: &Function,
    trace: ArgumentTraceV1<'_>,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if instance.role != SemanticKirFunctionRoleV1::InternalHelper
        || !ordinary_source_has_raw_wave64_shuffle_v1(owner.source_semantic())
    {
        return validate_parameter_correspondence_v1(
            owner.source_semantic(),
            instance,
            target,
            trace,
            budget,
        );
    }
    use wave64_reference_call_v1::{ReferenceErrorV1, with_checked_reference_helper_v1};
    let mut callback_error = None;
    let result = with_checked_reference_helper_v1(
        owner,
        instance.semantic_function,
        budget,
        |helper, budget| {
            let checked = (|| {
                let source = helper.representative();
                budget.charge_work(8)?;
                if !std::ptr::eq(source.owner(), owner)
                    || source.helper() != instance.semantic_function
                    || helper.incoming_count() == 0
                    || target.id != instance.kernel_ir_function
                    || target.role != fe2o3_kernel_ir::FunctionRole::InternalHelper
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let declaration = source.helper_declaration();
                check_argument_function_abi_v1(
                    declaration,
                    instance.semantic_function,
                    instance.role,
                )?;
                let mut locals = [None; 3];
                for (local, row) in declaration.locals().iter().enumerate() {
                    budget.charge_work(2)?;
                    match row.role() {
                        SemanticLocalRoleV1::Argument(argument) if argument < 3 => {
                            if locals[argument as usize]
                                .replace((local, row.ty()))
                                .is_some()
                            {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            }
                        }
                        SemanticLocalRoleV1::Argument(_)
                        | SemanticLocalRoleV1::RustCallTupleField { .. } => {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        _ => {}
                    }
                }
                let [Some(receiver), Some(context), Some(value)] = locals else {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                };
                for (ordinal, row) in [receiver, context, value].iter().enumerate() {
                    if declaration.abi().source_input_types().get(ordinal) != Some(&row.1) {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                }
                let rank = match source.argument(0) {
                    Some(wave64_reference_call_v1::ArgumentV1::Snapshot {
                        reference_type, ..
                    }) if reference_type == receiver.1 => false,
                    Some(wave64_reference_call_v1::ArgumentV1::Rank { scalar_type })
                        if scalar_type == receiver.1 =>
                    {
                        true
                    }
                    _ => return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch),
                };
                if !matches!(source.argument(1),
                    Some(wave64_reference_call_v1::ArgumentV1::Context { reference_type, .. }) if reference_type == context.1)
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let ids = [
                    ValueId(u32::try_from(receiver.0).map_err(|_| ArgumentResourceV1::Arithmetic)?),
                    ValueId(u32::try_from(value.0).map_err(|_| ArgumentResourceV1::Arithmetic)?),
                ];
                validate_wave64_nominal_physical_signature_v1(target, ids)?;
                if !trace.components.is_empty() || !trace.ignored.is_empty() {
                    // In particular neither the context reference nor snapshot
                    // pointer may appear as a fictitious ignored FnABI value.
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let expected_count = 1 + usize::from(rank);
                if trace.direct.len() != expected_count {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let mut next = 0;
                if rank {
                    check_wave64_direct_parameter_trace_v1(
                        trace.direct[next],
                        instance,
                        receiver.0,
                        ids[0],
                    )?;
                    next += 1;
                }
                check_wave64_direct_parameter_trace_v1(
                    trace.direct[next],
                    instance,
                    value.0,
                    ids[1],
                )?;
                Ok(())
            })();
            if let Err(error) = checked {
                callback_error = Some(error);
                return Err(ReferenceErrorV1::Unavailable(
                    "Wave64 nominal entry replay differs",
                ));
            }
            Ok(())
        },
    );
    if let Some(error) = callback_error {
        return Err(error);
    }
    match result {
        Ok(()) => Ok(()),
        Err(ReferenceErrorV1::Resource(error)) => Err(error.into()),
        Err(ReferenceErrorV1::CallbackPanicked) => Err(unsupported(
            0,
            None,
            None,
            "Wave64 nominal entry replay callback unwound",
        )),
        Err(ReferenceErrorV1::Unavailable(_)) => validate_parameter_correspondence_v1(
            owner.source_semantic(),
            instance,
            target,
            trace,
            budget,
        ),
    }
}

fn validate_wave64_nominal_physical_signature_v1(
    target: &Function,
    ids: [ValueId; 2],
) -> Result<(), ProductionSemanticKirErrorV1> {
    let body = target
        .body
        .as_ref()
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    if target.role != fe2o3_kernel_ir::FunctionRole::InternalHelper
        || target.signature.parameters
            != [Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::F32)]
        || target.signature.results != [Type::Scalar(ScalarType::F32)]
        || body.parameters != ids
        || ids[0] == ids[1]
    {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    Ok(())
}
fn check_wave64_direct_parameter_trace_v1(
    row: SemanticKirParameterBindingV1,
    instance: &SemanticKirFunctionCorrespondenceV1,
    local: usize,
    value: ValueId,
) -> Result<(), ProductionSemanticKirErrorV1> {
    if row.correspondence_owner != instance.correspondence_owner
        || row.semantic_function != instance.semantic_function
        || row.semantic_local.index() as usize != local
        || row.kernel_ir_value != value
    {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod wave64_nominal_parameter_replay_tests_v1 {
    use super::*;
    #[test]
    fn nominal_signature_checks_exact_runtime_carriers_and_formal_identities() {
        let function = Function::internal_helper(
            FunctionId::new("wave64_nominal_test"),
            Signature::new(
                vec![Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::F32)],
                vec![Type::Scalar(ScalarType::F32)],
            ),
            vec![ValueId(1), ValueId(3)],
            Vec::new(),
        );
        assert!(
            validate_wave64_nominal_physical_signature_v1(&function, [ValueId(1), ValueId(3)])
                .is_ok()
        );
        for fault in 0..6 {
            let mut bad = function.clone();
            match fault {
                0 => bad.signature.parameters.swap(0, 1),
                1 => bad.signature.parameters.push(Type::Unit),
                2 => bad.signature.results.clear(),
                3 => bad.body.as_mut().unwrap().parameters.swap(0, 1),
                4 => bad.role = fe2o3_kernel_ir::FunctionRole::KernelEntry,
                5 => bad.body = None,
                _ => unreachable!(),
            }
            assert!(
                validate_wave64_nominal_physical_signature_v1(&bad, [ValueId(1), ValueId(3)])
                    .is_err()
            );
        }
    }
}
