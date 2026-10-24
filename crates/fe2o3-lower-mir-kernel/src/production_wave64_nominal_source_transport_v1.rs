// All local descriptors are rederived from the SAME borrowed immutable source
// owner. They are arithmetic transport, not reference layout/EXEC authority.
struct CheckedWave64ReferenceLocalsV1<'a> {
    owner: Option<&'a ProductionSemanticSsaOwnerV1>,
    function: SemanticFunctionIdV1,
    locals: BTreeMap<u32, Wave64NominalReferenceDescriptorV1>,
    retained_snapshots: BTreeSet<u32>,
    rank_locals: BTreeSet<u32>,
    borrows: BTreeMap<(u32, u32), Wave64NominalReferenceDescriptorV1>,
    context_borrows: BTreeMap<(u32, u32), Wave64NominalReferenceDescriptorV1>,
}
impl CheckedWave64ReferenceLocalsV1<'_> {
    fn require_source_v1(
        &self,
        function: &SemanticFunctionDeclV1,
        id: SemanticFunctionIdV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if self.function != id
            || self.owner.is_some_and(|owner| {
                owner
                    .source_semantic()
                    .functions()
                    .get(id.index() as usize)
                    .is_none_or(|original| !std::ptr::eq(original, function))
            })
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        Ok(())
    }
}
fn wave64_query_result_v1(
    result: Result<(), wave64_reference_call_v1::ReferenceErrorV1>,
    callback_error: Option<ProductionSemanticKirErrorV1>,
) -> Result<bool, ProductionSemanticKirErrorV1> {
    if let Some(error) = callback_error {
        return Err(error);
    }
    match result {
        Ok(()) => Ok(true),
        Err(wave64_reference_call_v1::ReferenceErrorV1::Resource(error)) => Err(error.into()),
        Err(wave64_reference_call_v1::ReferenceErrorV1::Unavailable(_)) => Ok(false),
        Err(wave64_reference_call_v1::ReferenceErrorV1::CallbackPanicked) => Err(unsupported(
            0,
            None,
            None,
            "checked Wave64 source callback unwound",
        )),
    }
}
fn prepare_checked_wave64_reference_locals_v1<'a, 'work>(
    owner: Option<&'a ProductionSemanticSsaOwnerV1>,
    function: &SemanticFunctionDeclV1,
    id: SemanticFunctionIdV1,
    mut budget: Option<&mut (dyn SemanticEmissionBudgetV1 + 'work)>,
) -> Result<CheckedWave64ReferenceLocalsV1<'a>, ProductionSemanticKirErrorV1> {
    let mut output = CheckedWave64ReferenceLocalsV1 {
        owner,
        function: id,
        locals: BTreeMap::new(),
        retained_snapshots: BTreeSet::new(),
        rank_locals: BTreeSet::new(),
        borrows: BTreeMap::new(),
        context_borrows: BTreeMap::new(),
    };
    output.require_source_v1(function, id)?;
    let Some(owner) = owner else {
        return Ok(output);
    };
    if !ordinary_source_has_raw_wave64_shuffle_v1(owner.source_semantic()) {
        output.owner = None;
        return Ok(output);
    }
    let budget = budget
        .as_deref_mut()
        .ok_or(ArgumentResourceV1::Accounting)?;
    // Exact source/capture identity is independent of local type coordinates.
    if owner.plan_for_function(id).is_none() || owner.occurrences_v1().is_none() {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    budget.checked_wave64_helper_v1(owner, id, &mut |helper, budget| {
        if matches!(
            helper.representative().argument(0),
            Some(wave64_reference_call_v1::ArgumentV1::Rank { .. })
        ) {
            let mut local = None;
            for (index, row) in function.locals().iter().enumerate() {
                if row.role() == SemanticLocalRoleV1::Argument(0)
                    && local.replace(index as u32).is_some()
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
            }
            let local = local.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            reserve_execution_cfg_map_entry_v29::<u32, ()>(output.rank_locals.len(), budget)?;
            output.rank_locals.insert(local);
        }
        for argument in 0..2 {
            if let Some(descriptor) =
                Wave64NominalReferenceDescriptorV1::from_checked_formal_v1(helper, argument)?
            {
                require_closed_wave64_reference_holder_v1(
                    owner,
                    id,
                    descriptor.local,
                    None,
                    budget,
                )?;
                reserve_execution_cfg_map_entry_v29::<u32, Wave64NominalReferenceDescriptorV1>(
                    output.locals.len(),
                    budget,
                )?;
                if output.locals.insert(descriptor.local, descriptor).is_some() {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
            }
        }
        Ok(())
    })?;
    for (bi, block) in function.blocks().iter().enumerate() {
        for (si, statement) in block.statements().iter().enumerate() {
            budget.charge_work(2)?;
            let SemanticStatementKindV1::Assign(assignment) = statement.kind() else {
                continue;
            };
            if !matches!(
                assignment.value().kind(),
                SemanticRvalueKindV1::Borrow {
                    kind: SemanticBorrowKindV1::Shared,
                    ..
                }
            ) {
                continue;
            }
            budget.checked_wave64_borrow_v1(
                owner,
                id,
                SemanticBlockIdV1::from_index(bi as u32),
                si as u32,
                &mut |source, budget| {
                    let descriptor =
                        Wave64NominalReferenceDescriptorV1::from_checked_borrow_v1(source)?;
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        id,
                        descriptor.local,
                        Some((source.block(), source.statement())),
                        budget,
                    )?;
                    let SemanticRvalueKindV1::Borrow { place, .. } = source.source_rvalue().kind()
                    else {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    };
                    reserve_execution_cfg_map_entry_v29::<u32, Wave64NominalReferenceDescriptorV1>(
                        output.locals.len(),
                        budget,
                    )?;
                    if output
                        .locals
                        .insert(descriptor.local, descriptor)
                        .is_some_and(|old| old != descriptor)
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    reserve_execution_cfg_map_entry_v29::<u32, ()>(
                        output.retained_snapshots.len(),
                        budget,
                    )?;
                    output.retained_snapshots.insert(place.local().index());
                    reserve_execution_cfg_map_entry_v29::<
                        (u32, u32),
                        Wave64NominalReferenceDescriptorV1,
                    >(output.borrows.len(), budget)?;
                    if output
                        .borrows
                        .insert((bi as u32, si as u32), descriptor)
                        .is_some()
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    Ok(())
                },
            )?;
            budget.checked_wave64_context_borrow_v1(
                owner,
                id,
                SemanticBlockIdV1::from_index(bi as u32),
                si as u32,
                &mut |source, budget| {
                    let descriptor =
                        Wave64NominalReferenceDescriptorV1::from_checked_context_borrow_v1(source)?;
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        id,
                        descriptor.local,
                        Some((source.block(), source.statement())),
                        budget,
                    )?;
                    let (issuer_function, _, issuer_local) = source.issuer();
                    if issuer_function != id {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    reserve_execution_cfg_map_entry_v29::<u32, Wave64NominalReferenceDescriptorV1>(
                        output.locals.len(),
                        budget,
                    )?;
                    if output
                        .locals
                        .insert(descriptor.local, descriptor)
                        .is_some_and(|old| old != descriptor)
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    reserve_execution_cfg_map_entry_v29::<u32, ()>(
                        output.retained_snapshots.len(),
                        budget,
                    )?;
                    output.retained_snapshots.insert(issuer_local.index());
                    reserve_execution_cfg_map_entry_v29::<
                        (u32, u32),
                        Wave64NominalReferenceDescriptorV1,
                    >(output.context_borrows.len(), budget)?;
                    if output
                        .context_borrows
                        .insert((bi as u32, si as u32), descriptor)
                        .is_some()
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    Ok(())
                },
            )?;
        }
    }
    Ok(output)
}
fn checked_wave64_helper_parameter_plan_v1(
    owner: &ProductionSemanticSsaOwnerV1,
    root: SemanticFunctionIdV1,
    helper: SemanticFunctionIdV1,
    id: FunctionId,
    max_rows: usize,
    closure: &mut ReachableClosureBudgetV1,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<Option<LoweredFunctionPlanV1>, ProductionSemanticKirErrorV1> {
    if !ordinary_source_has_raw_wave64_shuffle_v1(owner.source_semantic()) {
        return Ok(None);
    }
    let mut output = None;
    let mut callback_error = None;
    let result = wave64_reference_call_v1::with_checked_reference_helper_v1(
        owner,
        helper,
        budget,
        |view, budget| {
            let result = (|| {
                let source = view.representative();
                let function = source.helper_declaration();
                check_argument_function_abi_v1(
                    function,
                    helper,
                    SemanticKirFunctionRoleV1::InternalHelper,
                )?;
                let mut parameters = Vec::new();
                for (local, row) in function.locals().iter().enumerate() {
                    match row.role() {
                        SemanticLocalRoleV1::Argument(argument) => {
                            parameters.push((argument, local, row.ty()))
                        }
                        SemanticLocalRoleV1::RustCallTupleField { .. } => {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        _ => {}
                    }
                }
                parameters.sort_unstable_by_key(|row| row.0);
                if parameters.len() != 3
                    || parameters
                        .iter()
                        .enumerate()
                        .any(|(i, row)| row.0 as usize != i)
                {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let mut types = Vec::new();
                let mut values = Vec::new();
                let mut selectors = Vec::new();
                let mut bindings = Vec::new();
                for &(argument, local, ty) in &parameters {
                    budget.charge_work(8)?;
                    let descriptor = Wave64NominalReferenceDescriptorV1::from_checked_formal_v1(
                        view,
                        argument as usize,
                    )?;
                    let component_types = match descriptor {
                        Some(d) => d.transport_types_v1(),
                        None => vec![lower_parameter_scalar_v1(
                            owner.source_semantic().types(),
                            ty,
                        )?],
                    };
                    closure.charge_parameter_expansion(
                        3,
                        types.len(),
                        component_types.len(),
                        max_rows,
                    )?;
                    let mut components = Vec::new();
                    for (component, component_type) in component_types.into_iter().enumerate() {
                        if component != 0 {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                        let value = ValueId(
                            u32::try_from(local).map_err(|_| ArgumentResourceV1::Arithmetic)?,
                        );
                        types.push(component_type.clone());
                        values.push(value);
                        components.push(ValueDef::new(value, component_type));
                        selectors.push(HelperCallArgumentV1 {
                            source_argument: argument,
                            tuple_field: None,
                            component: Some(0),
                        });
                    }
                    bindings.push(match descriptor {
                        Some(descriptor) => PlannedParameterLocalBindingV1::Wave64Nominal {
                            local,
                            semantic_type: ty,
                            descriptor,
                            values: components,
                        },
                        None => {
                            let [value] = components.as_slice() else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            PlannedParameterLocalBindingV1::Direct {
                                local,
                                value: value.id,
                                ty: value.ty.clone(),
                            }
                        }
                    });
                }
                if types != [Type::Scalar(ScalarType::U32), Type::Scalar(ScalarType::F32)] {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                let results =
                    helper_result_components_v1(owner.source_semantic().types(), function, helper)?
                        .components
                        .into_iter()
                        .map(|row| row.2)
                        .collect::<Vec<_>>();
                if results != [Type::Scalar(ScalarType::F32)] {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                output = Some(LoweredFunctionPlanV1 {
                    correspondence_owner: root,
                    semantic_function: helper,
                    kernel_ir_function: id,
                    role: SemanticKirFunctionRoleV1::InternalHelper,
                    parameter_declarations: parameters,
                    parameter_types: types,
                    parameter_values: values,
                    call_arguments: selectors,
                    parameter_local_bindings: bindings,
                    parameter_component_bindings: Vec::new(),
                    ignored_parameter_bindings: Vec::new(),
                    result_types: results,
                });
                Ok(())
            })();
            if let Err(error) = result {
                callback_error = Some(error);
                return Err(wave64_reference_call_v1::ReferenceErrorV1::Unavailable(
                    "nominal helper planner refused",
                ));
            }
            Ok(())
        },
    );
    // Preserve the original source-relation diagnostic while leaving the
    // existing generic fallback and every admission decision unchanged.
    if let Err(wave64_reference_call_v1::ReferenceErrorV1::Unavailable(reason)) = &result {
        eprintln!(
            "Wave64 nominal helper {} unavailable: {}",
            helper.index(),
            reason
        );
    }
    wave64_query_result_v1(result, callback_error)?;
    Ok(output)
}
impl SemanticFunctionLoweringV1<'_, '_> {
    fn try_lower_checked_wave64_snapshot_borrow_v1(
        &mut self,
        block: SemanticBlockIdV1,
        statement: Option<u32>,
        result_type: SemanticTypeIdV1,
        value: &SemanticRvalueKindV1,
    ) -> Result<Option<SemanticValueBindingV1>, ProductionSemanticKirErrorV1> {
        let Some(statement) = statement else {
            return Ok(None);
        };
        if let Some(expected) = self
            .wave64_references
            .context_borrows
            .get(&(block.index(), statement))
            .copied()
        {
            return self.lower_checked_wave64_context_borrow_v1(
                block,
                statement,
                result_type,
                value,
                expected,
            );
        }
        let Some(expected) = self
            .wave64_references
            .borrows
            .get(&(block.index(), statement))
            .copied()
        else {
            return Ok(None);
        };
        let owner = self
            .wave64_references
            .owner
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        let mut binding = None;
        self.with_emission_budget_v1(|this, budget| {
            let checked = budget.checked_wave64_borrow_v1(
                owner,
                this.semantic_function,
                block,
                statement,
                &mut |source, budget| {
                    budget.charge_work(12)?;
                    if !std::ptr::eq(source.source_rvalue().kind(), value)
                        || source.reference_type() != result_type
                        || Wave64NominalReferenceDescriptorV1::from_checked_borrow_v1(source)?
                            != expected
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    require_closed_wave64_reference_holder_v1(
                        owner,
                        this.semantic_function,
                        expected.local,
                        Some((block, statement)),
                        budget,
                    )?;
                    let (issuer, producer) = source.lane_producer();
                    if issuer != this.semantic_function {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    let Some(SemanticValueBindingV1::WaveLane { value, wave }) =
                        this.semantic_ssa_bindings.get(&producer)
                    else {
                        return Err(unsupported(
                            issuer.index(),
                            Some(block.index()),
                            Some(statement),
                            "original snapshot lane SSA producer has no emitted binding",
                        ));
                    };
                    if wave.width != 64 {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    binding = Some(expected.reconstruct_v1(
                        this.types,
                        result_type,
                        &[ValueDef::new(*value, Type::Scalar(ScalarType::U32))],
                    )?);
                    Ok(())
                },
            )?;
            if !checked || binding.is_none() {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            Ok(())
        })?;
        Ok(binding)
    }
    fn check_checked_wave64_call_bindings_v1(
        &mut self,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        bindings: &[SemanticValueBindingV1],
        arguments: &[ValueId],
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let owner = self
            .wave64_references
            .owner
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        self.with_emission_budget_v1(|this, budget| {
            let checked = budget.checked_wave64_call_v1(
                owner,
                this.semantic_function,
                block,
                &mut |source, budget| {
                    budget.charge_work(16)?;
                    if source.helper() != callee
                        || !std::ptr::eq(source.source_call(), call)
                        || bindings.len() != 3
                        || !matches!(bindings[1], SemanticValueBindingV1::CollectiveContext)
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    for argument in 0..3 {
                        let original = wave64_original_call_argument_ssa_v1(
                            owner,
                            this.semantic_function,
                            block,
                            argument,
                            budget,
                        )?;
                        let archived = this
                            .semantic_ssa_bindings
                            .get(&original)
                            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
                        if !same_wave64_original_binding_v1(
                            archived,
                            &bindings[argument as usize],
                            budget,
                            0,
                        )? {
                            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                        }
                    }
                    let lane = match source.argument(0) {
                        Some(wave64_reference_call_v1::ArgumentV1::Snapshot {
                            snapshot_type,
                            lane_field,
                            ..
                        }) => {
                            let Some(SemanticTypeShapeV1::Aggregate(original_fields)) = this
                                .types
                                .get(snapshot_type.index() as usize)
                                .map(SemanticTypeDeclV1::shape)
                            else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            let SemanticValueBindingV1::Aggregate(actual_fields) = &bindings[0]
                            else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            if original_fields.fields().len() != actual_fields.len() {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            }
                            for (index, field) in actual_fields.iter().enumerate() {
                                if index != lane_field as usize {
                                    visit_wave64_empty_source_zst_v1(
                                        field,
                                        &mut Wave64ReferenceCheckVisitorV1 { budget },
                                        0,
                                    )?;
                                }
                            }
                            let SemanticValueBindingV1::Aggregate(fields) = &bindings[0] else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            let Some(SemanticValueBindingV1::Value { id, ty }) =
                                fields.get(lane_field as usize)
                            else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            if *ty != Type::Scalar(ScalarType::U32) {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            }
                            *id
                        }
                        Some(wave64_reference_call_v1::ArgumentV1::Rank { .. }) => {
                            let SemanticValueBindingV1::Value { id, ty } = &bindings[0] else {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            };
                            if *ty != Type::Scalar(ScalarType::U32) {
                                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                            }
                            *id
                        }
                        _ => return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch),
                    };
                    let SemanticValueBindingV1::Value { id, ty } = &bindings[2] else {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    };
                    if *ty != Type::Scalar(ScalarType::F32) || arguments != [lane, *id] {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    Ok(())
                },
            )?;
            if !checked {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
            Ok(())
        })
    }
}

include!("production_wave64_nominal_source_transport_v1/descriptor_transport.rs");
include!("production_wave64_nominal_source_transport_v1/holder_lifetime.rs");
