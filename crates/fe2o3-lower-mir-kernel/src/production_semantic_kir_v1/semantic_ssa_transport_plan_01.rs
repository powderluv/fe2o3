include!("semantic_ssa_prepared_input_transport_v1.rs");

fn original_compiler_carrier_v29(
    local: SemanticLocalIdV1,
    compiler_issued_bindings: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    origins: &mut SemanticCapabilityOriginResolverV1<'_>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Option<(SemanticTypeIdV1, SemanticPromotedBindingV1)>, ProductionSemanticKirErrorV1> {
    budget.charge_work(2)?;
    let ty = origins
        .function
        .locals()
        .get(local.index() as usize)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
        .ty();
    charge_execution_cfg_lookup_v29(compiler_issued_bindings.len(), budget)?;
    let binding = match compiler_issued_bindings.get(&ty).copied() {
        Some(binding) => Some(binding),
        None => origins.resolve_with_budget_v29(local, &mut Some(budget))?,
    };
    // The ordinary transport resolver also follows transparent borrow aliases
    // to prepare ABI/SSA components. That is not compiler-carrier provenance:
    // source references retain their Loan/Address path, and value-alias origins
    // are already memoized by the original capability resolver above.
    Ok(binding
        .filter(|binding| !matches!(binding, SemanticPromotedBindingV1::Ordinary))
        .map(|binding| (ty, binding)))
}

#[cfg(test)]
fn promoted_transport_descriptor_v1(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    local: u32,
    compiler_issued_bindings: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    shared_promoted: &BTreeSet<u32>,
    capability_origins: &mut SemanticCapabilityOriginResolverV1<'_>,
    direct_parameters: &BTreeMap<u32, Type>,
) -> Result<(SemanticTypeIdV1, SemanticPromotedTransportV1), ProductionSemanticKirErrorV1> {
    promoted_transport_descriptor_with_inputs_v1(
        types,
        function,
        local,
        compiler_issued_bindings,
        shared_promoted,
        capability_origins,
        direct_parameters,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn promoted_transport_descriptor_with_inputs_v1<'work>(
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    local: u32,
    compiler_issued_bindings: &BTreeMap<SemanticTypeIdV1, SemanticPromotedBindingV1>,
    shared_promoted: &BTreeSet<u32>,
    capability_origins: &mut SemanticCapabilityOriginResolverV1<'_>,
    direct_parameters: &BTreeMap<u32, Type>,
    prepared: Option<&PreparedInputTransportV1<'_, '_>>,
    mut budget: Option<&mut (dyn SemanticEmissionBudgetV1 + 'work)>,
) -> Result<(SemanticTypeIdV1, SemanticPromotedTransportV1), ProductionSemanticKirErrorV1> {
    let mut current = local;
    if let Some(budget) = budget.as_deref_mut() {
        budget.reserve_storage(std::mem::size_of::<BTreeSet<u32>>())?;
    }
    let mut visited = BTreeSet::new();
    loop {
        if let Some(budget) = budget.as_deref_mut() {
            budget.charge_work(6)?;
            reserve_execution_cfg_map_entry_v29::<u32, ()>(visited.len(), budget)?;
            charge_execution_cfg_lookup_v29(compiler_issued_bindings.len(), budget)?;
            charge_execution_cfg_lookup_v29(direct_parameters.len(), budget)?;
        }
        capability_origins.charge_work(1)?;
        if !visited.insert(current) || visited.len() > MAX_SSA_VALUE_COMPONENTS_V1 {
            return Err(unsupported(
                0,
                None,
                None,
                "transparent-borrow SSA transport is cyclic or too deep",
            ));
        }
        let declaration = function
            .locals()
            .get(current as usize)
            .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
        if let Some(binding) = compiler_issued_bindings.get(&declaration.ty()).copied() {
            return Ok((
                declaration.ty(),
                SemanticPromotedTransportV1::Semantic(binding),
            ));
        }
        if let Some(binding) = capability_origins
            .resolve_with_budget_v29(SemanticLocalIdV1::from_index(current), &mut budget)?
        {
            return Ok((
                declaration.ty(),
                SemanticPromotedTransportV1::Semantic(binding),
            ));
        }
        if declaration.role().is_entry_argument() && direct_parameters.contains_key(&current) {
            return Ok((
                declaration.ty(),
                SemanticPromotedTransportV1::DirectParameter {
                    parameter_local: current,
                },
            ));
        }
        if declaration.role().is_entry_argument()
            && let Some(prepared) = prepared
            && let Some(ty) = prepared.ordinary(
                current,
                budget
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            )?
        {
            if !std::ptr::eq(prepared.cursor.function, function) || ty != declaration.ty() {
                return Err(invocation_entry_error_v1());
            }
            return Ok((
                ty,
                SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary),
            ));
        }

        if let Some(budget) = budget.as_deref_mut() {
            charge_execution_cfg_lookup_v29(capability_origins.invalidated_locals.len(), budget)?;
            charge_execution_cfg_lookup_v29(capability_origins.definitions.len(), budget)?;
            charge_execution_cfg_lookup_v29(shared_promoted.len(), budget)?;
        }
        let Some(definition) =
            capability_origins.transparent_definition(SemanticLocalIdV1::from_index(current))
        else {
            return Ok((
                declaration.ty(),
                SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary),
            ));
        };
        let source = match definition {
            SemanticTransparentDefinitionV1::Borrow { source, .. }
            | SemanticTransparentDefinitionV1::ValueAlias { source, .. } => source,
        };
        if !shared_promoted.contains(&source) {
            return Ok((
                declaration.ty(),
                SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary),
            ));
        }
        match definition {
            SemanticTransparentDefinitionV1::Borrow { referent_type, .. } => {
                let Some(reference) =
                    types
                        .get(declaration.ty().index() as usize)
                        .and_then(|declaration| match declaration.shape() {
                            SemanticTypeShapeV1::Pointer(pointer) => Some(pointer),
                            _ => None,
                        })
                else {
                    return Ok((
                        declaration.ty(),
                        SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary),
                    ));
                };
                if reference.pointee() != referent_type {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "transparent-borrow SSA transport pointee type changed",
                    ));
                }
            }
            SemanticTransparentDefinitionV1::ValueAlias { source_type, .. } => {
                if declaration.ty() != source_type {
                    return Err(unsupported(
                        0,
                        None,
                        None,
                        "transparent value-alias SSA transport type changed",
                    ));
                }
            }
        }
        current = source;
    }
}

struct SemanticSsaTransportInputV1<'a> {
    types: &'a [SemanticTypeDeclV1],
    callables: &'a [SemanticCallableDeclV1],
    function: &'a SemanticFunctionDeclV1,
    semantic_function: SemanticFunctionIdV1,
}

impl SemanticControlFlowSsaPlanV1 {
    #[cfg(test)]
    fn analyze(
        input: SemanticSsaTransportInputV1<'_>,
        semantic_ssa: &ProductionSemanticSsaFunctionPlanV1,
        option_dominance: &SemanticOptionDominanceV1,
        direct_parameters: &BTreeMap<u32, Type>,
        max_analysis_work: usize,
        max_analysis_storage: usize,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        Self::analyze_with_execution_v29(
            input,
            semantic_ssa,
            option_dominance,
            direct_parameters,
            max_analysis_work,
            max_analysis_storage,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
)
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_with_execution_v29<'work>(
        input: SemanticSsaTransportInputV1<'_>,
        semantic_ssa: &ProductionSemanticSsaFunctionPlanV1,
        option_dominance: &SemanticOptionDominanceV1,
        direct_parameters: &BTreeMap<u32, Type>,
        max_analysis_work: usize,
        max_analysis_storage: usize,
        mut emission_work: Option<&mut (dyn SemanticEmissionBudgetV1 + 'work)>,
        execution: Option<&ExecutionAvailabilityV29<'_>>,
        prepared: Option<&PreparedInputTransportV1<'_, '_>>,
        lifecycle: Option<&dyn ExecutionLifecycleConsumerV29>,
        nominal_parameters: Option<&[PlannedParameterLocalBindingV1]>,
        backing: Option<SourceFunctionBackingViewV29<'_>>,
        wave64_references: Option<&CheckedWave64ReferenceLocalsV1<'_>>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let SemanticSsaTransportInputV1 {
            types,
            callables,
            function,
            semantic_function,
        } = input;
        let representation = if execution.is_some_and(|cursor| cursor.references.is_some()) {
            ExecutionCfgRepresentationV29::OriginalSource
        } else {
            ExecutionCfgRepresentationV29::LegacyAbi
        };
        if semantic_ssa.function() != semantic_function
            || semantic_ssa.function_identity() != function.identity()
        {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        if let Some(prepared) = prepared {
            prepared.check(
                execution.ok_or_else(execution_availability_error_v29)?,
                function,
                semantic_ssa,
                emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            )?;
        }
        let mut compiler_issued_bindings = compiler_issued_ssa_bindings_v1(
            types,
            callables,
            function,
            semantic_function,
            lifecycle,
            emission_work.as_deref_mut(),
        )?;
        if let Some(parameters) = nominal_parameters {
            for parameter in parameters {
                if let PlannedParameterLocalBindingV1::Bf16Nominal {
                    local,
                    semantic_type,
                    descriptor,
                    ..
                } = parameter
                {
                    let declaration = function
                        .locals()
                        .get(*local)
                        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
                    if !declaration.role().is_entry_argument() || declaration.ty() != *semantic_type
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                    if compiler_issued_bindings
                        .insert(*semantic_type, *descriptor)
                        .is_some_and(|old| old != *descriptor)
                    {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                }
            }
        }
        let shared = semantic_ssa.plan();
        let retained_cross_edge = semantic_ssa
            .retained_cross_edge_variables()
            .iter()
            .map(|local| local.get())
            .collect::<BTreeSet<_>>();
        let shared_promoted = shared
            .promoted_variables()
            .iter()
            .map(|variable| variable.get())
            .collect::<BTreeSet<_>>();
        let mut private_slot_candidates = private_slot_candidate_locals_v1(
            function,
            &shared_promoted,
            &retained_cross_edge,
            representation,
        );
        if let Some(references) = wave64_references {
            references.require_source_v1(function, semantic_function)?;
            references.remove_checked_retained_snapshots_v1(&mut private_slot_candidates,emission_work.as_deref_mut().ok_or(ArgumentResourceV1::Accounting)?)?;
        }
        let mut retained_local_slots = BTreeMap::new();
        if let Some(backing) = backing {
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            let cursor = execution.ok_or_else(scoped_object_allocation_error_v29)?;
            backing.check(
                cursor.references.map(|references| references.plan),
                Some(cursor.instance),
                budget,
            )?;
            backing.visit_object_allocations(
                &private_slot_candidates,
                budget,
                |identity, slot, budget| {
                    reserve_execution_cfg_map_entry_v29::<
                        ScopedAllocationIdentityV29,
                        SemanticRetainedLocalSlotPlanV1,
                    >(retained_local_slots.len(), budget)?;
                    if retained_local_slots.insert(identity, slot).is_some() {
                        return Err(scoped_object_allocation_error_v29());
                    }
                    Ok(())
                },
            )?;
        }
        let mut has_retained_arrays = false;
        let mut unsupported_retained_locals = Vec::new();
        for local in private_slot_candidates {
            if let Some(budget) = emission_work.as_deref_mut() {
                charge_execution_cfg_lookup_v29(retained_local_slots.len(), budget)?;
            }
            let objects = ScopedAllocationIdentityV29::OriginalObject {
                local,
                generation: 0,
            }..=ScopedAllocationIdentityV29::OriginalObject {
                local,
                generation: u32::MAX,
            };
            if retained_local_slots.range(objects).next().is_some() {
                continue;
            }
            if let Some(cursor) = execution
                && let Some(prepared) = prepared
                && cursor.retained_installed_slot_omission_v1(
                    local,
                    prepared,
                    emission_work
                        .as_deref_mut()
                        .ok_or(ArgumentResourceV1::Accounting)?,
                )?
            {
                continue;
            }
            if let Some(cursor) = execution
                && let Some(references) = cursor.references
                && source_reference_existing_value_local_v29(
                    references.plan,
                    cursor.instance,
                    local,
                    emission_work
                        .as_deref_mut()
                        .ok_or(ArgumentResourceV1::Accounting)?,
                )?
            {
                continue;
            }
            let declaration = function
                .locals()
                .get(local as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            let representation = if execution.and_then(|cursor| cursor.references).is_some() {
                ExecutionCfgRepresentationV29::OriginalSource
            } else {
                ExecutionCfgRepresentationV29::LegacyAbi
            };
            let slot = if matches!(
                types[declaration.ty().index() as usize].shape(),
                SemanticTypeShapeV1::Array { .. }
            ) {
                if declaration.role().is_entry_argument() {
                    return Err(unsupported(
                        semantic_function.index(),
                        None,
                        None,
                        "retained by-value array arguments require source-effect-bound entry scatter",
                    ));
                }
                retained_array_slot_plan_with_representation_v29(
                    types,
                    declaration.ty(),
                    max_analysis_work,
                    representation,
                )
                .map_err(|error| match error {
                    ProductionSemanticKirErrorV1::Unsupported { detail, .. } => {
                        unsupported(semantic_function.index(), None, None, detail)
                    }
                    error => error,
                })?
            } else if let Some((kernel_type, alignment)) =
                retained_local_slot_type_with_representation_v29(
                    types,
                    declaration.ty(),
                    representation,
                )
            {
                SemanticRetainedLocalSlotPlanV1 {
                    semantic_type: declaration.ty(),
                    storage: SemanticRetainedStorageV29::ScalarArray {
                        kernel_type,
                        alignment,
                        array: None,
                    },
                }
            } else {
                unsupported_retained_locals.push((local, declaration.ty().index()));
                continue;
            };
            if let Some(backing) = backing {
                let budget = emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?;
                if backing.array_schema(local, budget)?.is_some()
                    && !matches!(
                        slot.storage,
                        SemanticRetainedStorageV29::ScalarArray { array: Some(_), .. }
                    )
                {
                    return Err(scoped_object_allocation_error_v29());
                }
            }
            has_retained_arrays |= slot.storage.scalar_array()?.2.is_some();
            retained_local_slots.insert(ScopedAllocationIdentityV29::LegacyLocal(local), slot);
        }
        if !unsupported_retained_locals.is_empty() {
            const MAX_RETAINED_LOCAL_DIAGNOSTICS_V1: usize = 32;
            let retained_count = unsupported_retained_locals.len();
            unsupported_retained_locals.truncate(MAX_RETAINED_LOCAL_DIAGNOSTICS_V1);
            let unsupported_retained_locals = unsupported_retained_locals
                .into_iter()
                .map(|(local, ty)| {
                    (
                        local,
                        ty,
                        first_retained_local_cause_v1(
                            function,
                            local,
                            callables,
                            &compiler_issued_bindings,
                        ),
                    )
                })
                .collect();
            return Err(ProductionSemanticKirErrorV1::RetainedLocalStorage {
                function: semantic_function.index(),
                retained_locals: unsupported_retained_locals,
                retained_count,
            });
        }
        let implicit_entry_locals = semantic_ssa
            .implicit_entry_variables()
            .iter()
            .map(|variable| variable.get())
            .collect::<BTreeSet<_>>();
        if implicit_entry_locals.len() != semantic_ssa.implicit_entry_variables().len() {
            return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
        }
        for local in &implicit_entry_locals {
            let declaration = function
                .locals()
                .get(*local as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            if !shared_promoted.contains(local)
                || declaration.role() != SemanticLocalRoleV1::Temporary
                || !authenticated_ambient_workgroup_lds_scope_zst_v1(
                    types,
                    callables,
                    declaration.ty(),
                )
                || compiler_issued_bindings.get(&declaration.ty())
                    != Some(&SemanticPromotedBindingV1::WorkgroupLdsScope)
            {
                return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
            }
        }
        let mut transported = BTreeSet::new();
        for block in shared.reverse_postorder() {
            let variables = shared
                .transport_variables(*block)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            for variable in variables {
                if !shared_promoted.contains(&variable.get()) {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                transported.insert(variable.get());
            }
        }
        let source_borrows = match (execution, emission_work.as_deref_mut()) {
            (Some(cursor), Some(budget)) if cursor.references.is_some() => {
                budget.reserve_storage(std::mem::size_of::<
                    Option<(
                        &ExecutionAvailabilityV29<'_>,
                        &mut dyn SemanticEmissionBudgetV1,
                    )>,
                >())?;
                Some((cursor, budget))
            }
            _ => None,
        };
        let mut capability_origins = SemanticCapabilityOriginResolverV1::new_with_source_v29(
            types,
            callables,
            function,
            option_dominance,
            &shared_promoted,
            max_analysis_work,
            max_analysis_storage,
            source_borrows,
        )?;
        let capability_storage = if let Some(cursor) = execution {
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            Some(CapabilityOriginStorageV29::new(
                &mut capability_origins,
                cursor,
                budget,
            )?)
        } else {
            None
        };
        if let Some(cursor) = execution {
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            cursor.check_ledger(budget)?;
            budget.reserve_storage(argument_sum_v1(&[
                std::mem::size_of::<ExecutionCfgCarriersV29>(),
                execution_cfg_carrier_index_headers_v29()?,
            ])?)?;
        }
        let mut cfg_carriers = ExecutionCfgCarriersV29::default();
        if let Some(cursor) = execution {
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            cfg_carriers.source_owner = Some(std::ptr::from_ref(function).addr());
            cfg_carriers.instance = Some(cursor.instance);
            cfg_carriers.ledger = Some(budget.work_ledger_identity_v1());
            if let Some(references) = cursor.references {
                references.check(budget)?;
                cfg_carriers.source_plan = Some(std::ptr::from_ref(references.plan).addr());
            }
            // Resolve each tracked original local once, including values that
            // are defined and used in one block. CFG restoration independently
            // requires the exact Plain source node in cfg_carriers.at().
            for local in 0..function.locals().len() {
                budget.charge_work(2)?;
                charge_execution_cfg_lookup_v29(shared_promoted.len(), budget)?;
                let local_index =
                    u32::try_from(local).map_err(|_| ArgumentResourceV1::Arithmetic)?;
                if cursor.cfg.nominal_locals[local] != 0 || !shared_promoted.contains(&local_index)
                {
                    continue;
                }
                let local = local_index;
                let Some((transport_type, binding)) = original_compiler_carrier_v29(
                    SemanticLocalIdV1::from_index(local),
                    &compiler_issued_bindings,
                    &mut capability_origins,
                    budget,
                )?
                else {
                    continue;
                };
                // A shared borrow of an original scalar witness also checks
                // its complete SSA holder, although the holder has no pointer.
                // Other non-reference capabilities retain their existing path.
                if !cursor.cfg.reference_locals[local as usize]
                    && !matches!(binding, SemanticPromotedBindingV1::IndexWitness { .. })
                {
                    continue;
                }
                let representation = if cursor.references.is_some() {
                    ExecutionCfgRepresentationV29::OriginalSource
                } else {
                    ExecutionCfgRepresentationV29::LegacyAbi
                };
                let kernel_types = binding.transport_types_with_allocation_v29(
                    types,
                    transport_type,
                    &mut CompilerCarrierAllocationV29::paid_with_representation(
                        budget,
                        representation,
                    )?,
                )?;
                cfg_carriers.append(
                    local,
                    ExecutionCfgCarrierV29 {
                        source_type: function.locals()[local as usize].ty(),
                        transport_type,
                        binding,
                        kernel_types: kernel_types.into_boxed_slice(),
                    },
                    budget,
                )?;
            }
        }
        let mut promoted = BTreeMap::new();
        for local in transported {
            let declaration = function
                .locals()
                .get(local as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            let reference =
                execution.is_some_and(|cursor| cursor.cfg.reference_locals[local as usize]);
            let source_enum =
                execution.is_some_and(|cursor| cursor.cfg.source_enum_locals[local as usize]);
            let nominal = reference
                || execution.is_some_and(|cursor| cursor.cfg.nominal_locals[local as usize] != 0);
            let checked_reference = wave64_references.and_then(|references| references.locals.get(&local)).copied();
            let (transport_semantic_type, binding) = if let Some(descriptor) = checked_reference {
                descriptor.exact_local_v1(semantic_function, local, declaration.ty())?;
                (declaration.ty(), SemanticPromotedTransportV1::CheckedWave64Reference(descriptor))
            } else if source_enum {
                (declaration.ty(), SemanticPromotedTransportV1::SourceEnumTag)
            } else if nominal {
                (declaration.ty(), SemanticPromotedTransportV1::Execution)
            } else {
                promoted_transport_descriptor_with_inputs_v1(
                    types,
                    function,
                    local,
                    &compiler_issued_bindings,
                    &shared_promoted,
                    &mut capability_origins,
                    direct_parameters,
                    prepared,
                    emission_work.as_deref_mut(),
                )?
            };
            let representation = if execution.is_some_and(|cursor| cursor.references.is_some()) {
                ExecutionCfgRepresentationV29::OriginalSource
            } else {
                ExecutionCfgRepresentationV29::LegacyAbi
            };
            let kernel_types = if nominal {
                let budget = emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?;
                execution.unwrap().check_ledger(budget)?;
                reserve_execution_cfg_map_entry_v29::<u32, SemanticPromotedLocalV1>(
                    promoted.len(),
                    budget,
                )?;
                if source_enum {
                    source_reference_cfg_enum_tag_types_v55(execution.unwrap(), local, budget)?
                } else if reference {
                    if let Some(carrier) = cfg_carriers.lookup(local, budget)? {
                        carrier.types(budget)?
                    } else {
                        source_reference_cfg_local_types_v29(execution.unwrap(), local, budget)?
                    }
                } else {
                    execution_cfg_types_with_representation_v29(
                        types,
                        transport_semantic_type,
                        representation,
                        budget,
                    )?
                }
            } else if let Some(descriptor)=checked_reference {
                let references=wave64_references.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
                let mut selected=None;
                references.with_descriptor_v1(descriptor,emission_work.as_deref_mut().ok_or(ArgumentResourceV1::Accounting)?,&mut |guard,_|{selected=Some(guard.transport_types_v1());Ok(())})?;
                selected.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
            } else {
                binding.transport_types_with_representation_v29(
                    types,
                    transport_semantic_type,
                    direct_parameters,
                    representation,
                )?
            };
            let ordinary_empty = kernel_types.is_empty()
                && matches!(
                    binding,
                    SemanticPromotedTransportV1::Semantic(SemanticPromotedBindingV1::Ordinary)
                )
                && types[transport_semantic_type.index() as usize]
                    .layout()
                    .size_bytes()
                    == Some(0)
                && binding_from_value_defs(types, transport_semantic_type, &[])?
                    .values()
                    .map_err(|detail| unsupported(semantic_function.index(), None, None, detail))?
                    .is_empty();
            if kernel_types.is_empty()
                && !nominal
                && !ordinary_empty
                && !matches!(
                    binding,
                    SemanticPromotedTransportV1::Semantic(
                        SemanticPromotedBindingV1::MathContext
                            | SemanticPromotedBindingV1::CollectiveContext
                            | SemanticPromotedBindingV1::WorkgroupLdsScope
                            | SemanticPromotedBindingV1::MatrixContext
                            | SemanticPromotedBindingV1::GridLeader { .. }
                    )
                )
            {
                return Err(unsupported(
                    semantic_function.index(),
                    None,
                    None,
                    "SSA transport produced no Kernel IR block-parameter types",
                ));
            }
            promoted.insert(
                local,
                SemanticPromotedLocalV1 {
                    semantic_type: declaration.ty(),
                    transport_semantic_type,
                    transport: binding,
                    kernel_types: kernel_types.into_boxed_slice(),
                },
            );
        }
        if let Some(storage) = capability_storage {
            storage.finish(
                capability_origins,
                emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            )?;
        } else {
            drop(capability_origins);
        }
        let mut live_in = BTreeMap::new();
        for block in 0..function.blocks().len() as u32 {
            let block_id = SsaBlockIdV1::new(block);
            let locals = if !shared.is_reachable(block_id) {
                Vec::new()
            } else {
                shared
                    .transport_variables(block_id)
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
                    .iter()
                    .map(|variable| variable.get())
                    .filter(|local| promoted.contains_key(local))
                    .collect()
            };
            live_in.insert(block, locals);
        }
        let mut edge_arguments = BTreeMap::new();
        let mut edge_definitions = BTreeMap::new();
        for block in shared.reverse_postorder() {
            let source = function
                .blocks()
                .get(block.get() as usize)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
            for ordinal in 0..source.terminator().kind().edge_count() {
                let edge = fe2o3_mir_model::SsaEdgeIdV1::new(*block, ordinal as u32);
                let arguments = shared
                    .edge_arguments(edge)
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
                    .iter()
                    .copied()
                    .filter(|argument| promoted.contains_key(&argument.variable().get()))
                    .collect();
                let definitions = shared
                    .edge_definitions(edge)
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
                    .to_vec();
                edge_definitions.insert((block.get(), ordinal as u32), definitions);
                edge_arguments.insert((block.get(), ordinal as u32), arguments);
            }
        }

        let entry_definitions = shared
            .entry_definitions()
            .iter()
            .copied()
            .map(|definition| (definition.variable().get(), definition.value()))
            .collect();
        let mut block_entry_values = BTreeMap::new();
        let mut definition_values = BTreeMap::<(u32, u32), Vec<SsaValueV1>>::new();
        for block in shared.reverse_postorder() {
            let mut seen = BTreeSet::new();
            for (_, event) in shared
                .resolved_events(*block)
                .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?
            {
                let (variable, entry_value, definition) = match *event {
                    SsaResolvedEventV1::Use { variable, value } => (variable, Some(value), None),
                    SsaResolvedEventV1::Define { variable, value } => (variable, None, Some(value)),
                    SsaResolvedEventV1::Kill { variable, previous } => (variable, previous, None),
                };
                let local = variable.get();
                if !shared_promoted.contains(&local) {
                    continue;
                }
                if seen.insert(local)
                    && let Some(value) = entry_value
                {
                    block_entry_values.insert((block.get(), local), value);
                }
                if let Some(value) = definition {
                    definition_values
                        .entry((block.get(), local))
                        .or_default()
                        .push(value);
                }
            }
        }
        let reachable = shared
            .reverse_postorder()
            .iter()
            .map(|block| block.get())
            .collect::<BTreeSet<_>>();
        let retained_initialized_at_entry =
            retained_local_initialization_entries_with_representation_v29(
                function,
                &retained_local_slots,
                &reachable,
                max_analysis_work,
                max_analysis_storage,
                representation,
            )?;
        Ok(Self {
            representation,
            cfg_carriers,
            has_retained_arrays,
            compiler_issued_bindings,
            implicit_entry_locals,
            ssa_value_locals: shared_promoted,
            promoted,
            live_in,
            block_entry_values,
            entry_definitions,
            definition_values,
            edge_definitions,
            edge_arguments,
            retained_local_slots,
            retained_initialized_at_entry,
        })
    }

    fn live_in(&self, block: u32) -> &[u32] {
        self.live_in.get(&block).map_or(&[], Vec::as_slice)
    }

    fn entry_value(
        &self,
        function: &SemanticFunctionDeclV1,
        block: u32,
        local: u32,
    ) -> Option<SsaValueV1> {
        if self.live_in(block).contains(&local) {
            return Some(SsaValueV1::BlockArgument {
                block: SsaBlockIdV1::new(block),
                variable: fe2o3_mir_model::SsaVariableIdV1::new(local),
            });
        }
        if block == function.entry().index() {
            return self.entry_definitions.get(&local).copied();
        }
        self.block_entry_values.get(&(block, local)).copied()
    }
}
