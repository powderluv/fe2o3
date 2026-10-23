impl<'a, 'service> SemanticFunctionLoweringV1<'a, 'service> {
    #[allow(clippy::too_many_arguments)]
    fn new_interprocedural(
        types: &'a [SemanticTypeDeclV1],
        callables: &'a [SemanticCallableDeclV1],
        function: &'a SemanticFunctionDeclV1,
        semantic_ssa: &ProductionSemanticSsaFunctionPlanV1,
        correspondence_owner: SemanticFunctionIdV1,
        semantic_function: SemanticFunctionIdV1,
        defined_function_ids: impl Into<
            EmissionReadOnlyV1<'a, BTreeMap<SemanticFunctionIdV1, FunctionId>>,
        >,
        defined_function_signatures: impl Into<ExecutionSignatureSourceV29<'a>>,
        result_types: impl Into<EmissionReadOnlyV1<'a, Vec<Type>>>,
        parameters: SemanticParameterBindingsV1<'_>,
        assert_failure_block: Option<BlockId>,
        required_workgroup: Option<[u32; 3]>,
        infallible_asserts: InfallibleAssertDecisionsV1<'a>,
        launch_rank: u8,
        authenticated_ranked_control: bool,
        max_operations: usize,
        mut private_array_work: PrivateArrayRecorderWorkV1<'service>,
        private_array_sources: Option<PrivateArraySourcesV1<'_>>,
        call_returns: CallReturnBufferV1,
        mut emission_work: Option<&'service mut dyn SemanticEmissionBudgetV1>,
        emission_placement: SemanticEmissionPlacementV1,
        mut execution: Option<ExecutionAvailabilityV29<'a>>,
        lifecycle: Option<&'service mut (dyn ExecutionLifecycleConsumerV29 + 'service)>,
        wave64_source_owner: Option<&'a ProductionSemanticSsaOwnerV1>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        infallible_asserts.require_parts(
            types,
            callables,
            function,
            correspondence_owner,
            semantic_function,
        )?;
        if let Some(execution) = &execution {
            execution.check_source(function, semantic_ssa)?;
            if !std::ptr::eq(execution.cfg.types, types) {
                return Err(execution_cfg_error_v29());
            }
        }
        if let Some(consumer) = lifecycle.as_deref() {
            let cursor = execution
                .as_ref()
                .ok_or_else(execution_availability_error_v29)?;
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            cursor.check_ledger(budget)?;
            let result = consumer.check_instance(cursor, budget);
            cursor.check_ledger(budget)?;
            result?;
        }
        let representation = if execution
            .as_ref()
            .is_some_and(|cursor| cursor.references.is_some())
        {
            ExecutionCfgRepresentationV29::OriginalSource
        } else {
            ExecutionCfgRepresentationV29::LegacyAbi
        };
        if wave64_source_owner.is_some_and(|owner| !std::ptr::eq(owner.source_semantic().types(), types) || !std::ptr::eq(owner.source_semantic().callables(), callables)) { return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch); }
        let wave64_references = prepare_checked_wave64_reference_locals_v1(wave64_source_owner, function, semantic_function, emission_work.as_deref_mut())?;
        let mut locals = vec![None; function.locals().len()];
        let option_producers = semantic_option_producers_v1(function, callables)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        let option_dominance = SemanticOptionDominanceV1::analyze(function, &option_producers)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        let enum_payload_dominance = SemanticEnumPayloadDominanceV1::analyze(function, types)
            .map_err(|error| unsupported(0, None, None, error.detail()))?;
        if let Some(parameter_local_bindings) = parameters.local_bindings {
            for binding in parameter_local_bindings {
                let (local, value) = match binding {
                    PlannedParameterLocalBindingV1::Wave64Nominal { local, semantic_type, descriptor, values } => {
                        descriptor.exact_local_v1(semantic_function, *local as u32, *semantic_type)?;
                        if wave64_references.locals.get(&(*local as u32)) != Some(descriptor) { return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch); }
                        let mut reconstructed=None;
                        wave64_references.with_descriptor_v1(*descriptor,emission_work.as_deref_mut().ok_or(ArgumentResourceV1::Accounting)?,&mut |guard,_|{reconstructed=Some(guard.reconstruct_v1(types,*semantic_type,values)?);Ok(())})?;
                        (*local,reconstructed.ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?)
                    }
                    PlannedParameterLocalBindingV1::Bf16Nominal {
                        local,
                        semantic_type,
                        descriptor,
                        values,
                    } => (
                        *local,
                        descriptor.binding_from_transport(types, *semantic_type, values)?,
                    ),
                    PlannedParameterLocalBindingV1::Direct { local, value, ty } => (
                        *local,
                        SemanticValueBindingV1::Value {
                            id: *value,
                            ty: ty.clone(),
                        },
                    ),
                    PlannedParameterLocalBindingV1::Flattened {
                        local,
                        semantic_type,
                        values,
                    } => (
                        *local,
                        binding_from_value_defs_with_representation_v29(
                            types,
                            *semantic_type,
                            values,
                            representation,
                        )?,
                    ),
                };
                locals[local] = Some(value);
            }
        } else {
            for ((_, local, _), (value, ty)) in parameters
                .declarations
                .iter()
                .zip(parameters.values.iter().zip(parameters.types))
            {
                locals[*local] = Some(SemanticValueBindingV1::Value {
                    id: *value,
                    ty: ty.clone(),
                });
            }
        }
        let nominal_floor = if let Some(cursor) = execution.as_mut() {
            cursor.install_call_parameters_v29(
                &parameters,
                &mut locals,
                emission_work.as_deref_mut(),
            )?
        } else {
            0
        };
        if let Some(cursor) = execution.as_mut()
            && !semantic_ssa.plan().entry_arguments().is_empty()
            && cursor.invocation_inputs.is_none()
        {
            let budget = emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            cursor.check_ledger(budget)?;
            cursor.invocation_inputs =
                Some(invocation_root_inputs_v1(function, &parameters, budget)?);
        }
        #[cfg(test)]
        if let Some(cursor) = &execution {
            // Inert fixture inputs only; no producer is constructed in production.
            for (local, binding) in &cursor.entry_seeds {
                let local = *local as usize;
                if !function
                    .locals()
                    .get(local)
                    .is_some_and(|local| local.role().is_entry_argument())
                    || cursor.cfg.nominal_locals[local] == 0
                    || locals[local].is_some()
                {
                    return Err(execution_cfg_error_v29());
                }
                locals[local] = Some(binding.clone());
            }
        }
        let parameter_floor = emission_placement.value_floor(parameters.values)?;
        let mut direct_parameters = BTreeMap::new();
        if let Some(parameter_local_bindings) = parameters.local_bindings {
            for binding in parameter_local_bindings {
                if let PlannedParameterLocalBindingV1::Direct { local, ty, .. } = binding {
                    let local = u32::try_from(*local).map_err(|_| {
                        unsupported(0, None, None, "parameter local does not fit Kernel IR")
                    })?;
                    if direct_parameters.insert(local, ty.clone()).is_some() {
                        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                    }
                }
            }
        }
        let mut next_value = u32::try_from(function.locals().len())
            .map_err(|_| unsupported(0, None, None, "local count does not fit Kernel IR"))?;
        next_value = next_value.max(parameter_floor).max(nominal_floor);
        let defined_function_signatures = defined_function_signatures.into();
        let backing = match &defined_function_signatures {
            ExecutionSignatureSourceV29::Legacy(_) => None,
            ExecutionSignatureSourceV29::Scoped(layouts) => {
                let budget = emission_work
                    .as_deref_mut()
                    .ok_or(ArgumentResourceV1::Accounting)?;
                source_reference_owned_prepay_v29::<Option<SourceFunctionBackingViewV29<'_>>>(
                    layouts.references,
                    budget,
                )?;
                SourceFunctionBackingViewV29::new(
                    &defined_function_signatures,
                    execution.as_ref(),
                    semantic_function,
                    budget,
                )?
            }
        };
        let control_flow_ssa = with_prepared_input_transport_v1(
            execution.as_ref(),
            function,
            semantic_ssa,
            parameters.values,
            parameters.types,
            &locals,
            emission_work.as_deref_mut(),
            |prepared, budget| {
                SemanticControlFlowSsaPlanV1::analyze_with_execution_v29(
                    SemanticSsaTransportInputV1 {
                        types,
                        callables,
                        function,
                        semantic_function,
                    },
                    semantic_ssa,
                    &option_dominance,
                    &direct_parameters,
                    max_operations,
                    max_operations,
                    budget,
                    execution.as_ref(),
                    prepared,
                    lifecycle.as_deref(),
                    parameters.local_bindings,
                    backing,
                    Some(&wave64_references),
)
            },
        )?;
        let workgroup_pipeline_contracts = workgroup_pipeline_type_contracts_v1(
            types,
            callables,
            &control_flow_ssa.compiler_issued_bindings,
        )?;
        let authenticated_loop_induction_bounds = if authenticated_ranked_control {
            authenticated_loop_induction_bounds_v1(types, function)?
        } else {
            BTreeMap::new()
        };
        let promoted_enum_variant_by_value = analyze_promoted_enum_variants_v1(
            types,
            function,
            &control_flow_ssa,
            max_operations,
            max_operations,
        )?;
        let mut retained_local_slots = BTreeMap::new();
        if let Some(budget) = emission_work.as_deref_mut() {
            budget.reserve_storage(std::mem::size_of::<
                BTreeMap<ScopedAllocationIdentityV29, SemanticRetainedLocalSlotV1>,
            >())?;
        }
        let private_array_enabled = control_flow_ssa.has_retained_arrays;
        for (identity, plan) in &control_flow_ssa.retained_local_slots {
            let pointer = ValueId(next_value);
            next_value = next_value.checked_add(1).ok_or_else(|| {
                unsupported(0, None, None, "retained-local slot identity overflow")
            })?;
            let storage = match emission_work.as_deref_mut() {
                Some(budget) => {
                    reserve_execution_cfg_map_entry_v29::<
                        ScopedAllocationIdentityV29,
                        SemanticRetainedLocalSlotV1,
                    >(retained_local_slots.len(), budget)?;
                    clone_retained_storage_v29(&plan.storage, budget)?
                }
                None => plan.storage.clone(),
            };
            retained_local_slots.insert(
                *identity,
                SemanticRetainedLocalSlotV1 {
                    pointer,
                    semantic_type: plan.semantic_type,
                    storage,
                },
            );
        }
        let mut block_parameters = BTreeMap::new();
        for block in 0..function.blocks().len() as u32 {
            if block == function.entry().index() && control_flow_ssa.live_in(block).is_empty() {
                continue;
            }
            let mut parameters = BTreeMap::new();
            for local in control_flow_ssa.live_in(block) {
                let promoted = control_flow_ssa
                    .promoted
                    .get(local)
                    .expect("live-in local must be promoted");
                let nominal = matches!(
                    promoted.transport,
                    SemanticPromotedTransportV1::Execution
                        | SemanticPromotedTransportV1::SourceEnumTag
                );
                let destination_types = if promoted.transport
                    == SemanticPromotedTransportV1::Execution
                    && execution.as_ref().is_some_and(|cursor| {
                        cursor.cfg.reference_locals[*local as usize] && cursor.references.is_some()
                    }) {
                    let budget = emission_work
                        .as_deref_mut()
                        .ok_or(ArgumentResourceV1::Accounting)?;
                    budget.reserve_storage(std::mem::size_of::<Option<Vec<Type>>>())?;
                    let selected = source_descriptor_cfg_types_v29(
                        execution.as_ref().ok_or_else(execution_cfg_error_v29)?,
                        &control_flow_ssa.cfg_carriers,
                        SemanticBlockIdV1::from_index(block),
                        *local,
                        budget,
                    )?;
                    if selected.len() != promoted.kernel_types.len() {
                        return Err(execution_cfg_error_v29());
                    }
                    Some(selected)
                } else {
                    None
                };
                let physical_types = destination_types
                    .as_deref()
                    .unwrap_or(&promoted.kernel_types);
                let mut components = if nominal {
                    let budget = emission_work
                        .as_deref_mut()
                        .ok_or(ArgumentResourceV1::Accounting)?;
                    reserve_execution_cfg_map_entry_v29::<u32, Vec<ValueDef>>(
                        parameters.len(),
                        budget,
                    )?;
                    reserve_execution_cfg_map_entry_v29::<u32, BTreeMap<u32, Vec<ValueDef>>>(
                        block_parameters.len(),
                        budget,
                    )?;
                    emission_vec_v1(promoted.kernel_types.len(), budget)?
                } else {
                    Vec::with_capacity(promoted.kernel_types.len())
                };
                for ty in physical_types {
                    let ty = if nominal {
                        execution_cfg_clone_type_v29(
                            ty,
                            emission_work
                                .as_deref_mut()
                                .ok_or(ArgumentResourceV1::Accounting)?,
                        )?
                    } else {
                        ty.clone()
                    };
                    components.push(ValueDef::new(ValueId(next_value), ty));
                    next_value = next_value.checked_add(1).ok_or_else(|| {
                        unsupported(0, Some(block), None, "block-parameter identity overflow")
                    })?;
                }
                parameters.insert(*local, components);
            }
            block_parameters.insert(block, parameters);
        }
        let enum_payload_sources =
            plan_unique_enum_payload_sources_v1(types, function, &control_flow_ssa);
        let (enum_payload_storage, enum_payload_requires_compile_time_custody) =
            plan_enum_payload_storage_v1(
                types,
                function,
                &control_flow_ssa,
                &enum_payload_sources,
                &mut next_value,
                execution.as_ref(),
                emission_work.as_deref_mut(),
            )?;
        let pending_semantic_ssa_definitions = control_flow_ssa
            .definition_values
            .iter()
            .map(|(site, values)| (*site, values.iter().copied().collect()))
            .collect();
        let mut private_array_outer = match &private_array_sources {
            Some(PrivateArraySourcesV1::Pending(payload)) => *payload,
            _ => PrivateArrayPayloadV1::default(),
        };
        if private_array_enabled {
            private_array_work.activate()?;
            if let Some(sources) = private_array_sources {
                private_array_outer = sources.payload(&mut private_array_work)?;
            }
        }
        let mut emitted_unsigned_exclusive_bounds = BTreeMap::new();
        for (local,binding) in locals.iter().enumerate() {
            if let Some(binding) = binding {
                if wave64_references.rank_locals.contains(&(local as u32)) || wave64_references.locals.contains_key(&(local as u32)) { wave64_references.seed_arithmetic_bound_v1(local as u32,binding,emission_work.as_deref_mut().ok_or(ArgumentResourceV1::Accounting)?,&mut emitted_unsigned_exclusive_bounds)?; }
            }
        }
        Ok(Self {
            wave64_references,
            ordered_composition: false,
            fixed_array_analysis: None,
            private_arrays: PrivateArrayFunctionRecorderV1::new_with_representation_v29(
                private_array_work,
                private_array_enabled,
                max_operations,
                private_array_outer,
                emission_placement,
                if execution
                    .as_ref()
                    .and_then(|cursor| cursor.references)
                    .is_some()
                {
                    ExecutionCfgRepresentationV29::OriginalSource
                } else {
                    ExecutionCfgRepresentationV29::LegacyAbi
                },
            ),
            types,
            callables,
            function,
            correspondence_owner,
            semantic_function,
            defined_function_ids: defined_function_ids.into(),
            defined_function_signatures,
            result_types: result_types.into(),
            locals,
            retained_local_slots,
            retained_local_allocas_emitted: false,
            invocation_preheader: None,
            retained_local_initialized: BTreeSet::new(),
            option_dominance,
            enum_payload_dominance,
            enum_payload_storage,
            enum_payload_sources,
            enum_payload_requires_compile_time_custody,
            enum_payload_compile_time_custody: BTreeMap::new(),
            enum_payload_allocas_emitted: false,
            control_flow_ssa,
            workgroup_pipeline_contracts,
            promoted_enum_variant_by_value,
            block_parameters,
            semantic_ssa_bindings: SemanticSsaBindingsV1::default(),
            semantic_ssa_archive_credit: None,
            semantic_rvalue_bindings: BTreeMap::new(),
            pending_semantic_ssa_definitions,
            next_value,
            emission_placement,
            emission_work,
            scoped_memory: execution
                .as_ref()
                .map(|cursor| ScopedMemoryRecorderV29::new(cursor, emission_placement)),
            execution,
            execution_calls: None,
            lifecycle,
            assert_failure_block,
            required_workgroup,
            infallible_asserts,
            launch_rank,
            max_operations,
            emitted_operations: 0,
            emitted_workgroup_memory_extents: BTreeMap::new(),
            emitted_unsigned_constants: BTreeMap::new(),
            emitted_u32_constants: BTreeMap::new(),
            emitted_u32_bitand_masks: BTreeMap::new(),
            authenticated_loop_induction_bounds,
            emitted_unsigned_exclusive_bounds,
            generated_terminator_values: Vec::new(),
            call_returns,
        })
    }
}
