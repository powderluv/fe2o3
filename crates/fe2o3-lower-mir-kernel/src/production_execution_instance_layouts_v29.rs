// A physical schema for each original invocation, not a source value or access
// proof. Source replay still reconstructs these layouts from the original C1 plan.
struct ExecutionInstanceLayoutV29 {
    function: SemanticFunctionIdV1,
    incoming: ProductionCallOccurrenceV1,
    original_call: usize,
    declarations: Vec<(u32, usize, SemanticTypeIdV1)>,
    signature: LoweredFunctionSignatureV1,
}

struct ExecutionInstanceLayoutsV29<'a, 'source> {
    references: &'a SourceReferencePlanV29<'a, 'source>,
    source: ExecutionCallSourceV29,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    slot: usize,
    floor: usize,
    rows: Vec<Option<ExecutionInstanceLayoutV29>>,
    calls: BTreeMap<(usize, u32), ProductionCallInstanceIdV1>,
    // Lookup into the one C1 allocation census, not a second backing owner.
    backing: BTreeMap<(usize, u32, u32), SourceBackingAllocationV29>,
}

enum ExecutionSignatureSourceV29<'a> {
    Legacy(EmissionReadOnlyV1<'a, BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>>),
    Scoped(&'a ExecutionInstanceLayoutsV29<'a, 'a>),
}

#[derive(Clone, Copy)]
struct SourceFunctionBackingViewV29<'a> {
    layouts: &'a ExecutionInstanceLayoutsV29<'a, 'a>,
    instance: ProductionCallInstanceIdV1,
}

impl<'a> SourceFunctionBackingViewV29<'a> {
    fn new(
        signatures: &ExecutionSignatureSourceV29<'a>,
        cursor: Option<&ExecutionAvailabilityV29<'a>>,
        function: SemanticFunctionIdV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<Option<Self>, ProductionSemanticKirErrorV1> {
        let ExecutionSignatureSourceV29::Scoped(layouts) = signatures else {
            return Ok(None);
        };
        layouts.check(budget)?;
        let cursor = cursor.ok_or_else(execution_instance_layout_error_v29)?;
        let references = cursor
            .references
            .ok_or_else(execution_instance_layout_error_v29)?;
        budget.source_reference_charge_v29(layouts.references, 6)?;
        let original = layouts
            .references
            .instances
            .instance(cursor.instance)
            .ok_or_else(execution_instance_layout_error_v29)?;
        if !std::ptr::eq(layouts.references, references.plan)
            || function != original.function()
            || cursor.function_id != function
            || !std::ptr::eq(cursor.function, original.declaration())
            || !std::ptr::eq(cursor.ssa, original.ssa())
        {
            return Err(execution_instance_layout_error_v29());
        }
        Ok(Some(Self {
            layouts,
            instance: cursor.instance,
        }))
    }

    fn check(
        self,
        plan: Option<&SourceReferencePlanV29<'_, '_>>,
        instance: Option<ProductionCallInstanceIdV1>,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.layouts.check(budget)?;
        budget.source_reference_charge_v29(self.layouts.references, 2)?;
        if instance != Some(self.instance)
            || !plan.is_some_and(|plan| std::ptr::eq(plan, self.layouts.references))
        {
            return Err(execution_instance_layout_error_v29());
        }
        Ok(())
    }

    fn cell(
        self,
        local: SemanticLocalIdV1,
        generation: u32,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<Option<(usize, SourceReferenceScalarCellV29)>, ProductionSemanticKirErrorV1> {
        self.layouts
            .backing_cell(self.instance, local, generation, budget)
    }
}

impl<'a> From<EmissionReadOnlyV1<'a, BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>>>
    for ExecutionSignatureSourceV29<'a>
{
    fn from(
        value: EmissionReadOnlyV1<'a, BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>>,
    ) -> Self {
        Self::Legacy(value)
    }
}

impl<'a> From<&'a BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>>
    for ExecutionSignatureSourceV29<'a>
{
    fn from(value: &'a BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>) -> Self {
        Self::Legacy(EmissionReadOnlyV1::Borrowed(value))
    }
}

impl From<BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>>
    for ExecutionSignatureSourceV29<'_>
{
    fn from(value: BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>) -> Self {
        Self::Legacy(EmissionReadOnlyV1::Owned(value))
    }
}

impl<'a> From<&'a ExecutionInstanceLayoutsV29<'a, 'a>> for ExecutionSignatureSourceV29<'a> {
    fn from(value: &'a ExecutionInstanceLayoutsV29<'a, 'a>) -> Self {
        Self::Scoped(value)
    }
}

fn execution_instance_layout_error_v29() -> ProductionSemanticKirErrorV1 {
    unsupported(
        0,
        None,
        None,
        "physical layout differs from its original call instance",
    )
}

fn execution_instance_layout_headers_v29<R>() -> Result<usize, ArgumentResourceV1> {
    use std::mem::size_of;
    type Panic = Box<dyn std::any::Any + Send>;
    argument_sum_v1(&[
        source_reference_emission_headers_v29::<ExecutionInstanceLayoutsV29<'_, '_>>()?,
        size_of::<Option<source_storage_v29::SourceStorageRootGrowthV29<'_, '_, '_>>>(),
        size_of::<Result<Result<R, ProductionSemanticKirErrorV1>, Panic>>(),
        size_of::<Result<R, ProductionSemanticKirErrorV1>>(),
        size_of::<Option<ProductionSemanticKirErrorV1>>(),
        size_of::<Result<Result<R, ProductionSemanticKirErrorV1>, Panic>>(),
        size_of::<
            std::panic::AssertUnwindSafe<Result<Result<R, ProductionSemanticKirErrorV1>, Panic>>,
        >(),
        source_reference_cleanup_headers_v29()?,
    ])
}

fn with_execution_instance_layouts_v29<'a, 'source, 'work, R>(
    references: &'a SourceReferencePlanV29<'a, 'source>,
    budget: &mut ArgumentBudgetV1<'work>,
    consume: impl FnOnce(
        &ExecutionInstanceLayoutsV29<'a, 'source>,
        &mut ArgumentBudgetV1<'work>,
    ) -> Result<R, ProductionSemanticKirErrorV1>,
) -> Result<R, ProductionSemanticKirErrorV1> {
    let floor = budget.storage();
    let ledger = budget.work_ledger_identity_v1();
    let slot = budget as *mut ArgumentBudgetV1<'_> as usize;
    let mut required = floor;
    let mut owned = None;
    let mut growth = None;
    let mut outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        references.check_owner(references.instances, budget)?;
        budget.charge_work(1 + SOURCE_REFERENCE_PAYLOAD_ATTEMPTS_V29)?;
        budget.reserve_storage(execution_instance_layout_headers_v29::<R>()?)?;
        required = budget.storage();
        if let Some(root) = references.storage_root.as_ref() {
            budget.charge_work(source_storage_v29::SOURCE_STORAGE_ROOT_GROWTH_WORK_V29)?;
            growth = Some(
                root.capture_retained_growth()
                    .ok_or(ArgumentResourceV1::Accounting)?,
            );
        }
        let owner = ExecutionInstanceLayoutsV29::new(references, budget)?;
        required = owner.floor;
        owned = Some(
            required
                .checked_sub(floor)
                .ok_or(ArgumentResourceV1::Accounting)?,
        );
        let result = consume(&owner, budget);
        drop(owner);
        result
    }));
    // A construction failure has no escaping output. Callback-owned output and
    // C2 growth are never treated as disposable layout storage.
    let bytes = owned.or_else(|| budget.storage().checked_sub(floor));
    let permits = bytes.is_some_and(|bytes| {
        growth
            .as_ref()
            .is_none_or(|growth| growth.permits_refund(floor, required, budget.storage(), bytes))
            && budget.permits_prepared_input_refund_v1(
                Some(references),
                slot,
                ledger,
                required,
                bytes,
            )
    });
    if let Ok(Err(error)) = &outcome {
        source_reference_record_failure_v29(references, error);
    }
    let mut first = references.failure.first_error();
    if !permits {
        if first.is_none() && matches!(outcome, Ok(Ok(_))) {
            references
                .failure
                .record_resource(ArgumentResourceV1::Accounting);
            first = Some(ArgumentResourceV1::Accounting.into());
        }
        if let Some(root) = references.storage_root.as_ref() {
            root.deny_active_root_refund();
        }
    }
    // Select the sticky failure before disposing of a rejected callback value.
    // Its destructor cannot replace that error or run after its paid envelope.
    if outcome.is_ok() && first.is_some() {
        let rejected = std::mem::replace(
            &mut outcome,
            Ok(Err(first
                .take()
                .unwrap_or_else(|| ArgumentResourceV1::Accounting.into()))),
        );
        if source_reference_discard_v29(rejected) {
            if let Some(root) = references.storage_root.as_ref() {
                root.deny_active_root_refund();
            }
        }
    }
    drop(first);
    let released = scoped_emission_refund_v29(
        Some(references),
        ledger,
        slot,
        floor,
        required,
        growth,
        bytes,
        budget,
    );
    match outcome {
        Ok(Ok(value)) if released => Ok(value),
        Ok(Ok(value)) => {
            // The concrete scope cannot change between the pure postflight and
            // settlement except through hostile cleanup. Preserve the refusal.
            source_reference_discard_v29(value);
            Err(ArgumentResourceV1::Accounting.into())
        }
        Ok(Err(error)) => {
            source_reference_record_failure_v29(references, &error);
            Err(error)
        }
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

impl<'a, 'source> ExecutionInstanceLayoutsV29<'a, 'source> {
    fn new(
        references: &'a SourceReferencePlanV29<'a, 'source>,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let result = (|| {
            let selected = source_reference_check_selected_backing_v29(references, budget)?;
            let instances = references.instances;
            let count = instances.instances().len();
            budget.charge_work(count)?;
            let mut rows = emission_vec_v1(count, budget)?;
            let mut calls = BTreeMap::new();
            let mut backing = BTreeMap::new();
            if selected {
                let scratch = budget.storage();
                let eligible = source_array_eligibility_v29(references, budget)?;
                let eligibility_bytes = budget
                    .storage()
                    .checked_sub(scratch)
                    .ok_or(ArgumentResourceV1::Accounting)?;
                for (index, cell) in references.cells.rows.iter().enumerate() {
                    budget.charge_work(2)?;
                    let key = (cell.instance.index(), cell.local.index(), cell.generation);
                    charge_execution_cfg_lookup_v29(backing.len(), budget)?;
                    reserve_execution_cfg_map_entry_v29::<
                        (usize, u32, u32),
                        SourceBackingAllocationV29,
                    >(backing.len(), budget)?;
                    if backing
                        .insert(
                            key,
                            SourceBackingAllocationV29 {
                                cell: index,
                                array: *eligible
                                    .get(index)
                                    .ok_or_else(execution_instance_layout_error_v29)?,
                            },
                        )
                        .is_some()
                    {
                        return Err(execution_instance_layout_error_v29());
                    }
                }
                drop(eligible);
                budget.release_storage(eligibility_bytes)?;
                intersect_source_array_generations_v29(references, &mut backing, budget)?;
                let existing_receivers = source_existing_receiver_rows_v29(references, budget)?;
                for activation in &references.storage_activations {
                    budget.charge_work(1)?;
                    charge_execution_cfg_lookup_v29(backing.len(), budget)?;
                    if !backing.contains_key(&(
                        activation.instance.index(),
                        activation.local.index(),
                        activation.generation,
                    )) && !source_existing_receiver_entry_v29(
                        references,
                        &existing_receivers,
                        *activation,
                        budget,
                    )? {
                        return Err(execution_instance_layout_error_v29());
                    }
                }
            }
            for index in 0..count {
                budget.charge_work(3)?;
                let instance = instances
                    .id_at(index)
                    .ok_or_else(execution_instance_layout_error_v29)?;
                let active = instances
                    .instance_reachable(instance)
                    .ok_or_else(execution_instance_layout_error_v29)?;
                if instance == instances.root() || !active {
                    if instance == instances.root() && !active {
                        return Err(execution_instance_layout_error_v29());
                    }
                    rows.push(None);
                    continue;
                }
                let original = instances
                    .instance(instance)
                    .ok_or_else(execution_instance_layout_error_v29)?;
                let incoming = instances
                    .incoming(instance)
                    .ok_or_else(execution_instance_layout_error_v29)?;
                let occurrence = incoming.occurrence();
                if incoming.child() != Some(instance)
                    || instances.block_reachable(occurrence.caller, occurrence.block) != Some(true)
                {
                    return Err(execution_instance_layout_error_v29());
                }
                let key = (occurrence.caller.index(), occurrence.block.index());
                charge_execution_cfg_lookup_v29(calls.len(), budget)?;
                if calls.contains_key(&key) {
                    return Err(execution_instance_layout_error_v29());
                }
                reserve_execution_cfg_map_entry_v29::<(usize, u32), ProductionCallInstanceIdV1>(
                    calls.len(),
                    budget,
                )?;
                calls.insert(key, instance);
                let layout =
                    execution_function_layout_v29(instances, instance, Some(references), budget)?;
                let inputs = original.declaration().abi().source_input_types();
                budget.charge_work(inputs.len())?;
                let mut semantic_types = emission_vec_v1(inputs.len(), budget)?;
                semantic_types.extend_from_slice(inputs);
                rows.push(Some(ExecutionInstanceLayoutV29 {
                    function: original.function(),
                    incoming: occurrence,
                    original_call: std::ptr::from_ref(incoming.source()) as usize,
                    declarations: layout.parameter_declarations,
                    signature: LoweredFunctionSignatureV1 {
                        bf16_nominal: false,
                wave64_nominal: false,
                        parameter_semantic_types: semantic_types,
                        call_arguments: layout.call_arguments,
                        parameter_types: layout.parameter_types,
                        result_types: layout.result_types,
                        result_semantic_type: original.declaration().abi().source_output_type(),
                    },
                }));
            }
            let source = ExecutionCallSourceV29::from_instances(instances, budget)?;
            Ok(Self {
                references,
                source,
                ledger: budget.work_ledger_identity_v1(),
                slot: budget as *mut ArgumentBudgetV1<'_> as usize,
                floor: budget.storage(),
                rows,
                calls,
                backing,
            })
        })();
        result.inspect_err(|error| source_reference_record_failure_v29(references, error))
    }

    fn root_plan(
        &self,
        function: FunctionId,
        max_operations: usize,
        closure: &mut ReachableClosureBudgetV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
        self.check(budget)?;
        let instances = self.references.instances;
        let original = instances
            .instance(instances.root())
            .ok_or_else(execution_instance_layout_error_v29)?;
        if self
            .rows
            .get(instances.root().index())
            .is_none_or(Option::is_some)
        {
            return Err(execution_instance_layout_error_v29());
        }
        kernel_entry_plan_with_descriptors_v29(
            instances.owner().source_semantic(),
            original.function(),
            original.function(),
            function,
            max_operations,
            closure,
            self.references,
            budget,
        )
    }

    fn backing_cell(
        &self,
        instance: ProductionCallInstanceIdV1,
        local: SemanticLocalIdV1,
        generation: u32,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<Option<(usize, SourceReferenceScalarCellV29)>, ProductionSemanticKirErrorV1> {
        self.check(budget)?;
        budget.source_reference_charge_v29(self.references, 4)?;
        if self.references.instances.instance_reachable(instance) != Some(true) {
            return Err(execution_instance_layout_error_v29());
        }
        let original = self
            .references
            .instances
            .instance(instance)
            .and_then(|instance| instance.declaration().locals().get(local.index() as usize))
            .ok_or_else(execution_instance_layout_error_v29)?;
        charge_execution_cfg_lookup_v29(self.backing.len(), budget)?;
        let Some(&selected) = self
            .backing
            .get(&(instance.index(), local.index(), generation))
        else {
            return Ok(None);
        };
        let index = selected.cell;
        let row = *self
            .references
            .cells
            .rows
            .get(index)
            .ok_or_else(execution_instance_layout_error_v29)?;
        if row.instance != instance
            || row.local != local
            || row.generation != generation
            || original.ty() != row.ty
        {
            return Err(execution_instance_layout_error_v29());
        }
        Ok(Some((index, row)))
    }

    fn check(
        &self,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        budget.source_reference_owner_v29(self.references)?;
        budget.source_reference_charge_v29(self.references, 4)?;
        if self.ledger != budget.work_ledger_identity_v1()
            || budget.prepared_input_slot_v1() != Some(self.slot)
            || budget.storage() < self.floor
            || self.rows.len() != self.references.instances.instances().len()
        {
            self.references
                .failure
                .record_resource(ArgumentResourceV1::Accounting);
            if let Some(root) = self.references.storage_root.as_ref() {
                root.deny_active_root_refund();
            }
            return Err(ArgumentResourceV1::Accounting.into());
        }
        Ok(())
    }

    fn row(
        &self,
        instance: ProductionCallInstanceIdV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<&ExecutionInstanceLayoutV29, ProductionSemanticKirErrorV1> {
        self.check(budget)?;
        budget.source_reference_charge_v29(self.references, 5)?;
        let instances = self.references.instances;
        let original = instances
            .instance(instance)
            .ok_or_else(execution_instance_layout_error_v29)?;
        let incoming = instances
            .incoming(instance)
            .ok_or_else(execution_instance_layout_error_v29)?;
        let row = self
            .rows
            .get(instance.index())
            .and_then(Option::as_ref)
            .ok_or_else(execution_instance_layout_error_v29)?;
        if instances.instance_reachable(instance) != Some(true)
            || row.function != original.function()
            || incoming.child() != Some(instance)
            || incoming.occurrence() != row.incoming
            || std::ptr::from_ref(incoming.source()) as usize != row.original_call
        {
            return Err(execution_instance_layout_error_v29());
        }
        Ok(row)
    }

    fn check_cursor(
        &self,
        cursor: &ExecutionAvailabilityV29<'_>,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        self.check(budget)?;
        let caller = self
            .references
            .instances
            .instance(cursor.instance)
            .ok_or_else(execution_instance_layout_error_v29)?;
        budget.source_reference_charge_v29(self.references, 5)?;
        if cursor.source != self.source
            || cursor
                .references
                .is_none_or(|emission| !std::ptr::eq(emission.plan, self.references))
            || !std::ptr::eq(cursor.function, caller.declaration())
            || !std::ptr::eq(cursor.ssa, caller.ssa())
            || cursor.function_id != caller.function()
            || self
                .references
                .instances
                .instance_reachable(cursor.instance)
                != Some(true)
        {
            return Err(execution_instance_layout_error_v29());
        }
        cursor.check_ledger(budget)
    }

    fn call_row(
        &self,
        cursor: &ExecutionAvailabilityV29<'_>,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<&ExecutionInstanceLayoutV29, ProductionSemanticKirErrorV1> {
        let result = (|| {
            self.check_cursor(cursor, budget)?;
            if cursor.source_call_control_v29(block, call, budget)?
                == ProductionCallControlV1::Unreachable
            {
                return Err(execution_instance_layout_error_v29());
            }
            charge_execution_cfg_lookup_v29(self.calls.len(), budget)?;
            let child = *self
                .calls
                .get(&(cursor.instance.index(), block.index()))
                .ok_or_else(execution_instance_layout_error_v29)?;
            let row = self.row(child, budget)?;
            if row.function != callee || row.original_call != std::ptr::from_ref(call) as usize {
                return Err(execution_instance_layout_error_v29());
            }
            Ok(row)
        })();
        result.inspect_err(|error| source_reference_record_failure_v29(self.references, error))
    }

    fn signature_for_call_v29(
        &self,
        cursor: &ExecutionAvailabilityV29<'_>,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<LoweredFunctionSignatureV1, ProductionSemanticKirErrorV1> {
        let row = self.call_row(cursor, block, call, callee, budget)?;
        clone_execution_function_signature_v29(&row.signature, budget)
            .inspect_err(|error| source_reference_record_failure_v29(self.references, error))
    }

    fn instance_plan_v29(
        &self,
        instances: &ExecutionInstancesV29<'_>,
        instance: ProductionCallInstanceIdV1,
        function: FunctionId,
        placement: SemanticEmissionPlacementV1,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<LoweredFunctionPlanV1, ProductionSemanticKirErrorV1> {
        let result = (|| {
            self.references.check_owner(instances, budget)?;
            let row = self.row(instance, budget)?;
            source_reference_owned_prepay_v29::<LoweredFunctionPlanV1>(self.references, budget)?;
            budget.charge_work(row.declarations.len())?;
            let mut declarations = emission_vec_v1(row.declarations.len(), budget)?;
            declarations.extend_from_slice(&row.declarations);
            let signature = clone_execution_function_signature_v29(&row.signature, budget)?;
            let semantic_bytes = argument_product_v1(
                signature.parameter_semantic_types.capacity(),
                std::mem::size_of::<SemanticTypeIdV1>(),
            )?;
            drop(signature.parameter_semantic_types);
            budget.release_storage(semantic_bytes)?;
            let layout = ExecutionFunctionLayoutV29 {
                parameter_declarations: declarations,
                parameter_types: signature.parameter_types,
                call_arguments: signature.call_arguments,
                result_types: signature.result_types,
            };
            execution_instance_plan_from_layout_v29(
                instances, instance, function, placement, layout, budget,
            )
        })();
        result.inspect_err(|error| source_reference_record_failure_v29(self.references, error))
    }
}

impl ExecutionSignatureSourceV29<'_> {
    fn call_return_buffer_v29(
        &self,
        cursor: Option<&ExecutionAvailabilityV29<'_>>,
        function: &SemanticFunctionDeclV1,
        callables: &[SemanticCallableDeclV1],
        return_width: usize,
        budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<CallReturnBufferV1, ProductionSemanticKirErrorV1> {
        match self {
            Self::Legacy(map) => {
                CallReturnBufferV1::for_function(function, callables, map, return_width, budget)
            }
            Self::Scoped(owner) => {
                let cursor = cursor.ok_or_else(execution_instance_layout_error_v29)?;
                owner.check_cursor(cursor, budget)?;
                budget.source_reference_charge_v29(owner.references, 2)?;
                if !std::ptr::eq(cursor.function, function)
                    || !std::ptr::eq(
                        callables,
                        owner
                            .references
                            .instances
                            .owner()
                            .source_semantic()
                            .callables(),
                    )
                {
                    return Err(execution_instance_layout_error_v29());
                }
                budget.charge_work(function.blocks().len())?;
                CallReturnBufferV1::for_function_widths_v29(
                    function,
                    callables,
                    return_width,
                    budget,
                    |block, call, callee, budget| {
                        let control = cursor.source_call_control_v29(block, call, budget)?;
                        if control == ProductionCallControlV1::Unreachable {
                            return Ok(0);
                        }
                        let row = owner.call_row(cursor, block, call, callee, budget)?;
                        // The typed Call still uses its ABI result shape, but no
                        // semantic return payload exists on a suppressed edge.
                        Ok(if control == ProductionCallControlV1::MayReturn {
                            row.signature.result_types.len()
                        } else {
                            0
                        })
                    },
                )
                .inspect_err(|error| source_reference_record_failure_v29(owner.references, error))
            }
        }
    }

    fn signature_for_call_v29(
        &self,
        cursor: Option<&ExecutionAvailabilityV29<'_>>,
        caller: SemanticFunctionIdV1,
        block: SemanticBlockIdV1,
        call: &SemanticDirectCallV1,
        callee: SemanticFunctionIdV1,
        scoped: bool,
        budget: Option<&mut dyn SemanticEmissionBudgetV1>,
    ) -> Result<LoweredFunctionSignatureV1, ProductionSemanticKirErrorV1> {
        match self {
            Self::Scoped(owner) if scoped => owner.signature_for_call_v29(
                cursor.ok_or_else(execution_instance_layout_error_v29)?,
                block,
                call,
                callee,
                budget.ok_or(ArgumentResourceV1::Accounting)?,
            ),
            Self::Legacy(map) => {
                let signature = map.get(&callee).ok_or_else(|| {
                    unsupported(
                        caller.index(),
                        Some(block.index()),
                        None,
                        "defined call target has no exact KIR signature",
                    )
                })?;
                if scoped {
                    clone_execution_function_signature_v29(
                        signature,
                        budget.ok_or(ArgumentResourceV1::Accounting)?,
                    )
                } else {
                    Ok(signature.clone())
                }
            }
            _ => Err(execution_instance_layout_error_v29()),
        }
    }
}
