fn emission_service_error_v1() -> ProductionSemanticKirErrorV1 {
    unsupported(
        0,
        None,
        None,
        "emission services differ from their original function frame",
    )
}

struct EmissionServicesV1<'service> {
    budget: Option<&'service mut dyn SemanticEmissionBudgetV1>,
    calls: Option<&'service mut (dyn ExecutionDefinedCallConsumerV29 + 'service)>,
    lifecycle: Option<&'service mut (dyn ExecutionLifecycleConsumerV29 + 'service)>,
    private: PrivateArrayRecorderWorkV1<'service>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct EmissionServiceSlotsV1 {
    budget: usize,
    private: usize,
    calls: Option<usize>,
    lifecycle: Option<usize>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct EmissionServiceSourceV1 {
    source: ExecutionCallSourceV29,
    instance: ProductionCallInstanceIdV1,
    function: usize,
    ssa: usize,
    references: Option<usize>,
    identities: Option<usize>,
}

type EmissionServiceGrowthV1<'source> =
    source_storage_v29::SourceStorageRootGrowthV29<'source, 'source, 'source>;

fn capture_emission_service_growth_v1<'source>(
    cursor: Option<&ExecutionAvailabilityV29<'source>>,
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Option<EmissionServiceGrowthV1<'source>>, ProductionSemanticKirErrorV1> {
    let Some(references) = cursor.and_then(|cursor| cursor.references) else {
        return Ok(None);
    };
    let Some(root) = references.plan.storage_root.as_ref() else {
        return Ok(None);
    };
    budget.source_reference_owner_v29(references.plan)?;
    budget.charge_work(source_storage_v29::SOURCE_STORAGE_ROOT_GROWTH_WORK_V29)?;
    root.capture_retained_growth()
        .map(Some)
        .ok_or_else(|| ArgumentResourceV1::Accounting.into())
}

struct EmissionServiceSealV1<'source> {
    slots: EmissionServiceSlotsV1,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    storage: usize,
    work: usize,
    credit: usize,
    function: usize,
    types: usize,
    type_count: usize,
    callables: usize,
    callable_count: usize,
    owner: SemanticFunctionIdV1,
    semantic_function: SemanticFunctionIdV1,
    source: Option<EmissionServiceSourceV1>,
    placement: SemanticEmissionPlacementV1,
    next_value: u32,
    emitted_operations: usize,
    max_operations: usize,
    private_state: EmissionPrivateStateV1,
    growth: Option<EmissionServiceGrowthV1<'source>>,
}

struct EmissionPrivateStateV1 {
    roots: usize,
    operations: usize,
    active: Option<(
        usize,
        usize,
        usize,
        Option<fe2o3_kernel_ir::CanonicalKernelIrWorkLimitV1>,
    )>,
}

impl EmissionPrivateStateV1 {
    fn capture(
        work: &PrivateArrayRecorderWorkV1<'_>,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let PrivateArrayRecorderWorkV1::Shared(work) = work else {
            return Err(emission_service_error_v1());
        };
        Ok(Self {
            roots: work.roots,
            operations: work.operations,
            active: work.active.as_ref().map(|active| {
                (
                    &active.work as *const fe2o3_kernel_ir::CanonicalKernelIrWorkBudgetV1 as usize,
                    active.work.limit(),
                    active.work.work(),
                    active.first_denial,
                )
            }),
        })
    }

    fn admits(&self, current: &Self) -> bool {
        self.roots == current.roots
            && self.operations == current.operations
            && match (self.active, current.active) {
                (None, _) => true,
                (Some((ledger, limit, work, denial)), Some((now, cap, spent, failure))) => {
                    ledger == now
                        && limit == cap
                        && spent >= work
                        && denial.is_none_or(|denial| failure == Some(denial))
                }
                (Some(_), None) => false,
            }
    }
}

struct DetachedEmissionStateV1<'source> {
    lowering: SemanticFunctionLoweringV1<'source, 'static>,
    seal: EmissionServiceSealV1<'source>,
}

fn emission_service_headers_v1() -> Result<usize, ProductionSemanticKirErrorV1> {
    argument_sum_v1(&[
        std::mem::size_of::<EmissionServiceSealV1<'static>>(),
        std::mem::size_of::<EmissionServicesV1<'static>>(),
        std::mem::size_of::<
            Result<
                (
                    DetachedEmissionStateV1<'static>,
                    EmissionServicesV1<'static>,
                ),
                ProductionSemanticKirErrorV1,
            >,
        >(),
    ])
    .map_err(Into::into)
}

fn emission_private_slot_v1(
    work: &PrivateArrayRecorderWorkV1<'_>,
) -> Result<usize, ProductionSemanticKirErrorV1> {
    match work {
        PrivateArrayRecorderWorkV1::Shared(work) => {
            Ok(&**work as *const PrivateArrayLazyBudgetV1 as usize)
        }
        PrivateArrayRecorderWorkV1::Detached => Err(emission_service_error_v1()),
        #[cfg(test)]
        PrivateArrayRecorderWorkV1::Owned(_) => Err(emission_service_error_v1()),
    }
}

fn emission_service_source_v1(cursor: &ExecutionAvailabilityV29<'_>) -> EmissionServiceSourceV1 {
    EmissionServiceSourceV1 {
        source: cursor.source,
        instance: cursor.instance,
        function: cursor.function as *const SemanticFunctionDeclV1 as usize,
        ssa: cursor.ssa as *const ProductionSemanticSsaFunctionPlanV1 as usize,
        references: cursor
            .references
            .map(|row| row as *const SourceReferenceEmissionV29<'_, '_> as usize),
        identities: cursor
            .identities
            .map(|(row, _)| row as *const ExecutionIdentityPlanV1<'_, '_> as usize),
    }
}

impl<'source, 'service> SemanticFunctionLoweringV1<'source, 'service> {
    fn capture_service_seal_v1(
        &mut self,
    ) -> Result<EmissionServiceSealV1<'source>, ProductionSemanticKirErrorV1> {
        self.capture_service_seal_at_v1(None)
    }

    fn capture_service_seal_at_v1(
        &mut self,
        call: Option<&ExecutionCallRequestV1>,
    ) -> Result<EmissionServiceSealV1<'source>, ProductionSemanticKirErrorV1> {
        self.with_emission_budget_v1(|this, budget| {
            budget.charge_work(8)?;
            if this.scoped_memory.as_ref().is_some_and(|row| {
                row.frame.is_some() || row.read_payload.is_some() || row.store_payload.is_some()
            }) || this.private_arrays.frame.is_some()
                || this.private_arrays.pending.is_some()
            {
                return Err(emission_service_error_v1());
            }
            match call {
                None if this
                    .execution
                    .as_ref()
                    .is_some_and(|cursor| cursor.block.is_some()) =>
                {
                    return Err(emission_service_error_v1());
                }
                Some(request) => request.check_position_v1(this, budget)?,
                None => {}
            }
            if let Some(cursor) = &this.execution {
                cursor.check_ledger(budget)?;
                if let Some(references) = cursor.references {
                    budget.source_reference_owner_v29(references.plan)?;
                }
                if let Some(consumer) = this.lifecycle.as_deref() {
                    consumer.check_instance(cursor, budget)?;
                }
            } else if this.lifecycle.is_some() || this.execution_calls.is_some() {
                return Err(emission_service_error_v1());
            }
            let slots = EmissionServiceSlotsV1 {
                budget: budget
                    .emission_service_slot_v1()
                    .ok_or_else(emission_service_error_v1)?,
                private: emission_private_slot_v1(&this.private_arrays.work)?,
                calls: this.execution_calls.as_deref().map(|row| {
                    row as *const dyn ExecutionDefinedCallConsumerV29 as *const () as usize
                }),
                lifecycle: this.lifecycle.as_deref().map(|row| {
                    row as *const dyn ExecutionLifecycleConsumerV29 as *const () as usize
                }),
            };
            budget.charge_work(6)?;
            let private_state = EmissionPrivateStateV1::capture(&this.private_arrays.work)?;
            let work = budget
                .emission_service_work_v1()
                .ok_or_else(emission_service_error_v1)?;
            let credit = emission_service_headers_v1()?;
            budget.reserve_storage(credit)?;
            let growth = capture_emission_service_growth_v1(this.execution.as_ref(), budget)?;
            Ok(EmissionServiceSealV1 {
                slots,
                ledger: budget.work_ledger_identity_v1(),
                storage: budget.storage(),
                work,
                credit,
                function: this.function as *const SemanticFunctionDeclV1 as usize,
                types: this.types.as_ptr() as usize,
                type_count: this.types.len(),
                callables: this.callables.as_ptr() as usize,
                callable_count: this.callables.len(),
                owner: this.correspondence_owner,
                semantic_function: this.semantic_function,
                source: this.execution.as_ref().map(emission_service_source_v1),
                placement: this.emission_placement,
                next_value: this.next_value,
                emitted_operations: this.emitted_operations,
                max_operations: this.max_operations,
                private_state,
                growth,
            })
        })
    }

    fn detach_services_v1(
        mut self,
    ) -> Result<
        (
            DetachedEmissionStateV1<'source>,
            EmissionServicesV1<'service>,
        ),
        ProductionSemanticKirErrorV1,
    > {
        let seal = self.capture_service_seal_v1()?;
        let (lowering, services) = self.replace_services_v1(EmissionServicesV1 {
            budget: None,
            calls: None,
            lifecycle: None,
            private: PrivateArrayRecorderWorkV1::Detached,
        });
        Ok((DetachedEmissionStateV1 { lowering, seal }, services))
    }

    fn detach_call_services_v1(
        mut self,
        request: &ExecutionCallRequestV1,
    ) -> Result<
        (
            DetachedEmissionStateV1<'source>,
            EmissionServicesV1<'service>,
        ),
        ProductionSemanticKirErrorV1,
    > {
        let seal = self.capture_service_seal_at_v1(Some(request))?;
        let (lowering, services) = self.replace_services_v1(EmissionServicesV1 {
            budget: None,
            calls: None,
            lifecycle: None,
            private: PrivateArrayRecorderWorkV1::Detached,
        });
        Ok((DetachedEmissionStateV1 { lowering, seal }, services))
    }

    fn replace_services_v1<'next>(
        self,
        services: EmissionServicesV1<'next>,
    ) -> (
        SemanticFunctionLoweringV1<'source, 'next>,
        EmissionServicesV1<'service>,
    ) {
        let Self {
            wave64_references,
            ordered_composition,
            emission_work,
            scoped_memory,
            fixed_array_analysis,
            private_arrays,
            types,
            callables,
            function,
            correspondence_owner,
            semantic_function,
            defined_function_ids,
            defined_function_signatures,
            result_types,
            locals,
            retained_local_slots,
            retained_local_allocas_emitted,
            invocation_preheader,
            retained_local_initialized,
            option_dominance,
            enum_payload_dominance,
            enum_payload_storage,
            enum_payload_sources,
            enum_payload_requires_compile_time_custody,
            enum_payload_compile_time_custody,
            enum_payload_allocas_emitted,
            control_flow_ssa,
            workgroup_pipeline_contracts,
            promoted_enum_variant_by_value,
            block_parameters,
            semantic_ssa_bindings,
            semantic_rvalue_bindings,
            semantic_ssa_archive_credit,
            pending_semantic_ssa_definitions,
            next_value,
            emission_placement,
            execution,
            execution_calls,
            lifecycle,
            assert_failure_block,
            required_workgroup,
            infallible_asserts,
            launch_rank,
            max_operations,
            emitted_operations,
            emitted_workgroup_memory_extents,
            emitted_unsigned_constants,
            emitted_u32_constants,
            emitted_u32_bitand_masks,
            authenticated_loop_induction_bounds,
            emitted_unsigned_exclusive_bounds,
            generated_terminator_values,
            call_returns,
        } = self;
        let (private_arrays, private) = private_arrays.replace_work(services.private);
        (
            SemanticFunctionLoweringV1 {
                wave64_references,
                ordered_composition,
                emission_work: services.budget,
                execution_calls: services.calls,
                lifecycle: services.lifecycle,
                scoped_memory,
                fixed_array_analysis,
                private_arrays,
                types,
                callables,
                function,
                correspondence_owner,
                semantic_function,
                defined_function_ids,
                defined_function_signatures,
                result_types,
                locals,
                retained_local_slots,
                retained_local_allocas_emitted,
                invocation_preheader,
                retained_local_initialized,
                option_dominance,
                enum_payload_dominance,
                enum_payload_storage,
                enum_payload_sources,
                enum_payload_requires_compile_time_custody,
                enum_payload_compile_time_custody,
                enum_payload_allocas_emitted,
                control_flow_ssa,
                workgroup_pipeline_contracts,
                promoted_enum_variant_by_value,
                block_parameters,
                semantic_ssa_bindings,
                semantic_rvalue_bindings,
                semantic_ssa_archive_credit,
                pending_semantic_ssa_definitions,
                next_value,
                emission_placement,
                execution,
                assert_failure_block,
                required_workgroup,
                infallible_asserts,
                launch_rank,
                max_operations,
                emitted_operations,
                emitted_workgroup_memory_extents,
                emitted_unsigned_constants,
                emitted_u32_constants,
                emitted_u32_bitand_masks,
                authenticated_loop_induction_bounds,
                emitted_unsigned_exclusive_bounds,
                generated_terminator_values,
                call_returns,
            },
            EmissionServicesV1 {
                budget: emission_work,
                calls: execution_calls,
                lifecycle,
                private,
            },
        )
    }
}

impl<'source> DetachedEmissionStateV1<'source> {
    fn deny_original_root_refund(&self) {
        if let Some(root) = self
            .lowering
            .execution
            .as_ref()
            .and_then(|cursor| cursor.references)
            .and_then(|emission| emission.plan.storage_root.as_ref())
        {
            root.deny_active_root_refund();
        }
    }

    fn attach<'service>(
        self,
        mut services: EmissionServicesV1<'service>,
    ) -> Result<SemanticFunctionLoweringV1<'source, 'service>, ProductionSemanticKirErrorV1> {
        let Some(budget) = services.budget.as_deref_mut() else {
            self.deny_original_root_refund();
            return Err(emission_service_error_v1());
        };
        let original_plan = self
            .lowering
            .execution
            .as_ref()
            .and_then(|cursor| cursor.references)
            .map(|emission| emission.plan);
        if !budget.permits_prepared_input_refund_v1(
            original_plan,
            self.seal.slots.budget,
            self.seal.ledger,
            self.seal.storage,
            0,
        ) || !self.seal.growth.as_ref().is_none_or(|growth| {
            self.seal
                .storage
                .checked_sub(self.seal.credit)
                .is_some_and(|entry| {
                    growth.permits_refund(entry, self.seal.storage, budget.storage(), 0)
                })
        }) {
            self.deny_original_root_refund();
            return Err(emission_service_error_v1());
        }
        let slots =
            EmissionServiceSlotsV1 {
                budget: budget
                    .emission_service_slot_v1()
                    .ok_or_else(emission_service_error_v1)?,
                private: emission_private_slot_v1(&services.private)?,
                calls: services.calls.as_deref().map(|row| {
                    row as *const dyn ExecutionDefinedCallConsumerV29 as *const () as usize
                }),
                lifecycle: services.lifecycle.as_deref().map(|row| {
                    row as *const dyn ExecutionLifecycleConsumerV29 as *const () as usize
                }),
            };
        if slots != self.seal.slots
            || budget.work_ledger_identity_v1() != self.seal.ledger
            || budget.storage() < self.seal.storage
            || budget
                .emission_service_work_v1()
                .is_none_or(|work| work < self.seal.work)
        {
            return Err(emission_service_error_v1());
        }
        budget.charge_work(12)?;
        let state = &self.lowering;
        if state.function as *const SemanticFunctionDeclV1 as usize != self.seal.function
            || state.types.as_ptr() as usize != self.seal.types
            || state.types.len() != self.seal.type_count
            || state.callables.as_ptr() as usize != self.seal.callables
            || state.callables.len() != self.seal.callable_count
            || state.correspondence_owner != self.seal.owner
            || state.semantic_function != self.seal.semantic_function
            || state.execution.as_ref().map(emission_service_source_v1) != self.seal.source
            || state.emission_placement != self.seal.placement
            || state.next_value != self.seal.next_value
            || state.emitted_operations != self.seal.emitted_operations
            || state.max_operations != self.seal.max_operations
            || state.emission_work.is_some()
            || state.execution_calls.is_some()
            || state.lifecycle.is_some()
            || !matches!(
                &state.private_arrays.work,
                PrivateArrayRecorderWorkV1::Detached
            )
        {
            return Err(emission_service_error_v1());
        }
        budget.charge_work(6)?;
        if !self
            .seal
            .private_state
            .admits(&EmissionPrivateStateV1::capture(&services.private)?)
        {
            return Err(emission_service_error_v1());
        }
        if let Some(cursor) = &state.execution {
            cursor.check_ledger(budget)?;
            if let Some(references) = cursor.references {
                budget.source_reference_owner_v29(references.plan)?;
            }
            if let Some(consumer) = services.lifecycle.as_deref() {
                consumer.check_instance(cursor, budget)?;
            }
        }
        if self.seal.growth.is_some() {
            budget.charge_work(4)?;
        }
        if !self.seal.growth.as_ref().is_none_or(|growth| {
            self.seal
                .storage
                .checked_sub(self.seal.credit)
                .is_some_and(|entry| {
                    growth.permits_refund(
                        entry,
                        self.seal.storage,
                        budget.storage(),
                        self.seal.credit,
                    )
                })
        }) {
            self.deny_original_root_refund();
            return Err(ArgumentResourceV1::Accounting.into());
        }
        let credit = self.seal.credit;
        let required = self.seal.storage;
        let slot = self.seal.slots.budget;
        let ledger = self.seal.ledger;
        drop(self.seal);
        let (mut lowering, old) = self.lowering.replace_services_v1(services);
        drop(old);
        lowering.with_emission_budget_v1(|this, budget| {
            let plan = this
                .execution
                .as_ref()
                .and_then(|cursor| cursor.references)
                .map(|row| row.plan);
            budget.release_emission_service_storage_v1(plan, slot, ledger, required, credit)
        })?;
        Ok(lowering)
    }
}
