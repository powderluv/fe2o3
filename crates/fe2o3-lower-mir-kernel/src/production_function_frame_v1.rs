enum FunctionEmissionPlanV1<'plan> {
    Borrowed(&'plan LoweredFunctionPlanV1),
    Owned(LoweredFunctionPlanV1),
}

#[cfg(test)]
fn copy_execution_observation_seeds_v1(
    seeds: &[Option<SemanticExecutionBindingV29>],
    budget: &mut dyn SemanticEmissionBudgetV1,
) -> Result<Vec<Option<SemanticExecutionBindingV29>>, ProductionSemanticKirErrorV1> {
    let mut retained = emission_vec_v1(seeds.len(), budget)?;
    budget.charge_work(seeds.len())?;
    retained.extend_from_slice(seeds);
    Ok(retained)
}

#[cfg(test)]
std::thread_local! {
    static DETACH_EMISSION_BLOCKS_V1: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct FunctionEmissionPlanPartsV1<'plan> {
    assembly: EmissionReadOnlyV1<'plan, LoweredFunctionPlanV1>,
    result_types: EmissionReadOnlyV1<'plan, Vec<Type>>,
}

struct FunctionPlanStorageV1<'source> {
    source: Option<&'source SourceReferencePlanV29<'source, 'source>>,
    growth: Option<EmissionServiceGrowthV1<'source>>,
    entry: usize,
    required: usize,
    credit: usize,
    slot: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
}

type FunctionEmissionOutcomeV1 = Result<
    Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1>,
    Box<dyn std::any::Any + Send>,
>;

fn function_frame_plan_headers_v1() -> Result<usize, ProductionSemanticKirErrorV1> {
    argument_sum_v1(&[
        std::mem::size_of::<FunctionEmissionPlanV1<'static>>(),
        std::mem::size_of::<FunctionEmissionPlanPartsV1<'static>>(),
        std::mem::size_of::<FunctionPlanStorageV1<'static>>(),
        std::mem::size_of::<FunctionEmissionOutcomeV1>(),
        std::mem::size_of::<PreparedFunctionFrameV1<'static, 'static>>(),
        std::mem::size_of::<
            Result<PreparedFunctionFrameV1<'static, 'static>, ProductionSemanticKirErrorV1>,
        >(),
        std::mem::size_of::<UnassembledFunctionFrameV1<'static>>(),
        std::mem::size_of::<
            Result<UnassembledFunctionFrameV1<'static>, ProductionSemanticKirErrorV1>,
        >(),
    ])
    .map_err(Into::into)
}

impl<'source> FunctionPlanStorageV1<'source> {
    fn new(
        cursor: Option<&ExecutionAvailabilityV29<'source>>,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let slot = budget
            .emission_service_slot_v1()
            .ok_or_else(emission_service_error_v1)?;
        let source = cursor
            .and_then(|cursor| cursor.references)
            .map(|emission| emission.plan);
        let entry = budget.storage();
        let credit = function_frame_plan_headers_v1()?;
        budget.reserve_storage(credit)?;
        let growth = capture_emission_service_growth_v1(cursor, budget)?;
        Ok(Self {
            source,
            growth,
            entry,
            credit,
            required: budget.storage(),
            slot,
            ledger: budget.work_ledger_identity_v1(),
        })
    }

    fn finish(
        self,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let allowed = budget.permits_prepared_input_refund_v1(
            self.source,
            self.slot,
            self.ledger,
            self.required,
            self.credit,
        ) && self.growth.as_ref().is_none_or(|growth| {
            growth.permits_refund(self.entry, self.required, budget.storage(), self.credit)
        });
        let source = self.source;
        let credit = self.credit;
        drop(self);
        if !allowed || budget.release_storage(credit).is_err() {
            if let Some(root) = source.and_then(|plan| plan.storage_root.as_ref()) {
                root.deny_active_root_refund();
            }
            return Err(ArgumentResourceV1::Accounting.into());
        }
        Ok(())
    }

    fn abandon(self, budget: &dyn SemanticEmissionBudgetV1) {
        let allowed = budget.permits_prepared_input_refund_v1(
            self.source,
            self.slot,
            self.ledger,
            self.required,
            0,
        ) && self.growth.as_ref().is_none_or(|growth| {
            growth.permits_refund(self.entry, self.required, budget.storage(), 0)
        });
        let source = self.source;
        drop(self);
        if !allowed {
            if let Some(root) = source.and_then(|plan| plan.storage_root.as_ref()) {
                root.deny_active_root_refund();
            }
        }
    }
}

impl<'plan> FunctionEmissionPlanV1<'plan> {
    fn split(self) -> FunctionEmissionPlanPartsV1<'plan> {
        match self {
            Self::Borrowed(plan) => FunctionEmissionPlanPartsV1 {
                assembly: EmissionReadOnlyV1::Borrowed(plan),
                result_types: EmissionReadOnlyV1::Borrowed(&plan.result_types),
            },
            Self::Owned(mut plan) => {
                let result_types = std::mem::take(&mut plan.result_types);
                FunctionEmissionPlanPartsV1 {
                    assembly: EmissionReadOnlyV1::Owned(plan),
                    result_types: EmissionReadOnlyV1::Owned(result_types),
                }
            }
        }
    }
}

struct FunctionFrameBodyV1 {
    order: Vec<SemanticBlockIdV1>,
    target_blocks: Vec<BasicBlock>,
    blocks: Vec<SemanticKirBlockCorrespondenceV1>,
    statements: Vec<SemanticKirStatementOperationSpanV1>,
    terminators: Vec<SemanticKirTerminatorOperationSpanV1>,
    synthetic: Vec<SemanticKirSyntheticOperationSpanV1>,
}

struct FunctionFrameControlV1<'source> {
    next_block: usize,
    failed: bool,
    credit: usize,
    required: usize,
    slot: usize,
    ledger: fe2o3_kernel_ir::CanonicalKernelIrWorkLedgerIdentityV1,
    growth: Option<EmissionServiceGrowthV1<'source>>,
    backing: Option<SourceFunctionBackingViewV29<'source>>,
}

impl FunctionFrameControlV1<'_> {
    fn check(
        &self,
        plan: Option<&SourceReferencePlanV29<'_, '_>>,
        instance: Option<ProductionCallInstanceIdV1>,
        budget: &mut dyn SemanticEmissionBudgetV1,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if let Some(backing) = self.backing {
            backing.check(plan, instance, budget)?;
        }
        budget.release_emission_service_storage_v1(
            plan,
            self.slot,
            self.ledger,
            self.required,
            0,
        )?;
        if let Some(growth) = &self.growth {
            budget.charge_work(4)?;
            if !self.required.checked_sub(self.credit).is_some_and(|entry| {
                growth.permits_refund(entry, self.required, budget.storage(), 0)
            }) {
                if let Some(root) = plan.and_then(|plan| plan.storage_root.as_ref()) {
                    root.deny_active_root_refund();
                }
                return Err(ArgumentResourceV1::Accounting.into());
            }
        }
        Ok(())
    }
}

struct FunctionFrameV1<'source, 'service, 'assembly> {
    lowering: SemanticFunctionLoweringV1<'source, 'service>,
    assembly: EmissionReadOnlyV1<'assembly, LoweredFunctionPlanV1>,
    body: FunctionFrameBodyV1,
    control: FunctionFrameControlV1<'source>,
}

struct DetachedFunctionFrameV1<'source, 'assembly> {
    lowering: DetachedEmissionStateV1<'source>,
    assembly: EmissionReadOnlyV1<'assembly, LoweredFunctionPlanV1>,
    body: FunctionFrameBodyV1,
    control: FunctionFrameControlV1<'source>,
}

fn function_frame_headers_v1() -> Result<usize, ProductionSemanticKirErrorV1> {
    argument_sum_v1(&[
        std::mem::size_of::<FunctionFrameControlV1<'static>>(),
        std::mem::size_of::<Result<bool, ProductionSemanticKirErrorV1>>(),
        std::mem::size_of::<EmissionReadOnlyV1<'static, LoweredFunctionPlanV1>>()
            .checked_sub(std::mem::size_of::<&LoweredFunctionPlanV1>())
            .ok_or(ArgumentResourceV1::Arithmetic)?,
    ])
    .map_err(Into::into)
}

impl<'source, 'service, 'assembly> FunctionFrameV1<'source, 'service, 'assembly> {
    fn new(
        lowering: SemanticFunctionLoweringV1<'source, 'service>,
        assembly: &'assembly LoweredFunctionPlanV1,
        body: FunctionFrameBodyV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        Self::new_with_assembly(lowering, EmissionReadOnlyV1::Borrowed(assembly), body)
    }

    fn new_with_assembly(
        mut lowering: SemanticFunctionLoweringV1<'source, 'service>,
        assembly: EmissionReadOnlyV1<'assembly, LoweredFunctionPlanV1>,
        body: FunctionFrameBodyV1,
    ) -> Result<Self, ProductionSemanticKirErrorV1> {
        let control = lowering.with_emission_budget_v1(|this, budget| {
            budget.charge_work(4)?;
            if this.correspondence_owner != assembly.correspondence_owner
                || this.semantic_function != assembly.semantic_function
                || this
                    .execution
                    .as_ref()
                    .is_some_and(|cursor| cursor.block.is_some())
                || this.private_arrays.frame.is_some()
            {
                return Err(emission_service_error_v1());
            }
            let slot = budget
                .emission_service_slot_v1()
                .ok_or_else(emission_service_error_v1)?;
            let credit = function_frame_headers_v1()?;
            budget.reserve_storage(credit)?;
            let growth = capture_emission_service_growth_v1(this.execution.as_ref(), budget)?;
            let backing = SourceFunctionBackingViewV29::new(
                &this.defined_function_signatures,
                this.execution.as_ref(),
                this.semantic_function,
                budget,
            )?;
            Ok(FunctionFrameControlV1 {
                next_block: 0,
                failed: false,
                credit,
                required: budget.storage(),
                slot,
                ledger: budget.work_ledger_identity_v1(),
                growth,
                backing,
            })
        })?;
        Ok(Self {
            lowering,
            assembly,
            body,
            control,
        })
    }

    // A step completes one original block. Any refusal poisons this frame before
    // a partially emitted block could become a legal service boundary.
    fn step(
        &mut self,
        assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    ) -> Result<bool, ProductionSemanticKirErrorV1> {
        match self.step_inner_v1(assert_origins, false)? {
            FunctionFrameStepV1::Done => Ok(false),
            FunctionFrameStepV1::Block => Ok(true),
            FunctionFrameStepV1::AwaitChild(_) => Err(emission_service_error_v1()),
        }
    }

    fn step_inner_v1(
        &mut self,
        assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
        suspend_children: bool,
    ) -> Result<FunctionFrameStepV1, ProductionSemanticKirErrorV1> {
        if self.control.failed {
            return Err(emission_service_error_v1());
        }
        self.control.failed = true;
        let control = &self.control;
        self.lowering.with_emission_budget_v1(|this, budget| {
            let plan = this
                .execution
                .as_ref()
                .and_then(|cursor| cursor.references)
                .map(|emission| emission.plan);
            control.check(
                plan,
                this.execution.as_ref().map(|cursor| cursor.instance),
                budget,
            )?;
            budget.charge_work(2)
        })?;
        while let Some(&block) = self.body.order.get(self.control.next_block) {
            let active = self.lowering.with_emission_budget_v1(|this, budget| {
                match this.execution.as_ref() {
                    Some(cursor) => cursor.source_block_reachable_v29(block, budget),
                    None => Ok(true),
                }
            })?;
            if active {
                break;
            }
            self.control.next_block = self
                .control
                .next_block
                .checked_add(1)
                .ok_or(ArgumentResourceV1::Arithmetic)?;
        }
        let Some(&semantic_block) = self.body.order.get(self.control.next_block) else {
            if self.control.next_block != self.body.order.len() {
                return Err(emission_service_error_v1());
            }
            self.control.failed = false;
            return Ok(FunctionFrameStepV1::Done);
        };
        let lowering = &mut self.lowering;
        let plan = &*self.assembly;
        let function = lowering.function;
        let index = usize::try_from(semantic_block.index()).map_err(|_| {
            unsupported(
                plan.semantic_function.index(),
                None,
                None,
                "block identity does not fit this host",
            )
        })?;
        let source = function.blocks().get(index).ok_or_else(|| {
            unsupported(
                plan.semantic_function.index(),
                Some(semantic_block.index()),
                None,
                "block is missing",
            )
        })?;
        let mut target = BasicBlock::new(lowering.kernel_block_id_v1(semantic_block)?);
        let prologue = lowering.begin_block(semantic_block, &mut target)?;
        if prologue.retained_local_storage != 0 {
            self.body
                .synthetic
                .push(SemanticKirSyntheticOperationSpanV1 {
                    correspondence_owner: plan.correspondence_owner,
                    semantic_function: plan.semantic_function,
                    rule: SemanticKirSyntheticOperationRuleV1::RetainedLocalStorage,
                    kernel_ir_block: target.id,
                    first_operation_ordinal: 0,
                    operation_count: prologue.retained_local_storage,
                });
        }
        if prologue.enum_payload_storage != 0 {
            self.body
                .synthetic
                .push(SemanticKirSyntheticOperationSpanV1 {
                    correspondence_owner: plan.correspondence_owner,
                    semantic_function: plan.semantic_function,
                    rule: SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage,
                    kernel_ir_block: target.id,
                    first_operation_ordinal: prologue.retained_local_storage,
                    operation_count: prologue.enum_payload_storage,
                });
        }
        for (statement, operation) in source.statements().iter().enumerate() {
            let statement = u32::try_from(statement).map_err(|_| {
                unsupported(
                    plan.semantic_function.index(),
                    Some(semantic_block.index()),
                    None,
                    "statement ordinal is too large",
                )
            })?;
            let first = target.operations.len();
            lowering.lower_statement(
                semantic_block,
                Some(statement),
                operation.kind(),
                &mut target.operations,
            )?;
            let (first_operation_ordinal, operation_count) = measured_operation_span(
                first,
                target.operations.len(),
                target.id,
                Some(statement),
            )?;
            self.body
                .statements
                .push(SemanticKirStatementOperationSpanV1 {
                    correspondence_owner: plan.correspondence_owner,
                    semantic_function: plan.semantic_function,
                    semantic_block,
                    statement_ordinal: statement,
                    kernel_ir_block: target.id,
                    first_operation_ordinal,
                    operation_count,
                });
        }
        let terminator_first = target.operations.len();
        if suspend_children
            && let SemanticTerminatorKindV1::Call(call) = source.terminator().kind()
            && let Some(SemanticCallableDeclV1::Defined { function: callee }) =
                lowering.callables.get(call.callee().index() as usize)
        {
            let callee = *callee;
            let credit = argument_sum_v1(&[
                std::mem::size_of::<AwaitingDefinedCallV1>(),
                std::mem::size_of::<SuspendedCallStartV1>(),
                std::mem::size_of::<FunctionFrameStepV1>(),
                std::mem::size_of::<Result<FunctionFrameStepV1, ProductionSemanticKirErrorV1>>(),
            ])?;
            let budget = lowering
                .emission_work
                .as_deref_mut()
                .ok_or(ArgumentResourceV1::Accounting)?;
            budget.charge_work(4)?;
            budget.reserve_storage(credit)?;
            let (prefix, request) = lowering.request_defined_call_v1(
                semantic_block,
                call,
                callee,
                &mut target.operations,
            )?;
            // The frame remains non-steppable and non-finishable until this
            // exact terminator's checked child completion is consumed.
            return Ok(FunctionFrameStepV1::AwaitChild(AwaitingDefinedCallV1 {
                prefix,
                request,
                target,
                terminator_first,
                credit,
            }));
        }
        target.terminator = Some(lowering.lower_terminator(
            semantic_block,
            source.terminator().kind(),
            &mut target.operations,
        )?);
        self.finish_emitted_block_v1(semantic_block, target, terminator_first, assert_origins)?;
        Ok(FunctionFrameStepV1::Block)
    }

    fn finish_emitted_block_v1(
        &mut self,
        semantic_block: SemanticBlockIdV1,
        target: BasicBlock,
        terminator_first: usize,
        assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        if !self.control.failed
            || self.body.order.get(self.control.next_block) != Some(&semantic_block)
            || target.terminator.is_none()
        {
            return Err(emission_service_error_v1());
        }
        let lowering = &mut self.lowering;
        let plan = &*self.assembly;
        let source = lowering
            .function
            .blocks()
            .get(semantic_block.index() as usize)
            .ok_or_else(emission_service_error_v1)?;
        if lowering.execution.is_some() {
            lowering.with_emission_budget_v1(|this, budget| {
                this.execution.as_mut().unwrap().finish_block(budget)
            })?;
        }
        let (first_operation_ordinal, operation_count) =
            measured_operation_span(terminator_first, target.operations.len(), target.id, None)?;
        self.body
            .terminators
            .push(SemanticKirTerminatorOperationSpanV1 {
                correspondence_owner: plan.correspondence_owner,
                semantic_function: plan.semantic_function,
                semantic_block,
                kernel_ir_block: target.id,
                first_operation_ordinal,
                operation_count,
            });
        if let Some(origins) = assert_origins {
            origins.record(
                SemanticKirTerminatorOperationSpanV1 {
                    correspondence_owner: plan.correspondence_owner,
                    semantic_function: plan.semantic_function,
                    semantic_block,
                    kernel_ir_block: target.id,
                    first_operation_ordinal,
                    operation_count,
                },
                &plan.kernel_ir_function,
                source.terminator().kind(),
                target
                    .terminator
                    .as_ref()
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?,
                lowering
                    .infallible_asserts
                    .contains(&semantic_block.index()),
            )?;
        }
        self.body.target_blocks.push(target);
        self.body.blocks.push(SemanticKirBlockCorrespondenceV1 {
            correspondence_owner: plan.correspondence_owner,
            semantic_function: plan.semantic_function,
            semantic_block,
            kernel_ir_block: lowering.kernel_block_id_v1(semantic_block)?,
            source_statement_count: u32::try_from(source.statements().len()).map_err(|_| {
                unsupported(
                    plan.semantic_function.index(),
                    Some(semantic_block.index()),
                    None,
                    "statement count is too large",
                )
            })?,
        });
        self.control.next_block = self
            .control
            .next_block
            .checked_add(1)
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        self.control.failed = false;
        Ok(())
    }

    fn resume_defined_call_v1(
        &mut self,
        pending: AwaitingDefinedCallV1,
        completed: CheckedExecutionReturnV1,
        assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    ) -> Result<(), ProductionSemanticKirErrorV1> {
        let block = pending.request.occurrence.block;
        if !self.control.failed
            || self.body.order.get(self.control.next_block) != Some(&block)
            || pending.target.terminator.is_some()
            || pending.prefix.block != block
        {
            return Err(emission_service_error_v1());
        }
        self.lowering.with_emission_budget_v1(|this, budget| {
            pending.request.check_position_v1(this, budget)?;
            budget.charge_work(7)?;
            if completed.source != pending.request.source
                || completed.occurrence != pending.request.occurrence
                || completed.child != pending.request.child
                || completed.original_call != pending.request.original_call
                || completed.callee != pending.request.callee
                || completed.normal_return != pending.request.normal_return
                || pending.prefix.normal_return != pending.request.normal_return
                || pending.target.id != this.kernel_block_id_v1(block)?
            {
                return Err(execution_call_error_v29());
            }
            Ok(())
        })?;
        let AwaitingDefinedCallV1 {
            prefix,
            request,
            mut target,
            terminator_first,
            credit: pending_credit,
        } = pending;
        let ExecutionCallRequestV1 {
            target: callee_id,
            arguments,
            credit: request_credit,
            ..
        } = request;
        let CheckedExecutionReturnV1 {
            nominal,
            credit: completion_credit,
            ..
        } = completed;
        let original = self
            .lowering
            .function
            .blocks()
            .get(block.index() as usize)
            .ok_or_else(execution_call_error_v29)?;
        let SemanticTerminatorKindV1::Call(call) = original.terminator().kind() else {
            return Err(execution_call_error_v29());
        };
        let (terminator, consumed_credit) = self.lowering.with_scoped_memory_frame_v29(
            ScopedMemoryFrameV29 {
                site: execution_site_v29(block, None),
                role: None,
            },
            |this| {
                let (terminator, consumed_credit) = this.finish_defined_call_prefix_v1(
                    call,
                    prefix,
                    callee_id,
                    arguments,
                    nominal,
                    &mut target.operations,
                )?;
                this.record_source_descriptor_guard_v29(block, &terminator)?;
                Ok((terminator, consumed_credit))
            },
        )?;
        target.terminator = Some(terminator);
        self.finish_emitted_block_v1(block, target, terminator_first, assert_origins)?;
        self.lowering
            .emission_work
            .as_deref_mut()
            .ok_or(ArgumentResourceV1::Accounting)?
            .release_storage(argument_sum_v1(&[
                pending_credit,
                request_credit,
                completion_credit,
                consumed_credit,
            ])?)?;
        Ok(())
    }

    fn detach(
        mut self,
    ) -> Result<
        (
            DetachedFunctionFrameV1<'source, 'assembly>,
            EmissionServicesV1<'service>,
        ),
        ProductionSemanticKirErrorV1,
    > {
        if self.control.failed
            || self.control.next_block == 0
            || self.control.next_block > self.body.order.len()
        {
            return Err(emission_service_error_v1());
        }
        let control = &self.control;
        self.lowering.with_emission_budget_v1(|this, budget| {
            let plan = this
                .execution
                .as_ref()
                .and_then(|cursor| cursor.references)
                .map(|emission| emission.plan);
            control.check(
                plan,
                this.execution.as_ref().map(|cursor| cursor.instance),
                budget,
            )
        })?;
        let (lowering, services) = self.lowering.detach_services_v1()?;
        Ok((
            DetachedFunctionFrameV1 {
                lowering,
                assembly: self.assembly,
                body: self.body,
                control: self.control,
            },
            services,
        ))
    }

    fn detach_call_v1(
        mut self,
        pending: &AwaitingDefinedCallV1,
    ) -> Result<
        (
            DetachedFunctionFrameV1<'source, 'assembly>,
            EmissionServicesV1<'service>,
            SuspendedCallStartV1,
        ),
        ProductionSemanticKirErrorV1,
    > {
        if !self.control.failed
            || self.body.order.get(self.control.next_block)
                != Some(&pending.request.occurrence.block)
            || pending.target.terminator.is_some()
        {
            return Err(emission_service_error_v1());
        }
        let control = &self.control;
        self.lowering.with_emission_budget_v1(|this, budget| {
            pending.request.check_position_v1(this, budget)?;
            let plan = this
                .execution
                .as_ref()
                .and_then(|cursor| cursor.references)
                .map(|row| row.plan);
            control.check(
                plan,
                this.execution.as_ref().map(|cursor| cursor.instance),
                budget,
            )?;
            budget.charge_work(5)?;
            Ok(())
        })?;
        let blocks = u32::try_from(self.lowering.function.blocks().len())
            .map_err(|_| ArgumentResourceV1::Arithmetic)?;
        let next_block = self
            .lowering
            .emission_placement
            .first_block
            .checked_add(blocks)
            .and_then(|next| {
                next.checked_add(u32::from(self.lowering.assert_failure_block.is_some()))
            })
            .and_then(|next| {
                next.checked_add(u32::from(self.lowering.invocation_preheader.is_some()))
            })
            .ok_or(ArgumentResourceV1::Arithmetic)?;
        let start = SuspendedCallStartV1 {
            first_value: self.lowering.next_value,
            next_block,
            remaining_operations: self
                .lowering
                .max_operations
                .checked_sub(self.lowering.emitted_operations)
                .ok_or_else(emission_service_error_v1)?,
            private_payload: self.lowering.private_arrays.live_payload_v1(0, 0, 0, 0)?,
        };
        let (lowering, services) = self.lowering.detach_call_services_v1(&pending.request)?;
        Ok((
            DetachedFunctionFrameV1 {
                lowering,
                assembly: self.assembly,
                body: self.body,
                control: self.control,
            },
            services,
            start,
        ))
    }

    fn finish_body(
        self,
    ) -> Result<
        (
            SemanticFunctionLoweringV1<'source, 'service>,
            FunctionFrameBodyV1,
        ),
        ProductionSemanticKirErrorV1,
    > {
        let (lowering, assembly, body) = self.finish_parts()?;
        drop(assembly);
        Ok((lowering, body))
    }

    fn finish_parts(
        self,
    ) -> Result<
        (
            SemanticFunctionLoweringV1<'source, 'service>,
            EmissionReadOnlyV1<'assembly, LoweredFunctionPlanV1>,
            FunctionFrameBodyV1,
        ),
        ProductionSemanticKirErrorV1,
    > {
        if self.control.failed || self.control.next_block != self.body.order.len() {
            return Err(emission_service_error_v1());
        }
        let Self {
            mut lowering,
            assembly,
            body,
            control,
        } = self;
        let credit = control.credit;
        let required = control.required;
        let slot = control.slot;
        let ledger = control.ledger;
        lowering.with_emission_budget_v1(|this, budget| {
            let plan = this
                .execution
                .as_ref()
                .and_then(|cursor| cursor.references)
                .map(|row| row.plan);
            if let Some(backing) = control.backing {
                backing.check(
                    plan,
                    this.execution.as_ref().map(|cursor| cursor.instance),
                    budget,
                )?;
            }
            budget.release_emission_service_storage_v1(plan, slot, ledger, required, 0)?;
            budget.charge_work(2)?;
            let growth_permits = control.growth.as_ref().is_none_or(|growth| {
                required.checked_sub(credit).is_some_and(|entry| {
                    growth.permits_refund(entry, required, budget.storage(), credit)
                })
            });
            drop(control);
            if !growth_permits {
                if let Some(root) = plan.and_then(|plan| plan.storage_root.as_ref()) {
                    root.deny_active_root_refund();
                }
                return Err(ArgumentResourceV1::Accounting.into());
            }
            budget.release_emission_service_storage_v1(plan, slot, ledger, required, credit)
        })?;
        Ok((lowering, assembly, body))
    }
}

impl<'source, 'assembly> DetachedFunctionFrameV1<'source, 'assembly> {
    fn attach<'service>(
        self,
        mut services: EmissionServicesV1<'service>,
    ) -> Result<FunctionFrameV1<'source, 'service, 'assembly>, ProductionSemanticKirErrorV1> {
        let Some(budget) = services.budget.as_deref_mut() else {
            self.lowering.deny_original_root_refund();
            return Err(emission_service_error_v1());
        };
        let plan = self
            .lowering
            .lowering
            .execution
            .as_ref()
            .and_then(|cursor| cursor.references)
            .map(|emission| emission.plan);
        self.control.check(
            plan,
            self.lowering
                .lowering
                .execution
                .as_ref()
                .map(|cursor| cursor.instance),
            budget,
        )?;
        Ok(FunctionFrameV1 {
            lowering: self.lowering.attach(services)?,
            assembly: self.assembly,
            body: self.body,
            control: self.control,
        })
    }
}

struct FunctionFrameOutputContextV1<'source> {
    function: &'source SemanticFunctionDeclV1,
    semantic_ssa: &'source ProductionSemanticSsaFunctionPlanV1,
    retained_header: bool,
    source_call_instance: Option<ProductionCallInstanceIdV1>,
    backing: Option<SourceFunctionBackingViewV29<'source>>,
    initialization_subject: Option<ScopedInitializationSubjectV29>,
    instance_assert_origins: Option<InstanceAssertCaptureV1>,
    has_runtime_assert: bool,
    invocation_entry: Option<InvocationEntryRelationV1>,
    parameter_bindings: Vec<SemanticKirParameterBindingV1>,
}

struct PreparedFunctionFrameV1<'source, 'service> {
    frame: FunctionFrameV1<'source, 'service, 'source>,
    context: FunctionFrameOutputContextV1<'source>,
}

// All mutable service borrows have ended before this source-owned artifact is
// handed back to the caller for assembly with the original concrete ledger.
struct UnassembledFunctionFrameV1<'source> {
    assembly: EmissionReadOnlyV1<'source, LoweredFunctionPlanV1>,
    context: FunctionFrameOutputContextV1<'source>,
    result_types: EmissionReadOnlyV1<'source, Vec<Type>>,
    infallible_asserts: InfallibleAssertDecisionsV1<'source>,
    target_blocks: Vec<BasicBlock>,
    blocks: Vec<SemanticKirBlockCorrespondenceV1>,
    statement_operation_spans: Vec<SemanticKirStatementOperationSpanV1>,
    terminator_operation_spans: Vec<SemanticKirTerminatorOperationSpanV1>,
    synthetic_operation_spans: Vec<SemanticKirSyntheticOperationSpanV1>,
    retained_local_slots: BTreeMap<ScopedAllocationIdentityV29, SemanticRetainedLocalSlotV1>,
    initialized_at_entry: BTreeMap<u32, BTreeSet<u32>>,
    generated_terminator_values: Vec<SemanticKirGeneratedTerminatorValuesV1>,
    call_returns: CallReturnBufferV1,
    private_arrays: PrivateArrayFunctionRowsV1,
    scoped_memory_anchors: Option<ScopedMemoryAnchorsV29>,
    lifecycle_events: Option<PendingLifecycleEventsV29>,
    emitted_operations: usize,
    next_value: u32,
    execution_observation: Option<ExecutionArchiveV29>,
    direct_call_inputs: Option<Vec<InvocationInputRowV1>>,
}

#[allow(clippy::too_many_arguments)]
fn prepare_function_frame_v1<'source, 'service>(
    semantic: &'source fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    plan: FunctionEmissionPlanV1<'source>,
    semantic_ssa: &'source ProductionSemanticSsaFunctionPlanV1,
    defined_function_ids: &'source BTreeMap<SemanticFunctionIdV1, FunctionId>,
    defined_function_signatures: impl Into<ExecutionSignatureSourceV29<'source>>,
    required_workgroup: Option<[u32; 3]>,
    infallible_asserts: InfallibleAssertDecisionsV1<'source>,
    launch_rank: u8,
    authenticated_ranked_control: bool,
    max_operations: usize,
    has_assert_observer: bool,
    private_array_work: &'service mut PrivateArrayLazyBudgetV1,
    private_array_sources: Option<PrivateArraySourcesV1<'source>>,
    call_budget: &'service mut ArgumentBudgetV1<'_>,
    placement: SemanticEmissionPlacementV1,
    execution: Option<ExecutionAvailabilityV29<'source>>,
    execution_calls: Option<&'service mut (dyn ExecutionDefinedCallConsumerV29 + 'service)>,
    lifecycle: Option<&'service mut (dyn ExecutionLifecycleConsumerV29 + 'service)>,
) -> Result<PreparedFunctionFrameV1<'source, 'service>, ProductionSemanticKirErrorV1> {
    let defined_function_signatures = defined_function_signatures.into();
    let FunctionEmissionPlanPartsV1 {
        assembly,
        result_types,
    } = plan.split();
    let plan = &*assembly;
    let result_count = result_types.len();
    infallible_asserts.require_source(semantic, plan)?;
    // Diagnostic cursors lack the scoped output owner that retains this charge.
    let retained_header = execution.is_some() && execution_calls.is_some() && lifecycle.is_some();
    let source_call_instance = execution.as_ref().map(|cursor| cursor.instance);
    let initialization_subject = execution
        .as_ref()
        .map(ScopedInitializationSubjectV29::from_cursor);
    if initialization_subject.is_some_and(|subject| {
        subject.source.root != plan.correspondence_owner
            || subject.function != plan.semantic_function
    }) {
        return Err(scoped_initialization_error_v29());
    }
    if source_call_instance.is_some() && has_assert_observer {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    let instance_assert_origins = execution
        .as_ref()
        .map(|cursor| InstanceAssertCaptureV1::new(cursor, placement));
    let function = semantic
        .functions()
        .get(plan.semantic_function.index() as usize)
        .ok_or_else(|| {
            unsupported(
                plan.semantic_function.index(),
                None,
                None,
                "lowered semantic function is missing",
            )
        })?;
    let has_runtime_assert = semantic_requires_runtime_assert_failure(
        function,
        semantic.callables(),
        &infallible_asserts,
    );
    let invocation_scope = OwnedInvocationEntryPlanV1::new(
        function,
        semantic_ssa,
        placement,
        has_runtime_assert,
        execution.as_ref(),
        call_budget,
    )?;
    let invocation = invocation_scope.plan();
    if invocation.layout.preheader.is_some() && execution.is_none() {
        return Err(unsupported(
            plan.semantic_function.index(),
            Some(function.entry().index()),
            None,
            "legacy source correspondence cannot represent an invocation-entry relation",
        ));
    }
    let statement_count = function
        .blocks()
        .iter()
        .try_fold(0_usize, |count, block| {
            count.checked_add(block.statements().len())
        })
        .ok_or(ProductionSemanticKirErrorV1::ResourceLimit {
            resource: ProductionSemanticKirResourceV1::Statements,
            actual: usize::MAX,
            limit: usize::MAX,
        })?;
    let mut parameter_bindings = Vec::new();
    parameter_bindings
        .try_reserve_exact(plan.parameter_declarations.len())
        .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
            resource: ProductionSemanticKirResourceV1::DebugBindings,
        })?;
    parameter_bindings.extend(plan.parameter_local_bindings.iter().filter_map(|binding| {
        match binding {
            PlannedParameterLocalBindingV1::Direct { local, value, .. } => {
                Some(SemanticKirParameterBindingV1 {
                    correspondence_owner: plan.correspondence_owner,
                    semantic_function: plan.semantic_function,
                    semantic_local: SemanticLocalIdV1::from_index(*local as u32),
                    kernel_ir_value: *value,
                })
            }
            PlannedParameterLocalBindingV1::Flattened { .. }
            | PlannedParameterLocalBindingV1::Bf16Nominal { .. }
            | PlannedParameterLocalBindingV1::Wave64Nominal { .. } => None,
        }
    }));
    let failure_block = has_runtime_assert
        .then(|| placement.block(function.blocks().len() as u32))
        .transpose()?;
    let call_returns = defined_function_signatures.call_return_buffer_v29(
        execution.as_ref(),
        function,
        semantic.callables(),
        result_count,
        call_budget,
    )?;
    let mut lowering = SemanticFunctionLoweringV1::new_interprocedural(
        semantic.types(),
        semantic.callables(),
        function,
        semantic_ssa,
        plan.correspondence_owner,
        plan.semantic_function,
        defined_function_ids,
        defined_function_signatures,
        result_types,
        SemanticParameterBindingsV1 {
            declarations: &plan.parameter_declarations,
            values: &plan.parameter_values,
            types: &plan.parameter_types,
            local_bindings: Some(&plan.parameter_local_bindings),
        },
        failure_block,
        required_workgroup,
        infallible_asserts,
        launch_rank,
        authenticated_ranked_control,
        max_operations,
        PrivateArrayRecorderWorkV1::Shared(private_array_work),
        private_array_sources,
        call_returns,
        Some(call_budget),
        placement,
        execution,
        match lifecycle {
            Some(consumer) => Some(&mut *consumer as &mut (dyn ExecutionLifecycleConsumerV29 + '_)),
            None => None,
        },
        None,
)?;
    lowering.execution_calls = match execution_calls {
        Some(consumer) => Some(&mut *consumer as &mut (dyn ExecutionDefinedCallConsumerV29 + '_)),
        None => None,
    };
    lowering.invocation_preheader = invocation.layout.preheader;

    let order = semantic_ssa
        .plan()
        .reverse_postorder()
        .iter()
        .map(|block| SemanticBlockIdV1::from_index(block.get()))
        .collect::<Vec<_>>();
    let target_block_capacity = argument_sum_v1(&[
        order.len(),
        usize::from(has_runtime_assert),
        usize::from(invocation.layout.preheader.is_some()),
    ])?;
    if invocation.layout.preheader.is_some() {
        lowering.with_emission_budget_v1(|_, budget| {
            budget.reserve_storage(std::mem::size_of::<BasicBlock>())
        })?;
    }
    let mut target_blocks = Vec::new();
    target_blocks
        .try_reserve_exact(target_block_capacity)
        .map_err(|_| ProductionSemanticKirErrorV1::AllocationFailure {
            resource: ProductionSemanticKirResourceV1::Blocks,
        })?;
    if invocation.layout.preheader.is_some() {
        lowering.with_emission_budget_v1(|_, budget| {
            budget.reserve_storage(argument_product_v1(
                target_blocks
                    .capacity()
                    .saturating_sub(target_block_capacity),
                std::mem::size_of::<BasicBlock>(),
            )?)
        })?;
    }
    let blocks = Vec::with_capacity(order.len());
    let statement_operation_spans = Vec::with_capacity(statement_count);
    let terminator_operation_spans = Vec::with_capacity(order.len());
    let mut synthetic_operation_spans = Vec::with_capacity(usize::from(has_runtime_assert));
    let invocation_entry = if let Some(preheader) = invocation.layout.preheader {
        let mut target = BasicBlock::new(preheader);
        let prologue = lowering.begin_invocation_v1(invocation, &mut target)?;
        if prologue.retained_local_storage != 0 {
            synthetic_operation_spans.push(SemanticKirSyntheticOperationSpanV1 {
                correspondence_owner: plan.correspondence_owner,
                semantic_function: plan.semantic_function,
                rule: SemanticKirSyntheticOperationRuleV1::RetainedLocalStorage,
                kernel_ir_block: target.id,
                first_operation_ordinal: 0,
                operation_count: prologue.retained_local_storage,
            });
        }
        if prologue.enum_payload_storage != 0 {
            synthetic_operation_spans.push(SemanticKirSyntheticOperationSpanV1 {
                correspondence_owner: plan.correspondence_owner,
                semantic_function: plan.semantic_function,
                rule: SemanticKirSyntheticOperationRuleV1::EnumPayloadStorage,
                kernel_ir_block: target.id,
                first_operation_ordinal: prologue.retained_local_storage,
                operation_count: prologue.enum_payload_storage,
            });
        }
        let relation =
            lowering.emit_invocation_edge_v1(invocation, initialization_subject, &mut target)?;
        target_blocks.push(target);
        Some(relation)
    } else {
        None
    };
    lowering.with_emission_budget_v1(|_, budget| invocation_scope.settle(budget))?;

    let frame = FunctionFrameV1::new_with_assembly(
        lowering,
        assembly,
        FunctionFrameBodyV1 {
            order,
            target_blocks,
            blocks,
            statements: statement_operation_spans,
            terminators: terminator_operation_spans,
            synthetic: synthetic_operation_spans,
        },
    )?;
    let backing = frame.control.backing;
    Ok(PreparedFunctionFrameV1 {
        frame,
        context: FunctionFrameOutputContextV1 {
            function,
            semantic_ssa,
            retained_header,
            source_call_instance,
            backing,
            initialization_subject,
            instance_assert_origins,
            has_runtime_assert,
            invocation_entry,
            parameter_bindings,
        },
    })
}

impl<'source> PreparedFunctionFrameV1<'source, '_> {
    fn finish_lowering(
        self,
        assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    ) -> Result<UnassembledFunctionFrameV1<'source>, ProductionSemanticKirErrorV1> {
        let (mut lowering, assembly, body) = self.frame.finish_parts()?;
        let plan = &*assembly;
        let FunctionFrameBodyV1 {
            order,
            mut target_blocks,
            blocks,
            statements: statement_operation_spans,
            terminators: terminator_operation_spans,
            synthetic: mut synthetic_operation_spans,
        } = body;
        drop(order);
        lowering.with_emission_budget_v1(|this, budget| {
            let Some(cursor) = this.execution.as_ref() else {
                return require_semantic_ssa_definitions_consumed_v1(
                    this.semantic_function.index(),
                    &this.pending_semantic_ssa_definitions,
                );
            };
            for ((block, _), definitions) in &this.pending_semantic_ssa_definitions {
                budget.charge_work(1)?;
                if !definitions.is_empty()
                    && cursor
                        .source_block_reachable_v29(SemanticBlockIdV1::from_index(*block), budget)?
                {
                    return Err(unsupported(
                        this.semantic_function.index(),
                        Some(*block),
                        None,
                        "semantic SSA definition plan was not fully consumed",
                    ));
                }
            }
            Ok(())
        })?;
        if lowering.execution.is_some() {
            lowering.with_emission_budget_v1(|this, budget| {
                this.execution.as_mut().unwrap().finish(budget)
            })?;
        }
        if let Some(failure_block) = lowering.assert_failure_block {
            let mut block = BasicBlock::new(failure_block);
            let first = block.operations.len();
            lowering.push_operation(&mut block.operations, || {
                AmdGpuDiagnosticOperation::Trap.operation(None)
            })?;
            let (first_operation_ordinal, operation_count) =
                measured_operation_span(first, block.operations.len(), failure_block, None)?;
            synthetic_operation_spans.push(SemanticKirSyntheticOperationSpanV1 {
                correspondence_owner: plan.correspondence_owner,
                semantic_function: plan.semantic_function,
                rule: SemanticKirSyntheticOperationRuleV1::RuntimeAssertFailureTrap,
                kernel_ir_block: failure_block,
                first_operation_ordinal,
                operation_count,
            });
            block.terminator = Some(Terminator::Unreachable);
            target_blocks.push(block);
        }
        lowering.finish_scoped_payloads_v29(&target_blocks)?;
        let lifecycle_events = lowering.take_execution_lifecycle_events_v29()?;
        let scoped_memory_anchors = lowering
            .scoped_memory
            .take()
            .map(|recorder| {
                if recorder.frame.is_some() {
                    return Err(scoped_memory_error_v29());
                }
                Ok(recorder.anchors)
            })
            .transpose()?;
        let emitted_operations = lowering.emitted_operations;
        let next_value = lowering.next_value;
        if let Some(origins) = assert_origins
            && let Some(capture) = origins.scalar_capture.as_mut()
        {
            capture
                .record_function(plan, &lowering, &target_blocks, origins.budget)
                .map_err(|error| match error {
                    scalar_ssa_emission_v1::ProductionScalarSsaEmissionErrorV1::Resource(error) => {
                        ProductionSemanticKirErrorV1::from(error)
                    }
                    _ => ProductionSemanticKirErrorV1::CorrespondenceMismatch,
                })?;
        }
        let direct_call_inputs = lowering
            .with_emission_budget_v1(|this, budget| this.take_direct_call_inputs_v26(budget))?;
        let execution_observation = lowering.with_emission_budget_v1(|this, budget| {
            this.take_execution_archive_v29(&target_blocks, &synthetic_operation_spans, budget)
        })?;
        let infallible_asserts = lowering.infallible_asserts;
        let result_types = lowering.result_types;
        drop(lowering.emission_work.take());
        let retained_local_slots = lowering.retained_local_slots;
        let initialized_at_entry = lowering.control_flow_ssa.retained_initialized_at_entry;
        let generated_terminator_values = lowering.generated_terminator_values;
        let call_returns = lowering.call_returns;
        let private_arrays = lowering.private_arrays.into_rows()?;

        Ok(UnassembledFunctionFrameV1 {
            assembly,
            context: self.context,
            result_types,
            infallible_asserts,
            target_blocks,
            blocks,
            statement_operation_spans,
            terminator_operation_spans,
            synthetic_operation_spans,
            retained_local_slots,
            initialized_at_entry,
            generated_terminator_values,
            call_returns,
            private_arrays,
            scoped_memory_anchors,
            lifecycle_events,
            emitted_operations,
            next_value,
            execution_observation,
            direct_call_inputs,
        })
    }
}

impl UnassembledFunctionFrameV1<'_> {
    fn assemble(
        self,
        call_budget: &mut ArgumentBudgetV1<'_>,
    ) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
        let Self {
            assembly,
            mut context,
            result_types,
            infallible_asserts,
            target_blocks,
            blocks,
            statement_operation_spans,
            terminator_operation_spans,
            synthetic_operation_spans,
            retained_local_slots,
            initialized_at_entry,
            generated_terminator_values,
            mut call_returns,
            private_arrays,
            scoped_memory_anchors,
            lifecycle_events,
            emitted_operations,
            next_value,
            execution_observation,
            direct_call_inputs,
        } = self;
        let plan = &*assembly;
        let function = context.function;
        let scoped_initialization = context
            .initialization_subject
            .map(|subject| {
                capture_scoped_initialization_v29(
                    subject,
                    function,
                    context.semantic_ssa,
                    &retained_local_slots,
                    &initialized_at_entry,
                    call_budget,
                )
            })
            .transpose()?;
        let scoped_slot_origins = context
            .source_call_instance
            .map(|_| {
                capture_scoped_slot_origins_v29(&retained_local_slots, context.backing, call_budget)
            })
            .transpose()?;
        drop(retained_local_slots);
        // The shared emitter's final blocks are still untouched. Capture with the
        // same now-reborrowable ledger, before any helper expansion changes placement.
        if let Some(origins) = &mut context.instance_assert_origins {
            call_budget.charge_work(terminator_operation_spans.len())?;
            for (span, target) in terminator_operation_spans.iter().zip(
                target_blocks
                    .iter()
                    .skip(usize::from(context.invocation_entry.is_some())),
            ) {
                let source = function
                    .blocks()
                    .get(span.semantic_block.index() as usize)
                    .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
                if target.id != span.kernel_ir_block {
                    return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
                }
                origins.record(
                    *span,
                    &plan.kernel_ir_function,
                    source.terminator().kind(),
                    target
                        .terminator
                        .as_ref()
                        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?,
                    infallible_asserts.contains(&span.semantic_block.index()),
                    call_budget,
                )?;
            }
        }
        call_returns.order_blocks(call_budget)?;
        let operation_capabilities = target_blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .flat_map(Operation::required_capabilities)
            .collect::<BTreeSet<_>>();
        let diagnostic_declarations = target_blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter_map(|operation| match &operation.kind {
                OperationKind::Call { callee, arguments } => {
                    AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                }
                _ => None,
            })
            .map(|operation| {
                let declaration = operation.declaration();
                (declaration.id.clone(), declaration)
            })
            .collect::<BTreeMap<_, _>>();
        let float_declarations = target_blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter_map(|operation| match &operation.kind {
                OperationKind::Call { callee, arguments } => {
                    FloatOperation::from_intrinsic_call(callee, arguments)
                }
                _ => None,
            })
            .map(|operation| {
                let declaration = operation.declaration();
                (declaration.id.clone(), declaration)
            })
            .collect::<BTreeMap<_, _>>();
        let (id, signature, parameters) = if context.retained_header {
            copy_emitted_function_header_v29(
                &plan.kernel_ir_function,
                &plan.parameter_types,
                &result_types,
                &plan.parameter_values,
                call_budget,
            )?
        } else {
            (
                plan.kernel_ir_function.clone(),
                Signature::new(plan.parameter_types.clone(), (*result_types).clone()),
                plan.parameter_values.clone(),
            )
        };
        let mut lowered = match plan.role {
            SemanticKirFunctionRoleV1::KernelEntry => {
                Function::kernel_entry(id, signature, parameters, target_blocks)
            }
            SemanticKirFunctionRoleV1::InternalHelper => {
                Function::internal_helper(id, signature, parameters, target_blocks)
            }
        };
        if context.has_runtime_assert {
            lowered
                .required_capabilities
                .extend(AmdGpuDiagnosticOperation::Trap.required_capabilities());
        }
        lowered
            .required_capabilities
            .extend(operation_capabilities.iter().cloned());
        Ok(LoweredFunctionResultV1 {
            next_value,
            execution_observation,
            direct_call_inputs,
            source_call_instance: context.source_call_instance,
            invocation_entry: context.invocation_entry,
            scoped_slot_origins,
            scoped_initialization,
            scoped_memory_anchors,
            instance_assert_origins: context.instance_assert_origins,
            lifecycle_events,
            private_arrays,
            function: lowered,
            operation_capabilities,
            diagnostic_declarations,
            float_declarations,
            blocks,
            statement_operation_spans,
            terminator_operation_spans,
            generated_terminator_values,
            call_returns,
            synthetic_operation_spans,
            parameter_bindings: context.parameter_bindings,
            parameter_component_bindings: plan.parameter_component_bindings.clone(),
            ignored_parameter_bindings: plan.ignored_parameter_bindings.clone(),
            emitted_operations,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_function_frame_v1<'source, 'facts: 'source>(
    semantic: &'source fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    plan: FunctionEmissionPlanV1<'source>,
    semantic_ssa: &'source ProductionSemanticSsaFunctionPlanV1,
    defined_function_ids: &'source BTreeMap<SemanticFunctionIdV1, FunctionId>,
    defined_function_signatures: impl Into<ExecutionSignatureSourceV29<'source>>,
    required_workgroup: Option<[u32; 3]>,
    infallible_asserts: impl Into<InfallibleAssertDecisionsV1<'facts>>,
    launch_rank: u8,
    authenticated_ranked_control: bool,
    max_operations: usize,
    mut assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    private_array_work: &mut PrivateArrayLazyBudgetV1,
    private_array_sources: Option<PrivateArraySourcesV1<'source>>,
    call_budget: &mut ArgumentBudgetV1<'_>,
    placement: SemanticEmissionPlacementV1,
    execution: Option<ExecutionAvailabilityV29<'source>>,
    execution_calls: Option<&mut dyn ExecutionDefinedCallConsumerV29>,
    lifecycle: Option<&mut dyn ExecutionLifecycleConsumerV29>,
    detach_blocks: bool,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    let plan_storage = FunctionPlanStorageV1::new(execution.as_ref(), call_budget)?;
    let outcome: FunctionEmissionOutcomeV1 =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut prepared = prepare_function_frame_v1(
                semantic,
                plan,
                semantic_ssa,
                defined_function_ids,
                defined_function_signatures,
                required_workgroup,
                infallible_asserts.into(),
                launch_rank,
                authenticated_ranked_control,
                max_operations,
                assert_origins.is_some(),
                private_array_work,
                private_array_sources,
                call_budget,
                placement,
                execution,
                match execution_calls {
                    Some(consumer) => {
                        Some(&mut *consumer as &mut (dyn ExecutionDefinedCallConsumerV29 + '_))
                    }
                    None => None,
                },
                match lifecycle {
                    Some(consumer) => {
                        Some(&mut *consumer as &mut (dyn ExecutionLifecycleConsumerV29 + '_))
                    }
                    None => None,
                },
            )?;
            while prepared.frame.step(assert_origins.as_deref_mut())? {
                if detach_blocks {
                    let (detached, services) = prepared.frame.detach()?;
                    prepared.frame = detached.attach(services)?;
                }
            }
            let unassembled = prepared.finish_lowering(assert_origins.as_deref_mut())?;
            unassembled.assemble(call_budget)
        }));
    match outcome {
        Ok(Ok(value)) => {
            plan_storage.finish(call_budget)?;
            Ok(value)
        }
        Ok(Err(error)) => {
            plan_storage.abandon(call_budget);
            Err(error)
        }
        Err(payload) => {
            plan_storage.abandon(call_budget);
            std::panic::resume_unwind(payload)
        }
    }
}
