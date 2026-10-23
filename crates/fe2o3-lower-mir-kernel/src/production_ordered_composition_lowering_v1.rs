// Legacy signatures preserve their exact None/default route. Only the private
// composition caller supplies the source-validated additive permit.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn lower_one_semantic_function_v1<'facts>(
    semantic: &fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    plan: &LoweredFunctionPlanV1,
    semantic_ssa: &ProductionSemanticSsaFunctionPlanV1,
    defined_function_ids: &BTreeMap<SemanticFunctionIdV1, FunctionId>,
    defined_function_signatures: &BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>,
    required_workgroup: Option<[u32; 3]>,
    infallible_asserts: impl Into<InfallibleAssertDecisionsV1<'facts>>,
    launch_rank: u8,
    authenticated_ranked_control: bool,
    max_operations: usize,
    assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    private_array_work: &mut PrivateArrayLazyBudgetV1,
    private_array_sources: Option<(&PrivateArrayMergeV1, Option<&PrivateArrayMergeV1>)>,
    call_budget: &mut ArgumentBudgetV1<'_>,
    placement: SemanticEmissionPlacementV1,
    execution: Option<ExecutionAvailabilityV29<'_>>,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    lower_one_semantic_function_for_composition_v1(
        semantic,
        plan,
        semantic_ssa,
        defined_function_ids,
        defined_function_signatures,
        required_workgroup,
        infallible_asserts,
        launch_rank,
        authenticated_ranked_control,
        max_operations,
        assert_origins,
        private_array_work,
        private_array_sources,
        call_budget,
        placement,
        execution,
        None,
        None,
        None,
)
}

#[allow(clippy::too_many_arguments)]
fn lower_one_semantic_function_with_calls_v29<'facts>(
    semantic: &fe2o3_mir_model::semantic_mir_v1::AdmittedInertSemanticMirV1,
    plan: &LoweredFunctionPlanV1,
    semantic_ssa: &ProductionSemanticSsaFunctionPlanV1,
    defined_function_ids: &BTreeMap<SemanticFunctionIdV1, FunctionId>,
    defined_function_signatures: &BTreeMap<SemanticFunctionIdV1, LoweredFunctionSignatureV1>,
    required_workgroup: Option<[u32; 3]>,
    infallible_asserts: impl Into<InfallibleAssertDecisionsV1<'facts>>,
    launch_rank: u8,
    authenticated_ranked_control: bool,
    max_operations: usize,
    assert_origins: Option<&mut AssertOriginEmissionV1<'_, '_>>,
    private_array_work: &mut PrivateArrayLazyBudgetV1,
    private_array_sources: Option<PrivateArraySourcesV1<'_>>,
    call_budget: &mut ArgumentBudgetV1<'_>,
    placement: SemanticEmissionPlacementV1,
    execution: Option<ExecutionAvailabilityV29<'_>>,
    execution_calls: Option<&mut dyn ExecutionDefinedCallConsumerV29>,
    lifecycle: Option<&mut dyn ExecutionLifecycleConsumerV29>,
) -> Result<LoweredFunctionResultV1, ProductionSemanticKirErrorV1> {
    lower_one_semantic_function_with_composition_v1(
        semantic,
        plan,
        semantic_ssa,
        defined_function_ids,
        defined_function_signatures,
        required_workgroup,
        infallible_asserts,
        launch_rank,
        authenticated_ranked_control,
        max_operations,
        assert_origins,
        private_array_work,
        private_array_sources,
        call_budget,
        placement,
        execution,
        execution_calls,
        lifecycle,
        None,
        None,
        None,
)
}

fn propagate_ordered_composition_capabilities_v1(
    module: &mut Module,
    semantic: &AdmittedInertSemanticMirV1,
    permit: OrderedCompositionPermitV1,
    symbol: &str,
    budget: &mut ArgumentBudgetV1<'_>,
) -> Result<(), ProductionSemanticKirErrorV1> {
    budget.charge_work(256)?;
    if !permit.matches(semantic)
        || module.functions.len() > 3
        || module.required_capabilities.len() > 4
    {
        return Err(ProductionSemanticKirErrorV1::CorrespondenceMismatch);
    }
    // These are ordinary lowering's existing capability allocations, bounded by
    // the source operation limits. Only actual function-operation requirements
    // are propagated; this neither authenticates source nor changes effects.
    let entry = module
        .functions
        .iter_mut()
        .find(|f| f.id.as_str() == symbol)
        .filter(|f| f.role == fe2o3_kernel_ir::FunctionRole::KernelEntry)
        .ok_or(ProductionSemanticKirErrorV1::CorrespondenceMismatch)?;
    entry
        .required_capabilities
        .extend(module.required_capabilities.iter().cloned());
    Ok(())
}
