use crate::{AnalysisReport, Diagnostic, UnsupportedReason, Variation};
use fe2o3_kernel_ir::{
    AddressSpace, AmdGpuDiagnosticOperation, Axis, BasicBlock, BinaryOp, BlockId, CastKind,
    CheckedBinaryOperator, ComparePredicate, Constant, FloatOperation, Function, FunctionBody,
    FunctionId, FunctionRole, IndexKind, IntrinsicKind, MAX_INTERPROCEDURAL_EFFECT_CALL_EDGES_V1,
    MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1, Module, Operation, OperationKind, ScalarType,
    Terminator, Type, UnaryOp, ValueId, WaveOperationKind, WorkgroupSize,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

mod contextual_control;
mod physical_context_v2;
pub use physical_context_v2::{UniformityPhysicalLaunchErrorV2, UniformityPhysicalLaunchV2};

/// Conservatively classifies SSA values and barrier control in one function.
///
/// The caller should run the kernel IR verifier first. This function still
/// fails closed for malformed input: unknown values become [`Variation::Varying`]
/// and produce diagnostics. Function parameters are varying because kernel IR
/// v1 has no uniform-argument metadata. Loads and atomic results are varying
/// because it has no immutable-region or inter-thread value summaries. Calls
/// are unsupported unless they are closed compiler intrinsics or reachable
/// helpers whose bodies can be summarized as pure.
/// Postdominance is established only for CFG regions with a path to an exit.
/// Divergent loop-exiting control remains active beyond the loop, while
/// branches that reconverge within the same natural-loop nest use their normal
/// postdominator. This pass checks barrier reachability only; compatible order
/// among distinct dynamic barrier instances remains an unsupported obligation.
///
/// The returned report is analysis evidence only and grants no assurance.
pub fn analyze_function(function: &Function) -> AnalysisReport {
    analyze_function_with_contract(
        function,
        &[],
        &BTreeSet::new(),
        &BTreeSet::new(),
        None,
        None,
    )
}

/// Analyzes an actual module-owned helper without assuming uniform parameters.
///
/// Every formal parameter of this helper is varying. Existing context-free,
/// argument-sensitive summaries may classify calls to pure helpers in `module`;
/// unknown, recursive, and context-dependent callees remain unsupported.
/// The report contains the real helper value and block-control facts. It has no
/// kernel launch, workgroup-size, physical-participation, or entry-call context.
/// A caller must separately prove any convergence obligation at the call site.
///
/// Returns `None` for foreign, ambiguous, non-helper, or kernel-aliased owners.
/// As with the other analyses, the caller should first verify the module. This
/// query grants no helper-admission, execution, or assurance authority.
pub fn analyze_module_helper_v1(module: &Module, function: &Function) -> Option<AnalysisReport> {
    if function.role != FunctionRole::InternalHelper
        || module.functions.len() > MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1
        || module
            .kernels
            .iter()
            .any(|kernel| kernel.entry == function.id)
    {
        return None;
    }
    let mut matches = module
        .functions
        .iter()
        .filter(|candidate| candidate.id == function.id);
    if !matches
        .next()
        .is_some_and(|member| std::ptr::eq(member, function))
        || matches.next().is_some()
    {
        return None;
    }
    let (summarized_calls, uniform_input_calls) = summarize_uniform_helpers(module, function);
    let parameters = vec![Variation::Varying; function.signature.parameters.len()];
    Some(analyze_function_with_contract(
        function,
        &parameters,
        &summarized_calls,
        &uniform_input_calls,
        None,
        None,
    ))
}

/// Classifies one kernel entry using uniform ABI parameters and conservative
/// summaries for reachable pure helpers.
pub fn analyze_kernel_entry(module: &Module, function: &Function) -> AnalysisReport {
    analyze_kernel_entry_with_physical_context_v2(module, function, None)
}

fn analyze_kernel_entry_with_physical_context_v2(
    module: &Module,
    function: &Function,
    physical: Option<&UniformityPhysicalLaunchV2<'_>>,
) -> AnalysisReport {
    let (summarized_calls, uniform_input_calls) = summarize_uniform_helpers(module, function);
    let parameters = vec![Variation::GridUniform; function.signature.parameters.len()];
    let mut matching_contracts = module
        .kernels
        .iter()
        .filter(|kernel| kernel.entry == function.id)
        .map(|kernel| kernel.workgroup_size);
    let workgroup_size = matching_contracts
        .next()
        .filter(|first| matching_contracts.all(|contract| contract == *first))
        .flatten();
    analyze_function_with_physical_contract_v2(
        function,
        &parameters,
        &summarized_calls,
        &uniform_input_calls,
        workgroup_size,
        contextual_control::exact_d1_workgroup(module, function),
        physical,
    )
}

fn analyze_function_with_contract(
    function: &Function,
    parameter_variations: &[Variation],
    summarized_calls: &BTreeSet<fe2o3_kernel_ir::FunctionId>,
    uniform_input_calls: &BTreeSet<fe2o3_kernel_ir::FunctionId>,
    workgroup_size: Option<WorkgroupSize>,
    exact_d1_workgroup: Option<u32>,
) -> AnalysisReport {
    analyze_function_with_physical_contract_v2(
        function,
        parameter_variations,
        summarized_calls,
        uniform_input_calls,
        workgroup_size,
        exact_d1_workgroup,
        None,
    )
}

fn analyze_function_with_physical_contract_v2(
    function: &Function,
    parameter_variations: &[Variation],
    summarized_calls: &BTreeSet<fe2o3_kernel_ir::FunctionId>,
    uniform_input_calls: &BTreeSet<fe2o3_kernel_ir::FunctionId>,
    workgroup_size: Option<WorkgroupSize>,
    exact_d1_workgroup: Option<u32>,
    physical: Option<&UniformityPhysicalLaunchV2<'_>>,
) -> AnalysisReport {
    let Some(body) = &function.body else {
        let mut report = AnalysisReport {
            function: function.id.clone(),
            values: BTreeMap::new(),
            block_controls: BTreeMap::new(),
            diagnostics: Vec::new(),
        };
        report.diagnostics.push(Diagnostic::Unsupported {
            block: None,
            operation_index: None,
            reason: UnsupportedReason::FunctionDeclaration,
        });
        return report;
    };

    Analyzer::new_with_physical_context_v2(
        function,
        body,
        parameter_variations,
        summarized_calls,
        uniform_input_calls,
        workgroup_size,
        exact_d1_workgroup,
        physical,
    )
    .run()
}

#[derive(Debug, Default)]
struct UniformHelperCandidate {
    callees: BTreeSet<FunctionId>,
    call_edges: usize,
    requires_uniform_inputs: bool,
    structurally_supported: bool,
}

/// Finds context-free helper calls whose results cannot be more varying than
/// their actual arguments. Collection and evaluation are iterative so hostile
/// call depth cannot consume the host stack.
fn summarize_uniform_helpers(
    module: &Module,
    entry: &Function,
) -> (BTreeSet<FunctionId>, BTreeSet<FunctionId>) {
    let mut summarized = BTreeSet::new();
    let mut uniform_inputs_required = BTreeSet::new();
    let mut pending = BTreeSet::new();
    let mut call_edges = 0usize;
    let Some(entry_body) = &entry.body else {
        return (summarized, uniform_inputs_required);
    };
    collect_entry_calls(entry_body, &mut pending, &mut summarized, &mut call_edges);
    if module.functions.len() > MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1
        || call_edges > MAX_INTERPROCEDURAL_EFFECT_CALL_EDGES_V1
    {
        return (summarized, uniform_inputs_required);
    }

    let mut candidates = BTreeMap::<FunctionId, UniformHelperCandidate>::new();
    while let Some(function_id) = pending.pop_first() {
        if candidates.contains_key(&function_id) {
            continue;
        }
        if candidates.len() == MAX_INTERPROCEDURAL_EFFECT_FUNCTIONS_V1 {
            return (summarized, uniform_inputs_required);
        }
        let candidate = collect_uniform_helper_candidate(module, &function_id, &mut summarized);
        call_edges = call_edges.saturating_add(candidate.call_edges);
        if call_edges > MAX_INTERPROCEDURAL_EFFECT_CALL_EDGES_V1 {
            return (summarized, uniform_inputs_required);
        }
        pending.extend(candidate.callees.iter().cloned());
        candidates.insert(function_id, candidate);
    }

    let mut callers = candidates
        .keys()
        .cloned()
        .map(|function| (function, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    let mut remaining_callees = BTreeMap::new();
    for (caller, candidate) in &candidates {
        remaining_callees.insert(caller.clone(), candidate.callees.len());
        for callee in &candidate.callees {
            if let Some(dependents) = callers.get_mut(callee) {
                dependents.insert(caller.clone());
            }
        }
    }
    let mut ready = remaining_callees
        .iter()
        .filter_map(|(function, remaining)| (*remaining == 0).then_some(function.clone()))
        .collect::<BTreeSet<_>>();

    while let Some(function_id) = ready.pop_first() {
        let candidate = &candidates[&function_id];
        if candidate.structurally_supported
            && candidate
                .callees
                .iter()
                .all(|callee| summarized.contains(callee))
            && uniform_helper_returns_are_proven(
                module,
                &function_id,
                &summarized,
                &uniform_inputs_required,
            )
        {
            summarized.insert(function_id.clone());
            if candidate.requires_uniform_inputs
                || candidate
                    .callees
                    .iter()
                    .any(|callee| uniform_inputs_required.contains(callee))
            {
                uniform_inputs_required.insert(function_id.clone());
            }
        }
        for caller in &callers[&function_id] {
            let remaining = remaining_callees
                .get_mut(caller)
                .expect("candidate caller has dependency accounting");
            *remaining -= 1;
            if *remaining == 0 {
                ready.insert(caller.clone());
            }
        }
    }
    (summarized, uniform_inputs_required)
}

fn collect_entry_calls(
    body: &FunctionBody,
    pending: &mut BTreeSet<FunctionId>,
    summarized: &mut BTreeSet<FunctionId>,
    call_edges: &mut usize,
) {
    for operation in body.blocks.iter().flat_map(|block| &block.operations) {
        let OperationKind::Call { callee, arguments } = &operation.kind else {
            continue;
        };
        *call_edges = call_edges.saturating_add(1);
        if FloatOperation::from_intrinsic_call(callee, arguments).is_some() {
            summarized.insert(callee.clone());
        } else if AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments).is_none() {
            pending.insert(callee.clone());
        }
    }
}

fn collect_uniform_helper_candidate(
    module: &Module,
    function_id: &FunctionId,
    summarized: &mut BTreeSet<FunctionId>,
) -> UniformHelperCandidate {
    let mut matches = module
        .functions
        .iter()
        .filter(|function| function.id == *function_id);
    let Some(function) = matches.next() else {
        return UniformHelperCandidate::default();
    };
    if matches.next().is_some() || function.role != FunctionRole::InternalHelper {
        return UniformHelperCandidate::default();
    }
    let Some(body) = &function.body else {
        return UniformHelperCandidate::default();
    };

    let mut candidate = UniformHelperCandidate {
        callees: BTreeSet::new(),
        call_edges: 0,
        requires_uniform_inputs: false,
        structurally_supported: true,
    };
    for block in &body.blocks {
        if block.terminator.is_none()
            || matches!(block.terminator, Some(Terminator::Unreachable))
                && !block.operations.last().is_some_and(|operation| {
                    let OperationKind::Call { callee, arguments } = &operation.kind else {
                        return false;
                    };
                    AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                        .is_some_and(|diagnostic| diagnostic.is_terminating())
                })
        {
            candidate.structurally_supported = false;
        }
        for operation in &block.operations {
            match &operation.kind {
                OperationKind::Call { callee, arguments }
                    if FloatOperation::from_intrinsic_call(callee, arguments).is_some() =>
                {
                    candidate.call_edges = candidate.call_edges.saturating_add(1);
                    summarized.insert(callee.clone());
                }
                OperationKind::Call { callee, arguments } => {
                    candidate.call_edges = candidate.call_edges.saturating_add(1);
                    if let Some(diagnostic) =
                        AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments)
                    {
                        if diagnostic.is_terminating() {
                            candidate.requires_uniform_inputs = true;
                        } else {
                            candidate.structurally_supported = false;
                        }
                    } else {
                        candidate.callees.insert(callee.clone());
                    }
                }
                OperationKind::Intrinsic(intrinsic) => match intrinsic.kind {
                    IntrinsicKind::LaunchExtent { .. }
                    | IntrinsicKind::InvocationIndex {
                        kind: IndexKind::WorkgroupSize | IndexKind::WorkgroupCount,
                        ..
                    } => {}
                    IntrinsicKind::InvocationIndex { .. } => {
                        candidate.structurally_supported = false;
                    }
                },
                OperationKind::Storage(_)
                | OperationKind::Execution(_)
                | OperationKind::VerificationContract(_)
                | OperationKind::Alloca { .. }
                | OperationKind::Atomic(_)
                | OperationKind::Barrier(_)
                | OperationKind::Fence(_)
                | OperationKind::Gfx950LdsTranspose(_)
                | OperationKind::InlineAssembly(_)
                | OperationKind::Matrix(_)
                | OperationKind::Wave(_)
                | OperationKind::WorkgroupBarrier(_)
                | OperationKind::WorkgroupMemory(_) => {
                    candidate.structurally_supported = false;
                }
                _ if !operation.memory_effects().is_empty() => {
                    candidate.structurally_supported = false;
                }
                _ => {}
            }
        }
    }
    candidate
}

fn uniform_helper_returns_are_proven(
    module: &Module,
    function_id: &FunctionId,
    summarized: &BTreeSet<FunctionId>,
    uniform_inputs_required: &BTreeSet<FunctionId>,
) -> bool {
    let Some(function) = module.function(function_id) else {
        return false;
    };
    let parameters = vec![Variation::GridUniform; function.signature.parameters.len()];
    let report = analyze_function_with_contract(
        function,
        &parameters,
        summarized,
        uniform_inputs_required,
        None,
        None,
    );
    if !report.diagnostics().is_empty() {
        return false;
    }
    function.body.as_ref().is_some_and(|body| {
        body.blocks.iter().all(|block| match &block.terminator {
            Some(Terminator::Return { values }) => values
                .iter()
                .all(|value| report.value(*value) == Variation::GridUniform),
            Some(_) => true,
            None => false,
        })
    })
}

struct Analyzer<'a> {
    body: &'a FunctionBody,
    reachable: BTreeSet<BlockId>,
    incoming: BTreeMap<BlockId, Vec<Edge>>,
    control_regions: BTreeMap<BlockId, BTreeSet<BlockId>>,
    control_unknown: BTreeSet<BlockId>,
    trivial_phi_representatives: BTreeMap<ValueId, ValueId>,
    private_load_slots: BTreeMap<ValueId, ValueId>,
    private_slot_stores: BTreeMap<ValueId, Vec<PrivateStore>>,
    uniform_recurrence_edges: BTreeSet<(ValueId, BlockId, ValueId)>,
    proven_no_overflow: BTreeSet<ValueId>,
    parameter_variations: &'a [Variation],
    summarized_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
    uniform_input_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
    workgroup_size: Option<WorkgroupSize>,
    exact_d1_workgroup: Option<u32>,
    value_definitions: BTreeMap<ValueId, &'a Operation>,
    report: AnalysisReport,
}

#[derive(Clone, Debug)]
struct Edge {
    source: BlockId,
    arguments: Vec<ValueId>,
    discriminator: Option<ValueId>,
}

#[derive(Clone, Copy, Debug)]
struct PrivateStore {
    block: BlockId,
    operation_index: usize,
    value: ValueId,
}

impl<'a> Analyzer<'a> {
    #[cfg(test)]
    fn new(
        function: &'a Function,
        body: &'a FunctionBody,
        parameter_variations: &'a [Variation],
        summarized_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
        uniform_input_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
        workgroup_size: Option<WorkgroupSize>,
        exact_d1_workgroup: Option<u32>,
    ) -> Self {
        Self::new_with_physical_context_v2(
            function,
            body,
            parameter_variations,
            summarized_calls,
            uniform_input_calls,
            workgroup_size,
            exact_d1_workgroup,
            None,
        )
    }

    fn new_with_physical_context_v2(
        function: &'a Function,
        body: &'a FunctionBody,
        parameter_variations: &'a [Variation],
        summarized_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
        uniform_input_calls: &'a BTreeSet<fe2o3_kernel_ir::FunctionId>,
        workgroup_size: Option<WorkgroupSize>,
        exact_d1_workgroup: Option<u32>,
        physical: Option<&UniformityPhysicalLaunchV2<'_>>,
    ) -> Self {
        let mut report = AnalysisReport {
            function: function.id.clone(),
            values: BTreeMap::new(),
            block_controls: BTreeMap::new(),
            diagnostics: Vec::new(),
        };
        let mut blocks = BTreeMap::new();
        let mut malformed = false;
        for block in &body.blocks {
            if blocks.insert(block.id, block).is_some() {
                malformed = true;
            }
        }
        if malformed || body.blocks.is_empty() {
            report.diagnostics.push(Diagnostic::Unsupported {
                block: None,
                operation_index: None,
                reason: UnsupportedReason::MalformedControlFlow,
            });
        }

        let (value_types, malformed_values) = index_value_types(function, body);
        if malformed_values {
            report.diagnostics.push(Diagnostic::Unsupported {
                block: None,
                operation_index: None,
                reason: UnsupportedReason::MalformedControlFlow,
            });
        }
        let reachable = reachable_blocks(body, &blocks);
        let (incoming, malformed_edges) = incoming_edges(body, &blocks, &reachable, &value_types);
        if malformed_edges {
            report.diagnostics.push(Diagnostic::Unsupported {
                block: None,
                operation_index: None,
                reason: UnsupportedReason::MalformedControlFlow,
            });
        }
        let mut control_flow_malformed = false;
        let natural_loop_nests = match crate::analyze_control_flow(function) {
            Ok(control_flow) => reachable
                .iter()
                .map(|block| {
                    (
                        *block,
                        control_flow
                            .containing_natural_loops(*block)
                            .unwrap_or_default()
                            .iter()
                            .copied()
                            .collect::<BTreeSet<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
            Err(_) => {
                control_flow_malformed = true;
                report.diagnostics.push(Diagnostic::Unsupported {
                    block: None,
                    operation_index: None,
                    reason: UnsupportedReason::MalformedControlFlow,
                });
                BTreeMap::new()
            }
        };
        let trivial_phi_representatives =
            trivial_phi_representatives(body, &incoming, &value_types);
        let dominators = compute_dominators(body, &reachable, &incoming);
        let uniform_recurrence_edges = uniform_recurrence_edges(
            function,
            body,
            &incoming,
            &dominators,
            &trivial_phi_representatives,
        );
        let (private_load_slots, private_slot_stores) = private_storage_facts(
            body,
            &reachable,
            &dominators,
            &trivial_phi_representatives,
            &value_types,
        );
        let known_integer_values = KnownIntegerValueAnalysis::new(
            body,
            &incoming,
            &private_load_slots,
            &private_slot_stores,
        )
        .solve_terminator_selectors();
        let proven_no_overflow = prove_unsigned_checked_arithmetic(
            function,
            body,
            &reachable,
            &incoming,
            &dominators,
            &trivial_phi_representatives,
            &private_load_slots,
            &private_slot_stores,
        );
        let mut value_definitions = BTreeMap::new();
        for block in &body.blocks {
            for operation in &block.operations {
                for result in &operation.results {
                    value_definitions.insert(result.id, operation);
                }
            }
        }
        let mut effective_successors = effective_successors(body, &known_integer_values);
        if !malformed && !malformed_values && !malformed_edges && !control_flow_malformed {
            if let Some(physical) = physical {
                physical_context_v2::refine_successors(
                    physical,
                    body,
                    &incoming,
                    &dominators,
                    &value_types,
                    &value_definitions,
                    &mut effective_successors,
                );
            } else {
                contextual_control::refine_successors(
                    body,
                    &incoming,
                    &dominators,
                    &value_types,
                    &value_definitions,
                    &mut effective_successors,
                );
            }
        }
        let postdominance_available =
            postdominance_available(&blocks, &reachable, &effective_successors);
        let mut control_unknown = reachable
            .difference(&postdominance_available)
            .copied()
            .collect::<BTreeSet<_>>();
        if control_flow_malformed {
            control_unknown.extend(reachable.iter().copied());
        }
        if !control_unknown.is_empty() {
            report.diagnostics.push(Diagnostic::Unsupported {
                block: None,
                operation_index: None,
                reason: UnsupportedReason::PostdominanceUnavailable {
                    blocks: control_unknown.iter().copied().collect(),
                },
            });
        }
        let control_regions = control_regions(
            body,
            &blocks,
            &reachable,
            &postdominance_available,
            &natural_loop_nests,
            &effective_successors,
        );
        Self {
            body,
            reachable,
            incoming,
            control_regions,
            control_unknown,
            trivial_phi_representatives,
            private_load_slots,
            private_slot_stores,
            uniform_recurrence_edges,
            proven_no_overflow,
            parameter_variations,
            summarized_calls,
            uniform_input_calls,
            workgroup_size,
            exact_d1_workgroup,
            value_definitions,
            report,
        }
    }

    fn run(mut self) -> AnalysisReport {
        self.collect_unsupported_diagnostics();
        self.initialize_facts();
        self.solve();
        self.diagnose_nonuniform_helper_calls();
        self.diagnose_barriers();
        self.report
    }

    fn collect_unsupported_diagnostics(&mut self) {
        let mut defined = BTreeSet::new();
        defined.extend(self.body.parameters.iter().copied());
        for block in &self.body.blocks {
            defined.extend(block.parameters.iter().map(|parameter| parameter.id));
            defined.extend(
                block
                    .operations
                    .iter()
                    .flat_map(|operation| operation.results.iter().map(|result| result.id)),
            );
        }

        let mut unknown = BTreeSet::new();
        for block in &self.body.blocks {
            for (operation_index, operation) in block.operations.iter().enumerate() {
                if let OperationKind::Call { callee, arguments } = &operation.kind
                    && !self.summarized_calls.contains(callee)
                    && AmdGpuDiagnosticOperation::from_intrinsic_call(callee, arguments).is_none()
                {
                    self.report.diagnostics.push(Diagnostic::Unsupported {
                        block: Some(block.id),
                        operation_index: Some(operation_index),
                        reason: UnsupportedReason::CallWithoutSummary {
                            callee: callee.clone(),
                        },
                    });
                }
                unknown.extend(
                    operation
                        .kind
                        .operands()
                        .into_iter()
                        .filter(|value| !defined.contains(value)),
                );
            }
            if let Some(terminator) = &block.terminator {
                unknown.extend(
                    terminator
                        .operands()
                        .into_iter()
                        .filter(|value| !defined.contains(value)),
                );
            }
        }
        self.report
            .diagnostics
            .extend(unknown.into_iter().map(|value| Diagnostic::Unsupported {
                block: None,
                operation_index: None,
                reason: UnsupportedReason::UnknownValue { value },
            }));
    }

    fn initialize_facts(&mut self) {
        for (index, parameter) in self.body.parameters.iter().enumerate() {
            self.report.values.insert(
                *parameter,
                self.parameter_variations
                    .get(index)
                    .copied()
                    .unwrap_or(Variation::Varying),
            );
        }
        for block in &self.body.blocks {
            let reachable = self.reachable.contains(&block.id);
            let initial = if reachable {
                Variation::GridUniform
            } else {
                Variation::Varying
            };
            if reachable {
                let control = if self.control_unknown.contains(&block.id) {
                    Variation::Varying
                } else {
                    Variation::GridUniform
                };
                self.report.block_controls.insert(block.id, control);
            }
            for parameter in &block.parameters {
                self.report.values.insert(parameter.id, initial);
            }
            for result in block
                .operations
                .iter()
                .flat_map(|operation| &operation.results)
            {
                self.report.values.insert(result.id, initial);
            }
        }
    }

    fn solve(&mut self) {
        loop {
            let mut changed = self.update_block_controls();
            for block in &self.body.blocks {
                if !self.reachable.contains(&block.id) {
                    continue;
                }
                changed |= self.update_block_parameters(block);
                for operation in &block.operations {
                    for result in &operation.results {
                        let variation = self.operation_result_variation(operation, result.id);
                        changed |= raise(&mut self.report.values, result.id, variation);
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn update_block_controls(&mut self) -> bool {
        let mut next = self
            .reachable
            .iter()
            .map(|block| {
                let control = if self.control_unknown.contains(block) {
                    Variation::Varying
                } else {
                    Variation::GridUniform
                };
                (*block, control)
            })
            .collect::<BTreeMap<_, _>>();

        for block in &self.body.blocks {
            if !self.reachable.contains(&block.id) {
                continue;
            }
            let Some(discriminator) = block.terminator.as_ref().and_then(discriminator) else {
                continue;
            };
            let source_control = self
                .report
                .block_controls
                .get(&block.id)
                .copied()
                .unwrap_or(Variation::Varying);
            let branch_control = source_control.join(self.value(discriminator));
            if let Some(region) = self.control_regions.get(&block.id) {
                for controlled in region {
                    raise(&mut next, *controlled, branch_control);
                }
            }
        }

        let mut changed = false;
        for (block, variation) in next {
            changed |= raise(&mut self.report.block_controls, block, variation);
        }
        changed
    }

    fn update_block_parameters(&mut self, block: &BasicBlock) -> bool {
        let Some(edges) = self.incoming.get(&block.id).cloned() else {
            let mut changed = false;
            for parameter in &block.parameters {
                changed |= raise(&mut self.report.values, parameter.id, Variation::Varying);
            }
            return changed;
        };

        let mut changed = false;
        for (index, parameter) in block.parameters.iter().enumerate() {
            let distinct_origins = edges
                .iter()
                .filter_map(|edge| edge.arguments.get(index))
                .map(|value| self.trivial_phi_representative(*value))
                .collect::<BTreeSet<_>>();
            let control_can_select_value = distinct_origins.len() != 1;
            let mut variation = Variation::GridUniform;
            for edge in &edges {
                let argument = edge
                    .arguments
                    .get(index)
                    .map(|value| self.value(*value))
                    .unwrap_or(Variation::Varying);
                variation = variation.join(argument);
                if control_can_select_value {
                    let edge_control = edge
                        .discriminator
                        .map(|value| self.value(value))
                        .unwrap_or(Variation::GridUniform);
                    let recurrence_edge = edge.arguments.get(index).is_some_and(|argument| {
                        self.uniform_recurrence_edges.contains(&(
                            parameter.id,
                            edge.source,
                            *argument,
                        ))
                    });
                    let source_control = if recurrence_edge {
                        self.direct_control_variation(edge.source)
                    } else {
                        self.report
                            .block_controls
                            .get(&edge.source)
                            .copied()
                            .unwrap_or(Variation::Varying)
                    };
                    variation = variation.join(source_control).join(edge_control);
                }
            }
            changed |= raise(&mut self.report.values, parameter.id, variation);
        }
        changed
    }

    fn direct_control_variation(&self, block: BlockId) -> Variation {
        if self.control_unknown.contains(&block) {
            return Variation::Varying;
        }
        self.control_regions
            .iter()
            .filter(|(_, region)| region.contains(&block))
            .filter_map(|(source, _)| {
                self.body
                    .blocks
                    .iter()
                    .find(|candidate| candidate.id == *source)
                    .and_then(|candidate| candidate.terminator.as_ref())
                    .and_then(discriminator)
            })
            .map(|selector| self.value(selector))
            .fold(Variation::GridUniform, Variation::join)
    }

    fn operation_variation(&self, operation: &Operation) -> Variation {
        match &operation.kind {
            OperationKind::Storage(_)
            | OperationKind::Execution(_)
            | OperationKind::VerificationContract(_)
            | OperationKind::VectorLoad(_)
            | OperationKind::VectorStore(_) => Variation::Varying,
            OperationKind::VectorLayoutConvert(_) => {
                join_values(operation.kind.operands(), &self.report.values)
            }
            OperationKind::Constant(_) => Variation::GridUniform,
            OperationKind::Intrinsic(intrinsic) => match intrinsic.kind {
                IntrinsicKind::LaunchExtent { .. } => Variation::GridUniform,
                IntrinsicKind::InvocationIndex { kind, .. } => match kind {
                    IndexKind::Global | IndexKind::Local => Variation::Varying,
                    IndexKind::Workgroup => Variation::WorkgroupUniform,
                    IndexKind::WorkgroupSize | IndexKind::WorkgroupCount => Variation::GridUniform,
                },
            },
            OperationKind::Call { callee, .. } if self.summarized_calls.contains(callee) => {
                join_values(operation.kind.operands(), &self.report.values)
            }
            OperationKind::Load { .. } => self
                .private_load_variation(operation)
                .unwrap_or(Variation::Varying),
            OperationKind::Call { .. }
            | OperationKind::GuardedLoad { .. }
            | OperationKind::Atomic(_) => Variation::Varying,
            OperationKind::MemoryIntrinsic(
                fe2o3_kernel_ir::MemoryIntrinsicOperation::PointerDistance { .. },
            ) => join_values(operation.kind.operands(), &self.report.values),
            OperationKind::MemoryIntrinsic(_) => Variation::Varying,
            OperationKind::InlineAssembly(_)
            | OperationKind::Gfx942OrderedRegion(_)
            | OperationKind::Gfx942OrderedProgram(_)
            | OperationKind::Gfx942CompleteBodyDeclaration(_)
            | OperationKind::Gfx942CompleteBodyStep(_)
            | OperationKind::Gfx942PhysicalEntryDeclaration(_)
            | OperationKind::Gfx942PhysicalEntryStep(_)
            | OperationKind::Gfx942PhysicalGlobalCopyDeclaration(_)
            | OperationKind::Gfx942PhysicalGlobalCopyStep(_)
            | OperationKind::Gfx942PhysicalLdsExchangeDeclaration(_)
            | OperationKind::Gfx942PhysicalLdsExchangeStep(_) => Variation::Varying,
            OperationKind::Matrix(_) => Variation::Varying,
            OperationKind::Gfx950LdsTranspose(transpose) => match transpose.kind {
                fe2o3_kernel_ir::Gfx950LdsTransposeOperationKindV1::Current { .. }
                | fe2o3_kernel_ir::Gfx950LdsTransposeOperationKindV1::Stage { .. }
                | fe2o3_kernel_ir::Gfx950LdsTransposeOperationKindV1::Publish { .. } => {
                    Variation::WorkgroupUniform
                }
                fe2o3_kernel_ir::Gfx950LdsTransposeOperationKindV1::Read { .. } => {
                    Variation::Varying
                }
            },
            OperationKind::Alloca {
                count,
                address_space,
                ..
            } => {
                let allocation = match address_space {
                    AddressSpace::Workgroup => Variation::WorkgroupUniform,
                    AddressSpace::Private
                    | AddressSpace::Global
                    | AddressSpace::Constant
                    | AddressSpace::Generic => Variation::Varying,
                };
                allocation.join(join_values(count.iter().copied(), &self.report.values))
            }
            OperationKind::WorkgroupMemory(_) => Variation::WorkgroupUniform,
            OperationKind::Wave(wave) => match wave.kind {
                WaveOperationKind::LaneId => Variation::Varying,
                WaveOperationKind::Ballot { predicate }
                | WaveOperationKind::Any { predicate }
                | WaveOperationKind::All { predicate } => {
                    subgroup_collective_variation(self.value(predicate))
                }
                WaveOperationKind::ShuffleIndex {
                    value,
                    source_lane,
                    tile_width,
                } => {
                    let value = self.value(value);
                    if value.is_uniform_for(fe2o3_kernel_ir::SynchronizationScope::Subgroup) {
                        value
                    } else if tile_width == wave.width.lanes()
                        && self
                            .value(source_lane)
                            .is_uniform_for(fe2o3_kernel_ir::SynchronizationScope::Subgroup)
                    {
                        Variation::SubgroupUniform
                    } else {
                        Variation::Varying
                    }
                }
                WaveOperationKind::ReduceF32 {
                    value, tile_width, ..
                } => {
                    let value = self.value(value);
                    if value.is_uniform_for(fe2o3_kernel_ir::SynchronizationScope::Subgroup) {
                        value
                    } else if tile_width == wave.width.lanes() {
                        Variation::SubgroupUniform
                    } else {
                        Variation::Varying
                    }
                }
                WaveOperationKind::BroadcastF32 {
                    value, tile_width, ..
                } => {
                    let value = self.value(value);
                    if value.is_uniform_for(fe2o3_kernel_ir::SynchronizationScope::Subgroup) {
                        value
                    } else if tile_width == wave.width.lanes() {
                        Variation::SubgroupUniform
                    } else {
                        Variation::Varying
                    }
                }
            },
            OperationKind::Store { .. }
            | OperationKind::GuardedStore { .. }
            | OperationKind::Barrier(_)
            | OperationKind::Fence(_)
            | OperationKind::WorkgroupBarrier(_) => Variation::Varying,
            OperationKind::Binary {
                op: BinaryOp::Divide,
                lhs,
                rhs,
            } if self.is_workgroup_index_quotient(*lhs, *rhs) => Variation::WorkgroupUniform,
            OperationKind::Binary {
                op: BinaryOp::Divide,
                lhs,
                rhs,
            } if self.is_wave64_index_quotient(*lhs, *rhs) => Variation::SubgroupUniform,
            OperationKind::Compare {
                predicate,
                lhs,
                rhs,
            } if matches!(operation.results.as_slice(), [result] if result.ty == Type::BOOL)
                && contextual_control::aligned_global_comparison(
                    self.exact_d1_workgroup,
                    &self.value_definitions,
                    *predicate,
                    *lhs,
                    *rhs,
                ) =>
            {
                Variation::WorkgroupUniform
            }
            OperationKind::Unary { .. }
            | OperationKind::Binary { .. }
            | OperationKind::Compare { .. }
            | OperationKind::Cast { .. }
            | OperationKind::Select { .. }
            | OperationKind::SliceLength { .. }
            | OperationKind::SliceData { .. }
            | OperationKind::GetElementPointer { .. } => {
                join_values(operation.kind.operands(), &self.report.values)
            }
        }
    }

    fn is_workgroup_index_quotient(&self, lhs: ValueId, rhs: ValueId) -> bool {
        let Some(workgroup_size) = self.workgroup_size else {
            return false;
        };
        let rhs = self.trivial_phi_representative(rhs);
        let Some(axis) = self.global_invocation_index_axis(lhs) else {
            return false;
        };
        let extent = match axis {
            Axis::X => workgroup_size.x,
            Axis::Y => workgroup_size.y,
            Axis::Z => workgroup_size.z,
        };
        extent != 0 && self.unsigned_constant(rhs) == Some(u64::from(extent))
    }

    fn is_wave64_index_quotient(&self, lhs: ValueId, rhs: ValueId) -> bool {
        let rhs = self.trivial_phi_representative(rhs);
        if self.workgroup_size != Some(WorkgroupSize::new(256, 1, 1))
            || self.unsigned_constant(rhs) != Some(64)
        {
            return false;
        }
        self.global_invocation_index_axis(lhs) == Some(Axis::X)
    }

    fn global_invocation_index_axis(&self, value: ValueId) -> Option<Axis> {
        let mut value = value;
        let mut expected_type = None;
        let mut visited = BTreeSet::new();
        let mut remaining = self.value_definitions.len().checked_add(1)?;
        loop {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            value = self.trivial_phi_representative(value);
            if !visited.insert(value) {
                return None;
            }

            let operation = self.value_definitions.get(&value).copied()?;
            let [result] = operation.results.as_slice() else {
                return None;
            };
            if result.id != value
                || expected_type
                    .as_ref()
                    .is_some_and(|expected| expected != &result.ty)
            {
                return None;
            }

            match &operation.kind {
                OperationKind::Intrinsic(fe2o3_kernel_ir::IntrinsicOperation {
                    kind:
                        IntrinsicKind::InvocationIndex {
                            kind: IndexKind::Global,
                            axis,
                        },
                    result_type,
                }) if result.ty == Type::INDEX && result_type == &Type::INDEX => {
                    return Some(*axis);
                }
                OperationKind::Cast {
                    kind: CastKind::Bitcast,
                    value: source,
                    to,
                } if result.ty == *to => {
                    expected_type = match to {
                        Type::Scalar(ScalarType::U64) => Some(Type::INDEX),
                        Type::Scalar(ScalarType::Index) => Some(Type::Scalar(ScalarType::U64)),
                        _ => return None,
                    };
                    value = *source;
                }
                _ => return None,
            }
        }
    }

    fn unsigned_constant(&self, value: ValueId) -> Option<u64> {
        let mut value = value;
        let mut visited = BTreeSet::new();
        let mut remaining = self.value_definitions.len().checked_add(1)?;
        loop {
            if remaining == 0 {
                return None;
            }
            remaining -= 1;
            value = self.trivial_phi_representative(value);
            if !visited.insert(value) {
                return None;
            }
            let operation = self.value_definitions.get(&value).copied()?;
            let result = operation.results.iter().find(|result| result.id == value)?;
            match &operation.kind {
                OperationKind::Constant(constant) if result.ty == constant.ty() => {
                    return match constant {
                        Constant::U8(value) => Some(u64::from(*value)),
                        Constant::U16(value) => Some(u64::from(*value)),
                        Constant::U32(value) => Some(u64::from(*value)),
                        Constant::U64(value) | Constant::Index(value) => Some(*value),
                        Constant::I8(value) => u64::try_from(*value).ok(),
                        Constant::I16(value) => u64::try_from(*value).ok(),
                        Constant::I32(value) => u64::try_from(*value).ok(),
                        Constant::I64(value) => u64::try_from(*value).ok(),
                        Constant::Bool(_)
                        | Constant::F16Bits(_)
                        | Constant::Bf16Bits(_)
                        | Constant::F32Bits(_)
                        | Constant::F64Bits(_) => None,
                    };
                }
                OperationKind::Cast {
                    kind: CastKind::ZeroExtend,
                    value: source,
                    to,
                } if result.ty == *to && self.zero_extend_preserves_unsigned_value(*source, to) => {
                    value = *source;
                }
                _ => return None,
            }
        }
    }

    fn zero_extend_preserves_unsigned_value(&self, value: ValueId, to: &Type) -> bool {
        let value = self.trivial_phi_representative(value);
        let Some(source) = self
            .value_definitions
            .get(&value)
            .and_then(|operation| operation.results.iter().find(|result| result.id == value))
            .map(|result| &result.ty)
        else {
            return false;
        };
        let (Type::Scalar(source), Type::Scalar(destination)) = (source, to) else {
            return false;
        };
        let source_width = match source {
            ScalarType::U8 => 8,
            ScalarType::U16 => 16,
            ScalarType::U32 => 32,
            ScalarType::U64 => 64,
            ScalarType::U128
            | ScalarType::Index
            | ScalarType::Bool
            | ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::I128
            | ScalarType::F16
            | ScalarType::Bf16
            | ScalarType::F32
            | ScalarType::F64 => return false,
        };
        match destination {
            ScalarType::U16 => source_width < 16,
            ScalarType::U32 => source_width < 32,
            ScalarType::U64 => source_width < 64,
            ScalarType::U128 => source_width < 128,
            // KIR's verified ZeroExtend-to-Index contract is the only
            // target-neutral value-preserving Index conversion admitted here.
            ScalarType::Index => source_width <= 32,
            ScalarType::Bool
            | ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I64
            | ScalarType::I128
            | ScalarType::U8
            | ScalarType::F16
            | ScalarType::Bf16
            | ScalarType::F32
            | ScalarType::F64 => false,
        }
    }

    fn operation_result_variation(&self, operation: &Operation, result: ValueId) -> Variation {
        if self.proven_no_overflow.contains(&result) {
            Variation::GridUniform
        } else {
            self.operation_variation(operation)
        }
    }

    fn diagnose_barriers(&mut self) {
        for block in &self.body.blocks {
            if !self.reachable.contains(&block.id) {
                continue;
            }
            let control = self
                .report
                .block_controls
                .get(&block.id)
                .copied()
                .unwrap_or(Variation::Varying);
            for (operation_index, operation) in block.operations.iter().enumerate() {
                let execution_scope = match &operation.kind {
                    OperationKind::Barrier(barrier) => barrier.execution_scope,
                    OperationKind::WorkgroupBarrier(_) => {
                        fe2o3_kernel_ir::SynchronizationScope::Workgroup
                    }
                    _ => continue,
                };
                if !control.is_uniform_for(execution_scope) {
                    self.report.diagnostics.push(Diagnostic::DivergentBarrier {
                        block: block.id,
                        operation_index,
                        execution_scope,
                        control,
                    });
                }
            }
        }
    }

    fn diagnose_nonuniform_helper_calls(&mut self) {
        for block in &self.body.blocks {
            if !self.reachable.contains(&block.id) {
                continue;
            }
            for (operation_index, operation) in block.operations.iter().enumerate() {
                let OperationKind::Call { callee, arguments } = &operation.kind else {
                    continue;
                };
                if self.uniform_input_calls.contains(callee)
                    && (self.report.block_control(block.id) != Variation::GridUniform
                        || arguments
                            .iter()
                            .any(|argument| self.value(*argument) != Variation::GridUniform))
                {
                    self.report.diagnostics.push(Diagnostic::Unsupported {
                        block: Some(block.id),
                        operation_index: Some(operation_index),
                        reason: UnsupportedReason::CallWithoutSummary {
                            callee: callee.clone(),
                        },
                    });
                }
            }
        }
    }

    fn value(&self, value: ValueId) -> Variation {
        self.report
            .values
            .get(&value)
            .copied()
            .unwrap_or(Variation::Varying)
    }

    fn private_load_variation(&self, operation: &Operation) -> Option<Variation> {
        let result = operation.results.first()?.id;
        let slot = self.private_load_slots.get(&result)?;
        let stores = self.private_slot_stores.get(slot)?;
        Some(
            stores
                .iter()
                .fold(Variation::GridUniform, |variation, store| {
                    variation.join(self.value(store.value)).join(
                        self.report
                            .block_controls
                            .get(&store.block)
                            .copied()
                            .unwrap_or(Variation::Varying),
                    )
                }),
        )
    }

    fn trivial_phi_representative(&self, mut value: ValueId) -> ValueId {
        let mut remaining = self.trivial_phi_representatives.len();
        while remaining != 0 {
            let Some(next) = self.trivial_phi_representatives.get(&value).copied() else {
                break;
            };
            if next == value {
                break;
            }
            value = next;
            remaining -= 1;
        }
        value
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UnsignedRange {
    min: u128,
    max: u128,
}

impl UnsignedRange {
    const fn exact(value: u128) -> Self {
        Self {
            min: value,
            max: value,
        }
    }

    fn join(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn intersect(self, other: Self) -> Self {
        let intersection = Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        };
        if intersection.min <= intersection.max {
            intersection
        } else {
            // An inconsistent path fact must not manufacture a proof.
            self
        }
    }
}

#[derive(Clone, Copy)]
struct OperationDefinition<'a> {
    block: BlockId,
    result_index: usize,
    operation: &'a Operation,
}

const MAX_KNOWN_INTEGER_VALUES: usize = 32;
const MAX_RELATIONAL_PROOF_WORK: usize = 65_536;

fn charge_relational_work(work: &mut usize, amount: usize) -> Option<()> {
    *work = work.checked_add(amount)?;
    (*work <= MAX_RELATIONAL_PROOF_WORK).then_some(())
}

struct KnownIntegerValueAnalysis<'a> {
    body: &'a FunctionBody,
    incoming: &'a BTreeMap<BlockId, Vec<Edge>>,
    private_load_slots: &'a BTreeMap<ValueId, ValueId>,
    private_slot_stores: &'a BTreeMap<ValueId, Vec<PrivateStore>>,
    operation_definitions: BTreeMap<ValueId, (usize, &'a Operation)>,
    block_parameters: BTreeMap<ValueId, (BlockId, usize)>,
    cache: BTreeMap<ValueId, Option<BTreeSet<u64>>>,
    visiting: BTreeSet<ValueId>,
}

impl<'a> KnownIntegerValueAnalysis<'a> {
    fn new(
        body: &'a FunctionBody,
        incoming: &'a BTreeMap<BlockId, Vec<Edge>>,
        private_load_slots: &'a BTreeMap<ValueId, ValueId>,
        private_slot_stores: &'a BTreeMap<ValueId, Vec<PrivateStore>>,
    ) -> Self {
        let mut operation_definitions = BTreeMap::new();
        let mut block_parameters = BTreeMap::new();
        for block in &body.blocks {
            for (index, parameter) in block.parameters.iter().enumerate() {
                block_parameters.insert(parameter.id, (block.id, index));
            }
            for operation in &block.operations {
                for (result_index, result) in operation.results.iter().enumerate() {
                    operation_definitions.insert(result.id, (result_index, operation));
                }
            }
        }
        Self {
            body,
            incoming,
            private_load_slots,
            private_slot_stores,
            operation_definitions,
            block_parameters,
            cache: BTreeMap::new(),
            visiting: BTreeSet::new(),
        }
    }

    fn solve_terminator_selectors(mut self) -> BTreeMap<ValueId, BTreeSet<u64>> {
        let selectors = self
            .body
            .blocks
            .iter()
            .filter_map(|block| block.terminator.as_ref().and_then(discriminator))
            .collect::<BTreeSet<_>>();
        selectors
            .into_iter()
            .filter_map(|selector| self.values(selector).map(|values| (selector, values)))
            .collect()
    }

    fn values(&mut self, value: ValueId) -> Option<BTreeSet<u64>> {
        if let Some(values) = self.cache.get(&value) {
            return values.clone();
        }
        if !self.visiting.insert(value) {
            return None;
        }
        let values = self.derive_values(value);
        self.visiting.remove(&value);
        self.cache.insert(value, values.clone());
        values
    }

    fn derive_values(&mut self, value: ValueId) -> Option<BTreeSet<u64>> {
        if let Some((block, index)) = self.block_parameters.get(&value).copied() {
            let edges = self.incoming.get(&block)?.clone();
            if edges.is_empty() {
                return None;
            }
            let mut values = BTreeSet::new();
            for edge in edges {
                let argument = *edge.arguments.get(index)?;
                extend_known_values(&mut values, self.values(argument)?)?;
            }
            return Some(values);
        }

        let (result_index, operation) = self.operation_definitions.get(&value).copied()?;
        if result_index != 0 {
            return None;
        }
        let kind = operation.kind.clone();
        match kind {
            OperationKind::Constant(constant) => {
                Some(BTreeSet::from([known_u64_constant(&constant)?]))
            }
            OperationKind::Cast {
                kind: CastKind::ZeroExtend | CastKind::Bitcast,
                value,
                ..
            } => self.values(value),
            OperationKind::Select {
                true_value,
                false_value,
                ..
            } => {
                let mut values = self.values(true_value)?;
                extend_known_values(&mut values, self.values(false_value)?)?;
                Some(values)
            }
            OperationKind::Load { .. } => {
                let slot = *self.private_load_slots.get(&value)?;
                let stores = self.private_slot_stores.get(&slot)?.clone();
                if stores.is_empty() {
                    return None;
                }
                let mut values = BTreeSet::new();
                for store in stores {
                    extend_known_values(&mut values, self.values(store.value)?)?;
                }
                Some(values)
            }
            _ => None,
        }
    }
}

fn extend_known_values(destination: &mut BTreeSet<u64>, source: BTreeSet<u64>) -> Option<()> {
    if destination.len().saturating_add(source.len()) > MAX_KNOWN_INTEGER_VALUES {
        return None;
    }
    destination.extend(source);
    (destination.len() <= MAX_KNOWN_INTEGER_VALUES).then_some(())
}

fn known_u64_constant(constant: &Constant) -> Option<u64> {
    match constant {
        Constant::Bool(value) => Some(u64::from(*value)),
        Constant::I8(value) => u64::try_from(*value).ok(),
        Constant::I16(value) => u64::try_from(*value).ok(),
        Constant::I32(value) => u64::try_from(*value).ok(),
        Constant::I64(value) => u64::try_from(*value).ok(),
        Constant::U8(value) => Some((*value).into()),
        Constant::U16(value) => Some((*value).into()),
        Constant::U32(value) => Some((*value).into()),
        Constant::U64(value) | Constant::Index(value) => Some(*value),
        Constant::F16Bits(_)
        | Constant::Bf16Bits(_)
        | Constant::F32Bits(_)
        | Constant::F64Bits(_) => None,
    }
}

fn effective_successors(
    body: &FunctionBody,
    known_values: &BTreeMap<ValueId, BTreeSet<u64>>,
) -> BTreeMap<BlockId, BTreeSet<BlockId>> {
    body.blocks
        .iter()
        .filter_map(|block| {
            let terminator = block.terminator.as_ref()?;
            let all = terminator.successors().into_iter().collect::<BTreeSet<_>>();
            let Some(selector) = discriminator(terminator) else {
                return Some((block.id, all));
            };
            let Some(values) = known_values
                .get(&selector)
                .filter(|values| !values.is_empty())
            else {
                return Some((block.id, all));
            };
            let selected = match terminator {
                Terminator::Switch {
                    cases,
                    default_target,
                    ..
                } => values
                    .iter()
                    .map(|value| {
                        cases
                            .iter()
                            .find(|case| case.value == *value)
                            .map_or(*default_target, |case| case.target)
                    })
                    .collect(),
                Terminator::IntegerSwitch {
                    cases,
                    default_target,
                    ..
                } => values
                    .iter()
                    .map(|value| {
                        cases
                            .iter()
                            .find(|case| known_u64_constant(&case.value) == Some(*value))
                            .map_or(*default_target, |case| case.target)
                    })
                    .collect(),
                _ => all.clone(),
            };
            Some((block.id, selected))
        })
        .collect()
}

fn successors_for(
    block: BlockId,
    terminator: &Terminator,
    effective_successors: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> Vec<BlockId> {
    effective_successors
        .get(&block)
        .map(|successors| successors.iter().copied().collect())
        .unwrap_or_else(|| terminator.successors())
}

#[derive(Clone, Copy)]
struct RangeGuard {
    block: BlockId,
    then_target: BlockId,
    else_target: BlockId,
    then_edge_is_exclusive: bool,
    else_edge_is_exclusive: bool,
    predicate: ComparePredicate,
    lhs: ValueId,
    rhs: ValueId,
}

struct UnsignedRangeAnalysis<'a> {
    body: &'a FunctionBody,
    reachable: &'a BTreeSet<BlockId>,
    incoming: &'a BTreeMap<BlockId, Vec<Edge>>,
    dominators: &'a BTreeMap<BlockId, BTreeSet<BlockId>>,
    trivial_phi_representatives: &'a BTreeMap<ValueId, ValueId>,
    private_load_slots: &'a BTreeMap<ValueId, ValueId>,
    private_slot_stores: &'a BTreeMap<ValueId, Vec<PrivateStore>>,
    value_types: BTreeMap<ValueId, Type>,
    operation_definitions: BTreeMap<ValueId, OperationDefinition<'a>>,
    block_parameters: BTreeMap<ValueId, (BlockId, usize)>,
    guards: Vec<RangeGuard>,
    cache: BTreeMap<(ValueId, BlockId), UnsignedRange>,
    visiting: BTreeSet<(ValueId, BlockId)>,
}

#[allow(clippy::too_many_arguments)]
fn prove_unsigned_checked_arithmetic(
    function: &Function,
    body: &FunctionBody,
    reachable: &BTreeSet<BlockId>,
    incoming: &BTreeMap<BlockId, Vec<Edge>>,
    dominators: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    trivial_phi_representatives: &BTreeMap<ValueId, ValueId>,
    private_load_slots: &BTreeMap<ValueId, ValueId>,
    private_slot_stores: &BTreeMap<ValueId, Vec<PrivateStore>>,
) -> BTreeSet<ValueId> {
    let mut analysis = UnsignedRangeAnalysis::new(
        function,
        body,
        reachable,
        incoming,
        dominators,
        trivial_phi_representatives,
        private_load_slots,
        private_slot_stores,
    );
    let sites = body
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id))
        .flat_map(|block| {
            block.operations.iter().filter_map(|operation| {
                let OperationKind::Binary {
                    op: BinaryOp::Checked(operator),
                    lhs,
                    rhs,
                } = operation.kind
                else {
                    return None;
                };
                (operation.results.len() == 2).then_some((
                    block.id,
                    operator,
                    lhs,
                    rhs,
                    operation.results[0].ty.clone(),
                    operation.results[1].id,
                ))
            })
        })
        .collect::<Vec<_>>();

    let mut proven = BTreeSet::new();
    for (block, operator, lhs, rhs, result_type, overflow) in sites {
        let accepted =
            analysis.checked_operation_cannot_overflow(block, operator, lhs, rhs, &result_type);
        if accepted {
            proven.insert(overflow);
        }
    }
    proven
}

impl<'a> UnsignedRangeAnalysis<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        function: &Function,
        body: &'a FunctionBody,
        reachable: &'a BTreeSet<BlockId>,
        incoming: &'a BTreeMap<BlockId, Vec<Edge>>,
        dominators: &'a BTreeMap<BlockId, BTreeSet<BlockId>>,
        trivial_phi_representatives: &'a BTreeMap<ValueId, ValueId>,
        private_load_slots: &'a BTreeMap<ValueId, ValueId>,
        private_slot_stores: &'a BTreeMap<ValueId, Vec<PrivateStore>>,
    ) -> Self {
        let mut value_types = BTreeMap::new();
        value_types.extend(
            body.parameters
                .iter()
                .copied()
                .zip(function.signature.parameters.iter().cloned()),
        );
        let mut operation_definitions = BTreeMap::new();
        let mut block_parameters = BTreeMap::new();
        for block in &body.blocks {
            for (index, parameter) in block.parameters.iter().enumerate() {
                value_types.insert(parameter.id, parameter.ty.clone());
                block_parameters.insert(parameter.id, (block.id, index));
            }
            for operation in &block.operations {
                for (result_index, result) in operation.results.iter().enumerate() {
                    value_types.insert(result.id, result.ty.clone());
                    operation_definitions.insert(
                        result.id,
                        OperationDefinition {
                            block: block.id,
                            result_index,
                            operation,
                        },
                    );
                }
            }
        }

        let mut analysis = Self {
            body,
            reachable,
            incoming,
            dominators,
            trivial_phi_representatives,
            private_load_slots,
            private_slot_stores,
            value_types,
            operation_definitions,
            block_parameters,
            guards: Vec::new(),
            cache: BTreeMap::new(),
            visiting: BTreeSet::new(),
        };
        analysis.guards = analysis.collect_guards();
        analysis
    }

    fn collect_guards(&self) -> Vec<RangeGuard> {
        self.body
            .blocks
            .iter()
            .filter(|block| self.reachable.contains(&block.id))
            .filter_map(|block| {
                let Terminator::ConditionalBranch {
                    condition,
                    then_target,
                    else_target,
                    ..
                } = block.terminator.as_ref()?
                else {
                    return None;
                };
                let (predicate, lhs, rhs) = self.comparison_for_condition(*condition, block.id)?;
                Some(RangeGuard {
                    block: block.id,
                    then_target: *then_target,
                    else_target: *else_target,
                    then_edge_is_exclusive: self.edge_is_exclusive(block.id, *then_target),
                    else_edge_is_exclusive: self.edge_is_exclusive(block.id, *else_target),
                    predicate,
                    lhs,
                    rhs,
                })
            })
            .collect()
    }

    fn edge_is_exclusive(&self, source: BlockId, target: BlockId) -> bool {
        self.incoming
            .get(&target)
            .is_some_and(|edges| matches!(edges.as_slice(), [edge] if edge.source == source))
    }

    fn comparison_for_condition(
        &self,
        mut value: ValueId,
        use_block: BlockId,
    ) -> Option<(ComparePredicate, ValueId, ValueId)> {
        let mut negate = false;
        let mut seen = BTreeSet::new();
        while seen.insert(value) {
            value = resolve_representative(self.trivial_phi_representatives, value);
            if let Some(slot) = self.private_load_slots.get(&value)
                && let Some([store]) = self.private_slot_stores.get(slot).map(Vec::as_slice)
                && self.dominates(store.block, use_block)
            {
                value = store.value;
                continue;
            }
            let definition = self.operation_definitions.get(&value)?;
            match &definition.operation.kind {
                OperationKind::Compare {
                    predicate,
                    lhs,
                    rhs,
                } => {
                    let predicate = if negate {
                        invert_predicate(*predicate)
                    } else {
                        *predicate
                    };
                    return Some((predicate, *lhs, *rhs));
                }
                OperationKind::Unary {
                    op: UnaryOp::Not,
                    operand,
                } if self.value_types.get(operand) == Some(&Type::BOOL) => {
                    negate = !negate;
                    value = *operand;
                }
                _ => return None,
            }
        }
        None
    }

    fn checked_operation_cannot_overflow(
        &mut self,
        block: BlockId,
        operator: CheckedBinaryOperator,
        lhs: ValueId,
        rhs: ValueId,
        result_type: &Type,
    ) -> bool {
        let Some(type_range) = unsigned_type_range(result_type) else {
            return false;
        };
        let lhs_value = lhs;
        let rhs_value = rhs;
        if let (Some(lhs), Some(rhs)) = (self.range_at(lhs, block), self.range_at(rhs, block))
            && checked_result_range(operator, lhs, rhs, type_range.max).is_some()
        {
            return true;
        }
        match operator {
            CheckedBinaryOperator::Subtract => {
                self.scaled_quotient_subtraction_is_nonnegative(block, lhs_value, rhs_value)
            }
            CheckedBinaryOperator::Add => {
                self.scaled_quotient_residual_addition_fits(block, lhs_value, rhs_value)
                    || self.scaled_quotient_residual_addition_fits(block, rhs_value, lhs_value)
            }
            CheckedBinaryOperator::Multiply => false,
        }
    }

    /// Proves that adding `offset` to an authenticated quotient residual fits
    /// when the residual's intra-scale term plus `offset` is below `scale`.
    fn scaled_quotient_residual_addition_fits(
        &mut self,
        query_block: BlockId,
        residual: ValueId,
        offset: ValueId,
    ) -> bool {
        let mut work = 0_usize;
        let Some(offset_range) = self.range_at(offset, query_block) else {
            return false;
        };
        let Some((residual_lhs, residual_rhs)) = self.checked_binary_operands(
            residual,
            CheckedBinaryOperator::Subtract,
            query_block,
            &mut work,
        ) else {
            return false;
        };
        let Some(rhs_product) = self.checked_binary_operands(
            residual_rhs,
            CheckedBinaryOperator::Multiply,
            query_block,
            &mut work,
        ) else {
            return false;
        };
        let lhs_terms = if let Some((lhs, rhs)) = self.checked_binary_operands(
            residual_lhs,
            CheckedBinaryOperator::Add,
            query_block,
            &mut work,
        ) {
            vec![(lhs, Some(rhs)), (rhs, Some(lhs))]
        } else {
            vec![(residual_lhs, None)]
        };

        for (lhs_base, extra) in lhs_terms {
            let Some(lhs_product) = self.checked_binary_operands(
                lhs_base,
                CheckedBinaryOperator::Multiply,
                query_block,
                &mut work,
            ) else {
                continue;
            };
            for (numerator, scale) in [lhs_product, (lhs_product.1, lhs_product.0)] {
                let Some(scale_range) = self.range_at(scale, query_block) else {
                    continue;
                };
                let extra_range = match extra {
                    Some(extra) => self.range_at(extra, query_block),
                    None => Some(UnsignedRange::exact(0)),
                };
                let Some(extra_range) = extra_range else {
                    continue;
                };
                if extra_range
                    .max
                    .checked_add(offset_range.max)
                    .is_none_or(|sum| sum >= scale_range.min)
                {
                    continue;
                }
                for (quotient, multiple) in [rhs_product, (rhs_product.1, rhs_product.0)] {
                    let Some((quotient_numerator, divisor)) =
                        self.binary_operands(quotient, BinaryOp::Divide, query_block, &mut work)
                    else {
                        continue;
                    };
                    if self.values_equivalent(numerator, quotient_numerator, query_block, &mut work)
                        && self.value_is_nonzero(divisor, query_block, &mut work)
                        && self.multiple_is_exact_scaled_divisor(
                            multiple,
                            divisor,
                            scale,
                            query_block,
                            &mut work,
                        )
                    {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Proves the unsigned Euclidean relation
    ///
    /// `(numerator * scale + extra) >= (numerator / divisor) * multiple`
    ///
    /// only when `multiple == divisor * scale`. The exact product identity may
    /// be present directly or authenticated by a dominating `multiple % scale
    /// == 0` edge together with `divisor == multiple / scale`. Every checked
    /// product/addition must also have its own exact non-overflow edge dominate
    /// the subtraction. The bounded structural search fails closed.
    fn scaled_quotient_subtraction_is_nonnegative(
        &mut self,
        query_block: BlockId,
        lhs: ValueId,
        rhs: ValueId,
    ) -> bool {
        let mut work = 0_usize;
        let Some(lhs_bases) = self.nonnegative_add_bases(lhs, query_block, &mut work) else {
            return false;
        };
        let Some(rhs_product) = self.checked_binary_operands(
            rhs,
            CheckedBinaryOperator::Multiply,
            query_block,
            &mut work,
        ) else {
            return false;
        };

        for lhs_base in lhs_bases {
            let Some(lhs_product) = self.checked_binary_operands(
                lhs_base,
                CheckedBinaryOperator::Multiply,
                query_block,
                &mut work,
            ) else {
                continue;
            };
            for (numerator, scale) in [lhs_product, (lhs_product.1, lhs_product.0)] {
                for (quotient, multiple) in [rhs_product, (rhs_product.1, rhs_product.0)] {
                    let Some((quotient_numerator, divisor)) =
                        self.binary_operands(quotient, BinaryOp::Divide, query_block, &mut work)
                    else {
                        continue;
                    };
                    if !self.values_equivalent(
                        numerator,
                        quotient_numerator,
                        query_block,
                        &mut work,
                    ) || !self.value_is_nonzero(divisor, query_block, &mut work)
                    {
                        continue;
                    }
                    if self.multiple_is_exact_scaled_divisor(
                        multiple,
                        divisor,
                        scale,
                        query_block,
                        &mut work,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn nonnegative_add_bases(
        &self,
        value: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> Option<Vec<ValueId>> {
        let Some((lhs, rhs)) =
            self.checked_binary_operands(value, CheckedBinaryOperator::Add, query_block, work)
        else {
            charge_relational_work(work, 1)?;
            return Some(vec![value]);
        };
        if self
            .value_types
            .get(&lhs)
            .and_then(unsigned_type_range)
            .is_none()
            || self
                .value_types
                .get(&rhs)
                .and_then(unsigned_type_range)
                .is_none()
        {
            return None;
        }
        Some(vec![lhs, rhs])
    }

    fn checked_binary_operands(
        &self,
        value: ValueId,
        operator: CheckedBinaryOperator,
        query_block: BlockId,
        work: &mut usize,
    ) -> Option<(ValueId, ValueId)> {
        charge_relational_work(work, 1)?;
        let value = self.normalized_unsigned_value(value, query_block, work)?;
        let definition = self.operation_definitions.get(&value)?;
        let OperationKind::Binary {
            op: BinaryOp::Checked(actual),
            lhs,
            rhs,
        } = &definition.operation.kind
        else {
            return None;
        };
        (definition.result_index == 0
            && *actual == operator
            && self.checked_result_succeeds_before(value, query_block, work))
        .then_some((*lhs, *rhs))
    }

    fn binary_operands(
        &self,
        value: ValueId,
        operator: BinaryOp,
        query_block: BlockId,
        work: &mut usize,
    ) -> Option<(ValueId, ValueId)> {
        charge_relational_work(work, 1)?;
        let value = self.normalized_unsigned_value(value, query_block, work)?;
        let definition = self.operation_definitions.get(&value)?;
        let OperationKind::Binary { op, lhs, rhs } = &definition.operation.kind else {
            return None;
        };
        (definition.result_index == 0 && *op == operator).then_some((*lhs, *rhs))
    }

    fn checked_result_succeeds_before(
        &self,
        value: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> bool {
        if charge_relational_work(work, 1).is_none() {
            return false;
        }
        let Some(definition) = self.operation_definitions.get(&value) else {
            return false;
        };
        let OperationKind::Binary {
            op: BinaryOp::Checked(_),
            ..
        } = &definition.operation.kind
        else {
            return false;
        };
        let [result, overflow] = definition.operation.results.as_slice() else {
            return false;
        };
        if definition.result_index != 0 || result.id != value || overflow.ty != Type::BOOL {
            return false;
        }
        let Some(terminator) = self
            .body
            .blocks
            .iter()
            .find(|block| block.id == definition.block)
            .and_then(|block| block.terminator.as_ref())
        else {
            return false;
        };
        match terminator {
            Terminator::ConditionalBranch {
                condition,
                else_target,
                ..
            } => {
                self.values_equivalent(*condition, overflow.id, definition.block, work)
                    && self.edge_is_exclusive(definition.block, *else_target)
                    && self.dominates(*else_target, query_block)
            }
            Terminator::Switch {
                selector, cases, ..
            } => {
                self.values_equivalent(*selector, overflow.id, definition.block, work)
                    && cases.iter().any(|case| {
                        case.value == 0
                            && self.edge_is_exclusive(definition.block, case.target)
                            && self.dominates(case.target, query_block)
                    })
            }
            Terminator::IntegerSwitch {
                selector, cases, ..
            } => {
                self.values_equivalent(*selector, overflow.id, definition.block, work)
                    && cases.iter().any(|case| {
                        known_u64_constant(&case.value) == Some(0)
                            && self.edge_is_exclusive(definition.block, case.target)
                            && self.dominates(case.target, query_block)
                    })
            }
            Terminator::Branch { .. } | Terminator::Return { .. } | Terminator::Unreachable => {
                false
            }
        }
    }

    fn multiple_is_exact_scaled_divisor(
        &mut self,
        multiple: ValueId,
        divisor: ValueId,
        scale: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> bool {
        if let Some(product) = self.checked_binary_operands(
            multiple,
            CheckedBinaryOperator::Multiply,
            query_block,
            work,
        ) {
            for (product_divisor, product_scale) in [product, (product.1, product.0)] {
                if self.values_equivalent(divisor, product_divisor, query_block, work)
                    && self.values_equivalent(scale, product_scale, query_block, work)
                {
                    return true;
                }
            }
        }

        let Some((base, divisor_scale)) =
            self.binary_operands(divisor, BinaryOp::Divide, query_block, work)
        else {
            return false;
        };
        self.values_equivalent(multiple, base, query_block, work)
            && self.values_equivalent(scale, divisor_scale, query_block, work)
            && self.value_is_nonzero(scale, query_block, work)
            && self.remainder_is_zero(base, scale, query_block, work)
    }

    fn remainder_is_zero(
        &mut self,
        numerator: ValueId,
        divisor: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> bool {
        let candidates = self
            .operation_definitions
            .iter()
            .filter_map(|(result, definition)| {
                let OperationKind::Binary {
                    op: BinaryOp::Remainder,
                    lhs,
                    rhs,
                } = &definition.operation.kind
                else {
                    return None;
                };
                (definition.result_index == 0).then_some((*result, *lhs, *rhs))
            })
            .collect::<Vec<_>>();
        if charge_relational_work(work, candidates.len()).is_none() {
            return false;
        }
        for (result, lhs, rhs) in candidates {
            if self.values_equivalent(numerator, lhs, query_block, work)
                && self.values_equivalent(divisor, rhs, query_block, work)
                && self.range_at(result, query_block) == Some(UnsignedRange::exact(0))
            {
                return true;
            }
        }
        false
    }

    fn value_is_nonzero(&mut self, value: ValueId, query_block: BlockId, work: &mut usize) -> bool {
        if self
            .range_at(value, query_block)
            .is_some_and(|range| range.min != 0)
        {
            return true;
        }
        let blocks = self
            .body
            .blocks
            .iter()
            .filter_map(|block| {
                block
                    .terminator
                    .as_ref()
                    .map(|terminator| (block.id, terminator))
            })
            .collect::<Vec<_>>();
        if charge_relational_work(work, blocks.len()).is_none() {
            return false;
        }
        for (source, terminator) in blocks {
            match terminator {
                Terminator::Switch {
                    selector,
                    cases,
                    default_target,
                    ..
                } if self.values_equivalent(*selector, value, source, work)
                    && cases.iter().any(|case| case.value == 0)
                    && self.edge_is_exclusive(source, *default_target)
                    && self.dominates(*default_target, query_block) =>
                {
                    return true;
                }
                Terminator::IntegerSwitch {
                    selector,
                    cases,
                    default_target,
                    ..
                } if self.values_equivalent(*selector, value, source, work)
                    && cases
                        .iter()
                        .any(|case| known_u64_constant(&case.value) == Some(0))
                    && self.edge_is_exclusive(source, *default_target)
                    && self.dominates(*default_target, query_block) =>
                {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    fn values_equivalent(
        &self,
        lhs: ValueId,
        rhs: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> bool {
        let Some(lhs) = self.normalized_unsigned_value(lhs, query_block, work) else {
            return false;
        };
        let Some(rhs) = self.normalized_unsigned_value(rhs, query_block, work) else {
            return false;
        };
        lhs == rhs
            || self
                .unsigned_constant_value(lhs)
                .zip(self.unsigned_constant_value(rhs))
                .is_some_and(|(lhs, rhs)| lhs == rhs)
    }

    fn normalized_unsigned_value(
        &self,
        mut value: ValueId,
        query_block: BlockId,
        work: &mut usize,
    ) -> Option<ValueId> {
        let mut seen = BTreeSet::new();
        while seen.insert(value) {
            charge_relational_work(work, 1)?;
            let representative = resolve_representative(self.trivial_phi_representatives, value);
            if representative != value {
                value = representative;
                continue;
            }
            if let Some(slot) = self.private_load_slots.get(&value)
                && let Some([store]) = self.private_slot_stores.get(slot).map(Vec::as_slice)
                && store.block != query_block
                && self.dominates(store.block, query_block)
            {
                value = store.value;
                continue;
            }
            let Some(definition) = self.operation_definitions.get(&value) else {
                break;
            };
            let OperationKind::Cast {
                kind,
                value: source,
                to,
            } = &definition.operation.kind
            else {
                break;
            };
            let source_type = self.value_types.get(source)?;
            let source_range = unsigned_type_range(source_type)?;
            let target_range = unsigned_type_range(to)?;
            let preserves_value = match kind {
                CastKind::ZeroExtend => source_range.max <= target_range.max,
                CastKind::Bitcast => source_range.max == target_range.max,
                CastKind::RestrictPointerAccess
                | CastKind::PointerToGeneric
                | CastKind::SliceToGeneric
                | CastKind::Truncate
                | CastKind::SignExtend
                | CastKind::FloatExtend
                | CastKind::FloatTruncate
                | CastKind::IntegerToFloat
                | CastKind::FloatToInteger => false,
            };
            if !preserves_value {
                break;
            }
            value = *source;
        }
        self.value_types
            .get(&value)
            .and_then(unsigned_type_range)
            .map(|_| value)
    }

    fn unsigned_constant_value(&self, value: ValueId) -> Option<u128> {
        let definition = self.operation_definitions.get(&value)?;
        let OperationKind::Constant(constant) = &definition.operation.kind else {
            return None;
        };
        unsigned_constant_range(constant)
            .and_then(|range| (range.min == range.max).then_some(range.min))
    }

    fn range_at(&mut self, value: ValueId, query_block: BlockId) -> Option<UnsignedRange> {
        let key = (value, query_block);
        if let Some(range) = self.cache.get(&key).copied() {
            return Some(range);
        }
        let type_range = self.value_types.get(&value).and_then(unsigned_type_range)?;
        if !self.visiting.insert(key) {
            return Some(type_range);
        }

        let mut range = self
            .expression_range(value, query_block)
            .unwrap_or(type_range)
            .intersect(type_range);
        let guards = self.guards.clone();
        for guard in guards {
            let then_dominates =
                guard.then_edge_is_exclusive && self.dominates(guard.then_target, query_block);
            let else_dominates =
                guard.else_edge_is_exclusive && self.dominates(guard.else_target, query_block);
            let predicate = match (then_dominates, else_dominates) {
                (true, false) => guard.predicate,
                (false, true) => invert_predicate(guard.predicate),
                _ => continue,
            };
            let value_identity = self.immutable_identity(value);
            let lhs_identity = self.immutable_identity(guard.lhs);
            let rhs_identity = self.immutable_identity(guard.rhs);
            let (predicate, other) = if value_identity == lhs_identity {
                (predicate, guard.rhs)
            } else if value_identity == rhs_identity {
                (swap_predicate(predicate), guard.lhs)
            } else {
                continue;
            };
            if let Some(other) = self.range_at(other, guard.block) {
                range = refine_unsigned_range(range, predicate, other);
            }
        }

        self.visiting.remove(&key);
        self.cache.insert(key, range);
        Some(range)
    }

    fn expression_range(&mut self, value: ValueId, query_block: BlockId) -> Option<UnsignedRange> {
        if let Some((block, index)) = self.block_parameters.get(&value).copied() {
            let edges = self.incoming.get(&block)?.clone();
            return edges
                .into_iter()
                .filter_map(|edge| {
                    edge.arguments
                        .get(index)
                        .copied()
                        .and_then(|argument| self.range_at(argument, edge.source))
                })
                .reduce(UnsignedRange::join);
        }

        let definition = self.operation_definitions.get(&value).copied()?;
        let kind = definition.operation.kind.clone();
        let result_type = self.value_types.get(&value)?.clone();
        match kind {
            OperationKind::Constant(constant) => unsigned_constant_range(&constant),
            OperationKind::Binary { op, lhs, rhs } if definition.result_index == 0 => {
                let lhs = self.range_at(lhs, query_block)?;
                let rhs = self.range_at(rhs, query_block)?;
                binary_result_range(op, lhs, rhs, unsigned_type_range(&result_type)?.max)
            }
            OperationKind::Cast {
                kind,
                value: source,
                ..
            } => {
                let source_type = self.value_types.get(&source)?.clone();
                let source = self.range_at(source, query_block)?;
                cast_result_range(kind, &source_type, &result_type, source)
            }
            OperationKind::Select {
                true_value,
                false_value,
                ..
            } => Some(
                self.range_at(true_value, query_block)?
                    .join(self.range_at(false_value, query_block)?),
            ),
            OperationKind::Load { .. } => {
                let slot = *self.private_load_slots.get(&value)?;
                self.private_slot_stores
                    .get(&slot)?
                    .clone()
                    .into_iter()
                    .filter_map(|store| self.range_at(store.value, query_block))
                    .reduce(UnsignedRange::join)
            }
            _ => None,
        }
    }

    fn immutable_identity(&self, mut value: ValueId) -> ValueId {
        let mut seen = BTreeSet::new();
        while seen.insert(value) {
            let representative = resolve_representative(self.trivial_phi_representatives, value);
            if representative != value {
                value = representative;
                continue;
            }
            let Some(slot) = self.private_load_slots.get(&value) else {
                break;
            };
            let Some([store]) = self.private_slot_stores.get(slot).map(Vec::as_slice) else {
                break;
            };
            value = store.value;
        }
        value
    }

    fn dominates(&self, dominator: BlockId, block: BlockId) -> bool {
        self.dominators
            .get(&block)
            .is_some_and(|dominators| dominators.contains(&dominator))
    }
}

fn unsigned_type_range(ty: &Type) -> Option<UnsignedRange> {
    let max = match ty.as_scalar()? {
        ScalarType::Bool => 1,
        ScalarType::U8 => u8::MAX.into(),
        ScalarType::U16 => u16::MAX.into(),
        ScalarType::U32 => u32::MAX.into(),
        ScalarType::U64 | ScalarType::Index => u64::MAX.into(),
        ScalarType::U128 => u128::MAX,
        ScalarType::I8
        | ScalarType::I16
        | ScalarType::I32
        | ScalarType::I64
        | ScalarType::I128
        | ScalarType::F16
        | ScalarType::Bf16
        | ScalarType::F32
        | ScalarType::F64 => return None,
    };
    Some(UnsignedRange { min: 0, max })
}

fn unsigned_constant_range(constant: &Constant) -> Option<UnsignedRange> {
    let value = match constant {
        Constant::Bool(value) => u128::from(*value),
        Constant::U8(value) => (*value).into(),
        Constant::U16(value) => (*value).into(),
        Constant::U32(value) => (*value).into(),
        Constant::U64(value) | Constant::Index(value) => (*value).into(),
        Constant::I8(_)
        | Constant::I16(_)
        | Constant::I32(_)
        | Constant::I64(_)
        | Constant::F16Bits(_)
        | Constant::Bf16Bits(_)
        | Constant::F32Bits(_)
        | Constant::F64Bits(_) => return None,
    };
    Some(UnsignedRange::exact(value))
}

fn binary_result_range(
    operator: BinaryOp,
    lhs: UnsignedRange,
    rhs: UnsignedRange,
    type_max: u128,
) -> Option<UnsignedRange> {
    match operator {
        BinaryOp::Add => checked_result_range(CheckedBinaryOperator::Add, lhs, rhs, type_max),
        BinaryOp::Subtract => {
            checked_result_range(CheckedBinaryOperator::Subtract, lhs, rhs, type_max)
        }
        BinaryOp::Multiply => {
            checked_result_range(CheckedBinaryOperator::Multiply, lhs, rhs, type_max)
        }
        BinaryOp::Checked(operator) => checked_result_range(operator, lhs, rhs, type_max),
        BinaryOp::Divide if rhs.min != 0 => Some(UnsignedRange {
            min: lhs.min / rhs.max,
            max: lhs.max / rhs.min,
        }),
        BinaryOp::Remainder if rhs.min != 0 => Some(UnsignedRange {
            min: 0,
            max: lhs.max.min(rhs.max - 1),
        }),
        BinaryOp::BitAnd => Some(UnsignedRange {
            min: 0,
            max: lhs.max.min(rhs.max).min(type_max),
        }),
        BinaryOp::Divide
        | BinaryOp::Remainder
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::ShiftLeft
        | BinaryOp::ShiftRight => None,
    }
}

fn checked_result_range(
    operator: CheckedBinaryOperator,
    lhs: UnsignedRange,
    rhs: UnsignedRange,
    type_max: u128,
) -> Option<UnsignedRange> {
    let range = match operator {
        CheckedBinaryOperator::Add => UnsignedRange {
            min: lhs.min.checked_add(rhs.min)?,
            max: lhs.max.checked_add(rhs.max)?,
        },
        CheckedBinaryOperator::Subtract if lhs.min >= rhs.max => UnsignedRange {
            min: lhs.min - rhs.max,
            max: lhs.max - rhs.min,
        },
        CheckedBinaryOperator::Multiply => UnsignedRange {
            min: lhs.min.checked_mul(rhs.min)?,
            max: lhs.max.checked_mul(rhs.max)?,
        },
        CheckedBinaryOperator::Subtract => return None,
    };
    (range.max <= type_max).then_some(range)
}

fn cast_result_range(
    kind: CastKind,
    source_type: &Type,
    target_type: &Type,
    source: UnsignedRange,
) -> Option<UnsignedRange> {
    let source_type = unsigned_type_range(source_type)?;
    let target_type = unsigned_type_range(target_type)?;
    match kind {
        CastKind::ZeroExtend if source_type.max <= target_type.max => Some(source),
        CastKind::Truncate if source.max <= target_type.max => Some(source),
        CastKind::Bitcast if source_type.max == target_type.max => Some(source),
        CastKind::Truncate => Some(target_type),
        CastKind::RestrictPointerAccess
        | CastKind::PointerToGeneric
        | CastKind::SliceToGeneric
        | CastKind::ZeroExtend
        | CastKind::SignExtend
        | CastKind::FloatExtend
        | CastKind::FloatTruncate
        | CastKind::IntegerToFloat
        | CastKind::FloatToInteger
        | CastKind::Bitcast => None,
    }
}

fn refine_unsigned_range(
    range: UnsignedRange,
    predicate: ComparePredicate,
    other: UnsignedRange,
) -> UnsignedRange {
    let refined = match predicate {
        ComparePredicate::Equal => range.intersect(other),
        ComparePredicate::NotEqual if other.min == 0 && other.max == 0 && range.min == 0 => {
            UnsignedRange {
                min: 1,
                max: range.max,
            }
        }
        ComparePredicate::NotEqual
            if other.min == range.max && other.max == range.max && range.min != range.max =>
        {
            UnsignedRange {
                min: range.min,
                max: range.max - 1,
            }
        }
        ComparePredicate::LessThan if other.max != 0 => UnsignedRange {
            min: range.min,
            max: range.max.min(other.max - 1),
        },
        ComparePredicate::LessThanOrEqual => UnsignedRange {
            min: range.min,
            max: range.max.min(other.max),
        },
        ComparePredicate::GreaterThan if other.min != u128::MAX => UnsignedRange {
            min: range.min.max(other.min + 1),
            max: range.max,
        },
        ComparePredicate::GreaterThanOrEqual => UnsignedRange {
            min: range.min.max(other.min),
            max: range.max,
        },
        ComparePredicate::NotEqual | ComparePredicate::LessThan | ComparePredicate::GreaterThan => {
            return range;
        }
    };
    range.intersect(refined)
}

const fn invert_predicate(predicate: ComparePredicate) -> ComparePredicate {
    match predicate {
        ComparePredicate::Equal => ComparePredicate::NotEqual,
        ComparePredicate::NotEqual => ComparePredicate::Equal,
        ComparePredicate::LessThan => ComparePredicate::GreaterThanOrEqual,
        ComparePredicate::LessThanOrEqual => ComparePredicate::GreaterThan,
        ComparePredicate::GreaterThan => ComparePredicate::LessThanOrEqual,
        ComparePredicate::GreaterThanOrEqual => ComparePredicate::LessThan,
    }
}

const fn swap_predicate(predicate: ComparePredicate) -> ComparePredicate {
    match predicate {
        ComparePredicate::Equal => ComparePredicate::Equal,
        ComparePredicate::NotEqual => ComparePredicate::NotEqual,
        ComparePredicate::LessThan => ComparePredicate::GreaterThan,
        ComparePredicate::LessThanOrEqual => ComparePredicate::GreaterThanOrEqual,
        ComparePredicate::GreaterThan => ComparePredicate::LessThan,
        ComparePredicate::GreaterThanOrEqual => ComparePredicate::LessThanOrEqual,
    }
}

fn private_storage_facts(
    body: &FunctionBody,
    reachable: &BTreeSet<BlockId>,
    dominators: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    trivial_phi_representatives: &BTreeMap<ValueId, ValueId>,
    value_types: &BTreeMap<ValueId, Type>,
) -> (
    BTreeMap<ValueId, ValueId>,
    BTreeMap<ValueId, Vec<PrivateStore>>,
) {
    let blocks = body
        .blocks
        .iter()
        .map(|block| (block.id, block))
        .collect::<BTreeMap<_, _>>();
    let slots = body
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id))
        .flat_map(|block| &block.operations)
        .filter(|operation| {
            matches!(
                operation.kind,
                OperationKind::Alloca {
                    count: None,
                    address_space: AddressSpace::Private,
                    ..
                }
            )
        })
        .flat_map(|operation| operation.results.iter().map(|result| result.id))
        .collect::<BTreeSet<_>>();
    let exact_slot = |value| {
        let origin = resolve_representative(trivial_phi_representatives, value);
        let same_type = value_types
            .get(&value)
            .zip(value_types.get(&origin))
            .is_some_and(|(value, origin)| value == origin);
        (same_type && slots.contains(&origin)).then_some(origin)
    };
    let mut escaped = BTreeSet::new();
    for block in body
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id))
    {
        for operation in &block.operations {
            for operand in operation.kind.operands() {
                if let Some(slot) = exact_slot(operand)
                    && !is_direct_private_slot_access(&operation.kind, operand)
                {
                    escaped.insert(slot);
                }
            }
        }
        if let Some(terminator) = &block.terminator {
            if let Some(selector) = discriminator(terminator)
                && let Some(slot) = exact_slot(selector)
            {
                escaped.insert(slot);
            }
            if let Terminator::Return { values } = terminator {
                escaped.extend(values.iter().filter_map(|value| exact_slot(*value)));
            }
            for (target, arguments) in terminator_edges(terminator) {
                let parameters = blocks.get(&target).map(|block| block.parameters.as_slice());
                let exact_edge = parameters.is_some_and(|parameters| {
                    edge_has_exact_types(&arguments, parameters, value_types)
                });
                for (index, argument) in arguments.into_iter().enumerate() {
                    let Some(slot) = exact_slot(argument) else {
                        continue;
                    };
                    let exact_transport = exact_edge
                        && parameters
                            .and_then(|parameters| parameters.get(index))
                            .is_some_and(|parameter| exact_slot(parameter.id) == Some(slot));
                    if !exact_transport {
                        escaped.insert(slot);
                    }
                }
            }
        }
    }

    let eligible = slots.difference(&escaped).copied().collect::<BTreeSet<_>>();
    let mut stores = BTreeMap::<ValueId, Vec<PrivateStore>>::new();
    let mut loads = Vec::new();
    for block in body
        .blocks
        .iter()
        .filter(|block| reachable.contains(&block.id))
    {
        for (operation_index, operation) in block.operations.iter().enumerate() {
            match &operation.kind {
                OperationKind::Store {
                    pointer,
                    value,
                    access,
                } if exact_slot(*pointer).is_some_and(|slot| eligible.contains(&slot))
                    && access.address_space == AddressSpace::Private =>
                {
                    let slot = exact_slot(*pointer).expect("eligible private pointer has a slot");
                    stores.entry(slot).or_default().push(PrivateStore {
                        block: block.id,
                        operation_index,
                        value: *value,
                    });
                }
                OperationKind::Load { pointer, access }
                    if exact_slot(*pointer).is_some_and(|slot| eligible.contains(&slot))
                        && access.address_space == AddressSpace::Private =>
                {
                    let slot = exact_slot(*pointer).expect("eligible private pointer has a slot");
                    loads.extend(
                        operation
                            .results
                            .iter()
                            .map(|result| (result.id, slot, block.id, operation_index)),
                    );
                }
                _ => {}
            }
        }
    }

    let private_load_slots = loads
        .into_iter()
        .filter(|(_, slot, load_block, load_index)| {
            stores.get(slot).is_some_and(|slot_stores| {
                slot_stores.iter().any(|store| {
                    dominators
                        .get(load_block)
                        .is_some_and(|set| set.contains(&store.block))
                        && (store.block != *load_block || store.operation_index < *load_index)
                })
            })
        })
        .map(|(result, slot, _, _)| (result, slot))
        .collect();
    (private_load_slots, stores)
}

fn is_direct_private_slot_access(kind: &OperationKind, slot: ValueId) -> bool {
    match kind {
        OperationKind::Load { pointer, access } => {
            *pointer == slot && access.address_space == AddressSpace::Private
        }
        OperationKind::Store {
            pointer,
            value,
            access,
        } => *pointer == slot && *value != slot && access.address_space == AddressSpace::Private,
        _ => false,
    }
}

const MAX_UNIFORM_RECURRENCE_PROOF_WORK: usize = 65_536;

/// Authenticates unsigned induction backedges of the form `next = checked_add(phi, step)`.
///
/// The returned edges may omit predecessor control when classifying the phi's
/// value: their argument is determined by the prior phi value and the step,
/// whose variation is still propagated normally. Reachability control is not
/// changed, so divergent loop exits continue to make barriers non-convergent.
/// Every backedge of a proven phi must use an exact checked-add result reached
/// through that operation's exclusive non-overflow edge.
fn uniform_recurrence_edges(
    function: &Function,
    body: &FunctionBody,
    incoming: &BTreeMap<BlockId, Vec<Edge>>,
    dominators: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    trivial_phi_representatives: &BTreeMap<ValueId, ValueId>,
) -> BTreeSet<(ValueId, BlockId, ValueId)> {
    let mut work = 0_usize;
    let mut blocks = BTreeMap::new();
    let mut value_types = BTreeMap::new();
    let mut operation_definitions = BTreeMap::new();

    if body.parameters.len() != function.signature.parameters.len() {
        return BTreeSet::new();
    }
    for (value, ty) in body
        .parameters
        .iter()
        .copied()
        .zip(function.signature.parameters.iter().cloned())
    {
        if value_types.insert(value, ty).is_some() {
            return BTreeSet::new();
        }
    }
    for block in &body.blocks {
        if blocks.insert(block.id, block).is_some() {
            return BTreeSet::new();
        }
        for parameter in &block.parameters {
            if value_types
                .insert(parameter.id, parameter.ty.clone())
                .is_some()
            {
                return BTreeSet::new();
            }
        }
        for operation in &block.operations {
            for (result_index, result) in operation.results.iter().enumerate() {
                if value_types.insert(result.id, result.ty.clone()).is_some()
                    || operation_definitions
                        .insert(
                            result.id,
                            OperationDefinition {
                                block: block.id,
                                result_index,
                                operation,
                            },
                        )
                        .is_some()
                {
                    return BTreeSet::new();
                }
            }
        }
    }

    let dominates = |dominator, block| {
        dominators
            .get(&block)
            .is_some_and(|set| set.contains(&dominator))
    };
    let edge_is_exclusive = |source, target| {
        incoming
            .get(&target)
            .is_some_and(|edges| matches!(edges.as_slice(), [edge] if edge.source == source))
    };
    let checked_add_succeeds_before = |value: ValueId,
                                       query_block: BlockId|
     -> Option<(ValueId, ValueId)> {
        let definition = operation_definitions.get(&value)?;
        let OperationKind::Binary {
            op: BinaryOp::Checked(CheckedBinaryOperator::Add),
            lhs,
            rhs,
        } = &definition.operation.kind
        else {
            return None;
        };
        let [result, overflow] = definition.operation.results.as_slice() else {
            return None;
        };
        if definition.result_index != 0
            || result.id != value
            || overflow.ty != Type::BOOL
            || !dominates(definition.block, query_block)
        {
            return None;
        }
        let terminator = blocks.get(&definition.block)?.terminator.as_ref()?;
        let exact_overflow = |selector: ValueId| {
            resolve_representative(trivial_phi_representatives, selector)
                == resolve_representative(trivial_phi_representatives, overflow.id)
        };
        let success_dominates =
            |target| edge_is_exclusive(definition.block, target) && dominates(target, query_block);
        let succeeds = match terminator {
            Terminator::ConditionalBranch {
                condition,
                else_target,
                ..
            } => exact_overflow(*condition) && success_dominates(*else_target),
            Terminator::Switch {
                selector, cases, ..
            } => {
                exact_overflow(*selector)
                    && cases
                        .iter()
                        .any(|case| case.value == 0 && success_dominates(case.target))
            }
            Terminator::IntegerSwitch {
                selector, cases, ..
            } => {
                exact_overflow(*selector)
                    && cases.iter().any(|case| {
                        known_u64_constant(&case.value) == Some(0) && success_dominates(case.target)
                    })
            }
            Terminator::Branch { .. } | Terminator::Return { .. } | Terminator::Unreachable => {
                false
            }
        };
        succeeds.then_some((*lhs, *rhs))
    };

    let mut proven = BTreeSet::new();
    for header in &body.blocks {
        let Some(edges) = incoming.get(&header.id) else {
            continue;
        };
        work = match work.checked_add(header.parameters.len().saturating_mul(edges.len())) {
            Some(next) if next <= MAX_UNIFORM_RECURRENCE_PROOF_WORK => next,
            _ => return BTreeSet::new(),
        };
        if edges
            .iter()
            .any(|edge| edge.arguments.len() != header.parameters.len())
        {
            continue;
        }
        let backedges = edges
            .iter()
            .filter(|edge| dominates(header.id, edge.source))
            .collect::<Vec<_>>();
        if backedges.is_empty() || backedges.len() == edges.len() {
            continue;
        }

        for (index, parameter) in header.parameters.iter().enumerate() {
            if unsigned_type_range(&parameter.ty).is_none() {
                continue;
            }
            let recurrence_edges = backedges
                .iter()
                .filter_map(|edge| {
                    let argument = *edge.arguments.get(index)?;
                    let recurrence_value =
                        resolve_representative(trivial_phi_representatives, argument);
                    let (lhs, rhs) = checked_add_succeeds_before(recurrence_value, edge.source)?;
                    let lhs_is_phi =
                        resolve_representative(trivial_phi_representatives, lhs) == parameter.id;
                    let rhs_is_phi =
                        resolve_representative(trivial_phi_representatives, rhs) == parameter.id;
                    let operands_match_type = value_types.get(&lhs) == Some(&parameter.ty)
                        && value_types.get(&rhs) == Some(&parameter.ty)
                        && value_types.get(&recurrence_value) == Some(&parameter.ty);
                    (operands_match_type && (lhs_is_phi || rhs_is_phi)).then_some((
                        parameter.id,
                        edge.source,
                        argument,
                    ))
                })
                .collect::<Vec<_>>();
            if recurrence_edges.len() == backedges.len() {
                proven.extend(recurrence_edges);
            }
        }
    }
    proven
}

fn compute_dominators(
    body: &FunctionBody,
    reachable: &BTreeSet<BlockId>,
    incoming: &BTreeMap<BlockId, Vec<Edge>>,
) -> BTreeMap<BlockId, BTreeSet<BlockId>> {
    let Some(entry) = body.blocks.first().map(|block| block.id) else {
        return BTreeMap::new();
    };
    let mut facts = reachable
        .iter()
        .map(|block| {
            let initial = if *block == entry {
                BTreeSet::from([entry])
            } else {
                reachable.clone()
            };
            (*block, initial)
        })
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for block in reachable.iter().copied().filter(|block| *block != entry) {
            let mut predecessor_sets = incoming
                .get(&block)
                .into_iter()
                .flat_map(|edges| edges.iter())
                .filter_map(|edge| facts.get(&edge.source));
            let mut next = predecessor_sets.next().cloned().unwrap_or_default();
            for predecessor_set in predecessor_sets {
                next.retain(|candidate| predecessor_set.contains(candidate));
            }
            next.insert(block);
            if facts.get(&block) != Some(&next) {
                facts.insert(block, next);
                changed = true;
            }
        }
        if !changed {
            return facts;
        }
    }
}

fn subgroup_collective_variation(input: Variation) -> Variation {
    if input.is_uniform_for(fe2o3_kernel_ir::SynchronizationScope::Subgroup) {
        input
    } else {
        Variation::SubgroupUniform
    }
}

fn raise<K: Ord + Copy>(facts: &mut BTreeMap<K, Variation>, key: K, value: Variation) -> bool {
    let current = facts.get(&key).copied().unwrap_or(Variation::GridUniform);
    let joined = current.join(value);
    if joined == current {
        false
    } else {
        facts.insert(key, joined);
        true
    }
}

fn join_values(
    values: impl IntoIterator<Item = ValueId>,
    facts: &BTreeMap<ValueId, Variation>,
) -> Variation {
    values
        .into_iter()
        .map(|value| facts.get(&value).copied().unwrap_or(Variation::Varying))
        .fold(Variation::GridUniform, Variation::join)
}

fn discriminator(terminator: &Terminator) -> Option<ValueId> {
    match terminator {
        Terminator::ConditionalBranch { condition, .. } => Some(*condition),
        Terminator::Switch { selector, .. } | Terminator::IntegerSwitch { selector, .. } => {
            Some(*selector)
        }
        Terminator::Branch { .. } | Terminator::Return { .. } | Terminator::Unreachable => None,
    }
}

fn reachable_blocks(
    body: &FunctionBody,
    blocks: &BTreeMap<BlockId, &BasicBlock>,
) -> BTreeSet<BlockId> {
    let Some(entry) = body.blocks.first() else {
        return BTreeSet::new();
    };
    let mut reachable = BTreeSet::new();
    let mut pending = VecDeque::from([entry.id]);
    while let Some(block_id) = pending.pop_front() {
        if !reachable.insert(block_id) {
            continue;
        }
        let Some(block) = blocks.get(&block_id) else {
            continue;
        };
        if let Some(terminator) = &block.terminator {
            for successor in terminator.successors() {
                if blocks.contains_key(&successor) && !reachable.contains(&successor) {
                    pending.push_back(successor);
                }
            }
        }
    }
    reachable
}

fn index_value_types(function: &Function, body: &FunctionBody) -> (BTreeMap<ValueId, Type>, bool) {
    let mut value_types = BTreeMap::new();
    let mut malformed = body.parameters.len() != function.signature.parameters.len();
    let mut insert = |value, ty| {
        if value_types.insert(value, ty).is_some() {
            malformed = true;
        }
    };

    for (value, ty) in body
        .parameters
        .iter()
        .copied()
        .zip(function.signature.parameters.iter().cloned())
    {
        insert(value, ty);
    }
    for block in &body.blocks {
        for parameter in &block.parameters {
            insert(parameter.id, parameter.ty.clone());
        }
        for result in block
            .operations
            .iter()
            .flat_map(|operation| &operation.results)
        {
            insert(result.id, result.ty.clone());
        }
    }

    (value_types, malformed)
}

fn edge_has_exact_types(
    arguments: &[ValueId],
    parameters: &[fe2o3_kernel_ir::ValueDef],
    value_types: &BTreeMap<ValueId, Type>,
) -> bool {
    arguments.len() == parameters.len()
        && arguments
            .iter()
            .zip(parameters)
            .all(|(argument, parameter)| {
                value_types
                    .get(argument)
                    .is_some_and(|argument_type| argument_type == &parameter.ty)
            })
}

fn incoming_edges(
    body: &FunctionBody,
    blocks: &BTreeMap<BlockId, &BasicBlock>,
    reachable: &BTreeSet<BlockId>,
    value_types: &BTreeMap<ValueId, Type>,
) -> (BTreeMap<BlockId, Vec<Edge>>, bool) {
    let mut incoming = BTreeMap::<BlockId, Vec<Edge>>::new();
    let mut malformed_targets = BTreeSet::new();
    let mut malformed = false;
    for block in &body.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let Some(terminator) = &block.terminator else {
            malformed = true;
            continue;
        };
        let edge_discriminator = discriminator(terminator);
        for (target, arguments) in terminator_edges(terminator) {
            let Some(target_block) = blocks.get(&target) else {
                malformed = true;
                continue;
            };
            if !edge_has_exact_types(&arguments, &target_block.parameters, value_types) {
                malformed = true;
                malformed_targets.insert(target);
                incoming.remove(&target);
                continue;
            }
            if malformed_targets.contains(&target) {
                continue;
            }
            incoming.entry(target).or_default().push(Edge {
                source: block.id,
                arguments,
                discriminator: edge_discriminator,
            });
        }
    }
    (incoming, malformed)
}

fn trivial_phi_representatives(
    body: &FunctionBody,
    incoming: &BTreeMap<BlockId, Vec<Edge>>,
    value_types: &BTreeMap<ValueId, Type>,
) -> BTreeMap<ValueId, ValueId> {
    let mut representatives = BTreeMap::new();
    loop {
        let mut changed = false;
        for block in &body.blocks {
            let Some(edges) = incoming.get(&block.id) else {
                continue;
            };
            if edges
                .iter()
                .any(|edge| !edge_has_exact_types(&edge.arguments, &block.parameters, value_types))
            {
                continue;
            }
            for (index, parameter) in block.parameters.iter().enumerate() {
                let origins = edges
                    .iter()
                    .filter_map(|edge| edge.arguments.get(index).copied())
                    .map(|value| resolve_representative(&representatives, value))
                    .filter(|value| *value != parameter.id)
                    .collect::<BTreeSet<_>>();
                if origins.len() == 1 {
                    let origin = *origins.first().expect("singleton origin");
                    if representatives.get(&parameter.id) != Some(&origin) {
                        representatives.insert(parameter.id, origin);
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            return representatives;
        }
    }
}

fn resolve_representative(
    representatives: &BTreeMap<ValueId, ValueId>,
    mut value: ValueId,
) -> ValueId {
    let mut remaining = representatives.len();
    while remaining != 0 {
        let Some(next) = representatives.get(&value).copied() else {
            break;
        };
        if next == value {
            break;
        }
        value = next;
        remaining -= 1;
    }
    value
}

fn terminator_edges(terminator: &Terminator) -> Vec<(BlockId, Vec<ValueId>)> {
    match terminator {
        Terminator::Branch { target, arguments } => vec![(*target, arguments.clone())],
        Terminator::ConditionalBranch {
            then_target,
            then_arguments,
            else_target,
            else_arguments,
            ..
        } => vec![
            (*then_target, then_arguments.clone()),
            (*else_target, else_arguments.clone()),
        ],
        Terminator::Switch {
            cases,
            default_target,
            default_arguments,
            ..
        } => cases
            .iter()
            .map(|case| (case.target, case.arguments.clone()))
            .chain([(*default_target, default_arguments.clone())])
            .collect(),
        Terminator::IntegerSwitch {
            cases,
            default_target,
            default_arguments,
            ..
        } => cases
            .iter()
            .map(|case| (case.target, case.arguments.clone()))
            .chain([(*default_target, default_arguments.clone())])
            .collect(),
        Terminator::Return { .. } | Terminator::Unreachable => Vec::new(),
    }
}

fn control_regions(
    body: &FunctionBody,
    blocks: &BTreeMap<BlockId, &BasicBlock>,
    reachable: &BTreeSet<BlockId>,
    postdominance_available: &BTreeSet<BlockId>,
    natural_loop_nests: &BTreeMap<BlockId, BTreeSet<BlockId>>,
    effective_successors: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> BTreeMap<BlockId, BTreeSet<BlockId>> {
    let postdominators = postdominators(blocks, postdominance_available, effective_successors);
    let mut regions = BTreeMap::new();
    for block in &body.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let Some(terminator) = &block.terminator else {
            continue;
        };
        if discriminator(terminator).is_none() {
            continue;
        }
        let mut stop = immediate_postdominator(block.id, &postdominators);
        let collect_region = |stop: Option<BlockId>| {
            let mut region = BTreeSet::new();
            let mut pending =
                VecDeque::from(successors_for(block.id, terminator, effective_successors));
            while let Some(candidate) = pending.pop_front() {
                if Some(candidate) == stop
                    || !reachable.contains(&candidate)
                    || !region.insert(candidate)
                {
                    continue;
                }
                if let Some(candidate_block) = blocks.get(&candidate)
                    && let Some(candidate_terminator) = &candidate_block.terminator
                {
                    pending.extend(successors_for(
                        candidate,
                        candidate_terminator,
                        effective_successors,
                    ));
                }
            }
            region
        };
        let mut region = collect_region(stop);
        if let Some(stop_block) = stop {
            let source_loops = natural_loop_nests.get(&block.id);
            let stop_loops = natural_loop_nests.get(&stop_block);
            let exits_containing_loop = source_loops.is_some_and(|source_loops| {
                source_loops
                    .iter()
                    .any(|header| !stop_loops.is_some_and(|stop_loops| stop_loops.contains(header)))
            });
            if exits_containing_loop {
                stop = None;
                region = collect_region(stop);
            }
        }
        regions.insert(block.id, region);
    }
    regions
}

fn postdominance_available(
    blocks: &BTreeMap<BlockId, &BasicBlock>,
    reachable: &BTreeSet<BlockId>,
    effective_successors: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> BTreeSet<BlockId> {
    let mut available = BTreeSet::new();
    loop {
        let mut changed = false;
        for block_id in reachable.iter().rev() {
            if available.contains(block_id) {
                continue;
            }
            let successors = blocks
                .get(block_id)
                .and_then(|block| block.terminator.as_ref())
                .map(|terminator| successors_for(*block_id, terminator, effective_successors))
                .unwrap_or_default();
            if successors.is_empty()
                || successors
                    .iter()
                    .any(|successor| available.contains(successor))
            {
                available.insert(*block_id);
                changed = true;
            }
        }
        if !changed {
            return available;
        }
    }
}

fn postdominators(
    blocks: &BTreeMap<BlockId, &BasicBlock>,
    reachable: &BTreeSet<BlockId>,
    effective_successors: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> BTreeMap<BlockId, BTreeSet<BlockId>> {
    let all = reachable.clone();
    let mut facts = reachable
        .iter()
        .map(|block| (*block, all.clone()))
        .collect::<BTreeMap<_, _>>();

    loop {
        let mut changed = false;
        for block_id in reachable.iter().rev() {
            let successors = blocks
                .get(block_id)
                .and_then(|block| block.terminator.as_ref())
                .map(|terminator| successors_for(*block_id, terminator, effective_successors))
                .unwrap_or_default()
                .into_iter()
                .filter(|successor| reachable.contains(successor))
                .collect::<Vec<_>>();
            let mut next = if successors.is_empty() {
                BTreeSet::new()
            } else {
                let mut successor_sets = successors.iter().map(|successor| &facts[successor]);
                let mut intersection = successor_sets.next().cloned().unwrap_or_default();
                for successor_set in successor_sets {
                    intersection.retain(|candidate| successor_set.contains(candidate));
                }
                intersection
            };
            next.insert(*block_id);
            if facts.get(block_id) != Some(&next) {
                facts.insert(*block_id, next);
                changed = true;
            }
        }
        if !changed {
            return facts;
        }
    }
}

fn immediate_postdominator(
    block: BlockId,
    postdominators: &BTreeMap<BlockId, BTreeSet<BlockId>>,
) -> Option<BlockId> {
    let strict = postdominators
        .get(&block)?
        .iter()
        .copied()
        .filter(|candidate| *candidate != block)
        .collect::<Vec<_>>();
    strict.iter().copied().find(|candidate| {
        strict.iter().all(|other| {
            candidate == other
                || !postdominators
                    .get(other)
                    .is_some_and(|dominators| dominators.contains(candidate))
        })
    })
}

#[cfg(test)]
mod source_ordered_definitions_v1_tests;

#[cfg(test)]
mod storage_variation_tests {
    use super::*;

    use fe2o3_kernel_ir::{
        AccessMode, MemoryAccess, Signature, StorageLayoutIdV1, StorageOperationV1,
        StorageProjectionV1, ValueDef,
    };
    #[test]
    fn storage_results_are_varying_and_do_not_make_a_pure_helper_summary() {
        let pointer = Type::pointer(
            Type::StorageObject(StorageLayoutIdV1(0)),
            AddressSpace::Private,
            AccessMode::ReadWrite,
        );
        let mut block = BasicBlock::new(BlockId(0));
        block.operations.push(Operation::new(
            vec![ValueDef::new(ValueId(1), pointer.clone())],
            OperationKind::Storage(StorageOperationV1::Project {
                base: ValueId(0),
                step: StorageProjectionV1::Field(0),
            }),
        ));
        block.operations.push(Operation::new(
            vec![ValueDef::new(ValueId(2), Type::Scalar(ScalarType::U32))],
            OperationKind::Storage(StorageOperationV1::ReadValue {
                address: ValueId(1),
                access: MemoryAccess::new(AddressSpace::Private, 4),
            }),
        ));
        block.operations.push(Operation::new(
            vec![ValueDef::new(ValueId(3), Type::Scalar(ScalarType::U32))],
            OperationKind::Constant(Constant::U32(9)),
        ));
        block.terminator = Some(Terminator::Return { values: vec![] });
        let function = Function::internal_helper(
            "storage",
            Signature::new(vec![pointer], vec![]),
            vec![ValueId(0)],
            vec![block],
        );
        let report = analyze_function(&function);
        assert_eq!(report.values.get(&ValueId(1)), Some(&Variation::Varying));
        assert_eq!(report.values.get(&ValueId(2)), Some(&Variation::Varying));
        assert_eq!(
            report.values.get(&ValueId(3)),
            Some(&Variation::GridUniform)
        );
        let id = function.id.clone();
        let mut module = Module::new("storage_variation");
        module.functions.push(function);
        // Isolate the effect-free projection: a read-effect fallback is not enough.
        module.functions[0].body.as_mut().unwrap().blocks[0]
            .operations
            .retain(|operation| {
                matches!(
                    operation.kind,
                    OperationKind::Storage(StorageOperationV1::Project { .. })
                )
            });
        assert!(
            !collect_uniform_helper_candidate(&module, &id, &mut BTreeSet::new())
                .structurally_supported
        );
    }
}
