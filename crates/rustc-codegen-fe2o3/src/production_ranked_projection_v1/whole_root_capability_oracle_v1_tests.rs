//! Independent original-source DATA oracle for the closed genuine observer.
//! No candidate scheduler/state is used to derive expectations.
use super::*;
use crate::production_ranked_projection_v1::bf16_nominal_dense_v1::NominalCapabilityLedgerV1;
use crate::production_ranked_projection_v1::bf16_nominal_source_algorithms_v1::constant_locals_with_resources_v1;
use crate::production_ranked_projection_v1::*;
use fe2o3_mir_model::semantic_mir_v1::{
    SemanticAssignmentV1, SemanticBasicBlockV1, SemanticControlFlowEdgeV1, SemanticOperandV1,
    SemanticRvalueKindV1, SemanticRvalueV1, SemanticStatementV1,
};
use std::collections::{HashMap, HashSet, VecDeque};

fn traced_original(
    callables: &[SemanticCallableDeclV1],
    function: &SemanticFunctionDeclV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    local_allocations: &[Option<AllocationContractV1>],
    constants: &[Option<u64>],
    pipeline_owners: &[Option<usize>],
    pipeline_payloads: &HashMap<usize, ProjectedMfmaOperandV1>,
    entry: usize,
    work: &mut usize,
) -> std::result::Result<
    (Vec<Option<ProjectedCapabilityStateV1>>, Vec<usize>),
    ProductionRankedProjectionErrorV1,
> {
    let mut entries: Vec<Option<ProjectedCapabilityStateV1>> = vec![None; function.blocks().len()];
    entries[entry] = Some(HashMap::new());
    let mut worklist = VecDeque::from([entry]);
    let mut stored_entries = 0_usize;
    let mut visits = Vec::new();
    while let Some(block_index) = worklist.pop_front() {
        visits.push(block_index);
        let entry_state = entries.get(block_index).and_then(Option::as_ref).ok_or(
            ProductionRankedProjectionErrorV1::Unsupported(
                "a queued capability dataflow block has no entry state",
            ),
        )?;
        charge_capability_dataflow_work_v1(
            work,
            entry_state.len().checked_add(1).ok_or(
                ProductionRankedProjectionErrorV1::Unsupported("capability clone work overflow"),
            )?,
        )?;
        let mut state = try_clone_capability_state_v1(entry_state)?;
        transfer_capability_statements_v1(
            function,
            block_index,
            &mut state,
            enum_payload_dominance,
        )?;
        transfer_capability_terminator_v1(
            callables,
            function,
            block_index,
            &mut state,
            local_allocations,
            constants,
            pipeline_owners,
            pipeline_payloads,
            false,
        )?;
        charge_capability_dataflow_work_v1(
            work,
            function.blocks()[block_index]
                .statements()
                .len()
                .checked_add(state.len())
                .ok_or(ProductionRankedProjectionErrorV1::Unsupported(
                    "capability dataflow work overflow",
                ))?,
        )?;
        if state.len() > MAX_PROJECTED_CAPABILITY_STATE_ENTRIES_V1 {
            return Err(ProductionRankedProjectionErrorV1::Unsupported(
                "capability dataflow exceeds the charged projection limit",
            ));
        }
        let successors = charged_unique_capability_successors_v1(
            function.blocks()[block_index].terminator().kind(),
            state.len(),
            work,
        )?;
        for target in successors {
            let target_entry =
                entries
                    .get_mut(target)
                    .ok_or(ProductionRankedProjectionErrorV1::Unsupported(
                        "a capability CFG edge outside the semantic function",
                    ))?;
            let changed = match target_entry {
                None => {
                    let next_stored_entries =
                        checked_capability_stored_entries_v1(stored_entries, state.len())?;
                    let successor_state = try_clone_capability_state_v1(&state)?;
                    stored_entries = next_stored_entries;
                    *target_entry = Some(successor_state);
                    true
                }
                Some(existing) => {
                    let additional = state
                        .keys()
                        .filter(|key| !existing.contains_key(key))
                        .count();
                    let next_stored_entries =
                        checked_capability_stored_entries_v1(stored_entries, additional)?;
                    let changed = merge_capability_states_v1(existing, &state)?;
                    stored_entries = next_stored_entries;
                    changed
                }
            };
            if changed {
                worklist.push_back(target);
            }
        }
    }
    Ok((entries, visits))
}

fn project_original_with_work(
    callables: &[SemanticCallableDeclV1],
    types: &[SemanticTypeDeclV1],
    function: &SemanticFunctionDeclV1,
    enum_payload_dominance: &SemanticEnumPayloadDominanceV1,
    local_allocations: &[Option<AllocationContractV1>],
    constants: &[Option<u64>],
) -> std::result::Result<(ProjectedCapabilityEffectsV1, usize), ProductionRankedProjectionErrorV1> {
    let block_count = function.blocks().len();
    let entry = function.entry().index() as usize;
    if entry >= block_count {
        return Err(ProductionRankedProjectionErrorV1::Unsupported(
            "a capability projection entry outside the semantic CFG",
        ));
    }
    let mut work = 0_usize;
    let pipeline_owners = workgroup_pipeline_local_owners_v1(callables, function)?;
    let initial_entries = propagate_capability_dataflow_v1(
        callables,
        function,
        enum_payload_dominance,
        local_allocations,
        constants,
        &pipeline_owners,
        &HashMap::new(),
        entry,
        &mut work,
    )?;
    let pipeline_payloads = collect_workgroup_pipeline_payloads_v1(
        callables,
        types,
        function,
        enum_payload_dominance,
        &pipeline_owners,
        &initial_entries,
        &mut work,
    )?;
    let entries = propagate_capability_dataflow_v1(
        callables,
        function,
        enum_payload_dominance,
        local_allocations,
        constants,
        &pipeline_owners,
        &pipeline_payloads,
        entry,
        &mut work,
    )?;

    let mut layouts = vec![None; block_count];
    let mut global_reads = vec![None; block_count];
    let mut transpose_workgroups = vec![None; block_count];
    let mut read_views = vec![None; block_count];
    for (block_index, entry_state) in entries.into_iter().enumerate() {
        let Some(mut state) = entry_state else {
            continue;
        };
        charge_capability_dataflow_work_v1(
            &mut work,
            state
                .len()
                .checked_add(function.blocks()[block_index].statements().len())
                .and_then(|work| work.checked_add(1))
                .ok_or(ProductionRankedProjectionErrorV1::Unsupported(
                    "capability replay work overflow",
                ))?,
        )?;
        transfer_capability_statements_v1(
            function,
            block_index,
            &mut state,
            enum_payload_dominance,
        )?;
        let effects = transfer_capability_terminator_v1(
            callables,
            function,
            block_index,
            &mut state,
            local_allocations,
            constants,
            &pipeline_owners,
            &pipeline_payloads,
            true,
        )?;
        layouts[block_index] = effects.layout;
        global_reads[block_index] = effects.global_read;
        transpose_workgroups[block_index] = effects.transpose_workgroup;
        read_views[block_index] = effects.read_view;
    }
    Ok((
        ProjectedCapabilityEffectsV1 {
            layouts,
            global_reads,
            transpose_workgroups,
            read_views,
        },
        work,
    ))
}

#[derive(Clone, Copy, Debug)]
struct OracleBound {
    blocks: usize,
    locals: usize,
    map_entries: usize,
    visits: usize,
    storage: usize,
    work: usize,
}
fn plus(a: usize, b: usize) -> BResult<usize> {
    a.checked_add(b).ok_or_else(arithmetic)
}
fn times(a: usize, b: usize) -> BResult<usize> {
    a.checked_mul(b).ok_or_else(arithmetic)
}
fn differs() -> Backend {
    Backend::Incomplete("actual capability prefix original DATA differs")
}
fn slot_bytes<T>(count: usize) -> BResult<usize> {
    times(count, size_of::<T>())
}

/// Logical source-owner/slot and traversal policy, not native heap/RSS/hash probes.
/// All bounds are derived from the actual source before any original FIFO runs.
fn oracle_bound(
    function: &SemanticFunctionDeclV1,
    resources: &mut Prep<'_, '_>,
) -> BResult<OracleBound> {
    resources.work(64)?;
    resources.reserve_storage(bound_frame())?;
    let blocks = function.blocks().len();
    let locals = function.locals().len();
    if !(1..=32).contains(&blocks) || !(1..=4096).contains(&locals) {
        return Err(Backend::Unsupported(
            "original capability oracle exceeds closed source profile",
        ));
    }
    let entry = function.entry().index() as usize;
    if entry >= blocks {
        return Err(differs());
    }
    let mut edges = [0u32; 32];
    let mut indegree = [0usize; 32];
    let mut order = [0usize; 32];
    let mut paths = [0usize; 32];
    let mut max_transfer = 0usize;
    let mut replay_work = 0usize;
    for (block_index, block) in function.blocks().iter().enumerate() {
        resources.work(16)?;
        let mut transfer = plus(locals, 256)?;
        for statement in block.statements() {
            resources.work(64)?;
            transfer = plus(transfer, 64)?;
            if let SemanticStatementKindV1::Assign(assignment) = statement.kind() {
                assignment.value().kind().try_visit_operands(|_| {
                    resources.work(8)?;
                    transfer = plus(transfer, 8)?;
                    Ok::<(), Backend>(())
                })?;
            }
        }
        let arguments = match block.terminator().kind() {
            SemanticTerminatorKindV1::Call(call) => call.arguments().len(),
            SemanticTerminatorKindV1::TailCall(call) => call.arguments().len(),
            _ => 1,
        };
        transfer = plus(transfer, times(arguments, 8)?)?;
        replay_work = plus(replay_work, plus(transfer, plus(locals, 1)?)?)?;
        let mut raw_edges = 0usize;
        block.terminator().kind().try_for_each_edge(|edge| {
            resources.work(8)?;
            raw_edges = plus(raw_edges, 1)?;
            let target = edge.target().index() as usize;
            if target >= blocks {
                return Err(differs());
            }
            let bit = 1u32 << target;
            if edges[block_index] & bit == 0 {
                edges[block_index] |= bit;
                indegree[target] = plus(indegree[target], 1)?;
            }
            Ok(())
        })?;
        // Per visit: original transfer policy, all raw edges, sorted successor
        // B^2 policy, full existing/incoming membership+merge, clone and scans.
        let per_visit = plus(
            plus(transfer, raw_edges)?,
            plus(
                times(blocks, blocks)?,
                plus(
                    times(blocks, plus(times(locals, 3)?, 1)?)?,
                    plus(locals, 16)?,
                )?,
            )?,
        )?;
        max_transfer = max_transfer.max(per_visit);
    }
    resources.work(plus(times(blocks, blocks)?, times(blocks, 4)?)?)?;
    let mut count = 0usize;
    for block in 0..blocks {
        if indegree[block] == 0 {
            order[count] = block;
            count += 1;
        }
    }
    let mut cursor = 0usize;
    while cursor < count {
        let block = order[cursor];
        cursor += 1;
        let mut successors = edges[block];
        while successors != 0 {
            let target = successors.trailing_zeros() as usize;
            successors &= successors - 1;
            indegree[target] = indegree[target].checked_sub(1).ok_or_else(accounting)?;
            if indegree[target] == 0 {
                order[count] = target;
                count += 1;
            }
        }
    }
    if count != blocks {
        return Err(Backend::Unsupported(
            "original capability oracle requires an acyclic source",
        ));
    }
    let map_entries = times(blocks - 1, locals)?.min(MAX_PROJECTED_CAPABILITY_STATE_ENTRIES_V1);
    let structural_visits = plus(blocks, times(map_entries, 2)?)?;
    // Independent tighter DAG bound: at most one enqueue per visited
    // predecessor and unique edge. Saturation at the structural bound is safe.
    resources.work(plus(times(blocks, blocks)?, times(blocks, 2)?)?)?;
    paths[entry] = 1;
    for index in 0..count {
        let block = order[index];
        let mut successors = edges[block];
        while successors != 0 {
            let target = successors.trailing_zeros() as usize;
            successors &= successors - 1;
            paths[target] = paths[target]
                .saturating_add(paths[block])
                .min(structural_visits);
        }
    }
    let mut path_visits = 0usize;
    for visits in paths.iter().take(blocks) {
        path_visits = path_visits.saturating_add(*visits).min(structural_visits);
    }
    let visits = path_visits.min(structural_visits);
    let storage = oracle_payload_bytes(blocks, locals, map_entries, visits)?;
    // Six FIFO traversals: two traced, two untouched validators and two in the
    // complete original driver. Final replay, owners, binding and full compares
    // are separate. This is the declared policy's units, not CPU instructions.
    let work = plus(
        times(times(6, visits)?, max_transfer)?,
        plus(
            times(replay_work, 2)?,
            plus(
                times(times(blocks, locals)?, 16)?,
                plus(
                    times(visits, 12)?,
                    plus(times(blocks, 32)?, times(locals, 16)?)?,
                )?,
            )?,
        )?,
    )?;
    Ok(OracleBound {
        blocks,
        locals,
        map_entries,
        visits,
        storage,
        work,
    })
}
fn oracle_payload_bytes(
    blocks: usize,
    locals: usize,
    keys: usize,
    visits: usize,
) -> BResult<usize> {
    let rows = [
        slot_bytes::<(usize, ProjectedCapabilityValueV1)>(plus(
            times(keys, 4)?,
            times(locals, 2)?,
        )?)?,
        slot_bytes::<Option<ProjectedCapabilityStateV1>>(times(blocks, 4)?)?,
        slot_bytes::<usize>(times(visits, 3)?)?, // two retained traces + active FIFO
        slot_bytes::<usize>(times(blocks, 2)?)?, // successor set + sorted vector
        slot_bytes::<Option<usize>>(times(locals, 2)?)?, // retained/full-driver owners
        slot_bytes::<Option<ProductionRankedOperationV1>>(blocks)?,
        slot_bytes::<Option<AllocationContractV1>>(blocks)?,
        slot_bytes::<Option<ProjectedTransposeWorkgroupEffectV1>>(blocks)?,
        slot_bytes::<Option<ProjectedReadViewAccessV1>>(blocks)?,
        slot_bytes::<Option<ProjectedCapabilityReadEffectV1>>(blocks)?,
        size_of::<ProductionRankedOperationV1>(), // original empty-reference prefix
    ];
    rows.into_iter().try_fold(0usize, plus)
}

pub(super) struct OriginalCapabilityOracleV1 {
    phase: PrefixPhase,
    ledger: Option<Ledger>,
    bound: Option<OracleBound>,
    constants: Vec<Option<u64>>,
    enumeration: Option<SemanticEnumPayloadDominanceV1>,
    scalar: Option<AssertionDefinitionInventoryV1>,
    provenance: Option<LocalProvenanceV1>,
    allocations: Vec<Option<AllocationContractV1>>,
    prefix: Option<RootEntryPrefixV1>,
    owners: Vec<Option<usize>>,
    payloads: HashMap<usize, ProjectedMfmaOperandV1>,
    initial: Vec<Option<ProjectedCapabilityStateV1>>,
    repeated: Vec<Option<ProjectedCapabilityStateV1>>,
    first_trace: Vec<usize>,
    repeated_trace: Vec<usize>,
    effects: Option<ProjectedCapabilityEffectsV1>,
    bound_reads: Vec<Option<ProjectedCapabilityReadEffectV1>>,
    original_work: usize,
    call_block: usize,
}
impl OriginalCapabilityOracleV1 {
    pub(super) fn new() -> Self {
        Self {
            phase: PrefixPhase::Fresh,
            ledger: None,
            bound: None,
            constants: Vec::new(),
            enumeration: None,
            scalar: None,
            provenance: None,
            allocations: Vec::new(),
            prefix: None,
            owners: Vec::new(),
            payloads: HashMap::new(),
            initial: Vec::new(),
            repeated: Vec::new(),
            first_trace: Vec::new(),
            repeated_trace: Vec::new(),
            effects: None,
            bound_reads: Vec::new(),
            original_work: 0,
            call_block: 0,
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        &mut self,
        function: &SemanticFunctionDeclV1,
        callables: &[SemanticCallableDeclV1],
        types: &[SemanticTypeDeclV1],
        checked: &CheckedBf16NominalCallV1<'_>,
        owner: &ProductionPreRankedKirOwnerV1,
        actual_inputs: &ActualRetainedRankedInputsV1<'_>,
        candidate_prefix: &RootEntryPrefixV1,
        candidate_constants: &[Option<u64>],
        resources: &mut Prep<'_, '_>,
    ) -> BResult<()> {
        let fresh = self.phase == PrefixPhase::Fresh;
        self.phase = PrefixPhase::Terminal;
        if !fresh || !resources.is_metered() || resources.has_denial() {
            return Err(accounting());
        }
        resources.work(64)?;
        resources.reserve_storage(oracle_frame()?)?;
        self.ledger = resources.original_ledger_v1();
        if self.ledger.is_none() {
            return Err(accounting());
        }
        self.bound = Some(oracle_bound(function, resources)?);
        let bound = self.bound.ok_or_else(accounting)?;
        resources.work(bound.work)?;
        resources.reserve_storage(bound.storage)?;
        // Complete independent originals. Their original local partial-scratch
        // drop behavior is unchanged; every completed result attaches here.
        self.constants = constant_locals_with_resources_v1(function, resources)?;
        resources.work(times(bound.locals, 2)?)?;
        if self.constants.as_slice() != candidate_constants {
            return Err(differs());
        }
        let selected = select_actual_capability_prefix_inputs_v1(
            owner,
            checked,
            function,
            actual_inputs,
            resources,
        )?;
        // This genuine observer closes the current actual empty-reference ABI.
        // Nonempty selected references remain a separately required source case.
        if !actual_inputs.bindings().as_slice().is_empty() || !selected.references().is_empty() {
            return Err(Backend::Unsupported(
                "actual capability oracle requires empty actual reference bindings",
            ));
        }
        self.prefix = Some(crate::production_ranked_projection_v1::root_entry_prefix_preparation_v1::prepare_root_entry_prefix_legacy_v1(
            selected.source_root(), actual_inputs.bindings())?);
        let prefix = self.prefix.as_ref().ok_or_else(accounting)?;
        if prefix.entry_operations != candidate_prefix.entry_operations
            || prefix.next_value != candidate_prefix.next_value
            || prefix.reserved_reference_values != candidate_prefix.reserved_reference_values
        {
            return Err(differs());
        }
        self.enumeration = Some(
            SemanticEnumPayloadDominanceV1::analyze_with_meter_v1(
                function,
                types,
                &mut ModelMeter(resources),
            )
            .map_err(model_error)?,
        );
        self.scalar = Some(assertion_definition_inventory_with_resources_v1(
            function, resources,
        )?);
        let scalar = self.scalar.as_ref().ok_or_else(accounting)?;
        self.provenance = Some(local_provenance_with_resources_v1(
            callables,
            types,
            function,
            &scalar.counts,
            &scalar.address_escaped,
            resources,
        )?);
        self.allocations = local_allocation_contracts_with_resources_v1(
            types,
            function,
            &self
                .provenance
                .as_ref()
                .ok_or_else(accounting)?
                .allocation_origins,
            resources,
        )?;
        let enumeration = self.enumeration.as_ref().ok_or_else(accounting)?;
        self.owners = workgroup_pipeline_local_owners_v1(callables, function)?;
        if self.owners.len() != bound.locals || self.owners.iter().any(Option::is_some) {
            return Err(Backend::Unsupported(
                "actual capability original owner scan found a pipeline",
            ));
        }
        let entry = function.entry().index() as usize;
        let mut traced_work = 0usize;
        (self.initial, self.first_trace) = traced_original(
            callables,
            function,
            enumeration,
            &self.allocations,
            &self.constants,
            &self.owners,
            &self.payloads,
            entry,
            &mut traced_work,
        )?;
        self.payloads = collect_workgroup_pipeline_payloads_v1(
            callables,
            types,
            function,
            enumeration,
            &self.owners,
            &self.initial,
            &mut traced_work,
        )?;
        if !self.payloads.is_empty() {
            return Err(Backend::Unsupported(
                "actual capability original payload scan found a pipeline",
            ));
        }
        (self.repeated, self.repeated_trace) = traced_original(
            callables,
            function,
            enumeration,
            &self.allocations,
            &self.constants,
            &self.owners,
            &self.payloads,
            entry,
            &mut traced_work,
        )?;
        if self.first_trace.len() > bound.visits || self.repeated_trace.len() > bound.visits {
            return Err(accounting());
        }
        // Untouched original API validates the trace-only adaptation's complete
        // states, independently of the candidate. Both passes share its counter.
        let mut validation_work = 0usize;
        let original_initial = propagate_capability_dataflow_v1(
            callables,
            function,
            enumeration,
            &self.allocations,
            &self.constants,
            &self.owners,
            &self.payloads,
            entry,
            &mut validation_work,
        )?;
        if original_initial != self.initial {
            return Err(differs());
        }
        drop(original_initial);
        let original_payloads = collect_workgroup_pipeline_payloads_v1(
            callables,
            types,
            function,
            enumeration,
            &self.owners,
            &self.initial,
            &mut validation_work,
        )?;
        if original_payloads != self.payloads {
            return Err(differs());
        }
        drop(original_payloads);
        let original_repeated = propagate_capability_dataflow_v1(
            callables,
            function,
            enumeration,
            &self.allocations,
            &self.constants,
            &self.owners,
            &self.payloads,
            entry,
            &mut validation_work,
        )?;
        if original_repeated != self.repeated || validation_work != traced_work {
            return Err(differs());
        }
        drop(original_repeated);
        let (effects, original_work) = project_original_with_work(
            callables,
            types,
            function,
            enumeration,
            &self.allocations,
            &self.constants,
        )?;
        self.effects = Some(effects);
        self.original_work = original_work;
        self.bound_reads = bind_capability_read_effects_to_call_blocks_v1(
            function,
            &self.effects.as_ref().ok_or_else(accounting)?.global_reads,
        )?;
        self.call_block = usize::MAX;
        for (block, body) in function.blocks().iter().enumerate() {
            if let SemanticTerminatorKindV1::Call(call) = body.terminator().kind() {
                if std::ptr::eq(call, checked.source_call()) {
                    if self.call_block != usize::MAX {
                        return Err(differs());
                    }
                    self.call_block = block;
                }
            }
        }
        if self.call_block == usize::MAX {
            return Err(differs());
        }
        if resources.has_denial() || resources.original_ledger_v1() != self.ledger {
            return Err(accounting());
        }
        self.phase = PrefixPhase::Complete;
        Ok(())
    }
    pub(super) fn compare(
        &self,
        function: &SemanticFunctionDeclV1,
        actual: CompletedNominalFifoDriverV1<'_>,
        visits: [usize; 3],
        consumer: &mut dyn NominalCapabilityConsumerV1,
    ) -> Result<PrefixObservation> {
        let ledger = consumer.ledger_v1();
        let bound = self.bound.ok_or(Resource::Accounting)?;
        if self.phase != PrefixPhase::Complete
            || self.ledger != Some((ledger.slot, ledger.identity))
            || ledger.denied_work
            || ledger.denied_storage
            || function.blocks().len() != bound.blocks
            || function.locals().len() != bound.locals
        {
            return Err(Resource::Accounting.into());
        }
        // Complete comparisons were prepaid in the original ledger, before the
        // oracle traversals. No candidate value determines an expectation.
        compare_flat(
            &self.initial,
            actual.initial_entries,
            actual.initial_reached,
            bound.locals,
        )
        .map_err(query_error)?;
        compare_flat(
            &self.repeated,
            actual.repeated_entries,
            actual.repeated_reached,
            bound.locals,
        )
        .map_err(query_error)?;
        let effects = self.effects.as_ref().ok_or(Resource::Accounting)?;
        let expected_visits = [
            self.first_trace
                .iter()
                .filter(|&&block| block == self.call_block)
                .count(),
            self.repeated_trace
                .iter()
                .filter(|&&block| block == self.call_block)
                .count(),
            1,
        ];
        let reached = self.repeated.iter().filter(|entry| entry.is_some()).count();
        if self.first_trace.as_slice() != actual.initial_visits
            || self.repeated_trace.as_slice() != actual.repeated_visits
            || effects.layouts.as_slice() != actual.layouts
            || effects.global_reads.as_slice() != actual.global_reads
            || effects.transpose_workgroups.as_slice() != actual.transpose_workgroups
            || effects.read_views.as_slice() != actual.read_views
            || self.bound_reads.as_slice() != actual.bound_reads
            || self.original_work != actual.original_work
            || visits != expected_visits
            || actual.run.query_visits != expected_visits
            || actual.run.authenticated_visits != expected_visits
            || actual.run.reached_blocks
                != [self.first_trace.len(), self.repeated_trace.len(), reached]
            || actual.run.array_destination_has_origin
            || actual.owner_scan_blocks != bound.blocks
            || actual.payload_scan_blocks != bound.blocks
            || actual.alias_scan_locals != bound.locals
            || expected_visits[0] == 0
            || expected_visits[1] == 0
        {
            return Err(QueryError::Unavailable(
                "actual capability original DATA or real query census differs",
            ));
        }
        Ok(PrefixObservation {
            constants: self.constants.len(),
            nonempty_constants: self
                .constants
                .iter()
                .filter(|value| value.is_some())
                .count(),
            blocks: bound.blocks,
            layouts: effects
                .layouts
                .iter()
                .filter(|value| value.is_some())
                .count(),
            bound_reads: self
                .bound_reads
                .iter()
                .filter(|value| value.is_some())
                .count(),
            first_visits: self.first_trace.len(),
            repeated_visits: self.repeated_trace.len(),
            original_work: self.original_work,
            query_visits: expected_visits,
        })
    }
}
fn compare_flat(
    expected: &[Option<ProjectedCapabilityStateV1>],
    flat: &[Option<ProjectedCapabilityValueV1>],
    reached: &[bool],
    locals: usize,
) -> BResult<()> {
    if flat.len() != times(expected.len(), locals)? || reached.len() != expected.len() {
        return Err(differs());
    }
    for (block, entry) in expected.iter().enumerate() {
        if reached[block] != entry.is_some() {
            return Err(differs());
        }
        if let Some(state) = entry {
            if state.keys().any(|&key| key >= locals) {
                return Err(differs());
            }
        }
        for local in 0..locals {
            let original = entry.as_ref().and_then(|state| state.get(&local)).copied();
            if flat[block * locals + local] != original {
                return Err(differs());
            }
        }
    }
    Ok(())
}

fn bound_frame() -> usize {
    size_of::<(
        &SemanticFunctionDeclV1,
        &mut Prep<'static, 'static>,
        OracleBound,
        [u32; 32],
        [usize; 32],
        [usize; 32],
        [usize; 32],
        [usize; 26],
        Option<usize>,
        BResult<usize>,
        BResult<OracleBound>,
        BResult<()>,
        std::iter::Enumerate<std::slice::Iter<'static, SemanticBasicBlockV1>>,
        &SemanticBasicBlockV1,
        std::slice::Iter<'static, SemanticStatementV1>,
        &SemanticStatementV1,
        &SemanticAssignmentV1,
        &SemanticRvalueV1,
        &SemanticRvalueKindV1,
        &SemanticOperandV1,
        &SemanticTerminatorKindV1,
        &SemanticDirectCallV1,
        usize,
        u32,
        SemanticControlFlowEdgeV1,
        SemanticControlFlowEdgeV1,
        SemanticBlockIdV1,
        SemanticBlockIdV1,
        u32,
        usize,
        &mut [u32; 32],
        &mut [usize; 32],
        &mut usize,
        &mut usize,
        std::ops::Range<usize>,
        std::ops::Range<usize>,
        std::iter::Take<std::slice::Iter<'static, usize>>,
        &usize,
        [usize; 11],
        std::array::IntoIter<usize, 11>,
        BResult<[usize; 11]>,
        Backend,
        bool,
    )>()
}
fn oracle_frame() -> BResult<usize> {
    let rows = [
        size_of::<OriginalCapabilityOracleV1>(),
        size_of::<(
            PrefixPhase,
            Option<Ledger>,
            Option<OracleBound>,
            usize,
            usize,
            Vec<Option<u64>>,
            Option<SemanticEnumPayloadDominanceV1>,
            Option<AssertionDefinitionInventoryV1>,
            Option<LocalProvenanceV1>,
            Vec<Option<AllocationContractV1>>,
            Option<RootEntryPrefixV1>,
            Vec<Option<usize>>,
            HashMap<usize, ProjectedMfmaOperandV1>,
            Vec<Option<ProjectedCapabilityStateV1>>,
            Vec<Option<ProjectedCapabilityStateV1>>,
            Vec<usize>,
            Vec<usize>,
            Option<ProjectedCapabilityEffectsV1>,
            Vec<Option<ProjectedCapabilityReadEffectV1>>,
        )>(),
        size_of::<(
            &mut OriginalCapabilityOracleV1,
            &OriginalCapabilityOracleV1,
            &SemanticFunctionDeclV1,
            &[SemanticCallableDeclV1],
            &[SemanticTypeDeclV1],
            &CheckedBf16NominalCallV1<'static>,
            &ProductionPreRankedKirOwnerV1,
            &ActualRetainedRankedInputsV1<'static>,
            &RootEntryPrefixV1,
            &[Option<u64>],
            &mut Prep<'static, 'static>,
        )>(),
        size_of::<(
            BResult<Vec<Option<u64>>>,
            BResult<SemanticEnumPayloadDominanceV1>,
            SemanticEnumPayloadMeteredErrorV1<Backend>,
            ModelMeter<'static, 'static, 'static>,
            BResult<AssertionDefinitionInventoryV1>,
            BResult<LocalProvenanceV1>,
            BResult<Vec<Option<AllocationContractV1>>>,
            BResult<RootEntryPrefixV1>,
            &AssertionDefinitionInventoryV1,
            &LocalProvenanceV1,
            &SemanticEnumPayloadDominanceV1,
            &RootEntryPrefixV1,
        )>(),
        size_of::<(
            BResult<Vec<Option<usize>>>,
            BResult<HashMap<usize, ProjectedMfmaOperandV1>>,
            BResult<(Vec<Option<ProjectedCapabilityStateV1>>, Vec<usize>)>,
            BResult<Vec<Option<ProjectedCapabilityStateV1>>>,
            BResult<(ProjectedCapabilityEffectsV1, usize)>,
            BResult<Vec<Option<ProjectedCapabilityReadEffectV1>>>,
            ProjectedCapabilityEffectsV1,
            &ProjectedCapabilityEffectsV1,
        )>(),
        // Complete original scheduling owners/call frames, including transient
        // maps/queues and full-driver locals while both traced tables are live.
        size_of::<(
            [Vec<Option<ProjectedCapabilityStateV1>>; 4],
            [ProjectedCapabilityStateV1; 4],
            [HashMap<usize, ProjectedMfmaOperandV1>; 4],
            [VecDeque<usize>; 2],
            [HashSet<usize>; 2],
            [Vec<usize>; 4],
            [Vec<Option<usize>>; 2],
            Vec<Option<ProductionRankedOperationV1>>,
            Vec<Option<AllocationContractV1>>,
            Vec<Option<ProjectedTransposeWorkgroupEffectV1>>,
            Vec<Option<ProjectedReadViewAccessV1>>,
            Vec<Option<ProjectedCapabilityReadEffectV1>>,
            RootEntryPrefixV1,
        )>(),
        size_of::<(
            [&[SemanticCallableDeclV1]; 6],
            [&SemanticFunctionDeclV1; 10],
            [&SemanticEnumPayloadDominanceV1; 6],
            [&[Option<AllocationContractV1>]; 6],
            [&[Option<u64>]; 6],
            [&[Option<usize>]; 6],
            [&HashMap<usize, ProjectedMfmaOperandV1>; 6],
            [&mut usize; 8],
            [&ProjectedCapabilityStateV1; 8],
            [&mut ProjectedCapabilityStateV1; 4],
            [&mut Option<ProjectedCapabilityStateV1>; 2],
            [usize; 36],
            [bool; 8],
        )>(),
        size_of::<(
            std::collections::hash_map::Keys<'static, usize, ProjectedCapabilityValueV1>,
            std::collections::hash_map::Iter<'static, usize, ProjectedCapabilityValueV1>,
            std::collections::hash_map::IterMut<'static, usize, ProjectedCapabilityValueV1>,
            std::vec::IntoIter<usize>,
            std::vec::IntoIter<Option<ProjectedCapabilityStateV1>>,
            std::iter::Enumerate<std::vec::IntoIter<Option<ProjectedCapabilityStateV1>>>,
            &usize,
            &usize,
            &ProjectedCapabilityValueV1,
            &mut ProjectedCapabilityValueV1,
            Option<&ProjectedCapabilityValueV1>,
            Option<ProjectedCapabilityValueV1>,
            (usize, ProjectedCapabilityValueV1),
            SemanticControlFlowEdgeV1,
            SemanticControlFlowEdgeV1,
            SemanticBlockIdV1,
            SemanticBlockIdV1,
            u32,
            usize,
        )>(),
        size_of::<(
            CompletedNominalFifoDriverV1<'static>,
            [usize; 3],
            &mut dyn NominalCapabilityConsumerV1,
            NominalCapabilityLedgerV1,
            OracleBound,
            Ledger,
            BResult<()>,
            Result<PrefixObservation>,
            PrefixObservation,
            Option<PrefixObservation>,
            Backend,
            &Backend,
            QueryError,
            Resource,
            Option<usize>,
            BResult<usize>,
            Result<usize>,
        )>(),
        size_of::<(
            [&[Option<ProjectedCapabilityStateV1>]; 3],
            &[Option<ProjectedCapabilityValueV1>],
            &[bool],
            std::iter::Enumerate<std::slice::Iter<'static, Option<ProjectedCapabilityStateV1>>>,
            &Option<ProjectedCapabilityStateV1>,
            Option<&ProjectedCapabilityStateV1>,
            &ProjectedCapabilityStateV1,
            std::ops::Range<usize>,
            usize,
            usize,
            Option<&ProjectedCapabilityValueV1>,
            Option<ProjectedCapabilityValueV1>,
            &usize,
            std::slice::Iter<'static, usize>,
            &usize,
            &&usize,
            std::slice::Iter<'static, Option<u64>>,
            &Option<u64>,
            &&Option<u64>,
            std::slice::Iter<'static, Option<ProductionRankedOperationV1>>,
            &Option<ProductionRankedOperationV1>,
            &&Option<ProductionRankedOperationV1>,
            std::slice::Iter<'static, Option<ProjectedCapabilityReadEffectV1>>,
            &Option<ProjectedCapabilityReadEffectV1>,
            &&Option<ProjectedCapabilityReadEffectV1>,
        )>(),
        size_of::<(
            [usize; 18],
            bool,
            Ledger,
            Option<Ledger>,
            Option<OracleBound>,
            &Vec<Option<u64>>,
            &[Option<u64>],
            &Vec<u8>,
            &[u8],
            &Vec<bool>,
            &[bool],
            &Vec<Option<u32>>,
            &[Option<u32>],
            &Vec<Option<AllocationContractV1>>,
            &[Option<AllocationContractV1>],
            &Vec<Option<usize>>,
            &[Option<usize>],
            std::slice::Iter<'static, Option<usize>>,
            &Option<usize>,
        )>(),
        size_of::<(
            std::iter::Enumerate<std::slice::Iter<'static, SemanticBasicBlockV1>>,
            &SemanticBasicBlockV1,
            &SemanticDirectCallV1,
            &SemanticDirectCallV1,
            &crate::reference_effect_v1::AuthenticatedReferenceEffectBindingsV1,
            &[crate::reference_effect_v1::AuthenticatedReferenceEffectBindingV1],
            &ProductionRankedOperationV1,
            &[ProductionRankedOperationV1],
            ActualSelectedInputsV1<'static>,
            BResult<ActualSelectedInputsV1<'static>>,
            &ActualSelectedInputsV1<'static>,
            ProductionSourceLaunchRootV1,
        )>(),
        // Explicit original oracle scheduling/helper carriers, separate from B.
        size_of::<(
            std::collections::hash_set::IntoIter<usize>,
            std::collections::hash_map::Entry<'static, usize, ProjectedCapabilityValueV1>,
            std::collections::hash_map::VacantEntry<'static, usize, ProjectedCapabilityValueV1>,
            std::collections::TryReserveError,
            std::result::Result<(), std::collections::TryReserveError>,
            [ProjectedCapabilityValueV1; 3],
            [ProjectedMfmaAccumulatorV1; 3],
            u64,
            ProjectedCapabilityTerminatorEffectsV1,
            BResult<ProjectedCapabilityTerminatorEffectsV1>,
            &mut ProjectedCapabilityValueV1,
            &SemanticTerminatorKindV1,
            &mut HashSet<usize>,
            &mut usize,
            &[usize],
            &Vec<usize>,
            std::slice::Iter<'static, usize>,
            &usize,
            &&usize,
            [usize; 12],
            Option<usize>,
            BResult<usize>,
            BResult<bool>,
            BResult<ProjectedCapabilityStateV1>,
            BResult<Vec<usize>>,
        )>(),
        size_of::<(
            [usize; 14],
            std::array::IntoIter<usize, 14>,
            BResult<[usize; 14]>,
            usize,
            usize,
            BResult<usize>,
            Backend,
        )>(),
    ];
    rows.into_iter().try_fold(0usize, plus)
}
