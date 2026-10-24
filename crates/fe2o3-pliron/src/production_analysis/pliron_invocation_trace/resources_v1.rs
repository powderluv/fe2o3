use crate::production_analysis::pliron_pipeline::invocation_receipt_v1::{
    InvocationObserverV1, observe_resource_preflight_v1, require_observed_v1,
};

/// Precharges the full operation scan that extracts and validates the single
/// fixed-size execution-layout contract. This phase must run before the
/// resulting layout is used to derive an invocation-trace expansion bound.
pub(crate) fn preflight_execution_layout_resource_upper_bound_v1(
    census: ProductionAnalysisInputCensusV1,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionAnalysisResourceUpperBoundV1, ProductionAnalysisResourceLimitV1> {
    preflight_execution_layout_resource_upper_bound_with_observation_v1(census, limits, None)
}

pub(crate) fn preflight_execution_layout_resource_upper_bound_with_observation_v1(
    census: ProductionAnalysisInputCensusV1,
    limits: ProductionAnalysisResourceLimitsV1,
    observer: Option<&InvocationObserverV1<'_, '_>>,
) -> Result<ProductionAnalysisResourceUpperBoundV1, ProductionAnalysisResourceLimitV1> {
    observe_resource_preflight_v1(observer, |observer| {
        let phase = ProductionAnalysisResourcePhaseV1::LaunchContract;
        let work = census
            .operations
            .checked_add(12)
            .ok_or(ProductionAnalysisResourceLimitV1 {
                phase,
                resource: "execution layout resource upper bound",
            })?;
        let bound = ProductionAnalysisResourceUpperBoundV1::checked_phase(phase, work, 10, 0)?;
        require_observed_v1(limits, phase, Ok(bound), observer)
    })
}

const TRACE_BLOCK_STATE_CENSUS_TEMPORARY_V1: usize = 3;

// The visited-key set is local to one invocation. Its wide entry key is
// allocated once if no successor occurrence can target entry. Every subsequent
// retained or pending key follows a successful global step charge. We retain
// the coarse bound for entry reentry, including unreachable/duplicate edges.
#[allow(clippy::too_many_arguments)]
fn preflight_single_entry_key_storage_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    census: ProductionAnalysisInputCensusV1,
    block_state: &mut InvocationTraceBlockStateCensusV1,
    invocations: usize,
    limits: ProductionAnalysisResourceLimitsV1,
    observer: Option<&InvocationObserverV1<'_, '_>>,
) -> Result<Option<usize>, ProductionAnalysisResourceLimitV1> {
    if invocations == 0 {
        return Ok(None);
    }
    let entry = inventory
        .blocks()
        .first()
        .copied()
        .ok_or_else(trace_block_state_census_error_v1)?;
    let entry_arguments = entry.deref(context).get_num_arguments();
    if entry_arguments == 0 || entry_arguments < block_state.max_block_arguments {
        return Ok(None);
    }
    if entry_arguments != block_state.max_block_arguments {
        return Err(trace_block_state_census_error_v1());
    }
    let phase = ProductionAnalysisResourcePhaseV1::InvocationTrace;
    let work = checked_trace_sum_v1(&[
        block_state.work,
        census.blocks,
        census.operations,
        census.successors,
        1,
    ])?;
    // Existing census output plus all simultaneously live scalar scan state.
    // The final trace peak already exceeds this fixed scratch (even at M=1).
    require_observed_v1(
        limits,
        phase,
        ProductionAnalysisResourceUpperBoundV1::checked_phase(phase, work, 0, 12),
        observer,
    )?;
    if inventory.blocks().len() != census.blocks
        || inventory.operations().len() != census.operations
    {
        return Err(trace_block_state_census_error_v1());
    }
    let mut other_arguments = 0;
    for block in inventory.blocks().iter().skip(1) {
        other_arguments = other_arguments.max(block.deref(context).get_num_arguments());
    }
    let mut successor_count = 0usize;
    let mut entry_has_incoming = false;
    for site in inventory.operations() {
        let raw = site.pointer().deref(context);
        let count = raw.get_num_successors();
        // Underdeclared counts must stop before any uncharged edge lookup.
        if count
            > census
                .successors
                .checked_sub(successor_count)
                .ok_or_else(trace_block_state_census_error_v1)?
        {
            return Err(trace_block_state_census_error_v1());
        }
        for ordinal in 0..count {
            entry_has_incoming |= raw.get_successor(ordinal) == entry;
        }
        successor_count = successor_count
            .checked_add(count)
            .ok_or_else(trace_resource_overflow_v1)?;
    }
    if successor_count != census.successors {
        return Err(trace_block_state_census_error_v1());
    }
    block_state.work = work;
    if entry_has_incoming || entry_arguments <= other_arguments {
        return Ok(None);
    }
    // At most M keys, of which at most one has E arguments. Prior
    // invocations' visited sets are dropped, so the E-m delta occurs once.
    Ok(Some(checked_trace_sum_v1(&[
        checked_trace_product_v1(
            MAX_PLIRON_TRACE_TOTAL_STEPS_V1,
            other_arguments
                .checked_add(2)
                .ok_or_else(trace_resource_overflow_v1)?,
        )?,
        entry_arguments
            .checked_sub(other_arguments)
            .ok_or_else(trace_resource_overflow_v1)?,
    ])?))
}

/// Native launch cardinality enters the same legacy scheduler preflight. The
/// native byte ledger separately pays correspondence, fold scratch and events.
pub(crate) fn preflight_native_trace_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    census: ProductionAnalysisInputCensusV1,
    geometry: native_input_v1::NativeTraceGeometryV1,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionAnalysisResourceUpperBoundV1, ProductionAnalysisResourceLimitV1> {
    let invocations = geometry
        .global_extents
        .into_iter()
        .try_fold(1_u64, u64::checked_mul)
        .filter(|v| *v <= MAX_PLIRON_RACE_INVOCATIONS_V1)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(trace_resource_overflow_v1)?;
    let mut block_state =
        collect_trace_block_state_census_observed_v1(context, inventory, census, limits, None)?;
    let entry_key_storage = preflight_single_entry_key_storage_v1(
        context,
        inventory,
        census,
        &mut block_state,
        invocations,
        limits,
        None,
    )?;
    Ok(
        invocation_trace_resource_upper_bound_for_block_state_observed_v1(
            census,
            block_state,
            invocations,
            3,
            limits,
            None,
            entry_key_storage,
        )?
        .upper_bound(),
    )
}

struct InvocationTraceBlockStateCensusV1 {
    max_block_arguments: usize,
    work: usize,
}

fn trace_block_state_census_error_v1() -> ProductionAnalysisResourceLimitV1 {
    ProductionAnalysisResourceLimitV1 {
        phase: ProductionAnalysisResourcePhaseV1::InvocationTrace,
        resource: "invocation trace block-state census mismatch",
    }
}

// The analysis manager supplies the same immutable inventory to preflight and
// tracing. Aggregate counts check the scan domain; they are not owner identity.
#[cfg(test)]
fn collect_trace_block_state_census_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    census: ProductionAnalysisInputCensusV1,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<InvocationTraceBlockStateCensusV1, ProductionAnalysisResourceLimitV1> {
    collect_trace_block_state_census_observed_v1(context, inventory, census, limits, None)
}

fn collect_trace_block_state_census_observed_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    census: ProductionAnalysisInputCensusV1,
    limits: ProductionAnalysisResourceLimitsV1,
    observer: Option<&InvocationObserverV1<'_, '_>>,
) -> Result<InvocationTraceBlockStateCensusV1, ProductionAnalysisResourceLimitV1> {
    let phase = ProductionAnalysisResourcePhaseV1::InvocationTrace;
    let work = census
        .blocks
        .checked_add(1)
        .ok_or_else(trace_resource_overflow_v1)?;
    // Two result fields and the running sum; no roster or key is allocated.
    require_observed_v1(
        limits,
        phase,
        ProductionAnalysisResourceUpperBoundV1::checked_phase(
            phase,
            work,
            0,
            TRACE_BLOCK_STATE_CENSUS_TEMPORARY_V1,
        ),
        observer,
    )?;
    if inventory.blocks().len() != census.blocks {
        return Err(trace_block_state_census_error_v1());
    }
    let mut result = InvocationTraceBlockStateCensusV1 {
        max_block_arguments: 0,
        work,
    };
    let mut arguments = 0_usize;
    for block in inventory.blocks() {
        let arity = block.deref(context).get_num_arguments();
        arguments = arguments
            .checked_add(arity)
            .ok_or_else(trace_resource_overflow_v1)?;
        result.max_block_arguments = result.max_block_arguments.max(arity);
    }
    if arguments != census.block_arguments {
        return Err(trace_block_state_census_error_v1());
    }
    Ok(result)
}

/// Preflights the exact-launch trace expansion before its first allocation.
///
/// `None` means the trace will reject the same dynamic, overflowing, or
/// over-limit launch before constructing a trace. A statically admitted launch
/// precharges the global step cap so cyclic CFGs remain bounded. After a
/// successful cache construction, `exact_admission` replaces that broad event
/// cardinality with the authenticated sum of cached events for downstream
/// analyses.
pub(crate) fn preflight_invocation_trace_resource_upper_bound_v1(
    context: &Context,
    inventory: Option<&BoundedPlironFunctionInventoryV1>,
    census: ProductionAnalysisInputCensusV1,
    sparse: Option<&crate::SparseIndexAnalysisV1>,
    layout: Option<PlironExecutionLayoutV1>,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionInvocationTraceResourcePreflightV1, ProductionAnalysisResourceLimitV1> {
    preflight_invocation_trace_resource_upper_bound_with_observation_v1(
        context, inventory, census, sparse, layout, limits, None,
    )
}

pub(crate) fn preflight_invocation_trace_resource_upper_bound_with_observation_v1(
    context: &Context,
    inventory: Option<&BoundedPlironFunctionInventoryV1>,
    census: ProductionAnalysisInputCensusV1,
    sparse: Option<&crate::SparseIndexAnalysisV1>,
    layout: Option<PlironExecutionLayoutV1>,
    limits: ProductionAnalysisResourceLimitsV1,
    observer: Option<&InvocationObserverV1<'_, '_>>,
) -> Result<ProductionInvocationTraceResourcePreflightV1, ProductionAnalysisResourceLimitV1> {
    observe_resource_preflight_v1(observer, |observer| {
        let exact_shape =
            sparse.and_then(|sparse| static_invocation_shape_for_resource_v1(sparse, layout));
        if let Some((invocations, launch_rank)) = exact_shape {
            let inventory = inventory.ok_or(ProductionAnalysisResourceLimitV1 {
                phase: ProductionAnalysisResourcePhaseV1::InvocationTrace,
                resource: "invocation trace inventory unavailable",
            })?;
            let mut block_state = collect_trace_block_state_census_observed_v1(
                context, inventory, census, limits, observer,
            )?;
            let entry_key_storage = preflight_single_entry_key_storage_v1(
                context,
                inventory,
                census,
                &mut block_state,
                invocations,
                limits,
                observer,
            )?;
            let exact_admission =
                invocation_trace_resource_upper_bound_for_block_state_observed_v1(
                    census,
                    block_state,
                    invocations,
                    launch_rank,
                    limits,
                    observer,
                    entry_key_storage,
                )?;
            return Ok(ProductionInvocationTraceResourcePreflightV1 {
                attempt_upper_bound: exact_admission.upper_bound(),
                exact_admission: Some(exact_admission),
            });
        }

        // Rejected attempts still scan for scoped operations, validate the bounded
        // launch vector, and retain a cache error. Admit that work even though no
        // exact invocation expansion is available to downstream analyses.
        let launch_rank = layout
            .map(|_| 3)
            .or_else(|| sparse.map(|sparse| sparse.launch_extents().len()))
            .unwrap_or(MAX_RANKED_MEMORY_RANK);
        let work = checked_trace_sum_v1(&[
            census.operations,
            checked_trace_product_v1(launch_rank, 2)?,
            16,
        ])?;
        let bound = ProductionAnalysisResourceUpperBoundV1::checked_phase(
            ProductionAnalysisResourcePhaseV1::InvocationTrace,
            work,
            8,
            launch_rank,
        )?;
        let attempt_upper_bound = require_observed_v1(
            limits,
            ProductionAnalysisResourcePhaseV1::InvocationTrace,
            Ok(bound),
            observer,
        )?;
        Ok(ProductionInvocationTraceResourcePreflightV1 {
            attempt_upper_bound,
            exact_admission: None,
        })
    })
}

pub(super) fn static_invocation_shape_for_resource_v1(
    sparse: &crate::SparseIndexAnalysisV1,
    layout: Option<PlironExecutionLayoutV1>,
) -> Option<(usize, usize)> {
    let launch_extents = match layout {
        Some(ref layout) => layout.global_extents.as_slice(),
        None => sparse.launch_extents(),
    };
    if launch_extents.contains(&0) {
        return None;
    }
    let invocations = launch_extents
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent))?;
    if invocations > MAX_PLIRON_RACE_INVOCATIONS_V1 {
        return None;
    }
    Some((usize::try_from(invocations).ok()?, launch_extents.len()))
}

// Existing synthetic resource tests retain the coarse count-only oracle.
// Production can obtain a narrowed block-state bound only from the inventory.
#[cfg(test)]
pub(super) fn invocation_trace_resource_upper_bound_for_shape_v1(
    census: ProductionAnalysisInputCensusV1,
    invocations: usize,
    launch_rank: usize,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionInvocationTraceResourceAdmissionV1, ProductionAnalysisResourceLimitV1> {
    invocation_trace_resource_upper_bound_for_block_state_v1(
        census,
        InvocationTraceBlockStateCensusV1 {
            max_block_arguments: census.block_arguments,
            work: 0,
        },
        invocations,
        launch_rank,
        limits,
    )
}

#[cfg(test)]
fn invocation_trace_resource_upper_bound_for_block_state_v1(
    census: ProductionAnalysisInputCensusV1,
    block_state: InvocationTraceBlockStateCensusV1,
    invocations: usize,
    launch_rank: usize,
    limits: ProductionAnalysisResourceLimitsV1,
) -> Result<ProductionInvocationTraceResourceAdmissionV1, ProductionAnalysisResourceLimitV1> {
    invocation_trace_resource_upper_bound_for_block_state_observed_v1(
        census,
        block_state,
        invocations,
        launch_rank,
        limits,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn invocation_trace_resource_upper_bound_for_block_state_observed_v1(
    census: ProductionAnalysisInputCensusV1,
    block_state: InvocationTraceBlockStateCensusV1,
    invocations: usize,
    launch_rank: usize,
    limits: ProductionAnalysisResourceLimitsV1,
    observer: Option<&InvocationObserverV1<'_, '_>>,
    entry_key_storage: Option<usize>,
) -> Result<ProductionInvocationTraceResourceAdmissionV1, ProductionAnalysisResourceLimitV1> {
    // A finite loop may revisit the same block with different block arguments.
    // The global step meter, rather than the static block/operation census, is
    // therefore the only sound expansion bound. An over-limit trace performs
    // the rejecting charge at MAX + 1 before it returns; no event is appended
    // by that failed step.
    let charged_trace_steps = usize::from(invocations != 0)
        .checked_mul(
            MAX_PLIRON_TRACE_TOTAL_STEPS_V1
                .checked_add(1)
                .ok_or_else(trace_resource_overflow_v1)?,
        )
        .ok_or_else(trace_resource_overflow_v1)?;
    let event_count = usize::from(invocations != 0)
        .checked_mul(MAX_PLIRON_TRACE_TOTAL_STEPS_V1)
        .ok_or_else(trace_resource_overflow_v1)?;
    let event_items = MAX_RANKED_MEMORY_RANK
        .checked_mul(2)
        .and_then(|nested| nested.checked_add(1))
        .ok_or_else(trace_resource_overflow_v1)?;

    // Work includes fixed prerequisite scans, invocation decoding, every
    // charged CFG step, and a launch-rank allowance for value/index handling.
    let fixed_scans = checked_trace_product_v1(census.operations, 4)?;
    let invocation_decode = checked_trace_product_v1(invocations, launch_rank)?;
    // A traced operation can evaluate each operand or outgoing block
    // argument. Each query admits the rejecting visit after its 65-value cap,
    // while a separate global meter admits its own MAX + 1 rejection.
    let values_per_step = census
        .max_operation_arity
        .checked_add(census.block_arguments)
        .and_then(|values| values.checked_add(1))
        .ok_or_else(trace_resource_overflow_v1)?;
    let evaluation_queries = checked_trace_product_v1(charged_trace_steps, values_per_step)?;
    let charged_query_visits = MAX_PLIRON_TRACE_VALUE_VISITS_PER_QUERY_V1
        .checked_add(1)
        .ok_or_else(trace_resource_overflow_v1)?;
    let charged_global_visits = usize::from(invocations != 0)
        .checked_mul(
            MAX_PLIRON_TRACE_TOTAL_STEPS_V1
                .checked_add(1)
                .ok_or_else(trace_resource_overflow_v1)?,
        )
        .ok_or_else(trace_resource_overflow_v1)?;
    let evaluation_visits = checked_trace_product_v1(evaluation_queries, charged_query_visits)?
        .min(charged_global_visits);
    let admitted_queries = evaluation_queries.min(charged_global_visits);
    let evaluation_work_per_visit = MAX_RANKED_MEMORY_RANK
        .checked_add(TRACE_VALUE_FIXED_WORK_PER_VISIT_V1)
        .and_then(|work| work.checked_add(TRACE_VALUE_STACK_WORK_PER_VISIT_V1))
        .ok_or_else(trace_resource_overflow_v1)?;
    let evaluation_work = checked_trace_sum_v1(&[
        checked_trace_product_v1(evaluation_visits, evaluation_work_per_visit)?,
        checked_trace_product_v1(admitted_queries, TRACE_VALUE_QUERY_SETUP_WORK_V1)?,
    ])?;
    let work = checked_trace_sum_v1(&[
        fixed_scans,
        census.blocks,
        block_state.work,
        invocation_decode,
        charged_trace_steps,
        evaluation_work,
    ])?;

    // The returned cache retains trace headers, invocation coordinates, and
    // all event/nested-index items. The temporary term covers the block map,
    // current environment, visited block-state keys, and launch extents.
    let retained = checked_trace_sum_v1(&[
        invocations,
        invocation_decode,
        checked_trace_product_v1(event_count, event_items)?,
    ])?;
    // Each key contains only the current block's arguments. Its allocation
    // follows a successful global step charge, including a duplicate candidate
    // while previous keys remain live; MAX + 1 fails before allocating a key.
    let visited_state_storage = match entry_key_storage {
        Some(storage) => storage,
        None => checked_trace_product_v1(
            event_count,
            block_state
                .max_block_arguments
                .checked_add(2)
                .ok_or_else(trace_resource_overflow_v1)?,
        )?,
    };
    let evaluation_stack_frames = MAX_PLIRON_TRACE_VALUE_VISITS_PER_QUERY_V1
        .checked_mul(TRACE_VALUE_STACK_FRAMES_PER_VISIT_V1)
        .and_then(|frames| frames.checked_add(TRACE_VALUE_STACK_FIXED_FRAMES_V1))
        .ok_or_else(trace_resource_overflow_v1)?;
    let evaluation_temporary = checked_trace_sum_v1(&[
        checked_trace_product_v1(
            evaluation_stack_frames,
            TRACE_VALUE_STACK_STORAGE_PER_FRAME_V1,
        )?,
        checked_trace_product_v1(
            MAX_PLIRON_TRACE_VALUE_VISITS_PER_QUERY_V1,
            TRACE_VALUE_CACHE_STORAGE_PER_VISIT_V1,
        )?,
        checked_trace_product_v1(
            MAX_PLIRON_TRACE_VALUE_VISITS_PER_QUERY_V1,
            TRACE_VALUE_ACTIVE_STORAGE_PER_VISIT_V1,
        )?,
        MAX_RANKED_MEMORY_RANK,
    ])?;
    let temporary = checked_trace_sum_v1(&[
        checked_trace_product_v1(census.blocks, 2)?,
        census.results,
        checked_trace_product_v1(census.block_arguments, 3)?,
        visited_state_storage,
        launch_rank,
        evaluation_temporary,
    ])?;
    // The fixed census temporaries end before trace allocation starts.
    let temporary = temporary.max(TRACE_BLOCK_STATE_CENSUS_TEMPORARY_V1);
    let bound = ProductionAnalysisResourceUpperBoundV1::checked_phase(
        ProductionAnalysisResourcePhaseV1::InvocationTrace,
        work,
        retained,
        temporary,
    )?;
    let upper_bound = require_observed_v1(
        limits,
        ProductionAnalysisResourcePhaseV1::InvocationTrace,
        Ok(bound),
        observer,
    )?;
    Ok(ProductionInvocationTraceResourceAdmissionV1 {
        upper_bound,
        invocation_count: invocations,
        launch_rank,
        event_upper_bound: event_count,
    })
}

include!("block_state_resource_tests_v1.rs");
