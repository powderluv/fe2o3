struct DynamicScheduleInputV1<'a> {
    schedule: &'a [EventSiteV1],
    accesses: &'a [AccessSiteV1],
    summary: &'a CanonicalEpochLoopV1,
    domain_reads: &'a [PlironOperationSiteV1],
}

fn verify_dynamic_schedule(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    buffers: u32,
    distance: u32,
    input: DynamicScheduleInputV1<'_>,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(usize, usize, bool), PlironPipelineProtocolFindingV1> {
    let DynamicScheduleInputV1 {
        schedule,
        accesses,
        summary,
        domain_reads,
    } = input;
    let mut prologue = Vec::new();
    let mut body = Vec::new();
    let mut drain = Vec::new();
    let prologue_start = summary
        .prologue
        .iter()
        .position(|block| *block == pipeline.block())
        .unwrap_or(0);
    let prologue_positions = summary.prologue[prologue_start..]
        .iter()
        .enumerate()
        .map(|(position, block)| (*block, position))
        .collect::<HashMap<_, _>>();
    let body_positions = summary
        .body
        .iter()
        .enumerate()
        .map(|(position, block)| (*block, position))
        .collect::<HashMap<_, _>>();
    let drain_positions = summary
        .drain
        .iter()
        .enumerate()
        .map(|(position, block)| (*block, position))
        .collect::<HashMap<_, _>>();
    for event in schedule {
        if let Some(position) = prologue_positions.get(&event.site.block()).copied() {
            prologue.push((position, *event));
        } else if let Some(position) = body_positions.get(&event.site.block()).copied() {
            body.push((position, *event));
        } else if let Some(position) = drain_positions.get(&event.site.block()).copied() {
            drain.push((position, *event));
        } else {
            return Err(invalid(
                pipeline,
                Some(event.site),
                "event is outside the canonical prologue, loop body, or drain block",
            ));
        }
    }
    prologue.sort_by_key(|(position, event)| (*position, event.site.operation()));
    body.sort_by_key(|(position, event)| (*position, event.site.operation()));
    drain.sort_by_key(|(position, event)| (*position, event.site.operation()));
    let prologue = prologue
        .into_iter()
        .map(|(_, event)| event)
        .collect::<Vec<_>>();
    let drain = drain
        .into_iter()
        .map(|(_, event)| event)
        .collect::<Vec<_>>();

    let expected_prologue = usize::try_from(distance).unwrap_or(usize::MAX) * 2;
    if prologue.len() != expected_prologue {
        return Err(invalid(
            pipeline,
            prologue.first().map(|event| event.site),
            &format!(
                "dynamic pipeline requires {expected_prologue} prologue events for prefetch distance {distance}, found {}",
                prologue.len()
            ),
        ));
    }
    for epoch in 0..u64::from(distance) {
        require_event(
            context,
            pipeline,
            &prologue[(epoch as usize) * 2],
            PipelineEventKindAttr::Stage,
            EpochExpectationV1::Constant(epoch),
            buffers,
            equivalence_resources,
        )?;
        require_event(
            context,
            pipeline,
            &prologue[(epoch as usize) * 2 + 1],
            PipelineEventKindAttr::Commit,
            EpochExpectationV1::Constant(epoch),
            buffers,
            equivalence_resources,
        )?;
    }

    let body = body.into_iter().map(|(_, event)| event).collect::<Vec<_>>();
    if body.len() != 5 {
        return Err(invalid(
            pipeline,
            body.first().map(|event| event.site),
            &format!(
                "dynamic loop body requires exactly five lifecycle events, found {}",
                body.len()
            ),
        ));
    }
    let body_induction = |event: &EventSiteV1| {
        summary
            .inductions
            .get(&event.site.block())
            .copied()
            .expect("body events are classified only in summarized blocks")
    };
    let expected = [
        (PipelineEventKindAttr::Stage, u64::from(distance)),
        (PipelineEventKindAttr::Commit, u64::from(distance)),
        (PipelineEventKindAttr::Wait, 0),
        (PipelineEventKindAttr::Consume, 0),
        (PipelineEventKindAttr::Release, 0),
    ];
    for (event, (kind, offset)) in body.iter().zip(expected) {
        require_event(
            context,
            pipeline,
            event,
            kind,
            EpochExpectationV1::Offset(body_induction(event), offset),
            buffers,
            equivalence_resources,
        )?;
    }

    let expected_drain = usize::try_from(distance).unwrap_or(usize::MAX) * 3;
    if drain.len() != expected_drain {
        return Err(invalid(
            pipeline,
            drain.first().map(|event| event.site),
            &format!(
                "dynamic pipeline requires {expected_drain} drain events for prefetch distance {distance}, found {}",
                drain.len()
            ),
        ));
    }
    for offset in 0..u64::from(distance) {
        for (inner, kind) in [
            PipelineEventKindAttr::Wait,
            PipelineEventKindAttr::Discard,
            PipelineEventKindAttr::Release,
        ]
        .into_iter()
        .enumerate()
        {
            require_event(
                context,
                pipeline,
                &drain[(offset as usize) * 3 + inner],
                kind,
                EpochExpectationV1::Offset(summary.bound, offset),
                buffers,
                equivalence_resources,
            )?;
        }
    }
    verify_dynamic_accesses(
        context,
        pipeline,
        buffers,
        schedule,
        accesses,
        summary,
        domain_reads,
        equivalence_resources,
    )
}

#[derive(Clone, Copy)]
enum EpochExpectationV1 {
    Constant(u64),
    Offset(Value, u64),
}

fn require_event(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    event: &EventSiteV1,
    kind: PipelineEventKindAttr,
    epoch: EpochExpectationV1,
    buffers: u32,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(), PlironPipelineProtocolFindingV1> {
    if event.kind != Some(kind) {
        return Err(invalid(
            pipeline,
            Some(event.site),
            &format!("expected {kind:?}, found {:?}", event.kind),
        ));
    }
    let epoch_matches = match epoch {
        EpochExpectationV1::Constant(expected) => {
            index_constant(context, event.epoch) == Some(expected)
        }
        EpochExpectationV1::Offset(base, offset) => {
            index_offset(context, event.epoch, base, equivalence_resources) == Some(offset)
        }
    };
    if !epoch_matches {
        return Err(invalid(
            pipeline,
            Some(event.site),
            "event uses the wrong epoch for its pipeline phase",
        ));
    }
    if !slot_is_epoch_modulo(
        context,
        event.slot,
        event.epoch,
        buffers,
        equivalence_resources,
    ) {
        return Err(invalid(
            pipeline,
            Some(event.site),
            &format!("slot is not epoch % {buffers}"),
        ));
    }
    Ok(())
}

#[derive(Clone)]
enum DynamicPipelineActionV1 {
    Event(EventSiteV1),
    Access(AccessSiteV1),
}

impl DynamicPipelineActionV1 {
    const fn site(&self) -> PlironOperationSiteV1 {
        match self {
            Self::Event(event) => event.site,
            Self::Access(access) => access.site,
        }
    }
}

fn verify_dynamic_accesses(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    buffers: u32,
    schedule: &[EventSiteV1],
    accesses: &[AccessSiteV1],
    summary: &CanonicalEpochLoopV1,
    domain_reads: &[PlironOperationSiteV1],
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(usize, usize, bool), PlironPipelineProtocolFindingV1> {
    if accesses.is_empty() {
        return Ok((0, 0, true));
    }
    let prologue_start = summary
        .prologue
        .iter()
        .position(|block| *block == pipeline.block())
        .unwrap_or(0);
    let mut ordered_blocks = summary.prologue[prologue_start..].to_vec();
    ordered_blocks.extend(summary.body.iter().copied());
    ordered_blocks.extend(summary.drain.iter().copied());
    let positions = ordered_blocks
        .iter()
        .enumerate()
        .map(|(position, block)| (*block, position))
        .collect::<HashMap<_, _>>();
    let order_key = |site: PlironOperationSiteV1| {
        positions
            .get(&site.block())
            .copied()
            .map(|position| (position, site.operation()))
    };
    let mut actions = Vec::with_capacity(accesses.len() + schedule.len());
    for access in accesses {
        if order_key(access.site).is_none() {
            return Err(invalid(
                pipeline,
                Some(access.site),
                "pipeline-owned storage is accessed outside the canonical prologue, loop body, or drain block",
            ));
        }
        actions.push(DynamicPipelineActionV1::Access(access.clone()));
    }
    for event in schedule {
        actions.push(DynamicPipelineActionV1::Event(*event));
    }
    actions.sort_by_key(|action| {
        order_key(action.site()).unwrap_or((usize::MAX, action.site().operation()))
    });

    let mut staging = None::<(Value, usize)>;
    let mut consuming = None::<(Value, usize)>;
    // Keep occurrence order: shared memo warming and bounded equivalence
    // attempts must not depend on a randomized hash-set traversal. Duplicate
    // rows do not change the bidirectional membership predicate below.
    let mut staging_coordinates = Vec::<Vec<Value>>::new();
    let mut consuming_coordinates = Vec::<Vec<Value>>::new();
    let mut canonical_coordinates = None::<Vec<Vec<Value>>>;
    let mut empty_staging_windows = 0_usize;
    let mut empty_consuming_windows = 0_usize;
    let mut staged_writes = 0;
    let mut consuming_reads = 0;
    let mut coverage_reads = 0;
    for action in actions {
        match action {
            DynamicPipelineActionV1::Event(event) => match event.kind {
                Some(PipelineEventKindAttr::Stage) => {
                    staging = Some((event.epoch, event.site.block()));
                    staging_coordinates.clear();
                }
                Some(PipelineEventKindAttr::Commit) => {
                    if staging_coordinates.is_empty() {
                        empty_staging_windows += 1;
                    } else {
                        require_matching_pipeline_coordinates_v1(
                            context,
                            pipeline,
                            event.site,
                            "staging",
                            &staging_coordinates,
                            &mut canonical_coordinates,
                            equivalence_resources,
                        )?;
                    }
                    staging = None;
                }
                Some(PipelineEventKindAttr::Consume) => {
                    consuming = Some((event.epoch, event.site.block()));
                    consuming_coordinates.clear();
                }
                Some(PipelineEventKindAttr::Release) => {
                    if consuming.is_some() {
                        if consuming_coordinates.is_empty() {
                            empty_consuming_windows += 1;
                        } else {
                            require_matching_pipeline_coordinates_v1(
                                context,
                                pipeline,
                                event.site,
                                "consuming",
                                &consuming_coordinates,
                                &mut canonical_coordinates,
                                equivalence_resources,
                            )?;
                        }
                    }
                    consuming_coordinates.clear();
                    consuming = None;
                }
                Some(PipelineEventKindAttr::Wait | PipelineEventKindAttr::Discard) | None => {}
            },
            DynamicPipelineActionV1::Access(access) => {
                let expected = match access.kind {
                    AccessKindAttr::Write => staging.map(|(epoch, block)| (epoch, block, true)),
                    AccessKindAttr::Read => consuming.map(|(epoch, block)| (epoch, block, false)),
                    _ => None,
                };
                let Some((epoch, epoch_block, is_write)) = expected else {
                    return Err(invalid(
                        pipeline,
                        Some(access.site),
                        &format!(
                            "{:?} access is outside its legal stage-to-commit or consume-to-release window",
                            access.kind
                        ),
                    ));
                };
                if !slot_is_epoch_modulo_across_loop_blocks(
                    context,
                    PipelineBlockValueV1 {
                        value: access.slot,
                        block: access.site.block(),
                    },
                    PipelineBlockValueV1 {
                        value: epoch,
                        block: epoch_block,
                    },
                    buffers,
                    summary,
                    equivalence_resources,
                ) {
                    return Err(invalid(
                        pipeline,
                        Some(access.site),
                        &format!(
                            "{:?} access does not use the live epoch modulo {buffers} as its leading ring index",
                            access.kind
                        ),
                    ));
                }
                if is_write {
                    staging_coordinates.push(access.indices[1..].to_vec());
                    staged_writes += 1;
                } else {
                    consuming_reads += 1;
                    if summary.optional_blocks.contains(&access.site.block()) {
                        // A zero-trip or bypassable child cannot establish
                        // coverage. Still authenticate its read against the
                        // already initialized tile and its live consume epoch.
                        let initialized = canonical_coordinates.as_ref().is_some_and(|rows| {
                            rows.iter().any(|row| {
                                row.len() == access.indices.len() - 1
                                    && row.iter().zip(&access.indices[1..]).all(|(left, right)| {
                                        index_values_equivalent(
                                            context,
                                            *left,
                                            *right,
                                            equivalence_resources,
                                        )
                                    })
                            })
                        });
                        if !initialized && !domain_reads.contains(&access.site) {
                            return Err(invalid(
                                pipeline,
                                Some(access.site),
                                "optional nested read is not in the initialized symbolic tile",
                            ));
                        }
                    } else {
                        consuming_coordinates.push(access.indices[1..].to_vec());
                        coverage_reads += 1;
                    }
                }
            }
        }
    }
    if (consuming_reads != 0 && empty_staging_windows != 0)
        || (coverage_reads != 0 && empty_consuming_windows != 0)
    {
        return Err(invalid(
            pipeline,
            None,
            "a consumed symbolic tile is not initialized in every prologue and steady-state epoch",
        ));
    }
    Ok((staged_writes, consuming_reads, true))
}

fn require_matching_pipeline_coordinates_v1(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    event: PlironOperationSiteV1,
    phase: &str,
    coordinates: &[Vec<Value>],
    canonical: &mut Option<Vec<Vec<Value>>>,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(), PlironPipelineProtocolFindingV1> {
    debug_assert!(!coordinates.is_empty());
    match canonical {
        None => *canonical = Some(coordinates.to_vec()),
        Some(expected)
            if coordinate_sets_equivalent_v1(
                context,
                expected,
                coordinates,
                equivalence_resources,
            ) => {}
        Some(_) => {
            return Err(invalid(
                pipeline,
                Some(event),
                &format!(
                    "{phase} epoch coordinates do not match the staged/consumed symbolic tile"
                ),
            ));
        }
    }
    Ok(())
}

fn coordinate_sets_equivalent_v1(
    context: &Context,
    left: &[Vec<Value>],
    right: &[Vec<Value>],
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> bool {
    if equivalence_resources.exhausted() {
        return false;
    }
    let mut contains_equivalent = |haystack: &[Vec<Value>], needle: &[Value]| {
        for candidate in haystack {
            if equivalence_resources.exhausted() {
                return false;
            }
            if candidate.len() == needle.len()
                && candidate
                    .iter()
                    .copied()
                    .zip(needle.iter().copied())
                    .all(|(left, right)| {
                        index_values_equivalent(context, left, right, equivalence_resources)
                    })
            {
                return true;
            }
        }
        false
    };
    left.iter()
        .all(|coordinate| contains_equivalent(right, coordinate))
        && right
            .iter()
            .all(|coordinate| contains_equivalent(left, coordinate))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SlotStateV1 {
    Free,
    Staged(u64),
    Committed(u64),
    Ready(u64),
    Consuming(u64),
    Discarding(u64),
}

// This private refinement is only for exact, guarded reads of a completely
// published one-dimensional workgroup tile. It does not equate coordinates.
#[allow(clippy::too_many_arguments)]
fn full_workgroup_initialized_reads_v1(
    context: &Context,
    control: &PipelineControlContextV1<'_>,
    indices: &mut ConcreteIndexContextV1<'_>,
    pipeline: PlironOperationSiteV1,
    summary: &CanonicalEpochLoopV1,
    schedule: &[EventSiteV1],
    accesses: &[AccessSiteV1],
    roots: &HashSet<Value>,
    visit_limit: usize,
    resources: &mut EquivalenceResourceMeterV1,
) -> Result<Vec<PlironOperationSiteV1>, PlironPipelineProtocolFindingV1> {
    use dialect_kernel::{InvocationIndexOp, MemorySpaceAttr};
    // Existing per-access setup allowance covers this allocation-free selector.
    // Constant/pointwise routes do not admit or reserve the new refinement.
    if summary.repeated_blocks.is_empty()
        || !accesses.iter().any(|access| {
            access.kind == AccessKindAttr::Read
                && summary.optional_blocks.contains(&access.site.block())
                && access
                    .indices
                    .get(1)
                    .is_some_and(|value| value.defining_op().is_none())
        })
    {
        return Ok(Vec::new());
    }
    let invocation_modulus = |coordinate: Value| {
        let definition = coordinate.defining_op()?;
        let operation = Operation::get_op_dyn(definition, context);
        let binary = operation.downcast_ref::<IndexBinaryOp>()?;
        if binary.kind(context) != Some(IndexBinaryKindAttr::Remainder) {
            return None;
        }
        let invocation = binary.lhs(context).defining_op()?;
        let invocation_op = Operation::get_op_dyn(invocation, context);
        let invocation = invocation_op.downcast_ref::<InvocationIndexOp>()?;
        if invocation.dimension(context) != Some(0) {
            return None;
        }
        Some((
            index_constant(context, binary.rhs(context))?,
            invocation.launch_extent(context)?,
        ))
    };
    if !accesses.iter().any(|access| {
        access.kind == AccessKindAttr::Write
            && access.indices.len() == 2
            && invocation_modulus(access.indices[1]).is_some()
    }) {
        return Ok(Vec::new());
    }

    // Admit before geometry/CFG scans, allocation, uniformity walks or guard
    // queries. The descriptor includes the already-live pipeline scratch.
    let bound = pipeline_full_writer_domain_resource_bound_v1(indices.census)
        .and_then(|bound| {
            let limits = indices
                .analyses
                .remaining_resource_limits(ProductionAnalysisResourcePhaseV1::PipelineProtocol)?;
            limits.require(ProductionAnalysisResourcePhaseV1::PipelineProtocol, bound)
        })
        .and_then(|bound| {
            indices.analyses.admit_retained_resource_upper_bound(
                ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                bound,
            )
        });
    if bound.is_err() {
        indices.unavailable = true;
        return Err(PlironPipelineProtocolFindingV1::AnalysisIncomplete {
            detail: "full workgroup writer-domain refinement could not be admitted".to_owned(),
        });
    }
    let inventory = control.inventory;
    let discovery = control.discovery;
    let create_op = Operation::get_op_dyn(pipeline.pointer(), context);
    let Some(create) = create_op.downcast_ref::<PipelineCreateOp>() else {
        return Ok(Vec::new());
    };
    let view = create.view(context);
    let Some(definition) = view.defining_op() else {
        return Ok(Vec::new());
    };
    let view_op = Operation::get_op_dyn(definition, context);
    let Some(ranked_view) = view_op.downcast_ref::<RankedViewOp>() else {
        return Ok(Vec::new());
    };
    let Some(ty) = ranked_view.view_type(context) else {
        return Ok(Vec::new());
    };
    let view_type = ty.deref(context);
    let [ring, extent] = view_type.shape() else {
        return Ok(Vec::new());
    };
    if ranked_view.memory_space(context) != Some(MemorySpaceAttr::Workgroup)
        || *extent == 0
        || *extent == dialect_kernel::DYNAMIC_EXTENT
        || Some(*ring)
            != create
                .pipeline_type(context)
                .map(|ty| u64::from(ty.deref(context).buffers()))
    {
        return Ok(Vec::new());
    }
    let Ok(Some(layout)) =
        super::pliron_invocation_trace::pliron_execution_layout_with_inventory_v1(
            context, inventory,
        )
    else {
        return Ok(Vec::new());
    };
    if layout.execution_domain != dialect_gpu::ExecutionDomainAttr::FullPhysicalWorkgroups
        || layout.workgroup_extents != [*extent, 1, 1]
        || layout.global_extents[0] == 0
        || !layout.global_extents[0].is_multiple_of(*extent)
        || layout.global_extents[1..] != [1, 1]
    {
        return Ok(Vec::new());
    }
    let Some(latch) = summary.body.last().copied() else {
        return Ok(Vec::new());
    };
    // All writes, not just one promising write, must be lane-bijective and
    // execute on every required writer path. Other writes cannot borrow coverage.
    if accesses
        .iter()
        .filter(|access| access.kind == AccessKindAttr::Write)
        .any(|access| {
            let anchor = if summary.body_members.contains(&access.site.block()) {
                latch
            } else {
                summary.header
            };
            access.indices.len() != 2
                || !discovery.dominators[anchor].contains(&access.site.block())
                || invocation_modulus(access.indices[1]).is_none_or(|(modulus, declared)| {
                    modulus != *extent || (declared != 0 && declared != layout.global_extents[0])
                })
        })
    {
        return Ok(Vec::new());
    }
    if !full_workgroup_epoch_entry_v1(
        context,
        inventory,
        discovery,
        summary,
        schedule,
        roots,
        visit_limit,
        indices.census,
        resources,
    ) {
        return Ok(Vec::new());
    }
    let mut proven = Vec::new();
    for access in accesses {
        if access.kind != AccessKindAttr::Read
            || access.indices.len() != 2
            || !summary.optional_blocks.contains(&access.site.block())
        {
            continue;
        }
        // Roster, owner, view, type and coordinate are the same immutable op
        // examined by ordinary bounds and the action/epoch verifier below.
        let op = Operation::get_op_dyn(access.site.pointer(), context);
        let Some(read) = op.downcast_ref::<RankedAccessOp>() else {
            continue;
        };
        if read.view(context) == view
            && read.indices(context) == access.indices
            && full_workgroup_guarded_coordinate_v1(
                context,
                inventory,
                access.site.block(),
                access.indices[1],
                *extent,
            )
        {
            proven.push(access.site);
        }
    }
    Ok(proven)
}

fn full_workgroup_guarded_coordinate_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    read_block: usize,
    coordinate: Value,
    extent: u64,
) -> bool {
    use crate::production_analysis::pliron_control_edges_v1::ControlViewV1;
    let target = inventory.blocks()[read_block];
    let mut incoming = 0;
    for block in inventory.blocks().iter().copied() {
        let Some(terminator) = block.deref(context).get_terminator(context) else {
            return false;
        };
        let Ok(control) = ControlViewV1::observe(context, terminator) else {
            return false;
        };
        for ordinal in 0..control.successor_count() {
            let Ok(edge) = control.edge(ordinal) else {
                return false;
            };
            if edge.target() != target {
                continue;
            }
            incoming += 1;
            if incoming > 1 {
                return false;
            }
            // This refinement admits only the actual immediate true-edge shape.
            // Every occurrence is examined, including duplicate successor targets.
            let Some((index, bound)) = edge.index_less_than_guard() else {
                return false;
            };
            if !dialect_kernel::is_index_type(index, context)
                || !dialect_kernel::is_index_type(coordinate, context)
                || index_constant(context, bound).is_none_or(|bound| bound > extent)
            {
                return false;
            }
            let mut exact = false;
            for argument in 0..edge.argument_count() {
                let Ok((incoming, formal)) = edge.argument_at(argument) else {
                    return false;
                };
                if formal == coordinate && incoming == index {
                    exact = true;
                }
            }
            if !exact {
                return false;
            }
        }
    }
    incoming == 1
}

// Prove participant uniformity independently of the certificate this creates.
// Every preheader cycle must have the existing finite read-only loop proof.
fn full_workgroup_epoch_entry_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    discovery: &EpochLoopDiscoveryV1,
    summary: &CanonicalEpochLoopV1,
    schedule: &[EventSiteV1],
    roots: &HashSet<Value>,
    visit_limit: usize,
    census: ProductionAnalysisInputCensusV1,
    resources: &mut EquivalenceResourceMeterV1,
) -> bool {
    let mut prefix = HashSet::new();
    let mut pending = vec![0];
    while let Some(block) = pending.pop() {
        if block == summary.header || !prefix.insert(block) {
            continue;
        }
        pending.extend(discovery.cfg_successors[block].iter().copied());
    }
    // Reuse the existing finite read-only loop contract for the actual
    // constant-trip prelude loops. Replaying it uses a separately paid instance
    // of the existing equivalence meter; baseline/no-new-route budgets do not move.
    let Ok((queries, pairs)) = pipeline_full_writer_domain_equivalence_limits_v1(census) else {
        return false;
    };
    let Ok(mut finite_resources) =
        EquivalenceResourceMeterV1::new_with_observation_v1(queries, pairs, resources.observer)
    else {
        return false;
    };
    let mut predecessors = vec![Vec::new(); inventory.blocks().len()];
    for (source, targets) in discovery.cfg_successors.iter().enumerate() {
        for target in targets {
            predecessors[*target].push(source);
        }
    }
    let mut proof_edges = discovery.cfg_successors.clone();
    for child in &discovery.loops {
        if !prefix.contains(&child.header)
            || !child
                .body_members
                .iter()
                .all(|block| prefix.contains(block))
        {
            continue;
        }
        let (Some(latch), Some(exit)) = (child.body.last().copied(), child.drain.first().copied())
        else {
            return false;
        };
        if repeatable_epoch_child_v1(
            context,
            inventory,
            child,
            latch,
            exit,
            &predecessors,
            &discovery.cfg_successors,
            &mut finite_resources,
        ) {
            proof_edges[latch].retain(|target| *target != child.header);
            proof_edges[latch].push(exit);
        }
    }
    if finite_resources.exhausted() {
        return false;
    }
    let Some(order) = acyclic_loop_body_order_v1(&prefix, &proof_edges) else {
        return false;
    };
    let mut reaches = vec![false; inventory.blocks().len()];
    let mut must_reach = reaches.clone();
    reaches[summary.header] = true;
    must_reach[summary.header] = true;
    for block in order.iter().rev().copied() {
        let successors = &proof_edges[block];
        reaches[block] = successors.iter().any(|target| reaches[*target]);
        must_reach[block] =
            !successors.is_empty() && successors.iter().all(|target| must_reach[*target]);
    }
    for block in order {
        if !reaches[block] || proof_edges[block].iter().all(|target| reaches[*target]) {
            continue;
        }
        if !discovery.dominators[summary.header].contains(&block) {
            return false;
        }
        let Some(term) = inventory.blocks()[block]
            .deref(context)
            .get_terminator(context)
        else {
            return false;
        };
        let op = Operation::get_op_dyn(term, context);
        let raw = term.deref(context);
        let comparison = op.downcast_ref::<IndexLessThanBranchArgsOp>().is_some()
            || op.downcast_ref::<IndexEqualBranchArgsOp>().is_some()
            || op
                .downcast_ref::<dialect_kernel::IndexLessThanBranchOp>()
                .is_some()
            || op
                .downcast_ref::<dialect_kernel::IndexEqualBranchOp>()
                .is_some();
        if !comparison || raw.get_num_operands() < 2 {
            return false;
        }
        for operand in 0..2 {
            if is_uniform_value_with_observation_v1(
                context,
                raw.get_operand(operand),
                roots,
                visit_limit,
                resources.observer,
            ) != Ok(true)
            {
                return false;
            }
        }
    }
    // Once staging begins there is no exit before the uniform epoch loop;
    // all prologue publication events execute on every path into that loop.
    schedule
        .iter()
        .filter(|event| summary.prologue.contains(&event.site.block()))
        .all(|event| {
            discovery.dominators[summary.header].contains(&event.site.block())
                && must_reach[event.site.block()]
        })
}
