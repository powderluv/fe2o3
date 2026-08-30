//! Workgroup-memory initialization, publication, and race verification.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use dialect_gpu::{
    AddressSpaceAttr, BarrierOp, ExecutionDomainAttr, HierarchyAttr, MemoryOrderAttr,
    MemoryScopeAttr,
};
use dialect_kernel::{
    AccessKindAttr, AllocationEffectOp, AnalysisSplitOp, BranchArgsOp, BranchOp,
    IndexEqualBranchArgsOp, IndexEqualBranchOp, IndexLessThanBranchArgsOp, IndexLessThanBranchOp,
    MemorySpaceAttr, PipelineCreateOp, RankedAccessOp, RankedViewOp, ReturnOp, TrapOp,
    is_supported_allocation_effect_contract_v1,
};
#[cfg(test)]
use dialect_kernel::{
    GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    operation::Operation,
    value::Value,
};

use crate::pliron_analysis_manager::{PlironAnalysisManagerV1, PlironMemoryOrderAnalysisFailureV1};
use crate::pliron_barrier::run_pliron_barrier_convergence_check_with_analyses_v1;
use crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1;
use crate::pliron_invocation_trace::{
    PlironTraceLocationV1, pliron_execution_layout_with_inventory_v1,
};
use crate::pliron_memory_order::{PlironMemoryOrderFailureV1, PlironMemoryOrderIssueV1};
use crate::pliron_pipeline_protocol::run_pliron_pipeline_protocol_check_with_analyses_v1;
use crate::pliron_ranked_bounds::run_pliron_ranked_bounds_check_with_analyses_v1;
use crate::{KernelCheckPassKindV1, KernelCheckStatusV1, trace_failure_detail};

pub const MAX_PLIRON_WORKGROUP_FINDINGS_V1: usize = 4_096;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironWorkgroupMemoryFindingV1 {
    BoundsPrerequisiteRejected,
    BarrierPrerequisiteRejected,
    AnalysisIncomplete {
        detail: String,
    },
    ReadBeforeInitialization {
        invocation: Vec<u64>,
        block: usize,
        operation: usize,
        indices: Vec<u64>,
    },
    ConflictingEffects {
        indices: Vec<u64>,
        first_invocation: Vec<u64>,
        first_block: usize,
        first_operation: usize,
        first_access: AccessKindAttr,
        second_invocation: Vec<u64>,
        second_block: usize,
        second_operation: usize,
        second_access: AccessKindAttr,
    },
    FindingLimitExceeded,
}

impl PlironWorkgroupMemoryFindingV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::ReadBeforeInitialization { .. } | Self::ConflictingEffects { .. } => {
                KernelCheckStatusV1::Rejected
            }
            Self::BoundsPrerequisiteRejected
            | Self::BarrierPrerequisiteRejected
            | Self::AnalysisIncomplete { .. }
            | Self::FindingLimitExceeded => KernelCheckStatusV1::Incomplete,
        }
    }
}

impl fmt::Display for PlironWorkgroupMemoryFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundsPrerequisiteRejected => formatter.write_str(
                "error[FE2O3-WORKGROUP-000]: bounds prerequisite rejected before workgroup-memory analysis",
            ),
            Self::BarrierPrerequisiteRejected => formatter.write_str(
                "error[FE2O3-WORKGROUP-000]: barrier-convergence prerequisite rejected before workgroup-memory analysis",
            ),
            Self::AnalysisIncomplete { detail } => write!(
                formatter,
                "error[FE2O3-WORKGROUP-003]: cannot prove workgroup-memory safety: {detail}",
            ),
            Self::ReadBeforeInitialization {
                invocation,
                block,
                operation,
                indices,
            } => write!(
                formatter,
                "error[FE2O3-WORKGROUP-001]: invocation {invocation:?} reads uninitialized workgroup address {indices:?} at block {block} op {operation}; failed proof: the address is not initialized by this invocation and no convergent workgroup-memory barrier published a prior write; help: initialize the address and publish it with a workgroup acquire-release barrier before the read",
            ),
            Self::ConflictingEffects {
                indices,
                first_invocation,
                first_block,
                first_operation,
                first_access,
                second_invocation,
                second_block,
                second_operation,
                second_access,
            } => write!(
                formatter,
                "error[FE2O3-WORKGROUP-002]: conflicting {first_access:?}/{second_access:?} workgroup-memory effects at address {indices:?}; invocation {first_invocation:?} block {first_block} op {first_operation} conflicts with invocation {second_invocation:?} block {second_block} op {second_operation}; help: use disjoint coordinates, a convergent workgroup barrier between epochs, or compatible atomic operations",
            ),
            Self::FindingLimitExceeded => formatter.write_str(
                "error[FE2O3-WORKGROUP-003]: workgroup-memory finding limit exceeded",
            ),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironWorkgroupMemoryReportV1 {
    findings: Vec<PlironWorkgroupMemoryFindingV1>,
}

impl PlironWorkgroupMemoryReportV1 {
    pub const fn pass(&self) -> KernelCheckPassKindV1 {
        KernelCheckPassKindV1::WorkgroupMemory
    }

    pub fn status(&self) -> KernelCheckStatusV1 {
        self.findings
            .iter()
            .fold(KernelCheckStatusV1::Clean, |status, finding| {
                status.join(finding.status())
            })
    }

    pub fn findings(&self) -> &[PlironWorkgroupMemoryFindingV1] {
        &self.findings
    }

    pub fn is_clean(&self) -> bool {
        self.status() == KernelCheckStatusV1::Clean
    }

    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironWorkgroupMemoryCheckErrorV1 {
    report: PlironWorkgroupMemoryReportV1,
}

impl PlironWorkgroupMemoryCheckErrorV1 {
    pub fn report(&self) -> &PlironWorkgroupMemoryReportV1 {
        &self.report
    }
}

impl fmt::Display for PlironWorkgroupMemoryCheckErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, finding) in self.report.findings.iter().enumerate() {
            if index != 0 {
                formatter.write_str("\n")?;
            }
            finding.fmt(formatter)?;
        }
        Ok(())
    }
}

impl std::error::Error for PlironWorkgroupMemoryCheckErrorV1 {}

pub fn run_pliron_workgroup_memory_check_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironWorkgroupMemoryReportV1 {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    if !run_pliron_ranked_bounds_check_with_analyses_v1(context, function, &mut analyses).is_clean()
    {
        return one(PlironWorkgroupMemoryFindingV1::BoundsPrerequisiteRejected);
    }
    if !run_pliron_barrier_convergence_check_with_analyses_v1(context, function, &mut analyses)
        .is_clean()
    {
        return one(PlironWorkgroupMemoryFindingV1::BarrierPrerequisiteRejected);
    }
    run_pliron_workgroup_memory_check_with_analyses_v1(context, function, &mut analyses)
}

pub(crate) fn run_pliron_workgroup_memory_check_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> PlironWorkgroupMemoryReportV1 {
    analyses.prepare_function_inventory(context, function);
    let inventory = match analyses.function_inventory_handle() {
        Ok(inventory) => inventory,
        Err(_) => {
            return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                detail: "the bounded function inventory limit was exceeded".to_owned(),
            });
        }
    };
    let (workgroup_access_views, pipeline_views, collective_effect_sites) =
        inventory.operations().iter().fold(
            (
                HashSet::<Value>::new(),
                Vec::new(),
                HashSet::<PlironTraceLocationV1>::new(),
            ),
            |(mut accesses, mut pipelines, mut collective_effects), site| {
                let operation = Operation::get_op_dyn(site.pointer(), context);
                if let Some(access) = operation.downcast_ref::<RankedAccessOp>()
                    && access
                        .view(context)
                        .defining_op()
                        .is_some_and(|definition| {
                            Operation::get_op_dyn(definition, context)
                                .downcast_ref::<RankedViewOp>()
                                .is_some_and(|view| {
                                    view.memory_space(context) == Some(MemorySpaceAttr::Workgroup)
                                })
                        })
                {
                    accesses.insert(access.view(context));
                }
                if let Some(create) = operation.downcast_ref::<PipelineCreateOp>() {
                    pipelines.push((site.block(), site.operation(), create.view(context)));
                }
                if operation
                    .downcast_ref::<AllocationEffectOp>()
                    .is_some_and(|effect| {
                        effect.memory_space(context) == Some(MemorySpaceAttr::Workgroup)
                    })
                {
                    collective_effects.insert(PlironTraceLocationV1 {
                        block: site.block(),
                        operation: site.operation(),
                    });
                }
                (accesses, pipelines, collective_effects)
            },
        );
    if !collective_effect_sites.is_empty() {
        let layout = match pliron_execution_layout_with_inventory_v1(context, &inventory) {
            Ok(Some(layout)) => layout,
            Ok(None) => {
                return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                    detail: "the collective transpose lifecycle has no execution layout".to_owned(),
                });
            }
            Err(failure) => {
                return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                    detail: trace_failure_detail(failure),
                });
            }
        };
        let workgroup_size = layout
            .workgroup_extents
            .into_iter()
            .try_fold(1_u64, u64::checked_mul);
        if workgroup_size != Some(64)
            || layout.subgroup_size != 64
            || layout.execution_domain != ExecutionDomainAttr::FullPhysicalWorkgroups
        {
            return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                detail: "the coordinate-free gfx950 transpose tile requires one full physical Wave64 per workgroup".to_owned(),
            });
        }
        if let Err(detail) = validate_collective_transpose_lifecycle_v1(
            context,
            &inventory,
            &collective_effect_sites,
        ) {
            return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete { detail });
        }
    }
    if workgroup_access_views.is_empty() {
        return PlironWorkgroupMemoryReportV1 { findings: vec![] };
    }
    if !pipeline_views.is_empty() {
        let pipeline =
            run_pliron_pipeline_protocol_check_with_analyses_v1(context, function, analyses);
        if pipeline.is_clean() {
            let certified = pipeline
                .certificates()
                .iter()
                .filter(|certificate| certificate.access_refinement_proven())
                .filter_map(|certificate| {
                    pipeline_views
                        .iter()
                        .find(|(block, operation, _)| {
                            *block == certificate.pipeline_block()
                                && *operation == certificate.pipeline_operation()
                        })
                        .map(|(_, _, view)| *view)
                })
                .collect::<HashSet<_>>();
            if workgroup_access_views.is_subset(&certified) {
                return PlironWorkgroupMemoryReportV1 { findings: vec![] };
            }
        }
    }
    analyses.prepare_memory_order(context, function);
    let memory_order = match analyses.memory_order() {
        Ok(analysis) => analysis,
        Err(failure) => {
            return one(PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                detail: memory_order_failure_detail(failure),
            });
        }
    };
    let mut findings = Vec::new();
    for issue in memory_order.issues() {
        let finding = match issue {
            PlironMemoryOrderIssueV1::ReadBeforeInitialization {
                invocation,
                location,
                address,
            } => PlironWorkgroupMemoryFindingV1::ReadBeforeInitialization {
                invocation: invocation.clone(),
                block: location.block(),
                operation: location.operation(),
                indices: address.indices().to_vec(),
            },
            PlironMemoryOrderIssueV1::ConflictingEffects {
                address,
                first_invocation,
                first_location,
                first_access,
                second_invocation,
                second_location,
                second_access,
            } => PlironWorkgroupMemoryFindingV1::ConflictingEffects {
                indices: address.indices().to_vec(),
                first_invocation: first_invocation.clone(),
                first_block: first_location.block(),
                first_operation: first_location.operation(),
                first_access: *first_access,
                second_invocation: second_invocation.clone(),
                second_block: second_location.block(),
                second_operation: second_location.operation(),
                second_access: *second_access,
            },
            PlironMemoryOrderIssueV1::AtomicReadFromUnresolved {
                invocation,
                location,
                address,
                detail,
            } => PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                detail: format!(
                    "invocation {invocation:?} block {} op {} cannot derive read-from for workgroup address {:?}: {detail}",
                    location.block(),
                    location.operation(),
                    address.indices(),
                ),
            },
        };
        if findings.len() == MAX_PLIRON_WORKGROUP_FINDINGS_V1 {
            return one(PlironWorkgroupMemoryFindingV1::FindingLimitExceeded);
        }
        findings.push(finding);
    }
    PlironWorkgroupMemoryReportV1 { findings }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CollectiveTransposePathEventV1 {
    Allocation {
        location: PlironTraceLocationV1,
        access: AccessKindAttr,
        allocation_origin: u64,
        noalias_class: u64,
    },
    Barrier {
        location: PlironTraceLocationV1,
        execution_scope: HierarchyAttr,
        memory_scope: MemoryScopeAttr,
        address_space: AddressSpaceAttr,
        order: MemoryOrderAttr,
    },
}

const MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1: usize = 3;
const MAX_COLLECTIVE_TRANSPOSE_TRAP_TRACES_V1: usize = 8;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CollectiveTransposePathSummaryV1 {
    normal: Option<Vec<CollectiveTransposePathEventV1>>,
    trapped: Vec<Vec<CollectiveTransposePathEventV1>>,
}

fn prepend_collective_path_v1(
    local: &[CollectiveTransposePathEventV1],
    summary: CollectiveTransposePathSummaryV1,
) -> Result<CollectiveTransposePathSummaryV1, String> {
    let prepend = |mut suffix: Vec<CollectiveTransposePathEventV1>| {
        let total = local
            .len()
            .checked_add(suffix.len())
            .ok_or_else(|| "collective transpose path length overflowed".to_owned())?;
        if total > MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1 {
            return Err(format!(
                "collective transpose path has more than {MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1} events"
            ));
        }
        let mut trace = Vec::with_capacity(total);
        trace.extend_from_slice(local);
        trace.append(&mut suffix);
        Ok(trace)
    };
    let normal = summary.normal.map(&prepend).transpose()?;
    let trapped = summary
        .trapped
        .into_iter()
        .map(prepend)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CollectiveTransposePathSummaryV1 { normal, trapped })
}

fn merge_collective_path_v1(
    complete: &mut CollectiveTransposePathSummaryV1,
    candidate: CollectiveTransposePathSummaryV1,
) -> Result<(), String> {
    if let Some(candidate_normal) = candidate.normal {
        if let Some(first_normal) = &complete.normal
            && first_normal != &candidate_normal
        {
            return Err("normal paths have different collective transpose traces".to_owned());
        }
        complete.normal = Some(candidate_normal);
    }
    for candidate_trap in candidate.trapped {
        if !complete.trapped.contains(&candidate_trap) {
            if complete.trapped.len() == MAX_COLLECTIVE_TRANSPOSE_TRAP_TRACES_V1 {
                return Err(format!(
                    "collective transpose CFG has more than {MAX_COLLECTIVE_TRANSPOSE_TRAP_TRACES_V1} distinct terminal trap traces"
                ));
            }
            complete.trapped.push(candidate_trap);
        }
    }
    Ok(())
}

fn validate_collective_trap_paths_v1(
    normal: &[CollectiveTransposePathEventV1],
    trapped: &[Vec<CollectiveTransposePathEventV1>],
) -> Result<(), String> {
    if trapped
        .iter()
        .any(|trace| !trace.is_empty() && trace.as_slice() != normal)
    {
        return Err(
            "a terminal trap path partially executes the collective transpose lifecycle".to_owned(),
        );
    }
    Ok(())
}

fn collective_block_events_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    block: usize,
    terminator: Ptr<Operation>,
) -> Result<Vec<CollectiveTransposePathEventV1>, String> {
    let mut events = Vec::with_capacity(MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1);
    for site in inventory.block_operations(block) {
        if site.pointer() == terminator {
            continue;
        }
        let operation = Operation::get_op_dyn(site.pointer(), context);
        if let Some(effect) = operation.downcast_ref::<AllocationEffectOp>()
            && effect.memory_space(context) == Some(MemorySpaceAttr::Workgroup)
        {
            let (Some(access), Some(allocation_origin), Some(noalias_class)) = (
                effect.kind(context),
                effect.allocation_origin(context),
                effect.noalias_class(context),
            ) else {
                return Err(format!(
                    "block {block} op {} has a malformed collective transpose effect",
                    site.operation()
                ));
            };
            if !is_supported_allocation_effect_contract_v1(
                access,
                MemorySpaceAttr::Workgroup,
                allocation_origin,
                noalias_class,
            ) {
                return Err(format!(
                    "block {block} op {} uses a non-reserved collective transpose identity",
                    site.operation()
                ));
            }
            if events.len() == MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1 {
                return Err(format!(
                    "block {block} has more than {MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1} collective transpose events"
                ));
            }
            events.push(CollectiveTransposePathEventV1::Allocation {
                location: PlironTraceLocationV1 {
                    block,
                    operation: site.operation(),
                },
                access,
                allocation_origin,
                noalias_class,
            });
        } else if let Some(barrier) = operation.downcast_ref::<BarrierOp>() {
            let (Some(execution_scope), Some(memory_scope), Some(address_space), Some(order)) = (
                barrier.execution_scope(context),
                barrier.memory_scope(context),
                barrier.address_space(context),
                barrier.order(context),
            ) else {
                return Err(format!(
                    "block {block} op {} has a malformed collective transpose barrier",
                    site.operation()
                ));
            };
            if events.len() == MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1 {
                return Err(format!(
                    "block {block} has more than {MAX_COLLECTIVE_TRANSPOSE_PATH_EVENTS_V1} collective transpose events"
                ));
            }
            events.push(CollectiveTransposePathEventV1::Barrier {
                location: PlironTraceLocationV1 {
                    block,
                    operation: site.operation(),
                },
                execution_scope,
                memory_scope,
                address_space,
                order,
            });
        }
    }
    Ok(events)
}

fn validate_collective_transpose_lifecycle_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
    expected_sites: &HashSet<PlironTraceLocationV1>,
) -> Result<(), String> {
    let blocks = inventory.blocks();
    let block_indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect::<HashMap<Ptr<BasicBlock>, usize>>();
    let mut successors = vec![Vec::new(); blocks.len()];
    let mut predecessors = vec![Vec::new(); blocks.len()];
    let mut local_events = Vec::with_capacity(blocks.len());
    let mut terminal_kinds = vec![0_u8; blocks.len()];

    for (block_index, block) in blocks.iter().copied().enumerate() {
        let terminator = block
            .deref(context)
            .get_terminator(context)
            .ok_or_else(|| format!("block {block_index} has no terminator"))?;
        local_events.push(collective_block_events_v1(
            context,
            inventory,
            block_index,
            terminator,
        )?);
        let terminator_op = Operation::get_op_dyn(terminator, context);
        let raw = terminator_op.get_operation().deref(context);
        if terminator_op.downcast_ref::<ReturnOp>().is_some() {
            terminal_kinds[block_index] = 1;
        } else if terminator_op.downcast_ref::<TrapOp>().is_some() {
            terminal_kinds[block_index] = 2;
        } else if terminator_op.downcast_ref::<BranchOp>().is_some()
            || terminator_op.downcast_ref::<BranchArgsOp>().is_some()
            || terminator_op
                .downcast_ref::<IndexLessThanBranchOp>()
                .is_some()
            || terminator_op
                .downcast_ref::<IndexLessThanBranchArgsOp>()
                .is_some()
            || terminator_op.downcast_ref::<IndexEqualBranchOp>().is_some()
            || terminator_op
                .downcast_ref::<IndexEqualBranchArgsOp>()
                .is_some()
            || terminator_op.downcast_ref::<AnalysisSplitOp>().is_some()
        {
            for successor in raw.successors() {
                let target = block_indices.get(&successor).copied().ok_or_else(|| {
                    format!("block {block_index} targets a block outside the kernel")
                })?;
                successors[block_index].push(target);
                predecessors[target].push(block_index);
            }
            if successors[block_index].is_empty() {
                return Err(format!("block {block_index} has no CFG successor"));
            }
        } else {
            return Err(format!(
                "block {block_index} has an unsupported collective transpose terminator"
            ));
        }
    }

    let mut remaining = successors.iter().map(Vec::len).collect::<Vec<_>>();
    let mut summaries = vec![None; blocks.len()];
    let mut ready = VecDeque::new();
    for block in 0..blocks.len() {
        match terminal_kinds[block] {
            1 => {
                summaries[block] = Some(CollectiveTransposePathSummaryV1 {
                    normal: Some(local_events[block].clone()),
                    trapped: Vec::new(),
                });
                ready.push_back(block);
            }
            2 => {
                summaries[block] = Some(CollectiveTransposePathSummaryV1 {
                    normal: None,
                    trapped: vec![local_events[block].clone()],
                });
                ready.push_back(block);
            }
            _ => {}
        }
    }
    while let Some(completed) = ready.pop_front() {
        for predecessor in predecessors[completed].iter().copied() {
            remaining[predecessor] = remaining[predecessor]
                .checked_sub(1)
                .ok_or_else(|| "collective transpose CFG accounting underflowed".to_owned())?;
            if remaining[predecessor] != 0 {
                continue;
            }
            let mut summary = CollectiveTransposePathSummaryV1::default();
            for successor in successors[predecessor].iter().copied() {
                let suffix = summaries[successor].clone().ok_or_else(|| {
                    format!("block {predecessor} has an unresolved cyclic CFG successor")
                })?;
                let candidate = prepend_collective_path_v1(&local_events[predecessor], suffix)?;
                merge_collective_path_v1(&mut summary, candidate)?;
            }
            summaries[predecessor] = Some(summary);
            ready.push_back(predecessor);
        }
    }

    let summary = summaries.first().and_then(Option::as_ref).ok_or_else(|| {
        "the collective transpose entry participates in cyclic control flow".to_owned()
    })?;
    let normal = summary
        .normal
        .as_deref()
        .ok_or_else(|| "the collective transpose CFG has no normal return path".to_owned())?;
    validate_collective_trap_paths_v1(normal, &summary.trapped)?;
    let [
        CollectiveTransposePathEventV1::Allocation {
            location: write_location,
            access: AccessKindAttr::Write,
            allocation_origin: write_origin,
            noalias_class: write_class,
        },
        CollectiveTransposePathEventV1::Barrier {
            execution_scope: HierarchyAttr::Workgroup,
            memory_scope: MemoryScopeAttr::Workgroup,
            address_space: AddressSpaceAttr::Workgroup,
            order: MemoryOrderAttr::AcquireRelease,
            ..
        },
        CollectiveTransposePathEventV1::Allocation {
            location: read_location,
            access: AccessKindAttr::Read,
            allocation_origin: read_origin,
            noalias_class: read_class,
        },
    ] = normal
    else {
        return Err(
            "every normal path must execute exactly stage, workgroup acquire-release publication, then read"
                .to_owned(),
        );
    };
    if (write_origin, write_class) != (read_origin, read_class) {
        return Err("the collective transpose stage and read formats differ".to_owned());
    }
    let observed_sites = HashSet::from([*write_location, *read_location]);
    if observed_sites != *expected_sites || expected_sites.len() != 2 {
        return Err(
            "the collective transpose path does not execute every reserved effect exactly once"
                .to_owned(),
        );
    }
    Ok(())
}
fn memory_order_failure_detail(failure: PlironMemoryOrderAnalysisFailureV1) -> String {
    match failure {
        PlironMemoryOrderAnalysisFailureV1::Trace(failure) => trace_failure_detail(failure),
        PlironMemoryOrderAnalysisFailureV1::Provenance(failure) => failure.to_string(),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::UnresolvedAddress { location },
        ) => format!(
            "workgroup address at block {} op {} is unresolved",
            location.block(),
            location.operation(),
        ),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::MismatchedBarrierPhase {
                grid,
                workgroup,
                epoch,
            },
        ) => format!(
            "grid {grid} workgroup {workgroup} has mismatched workgroup-barrier participation at memory epoch {epoch}"
        ),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::SubgroupPublicationUnsupported { .. },
        ) => "subgroup-local LDS publication requires a retained per-subgroup epoch/read-from relation; a subgroup barrier never publishes to sibling waves".to_owned(),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::FencePublicationUnsupported { .. },
        ) => "fence-mediated LDS publication requires a retained read-from/synchronizes-with relation; a non-collective fence is not a workgroup barrier".to_owned(),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::VersionLimitExceeded,
        ) => "workgroup memory-version limit exceeded".to_owned(),
        PlironMemoryOrderAnalysisFailureV1::MemoryOrder(
            PlironMemoryOrderFailureV1::IssueLimitExceeded,
        ) => "workgroup memory-order issue limit exceeded".to_owned(),
    }
}

pub(crate) fn require_pliron_workgroup_memory_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> Result<PlironWorkgroupMemoryReportV1, PlironWorkgroupMemoryCheckErrorV1> {
    let report = run_pliron_workgroup_memory_check_with_analyses_v1(context, function, analyses);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironWorkgroupMemoryCheckErrorV1 { report })
    }
}

pub fn require_pliron_workgroup_memory_safety_before_lowering_v1(
    context: &Context,
    function: &FuncOp,
) -> Result<PlironWorkgroupMemoryReportV1, PlironWorkgroupMemoryCheckErrorV1> {
    let report = run_pliron_workgroup_memory_check_v1(context, function);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironWorkgroupMemoryCheckErrorV1 { report })
    }
}

fn one(finding: PlironWorkgroupMemoryFindingV1) -> PlironWorkgroupMemoryReportV1 {
    PlironWorkgroupMemoryReportV1 {
        findings: vec![finding],
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    fn conflict() -> PlironWorkgroupMemoryFindingV1 {
        PlironWorkgroupMemoryFindingV1::ConflictingEffects {
            indices: vec![0],
            first_invocation: vec![0],
            first_block: 0,
            first_operation: 0,
            first_access: AccessKindAttr::Write,
            second_invocation: vec![1],
            second_block: 0,
            second_operation: 0,
            second_access: AccessKindAttr::Read,
        }
    }

    #[test]
    fn every_workgroup_memory_finding_has_the_shared_status() {
        let incomplete = [
            PlironWorkgroupMemoryFindingV1::BoundsPrerequisiteRejected,
            PlironWorkgroupMemoryFindingV1::BarrierPrerequisiteRejected,
            PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                detail: "unresolved".to_owned(),
            },
            PlironWorkgroupMemoryFindingV1::FindingLimitExceeded,
        ];
        for finding in incomplete {
            assert_eq!(finding.status(), KernelCheckStatusV1::Incomplete);
        }

        let rejected = [
            PlironWorkgroupMemoryFindingV1::ReadBeforeInitialization {
                invocation: vec![0],
                block: 0,
                operation: 0,
                indices: vec![0],
            },
            conflict(),
        ];
        for finding in rejected {
            assert_eq!(finding.status(), KernelCheckStatusV1::Rejected);
        }
    }

    #[test]
    fn rejected_workgroup_finding_dominates_an_incomplete_finding() {
        let report = PlironWorkgroupMemoryReportV1 {
            findings: vec![
                PlironWorkgroupMemoryFindingV1::AnalysisIncomplete {
                    detail: "unresolved".to_owned(),
                },
                conflict(),
            ],
        };
        assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
        assert!(!report.is_clean());
        assert_eq!(
            PlironWorkgroupMemoryReportV1 { findings: vec![] }.status(),
            KernelCheckStatusV1::Clean
        );
    }

    #[test]
    fn collective_transpose_path_summaries_are_bounded_to_the_valid_trace_length() {
        let event = CollectiveTransposePathEventV1::Allocation {
            location: PlironTraceLocationV1 {
                block: 0,
                operation: 0,
            },
            access: AccessKindAttr::Write,
            allocation_origin: GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            noalias_class: GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        };
        let summary = CollectiveTransposePathSummaryV1 {
            normal: Some(Vec::new()),
            trapped: Vec::new(),
        };
        let error = prepend_collective_path_v1(&[event; 4], summary).unwrap_err();
        assert!(error.contains("more than 3 events"));
    }

    #[test]
    fn collective_transpose_trap_paths_may_precede_or_complete_the_normal_trace() {
        let write = CollectiveTransposePathEventV1::Allocation {
            location: PlironTraceLocationV1 {
                block: 0,
                operation: 0,
            },
            access: AccessKindAttr::Write,
            allocation_origin: GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            noalias_class: GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        };
        let barrier = CollectiveTransposePathEventV1::Barrier {
            location: PlironTraceLocationV1 {
                block: 1,
                operation: 0,
            },
            execution_scope: HierarchyAttr::Workgroup,
            memory_scope: MemoryScopeAttr::Workgroup,
            address_space: AddressSpaceAttr::Workgroup,
            order: MemoryOrderAttr::AcquireRelease,
        };
        let read = CollectiveTransposePathEventV1::Allocation {
            location: PlironTraceLocationV1 {
                block: 2,
                operation: 0,
            },
            access: AccessKindAttr::Read,
            allocation_origin: GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
            noalias_class: GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
        };
        let mut merged = CollectiveTransposePathSummaryV1 {
            normal: Some(vec![write, barrier, read]),
            trapped: Vec::new(),
        };
        merge_collective_path_v1(
            &mut merged,
            CollectiveTransposePathSummaryV1 {
                normal: None,
                trapped: vec![Vec::new(), vec![write, barrier, read]],
            },
        )
        .unwrap();
        validate_collective_trap_paths_v1(merged.normal.as_deref().unwrap(), &merged.trapped)
            .unwrap();

        let error = validate_collective_trap_paths_v1(
            merged.normal.as_deref().unwrap(),
            &[vec![write, barrier]],
        )
        .unwrap_err();
        assert!(error.contains("partially executes"));
    }
}
