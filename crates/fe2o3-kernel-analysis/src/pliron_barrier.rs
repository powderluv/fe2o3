//! Barrier-convergence verification over bounded PLIRON invocation traces.

use std::{collections::HashMap, fmt};

use dialect_gpu::{
    AddressSpaceAttr, BarrierOp, ExecutionDomainAttr, HierarchyAttr, MemoryOrderAttr,
    MemoryScopeAttr,
};
use dialect_kernel::{
    AnalysisSplitOp, BranchArgsOp, BranchOp, IndexEqualBranchArgsOp, IndexEqualBranchOp,
    IndexLessThanBranchArgsOp, IndexLessThanBranchOp, PipelineEventKindAttr, PipelineEventOp,
    ReturnOp, TensorLayoutOp, TrapOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    operation::Operation,
};

use crate::pliron_analysis_manager::PlironAnalysisManagerV1;
use crate::pliron_invocation_trace::{
    PlironInvocationTraceV1, PlironTraceEventV1, PlironTraceFailureV1, PlironTraceLocationV1,
};
use crate::pliron_pipeline_protocol::run_pliron_pipeline_protocol_check_with_analyses_v1;
use crate::pliron_ranked_bounds::run_pliron_ranked_bounds_check_with_analyses_v1;
use crate::pliron_simt_protocol::{PlironProtocolEventV1, PlironSimtProtocolIssueV1};
use crate::{KernelCheckPassKindV1, KernelCheckStatusV1};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironBarrierFindingV1 {
    BoundsPrerequisiteRejected,
    AnalysisIncomplete {
        detail: String,
    },
    DivergentBarrierTrace {
        first_invocation: Vec<u64>,
        first_trace: Vec<(usize, usize)>,
        second_invocation: Vec<u64>,
        second_trace: Vec<(usize, usize)>,
    },
    DivergentBarrierPaths {
        first_trace: Vec<(usize, usize)>,
        second_trace: Vec<(usize, usize)>,
    },
    SimtProtocolViolation {
        issue: Box<PlironSimtProtocolIssueV1>,
    },
}

impl PlironBarrierFindingV1 {
    pub fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::DivergentBarrierTrace { .. } | Self::DivergentBarrierPaths { .. } => {
                KernelCheckStatusV1::Rejected
            }
            Self::BoundsPrerequisiteRejected | Self::AnalysisIncomplete { .. } => {
                KernelCheckStatusV1::Incomplete
            }
            Self::SimtProtocolViolation { issue } => match issue.as_ref() {
                PlironSimtProtocolIssueV1::ResourceLimitExceeded => KernelCheckStatusV1::Incomplete,
                PlironSimtProtocolIssueV1::PhaseMismatch { .. }
                | PlironSimtProtocolIssueV1::PartialTensorParticipation { .. }
                | PlironSimtProtocolIssueV1::ClaimedActiveMaskMismatch { .. } => {
                    KernelCheckStatusV1::Rejected
                }
            },
        }
    }
}

impl fmt::Display for PlironBarrierFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundsPrerequisiteRejected => formatter.write_str(
                "error[FE2O3-BARRIER-000]: bounds prerequisite rejected before barrier-convergence analysis",
            ),
            Self::AnalysisIncomplete { detail } => write!(
                formatter,
                "error[FE2O3-BARRIER-002]: cannot prove barrier convergence: {detail}",
            ),
            Self::DivergentBarrierTrace {
                first_invocation,
                first_trace,
                second_invocation,
                second_trace,
            } => write!(
                formatter,
                "error[FE2O3-BARRIER-001]: divergent collective barrier trace; invocation {first_invocation:?} executes {}, while invocation {second_invocation:?} executes {}; failed proof: every participating invocation reaches the same barriers in the same order; help: move the barrier out of invocation-varying control flow",
                describe_trace(first_trace),
                describe_trace(second_trace),
            ),
            Self::DivergentBarrierPaths {
                first_trace,
                second_trace,
            } => write!(
                formatter,
                "error[FE2O3-BARRIER-001]: divergent collective barrier paths execute {} and {}; failed proof: every possible invocation path must reach the same barriers in the same order; help: move the barrier after the branch reconverges",
                describe_trace(first_trace),
                describe_trace(second_trace),
            ),
            Self::SimtProtocolViolation { issue } => format_protocol_issue(formatter, issue),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironBarrierReportV1 {
    findings: Vec<PlironBarrierFindingV1>,
}

impl PlironBarrierReportV1 {
    pub const fn pass(&self) -> KernelCheckPassKindV1 {
        KernelCheckPassKindV1::BarrierConvergence
    }

    pub fn status(&self) -> KernelCheckStatusV1 {
        self.findings
            .iter()
            .fold(KernelCheckStatusV1::Clean, |status, finding| {
                status.join(finding.status())
            })
    }

    pub fn findings(&self) -> &[PlironBarrierFindingV1] {
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
pub struct PlironBarrierCheckErrorV1 {
    report: PlironBarrierReportV1,
}

impl PlironBarrierCheckErrorV1 {
    pub fn report(&self) -> &PlironBarrierReportV1 {
        &self.report
    }
}

impl fmt::Display for PlironBarrierCheckErrorV1 {
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

impl std::error::Error for PlironBarrierCheckErrorV1 {}

pub fn run_pliron_barrier_convergence_check_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironBarrierReportV1 {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    if !run_pliron_ranked_bounds_check_with_analyses_v1(context, function, &mut analyses).is_clean()
    {
        return report(PlironBarrierFindingV1::BoundsPrerequisiteRejected);
    }
    run_pliron_barrier_convergence_check_with_analyses_v1(context, function, &mut analyses)
}

pub(crate) fn run_pliron_barrier_convergence_check_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> PlironBarrierReportV1 {
    analyses.prepare_function_inventory(context, function);
    let inventory = match analyses.function_inventory_handle() {
        Ok(inventory) => inventory,
        Err(_) => {
            return report(PlironBarrierFindingV1::AnalysisIncomplete {
                detail: "the bounded function inventory limit was exceeded".to_owned(),
            });
        }
    };
    let mut has_barrier = false;
    let mut has_tensor = false;
    for site in inventory.operations() {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        has_barrier |= operation.downcast_ref::<BarrierOp>().is_some();
        has_tensor |= operation.downcast_ref::<TensorLayoutOp>().is_some();
    }
    if !has_barrier && !has_tensor {
        return PlironBarrierReportV1 { findings: vec![] };
    }
    analyses.prepare_simt_protocol(context, function);
    if let Ok(protocol) = analyses.simt_protocol()
        && let Some(issue) = protocol.issues().first()
    {
        return report(PlironBarrierFindingV1::SimtProtocolViolation {
            issue: Box::new(issue.clone()),
        });
    }
    // The existing all-path barrier proof can still decide some dynamic or
    // cyclic cases for which exact active-mask tracing is unavailable.
    if !has_barrier {
        // Tensor-layout analysis retains the static convergence proof when an
        // exact active-mask trace is unavailable. This stage only adds a
        // counterexample when the exact SIMT trace succeeds.
        return PlironBarrierReportV1 { findings: vec![] };
    }
    let trace_failure = match analyses.exact_trace() {
        Ok(traces) => {
            if traces.is_empty() {
                return report(PlironBarrierFindingV1::AnalysisIncomplete {
                    detail: "the launch domain is empty".to_owned(),
                });
            }
            if let Some(finding) = divergent_scope_trace(traces, HierarchyAttr::Workgroup)
                .or_else(|| divergent_scope_trace(traces, HierarchyAttr::Subgroup))
            {
                return report(finding);
            }
            return PlironBarrierReportV1 { findings: vec![] };
        }
        Err(failure) => failure,
    };
    if matches!(
        trace_failure,
        PlironTraceFailureV1::MissingExecutionLayout
            | PlironTraceFailureV1::InvalidExecutionLayout
            | PlironTraceFailureV1::UnsupportedGridSynchronization { .. }
            | PlironTraceFailureV1::PartialBarrierParticipants { .. }
    ) {
        return report(PlironBarrierFindingV1::AnalysisIncomplete {
            detail: trace_failure_detail(trace_failure),
        });
    }
    if matches!(trace_failure, PlironTraceFailureV1::DynamicLaunch { .. })
        && !matches!(
            analyses.execution_layout(),
            Ok(Some(layout))
                if layout.execution_domain == ExecutionDomainAttr::FullPhysicalWorkgroups
        )
    {
        return report(PlironBarrierFindingV1::AnalysisIncomplete {
            detail: "dynamic barrier convergence requires authenticated full physical workgroups"
                .to_owned(),
        });
    }
    match summarize_all_barrier_paths(context, &inventory) {
        BarrierPathSummaryV1::Unique => PlironBarrierReportV1 { findings: vec![] },
        BarrierPathSummaryV1::Divergent {
            first_trace,
            second_trace,
        } => report(PlironBarrierFindingV1::DivergentBarrierPaths {
            first_trace,
            second_trace,
        }),
        BarrierPathSummaryV1::Incomplete(path_detail) => {
            let epoch_detail = match pipeline_barriers_have_uniform_epoch_proof(
                context, function, &inventory, analyses,
            ) {
                Ok(()) => return PlironBarrierReportV1 { findings: vec![] },
                Err(detail) => detail,
            };
            report(PlironBarrierFindingV1::AnalysisIncomplete {
                detail: format!(
                    "{}; all-path convergence proof also failed: {path_detail}; uniform epoch proof also failed: {epoch_detail}",
                    trace_failure_detail(trace_failure),
                ),
            })
        }
    }
}

fn pipeline_barriers_have_uniform_epoch_proof(
    context: &Context,
    function: &FuncOp,
    inventory: &crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1,
    analyses: &mut PlironAnalysisManagerV1,
) -> Result<(), String> {
    let protocol = run_pliron_pipeline_protocol_check_with_analyses_v1(context, function, analyses);
    if !protocol.is_clean() {
        return Err(protocol
            .findings()
            .first()
            .map(ToString::to_string)
            .unwrap_or_else(|| "pipeline protocol rejected without a finding".to_owned()));
    }
    if protocol.certificates().is_empty() {
        return Err("the function has no staged-pipeline certificate".to_owned());
    }
    let mut certificates = HashMap::new();
    for certificate in protocol.certificates() {
        let Some(site) = inventory.operations().iter().find(|site| {
            site.block() == certificate.pipeline_block()
                && site.operation() == certificate.pipeline_operation()
        }) else {
            return Err("a pipeline certificate has no exact creation site".to_owned());
        };
        if certificate.dynamic_loop().is_none() || !certificate.access_refinement_proven() {
            return Err(
                "a pipeline certificate lacks a uniform dynamic-loop/access refinement proof"
                    .to_owned(),
            );
        }
        if certificates.insert(site.pointer(), certificate).is_some() {
            return Err("two pipeline certificates name the same creation".to_owned());
        }
    }
    let mut barriers = 0_usize;
    for site in inventory.operations() {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(barrier) = operation.downcast_ref::<BarrierOp>() else {
            continue;
        };
        barriers = barriers.saturating_add(1);
        if barrier.execution_scope(context) != Some(HierarchyAttr::Workgroup)
            || barrier.memory_scope(context) != Some(MemoryScopeAttr::Workgroup)
            || barrier.address_space(context) != Some(AddressSpaceAttr::Workgroup)
            || barrier.order(context) != Some(MemoryOrderAttr::AcquireRelease)
        {
            return Err("a barrier paired with a pipeline wait changed its exact workgroup acquire-release contract".to_owned());
        }
        let block_operations = inventory.block_operations(site.block());
        if block_operations
            .iter()
            .filter(|candidate| {
                Operation::get_op_dyn(candidate.pointer(), context)
                    .downcast_ref::<BarrierOp>()
                    .is_some()
            })
            .count()
            != 1
        {
            return Err(format!(
                "pipeline-wait block {} contains more than one barrier",
                site.block()
            ));
        }
        let matched_waits = block_operations
            .iter()
            .filter(|candidate| {
                let operation = Operation::get_op_dyn(candidate.pointer(), context);
                let Some(event) = operation.downcast_ref::<PipelineEventOp>() else {
                    return false;
                };
                if event.kind(context) != Some(PipelineEventKindAttr::Wait) {
                    return false;
                }
                let Some(owner) = event.pipeline(context).defining_op() else {
                    return false;
                };
                let Some(certificate) = certificates.get(&owner) else {
                    return false;
                };
                let Some(summary) = certificate.dynamic_loop() else {
                    return false;
                };
                summary.body().contains(&site.block())
                    || summary.prologue_blocks().contains(&site.block())
                    || summary.drain_blocks().contains(&site.block())
            })
            .count();
        if matched_waits != 1 {
            return Err(format!(
                "barrier in block {} has {matched_waits} certified pipeline waits",
                site.block()
            ));
        }
    }
    if barriers == 0 {
        Err("the function has no barrier to justify".to_owned())
    } else {
        Ok(())
    }
}

fn describe_protocol_sequence(sequence: &[PlironProtocolEventV1]) -> String {
    if sequence.is_empty() {
        return "no collective events".to_owned();
    }
    sequence
        .iter()
        .map(|event| {
            format!(
                "{:?}@block {} op {}",
                event.kind(),
                event.location().block(),
                event.location().operation(),
            )
        })
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn format_protocol_issue(
    formatter: &mut fmt::Formatter<'_>,
    issue: &PlironSimtProtocolIssueV1,
) -> fmt::Result {
    match issue {
        PlironSimtProtocolIssueV1::PhaseMismatch {
            grid,
            workgroup,
            subgroup,
            first_invocation,
            first,
            second_invocation,
            second,
        } => write!(
            formatter,
            "error[FE2O3-PROTOCOL-001]: collective phase mismatch in grid {grid} workgroup {workgroup} subgroup {subgroup}; invocation {first_invocation:?} executes {}, while invocation {second_invocation:?} executes {}; failed proof: all active lanes must execute the same tensor/barrier protocol in the same order; help: reconverge control flow before the collective and keep collective phases in one uniform sequence",
            describe_protocol_sequence(first),
            describe_protocol_sequence(second),
        ),
        PlironSimtProtocolIssueV1::PartialTensorParticipation {
            grid,
            workgroup,
            subgroup,
            location,
            expected_lanes,
            actual_lanes,
        } => write!(
            formatter,
            "error[FE2O3-PROTOCOL-002]: tensor collective at block {} op {} in grid {grid} workgroup {workgroup} subgroup {subgroup} requires {expected_lanes} active lanes, but the actual CFG paths reach it with lanes {actual_lanes:?}; failed proof: the physical subgroup participates as one active mask; help: move the tensor instruction after subgroup reconvergence and predicate its inputs instead of the collective",
            location.block(),
            location.operation(),
        ),
        PlironSimtProtocolIssueV1::ClaimedActiveMaskMismatch {
            location,
            claimed_active_lanes,
            actual_active_lanes,
        } => write!(
            formatter,
            "error[FE2O3-PROTOCOL-003]: tensor collective at block {} op {} claims {claimed_active_lanes} active lanes, but CFG-derived execution has {actual_active_lanes}; failed proof: retained participation metadata matches the executed active mask; help: derive the tensor site after reconvergence and regenerate its compiler-owned participation metadata",
            location.block(),
            location.operation(),
        ),
        PlironSimtProtocolIssueV1::ResourceLimitExceeded => formatter.write_str(
            "error[FE2O3-PROTOCOL-004]: SIMT protocol analysis exceeded its bounded issue limit",
        ),
    }
}

enum BarrierPathSummaryV1 {
    Unique,
    Divergent {
        first_trace: Vec<(usize, usize)>,
        second_trace: Vec<(usize, usize)>,
    },
    Incomplete(String),
}

const MAX_FALLBACK_BARRIER_CFG_BLOCKS_V1: usize = 512;
const MAX_FALLBACK_BARRIER_PATH_EVENTS_V1: usize = 256;

fn summarize_all_barrier_paths(
    context: &Context,
    inventory: &crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1,
) -> BarrierPathSummaryV1 {
    let blocks = inventory.blocks();
    let block_indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect::<HashMap<Ptr<BasicBlock>, usize>>();
    if blocks.is_empty() {
        return BarrierPathSummaryV1::Incomplete("the kernel CFG is empty".to_owned());
    }
    if blocks.len() > MAX_FALLBACK_BARRIER_CFG_BLOCKS_V1 {
        return BarrierPathSummaryV1::Incomplete(format!(
            "the fallback barrier CFG has {} blocks, exceeding the bounded limit of {MAX_FALLBACK_BARRIER_CFG_BLOCKS_V1}",
            blocks.len()
        ));
    }
    let mut states = vec![0_u8; blocks.len()];
    let mut summaries = vec![None; blocks.len()];
    match summarize_barrier_paths_from(
        context,
        inventory,
        blocks,
        &block_indices,
        0,
        &mut states,
        &mut summaries,
    ) {
        Ok(_) => BarrierPathSummaryV1::Unique,
        Err(BarrierPathFailureV1::Divergent {
            first_trace,
            second_trace,
        }) => BarrierPathSummaryV1::Divergent {
            first_trace,
            second_trace,
        },
        Err(BarrierPathFailureV1::Incomplete(detail)) => BarrierPathSummaryV1::Incomplete(detail),
    }
}

enum BarrierPathFailureV1 {
    Divergent {
        first_trace: Vec<(usize, usize)>,
        second_trace: Vec<(usize, usize)>,
    },
    Incomplete(String),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct BarrierPathBlockSummaryV1 {
    normal: Option<Vec<(usize, usize)>>,
    trapped_prefix: Option<Vec<(usize, usize)>>,
}

fn prepend_barrier_path_v1(
    local: &[(usize, usize)],
    summary: BarrierPathBlockSummaryV1,
) -> Result<BarrierPathBlockSummaryV1, BarrierPathFailureV1> {
    let prepend = |mut suffix: Vec<(usize, usize)>| {
        let total = local.len().checked_add(suffix.len()).ok_or_else(|| {
            BarrierPathFailureV1::Incomplete("fallback barrier path length overflowed".to_owned())
        })?;
        if total > MAX_FALLBACK_BARRIER_PATH_EVENTS_V1 {
            return Err(BarrierPathFailureV1::Incomplete(format!(
                "a fallback barrier path has more than {MAX_FALLBACK_BARRIER_PATH_EVENTS_V1} events"
            )));
        }
        let mut trace = Vec::with_capacity(total);
        trace.extend_from_slice(local);
        trace.append(&mut suffix);
        Ok(trace)
    };
    Ok(BarrierPathBlockSummaryV1 {
        normal: summary.normal.map(&prepend).transpose()?,
        trapped_prefix: summary.trapped_prefix.map(prepend).transpose()?,
    })
}

fn merge_barrier_path_summary_v1(
    complete: &mut BarrierPathBlockSummaryV1,
    candidate: BarrierPathBlockSummaryV1,
) -> Result<(), BarrierPathFailureV1> {
    if let Some(candidate_normal) = candidate.normal {
        if let Some(first_normal) = &complete.normal
            && first_normal != &candidate_normal
        {
            return Err(BarrierPathFailureV1::Divergent {
                first_trace: first_normal.clone(),
                second_trace: candidate_normal,
            });
        }
        complete.normal = Some(candidate_normal);
    }
    if let Some(candidate_trap) = candidate.trapped_prefix {
        match &complete.trapped_prefix {
            Some(first_trap) if first_trap.starts_with(&candidate_trap) => {}
            Some(first_trap) if candidate_trap.starts_with(first_trap) => {
                complete.trapped_prefix = Some(candidate_trap);
            }
            Some(first_trap) => {
                return Err(BarrierPathFailureV1::Divergent {
                    first_trace: first_trap.clone(),
                    second_trace: candidate_trap,
                });
            }
            None => complete.trapped_prefix = Some(candidate_trap),
        }
    }
    if let (Some(normal), Some(trapped_prefix)) = (&complete.normal, &complete.trapped_prefix)
        && !normal.starts_with(trapped_prefix)
    {
        return Err(BarrierPathFailureV1::Divergent {
            first_trace: normal.clone(),
            second_trace: trapped_prefix.clone(),
        });
    }
    Ok(())
}

fn summarize_barrier_paths_from(
    context: &Context,
    inventory: &crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1,
    blocks: &[Ptr<BasicBlock>],
    block_indices: &HashMap<Ptr<BasicBlock>, usize>,
    block_index: usize,
    states: &mut [u8],
    summaries: &mut [Option<BarrierPathBlockSummaryV1>],
) -> Result<BarrierPathBlockSummaryV1, BarrierPathFailureV1> {
    match states.get(block_index).copied() {
        Some(2) => {
            return Ok(summaries[block_index]
                .as_ref()
                .expect("completed barrier path summary")
                .clone());
        }
        Some(1) => {
            return Err(BarrierPathFailureV1::Incomplete(format!(
                "block {block_index} participates in cyclic control flow"
            )));
        }
        Some(_) => {}
        None => {
            return Err(BarrierPathFailureV1::Incomplete(
                "a CFG successor is outside the kernel".to_owned(),
            ));
        }
    }
    states[block_index] = 1;
    let block = blocks[block_index];
    let terminator = block
        .deref(context)
        .get_terminator(context)
        .ok_or_else(|| {
            BarrierPathFailureV1::Incomplete(format!("block {block_index} has no terminator"))
        })?;
    let mut local = Vec::with_capacity(MAX_FALLBACK_BARRIER_PATH_EVENTS_V1);
    for site in inventory.block_operations(block_index) {
        let operation_index = site.operation();
        let operation = site.pointer();
        if operation == terminator {
            continue;
        }
        if Operation::get_op_dyn(operation, context)
            .downcast_ref::<BarrierOp>()
            .is_some()
        {
            if local.len() == MAX_FALLBACK_BARRIER_PATH_EVENTS_V1 {
                return Err(BarrierPathFailureV1::Incomplete(format!(
                    "block {block_index} has more than {MAX_FALLBACK_BARRIER_PATH_EVENTS_V1} barriers"
                )));
            }
            local.push((block_index, operation_index));
        }
    }
    let terminator = Operation::get_op_dyn(terminator, context);
    let raw = terminator.get_operation().deref(context);
    let is_return = terminator.downcast_ref::<ReturnOp>().is_some();
    let is_trap = terminator.downcast_ref::<TrapOp>().is_some();
    let successors = if is_return || is_trap {
        Vec::new()
    } else if terminator.downcast_ref::<BranchOp>().is_some()
        || terminator.downcast_ref::<BranchArgsOp>().is_some()
        || terminator.downcast_ref::<IndexLessThanBranchOp>().is_some()
        || terminator
            .downcast_ref::<IndexLessThanBranchArgsOp>()
            .is_some()
        || terminator.downcast_ref::<IndexEqualBranchOp>().is_some()
        || terminator
            .downcast_ref::<IndexEqualBranchArgsOp>()
            .is_some()
        || terminator.downcast_ref::<AnalysisSplitOp>().is_some()
    {
        raw.successors()
            .map(|successor| {
                block_indices.get(&successor).copied().ok_or_else(|| {
                    BarrierPathFailureV1::Incomplete(format!(
                        "block {block_index} targets a block outside the kernel"
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err(BarrierPathFailureV1::Incomplete(format!(
            "block {block_index} has an unsupported terminator"
        )));
    };
    let mut complete = BarrierPathBlockSummaryV1::default();
    for successor in successors {
        let suffix = summarize_barrier_paths_from(
            context,
            inventory,
            blocks,
            block_indices,
            successor,
            states,
            summaries,
        )?;
        let candidate = prepend_barrier_path_v1(&local, suffix)?;
        merge_barrier_path_summary_v1(&mut complete, candidate)?;
    }
    if is_return {
        complete.normal = Some(local);
    } else if is_trap {
        complete.trapped_prefix = Some(local);
    }
    states[block_index] = 2;
    summaries[block_index] = Some(complete.clone());
    Ok(complete)
}

pub(crate) fn require_pliron_barrier_convergence_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> Result<PlironBarrierReportV1, PlironBarrierCheckErrorV1> {
    let report = run_pliron_barrier_convergence_check_with_analyses_v1(context, function, analyses);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironBarrierCheckErrorV1 { report })
    }
}

pub fn require_pliron_barrier_convergence_before_lowering_v1(
    context: &Context,
    function: &FuncOp,
) -> Result<PlironBarrierReportV1, PlironBarrierCheckErrorV1> {
    let report = run_pliron_barrier_convergence_check_v1(context, function);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironBarrierCheckErrorV1 { report })
    }
}

fn barrier_trace(
    trace: &PlironInvocationTraceV1,
    scope: HierarchyAttr,
) -> Vec<(PlironTraceLocationV1, AddressSpaceAttr)> {
    trace
        .events
        .iter()
        .filter_map(|event| match event {
            PlironTraceEventV1::Barrier {
                location,
                execution_scope,
                address_space,
                ..
            } if *execution_scope == scope => Some((*location, *address_space)),
            PlironTraceEventV1::Barrier { .. }
            | PlironTraceEventV1::Fence { .. }
            | PlironTraceEventV1::TensorInstruction { .. }
            | PlironTraceEventV1::Trap { .. }
            | PlironTraceEventV1::Memory { .. }
            | PlironTraceEventV1::CollectiveAllocation { .. } => None,
        })
        .collect()
}

fn divergent_scope_trace(
    traces: &[PlironInvocationTraceV1],
    scope: HierarchyAttr,
) -> Option<PlironBarrierFindingV1> {
    let mut first_by_group: HashMap<(u64, u64, Option<u64>), &PlironInvocationTraceV1> =
        HashMap::new();
    for trace in traces {
        let group = (
            trace.grid,
            trace.workgroup,
            (scope == HierarchyAttr::Subgroup).then_some(trace.subgroup),
        );
        let Some(first) = first_by_group.get(&group).copied() else {
            first_by_group.insert(group, trace);
            continue;
        };
        let first_barriers = barrier_trace(first, scope);
        let barriers = barrier_trace(trace, scope);
        if barriers != first_barriers {
            return Some(PlironBarrierFindingV1::DivergentBarrierTrace {
                first_invocation: first.invocation.clone(),
                first_trace: first_barriers
                    .iter()
                    .map(|(location, _)| (location.block, location.operation))
                    .collect(),
                second_invocation: trace.invocation.clone(),
                second_trace: barriers
                    .iter()
                    .map(|(location, _)| (location.block, location.operation))
                    .collect(),
            });
        }
    }
    None
}

fn report(finding: PlironBarrierFindingV1) -> PlironBarrierReportV1 {
    PlironBarrierReportV1 {
        findings: vec![finding],
    }
}

fn describe_trace(trace: &[(usize, usize)]) -> String {
    if trace.is_empty() {
        return "no barrier".to_owned();
    }
    trace
        .iter()
        .map(|(block, operation)| format!("barrier(block {block}, op {operation})"))
        .collect::<Vec<_>>()
        .join(" -> ")
}

pub(crate) fn trace_failure_detail(failure: PlironTraceFailureV1) -> String {
    match failure {
        PlironTraceFailureV1::Sparse(failure) => {
            format!("sparse index analysis failed: {failure:?}")
        }
        PlironTraceFailureV1::DynamicLaunch { dimension } => {
            format!("launch dimension {dimension} is dynamic")
        }
        PlironTraceFailureV1::LaunchTooLarge { invocations } => {
            format!("launch domain has {invocations} invocations")
        }
        PlironTraceFailureV1::UnresolvedBranch { block } => {
            format!("branch in block {block} has an unresolved condition")
        }
        PlironTraceFailureV1::ForeignView { block, operation } => {
            format!("memory view at block {block} op {operation} is unresolved")
        }
        PlironTraceFailureV1::UnsupportedTerminator { block } => {
            format!("block {block} has an unsupported terminator")
        }
        PlironTraceFailureV1::CyclicControlFlow { block } => {
            format!(
                "block {block} participates in cyclic control flow; progress-dependent spin synchronization is unsupported"
            )
        }
        PlironTraceFailureV1::MissingExecutionLayout => {
            "scoped synchronization lacks a retained gpu.execution_layout".to_owned()
        }
        PlironTraceFailureV1::InvalidExecutionLayout => {
            "gpu.execution_layout is malformed, duplicated, or outside the entry block".to_owned()
        }
        PlironTraceFailureV1::UnsupportedGridSynchronization { block, operation } => {
            format!(
                "ordinary grid-wide barriers are unsupported at block {block} op {operation}; use disjoint workgroup ownership or legal device-scope atomics"
            )
        }
        PlironTraceFailureV1::PartialBarrierParticipants {
            scope,
            dimension,
            global_extent,
            workgroup_extent,
        } => format!(
            "{scope:?} barrier has global extent {global_extent} on axis {dimension}, which is not a multiple of workgroup extent {workgroup_extent}; rounded physical lanes and their activity paths are not represented"
        ),
        PlironTraceFailureV1::ResourceLimit => "trace resource limit exceeded".to_owned(),
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    #[test]
    fn every_barrier_finding_has_the_shared_status() {
        let incomplete = [
            PlironBarrierFindingV1::BoundsPrerequisiteRejected,
            PlironBarrierFindingV1::AnalysisIncomplete {
                detail: "unresolved".to_owned(),
            },
        ];
        for finding in incomplete {
            assert_eq!(finding.status(), KernelCheckStatusV1::Incomplete);
        }

        let rejected = [
            PlironBarrierFindingV1::DivergentBarrierTrace {
                first_invocation: vec![0],
                first_trace: vec![(0, 0)],
                second_invocation: vec![1],
                second_trace: vec![],
            },
            PlironBarrierFindingV1::DivergentBarrierPaths {
                first_trace: vec![(0, 0)],
                second_trace: vec![],
            },
        ];
        for finding in rejected {
            assert_eq!(finding.status(), KernelCheckStatusV1::Rejected);
        }
    }

    #[test]
    fn rejected_barrier_finding_dominates_an_incomplete_finding() {
        let report = PlironBarrierReportV1 {
            findings: vec![
                PlironBarrierFindingV1::AnalysisIncomplete {
                    detail: "unresolved".to_owned(),
                },
                PlironBarrierFindingV1::DivergentBarrierPaths {
                    first_trace: vec![(0, 0)],
                    second_trace: vec![],
                },
            ],
        };
        assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
        assert!(!report.is_clean());
        assert_eq!(
            PlironBarrierReportV1 { findings: vec![] }.status(),
            KernelCheckStatusV1::Clean
        );
    }

    #[test]
    fn fallback_barrier_path_event_vectors_are_bounded() {
        let summary = BarrierPathBlockSummaryV1 {
            normal: Some(vec![(0, 0); MAX_FALLBACK_BARRIER_PATH_EVENTS_V1]),
            trapped_prefix: None,
        };
        let error = prepend_barrier_path_v1(&[(1, 0)], summary).unwrap_err();
        assert!(
            matches!(error, BarrierPathFailureV1::Incomplete(detail) if detail.contains("more than 256 events"))
        );
    }
}
