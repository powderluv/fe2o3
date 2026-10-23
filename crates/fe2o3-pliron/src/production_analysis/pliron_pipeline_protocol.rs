//! Epoch-aware verification for target-neutral staged workgroup pipelines.
//!
//! Constant schedules are interpreted exactly. Runtime-bounded canonical
//! loops are proved from a finite epoch-window invariant, so verification
//! never unrolls a dynamic trip count.

use std::{
    collections::{HashMap, HashSet},
    fmt,
};

use dialect_kernel::{
    AccessKindAttr, AllocationEffectOp, AnalysisSplitOp, BranchArgsOp, DeterministicJoinOp,
    IndexBinaryKindAttr, IndexBinaryOp, IndexConstantOp, IndexEqualBranchArgsOp,
    IndexLessThanBranchArgsOp, IndexUnsignedCastOp, PipelineCreateOp, PipelineEventKindAttr,
    PipelineEventOp, RankedAccessOp, RankedViewOp,
};
use dialect_proof::RequireEffectRefinementOp;
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    common_traits::Named,
    context::{Context, Ptr},
    operation::Operation,
    value::Value,
};

use crate::production_analysis::pliron_analysis_manager::PlironAnalysisManagerV1;
use crate::production_analysis::pliron_function_inventory::{
    BoundedPlironFunctionInventoryV1, PlironOperationSiteV1,
};
use crate::production_analysis::pliron_resource_envelope::{
    ProductionAnalysisInputCensusV1, ProductionAnalysisResourceLimitV1,
    ProductionAnalysisResourceLimitsV1, ProductionAnalysisResourcePhaseV1,
    ProductionAnalysisResourceUpperBoundV1,
};
use crate::{KernelCheckPassKindV1, KernelCheckStatusV1};

pub const MAX_PLIRON_PIPELINE_FINDINGS_V1: usize = 4_096;
const MAX_PLIRON_PIPELINE_VALUE_NAME_BYTES_V1: usize = 240;
const MAX_PLIRON_PIPELINE_DIAGNOSTIC_BYTES_V1: usize = 512;
const MAX_EQUIVALENCE_WORK_V1: usize = 256;
const EQUIVALENCE_REJECTING_ATTEMPTS_V1: usize = MAX_EQUIVALENCE_WORK_V1 + 1;

include!("pliron_pipeline_protocol/resources_v1.rs");

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironPipelineProtocolFindingV1 {
    AnalysisIncomplete {
        detail: String,
    },
    OrphanEvent {
        block: usize,
        operation: usize,
    },
    AliasedStorage {
        first_block: usize,
        first_operation: usize,
        second_block: usize,
        second_operation: usize,
    },
    InvalidSchedule {
        pipeline_block: usize,
        pipeline_operation: usize,
        event_block: Option<usize>,
        event_operation: Option<usize>,
        detail: String,
    },
    FindingLimitExceeded,
}

impl PlironPipelineProtocolFindingV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::AnalysisIncomplete { .. } | Self::FindingLimitExceeded => {
                KernelCheckStatusV1::Incomplete
            }
            Self::OrphanEvent { .. }
            | Self::AliasedStorage { .. }
            | Self::InvalidSchedule { .. } => KernelCheckStatusV1::Rejected,
        }
    }
}

impl fmt::Display for PlironPipelineProtocolFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnalysisIncomplete { detail } => write!(
                formatter,
                "error[FE2O3-PIPELINE-003]: cannot prove staged pipeline safety: {detail}"
            ),
            Self::OrphanEvent { block, operation } => write!(
                formatter,
                "error[FE2O3-PIPELINE-001]: pipeline event at block {block} op {operation} does not use a pipeline created by kernel.pipeline_create; help: bind the event to the pipeline that owns the staged workgroup view"
            ),
            Self::AliasedStorage {
                first_block,
                first_operation,
                second_block,
                second_operation,
            } => write!(
                formatter,
                "error[FE2O3-PIPELINE-001]: pipeline at block {second_block} op {second_operation} may alias staged storage already owned by pipeline block {first_block} op {first_operation}; help: use one lifecycle for that view or provide compiler-derived disjoint allocation provenance"
            ),
            Self::InvalidSchedule {
                pipeline_block,
                pipeline_operation,
                event_block,
                event_operation,
                detail,
            } => {
                write!(
                    formatter,
                    "error[FE2O3-PIPELINE-001]: invalid staged pipeline created at block {pipeline_block} op {pipeline_operation}"
                )?;
                if let (Some(block), Some(operation)) = (event_block, event_operation) {
                    write!(formatter, ", detected at block {block} op {operation}")?;
                }
                write!(
                    formatter,
                    ": {detail}; help: stage then commit each future epoch, wait then consume or discard it, release it exactly once, and use slot = epoch % buffer_count"
                )
            }
            Self::FindingLimitExceeded => formatter
                .write_str("error[FE2O3-PIPELINE-003]: staged-pipeline finding limit exceeded"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EpochAwareLoopSummaryV1 {
    prologue: usize,
    prologue_blocks: Vec<usize>,
    header: usize,
    body: Vec<usize>,
    exit: usize,
    drain_blocks: Vec<usize>,
    induction: String,
    bound: String,
    step: u64,
    prefetched_epochs: u32,
    live_epoch_window: u32,
    drained_epochs: u32,
}

impl EpochAwareLoopSummaryV1 {
    pub const fn prologue(&self) -> usize {
        self.prologue
    }
    pub const fn header(&self) -> usize {
        self.header
    }
    pub fn prologue_blocks(&self) -> &[usize] {
        &self.prologue_blocks
    }
    pub fn body(&self) -> &[usize] {
        &self.body
    }
    pub const fn exit(&self) -> usize {
        self.exit
    }
    pub fn drain_blocks(&self) -> &[usize] {
        &self.drain_blocks
    }
    pub fn induction(&self) -> &str {
        &self.induction
    }
    pub fn bound(&self) -> &str {
        &self.bound
    }
    pub const fn step(&self) -> u64 {
        self.step
    }
    pub const fn prefetched_epochs(&self) -> u32 {
        self.prefetched_epochs
    }
    pub const fn live_epoch_window(&self) -> u32 {
        self.live_epoch_window
    }
    pub const fn drained_epochs(&self) -> u32 {
        self.drained_epochs
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironPipelineProtocolCertificateV1 {
    pipeline_block: usize,
    pipeline_operation: usize,
    buffers: u32,
    prefetch_distance: u32,
    dynamic_loop: Option<EpochAwareLoopSummaryV1>,
    concrete_epochs: usize,
    staged_writes: usize,
    consuming_reads: usize,
    access_refinement_proven: bool,
}

impl PlironPipelineProtocolCertificateV1 {
    pub const fn pipeline_block(&self) -> usize {
        self.pipeline_block
    }
    pub const fn pipeline_operation(&self) -> usize {
        self.pipeline_operation
    }
    pub const fn buffers(&self) -> u32 {
        self.buffers
    }
    pub const fn prefetch_distance(&self) -> u32 {
        self.prefetch_distance
    }
    pub const fn dynamic_loop(&self) -> Option<&EpochAwareLoopSummaryV1> {
        self.dynamic_loop.as_ref()
    }
    pub const fn concrete_epochs(&self) -> usize {
        self.concrete_epochs
    }
    pub const fn staged_writes(&self) -> usize {
        self.staged_writes
    }
    pub const fn consuming_reads(&self) -> usize {
        self.consuming_reads
    }
    pub const fn access_refinement_proven(&self) -> bool {
        self.access_refinement_proven
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironPipelineProtocolReportV1 {
    findings: Vec<PlironPipelineProtocolFindingV1>,
    certificates: Vec<PlironPipelineProtocolCertificateV1>,
}

impl PlironPipelineProtocolReportV1 {
    pub const fn pass(&self) -> KernelCheckPassKindV1 {
        KernelCheckPassKindV1::PipelineProtocol
    }
    pub fn status(&self) -> KernelCheckStatusV1 {
        self.findings
            .iter()
            .fold(KernelCheckStatusV1::Clean, |status, finding| {
                status.join(finding.status())
            })
    }
    pub fn is_clean(&self) -> bool {
        self.status() == KernelCheckStatusV1::Clean
    }
    pub fn findings(&self) -> &[PlironPipelineProtocolFindingV1] {
        &self.findings
    }
    pub fn certificates(&self) -> &[PlironPipelineProtocolCertificateV1] {
        &self.certificates
    }
    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironPipelineProtocolCheckErrorV1 {
    report: PlironPipelineProtocolReportV1,
}

impl PlironPipelineProtocolCheckErrorV1 {
    pub fn report(&self) -> &PlironPipelineProtocolReportV1 {
        &self.report
    }
}

impl fmt::Display for PlironPipelineProtocolCheckErrorV1 {
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

impl std::error::Error for PlironPipelineProtocolCheckErrorV1 {}

#[cfg(test)]
pub(crate) fn run_pliron_pipeline_protocol_check_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironPipelineProtocolReportV1 {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    run_pliron_pipeline_protocol_check_with_analyses_v1(context, function, &mut analyses)
}

pub(crate) fn require_pliron_pipeline_protocol_with_observation_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
    observer: ProtocolObserverV1<'_, '_, '_>,
) -> Result<PlironPipelineProtocolReportV1, PlironPipelineProtocolCheckErrorV1> {
    let report =
        run_pliron_pipeline_protocol_with_observation_v1(context, function, analyses, observer);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironPipelineProtocolCheckErrorV1 { report })
    }
}

#[cfg(test)]
pub(crate) fn run_pliron_pipeline_protocol_check_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> PlironPipelineProtocolReportV1 {
    run_pliron_pipeline_protocol_with_observation_v1(context, function, analyses, None)
}

pub(crate) fn run_pliron_pipeline_protocol_with_observation_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
    observer: ProtocolObserverV1<'_, '_, '_>,
) -> PlironPipelineProtocolReportV1 {
    match observer {
        None => run_pliron_pipeline_protocol_observed_inner_v1(context, function, analyses, None),
        Some(observer) => observer.with_projection(&Ok, |nested| {
            run_pliron_pipeline_protocol_observed_inner_v1(
                context,
                function,
                analyses,
                Some(nested),
            )
        }),
    }
}

fn run_pliron_pipeline_protocol_observed_inner_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
    observer: ProtocolObserverV1<'_, '_, '_>,
) -> PlironPipelineProtocolReportV1 {
    let authenticated_census = analyses.input_census();
    analyses.prepare_function_inventory(context, function);
    let inventory = match analyses.function_inventory_handle() {
        Ok(inventory) => inventory,
        Err(failure) => {
            observe_protocol_quota_v1(observer, failure.resource());
            return report(PlironPipelineProtocolFindingV1::AnalysisIncomplete {
                detail: "the bounded function inventory limit was exceeded".to_owned(),
            });
        }
    };
    let census = match authenticated_census
        .or_else(|| pipeline_protocol_inventory_census_v1(context, &inventory))
    {
        Some(census) => census,
        None => {
            observe_protocol_quota_v1(observer, "pipeline protocol census counter overflow");
            return equivalence_resource_failure_report_v1();
        }
    };
    if census.pipeline_creates == 0 {
        let mut findings = Vec::new();
        for site in inventory.operations() {
            let operation = Operation::get_op_dyn(site.pointer(), context);
            if operation.downcast_ref::<PipelineEventOp>().is_some() {
                push_finding(
                    &mut findings,
                    PlironPipelineProtocolFindingV1::OrphanEvent {
                        block: site.block(),
                        operation: site.operation(),
                    },
                );
            }
        }
        return PlironPipelineProtocolReportV1 {
            findings,
            certificates: Vec::new(),
        };
    }
    // The production descriptor prepays this phase, including its nested
    // barrier/workgroup-memory attempts. A census alone is not that admission;
    // those closed callers own the descriptor. Standalone diagnostics have no
    // descriptor and reserve their own cumulative phase before discovery.
    if authenticated_census.is_none() {
        let admission = analyses
            .remaining_resource_limits(ProductionAnalysisResourcePhaseV1::PipelineProtocol)
            .and_then(|limits| {
                let bound = preflight_pipeline_protocol_resource_upper_bound_v1(census, limits)?;
                match observer {
                    None => Ok(bound),
                    Some(observer) => observer.require(
                        limits,
                        ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                        Ok(bound),
                    ),
                }
            })
            .and_then(|bound| {
                analyses.admit_retained_resource_upper_bound(
                    ProductionAnalysisResourcePhaseV1::PipelineProtocol,
                    bound,
                )
            });
        if let Err(error) = admission {
            if let Some(observer) = observer {
                observer.deny(error);
            }
            return report(PlironPipelineProtocolFindingV1::AnalysisIncomplete {
                detail: "pipeline protocol exceeded its cumulative work or storage limit"
                    .to_owned(),
            });
        }
    }
    let unique_pair_limit = match pipeline_equivalence_unique_pair_upper_bound_v1(census) {
        Ok(limit) => limit,
        Err(error) => {
            if let Some(observer) = observer {
                observer.deny(error);
            }
            return equivalence_resource_failure_report_v1();
        }
    };
    let mut equivalence_resources = match pipeline_equivalence_query_upper_bound_v1(census)
        .map_err(|error| {
            if let Some(observer) = observer {
                observer.deny(error);
            }
            EquivalenceVisitLimitV1
        })
        .and_then(|query_limit| {
            EquivalenceResourceMeterV1::new_with_observation_v1(
                query_limit,
                unique_pair_limit,
                observer,
            )
        }) {
        Ok(resources) => resources,
        Err(_) => return equivalence_resource_failure_report_v1(),
    };
    let loop_discovery = discover_epoch_loops(context, &inventory, &mut equivalence_resources);
    if equivalence_resources.exhausted() {
        return equivalence_resource_failure_report_v1();
    }
    let uniform_roots = function
        .get_entry_block(context)
        .deref(context)
        .arguments()
        .collect::<HashSet<_>>();
    let mut creates = Vec::new();
    let mut create_owners = HashSet::new();
    let mut first_storage_owner = None;
    let mut unknown_class_owner = None;
    let mut view_owners = HashMap::<Value, PlironOperationSiteV1>::new();
    let mut origin_owners = HashMap::<u64, PlironOperationSiteV1>::new();
    let mut class_owners = HashMap::<u64, PlironOperationSiteV1>::new();
    let mut findings = Vec::new();
    for site in inventory.operations() {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(create) = operation.downcast_ref::<PipelineCreateOp>() else {
            continue;
        };
        let view = create.view(context);
        let (origin, noalias_class) = pipeline_storage_contract(context, view);
        let first = view_owners
            .get(&view)
            .or_else(|| origin.and_then(|origin| origin_owners.get(&origin)))
            .or_else(|| noalias_class.and_then(|class| class_owners.get(&class)))
            .or(match noalias_class {
                Some(_) => unknown_class_owner.as_ref(),
                None => first_storage_owner.as_ref(),
            })
            .copied();
        if let Some(first) = first {
            push_finding(
                &mut findings,
                PlironPipelineProtocolFindingV1::AliasedStorage {
                    first_block: first.block(),
                    first_operation: first.operation(),
                    second_block: site.block(),
                    second_operation: site.operation(),
                },
            );
        }
        first_storage_owner.get_or_insert(*site);
        view_owners.entry(view).or_insert(*site);
        if let Some(origin) = origin {
            origin_owners.entry(origin).or_insert(*site);
        }
        if let Some(noalias_class) = noalias_class {
            class_owners.entry(noalias_class).or_insert(*site);
        } else {
            unknown_class_owner.get_or_insert(*site);
        }
        creates.push((site.pointer(), *site, create.pipeline_type(context), view));
        create_owners.insert(site.pointer());
    }
    let mut events = HashMap::<Ptr<Operation>, Vec<EventSiteV1>>::new();
    for site in inventory.operations() {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(event) = operation.downcast_ref::<PipelineEventOp>() else {
            continue;
        };
        let Some(owner) = event.pipeline(context).defining_op() else {
            push_finding(
                &mut findings,
                PlironPipelineProtocolFindingV1::OrphanEvent {
                    block: site.block(),
                    operation: site.operation(),
                },
            );
            continue;
        };
        if !create_owners.contains(&owner) {
            push_finding(
                &mut findings,
                PlironPipelineProtocolFindingV1::OrphanEvent {
                    block: site.block(),
                    operation: site.operation(),
                },
            );
            continue;
        }
        events.entry(owner).or_default().push(EventSiteV1 {
            site: *site,
            kind: event.kind(context),
            epoch: event.epoch(context),
            slot: event.slot(context),
        });
    }
    let mut accesses = HashMap::<Value, Vec<AccessSiteV1>>::new();
    for site in inventory.operations() {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(access) = operation.downcast_ref::<RankedAccessOp>() else {
            continue;
        };
        let Some(kind) = access.kind(context) else {
            continue;
        };
        let indices = access.indices(context);
        let Some(slot) = indices.first().copied() else {
            continue;
        };
        accesses
            .entry(access.view(context))
            .or_default()
            .push(AccessSiteV1 {
                site: *site,
                kind,
                slot,
                indices,
            });
    }
    let mut certificates = Vec::new();
    let mut control = PipelineControlContextV1 {
        inventory: &inventory,
        discovery: &loop_discovery,
        concrete: None,
    };
    let mut concrete_indices = ConcreteIndexContextV1 {
        function,
        analyses,
        census,
        unavailable: false,
    };
    let uniformity_visit_limit = inventory.operations().len();
    for (pointer, site, pipeline_type, view) in creates {
        let Some(pipeline_type) = pipeline_type else {
            push_finding(
                &mut findings,
                invalid(site, None, "pipeline configuration type is malformed"),
            );
            continue;
        };
        let pipeline_type = pipeline_type.deref(context);
        let schedule = events.remove(&pointer).unwrap_or_default();
        if schedule.is_empty() {
            push_finding(
                &mut findings,
                invalid(site, None, "pipeline has no lifecycle events"),
            );
            continue;
        }
        match verify_one_pipeline(
            context,
            &mut control,
            &mut concrete_indices,
            site,
            pipeline_type.buffers(),
            pipeline_type.prefetch_distance(),
            &schedule,
            accesses.get(&view).map_or(&[], Vec::as_slice),
            &uniform_roots,
            uniformity_visit_limit,
            &mut equivalence_resources,
        ) {
            Ok(certificate) => certificates.push(certificate),
            Err(finding) if concrete_indices.unavailable => {
                // Sparse preflight can scan the merge census before rejecting
                // its full bound. Do not continue any protocol work after this
                // uncommitted failed initialization, or retry another create.
                return PlironPipelineProtocolReportV1 {
                    findings: vec![finding],
                    certificates: vec![],
                };
            }
            Err(finding) => push_finding(&mut findings, finding),
        }
        if equivalence_resources.exhausted() {
            return equivalence_resource_failure_report_v1();
        }
    }
    PlironPipelineProtocolReportV1 {
        findings,
        certificates,
    }
}

fn pipeline_protocol_inventory_census_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
) -> Option<ProductionAnalysisInputCensusV1> {
    let mut operands = 0_usize;
    let mut successors = 0_usize;
    let mut max_operation_arity = 0_usize;
    let mut results = 0_usize;
    let mut pipeline_creates = 0_usize;
    let mut pipeline_events = 0_usize;
    let mut ranked_accesses = 0_usize;
    let mut allocation_effects = 0_usize;
    let mut effect_refinement_contracts = 0_usize;
    let mut index_lt_branch_candidates = 0_usize;
    for site in inventory.operations() {
        let raw = site.pointer().deref(context);
        let operand_arity = raw.get_num_operands();
        let arity = operand_arity.checked_add(raw.get_num_results())?;
        operands = operands.checked_add(operand_arity)?;
        successors = successors.checked_add(raw.get_num_successors())?;
        results = results.checked_add(raw.get_num_results())?;
        max_operation_arity = max_operation_arity.max(arity);
        let operation = Operation::get_op_dyn(site.pointer(), context);
        pipeline_creates += usize::from(operation.downcast_ref::<PipelineCreateOp>().is_some());
        pipeline_events += usize::from(operation.downcast_ref::<PipelineEventOp>().is_some());
        ranked_accesses += usize::from(operation.downcast_ref::<RankedAccessOp>().is_some());
        allocation_effects += usize::from(operation.downcast_ref::<AllocationEffectOp>().is_some());
        effect_refinement_contracts += usize::from(
            operation
                .downcast_ref::<RequireEffectRefinementOp>()
                .is_some(),
        );
        index_lt_branch_candidates += usize::from(
            operation
                .downcast_ref::<IndexLessThanBranchArgsOp>()
                .is_some(),
        );
    }
    let block_arguments = inventory
        .blocks()
        .iter()
        .try_fold(0_usize, |total, block| {
            total.checked_add(block.deref(context).get_num_arguments())
        })?;
    Some(ProductionAnalysisInputCensusV1 {
        blocks: inventory.blocks().len(),
        operations: inventory.operations().len(),
        operands,
        results,
        successors,
        block_arguments,
        max_operation_arity,
        pipeline_creates,
        pipeline_events,
        ranked_accesses,
        allocation_effects,
        effect_refinement_contracts,
        index_lt_branch_candidates,
        ..ProductionAnalysisInputCensusV1::default()
    })
}

fn equivalence_resource_failure_report_v1() -> PlironPipelineProtocolReportV1 {
    report(PlironPipelineProtocolFindingV1::AnalysisIncomplete {
        detail: "pipeline expression equivalence exceeded its authenticated resource limit"
            .to_owned(),
    })
}

fn pipeline_storage_contract(context: &Context, view: Value) -> (Option<u64>, Option<u64>) {
    let contract = view.defining_op().and_then(|definition| {
        Operation::get_op_dyn(definition, context)
            .downcast_ref::<RankedViewOp>()
            .map(|view| (view.allocation_origin(context), view.noalias_class(context)))
    });
    contract.map_or((None, None), |(origin, noalias_class)| {
        (
            origin.filter(|origin| *origin != 0),
            noalias_class.filter(|class| *class != 0),
        )
    })
}

#[derive(Clone, Copy)]
struct EventSiteV1 {
    site: PlironOperationSiteV1,
    kind: Option<PipelineEventKindAttr>,
    epoch: Value,
    slot: Value,
}

#[derive(Clone)]
struct AccessSiteV1 {
    site: PlironOperationSiteV1,
    kind: AccessKindAttr,
    slot: Value,
    indices: Vec<Value>,
}

#[derive(Clone)]
struct CanonicalEpochLoopV1 {
    prologue: Vec<usize>,
    header: usize,
    body: Vec<usize>,
    body_members: HashSet<usize>,
    drain: Vec<usize>,
    inductions: HashMap<usize, Value>,
    header_induction: Value,
    bound: Value,
    repeated_blocks: HashSet<usize>,
    optional_blocks: HashSet<usize>,
}

struct EpochLoopDiscoveryV1 {
    loops: Vec<CanonicalEpochLoopV1>,
    dominators: Vec<HashSet<usize>>,
    cfg_successors: Vec<Vec<usize>>,
}

include!("pliron_pipeline_protocol/concrete_cfg_v1.rs");
include!("pliron_pipeline_protocol/concrete_index_facts_v1.rs");

#[allow(clippy::too_many_arguments)]
fn verify_one_pipeline(
    context: &Context,
    control: &mut PipelineControlContextV1<'_>,
    concrete_indices: &mut ConcreteIndexContextV1<'_>,
    pipeline: PlironOperationSiteV1,
    buffers: u32,
    distance: u32,
    schedule: &[EventSiteV1],
    accesses: &[AccessSiteV1],
    uniform_roots: &HashSet<Value>,
    uniformity_visit_limit: usize,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<PlironPipelineProtocolCertificateV1, PlironPipelineProtocolFindingV1> {
    let discovery = control.discovery;
    let mut containing = discovery
        .loops
        .iter()
        .filter(|summary| {
            schedule
                .iter()
                .any(|event| summary.body_members.contains(&event.site.block()))
        })
        .collect::<Vec<_>>();
    if containing.is_empty() {
        let same_block = schedule
            .iter()
            .map(|event| event.site)
            .chain(accesses.iter().map(|access| access.site))
            .all(|site| site.block() == pipeline.block());
        let (epochs, staged_writes, consuming_reads, access_refinement_proven) = if same_block {
            let facts = concrete_indices.facts_for(context, schedule, accesses)?;
            verify_concrete_schedule(
                context,
                pipeline,
                buffers,
                schedule,
                accesses,
                facts,
                equivalence_resources,
            )?
        } else {
            let actions = verify_cross_block_concrete_trace_v1(
                context, control, pipeline, schedule, accesses,
            )?;
            let facts = concrete_indices.facts_for(context, schedule, accesses)?;
            verify_ordered_concrete_schedule(
                context,
                pipeline,
                buffers,
                &actions,
                facts,
                equivalence_resources,
            )?
        };
        return Ok(PlironPipelineProtocolCertificateV1 {
            pipeline_block: pipeline.block(),
            pipeline_operation: pipeline.operation(),
            buffers,
            prefetch_distance: distance,
            dynamic_loop: None,
            concrete_epochs: epochs,
            staged_writes,
            consuming_reads,
            access_refinement_proven,
        });
    }
    containing.sort_by_key(|summary| summary.body.len());
    let Some(summary) = containing.first().copied() else {
        unreachable!("the empty case returned above")
    };
    if containing[1..].iter().any(|outer| {
        !summary
            .body_members
            .iter()
            .all(|block| outer.body_members.contains(block))
    }) {
        return Err(invalid(
            pipeline,
            None,
            "one pipeline lifecycle crosses non-nested dynamic loops",
        ));
    }
    if summary.body_members.contains(&pipeline.block())
        || summary.header == pipeline.block()
        || !pipeline_creation_dominates_schedule(
            &discovery.dominators,
            pipeline,
            schedule,
            accesses,
        )
    {
        return Err(invalid(
            pipeline,
            None,
            "the dynamic pipeline creation does not execute once and dominate its complete lifecycle",
        ));
    }
    let mut prologue_blocks = Vec::with_capacity(summary.prologue.len() + 1);
    prologue_blocks.push(pipeline.block());
    prologue_blocks.extend(
        summary
            .prologue
            .iter()
            .copied()
            .filter(|block| *block != pipeline.block()),
    );
    let uniform_bound = is_uniform_value_with_observation_v1(
        context,
        summary.bound,
        uniform_roots,
        uniformity_visit_limit,
        equivalence_resources.observer,
    );
    if uniform_bound != Ok(true) {
        let detail = if uniform_bound.is_err() {
            "the runtime loop bound uniformity proof exceeded its authenticated operation visit limit"
        } else {
            "the runtime loop bound is not proved workgroup-uniform"
        };
        return Err(invalid(pipeline, None, detail));
    }
    if !summary.repeated_blocks.is_empty() {
        let Some(latch) = summary.body.last().copied() else {
            return Err(invalid(pipeline, None, "nested epoch body has no latch"));
        };
        // Nested repetition may not turn a conditional event into authority
        // for a sibling path. All steady-state events execute on every path.
        if accesses.iter().any(|access| {
            access.kind == AccessKindAttr::Write
                && summary.body_members.contains(&access.site.block())
                && !discovery.dominators[latch].contains(&access.site.block())
        }) {
            return Err(invalid(
                pipeline,
                None,
                "nested epoch has a bypassable staging write",
            ));
        }
        if schedule.iter().any(|event| {
            summary.body_members.contains(&event.site.block())
                && !discovery.dominators[latch].contains(&event.site.block())
        }) {
            return Err(invalid(
                pipeline,
                None,
                "nested epoch has a bypassable pipeline event",
            ));
        }
    }
    let domain_reads = full_workgroup_initialized_reads_v1(
        context,
        control,
        concrete_indices,
        pipeline,
        summary,
        schedule,
        accesses,
        uniform_roots,
        uniformity_visit_limit,
        equivalence_resources,
    )?;
    let (staged_writes, consuming_reads, access_refinement_proven) = verify_dynamic_schedule(
        context,
        pipeline,
        buffers,
        distance,
        DynamicScheduleInputV1 {
            schedule,
            accesses,
            summary,
            domain_reads: &domain_reads,
        },
        equivalence_resources,
    )?;
    Ok(PlironPipelineProtocolCertificateV1 {
        pipeline_block: pipeline.block(),
        pipeline_operation: pipeline.operation(),
        buffers,
        prefetch_distance: distance,
        dynamic_loop: Some(EpochAwareLoopSummaryV1 {
            prologue: pipeline.block(),
            prologue_blocks,
            header: summary.header,
            body: summary.body.clone(),
            exit: summary.drain[0],
            drain_blocks: summary.drain.clone(),
            induction: bounded_pipeline_value_name_v1(summary.header_induction, context),
            bound: bounded_pipeline_value_name_v1(summary.bound, context),
            step: 1,
            prefetched_epochs: distance,
            live_epoch_window: distance + 1,
            drained_epochs: distance,
        }),
        concrete_epochs: 0,
        staged_writes,
        consuming_reads,
        access_refinement_proven,
    })
}

fn bounded_pipeline_value_name_v1(value: Value, context: &Context) -> String {
    let identifier = value.unique_name(context);
    let source = identifier.as_ref();
    if source.len() <= MAX_PLIRON_PIPELINE_VALUE_NAME_BYTES_V1 {
        return source.to_owned();
    }
    let mut end = MAX_PLIRON_PIPELINE_VALUE_NAME_BYTES_V1;
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    let mut bounded = String::with_capacity(end + 3);
    bounded.push_str(&source[..end]);
    bounded.push_str("...");
    bounded
}

fn pipeline_creation_dominates_schedule(
    dominators: &[HashSet<usize>],
    pipeline: PlironOperationSiteV1,
    schedule: &[EventSiteV1],
    accesses: &[AccessSiteV1],
) -> bool {
    schedule
        .iter()
        .map(|event| event.site)
        .chain(accesses.iter().map(|access| access.site))
        .all(|site| {
            if site.block() == pipeline.block() {
                return site.operation() > pipeline.operation();
            }
            dominators
                .get(site.block())
                .is_some_and(|target| target.contains(&pipeline.block()))
        })
}

enum UniformityFrameV1 {
    Enter(Value),
    Evaluate {
        value: Value,
        inputs: Vec<Value>,
        next: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UniformityVisitLimitV1;

fn is_uniform_value_with_observation_v1(
    context: &Context,
    value: Value,
    uniform_roots: &HashSet<Value>,
    visit_limit: usize,
    observer: ProtocolObserverV1<'_, '_, '_>,
) -> Result<bool, UniformityVisitLimitV1> {
    let result = match observer {
        None => is_uniform_value(context, value, uniform_roots, visit_limit),
        Some(observer) => observer.with_projection(&Ok, |_| {
            is_uniform_value(context, value, uniform_roots, visit_limit)
        }),
    };
    if result.is_err() {
        observe_protocol_quota_v1(observer, "uniformity operation visit quota");
    }
    result
}

fn is_uniform_value(
    context: &Context,
    value: Value,
    uniform_roots: &HashSet<Value>,
    visit_limit: usize,
) -> Result<bool, UniformityVisitLimitV1> {
    let mut visits = 0_usize;
    let mut memo = HashMap::<Value, bool>::new();
    let mut active = HashSet::<Value>::new();
    let mut pending = vec![UniformityFrameV1::Enter(value)];
    while let Some(frame) = pending.pop() {
        match frame {
            UniformityFrameV1::Enter(value) => {
                if uniform_roots.contains(&value) {
                    memo.insert(value, true);
                    continue;
                }
                if memo.contains_key(&value) {
                    continue;
                }
                visits = visits.checked_add(1).ok_or(UniformityVisitLimitV1)?;
                if visits > visit_limit {
                    return Err(UniformityVisitLimitV1);
                }
                if !active.insert(value) {
                    memo.insert(value, false);
                    continue;
                }
                let Some(definition) = value.defining_op() else {
                    active.remove(&value);
                    memo.insert(value, false);
                    continue;
                };
                let operation = Operation::get_op_dyn(definition, context);
                let inputs = if operation.downcast_ref::<IndexConstantOp>().is_some() {
                    active.remove(&value);
                    memo.insert(value, true);
                    continue;
                } else if let Some(binary) = operation.downcast_ref::<IndexBinaryOp>() {
                    vec![binary.lhs(context), binary.rhs(context)]
                } else if let Some(cast) = operation.downcast_ref::<IndexUnsignedCastOp>() {
                    vec![cast.source(context)]
                } else if let Some(join) = operation.downcast_ref::<DeterministicJoinOp>() {
                    let dependencies = join.dependencies(context);
                    if dependencies.is_empty() {
                        active.remove(&value);
                        memo.insert(value, false);
                        continue;
                    }
                    dependencies
                } else {
                    active.remove(&value);
                    memo.insert(value, false);
                    continue;
                };
                pending.push(UniformityFrameV1::Evaluate {
                    value,
                    inputs,
                    next: 0,
                });
            }
            UniformityFrameV1::Evaluate {
                value,
                inputs,
                next,
            } if next < inputs.len() => {
                let input = inputs[next];
                pending.push(UniformityFrameV1::Evaluate {
                    value,
                    inputs,
                    next: next + 1,
                });
                pending.push(UniformityFrameV1::Enter(input));
            }
            UniformityFrameV1::Evaluate { value, inputs, .. } => {
                active.remove(&value);
                let uniform = memo.get(&value) != Some(&false)
                    && inputs.iter().all(|input| memo.get(input) == Some(&true));
                memo.insert(value, uniform);
            }
        }
    }
    Ok(memo.get(&value) == Some(&true))
}

include!("pliron_pipeline_protocol/dynamic_schedule_v1.rs");

fn verify_concrete_schedule(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    buffers: u32,
    schedule: &[EventSiteV1],
    accesses: &[AccessSiteV1],
    facts: Option<&SparseIndexAnalysisV1>,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(usize, usize, usize, bool), PlironPipelineProtocolFindingV1> {
    let first_block = schedule[0].site.block();
    if first_block != pipeline.block() {
        return Err(invalid(
            pipeline,
            None,
            "a straight-line pipeline lifecycle is not in its creation block, so per-entry lifecycle ownership is unproved",
        ));
    }
    if schedule
        .iter()
        .any(|event| event.site.block() != first_block)
    {
        return Err(invalid(
            pipeline,
            None,
            "non-loop pipeline events span multiple blocks, so execution order is not unique",
        ));
    }
    if let Some(access) = accesses
        .iter()
        .find(|access| access.site.block() != first_block)
    {
        return Err(invalid(
            pipeline,
            Some(access.site),
            "a straight-line pipeline access is outside the entry block",
        ));
    }
    let mut actions = schedule
        .iter()
        .map(ConcreteActionV1::Event)
        .chain(accesses.iter().map(ConcreteActionV1::Access))
        .collect::<Vec<_>>();
    actions.sort_by_key(|action| match action {
        ConcreteActionV1::Event(event) => (event.site.operation(), 1_u8),
        ConcreteActionV1::Access(access) => (access.site.operation(), 0_u8),
    });
    verify_ordered_concrete_schedule(
        context,
        pipeline,
        buffers,
        &actions,
        facts,
        equivalence_resources,
    )
}

fn concrete_coordinate_initialized_v1(
    context: &Context,
    writes: &[&[Value]],
    coordinate: &[Value],
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> bool {
    if equivalence_resources.exhausted() {
        return false;
    }
    for written in writes {
        if equivalence_resources.exhausted() {
            return false;
        }
        if written.len() == coordinate.len()
            && written
                .iter()
                .copied()
                .zip(coordinate.iter().copied())
                .all(|(left, right)| {
                    index_values_equivalent(context, left, right, equivalence_resources)
                })
        {
            return true;
        }
    }
    false
}

fn verify_ordered_concrete_schedule(
    context: &Context,
    pipeline: PlironOperationSiteV1,
    buffers: u32,
    actions: &[ConcreteActionV1<'_>],
    facts: Option<&SparseIndexAnalysisV1>,
    equivalence_resources: &mut EquivalenceResourceMeterV1,
) -> Result<(usize, usize, usize, bool), PlironPipelineProtocolFindingV1> {
    let mut slots = vec![SlotStateV1::Free; buffers as usize];
    let mut epochs = HashSet::new();
    let mut staged_writes = 0;
    let mut consuming_reads = 0;
    // Source-ordered borrowed coordinates make resource-sensitive matching
    // deterministic. Repeated writes retain at most one row per actual access;
    // the existing A-row/A^2-query envelope also covered the former owned sets.
    let mut initialized = HashMap::<u64, Vec<&[Value]>>::new();
    for action in actions.iter().copied() {
        if let ConcreteActionV1::Access(access) = action {
            let Some(slot) = concrete_index_constant_v1(context, facts, access.slot) else {
                return Err(invalid(
                    pipeline,
                    Some(access.site),
                    "a straight-line pipeline access has a symbolic ring slot",
                ));
            };
            let Some(state) = usize::try_from(slot)
                .ok()
                .and_then(|slot| slots.get(slot))
                .copied()
            else {
                return Err(invalid(
                    pipeline,
                    Some(access.site),
                    &format!("pipeline access slot {slot} is outside {buffers} buffers"),
                ));
            };
            match access.kind {
                AccessKindAttr::Write if matches!(state, SlotStateV1::Staged(_)) => {
                    let SlotStateV1::Staged(epoch) = state else {
                        unreachable!("the guarded state is staged")
                    };
                    initialized
                        .entry(epoch)
                        .or_default()
                        .push(&access.indices[1..]);
                    staged_writes += 1;
                }
                AccessKindAttr::Read if matches!(state, SlotStateV1::Consuming(_)) => {
                    let SlotStateV1::Consuming(epoch) = state else {
                        unreachable!("the guarded state is consuming")
                    };
                    if !initialized.get(&epoch).is_some_and(|writes| {
                        concrete_coordinate_initialized_v1(
                            context,
                            writes,
                            &access.indices[1..],
                            equivalence_resources,
                        )
                    }) {
                        return Err(invalid(
                            pipeline,
                            Some(access.site),
                            "pipeline read coordinate was not initialized in the same epoch",
                        ));
                    }
                    consuming_reads += 1;
                }
                _ => {
                    return Err(invalid(
                        pipeline,
                        Some(access.site),
                        &format!(
                            "{:?} access to slot {slot} occurs while that slot is {state:?}",
                            access.kind
                        ),
                    ));
                }
            }
            continue;
        }
        let ConcreteActionV1::Event(event) = action else {
            unreachable!("access actions continue above")
        };
        let Some(kind) = event.kind else {
            return Err(invalid(
                pipeline,
                Some(event.site),
                "event kind is malformed",
            ));
        };
        let Some(epoch) = concrete_index_constant_v1(context, facts, event.epoch) else {
            return Err(invalid(
                pipeline,
                Some(event.site),
                "symbolic event is not enclosed by a supported runtime-bounded loop",
            ));
        };
        let Some(slot) = concrete_index_constant_v1(context, facts, event.slot) else {
            return Err(invalid(
                pipeline,
                Some(event.site),
                "constant epoch has a non-constant ring slot",
            ));
        };
        if slot != epoch % u64::from(buffers) {
            return Err(invalid(
                pipeline,
                Some(event.site),
                &format!("slot {slot} is not epoch {epoch} % {buffers}"),
            ));
        }
        let state = &mut slots[slot as usize];
        let next = match (kind, *state) {
            (PipelineEventKindAttr::Stage, SlotStateV1::Free) => {
                epochs.insert(epoch);
                SlotStateV1::Staged(epoch)
            }
            (PipelineEventKindAttr::Commit, SlotStateV1::Staged(current)) if current == epoch => {
                SlotStateV1::Committed(epoch)
            }
            (PipelineEventKindAttr::Wait, SlotStateV1::Committed(current)) if current == epoch => {
                SlotStateV1::Ready(epoch)
            }
            (PipelineEventKindAttr::Consume, SlotStateV1::Ready(current)) if current == epoch => {
                SlotStateV1::Consuming(epoch)
            }
            (PipelineEventKindAttr::Discard, SlotStateV1::Ready(current)) if current == epoch => {
                SlotStateV1::Discarding(epoch)
            }
            (PipelineEventKindAttr::Release, SlotStateV1::Consuming(current))
            | (PipelineEventKindAttr::Release, SlotStateV1::Discarding(current))
                if current == epoch =>
            {
                SlotStateV1::Free
            }
            _ => {
                return Err(invalid(
                    pipeline,
                    Some(event.site),
                    &format!(
                        "{kind:?} for epoch {epoch} is illegal while slot {slot} is {state:?}"
                    ),
                ));
            }
        };
        *state = next;
    }
    if slots.iter().any(|state| *state != SlotStateV1::Free) {
        return Err(invalid(
            pipeline,
            None,
            "pipeline exits with committed or consuming epochs that were not released",
        ));
    }
    Ok((epochs.len(), staged_writes, consuming_reads, true))
}

include!("pliron_pipeline_protocol/epoch_loops_v1.rs");

include!("pliron_pipeline_protocol/index_equivalence_v1.rs");

fn invalid(
    pipeline: PlironOperationSiteV1,
    event: Option<PlironOperationSiteV1>,
    detail: &str,
) -> PlironPipelineProtocolFindingV1 {
    PlironPipelineProtocolFindingV1::InvalidSchedule {
        pipeline_block: pipeline.block(),
        pipeline_operation: pipeline.operation(),
        event_block: event.map(PlironOperationSiteV1::block),
        event_operation: event.map(PlironOperationSiteV1::operation),
        detail: detail.to_owned(),
    }
}

fn push_finding(
    findings: &mut Vec<PlironPipelineProtocolFindingV1>,
    finding: PlironPipelineProtocolFindingV1,
) {
    if findings.len() < MAX_PLIRON_PIPELINE_FINDINGS_V1 {
        findings.push(finding);
    } else if !matches!(
        findings.last(),
        Some(PlironPipelineProtocolFindingV1::FindingLimitExceeded)
    ) {
        findings.push(PlironPipelineProtocolFindingV1::FindingLimitExceeded);
    }
}

fn report(finding: PlironPipelineProtocolFindingV1) -> PlironPipelineProtocolReportV1 {
    PlironPipelineProtocolReportV1 {
        findings: vec![finding],
        certificates: Vec::new(),
    }
}

include!("pliron_pipeline_protocol/resource_tests.rs");

#[cfg(test)]
#[path = "pliron_pipeline_protocol/opaque_join_equivalence_v1_tests.rs"]
mod opaque_join_equivalence_v1_tests;

#[cfg(test)]
#[path = "pliron_pipeline_protocol/dynamic_coordinate_order_v1_tests.rs"]
mod dynamic_coordinate_order_v1_tests;

include!("pliron_pipeline_protocol/concrete_cfg_v1_tests.rs");

#[cfg(test)]
#[path = "pliron_pipeline_protocol/concrete_index_facts_v1_tests.rs"]
mod concrete_index_facts_v1_tests;

#[cfg(test)]
#[path = "pliron_pipeline_protocol/nested_epoch_v1_tests.rs"]
mod nested_epoch_v1_tests;
