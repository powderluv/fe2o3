//! Bounded, workload-neutral verification of cooperative tensor distributions.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    error::Error,
    fmt,
};

use dialect_kernel::{
    AnalysisSplitOp, BranchArgsOp, BranchOp, DeterministicJoinOp, IndexBinaryKindAttr,
    IndexBinaryOp, IndexConstantOp, IndexEqualBranchArgsOp, IndexEqualBranchOp,
    IndexLessThanBranchArgsOp, IndexLessThanBranchOp, IndexUnsignedCastOp, InvocationIndexOp,
    MAX_DETERMINISTIC_JOIN_INPUTS_V1, ReturnOp, TensorConvergenceAttr, TensorLayoutOp, TrapOp,
};
use fe2o3_kernel_ir::{
    TensorFragmentLayoutV1, TensorInstructionProfileV1, TensorLayoutFindingV1, TensorOperandRoleV1,
    verify_tensor_layout_contract_v1,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    context::{Context, Ptr},
    operation::Operation,
    value::Value,
};

use crate::pliron_analysis_manager::PlironAnalysisManagerV1;
use crate::pliron_barrier::trace_failure_detail;
use crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1;
use crate::pliron_invocation_trace::{
    PlironExecutionLayoutV1, PlironInvocationTraceV1, PlironTraceEventV1, PlironTraceFailureV1,
    PlironTraceLocationV1,
};
use crate::{
    KernelCheckPassKindV1, KernelCheckStatusV1, SparseIndexAnalysisV1, SparseIndexFactV1,
    SparseIndexFailureV1,
};

pub const MAX_PLIRON_TENSOR_LAYOUT_OPERATIONS_V1: usize = 16_384;
pub const MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1: usize = 256;
pub const MAX_PLIRON_TENSOR_UNIFORMITY_VALUES_V1: usize = 65_536;
pub const MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1: usize = 1_048_576;
pub const MAX_PLIRON_TENSOR_DATAFLOW_ROOTS_V1: usize = 16_384;
pub const MAX_PLIRON_TENSOR_DATAFLOW_EDGES_V1: usize = 65_536;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PlironTensorLayoutLocationV1 {
    pub block: usize,
    pub operation: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlironTensorLayoutFactV1 {
    pub layout: TensorFragmentLayoutV1,
    pub subgroup_width: u16,
    pub profile: TensorInstructionProfileV1,
    pub producer: PlironTensorLayoutLocationV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironTensorLayoutDataflowIssueV1 {
    MergeConflict {
        root: [u64; 4],
        first: PlironTensorLayoutFactV1,
        second: PlironTensorLayoutFactV1,
    },
    ConsumerMismatch {
        root: [u64; 4],
        producer: PlironTensorLayoutFactV1,
        consumer: PlironTensorLayoutLocationV1,
        consumer_profile: TensorInstructionProfileV1,
        operand: TensorOperandRoleV1,
        expected: TensorFragmentLayoutV1,
        expected_subgroup_width: u16,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironTensorLayoutDataflowFailureV1 {
    ResourceLimit,
    MalformedSite { block: usize, operation: usize },
}

/// Whole-function layout facts for compiler-derived cooperative-tensor roots.
///
/// A root may have multiple CFG producers. Equal layouts join; unequal layouts
/// become an explicit conflict. Missing producer facts denote external checked
/// loads or zero initializers, not proof of an arbitrary layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironTensorLayoutDataflowAnalysisV1 {
    facts: BTreeMap<[u64; 4], PlironTensorLayoutFactV1>,
    conflicted_roots: HashSet<[u64; 4]>,
    issues: Vec<PlironTensorLayoutDataflowIssueV1>,
    bound_sites: usize,
}

impl PlironTensorLayoutDataflowAnalysisV1 {
    pub fn fact(&self, root: [u64; 4]) -> Option<PlironTensorLayoutFactV1> {
        (!self.conflicted_roots.contains(&root))
            .then(|| self.facts.get(&root).copied())
            .flatten()
    }

    pub fn issues(&self) -> &[PlironTensorLayoutDataflowIssueV1] {
        &self.issues
    }

    pub const fn bound_site_count(&self) -> usize {
        self.bound_sites
    }
}

#[derive(Clone, Copy)]
struct TensorDataflowSiteV1 {
    location: PlironTensorLayoutLocationV1,
    roots: dialect_kernel::TensorDataflowRootsV1,
    contract: fe2o3_kernel_ir::TensorLayoutContractV1,
}

pub(crate) fn analyze_pliron_tensor_layout_dataflow_with_inventory_v1(
    context: &Context,
    inventory: &BoundedPlironFunctionInventoryV1,
) -> Result<PlironTensorLayoutDataflowAnalysisV1, PlironTensorLayoutDataflowFailureV1> {
    let mut sites = Vec::new();
    let mut operation_count = 0usize;
    for site in inventory.operations() {
        let block = site.block();
        let operation = site.operation();
        operation_count = operation_count.saturating_add(1);
        if operation_count > MAX_PLIRON_TENSOR_LAYOUT_OPERATIONS_V1 {
            return Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit);
        }
        let operation_ref = Operation::get_op_dyn(site.pointer(), context);
        let Some(tensor) = operation_ref.downcast_ref::<TensorLayoutOp>() else {
            continue;
        };
        let roots = tensor
            .dataflow_roots(context)
            .map_err(|_| PlironTensorLayoutDataflowFailureV1::MalformedSite { block, operation })?;
        let Some(roots) = roots else {
            continue;
        };
        let contract = tensor
            .contract(context)
            .map_err(|_| PlironTensorLayoutDataflowFailureV1::MalformedSite { block, operation })?;
        if sites.len() == MAX_PLIRON_TENSOR_DATAFLOW_ROOTS_V1 {
            return Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit);
        }
        sites.push(TensorDataflowSiteV1 {
            location: PlironTensorLayoutLocationV1 { block, operation },
            roots,
            contract,
        });
    }

    let mut facts = BTreeMap::new();
    let mut conflicted_roots = HashSet::new();
    let mut issues = Vec::new();
    for site in &sites {
        let fact = PlironTensorLayoutFactV1 {
            layout: site.contract.accumulator,
            subgroup_width: site.contract.subgroup_width,
            profile: site.contract.profile,
            producer: site.location,
        };
        match facts.get(&site.roots.result).copied() {
            None => {
                facts.insert(site.roots.result, fact);
            }
            Some(first)
                if first.layout == fact.layout && first.subgroup_width == fact.subgroup_width => {}
            Some(first) => {
                conflicted_roots.insert(site.roots.result);
                issues.push(PlironTensorLayoutDataflowIssueV1::MergeConflict {
                    root: site.roots.result,
                    first,
                    second: fact,
                });
            }
        }
        if facts.len() > MAX_PLIRON_TENSOR_DATAFLOW_ROOTS_V1
            || issues.len() > MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1
        {
            return Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit);
        }
    }

    let mut edge_count = 0usize;
    for site in &sites {
        for (root, operand, expected) in [
            (site.roots.lhs, TensorOperandRoleV1::A, site.contract.a),
            (site.roots.rhs, TensorOperandRoleV1::B, site.contract.b),
            (
                site.roots.accumulator,
                TensorOperandRoleV1::Accumulator,
                site.contract.accumulator,
            ),
        ] {
            edge_count = edge_count.saturating_add(1);
            if edge_count > MAX_PLIRON_TENSOR_DATAFLOW_EDGES_V1 {
                return Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit);
            }
            let Some(producer) = facts.get(&root).copied() else {
                continue;
            };
            if conflicted_roots.contains(&root) {
                continue;
            }
            if producer.layout != expected
                || producer.subgroup_width != site.contract.subgroup_width
            {
                issues.push(PlironTensorLayoutDataflowIssueV1::ConsumerMismatch {
                    root,
                    producer,
                    consumer: site.location,
                    consumer_profile: site.contract.profile,
                    operand,
                    expected,
                    expected_subgroup_width: site.contract.subgroup_width,
                });
            }
            if issues.len() > MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
                return Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit);
            }
        }
    }

    Ok(PlironTensorLayoutDataflowAnalysisV1 {
        facts,
        conflicted_roots,
        issues,
        bound_sites: sites.len(),
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironTensorLayoutFindingV1 {
    Contract {
        block: usize,
        operation: usize,
        finding: TensorLayoutFindingV1,
    },
    ActiveLaneMismatch {
        block: usize,
        operation: usize,
        expected: u64,
        actual: u32,
    },
    ExecutionLayoutMismatch {
        block: usize,
        operation: usize,
        declared: u64,
        required: u64,
    },
    ConvergenceMismatch {
        block: usize,
        operation: usize,
        actual: TensorConvergenceAttr,
    },
    MalformedContract {
        block: usize,
        operation: usize,
    },
    DivergentInstructionTrace {
        first_invocation: Vec<u64>,
        first_trace: Vec<(usize, usize)>,
        second_invocation: Vec<u64>,
        second_trace: Vec<(usize, usize)>,
    },
    PartialSubgroupParticipation {
        grid: u64,
        workgroup: u64,
        subgroup: u64,
        expected: u64,
        actual: usize,
    },
    DivergentSubgroupControl {
        block: usize,
        operation: usize,
        controller: usize,
    },
    ConvergenceAnalysisIncomplete {
        detail: String,
    },
    Dataflow(Box<PlironTensorLayoutDataflowIssueV1>),
    ResourceLimitExceeded,
}

impl PlironTensorLayoutFindingV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::Contract { finding, .. } if finding.is_incomplete() => {
                KernelCheckStatusV1::Incomplete
            }
            Self::ConvergenceAnalysisIncomplete { .. } | Self::ResourceLimitExceeded => {
                KernelCheckStatusV1::Incomplete
            }
            Self::Contract { .. }
            | Self::ActiveLaneMismatch { .. }
            | Self::ExecutionLayoutMismatch { .. }
            | Self::ConvergenceMismatch { .. }
            | Self::MalformedContract { .. }
            | Self::DivergentInstructionTrace { .. }
            | Self::PartialSubgroupParticipation { .. }
            | Self::DivergentSubgroupControl { .. }
            | Self::Dataflow(_) => KernelCheckStatusV1::Rejected,
        }
    }
}

impl fmt::Display for PlironTensorLayoutFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract {
                block,
                operation,
                finding,
            } if finding.is_incomplete() => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-002]: tensor layout analysis is incomplete at block {block} op {operation}: {finding}",
            ),
            Self::Contract {
                block,
                operation,
                finding,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor layout rejected at block {block} op {operation}: {finding}",
            ),
            Self::ActiveLaneMismatch {
                block,
                operation,
                expected,
                actual,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor layout rejected at block {block} op {operation}: authenticated execution requires {expected} active lanes, found {actual}",
            ),
            Self::ExecutionLayoutMismatch {
                block,
                operation,
                declared,
                required,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor layout rejected at block {block} op {operation}: authenticated execution layout declares subgroup width {declared}, but the tensor contract requires {required}",
            ),
            Self::ConvergenceMismatch {
                block,
                operation,
                actual,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor layout rejected at block {block} op {operation}: exact uniform subgroup convergence is required, found {actual:?}",
            ),
            Self::MalformedContract { block, operation } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor layout rejected malformed contract at block {block} op {operation}",
            ),
            Self::DivergentInstructionTrace {
                first_invocation,
                first_trace,
                second_invocation,
                second_trace,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: divergent tensor-instruction trace; invocation {first_invocation:?} executes {first_trace:?}, while invocation {second_invocation:?} executes {second_trace:?}; every subgroup participant must execute the same tensor instructions in the same order",
            ),
            Self::PartialSubgroupParticipation {
                grid,
                workgroup,
                subgroup,
                expected,
                actual,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor subgroup ({grid}, {workgroup}, {subgroup}) has {actual} retained participants; the authenticated execution layout requires all {expected} lanes",
            ),
            Self::DivergentSubgroupControl {
                block,
                operation,
                controller,
            } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-001]: tensor instruction at block {block} op {operation} is control-dependent on subgroup-varying branch block {controller}",
            ),
            Self::ConvergenceAnalysisIncomplete { detail } => write!(
                formatter,
                "error[FE2O3-TENSOR-LAYOUT-002]: tensor convergence analysis is incomplete: {detail}",
            ),
            Self::Dataflow(issue) => display_dataflow_issue(formatter, issue),
            Self::ResourceLimitExceeded => formatter.write_str(
                "error[FE2O3-TENSOR-LAYOUT-003]: tensor layout analysis resource limit exceeded; help: split the kernel or reduce tensor/control-flow graph size so the bounded analysis can complete",
            ),
        }
    }
}

fn display_dataflow_issue(
    formatter: &mut fmt::Formatter<'_>,
    issue: &PlironTensorLayoutDataflowIssueV1,
) -> fmt::Result {
    match issue {
        PlironTensorLayoutDataflowIssueV1::MergeConflict {
            root,
            first,
            second,
        } => write!(
            formatter,
            "error[FE2O3-TENSOR-LAYOUT-004]: incompatible tensor layouts reach value root {} from block {} op {} ({}) and block {} op {} ({}); help: make every control-flow producer use the same fragment layout, or insert an explicit checked conversion before the join",
            display_root(*root),
            first.producer.block,
            first.producer.operation,
            describe_layout(*first),
            second.producer.block,
            second.producer.operation,
            describe_layout(*second),
        ),
        PlironTensorLayoutDataflowIssueV1::ConsumerMismatch {
            root,
            producer,
            consumer,
            consumer_profile,
            operand,
            expected,
            expected_subgroup_width,
        } => write!(
            formatter,
            "error[FE2O3-TENSOR-LAYOUT-005]: tensor value root {} is produced at block {} op {} as {}, but block {} op {} uses it as {operand:?} for profile {consumer_profile:?}, which requires {}; help: {}",
            display_root(*root),
            producer.producer.block,
            producer.producer.operation,
            describe_layout(*producer),
            consumer.block,
            consumer.operation,
            describe_fragment(*expected, *expected_subgroup_width),
            layout_mismatch_repair(*producer, *operand, *consumer_profile),
        ),
    }
}

fn display_root(root: [u64; 4]) -> String {
    format!(
        "{:016x}{:016x}{:016x}{:016x}",
        root[0], root[1], root[2], root[3]
    )
}

fn describe_layout(fact: PlironTensorLayoutFactV1) -> String {
    format!(
        "profile {:?}, {}",
        fact.profile,
        describe_fragment(fact.layout, fact.subgroup_width)
    )
}

fn describe_fragment(layout: TensorFragmentLayoutV1, subgroup_width: u16) -> String {
    format!(
        "{:?} {:?} {}x{} fragment with {} components across wave{}",
        layout.role,
        layout.element,
        layout.shape[0],
        layout.shape[1],
        layout.fragment_elements,
        subgroup_width,
    )
}

fn layout_mismatch_repair(
    producer: PlironTensorLayoutFactV1,
    operand: TensorOperandRoleV1,
    consumer_profile: TensorInstructionProfileV1,
) -> String {
    if operand == TensorOperandRoleV1::Accumulator {
        format!(
            "select a consumer instruction whose accumulator ABI accepts profile {:?}, or explicitly convert the accumulator before profile {consumer_profile:?}",
            producer.profile,
        )
    } else {
        format!(
            "insert a checked conversion/repack from the produced accumulator layout to the required {operand:?} fragment, or choose a consumer instruction whose {operand:?} ABI accepts profile {:?}",
            producer.profile,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironTensorLayoutReportV1 {
    findings: Vec<PlironTensorLayoutFindingV1>,
}

impl PlironTensorLayoutReportV1 {
    pub const fn pass(&self) -> KernelCheckPassKindV1 {
        KernelCheckPassKindV1::TensorLayout
    }

    pub fn status(&self) -> KernelCheckStatusV1 {
        self.findings
            .iter()
            .fold(KernelCheckStatusV1::Clean, |status, finding| {
                status.join(finding.status())
            })
    }

    pub fn findings(&self) -> &[PlironTensorLayoutFindingV1] {
        &self.findings
    }

    pub fn is_clean(&self) -> bool {
        self.status() == KernelCheckStatusV1::Clean
    }

    /// Contract consistency is not a source-to-IR or producer/dominance proof.
    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    /// Raw ranked declarations never authorize artifact publication or launch.
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironTensorLayoutCheckErrorV1 {
    report: PlironTensorLayoutReportV1,
}

impl PlironTensorLayoutCheckErrorV1 {
    pub const fn report(&self) -> &PlironTensorLayoutReportV1 {
        &self.report
    }
}

impl fmt::Display for PlironTensorLayoutCheckErrorV1 {
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

impl Error for PlironTensorLayoutCheckErrorV1 {}

pub fn run_pliron_tensor_layout_check_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironTensorLayoutReportV1 {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    run_pliron_tensor_layout_check_with_analyses_v1(context, function, &mut analyses)
}

pub(crate) fn run_pliron_tensor_layout_check_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> PlironTensorLayoutReportV1 {
    analyses.prepare_function_inventory(context, function);
    let inventory = match analyses.function_inventory_handle() {
        Ok(inventory) => inventory,
        Err(_) => return report(vec![PlironTensorLayoutFindingV1::ResourceLimitExceeded]),
    };
    let mut findings = Vec::new();
    let mut operation_count = 0;
    let mut tensor_sites = Vec::new();
    for site in inventory.operations() {
        let block_index = site.block();
        let operation_index = site.operation();
        operation_count += 1;
        if operation_count > MAX_PLIRON_TENSOR_LAYOUT_OPERATIONS_V1
            || findings.len() >= MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1
        {
            findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
            return report(findings);
        }
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(layout) = operation.downcast_ref::<TensorLayoutOp>() else {
            continue;
        };
        let contract = layout.contract(context);
        tensor_sites.push((
            block_index,
            operation_index,
            layout.active_lanes(context),
            contract
                .as_ref()
                .ok()
                .map(|contract| u64::from(contract.subgroup_width)),
        ));
        let Ok(contract) = contract else {
            findings.push(PlironTensorLayoutFindingV1::MalformedContract {
                block: block_index,
                operation: operation_index,
            });
            continue;
        };
        if layout.convergence(context) != Some(TensorConvergenceAttr::UniformSubgroup) {
            findings.push(PlironTensorLayoutFindingV1::ConvergenceMismatch {
                block: block_index,
                operation: operation_index,
                actual: layout
                    .convergence(context)
                    .unwrap_or(TensorConvergenceAttr::Opaque),
            });
        }
        for finding in verify_tensor_layout_contract_v1(&contract) {
            if findings.len() >= MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
                findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
                return report(findings);
            }
            findings.push(PlironTensorLayoutFindingV1::Contract {
                block: block_index,
                operation: operation_index,
                finding,
            });
        }
    }
    analyses.prepare_tensor_layout_dataflow(context, function);
    match analyses.tensor_layout_dataflow() {
        Ok(dataflow) => {
            for issue in dataflow.issues() {
                if findings.len() >= MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
                    findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
                    return report(findings);
                }
                findings.push(PlironTensorLayoutFindingV1::Dataflow(Box::new(
                    issue.clone(),
                )));
            }
        }
        Err(PlironTensorLayoutDataflowFailureV1::ResourceLimit) => {
            findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
            return report(findings);
        }
        Err(PlironTensorLayoutDataflowFailureV1::MalformedSite { block, operation }) => {
            findings.push(PlironTensorLayoutFindingV1::MalformedContract { block, operation });
            return report(findings);
        }
    }
    if !tensor_sites.is_empty() {
        analyses.prepare_execution_layout(context, function);
        let layout = match analyses.execution_layout() {
            Ok(Some(layout)) => layout,
            Ok(None) => {
                findings.push(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "tensor instructions require one authenticated gpu.execution_layout in the entry block"
                        .to_owned(),
                });
                return report(findings);
            }
            Err(failure) => {
                findings.push(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: trace_failure_detail(failure),
                });
                return report(findings);
            }
        };
        for (block, operation, active_lanes, contract_width) in &tensor_sites {
            if let Some(required) = contract_width
                && layout.subgroup_size != *required
            {
                if findings.len() >= MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
                    findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
                    return report(findings);
                }
                findings.push(PlironTensorLayoutFindingV1::ExecutionLayoutMismatch {
                    block: *block,
                    operation: *operation,
                    declared: layout.subgroup_size,
                    required: *required,
                });
            }
            if let Some(actual) = active_lanes
                && u64::from(*actual) != layout.subgroup_size
            {
                if findings.len() >= MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
                    findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
                    return report(findings);
                }
                findings.push(PlironTensorLayoutFindingV1::ActiveLaneMismatch {
                    block: *block,
                    operation: *operation,
                    expected: layout.subgroup_size,
                    actual: *actual,
                });
            }
        }
        analyses.prepare_exact_trace(context, function);
        match analyses.exact_trace() {
            Ok(traces) => {
                if let Some(finding) = exact_subgroup_trace_finding(traces, layout.subgroup_size) {
                    findings.push(finding);
                }
            }
            Err(PlironTraceFailureV1::Sparse(SparseIndexFailureV1::ResourceLimit { .. })) => {
                findings.push(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
            }
            Err(
                PlironTraceFailureV1::DynamicLaunch { .. }
                | PlironTraceFailureV1::LaunchTooLarge { .. }
                | PlironTraceFailureV1::UnresolvedBranch { .. }
                | PlironTraceFailureV1::CyclicControlFlow { .. }
                | PlironTraceFailureV1::UnsupportedTerminator { .. },
            ) => match symbolic_subgroup_convergence(
                context,
                function,
                &inventory,
                layout,
                analyses,
                &tensor_sites
                    .iter()
                    .map(|(block, operation, _, _)| (*block, *operation))
                    .collect::<Vec<_>>(),
            ) {
                Ok(()) => {}
                Err(finding) => findings.push(finding),
            },
            Err(failure) => {
                findings.push(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: trace_failure_detail(failure),
                });
            }
        }
    }
    report(findings)
}

fn exact_subgroup_trace_finding(
    traces: &[PlironInvocationTraceV1],
    subgroup_size: u64,
) -> Option<PlironTensorLayoutFindingV1> {
    let mut groups = BTreeMap::<(u64, u64, u64), Vec<&PlironInvocationTraceV1>>::new();
    for trace in traces {
        groups
            .entry((trace.grid, trace.workgroup, trace.subgroup))
            .or_default()
            .push(trace);
    }
    for ((grid, workgroup, subgroup), group) in groups {
        if !group.iter().any(|trace| !tensor_trace(trace).is_empty()) {
            continue;
        }
        let lanes = group.iter().map(|trace| trace.lane).collect::<HashSet<_>>();
        let complete = lanes.len() == subgroup_size as usize
            && group.len() == subgroup_size as usize
            && (0..subgroup_size).all(|lane| lanes.contains(&lane));
        if !complete {
            return Some(PlironTensorLayoutFindingV1::PartialSubgroupParticipation {
                grid,
                workgroup,
                subgroup,
                expected: subgroup_size,
                actual: lanes.len(),
            });
        }
        let Some(first) = group.first().copied() else {
            continue;
        };
        let first_tensor = tensor_trace(first);
        for trace in group.iter().copied().skip(1) {
            let tensor = tensor_trace(trace);
            if tensor != first_tensor {
                return Some(PlironTensorLayoutFindingV1::DivergentInstructionTrace {
                    first_invocation: first.invocation.clone(),
                    first_trace: first_tensor
                        .iter()
                        .map(|location| (location.block, location.operation))
                        .collect(),
                    second_invocation: trace.invocation.clone(),
                    second_trace: tensor
                        .iter()
                        .map(|location| (location.block, location.operation))
                        .collect(),
                });
            }
        }
    }
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubgroupBranchUniformityV1 {
    Uniform,
    Varying,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SubgroupValueUniformityV1 {
    Uniform,
    Unknown,
    Varying,
}

impl SubgroupValueUniformityV1 {
    const fn rank(self) -> u8 {
        match self {
            Self::Uniform => 0,
            Self::Unknown => 1,
            Self::Varying => 2,
        }
    }

    fn merge(inputs: impl IntoIterator<Item = Self>) -> Self {
        inputs
            .into_iter()
            .max_by_key(|uniformity| uniformity.rank())
            .unwrap_or(Self::Unknown)
    }
}

enum SubgroupValueDefinitionV1 {
    Fixed(SubgroupValueUniformityV1),
    Merge(Vec<Value>),
}

struct PlironSubgroupUniformityV1 {
    facts: HashMap<Value, SubgroupValueUniformityV1>,
}

impl PlironSubgroupUniformityV1 {
    fn fact(&self, value: Value) -> SubgroupValueUniformityV1 {
        self.facts
            .get(&value)
            .copied()
            .unwrap_or(SubgroupValueUniformityV1::Unknown)
    }
}

struct SymbolicTensorCfgV1 {
    successors: Vec<Vec<usize>>,
    branch_uniformity: Vec<SubgroupBranchUniformityV1>,
    reachable: Vec<bool>,
}

struct TensorControlRegionV1 {
    blocks: Vec<u64>,
    has_cycle: bool,
}

impl TensorControlRegionV1 {
    fn contains(&self, block: usize) -> bool {
        self.blocks
            .get(block / u64::BITS as usize)
            .is_some_and(|word| word & (1_u64 << (block % u64::BITS as usize)) != 0)
    }
}

fn symbolic_subgroup_convergence(
    context: &Context,
    function: &FuncOp,
    inventory: &BoundedPlironFunctionInventoryV1,
    layout: PlironExecutionLayoutV1,
    analyses: &mut PlironAnalysisManagerV1,
    tensor_sites: &[(usize, usize)],
) -> Result<(), PlironTensorLayoutFindingV1> {
    let potentially_partial = layout
        .global_extents
        .iter()
        .zip(layout.workgroup_extents)
        .any(|(global, workgroup)| *global == 0 || !global.is_multiple_of(workgroup));
    if potentially_partial
        && layout.execution_domain != dialect_gpu::ExecutionDomainAttr::FullPhysicalWorkgroups
    {
        return Err(
            PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "symbolic tensor convergence cannot establish full subgroup participation for a partial workgroup"
                    .to_owned(),
            },
        );
    }
    analyses.prepare_sparse_indices(context, function);
    let sparse = analyses.sparse_indices().map_err(|failure| match failure {
        SparseIndexFailureV1::ResourceLimit { .. } => {
            PlironTensorLayoutFindingV1::ResourceLimitExceeded
        }
        failure => PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: format!("sparse predicate analysis failed: {failure:?}"),
        },
    })?;
    let uniformity = analyze_pliron_subgroup_uniformity(context, function, inventory, layout)?;
    let blocks = inventory.blocks();
    if blocks.is_empty() || blocks.len() > MAX_PLIRON_TENSOR_LAYOUT_OPERATIONS_V1 {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    let block_indices = blocks
        .iter()
        .copied()
        .enumerate()
        .map(|(index, block)| (block, index))
        .collect::<HashMap<_, _>>();
    let entry = function.get_entry_block(context);
    let mut successors = Vec::with_capacity(blocks.len());
    let mut branch_uniformity = Vec::with_capacity(blocks.len());
    for (block_index, block) in blocks.iter().copied().enumerate() {
        let terminator = block
            .deref(context)
            .get_terminator(context)
            .ok_or_else(
                || PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!("block {block_index} has no terminator"),
                },
            )?;
        let terminator = Operation::get_op_dyn(terminator, context);
        let raw = terminator.get_operation().deref(context);
        let (kind, expected_successors) = if terminator.downcast_ref::<ReturnOp>().is_some()
            || terminator.downcast_ref::<TrapOp>().is_some()
        {
            (SubgroupBranchUniformityV1::Uniform, 0)
        } else if terminator.downcast_ref::<BranchOp>().is_some()
            || terminator.downcast_ref::<BranchArgsOp>().is_some()
        {
            (SubgroupBranchUniformityV1::Uniform, 1)
        } else if let Some(branch) = terminator.downcast_ref::<IndexLessThanBranchOp>() {
            if raw.get_num_operands() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} index comparison has a malformed operand count"
                    ),
                });
            }
            (
                classify_subgroup_predicate(
                    entry,
                    layout,
                    sparse,
                    &uniformity,
                    branch.lhs(context),
                    branch.rhs(context),
                ),
                2,
            )
        } else if let Some(branch) = terminator.downcast_ref::<IndexLessThanBranchArgsOp>() {
            if raw.get_num_successors() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} typed index comparison has a malformed successor count"
                    ),
                });
            }
            let expected_operands = 2
                + raw.get_successor(0).deref(context).get_num_arguments()
                + raw.get_successor(1).deref(context).get_num_arguments();
            if raw.get_num_operands() != expected_operands {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} typed index comparison has a malformed operand count"
                    ),
                });
            }
            (
                classify_subgroup_predicate(
                    entry,
                    layout,
                    sparse,
                    &uniformity,
                    branch.lhs(context),
                    branch.rhs(context),
                ),
                2,
            )
        } else if let Some(branch) = terminator.downcast_ref::<IndexEqualBranchOp>() {
            if raw.get_num_operands() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} equality comparison has a malformed operand count"
                    ),
                });
            }
            (
                classify_subgroup_equality(
                    entry,
                    layout,
                    sparse,
                    &uniformity,
                    branch.lhs(context),
                    branch.rhs(context),
                ),
                2,
            )
        } else if let Some(branch) = terminator.downcast_ref::<IndexEqualBranchArgsOp>() {
            if raw.get_num_successors() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} typed equality comparison has a malformed successor count"
                    ),
                });
            }
            let expected_operands = 2
                + raw.get_successor(0).deref(context).get_num_arguments()
                + raw.get_successor(1).deref(context).get_num_arguments();
            if raw.get_num_operands() != expected_operands {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: format!(
                        "block {block_index} typed equality comparison has a malformed operand count"
                    ),
                });
            }
            (
                classify_subgroup_equality(
                    entry,
                    layout,
                    sparse,
                    &uniformity,
                    branch.lhs(context),
                    branch.rhs(context),
                ),
                2,
            )
        } else if let Some(split) = terminator.downcast_ref::<AnalysisSplitOp>() {
            let dependencies = split.control_dependencies(context);
            let kind = if dependencies.is_empty() {
                SubgroupBranchUniformityV1::Unknown
            } else {
                match SubgroupValueUniformityV1::merge(
                    dependencies
                        .into_iter()
                        .map(|dependency| uniformity.fact(dependency)),
                ) {
                    SubgroupValueUniformityV1::Uniform => SubgroupBranchUniformityV1::Uniform,
                    SubgroupValueUniformityV1::Varying => SubgroupBranchUniformityV1::Varying,
                    SubgroupValueUniformityV1::Unknown => SubgroupBranchUniformityV1::Unknown,
                }
            };
            (kind, 2)
        } else {
            return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: format!("block {block_index} has an unsupported terminator"),
            });
        };
        if raw.get_num_successors() != expected_successors {
            return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: format!(
                    "block {block_index} terminator has {} successors, expected {expected_successors}",
                    raw.get_num_successors()
                ),
            });
        }
        let targets = raw
            .successors()
            .map(|successor| {
                block_indices.get(&successor).copied().ok_or_else(|| {
                    PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: format!("block {block_index} targets a block outside the kernel"),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        successors.push(targets);
        branch_uniformity.push(kind);
    }

    let mut reachable = vec![false; blocks.len()];
    let entry_index = block_indices.get(&entry).copied().ok_or_else(|| {
        PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "the kernel entry block is outside its body region".to_owned(),
        }
    })?;
    let mut worklist = VecDeque::from([entry_index]);
    while let Some(block) = worklist.pop_front() {
        if reachable[block] {
            continue;
        }
        reachable[block] = true;
        worklist.extend(successors[block].iter().copied());
    }
    let cfg = SymbolicTensorCfgV1 {
        successors,
        branch_uniformity,
        reachable,
    };

    let tensor_blocks = tensor_sites.iter().copied().fold(
        BTreeMap::<usize, usize>::new(),
        |mut blocks, (block, operation)| {
            blocks.entry(block).or_insert(operation);
            blocks
        },
    );
    if tensor_blocks.len() > MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    let mut convergence_work = 0_usize;
    let predecessors = bounded_predecessors(&cfg.successors, &mut convergence_work)?;
    let postdominators = bounded_postdominators(
        &cfg.successors,
        &cfg.reachable,
        &predecessors,
        &mut convergence_work,
    )?;
    let control_regions = bounded_control_regions(
        &cfg.successors,
        &cfg.reachable,
        &cfg.branch_uniformity,
        &postdominators,
        &mut convergence_work,
    )?;
    let tensor_blocks = tensor_blocks.into_iter().collect::<Vec<_>>();
    let mut tensor_block_ids = Vec::new();
    tensor_block_ids
        .try_reserve_exact(tensor_blocks.len())
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    tensor_block_ids.extend(tensor_blocks.iter().map(|(block, _)| *block));
    let tensor_reachability = bounded_tensor_reachability(
        &cfg.successors,
        &predecessors,
        &tensor_block_ids,
        &mut convergence_work,
    )?;
    let edge_count = cfg
        .successors
        .iter()
        .try_fold(0_usize, |count, targets| count.checked_add(targets.len()));
    let Some(edge_count) = edge_count else {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    };
    let controller_query_work = cfg
        .successors
        .len()
        .checked_mul(2)
        .and_then(|blocks| blocks.checked_add(edge_count))
        .and_then(|per_tensor| per_tensor.checked_mul(tensor_blocks.len()))
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    charge_convergence_work(&mut convergence_work, controller_query_work)?;
    for (tensor_index, (tensor_block, tensor_operation)) in tensor_blocks.into_iter().enumerate() {
        if tensor_block >= cfg.successors.len() || !cfg.reachable[tensor_block] {
            continue;
        }
        for controller in 0..cfg.successors.len() {
            let kind = cfg.branch_uniformity[controller];
            let mut controls_future_tensor = false;
            for successor in cfg.successors[controller].iter().copied() {
                if tensor_reachability.block_reaches(successor, tensor_index)? {
                    controls_future_tensor = true;
                    break;
                }
            }
            if !cfg.reachable[controller]
                || !tensor_reachability.block_reaches(controller, tensor_index)?
                || !controls_future_tensor
                || kind == SubgroupBranchUniformityV1::Uniform
            {
                continue;
            }
            let Some(region) = &control_regions[controller] else {
                continue;
            };
            if !region.contains(tensor_block) && !region.has_cycle {
                continue;
            }
            return Err(match kind {
                SubgroupBranchUniformityV1::Varying => {
                    PlironTensorLayoutFindingV1::DivergentSubgroupControl {
                        block: tensor_block,
                        operation: tensor_operation,
                        controller,
                    }
                }
                SubgroupBranchUniformityV1::Unknown => {
                    PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: format!(
                            "tensor instruction at block {tensor_block} op {tensor_operation} is control-dependent on unresolved branch block {controller}"
                        ),
                    }
                }
                SubgroupBranchUniformityV1::Uniform => {
                    PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: format!(
                            "uniform controller block {controller} reached the divergent-control rejection boundary"
                        ),
                    }
                }
            });
        }
    }
    Ok(())
}

fn classify_subgroup_predicate(
    entry: Ptr<BasicBlock>,
    layout: PlironExecutionLayoutV1,
    sparse: &SparseIndexAnalysisV1,
    uniformity: &PlironSubgroupUniformityV1,
    lhs: Value,
    rhs: Value,
) -> SubgroupBranchUniformityV1 {
    if lhs == rhs {
        return SubgroupBranchUniformityV1::Uniform;
    }
    let lhs_is_entry_argument = value_is_entry_argument(lhs, entry);
    let rhs_is_entry_argument = value_is_entry_argument(rhs, entry);
    let lhs_uniformity = uniformity.fact(lhs);
    let rhs_uniformity = uniformity.fact(rhs);
    let lhs = sparse.fact(lhs);
    let rhs = sparse.fact(rhs);
    if (lhs_is_entry_argument
        || lhs_uniformity == SubgroupValueUniformityV1::Uniform
        || sparse_fact_is_subgroup_uniform(&lhs, layout))
        && (rhs_is_entry_argument
            || rhs_uniformity == SubgroupValueUniformityV1::Uniform
            || sparse_fact_is_subgroup_uniform(&rhs, layout))
    {
        return SubgroupBranchUniformityV1::Uniform;
    }
    if let (Some(lhs), Some(rhs)) = (lhs.affine(), rhs.affine()) {
        let differing_lane_coefficient = lhs
            .coefficients()
            .iter()
            .zip(rhs.coefficients())
            .enumerate()
            .any(|(dimension, (lhs, rhs))| {
                lhs != rhs && !invocation_axis_is_subgroup_uniform(dimension, layout)
            });
        if !differing_lane_coefficient
            && affine_is_total_over_layout(lhs, layout)
            && affine_is_total_over_layout(rhs, layout)
        {
            return SubgroupBranchUniformityV1::Uniform;
        }
    }
    if let Some(classification) = classify_coordinate_cutoff(&lhs, &rhs, layout, true) {
        return classification;
    }
    if let Some(classification) = classify_coordinate_cutoff(&rhs, &lhs, layout, false) {
        return classification;
    }
    if matches!(
        (lhs_uniformity, rhs_uniformity),
        (
            SubgroupValueUniformityV1::Varying,
            SubgroupValueUniformityV1::Uniform
        ) | (
            SubgroupValueUniformityV1::Uniform,
            SubgroupValueUniformityV1::Varying
        )
    ) {
        return SubgroupBranchUniformityV1::Varying;
    }
    SubgroupBranchUniformityV1::Unknown
}

fn classify_subgroup_equality(
    entry: Ptr<BasicBlock>,
    layout: PlironExecutionLayoutV1,
    sparse: &SparseIndexAnalysisV1,
    uniformity: &PlironSubgroupUniformityV1,
    lhs: Value,
    rhs: Value,
) -> SubgroupBranchUniformityV1 {
    if lhs == rhs {
        return SubgroupBranchUniformityV1::Uniform;
    }
    let lhs_uniformity = uniformity.fact(lhs);
    let rhs_uniformity = uniformity.fact(rhs);
    let lhs_fact = sparse.fact(lhs);
    let rhs_fact = sparse.fact(rhs);
    let lhs_is_uniform = value_is_entry_argument(lhs, entry)
        || lhs_uniformity == SubgroupValueUniformityV1::Uniform
        || sparse_fact_is_subgroup_uniform(&lhs_fact, layout);
    let rhs_is_uniform = value_is_entry_argument(rhs, entry)
        || rhs_uniformity == SubgroupValueUniformityV1::Uniform
        || sparse_fact_is_subgroup_uniform(&rhs_fact, layout);
    if lhs_is_uniform && rhs_is_uniform {
        return SubgroupBranchUniformityV1::Uniform;
    }
    if let (Some(lhs), Some(rhs)) = (lhs_fact.affine(), rhs_fact.affine())
        && lhs
            .coefficients()
            .iter()
            .zip(rhs.coefficients())
            .enumerate()
            .all(|(dimension, (lhs, rhs))| {
                lhs == rhs || invocation_axis_is_subgroup_uniform(dimension, layout)
            })
        && affine_is_total_over_layout(lhs, layout)
        && affine_is_total_over_layout(rhs, layout)
    {
        return SubgroupBranchUniformityV1::Uniform;
    }
    if matches!(
        (lhs_uniformity, rhs_uniformity),
        (
            SubgroupValueUniformityV1::Varying,
            SubgroupValueUniformityV1::Uniform
        ) | (
            SubgroupValueUniformityV1::Uniform,
            SubgroupValueUniformityV1::Varying
        )
    ) {
        return SubgroupBranchUniformityV1::Varying;
    }
    SubgroupBranchUniformityV1::Unknown
}

fn affine_is_total_over_layout(
    affine: &crate::SparseAffineIndexV1,
    layout: PlironExecutionLayoutV1,
) -> bool {
    let mut maximum = affine.constant_term();
    for (dimension, coefficient) in affine.coefficients().iter().copied().enumerate() {
        if coefficient == 0 {
            continue;
        }
        let Some(extent) = layout.global_extents.get(dimension).copied() else {
            return false;
        };
        let Some(coordinate) = extent.checked_sub(1) else {
            return false;
        };
        let Some(term) = coefficient.checked_mul(coordinate) else {
            return false;
        };
        let Some(next) = maximum.checked_add(term) else {
            return false;
        };
        maximum = next;
    }
    true
}

fn analyze_pliron_subgroup_uniformity(
    context: &Context,
    function: &FuncOp,
    inventory: &BoundedPlironFunctionInventoryV1,
    layout: PlironExecutionLayoutV1,
) -> Result<PlironSubgroupUniformityV1, PlironTensorLayoutFindingV1> {
    let entry = function.get_entry_block(context);
    let mut definitions = HashMap::<Value, SubgroupValueDefinitionV1>::new();
    let mut definition_order = Vec::new();
    let mut dependents = HashMap::<Value, Vec<Value>>::new();
    let mut block_arguments = HashMap::<Ptr<BasicBlock>, Vec<Value>>::new();
    let mut collection_work = 0_usize;
    for (block_index, block) in inventory.blocks().iter().copied().enumerate() {
        let argument_count = block.deref(context).get_num_arguments();
        charge_uniformity_collection(&mut collection_work, argument_count)?;
        ensure_uniformity_value_capacity(definitions.len(), argument_count)?;
        let arguments = block.deref(context).arguments().collect::<Vec<_>>();
        for argument in &arguments {
            definition_order.push(*argument);
            definitions.insert(
                *argument,
                if block == entry {
                    SubgroupValueDefinitionV1::Fixed(SubgroupValueUniformityV1::Uniform)
                } else {
                    SubgroupValueDefinitionV1::Merge(Vec::new())
                },
            );
        }
        block_arguments.insert(block, arguments);
        for site in inventory.block_operations(block_index) {
            let operation = site.pointer();
            let dynamic = Operation::get_op_dyn(operation, context);
            let raw = operation.deref(context);
            let result_count = raw.get_num_results();
            charge_uniformity_collection(
                &mut collection_work,
                raw.get_num_operands().saturating_add(result_count),
            )?;
            ensure_uniformity_value_capacity(definitions.len(), result_count)?;
            for result_index in 0..result_count {
                let result = raw.get_result(result_index);
                let definition = if dynamic.downcast_ref::<IndexConstantOp>().is_some() {
                    SubgroupValueDefinitionV1::Fixed(SubgroupValueUniformityV1::Uniform)
                } else if let Some(invocation) = dynamic.downcast_ref::<InvocationIndexOp>() {
                    let uniformity = match invocation
                        .dimension(context)
                        .and_then(|dimension| usize::try_from(dimension).ok())
                    {
                        Some(dimension)
                            if invocation_axis_is_subgroup_uniform(dimension, layout) =>
                        {
                            SubgroupValueUniformityV1::Uniform
                        }
                        Some(_) => SubgroupValueUniformityV1::Varying,
                        None => SubgroupValueUniformityV1::Unknown,
                    };
                    SubgroupValueDefinitionV1::Fixed(uniformity)
                } else if let Some(binary) = dynamic.downcast_ref::<IndexBinaryOp>()
                    && raw.get_num_operands() == 2
                {
                    if invocation_quotient_is_subgroup_uniform(context, binary, layout) {
                        SubgroupValueDefinitionV1::Fixed(SubgroupValueUniformityV1::Uniform)
                    } else {
                        SubgroupValueDefinitionV1::Merge(vec![
                            binary.lhs(context),
                            binary.rhs(context),
                        ])
                    }
                } else if let Some(cast) = dynamic.downcast_ref::<IndexUnsignedCastOp>() {
                    SubgroupValueDefinitionV1::Merge(vec![cast.source(context)])
                } else if let Some(join) = dynamic.downcast_ref::<DeterministicJoinOp>() {
                    let dependencies = join.dependencies(context);
                    if dependencies.is_empty() {
                        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                            detail: "deterministic join has no explicit dependencies".to_owned(),
                        });
                    }
                    if dependencies.len() > MAX_DETERMINISTIC_JOIN_INPUTS_V1 {
                        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
                    }
                    SubgroupValueDefinitionV1::Merge(dependencies)
                } else {
                    SubgroupValueDefinitionV1::Fixed(SubgroupValueUniformityV1::Unknown)
                };
                definition_order.push(result);
                definitions.insert(result, definition);
            }
        }
    }
    for block in inventory.blocks() {
        let Some(terminator) = block.deref(context).get_terminator(context) else {
            continue;
        };
        let dynamic = Operation::get_op_dyn(terminator, context);
        let raw = terminator.deref(context);
        if dynamic
            .downcast_ref::<IndexLessThanBranchArgsOp>()
            .is_some()
            || dynamic.downcast_ref::<IndexEqualBranchArgsOp>().is_some()
        {
            if raw.get_num_successors() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "typed conditional edge has a malformed successor count".to_owned(),
                });
            }
            let expected_operands = 2
                + raw.get_successor(0).deref(context).get_num_arguments()
                + raw.get_successor(1).deref(context).get_num_arguments();
            if raw.get_num_operands() != expected_operands {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "typed conditional edge has a malformed operand count".to_owned(),
                });
            }
        }
        if let Some(split) = dynamic.downcast_ref::<AnalysisSplitOp>() {
            if raw.get_num_successors() != 2 {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "analysis split has a malformed successor count".to_owned(),
                });
            }
            let expected_operands = split.control_dependencies(context).len()
                + raw.get_successor(0).deref(context).get_num_arguments()
                + raw.get_successor(1).deref(context).get_num_arguments();
            if raw.get_num_operands() != expected_operands {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "analysis split has a malformed operand count".to_owned(),
                });
            }
        }
        for (successor_index, successor) in raw.successors().enumerate() {
            let Some(arguments) = block_arguments.get(&successor) else {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "a branch targets a block outside the kernel".to_owned(),
                });
            };
            if arguments.is_empty() {
                continue;
            }
            charge_uniformity_collection(&mut collection_work, arguments.len())?;
            let incoming = if let Some(branch) = dynamic.downcast_ref::<BranchArgsOp>() {
                branch.arguments(context)
            } else if let Some(branch) = dynamic.downcast_ref::<IndexLessThanBranchArgsOp>() {
                if successor_index == 0 {
                    branch.true_arguments(context)
                } else {
                    branch.false_arguments(context)
                }
            } else if let Some(branch) = dynamic.downcast_ref::<IndexEqualBranchArgsOp>() {
                if successor_index == 0 {
                    branch.true_arguments(context)
                } else {
                    branch.false_arguments(context)
                }
            } else if let Some(split) = dynamic.downcast_ref::<AnalysisSplitOp>() {
                if successor_index == 0 {
                    split.first_arguments(context)
                } else {
                    split.second_arguments(context)
                }
            } else {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "a block argument has a predecessor without typed edge operands"
                        .to_owned(),
                });
            };
            if incoming.len() != arguments.len() {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "typed edge operand and block argument counts differ".to_owned(),
                });
            }
            for (argument, incoming) in arguments.iter().zip(incoming) {
                let Some(SubgroupValueDefinitionV1::Merge(values)) = definitions.get_mut(argument)
                else {
                    return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: "an entry argument cannot receive a CFG edge operand".to_owned(),
                    });
                };
                values.push(incoming);
            }
        }
    }

    for value in &definition_order {
        let Some(definition) = definitions.get(value) else {
            return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "ordered subgroup-uniformity definition disappeared".to_owned(),
            });
        };
        if let SubgroupValueDefinitionV1::Merge(inputs) = definition {
            charge_uniformity_collection(&mut collection_work, inputs.len())?;
            for input in inputs {
                dependents.entry(*input).or_default().push(*value);
            }
        }
    }
    let mut facts = definition_order
        .iter()
        .copied()
        .map(|value| (value, SubgroupValueUniformityV1::Uniform))
        .collect::<HashMap<_, _>>();
    let mut worklist = definition_order.into_iter().collect::<VecDeque<_>>();
    let mut work_units = 0_usize;
    while let Some(value) = worklist.pop_front() {
        work_units = work_units.saturating_add(1);
        if work_units > MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 {
            return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
        }
        let Some(definition) = definitions.get(&value) else {
            return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "queued subgroup-uniformity definition disappeared".to_owned(),
            });
        };
        let next = match definition {
            SubgroupValueDefinitionV1::Fixed(uniformity) => *uniformity,
            SubgroupValueDefinitionV1::Merge(inputs) => {
                SubgroupValueUniformityV1::merge(inputs.iter().map(|input| {
                    facts
                        .get(input)
                        .copied()
                        .unwrap_or(SubgroupValueUniformityV1::Unknown)
                }))
            }
        };
        let Some(current) = facts.get_mut(&value) else {
            return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "subgroup-uniformity definition has no lattice fact".to_owned(),
            });
        };
        if next.rank() <= current.rank() {
            continue;
        }
        *current = next;
        worklist.extend(dependents.get(&value).into_iter().flatten().copied());
    }
    Ok(PlironSubgroupUniformityV1 { facts })
}

fn charge_uniformity_collection(
    work: &mut usize,
    amount: usize,
) -> Result<(), PlironTensorLayoutFindingV1> {
    *work = work
        .checked_add(amount)
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    if *work > MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    Ok(())
}

fn ensure_uniformity_value_capacity(
    current: usize,
    additional: usize,
) -> Result<(), PlironTensorLayoutFindingV1> {
    if current
        .checked_add(additional)
        .is_none_or(|count| count > MAX_PLIRON_TENSOR_UNIFORMITY_VALUES_V1)
    {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    Ok(())
}

fn invocation_quotient_is_subgroup_uniform(
    context: &Context,
    binary: &IndexBinaryOp,
    layout: PlironExecutionLayoutV1,
) -> bool {
    if binary.kind(context) != Some(IndexBinaryKindAttr::Divide) || layout.subgroup_size == 0 {
        return false;
    }
    let Some(divisor) = binary.rhs(context).defining_op().and_then(|operation| {
        Operation::get_op_dyn(operation, context)
            .downcast_ref::<IndexConstantOp>()
            .and_then(|constant| constant.value(context))
    }) else {
        return false;
    };
    if divisor == 0 || !divisor.is_multiple_of(layout.subgroup_size) {
        return false;
    }
    let Some(invocation) = binary.lhs(context).defining_op().and_then(|operation| {
        Operation::get_op_dyn(operation, context)
            .downcast_ref::<InvocationIndexOp>()
            .cloned()
    }) else {
        return false;
    };
    if invocation.dimension(context) != Some(0) {
        return false;
    }
    let workgroup_extent = layout.workgroup_extents[0];
    workgroup_extent >= layout.subgroup_size
        && workgroup_extent.is_multiple_of(layout.subgroup_size)
}

fn value_is_entry_argument(value: Value, entry: Ptr<BasicBlock>) -> bool {
    value.defining_block() == Some(entry)
}

fn sparse_fact_is_subgroup_uniform(
    fact: &SparseIndexFactV1,
    layout: PlironExecutionLayoutV1,
) -> bool {
    match fact {
        SparseIndexFactV1::Affine(affine) => {
            affine
                .coefficients()
                .iter()
                .enumerate()
                .all(|(dimension, coefficient)| {
                    *coefficient == 0 || invocation_axis_is_subgroup_uniform(dimension, layout)
                })
        }
        SparseIndexFactV1::Remainder { dividend, modulus } => {
            *modulus == 1
                || dividend
                    .coefficients()
                    .iter()
                    .enumerate()
                    .all(|(dimension, coefficient)| {
                        *coefficient == 0 || invocation_axis_is_subgroup_uniform(dimension, layout)
                    })
        }
        SparseIndexFactV1::Unknown
        | SparseIndexFactV1::MachineOverflow(_)
        | SparseIndexFactV1::CheckedTiled2D(_)
        | SparseIndexFactV1::CheckedRowStriped2D(_) => false,
    }
}

fn invocation_axis_is_subgroup_uniform(dimension: usize, layout: PlironExecutionLayoutV1) -> bool {
    let Some(&extent) = layout.workgroup_extents.get(dimension) else {
        return false;
    };
    if extent == 1 {
        return true;
    }
    let Some(stride) = layout.workgroup_extents[..dimension]
        .iter()
        .try_fold(1_u64, |stride, extent| stride.checked_mul(*extent))
    else {
        return false;
    };
    stride >= layout.subgroup_size && stride.is_multiple_of(layout.subgroup_size)
}

fn classify_coordinate_cutoff(
    coordinate: &SparseIndexFactV1,
    cutoff: &SparseIndexFactV1,
    layout: PlironExecutionLayoutV1,
    coordinate_is_lhs: bool,
) -> Option<SubgroupBranchUniformityV1> {
    let affine = coordinate.affine()?;
    if affine.constant_term() != 0 {
        return None;
    }
    let mut dimensions = affine
        .coefficients()
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coefficient)| *coefficient != 0);
    let (dimension, coefficient) = dimensions.next()?;
    if coefficient != 1 || dimensions.next().is_some() {
        return None;
    }
    let cutoff = cutoff.constant_value()?;
    let global_extent = layout.global_extents.get(dimension).copied()?;
    if (coordinate_is_lhs && cutoff == 0)
        || (!coordinate_is_lhs
            && (cutoff == u64::MAX
                || (global_extent != 0 && cutoff >= global_extent.saturating_sub(1))))
        || (coordinate_is_lhs && global_extent != 0 && cutoff >= global_extent)
        || invocation_axis_is_subgroup_uniform(dimension, layout)
    {
        return Some(SubgroupBranchUniformityV1::Uniform);
    }
    let boundary = if coordinate_is_lhs {
        cutoff
    } else {
        cutoff.checked_add(1)?
    };
    let extent = *layout.workgroup_extents.get(dimension)?;
    let stride = layout.workgroup_extents[..dimension]
        .iter()
        .try_fold(1_u64, |stride, extent| stride.checked_mul(*extent))?;
    let period = extent.checked_mul(stride)?;
    let transition = (boundary % extent).checked_mul(stride)?;
    if period.is_multiple_of(layout.subgroup_size) {
        return Some(if transition.is_multiple_of(layout.subgroup_size) {
            SubgroupBranchUniformityV1::Uniform
        } else {
            SubgroupBranchUniformityV1::Varying
        });
    }
    None
}

fn charge_convergence_work(
    work: &mut usize,
    amount: usize,
) -> Result<(), PlironTensorLayoutFindingV1> {
    *work = work
        .checked_add(amount)
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    if *work > MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1 {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    Ok(())
}

fn bounded_predecessors(
    successors: &[Vec<usize>],
    work: &mut usize,
) -> Result<Vec<Vec<usize>>, PlironTensorLayoutFindingV1> {
    let edge_count = successors
        .iter()
        .try_fold(0_usize, |count, targets| count.checked_add(targets.len()))
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    charge_convergence_work(work, successors.len())?;
    charge_convergence_work(work, edge_count)?;
    let mut predecessors = vec![Vec::new(); successors.len()];
    for (block, targets) in successors.iter().enumerate() {
        for target in targets {
            if *target >= successors.len() {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "convergence edge is outside the kernel CFG".to_owned(),
                });
            }
            predecessors[*target].push(block);
        }
    }
    Ok(predecessors)
}

struct TensorReachabilityV1 {
    component_of: Vec<usize>,
    tensors_by_component: Vec<Vec<u64>>,
}

impl TensorReachabilityV1 {
    fn block_reaches(
        &self,
        block: usize,
        tensor: usize,
    ) -> Result<bool, PlironTensorLayoutFindingV1> {
        let component = self.component_of.get(block).copied().ok_or_else(|| {
            PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "tensor reachability queried a block outside the kernel CFG".to_owned(),
            }
        })?;
        let words = self.tensors_by_component.get(component).ok_or_else(|| {
            PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "tensor reachability lost a block SCC".to_owned(),
            }
        })?;
        let word = words.get(tensor / u64::BITS as usize).ok_or_else(|| {
            PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "tensor reachability queried an unknown tensor site".to_owned(),
            }
        })?;
        Ok(word & (1_u64 << (tensor % u64::BITS as usize)) != 0)
    }
}

fn bounded_tensor_reachability(
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
    tensor_blocks: &[usize],
    work: &mut usize,
) -> Result<TensorReachabilityV1, PlironTensorLayoutFindingV1> {
    let block_count = successors.len();
    if predecessors.len() != block_count {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "tensor reachability inputs do not match the kernel CFG".to_owned(),
        });
    }
    if tensor_blocks.len() > MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1 {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    }
    for targets in successors {
        for target in targets.iter().copied() {
            if target >= block_count {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "tensor reachability edge is outside the kernel CFG".to_owned(),
                });
            }
        }
    }
    if tensor_blocks.iter().any(|block| *block >= block_count) {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "tensor block is outside the kernel CFG".to_owned(),
        });
    }

    charge_convergence_work(work, block_count.saturating_mul(3))?;
    let mut visited = vec![false; block_count];
    let mut finish_order = Vec::new();
    finish_order
        .try_reserve_exact(block_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    let mut dfs = Vec::new();
    dfs.try_reserve_exact(block_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    for root in 0..block_count {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        dfs.push((root, 0_usize));
        while let Some((block, successor_index)) = dfs.last_mut() {
            charge_convergence_work(work, 1)?;
            if let Some(successor) = successors[*block].get(*successor_index).copied() {
                *successor_index += 1;
                if !visited[successor] {
                    visited[successor] = true;
                    dfs.push((successor, 0));
                }
            } else {
                finish_order.push(*block);
                dfs.pop();
            }
        }
    }

    let mut component_of = vec![usize::MAX; block_count];
    let mut component_count = 0_usize;
    let mut component_worklist = Vec::new();
    component_worklist
        .try_reserve_exact(block_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    for root in finish_order.into_iter().rev() {
        if component_of[root] != usize::MAX {
            continue;
        }
        let component = component_count;
        component_count = component_count
            .checked_add(1)
            .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
        component_of[root] = component;
        component_worklist.push(root);
        while let Some(block) = component_worklist.pop() {
            charge_convergence_work(work, 1)?;
            for predecessor in predecessors[block].iter().copied() {
                charge_convergence_work(work, 1)?;
                if predecessor >= block_count {
                    return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: "tensor reachability predecessor is outside the kernel CFG"
                            .to_owned(),
                    });
                }
                if component_of[predecessor] == usize::MAX {
                    component_of[predecessor] = component;
                    component_worklist.push(predecessor);
                }
            }
        }
    }
    if component_of
        .iter()
        .any(|component| *component == usize::MAX)
    {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "tensor reachability did not assign every block to an SCC".to_owned(),
        });
    }

    charge_convergence_work(work, component_count)?;
    let edge_count = successors
        .iter()
        .try_fold(0_usize, |count, targets| count.checked_add(targets.len()));
    let Some(edge_count) = edge_count else {
        return Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded);
    };
    let mut component_edges = HashSet::new();
    component_edges
        .try_reserve(edge_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    for (block, targets) in successors.iter().enumerate() {
        let source_component = component_of[block];
        for target in targets.iter().copied() {
            let target_component = component_of[target];
            if source_component != target_component
                && component_edges.insert((source_component, target_component))
            {
                charge_convergence_work(work, 1)?;
            }
        }
    }

    let mut component_outdegree = vec![0_usize; component_count];
    for (source, _) in component_edges.iter().copied() {
        component_outdegree[source] = component_outdegree[source]
            .checked_add(1)
            .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    }
    let mut component_successors = Vec::new();
    component_successors
        .try_reserve_exact(component_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    for degree in component_outdegree {
        let mut targets = Vec::new();
        targets
            .try_reserve_exact(degree)
            .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
        component_successors.push(targets);
    }
    for (source, target) in component_edges {
        component_successors[source].push(target);
    }

    charge_convergence_work(work, component_count.saturating_mul(2))?;
    let mut indegree = vec![0_usize; component_count];
    for targets in &component_successors {
        for target in targets.iter().copied() {
            indegree[target] = indegree[target]
                .checked_add(1)
                .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
        }
    }
    let mut ready = VecDeque::new();
    ready
        .try_reserve_exact(component_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    ready.extend(
        indegree
            .iter()
            .enumerate()
            .filter_map(|(component, degree)| (*degree == 0).then_some(component)),
    );
    let mut topological = Vec::new();
    topological
        .try_reserve_exact(component_count)
        .map_err(|_| PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    while let Some(component) = ready.pop_front() {
        charge_convergence_work(work, 1)?;
        topological.push(component);
        for successor in component_successors[component].iter().copied() {
            let degree = indegree.get_mut(successor).ok_or_else(|| {
                PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "tensor reachability SCC edge is outside the condensation graph"
                        .to_owned(),
                }
            })?;
            *degree = degree.checked_sub(1).ok_or_else(|| {
                PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "tensor reachability SCC indegree underflowed".to_owned(),
                }
            })?;
            if *degree == 0 {
                ready.push_back(successor);
            }
        }
    }
    if topological.len() != component_count {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "tensor reachability condensation graph is cyclic".to_owned(),
        });
    }

    let word_count = tensor_blocks.len().div_ceil(u64::BITS as usize);
    let storage = component_count
        .checked_mul(word_count)
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    charge_convergence_work(work, storage)?;
    let mut tensors_by_component = vec![vec![0_u64; word_count]; component_count];
    for (tensor, block) in tensor_blocks.iter().copied().enumerate() {
        let component = component_of[block];
        tensors_by_component[component][tensor / u64::BITS as usize] |=
            1_u64 << (tensor % u64::BITS as usize);
    }
    for component in topological.into_iter().rev() {
        for successor in component_successors[component].iter().copied() {
            charge_convergence_work(work, word_count)?;
            for word in 0..word_count {
                tensors_by_component[component][word] |= tensors_by_component[successor][word];
            }
        }
    }

    Ok(TensorReachabilityV1 {
        component_of,
        tensors_by_component,
    })
}

fn bounded_postdominators(
    successors: &[Vec<usize>],
    reachable: &[bool],
    predecessors: &[Vec<usize>],
    work: &mut usize,
) -> Result<Vec<Option<Vec<u64>>>, PlironTensorLayoutFindingV1> {
    let block_count = successors.len();
    let word_count = block_count.div_ceil(u64::BITS as usize);
    if predecessors.len() != block_count || reachable.len() != block_count {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "postdominator inputs do not match the kernel CFG".to_owned(),
        });
    }
    charge_convergence_work(work, block_count)?;
    charge_convergence_work(work, word_count)?;
    for (block, targets) in successors.iter().enumerate() {
        if !reachable.get(block).copied().unwrap_or(false) {
            continue;
        }
        for target in targets.iter().copied() {
            if target >= block_count {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "postdominator edge is outside the kernel CFG".to_owned(),
                });
            }
            let _ = reachable[target];
        }
    }

    charge_convergence_work(work, block_count)?;
    let mut can_reach_exit = vec![false; block_count];
    let mut worklist = VecDeque::new();
    for block in 0..block_count {
        if reachable[block] && successors[block].is_empty() {
            can_reach_exit[block] = true;
            worklist.push_back(block);
        }
    }
    while let Some(block) = worklist.pop_front() {
        for predecessor in predecessors[block].iter().copied() {
            charge_convergence_work(work, 1)?;
            let Some(predecessor_reaches_exit) = can_reach_exit.get_mut(predecessor) else {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "postdominator predecessor is outside the kernel CFG".to_owned(),
                });
            };
            if !*predecessor_reaches_exit {
                *predecessor_reaches_exit = true;
                worklist.push_back(predecessor);
            }
        }
    }

    charge_convergence_work(work, word_count)?;
    let mut universe = vec![0_u64; word_count];
    for block in 0..block_count {
        if can_reach_exit[block] {
            universe[block / u64::BITS as usize] |= 1_u64 << (block % u64::BITS as usize);
        }
    }
    let initial_words = can_reach_exit
        .iter()
        .filter(|available| **available)
        .count()
        .saturating_mul(word_count);
    charge_convergence_work(work, initial_words)?;
    let mut facts = can_reach_exit
        .iter()
        .map(|available| available.then(|| universe.clone()))
        .collect::<Vec<_>>();
    loop {
        let mut changed = false;
        for block in (0..block_count).rev() {
            if !can_reach_exit[block] {
                continue;
            }
            let mut next = if successors[block].is_empty()
                || successors[block]
                    .iter()
                    .any(|successor| !can_reach_exit[*successor])
            {
                charge_convergence_work(work, word_count)?;
                vec![0_u64; word_count]
            } else {
                let mut targets = successors[block].iter();
                let Some(first) = targets.next().copied() else {
                    return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: "a non-terminal postdominator block has no successor".to_owned(),
                    });
                };
                charge_convergence_work(work, word_count)?;
                let Some(first_fact) = facts[first].as_ref() else {
                    return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: "an exit-reachable successor has no postdominator fact".to_owned(),
                    });
                };
                let mut intersection = first_fact.clone();
                for successor in targets {
                    let Some(successor) = facts[*successor].as_ref() else {
                        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                            detail: "an exit-reachable successor has no postdominator fact"
                                .to_owned(),
                        });
                    };
                    for (word, successor_word) in intersection.iter_mut().zip(successor) {
                        charge_convergence_work(work, 1)?;
                        *word &= successor_word;
                    }
                }
                intersection
            };
            next[block / u64::BITS as usize] |= 1_u64 << (block % u64::BITS as usize);
            let Some(slot) = facts[block].as_mut() else {
                return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                    detail: "an exit-reachable block has no postdominator fact".to_owned(),
                });
            };
            charge_convergence_work(work, word_count)?;
            if *slot != next {
                *slot = next;
                changed = true;
            }
        }
        if !changed {
            return Ok(facts);
        }
        charge_convergence_work(work, block_count)?;
    }
}

fn immediate_postdominator(
    controller: usize,
    facts: &[Option<Vec<u64>>],
    depths: &[u32],
    work: &mut usize,
) -> Result<Option<usize>, PlironTensorLayoutFindingV1> {
    if depths.len() != facts.len() {
        return Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
            detail: "postdominator depths do not match the kernel CFG".to_owned(),
        });
    }
    let Some(fact) = facts.get(controller).and_then(Option::as_ref) else {
        return Ok(None);
    };
    let mut best = None;
    let mut best_depth = 0_u32;
    let mut tied = false;
    for (word_index, word) in fact.iter().copied().enumerate() {
        let mut candidates = word;
        while candidates != 0 {
            charge_convergence_work(work, 1)?;
            let bit = candidates.trailing_zeros() as usize;
            candidates &= candidates - 1;
            let candidate = word_index * u64::BITS as usize + bit;
            if candidate == controller || candidate >= facts.len() {
                continue;
            }
            let depth = depths[candidate];
            if depth > best_depth {
                best = Some(candidate);
                best_depth = depth;
                tied = false;
            } else if depth == best_depth {
                tied = true;
            }
        }
    }
    Ok((!tied).then_some(best).flatten())
}

fn bounded_control_regions(
    successors: &[Vec<usize>],
    reachable: &[bool],
    branch_uniformity: &[SubgroupBranchUniformityV1],
    postdominators: &[Option<Vec<u64>>],
    work: &mut usize,
) -> Result<Vec<Option<TensorControlRegionV1>>, PlironTensorLayoutFindingV1> {
    let block_count = successors.len();
    let word_count = block_count.div_ceil(u64::BITS as usize);
    let controller_count = branch_uniformity
        .iter()
        .enumerate()
        .filter(|(block, uniformity)| {
            reachable[*block]
                && **uniformity != SubgroupBranchUniformityV1::Uniform
                && successors[*block].len() > 1
        })
        .count();
    let allocation_work = controller_count
        .checked_mul(word_count)
        .and_then(|work| work.checked_add(block_count))
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    charge_convergence_work(work, allocation_work)?;
    let depth_work = block_count
        .checked_mul(word_count)
        .ok_or(PlironTensorLayoutFindingV1::ResourceLimitExceeded)?;
    charge_convergence_work(work, depth_work)?;
    let postdominator_depths = postdominators
        .iter()
        .map(|fact| {
            fact.as_ref()
                .map(|words| words.iter().map(|word| word.count_ones()).sum())
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let mut regions = (0..block_count).map(|_| None).collect::<Vec<_>>();
    for controller in 0..block_count {
        if !reachable[controller]
            || branch_uniformity[controller] == SubgroupBranchUniformityV1::Uniform
            || successors[controller].len() < 2
        {
            continue;
        }
        let reconvergence =
            immediate_postdominator(controller, postdominators, &postdominator_depths, work)?;
        let mut blocks = vec![0_u64; word_count];
        let mut region_blocks = Vec::new();
        let mut worklist = successors[controller]
            .iter()
            .copied()
            .collect::<VecDeque<_>>();
        while let Some(block) = worklist.pop_front() {
            charge_convergence_work(work, 1)?;
            let word = &mut blocks[block / u64::BITS as usize];
            let mask = 1_u64 << (block % u64::BITS as usize);
            if Some(block) == reconvergence || !reachable[block] || *word & mask != 0 {
                continue;
            }
            *word |= mask;
            charge_convergence_work(work, 1)?;
            region_blocks.push(block);
            worklist.extend(successors[block].iter().copied());
        }
        let contains = |block: usize| {
            blocks[block / u64::BITS as usize] & (1_u64 << (block % u64::BITS as usize)) != 0
        };
        charge_convergence_work(work, region_blocks.len())?;
        let mut indegree = region_blocks
            .iter()
            .copied()
            .map(|block| (block, 0_usize))
            .collect::<HashMap<_, _>>();
        for block in region_blocks.iter().copied() {
            for successor in successors[block].iter().copied() {
                if contains(successor) {
                    charge_convergence_work(work, 1)?;
                    let degree = indegree.get_mut(&successor).ok_or_else(|| {
                        PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                            detail: "control region membership is internally inconsistent"
                                .to_owned(),
                        }
                    })?;
                    *degree = degree.saturating_add(1);
                }
            }
        }
        let mut acyclic = indegree
            .iter()
            .filter_map(|(block, indegree)| (*indegree == 0).then_some(*block))
            .collect::<VecDeque<_>>();
        let mut consumed = 0_usize;
        while let Some(block) = acyclic.pop_front() {
            consumed += 1;
            for successor in successors[block].iter().copied() {
                if !contains(successor) {
                    continue;
                }
                charge_convergence_work(work, 1)?;
                let degree = indegree.get_mut(&successor).ok_or_else(|| {
                    PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                        detail: "control region membership is internally inconsistent".to_owned(),
                    }
                })?;
                *degree -= 1;
                if *degree == 0 {
                    acyclic.push_back(successor);
                }
            }
        }
        regions[controller] = Some(TensorControlRegionV1 {
            blocks,
            has_cycle: consumed != region_blocks.len(),
        });
    }
    Ok(regions)
}

fn tensor_trace(trace: &PlironInvocationTraceV1) -> Vec<PlironTraceLocationV1> {
    trace
        .events
        .iter()
        .filter_map(|event| match event {
            PlironTraceEventV1::TensorInstruction { location, .. } => Some(*location),
            PlironTraceEventV1::Barrier { .. }
            | PlironTraceEventV1::Fence { .. }
            | PlironTraceEventV1::Trap { .. }
            | PlironTraceEventV1::Memory { .. }
            | PlironTraceEventV1::CollectiveAllocation { .. } => None,
        })
        .collect()
}

pub fn require_pliron_tensor_layout_before_lowering_v1(
    context: &Context,
    function: &FuncOp,
) -> Result<PlironTensorLayoutReportV1, PlironTensorLayoutCheckErrorV1> {
    let report = run_pliron_tensor_layout_check_v1(context, function);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironTensorLayoutCheckErrorV1 { report })
    }
}

pub(crate) fn require_pliron_tensor_layout_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> Result<PlironTensorLayoutReportV1, PlironTensorLayoutCheckErrorV1> {
    let report = run_pliron_tensor_layout_check_with_analyses_v1(context, function, analyses);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(PlironTensorLayoutCheckErrorV1 { report })
    }
}

fn report(findings: Vec<PlironTensorLayoutFindingV1>) -> PlironTensorLayoutReportV1 {
    PlironTensorLayoutReportV1 { findings }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rejected_contract() -> PlironTensorLayoutFindingV1 {
        PlironTensorLayoutFindingV1::Contract {
            block: 0,
            operation: 0,
            finding: TensorLayoutFindingV1::TailMaskMismatch,
        }
    }

    #[test]
    fn every_tensor_layout_finding_has_the_shared_status() {
        let incomplete = [
            PlironTensorLayoutFindingV1::Contract {
                block: 0,
                operation: 0,
                finding: TensorLayoutFindingV1::UnsupportedProfile,
            },
            PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete {
                detail: "unresolved".to_owned(),
            },
            PlironTensorLayoutFindingV1::ResourceLimitExceeded,
        ];
        for finding in incomplete {
            assert_eq!(finding.status(), KernelCheckStatusV1::Incomplete);
        }

        let rejected = [
            rejected_contract(),
            PlironTensorLayoutFindingV1::ActiveLaneMismatch {
                block: 0,
                operation: 0,
                expected: 64,
                actual: 32,
            },
            PlironTensorLayoutFindingV1::ExecutionLayoutMismatch {
                block: 0,
                operation: 0,
                declared: 32,
                required: 64,
            },
            PlironTensorLayoutFindingV1::ConvergenceMismatch {
                block: 0,
                operation: 0,
                actual: TensorConvergenceAttr::Divergent,
            },
            PlironTensorLayoutFindingV1::MalformedContract {
                block: 0,
                operation: 0,
            },
            PlironTensorLayoutFindingV1::DivergentInstructionTrace {
                first_invocation: vec![0],
                first_trace: vec![(0, 0)],
                second_invocation: vec![1],
                second_trace: vec![],
            },
            PlironTensorLayoutFindingV1::PartialSubgroupParticipation {
                grid: 0,
                workgroup: 0,
                subgroup: 0,
                expected: 64,
                actual: 63,
            },
            PlironTensorLayoutFindingV1::DivergentSubgroupControl {
                block: 0,
                operation: 0,
                controller: 1,
            },
        ];
        for finding in rejected {
            assert_eq!(finding.status(), KernelCheckStatusV1::Rejected);
        }
    }

    #[test]
    fn rejected_tensor_finding_dominates_an_incomplete_finding() {
        let mixed = report(vec![
            PlironTensorLayoutFindingV1::ResourceLimitExceeded,
            rejected_contract(),
        ]);
        assert_eq!(mixed.status(), KernelCheckStatusV1::Rejected);
        assert!(!mixed.is_clean());
        assert_eq!(report(vec![]).status(), KernelCheckStatusV1::Clean);
    }

    #[test]
    fn shared_tensor_reachability_condenses_cycles_and_multiple_sites() {
        let successors = vec![vec![1], vec![2], vec![1, 3], vec![4], vec![]];
        let mut work = 0;
        let predecessors = bounded_predecessors(&successors, &mut work).unwrap();
        let reachability =
            bounded_tensor_reachability(&successors, &predecessors, &[2, 4], &mut work).unwrap();

        for block in [0, 1, 2] {
            assert!(reachability.block_reaches(block, 0).unwrap());
            assert!(reachability.block_reaches(block, 1).unwrap());
        }
        for block in [3, 4] {
            assert!(!reachability.block_reaches(block, 0).unwrap());
            assert!(reachability.block_reaches(block, 1).unwrap());
        }
        assert!(work < MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    }

    #[test]
    fn shared_tensor_reachability_handles_the_site_limit_in_four_words() {
        let block_count = MAX_PLIRON_TENSOR_LAYOUT_FINDINGS_V1;
        let successors = (0..block_count)
            .map(|block| {
                (block + 1 < block_count)
                    .then_some(block + 1)
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let tensor_blocks = (0..block_count).collect::<Vec<_>>();
        let mut work = 0;
        let predecessors = bounded_predecessors(&successors, &mut work).unwrap();
        let reachability =
            bounded_tensor_reachability(&successors, &predecessors, &tensor_blocks, &mut work)
                .unwrap();

        assert!(reachability.block_reaches(0, block_count - 1).unwrap());
        assert!(!reachability.block_reaches(block_count - 1, 0).unwrap());
        assert_eq!(reachability.tensors_by_component[0].len(), 4);
        assert!(work < MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1);
    }

    #[test]
    fn shared_tensor_reachability_rejects_malformed_inputs_without_panicking() {
        let mut work = 0;
        assert!(matches!(
            bounded_tensor_reachability(&[vec![]], &[], &[0], &mut work),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("do not match")
        ));

        let mut work = 0;
        assert!(matches!(
            bounded_tensor_reachability(&[vec![1]], &[vec![]], &[0], &mut work),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("edge is outside")
        ));

        let mut work = 0;
        assert!(matches!(
            bounded_tensor_reachability(&[vec![]], &[vec![]], &[1], &mut work),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("tensor block is outside")
        ));
    }

    #[test]
    fn shared_tensor_reachability_obeys_the_convergence_budget() {
        let mut work = MAX_PLIRON_TENSOR_UNIFORMITY_WORK_UNITS_V1;
        assert!(matches!(
            bounded_tensor_reachability(&[vec![]], &[vec![]], &[0], &mut work),
            Err(PlironTensorLayoutFindingV1::ResourceLimitExceeded)
        ));
    }

    #[test]
    fn tensor_reachability_queries_fail_typed_at_both_boundaries() {
        let reachability = TensorReachabilityV1 {
            component_of: vec![0],
            tensors_by_component: vec![vec![1]],
        };
        assert!(matches!(
            reachability.block_reaches(1, 0),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("block outside")
        ));
        assert!(matches!(
            reachability.block_reaches(0, 64),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("unknown tensor site")
        ));
    }

    #[test]
    fn postdominators_reject_malformed_inputs_without_panicking() {
        let mut work = 0;
        assert!(matches!(
            bounded_postdominators(&[vec![]], &[], &[vec![]], &mut work),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("do not match")
        ));

        let mut work = 0;
        assert!(matches!(
            bounded_postdominators(&[vec![]], &[true], &[vec![1]], &mut work),
            Err(PlironTensorLayoutFindingV1::ConvergenceAnalysisIncomplete { detail })
                if detail.contains("predecessor is outside")
        ));
    }
}
