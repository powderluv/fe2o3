//! Workload-neutral termination and progress checks over ranked PLIRON CFGs.
//!
//! The analysis proves only a closed canonical induction form. Other cyclic
//! control flow is rejected when nontermination is structural, or reported as
//! incomplete when a ranking function would require a stronger solver.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};

use dialect_kernel::{
    AnalysisSplitOp, BranchArgsOp, BranchOp, IndexBinaryKindAttr, IndexBinaryOp, IndexConstantOp,
    IndexEqualBranchArgsOp, IndexLessThanBranchArgsOp, IndexUnsignedCastOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::ops::FuncOp,
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::{Op, OpBox},
    operation::{Operation, verify_operation},
    printable::Printable,
};

use crate::KernelCheckStatusV1;

/// Maximum aggregate nested blocks inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_BLOCKS_V1: usize = 4_096;
/// Maximum aggregate successor records inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_EDGES_V1: usize = 16_384;
/// Maximum aggregate operations below the function root.
pub const MAX_PLIRON_PROGRESS_OPERATIONS_V1: usize = 65_536;
/// Maximum aggregate regions inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_REGIONS_V1: usize = 4_096;
/// Maximum aggregate operation operands inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_OPERANDS_V1: usize = 65_536;
/// Maximum aggregate operation results inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_RESULTS_V1: usize = 65_536;
/// Maximum aggregate operation and block attribute entries.
pub const MAX_PLIRON_PROGRESS_ATTRIBUTES_V1: usize = 65_536;
/// Maximum aggregate block arguments inventoried before recursive verification.
pub const MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1: usize = 65_536;
/// Maximum operation nesting depth admitted to PLIRON's recursive verifier.
pub const MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1: usize = 128;
/// Maximum cumulative work units for inventory, graph construction, and analysis.
pub const MAX_PLIRON_PROGRESS_WORK_UNITS_V1: usize = 262_144;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironProgressCertificateV1 {
    header: usize,
    body: usize,
    exit: usize,
    induction: String,
    bound: String,
    step: u64,
}

impl PlironProgressCertificateV1 {
    pub const fn header(&self) -> usize {
        self.header
    }
    pub const fn body(&self) -> usize {
        self.body
    }
    pub const fn exit(&self) -> usize {
        self.exit
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
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlironProgressFindingV1 {
    StructuralPrerequisiteRejected {
        reason: String,
    },
    ResourceLimitExceeded {
        resource: &'static str,
        actual: usize,
        limit: usize,
    },
    NonTerminatingCycle {
        blocks: Vec<usize>,
        reason: &'static str,
        counterexample: String,
    },
    ProgressIncomplete {
        blocks: Vec<usize>,
        reason: &'static str,
    },
}

impl PlironProgressFindingV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::NonTerminatingCycle { .. } | Self::StructuralPrerequisiteRejected { .. } => {
                KernelCheckStatusV1::Rejected
            }
            Self::ResourceLimitExceeded { .. } | Self::ProgressIncomplete { .. } => {
                KernelCheckStatusV1::Incomplete
            }
        }
    }
}

impl fmt::Display for PlironProgressFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StructuralPrerequisiteRejected { reason } => write!(
                formatter,
                "error[FE2O3-PROGRESS-000]: PLIRON structural verification failed before progress analysis: {reason}; help: repair the malformed operation, type, SSA operand, block argument, or CFG edge"
            ),
            Self::ResourceLimitExceeded {
                resource,
                actual,
                limit,
            } => write!(
                formatter,
                "error[FE2O3-PROGRESS-003]: progress analysis has {actual} {resource}, exceeding limit {limit}; help: simplify or split the kernel so its bounded structural inventory fits the reported resource limit"
            ),
            Self::NonTerminatingCycle {
                blocks,
                reason,
                counterexample,
            } => write!(
                formatter,
                "error[FE2O3-PROGRESS-001]: control-flow cycle {blocks:?} does not terminate: {reason}; counterexample: {counterexample}; help: add an exit controlled by a finite induction variable and advance it on every backedge"
            ),
            Self::ProgressIncomplete { blocks, reason } => write!(
                formatter,
                "error[FE2O3-PROGRESS-002]: termination proof for control-flow cycle {blocks:?} is incomplete: {reason}; help: express the loop as `i < bound` with a positive constant backedge step and a statically proved no-wrap update, or provide a future supported ranking-function contract"
            ),
        }
    }
}

#[derive(Default)]
struct ProgressWorkBudgetV1 {
    work_units: usize,
}

impl ProgressWorkBudgetV1 {
    fn charge(&mut self, units: usize) -> Result<(), PlironProgressFindingV1> {
        let actual = self.work_units.saturating_add(units);
        if actual > MAX_PLIRON_PROGRESS_WORK_UNITS_V1 {
            return Err(PlironProgressFindingV1::ResourceLimitExceeded {
                resource: "work units",
                actual,
                limit: MAX_PLIRON_PROGRESS_WORK_UNITS_V1,
            });
        }
        self.work_units = actual;
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlironProgressReportV1 {
    findings: Vec<PlironProgressFindingV1>,
    certificates: Vec<PlironProgressCertificateV1>,
}

impl PlironProgressReportV1 {
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
    pub fn findings(&self) -> &[PlironProgressFindingV1] {
        &self.findings
    }
    pub fn certificates(&self) -> &[PlironProgressCertificateV1] {
        &self.certificates
    }
    pub const fn grants_launch_or_liveness_authority(&self) -> bool {
        false
    }
    pub(crate) fn clean() -> Self {
        Self {
            findings: Vec::new(),
            certificates: Vec::new(),
        }
    }
}

pub fn run_pliron_progress_check_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironProgressReportV1 {
    match catch_unwind(AssertUnwindSafe(|| {
        run_pliron_progress_check_inner_v1(context, function)
    })) {
        Ok(report) => report,
        Err(payload) => report(structural_rejection(format!(
            "bounded structural preflight panicked: {}",
            panic_detail(payload)
        ))),
    }
}

fn run_pliron_progress_check_inner_v1(
    context: &Context,
    function: &FuncOp,
) -> PlironProgressReportV1 {
    let inventory = match bounded_structural_inventory(context, function) {
        Ok(inventory) => inventory,
        Err(finding) => return report(finding),
    };
    let mut work = ProgressWorkBudgetV1::default();
    if let Err(finding) = work.charge(inventory.verification_work()) {
        return report(finding);
    }
    match catch_unwind(AssertUnwindSafe(|| {
        verify_operation(function.get_operation(), context)
    })) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            return report(structural_rejection(format!(
                "the PLIRON verifier rejected the function at {}",
                bounded_detail(format!("{}", error.disp(context)))
            )));
        }
        Err(payload) => {
            return report(structural_rejection(format!(
                "the PLIRON verifier panicked: {}",
                panic_detail(payload)
            )));
        }
    }

    let blocks = inventory.root_blocks;
    let block_indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect::<HashMap<_, _>>();
    let graph = match build_root_graph(context, &blocks, &block_indices) {
        Ok(graph) => graph,
        Err(finding) => return report(finding),
    };
    let edge_count = graph.edges.iter().map(Vec::len).sum::<usize>();
    let graph_work = blocks
        .len()
        .checked_add(edge_count)
        .and_then(|units| units.checked_mul(8))
        .unwrap_or(usize::MAX);
    if let Err(finding) = work.charge(graph_work) {
        return report(finding);
    }

    let reachable = reachable_blocks(&graph.edges);
    let definitely_reachable = reachable_blocks(&graph.unconditional_edges);
    let dominators = progress_dominators_v1(&graph.edges, &graph.predecessors);
    let mut findings = Vec::new();
    let mut certificates = Vec::new();
    for mut component in strongly_connected_components(&graph.edges) {
        component.sort_unstable();
        let component_work = component.len().saturating_mul(4);
        if let Err(finding) = work.charge(component_work) {
            return report(finding);
        }
        if !component.iter().any(|block| reachable[*block]) || !is_cycle(&component, &graph.edges) {
            continue;
        }
        let component_members = component.iter().copied().collect::<HashSet<_>>();
        let has_exit = component.iter().any(|block| {
            graph.edges[*block]
                .iter()
                .any(|successor| !component_members.contains(successor))
        });
        if !has_exit {
            if component.iter().any(|block| definitely_reachable[*block]) {
                findings.push(PlironProgressFindingV1::NonTerminatingCycle {
                    blocks: component,
                    reason: "the strongly connected component has no exit edge",
                    counterexample: "an unconditional path from entry reaches the cycle, and every successor remains in it".to_owned(),
                });
            } else {
                findings.push(PlironProgressFindingV1::ProgressIncomplete {
                    blocks: component,
                    reason: "the exit-free cycle is only conditionally reachable and no feasible incoming witness was reconstructed",
                });
            }
            continue;
        }
        match canonical_positive_induction_loop(
            context,
            &blocks,
            &block_indices,
            &inventory.root_operation_blocks,
            &graph.predecessors,
            &graph.edges,
            &graph.incoming,
            &component,
            &component_members,
        ) {
            CanonicalLoopResultV1::Proved(certificate) => certificates.push(certificate),
            CanonicalLoopResultV1::Inactive => {}
            CanonicalLoopResultV1::Rejected {
                reason,
                counterexample,
            } => {
                findings.push(PlironProgressFindingV1::NonTerminatingCycle {
                    blocks: component,
                    reason,
                    counterexample,
                });
            }
            CanonicalLoopResultV1::Incomplete(reason) => {
                match prove_nested_positive_induction_loops_v1(
                    context,
                    &blocks,
                    &block_indices,
                    &inventory.root_operation_blocks,
                    &dominators,
                    &graph.predecessors,
                    &graph.edges,
                    &component,
                    &component_members,
                ) {
                    Ok(mut nested) => certificates.append(&mut nested),
                    Err(()) => findings.push(PlironProgressFindingV1::ProgressIncomplete {
                        blocks: component,
                        reason,
                    }),
                }
            }
        }
    }
    PlironProgressReportV1 {
        findings,
        certificates,
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_nested_positive_induction_loops_v1(
    context: &Context,
    blocks: &[Ptr<BasicBlock>],
    block_indices: &HashMap<Ptr<BasicBlock>, usize>,
    operation_blocks: &HashMap<Ptr<Operation>, usize>,
    dominators: &[HashSet<usize>],
    predecessors: &[Vec<usize>],
    edges: &[Vec<usize>],
    component: &[usize],
    component_members: &HashSet<usize>,
) -> Result<Vec<PlironProgressCertificateV1>, ()> {
    let mut backedges = Vec::new();
    for source in component.iter().copied() {
        for target in edges[source].iter().copied() {
            if component_members.contains(&target) && dominators[source].contains(&target) {
                backedges.push((source, target));
            }
        }
    }
    if backedges.is_empty() {
        return Err(());
    }

    let mut certificates = Vec::with_capacity(backedges.len());
    let mut proved_backedges = HashSet::new();
    for (latch, header_index) in backedges {
        let header = blocks[header_index];
        let header_ref = header.deref(context);
        let terminator = header_ref.get_terminator(context).ok_or(())?;
        let operation = Operation::get_op_dyn(terminator, context);
        let branch = operation
            .downcast_ref::<IndexLessThanBranchArgsOp>()
            .ok_or(())?;
        let induction = branch.lhs(context);
        let induction_argument = (0..header_ref.get_num_arguments())
            .find(|argument| header_ref.get_argument(*argument) == induction)
            .ok_or(())?;
        let successors = operation
            .get_operation()
            .deref(context)
            .successors()
            .collect::<Vec<_>>();
        let [body, exit] = successors.as_slice() else {
            return Err(());
        };
        let body_index = *block_indices.get(body).ok_or(())?;
        let exit_index = *block_indices.get(exit).ok_or(())?;

        let mut natural_loop = HashSet::from([header_index, latch]);
        let mut pending = vec![latch];
        while let Some(block) = pending.pop() {
            for predecessor in predecessors[block].iter().copied() {
                if predecessor != header_index
                    && dominators[predecessor].contains(&header_index)
                    && natural_loop.insert(predecessor)
                {
                    pending.push(predecessor);
                }
            }
        }
        if !natural_loop.contains(&body_index) || natural_loop.contains(&exit_index) {
            return Err(());
        }
        for block in natural_loop.iter().copied() {
            if block != header_index
                && predecessors[block]
                    .iter()
                    .any(|predecessor| !natural_loop.contains(predecessor))
            {
                return Err(());
            }
        }
        let entries = predecessors[header_index]
            .iter()
            .copied()
            .filter(|predecessor| !natural_loop.contains(predecessor))
            .collect::<Vec<_>>();
        let [entry] = entries.as_slice() else {
            return Err(());
        };
        let entry_arguments = progress_edge_arguments_v1(context, blocks[*entry], header)?;
        if entry_arguments
            .get(induction_argument)
            .and_then(|value| index_constant(context, *value))
            != Some(0)
        {
            return Err(());
        }
        if branch
            .rhs(context)
            .defining_op()
            .and_then(|definition| operation_blocks.get(&definition).copied())
            .is_some_and(|block| natural_loop.contains(&block))
        {
            return Err(());
        }

        let inductions = propagate_loop_induction_v1(
            context,
            blocks,
            edges,
            &natural_loop,
            header_index,
            induction,
        )?;
        let latch_induction = *inductions.get(&latch).ok_or(())?;
        let latch_arguments = progress_edge_arguments_v1(context, blocks[latch], header)?;
        let next = *latch_arguments.get(induction_argument).ok_or(())?;
        let step = progress_index_offset_v1(context, next, latch_induction).ok_or(())?;
        if step == 0 {
            return Err(());
        }
        if step > 1 {
            let upper_bound = index_constant(context, branch.rhs(context))
                .or_else(|| unsigned_cast_upper_bound(context, branch.rhs(context)))
                .ok_or(())?;
            if upper_bound != 0 && (upper_bound - 1).checked_add(step).is_none() {
                return Err(());
            }
        }
        proved_backedges.insert((latch, header_index));
        certificates.push(PlironProgressCertificateV1 {
            header: header_index,
            body: body_index,
            exit: exit_index,
            induction: induction.unique_name(context).to_string(),
            bound: branch.rhs(context).unique_name(context).to_string(),
            step,
        });
    }
    if !is_acyclic_without_edges_v1(component_members, edges, &proved_backedges) {
        return Err(());
    }
    certificates.sort_by_key(|certificate| certificate.header);
    Ok(certificates)
}

fn progress_dominators_v1(
    edges: &[Vec<usize>],
    predecessors: &[Vec<usize>],
) -> Vec<HashSet<usize>> {
    let reachable = reachable_blocks(edges);
    let all = reachable
        .iter()
        .enumerate()
        .filter_map(|(block, reachable)| (*reachable).then_some(block))
        .collect::<HashSet<_>>();
    let mut dominators = vec![HashSet::new(); edges.len()];
    for block in all.iter().copied() {
        dominators[block] = all.clone();
    }
    if !edges.is_empty() {
        dominators[0] = HashSet::from([0]);
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in all.iter().copied().filter(|block| *block != 0) {
            let mut incoming = predecessors[block]
                .iter()
                .copied()
                .filter(|predecessor| reachable[*predecessor]);
            let Some(first) = incoming.next() else {
                continue;
            };
            let mut next = dominators[first].clone();
            for predecessor in incoming {
                next.retain(|dominator| dominators[predecessor].contains(dominator));
            }
            next.insert(block);
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
    }
    dominators
}

fn propagate_loop_induction_v1(
    context: &Context,
    blocks: &[Ptr<BasicBlock>],
    edges: &[Vec<usize>],
    members: &HashSet<usize>,
    header: usize,
    induction: pliron::value::Value,
) -> Result<HashMap<usize, pliron::value::Value>, ()> {
    let mut inductions = HashMap::from([(header, induction)]);
    for _ in 0..members.len() {
        let mut changed = false;
        for source in members.iter().copied().collect::<Vec<_>>() {
            let Some(source_induction) = inductions.get(&source).copied() else {
                continue;
            };
            for target in edges[source].iter().copied() {
                if target == header || !members.contains(&target) {
                    continue;
                }
                let arguments =
                    progress_edge_arguments_v1(context, blocks[source], blocks[target])?;
                let matching = arguments
                    .iter()
                    .enumerate()
                    .filter(|(_, argument)| **argument == source_induction)
                    .map(|(argument, _)| argument)
                    .collect::<Vec<_>>();
                let [argument] = matching.as_slice() else {
                    return Err(());
                };
                let target_induction = blocks[target].deref(context).get_argument(*argument);
                if let Some(existing) = inductions.get(&target) {
                    if *existing != target_induction {
                        return Err(());
                    }
                } else {
                    inductions.insert(target, target_induction);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    Ok(inductions)
}

fn progress_edge_arguments_v1(
    context: &Context,
    source: Ptr<BasicBlock>,
    target: Ptr<BasicBlock>,
) -> Result<Vec<pliron::value::Value>, ()> {
    let terminator = source.deref(context).get_terminator(context).ok_or(())?;
    let operation = Operation::get_op_dyn(terminator, context);
    let successor = operation
        .get_operation()
        .deref(context)
        .successors()
        .position(|successor| successor == target)
        .ok_or(())?;
    if let Some(branch) = operation.downcast_ref::<BranchArgsOp>() {
        return (successor == 0)
            .then(|| branch.arguments(context))
            .ok_or(());
    }
    if let Some(branch) = operation.downcast_ref::<IndexLessThanBranchArgsOp>() {
        return match successor {
            0 => Ok(branch.true_arguments(context)),
            1 => Ok(branch.false_arguments(context)),
            _ => Err(()),
        };
    }
    if let Some(branch) = operation.downcast_ref::<IndexEqualBranchArgsOp>() {
        return match successor {
            0 => Ok(branch.true_arguments(context)),
            1 => Ok(branch.false_arguments(context)),
            _ => Err(()),
        };
    }
    if let Some(split) = operation.downcast_ref::<AnalysisSplitOp>() {
        return match successor {
            0 => Ok(split.first_arguments(context)),
            1 => Ok(split.second_arguments(context)),
            _ => Err(()),
        };
    }
    (target.deref(context).get_num_arguments() == 0)
        .then(Vec::new)
        .ok_or(())
}

fn progress_index_offset_v1(
    context: &Context,
    value: pliron::value::Value,
    base: pliron::value::Value,
) -> Option<u64> {
    if value == base {
        return Some(0);
    }
    let definition = value.defining_op()?;
    let operation = Operation::get_op_dyn(definition, context);
    let add = operation.downcast_ref::<IndexBinaryOp>()?;
    if add.kind(context) != Some(IndexBinaryKindAttr::Add) {
        return None;
    }
    if add.lhs(context) == base {
        index_constant(context, add.rhs(context))
    } else if add.rhs(context) == base {
        index_constant(context, add.lhs(context))
    } else {
        None
    }
}

fn is_acyclic_without_edges_v1(
    members: &HashSet<usize>,
    edges: &[Vec<usize>],
    removed: &HashSet<(usize, usize)>,
) -> bool {
    let mut incoming = members
        .iter()
        .copied()
        .map(|block| (block, 0_usize))
        .collect::<HashMap<_, _>>();
    for source in members.iter().copied() {
        for target in edges[source].iter().copied() {
            if members.contains(&target) && !removed.contains(&(source, target)) {
                let Some(count) = incoming.get_mut(&target) else {
                    return false;
                };
                *count += 1;
            }
        }
    }
    let mut ready = incoming
        .iter()
        .filter_map(|(block, count)| (*count == 0).then_some(*block))
        .collect::<Vec<_>>();
    let mut visited = 0_usize;
    while let Some(source) = ready.pop() {
        visited += 1;
        for target in edges[source].iter().copied() {
            if !members.contains(&target) || removed.contains(&(source, target)) {
                continue;
            }
            let Some(count) = incoming.get_mut(&target) else {
                return false;
            };
            let Some(next) = count.checked_sub(1) else {
                return false;
            };
            *count = next;
            if next == 0 {
                ready.push(target);
            }
        }
    }
    visited == members.len()
}

#[derive(Default)]
struct StructuralInventoryV1 {
    regions: usize,
    blocks: usize,
    operations: usize,
    operands: usize,
    results: usize,
    attributes: usize,
    block_arguments: usize,
    edges: usize,
    root_blocks: Vec<Ptr<BasicBlock>>,
    root_operation_blocks: HashMap<Ptr<Operation>, usize>,
}

impl StructuralInventoryV1 {
    fn verification_work(&self) -> usize {
        self.regions
            .checked_add(self.blocks)
            .and_then(|n| n.checked_add(self.operations))
            .and_then(|n| n.checked_add(self.operands))
            .and_then(|n| n.checked_add(self.results))
            .and_then(|n| n.checked_add(self.attributes))
            .and_then(|n| n.checked_add(self.block_arguments))
            .and_then(|n| n.checked_add(self.edges))
            .unwrap_or(usize::MAX)
    }
}

fn bounded_structural_inventory(
    context: &Context,
    function: &FuncOp,
) -> Result<StructuralInventoryV1, PlironProgressFindingV1> {
    let root = function.get_operation();
    let root_region = {
        let root_ref = root.try_deref(context).map_err(|error| {
            structural_rejection(format!(
                "the function root cannot be borrowed from the supplied context: {}",
                bounded_detail(format!("{}", error.disp(context)))
            ))
        })?;
        let mut regions = root_ref.regions();
        let Some(root_region) = regions.next() else {
            return Err(structural_rejection(
                "the function root has no body region".to_owned(),
            ));
        };
        root_region
    };
    let mut inventory = StructuralInventoryV1::default();
    let mut pending = vec![(root, 0_usize)];
    while let Some((operation, depth)) = pending.pop() {
        if depth > MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1 {
            return Err(resource_limit(
                "operation nesting depth",
                depth,
                MAX_PLIRON_PROGRESS_NESTING_DEPTH_V1,
            ));
        }
        let operation_ref = operation.try_deref(context).map_err(|error| {
            structural_rejection(format!(
                "an operation cannot be borrowed during structural inventory: {}",
                bounded_detail(format!("{}", error.disp(context)))
            ))
        })?;
        add_count(
            &mut inventory.operands,
            operation_ref.get_num_operands(),
            MAX_PLIRON_PROGRESS_OPERANDS_V1,
            "operands",
        )?;
        add_count(
            &mut inventory.results,
            operation_ref.get_num_results(),
            MAX_PLIRON_PROGRESS_RESULTS_V1,
            "results",
        )?;
        add_count(
            &mut inventory.attributes,
            operation_ref.attributes.0.len(),
            MAX_PLIRON_PROGRESS_ATTRIBUTES_V1,
            "attributes",
        )?;
        add_count(
            &mut inventory.edges,
            operation_ref.get_num_successors(),
            MAX_PLIRON_PROGRESS_EDGES_V1,
            "CFG edges",
        )?;
        for region in operation_ref.regions() {
            add_count(
                &mut inventory.regions,
                1,
                MAX_PLIRON_PROGRESS_REGIONS_V1,
                "regions",
            )?;
            let is_root_region = region == root_region;
            let region_ref = region.try_deref(context).map_err(|error| {
                structural_rejection(format!(
                    "a region cannot be borrowed during structural inventory: {}",
                    bounded_detail(format!("{}", error.disp(context)))
                ))
            })?;
            for block in region_ref.iter(context) {
                add_count(
                    &mut inventory.blocks,
                    1,
                    MAX_PLIRON_PROGRESS_BLOCKS_V1,
                    "basic blocks",
                )?;
                let block_ref = block.try_deref(context).map_err(|error| {
                    structural_rejection(format!(
                        "a block cannot be borrowed during structural inventory: {}",
                        bounded_detail(format!("{}", error.disp(context)))
                    ))
                })?;
                add_count(
                    &mut inventory.block_arguments,
                    block_ref.get_num_arguments(),
                    MAX_PLIRON_PROGRESS_BLOCK_ARGUMENTS_V1,
                    "block arguments",
                )?;
                add_count(
                    &mut inventory.attributes,
                    block_ref.attributes.0.len(),
                    MAX_PLIRON_PROGRESS_ATTRIBUTES_V1,
                    "attributes",
                )?;
                let root_block_index = if is_root_region {
                    let index = inventory.root_blocks.len();
                    inventory.root_blocks.push(block);
                    Some(index)
                } else {
                    None
                };
                for child in block_ref.iter(context) {
                    add_count(
                        &mut inventory.operations,
                        1,
                        MAX_PLIRON_PROGRESS_OPERATIONS_V1,
                        "operations",
                    )?;
                    if let Some(index) = root_block_index {
                        inventory.root_operation_blocks.insert(child, index);
                    }
                    pending.push((child, depth.saturating_add(1)));
                }
            }
        }
    }
    Ok(inventory)
}

fn add_count(
    count: &mut usize,
    amount: usize,
    limit: usize,
    resource: &'static str,
) -> Result<(), PlironProgressFindingV1> {
    let actual = count.checked_add(amount).unwrap_or(usize::MAX);
    if actual > limit {
        return Err(resource_limit(resource, actual, limit));
    }
    *count = actual;
    Ok(())
}

fn resource_limit(resource: &'static str, actual: usize, limit: usize) -> PlironProgressFindingV1 {
    PlironProgressFindingV1::ResourceLimitExceeded {
        resource,
        actual,
        limit,
    }
}

fn structural_rejection(reason: String) -> PlironProgressFindingV1 {
    PlironProgressFindingV1::StructuralPrerequisiteRejected {
        reason: bounded_detail(reason),
    }
}

fn bounded_detail(detail: String) -> String {
    const MAX_DETAIL_CHARS: usize = 1_024;
    if detail.chars().count() <= MAX_DETAIL_CHARS {
        detail
    } else {
        detail.chars().take(MAX_DETAIL_CHARS).collect::<String>() + "..."
    }
}

fn panic_detail(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        bounded_detail((*message).to_owned())
    } else if let Some(message) = payload.downcast_ref::<String>() {
        bounded_detail(message.clone())
    } else {
        "an untyped panic escaped a Rust IR access boundary".to_owned()
    }
}

#[derive(Clone, Copy)]
struct IncomingEdgeV1 {
    source: usize,
    initial: Option<u64>,
}

struct RootGraphV1 {
    edges: Vec<Vec<usize>>,
    unconditional_edges: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
    incoming: Vec<Vec<IncomingEdgeV1>>,
}

fn build_root_graph(
    context: &Context,
    blocks: &[Ptr<BasicBlock>],
    block_indices: &HashMap<Ptr<BasicBlock>, usize>,
) -> Result<RootGraphV1, PlironProgressFindingV1> {
    let mut edges = vec![Vec::new(); blocks.len()];
    let mut unconditional_edges = vec![Vec::new(); blocks.len()];
    let mut predecessors = vec![Vec::new(); blocks.len()];
    let mut incoming = vec![Vec::new(); blocks.len()];
    for (source, block) in blocks.iter().copied().enumerate() {
        let Some(terminator) = block.deref(context).get_terminator(context) else {
            return Err(structural_rejection(format!(
                "block {source} has no registered terminator after structural verification"
            )));
        };
        let operation = Operation::get_op_dyn(terminator, context);
        let initial = operation
            .downcast_ref::<BranchArgsOp>()
            .and_then(|branch| branch.arguments(context).first().copied())
            .and_then(|value| index_constant(context, value));
        let is_unconditional = operation.downcast_ref::<BranchOp>().is_some()
            || operation.downcast_ref::<BranchArgsOp>().is_some();
        for successor in operation.get_operation().deref(context).successors() {
            let Some(target) = block_indices.get(&successor).copied() else {
                return Err(structural_rejection(format!(
                    "block {source} has a successor outside the function after structural verification"
                )));
            };
            edges[source].push(target);
            predecessors[target].push(source);
            incoming[target].push(IncomingEdgeV1 { source, initial });
            if is_unconditional {
                unconditional_edges[source].push(target);
            }
        }
    }
    Ok(RootGraphV1 {
        edges,
        unconditional_edges,
        predecessors,
        incoming,
    })
}

enum CanonicalLoopResultV1 {
    Proved(PlironProgressCertificateV1),
    Inactive,
    Rejected {
        reason: &'static str,
        counterexample: String,
    },
    Incomplete(&'static str),
}

#[allow(clippy::too_many_arguments)]
fn canonical_positive_induction_loop(
    context: &Context,
    blocks: &[Ptr<BasicBlock>],
    block_indices: &HashMap<Ptr<BasicBlock>, usize>,
    operation_blocks: &HashMap<Ptr<Operation>, usize>,
    predecessors: &[Vec<usize>],
    edges: &[Vec<usize>],
    incoming: &[Vec<IncomingEdgeV1>],
    component: &[usize],
    component_members: &HashSet<usize>,
) -> CanonicalLoopResultV1 {
    for header_index in component {
        let header = blocks[*header_index];
        let header_block = header.deref(context);
        let Some(terminator) = header_block.get_terminator(context) else {
            continue;
        };
        let terminator = Operation::get_op_dyn(terminator, context);
        let Some(branch) = terminator.downcast_ref::<IndexLessThanBranchArgsOp>() else {
            continue;
        };
        if header_block.get_num_arguments() != 1
            || branch.lhs(context) != header_block.get_argument(0)
            || branch.true_arguments(context).as_slice() != [branch.lhs(context)]
        {
            continue;
        }
        let successors = branch
            .get_operation()
            .deref(context)
            .successors()
            .collect::<Vec<_>>();
        let [body, exit] = successors.as_slice() else {
            continue;
        };
        let (Some(body_index), Some(exit_index)) = (
            block_indices.get(body).copied(),
            block_indices.get(exit).copied(),
        ) else {
            continue;
        };
        if !component_members.contains(&body_index) || component_members.contains(&exit_index) {
            continue;
        }
        let internal_header_predecessors = predecessors[*header_index]
            .iter()
            .copied()
            .filter(|predecessor| component_members.contains(predecessor))
            .collect::<Vec<_>>();
        let external_header_predecessors = predecessors[*header_index]
            .iter()
            .copied()
            .filter(|predecessor| !component_members.contains(predecessor))
            .collect::<Vec<_>>();
        if internal_header_predecessors.len() != 1 || external_header_predecessors.len() != 1 {
            return CanonicalLoopResultV1::Incomplete(
                "the loop header does not have exactly one external entry and one internal recurrence",
            );
        }
        if branch
            .rhs(context)
            .defining_op()
            .and_then(|definition| operation_blocks.get(&definition))
            .is_some_and(|block| component_members.contains(block))
        {
            return CanonicalLoopResultV1::Incomplete(
                "the loop bound depends on a value defined inside the cycle",
            );
        }
        let latch_index = internal_header_predecessors[0];
        if component.iter().copied().any(|block| {
            block != *header_index
                && predecessors[block]
                    .iter()
                    .any(|predecessor| !component_members.contains(predecessor))
        }) {
            return CanonicalLoopResultV1::Incomplete(
                "a loop body block has an external predecessor that bypasses the guarded header",
            );
        }
        if component
            .iter()
            .copied()
            .any(|block| blocks[block].deref(context).get_num_arguments() != 1)
        {
            return CanonicalLoopResultV1::Incomplete(
                "every block in the guarded recurrence must carry exactly one induction argument",
            );
        }
        let removed_backedge = HashSet::from([(latch_index, *header_index)]);
        if !is_acyclic_without_edges_v1(component_members, edges, &removed_backedge) {
            return CanonicalLoopResultV1::Incomplete(
                "the loop body retains a control-flow cycle that can bypass the induction update",
            );
        }

        let mut next = None;
        let mut latch_induction = None;
        for source_index in component.iter().copied() {
            let source_block = blocks[source_index].deref(context);
            let source_induction = source_block.get_argument(0);
            let Some(terminator) = source_block.get_terminator(context) else {
                return CanonicalLoopResultV1::Incomplete(
                    "a loop recurrence block has no terminator",
                );
            };
            let terminator = Operation::get_op_dyn(terminator, context);
            let successors = terminator
                .get_operation()
                .deref(context)
                .successors()
                .collect::<Vec<_>>();
            for (ordinal, successor) in successors.iter().copied().enumerate() {
                let Some(successor_index) = block_indices.get(&successor).copied() else {
                    return CanonicalLoopResultV1::Incomplete(
                        "a loop recurrence edge leaves the kernel function",
                    );
                };
                if !component_members.contains(&successor_index) {
                    continue;
                }
                let Some(arguments) = successor_arguments(context, &terminator, ordinal) else {
                    return CanonicalLoopResultV1::Incomplete(
                        "an internal loop edge does not expose exact SSA successor arguments",
                    );
                };
                let [forwarded] = arguments.as_slice() else {
                    return CanonicalLoopResultV1::Incomplete(
                        "an internal loop edge does not carry exactly one induction value",
                    );
                };
                if source_index == latch_index && successor_index == *header_index {
                    next = Some(*forwarded);
                    latch_induction = Some(source_induction);
                } else if *forwarded != source_induction {
                    return CanonicalLoopResultV1::Incomplete(
                        "an internal loop edge does not forward the induction value unchanged",
                    );
                }
            }
        }
        let (Some(latch_induction), Some(next)) = (latch_induction, next) else {
            return CanonicalLoopResultV1::Incomplete(
                "the unique loop latch does not carry an authenticated induction update",
            );
        };
        if next == latch_induction {
            return zero_step_result(
                context,
                component_members,
                &incoming[*header_index],
                branch.rhs(context),
                "the induction variable is unchanged on the backedge",
            );
        }
        let Some(increment_definition) = next.defining_op() else {
            return CanonicalLoopResultV1::Incomplete(
                "the backedge value is not a locally reconstructed induction update",
            );
        };
        let increment = Operation::get_op_dyn(increment_definition, context);
        let Some(increment) = increment.downcast_ref::<IndexBinaryOp>() else {
            return CanonicalLoopResultV1::Incomplete(
                "the induction update is not target-neutral index addition",
            );
        };
        if increment.kind(context) != Some(IndexBinaryKindAttr::Add)
            || increment.lhs(context) != latch_induction
        {
            return CanonicalLoopResultV1::Incomplete("the induction update is not `i + constant`");
        }
        let Some(step_definition) = increment.rhs(context).defining_op() else {
            return CanonicalLoopResultV1::Incomplete("the induction step is not constant");
        };
        let step = Operation::get_op_dyn(step_definition, context);
        let Some(step) = step.downcast_ref::<IndexConstantOp>() else {
            return CanonicalLoopResultV1::Incomplete("the induction step is not constant");
        };
        let step = match step.value(context) {
            Some(0) => {
                return zero_step_result(
                    context,
                    component_members,
                    &incoming[*header_index],
                    branch.rhs(context),
                    "the induction step is zero",
                );
            }
            Some(step) => step,
            None => return CanonicalLoopResultV1::Incomplete("the induction step is malformed"),
        };
        if step > 1 {
            let upper_bound = index_constant(context, branch.rhs(context))
                .or_else(|| unsigned_cast_upper_bound(context, branch.rhs(context)));
            let Some(bound) = upper_bound else {
                return CanonicalLoopResultV1::Incomplete(
                    "a symbolic bound with a non-unit step needs a no-wrap range proof",
                );
            };
            if bound != 0 && (bound - 1).checked_add(step).is_none() {
                return CanonicalLoopResultV1::Incomplete(
                    "the largest guarded induction value plus the step can overflow u64",
                );
            }
        }
        return CanonicalLoopResultV1::Proved(PlironProgressCertificateV1 {
            header: *header_index,
            body: body_index,
            exit: exit_index,
            induction: branch.lhs(context).unique_name(context).to_string(),
            bound: branch.rhs(context).unique_name(context).to_string(),
            step,
        });
    }
    CanonicalLoopResultV1::Incomplete(
        "the cycle has no supported `i < bound; i := i + positive_constant` header and backedge",
    )
}

fn successor_arguments(
    context: &Context,
    terminator: &OpBox,
    ordinal: usize,
) -> Option<Vec<pliron::value::Value>> {
    if let Some(branch) = terminator.downcast_ref::<BranchArgsOp>() {
        return (ordinal == 0).then(|| branch.arguments(context));
    }
    if let Some(branch) = terminator.downcast_ref::<IndexLessThanBranchArgsOp>() {
        return match ordinal {
            0 => Some(branch.true_arguments(context)),
            1 => Some(branch.false_arguments(context)),
            _ => None,
        };
    }
    if let Some(branch) = terminator.downcast_ref::<IndexEqualBranchArgsOp>() {
        return match ordinal {
            0 => Some(branch.true_arguments(context)),
            1 => Some(branch.false_arguments(context)),
            _ => None,
        };
    }
    if let Some(branch) = terminator.downcast_ref::<AnalysisSplitOp>() {
        return match ordinal {
            0 => Some(branch.first_arguments(context)),
            1 => Some(branch.second_arguments(context)),
            _ => None,
        };
    }
    None
}

fn zero_step_result(
    context: &Context,
    component_members: &HashSet<usize>,
    incoming: &[IncomingEdgeV1],
    bound: pliron::value::Value,
    reason: &'static str,
) -> CanonicalLoopResultV1 {
    let Some(bound) = index_constant(context, bound) else {
        return CanonicalLoopResultV1::Incomplete(
            "the zero-step cycle needs a feasible true-edge witness, but its bound is symbolic",
        );
    };
    let mut saw_predecessor = false;
    let mut saw_unknown = false;
    for edge in incoming {
        if component_members.contains(&edge.source) {
            continue;
        }
        saw_predecessor = true;
        let Some(initial) = edge.initial else {
            saw_unknown = true;
            continue;
        };
        if initial < bound {
            return CanonicalLoopResultV1::Rejected {
                reason,
                counterexample: format!(
                    "the live incoming edge carries i = {initial} and bound = {bound}, so the true edge repeats forever"
                ),
            };
        }
    }
    if saw_predecessor && !saw_unknown {
        CanonicalLoopResultV1::Inactive
    } else {
        CanonicalLoopResultV1::Incomplete(
            "the zero-step cycle has no reconstructed feasible incoming value",
        )
    }
}

fn index_constant(context: &Context, value: pliron::value::Value) -> Option<u64> {
    let definition = value.defining_op()?;
    Operation::get_op_dyn(definition, context)
        .downcast_ref::<IndexConstantOp>()?
        .value(context)
}

fn unsigned_cast_upper_bound(context: &Context, value: pliron::value::Value) -> Option<u64> {
    let definition = value.defining_op()?;
    let operation = Operation::get_op_dyn(definition, context);
    let cast = operation.downcast_ref::<IndexUnsignedCastOp>()?;
    (cast.result(context) == value && cast.verify(context).is_ok())
        .then(|| cast.inclusive_upper_bound(context))
        .flatten()
}

fn reachable_blocks(edges: &[Vec<usize>]) -> Vec<bool> {
    let mut reachable = vec![false; edges.len()];
    if edges.is_empty() {
        return reachable;
    }
    let mut stack = vec![0];
    while let Some(block) = stack.pop() {
        if reachable[block] {
            continue;
        }
        reachable[block] = true;
        stack.extend(edges[block].iter().copied());
    }
    reachable
}

fn strongly_connected_components(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut reverse = vec![Vec::new(); edges.len()];
    for (from, successors) in edges.iter().enumerate() {
        for successor in successors {
            reverse[*successor].push(from);
        }
    }
    let mut visited = vec![false; edges.len()];
    let mut order = Vec::with_capacity(edges.len());
    for root in 0..edges.len() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut stack = vec![(root, 0_usize)];
        while let Some((block, next)) = stack.pop() {
            if let Some(successor) = edges[block].get(next).copied() {
                stack.push((block, next + 1));
                if !visited[successor] {
                    visited[successor] = true;
                    stack.push((successor, 0));
                }
            } else {
                order.push(block);
            }
        }
    }
    visited.fill(false);
    let mut components = Vec::new();
    for root in order.into_iter().rev() {
        if visited[root] {
            continue;
        }
        visited[root] = true;
        let mut component = Vec::new();
        let mut stack = vec![root];
        while let Some(block) = stack.pop() {
            component.push(block);
            for predecessor in &reverse[block] {
                if !visited[*predecessor] {
                    visited[*predecessor] = true;
                    stack.push(*predecessor);
                }
            }
        }
        components.push(component);
    }
    components
}

fn is_cycle(component: &[usize], edges: &[Vec<usize>]) -> bool {
    component.len() > 1
        || component
            .first()
            .is_some_and(|block| edges[*block].contains(block))
}

fn report(finding: PlironProgressFindingV1) -> PlironProgressReportV1 {
    PlironProgressReportV1 {
        findings: vec![finding],
        certificates: Vec::new(),
    }
}

#[cfg(test)]
mod resource_budget_tests {
    use super::*;

    #[test]
    fn work_budget_accepts_the_boundary_and_rejects_one_more_unit() {
        let mut budget = ProgressWorkBudgetV1::default();
        budget.charge(MAX_PLIRON_PROGRESS_WORK_UNITS_V1).unwrap();
        assert_eq!(
            budget.charge(1),
            Err(PlironProgressFindingV1::ResourceLimitExceeded {
                resource: "work units",
                actual: MAX_PLIRON_PROGRESS_WORK_UNITS_V1 + 1,
                limit: MAX_PLIRON_PROGRESS_WORK_UNITS_V1,
            })
        );
    }
}
