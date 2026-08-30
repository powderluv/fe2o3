//! Generic concurrent-effect verification for ranked Pliron memory.
//!
//! Sparse SSA propagation supplies index formulas. A conservative symbolic
//! fast path proves equal full-rank affine maps injective for any launch size.
//! Remaining cases are evaluated over a bounded static launch domain and
//! indexed by logical allocation plus element coordinate. Exact fallback is
//! O(invocations * effects * rank), never pairwise in invocation count.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use dialect_gpu::{AddressSpaceAttr, FenceOp};
use dialect_kernel::{
    AccessKindAttr, AllocationEffectOp, AtomicOrderingAttr, AtomicScopeAttr, IndexConstantOp,
    IndexEqualBranchArgsOp, IndexEqualBranchOp, IndexLessThanBranchArgsOp, IndexLessThanBranchOp,
    InvocationIndexOp, MAX_RANKED_MEMORY_RANK, MemorySpaceAttr, RankedAccessOp, RankedViewOp,
    is_supported_allocation_effect_contract_v1,
};
use pliron::{
    builtin::ops::FuncOp, common_traits::Named, context::Context, operation::Operation,
    value::Value,
};

use crate::pliron_analysis_manager::PlironAnalysisManagerV1;
use crate::pliron_analysis_witness::evaluate_raw_index_at_invocation_v1;
use crate::pliron_invocation_trace::PlironTraceFailureV1;
use crate::pliron_ranked_bounds::run_pliron_ranked_bounds_check_with_analyses_v1;
use crate::pliron_sparse_index::SparseAffineIndexV1;
use crate::{
    KernelCheckPassKindV1, KernelCheckStatusV1, MAX_PRESBURGER_WORK_UNITS_V1,
    PlironPresburgerAnalysisV1, PresburgerCollisionDecisionV1, PresburgerMachineIntSemanticsV1,
    PresburgerMachineRangeDecisionV1, SparseIndexAnalysisV1, SparseIndexFailureV1,
};

pub const MAX_PLIRON_RACE_INVOCATIONS_V1: u64 = 65_536;
pub const MAX_PLIRON_RACE_EFFECT_INSTANCES_V1: usize = 1_048_576;
pub const MAX_PLIRON_RACE_FINDINGS_V1: usize = 4_096;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RankedRaceLocationV1 {
    block: usize,
    operation: usize,
}

impl RankedRaceLocationV1 {
    pub const fn block(self) -> usize {
        self.block
    }

    pub const fn operation(self) -> usize {
        self.operation
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankedRaceWitnessV1 {
    location: RankedRaceLocationV1,
    access: AccessKindAttr,
    invocation: Vec<u64>,
    grid: u64,
    workgroup: Option<u64>,
    subgroup: Option<u64>,
    lane: Option<u64>,
    atomic_scope: Option<AtomicScopeAttr>,
}

impl RankedRaceWitnessV1 {
    pub const fn location(&self) -> RankedRaceLocationV1 {
        self.location
    }

    pub const fn access(&self) -> AccessKindAttr {
        self.access
    }

    pub fn invocation(&self) -> &[u64] {
        &self.invocation
    }

    pub const fn grid(&self) -> u64 {
        self.grid
    }

    pub const fn workgroup(&self) -> Option<u64> {
        self.workgroup
    }

    pub const fn subgroup(&self) -> Option<u64> {
        self.subgroup
    }

    pub const fn lane(&self) -> Option<u64> {
        self.lane
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RankedRaceFindingV1 {
    BoundsPrerequisiteRejected,
    SparseIndexAnalysisFailed {
        detail: String,
    },
    DynamicLaunchExtent {
        dimension: usize,
    },
    LaunchDomainTooLarge {
        invocations: u64,
        limit: u64,
    },
    UnresolvedIndex {
        block: usize,
        operation: usize,
        dimension: usize,
        value: String,
    },
    EffectInstanceLimitExceeded {
        actual: usize,
        limit: usize,
    },
    FindingLimitExceeded {
        actual: usize,
        limit: usize,
    },
    ConflictingEffects {
        view: String,
        indices: Vec<u64>,
        first: RankedRaceWitnessV1,
        second: RankedRaceWitnessV1,
    },
    ExecutionLayoutUnavailable {
        detail: String,
    },
    AllocationContractUnavailable {
        detail: String,
    },
    InsufficientAtomicScope {
        view: String,
        indices: Vec<u64>,
        first: RankedRaceWitnessV1,
        second: RankedRaceWitnessV1,
    },
    HappensBeforeIncomplete {
        view: String,
        detail: String,
    },
}

impl fmt::Display for RankedRaceFindingV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundsPrerequisiteRejected => formatter.write_str(
                "error[FE2O3-RACE-000]: ranked bounds prerequisite rejected before race analysis",
            ),
            Self::SparseIndexAnalysisFailed { detail } => write!(
                formatter,
                "error[FE2O3-RACE-003]: sparse index analysis failed before race analysis: {detail}",
            ),
            Self::DynamicLaunchExtent { dimension } => write!(
                formatter,
                "error[FE2O3-RACE-002]: cannot prove race freedom for dynamic launch dimension {dimension}; help: retain a bounded launch contract or supply a symbolic disjointness proof",
            ),
            Self::LaunchDomainTooLarge { invocations, limit } => write!(
                formatter,
                "error[FE2O3-RACE-003]: static launch has {invocations} invocations, exceeding exact race-analysis limit {limit}",
            ),
            Self::UnresolvedIndex {
                block,
                operation,
                dimension,
                value,
            } => write!(
                formatter,
                "error[FE2O3-RACE-002]: cannot prove race freedom at block {block} op {operation}; access dimension {dimension} has unresolved index {value}; checked structured index markers are currently incomplete because they carry no independently validated success/value contract; help: express the address with explicit affine index operations plus finite no-wrap and extent guards",
            ),
            Self::EffectInstanceLimitExceeded { actual, limit } => write!(
                formatter,
                "error[FE2O3-RACE-003]: concurrent effect instance count {actual} exceeds analysis limit {limit}",
            ),
            Self::FindingLimitExceeded { actual, limit } => write!(
                formatter,
                "error[FE2O3-RACE-003]: race finding count {actual} exceeds analysis limit {limit}",
            ),
            Self::ConflictingEffects {
                view,
                indices,
                first,
                second,
            } => write!(
                formatter,
                "error[FE2O3-RACE-001]: potentially conflicting incompatible {:?}/{:?} effects on {view}{indices:?}; first writer/reader: invocation {:?} at block {} op {}; second writer/reader: invocation {:?} at block {} op {}; failed proof: distinct concurrent invocations do not imply disjoint memory coordinates; help: include an invocation-owned coordinate, use a disjoint view, or use a compatible atomic operation",
                first.access,
                second.access,
                first.invocation,
                first.location.block,
                first.location.operation,
                second.invocation,
                second.location.block,
                second.location.operation,
            ),
            Self::ExecutionLayoutUnavailable { detail } => write!(
                formatter,
                "error[FE2O3-RACE-002]: scoped concurrency analysis is incomplete: {detail}",
            ),
            Self::AllocationContractUnavailable { detail } => write!(
                formatter,
                "error[FE2O3-RACE-002]: allocation alias analysis is incomplete: {detail}",
            ),
            Self::InsufficientAtomicScope {
                view,
                indices,
                first,
                second,
            } => write!(
                formatter,
                "error[FE2O3-RACE-004]: overlapping atomic effects on {view}{indices:?} use scopes {:?}/{:?} that do not cover invocations {:?}/{:?}; failed proof: cross-workgroup overlap requires compatible device-scope atomics",
                first.atomic_scope, second.atomic_scope, first.invocation, second.invocation,
            ),
            Self::HappensBeforeIncomplete { view, detail } => write!(
                formatter,
                "error[FE2O3-RACE-002]: happens-before analysis for conflicting ordinary effects on {view} is incomplete: {detail}",
            ),
        }
    }
}

impl RankedRaceFindingV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::ConflictingEffects { .. } | Self::InsufficientAtomicScope { .. } => {
                KernelCheckStatusV1::Rejected
            }
            Self::BoundsPrerequisiteRejected
            | Self::SparseIndexAnalysisFailed { .. }
            | Self::DynamicLaunchExtent { .. }
            | Self::LaunchDomainTooLarge { .. }
            | Self::UnresolvedIndex { .. }
            | Self::EffectInstanceLimitExceeded { .. }
            | Self::FindingLimitExceeded { .. }
            | Self::ExecutionLayoutUnavailable { .. }
            | Self::AllocationContractUnavailable { .. }
            | Self::HappensBeforeIncomplete { .. } => KernelCheckStatusV1::Incomplete,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RankedRaceReportV1 {
    findings: Vec<RankedRaceFindingV1>,
}

impl RankedRaceReportV1 {
    pub const fn pass(&self) -> KernelCheckPassKindV1 {
        KernelCheckPassKindV1::RaceFreedom
    }

    pub fn status(&self) -> KernelCheckStatusV1 {
        self.findings
            .iter()
            .fold(KernelCheckStatusV1::Clean, |status, finding| {
                status.join(finding.status())
            })
    }

    pub fn findings(&self) -> &[RankedRaceFindingV1] {
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
pub struct RankedRaceCheckErrorV1 {
    report: RankedRaceReportV1,
}

impl RankedRaceCheckErrorV1 {
    pub fn report(&self) -> &RankedRaceReportV1 {
        &self.report
    }
}

impl fmt::Display for RankedRaceCheckErrorV1 {
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

impl std::error::Error for RankedRaceCheckErrorV1 {}

#[derive(Clone)]
struct EffectV1 {
    identity: EffectIdentityV1,
    view_name: String,
    kind: AccessKindAttr,
    location: RankedRaceLocationV1,
    indices: Vec<Value>,
    checked_success: Option<Value>,
    atomic_scope: Option<AtomicScopeAttr>,
    atomic_ordering: Option<AtomicOrderingAttr>,
    noalias_class: u64,
    conservative: bool,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum EffectIdentityV1 {
    View(Value),
    Allocation(u64),
    AllocationSite(RankedRaceLocationV1),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AddressKeyV1 {
    allocation_class: u64,
    indices: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
struct WitnessPairV1 {
    first: Option<RankedRaceWitnessV1>,
    second: Option<RankedRaceWitnessV1>,
}

impl WitnessPairV1 {
    fn different_from(&self, invocation: &[u64]) -> Option<&RankedRaceWitnessV1> {
        self.first
            .as_ref()
            .filter(|witness| witness.invocation != invocation)
            .or_else(|| {
                self.second
                    .as_ref()
                    .filter(|witness| witness.invocation != invocation)
            })
    }

    fn insert(&mut self, witness: RankedRaceWitnessV1) {
        if self.first.is_none() {
            self.first = Some(witness);
        } else if self
            .first
            .as_ref()
            .is_some_and(|first| first.invocation != witness.invocation)
            && self.second.is_none()
        {
            self.second = Some(witness);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct AddressStateV1 {
    reads: WitnessPairV1,
    writes: WitnessPairV1,
    atomic_reads: WitnessPairV1,
    atomic_writes: WitnessPairV1,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ConflictClassV1 {
    identity: EffectIdentityV1,
    first: RankedRaceLocationV1,
    second: RankedRaceLocationV1,
    first_kind: AccessKindAttr,
    second_kind: AccessKindAttr,
}

pub fn run_pliron_ranked_race_check_v1(context: &Context, function: &FuncOp) -> RankedRaceReportV1 {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    if !run_pliron_ranked_bounds_check_with_analyses_v1(context, function, &mut analyses).is_clean()
    {
        return one(RankedRaceFindingV1::BoundsPrerequisiteRejected);
    }
    run_pliron_ranked_race_check_with_analyses_v1(context, function, &mut analyses)
}

pub(crate) fn run_pliron_ranked_race_check_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> RankedRaceReportV1 {
    analyses.prepare_sparse_indices(context, function);
    analyses.prepare_presburger(context, function);
    analyses.prepare_provenance_alias(context, function);
    if let Err(failure) = analyses.sparse_indices() {
        return one(RankedRaceFindingV1::SparseIndexAnalysisFailed {
            detail: sparse_failure(failure),
        });
    }
    analyses.prepare_execution_layout(context, function);
    let sparse = match analyses.sparse_indices() {
        Ok(sparse) => sparse,
        Err(failure) => {
            return one(RankedRaceFindingV1::SparseIndexAnalysisFailed {
                detail: sparse_failure(failure),
            });
        }
    };
    let presburger = match analyses.presburger() {
        Ok(presburger) => presburger,
        Err(failure) => {
            return one(RankedRaceFindingV1::SparseIndexAnalysisFailed {
                detail: sparse_failure(failure),
            });
        }
    };
    let provenance = match analyses.provenance_alias() {
        Ok(provenance) => provenance,
        Err(failure) => {
            return one(RankedRaceFindingV1::AllocationContractUnavailable {
                detail: failure.to_string(),
            });
        }
    };
    if let Err(failure) = provenance.validate_space(MemorySpaceAttr::Global) {
        return one(RankedRaceFindingV1::AllocationContractUnavailable {
            detail: failure.to_string(),
        });
    }
    let inventory = analyses
        .function_inventory_handle()
        .expect("race prerequisites prepare the function inventory");
    let mut effects = Vec::new();
    let mut has_global_fence = false;
    for site in inventory.operations() {
        let block_index = site.block();
        let operation_index = site.operation();
        let operation = Operation::get_op_dyn(site.pointer(), context);
        if operation
            .downcast_ref::<FenceOp>()
            .is_some_and(|fence| fence.address_space(context) == Some(AddressSpaceAttr::Global))
        {
            has_global_fence = true;
        }
        if let Some(effect) = operation.downcast_ref::<AllocationEffectOp>() {
            let Some(kind) = effect.kind(context) else {
                return one(RankedRaceFindingV1::AllocationContractUnavailable {
                    detail: "a whole-allocation effect has no access kind".to_owned(),
                });
            };
            let Some(memory_space) = effect.memory_space(context) else {
                return one(RankedRaceFindingV1::AllocationContractUnavailable {
                    detail: "a whole-allocation effect has no memory space".to_owned(),
                });
            };
            let allocation_origin = effect.allocation_origin(context).unwrap_or(0);
            let noalias_class = effect.noalias_class(context).unwrap_or(0);
            match memory_space {
                MemorySpaceAttr::Global => {}
                MemorySpaceAttr::Workgroup
                    if is_supported_allocation_effect_contract_v1(
                        kind,
                        memory_space,
                        allocation_origin,
                        noalias_class,
                    ) =>
                {
                    // The production source join authenticates the typed
                    // transpose lifecycle; no global race is represented.
                    continue;
                }
                MemorySpaceAttr::Private | MemorySpaceAttr::Workgroup => {
                    return one(RankedRaceFindingV1::AllocationContractUnavailable {
                        detail:
                            "a whole-allocation effect uses an unsupported non-global memory space"
                                .to_owned(),
                    });
                }
            }
            let location = RankedRaceLocationV1 {
                block: block_index,
                operation: operation_index,
            };
            effects.push(EffectV1 {
                identity: if allocation_origin == 0 {
                    EffectIdentityV1::AllocationSite(location)
                } else {
                    EffectIdentityV1::Allocation(allocation_origin)
                },
                view_name: format!("allocation origin {allocation_origin}"),
                kind,
                location,
                indices: vec![],
                checked_success: None,
                atomic_scope: None,
                atomic_ordering: None,
                noalias_class,
                conservative: true,
            });
            continue;
        }
        let Some(access) = operation.downcast_ref::<RankedAccessOp>() else {
            continue;
        };
        let view = access.view(context);
        let Some(definition) = view.defining_op() else {
            return one(RankedRaceFindingV1::UnresolvedIndex {
                block: block_index,
                operation: operation_index,
                dimension: 0,
                value: "view-without-definition".to_owned(),
            });
        };
        let definition = Operation::get_op_dyn(definition, context);
        let Some(view_op) = definition.downcast_ref::<RankedViewOp>() else {
            return one(RankedRaceFindingV1::UnresolvedIndex {
                block: block_index,
                operation: operation_index,
                dimension: 0,
                value: "foreign-view-definition".to_owned(),
            });
        };
        match view_op.memory_space(context) {
            Some(MemorySpaceAttr::Private) => continue,
            // Workgroup effects are checked with barrier epochs by the
            // mandatory workgroup-memory pass that follows this pass.
            Some(MemorySpaceAttr::Workgroup) => continue,
            Some(MemorySpaceAttr::Global) => {}
            None => {
                return one(RankedRaceFindingV1::UnresolvedIndex {
                    block: block_index,
                    operation: operation_index,
                    dimension: 0,
                    value: "view-without-memory-space".to_owned(),
                });
            }
        }
        let Some(kind) = access.kind(context) else {
            return one(RankedRaceFindingV1::UnresolvedIndex {
                block: block_index,
                operation: operation_index,
                dimension: 0,
                value: "access-without-kind".to_owned(),
            });
        };
        effects.push(EffectV1 {
            identity: EffectIdentityV1::View(view),
            view_name: view.unique_name(context).to_string(),
            kind,
            location: RankedRaceLocationV1 {
                block: block_index,
                operation: operation_index,
            },
            indices: access.indices(context),
            checked_success: access.checked_success(context),
            atomic_scope: access.atomic_scope(context),
            atomic_ordering: access.atomic_ordering(context),
            noalias_class: view_op.noalias_class(context).unwrap_or(0),
            conservative: false,
        });
    }

    for effect in &mut effects {
        effect.noalias_class =
            provenance.canonical_class(MemorySpaceAttr::Global, effect.noalias_class);
    }
    let classes_with_writes = effects
        .iter()
        .filter_map(|effect| effect.kind.writes_memory().then_some(effect.noalias_class))
        .collect::<HashSet<_>>();
    // Read-only allocation classes cannot participate in a data race. Keep
    // reads that may alias a write, but do not require unrelated input-only
    // address calculations to be recoverable by the race proof.
    effects.retain(|effect| classes_with_writes.contains(&effect.noalias_class));
    let layout = match analyses.execution_layout() {
        Ok(layout) => layout,
        Err(failure) => {
            return one(RankedRaceFindingV1::ExecutionLayoutUnavailable {
                detail: match failure {
                    PlironTraceFailureV1::InvalidExecutionLayout => {
                        "gpu.execution_layout is malformed or duplicated".to_owned()
                    }
                    _ => format!("execution layout extraction failed: {failure:?}"),
                },
            });
        }
    };
    let launch_extents = if let Some(layout) = layout {
        for dimension in 0..sparse.launch_extents().len().max(3) {
            if let Some(declared) = sparse.declared_launch_extent(dimension) {
                let Some(layout_extent) = layout.global_extents.get(dimension).copied() else {
                    return one(RankedRaceFindingV1::ExecutionLayoutUnavailable {
                        detail: format!(
                            "invocation coordinate axis {dimension} is outside the three-dimensional gpu.execution_layout"
                        ),
                    });
                };
                if declared != 0 && layout_extent != declared {
                    return one(RankedRaceFindingV1::ExecutionLayoutUnavailable {
                        detail: format!(
                            "invocation coordinate axis {dimension} declares extent {declared}, inconsistent with gpu.execution_layout"
                        ),
                    });
                }
            }
        }
        layout.global_extents.to_vec()
    } else if sparse.has_declared_launch_extent() {
        sparse.launch_extents().to_vec()
    } else if effects.is_empty() {
        vec![1]
    } else {
        return one(RankedRaceFindingV1::ExecutionLayoutUnavailable {
            detail: "concurrent memory effects require a declared execution domain even when the kernel does not read an invocation coordinate".to_owned(),
        });
    };

    let invocation_bounds = invocation_upper_bounds_by_block(context, function, &inventory);
    if symbolically_proves_disjoint(
        context,
        function,
        &effects,
        sparse,
        &launch_extents,
        invocation_bounds.as_deref(),
    ) || presburger_proves_no_conflicts(&effects, sparse, presburger, &launch_extents)
    {
        return clean();
    }
    let release_signal_views = effects
        .iter()
        .filter_map(|effect| {
            (effect.kind.is_atomic()
                && effect.kind.writes_memory()
                && matches!(
                    effect.atomic_ordering,
                    Some(
                        AtomicOrderingAttr::Release
                            | AtomicOrderingAttr::AcquireRelease
                            | AtomicOrderingAttr::SequentiallyConsistent
                    )
                )
                && effect
                    .atomic_scope
                    .is_some_and(|scope| scope.rank() >= AtomicScopeAttr::Agent.rank()))
            .then_some(effect.noalias_class)
        })
        .collect::<HashSet<_>>();
    let acquire_signal_views = effects
        .iter()
        .filter_map(|effect| {
            (effect.kind.is_atomic()
                && effect.kind.reads_memory()
                && matches!(
                    effect.atomic_ordering,
                    Some(
                        AtomicOrderingAttr::Acquire
                            | AtomicOrderingAttr::AcquireRelease
                            | AtomicOrderingAttr::SequentiallyConsistent
                    )
                )
                && effect
                    .atomic_scope
                    .is_some_and(|scope| scope.rank() >= AtomicScopeAttr::Agent.rank()))
            .then_some(effect.noalias_class)
        })
        .collect::<HashSet<_>>();
    let atomic_signal_views = release_signal_views
        .intersection(&acquire_signal_views)
        .copied()
        .collect::<HashSet<_>>();
    if let Some(dimension) = launch_extents.iter().position(|extent| *extent == 0) {
        return one(RankedRaceFindingV1::DynamicLaunchExtent { dimension });
    }
    let Some(invocation_count) = launch_extents
        .iter()
        .try_fold(1_u64, |total, extent| total.checked_mul(*extent))
    else {
        return one(RankedRaceFindingV1::LaunchDomainTooLarge {
            invocations: u64::MAX,
            limit: MAX_PLIRON_RACE_INVOCATIONS_V1,
        });
    };
    if invocation_count > MAX_PLIRON_RACE_INVOCATIONS_V1 {
        return one(RankedRaceFindingV1::LaunchDomainTooLarge {
            invocations: invocation_count,
            limit: MAX_PLIRON_RACE_INVOCATIONS_V1,
        });
    }
    if invocation_count <= 1 {
        return clean();
    }
    if let Some(effect) = effects
        .iter()
        .find(|effect| effect.conservative && classes_with_writes.contains(&effect.noalias_class))
    {
        return one(RankedRaceFindingV1::AllocationContractUnavailable {
            detail: format!(
                "whole-allocation read on {} may overlap a writable effect in alias class {} across concurrent invocations",
                effect.view_name, effect.noalias_class
            ),
        });
    }

    let zero_invocation = vec![0; launch_extents.len()];
    let mut raw_evaluation_steps = 0;
    for effect in &effects {
        for (dimension, index) in effect.indices.iter().copied().enumerate() {
            if sparse.fact(index).evaluate(&zero_invocation).is_none()
                && evaluate_raw_index_at_invocation_v1(
                    context,
                    index,
                    &zero_invocation,
                    &mut raw_evaluation_steps,
                )
                .is_none()
            {
                return one(RankedRaceFindingV1::UnresolvedIndex {
                    block: effect.location.block,
                    operation: effect.location.operation,
                    dimension,
                    value: index.unique_name(context).to_string(),
                });
            }
        }
    }

    let mut addresses: HashMap<AddressKeyV1, AddressStateV1> = HashMap::new();
    let mut findings = Vec::new();
    let mut conflict_classes = HashSet::new();
    let mut effect_instances = 0_usize;
    for linear_invocation in 0..invocation_count {
        let invocation = decode_invocation(linear_invocation, &launch_extents);
        for effect in &effects {
            effect_instances = effect_instances.saturating_add(1);
            if effect_instances > MAX_PLIRON_RACE_EFFECT_INSTANCES_V1 {
                return one(RankedRaceFindingV1::EffectInstanceLimitExceeded {
                    actual: effect_instances,
                    limit: MAX_PLIRON_RACE_EFFECT_INSTANCES_V1,
                });
            }
            let Some(indices) = effect
                .indices
                .iter()
                .map(|index| {
                    sparse.fact(*index).evaluate(&invocation).or_else(|| {
                        evaluate_raw_index_at_invocation_v1(
                            context,
                            *index,
                            &invocation,
                            &mut raw_evaluation_steps,
                        )
                    })
                })
                .collect::<Option<Vec<_>>>()
            else {
                let (dimension, value) = effect
                    .indices
                    .iter()
                    .copied()
                    .enumerate()
                    .find(|(_, index)| sparse.fact(*index).evaluate(&invocation).is_none())
                    .expect("failed index evaluation identifies an unresolved index");
                return one(RankedRaceFindingV1::UnresolvedIndex {
                    block: effect.location.block,
                    operation: effect.location.operation,
                    dimension,
                    value: value.unique_name(context).to_string(),
                });
            };
            let key = AddressKeyV1 {
                allocation_class: effect.noalias_class,
                indices,
            };
            let scoped_identity = layout.and_then(|layout| layout.scoped_identity(&invocation));
            let witness = RankedRaceWitnessV1 {
                location: effect.location,
                access: effect.kind,
                invocation: invocation.clone(),
                grid: layout.map_or(0, |layout| layout.grid),
                workgroup: scoped_identity.map(|identity| identity.0),
                subgroup: scoped_identity.map(|identity| identity.1),
                lane: scoped_identity.map(|identity| identity.2),
                atomic_scope: effect.atomic_scope,
            };
            let state = addresses.entry(key.clone()).or_default();
            let conflict = conflicting_witness(state, effect.kind, &witness).cloned();
            if let Some(first) = conflict {
                let class = ConflictClassV1 {
                    identity: effect.identity,
                    first: first.location,
                    second: effect.location,
                    first_kind: first.access,
                    second_kind: effect.kind,
                };
                if conflict_classes.insert(class) {
                    let finding = if first.access.is_atomic() && witness.access.is_atomic() {
                        if layout.is_none() {
                            RankedRaceFindingV1::ExecutionLayoutUnavailable {
                                detail: format!(
                                    "overlapping narrow-scope atomics on {} require retained workgroup identity",
                                    effect.view_name
                                ),
                            }
                        } else {
                            RankedRaceFindingV1::InsufficientAtomicScope {
                                view: effect.view_name.clone(),
                                indices: key.indices,
                                first,
                                second: witness.clone(),
                            }
                        }
                    } else if has_global_fence
                        || atomic_signal_views
                            .iter()
                            .any(|class| *class != effect.noalias_class)
                    {
                        RankedRaceFindingV1::HappensBeforeIncomplete {
                            view: effect.view_name.clone(),
                            detail: if atomic_signal_views
                                .iter()
                                .any(|class| *class != effect.noalias_class)
                            {
                                "release/acquire atomics require an authenticated read-from relation before they can publish ordinary memory across invocations".to_owned()
                            } else {
                                "a non-collective fence alone does not establish a cross-invocation synchronizes-with edge".to_owned()
                            },
                        }
                    } else {
                        RankedRaceFindingV1::ConflictingEffects {
                            view: effect.view_name.clone(),
                            indices: key.indices,
                            first,
                            second: witness.clone(),
                        }
                    };
                    findings.push(finding);
                    if findings.len() > MAX_PLIRON_RACE_FINDINGS_V1 {
                        return one(RankedRaceFindingV1::FindingLimitExceeded {
                            actual: findings.len(),
                            limit: MAX_PLIRON_RACE_FINDINGS_V1,
                        });
                    }
                }
            }
            insert_witness(state, witness);
        }
    }
    RankedRaceReportV1 { findings }
}

pub(crate) fn require_pliron_ranked_race_freedom_with_analyses_v1(
    context: &Context,
    function: &FuncOp,
    analyses: &mut PlironAnalysisManagerV1,
) -> Result<RankedRaceReportV1, RankedRaceCheckErrorV1> {
    let report = run_pliron_ranked_race_check_with_analyses_v1(context, function, analyses);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(RankedRaceCheckErrorV1 { report })
    }
}

pub fn require_pliron_ranked_race_freedom_before_lowering_v1(
    context: &Context,
    function: &FuncOp,
) -> Result<RankedRaceReportV1, RankedRaceCheckErrorV1> {
    let report = run_pliron_ranked_race_check_v1(context, function);
    if report.is_clean() {
        Ok(report)
    } else {
        Err(RankedRaceCheckErrorV1 { report })
    }
}

fn conflicting_witness<'a>(
    state: &'a AddressStateV1,
    access: AccessKindAttr,
    witness: &RankedRaceWitnessV1,
) -> Option<&'a RankedRaceWitnessV1> {
    match access {
        AccessKindAttr::Read => state
            .writes
            .different_from(&witness.invocation)
            .or_else(|| state.atomic_writes.different_from(&witness.invocation)),
        AccessKindAttr::Write => state
            .writes
            .different_from(&witness.invocation)
            .or_else(|| state.reads.different_from(&witness.invocation))
            .or_else(|| state.atomic_reads.different_from(&witness.invocation))
            .or_else(|| state.atomic_writes.different_from(&witness.invocation)),
        AccessKindAttr::AtomicRead => state
            .writes
            .different_from(&witness.invocation)
            .or_else(|| incompatible_atomic(&state.atomic_writes, witness)),
        AccessKindAttr::AtomicWrite | AccessKindAttr::AtomicReadModifyWrite => state
            .writes
            .different_from(&witness.invocation)
            .or_else(|| state.reads.different_from(&witness.invocation))
            .or_else(|| incompatible_atomic(&state.atomic_reads, witness))
            .or_else(|| incompatible_atomic(&state.atomic_writes, witness)),
    }
}

fn incompatible_atomic<'a>(
    state: &'a WitnessPairV1,
    witness: &RankedRaceWitnessV1,
) -> Option<&'a RankedRaceWitnessV1> {
    [state.first.as_ref(), state.second.as_ref()]
        .into_iter()
        .flatten()
        .find(|other| {
            other.invocation != witness.invocation
                && (!atomic_scope_covers_pair(other.atomic_scope, other, witness)
                    || !atomic_scope_covers_pair(witness.atomic_scope, witness, other))
        })
}

fn atomic_scope_covers_pair(
    scope: Option<AtomicScopeAttr>,
    first: &RankedRaceWitnessV1,
    second: &RankedRaceWitnessV1,
) -> bool {
    if first.invocation == second.invocation {
        return true;
    }
    match (first.workgroup, second.workgroup) {
        (Some(first), Some(second)) if first == second => matches!(
            scope,
            Some(
                AtomicScopeAttr::Workgroup
                    | AtomicScopeAttr::Agent
                    | AtomicScopeAttr::Device
                    | AtomicScopeAttr::System
            )
        ),
        _ => matches!(
            scope,
            Some(AtomicScopeAttr::Agent | AtomicScopeAttr::Device | AtomicScopeAttr::System)
        ),
    }
}

fn insert_witness(state: &mut AddressStateV1, witness: RankedRaceWitnessV1) {
    match witness.access {
        AccessKindAttr::Read => state.reads.insert(witness),
        AccessKindAttr::Write => state.writes.insert(witness),
        AccessKindAttr::AtomicRead => state.atomic_reads.insert(witness),
        AccessKindAttr::AtomicWrite | AccessKindAttr::AtomicReadModifyWrite => {
            state.atomic_writes.insert(witness);
        }
    }
}

fn symbolically_proves_disjoint(
    context: &Context,
    function: &FuncOp,
    effects: &[EffectV1],
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
    invocation_bounds: Option<&[[Option<u64>; MAX_RANKED_MEMORY_RANK]]>,
) -> bool {
    let mut by_view: HashMap<u64, Vec<&EffectV1>> = HashMap::new();
    for effect in effects {
        by_view
            .entry(effect.noalias_class)
            .or_default()
            .push(effect);
    }
    for effects in by_view.values() {
        if !effect_pair_inventory_fits_budget(effects.len()) {
            return false;
        }
        for first_index in 0..effects.len() {
            for second_index in first_index..effects.len() {
                let first = effects[first_index];
                let second = effects[second_index];
                if !access_kinds_need_disjoint_coordinates(first.kind, second.kind)
                    || atomics_are_device_compatible(first, second)
                {
                    continue;
                }
                if !effect_pair_symbolically_disjoint(
                    context,
                    function,
                    first,
                    second,
                    sparse,
                    launch_extents,
                    invocation_bounds,
                ) {
                    return false;
                }
            }
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn effect_pair_symbolically_disjoint(
    context: &Context,
    function: &FuncOp,
    first: &EffectV1,
    second: &EffectV1,
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
    invocation_bounds: Option<&[[Option<u64>; MAX_RANKED_MEMORY_RANK]]>,
) -> bool {
    if effect_affine_map_is_injective(first, sparse, launch_extents, invocation_bounds)
        && effect_affine_map_is_injective(second, sparse, launch_extents, invocation_bounds)
        && same_index_formula(&first.indices, &second.indices, sparse)
    {
        return true;
    }
    checked_tiled_pair_is_disjoint(context, function, first, second, sparse, launch_extents)
        || checked_row_striped_pair_is_disjoint(
            context,
            function,
            first,
            second,
            sparse,
            launch_extents,
        )
        || one_dimensional_affine_residues_are_disjoint(
            first,
            second,
            sparse,
            launch_extents,
            invocation_bounds,
        )
}

fn checked_tiled_pair_is_disjoint(
    context: &Context,
    function: &FuncOp,
    first: &EffectV1,
    second: &EffectV1,
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
) -> bool {
    let ([first_index], [second_index]) = (first.indices.as_slice(), second.indices.as_slice())
    else {
        return false;
    };
    if first.checked_success.is_none() || second.checked_success.is_none() {
        return false;
    }
    let first_fact = sparse.fact(*first_index);
    let second_fact = sparse.fact(*second_index);
    let (Some(first), Some(second)) = (
        first_fact.checked_tiled_2d(),
        second_fact.checked_tiled_2d(),
    ) else {
        return false;
    };
    let first_component = sparse.fact(first.component()).constant_value();
    let second_component = sparse.fact(second.component()).constant_value();
    first.geometry() == second.geometry()
        && first.runtime_layout() == second.runtime_layout()
        && checked_runtime_layout_is_uniform(context, function, &first.runtime_layout(), sparse)
        && first.invocation() == second.invocation()
        && first_component.is_some_and(|component| component < first.geometry()[3])
        && second_component.is_some_and(|component| component < second.geometry()[3])
        && checked_invocation_is_injective(first.invocation(), launch_extents)
}

fn checked_row_striped_pair_is_disjoint(
    context: &Context,
    function: &FuncOp,
    first: &EffectV1,
    second: &EffectV1,
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
) -> bool {
    let ([first_index], [second_index]) = (first.indices.as_slice(), second.indices.as_slice())
    else {
        return false;
    };
    if first.checked_success.is_none() || second.checked_success.is_none() {
        return false;
    }
    let first_fact = sparse.fact(*first_index);
    let second_fact = sparse.fact(*second_index);
    let (Some(first), Some(second)) = (
        first_fact.checked_row_striped_2d(),
        second_fact.checked_row_striped_2d(),
    ) else {
        return false;
    };
    let first_component = sparse.fact(first.component()).constant_value();
    let second_component = sparse.fact(second.component()).constant_value();
    first.geometry() == second.geometry()
        && first.runtime_layout() == second.runtime_layout()
        && checked_runtime_layout_is_uniform(context, function, &first.runtime_layout(), sparse)
        && first.invocation() == second.invocation()
        && first_component.is_some_and(|component| component < first.geometry()[1])
        && second_component.is_some_and(|component| component < second.geometry()[1])
        && checked_invocation_is_injective(first.invocation(), launch_extents)
}

fn checked_runtime_layout_is_uniform(
    context: &Context,
    function: &FuncOp,
    layout: &[Value; 3],
    sparse: &SparseIndexAnalysisV1,
) -> bool {
    let entry = function.get_entry_block(context);
    layout.iter().all(|value| {
        (value.defining_op().is_none() && value.defining_block() == Some(entry))
            || sparse.fact(*value).affine().is_some_and(|affine| {
                affine
                    .coefficients()
                    .iter()
                    .all(|coefficient| *coefficient == 0)
            })
    })
}

fn checked_invocation_is_injective(
    invocation: &SparseAffineIndexV1,
    launch_extents: &[u64],
) -> bool {
    let facts = [invocation.clone()];
    affine_facts_are_injective(&facts, launch_extents)
        || affine_facts_contain_unit_coordinate_embedding(&facts, launch_extents)
}

/// Uses exact bounded relation images to discharge affine/remainder effect
/// families that the matrix-rank fast path cannot prove. This query only
/// returns true when every potentially conflicting pair has an empty
/// cross-invocation intersection. Unsupported facts and exhausted budgets fall
/// through to the existing exact trace path.
fn presburger_proves_no_conflicts(
    effects: &[EffectV1],
    sparse: &SparseIndexAnalysisV1,
    presburger: &PlironPresburgerAnalysisV1,
    launch_extents: &[u64],
) -> bool {
    let Some(invocations) = launch_extents.iter().try_fold(1_u128, |count, extent| {
        count.checked_mul(u128::from(*extent))
    }) else {
        return false;
    };
    // The existing address-indexed trace is O(invocations * effects) and is
    // preferable inside its admitted domain. Presburger map intersection is
    // reserved for domains that trace intentionally refuses.
    if invocations <= u128::from(MAX_PLIRON_RACE_INVOCATIONS_V1) {
        return false;
    }
    if !effect_pair_inventory_fits_budget(effects.len()) {
        return false;
    }
    let relevant_pairs = (0..effects.len())
        .flat_map(|first| (first..effects.len()).map(move |second| (first, second)))
        .filter(|(first, second)| {
            let first = &effects[*first];
            let second = &effects[*second];
            first.noalias_class == second.noalias_class
                && access_kinds_need_disjoint_coordinates(first.kind, second.kind)
                && !atomics_are_device_compatible(first, second)
        })
        .count();
    let estimated_work = invocations
        .checked_mul((launch_extents.len() as u128).saturating_add(1))
        .and_then(|work| work.checked_mul(2))
        .and_then(|work| work.checked_mul(relevant_pairs as u128));
    if estimated_work.is_none_or(|work| work > MAX_PRESBURGER_WORK_UNITS_V1 as u128) {
        return false;
    }
    for first_index in 0..effects.len() {
        for second_index in first_index..effects.len() {
            let first = &effects[first_index];
            let second = &effects[second_index];
            if first.noalias_class != second.noalias_class
                || !access_kinds_need_disjoint_coordinates(first.kind, second.kind)
                || atomics_are_device_compatible(first, second)
            {
                continue;
            }
            let first_facts = first
                .indices
                .iter()
                .map(|index| sparse.fact(*index).clone())
                .collect::<Vec<_>>();
            let second_facts = second
                .indices
                .iter()
                .map(|index| sparse.fact(*index).clone())
                .collect::<Vec<_>>();
            let (Ok(first_map), Ok(second_map)) = (
                presburger.map_for_facts_over_extents(&first_facts, launch_extents),
                presburger.map_for_facts_over_extents(&second_facts, launch_extents),
            ) else {
                return false;
            };
            if first_map.find_machine_overflow(PresburgerMachineIntSemanticsV1::unsigned_64())
                != PresburgerMachineRangeDecisionV1::Proved
                || second_map.find_machine_overflow(PresburgerMachineIntSemanticsV1::unsigned_64())
                    != PresburgerMachineRangeDecisionV1::Proved
            {
                return false;
            }
            if first_map.find_cross_collision(&second_map, true)
                != PresburgerCollisionDecisionV1::Proved
            {
                return false;
            }
        }
    }
    true
}

fn effect_pair_inventory_fits_budget(effect_count: usize) -> bool {
    let effect_count = effect_count as u128;
    effect_count
        .checked_add(1)
        .and_then(|next| effect_count.checked_mul(next))
        .map(|ordered| ordered / 2)
        .is_some_and(|pairs| pairs <= MAX_PRESBURGER_WORK_UNITS_V1 as u128)
}

fn access_kinds_need_disjoint_coordinates(first: AccessKindAttr, second: AccessKindAttr) -> bool {
    first.writes_memory() || second.writes_memory()
}

fn atomics_are_device_compatible(first: &EffectV1, second: &EffectV1) -> bool {
    first.kind.is_atomic()
        && second.kind.is_atomic()
        && [first.atomic_scope, second.atomic_scope]
            .into_iter()
            .all(|scope| {
                matches!(
                    scope,
                    Some(
                        AtomicScopeAttr::Agent | AtomicScopeAttr::Device | AtomicScopeAttr::System
                    )
                )
            })
}

fn affine_facts_contain_unit_coordinate_embedding(
    facts: &[SparseAffineIndexV1],
    launch_extents: &[u64],
) -> bool {
    let active_dimensions = launch_extents
        .iter()
        .enumerate()
        .filter_map(|(dimension, extent)| (*extent != 1).then_some(dimension))
        .collect::<Vec<_>>();
    active_dimensions.iter().all(|embedded_dimension| {
        facts.iter().any(|affine| {
            affine.constant_term() == 0
                && affine.coefficients().iter().copied().enumerate().all(
                    |(dimension, coefficient)| match launch_extents.get(dimension) {
                        None => coefficient == 0,
                        Some(1) => true,
                        Some(_) => coefficient == u64::from(dimension == *embedded_dimension),
                    },
                )
                && active_dimensions.iter().all(|dimension| {
                    affine.coefficients().get(*dimension).copied()
                        == Some(u64::from(dimension == embedded_dimension))
                })
        })
    })
}

fn same_index_formula(first: &[Value], second: &[Value], sparse: &SparseIndexAnalysisV1) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(first, second)| sparse.fact(*first) == sparse.fact(*second))
}

fn one_dimensional_affine_residues_are_disjoint(
    first: &EffectV1,
    second: &EffectV1,
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
    invocation_bounds: Option<&[[Option<u64>; MAX_RANKED_MEMORY_RANK]]>,
) -> bool {
    let ([first_index], [second_index]) = (first.indices.as_slice(), second.indices.as_slice())
    else {
        return false;
    };
    let first_fact = sparse.fact(*first_index);
    let second_fact = sparse.fact(*second_index);
    let (Some(first_affine), Some(second_affine)) = (first_fact.affine(), second_fact.affine())
    else {
        return false;
    };
    if first_affine.coefficients() != second_affine.coefficients() {
        return false;
    }
    let mut active = launch_extents
        .iter()
        .enumerate()
        .filter_map(|(dimension, extent)| (*extent != 1).then_some(dimension));
    let Some(dimension) = active.next() else {
        return false;
    };
    if active.next().is_some() {
        return false;
    }
    let Some(stride) = first_affine.coefficients().get(dimension).copied() else {
        return false;
    };
    if stride == 0
        || first_affine
            .coefficients()
            .iter()
            .enumerate()
            .any(|(candidate, coefficient)| candidate != dimension && *coefficient != 0)
    {
        return false;
    }
    let first_extents = effective_launch_extents(first, launch_extents, invocation_bounds);
    let second_extents = effective_launch_extents(second, launch_extents, invocation_bounds);
    affine_is_total_over_launch(first_affine, &first_extents)
        && affine_is_total_over_launch(second_affine, &second_extents)
        && first_affine.constant_term() % stride != second_affine.constant_term() % stride
}

fn affine_map_is_injective(
    indices: &[Value],
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
) -> bool {
    let facts = indices
        .iter()
        .map(|index| sparse.fact(*index).affine().cloned())
        .collect::<Option<Vec<_>>>();
    facts.is_some_and(|facts| {
        affine_facts_are_injective(&facts, launch_extents)
            || affine_facts_contain_unit_coordinate_embedding(&facts, launch_extents)
    })
}

fn effect_affine_map_is_injective(
    effect: &EffectV1,
    sparse: &SparseIndexAnalysisV1,
    launch_extents: &[u64],
    invocation_bounds: Option<&[[Option<u64>; MAX_RANKED_MEMORY_RANK]]>,
) -> bool {
    let effective_extents = effective_launch_extents(effect, launch_extents, invocation_bounds);
    affine_map_is_injective(&effect.indices, sparse, &effective_extents)
}

fn effective_launch_extents(
    effect: &EffectV1,
    launch_extents: &[u64],
    invocation_bounds: Option<&[[Option<u64>; MAX_RANKED_MEMORY_RANK]]>,
) -> Vec<u64> {
    let mut effective_extents = launch_extents.to_vec();
    if let Some(bounds) = invocation_bounds.and_then(|bounds| bounds.get(effect.location.block)) {
        for (dimension, extent) in effective_extents.iter_mut().enumerate() {
            if *extent == 0
                && let Some(bound) = bounds.get(dimension).copied().flatten()
            {
                *extent = bound;
            }
        }
    }
    effective_extents
}

fn invocation_upper_bounds_by_block(
    context: &Context,
    function: &FuncOp,
    inventory: &crate::pliron_function_inventory::BoundedPlironFunctionInventoryV1,
) -> Option<Vec<[Option<u64>; MAX_RANKED_MEMORY_RANK]>> {
    let blocks = inventory.blocks();
    let indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (*block, index))
        .collect::<HashMap<_, _>>();
    let entry = *indices.get(&function.get_entry_block(context))?;
    let empty: [Option<u64>; MAX_RANKED_MEMORY_RANK] = [None; MAX_RANKED_MEMORY_RANK];
    let mut inputs = vec![None; blocks.len()];
    inputs[entry] = Some(empty);
    let mut worklist = VecDeque::from([entry]);
    let mut work = 0_usize;

    while let Some(block_index) = worklist.pop_front() {
        work = work.checked_add(1)?;
        if work > MAX_PLIRON_RACE_EFFECT_INSTANCES_V1 {
            return None;
        }
        let source = inputs[block_index]?;
        let terminator = blocks[block_index].deref(context).get_terminator(context)?;
        let operation = Operation::get_op_dyn(terminator, context);
        let guard = invocation_upper_bound_guard(operation.as_ref(), context);
        let raw = terminator.deref(context);
        for (successor_index, successor) in raw.successors().enumerate() {
            let target = *indices.get(&successor)?;
            if target == entry {
                continue;
            }
            let mut candidate = source;
            if successor_index == 0
                && let Some((dimension, bound)) = guard
            {
                let slot = candidate.get_mut(dimension)?;
                *slot = Some(slot.map_or(bound, |current| current.min(bound)));
            }
            let merged = match inputs[target] {
                None => candidate,
                Some(current) => std::array::from_fn(|dimension| {
                    match (current[dimension], candidate[dimension]) {
                        (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
                        _ => None,
                    }
                }),
            };
            if inputs[target] != Some(merged) {
                inputs[target] = Some(merged);
                worklist.push_back(target);
            }
        }
    }

    Some(
        inputs
            .into_iter()
            .map(|bounds| bounds.unwrap_or(empty))
            .collect(),
    )
}

fn invocation_upper_bound_guard(
    operation: &dyn pliron::op::Op,
    context: &Context,
) -> Option<(usize, u64)> {
    let less_than = operation
        .downcast_ref::<IndexLessThanBranchOp>()
        .map(|branch| (branch.lhs(context), branch.rhs(context)))
        .or_else(|| {
            operation
                .downcast_ref::<IndexLessThanBranchArgsOp>()
                .map(|branch| (branch.lhs(context), branch.rhs(context)))
        });
    if let Some((lhs, rhs)) = less_than {
        return Some((
            invocation_dimension(lhs, context)?,
            index_constant(rhs, context)?,
        ));
    }
    let equal = operation
        .downcast_ref::<IndexEqualBranchOp>()
        .map(|branch| (branch.lhs(context), branch.rhs(context)))
        .or_else(|| {
            operation
                .downcast_ref::<IndexEqualBranchArgsOp>()
                .map(|branch| (branch.lhs(context), branch.rhs(context)))
        })?;
    let dimension = invocation_dimension(equal.0, context)
        .filter(|_| index_constant(equal.1, context) == Some(0))
        .or_else(|| {
            invocation_dimension(equal.1, context)
                .filter(|_| index_constant(equal.0, context) == Some(0))
        })?;
    Some((dimension, 1))
}

fn invocation_dimension(value: Value, context: &Context) -> Option<usize> {
    let operation = Operation::get_op_dyn(value.defining_op()?, context);
    usize::try_from(
        operation
            .downcast_ref::<InvocationIndexOp>()?
            .dimension(context)?,
    )
    .ok()
    .filter(|dimension| *dimension < MAX_RANKED_MEMORY_RANK)
}

fn index_constant(value: Value, context: &Context) -> Option<u64> {
    let operation = Operation::get_op_dyn(value.defining_op()?, context);
    operation.downcast_ref::<IndexConstantOp>()?.value(context)
}

fn affine_facts_are_injective(facts: &[SparseAffineIndexV1], launch_extents: &[u64]) -> bool {
    if !facts
        .iter()
        .all(|affine| affine_is_total_over_launch(affine, launch_extents))
    {
        return false;
    }
    let active_dimensions = launch_extents
        .iter()
        .enumerate()
        .filter_map(|(dimension, extent)| (*extent != 1).then_some(dimension))
        .collect::<Vec<_>>();
    if active_dimensions.is_empty() {
        return true;
    }
    let matrix = facts
        .iter()
        .map(|affine| {
            active_dimensions
                .iter()
                .map(|dimension| affine.coefficients()[*dimension])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    modular_rank(matrix) == active_dimensions.len()
}

fn affine_is_total_over_launch(affine: &SparseAffineIndexV1, launch_extents: &[u64]) -> bool {
    let mut maximum = affine.constant_term();
    for (dimension, coefficient) in affine.coefficients().iter().copied().enumerate() {
        if coefficient == 0 {
            continue;
        }
        let Some(maximum_coordinate) = launch_extents
            .get(dimension)
            .copied()
            .and_then(|extent| extent.checked_sub(1))
        else {
            return false;
        };
        let Some(contribution) = coefficient.checked_mul(maximum_coordinate) else {
            return false;
        };
        let Some(next) = maximum.checked_add(contribution) else {
            return false;
        };
        maximum = next;
    }
    true
}

// Full rank modulo a prime implies full rank over the integers. A rank loss
// modulo this prime is treated as unknown and falls back to exact analysis.
fn modular_rank(mut matrix: Vec<Vec<u64>>) -> usize {
    const PRIME: u64 = (1_u64 << 61) - 1;
    let row_count = matrix.len();
    let column_count = matrix.first().map_or(0, Vec::len);
    for row in &mut matrix {
        for value in row {
            *value %= PRIME;
        }
    }
    let mut rank = 0_usize;
    for column in 0..column_count {
        let Some(pivot) = (rank..row_count).find(|row| matrix[*row][column] != 0) else {
            continue;
        };
        matrix.swap(rank, pivot);
        let inverse = modular_power(matrix[rank][column], PRIME - 2, PRIME);
        for value in &mut matrix[rank][column..column_count] {
            *value = modular_multiply(*value, inverse, PRIME);
        }
        let pivot_row = matrix[rank].clone();
        for (row_index, row) in matrix.iter_mut().enumerate() {
            if row_index == rank || row[column] == 0 {
                continue;
            }
            let factor = row[column];
            for (value, pivot) in row[column..column_count]
                .iter_mut()
                .zip(&pivot_row[column..column_count])
            {
                let product = modular_multiply(factor, *pivot, PRIME);
                *value = (*value + PRIME - product) % PRIME;
            }
        }
        rank += 1;
        if rank == row_count {
            break;
        }
    }
    rank
}

fn modular_power(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1_u64;
    while exponent != 0 {
        if exponent & 1 != 0 {
            result = modular_multiply(result, base, modulus);
        }
        base = modular_multiply(base, base, modulus);
        exponent >>= 1;
    }
    result
}

fn modular_multiply(lhs: u64, rhs: u64, modulus: u64) -> u64 {
    ((u128::from(lhs) * u128::from(rhs)) % u128::from(modulus)) as u64
}

fn decode_invocation(mut linear: u64, extents: &[u64]) -> Vec<u64> {
    let mut invocation = Vec::with_capacity(extents.len());
    for extent in extents {
        invocation.push(linear % extent);
        linear /= extent;
    }
    invocation
}

fn sparse_failure(failure: SparseIndexFailureV1) -> String {
    match failure {
        SparseIndexFailureV1::ResourceLimit {
            resource,
            limit,
            actual,
        } => format!("{resource} count {actual} exceeds {limit}"),
        SparseIndexFailureV1::InconsistentLaunchExtent {
            dimension,
            first,
            second,
        } => format!(
            "invocation dimension {dimension} has inconsistent launch extents {first} and {second}"
        ),
        SparseIndexFailureV1::MalformedControlFlow { detail } => detail.to_owned(),
    }
}

fn one(finding: RankedRaceFindingV1) -> RankedRaceReportV1 {
    RankedRaceReportV1 {
        findings: vec![finding],
    }
}

fn clean() -> RankedRaceReportV1 {
    RankedRaceReportV1 {
        findings: Vec::new(),
    }
}

#[cfg(test)]
mod status_tests {
    use super::*;

    fn witness(
        access: AccessKindAttr,
        atomic_scope: Option<AtomicScopeAttr>,
    ) -> RankedRaceWitnessV1 {
        RankedRaceWitnessV1 {
            location: RankedRaceLocationV1 {
                block: 0,
                operation: 0,
            },
            access,
            invocation: vec![0],
            grid: 0,
            workgroup: Some(0),
            subgroup: Some(0),
            lane: Some(0),
            atomic_scope,
        }
    }

    fn conflict() -> RankedRaceFindingV1 {
        RankedRaceFindingV1::ConflictingEffects {
            view: "v0".to_owned(),
            indices: vec![0],
            first: witness(AccessKindAttr::Write, None),
            second: witness(AccessKindAttr::Read, None),
        }
    }

    #[test]
    fn every_race_finding_has_the_shared_status() {
        let incomplete = [
            RankedRaceFindingV1::BoundsPrerequisiteRejected,
            RankedRaceFindingV1::SparseIndexAnalysisFailed {
                detail: "unresolved".to_owned(),
            },
            RankedRaceFindingV1::DynamicLaunchExtent { dimension: 0 },
            RankedRaceFindingV1::LaunchDomainTooLarge {
                invocations: 2,
                limit: 1,
            },
            RankedRaceFindingV1::UnresolvedIndex {
                block: 0,
                operation: 0,
                dimension: 0,
                value: "i".to_owned(),
            },
            RankedRaceFindingV1::EffectInstanceLimitExceeded {
                actual: 2,
                limit: 1,
            },
            RankedRaceFindingV1::FindingLimitExceeded {
                actual: 2,
                limit: 1,
            },
            RankedRaceFindingV1::ExecutionLayoutUnavailable {
                detail: "missing".to_owned(),
            },
            RankedRaceFindingV1::AllocationContractUnavailable {
                detail: "missing".to_owned(),
            },
            RankedRaceFindingV1::HappensBeforeIncomplete {
                view: "v0".to_owned(),
                detail: "missing".to_owned(),
            },
        ];
        for finding in incomplete {
            assert_eq!(finding.status(), KernelCheckStatusV1::Incomplete);
        }

        let rejected = [
            conflict(),
            RankedRaceFindingV1::InsufficientAtomicScope {
                view: "v0".to_owned(),
                indices: vec![0],
                first: witness(
                    AccessKindAttr::AtomicWrite,
                    Some(AtomicScopeAttr::Workgroup),
                ),
                second: witness(AccessKindAttr::AtomicRead, Some(AtomicScopeAttr::Workgroup)),
            },
        ];
        for finding in rejected {
            assert_eq!(finding.status(), KernelCheckStatusV1::Rejected);
        }
    }

    #[test]
    fn rejected_race_finding_dominates_an_incomplete_finding() {
        let report = RankedRaceReportV1 {
            findings: vec![RankedRaceFindingV1::BoundsPrerequisiteRejected, conflict()],
        };
        assert_eq!(report.status(), KernelCheckStatusV1::Rejected);
        assert!(!report.is_clean());
        assert_eq!(clean().status(), KernelCheckStatusV1::Clean);
    }

    #[test]
    fn effect_pair_inventory_is_charged_before_enumeration() {
        assert!(effect_pair_inventory_fits_budget(1_447));
        assert!(!effect_pair_inventory_fits_budget(1_448));
        assert!(!effect_pair_inventory_fits_budget(usize::MAX));
    }
}
