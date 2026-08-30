//! Independently replayed witnesses for sealed production analysis reports.
//!
//! The envelope in this module is issued only from the compiler-owned report
//! custody session. It binds an exact PLIRON subject and pass checkpoint to a
//! typed witness, then re-derives the supported obligations from live IR. A
//! digest, a report commitment, or a caller-authored success bit is never
//! accepted as evidence.
//!
//! V1 completes only the explicitly documented ranked-bounds fragment. Other
//! passes and unsupported bounds forms remain `Incomplete`. A complete replay
//! validates this analysis report at this checkpoint; it grants no compiler
//! refinement, lowering, artifact, publication, or launch authority.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
};

use dialect_gpu::ExecutionLayoutOp;
use dialect_kernel::{
    DYNAMIC_EXTENT, IndexBinaryKindAttr, IndexBinaryOp, IndexConstantOp, InvocationIndexOp,
    RankedAccessOp, ranked_view_type,
};
use fe2o3_pliron_owner_core::{ContextIdentity, require_context_identity};
use pliron::{
    builtin::ops::FuncOp,
    common_traits::Verify,
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
    r#type::TypedHandle,
    value::Value,
};

use crate::pliron_analysis_manager::PlironAnalysisManagerV1;
use crate::{
    CapturedProductionAnalysisReportV1, KernelCheckStatusV1, PresburgerMapV1,
    ProductionAnalysisCheckpointV1, ProductionAnalysisConfigurationV1,
    ProductionAnalysisImplementationV1, ProductionAnalysisWitnessGapV1, RankedBoundsReportV1,
    SparseIndexFactV1, witness_gap,
};

const MAX_BOUNDS_WITNESS_INVOCATIONS_V1: u64 = 65_536;
const MAX_BOUNDS_WITNESS_EVALUATION_STEPS_V1: usize = 1_048_576;

/// Checker implementation recorded in a witness envelope.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProductionAnalysisWitnessCheckerV1 {
    BoundsExhaustiveRawIrReplayV1,
    UnsupportedV1,
}

/// Exact supported-fragment result of one independent replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionAnalysisWitnessCoverageV1 {
    Complete {
        obligation_count: usize,
    },
    Incomplete {
        gap: ProductionAnalysisWitnessGapV1,
        reason: String,
    },
}

impl ProductionAnalysisWitnessCoverageV1 {
    pub const fn status(&self) -> KernelCheckStatusV1 {
        match self {
            Self::Complete { .. } => KernelCheckStatusV1::Clean,
            Self::Incomplete { .. } => KernelCheckStatusV1::Incomplete,
        }
    }

    pub const fn is_complete(&self) -> bool {
        matches!(self, Self::Complete { .. })
    }

    pub const fn obligation_count(&self) -> usize {
        match self {
            Self::Complete { obligation_count } => *obligation_count,
            Self::Incomplete { .. } => 0,
        }
    }

    pub const fn gap(&self) -> Option<ProductionAnalysisWitnessGapV1> {
        match self {
            Self::Complete { .. } => None,
            Self::Incomplete { gap, .. } => Some(*gap),
        }
    }

    pub fn incomplete_reason(&self) -> Option<&str> {
        match self {
            Self::Complete { .. } => None,
            Self::Incomplete { reason, .. } => Some(reason),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BoundsPresburgerObligationV1 {
    block: usize,
    operation: usize,
    dimension: usize,
    extent: u64,
    checked_invocations: u64,
    normalized_map: PresburgerMapV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BoundsPresburgerWitnessV1 {
    obligations: Vec<BoundsPresburgerObligationV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProductionAnalysisWitnessPayloadV1 {
    Bounds(BoundsPresburgerWitnessV1),
    Incomplete {
        gap: ProductionAnalysisWitnessGapV1,
        reason: String,
    },
}

/// Compiler-issued, subject-bound witness envelope for one sealed report.
///
/// All authority-bearing identity fields and the payload are private. Public
/// consumers can inspect replay coverage, but cannot construct or modify an
/// envelope or convert it into another compiler capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionAnalysisWitnessEnvelopeV1 {
    context_identity: Option<ContextIdentity>,
    function: Ptr<Operation>,
    checkpoint: ProductionAnalysisCheckpointV1,
    implementation: ProductionAnalysisImplementationV1,
    configuration: ProductionAnalysisConfigurationV1,
    report: CapturedProductionAnalysisReportV1,
    checker: ProductionAnalysisWitnessCheckerV1,
    payload: ProductionAnalysisWitnessPayloadV1,
    coverage: ProductionAnalysisWitnessCoverageV1,
}

impl ProductionAnalysisWitnessEnvelopeV1 {
    pub const fn checkpoint(&self) -> ProductionAnalysisCheckpointV1 {
        self.checkpoint
    }

    pub const fn implementation(&self) -> ProductionAnalysisImplementationV1 {
        self.implementation
    }

    pub const fn configuration(&self) -> &ProductionAnalysisConfigurationV1 {
        &self.configuration
    }

    pub const fn checker(&self) -> ProductionAnalysisWitnessCheckerV1 {
        self.checker
    }

    pub const fn coverage(&self) -> &ProductionAnalysisWitnessCoverageV1 {
        &self.coverage
    }

    pub const fn status(&self) -> KernelCheckStatusV1 {
        self.coverage.status()
    }

    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    pub const fn grants_lowering_or_launch_authority(&self) -> bool {
        false
    }
}

/// Integrity failures from replaying an issued witness envelope.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionAnalysisWitnessValidationErrorV1 {
    SubjectMismatch,
    BindingMismatch {
        component: &'static str,
    },
    MutationEpochUnavailable,
    MutationEpochMismatch {
        expected: u64,
        observed: u64,
    },
    ReportMismatch,
    PayloadMismatch,
    BoundsCounterexample {
        block: usize,
        operation: usize,
        dimension: usize,
        invocation: Vec<u64>,
        index: u64,
        extent: u64,
    },
    BoundsMachineOverflow {
        block: usize,
        operation: usize,
        dimension: usize,
        invocation: Vec<u64>,
        operator: &'static str,
    },
}

impl ProductionAnalysisWitnessValidationErrorV1 {
    pub const fn code(&self) -> &'static str {
        "FE2O3-PRESERVE-045"
    }
}

impl fmt::Display for ProductionAnalysisWitnessValidationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "error[{}]: ", self.code())?;
        match self {
            Self::SubjectMismatch => formatter.write_str(
                "analysis witness belongs to a different PLIRON context or function",
            ),
            Self::BindingMismatch { component } => write!(
                formatter,
                "analysis witness {component} differs from its sealed report checkpoint",
            ),
            Self::MutationEpochUnavailable => formatter.write_str(
                "the PLIRON mutation-attempt epoch is unavailable while replaying an analysis witness",
            ),
            Self::MutationEpochMismatch { expected, observed } => write!(
                formatter,
                "analysis witness checkpoint mutation epoch changed from {expected} to {observed}",
            ),
            Self::ReportMismatch => formatter.write_str(
                "analysis witness report payload or status differs from the sealed report",
            ),
            Self::PayloadMismatch => formatter.write_str(
                "analysis witness omitted, substituted, reordered, or forged a live-IR obligation",
            ),
            Self::BoundsCounterexample {
                block,
                operation,
                dimension,
                invocation,
                index,
                extent,
            } => write!(
                formatter,
                "bounds witness replay found a counterexample at block {block} op {operation} dimension {dimension}: invocation {invocation:?} evaluates to {index}, outside extent {extent}",
            ),
            Self::BoundsMachineOverflow {
                block,
                operation,
                dimension,
                invocation,
                operator,
            } => write!(
                formatter,
                "bounds witness replay found checked unsigned {operator} overflow at block {block} op {operation} dimension {dimension} for invocation {invocation:?}",
            ),
        }
    }
}

impl std::error::Error for ProductionAnalysisWitnessValidationErrorV1 {}

enum SupportedWitnessBuildV1<T> {
    Complete(T),
    Incomplete(String),
}

fn current_mutation_epoch(
    context: &Context,
) -> Result<u64, ProductionAnalysisWitnessValidationErrorV1> {
    context
        .ir_mutation_attempt_epoch()
        .map(|epoch| epoch.value())
        .map_err(|_| ProductionAnalysisWitnessValidationErrorV1::MutationEpochUnavailable)
}

pub(crate) fn issue_and_validate_production_analysis_witness_v1(
    context: &Context,
    function: &FuncOp,
    checkpoint: ProductionAnalysisCheckpointV1,
    implementation: ProductionAnalysisImplementationV1,
    configuration: ProductionAnalysisConfigurationV1,
    report: CapturedProductionAnalysisReportV1,
) -> Result<ProductionAnalysisWitnessEnvelopeV1, ProductionAnalysisWitnessValidationErrorV1> {
    let observed_epoch = current_mutation_epoch(context)?;
    if observed_epoch != checkpoint.mutation_epoch() {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::MutationEpochMismatch {
                expected: checkpoint.mutation_epoch(),
                observed: observed_epoch,
            },
        );
    }

    let context_identity = require_context_identity(context).ok();
    let pass = report.pass();
    let (checker, payload, coverage) = match (&report, context_identity) {
        (CapturedProductionAnalysisReportV1::Bounds(_), None) => {
            let gap = witness_gap(pass);
            let reason = "bounds replay cannot complete because this PLIRON context has no compiler-owned ContextIdentity".to_owned();
            (
                ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1,
                ProductionAnalysisWitnessPayloadV1::Incomplete {
                    gap,
                    reason: reason.clone(),
                },
                ProductionAnalysisWitnessCoverageV1::Incomplete { gap, reason },
            )
        }
        (CapturedProductionAnalysisReportV1::Bounds(bounds), Some(_)) => {
            match build_bounds_presburger_witness(context, function, bounds)? {
                SupportedWitnessBuildV1::Complete(witness) => {
                    let obligation_count = witness.obligations.len();
                    (
                        ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1,
                        ProductionAnalysisWitnessPayloadV1::Bounds(witness),
                        ProductionAnalysisWitnessCoverageV1::Complete { obligation_count },
                    )
                }
                SupportedWitnessBuildV1::Incomplete(reason) => {
                    let gap = witness_gap(pass);
                    (
                        ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1,
                        ProductionAnalysisWitnessPayloadV1::Incomplete {
                            gap,
                            reason: reason.clone(),
                        },
                        ProductionAnalysisWitnessCoverageV1::Incomplete { gap, reason },
                    )
                }
            }
        }
        _ => {
            let gap = witness_gap(pass);
            let reason = format!(
                "{} witness replay is not implemented in V1; required evidence: {}",
                pass.name(),
                gap.required_evidence()
            );
            (
                ProductionAnalysisWitnessCheckerV1::UnsupportedV1,
                ProductionAnalysisWitnessPayloadV1::Incomplete {
                    gap,
                    reason: reason.clone(),
                },
                ProductionAnalysisWitnessCoverageV1::Incomplete { gap, reason },
            )
        }
    };

    let envelope = ProductionAnalysisWitnessEnvelopeV1 {
        context_identity,
        function: function.get_operation(),
        checkpoint,
        implementation,
        configuration,
        report,
        checker,
        payload,
        coverage,
    };
    validate_production_analysis_witness_v1(
        context,
        function,
        checkpoint,
        implementation,
        envelope.configuration(),
        &envelope.report,
        &envelope,
    )?;
    Ok(envelope)
}

fn validate_production_analysis_witness_v1(
    context: &Context,
    function: &FuncOp,
    checkpoint: ProductionAnalysisCheckpointV1,
    implementation: ProductionAnalysisImplementationV1,
    configuration: &ProductionAnalysisConfigurationV1,
    report: &CapturedProductionAnalysisReportV1,
    envelope: &ProductionAnalysisWitnessEnvelopeV1,
) -> Result<(), ProductionAnalysisWitnessValidationErrorV1> {
    if envelope.context_identity != require_context_identity(context).ok()
        || envelope.function != function.get_operation()
    {
        return Err(ProductionAnalysisWitnessValidationErrorV1::SubjectMismatch);
    }
    if envelope.checkpoint != checkpoint {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::BindingMismatch {
                component: "checkpoint",
            },
        );
    }
    if envelope.implementation != implementation {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::BindingMismatch {
                component: "implementation",
            },
        );
    }
    if envelope.configuration != *configuration {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::BindingMismatch {
                component: "configuration",
            },
        );
    }
    if envelope.report != *report
        || envelope.report.pass() != checkpoint.pass()
        || envelope.report.status() != report.status()
    {
        return Err(ProductionAnalysisWitnessValidationErrorV1::ReportMismatch);
    }
    let before = current_mutation_epoch(context)?;
    if before != checkpoint.mutation_epoch() {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::MutationEpochMismatch {
                expected: checkpoint.mutation_epoch(),
                observed: before,
            },
        );
    }

    match (&envelope.report, &envelope.payload, &envelope.coverage) {
        (
            CapturedProductionAnalysisReportV1::Bounds(bounds),
            ProductionAnalysisWitnessPayloadV1::Bounds(witness),
            ProductionAnalysisWitnessCoverageV1::Complete { obligation_count },
        ) if envelope.checker
            == ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1 =>
        {
            if envelope.context_identity.is_none() {
                return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
            }
            if *obligation_count != witness.obligations.len() {
                return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
            }
            match build_bounds_presburger_witness(context, function, bounds)? {
                SupportedWitnessBuildV1::Complete(replayed) if replayed == *witness => {}
                SupportedWitnessBuildV1::Complete(_) | SupportedWitnessBuildV1::Incomplete(_) => {
                    return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
                }
            }
        }
        (
            _,
            ProductionAnalysisWitnessPayloadV1::Incomplete {
                gap: payload_gap,
                reason: payload_reason,
            },
            ProductionAnalysisWitnessCoverageV1::Incomplete { gap, reason },
        ) if envelope.checker == ProductionAnalysisWitnessCheckerV1::UnsupportedV1
            || envelope.checker
                == ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1 =>
        {
            if payload_gap != gap
                || payload_reason != reason
                || gap.pass() != checkpoint.pass()
                || reason.is_empty()
            {
                return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
            }
        }
        _ => return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch),
    }

    let after = current_mutation_epoch(context)?;
    if after != before {
        return Err(
            ProductionAnalysisWitnessValidationErrorV1::MutationEpochMismatch {
                expected: before,
                observed: after,
            },
        );
    }
    Ok(())
}

fn build_bounds_presburger_witness(
    context: &Context,
    function: &FuncOp,
    report: &RankedBoundsReportV1,
) -> Result<
    SupportedWitnessBuildV1<BoundsPresburgerWitnessV1>,
    ProductionAnalysisWitnessValidationErrorV1,
> {
    let mut analyses = PlironAnalysisManagerV1::new(function);
    analyses.prepare_function_inventory(context, function);
    let inventory = match analyses.function_inventory_handle() {
        Ok(inventory) => inventory,
        Err(failure) => {
            return Ok(SupportedWitnessBuildV1::Incomplete(format!(
                "bounds witness function inventory {} count {} exceeds limit {}",
                failure.resource(),
                failure.actual(),
                failure.limit(),
            )));
        }
    };
    let blocks = inventory.blocks();
    if blocks.len() != 1 {
        return Ok(SupportedWitnessBuildV1::Incomplete(
            "bounds witness V1 cannot yet enumerate exhaustive CFG path domains or dominating guard facts"
                .to_owned(),
        ));
    }

    let block_operations = inventory.block_operations(0);
    let launch_extents = match raw_launch_extents(context, block_operations) {
        Ok(extents) => extents,
        Err(reason) => return Ok(SupportedWitnessBuildV1::Incomplete(reason)),
    };
    let invocation_count = match exhaustive_invocation_count(&launch_extents) {
        Ok(count) => count,
        Err(reason) => return Ok(SupportedWitnessBuildV1::Incomplete(reason)),
    };
    if report.status() != KernelCheckStatusV1::Clean {
        return Ok(SupportedWitnessBuildV1::Incomplete(
            "the raw launch domain is supported, but only a Clean bounds report can be replayed as a positive witness"
                .to_owned(),
        ));
    }

    // This transcript is useful for auditing the production analysis, but it
    // is deliberately not the authority for `Complete`. The separate raw-IR
    // evaluator below interprets the defining operation DAG directly and
    // enumerates every invocation in the finite launch box.
    analyses.prepare_sparse_indices(context, function);
    analyses.prepare_presburger(context, function);
    let sparse = match analyses.sparse_indices() {
        Ok(sparse) => sparse,
        Err(failure) => {
            return Ok(SupportedWitnessBuildV1::Incomplete(format!(
                "sparse-index witness construction is incomplete: {failure:?}"
            )));
        }
    };
    let presburger = match analyses.presburger() {
        Ok(presburger) => presburger,
        Err(failure) => {
            return Ok(SupportedWitnessBuildV1::Incomplete(format!(
                "Presburger witness construction is incomplete: {failure:?}"
            )));
        }
    };

    let mut obligations = Vec::new();
    let mut evaluation_steps = 0usize;
    for site in block_operations {
        let operation_index = site.operation();
        let operation = Operation::get_op_dyn(site.pointer(), context);
        let Some(access) = operation.downcast_ref::<RankedAccessOp>() else {
            continue;
        };
        let view = access.view(context);
        let Some(view_type) = ranked_view_type(view, context) else {
            return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
        };
        let view_type: TypedHandle<dialect_kernel::RankedViewType> = view_type;
        let view_type = view_type.deref(context);
        for (dimension, index) in access.indices(context).into_iter().enumerate() {
            let Some(extent) = view_type.shape().get(dimension).copied() else {
                return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
            };
            if extent == DYNAMIC_EXTENT {
                return Ok(SupportedWitnessBuildV1::Incomplete(format!(
                    "bounds witness V1 cannot enumerate dynamic extent at block 0 op {operation_index} dimension {dimension}"
                )));
            }

            if let Err(failure) = exhaustively_check_raw_index(
                context,
                index,
                &launch_extents,
                extent,
                &mut evaluation_steps,
                operation_index,
                dimension,
            ) {
                match failure {
                    RawBoundsReplayFailureV1::Incomplete(reason) => {
                        return Ok(SupportedWitnessBuildV1::Incomplete(reason));
                    }
                    RawBoundsReplayFailureV1::Counterexample(error) => return Err(error),
                }
            }

            let fact = sparse.fact(index);
            if matches!(fact, SparseIndexFactV1::MachineOverflow(_)) {
                return Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch);
            }
            let normalized_map = match presburger.map_for_facts(&[fact]) {
                Ok(map) => map,
                Err(failure) => {
                    return Ok(SupportedWitnessBuildV1::Incomplete(format!(
                        "bounds witness V1 cannot capture the Presburger transcript at block 0 op {operation_index} dimension {dimension}: {failure}"
                    )));
                }
            };
            obligations.push(BoundsPresburgerObligationV1 {
                block: 0,
                operation: operation_index,
                dimension,
                extent,
                checked_invocations: invocation_count,
                normalized_map,
            });
        }
    }
    Ok(SupportedWitnessBuildV1::Complete(
        BoundsPresburgerWitnessV1 { obligations },
    ))
}

fn raw_launch_extents(
    context: &Context,
    operations: &[crate::pliron_function_inventory::PlironOperationSiteV1],
) -> Result<Vec<u64>, String> {
    let mut by_dimension = BTreeMap::<usize, u64>::new();
    let mut execution_layout = None;
    for site in operations {
        let operation = Operation::get_op_dyn(site.pointer(), context);
        if let Some(layout) = operation.downcast_ref::<ExecutionLayoutOp>() {
            if execution_layout.is_some() {
                return Err("bounds witness V1 found more than one gpu.execution_layout".to_owned());
            }
            if layout.verify(context).is_err() {
                return Err("bounds witness V1 found a malformed gpu.execution_layout".to_owned());
            }
            let Some(global_extents) = layout.global_extents(context) else {
                return Err("bounds witness V1 found a malformed gpu.execution_layout".to_owned());
            };
            execution_layout = Some(global_extents);
            continue;
        }

        let Some(invocation) = operation.downcast_ref::<InvocationIndexOp>() else {
            continue;
        };
        let Some(dimension) = invocation
            .dimension(context)
            .and_then(|dimension| usize::try_from(dimension).ok())
        else {
            return Err("bounds witness V1 found a malformed invocation dimension".to_owned());
        };
        let Some(extent) = invocation.launch_extent(context) else {
            return Err("bounds witness V1 found a missing launch extent".to_owned());
        };
        if extent == DYNAMIC_EXTENT {
            return Err(format!(
                "bounds witness V1 cannot enumerate dynamic launch dimension {dimension}"
            ));
        }
        if by_dimension.insert(dimension, extent).is_some() {
            return Err(format!(
                "bounds witness V1 found duplicate invocation dimension {dimension}"
            ));
        }
    }

    if let Some(layout_extents) = execution_layout {
        for (dimension, layout_extent) in layout_extents.into_iter().enumerate() {
            if layout_extent == DYNAMIC_EXTENT {
                return Err(format!(
                    "bounds witness V1 cannot enumerate dynamic gpu.execution_layout axis {dimension}"
                ));
            }
            match by_dimension.get(&dimension) {
                Some(invocation_extent) if *invocation_extent == layout_extent => {}
                Some(invocation_extent) => {
                    return Err(format!(
                        "bounds witness V1 invocation dimension {dimension} extent {invocation_extent} is inconsistent with gpu.execution_layout extent {layout_extent}"
                    ));
                }
                None if layout_extent > 1 => {
                    return Err(format!(
                        "bounds witness V1 gpu.execution_layout has active axis {dimension} extent {layout_extent} without an invocation dimension"
                    ));
                }
                None => {}
            }
        }
        if let Some((dimension, _)) = by_dimension.range(3..).next() {
            return Err(format!(
                "bounds witness V1 invocation dimension {dimension} is outside the three-dimensional gpu.execution_layout"
            ));
        }
        return Ok(layout_extents.to_vec());
    }

    let dimension_count = by_dimension
        .last_key_value()
        .map_or(0, |(dimension, _)| dimension + 1);
    let mut extents = vec![1; dimension_count];
    for (dimension, extent) in by_dimension {
        extents[dimension] = extent;
    }
    Ok(extents)
}

fn exhaustive_invocation_count(extents: &[u64]) -> Result<u64, String> {
    let mut count = 1u64;
    for extent in extents {
        count = count.checked_mul(*extent).ok_or_else(|| {
            "bounds witness V1 launch-domain cardinality overflows u64".to_owned()
        })?;
        if count > MAX_BOUNDS_WITNESS_INVOCATIONS_V1 {
            return Err(format!(
                "bounds witness V1 needs {count} invocations, exceeding its exhaustive replay cap of {MAX_BOUNDS_WITNESS_INVOCATIONS_V1}"
            ));
        }
    }
    Ok(count)
}

enum RawBoundsReplayFailureV1 {
    Incomplete(String),
    Counterexample(ProductionAnalysisWitnessValidationErrorV1),
}

fn exhaustively_check_raw_index(
    context: &Context,
    index: Value,
    extents: &[u64],
    extent: u64,
    evaluation_steps: &mut usize,
    operation: usize,
    dimension: usize,
) -> Result<(), RawBoundsReplayFailureV1> {
    if extents.iter().any(|extent| *extent == 0) {
        return Ok(());
    }
    let mut invocation = vec![0u64; extents.len()];
    loop {
        let mut cache = HashMap::new();
        let mut active = HashSet::new();
        let evaluated = match evaluate_raw_index(
            context,
            index,
            &invocation,
            &mut cache,
            &mut active,
            evaluation_steps,
        ) {
            Ok(value) => value,
            Err(RawIndexEvaluationFailureV1::Incomplete(reason)) => {
                return Err(RawBoundsReplayFailureV1::Incomplete(format!(
                    "bounds witness V1 cannot interpret block 0 op {operation} dimension {dimension}: {reason}"
                )));
            }
            Err(RawIndexEvaluationFailureV1::Overflow(operator)) => {
                return Err(RawBoundsReplayFailureV1::Counterexample(
                    ProductionAnalysisWitnessValidationErrorV1::BoundsMachineOverflow {
                        block: 0,
                        operation,
                        dimension,
                        invocation,
                        operator,
                    },
                ));
            }
        };
        if evaluated >= extent {
            return Err(RawBoundsReplayFailureV1::Counterexample(
                ProductionAnalysisWitnessValidationErrorV1::BoundsCounterexample {
                    block: 0,
                    operation,
                    dimension,
                    invocation,
                    index: evaluated,
                    extent,
                },
            ));
        }
        if !increment_invocation(&mut invocation, extents) {
            return Ok(());
        }
    }
}

fn increment_invocation(invocation: &mut [u64], extents: &[u64]) -> bool {
    for dimension in (0..invocation.len()).rev() {
        invocation[dimension] += 1;
        if invocation[dimension] < extents[dimension] {
            return true;
        }
        invocation[dimension] = 0;
    }
    false
}

enum RawIndexEvaluationFailureV1 {
    Incomplete(&'static str),
    Overflow(&'static str),
}

pub(crate) fn evaluate_raw_index_at_invocation_v1(
    context: &Context,
    value: Value,
    invocation: &[u64],
    evaluation_steps: &mut usize,
) -> Option<u64> {
    evaluate_raw_index(
        context,
        value,
        invocation,
        &mut HashMap::new(),
        &mut HashSet::new(),
        evaluation_steps,
    )
    .ok()
}

fn evaluate_raw_index(
    context: &Context,
    value: Value,
    invocation: &[u64],
    cache: &mut HashMap<Value, u64>,
    active: &mut HashSet<Value>,
    evaluation_steps: &mut usize,
) -> Result<u64, RawIndexEvaluationFailureV1> {
    if let Some(value) = cache.get(&value) {
        return Ok(*value);
    }
    *evaluation_steps = evaluation_steps.saturating_add(1);
    if *evaluation_steps > MAX_BOUNDS_WITNESS_EVALUATION_STEPS_V1 {
        return Err(RawIndexEvaluationFailureV1::Incomplete(
            "raw-index evaluation exceeded its deterministic work cap",
        ));
    }
    if !active.insert(value) {
        return Err(RawIndexEvaluationFailureV1::Incomplete(
            "the raw index definition graph is cyclic",
        ));
    }
    let result = (|| {
        let Some(definition) = value.defining_op() else {
            return Err(RawIndexEvaluationFailureV1::Incomplete(
                "block arguments are outside the V1 raw-index fragment",
            ));
        };
        let operation = Operation::get_op_dyn(definition, context);
        if let Some(constant) = operation.downcast_ref::<IndexConstantOp>() {
            return constant
                .value(context)
                .ok_or(RawIndexEvaluationFailureV1::Incomplete(
                    "index constant has no value",
                ));
        }
        if let Some(index) = operation.downcast_ref::<InvocationIndexOp>() {
            let dimension = index
                .dimension(context)
                .and_then(|dimension| usize::try_from(dimension).ok())
                .ok_or(RawIndexEvaluationFailureV1::Incomplete(
                    "invocation index has no valid dimension",
                ))?;
            return invocation.get(dimension).copied().ok_or(
                RawIndexEvaluationFailureV1::Incomplete(
                    "invocation dimension is absent from the launch inventory",
                ),
            );
        }
        let Some(binary) = operation.downcast_ref::<IndexBinaryOp>() else {
            return Err(RawIndexEvaluationFailureV1::Incomplete(
                "index producer is not a supported constant, invocation, or binary operation",
            ));
        };
        let lhs = evaluate_raw_index(
            context,
            binary.lhs(context),
            invocation,
            cache,
            active,
            evaluation_steps,
        )?;
        let rhs = evaluate_raw_index(
            context,
            binary.rhs(context),
            invocation,
            cache,
            active,
            evaluation_steps,
        )?;
        match binary.kind(context) {
            Some(IndexBinaryKindAttr::Add) => lhs
                .checked_add(rhs)
                .ok_or(RawIndexEvaluationFailureV1::Overflow("addition")),
            Some(IndexBinaryKindAttr::Multiply) => lhs
                .checked_mul(rhs)
                .ok_or(RawIndexEvaluationFailureV1::Overflow("multiplication")),
            Some(IndexBinaryKindAttr::Remainder) if rhs != 0 => Ok(lhs % rhs),
            Some(IndexBinaryKindAttr::Divide) if rhs != 0 => Ok(lhs / rhs),
            Some(IndexBinaryKindAttr::Remainder) => Err(RawIndexEvaluationFailureV1::Incomplete(
                "remainder divisor is zero",
            )),
            Some(IndexBinaryKindAttr::Divide) => Err(RawIndexEvaluationFailureV1::Incomplete(
                "division divisor is zero",
            )),
            None => Err(RawIndexEvaluationFailureV1::Incomplete(
                "index binary operation has no kind",
            )),
        }
    })();
    active.remove(&value);
    if let Ok(result) = result {
        cache.insert(value, result);
    }
    result
}

#[cfg(test)]
mod tests {
    use dialect_kernel::{DIALECT_NAME, register_dialect};
    use fe2o3_pliron_owner_core::{ensure_context_identity, require_context_identity};
    use pliron::{
        builtin::ops::FuncOp, context::Context, dialect::DialectName, op::Op, operation::Operation,
        parsable::parse_from_str,
    };

    use super::*;
    use crate::{KernelCheckPassKindV1, require_production_pliron_checks_before_lowering_v2};

    const SAFE_AFFINE: &str = r#"
builtin.func @bounds_witness_safe: builtin.function <() -> ()>
{
  ^entry_block1v1():
    v0 = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space Global]: <() -> (kernel.ranked_view <32,false,[16]>)>;
    v1 = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 8]: <() -> (kernel.index )>;
    v2 = kernel.index_constant () [] [kernel_index_value: kernel.index_value 2]: <() -> (kernel.index )>;
    v3 = kernel.index_constant () [] [kernel_index_value: kernel.index_value 1]: <() -> (kernel.index )>;
    v4 = kernel.index_binary (v1, v2) [] [kernel_index_binary_kind: kernel.index_binary_kind Multiply]: <(kernel.index , kernel.index ) -> (kernel.index )>;
    v5 = kernel.index_binary (v4, v3) [] [kernel_index_binary_kind: kernel.index_binary_kind Add]: <(kernel.index , kernel.index ) -> (kernel.index )>;
    kernel.access (v0, v5) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <32,false,[16]>, kernel.index ) -> ()>;
    kernel.return () [] []: <() -> ()>
}
"#;

    fn setup() -> Context {
        let mut context = Context::new();
        register_dialect(
            &mut context,
            &DialectName::try_new(DIALECT_NAME).expect("valid dialect"),
        )
        .expect("kernel dialect");
        dialect_gpu::register_dialect(&mut context).expect("gpu dialect");
        dialect_proof::register_dialect(&mut context).expect("proof dialect");
        ensure_context_identity(&mut context).expect("context identity");
        context
    }

    fn parse_function(context: &mut Context) -> FuncOp {
        parse_source(context, SAFE_AFFINE)
    }

    fn parse_source(context: &mut Context, source: &str) -> FuncOp {
        let operation = parse_from_str(Operation::top_level_parser(), context, source)
            .expect("parse witness function");
        FuncOp::from_operation(operation)
    }

    fn bounds_envelope<'a>(
        report: &'a crate::ProductionPlironPreloweringReportV2,
    ) -> (
        &'a crate::ProductionAnalysisStageValidationV1,
        ProductionAnalysisWitnessEnvelopeV1,
    ) {
        let stage = &report.report_validation().stages()[1];
        (stage, stage.witness().clone())
    }

    fn replay_with_clean_bounds_report(
        context: &mut Context,
        source: &str,
    ) -> Result<
        SupportedWitnessBuildV1<BoundsPresburgerWitnessV1>,
        ProductionAnalysisWitnessValidationErrorV1,
    > {
        // The clean report is intentionally from a distinct valid subject.
        // These tests exercise the independent witness replay boundary, not
        // the primary bounds verifier that issued that report.
        let safe = parse_function(context);
        let safe_report = require_production_pliron_checks_before_lowering_v2(context, &safe)
            .expect("safe bounds report");
        let candidate = parse_source(context, source);
        build_bounds_presburger_witness(context, &candidate, safe_report.bounds())
    }

    fn expect_incomplete_bounds_replay(
        replay: Result<
            SupportedWitnessBuildV1<BoundsPresburgerWitnessV1>,
            ProductionAnalysisWitnessValidationErrorV1,
        >,
        expected_reason: &str,
    ) {
        match replay {
            Ok(SupportedWitnessBuildV1::Incomplete(reason)) => assert!(
                reason.contains(expected_reason),
                "incomplete reason {reason:?} did not contain {expected_reason:?}"
            ),
            Ok(SupportedWitnessBuildV1::Complete(_)) => {
                panic!("hostile bounds replay must never be Complete")
            }
            Err(error) => panic!("expected Incomplete bounds replay, got rejection: {error}"),
        }
    }

    fn affine_source_with_execution_layout(name: &str, global: [u64; 3]) -> String {
        let [global_x, global_y, global_z] = global;
        let layout = format!(
            "    gpu.execution_layout () [] [gpu_execution_grid_identity: gpu.grid_identity 7, gpu_execution_global_x: gpu.execution_extent {global_x}, gpu_execution_global_y: gpu.execution_extent {global_y}, gpu_execution_global_z: gpu.execution_extent {global_z}, gpu_execution_workgroup_x: gpu.execution_extent {global_x}, gpu_execution_workgroup_y: gpu.execution_extent {global_y}, gpu_execution_workgroup_z: gpu.execution_extent {global_z}, gpu_execution_subgroup_size: gpu.subgroup_size 4, gpu_execution_domain: gpu.execution_domain FullPhysicalWorkgroups]: <() -> ()>;"
        );
        SAFE_AFFINE
            .replace("@bounds_witness_safe", &format!("@{name}"))
            .replace(
                "  ^entry_block1v1():",
                &format!("  ^entry_block1v1():\n{layout}"),
            )
    }

    #[test]
    fn affine_bounds_witness_replays_every_access_dimension() {
        let context = &mut setup();
        let function = parse_function(context);
        let report = require_production_pliron_checks_before_lowering_v2(context, &function)
            .expect("safe affine function");
        let (stage, envelope) = bounds_envelope(&report);

        assert_eq!(
            stage.checkpoint().pass(),
            KernelCheckPassKindV1::MemoryBounds
        );
        assert_eq!(
            envelope.checker(),
            ProductionAnalysisWitnessCheckerV1::BoundsExhaustiveRawIrReplayV1
        );
        assert_eq!(envelope.coverage().obligation_count(), 1);
        assert!(envelope.coverage().is_complete());
        assert_eq!(
            stage.independent_validation_status(),
            KernelCheckStatusV1::Clean
        );
        assert!(!envelope.grants_compiler_refinement_authority());
        assert!(!envelope.grants_lowering_or_launch_authority());
    }

    #[test]
    fn subject_binding_report_and_obligation_mutations_fail_closed() {
        let context = &mut setup();
        let function = parse_function(context);
        let report = require_production_pliron_checks_before_lowering_v2(context, &function)
            .expect("safe affine function");
        let (stage, envelope) = bounds_envelope(&report);
        let captured = CapturedProductionAnalysisReportV1::Bounds(report.bounds().clone());
        let foreign_context = &mut setup();
        let foreign_identity = require_context_identity(foreign_context).expect("foreign identity");

        let validate = |candidate: &ProductionAnalysisWitnessEnvelopeV1| {
            validate_production_analysis_witness_v1(
                context,
                &function,
                stage.checkpoint(),
                stage.implementation(),
                stage.configuration(),
                &captured,
                candidate,
            )
        };
        validate(&envelope).expect("unaltered envelope");

        let mut forged_subject = envelope.clone();
        forged_subject.context_identity = Some(foreign_identity);
        assert!(matches!(
            validate(&forged_subject),
            Err(ProductionAnalysisWitnessValidationErrorV1::SubjectMismatch)
        ));

        let mut substituted_report = envelope.clone();
        substituted_report.report =
            CapturedProductionAnalysisReportV1::TensorLayout(report.tensor_layout().clone());
        assert!(matches!(
            validate(&substituted_report),
            Err(ProductionAnalysisWitnessValidationErrorV1::ReportMismatch)
        ));

        let mut omitted = envelope.clone();
        let ProductionAnalysisWitnessPayloadV1::Bounds(witness) = &mut omitted.payload else {
            panic!("bounds witness payload");
        };
        witness.obligations.clear();
        assert!(matches!(
            validate(&omitted),
            Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch)
        ));

        let mut forged_extent = envelope.clone();
        let ProductionAnalysisWitnessPayloadV1::Bounds(witness) = &mut forged_extent.payload else {
            panic!("bounds witness payload");
        };
        witness.obligations[0].extent += 1;
        assert!(matches!(
            validate(&forged_extent),
            Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch)
        ));

        let mut forged_transcript = envelope.clone();
        let ProductionAnalysisWitnessPayloadV1::Bounds(witness) = &mut forged_transcript.payload
        else {
            panic!("bounds witness payload");
        };
        witness.obligations[0].checked_invocations += 1;
        assert!(matches!(
            validate(&forged_transcript),
            Err(ProductionAnalysisWitnessValidationErrorV1::PayloadMismatch)
        ));
    }

    #[test]
    fn exhaustive_resource_cap_and_dynamic_launch_remain_incomplete() {
        let context = &mut setup();
        let capped_source = SAFE_AFFINE
            .replace("@bounds_witness_safe", "@bounds_witness_capped")
            .replace("kernel.launch_extent 8", "kernel.launch_extent 65537")
            .replace("kernel.index_value 2", "kernel.index_value 16")
            .replace("kernel.index_value 1", "kernel.index_value 0")
            .replace(
                "kernel.index_binary_kind Multiply",
                "kernel.index_binary_kind Remainder",
            )
            .replace("kernel.access (v0, v5)", "kernel.access (v0, v4)");
        let capped = parse_source(context, &capped_source);
        let capped_report = require_production_pliron_checks_before_lowering_v2(context, &capped)
            .expect("analysis accepts bounded remainder");
        let capped_coverage = capped_report.report_validation().stages()[1]
            .witness()
            .coverage();
        assert_eq!(capped_coverage.status(), KernelCheckStatusV1::Incomplete);
        assert!(
            capped_coverage
                .incomplete_reason()
                .expect("resource reason")
                .contains("exhaustive replay cap")
        );

        let dynamic_source = SAFE_AFFINE
            .replace("@bounds_witness_safe", "@bounds_witness_dynamic")
            .replace("kernel.launch_extent 8", "kernel.launch_extent 0")
            .replace("kernel.access (v0, v5)", "kernel.access (v0, v3)");
        let dynamic = parse_source(context, &dynamic_source);
        let dynamic_report = require_production_pliron_checks_before_lowering_v2(context, &dynamic)
            .expect("constant access is analysis-safe with a dynamic launch");
        let dynamic_coverage = dynamic_report.report_validation().stages()[1]
            .witness()
            .coverage();
        assert_eq!(dynamic_coverage.status(), KernelCheckStatusV1::Incomplete);
        assert!(
            dynamic_coverage
                .incomplete_reason()
                .expect("dynamic reason")
                .contains("dynamic launch dimension")
        );
    }

    #[test]
    fn raw_evaluator_replays_a_concrete_counterexample_independently() {
        let context = &mut setup();
        let safe = parse_function(context);
        let safe_report = require_production_pliron_checks_before_lowering_v2(context, &safe)
            .expect("safe report");
        let out_of_bounds_source = SAFE_AFFINE
            .replace("@bounds_witness_safe", "@bounds_witness_counterexample")
            .replace("[16]", "[15]");
        let out_of_bounds = parse_source(context, &out_of_bounds_source);

        assert!(matches!(
            build_bounds_presburger_witness(context, &out_of_bounds, safe_report.bounds()),
            Err(ProductionAnalysisWitnessValidationErrorV1::BoundsCounterexample {
                invocation,
                index: 15,
                extent: 15,
                ..
            }) if invocation == vec![7]
        ));
    }

    #[test]
    fn matching_execution_layout_preserves_the_supported_fragment() {
        let context = &mut setup();
        let source = affine_source_with_execution_layout("bounds_witness_layout_match", [8, 1, 1]);
        let replay = replay_with_clean_bounds_report(context, &source)
            .expect("matching layout is replayable");
        let SupportedWitnessBuildV1::Complete(witness) = replay else {
            panic!("matching static execution layout must remain supported")
        };

        assert_eq!(witness.obligations.len(), 1);
        assert_eq!(witness.obligations[0].checked_invocations, 8);
    }

    #[test]
    fn execution_layout_must_agree_with_invocation_inventory() {
        let context = &mut setup();
        let source =
            affine_source_with_execution_layout("bounds_witness_layout_mismatch", [4, 1, 1]);

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "inconsistent with gpu.execution_layout",
        );
    }

    #[test]
    fn execution_layout_active_axes_require_invocation_dimensions() {
        let context = &mut setup();
        let source =
            affine_source_with_execution_layout("bounds_witness_missing_active_axis", [8, 4, 1]);

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "active axis 1 extent 4 without an invocation dimension",
        );
    }

    #[test]
    fn duplicate_execution_layout_records_are_incomplete() {
        let context = &mut setup();
        let source =
            affine_source_with_execution_layout("bounds_witness_duplicate_layout", [8, 1, 1]);
        let layout = source
            .lines()
            .find(|line| line.contains("gpu.execution_layout"))
            .expect("layout line");
        let source = source.replacen(layout, &format!("{layout}\n{layout}"), 1);

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "more than one gpu.execution_layout",
        );
    }

    #[test]
    fn malformed_execution_layout_is_incomplete() {
        let context = &mut setup();
        let source =
            affine_source_with_execution_layout("bounds_witness_malformed_layout", [8, 1, 1])
                .replace("gpu.subgroup_size 4", "gpu.subgroup_size 3");

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "malformed gpu.execution_layout",
        );
    }

    #[test]
    fn zero_or_dynamic_invocation_extent_cannot_vacuously_complete() {
        let context = &mut setup();
        let source = SAFE_AFFINE
            .replace("@bounds_witness_safe", "@bounds_witness_zero_invocations")
            .replace("kernel.launch_extent 8", "kernel.launch_extent 0")
            .replace("kernel.index_value 1", "kernel.index_value 99")
            .replace("kernel.access (v0, v5)", "kernel.access (v0, v3)");

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "cannot enumerate dynamic launch dimension 0",
        );
    }

    #[test]
    fn zero_divisors_are_incomplete_for_divide_and_remainder() {
        for (name, operator) in [
            ("bounds_witness_divide_zero", "Divide"),
            ("bounds_witness_remainder_zero", "Remainder"),
        ] {
            let context = &mut setup();
            let source = SAFE_AFFINE
                .replace("@bounds_witness_safe", &format!("@{name}"))
                .replace("kernel.index_value 2", "kernel.index_value 0")
                .replace(
                    "kernel.index_binary_kind Multiply",
                    &format!("kernel.index_binary_kind {operator}"),
                )
                .replace("kernel.access (v0, v5)", "kernel.access (v0, v4)");

            expect_incomplete_bounds_replay(
                replay_with_clean_bounds_report(context, &source),
                "divisor is zero",
            );
        }
    }

    #[test]
    fn duplicate_invocation_dimensions_cannot_self_certify() {
        let context = &mut setup();
        let source = SAFE_AFFINE
            .replace("@bounds_witness_safe", "@bounds_witness_duplicate_dimension")
            .replace(
                "    v2 = kernel.index_constant",
                "    v6 = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 8]: <() -> (kernel.index )>;\n    v2 = kernel.index_constant",
            );

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, &source),
            "duplicate invocation dimension 0",
        );
    }

    #[test]
    fn unresolved_index_definition_is_incomplete() {
        let context = &mut setup();
        let source = r#"
builtin.func @bounds_witness_unresolved: builtin.function <(kernel.index ) -> ()>
{
  ^entry_block1v1(unresolved_v0: kernel.index ):
    v1 = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space Global]: <() -> (kernel.ranked_view <32,false,[16]>)>;
    v2 = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 8]: <() -> (kernel.index )>;
    kernel.access (v1, unresolved_v0) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <32,false,[16]>, kernel.index ) -> ()>;
    kernel.return () [] []: <() -> ()>
}
"#;

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, source),
            "block arguments are outside the V1 raw-index fragment",
        );
    }

    #[test]
    fn cumulative_multi_access_multi_dimension_work_is_capped() {
        let context = &mut setup();
        let source = r#"
builtin.func @bounds_witness_cumulative_cap: builtin.function <() -> ()>
{
  ^entry_block1v1():
    v0 = kernel.ranked_view () [] [kernel_memory_space: kernel.memory_space Global]: <() -> (kernel.ranked_view <32,false,[16,16]>)>;
    v1 = kernel.invocation_index () [] [kernel_invocation_dimension: kernel.invocation_dimension 0, kernel_launch_extent: kernel.launch_extent 65536]: <() -> (kernel.index )>;
    v2 = kernel.index_constant () [] [kernel_index_value: kernel.index_value 16]: <() -> (kernel.index )>;
    v3 = kernel.index_binary (v1, v2) [] [kernel_index_binary_kind: kernel.index_binary_kind Remainder]: <(kernel.index , kernel.index ) -> (kernel.index )>;
    kernel.access (v0, v3, v3) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <32,false,[16,16]>, kernel.index , kernel.index ) -> ()>;
    kernel.access (v0, v3, v3) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <32,false,[16,16]>, kernel.index , kernel.index ) -> ()>;
    kernel.access (v0, v3, v3) [] [kernel_access_kind: kernel.access_kind Read]: <(kernel.ranked_view <32,false,[16,16]>, kernel.index , kernel.index ) -> ()>;
    kernel.return () [] []: <() -> ()>
}
"#;

        expect_incomplete_bounds_replay(
            replay_with_clean_bounds_report(context, source),
            "raw-index evaluation exceeded its deterministic work cap",
        );
    }
}
