#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use std::{error::Error, fmt};

use pliron::{
    attribute::Attribute,
    builtin::{
        ATTR_KEY_DEBUG_INFO,
        op_interfaces::{NOpdsInterface, NRegionsInterface, NResultsInterface},
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface, op_interface_impl, pliron_attr, pliron_op, pliron_type},
    dialect::{Dialect, DialectName},
    identifier::Identifier,
    op::Op,
    operation::Operation,
    result::Result as PlironResult,
    r#type::{Type, TypedHandle},
    verify_err, verify_err_noloc, verify_error,
};

mod registration;

pub use registration::dialect_registration;

mod collective_semantics;

pub use collective_semantics::{
    CollectiveSemanticContractError, MAX_COLLECTIVE_SEMANTIC_STEPS_V1, RequireFiniteFoldOp,
    RequireFiniteRecurrenceOp, RequirePermutationGatherOp, SemanticCoverageBindingAttr,
    SemanticDomainBoundAttr, SemanticEvaluationOrderAttr, SemanticNumericalPolicyAttr,
    SemanticStepBoundAttr,
};
mod pipeline_protocol;
mod ranked_memory;
mod semantic_contract;
mod semantic_typed_contract;
mod semantic_typed_expression;
mod tensor_layout;

pub use pipeline_protocol::{
    MAX_PIPELINE_BUFFERS_V1, PipelineCreateOp, PipelineEventKindAttr, PipelineEventOp,
    PipelineProtocolError, PipelineType, pipeline_type,
};
pub use ranked_memory::{
    AccessKindAttr, AllocationEffectOp, AllocationOriginAttr, AnalysisSplitControlCountAttr,
    AnalysisSplitOp, AtomicOrderingAttr, AtomicScopeAttr, BranchArgsOp, BranchOp,
    CheckedRowStripedIndex2DOp, CheckedTiledIndex2DOp, DYNAMIC_EXTENT, DeterministicJoinOp,
    DimensionAttr, DimensionOp, GFX950_TRANSPOSE_FP4_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP4_WORKGROUP_NOALIAS_CLASS_V1,
    GFX950_TRANSPOSE_FP8_WORKGROUP_ALLOCATION_ORIGIN_V1,
    GFX950_TRANSPOSE_FP8_WORKGROUP_NOALIAS_CLASS_V1, IndexBinaryKindAttr, IndexBinaryOp,
    IndexConstantOp, IndexEqualBranchArgsOp, IndexEqualBranchOp, IndexLessThanBranchArgsOp,
    IndexLessThanBranchOp, IndexType, IndexUnknownOp, IndexUnsignedCastOp, IndexValueAttr,
    InvocationDimensionAttr, InvocationIndexOp, LaunchExtentAttr, MAX_DETERMINISTIC_JOIN_INPUTS_V1,
    MAX_RANKED_MEMORY_RANK, MemorySpaceAttr, NoAliasClassAttr, OwnershipContractOp,
    OwnershipCoverageAttr, OwnershipPartitionAttr, RankedAccessOp, RankedMemoryError, RankedViewOp,
    RankedViewType, ReturnOp, SUPPORTED_ELEMENT_WIDTHS, TrapOp, is_checked_access_capability_type,
    is_index_type, is_supported_allocation_effect_contract_v1, ranked_view_type,
};
pub use semantic_contract::{
    RequireEquivalentOp, SemanticBinaryKindAttr, SemanticBinaryOp, SemanticConstantAttr,
    SemanticConstantOp, SemanticContractError, SemanticExpressionCommitmentAttr,
    SemanticExpressionCommitmentOp, SemanticScalarType, SemanticSymbolAttr, SemanticSymbolOp,
};
pub use semantic_typed_contract::{
    MAX_TENSOR_RESULT_COMPONENTS_V1, SemanticExceptionalValueAttr, SemanticIeeeRoundingAttr,
    SemanticOverflowAttr, SemanticScalarKindAttr, SemanticTypedBinaryKindAttr,
    SemanticTypedBinaryOp, SemanticTypedCastKindAttr, SemanticTypedCastOp,
    SemanticTypedCompareKindAttr, SemanticTypedCompareOp, SemanticTypedConstantOp,
    SemanticTypedExpressionRootOp, SemanticTypedScalarV1, SemanticTypedSelectOp,
    SemanticTypedSymbolOp, SemanticTypedUnaryKindAttr, SemanticTypedUnaryOp,
    TensorResultComponentOp,
};
pub use semantic_typed_expression::{
    MAX_SEMANTIC_TYPED_EXPRESSION_DEPTH_V1, MAX_SEMANTIC_TYPED_EXPRESSION_NODES_V1,
    SemanticNumericalContractV1, SemanticTypedExpressionErrorV1, SemanticTypedExpressionStatsV1,
    SemanticTypedExpressionV1,
};
pub use tensor_layout::{
    TensorConvergenceAttr, TensorDataflowRootsV1, TensorFragmentAttr, TensorInstructionAttr,
    TensorLayoutDialectError, TensorLayoutOp, TensorValueRootAttr,
};

/// The Pliron namespace owned by this crate.
pub const DIALECT_NAME: &str = "kernel";

/// The largest structured iteration rank admitted by this shell.
pub const MAX_ITERATION_RANK: u32 = 8;

/// The operation attribute key carrying the iteration domain.
pub const ITERATION_DOMAIN_ATTR_KEY: &str = "kernel_iteration_domain";

const REGISTRATION_MARKER_KEY: &str = "fe2o3_dialect_kernel_registration_v1";

/// The semantic owner reported by kernel interfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SemanticOwner {
    /// Structured algorithm semantics are owned by `kernel`.
    Kernel,
}

/// A bounded construction or verification failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelError {
    /// An iteration rank was zero or exceeded [`MAX_ITERATION_RANK`].
    IterationRankOutOfBounds(u32),
    /// The algorithm root did not carry its required typed domain attribute.
    MissingIterationDomain,
    /// The result type rank and iteration-domain rank disagreed.
    IterationRankMismatch { result: u32, domain: u32 },
}

impl fmt::Display for KernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IterationRankOutOfBounds(rank) => write!(
                formatter,
                "iteration rank {rank} is outside 1..={MAX_ITERATION_RANK}"
            ),
            Self::MissingIterationDomain => {
                formatter.write_str("algorithm root is missing its iteration-domain attribute")
            }
            Self::IterationRankMismatch { result, domain } => write!(
                formatter,
                "algorithm result rank {result} does not match domain rank {domain}"
            ),
        }
    }
}

impl Error for KernelError {}

fn check_rank(rank: u32) -> Result<(), KernelError> {
    if (1..=MAX_ITERATION_RANK).contains(&rank) {
        Ok(())
    } else {
        Err(KernelError::IterationRankOutOfBounds(rank))
    }
}

/// A target-neutral structured algorithm type.
#[pliron_type(name = "kernel.algorithm", format = "`<` $rank `>`")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AlgorithmType {
    rank: u32,
}

impl AlgorithmType {
    /// Creates a uniqued algorithm type after enforcing the rank bound.
    pub fn new(context: &Context, rank: u32) -> Result<TypedHandle<Self>, KernelError> {
        check_rank(rank)?;
        Ok(Self::instantiate(Self { rank }, context))
    }

    /// Returns the iteration rank.
    pub const fn rank(&self) -> u32 {
        self.rank
    }
}

impl Verify for AlgorithmType {
    fn verify(&self, _context: &Context) -> PlironResult<()> {
        if let Err(error) = check_rank(self.rank) {
            return verify_err_noloc!(error);
        }
        Ok(())
    }
}

/// Typed metadata for the structured iteration domain.
#[pliron_attr(name = "kernel.iteration_domain", format = "`<` $rank `>`")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IterationDomainAttr {
    rank: u32,
}

impl IterationDomainAttr {
    /// Creates iteration-domain metadata after enforcing the rank bound.
    pub fn new(rank: u32) -> Result<Self, KernelError> {
        check_rank(rank)?;
        Ok(Self { rank })
    }

    /// Returns the iteration rank.
    pub const fn rank(&self) -> u32 {
        self.rank
    }
}

impl Verify for IterationDomainAttr {
    fn verify(&self, _context: &Context) -> PlironResult<()> {
        if let Err(error) = check_rank(self.rank) {
            return verify_err_noloc!(error);
        }
        Ok(())
    }
}

/// Interface implemented only by structured semantic operations owned here.
#[op_interface]
pub trait StructuredAlgorithmOp {
    /// Returns the dialect that owns this operation's semantics.
    fn semantic_owner(&self) -> SemanticOwner;

    /// Kernel semantics remain independent of a physical target.
    fn is_target_neutral(&self) -> bool {
        true
    }

    /// Verifies the fixed ownership namespace.
    fn verify(operation: &dyn Op, context: &Context) -> PlironResult<()>
    where
        Self: Sized,
    {
        if operation.get_opid().dialect.as_ref() != DIALECT_NAME {
            return verify_err!(
                operation.loc(context),
                "kernel interface on foreign operation"
            );
        }
        Ok(())
    }
}

/// A minimal root for one structured algorithm graph.
#[pliron_op(
    name = "kernel.algorithm_root",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>, NRegionsInterface<0>],
    results = (algorithm: AlgorithmType),
)]
pub struct AlgorithmOp;

#[op_interface_impl]
impl StructuredAlgorithmOp for AlgorithmOp {
    fn semantic_owner(&self) -> SemanticOwner {
        SemanticOwner::Kernel
    }
}

impl AlgorithmOp {
    /// Creates a verified-shape algorithm root.
    pub fn new(context: &mut Context, rank: u32) -> Result<Self, KernelError> {
        let algorithm_type = AlgorithmType::new(context, rank)?;
        let domain = IterationDomainAttr::new(rank)?;
        let operation = Operation::new(
            context,
            Self::get_concrete_op_info(),
            vec![algorithm_type.into()],
            vec![],
            vec![],
            0,
        );
        let algorithm = Self { op: operation };
        algorithm.set_iteration_domain(context, domain);
        Ok(algorithm)
    }

    /// Returns a clone of the typed iteration-domain attribute, if present.
    pub fn iteration_domain(&self, context: &Context) -> Option<IterationDomainAttr> {
        self.get_operation()
            .deref(context)
            .attributes
            .0
            .get(&iteration_domain_attr_key())
            .and_then(|attribute| attribute.downcast_ref::<IterationDomainAttr>())
            .cloned()
    }

    /// Replaces the domain metadata. Verification rechecks consistency.
    pub fn set_iteration_domain(&self, context: &Context, domain: IterationDomainAttr) {
        self.get_operation()
            .deref_mut(context)
            .attributes
            .0
            .insert(iteration_domain_attr_key(), Box::new(domain));
    }
}

impl Verify for AlgorithmOp {
    fn verify(&self, context: &Context) -> PlironResult<()> {
        verify_closed_shape(self, context)?;
        let domain = self
            .iteration_domain(context)
            .ok_or_else(|| verify_error!(self.loc(context), KernelError::MissingIterationDomain))?;
        let result_type = self.get_operation().deref(context).get_type(0);
        let result_type = result_type.deref(context);
        let algorithm_type = result_type.downcast_ref::<AlgorithmType>().ok_or_else(|| {
            verify_error!(self.loc(context), "algorithm result has a foreign type")
        })?;
        if algorithm_type.rank() != domain.rank() {
            return verify_err!(
                self.loc(context),
                KernelError::IterationRankMismatch {
                    result: algorithm_type.rank(),
                    domain: domain.rank(),
                }
            );
        }
        Ok(())
    }
}

fn verify_closed_shape(op: &dyn Op, context: &Context) -> PlironResult<()> {
    let operation = op.get_operation();
    let operation = operation.deref(context);
    let attributes_are_closed = operation.attributes.0.iter().all(|(key, attribute)| {
        key == &iteration_domain_attr_key()
            || (key == &*ATTR_KEY_DEBUG_INFO && is_debug_info(attribute.as_ref()))
    });
    if operation.get_num_operands() != 0
        || operation.get_num_results() != 1
        || operation.get_num_successors() != 0
        || operation.num_regions() != 0
        || !attributes_are_closed
    {
        return verify_err!(
            op.loc(context),
            "{} has malformed or unbounded structural payload",
            op.get_opid()
        );
    }
    Ok(())
}

fn is_debug_info(attribute: &dyn Attribute) -> bool {
    let id = attribute.get_attr_id();
    id.dialect.as_ref() == "builtin" && AsRef::<str>::as_ref(&id.name) == "debug_info"
}

fn iteration_domain_attr_key() -> Identifier {
    ITERATION_DOMAIN_ATTR_KEY
        .try_into()
        .expect("constant attribute key is a valid identifier")
}

#[derive(Debug)]
struct RegistrationMarker;

/// Result of explicit registration in one Pliron context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationOutcome {
    /// This call installed the explicit registration marker.
    Registered,
    /// This crate had already registered in the same context.
    AlreadyRegistered,
}

/// A fail-closed explicit registration error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RegistrationError {
    /// The requested namespace was not [`DIALECT_NAME`].
    WrongDialect,
    /// Another typed value already claimed this crate's marker key.
    MarkerCollision,
    /// The marker map referenced absent auxiliary data.
    CorruptMarker,
}

impl fmt::Display for RegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongDialect => {
                formatter.write_str("kernel registration requested for wrong dialect")
            }
            Self::MarkerCollision => formatter.write_str("kernel registration marker collision"),
            Self::CorruptMarker => formatter.write_str("kernel registration marker is corrupt"),
        }
    }
}

impl Error for RegistrationError {}

/// Explicitly registers every kernel entity, rejecting marker collisions.
pub fn register_dialect(
    context: &mut Context,
    requested: &DialectName,
) -> Result<RegistrationOutcome, RegistrationError> {
    if requested.as_ref() != DIALECT_NAME {
        return Err(RegistrationError::WrongDialect);
    }

    let marker_key: Identifier = REGISTRATION_MARKER_KEY
        .try_into()
        .expect("constant marker key is a valid identifier");
    if let Some(index) = context.aux_data_map.get(&marker_key).copied() {
        return match context.aux_data.get(index) {
            Some(marker) if marker.is::<RegistrationMarker>() => {
                Ok(RegistrationOutcome::AlreadyRegistered)
            }
            Some(_) => Err(RegistrationError::MarkerCollision),
            None => Err(RegistrationError::CorruptMarker),
        };
    }

    Dialect::register(context, requested);
    AlgorithmType::register(context);
    <IterationDomainAttr as Attribute>::register::<IterationDomainAttr>(context);
    RankedViewType::register(context);
    IndexType::register(context);
    SemanticScalarType::register(context);
    <SemanticScalarKindAttr as Attribute>::register::<SemanticScalarKindAttr>(context);
    <SemanticTypedUnaryKindAttr as Attribute>::register::<SemanticTypedUnaryKindAttr>(context);
    <SemanticTypedBinaryKindAttr as Attribute>::register::<SemanticTypedBinaryKindAttr>(context);
    <SemanticOverflowAttr as Attribute>::register::<SemanticOverflowAttr>(context);
    <SemanticTypedCompareKindAttr as Attribute>::register::<SemanticTypedCompareKindAttr>(context);
    <SemanticTypedCastKindAttr as Attribute>::register::<SemanticTypedCastKindAttr>(context);
    <SemanticNumericalPolicyAttr as Attribute>::register::<SemanticNumericalPolicyAttr>(context);
    <SemanticIeeeRoundingAttr as Attribute>::register::<SemanticIeeeRoundingAttr>(context);
    <SemanticExceptionalValueAttr as Attribute>::register::<SemanticExceptionalValueAttr>(context);
    <IndexValueAttr as Attribute>::register::<IndexValueAttr>(context);
    <DimensionAttr as Attribute>::register::<DimensionAttr>(context);
    <AccessKindAttr as Attribute>::register::<AccessKindAttr>(context);
    <AtomicOrderingAttr as Attribute>::register::<AtomicOrderingAttr>(context);
    <AtomicScopeAttr as Attribute>::register::<AtomicScopeAttr>(context);
    <MemorySpaceAttr as Attribute>::register::<MemorySpaceAttr>(context);
    <AllocationOriginAttr as Attribute>::register::<AllocationOriginAttr>(context);
    <NoAliasClassAttr as Attribute>::register::<NoAliasClassAttr>(context);
    <OwnershipCoverageAttr as Attribute>::register::<OwnershipCoverageAttr>(context);
    <OwnershipPartitionAttr as Attribute>::register::<OwnershipPartitionAttr>(context);
    <InvocationDimensionAttr as Attribute>::register::<InvocationDimensionAttr>(context);
    <LaunchExtentAttr as Attribute>::register::<LaunchExtentAttr>(context);
    <AnalysisSplitControlCountAttr as Attribute>::register::<AnalysisSplitControlCountAttr>(
        context,
    );
    <IndexBinaryKindAttr as Attribute>::register::<IndexBinaryKindAttr>(context);
    <SemanticSymbolAttr as Attribute>::register::<SemanticSymbolAttr>(context);
    <SemanticConstantAttr as Attribute>::register::<SemanticConstantAttr>(context);
    <SemanticExpressionCommitmentAttr as Attribute>::register::<SemanticExpressionCommitmentAttr>(
        context,
    );
    <SemanticBinaryKindAttr as Attribute>::register::<SemanticBinaryKindAttr>(context);
    <SemanticDomainBoundAttr as Attribute>::register::<SemanticDomainBoundAttr>(context);
    <SemanticStepBoundAttr as Attribute>::register::<SemanticStepBoundAttr>(context);
    <SemanticEvaluationOrderAttr as Attribute>::register::<SemanticEvaluationOrderAttr>(context);
    <SemanticCoverageBindingAttr as Attribute>::register::<SemanticCoverageBindingAttr>(context);
    AlgorithmOp::register(context);
    RankedViewOp::register(context);
    IndexConstantOp::register(context);
    IndexUnknownOp::register(context);
    InvocationIndexOp::register(context);
    IndexUnsignedCastOp::register(context);
    IndexBinaryOp::register(context);
    DeterministicJoinOp::register(context);
    CheckedTiledIndex2DOp::register(context);
    CheckedRowStripedIndex2DOp::register(context);
    DimensionOp::register(context);
    RankedAccessOp::register(context);
    OwnershipContractOp::register(context);
    AllocationEffectOp::register(context);
    IndexLessThanBranchOp::register(context);
    IndexLessThanBranchArgsOp::register(context);
    IndexEqualBranchOp::register(context);
    IndexEqualBranchArgsOp::register(context);
    AnalysisSplitOp::register(context);
    BranchOp::register(context);
    BranchArgsOp::register(context);
    ReturnOp::register(context);
    TrapOp::register(context);
    SemanticSymbolOp::register(context);
    SemanticConstantOp::register(context);
    SemanticExpressionCommitmentOp::register(context);
    SemanticBinaryOp::register(context);
    SemanticTypedSymbolOp::register(context);
    TensorResultComponentOp::register(context);
    SemanticTypedConstantOp::register(context);
    SemanticTypedUnaryOp::register(context);
    SemanticTypedBinaryOp::register(context);
    SemanticTypedCompareOp::register(context);
    SemanticTypedSelectOp::register(context);
    SemanticTypedCastOp::register(context);
    SemanticTypedExpressionRootOp::register(context);
    RequireEquivalentOp::register(context);
    RequireFiniteFoldOp::register(context);
    RequireFiniteRecurrenceOp::register(context);
    RequirePermutationGatherOp::register(context);
    TensorConvergenceAttr::register(context);
    TensorInstructionAttr::register(context);
    TensorFragmentAttr::register(context);
    TensorValueRootAttr::register(context);
    TensorLayoutOp::register(context);

    let marker = context.aux_data.insert(Box::new(RegistrationMarker));
    context.aux_data_map.insert(marker_key, marker);
    Ok(RegistrationOutcome::Registered)
}
