#[cfg(feature = "internal-proof-staging")]
use fe2o3_functional_proof::ImportedFunctionalRefinementProofV2;
use fe2o3_functional_proof::{
    FunctionalRefinementBindingV2, FunctionalRefinementBoundaryV2,
    FunctionalRefinementReceiptIdentityV2, FunctionalRefinementSubjectsV2,
    HARD_MAX_AGGREGATE_FUNCTIONAL_OUTPUTS_V1, HARD_MAX_PARALLEL_CALL_ARGUMENTS_V1,
    VerusToolchainIdentityV2,
};
use fe2o3_proof_contracts::DigestV1;
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    error::Error,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
};

use dialect_gpu::{
    AddressSpaceAttr, BarrierOp, ExecutionDomainAttr, ExecutionLayoutOp, FenceOp, HierarchyAttr,
    MemoryOrderAttr, MemoryScopeAttr,
};
use dialect_kernel::{
    AccessKindAttr, AllocationEffectOp, AnalysisSplitOp, AtomicOrderingAttr, AtomicScopeAttr,
    BranchArgsOp, BranchOp, CheckedRowStripedIndex2DOp, CheckedTiledIndex2DOp, DYNAMIC_EXTENT,
    DeterministicJoinOp, DimensionOp, IndexBinaryKindAttr, IndexBinaryOp, IndexConstantOp,
    IndexEqualBranchArgsOp, IndexEqualBranchOp, IndexLessThanBranchArgsOp, IndexLessThanBranchOp,
    IndexType, IndexUnknownOp, IndexUnsignedCastOp, InvocationIndexOp,
    MAX_COLLECTIVE_SEMANTIC_STEPS_V1, MAX_DETERMINISTIC_JOIN_INPUTS_V1, MAX_RANKED_MEMORY_RANK,
    MemorySpaceAttr, OwnershipContractOp, OwnershipCoverageAttr, OwnershipPartitionAttr,
    PipelineCreateOp, PipelineEventKindAttr, PipelineEventOp, RankedAccessOp, RankedViewOp,
    RankedViewType, RequireEquivalentOp, RequireFiniteFoldOp, RequireFiniteRecurrenceOp,
    RequirePermutationGatherOp, ReturnOp, SUPPORTED_ELEMENT_WIDTHS, SemanticBinaryKindAttr,
    SemanticBinaryOp, SemanticConstantOp, SemanticCoverageBindingAttr, SemanticEvaluationOrderAttr,
    SemanticExceptionalValueAttr, SemanticIeeeRoundingAttr, SemanticNumericalPolicyAttr,
    SemanticOverflowAttr, SemanticScalarKindAttr, SemanticSymbolOp, SemanticTypedBinaryKindAttr,
    SemanticTypedBinaryOp, SemanticTypedCastKindAttr, SemanticTypedCastOp,
    SemanticTypedCompareKindAttr, SemanticTypedCompareOp, SemanticTypedConstantOp,
    SemanticTypedExpressionRootOp, SemanticTypedScalarV1, SemanticTypedSelectOp,
    SemanticTypedSymbolOp, SemanticTypedUnaryKindAttr, SemanticTypedUnaryOp, TensorConvergenceAttr,
    TensorLayoutOp, TensorResultComponentOp, TrapOp, is_supported_allocation_effect_contract_v1,
};
use dialect_proof::{
    AbsoluteErrorF64BitsAttr, CoveredBoundaryAttr, EvidenceRefOp, EvidenceStatusAttr, ObligationOp,
    ProofIdAttr, PropertyAttr, RelativeErrorF64BitsAttr, RequireEffectRefinementOp,
    RequireNumericalRefinementOp, RequireRefinementOp, RequireTensorRefinementOp,
};
use fe2o3_kernel_analysis::{
    HierarchicalOwnershipReportV1, MAX_RANKED_BOUNDS_BLOCKS, MAX_RANKED_BOUNDS_OPERATIONS,
    PlironAtomicLegalityReportV1, PlironAtomicTargetCapabilityV1, PlironAtomicTargetContextErrorV1,
    PlironAtomicTargetContextV1, PlironBarrierReportV1, PlironPipelineProtocolReportV1,
    PlironSemanticRefinementReportV1, PlironTensorLayoutReportV1, PlironWorkgroupMemoryReportV1,
    ProductionPlironPreloweringErrorV2, ProductionPlironPreloweringReportV2, RankedBoundsReportV1,
    RankedRaceReportV1, require_production_pliron_checks_before_lowering_v2,
    require_production_pliron_checks_with_atomic_target_before_lowering_v2,
};
use fe2o3_kernel_ir::{MatrixElement, TensorLayoutContractV1};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        op_interfaces::{OneRegionInterface, SingleBlockRegionInterface},
        ops::{FuncOp, ModuleOp},
        types::FunctionType,
    },
    common_traits::Named,
    context::Ptr,
    identifier::Identifier,
    linked_list::ContainsLinkedList,
    op::Op,
    operation::Operation,
    r#type::TypeHandle,
    value::Value,
};

use super::{
    ProductionIeeeExceptionalValuePolicyV2, ProductionIeeeRoundingModeV2,
    ProductionNumericalContractV2, ProductionOverflowContractV2, ProductionSemanticBinaryOpV2,
    ProductionSemanticCastV2, ProductionSemanticComparisonV2, ProductionSemanticExpressionErrorV2,
    ProductionSemanticExpressionV2, ProductionSemanticScalarTypeV2, ProductionSemanticUnaryOpV2,
};

/// Compiler-derived provenance retained for one cooperative tensor call.
///
/// These are identities of typed capability roots, not user assertions. The
/// source projector obtains them from the dominating context, lane, operand,
/// and accumulator producers. Production semantic composition additionally
/// binds this record to the live layout, control-flow site, output relation,
/// hierarchy reports, and authenticated refinement receipt.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionCooperativeTensorBindingV1 {
    context_root: DigestV1,
    lane_root: DigestV1,
    lhs_root: DigestV1,
    rhs_root: DigestV1,
    accumulator_root: DigestV1,
    result_root: DigestV1,
    argument_count: u16,
}

impl ProductionCooperativeTensorBindingV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_root: DigestV1,
        lane_root: DigestV1,
        lhs_root: DigestV1,
        rhs_root: DigestV1,
        accumulator_root: DigestV1,
        result_root: DigestV1,
        argument_count: u16,
    ) -> Option<Self> {
        if [
            context_root,
            lane_root,
            lhs_root,
            rhs_root,
            accumulator_root,
            result_root,
        ]
        .into_iter()
        .any(DigestV1::is_zero)
            || argument_count == 0
            || argument_count > HARD_MAX_PARALLEL_CALL_ARGUMENTS_V1
        {
            return None;
        }
        Some(Self {
            context_root,
            lane_root,
            lhs_root,
            rhs_root,
            accumulator_root,
            result_root,
            argument_count,
        })
    }

    pub const fn context_root(self) -> DigestV1 {
        self.context_root
    }
    pub const fn lane_root(self) -> DigestV1 {
        self.lane_root
    }
    pub const fn lhs_root(self) -> DigestV1 {
        self.lhs_root
    }
    pub const fn rhs_root(self) -> DigestV1 {
        self.rhs_root
    }
    pub const fn accumulator_root(self) -> DigestV1 {
        self.accumulator_root
    }
    pub const fn result_root(self) -> DigestV1 {
        self.result_root
    }
    pub const fn argument_count(self) -> u16 {
        self.argument_count
    }
}

#[cfg(feature = "internal-proof-staging")]
use super::HARD_MAX_PRODUCTION_CONSTRUCTIONS;
use super::{
    ConstructedGraphStageV1, KernelChecksVerifiedGraphStageV1, ProductionConstructionKindV1,
    ProductionConstructionV1, ProductionPlironSessionV1, ProductionRootHandleV1,
    ProductionSessionErrorV1, ProductionStageHandleV1, RootIdentityV1,
};
use crate::{
    ContextBuildError, HARD_MAX_SESSION_OPERATION_TREE_ITEMS, NameError, NameKind, OperationHandle,
    OperationHandleError, ProductionSessionLimitsV1, validate_name,
};

mod ranked_index_constant_fold_v1;

pub use ranked_index_constant_fold_v1::ProductionRankedTranslationErrorV1;

pub const HARD_MAX_PRODUCTION_RANKED_ARGUMENTS: usize = 64;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProductionCollectiveSemanticKindV1 {
    FiniteFold,
    FiniteRecurrence,
    PermutationGather,
}

/// Closed finite semantic contract retained by the production ranked recipe.
///
/// This metadata does not prove a GPU implementation. The mandatory semantic
/// pass additionally requires an exact coverage theorem and an independently
/// authenticated MIR functional-refinement equality over the contract values.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionCollectiveSemanticContractV1 {
    kind: ProductionCollectiveSemanticKindV1,
    contract_identity: [u64; 4],
    source_domain_identity: [u64; 4],
    target_domain_identity: [u64; 4],
    domain_bound: u64,
    step_bound: u64,
    order: SemanticEvaluationOrderAttr,
    numerical_contract: ProductionNumericalContractV2,
    coverage: SemanticCoverageBindingAttr,
}

impl ProductionCollectiveSemanticContractV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: ProductionCollectiveSemanticKindV1,
        contract_identity: [u64; 4],
        source_domain_identity: [u64; 4],
        target_domain_identity: [u64; 4],
        domain_bound: u64,
        step_bound: u64,
        order: SemanticEvaluationOrderAttr,
        numerical_contract: ProductionNumericalContractV2,
        coverage: SemanticCoverageBindingAttr,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        let identities = [
            contract_identity,
            source_domain_identity,
            target_domain_identity,
        ];
        if identities.contains(&[0; 4])
            || contract_identity == source_domain_identity
            || contract_identity == target_domain_identity
            || domain_bound == 0
            || domain_bound > MAX_COLLECTIVE_SEMANTIC_STEPS_V1
            || step_bound == 0
            || step_bound > domain_bound
            || !numerical_contract.is_supported()
            || matches!(kind, ProductionCollectiveSemanticKindV1::PermutationGather)
                && source_domain_identity == target_domain_identity
            || !matches!(kind, ProductionCollectiveSemanticKindV1::PermutationGather)
                && source_domain_identity != target_domain_identity
        {
            return Err(ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract);
        }
        Ok(Self {
            kind,
            contract_identity,
            source_domain_identity,
            target_domain_identity,
            domain_bound,
            step_bound,
            order,
            numerical_contract,
            coverage,
        })
    }

    pub const fn kind(&self) -> ProductionCollectiveSemanticKindV1 {
        self.kind
    }
    pub const fn contract_identity(&self) -> [u64; 4] {
        self.contract_identity
    }
    pub const fn source_domain_identity(&self) -> [u64; 4] {
        self.source_domain_identity
    }
    pub const fn target_domain_identity(&self) -> [u64; 4] {
        self.target_domain_identity
    }
    pub const fn domain_bound(&self) -> u64 {
        self.domain_bound
    }
    pub const fn step_bound(&self) -> u64 {
        self.step_bound
    }
    pub const fn order(&self) -> SemanticEvaluationOrderAttr {
        self.order
    }
    pub const fn numerical_contract(&self) -> ProductionNumericalContractV2 {
        self.numerical_contract
    }
    pub const fn coverage(&self) -> SemanticCoverageBindingAttr {
        self.coverage
    }

    /// Contract metadata alone never proves an implementation or final value.
    pub const fn grants_gpu_implementation_refinement_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionRankedValueIdV1(u32);

impl ProductionRankedValueIdV1 {
    pub const fn new(identity: u32) -> Self {
        Self(identity)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ProductionRankedValueV1 {
    Argument(u32),
    BlockArgument { block: u32, argument: u32 },
    Local(ProductionRankedValueIdV1),
}

/// Exact receipt and semantic binding requested by one ranked recipe operation.
///
/// This cloneable request is not evidence. Only
/// [`compile_ranked_kernel_with_policy_checked_refinement_staging_v2`] can reconcile it with a consumed,
/// authenticated [`ImportedFunctionalRefinementProofV2`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionReferenceProofV2 {
    receipt_identity: FunctionalRefinementReceiptIdentityV2,
    binding: FunctionalRefinementBindingV2,
}

impl ProductionReferenceProofV2 {
    pub const fn request_exact(
        receipt_identity: FunctionalRefinementReceiptIdentityV2,
        binding: FunctionalRefinementBindingV2,
    ) -> Self {
        Self {
            receipt_identity,
            binding,
        }
    }

    pub const fn receipt_identity(&self) -> FunctionalRefinementReceiptIdentityV2 {
        self.receipt_identity
    }

    pub const fn binding(&self) -> FunctionalRefinementBindingV2 {
        self.binding
    }
}

const FUNCTIONAL_REFINEMENT_FORMULA_DOMAIN_V2: &[u8] =
    b"FE2O3/PLIRON/FUNCTIONAL-REFINEMENT-FORMULA/V2\0";
const EFFECT_REFINEMENT_CONTRACT_DOMAIN_V2: &[u8] = b"FE2O3/PLIRON/EFFECT-REFINEMENT-CONTRACT/V2\0";
const NUMERICAL_REFINEMENT_CONTRACT_DOMAIN_V2: &[u8] =
    b"FE2O3/PLIRON/NUMERICAL-REFINEMENT-CONTRACT/V2\0";
const TENSOR_REFINEMENT_CONTRACT_DOMAIN_V1: &[u8] = b"FE2O3/PLIRON/TENSOR-REFINEMENT-CONTRACT/V1\0";
pub const MAX_PRODUCTION_RANKED_EFFECT_INDICES_V2: usize = MAX_RANKED_MEMORY_RANK;
pub const MAX_PRODUCTION_TENSOR_COMPONENTS_V1: usize =
    dialect_kernel::MAX_TENSOR_RESULT_COMPONENTS_V1;
pub const MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1: usize =
    HARD_MAX_AGGREGATE_FUNCTIONAL_OUTPUTS_V1;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ProductionGpuWriteSiteV2 {
    block: u32,
    operation: u32,
}

impl ProductionGpuWriteSiteV2 {
    pub const fn new(block: u32, operation: u32) -> Self {
        Self { block, operation }
    }
    pub const fn block(self) -> u32 {
        self.block
    }
    pub const fn operation(self) -> u32 {
        self.operation
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionReferenceOutputSiteV2 {
    argument: u32,
    block: u32,
    statement: u32,
}

impl ProductionReferenceOutputSiteV2 {
    pub const fn new(argument: u32, block: u32, statement: u32) -> Self {
        Self {
            argument,
            block,
            statement,
        }
    }
    pub const fn argument(self) -> u32 {
        self.argument
    }
    pub const fn block(self) -> u32 {
        self.block
    }
    pub const fn statement(self) -> u32 {
        self.statement
    }
}

/// Workload-neutral normalized effect statement joined to one logical GPU write.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionEffectRefinementContractV2 {
    contract_identity: u64,
    gpu_write_site: ProductionGpuWriteSiteV2,
    reference_output_site: ProductionReferenceOutputSiteV2,
    view: ProductionRankedValueV1,
    indices: Vec<ProductionRankedValueV1>,
    gpu_coordinates: Vec<ProductionRankedValueV1>,
    reference_coordinates: Vec<ProductionRankedValueV1>,
    gpu_domain: ProductionRankedValueV1,
    reference_domain: ProductionRankedValueV1,
    gpu_precondition: ProductionRankedValueV1,
    reference_precondition: ProductionRankedValueV1,
    gpu_value: ProductionRankedValueV1,
    reference_value: ProductionRankedValueV1,
}

impl ProductionEffectRefinementContractV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract_identity: u64,
        gpu_write_site: ProductionGpuWriteSiteV2,
        reference_output_site: ProductionReferenceOutputSiteV2,
        view: ProductionRankedValueV1,
        indices: Vec<ProductionRankedValueV1>,
        gpu_coordinates: Vec<ProductionRankedValueV1>,
        reference_coordinates: Vec<ProductionRankedValueV1>,
        gpu_domain: ProductionRankedValueV1,
        reference_domain: ProductionRankedValueV1,
        gpu_precondition: ProductionRankedValueV1,
        reference_precondition: ProductionRankedValueV1,
        gpu_value: ProductionRankedValueV1,
        reference_value: ProductionRankedValueV1,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        if contract_identity == 0
            || indices.is_empty()
            || indices.len() > MAX_PRODUCTION_RANKED_EFFECT_INDICES_V2
            || gpu_coordinates.len() != indices.len()
            || reference_coordinates.len() != indices.len()
        {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        Ok(Self {
            contract_identity,
            gpu_write_site,
            reference_output_site,
            view,
            indices,
            gpu_coordinates,
            reference_coordinates,
            gpu_domain,
            reference_domain,
            gpu_precondition,
            reference_precondition,
            gpu_value,
            reference_value,
        })
    }

    pub const fn contract_identity(&self) -> u64 {
        self.contract_identity
    }
    pub const fn gpu_write_site(&self) -> ProductionGpuWriteSiteV2 {
        self.gpu_write_site
    }
    pub const fn reference_output_site(&self) -> ProductionReferenceOutputSiteV2 {
        self.reference_output_site
    }
    pub const fn view(&self) -> ProductionRankedValueV1 {
        self.view
    }
    pub fn indices(&self) -> &[ProductionRankedValueV1] {
        &self.indices
    }
    pub fn gpu_coordinates(&self) -> &[ProductionRankedValueV1] {
        &self.gpu_coordinates
    }
    pub fn reference_coordinates(&self) -> &[ProductionRankedValueV1] {
        &self.reference_coordinates
    }
    pub const fn gpu_domain(&self) -> ProductionRankedValueV1 {
        self.gpu_domain
    }
    pub const fn reference_domain(&self) -> ProductionRankedValueV1 {
        self.reference_domain
    }
    pub const fn gpu_precondition(&self) -> ProductionRankedValueV1 {
        self.gpu_precondition
    }
    pub const fn reference_precondition(&self) -> ProductionRankedValueV1 {
        self.reference_precondition
    }
    pub const fn gpu_value(&self) -> ProductionRankedValueV1 {
        self.gpu_value
    }
    pub const fn reference_value(&self) -> ProductionRankedValueV1 {
        self.reference_value
    }

    /// Non-authoritative shape identity. Production admission uses the full
    /// validated-kernel transcript rather than this request-local digest.
    pub fn request_shape_hash(&self) -> DigestV1 {
        let mut writer = CanonicalRefinementDigestV2::new(EFFECT_REFINEMENT_CONTRACT_DOMAIN_V2);
        writer.field(1, &self.contract_identity.to_le_bytes());
        writer.field(2, &self.gpu_write_site.block.to_le_bytes());
        writer.field(3, &self.gpu_write_site.operation.to_le_bytes());
        writer.field(4, &self.reference_output_site.argument.to_le_bytes());
        writer.field(5, &self.reference_output_site.block.to_le_bytes());
        writer.field(6, &self.reference_output_site.statement.to_le_bytes());
        writer.value(7, self.view);
        writer.values(8, &self.indices);
        writer.values(9, &self.gpu_coordinates);
        writer.values(10, &self.reference_coordinates);
        for (tag, value) in [
            (11, self.gpu_domain),
            (12, self.reference_domain),
            (13, self.gpu_precondition),
            (14, self.reference_precondition),
            (15, self.gpu_value),
            (16, self.reference_value),
        ] {
            writer.value(tag, value);
        }
        writer.finish()
    }
}

/// Workload-neutral finite-error theorem over two typed semantic roots.
///
/// At every logical point where `domain && precondition`, the theorem means
/// both floating results are finite and
/// `abs(actual - reference) <= absolute + relative * abs(reference)`.
/// Exceptional values therefore require a false precondition or a separate
/// exact-bit relation; they cannot satisfy this finite-error claim.
///
/// Construction validates only the closed claim shape. Authority comes from a
/// receipt whose obligation digest binds this contract to the complete ranked
/// graph and exact MIR subjects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionNumericalRefinementContractV2 {
    contract_identity: u64,
    actual: ProductionRankedValueV1,
    reference: ProductionRankedValueV1,
    domain: ProductionRankedValueV1,
    precondition: ProductionRankedValueV1,
    absolute_error_f64_bits: u64,
    relative_error_f64_bits: u64,
}

impl ProductionNumericalRefinementContractV2 {
    pub fn new(
        contract_identity: u64,
        actual: ProductionRankedValueV1,
        reference: ProductionRankedValueV1,
        domain: ProductionRankedValueV1,
        precondition: ProductionRankedValueV1,
        absolute_error_f64_bits: u64,
        relative_error_f64_bits: u64,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        let absolute = f64::from_bits(absolute_error_f64_bits);
        let relative = f64::from_bits(relative_error_f64_bits);
        if contract_identity == 0
            || !absolute.is_finite()
            || !relative.is_finite()
            || absolute < 0.0
            || relative < 0.0
            || (absolute == 0.0 && relative == 0.0)
        {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        Ok(Self {
            contract_identity,
            actual,
            reference,
            domain,
            precondition,
            absolute_error_f64_bits,
            relative_error_f64_bits,
        })
    }

    pub const fn contract_identity(self) -> u64 {
        self.contract_identity
    }
    pub const fn actual(self) -> ProductionRankedValueV1 {
        self.actual
    }
    pub const fn reference(self) -> ProductionRankedValueV1 {
        self.reference
    }
    pub const fn domain(self) -> ProductionRankedValueV1 {
        self.domain
    }
    pub const fn precondition(self) -> ProductionRankedValueV1 {
        self.precondition
    }
    pub const fn absolute_error_f64_bits(self) -> u64 {
        self.absolute_error_f64_bits
    }
    pub const fn relative_error_f64_bits(self) -> u64 {
        self.relative_error_f64_bits
    }

    pub fn request_shape_hash(self) -> DigestV1 {
        let mut writer = CanonicalRefinementDigestV2::new(NUMERICAL_REFINEMENT_CONTRACT_DOMAIN_V2);
        writer.field(1, &self.contract_identity.to_le_bytes());
        for (tag, value) in [
            (2, self.actual),
            (3, self.reference),
            (4, self.domain),
            (5, self.precondition),
        ] {
            writer.value(tag, value);
        }
        writer.field(6, &self.absolute_error_f64_bits.to_le_bytes());
        writer.field(7, &self.relative_error_f64_bits.to_le_bytes());
        writer.finish()
    }
}

/// Stable location of one live cooperative-tensor instruction in ranked IR.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProductionTensorInstructionSiteV1 {
    block: u32,
    operation: u32,
}

impl ProductionTensorInstructionSiteV1 {
    pub const fn new(block: u32, operation: u32) -> Self {
        Self { block, operation }
    }
    pub const fn block(self) -> u32 {
        self.block
    }
    pub const fn operation(self) -> u32 {
        self.operation
    }
}

/// Ordered binding from one tensor result component to one exact output write.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionTensorResultComponentV1 {
    component: u16,
    store_site: ProductionGpuWriteSiteV2,
    indices: Vec<ProductionRankedValueV1>,
    gpu_value: ProductionRankedValueV1,
    reference_value: ProductionRankedValueV1,
}

impl ProductionTensorResultComponentV1 {
    pub fn new(
        component: u16,
        store_site: ProductionGpuWriteSiteV2,
        indices: Vec<ProductionRankedValueV1>,
        gpu_value: ProductionRankedValueV1,
        reference_value: ProductionRankedValueV1,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        if indices.is_empty() || indices.len() > MAX_PRODUCTION_RANKED_EFFECT_INDICES_V2 {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        Ok(Self {
            component,
            store_site,
            indices,
            gpu_value,
            reference_value,
        })
    }

    pub const fn component(&self) -> u16 {
        self.component
    }
    pub const fn store_site(&self) -> ProductionGpuWriteSiteV2 {
        self.store_site
    }
    pub fn indices(&self) -> &[ProductionRankedValueV1] {
        &self.indices
    }
    pub const fn gpu_value(&self) -> ProductionRankedValueV1 {
        self.gpu_value
    }
    pub const fn reference_value(&self) -> ProductionRankedValueV1 {
        self.reference_value
    }
}

/// Claim-specific functional composition of one cooperative tensor instruction.
///
/// The supported V1 subset is deliberately finite: one live tensor instruction,
/// all of its declared result components, one logical output view, and exact
/// component stores. The claim states that the ordered component pair at ordinal
/// `i` is the scalar extraction `i` of `tensor_result_root`, that its GPU member
/// is the value consumed by the named store at the named output coordinate, and
/// that the ordered product of all pairs refines the aggregate `actual` to the
/// aggregate `reference` under `numerical_contract`. Construction grants no
/// authority. Production admission requires an independently imported receipt
/// over the complete ranked graph and this exact contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionTensorRefinementContractV1 {
    contract_identity: u64,
    tensor_site: ProductionTensorInstructionSiteV1,
    tensor_result_root: DigestV1,
    output_view: ProductionRankedValueV1,
    actual: ProductionRankedValueV1,
    reference: ProductionRankedValueV1,
    component_scalar: ProductionSemanticScalarTypeV2,
    numerical_contract: ProductionNumericalContractV2,
    components: Vec<ProductionTensorResultComponentV1>,
}

impl ProductionTensorRefinementContractV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        contract_identity: u64,
        tensor_site: ProductionTensorInstructionSiteV1,
        tensor_result_root: DigestV1,
        output_view: ProductionRankedValueV1,
        actual: ProductionRankedValueV1,
        reference: ProductionRankedValueV1,
        component_scalar: ProductionSemanticScalarTypeV2,
        numerical_contract: ProductionNumericalContractV2,
        components: Vec<ProductionTensorResultComponentV1>,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        if contract_identity == 0
            || tensor_result_root.is_zero()
            || components.is_empty()
            || components.len() > MAX_PRODUCTION_TENSOR_COMPONENTS_V1
            || !numerical_contract.is_supported()
            || !numerical_contract.admits_scalar(component_scalar)
        {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        let canonical_components = components
            .iter()
            .enumerate()
            .all(|(index, component)| usize::from(component.component) == index);
        let unique_stores = components
            .iter()
            .map(|component| component.store_site)
            .collect::<BTreeSet<_>>()
            .len()
            == components.len();
        let unique_gpu_values = components
            .iter()
            .map(|component| component.gpu_value)
            .collect::<BTreeSet<_>>()
            .len()
            == components.len();
        if !canonical_components || !unique_stores || !unique_gpu_values {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        Ok(Self {
            contract_identity,
            tensor_site,
            tensor_result_root,
            output_view,
            actual,
            reference,
            component_scalar,
            numerical_contract,
            components,
        })
    }

    pub const fn contract_identity(&self) -> u64 {
        self.contract_identity
    }
    pub const fn tensor_site(&self) -> ProductionTensorInstructionSiteV1 {
        self.tensor_site
    }
    pub const fn tensor_result_root(&self) -> DigestV1 {
        self.tensor_result_root
    }
    pub const fn output_view(&self) -> ProductionRankedValueV1 {
        self.output_view
    }
    pub const fn actual(&self) -> ProductionRankedValueV1 {
        self.actual
    }
    pub const fn reference(&self) -> ProductionRankedValueV1 {
        self.reference
    }
    pub const fn component_scalar(&self) -> ProductionSemanticScalarTypeV2 {
        self.component_scalar
    }
    pub const fn numerical_contract(&self) -> ProductionNumericalContractV2 {
        self.numerical_contract
    }
    pub fn components(&self) -> &[ProductionTensorResultComponentV1] {
        &self.components
    }
}

/// Derives the exact scalar-refinement transcript digest from validated recipe DAGs.
pub fn normalized_functional_refinement_formula_hash_for_kernel_v2(
    kernel: &ProductionRankedKernelV1,
    block_index: usize,
    operation_index: usize,
    actual: ProductionRankedValueV1,
    expected: ProductionRankedValueV1,
    subjects: FunctionalRefinementSubjectsV2,
) -> Result<DigestV1, ProductionRankedKernelErrorV1> {
    let mut writer = CanonicalRefinementDigestV2::new(FUNCTIONAL_REFINEMENT_FORMULA_DOMAIN_V2);
    writer.kernel_header(kernel, block_index, operation_index, subjects);
    writer.field(
        12,
        &super::middle_end_evidence_v4::derive_functional_refinement_graph_identity_v2(kernel),
    );
    writer.value(20, actual);
    writer.value(21, expected);
    Ok(writer.finish())
}

/// Derives the exact effect-refinement transcript digest from the validated
/// recipe DAG, correlated write, view/allocation, and ownership contract.
pub fn normalized_effect_refinement_hash_for_kernel_v2(
    kernel: &ProductionRankedKernelV1,
    block_index: usize,
    operation_index: usize,
    contract: &ProductionEffectRefinementContractV2,
    subjects: FunctionalRefinementSubjectsV2,
) -> Result<DigestV1, ProductionRankedKernelErrorV1> {
    let mut writer = CanonicalRefinementDigestV2::new(EFFECT_REFINEMENT_CONTRACT_DOMAIN_V2);
    writer.kernel_header(kernel, block_index, operation_index, subjects);
    writer.field(
        12,
        &super::middle_end_evidence_v4::derive_functional_refinement_graph_identity_v2(kernel),
    );
    writer.field(20, &contract.contract_identity.to_le_bytes());
    writer.field(21, &contract.gpu_write_site.block.to_le_bytes());
    writer.field(22, &contract.gpu_write_site.operation.to_le_bytes());
    writer.field(23, &contract.reference_output_site.argument.to_le_bytes());
    writer.field(24, &contract.reference_output_site.block.to_le_bytes());
    writer.field(25, &contract.reference_output_site.statement.to_le_bytes());
    writer.value(26, contract.view);
    for (index, value) in contract.indices.iter().copied().enumerate() {
        writer.value(30 + index as u16, value);
    }
    for (index, value) in contract.gpu_coordinates.iter().copied().enumerate() {
        writer.value(50 + index as u16, value);
    }
    for (index, value) in contract.reference_coordinates.iter().copied().enumerate() {
        writer.value(70 + index as u16, value);
    }
    for (tag, value) in [
        (90, contract.gpu_domain),
        (91, contract.reference_domain),
        (92, contract.gpu_precondition),
        (93, contract.reference_precondition),
        (94, contract.gpu_value),
        (95, contract.reference_value),
    ] {
        writer.value(tag, value);
    }

    let mut writes = Vec::new();
    let mut ownership = Vec::new();
    let mut unmodeled_matching_write = false;
    for (candidate_block, block) in kernel.blocks.iter().enumerate() {
        for (candidate_operation, operation) in block.operations.iter().enumerate() {
            match operation {
                ProductionRankedOperationV1::Access {
                    kind,
                    view,
                    indices,
                } if kind.writes_memory()
                    && *view == contract.view
                    && indices == &contract.indices =>
                {
                    unmodeled_matching_write = true;
                }
                ProductionRankedOperationV1::PredicatedAccess {
                    kind, view, index, ..
                } if kind.writes_memory()
                    && *view == contract.view
                    && [*index].as_slice() == contract.indices.as_slice() =>
                {
                    unmodeled_matching_write = true;
                }
                ProductionRankedOperationV1::ValueAccess {
                    kind,
                    view,
                    indices,
                    value,
                } if kind.writes_memory()
                    && *view == contract.view
                    && indices == &contract.indices
                    && *value == contract.gpu_value =>
                {
                    writes.push((
                        candidate_block,
                        candidate_operation,
                        access_kind_tag(*kind),
                        0,
                        0,
                    ));
                }
                ProductionRankedOperationV1::AtomicAccess {
                    kind,
                    ordering,
                    scope,
                    view,
                    indices,
                } if kind.writes_memory()
                    && *view == contract.view
                    && indices == &contract.indices =>
                {
                    unmodeled_matching_write = true;
                }
                ProductionRankedOperationV1::AtomicValueAccess {
                    kind,
                    ordering,
                    scope,
                    view,
                    indices,
                    value,
                } if kind.writes_memory()
                    && *view == contract.view
                    && indices == &contract.indices
                    && *value == contract.gpu_value =>
                {
                    writes.push((
                        candidate_block,
                        candidate_operation,
                        access_kind_tag(*kind),
                        atomic_ordering_tag(*ordering),
                        atomic_scope_tag(*scope),
                    ));
                }
                ProductionRankedOperationV1::OwnershipContract {
                    view,
                    coverage,
                    partition,
                } if *view == contract.view => {
                    ownership.push((
                        candidate_block,
                        candidate_operation,
                        ownership_coverage_tag(*coverage),
                        ownership_partition_tag(*partition),
                    ));
                }
                _ => {}
            }
        }
    }
    if unmodeled_matching_write || writes.len() != 1 || ownership.len() != 1 {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    }
    let write = writes[0];
    if write.0 != contract.gpu_write_site.block as usize
        || write.1 != contract.gpu_write_site.operation as usize
    {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    }
    for (tag, value) in [
        (80, write.0 as u64),
        (81, write.1 as u64),
        (82, u64::from(write.2)),
        (83, u64::from(write.3)),
        (84, u64::from(write.4)),
    ] {
        writer.field(tag, &value.to_le_bytes());
    }
    let owner = ownership[0];
    for (tag, value) in [
        (90, owner.0 as u64),
        (91, owner.1 as u64),
        (92, u64::from(owner.2)),
        (93, u64::from(owner.3)),
    ] {
        writer.field(tag, &value.to_le_bytes());
    }
    Ok(writer.finish())
}

/// Derives the exact claim-specific cooperative-tensor obligation from the
/// complete ranked graph. This function proves no arithmetic: its digest is the
/// statement authenticated by an independently imported receipt.
pub fn normalized_tensor_refinement_hash_for_kernel_v1(
    kernel: &ProductionRankedKernelV1,
    block_index: usize,
    operation_index: usize,
    contract: &ProductionTensorRefinementContractV1,
    subjects: FunctionalRefinementSubjectsV2,
) -> Result<DigestV1, ProductionRankedKernelErrorV1> {
    let mut tensor_sites = Vec::new();
    let mut component_definitions = Vec::new();
    let mut writes = Vec::new();
    let mut ownership = Vec::new();
    for (candidate_block, block) in kernel.blocks.iter().enumerate() {
        for (candidate_operation, operation) in block.operations.iter().enumerate() {
            match operation {
                ProductionRankedOperationV1::TensorLayout {
                    contract,
                    convergence,
                    active_lanes,
                    binding,
                } => tensor_sites.push((
                    candidate_block,
                    candidate_operation,
                    contract,
                    *convergence,
                    *active_lanes,
                    *binding,
                )),
                ProductionRankedOperationV1::TensorResultComponent {
                    result,
                    tensor_result_root,
                    component,
                    scalar,
                    numerical_contract,
                } => component_definitions.push((
                    ProductionRankedValueV1::Local(*result),
                    *tensor_result_root,
                    *component,
                    *scalar,
                    *numerical_contract,
                )),
                ProductionRankedOperationV1::ValueAccess {
                    kind,
                    view,
                    indices,
                    value,
                } if kind.writes_memory() && *view == contract.output_view => {
                    writes.push((
                        candidate_block,
                        candidate_operation,
                        *kind,
                        indices.clone(),
                        *value,
                    ));
                }
                ProductionRankedOperationV1::Access { kind, view, .. }
                | ProductionRankedOperationV1::AtomicAccess { kind, view, .. }
                    if kind.writes_memory() && *view == contract.output_view =>
                {
                    return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
                }
                ProductionRankedOperationV1::PredicatedAccess { kind, view, .. }
                    if kind.writes_memory() && *view == contract.output_view =>
                {
                    return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
                }
                ProductionRankedOperationV1::AtomicValueAccess { kind, view, .. }
                    if kind.writes_memory() && *view == contract.output_view =>
                {
                    return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
                }
                ProductionRankedOperationV1::OwnershipContract {
                    view,
                    coverage,
                    partition,
                } if *view == contract.output_view => {
                    ownership.push((*coverage, *partition));
                }
                _ => {}
            }
        }
    }
    let bound_result_roots = tensor_sites
        .iter()
        .filter_map(|(_, _, _, _, _, binding)| binding.map(|binding| binding.result_root()))
        .collect::<Vec<_>>();
    if bound_result_roots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len()
        != bound_result_roots.len()
    {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    }
    let matching_tensor_sites = tensor_sites
        .iter()
        .filter(|(block, operation, ..)| {
            *block == contract.tensor_site.block as usize
                && *operation == contract.tensor_site.operation as usize
        })
        .collect::<Vec<_>>();
    let [(_tensor_block, _tensor_operation, layout, convergence, active_lanes, Some(binding))] =
        matching_tensor_sites.as_slice()
    else {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    };
    if binding.result_root() != contract.tensor_result_root
        || tensor_layout_result_scalar_v1(layout) != Some(contract.component_scalar)
        || usize::from(layout.accumulator.fragment_elements) != contract.components.len()
        || component_definitions
            .iter()
            .filter(|(_, result_root, ..)| *result_root == contract.tensor_result_root)
            .count()
            != contract.components.len()
        || writes.len() != contract.components.len()
        || ownership.as_slice()
            != [(
                OwnershipCoverageAttr::TotalView,
                OwnershipPartitionAttr::ExactSets,
            )]
    {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    }
    for component in &contract.components {
        let matching_definition = component_definitions
            .iter()
            .filter(|(value, result_root, ordinal, scalar, numerical)| {
                *value == component.gpu_value
                    && *result_root == contract.tensor_result_root
                    && *ordinal == component.component
                    && *scalar == contract.component_scalar
                    && *numerical == contract.numerical_contract
            })
            .count();
        let matching = writes
            .iter()
            .filter(|(block, operation, kind, indices, value)| {
                *block == component.store_site.block as usize
                    && *operation == component.store_site.operation as usize
                    && *kind == AccessKindAttr::Write
                    && indices.as_slice() == component.indices.as_slice()
                    && *value == component.gpu_value
            })
            .count();
        if matching_definition != 1 || matching != 1 {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
    }

    let mut writer = CanonicalRefinementDigestV2::new(TENSOR_REFINEMENT_CONTRACT_DOMAIN_V1);
    writer.kernel_header(kernel, block_index, operation_index, subjects);
    writer.field(
        12,
        &super::middle_end_evidence_v4::derive_functional_refinement_graph_identity_v2(kernel),
    );
    writer.field(20, &contract.contract_identity.to_le_bytes());
    writer.field(21, &contract.tensor_site.block.to_le_bytes());
    writer.field(22, &contract.tensor_site.operation.to_le_bytes());
    writer.field(23, contract.tensor_result_root.as_bytes());
    writer.value(24, contract.output_view);
    writer.value(25, contract.actual);
    writer.value(26, contract.reference);
    let mut scalar = vec![];
    match contract.component_scalar {
        ProductionSemanticScalarTypeV2::Bool => scalar.push(1),
        ProductionSemanticScalarTypeV2::Integer { signed, bits } => {
            scalar.extend([2, u8::from(signed)]);
            scalar.extend(bits.to_le_bytes());
        }
        ProductionSemanticScalarTypeV2::Float { bits } => {
            scalar.push(3);
            scalar.extend(bits.to_le_bytes());
        }
    }
    writer.field(38, &scalar);
    let mut numerical = vec![];
    match contract.numerical_contract {
        ProductionNumericalContractV2::ExactBitVectorOperatorCongruence => numerical.push(1),
        ProductionNumericalContractV2::ExactIeee754OperatorCongruence {
            rounding,
            exceptional_values,
        } => numerical.extend([2, rounding as u8, exceptional_values as u8]),
        ProductionNumericalContractV2::ErrorBounded {
            absolute_error_f64_bits,
            relative_error_f64_bits,
        } => {
            numerical.push(3);
            numerical.extend(absolute_error_f64_bits.to_le_bytes());
            numerical.extend(relative_error_f64_bits.to_le_bytes());
        }
        ProductionNumericalContractV2::Relaxed => numerical.push(4),
    }
    writer.field(39, &numerical);

    let mut layout_digest = Sha256::new();
    super::middle_end_evidence_v4::hash_tensor_layout_contract(&mut layout_digest, layout);
    writer.field(27, &layout_digest.finalize());
    writer.field(
        28,
        &[match convergence {
            TensorConvergenceAttr::UniformSubgroup => 1,
            TensorConvergenceAttr::Divergent => 2,
            TensorConvergenceAttr::UniformWorkgroup => 3,
            TensorConvergenceAttr::Opaque => 4,
        }],
    );
    writer.field(29, &active_lanes.to_le_bytes());
    for (tag, root) in [
        (30, binding.context_root()),
        (31, binding.lane_root()),
        (32, binding.lhs_root()),
        (33, binding.rhs_root()),
        (34, binding.accumulator_root()),
        (35, binding.result_root()),
    ] {
        writer.field(tag, root.as_bytes());
    }
    writer.field(36, &binding.argument_count().to_le_bytes());
    writer.field(37, &(contract.components.len() as u64).to_le_bytes());
    for (index, component) in contract.components.iter().enumerate() {
        let base = 100
            + u16::try_from(index)
                .map_err(|_| ProductionRankedKernelErrorV1::InvalidReferenceContract)?
                * 8;
        writer.field(base, &component.component.to_le_bytes());
        writer.field(base + 1, &component.store_site.block.to_le_bytes());
        writer.field(base + 2, &component.store_site.operation.to_le_bytes());
        writer.values(base + 3, &component.indices);
        writer.value(base + 4, component.gpu_value);
        writer.value(base + 5, component.reference_value);
    }
    Ok(writer.finish())
}

fn tensor_layout_result_scalar_v1(
    layout: &TensorLayoutContractV1,
) -> Option<ProductionSemanticScalarTypeV2> {
    match layout.accumulator.element {
        MatrixElement::F32 => Some(ProductionSemanticScalarTypeV2::Float { bits: 32 }),
        // The semantic scalar model has no BF16 kind yet. Keep such result
        // layouts closed rather than conflating BF16 with IEEE binary16.
        MatrixElement::Bf16 | MatrixElement::Fp4E2M1 | MatrixElement::Fp8E4M3 => None,
    }
}

/// Derives the exact finite-error obligation from the complete ranked graph.
pub fn normalized_numerical_refinement_hash_for_kernel_v2(
    kernel: &ProductionRankedKernelV1,
    block_index: usize,
    operation_index: usize,
    contract: ProductionNumericalRefinementContractV2,
    subjects: FunctionalRefinementSubjectsV2,
) -> Result<DigestV1, ProductionRankedKernelErrorV1> {
    let mut writer = CanonicalRefinementDigestV2::new(NUMERICAL_REFINEMENT_CONTRACT_DOMAIN_V2);
    writer.kernel_header(kernel, block_index, operation_index, subjects);
    writer.field(
        12,
        &super::middle_end_evidence_v4::derive_functional_refinement_graph_identity_v2(kernel),
    );
    writer.field(20, &contract.contract_identity.to_le_bytes());
    for (tag, value) in [
        (21, contract.actual),
        (22, contract.reference),
        (23, contract.domain),
        (24, contract.precondition),
    ] {
        writer.value(tag, value);
    }
    writer.field(25, &contract.absolute_error_f64_bits.to_le_bytes());
    writer.field(26, &contract.relative_error_f64_bits.to_le_bytes());
    Ok(writer.finish())
}

struct CanonicalRefinementDigestV2(Sha256);

impl CanonicalRefinementDigestV2 {
    fn new(domain: &[u8]) -> Self {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        Self(digest)
    }

    fn field(&mut self, tag: u16, bytes: &[u8]) {
        self.0.update(tag.to_le_bytes());
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    fn kernel_header(
        &mut self,
        kernel: &ProductionRankedKernelV1,
        block_index: usize,
        operation_index: usize,
        subjects: FunctionalRefinementSubjectsV2,
    ) {
        self.field(1, b"2");
        self.field(2, kernel.function_name.as_bytes());
        self.field(3, &(kernel.argument_count as u64).to_le_bytes());
        self.field(4, &(block_index as u64).to_le_bytes());
        self.field(5, &(operation_index as u64).to_le_bytes());
        self.field(6, &[subjects.safe_reference_kind() as u8]);
        for (tag, digest) in [
            (7, subjects.safe_reference_identity()),
            (8, subjects.safe_reference_source_hash()),
            (9, subjects.safe_reference_mir_hash()),
            (10, subjects.kernel_subject_identity()),
            (11, subjects.kernel_mir_hash()),
        ] {
            self.field(tag, digest.as_bytes());
        }
    }

    fn value(&mut self, tag: u16, value: ProductionRankedValueV1) {
        let mut bytes = Vec::with_capacity(9);
        match value {
            ProductionRankedValueV1::Argument(index) => {
                bytes.push(1);
                bytes.extend_from_slice(&index.to_le_bytes());
            }
            ProductionRankedValueV1::BlockArgument { block, argument } => {
                bytes.push(2);
                bytes.extend_from_slice(&block.to_le_bytes());
                bytes.extend_from_slice(&argument.to_le_bytes());
            }
            ProductionRankedValueV1::Local(identity) => {
                bytes.push(3);
                bytes.extend_from_slice(&identity.get().to_le_bytes());
            }
        }
        self.field(tag, &bytes);
    }

    fn values(&mut self, tag: u16, values: &[ProductionRankedValueV1]) {
        let mut bytes = Vec::with_capacity(8 + values.len() * 9);
        bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            let mut item = [0_u8; 9];
            match value {
                ProductionRankedValueV1::Argument(index) => {
                    item[0] = 1;
                    item[1..5].copy_from_slice(&index.to_le_bytes());
                }
                ProductionRankedValueV1::BlockArgument { block, argument } => {
                    item[0] = 2;
                    item[1..5].copy_from_slice(&block.to_le_bytes());
                    item[5..9].copy_from_slice(&argument.to_le_bytes());
                }
                ProductionRankedValueV1::Local(identity) => {
                    item[0] = 3;
                    item[1..5].copy_from_slice(&identity.get().to_le_bytes());
                }
            }
            bytes.extend_from_slice(&item);
        }
        self.field(tag, &bytes);
    }

    fn finish(self) -> DigestV1 {
        DigestV1::from_untrusted_bytes(self.0.finalize().into())
    }
}

fn access_kind_tag(kind: AccessKindAttr) -> u8 {
    match kind {
        AccessKindAttr::Read => 1,
        AccessKindAttr::Write => 2,
        AccessKindAttr::AtomicRead => 3,
        AccessKindAttr::AtomicWrite => 4,
        AccessKindAttr::AtomicReadModifyWrite => 5,
    }
}
fn ownership_coverage_tag(coverage: OwnershipCoverageAttr) -> u8 {
    match coverage {
        OwnershipCoverageAttr::ExactView => 1,
        OwnershipCoverageAttr::ExactEffectDomain => 2,
        OwnershipCoverageAttr::TotalView => 3,
        OwnershipCoverageAttr::CollectiveContributions => 4,
    }
}
fn ownership_partition_tag(partition: OwnershipPartitionAttr) -> u8 {
    match partition {
        OwnershipPartitionAttr::ExactSets => 1,
        OwnershipPartitionAttr::DenseRectangles => 2,
    }
}
fn atomic_ordering_tag(ordering: AtomicOrderingAttr) -> u8 {
    match ordering {
        AtomicOrderingAttr::Relaxed => 1,
        AtomicOrderingAttr::Acquire => 2,
        AtomicOrderingAttr::Release => 3,
        AtomicOrderingAttr::AcquireRelease => 4,
        AtomicOrderingAttr::SequentiallyConsistent => 5,
    }
}
fn atomic_scope_tag(scope: AtomicScopeAttr) -> u8 {
    match scope {
        AtomicScopeAttr::SingleThread => 1,
        AtomicScopeAttr::Workgroup => 2,
        AtomicScopeAttr::Agent => 3,
        AtomicScopeAttr::Device => 4,
        AtomicScopeAttr::System => 5,
    }
}

/// Non-authoritative policy-checked staging summary for one imported receipt.
///
/// Signature and caller-selected policy checks do not establish proof execution or
/// compiler authority. Only the private aggregate exact-formula replay may do so.
///
/// ```compile_fail
/// use fe2o3_pliron::ProductionPolicyCheckedRefinementStagingV2;
///
/// fn duplicate(evidence: ProductionPolicyCheckedRefinementStagingV2) {
///     let _first = evidence;
///     let _second = evidence;
/// }
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct ProductionPolicyCheckedRefinementStagingV2 {
    receipt_identity: FunctionalRefinementReceiptIdentityV2,
    binding: FunctionalRefinementBindingV2,
    signer_identity: DigestV1,
    toolchain: VerusToolchainIdentityV2,
    execution_identity: DigestV1,
    boundary: FunctionalRefinementBoundaryV2,
}

/// Caller-selected policy for non-authoritative receipt staging.
///
/// The policy is deliberately not a compiler trust root. Its constructor is exposed
/// only to the workspace verifier and hostile tests through `internal-proof-staging`.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg(feature = "internal-proof-staging")]
pub struct ProductionRefinementStagingPolicyV2 {
    signer_identities: BTreeSet<DigestV1>,
    toolchain: VerusToolchainIdentityV2,
}

#[cfg(feature = "internal-proof-staging")]
impl ProductionRefinementStagingPolicyV2 {
    pub fn new(
        signer_identities: impl IntoIterator<Item = DigestV1>,
        toolchain: VerusToolchainIdentityV2,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        let signer_identities = signer_identities.into_iter().collect::<BTreeSet<_>>();
        if signer_identities.is_empty()
            || signer_identities.len() > HARD_MAX_PRODUCTION_CONSTRUCTIONS
            || signer_identities.iter().any(|identity| identity.is_zero())
        {
            return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
        }
        Ok(Self {
            signer_identities,
            toolchain,
        })
    }

    pub fn accepts_signer(&self, signer: DigestV1) -> bool {
        self.signer_identities.contains(&signer)
    }

    pub const fn toolchain(&self) -> VerusToolchainIdentityV2 {
        self.toolchain
    }
}

impl ProductionPolicyCheckedRefinementStagingV2 {
    #[cfg(feature = "internal-proof-staging")]
    fn from_imported(proof: ImportedFunctionalRefinementProofV2) -> Self {
        Self {
            receipt_identity: proof.receipt_identity(),
            binding: proof.binding(),
            signer_identity: proof.signer_identity(),
            toolchain: proof.toolchain(),
            execution_identity: proof.execution_identity(),
            boundary: proof.boundary(),
        }
    }

    pub const fn receipt_identity(&self) -> FunctionalRefinementReceiptIdentityV2 {
        self.receipt_identity
    }
    pub const fn binding(&self) -> FunctionalRefinementBindingV2 {
        self.binding
    }
    pub const fn signer_identity(&self) -> DigestV1 {
        self.signer_identity
    }
    pub const fn toolchain(&self) -> VerusToolchainIdentityV2 {
        self.toolchain
    }
    pub const fn execution_identity(&self) -> DigestV1 {
        self.execution_identity
    }
    pub const fn boundary(&self) -> FunctionalRefinementBoundaryV2 {
        self.boundary
    }
    pub const fn is_policy_checked_untrusted_staging(&self) -> bool {
        true
    }
    pub const fn grants_source_to_isa_authority(&self) -> bool {
        false
    }
    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionRankedOperationV1 {
    /// Retained linear invocation-to-scope mapping used by mandatory
    /// concurrency analysis. It carries no launch authority.
    ExecutionLayout {
        grid_identity: u64,
        global_extents: [u64; 3],
        workgroup_extents: [u64; 3],
        subgroup_size: u64,
        full_physical_workgroups: bool,
    },
    View {
        result: ProductionRankedValueIdV1,
        element_width: u32,
        writable: bool,
        shape: Vec<u64>,
        dynamic_extents: Vec<ProductionRankedValueV1>,
        allocation_origin: u64,
        noalias_class: u64,
    },
    ViewInSpace {
        result: ProductionRankedValueIdV1,
        element_width: u32,
        writable: bool,
        shape: Vec<u64>,
        dynamic_extents: Vec<ProductionRankedValueV1>,
        memory_space: MemorySpaceAttr,
        allocation_origin: u64,
        noalias_class: u64,
    },
    /// Creates a workload-neutral staged-storage lifecycle over one workgroup view.
    PipelineCreate {
        result: ProductionRankedValueIdV1,
        view: ProductionRankedValueV1,
        buffers: u32,
        prefetch_distance: u32,
    },
    /// One exact epoch transition in a staged-storage lifecycle.
    PipelineEvent {
        pipeline: ProductionRankedValueV1,
        epoch: ProductionRankedValueV1,
        slot: ProductionRankedValueV1,
        kind: PipelineEventKindAttr,
    },
    IndexConstant {
        result: ProductionRankedValueIdV1,
        value: u64,
    },
    /// Explicit unsigned-width conversion for an exact ranked index value.
    ///
    /// The maximum is derived from `bit_width`; no caller-selected bound is
    /// retained. This recipe is authoritative only when independently
    /// reconciled with the authenticated semantic-MIR subject by the source
    /// projector.
    IndexUnsignedCast {
        result: ProductionRankedValueIdV1,
        source: ProductionRankedValueV1,
        bit_width: u16,
    },
    IndexUnknown {
        result: ProductionRankedValueIdV1,
    },
    InvocationIndex {
        result: ProductionRankedValueIdV1,
        dimension: u32,
        launch_extent: u64,
    },
    IndexBinary {
        result: ProductionRankedValueIdV1,
        kind: IndexBinaryKindAttr,
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
    },
    /// Abstract result of a source-authenticated total deterministic operation.
    ///
    /// The recipe retains dependencies only. It does not itself authenticate
    /// source semantics or grant compiler, artifact, or launch authority.
    DeterministicJoin {
        result: ProductionRankedValueIdV1,
        dependencies: Vec<ProductionRankedValueV1>,
    },
    CheckedTiledIndex2D {
        result: ProductionRankedValueIdV1,
        invocation: ProductionRankedValueV1,
        component: ProductionRankedValueV1,
        rows: ProductionRankedValueV1,
        columns: ProductionRankedValueV1,
        row_stride: ProductionRankedValueV1,
        lanes_per_tile: u64,
        tile_rows: u64,
        tile_columns: u64,
        elements_per_lane: u64,
    },
    CheckedRowStripedIndex2D {
        result: ProductionRankedValueIdV1,
        invocation: ProductionRankedValueV1,
        component: ProductionRankedValueV1,
        rows: ProductionRankedValueV1,
        columns: ProductionRankedValueV1,
        row_stride: ProductionRankedValueV1,
        lanes_per_row: u64,
        elements_per_lane: u64,
    },
    /// Structural predicated tiled mapping. `success` carries an obligation
    /// tying the mapping to the destination physical extent. The recipe shape
    /// alone grants no source, refinement, artifact, or launch authority.
    PredicatedCheckedTiledIndex2D {
        result: ProductionRankedValueIdV1,
        success: ProductionRankedValueIdV1,
        invocation: ProductionRankedValueV1,
        component: ProductionRankedValueV1,
        rows: ProductionRankedValueV1,
        columns: ProductionRankedValueV1,
        row_stride: ProductionRankedValueV1,
        physical_extent: ProductionRankedValueV1,
        lanes_per_tile: u64,
        tile_rows: u64,
        tile_columns: u64,
        elements_per_lane: u64,
    },
    /// Structural predicated row-striped mapping with no authority by itself.
    PredicatedCheckedRowStripedIndex2D {
        result: ProductionRankedValueIdV1,
        success: ProductionRankedValueIdV1,
        invocation: ProductionRankedValueV1,
        component: ProductionRankedValueV1,
        rows: ProductionRankedValueV1,
        columns: ProductionRankedValueV1,
        row_stride: ProductionRankedValueV1,
        physical_extent: ProductionRankedValueV1,
        lanes_per_row: u64,
        elements_per_lane: u64,
    },
    Dimension {
        result: ProductionRankedValueIdV1,
        view: ProductionRankedValueV1,
        dimension: u32,
    },
    Access {
        kind: AccessKindAttr,
        view: ProductionRankedValueV1,
        indices: Vec<ProductionRankedValueV1>,
    },
    /// One non-atomic access structurally paired with the checked mapping that
    /// produced `index` and `success`. This shape grants no authority.
    PredicatedAccess {
        kind: AccessKindAttr,
        view: ProductionRankedValueV1,
        index: ProductionRankedValueV1,
        success: ProductionRankedValueV1,
    },
    /// A non-atomic access whose exact semantic write RHS is retained.
    ///
    /// Functional-effect refinement accepts only this value-carrying form for
    /// writes, so a detached proof formula cannot stand in for the actual RHS.
    ValueAccess {
        kind: AccessKindAttr,
        view: ProductionRankedValueV1,
        indices: Vec<ProductionRankedValueV1>,
        value: ProductionRankedValueV1,
    },
    AtomicAccess {
        kind: AccessKindAttr,
        ordering: AtomicOrderingAttr,
        scope: AtomicScopeAttr,
        view: ProductionRankedValueV1,
        indices: Vec<ProductionRankedValueV1>,
    },
    /// An atomic access whose exact semantic write RHS is retained.
    AtomicValueAccess {
        kind: AccessKindAttr,
        ordering: AtomicOrderingAttr,
        scope: AtomicScopeAttr,
        view: ProductionRankedValueV1,
        indices: Vec<ProductionRankedValueV1>,
        value: ProductionRankedValueV1,
    },
    /// Requests a workload-neutral proof of write ownership across invocation,
    /// subgroup, workgroup, and grid scopes for one global output view.
    OwnershipContract {
        view: ProductionRankedValueV1,
        coverage: OwnershipCoverageAttr,
        partition: OwnershipPartitionAttr,
    },
    /// Conservative allocation-level memory effect with no claimed coordinate.
    AllocationEffect {
        kind: AccessKindAttr,
        memory_space: MemorySpaceAttr,
        allocation_origin: u64,
        noalias_class: u64,
    },
    Barrier {
        execution_scope: HierarchyAttr,
        memory_scope: MemoryScopeAttr,
        address_space: AddressSpaceAttr,
        order: MemoryOrderAttr,
    },
    Fence {
        memory_scope: MemoryScopeAttr,
        address_space: AddressSpaceAttr,
        order: MemoryOrderAttr,
    },
    /// One tensor-instruction site, not a free-standing proof annotation.
    ///
    /// The source projector must derive this declaration from an authenticated
    /// semantic terminal and its dominating operand producers. Merely adding
    /// this recipe operation never grants source-refinement or artifact authority.
    TensorLayout {
        contract: TensorLayoutContractV1,
        convergence: TensorConvergenceAttr,
        active_lanes: u32,
        binding: Option<ProductionCooperativeTensorBindingV1>,
    },
    /// Exact typed SSA extraction from one authenticated tensor result root.
    TensorResultComponent {
        result: ProductionRankedValueIdV1,
        tensor_result_root: DigestV1,
        component: u16,
        scalar: ProductionSemanticScalarTypeV2,
        numerical_contract: ProductionNumericalContractV2,
    },
    SemanticSymbol {
        result: ProductionRankedValueIdV1,
        symbol: u32,
    },
    SemanticConstant {
        result: ProductionRankedValueIdV1,
        value: u64,
    },
    SemanticBinary {
        result: ProductionRankedValueIdV1,
        kind: SemanticBinaryKindAttr,
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
    },
    /// One closed, typed expression independently projected from source MIR.
    SemanticExpression {
        result: ProductionRankedValueIdV1,
        expression: ProductionSemanticExpressionV2,
        numerical_contract: ProductionNumericalContractV2,
    },
    /// One finite fold, recurrence, or permutation/gather contract.
    ///
    /// Witness 0 is the identity/initial value/mapping and witness 1 is the
    /// operator/transition/inverse mapping according to `contract.kind()`.
    CollectiveSemantics {
        contract: ProductionCollectiveSemanticContractV1,
        view: ProductionRankedValueV1,
        actual: ProductionRankedValueV1,
        expected: ProductionRankedValueV1,
        witness0: ProductionRankedValueV1,
        witness1: ProductionRankedValueV1,
    },
    RequireEquivalent {
        actual: ProductionRankedValueV1,
        expected: ProductionRankedValueV1,
    },
    /// Requires semantic equality backed by an exact authenticated V2 receipt.
    /// The request itself is inert until the V2 production entrypoint consumes
    /// and reconciles the corresponding imported proof.
    RequireAuthenticatedReferenceEquivalent {
        actual: ProductionRankedValueV1,
        expected: ProductionRankedValueV1,
        proof: ProductionReferenceProofV2,
    },
    /// Generator input before exact proof execution/import. Production compile
    /// rejects this variant until the consuming bind transition replaces it.
    RequestAuthenticatedReferenceEquivalent {
        actual: ProductionRankedValueV1,
        expected: ProductionRankedValueV1,
        subjects: FunctionalRefinementSubjectsV2,
    },
    /// Joins authenticated MIR evidence to one normalized write effect.
    RequireEffectRefinement {
        contract: ProductionEffectRefinementContractV2,
        proof: ProductionReferenceProofV2,
    },
    /// Generator input for an effect contract before exact proof import.
    RequestEffectRefinement {
        contract: ProductionEffectRefinementContractV2,
        subjects: FunctionalRefinementSubjectsV2,
    },
    /// Requires one authenticated finite-error theorem over exact typed roots.
    RequireNumericalRefinement {
        contract: ProductionNumericalRefinementContractV2,
        proof: ProductionReferenceProofV2,
    },
    /// Import input for a claim-specific finite-error theorem.
    ///
    /// The generic Verus generator rejects this request because scalar
    /// equality does not prove guarded finiteness or the stated inequality.
    RequestNumericalRefinement {
        contract: ProductionNumericalRefinementContractV2,
        subjects: FunctionalRefinementSubjectsV2,
    },
    /// Requires one independently proved cooperative-tensor composition.
    RequireTensorRefinement {
        contract: ProductionTensorRefinementContractV1,
        proof: ProductionReferenceProofV2,
    },
    /// Import input for a claim-specific tensor theorem. The generic scalar
    /// proof generator deliberately does not synthesize this theorem.
    RequestTensorRefinement {
        contract: ProductionTensorRefinementContractV1,
        subjects: FunctionalRefinementSubjectsV2,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionRankedTerminatorV1 {
    IndexLessThan {
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
        true_block: u32,
        false_block: u32,
    },
    IndexLessThanArgs {
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
        true_arguments: Vec<ProductionRankedValueV1>,
        false_arguments: Vec<ProductionRankedValueV1>,
        true_block: u32,
        false_block: u32,
    },
    IndexEqual {
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
        true_block: u32,
        false_block: u32,
    },
    IndexEqualArgs {
        lhs: ProductionRankedValueV1,
        rhs: ProductionRankedValueV1,
        true_arguments: Vec<ProductionRankedValueV1>,
        false_arguments: Vec<ProductionRankedValueV1>,
        true_block: u32,
        false_block: u32,
    },
    AnalysisSplit {
        control_dependencies: Vec<ProductionRankedValueV1>,
        first_block: u32,
        second_block: u32,
    },
    AnalysisSplitArgs {
        control_dependencies: Vec<ProductionRankedValueV1>,
        first_arguments: Vec<ProductionRankedValueV1>,
        second_arguments: Vec<ProductionRankedValueV1>,
        first_block: u32,
        second_block: u32,
    },
    Branch {
        target: u32,
    },
    BranchArgs {
        arguments: Vec<ProductionRankedValueV1>,
        target: u32,
    },
    BranchArgsAdd {
        value: ProductionRankedValueV1,
        step: ProductionRankedValueV1,
        target: u32,
    },
    BranchArgsAddAt {
        arguments: Vec<ProductionRankedValueV1>,
        add_argument: u32,
        step: ProductionRankedValueV1,
        target: u32,
    },
    Return,
    Trap,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionRankedBlockV1 {
    index_argument_count: u32,
    operations: Vec<ProductionRankedOperationV1>,
    terminator: ProductionRankedTerminatorV1,
}

impl ProductionRankedBlockV1 {
    pub fn new(
        operations: Vec<ProductionRankedOperationV1>,
        terminator: ProductionRankedTerminatorV1,
    ) -> Self {
        Self {
            index_argument_count: 0,
            operations,
            terminator,
        }
    }

    pub fn with_index_arguments(
        index_argument_count: u32,
        operations: Vec<ProductionRankedOperationV1>,
        terminator: ProductionRankedTerminatorV1,
    ) -> Self {
        Self {
            index_argument_count,
            operations,
            terminator,
        }
    }

    pub const fn index_argument_count(&self) -> u32 {
        self.index_argument_count
    }

    pub fn operations(&self) -> &[ProductionRankedOperationV1] {
        &self.operations
    }

    pub const fn terminator(&self) -> &ProductionRankedTerminatorV1 {
        &self.terminator
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductionRankedKernelV1 {
    function_name: String,
    argument_count: usize,
    blocks: Vec<ProductionRankedBlockV1>,
    tree_work: usize,
}

impl ProductionRankedKernelV1 {
    pub fn new(
        function_name: &str,
        argument_count: usize,
        blocks: Vec<ProductionRankedBlockV1>,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        validate_name(function_name, NameKind::Dialect)
            .map_err(ProductionRankedKernelErrorV1::InvalidFunctionName)?;
        let mut kernel = Self {
            function_name: function_name.to_owned(),
            argument_count,
            blocks,
            tree_work: 0,
        };
        kernel.tree_work = kernel.validate()?;
        kernel = ranked_index_constant_fold_v1::fold_and_validate_index_constants_v1(kernel)
            .map_err(ProductionRankedKernelErrorV1::InvalidTransformation)?;
        let transformed_tree_work = kernel.validate()?;
        if transformed_tree_work != kernel.tree_work {
            return Err(ProductionRankedKernelErrorV1::InvalidTransformation(
                ProductionRankedTranslationErrorV1::TreeWorkChanged,
            ));
        }
        Ok(kernel)
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }

    pub const fn argument_count(&self) -> usize {
        self.argument_count
    }

    pub fn blocks(&self) -> &[ProductionRankedBlockV1] {
        &self.blocks
    }

    /// Consumes one validated generator request and replaces only the exact
    /// addressed unbound operation with its imported receipt request.
    pub fn bind_functional_refinement_request_v2(
        mut self,
        block_index: usize,
        operation_index: usize,
        proof: ProductionReferenceProofV2,
    ) -> Result<Self, ProductionRankedKernelErrorV1> {
        let operation = self
            .blocks
            .get_mut(block_index)
            .and_then(|block| block.operations.get_mut(operation_index))
            .ok_or(ProductionRankedKernelErrorV1::InvalidReferenceContract)?;
        let replacement = match operation {
            ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent {
                actual,
                expected,
                subjects,
            } if *subjects == proof.binding().subjects() => {
                ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent {
                    actual: *actual,
                    expected: *expected,
                    proof,
                }
            }
            ProductionRankedOperationV1::RequestEffectRefinement { contract, subjects }
                if *subjects == proof.binding().subjects() =>
            {
                ProductionRankedOperationV1::RequireEffectRefinement {
                    contract: contract.clone(),
                    proof,
                }
            }
            ProductionRankedOperationV1::RequestNumericalRefinement { contract, subjects }
                if *subjects == proof.binding().subjects() =>
            {
                ProductionRankedOperationV1::RequireNumericalRefinement {
                    contract: *contract,
                    proof,
                }
            }
            ProductionRankedOperationV1::RequestTensorRefinement { contract, subjects }
                if *subjects == proof.binding().subjects() =>
            {
                ProductionRankedOperationV1::RequireTensorRefinement {
                    contract: contract.clone(),
                    proof,
                }
            }
            _ => return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract),
        };
        *operation = replacement;
        self.tree_work = self.validate()?;
        Ok(self)
    }

    fn validate(&self) -> Result<usize, ProductionRankedKernelErrorV1> {
        if self.argument_count > HARD_MAX_PRODUCTION_RANKED_ARGUMENTS {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "function argument",
                limit: HARD_MAX_PRODUCTION_RANKED_ARGUMENTS,
                actual: self.argument_count,
            });
        }
        if self.blocks.is_empty() || self.blocks.len() > MAX_RANKED_BOUNDS_BLOCKS {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "basic block",
                limit: MAX_RANKED_BOUNDS_BLOCKS,
                actual: self.blocks.len(),
            });
        }
        let tensor_sites = self
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|operation| {
                matches!(operation, ProductionRankedOperationV1::TensorLayout { .. })
            })
            .count();
        let tensor_claims = self
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|operation| {
                matches!(
                    operation,
                    ProductionRankedOperationV1::RequireTensorRefinement { .. }
                        | ProductionRankedOperationV1::RequestTensorRefinement { .. }
                )
            })
            .count();
        validate_tensor_refinement_resource_counts_v1(tensor_sites, tensor_claims)?;
        for expression in self.blocks.iter().flat_map(|block| {
            block
                .operations
                .iter()
                .filter_map(|operation| match operation {
                    ProductionRankedOperationV1::SemanticExpression { expression, .. } => {
                        Some(expression)
                    }
                    _ => None,
                })
        }) {
            expression
                .validate()
                .map_err(ProductionRankedKernelErrorV1::InvalidSemanticExpression)?;
            validate_live_semantic_loads(self, expression)?;
        }
        let operation_count = self.blocks.iter().try_fold(0_usize, |total, block| {
            let materialized = block
                .operations
                .iter()
                .try_fold(0_usize, |count, operation| {
                    count.checked_add(match operation {
                        ProductionRankedOperationV1::SemanticExpression { expression, .. } => {
                            expression.validate().ok()?.nodes.checked_add(1)?
                        }
                        _ if matches!(
                            operation,
                            ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent { .. }
                                | ProductionRankedOperationV1::RequireEffectRefinement { .. }
                                | ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent { .. }
                                | ProductionRankedOperationV1::RequestEffectRefinement { .. }
                                | ProductionRankedOperationV1::RequireNumericalRefinement { .. }
                                | ProductionRankedOperationV1::RequestNumericalRefinement { .. }
                                | ProductionRankedOperationV1::RequireTensorRefinement { .. }
                                | ProductionRankedOperationV1::RequestTensorRefinement { .. }
                        ) => 3,
                        _ => 1,
                    })
                })?;
            total
                .checked_add(materialized.checked_add(1)?)?
                .checked_add(usize::from(matches!(
                    block.terminator,
                    ProductionRankedTerminatorV1::BranchArgsAdd { .. }
                        | ProductionRankedTerminatorV1::BranchArgsAddAt { .. }
                )))
        });
        let Some(operation_count) = operation_count else {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "operation",
                limit: MAX_RANKED_BOUNDS_OPERATIONS,
                actual: usize::MAX,
            });
        };
        if operation_count > MAX_RANKED_BOUNDS_OPERATIONS {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "operation",
                limit: MAX_RANKED_BOUNDS_OPERATIONS,
                actual: operation_count,
            });
        }
        let tree_work = ranked_tree_work(self.blocks.len(), operation_count).ok_or(
            ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "operation tree work",
                limit: HARD_MAX_SESSION_OPERATION_TREE_ITEMS,
                actual: usize::MAX,
            },
        )?;
        if tree_work > HARD_MAX_SESSION_OPERATION_TREE_ITEMS {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource: "operation tree work",
                limit: HARD_MAX_SESSION_OPERATION_TREE_ITEMS,
                actual: tree_work,
            });
        }

        let mut locals = Vec::new();
        let mut local_definition_blocks = Vec::new();
        let mut saw_execution_layout = false;
        let mut allocation_classes = HashMap::new();
        let mut predicated_success_uses: BTreeMap<ProductionRankedValueIdV1, usize> =
            BTreeMap::new();
        let mut predicated_indices = BTreeMap::new();
        let mut total_block_arguments = 0_usize;
        for (block_index, block) in self.blocks.iter().enumerate() {
            if block_index == 0 && block.index_argument_count != 0
                || block.index_argument_count as usize > HARD_MAX_PRODUCTION_RANKED_ARGUMENTS
            {
                return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                    resource: "block argument",
                    limit: HARD_MAX_PRODUCTION_RANKED_ARGUMENTS,
                    actual: block.index_argument_count as usize,
                });
            }
            total_block_arguments = total_block_arguments
                .checked_add(block.index_argument_count as usize)
                .ok_or(ProductionRankedKernelErrorV1::ResourceLimit {
                    resource: "total block argument",
                    limit: MAX_RANKED_BOUNDS_OPERATIONS,
                    actual: usize::MAX,
                })?;
            if total_block_arguments > MAX_RANKED_BOUNDS_OPERATIONS {
                return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                    resource: "total block argument",
                    limit: MAX_RANKED_BOUNDS_OPERATIONS,
                    actual: total_block_arguments,
                });
            }
            for (operation_index, operation) in block.operations.iter().enumerate() {
                validate_scoped_operation_values_v1(
                    operation,
                    block_index,
                    &self.blocks,
                    &local_definition_blocks,
                )?;
                if let ProductionRankedOperationV1::ExecutionLayout {
                    global_extents,
                    workgroup_extents,
                    subgroup_size,
                    full_physical_workgroups,
                    ..
                } = operation
                {
                    let workgroup_size = workgroup_extents
                        .iter()
                        .try_fold(1_u64, |volume, extent| volume.checked_mul(*extent));
                    if block_index != 0
                        || operation_index != 0
                        || saw_execution_layout
                        || workgroup_extents.contains(&0)
                        || workgroup_size.is_none()
                        || *subgroup_size == 0
                        || workgroup_size.is_some_and(|size| *subgroup_size > size)
                        || workgroup_size.is_some_and(|size| !size.is_multiple_of(*subgroup_size))
                        || (*full_physical_workgroups
                            && global_extents.iter().zip(workgroup_extents).any(
                                |(global, workgroup)| {
                                    *global != 0 && !global.is_multiple_of(*workgroup)
                                },
                            ))
                    {
                        return Err(ProductionRankedKernelErrorV1::InvalidExecutionLayout);
                    }
                    saw_execution_layout = true;
                }
                if let ProductionRankedOperationV1::View {
                    allocation_origin,
                    noalias_class,
                    ..
                }
                | ProductionRankedOperationV1::ViewInSpace {
                    allocation_origin,
                    noalias_class,
                    ..
                }
                | ProductionRankedOperationV1::AllocationEffect {
                    allocation_origin,
                    noalias_class,
                    ..
                } = operation
                {
                    if *noalias_class != 0 && *allocation_origin == 0
                        || *allocation_origin != 0
                            && allocation_classes
                                .insert(*allocation_origin, *noalias_class)
                                .is_some_and(|previous| previous != *noalias_class)
                    {
                        return Err(ProductionRankedKernelErrorV1::InvalidAllocationContract);
                    }
                }
                let result = validate_operation(operation, self.argument_count, &locals)?;
                if let ProductionRankedOperationV1::Access { indices, .. }
                | ProductionRankedOperationV1::ValueAccess { indices, .. }
                | ProductionRankedOperationV1::AtomicAccess { indices, .. }
                | ProductionRankedOperationV1::AtomicValueAccess { indices, .. } = operation
                {
                    if let Some(index) = indices.iter().find_map(|value| {
                        let ProductionRankedValueV1::Local(index) = value else {
                            return None;
                        };
                        predicated_indices.contains_key(index).then_some(*index)
                    }) {
                        return Err(
                            ProductionRankedKernelErrorV1::InvalidPredicatedAccessIndexUse {
                                index,
                            },
                        );
                    }
                }
                if let ProductionRankedOperationV1::PredicatedAccess { success, .. } = operation {
                    let ProductionRankedValueV1::Local(success) = success else {
                        return Err(ProductionRankedKernelErrorV1::InvalidShape);
                    };
                    let uses = predicated_success_uses.get_mut(success).ok_or(
                        ProductionRankedKernelErrorV1::InvalidPredicatedAccessUse {
                            success: *success,
                            uses: 0,
                        },
                    )?;
                    *uses = uses.checked_add(1).ok_or(
                        ProductionRankedKernelErrorV1::ResourceLimit {
                            resource: "predicated success use",
                            limit: MAX_RANKED_BOUNDS_OPERATIONS,
                            actual: usize::MAX,
                        },
                    )?;
                }
                if let Some((identity, kind)) = result {
                    let expected = u32::try_from(locals.len()).map_err(|_| {
                        ProductionRankedKernelErrorV1::ResourceLimit {
                            resource: "local value",
                            limit: MAX_RANKED_BOUNDS_OPERATIONS,
                            actual: locals.len(),
                        }
                    })?;
                    if identity.get() != expected {
                        return Err(ProductionRankedKernelErrorV1::NonCanonicalValueId {
                            expected,
                            actual: identity.get(),
                        });
                    }
                    locals.push(kind);
                    local_definition_blocks.push(block_index);
                    let paired = match operation {
                        ProductionRankedOperationV1::PredicatedCheckedTiledIndex2D {
                            result,
                            success,
                            physical_extent,
                            ..
                        }
                        | ProductionRankedOperationV1::PredicatedCheckedRowStripedIndex2D {
                            result,
                            success,
                            physical_extent,
                            ..
                        } => Some((*result, *success, *physical_extent)),
                        _ => None,
                    };
                    if let Some((index, success, physical_extent)) = paired {
                        let expected = u32::try_from(locals.len()).map_err(|_| {
                            ProductionRankedKernelErrorV1::ResourceLimit {
                                resource: "local value",
                                limit: MAX_RANKED_BOUNDS_OPERATIONS,
                                actual: locals.len(),
                            }
                        })?;
                        if success.get() != expected {
                            return Err(ProductionRankedKernelErrorV1::NonCanonicalValueId {
                                expected,
                                actual: success.get(),
                            });
                        }
                        locals.push(RecipeValueKindV1::CheckedAccessSuccess {
                            index: ProductionRankedValueV1::Local(index),
                            physical_extent,
                        });
                        local_definition_blocks.push(block_index);
                        predicated_success_uses.insert(success, 0);
                        predicated_indices.insert(index, success);
                    }
                }
            }
            validate_terminator(
                &block.terminator,
                self.argument_count,
                &locals,
                &self.blocks,
                block_index,
                &local_definition_blocks,
            )?;
        }
        if let Some((success, uses)) = predicated_success_uses
            .into_iter()
            .find(|(_, uses)| *uses == 0)
        {
            return Err(ProductionRankedKernelErrorV1::InvalidPredicatedAccessUse {
                success,
                uses,
            });
        }
        Ok(tree_work)
    }
}

fn validate_tensor_refinement_resource_counts_v1(
    tensor_sites: usize,
    tensor_claims: usize,
) -> Result<(), ProductionRankedKernelErrorV1> {
    for (resource, actual) in [
        ("cooperative tensor instruction site", tensor_sites),
        ("cooperative tensor refinement claim", tensor_claims),
    ] {
        if actual > MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1 {
            return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                resource,
                limit: MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
                actual,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tensor_refinement_resource_tests {
    use super::*;

    #[test]
    fn tensor_site_and_claim_prescans_are_bounded_at_sixty_four() {
        assert_eq!(
            validate_tensor_refinement_resource_counts_v1(
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
            ),
            Ok(())
        );
        for (sites, claims, resource) in [
            (
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1 + 1,
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
                "cooperative tensor instruction site",
            ),
            (
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
                MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1 + 1,
                "cooperative tensor refinement claim",
            ),
        ] {
            assert_eq!(
                validate_tensor_refinement_resource_counts_v1(sites, claims),
                Err(ProductionRankedKernelErrorV1::ResourceLimit {
                    resource,
                    limit: MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1,
                    actual: MAX_PRODUCTION_TENSOR_REFINEMENT_SITES_V1 + 1,
                })
            );
        }
    }
}

#[cfg(test)]
mod unsigned_cast_recipe_tests {
    use super::*;

    fn cast(
        result: u32,
        source: ProductionRankedValueV1,
        bit_width: u16,
    ) -> ProductionRankedOperationV1 {
        ProductionRankedOperationV1::IndexUnsignedCast {
            result: ProductionRankedValueIdV1::new(result),
            source,
            bit_width,
        }
    }

    #[test]
    fn unsigned_cast_recipe_is_value_defining_and_width_closed() {
        let argument = ProductionRankedValueV1::Argument(0);
        assert!(
            ProductionRankedKernelV1::new(
                "valid_unsigned_cast",
                1,
                vec![ProductionRankedBlockV1::new(
                    vec![cast(0, argument, 32)],
                    ProductionRankedTerminatorV1::Return,
                )],
            )
            .is_ok()
        );

        assert_eq!(
            ProductionRankedKernelV1::new(
                "invalid_unsigned_cast",
                1,
                vec![ProductionRankedBlockV1::new(
                    vec![cast(0, argument, 7)],
                    ProductionRankedTerminatorV1::Return,
                )],
            ),
            Err(ProductionRankedKernelErrorV1::InvalidUnsignedCast),
        );

        assert!(
            ProductionRankedKernelV1::new(
                "block_local_unsigned_cast",
                1,
                vec![
                    ProductionRankedBlockV1::new(
                        vec![],
                        ProductionRankedTerminatorV1::Branch { target: 1 },
                    ),
                    ProductionRankedBlockV1::new(
                        vec![cast(0, argument, 32)],
                        ProductionRankedTerminatorV1::Return,
                    ),
                ],
            )
            .is_ok(),
        );

        for source in [
            ProductionRankedValueV1::BlockArgument {
                block: 1,
                argument: 0,
            },
            ProductionRankedValueV1::BlockArgument {
                block: 0,
                argument: 0,
            },
        ] {
            assert_eq!(
                ProductionRankedKernelV1::new(
                    "foreign_unsigned_cast_block_argument",
                    0,
                    vec![
                        ProductionRankedBlockV1::new(
                            vec![cast(0, source, 32)],
                            ProductionRankedTerminatorV1::Branch { target: 1 },
                        ),
                        ProductionRankedBlockV1::with_index_arguments(
                            1,
                            vec![],
                            ProductionRankedTerminatorV1::Return,
                        ),
                    ],
                ),
                Err(ProductionRankedKernelErrorV1::UndefinedValue(source)),
            );
        }
    }
}

fn validate_live_semantic_loads(
    kernel: &ProductionRankedKernelV1,
    expression: &ProductionSemanticExpressionV2,
) -> Result<(), ProductionRankedKernelErrorV1> {
    match expression {
        ProductionSemanticExpressionV2::Load(load) => {
            let operation = kernel
                .blocks()
                .get(load.block as usize)
                .and_then(|block| block.operations().get(load.operation as usize));
            let Some(ProductionRankedOperationV1::Access {
                kind,
                view,
                indices,
            }) = operation
            else {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            };
            if *kind != AccessKindAttr::Read
                || *view != load.view
                || indices.as_slice() != load.indices.as_ref()
            {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            let ProductionRankedValueV1::Local(view_identity) = load.view else {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            };
            let mut definitions = kernel
                .blocks()
                .iter()
                .flat_map(|block| block.operations())
                .filter_map(|operation| match operation {
                    ProductionRankedOperationV1::ViewInSpace {
                        result,
                        element_width,
                        allocation_origin,
                        memory_space,
                        ..
                    } if *result == view_identity => {
                        Some((*element_width, *allocation_origin, *memory_space))
                    }
                    ProductionRankedOperationV1::View {
                        result,
                        element_width,
                        allocation_origin,
                        ..
                    } if *result == view_identity => {
                        Some((*element_width, *allocation_origin, MemorySpaceAttr::Global))
                    }
                    _ => None,
                });
            let Some((element_width, allocation_origin, memory_space)) = definitions.next() else {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            };
            if definitions.next().is_some()
                || memory_space != MemorySpaceAttr::Global
                || allocation_origin != load.allocation_origin
                || element_width != u32::from(load.scalar.bit_width())
            {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            Ok(())
        }
        ProductionSemanticExpressionV2::Symbol { .. }
        | ProductionSemanticExpressionV2::Constant { .. } => Ok(()),
        ProductionSemanticExpressionV2::Unary { operand, .. }
        | ProductionSemanticExpressionV2::Cast { operand, .. } => {
            validate_live_semantic_loads(kernel, operand)
        }
        ProductionSemanticExpressionV2::Binary { lhs, rhs, .. }
        | ProductionSemanticExpressionV2::Compare { lhs, rhs, .. } => {
            validate_live_semantic_loads(kernel, lhs)?;
            validate_live_semantic_loads(kernel, rhs)
        }
        ProductionSemanticExpressionV2::Select {
            condition,
            when_true,
            when_false,
            ..
        } => {
            validate_live_semantic_loads(kernel, condition)?;
            validate_live_semantic_loads(kernel, when_true)?;
            validate_live_semantic_loads(kernel, when_false)
        }
    }
}

fn ranked_tree_work(block_count: usize, operation_count: usize) -> Option<usize> {
    // Module: root + region + block + function edge. Function: root + region,
    // blocks, and operation edges. Each child operation contributes its root.
    6_usize
        .checked_add(block_count)?
        .checked_add(operation_count.checked_mul(2)?)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionRankedKernelErrorV1 {
    InvalidFunctionName(NameError),
    ResourceLimit {
        resource: &'static str,
        limit: usize,
        actual: usize,
    },
    InvalidShape,
    InvalidPredicatedAccessUse {
        success: ProductionRankedValueIdV1,
        uses: usize,
    },
    InvalidPredicatedAccessIndexUse {
        index: ProductionRankedValueIdV1,
    },
    InvalidExecutionLayout,
    InvalidUnsignedCast,
    InvalidAllocationContract,
    InvalidPipelineContract,
    InvalidReferenceContract,
    InvalidSemanticExpression(ProductionSemanticExpressionErrorV2),
    InvalidCollectiveSemanticContract,
    InvalidTransformation(ProductionRankedTranslationErrorV1),
    UnsupportedElementWidth(u32),
    DynamicExtentCountMismatch {
        expected: usize,
        actual: usize,
    },
    UndefinedValue(ProductionRankedValueV1),
    NonCanonicalValueId {
        expected: u32,
        actual: u32,
    },
    ExpectedIndex(ProductionRankedValueV1),
    ExpectedSemantic(ProductionRankedValueV1),
    ExpectedView(ProductionRankedValueV1),
    DimensionOutOfBounds {
        dimension: u32,
        rank: usize,
    },
    AccessRankMismatch {
        expected: usize,
        actual: usize,
    },
    WriteThroughReadOnlyView,
    AtomicContractRequired,
    NonAtomicKindForAtomicAccess,
    InvalidBlockTarget(u32),
    CrossBlockDefinitionRequiresArgument {
        definition_block: usize,
        use_block: usize,
    },
    MissingKernelDialect,
    MissingGpuDialect,
    Materialization(&'static str),
}

/// Positive, non-vacuous accounting for typed semantic proof obligations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProductionTypedSemanticObligationSummaryV2 {
    pub expression_roots: usize,
    pub expression_nodes: usize,
    pub arithmetic_operations: usize,
    pub comparisons: usize,
    pub selects: usize,
    pub casts: usize,
    pub checked_operations: usize,
    pub statically_discharged_domain_roots: usize,
    pub exact_bitvector_operator_congruence_roots: usize,
    /// MIR/KIR operator-identity and congruence only. Never target-value authority.
    pub exact_ieee_operator_congruence_roots: usize,
}

impl ProductionTypedSemanticObligationSummaryV2 {
    pub const fn is_non_vacuous(self) -> bool {
        self.expression_roots != 0 && self.expression_nodes != 0
    }

    pub const fn grants_target_ieee_value_authority(self) -> bool {
        false
    }
}

pub fn typed_semantic_obligation_summary_v2(
    kernel: &ProductionRankedKernelV1,
) -> Result<ProductionTypedSemanticObligationSummaryV2, ProductionRankedKernelErrorV1> {
    let mut summary = ProductionTypedSemanticObligationSummaryV2::default();
    for operation in kernel.blocks().iter().flat_map(|block| block.operations()) {
        let ProductionRankedOperationV1::SemanticExpression {
            expression,
            numerical_contract,
            ..
        } = operation
        else {
            continue;
        };
        let stats = expression
            .validate()
            .map_err(ProductionRankedKernelErrorV1::InvalidSemanticExpression)?;
        expression
            .validate_static_domains()
            .map_err(ProductionRankedKernelErrorV1::InvalidSemanticExpression)?;
        if !numerical_contract.is_supported() || !numerical_contract.admits_expression(expression) {
            return Err(ProductionRankedKernelErrorV1::InvalidSemanticExpression(
                ProductionSemanticExpressionErrorV2::UnsupportedNumericalContract,
            ));
        }
        summary.expression_roots += 1;
        summary.expression_nodes += stats.nodes;
        summary.arithmetic_operations += stats.arithmetic_operations;
        summary.comparisons += stats.comparisons;
        summary.selects += stats.selects;
        summary.casts += stats.casts;
        summary.checked_operations += stats.checked_operations;
        summary.statically_discharged_domain_roots += 1;
        match numerical_contract {
            ProductionNumericalContractV2::ExactBitVectorOperatorCongruence => {
                summary.exact_bitvector_operator_congruence_roots += 1;
            }
            ProductionNumericalContractV2::ExactIeee754OperatorCongruence { .. } => {
                summary.exact_ieee_operator_congruence_roots += 1;
            }
            ProductionNumericalContractV2::Relaxed
            | ProductionNumericalContractV2::ErrorBounded { .. } => {
                unreachable!("unsupported numerical contracts were rejected above")
            }
        }
    }
    Ok(summary)
}

/// Exact reconciliation between retained typed recipes and live PLIRON typed roots.
///
/// The mandatory semantic pass reconstructs the closed typed SSA trees and validates
/// every root policy and commitment. This record establishes that materialization
/// retained the same ordered canonical transcript digests. It does not grant
/// target-value or lowering authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductionTypedSemanticCommitmentReconciliationV2 {
    recipe_expression_roots: usize,
    pliron_commitment_roots: usize,
    ordered_commitments_sha256: [u8; 32],
}

impl ProductionTypedSemanticCommitmentReconciliationV2 {
    pub const fn recipe_expression_roots(self) -> usize {
        self.recipe_expression_roots
    }

    pub const fn pliron_commitment_roots(self) -> usize {
        self.pliron_commitment_roots
    }

    pub const fn ordered_commitments_sha256(&self) -> &[u8; 32] {
        &self.ordered_commitments_sha256
    }

    pub const fn is_exact(self) -> bool {
        self.recipe_expression_roots == self.pliron_commitment_roots
    }

    pub const fn grants_arithmetic_interpretation_authority(self) -> bool {
        false
    }

    pub const fn grants_target_value_authority(self) -> bool {
        false
    }
}

/// Reconciles every retained typed expression with the corresponding operation
/// in the owner-held PLIRON graph, in deterministic block/operation order.
pub fn typed_semantic_commitment_reconciliation_v2(
    input: &ProductionRankedKernelLoweringInputV1,
) -> Result<ProductionTypedSemanticCommitmentReconciliationV2, ProductionRankedKernelErrorV1> {
    const DOMAIN: &[u8] = b"FE2O3/TYPED-SEMANTIC-COMMITMENT-RECONCILIATION/V2\0";

    let expected = input
        .kernel
        .blocks
        .iter()
        .flat_map(|block| block.operations())
        .filter_map(|operation| match operation {
            ProductionRankedOperationV1::SemanticExpression {
                expression,
                numerical_contract,
                ..
            } => Some(expression.materialized_pliron_transcript_sha256(*numerical_contract)),
            _ => None,
        })
        .collect::<Vec<_>>();

    let context = &input._session.inner.context;
    let root_pointer = input
        ._session
        .inner
        .operations
        .get(&input._root.operation.identity)
        .copied()
        .ok_or(ProductionRankedKernelErrorV1::Materialization(
            "live ranked root is absent during typed commitment reconciliation",
        ))?;
    let module = ModuleOp::from_operation(root_pointer);
    let module_operations = module
        .get_body(context, 0)
        .deref(context)
        .iter(context)
        .collect::<Vec<_>>();
    let [function_pointer] = module_operations.as_slice() else {
        return Err(ProductionRankedKernelErrorV1::Materialization(
            "live ranked module shape changed during typed commitment reconciliation",
        ));
    };
    let function = Operation::get_op::<FuncOp>(*function_pointer, context).ok_or(
        ProductionRankedKernelErrorV1::Materialization(
            "live ranked function is absent during typed commitment reconciliation",
        ),
    )?;
    let mut actual = Vec::new();
    for block in function.get_region(context).deref(context).iter(context) {
        for operation in block.deref(context).iter(context) {
            let Some(root) = Operation::get_op::<SemanticTypedExpressionRootOp>(operation, context)
            else {
                continue;
            };
            let words =
                root.commitment(context)
                    .ok_or(ProductionRankedKernelErrorV1::Materialization(
                        "live typed semantic root has no canonical commitment",
                    ))?;
            let mut digest = [0_u8; 32];
            for (chunk, word) in digest.chunks_exact_mut(8).zip(words) {
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            actual.push(digest);
        }
    }
    if actual != expected {
        return Err(ProductionRankedKernelErrorV1::Materialization(
            "retained typed semantic recipes do not match live PLIRON root commitments",
        ));
    }

    let mut digest = Sha256::new();
    digest.update(DOMAIN);
    digest.update((expected.len() as u64).to_le_bytes());
    for commitment in &expected {
        digest.update(commitment);
    }
    Ok(ProductionTypedSemanticCommitmentReconciliationV2 {
        recipe_expression_roots: expected.len(),
        pliron_commitment_roots: actual.len(),
        ordered_commitments_sha256: digest.finalize().into(),
    })
}

impl fmt::Display for ProductionRankedKernelErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFunctionName(_) => {
                formatter.write_str("invalid ranked-kernel function name")
            }
            Self::ResourceLimit {
                resource,
                limit,
                actual,
            } => {
                write!(
                    formatter,
                    "ranked-kernel {resource} count {actual} exceeds {limit}"
                )
            }
            Self::InvalidShape => write!(
                formatter,
                "ranked view rank must be within 1..={MAX_RANKED_MEMORY_RANK}"
            ),
            Self::InvalidPredicatedAccessUse { success, uses } => write!(
                formatter,
                "ranked predicated success value {} must be consumed at least once but has {uses} uses",
                success.get(),
            ),
            Self::InvalidPredicatedAccessIndexUse { index } => write!(
                formatter,
                "ranked predicated index value {} is consumed by an unpaired memory access",
                index.get(),
            ),
            Self::InvalidExecutionLayout => formatter.write_str(
                "ranked execution layout must be the unique first entry operation with nonzero workgroup axes and an integral subgroup width",
            ),
            Self::InvalidUnsignedCast => formatter.write_str(
                "ranked unsigned cast requires an index source and a width in {8, 16, 32, 64}",
            ),
            Self::InvalidAllocationContract => formatter.write_str(
                "ranked views/effects require supported memory semantics and consistent allocation origins/no-alias classes",
            ),
            Self::InvalidPipelineContract => formatter.write_str(
                "staged pipeline requires writable workgroup storage, 2..=8 buffers, a smaller positive prefetch distance, and index epoch/slot operands",
            ),
            Self::InvalidReferenceContract => formatter.write_str(
                "functional-reference proof identities must be nonzero and pairwise distinct",
            ),
            Self::InvalidSemanticExpression(error) => write!(
                formatter,
                "invalid typed semantic expression: {error}"
            ),
            Self::InvalidCollectiveSemanticContract => formatter.write_str(
                "finite collective semantic contract is malformed, unsupported, or unbounded",
            ),
            Self::InvalidTransformation(error) => write!(formatter, "{error}"),
            Self::UnsupportedElementWidth(width) => write!(
                formatter,
                "ranked view element width {width} is not one of {SUPPORTED_ELEMENT_WIDTHS:?}"
            ),
            Self::DynamicExtentCountMismatch { expected, actual } => write!(
                formatter,
                "ranked view requires {expected} dynamic extents but has {actual}"
            ),
            Self::UndefinedValue(value) => write!(
                formatter,
                "ranked recipe references undefined value {value:?}"
            ),
            Self::NonCanonicalValueId { expected, actual } => write!(
                formatter,
                "ranked recipe local value ID {actual} is noncanonical; expected {expected}"
            ),
            Self::ExpectedIndex(value) => write!(
                formatter,
                "ranked recipe expected index value, found {value:?}"
            ),
            Self::ExpectedSemantic(value) => write!(
                formatter,
                "ranked recipe expected semantic scalar value, found {value:?}"
            ),
            Self::ExpectedView(value) => write!(
                formatter,
                "ranked recipe expected view value, found {value:?}"
            ),
            Self::DimensionOutOfBounds { dimension, rank } => write!(
                formatter,
                "ranked dimension {dimension} is outside rank {rank}"
            ),
            Self::AccessRankMismatch { expected, actual } => write!(
                formatter,
                "ranked access requires {expected} indices but has {actual}"
            ),
            Self::WriteThroughReadOnlyView => {
                formatter.write_str("ranked write targets a read-only view")
            }
            Self::AtomicContractRequired => formatter
                .write_str("atomic ranked access requires the explicit AtomicAccess recipe"),
            Self::NonAtomicKindForAtomicAccess => {
                formatter.write_str("AtomicAccess recipe requires an atomic access kind")
            }
            Self::InvalidBlockTarget(target) => {
                write!(formatter, "ranked terminator targets absent block {target}")
            }
            Self::CrossBlockDefinitionRequiresArgument {
                definition_block,
                use_block,
            } => write!(
                formatter,
                "ranked recipe value defined in block {definition_block} is used directly by block {use_block}; pass it through explicit block arguments"
            ),
            Self::MissingKernelDialect => formatter.write_str(
                "production ranked construction requires the kernel dialect registration",
            ),
            Self::MissingGpuDialect => formatter.write_str(
                "production ranked barrier construction requires the gpu dialect registration",
            ),
            Self::Materialization(message) => formatter.write_str(message),
        }
    }
}

impl Error for ProductionRankedKernelErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidTransformation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RecipeValueKindV1 {
    Index,
    Pipeline {
        buffers: u32,
        prefetch_distance: u32,
    },
    CheckedAccessSuccess {
        index: ProductionRankedValueV1,
        physical_extent: ProductionRankedValueV1,
    },
    Semantic,
    TypedSemantic {
        scalar: super::ProductionSemanticScalarTypeV2,
        numerical_contract: ProductionNumericalContractV2,
    },
    View {
        rank: usize,
        writable: bool,
        dynamic_extent: Option<ProductionRankedValueV1>,
        memory_space: MemorySpaceAttr,
        leading_extent: u64,
    },
}

fn require_value(
    value: ProductionRankedValueV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<RecipeValueKindV1, ProductionRankedKernelErrorV1> {
    match value {
        ProductionRankedValueV1::Argument(argument)
            if usize::try_from(argument)
                .ok()
                .is_some_and(|argument| argument < argument_count) =>
        {
            Ok(RecipeValueKindV1::Index)
        }
        ProductionRankedValueV1::Local(identity) => locals
            .get(identity.get() as usize)
            .copied()
            .ok_or(ProductionRankedKernelErrorV1::UndefinedValue(value)),
        ProductionRankedValueV1::BlockArgument { .. } => Ok(RecipeValueKindV1::Index),
        ProductionRankedValueV1::Argument(_) => {
            Err(ProductionRankedKernelErrorV1::UndefinedValue(value))
        }
    }
}

fn require_index(
    value: ProductionRankedValueV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<(), ProductionRankedKernelErrorV1> {
    if matches!(
        require_value(value, argument_count, locals)?,
        RecipeValueKindV1::Index
    ) {
        Ok(())
    } else {
        Err(ProductionRankedKernelErrorV1::ExpectedIndex(value))
    }
}

fn require_view(
    value: ProductionRankedValueV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<(usize, bool), ProductionRankedKernelErrorV1> {
    match require_value(value, argument_count, locals)? {
        RecipeValueKindV1::View { rank, writable, .. } => Ok((rank, writable)),
        RecipeValueKindV1::Index
        | RecipeValueKindV1::Pipeline { .. }
        | RecipeValueKindV1::CheckedAccessSuccess { .. }
        | RecipeValueKindV1::Semantic
        | RecipeValueKindV1::TypedSemantic { .. } => {
            Err(ProductionRankedKernelErrorV1::ExpectedView(value))
        }
    }
}

fn require_semantic(
    value: ProductionRankedValueV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<(), ProductionRankedKernelErrorV1> {
    if matches!(
        require_value(value, argument_count, locals)?,
        RecipeValueKindV1::Semantic | RecipeValueKindV1::TypedSemantic { .. }
    ) {
        Ok(())
    } else {
        Err(ProductionRankedKernelErrorV1::ExpectedSemantic(value))
    }
}

fn require_typed_semantic(
    value: ProductionRankedValueV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<
    (
        super::ProductionSemanticScalarTypeV2,
        ProductionNumericalContractV2,
    ),
    ProductionRankedKernelErrorV1,
> {
    match require_value(value, argument_count, locals)? {
        RecipeValueKindV1::TypedSemantic {
            scalar,
            numerical_contract,
        } => Ok((scalar, numerical_contract)),
        _ => Err(ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract),
    }
}

fn validate_operation(
    operation: &ProductionRankedOperationV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<Option<(ProductionRankedValueIdV1, RecipeValueKindV1)>, ProductionRankedKernelErrorV1> {
    match operation {
        ProductionRankedOperationV1::ExecutionLayout { .. } => Ok(None),
        ProductionRankedOperationV1::View {
            result,
            element_width,
            writable,
            shape,
            dynamic_extents,
            ..
        }
        | ProductionRankedOperationV1::ViewInSpace {
            result,
            element_width,
            writable,
            shape,
            dynamic_extents,
            ..
        } => {
            if !(1..=MAX_RANKED_MEMORY_RANK).contains(&shape.len()) {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            }
            if !SUPPORTED_ELEMENT_WIDTHS.contains(element_width) {
                return Err(ProductionRankedKernelErrorV1::UnsupportedElementWidth(
                    *element_width,
                ));
            }
            let expected = shape
                .iter()
                .filter(|extent| **extent == DYNAMIC_EXTENT)
                .count();
            if dynamic_extents.len() != expected {
                return Err(ProductionRankedKernelErrorV1::DynamicExtentCountMismatch {
                    expected,
                    actual: dynamic_extents.len(),
                });
            }
            for extent in dynamic_extents {
                require_index(*extent, argument_count, locals)?;
            }
            Ok(Some((
                *result,
                RecipeValueKindV1::View {
                    rank: shape.len(),
                    writable: *writable,
                    dynamic_extent: (shape.as_slice() == [DYNAMIC_EXTENT])
                        .then(|| dynamic_extents[0]),
                    memory_space: match operation {
                        ProductionRankedOperationV1::View { .. } => MemorySpaceAttr::Global,
                        ProductionRankedOperationV1::ViewInSpace { memory_space, .. } => {
                            *memory_space
                        }
                        _ => unreachable!("matched ranked view recipe"),
                    },
                    leading_extent: shape[0],
                },
            )))
        }
        ProductionRankedOperationV1::PipelineCreate {
            result,
            view,
            buffers,
            prefetch_distance,
        } => {
            let RecipeValueKindV1::View {
                rank: _,
                writable: true,
                memory_space: MemorySpaceAttr::Workgroup,
                leading_extent,
                ..
            } = require_value(*view, argument_count, locals)?
            else {
                return Err(ProductionRankedKernelErrorV1::InvalidPipelineContract);
            };
            if leading_extent != u64::from(*buffers)
                || !(2..=dialect_kernel::MAX_PIPELINE_BUFFERS_V1).contains(buffers)
                || *prefetch_distance == 0
                || *prefetch_distance >= *buffers
            {
                return Err(ProductionRankedKernelErrorV1::InvalidPipelineContract);
            }
            Ok(Some((
                *result,
                RecipeValueKindV1::Pipeline {
                    buffers: *buffers,
                    prefetch_distance: *prefetch_distance,
                },
            )))
        }
        ProductionRankedOperationV1::PipelineEvent {
            pipeline,
            epoch,
            slot,
            ..
        } => {
            let RecipeValueKindV1::Pipeline {
                buffers,
                prefetch_distance,
            } = require_value(*pipeline, argument_count, locals)?
            else {
                return Err(ProductionRankedKernelErrorV1::InvalidPipelineContract);
            };
            if prefetch_distance == 0 || prefetch_distance >= buffers {
                return Err(ProductionRankedKernelErrorV1::InvalidPipelineContract);
            }
            require_index(*epoch, argument_count, locals)?;
            require_index(*slot, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::IndexConstant { result, .. } => {
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::IndexUnsignedCast {
            result,
            source,
            bit_width,
        } => {
            require_index(*source, argument_count, locals)?;
            if !matches!(*bit_width, 8 | 16 | 32 | 64) {
                return Err(ProductionRankedKernelErrorV1::InvalidUnsignedCast);
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::IndexUnknown { result } => {
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::InvocationIndex {
            result, dimension, ..
        } => {
            if usize::try_from(*dimension)
                .ok()
                .is_none_or(|dimension| dimension >= MAX_RANKED_MEMORY_RANK)
            {
                return Err(ProductionRankedKernelErrorV1::DimensionOutOfBounds {
                    dimension: *dimension,
                    rank: MAX_RANKED_MEMORY_RANK,
                });
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::IndexBinary {
            result, lhs, rhs, ..
        } => {
            require_index(*lhs, argument_count, locals)?;
            require_index(*rhs, argument_count, locals)?;
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::DeterministicJoin {
            result,
            dependencies,
        } => {
            if !(1..=MAX_DETERMINISTIC_JOIN_INPUTS_V1).contains(&dependencies.len()) {
                return Err(ProductionRankedKernelErrorV1::ResourceLimit {
                    resource: "deterministic dependency",
                    limit: MAX_DETERMINISTIC_JOIN_INPUTS_V1,
                    actual: dependencies.len(),
                });
            }
            for dependency in dependencies {
                require_index(*dependency, argument_count, locals)?;
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::CheckedTiledIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
        } => {
            for value in [invocation, component, rows, columns, row_stride] {
                require_index(*value, argument_count, locals)?;
            }
            if *lanes_per_tile == 0
                || *tile_rows == 0
                || *tile_columns == 0
                || *elements_per_lane == 0
                || !lanes_per_tile.is_multiple_of(*tile_columns)
                || lanes_per_tile.checked_mul(*elements_per_lane)
                    != tile_rows.checked_mul(*tile_columns)
                || (lanes_per_tile / tile_columns).checked_mul(*elements_per_lane)
                    != Some(*tile_rows)
            {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::CheckedRowStripedIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            lanes_per_row,
            elements_per_lane,
        } => {
            for value in [invocation, component, rows, columns, row_stride] {
                require_index(*value, argument_count, locals)?;
            }
            if *lanes_per_row == 0
                || *elements_per_lane == 0
                || (*elements_per_lane - 1)
                    .checked_mul(*lanes_per_row)
                    .and_then(|base| base.checked_add(*lanes_per_row - 1))
                    .is_none()
            {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::PredicatedCheckedTiledIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
            ..
        } => {
            for value in [
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ] {
                require_index(*value, argument_count, locals)?;
            }
            if *lanes_per_tile == 0
                || *tile_rows == 0
                || *tile_columns == 0
                || *elements_per_lane == 0
                || !lanes_per_tile.is_multiple_of(*tile_columns)
                || lanes_per_tile.checked_mul(*elements_per_lane)
                    != tile_rows.checked_mul(*tile_columns)
                || (lanes_per_tile / tile_columns).checked_mul(*elements_per_lane)
                    != Some(*tile_rows)
            {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::PredicatedCheckedRowStripedIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            lanes_per_row,
            elements_per_lane,
            ..
        } => {
            for value in [
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ] {
                require_index(*value, argument_count, locals)?;
            }
            if *lanes_per_row == 0
                || *elements_per_lane == 0
                || (*elements_per_lane - 1)
                    .checked_mul(*lanes_per_row)
                    .and_then(|base| base.checked_add(*lanes_per_row - 1))
                    .is_none()
            {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::Dimension {
            result,
            view,
            dimension,
        } => {
            let (rank, _) = require_view(*view, argument_count, locals)?;
            if usize::try_from(*dimension)
                .ok()
                .is_none_or(|dimension| dimension >= rank)
            {
                return Err(ProductionRankedKernelErrorV1::DimensionOutOfBounds {
                    dimension: *dimension,
                    rank,
                });
            }
            Ok(Some((*result, RecipeValueKindV1::Index)))
        }
        ProductionRankedOperationV1::Access {
            kind,
            view,
            indices,
        } => {
            if kind.is_atomic() {
                return Err(ProductionRankedKernelErrorV1::AtomicContractRequired);
            }
            validate_access(*kind, *view, indices, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::PredicatedAccess {
            kind,
            view,
            index,
            success,
        } => {
            if kind.is_atomic() {
                return Err(ProductionRankedKernelErrorV1::AtomicContractRequired);
            }
            validate_access(*kind, *view, &[*index], argument_count, locals)?;
            let view_kind = require_value(*view, argument_count, locals)?;
            let RecipeValueKindV1::View {
                rank: 1,
                dynamic_extent: Some(view_extent),
                ..
            } = view_kind
            else {
                return Err(ProductionRankedKernelErrorV1::InvalidShape);
            };
            require_index(*index, argument_count, locals)?;
            match require_value(*success, argument_count, locals)? {
                RecipeValueKindV1::CheckedAccessSuccess {
                    index: expected_index,
                    physical_extent,
                } if expected_index == *index && physical_extent == view_extent => Ok(None),
                _ => Err(ProductionRankedKernelErrorV1::InvalidShape),
            }
        }
        ProductionRankedOperationV1::ValueAccess {
            kind,
            view,
            indices,
            value,
        } => {
            if kind.is_atomic() || !kind.writes_memory() {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            validate_access(*kind, *view, indices, argument_count, locals)?;
            require_semantic(*value, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::AtomicAccess {
            kind,
            view,
            indices,
            ..
        } => {
            if !kind.is_atomic() {
                return Err(ProductionRankedKernelErrorV1::NonAtomicKindForAtomicAccess);
            }
            validate_access(*kind, *view, indices, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::AtomicValueAccess {
            kind,
            view,
            indices,
            value,
            ..
        } => {
            if !kind.is_atomic() || !kind.writes_memory() {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            validate_access(*kind, *view, indices, argument_count, locals)?;
            require_semantic(*value, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::OwnershipContract { view, .. } => {
            let (_, writable) = require_view(*view, argument_count, locals)?;
            if !writable {
                return Err(ProductionRankedKernelErrorV1::WriteThroughReadOnlyView);
            }
            Ok(None)
        }
        ProductionRankedOperationV1::AllocationEffect {
            kind,
            memory_space,
            allocation_origin,
            noalias_class,
        } => {
            if !is_supported_allocation_effect_contract_v1(
                *kind,
                *memory_space,
                *allocation_origin,
                *noalias_class,
            ) {
                return Err(ProductionRankedKernelErrorV1::InvalidAllocationContract);
            }
            Ok(None)
        }
        ProductionRankedOperationV1::Barrier { .. }
        | ProductionRankedOperationV1::Fence { .. }
        | ProductionRankedOperationV1::TensorLayout { .. } => Ok(None),
        ProductionRankedOperationV1::TensorResultComponent {
            result,
            tensor_result_root,
            component,
            scalar,
            numerical_contract,
        } => {
            if tensor_result_root.is_zero()
                || usize::from(*component) >= MAX_PRODUCTION_TENSOR_COMPONENTS_V1
                || !numerical_contract.is_supported()
                || !numerical_contract.admits_scalar(*scalar)
            {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            Ok(Some((
                *result,
                RecipeValueKindV1::TypedSemantic {
                    scalar: *scalar,
                    numerical_contract: *numerical_contract,
                },
            )))
        }
        ProductionRankedOperationV1::SemanticSymbol { result, symbol } => {
            if *symbol >= super::PRODUCTION_SEMANTIC_LOAD_SYMBOL_BASE_V2 {
                return Err(ProductionRankedKernelErrorV1::InvalidSemanticExpression(
                    ProductionSemanticExpressionErrorV2::ReservedSymbol,
                ));
            }
            Ok(Some((*result, RecipeValueKindV1::Semantic)))
        }
        ProductionRankedOperationV1::SemanticConstant { result, .. } => {
            Ok(Some((*result, RecipeValueKindV1::Semantic)))
        }
        ProductionRankedOperationV1::SemanticBinary {
            result, lhs, rhs, ..
        } => {
            require_semantic(*lhs, argument_count, locals)?;
            require_semantic(*rhs, argument_count, locals)?;
            Ok(Some((*result, RecipeValueKindV1::Semantic)))
        }
        ProductionRankedOperationV1::SemanticExpression {
            result,
            expression,
            numerical_contract,
        } => {
            expression
                .validate()
                .map_err(ProductionRankedKernelErrorV1::InvalidSemanticExpression)?;
            expression
                .validate_static_domains()
                .map_err(ProductionRankedKernelErrorV1::InvalidSemanticExpression)?;
            if !numerical_contract.is_supported()
                || !numerical_contract.admits_expression(expression)
            {
                return Err(ProductionRankedKernelErrorV1::InvalidSemanticExpression(
                    ProductionSemanticExpressionErrorV2::UnsupportedNumericalContract,
                ));
            }
            Ok(Some((
                *result,
                RecipeValueKindV1::TypedSemantic {
                    scalar: expression.scalar(),
                    numerical_contract: *numerical_contract,
                },
            )))
        }
        ProductionRankedOperationV1::CollectiveSemantics {
            contract,
            view,
            actual,
            expected,
            witness0,
            witness1,
        } => {
            let (_, writable) = require_view(*view, argument_count, locals)?;
            if !writable {
                return Err(ProductionRankedKernelErrorV1::WriteThroughReadOnlyView);
            }
            let actual_type = require_typed_semantic(*actual, argument_count, locals)?;
            let expected_type = require_typed_semantic(*expected, argument_count, locals)?;
            let witness0_type = require_typed_semantic(*witness0, argument_count, locals)?;
            let witness1_type = require_typed_semantic(*witness1, argument_count, locals)?;
            if actual_type != expected_type
                || actual_type.1 != contract.numerical_contract()
                || !contract.numerical_contract().admits_scalar(actual_type.0)
            {
                return Err(ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract);
            }
            match contract.kind() {
                ProductionCollectiveSemanticKindV1::FiniteFold
                | ProductionCollectiveSemanticKindV1::FiniteRecurrence => {
                    if witness0_type != actual_type || witness1_type != actual_type {
                        return Err(
                            ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract,
                        );
                    }
                }
                ProductionCollectiveSemanticKindV1::PermutationGather => {
                    if witness0_type != witness1_type
                        || !witness0_type.0.is_integer()
                        || witness0_type.1
                            != ProductionNumericalContractV2::ExactBitVectorOperatorCongruence
                    {
                        return Err(
                            ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract,
                        );
                    }
                }
            }
            Ok(None)
        }
        ProductionRankedOperationV1::RequireEquivalent { actual, expected }
        | ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent {
            actual,
            expected,
            ..
        }
        | ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent {
            actual,
            expected,
            ..
        } => {
            require_semantic(*actual, argument_count, locals)?;
            require_semantic(*expected, argument_count, locals)?;
            Ok(None)
        }
        ProductionRankedOperationV1::RequireEffectRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestEffectRefinement { contract, .. } => {
            require_view(contract.view(), argument_count, locals)?;
            for index in contract.indices() {
                require_index(*index, argument_count, locals)?;
            }
            for value in contract
                .gpu_coordinates()
                .iter()
                .chain(contract.reference_coordinates())
                .copied()
                .chain([
                    contract.gpu_domain(),
                    contract.reference_domain(),
                    contract.gpu_precondition(),
                    contract.reference_precondition(),
                    contract.gpu_value(),
                    contract.reference_value(),
                ])
            {
                require_semantic(value, argument_count, locals)?;
            }
            Ok(None)
        }
        ProductionRankedOperationV1::RequireNumericalRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestNumericalRefinement { contract, .. } => {
            let actual = require_typed_semantic(contract.actual(), argument_count, locals)?;
            let reference = require_typed_semantic(contract.reference(), argument_count, locals)?;
            let domain = require_typed_semantic(contract.domain(), argument_count, locals)?;
            let precondition =
                require_typed_semantic(contract.precondition(), argument_count, locals)?;
            let boolean = (
                ProductionSemanticScalarTypeV2::Bool,
                ProductionNumericalContractV2::ExactBitVectorOperatorCongruence,
            );
            if actual != reference
                || !actual.0.is_float()
                || domain != boolean
                || precondition != boolean
            {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            Ok(None)
        }
        ProductionRankedOperationV1::RequireTensorRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestTensorRefinement { contract, .. } => {
            require_view(contract.output_view(), argument_count, locals)?;
            let actual = require_typed_semantic(contract.actual(), argument_count, locals)?;
            let reference = require_typed_semantic(contract.reference(), argument_count, locals)?;
            if actual != reference
                || actual != (contract.component_scalar(), contract.numerical_contract())
            {
                return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
            }
            for component in contract.components() {
                for index in component.indices() {
                    require_index(*index, argument_count, locals)?;
                }
                let gpu = require_typed_semantic(component.gpu_value(), argument_count, locals)?;
                let sequential =
                    require_typed_semantic(component.reference_value(), argument_count, locals)?;
                if gpu != actual || sequential != reference {
                    return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
                }
            }
            Ok(None)
        }
    }
}

fn validate_access(
    kind: AccessKindAttr,
    view: ProductionRankedValueV1,
    indices: &[ProductionRankedValueV1],
    argument_count: usize,
    locals: &[RecipeValueKindV1],
) -> Result<(), ProductionRankedKernelErrorV1> {
    let (rank, writable) = require_view(view, argument_count, locals)?;
    if indices.len() != rank {
        return Err(ProductionRankedKernelErrorV1::AccessRankMismatch {
            expected: rank,
            actual: indices.len(),
        });
    }
    if kind.writes_memory() && !writable {
        return Err(ProductionRankedKernelErrorV1::WriteThroughReadOnlyView);
    }
    for index in indices {
        require_index(*index, argument_count, locals)?;
    }
    Ok(())
}

fn validate_scoped_value_v1(
    value: ProductionRankedValueV1,
    current_block: usize,
    blocks: &[ProductionRankedBlockV1],
    local_definition_blocks: &[usize],
) -> Result<(), ProductionRankedKernelErrorV1> {
    match value {
        ProductionRankedValueV1::BlockArgument { block, argument } => {
            if block as usize != current_block
                || blocks
                    .get(block as usize)
                    .is_none_or(|recipe| argument >= recipe.index_argument_count)
            {
                return Err(ProductionRankedKernelErrorV1::UndefinedValue(value));
            }
        }
        ProductionRankedValueV1::Local(identity) => {
            let definition = local_definition_blocks
                .get(identity.get() as usize)
                .copied()
                .ok_or(ProductionRankedKernelErrorV1::UndefinedValue(value))?;
            if definition != 0 && definition != current_block {
                return Err(
                    ProductionRankedKernelErrorV1::CrossBlockDefinitionRequiresArgument {
                        definition_block: definition,
                        use_block: current_block,
                    },
                );
            }
        }
        ProductionRankedValueV1::Argument(_) => {}
    }
    Ok(())
}

fn validate_scoped_operation_values_v1(
    operation: &ProductionRankedOperationV1,
    current_block: usize,
    blocks: &[ProductionRankedBlockV1],
    local_definition_blocks: &[usize],
) -> Result<(), ProductionRankedKernelErrorV1> {
    let validate =
        |value| validate_scoped_value_v1(value, current_block, blocks, local_definition_blocks);
    match operation {
        ProductionRankedOperationV1::View {
            dynamic_extents, ..
        }
        | ProductionRankedOperationV1::ViewInSpace {
            dynamic_extents, ..
        } => {
            for value in dynamic_extents {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::PipelineCreate { view, .. } => validate(*view)?,
        ProductionRankedOperationV1::PipelineEvent {
            pipeline,
            epoch,
            slot,
            ..
        } => {
            validate(*pipeline)?;
            validate(*epoch)?;
            validate(*slot)?;
        }
        ProductionRankedOperationV1::IndexBinary { lhs, rhs, .. }
        | ProductionRankedOperationV1::SemanticBinary { lhs, rhs, .. } => {
            validate(*lhs)?;
            validate(*rhs)?;
        }
        ProductionRankedOperationV1::IndexUnsignedCast { source, .. } => validate(*source)?,
        ProductionRankedOperationV1::DeterministicJoin { dependencies, .. } => {
            for dependency in dependencies {
                validate(*dependency)?;
            }
        }
        ProductionRankedOperationV1::CheckedTiledIndex2D {
            invocation,
            component,
            rows,
            columns,
            row_stride,
            ..
        } => {
            for value in [invocation, component, rows, columns, row_stride] {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::PredicatedCheckedTiledIndex2D {
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            ..
        } => {
            for value in [
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ] {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::CheckedRowStripedIndex2D {
            invocation,
            component,
            rows,
            columns,
            row_stride,
            ..
        } => {
            for value in [invocation, component, rows, columns, row_stride] {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::PredicatedCheckedRowStripedIndex2D {
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            ..
        } => {
            for value in [
                invocation,
                component,
                rows,
                columns,
                row_stride,
                physical_extent,
            ] {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::Dimension { view, .. }
        | ProductionRankedOperationV1::OwnershipContract { view, .. } => validate(*view)?,
        ProductionRankedOperationV1::Access { view, indices, .. }
        | ProductionRankedOperationV1::AtomicAccess { view, indices, .. } => {
            validate(*view)?;
            for value in indices {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::PredicatedAccess {
            view,
            index,
            success,
            ..
        } => {
            validate(*view)?;
            validate(*index)?;
            validate(*success)?;
        }
        ProductionRankedOperationV1::ValueAccess {
            view,
            indices,
            value,
            ..
        }
        | ProductionRankedOperationV1::AtomicValueAccess {
            view,
            indices,
            value,
            ..
        } => {
            validate(*view)?;
            for index in indices {
                validate(*index)?;
            }
            validate(*value)?;
        }
        ProductionRankedOperationV1::RequireEquivalent { actual, expected }
        | ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent {
            actual,
            expected,
            ..
        }
        | ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent {
            actual,
            expected,
            ..
        } => {
            validate(*actual)?;
            validate(*expected)?;
        }
        ProductionRankedOperationV1::CollectiveSemantics {
            view,
            actual,
            expected,
            witness0,
            witness1,
            ..
        } => {
            for value in [view, actual, expected, witness0, witness1] {
                validate(*value)?;
            }
        }
        ProductionRankedOperationV1::RequireEffectRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestEffectRefinement { contract, .. } => {
            validate(contract.view())?;
            for value in contract
                .indices()
                .iter()
                .chain(contract.gpu_coordinates())
                .chain(contract.reference_coordinates())
                .copied()
                .chain([
                    contract.gpu_domain(),
                    contract.reference_domain(),
                    contract.gpu_precondition(),
                    contract.reference_precondition(),
                    contract.gpu_value(),
                    contract.reference_value(),
                ])
            {
                validate(value)?;
            }
        }
        ProductionRankedOperationV1::RequireNumericalRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestNumericalRefinement { contract, .. } => {
            for value in [
                contract.actual(),
                contract.reference(),
                contract.domain(),
                contract.precondition(),
            ] {
                validate(value)?;
            }
        }
        ProductionRankedOperationV1::RequireTensorRefinement { contract, .. }
        | ProductionRankedOperationV1::RequestTensorRefinement { contract, .. } => {
            validate(contract.output_view())?;
            validate(contract.actual())?;
            validate(contract.reference())?;
            for component in contract.components() {
                for value in component.indices() {
                    validate(*value)?;
                }
                validate(component.gpu_value())?;
                validate(component.reference_value())?;
            }
        }
        ProductionRankedOperationV1::ExecutionLayout { .. }
        | ProductionRankedOperationV1::IndexConstant { .. }
        | ProductionRankedOperationV1::IndexUnknown { .. }
        | ProductionRankedOperationV1::InvocationIndex { .. }
        | ProductionRankedOperationV1::Barrier { .. }
        | ProductionRankedOperationV1::Fence { .. }
        | ProductionRankedOperationV1::TensorLayout { .. }
        | ProductionRankedOperationV1::TensorResultComponent { .. }
        | ProductionRankedOperationV1::AllocationEffect { .. }
        | ProductionRankedOperationV1::SemanticSymbol { .. }
        | ProductionRankedOperationV1::SemanticConstant { .. } => {}
        ProductionRankedOperationV1::SemanticExpression { .. } => {}
    }
    Ok(())
}

fn validate_scoped_terminator_values_v1(
    terminator: &ProductionRankedTerminatorV1,
    current_block: usize,
    blocks: &[ProductionRankedBlockV1],
    local_definition_blocks: &[usize],
) -> Result<(), ProductionRankedKernelErrorV1> {
    let validate =
        |value| validate_scoped_value_v1(value, current_block, blocks, local_definition_blocks);
    match terminator {
        ProductionRankedTerminatorV1::IndexLessThan { lhs, rhs, .. }
        | ProductionRankedTerminatorV1::IndexEqual { lhs, rhs, .. }
        | ProductionRankedTerminatorV1::BranchArgsAdd {
            value: lhs,
            step: rhs,
            ..
        } => {
            validate(*lhs)?;
            validate(*rhs)
        }
        ProductionRankedTerminatorV1::IndexLessThanArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            ..
        }
        | ProductionRankedTerminatorV1::IndexEqualArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            ..
        } => {
            validate(*lhs)?;
            validate(*rhs)?;
            for value in true_arguments.iter().chain(false_arguments) {
                validate(*value)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::BranchArgs { arguments, .. } => {
            for value in arguments {
                validate(*value)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::BranchArgsAddAt {
            arguments, step, ..
        } => {
            for value in arguments {
                validate(*value)?;
            }
            validate(*step)
        }
        ProductionRankedTerminatorV1::AnalysisSplitArgs {
            control_dependencies,
            first_arguments,
            second_arguments,
            ..
        } => {
            for value in control_dependencies
                .iter()
                .chain(first_arguments)
                .chain(second_arguments)
            {
                validate(*value)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::AnalysisSplit {
            control_dependencies,
            ..
        } => {
            for value in control_dependencies {
                validate(*value)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::Branch { .. }
        | ProductionRankedTerminatorV1::Return
        | ProductionRankedTerminatorV1::Trap => Ok(()),
    }
}

fn validate_terminator(
    terminator: &ProductionRankedTerminatorV1,
    argument_count: usize,
    locals: &[RecipeValueKindV1],
    blocks: &[ProductionRankedBlockV1],
    current_block: usize,
    local_definition_blocks: &[usize],
) -> Result<(), ProductionRankedKernelErrorV1> {
    let target = |target: u32| {
        usize::try_from(target)
            .ok()
            .filter(|target| *target < blocks.len())
            .map(|_| ())
            .ok_or(ProductionRankedKernelErrorV1::InvalidBlockTarget(target))
    };
    validate_scoped_terminator_values_v1(
        terminator,
        current_block,
        blocks,
        local_definition_blocks,
    )?;
    let target_without_arguments = |destination: u32| {
        target(destination)?;
        if blocks[destination as usize].index_argument_count != 0 {
            return Err(ProductionRankedKernelErrorV1::Materialization(
                "ranked branch omits required successor arguments",
            ));
        }
        Ok(())
    };
    match terminator {
        ProductionRankedTerminatorV1::IndexLessThan {
            lhs,
            rhs,
            true_block,
            false_block,
        }
        | ProductionRankedTerminatorV1::IndexEqual {
            lhs,
            rhs,
            true_block,
            false_block,
        } => {
            require_index(*lhs, argument_count, locals)?;
            require_index(*rhs, argument_count, locals)?;
            target_without_arguments(*true_block)?;
            target_without_arguments(*false_block)
        }
        ProductionRankedTerminatorV1::IndexLessThanArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            true_block,
            false_block,
        }
        | ProductionRankedTerminatorV1::IndexEqualArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            true_block,
            false_block,
        } => {
            require_index(*lhs, argument_count, locals)?;
            require_index(*rhs, argument_count, locals)?;
            target(*true_block)?;
            target(*false_block)?;
            if true_arguments.len() != blocks[*true_block as usize].index_argument_count as usize
                || false_arguments.len()
                    != blocks[*false_block as usize].index_argument_count as usize
            {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "ranked conditional branch arguments do not match successors",
                ));
            }
            for value in true_arguments.iter().chain(false_arguments) {
                require_index(*value, argument_count, locals)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::AnalysisSplit {
            control_dependencies,
            first_block,
            second_block,
        } => {
            for value in control_dependencies {
                require_index(*value, argument_count, locals)?;
            }
            target_without_arguments(*first_block)?;
            target_without_arguments(*second_block)
        }
        ProductionRankedTerminatorV1::AnalysisSplitArgs {
            control_dependencies,
            first_arguments,
            second_arguments,
            first_block,
            second_block,
        } => {
            target(*first_block)?;
            target(*second_block)?;
            if first_arguments.len() != blocks[*first_block as usize].index_argument_count as usize
                || second_arguments.len()
                    != blocks[*second_block as usize].index_argument_count as usize
            {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "ranked analysis split arguments do not match successors",
                ));
            }
            for value in control_dependencies
                .iter()
                .chain(first_arguments)
                .chain(second_arguments)
            {
                require_index(*value, argument_count, locals)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::Branch {
            target: destination,
        } => target_without_arguments(*destination),
        ProductionRankedTerminatorV1::BranchArgs {
            arguments,
            target: destination,
        } => {
            target(*destination)?;
            let expected = blocks[*destination as usize].index_argument_count as usize;
            if arguments.len() != expected {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "ranked branch argument count does not match its successor",
                ));
            }
            for argument in arguments {
                require_index(*argument, argument_count, locals)?;
            }
            Ok(())
        }
        ProductionRankedTerminatorV1::BranchArgsAdd {
            value,
            step,
            target: destination,
        } => {
            target(*destination)?;
            if blocks[*destination as usize].index_argument_count != 1 {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "ranked induction backedge requires one successor index argument",
                ));
            }
            require_index(*value, argument_count, locals)?;
            require_index(*step, argument_count, locals)
        }
        ProductionRankedTerminatorV1::BranchArgsAddAt {
            arguments,
            add_argument,
            step,
            target: destination,
        } => {
            target(*destination)?;
            let expected = blocks[*destination as usize].index_argument_count as usize;
            if arguments.len() != expected
                || usize::try_from(*add_argument)
                    .ok()
                    .is_none_or(|argument| argument >= arguments.len())
            {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "ranked induction backedge update does not match its successor arguments",
                ));
            }
            for argument in arguments {
                require_index(*argument, argument_count, locals)?;
            }
            require_index(*step, argument_count, locals)
        }
        ProductionRankedTerminatorV1::Return | ProductionRankedTerminatorV1::Trap => Ok(()),
    }
}

pub(super) struct ConstructedRootV1 {
    pub(super) identity: RootIdentityV1,
    pub(super) ranked_function: Option<Ptr<Operation>>,
    pub(super) ranked_kernel: Option<ProductionRankedKernelV1>,
    pub(super) ranked_view_names: BTreeMap<ProductionRankedValueV1, String>,
    pub(super) policy_checked_refinement_staging: Vec<ProductionPolicyCheckedRefinementStagingV2>,
    pub(super) production_pipeline_report: Option<ProductionPlironPreloweringReportV2>,
}

pub(super) struct MaterializedConstructionV1 {
    pub(super) operation: OperationHandle,
    pub(super) ranked_function: Option<Ptr<Operation>>,
    pub(super) ranked_kernel: Option<ProductionRankedKernelV1>,
    pub(super) ranked_view_names: BTreeMap<ProductionRankedValueV1, String>,
    pub(super) policy_checked_refinement_staging: Vec<ProductionPolicyCheckedRefinementStagingV2>,
}

impl ProductionConstructionV1 {
    pub fn ranked_kernel(
        root_name: &str,
        kernel: ProductionRankedKernelV1,
    ) -> Result<Self, NameError> {
        validate_name(root_name, NameKind::Dialect)?;
        Ok(Self {
            kind: ProductionConstructionKindV1::RankedKernel {
                root_name: root_name.to_owned(),
                kernel,
                policy_checked_refinement_staging: Vec::new(),
            },
        })
    }
}

impl ProductionPlironSessionV1 {
    fn run_production_pipeline_guarded(
        &mut self,
        function: Ptr<Operation>,
    ) -> Result<ProductionPlironPreloweringReportV2, ProductionSessionErrorV1> {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let function = FuncOp::from_operation(function);
            match self.atomic_target.as_ref() {
                Some(target) => {
                    require_production_pliron_checks_with_atomic_target_before_lowering_v2(
                        &self.inner.context,
                        &function,
                        target,
                    )
                }
                None => require_production_pliron_checks_before_lowering_v2(
                    &self.inner.context,
                    &function,
                ),
            }
        }));
        match result {
            Ok(Ok(report)) => Ok(report),
            Ok(Err(error)) => Err(production_pipeline_check_error(error)),
            Err(_) => {
                self.poisoned = true;
                Err(ProductionSessionErrorV1::Operation(
                    OperationHandleError::UpstreamPanicked,
                ))
            }
        }
    }

    pub(super) fn preflight_construction(
        &self,
        construction: &ProductionConstructionV1,
    ) -> Result<(), ProductionSessionErrorV1> {
        let tree_work = match &construction.kind {
            ProductionConstructionKindV1::BuiltinModule { .. } => 3,
            ProductionConstructionKindV1::RankedKernel { kernel, .. } => kernel.tree_work,
        };
        self.inner
            .require_internal_tree_capacity(tree_work)
            .map_err(ProductionSessionErrorV1::Operation)
    }

    pub(super) fn materialize_construction(
        &mut self,
        construction: ProductionConstructionV1,
        root_name: &str,
    ) -> Result<MaterializedConstructionV1, ProductionSessionErrorV1> {
        let result = catch_unwind(AssertUnwindSafe(|| match construction.kind {
            ProductionConstructionKindV1::BuiltinModule { .. } => self
                .inner
                .create_module(root_name)
                .map(|operation| MaterializedConstructionV1 {
                    operation,
                    ranked_function: None,
                    ranked_kernel: None,
                    ranked_view_names: BTreeMap::new(),
                    policy_checked_refinement_staging: Vec::new(),
                })
                .map_err(ProductionSessionErrorV1::Operation),
            ProductionConstructionKindV1::RankedKernel {
                kernel,
                policy_checked_refinement_staging,
                ..
            } => {
                self.materialize_ranked_kernel(root_name, kernel, policy_checked_refinement_staging)
            }
        }));
        match result {
            Ok(result) => result,
            Err(_) => Err(ProductionSessionErrorV1::Operation(
                OperationHandleError::UpstreamPanicked,
            )),
        }
    }

    fn materialize_ranked_kernel(
        &mut self,
        root_name: &str,
        kernel: ProductionRankedKernelV1,
        policy_checked_refinement_staging: Vec<ProductionPolicyCheckedRefinementStagingV2>,
    ) -> Result<MaterializedConstructionV1, ProductionSessionErrorV1> {
        if !self
            .inner
            .manifest()
            .registration_order()
            .iter()
            .any(|name| name == dialect_kernel::DIALECT_NAME)
        {
            return Err(ProductionSessionErrorV1::RankedRecipe(
                ProductionRankedKernelErrorV1::MissingKernelDialect,
            ));
        }
        let has_barrier = kernel.blocks.iter().any(|block| {
            block.operations.iter().any(|operation| {
                matches!(
                    operation,
                    ProductionRankedOperationV1::Barrier { .. }
                        | ProductionRankedOperationV1::Fence { .. }
                        | ProductionRankedOperationV1::ExecutionLayout { .. }
                )
            })
        });
        if has_barrier
            && !self
                .inner
                .manifest()
                .registration_order()
                .iter()
                .any(|name| name == dialect_gpu::DIALECT_NAME)
        {
            return Err(ProductionSessionErrorV1::RankedRecipe(
                ProductionRankedKernelErrorV1::MissingGpuDialect,
            ));
        }
        let has_reference = kernel.blocks.iter().any(|block| {
            block.operations.iter().any(|operation| {
                matches!(
                    operation,
                    ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent { .. }
                        | ProductionRankedOperationV1::RequireEffectRefinement { .. }
                )
            })
        });
        if has_reference
            && !self
                .inner
                .manifest()
                .registration_order()
                .iter()
                .any(|name| name == dialect_proof::DIALECT_NAME)
        {
            return Err(ProductionSessionErrorV1::RankedRecipe(
                ProductionRankedKernelErrorV1::Materialization(
                    "production reference construction requires the proof dialect registration",
                ),
            ));
        }
        let operation = self
            .inner
            .create_module(root_name)
            .map_err(ProductionSessionErrorV1::Operation)?;
        let root_pointer = self
            .inner
            .operations
            .get(&operation.identity)
            .copied()
            .ok_or(ProductionSessionErrorV1::Operation(
                OperationHandleError::StaleHandle,
            ))?;
        let module = ModuleOp::from_operation(root_pointer);
        let index: TypeHandle = IndexType::get(&self.inner.context).into();
        let function_type = FunctionType::get(
            &self.inner.context,
            vec![index; kernel.argument_count],
            vec![],
        );
        let function_name: Identifier = kernel.function_name.as_str().try_into().map_err(|_| {
            ProductionSessionErrorV1::RankedRecipe(ProductionRankedKernelErrorV1::Materialization(
                "validated function name could not be interned",
            ))
        })?;
        let function = FuncOp::new(&mut self.inner.context, function_name, function_type);
        module.append_operation(&mut self.inner.context, function.get_operation(), 0);

        let mut blocks = vec![function.get_entry_block(&self.inner.context)];
        for block_index in 1..kernel.blocks.len() {
            let label: Identifier =
                format!("bb{block_index}")
                    .as_str()
                    .try_into()
                    .map_err(|_| {
                        ProductionSessionErrorV1::RankedRecipe(
                            ProductionRankedKernelErrorV1::Materialization(
                                "generated block label could not be interned",
                            ),
                        )
                    })?;
            let block = BasicBlock::new(
                &mut self.inner.context,
                Some(label),
                vec![index; kernel.blocks[block_index].index_argument_count as usize],
            );
            block.insert_at_back(
                function.get_region(&self.inner.context),
                &self.inner.context,
            );
            blocks.push(block);
        }

        let arguments = blocks[0]
            .deref(&self.inner.context)
            .arguments()
            .collect::<Vec<_>>();
        let mut block_arguments = HashMap::new();
        for (block_index, block) in blocks.iter().copied().enumerate().skip(1) {
            for (argument_index, argument) in
                block.deref(&self.inner.context).arguments().enumerate()
            {
                block_arguments.insert((block_index as u32, argument_index as u32), argument);
            }
        }
        let mut locals = Vec::new();
        for (block_index, recipe_block) in kernel.blocks.iter().enumerate() {
            let block = blocks[block_index];
            for recipe in &recipe_block.operations {
                materialize_operation(
                    &mut self.inner.context,
                    block,
                    recipe,
                    &arguments,
                    &mut locals,
                    &block_arguments,
                    &policy_checked_refinement_staging,
                )
                .map_err(ProductionSessionErrorV1::RankedRecipe)?;
            }
            materialize_terminator(
                &mut self.inner.context,
                block,
                &recipe_block.terminator,
                &blocks,
                &arguments,
                &locals,
                &block_arguments,
            )
            .map_err(ProductionSessionErrorV1::RankedRecipe)?;
        }
        self.inner
            .finish_internal_root_construction(&operation)
            .map_err(ProductionSessionErrorV1::Operation)?;
        let ranked_view_names = kernel
            .blocks()
            .iter()
            .flat_map(|block| block.operations())
            .filter_map(|operation| match operation {
                ProductionRankedOperationV1::View { result, .. }
                | ProductionRankedOperationV1::ViewInSpace { result, .. } => {
                    let value = ProductionRankedValueV1::Local(*result);
                    let live = locals.get(result.get() as usize)?;
                    Some((value, live.unique_name(&self.inner.context).to_string()))
                }
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        let ranked_view_count = kernel
            .blocks()
            .iter()
            .flat_map(|block| block.operations())
            .filter(|operation| {
                matches!(
                    operation,
                    ProductionRankedOperationV1::View { .. }
                        | ProductionRankedOperationV1::ViewInSpace { .. }
                )
            })
            .count();
        if ranked_view_names.len() != ranked_view_count {
            return Err(ProductionSessionErrorV1::RankedRecipe(
                ProductionRankedKernelErrorV1::Materialization(
                    "ranked view identity binding is incomplete",
                ),
            ));
        }
        Ok(MaterializedConstructionV1 {
            operation,
            ranked_function: Some(function.get_operation()),
            ranked_kernel: Some(kernel),
            ranked_view_names,
            policy_checked_refinement_staging,
        })
    }

    /// Runs the fixed generic verifier pipeline in one prerequisite-aware
    /// sweep and returns only the final safety typestate.
    pub fn verify_production_ranked_kernel_pipeline(
        &mut self,
        stage: ProductionStageHandleV1<ConstructedGraphStageV1>,
        root: ProductionRootHandleV1<ConstructedGraphStageV1>,
    ) -> Result<
        (
            ProductionStageHandleV1<KernelChecksVerifiedGraphStageV1>,
            ProductionRootHandleV1<KernelChecksVerifiedGraphStageV1>,
        ),
        ProductionSessionErrorV1,
    > {
        self.validate_live()?;
        self.authenticate_owner(stage.owner)?;
        self.authenticate_owner(root.owner)?;
        let record = self
            .constructed_roots
            .get(&stage.identity)
            .ok_or(ProductionSessionErrorV1::StaleStage)?;
        if root.stage != stage.identity || root.identity != record.identity {
            return Err(ProductionSessionErrorV1::StageRootMismatch);
        }
        if record.production_pipeline_report.is_some() {
            return Err(ProductionSessionErrorV1::StaleStage);
        }
        let function = record
            .ranked_function
            .ok_or(ProductionSessionErrorV1::WrongConstructionKind)?;
        let expected_typed_roots = expected_typed_root_commitments(
            record
                .ranked_kernel
                .as_ref()
                .ok_or(ProductionSessionErrorV1::WrongConstructionKind)?,
        );
        let report = self.run_production_pipeline_guarded(function)?;
        if report.semantics().typed_root_commitments() != expected_typed_roots {
            return Err(ProductionSessionErrorV1::RankedGraphChanged);
        }
        let record = self
            .constructed_roots
            .get_mut(&stage.identity)
            .ok_or(ProductionSessionErrorV1::StaleStage)?;
        record.production_pipeline_report = Some(report);
        Ok((
            ProductionStageHandleV1 {
                owner: stage.owner,
                identity: stage.identity,
                _stage: std::marker::PhantomData,
            },
            ProductionRootHandleV1 {
                owner: root.owner,
                stage: root.stage,
                identity: root.identity,
                operation: root.operation,
                _stage: std::marker::PhantomData,
            },
        ))
    }

    pub fn prepare_ranked_lowering(
        mut self,
        stage: ProductionStageHandleV1<KernelChecksVerifiedGraphStageV1>,
        root: ProductionRootHandleV1<KernelChecksVerifiedGraphStageV1>,
    ) -> Result<ProductionRankedKernelLoweringInputV1, ProductionSessionErrorV1> {
        self.validate_live()?;
        self.authenticate_owner(stage.owner)?;
        self.authenticate_owner(root.owner)?;
        if let Err(error) = self.inner.operation_shape(&root.operation) {
            self.poisoned = true;
            return Err(ProductionSessionErrorV1::Operation(error));
        }
        let (expected_root, function, expected_report, expected_typed_roots) = {
            let record = self
                .constructed_roots
                .get(&stage.identity)
                .ok_or(ProductionSessionErrorV1::StaleStage)?;
            (
                record.identity,
                record.ranked_function,
                record.production_pipeline_report.clone(),
                expected_typed_root_commitments(
                    record
                        .ranked_kernel
                        .as_ref()
                        .ok_or(ProductionSessionErrorV1::WrongConstructionKind)?,
                ),
            )
        };
        if root.stage != stage.identity
            || root.identity != expected_root
            || expected_report.is_none()
        {
            return Err(ProductionSessionErrorV1::StageRootMismatch);
        }
        let function = function.ok_or(ProductionSessionErrorV1::WrongConstructionKind)?;
        let revalidated = match self.run_production_pipeline_guarded(function) {
            Ok(report) => report,
            Err(_) => {
                self.poisoned = true;
                return Err(ProductionSessionErrorV1::RankedGraphChanged);
            }
        };
        if revalidated.semantics().typed_root_commitments() != expected_typed_roots {
            self.poisoned = true;
            return Err(ProductionSessionErrorV1::RankedGraphChanged);
        }
        if expected_report.as_ref() != Some(&revalidated) {
            self.poisoned = true;
            return Err(ProductionSessionErrorV1::RankedGraphChanged);
        }
        let record = self
            .constructed_roots
            .remove(&stage.identity)
            .ok_or(ProductionSessionErrorV1::StaleStage)?;
        if root.stage != stage.identity || root.identity != record.identity {
            return Err(ProductionSessionErrorV1::StageRootMismatch);
        }
        let kernel = record
            .ranked_kernel
            .ok_or(ProductionSessionErrorV1::WrongConstructionKind)?;
        let report = record
            .production_pipeline_report
            .ok_or(ProductionSessionErrorV1::StageRootMismatch)?;
        if !report.is_clean() {
            return Err(ProductionSessionErrorV1::RankedRecipe(
                ProductionRankedKernelErrorV1::Materialization(
                    "safety-verified stage carried a rejected report",
                ),
            ));
        }
        Ok(ProductionRankedKernelLoweringInputV1 {
            kernel,
            production_pipeline_report: report,
            ranked_view_names: record.ranked_view_names,
            policy_checked_refinement_staging: record.policy_checked_refinement_staging,
            _session: self,
            _stage: stage,
            _root: root,
        })
    }
}

fn production_pipeline_check_error(
    error: ProductionPlironPreloweringErrorV2,
) -> ProductionSessionErrorV1 {
    match error {
        // Ranked recipe construction is target-agnostic. A target-contract
        // error can only enter through the separate targeted prelowering API.
        ProductionPlironPreloweringErrorV2::TargetContract(_) => {
            ProductionSessionErrorV1::RankedRecipe(ProductionRankedKernelErrorV1::Materialization(
                "target feasibility was requested outside target-agnostic ranked construction",
            ))
        }
        ProductionPlironPreloweringErrorV2::TensorLayout(error) => {
            ProductionSessionErrorV1::RankedTensorLayout(error)
        }
        ProductionPlironPreloweringErrorV2::Bounds(error) => {
            ProductionSessionErrorV1::RankedBounds(error)
        }
        ProductionPlironPreloweringErrorV2::Atomic(error) => {
            ProductionSessionErrorV1::RankedAtomic(error)
        }
        ProductionPlironPreloweringErrorV2::Race(error) => {
            ProductionSessionErrorV1::RankedRace(error)
        }
        ProductionPlironPreloweringErrorV2::Ownership(error) => {
            ProductionSessionErrorV1::RankedOwnership(error)
        }
        ProductionPlironPreloweringErrorV2::Barrier(error) => {
            ProductionSessionErrorV1::RankedBarrier(error)
        }
        ProductionPlironPreloweringErrorV2::PipelineProtocol(error) => {
            ProductionSessionErrorV1::RankedPipeline(error)
        }
        ProductionPlironPreloweringErrorV2::Workgroup(error) => {
            ProductionSessionErrorV1::RankedWorkgroup(error)
        }
        ProductionPlironPreloweringErrorV2::Semantic(error) => {
            ProductionSessionErrorV1::RankedSemantic(error)
        }
        ProductionPlironPreloweringErrorV2::Preservation(error) => {
            ProductionSessionErrorV1::RankedPassPreservation(error)
        }
        ProductionPlironPreloweringErrorV2::ReportValidation(error) => {
            ProductionSessionErrorV1::RankedReportValidation(error)
        }
    }
}

fn resolve_value(
    value: ProductionRankedValueV1,
    arguments: &[Value],
    locals: &[Value],
    block_arguments: &HashMap<(u32, u32), Value>,
) -> Result<Value, ProductionRankedKernelErrorV1> {
    match value {
        ProductionRankedValueV1::Argument(argument) => arguments
            .get(argument as usize)
            .copied()
            .ok_or(ProductionRankedKernelErrorV1::UndefinedValue(value)),
        ProductionRankedValueV1::Local(identity) => locals
            .get(identity.get() as usize)
            .copied()
            .ok_or(ProductionRankedKernelErrorV1::UndefinedValue(value)),
        ProductionRankedValueV1::BlockArgument { block, argument } => block_arguments
            .get(&(block, argument))
            .copied()
            .ok_or(ProductionRankedKernelErrorV1::UndefinedValue(value)),
    }
}

fn materialize_policy_checked_refinement_header(
    context: &mut pliron::context::Context,
    block: Ptr<BasicBlock>,
    retained: &[ProductionPolicyCheckedRefinementStagingV2],
    proof: &ProductionReferenceProofV2,
) -> Result<ProofIdAttr, ProductionRankedKernelErrorV1> {
    let imported = retained
        .iter()
        .find(|candidate| candidate.receipt_identity() == proof.receipt_identity())
        .filter(|candidate| candidate.binding() == proof.binding())
        .ok_or(ProductionRankedKernelErrorV1::Materialization(
            "policy-checked functional-refinement staging was not retained",
        ))?;
    let [obligation_words, subject_words, model_words, evidence_words] =
        policy_checked_proof_ids(imported)?;
    let obligation_id = ProofIdAttr::new(obligation_words);
    ObligationOp::new(
        context,
        obligation_id.clone(),
        ProofIdAttr::new(subject_words),
        ProofIdAttr::new(model_words),
        PropertyAttr::FunctionalRefinement,
    )
    .get_operation()
    .insert_at_back(block, context);
    EvidenceRefOp::new(
        context,
        ProofIdAttr::new(evidence_words),
        obligation_id.clone(),
        PropertyAttr::FunctionalRefinement,
        EvidenceStatusAttr::Checked,
        CoveredBoundaryAttr::Mir,
    )
    .get_operation()
    .insert_at_back(block, context);
    Ok(obligation_id)
}

fn materialize_operation(
    context: &mut pliron::context::Context,
    block: Ptr<BasicBlock>,
    recipe: &ProductionRankedOperationV1,
    arguments: &[Value],
    locals: &mut Vec<Value>,
    block_arguments: &HashMap<(u32, u32), Value>,
    policy_checked_refinement_staging: &[ProductionPolicyCheckedRefinementStagingV2],
) -> Result<(), ProductionRankedKernelErrorV1> {
    let (operation, result) = match recipe {
        ProductionRankedOperationV1::ExecutionLayout {
            grid_identity,
            global_extents,
            workgroup_extents,
            subgroup_size,
            full_physical_workgroups,
        } => {
            let execution_domain = if *full_physical_workgroups {
                ExecutionDomainAttr::FullPhysicalWorkgroups
            } else {
                ExecutionDomainAttr::PotentiallyPartial
            };
            let op = ExecutionLayoutOp::new_with_domain(
                context,
                *grid_identity,
                *global_extents,
                *workgroup_extents,
                *subgroup_size,
                execution_domain,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::View {
            result,
            element_width,
            writable,
            shape,
            dynamic_extents,
            allocation_origin,
            noalias_class,
        } => {
            let view_type = RankedViewType::new(context, *element_width, *writable, shape.clone())
                .map_err(|_| {
                    ProductionRankedKernelErrorV1::Materialization(
                        "validated ranked view failed materialization",
                    )
                })?;
            let dynamic_extents = dynamic_extents
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let op = RankedViewOp::new_in_space_with_allocation_contract(
                context,
                view_type,
                dynamic_extents,
                MemorySpaceAttr::Global,
                *allocation_origin,
                *noalias_class,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated ranked view operation failed materialization",
                )
            })?;
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::PipelineCreate {
            result,
            view,
            buffers,
            prefetch_distance,
        } => {
            let op = PipelineCreateOp::new(
                context,
                resolve_value(*view, arguments, locals, block_arguments)?,
                *buffers,
                *prefetch_distance,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated staged pipeline failed materialization",
                )
            })?;
            (op.get_operation(), Some((*result, op.pipeline(context))))
        }
        ProductionRankedOperationV1::PipelineEvent {
            pipeline,
            epoch,
            slot,
            kind,
        } => {
            let op = PipelineEventOp::new(
                context,
                resolve_value(*pipeline, arguments, locals, block_arguments)?,
                resolve_value(*epoch, arguments, locals, block_arguments)?,
                resolve_value(*slot, arguments, locals, block_arguments)?,
                *kind,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated staged pipeline event failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::PredicatedCheckedTiledIndex2D {
            result,
            success,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
        } => {
            let op = CheckedTiledIndex2DOp::new_predicated(
                context,
                resolve_value(*invocation, arguments, locals, block_arguments)?,
                resolve_value(*component, arguments, locals, block_arguments)?,
                resolve_value(*rows, arguments, locals, block_arguments)?,
                resolve_value(*columns, arguments, locals, block_arguments)?,
                resolve_value(*row_stride, arguments, locals, block_arguments)?,
                resolve_value(*physical_extent, arguments, locals, block_arguments)?,
                [
                    *lanes_per_tile,
                    *tile_rows,
                    *tile_columns,
                    *elements_per_lane,
                ],
            );
            if result.get() as usize != locals.len()
                || success.get() != result.get().saturating_add(1)
            {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "validated predicated checked results changed order",
                ));
            }
            locals.push(op.result(context));
            locals.push(op.success(context).ok_or(
                ProductionRankedKernelErrorV1::Materialization(
                    "predicated tiled operation omitted its success result",
                ),
            )?);
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::PredicatedCheckedRowStripedIndex2D {
            result,
            success,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            physical_extent,
            lanes_per_row,
            elements_per_lane,
        } => {
            let op = CheckedRowStripedIndex2DOp::new_predicated(
                context,
                resolve_value(*invocation, arguments, locals, block_arguments)?,
                resolve_value(*component, arguments, locals, block_arguments)?,
                resolve_value(*rows, arguments, locals, block_arguments)?,
                resolve_value(*columns, arguments, locals, block_arguments)?,
                resolve_value(*row_stride, arguments, locals, block_arguments)?,
                resolve_value(*physical_extent, arguments, locals, block_arguments)?,
                [*lanes_per_row, *elements_per_lane],
            );
            if result.get() as usize != locals.len()
                || success.get() != result.get().saturating_add(1)
            {
                return Err(ProductionRankedKernelErrorV1::Materialization(
                    "validated predicated checked results changed order",
                ));
            }
            locals.push(op.result(context));
            locals.push(op.success(context).ok_or(
                ProductionRankedKernelErrorV1::Materialization(
                    "predicated row-striped operation omitted its success result",
                ),
            )?);
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::CheckedTiledIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            lanes_per_tile,
            tile_rows,
            tile_columns,
            elements_per_lane,
        } => {
            let op = CheckedTiledIndex2DOp::new(
                context,
                resolve_value(*invocation, arguments, locals, block_arguments)?,
                resolve_value(*component, arguments, locals, block_arguments)?,
                resolve_value(*rows, arguments, locals, block_arguments)?,
                resolve_value(*columns, arguments, locals, block_arguments)?,
                resolve_value(*row_stride, arguments, locals, block_arguments)?,
                [
                    *lanes_per_tile,
                    *tile_rows,
                    *tile_columns,
                    *elements_per_lane,
                ],
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::CheckedRowStripedIndex2D {
            result,
            invocation,
            component,
            rows,
            columns,
            row_stride,
            lanes_per_row,
            elements_per_lane,
        } => {
            let op = CheckedRowStripedIndex2DOp::new(
                context,
                resolve_value(*invocation, arguments, locals, block_arguments)?,
                resolve_value(*component, arguments, locals, block_arguments)?,
                resolve_value(*rows, arguments, locals, block_arguments)?,
                resolve_value(*columns, arguments, locals, block_arguments)?,
                resolve_value(*row_stride, arguments, locals, block_arguments)?,
                [*lanes_per_row, *elements_per_lane],
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::ViewInSpace {
            result,
            element_width,
            writable,
            shape,
            dynamic_extents,
            memory_space,
            allocation_origin,
            noalias_class,
        } => {
            let view_type = RankedViewType::new(context, *element_width, *writable, shape.clone())
                .map_err(|_| {
                    ProductionRankedKernelErrorV1::Materialization(
                        "validated ranked view failed materialization",
                    )
                })?;
            let dynamic_extents = dynamic_extents
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let op = RankedViewOp::new_in_space_with_allocation_contract(
                context,
                view_type,
                dynamic_extents,
                *memory_space,
                *allocation_origin,
                *noalias_class,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated ranked view operation failed materialization",
                )
            })?;
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::IndexConstant { result, value } => {
            let op = IndexConstantOp::new(context, *value);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::IndexUnsignedCast {
            result,
            source,
            bit_width,
        } => {
            let op = IndexUnsignedCastOp::new(
                context,
                resolve_value(*source, arguments, locals, block_arguments)?,
                u64::from(*bit_width),
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::IndexUnknown { result } => {
            let op = IndexUnknownOp::new(context);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::InvocationIndex {
            result,
            dimension,
            launch_extent,
        } => {
            let op = InvocationIndexOp::new(context, *dimension, *launch_extent);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::IndexBinary {
            result,
            kind,
            lhs,
            rhs,
        } => {
            let op = IndexBinaryOp::new(
                context,
                *kind,
                resolve_value(*lhs, arguments, locals, block_arguments)?,
                resolve_value(*rhs, arguments, locals, block_arguments)?,
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::DeterministicJoin {
            result,
            dependencies,
        } => {
            let dependencies = dependencies
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let op = DeterministicJoinOp::new(context, dependencies);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::Dimension {
            result,
            view,
            dimension,
        } => {
            let op = DimensionOp::new(
                context,
                resolve_value(*view, arguments, locals, block_arguments)?,
                *dimension,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated dimension failed materialization",
                )
            })?;
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::Access {
            kind,
            view,
            indices,
        }
        | ProductionRankedOperationV1::ValueAccess {
            kind,
            view,
            indices,
            ..
        } => {
            let indices = indices
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let op = RankedAccessOp::new(
                context,
                *kind,
                resolve_value(*view, arguments, locals, block_arguments)?,
                indices,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated ranked access failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::PredicatedAccess {
            kind,
            view,
            index,
            success,
        } => {
            let op = RankedAccessOp::new_predicated(
                context,
                *kind,
                resolve_value(*view, arguments, locals, block_arguments)?,
                resolve_value(*index, arguments, locals, block_arguments)?,
                resolve_value(*success, arguments, locals, block_arguments)?,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated predicated ranked access failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::AtomicAccess {
            kind,
            ordering,
            scope,
            view,
            indices,
        }
        | ProductionRankedOperationV1::AtomicValueAccess {
            kind,
            ordering,
            scope,
            view,
            indices,
            ..
        } => {
            let indices = indices
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let op = RankedAccessOp::new_atomic(
                context,
                *kind,
                *ordering,
                *scope,
                resolve_value(*view, arguments, locals, block_arguments)?,
                indices,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated ranked atomic access failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::OwnershipContract {
            view,
            coverage,
            partition,
        } => {
            let op = OwnershipContractOp::new(
                context,
                resolve_value(*view, arguments, locals, block_arguments)?,
                *coverage,
                *partition,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated ownership contract failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::AllocationEffect {
            kind,
            memory_space,
            allocation_origin,
            noalias_class,
        } => {
            let op = AllocationEffectOp::new(
                context,
                *kind,
                *memory_space,
                *allocation_origin,
                *noalias_class,
            )
            .map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "validated allocation effect failed materialization",
                )
            })?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::Barrier {
            execution_scope,
            memory_scope,
            address_space,
            order,
        } => {
            let op = BarrierOp::new(
                context,
                *execution_scope,
                *memory_scope,
                *address_space,
                *order,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::Fence {
            memory_scope,
            address_space,
            order,
        } => {
            let op = FenceOp::new(context, *memory_scope, *address_space, *order);
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::TensorLayout {
            contract,
            convergence,
            active_lanes,
            binding,
        } => {
            let op = match binding {
                Some(binding) => TensorLayoutOp::new_with_dataflow_roots(
                    context,
                    contract,
                    *convergence,
                    *active_lanes,
                    dialect_kernel::TensorDataflowRootsV1 {
                        lhs: digest_words_v2(*binding.lhs_root().as_bytes()),
                        rhs: digest_words_v2(*binding.rhs_root().as_bytes()),
                        accumulator: digest_words_v2(*binding.accumulator_root().as_bytes()),
                        result: digest_words_v2(*binding.result_root().as_bytes()),
                    },
                ),
                None => TensorLayoutOp::new(context, contract, *convergence, *active_lanes),
            };
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::TensorResultComponent {
            result,
            tensor_result_root,
            component,
            scalar,
            ..
        } => {
            let op = TensorResultComponentOp::new(
                context,
                dialect_kernel::SemanticExpressionCommitmentAttr::new(digest_words_v2(
                    *tensor_result_root.as_bytes(),
                )),
                u32::from(*component),
                typed_scalar(*scalar)?,
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::SemanticSymbol { result, symbol } => {
            let op = SemanticSymbolOp::new(context, *symbol);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::SemanticConstant { result, value } => {
            let op = SemanticConstantOp::new(context, *value);
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::SemanticBinary {
            result,
            kind,
            lhs,
            rhs,
        } => {
            let op = SemanticBinaryOp::new(
                context,
                *kind,
                resolve_value(*lhs, arguments, locals, block_arguments)?,
                resolve_value(*rhs, arguments, locals, block_arguments)?,
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::SemanticExpression {
            result,
            expression,
            numerical_contract,
        } => {
            let digest = expression.materialized_pliron_transcript_sha256(*numerical_contract);
            let expression = materialize_typed_semantic_expression(context, block, expression)?;
            let (policy, rounding, exceptional_values) =
                typed_numerical_contract(*numerical_contract)?;
            let op = SemanticTypedExpressionRootOp::new(
                context,
                expression,
                policy,
                rounding,
                exceptional_values,
                digest_words_v2(digest),
            );
            (op.get_operation(), Some((*result, op.result(context))))
        }
        ProductionRankedOperationV1::CollectiveSemantics {
            contract,
            view,
            actual,
            expected,
            witness0,
            witness1,
        } => {
            let view = resolve_value(*view, arguments, locals, block_arguments)?;
            let actual = resolve_value(*actual, arguments, locals, block_arguments)?;
            let expected = resolve_value(*expected, arguments, locals, block_arguments)?;
            let witness0 = resolve_value(*witness0, arguments, locals, block_arguments)?;
            let witness1 = resolve_value(*witness1, arguments, locals, block_arguments)?;
            let contract_identity =
                dialect_kernel::SemanticExpressionCommitmentAttr::new(contract.contract_identity());
            let source_domain_identity = dialect_kernel::SemanticExpressionCommitmentAttr::new(
                contract.source_domain_identity(),
            );
            let numerical_policy = semantic_numerical_policy_v1(contract.numerical_contract())?;
            let op = match contract.kind() {
                ProductionCollectiveSemanticKindV1::FiniteFold => RequireFiniteFoldOp::new(
                    context,
                    view,
                    actual,
                    expected,
                    witness0,
                    witness1,
                    contract_identity,
                    source_domain_identity,
                    contract.domain_bound(),
                    contract.step_bound(),
                    contract.order(),
                    numerical_policy,
                    contract.coverage(),
                )
                .get_operation(),
                ProductionCollectiveSemanticKindV1::FiniteRecurrence => {
                    RequireFiniteRecurrenceOp::new(
                        context,
                        view,
                        actual,
                        expected,
                        witness0,
                        witness1,
                        contract_identity,
                        source_domain_identity,
                        contract.domain_bound(),
                        contract.step_bound(),
                        contract.order(),
                        numerical_policy,
                        contract.coverage(),
                    )
                    .get_operation()
                }
                ProductionCollectiveSemanticKindV1::PermutationGather => {
                    RequirePermutationGatherOp::new(
                        context,
                        view,
                        actual,
                        expected,
                        witness0,
                        witness1,
                        contract_identity,
                        source_domain_identity,
                        dialect_kernel::SemanticExpressionCommitmentAttr::new(
                            contract.target_domain_identity(),
                        ),
                        contract.domain_bound(),
                        contract.step_bound(),
                        contract.order(),
                        numerical_policy,
                        contract.coverage(),
                    )
                    .get_operation()
                }
            };
            (op, None)
        }
        ProductionRankedOperationV1::RequireEquivalent { actual, expected } => {
            let op = RequireEquivalentOp::new(
                context,
                resolve_value(*actual, arguments, locals, block_arguments)?,
                resolve_value(*expected, arguments, locals, block_arguments)?,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent {
            actual,
            expected,
            proof,
        } => {
            let obligation_id = materialize_policy_checked_refinement_header(
                context,
                block,
                policy_checked_refinement_staging,
                proof,
            )?;
            let op = RequireRefinementOp::new(
                context,
                obligation_id,
                resolve_value(*actual, arguments, locals, block_arguments)?,
                resolve_value(*expected, arguments, locals, block_arguments)?,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::RequireEffectRefinement { contract, proof } => {
            let obligation_id = materialize_policy_checked_refinement_header(
                context,
                block,
                policy_checked_refinement_staging,
                proof,
            )?;
            let op = RequireEffectRefinementOp::new(
                context,
                obligation_id,
                resolve_value(contract.view(), arguments, locals, block_arguments)?,
                contract
                    .indices()
                    .iter()
                    .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                    .collect::<Result<Vec<_>, _>>()?,
                contract
                    .gpu_coordinates()
                    .iter()
                    .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                    .collect::<Result<Vec<_>, _>>()?,
                contract
                    .reference_coordinates()
                    .iter()
                    .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                    .collect::<Result<Vec<_>, _>>()?,
                resolve_value(contract.gpu_domain(), arguments, locals, block_arguments)?,
                resolve_value(
                    contract.reference_domain(),
                    arguments,
                    locals,
                    block_arguments,
                )?,
                resolve_value(
                    contract.gpu_precondition(),
                    arguments,
                    locals,
                    block_arguments,
                )?,
                resolve_value(
                    contract.reference_precondition(),
                    arguments,
                    locals,
                    block_arguments,
                )?,
                resolve_value(contract.gpu_value(), arguments, locals, block_arguments)?,
                resolve_value(
                    contract.reference_value(),
                    arguments,
                    locals,
                    block_arguments,
                )?,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::RequireNumericalRefinement { contract, proof } => {
            let obligation_id = materialize_policy_checked_refinement_header(
                context,
                block,
                policy_checked_refinement_staging,
                proof,
            )?;
            let op = RequireNumericalRefinementOp::new(
                context,
                obligation_id,
                AbsoluteErrorF64BitsAttr(contract.absolute_error_f64_bits()),
                RelativeErrorF64BitsAttr(contract.relative_error_f64_bits()),
                resolve_value(contract.actual(), arguments, locals, block_arguments)?,
                resolve_value(contract.reference(), arguments, locals, block_arguments)?,
                resolve_value(contract.domain(), arguments, locals, block_arguments)?,
                resolve_value(contract.precondition(), arguments, locals, block_arguments)?,
            );
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::RequireTensorRefinement { contract, proof } => {
            let obligation_id = materialize_policy_checked_refinement_header(
                context,
                block,
                policy_checked_refinement_staging,
                proof,
            )?;
            let components = contract
                .components()
                .iter()
                .map(|component| {
                    Ok((
                        resolve_value(component.gpu_value(), arguments, locals, block_arguments)?,
                        resolve_value(
                            component.reference_value(),
                            arguments,
                            locals,
                            block_arguments,
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, ProductionRankedKernelErrorV1>>()?;
            let op = RequireTensorRefinementOp::try_new(
                context,
                obligation_id,
                ProofIdAttr::new(digest_words_v2(*contract.tensor_result_root().as_bytes())),
                resolve_value(contract.output_view(), arguments, locals, block_arguments)?,
                resolve_value(contract.actual(), arguments, locals, block_arguments)?,
                resolve_value(contract.reference(), arguments, locals, block_arguments)?,
                components,
            )
            .map_err(|_| ProductionRankedKernelErrorV1::InvalidReferenceContract)?;
            (op.get_operation(), None)
        }
        ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent { .. }
        | ProductionRankedOperationV1::RequestEffectRefinement { .. }
        | ProductionRankedOperationV1::RequestNumericalRefinement { .. }
        | ProductionRankedOperationV1::RequestTensorRefinement { .. } => {
            return Err(ProductionRankedKernelErrorV1::Materialization(
                "unbound functional-refinement request cannot be materialized",
            ));
        }
    };
    operation.insert_at_back(block, context);
    if let Some((identity, value)) = result {
        if identity.get() as usize != locals.len() {
            return Err(ProductionRankedKernelErrorV1::Materialization(
                "validated local value order changed before materialization",
            ));
        }
        locals.push(value);
    }
    Ok(())
}

fn semantic_numerical_policy_v1(
    contract: ProductionNumericalContractV2,
) -> Result<SemanticNumericalPolicyAttr, ProductionRankedKernelErrorV1> {
    match contract {
        ProductionNumericalContractV2::ExactBitVectorOperatorCongruence => {
            Ok(SemanticNumericalPolicyAttr::ExactBitVectorOperatorCongruence)
        }
        ProductionNumericalContractV2::ExactIeee754OperatorCongruence {
            rounding: super::ProductionIeeeRoundingModeV2::NearestTiesToEven,
            exceptional_values: super::ProductionIeeeExceptionalValuePolicyV2::PreserveExactBits,
        } => Ok(SemanticNumericalPolicyAttr::ExactIeeeNearestTiesToEvenPreserveBits),
        _ => Err(ProductionRankedKernelErrorV1::InvalidCollectiveSemanticContract),
    }
}

fn digest_as_proof_id(digest: DigestV1) -> [u64; 4] {
    let bytes = digest.as_bytes();
    std::array::from_fn(|index| {
        u64::from_le_bytes(
            bytes[index * 8..(index + 1) * 8]
                .try_into()
                .expect("digest quarters have fixed width"),
        )
    })
}

fn expected_typed_root_commitments(kernel: &ProductionRankedKernelV1) -> Vec<[u64; 4]> {
    kernel
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match operation {
            ProductionRankedOperationV1::SemanticExpression {
                expression,
                numerical_contract,
                ..
            } => Some(digest_words_v2(
                expression.materialized_pliron_transcript_sha256(*numerical_contract),
            )),
            _ => None,
        })
        .collect()
}

fn typed_scalar(
    scalar: ProductionSemanticScalarTypeV2,
) -> Result<SemanticTypedScalarV1, ProductionRankedKernelErrorV1> {
    let (kind, bits) = match scalar {
        ProductionSemanticScalarTypeV2::Bool => (SemanticScalarKindAttr::Bool, 1),
        ProductionSemanticScalarTypeV2::Integer {
            signed: false,
            bits,
        } => (SemanticScalarKindAttr::UnsignedInteger, bits),
        ProductionSemanticScalarTypeV2::Integer { signed: true, bits } => {
            (SemanticScalarKindAttr::SignedInteger, bits)
        }
        ProductionSemanticScalarTypeV2::Float { bits } => (SemanticScalarKindAttr::Float, bits),
    };
    SemanticTypedScalarV1::new(kind, bits).ok_or(ProductionRankedKernelErrorV1::Materialization(
        "validated typed semantic scalar failed PLIRON materialization",
    ))
}

fn typed_numerical_contract(
    contract: ProductionNumericalContractV2,
) -> Result<
    (
        SemanticNumericalPolicyAttr,
        SemanticIeeeRoundingAttr,
        SemanticExceptionalValueAttr,
    ),
    ProductionRankedKernelErrorV1,
> {
    match contract {
        ProductionNumericalContractV2::ExactBitVectorOperatorCongruence => Ok((
            SemanticNumericalPolicyAttr::ExactBitVectorOperatorCongruence,
            SemanticIeeeRoundingAttr::NearestTiesToEven,
            SemanticExceptionalValueAttr::PreserveExactBits,
        )),
        ProductionNumericalContractV2::ExactIeee754OperatorCongruence {
            rounding,
            exceptional_values,
        } => Ok((
            SemanticNumericalPolicyAttr::ExactIeeeNearestTiesToEvenPreserveBits,
            match rounding {
                ProductionIeeeRoundingModeV2::NearestTiesToEven => {
                    SemanticIeeeRoundingAttr::NearestTiesToEven
                }
                ProductionIeeeRoundingModeV2::TowardZero => SemanticIeeeRoundingAttr::TowardZero,
                ProductionIeeeRoundingModeV2::TowardPositive => {
                    SemanticIeeeRoundingAttr::TowardPositive
                }
                ProductionIeeeRoundingModeV2::TowardNegative => {
                    SemanticIeeeRoundingAttr::TowardNegative
                }
            },
            match exceptional_values {
                ProductionIeeeExceptionalValuePolicyV2::PreserveExactBits => {
                    SemanticExceptionalValueAttr::PreserveExactBits
                }
                ProductionIeeeExceptionalValuePolicyV2::CanonicalNan => {
                    SemanticExceptionalValueAttr::CanonicalNan
                }
            },
        )),
        ProductionNumericalContractV2::Relaxed
        | ProductionNumericalContractV2::ErrorBounded { .. } => {
            Err(ProductionRankedKernelErrorV1::Materialization(
                "unsupported numerical contract reached PLIRON materialization",
            ))
        }
    }
}

fn materialize_typed_semantic_expression(
    context: &mut pliron::context::Context,
    block: Ptr<BasicBlock>,
    expression: &ProductionSemanticExpressionV2,
) -> Result<Value, ProductionRankedKernelErrorV1> {
    let operation = match expression {
        ProductionSemanticExpressionV2::Symbol { symbol, scalar } => {
            SemanticTypedSymbolOp::new(context, *symbol, typed_scalar(*scalar)?).get_operation()
        }
        ProductionSemanticExpressionV2::Constant { scalar, bits } => {
            SemanticTypedConstantOp::new(context, *bits, typed_scalar(*scalar)?).get_operation()
        }
        ProductionSemanticExpressionV2::Load(load) => {
            SemanticTypedSymbolOp::new(context, load.proof_symbol(), typed_scalar(load.scalar)?)
                .get_operation()
        }
        ProductionSemanticExpressionV2::Unary {
            operation,
            scalar,
            operand,
        } => {
            let operand = materialize_typed_semantic_expression(context, block, operand)?;
            SemanticTypedUnaryOp::new(
                context,
                match operation {
                    ProductionSemanticUnaryOpV2::Not => SemanticTypedUnaryKindAttr::Not,
                    ProductionSemanticUnaryOpV2::Negate => SemanticTypedUnaryKindAttr::Negate,
                },
                typed_scalar(*scalar)?,
                operand,
            )
            .get_operation()
        }
        ProductionSemanticExpressionV2::Binary {
            operation,
            scalar,
            overflow,
            lhs,
            rhs,
        } => {
            let lhs = materialize_typed_semantic_expression(context, block, lhs)?;
            let rhs = materialize_typed_semantic_expression(context, block, rhs)?;
            SemanticTypedBinaryOp::new(
                context,
                match operation {
                    ProductionSemanticBinaryOpV2::Add => SemanticTypedBinaryKindAttr::Add,
                    ProductionSemanticBinaryOpV2::Subtract => SemanticTypedBinaryKindAttr::Subtract,
                    ProductionSemanticBinaryOpV2::Multiply => SemanticTypedBinaryKindAttr::Multiply,
                    ProductionSemanticBinaryOpV2::Divide => SemanticTypedBinaryKindAttr::Divide,
                    ProductionSemanticBinaryOpV2::Remainder => {
                        SemanticTypedBinaryKindAttr::Remainder
                    }
                    ProductionSemanticBinaryOpV2::BitXor => SemanticTypedBinaryKindAttr::BitXor,
                    ProductionSemanticBinaryOpV2::BitAnd => SemanticTypedBinaryKindAttr::BitAnd,
                    ProductionSemanticBinaryOpV2::BitOr => SemanticTypedBinaryKindAttr::BitOr,
                    ProductionSemanticBinaryOpV2::ShiftLeft => {
                        SemanticTypedBinaryKindAttr::ShiftLeft
                    }
                    ProductionSemanticBinaryOpV2::ShiftRight => {
                        SemanticTypedBinaryKindAttr::ShiftRight
                    }
                },
                match overflow {
                    ProductionOverflowContractV2::Wrapping => SemanticOverflowAttr::Wrapping,
                    ProductionOverflowContractV2::Checked => SemanticOverflowAttr::Checked,
                },
                typed_scalar(*scalar)?,
                lhs,
                rhs,
            )
            .get_operation()
        }
        ProductionSemanticExpressionV2::Compare {
            operation,
            operand_scalar,
            lhs,
            rhs,
        } => {
            let lhs = materialize_typed_semantic_expression(context, block, lhs)?;
            let rhs = materialize_typed_semantic_expression(context, block, rhs)?;
            SemanticTypedCompareOp::new(
                context,
                match operation {
                    ProductionSemanticComparisonV2::Equal => SemanticTypedCompareKindAttr::Equal,
                    ProductionSemanticComparisonV2::LessThan => {
                        SemanticTypedCompareKindAttr::LessThan
                    }
                    ProductionSemanticComparisonV2::LessOrEqual => {
                        SemanticTypedCompareKindAttr::LessOrEqual
                    }
                    ProductionSemanticComparisonV2::NotEqual => {
                        SemanticTypedCompareKindAttr::NotEqual
                    }
                    ProductionSemanticComparisonV2::GreaterOrEqual => {
                        SemanticTypedCompareKindAttr::GreaterOrEqual
                    }
                    ProductionSemanticComparisonV2::GreaterThan => {
                        SemanticTypedCompareKindAttr::GreaterThan
                    }
                },
                typed_scalar(*operand_scalar)?,
                lhs,
                rhs,
            )
            .get_operation()
        }
        ProductionSemanticExpressionV2::Select {
            scalar,
            condition,
            when_true,
            when_false,
        } => {
            let condition = materialize_typed_semantic_expression(context, block, condition)?;
            let when_true = materialize_typed_semantic_expression(context, block, when_true)?;
            let when_false = materialize_typed_semantic_expression(context, block, when_false)?;
            SemanticTypedSelectOp::new(
                context,
                typed_scalar(*scalar)?,
                condition,
                when_true,
                when_false,
            )
            .get_operation()
        }
        ProductionSemanticExpressionV2::Cast {
            kind,
            source,
            target,
            operand,
        } => {
            let operand = materialize_typed_semantic_expression(context, block, operand)?;
            SemanticTypedCastOp::new(
                context,
                match kind {
                    ProductionSemanticCastV2::Integer => SemanticTypedCastKindAttr::Integer,
                    ProductionSemanticCastV2::IntegerToFloat => {
                        SemanticTypedCastKindAttr::IntegerToFloat
                    }
                    ProductionSemanticCastV2::FloatToFloat => {
                        SemanticTypedCastKindAttr::FloatToFloat
                    }
                    ProductionSemanticCastV2::FloatToIntegerSaturating => {
                        SemanticTypedCastKindAttr::FloatToIntegerSaturating
                    }
                },
                typed_scalar(*source)?,
                typed_scalar(*target)?,
                operand,
            )
            .get_operation()
        }
    };
    let result = operation.deref(context).get_result(0);
    operation.insert_at_back(block, context);
    Ok(result)
}

fn policy_checked_proof_ids(
    evidence: &ProductionPolicyCheckedRefinementStagingV2,
) -> Result<[[u64; 4]; 4], ProductionRankedKernelErrorV1> {
    let inputs = [
        (
            b"FE2O3/PLIRON/PROOF-ID/OBLIGATION/V2\0".as_slice(),
            evidence.binding().normalized_obligation_effect_ir_hash(),
        ),
        (
            b"FE2O3/PLIRON/PROOF-ID/SUBJECT/V2\0".as_slice(),
            evidence.binding().kernel_mir_hash(),
        ),
        (
            b"FE2O3/PLIRON/PROOF-ID/MODEL/V2\0".as_slice(),
            evidence.binding().safe_reference_mir_hash(),
        ),
        (
            b"FE2O3/PLIRON/PROOF-ID/EVIDENCE/V2\0".as_slice(),
            evidence.receipt_identity().digest(),
        ),
    ];
    let identities = inputs.map(|(domain, input)| {
        let mut digest = Sha256::new();
        digest.update((domain.len() as u64).to_le_bytes());
        digest.update(domain);
        digest.update(input.as_bytes());
        digest_as_proof_id(DigestV1::from_untrusted_bytes(digest.finalize().into()))
    });
    if identities.iter().any(|identity| *identity == [0; 4])
        || identities
            .iter()
            .enumerate()
            .any(|(index, identity)| identities[..index].contains(identity))
    {
        return Err(ProductionRankedKernelErrorV1::InvalidReferenceContract);
    }
    Ok(identities)
}

fn digest_words_v2(digest: [u8; 32]) -> [u64; 4] {
    let mut words = [0_u64; 4];
    for (word, bytes) in words.iter_mut().zip(digest.chunks_exact(8)) {
        *word = u64::from_le_bytes(bytes.try_into().expect("eight-byte digest chunk"));
    }
    words
}

fn materialize_terminator(
    context: &mut pliron::context::Context,
    block: Ptr<BasicBlock>,
    terminator: &ProductionRankedTerminatorV1,
    blocks: &[Ptr<BasicBlock>],
    arguments: &[Value],
    locals: &[Value],
    block_arguments: &HashMap<(u32, u32), Value>,
) -> Result<(), ProductionRankedKernelErrorV1> {
    let operation = match terminator {
        ProductionRankedTerminatorV1::IndexLessThan {
            lhs,
            rhs,
            true_block,
            false_block,
        } => IndexLessThanBranchOp::new(
            context,
            resolve_value(*lhs, arguments, locals, block_arguments)?,
            resolve_value(*rhs, arguments, locals, block_arguments)?,
            blocks[*true_block as usize],
            blocks[*false_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::IndexLessThanArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            true_block,
            false_block,
        } => IndexLessThanBranchArgsOp::new(
            context,
            resolve_value(*lhs, arguments, locals, block_arguments)?,
            resolve_value(*rhs, arguments, locals, block_arguments)?,
            true_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            false_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            blocks[*true_block as usize],
            blocks[*false_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::IndexEqual {
            lhs,
            rhs,
            true_block,
            false_block,
        } => IndexEqualBranchOp::new(
            context,
            resolve_value(*lhs, arguments, locals, block_arguments)?,
            resolve_value(*rhs, arguments, locals, block_arguments)?,
            blocks[*true_block as usize],
            blocks[*false_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::IndexEqualArgs {
            lhs,
            rhs,
            true_arguments,
            false_arguments,
            true_block,
            false_block,
        } => IndexEqualBranchArgsOp::new(
            context,
            resolve_value(*lhs, arguments, locals, block_arguments)?,
            resolve_value(*rhs, arguments, locals, block_arguments)?,
            true_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            false_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            blocks[*true_block as usize],
            blocks[*false_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::AnalysisSplit {
            control_dependencies,
            first_block,
            second_block,
        } => AnalysisSplitOp::new_with_control_and_arguments(
            context,
            control_dependencies
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            vec![],
            vec![],
            blocks[*first_block as usize],
            blocks[*second_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::AnalysisSplitArgs {
            control_dependencies,
            first_arguments,
            second_arguments,
            first_block,
            second_block,
        } => AnalysisSplitOp::new_with_control_and_arguments(
            context,
            control_dependencies
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            first_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            second_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            blocks[*first_block as usize],
            blocks[*second_block as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::Branch { target } => {
            BranchOp::new(context, blocks[*target as usize]).get_operation()
        }
        ProductionRankedTerminatorV1::BranchArgs {
            arguments: edge_arguments,
            target,
        } => BranchArgsOp::new(
            context,
            edge_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?,
            blocks[*target as usize],
        )
        .get_operation(),
        ProductionRankedTerminatorV1::BranchArgsAdd {
            value,
            step,
            target,
        } => {
            let next = IndexBinaryOp::new(
                context,
                IndexBinaryKindAttr::Add,
                resolve_value(*value, arguments, locals, block_arguments)?,
                resolve_value(*step, arguments, locals, block_arguments)?,
            );
            next.get_operation().insert_at_back(block, context);
            BranchArgsOp::new(
                context,
                vec![next.result(context)],
                blocks[*target as usize],
            )
            .get_operation()
        }
        ProductionRankedTerminatorV1::BranchArgsAddAt {
            arguments: edge_arguments,
            add_argument,
            step,
            target,
        } => {
            let mut edge_arguments = edge_arguments
                .iter()
                .map(|value| resolve_value(*value, arguments, locals, block_arguments))
                .collect::<Result<Vec<_>, _>>()?;
            let add_argument = usize::try_from(*add_argument).map_err(|_| {
                ProductionRankedKernelErrorV1::Materialization(
                    "ranked induction update argument does not fit usize",
                )
            })?;
            let next = IndexBinaryOp::new(
                context,
                IndexBinaryKindAttr::Add,
                edge_arguments[add_argument],
                resolve_value(*step, arguments, locals, block_arguments)?,
            );
            next.get_operation().insert_at_back(block, context);
            edge_arguments[add_argument] = next.result(context);
            BranchArgsOp::new(context, edge_arguments, blocks[*target as usize]).get_operation()
        }
        ProductionRankedTerminatorV1::Return => ReturnOp::new(context).get_operation(),
        ProductionRankedTerminatorV1::Trap => TrapOp::new(context).get_operation(),
    };
    operation.insert_at_back(block, context);
    Ok(())
}

/// Move-only output of the closed construction, bounds, and race stages.
///
/// The value owns the complete production session and verified stage/root, so
/// the exact checked graph remains alive while no raw Pliron pointer is exposed.
/// It does not authenticate a source allocation or grant compiler/artifact
/// authority; later production stages must bind the graph to retained frontend
/// memory facts and consume this value without reconstructing it.
///
/// ```compile_fail
/// use fe2o3_pliron::ProductionRankedKernelLoweringInputV1;
///
/// fn duplicate(input: ProductionRankedKernelLoweringInputV1) {
///     let _second = input.clone();
/// }
/// ```
#[must_use = "safety-verified ranked input must be consumed by a checked lowering stage"]
pub struct ProductionRankedKernelLoweringInputV1 {
    kernel: ProductionRankedKernelV1,
    production_pipeline_report: ProductionPlironPreloweringReportV2,
    ranked_view_names: BTreeMap<ProductionRankedValueV1, String>,
    policy_checked_refinement_staging: Vec<ProductionPolicyCheckedRefinementStagingV2>,
    _session: ProductionPlironSessionV1,
    _stage: ProductionStageHandleV1<KernelChecksVerifiedGraphStageV1>,
    _root: ProductionRootHandleV1<KernelChecksVerifiedGraphStageV1>,
}

impl fmt::Debug for ProductionRankedKernelLoweringInputV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProductionRankedKernelLoweringInputV1")
            .field("function_name", &self.kernel.function_name())
            .field("argument_count", &self.kernel.argument_count())
            .field("block_count", &self.kernel.blocks().len())
            .finish_non_exhaustive()
    }
}

impl ProductionRankedKernelLoweringInputV1 {
    pub(super) fn revalidate_structure(&self) -> Result<(), ProductionRankedKernelErrorV1> {
        let tree_work = self.kernel.validate()?;
        if tree_work != self.kernel.tree_work {
            return Err(ProductionRankedKernelErrorV1::Materialization(
                "validated ranked-kernel tree work changed before evidence construction",
            ));
        }
        Ok(())
    }

    pub const fn kernel(&self) -> &ProductionRankedKernelV1 {
        &self.kernel
    }

    /// Compiler-captured live PLIRON SSA identity for one stable ranked value.
    /// The mapping is created during closed materialization and is never accepted
    /// from a contract or downstream caller.
    pub fn live_ranked_view_name(&self, view: ProductionRankedValueV1) -> Option<&str> {
        self.ranked_view_names.get(&view).map(String::as_str)
    }

    /// Indivisible lineage from the mandatory nine-pass production pipeline.
    pub const fn production_pipeline_report(&self) -> &ProductionPlironPreloweringReportV2 {
        &self.production_pipeline_report
    }

    /// Policy-checked, non-authoritative receipt staging retained for aggregate replay.
    ///
    /// A caller-selected policy can satisfy this structural check. This accessor never
    /// reports proof execution and the returned values grant no authority.
    pub fn retained_policy_checked_refinement_staging(
        &self,
    ) -> &[ProductionPolicyCheckedRefinementStagingV2] {
        &self.policy_checked_refinement_staging
    }

    pub const fn bounds_report(&self) -> &RankedBoundsReportV1 {
        self.production_pipeline_report.bounds()
    }

    pub const fn tensor_layout_report(&self) -> &PlironTensorLayoutReportV1 {
        self.production_pipeline_report.tensor_layout()
    }

    pub const fn atomic_report(&self) -> &PlironAtomicLegalityReportV1 {
        self.production_pipeline_report.atomics()
    }

    pub const fn race_report(&self) -> &RankedRaceReportV1 {
        self.production_pipeline_report.race()
    }

    pub const fn ownership_report(&self) -> &HierarchicalOwnershipReportV1 {
        self.production_pipeline_report.ownership()
    }

    pub const fn barrier_report(&self) -> &PlironBarrierReportV1 {
        self.production_pipeline_report.barriers()
    }

    pub const fn pipeline_protocol_report(&self) -> &PlironPipelineProtocolReportV1 {
        self.production_pipeline_report.pipeline_protocol()
    }

    pub const fn workgroup_report(&self) -> &PlironWorkgroupMemoryReportV1 {
        self.production_pipeline_report.workgroup()
    }

    pub const fn semantic_report(&self) -> &PlironSemanticRefinementReportV1 {
        self.production_pipeline_report.semantics()
    }

    pub const fn pass_preservation_report(
        &self,
    ) -> &fe2o3_kernel_analysis::PlironPassPreservationReportV1 {
        self.production_pipeline_report.preservation()
    }

    pub fn all_mandatory_reports_are_clean(&self) -> bool {
        self.production_pipeline_report.is_clean()
    }

    pub const fn grants_artifact_or_launch_authority(&self) -> bool {
        false
    }

    pub const fn grants_compiler_refinement_authority(&self) -> bool {
        false
    }

    pub const fn has_retained_policy_checked_refinement_staging(&self) -> bool {
        !self.policy_checked_refinement_staging.is_empty()
    }
}

#[derive(Debug)]
pub enum ProductionRankedCompileErrorV1 {
    Registration(NameError),
    AtomicTarget(PlironAtomicTargetContextErrorV1),
    Context(ContextBuildError),
    Session(ProductionSessionErrorV1),
}

impl fmt::Display for ProductionRankedCompileErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registration(_) => {
                formatter.write_str("kernel dialect registration construction failed")
            }
            Self::AtomicTarget(error) => {
                write!(
                    formatter,
                    "gfx942 atomic target construction failed: {error}"
                )
            }
            Self::Context(error) => write!(
                formatter,
                "production Pliron context construction failed: {error:?}"
            ),
            Self::Session(error) => error.fmt(formatter),
        }
    }
}

impl Error for ProductionRankedCompileErrorV1 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Registration(_) | Self::AtomicTarget(_) | Self::Context(_) => None,
            Self::Session(error) => Some(error),
        }
    }
}

/// Failure to join imported proof capabilities to exact ranked-IR requests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProductionFunctionalRefinementAdmissionErrorV2 {
    WrongConstructionKind,
    UnboundRequest,
    DuplicateImportedReceipt(FunctionalRefinementReceiptIdentityV2),
    DuplicateReceiptClaim(FunctionalRefinementReceiptIdentityV2),
    MissingImportedReceipt(FunctionalRefinementReceiptIdentityV2),
    UnusedImportedReceipt(FunctionalRefinementReceiptIdentityV2),
    BindingMismatch(FunctionalRefinementReceiptIdentityV2),
    ObligationEffectDigestMismatch(FunctionalRefinementReceiptIdentityV2),
    WrongBoundary(FunctionalRefinementReceiptIdentityV2),
    WrongSigner(FunctionalRefinementReceiptIdentityV2),
    WrongToolchain(FunctionalRefinementReceiptIdentityV2),
    InertImportedEvidence(FunctionalRefinementReceiptIdentityV2),
}

impl fmt::Display for ProductionFunctionalRefinementAdmissionErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongConstructionKind => formatter.write_str(
                "functional-refinement receipts can be admitted only for a ranked kernel",
            ),
            Self::UnboundRequest => formatter.write_str(
                "functional-refinement generator request must be bound to an imported receipt before production compilation",
            ),
            Self::DuplicateImportedReceipt(_) => {
                formatter.write_str("duplicate imported functional-refinement receipt")
            }
            Self::DuplicateReceiptClaim(_) => {
                formatter.write_str("functional-refinement receipt is claimed more than once")
            }
            Self::MissingImportedReceipt(_) => {
                formatter.write_str("ranked operation has no matching imported functional-refinement receipt")
            }
            Self::UnusedImportedReceipt(_) => {
                formatter.write_str("imported functional-refinement receipt is not claimed by the ranked kernel")
            }
            Self::BindingMismatch(_) => formatter.write_str(
                "ranked operation functional-refinement binding does not match the imported receipt",
            ),
            Self::ObligationEffectDigestMismatch(_) => formatter.write_str(
                "authenticated normalized obligation/effect digest does not match the ranked recipe",
            ),
            Self::WrongBoundary(_) => formatter.write_str(
                "production admission requires the safe-reference-MIR to kernel-MIR boundary",
            ),
            Self::WrongSigner(_) => formatter.write_str(
                "functional-refinement signer is not trusted by compiler configuration",
            ),
            Self::WrongToolchain(_) => formatter.write_str(
                "functional-refinement toolchain is not pinned by compiler configuration",
            ),
            Self::InertImportedEvidence(_) => formatter.write_str(
                "imported receipt did not grant exact functional-refinement evidence",
            ),
        }
    }
}

impl Error for ProductionFunctionalRefinementAdmissionErrorV2 {}

#[derive(Debug)]
pub enum ProductionRankedCompileErrorV2 {
    Proof(ProductionFunctionalRefinementAdmissionErrorV2),
    Pipeline(ProductionRankedCompileErrorV1),
}

impl fmt::Display for ProductionRankedCompileErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proof(error) => error.fmt(formatter),
            Self::Pipeline(error) => error.fmt(formatter),
        }
    }
}

impl Error for ProductionRankedCompileErrorV2 {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Proof(error) => Some(error),
            Self::Pipeline(error) => Some(error),
        }
    }
}

#[cfg(feature = "internal-proof-staging")]
fn admit_functional_refinement_v2(
    construction: &mut ProductionConstructionV1,
    imported: Vec<ImportedFunctionalRefinementProofV2>,
    policy: &ProductionRefinementStagingPolicyV2,
) -> Result<(), ProductionFunctionalRefinementAdmissionErrorV2> {
    let ProductionConstructionKindV1::RankedKernel {
        kernel,
        policy_checked_refinement_staging,
        ..
    } = &mut construction.kind
    else {
        return Err(ProductionFunctionalRefinementAdmissionErrorV2::WrongConstructionKind);
    };

    let mut available = BTreeMap::new();
    for proof in imported {
        let identity = proof.receipt_identity();
        if available.insert(identity, proof).is_some() {
            return Err(
                ProductionFunctionalRefinementAdmissionErrorV2::DuplicateImportedReceipt(identity),
            );
        }
    }
    let mut claimed = BTreeSet::new();
    let mut retained = Vec::new();
    let mut admit = |request: &ProductionReferenceProofV2,
                     expected_obligation_effect: DigestV1|
     -> Result<(), ProductionFunctionalRefinementAdmissionErrorV2> {
        let identity = request.receipt_identity();
        if !claimed.insert(identity) {
            return Err(
                ProductionFunctionalRefinementAdmissionErrorV2::DuplicateReceiptClaim(identity),
            );
        }
        let proof = available.remove(&identity).ok_or(
            ProductionFunctionalRefinementAdmissionErrorV2::MissingImportedReceipt(identity),
        )?;
        if proof.binding() != request.binding() {
            return Err(ProductionFunctionalRefinementAdmissionErrorV2::BindingMismatch(identity));
        }
        if proof.boundary() != FunctionalRefinementBoundaryV2::SafeReferenceMirToKernelMir {
            return Err(ProductionFunctionalRefinementAdmissionErrorV2::WrongBoundary(identity));
        }
        if !policy.accepts_signer(proof.signer_identity()) {
            return Err(ProductionFunctionalRefinementAdmissionErrorV2::WrongSigner(
                identity,
            ));
        }
        if proof.toolchain() != policy.toolchain() {
            return Err(ProductionFunctionalRefinementAdmissionErrorV2::WrongToolchain(identity));
        }
        if !proof.signature_and_policy_verified() {
            return Err(
                ProductionFunctionalRefinementAdmissionErrorV2::InertImportedEvidence(identity),
            );
        }
        if proof.binding().normalized_obligation_effect_ir_hash() != expected_obligation_effect {
            return Err(
                ProductionFunctionalRefinementAdmissionErrorV2::ObligationEffectDigestMismatch(
                    identity,
                ),
            );
        }
        retained.push(ProductionPolicyCheckedRefinementStagingV2::from_imported(
            proof,
        ));
        Ok(())
    };
    for (block_index, block) in kernel.blocks.iter().enumerate() {
        for (operation_index, operation) in block.operations.iter().enumerate() {
            match operation {
                ProductionRankedOperationV1::RequestAuthenticatedReferenceEquivalent { .. }
                | ProductionRankedOperationV1::RequestEffectRefinement { .. }
                | ProductionRankedOperationV1::RequestNumericalRefinement { .. }
                | ProductionRankedOperationV1::RequestTensorRefinement { .. } => {
                    return Err(ProductionFunctionalRefinementAdmissionErrorV2::UnboundRequest);
                }
                ProductionRankedOperationV1::RequireAuthenticatedReferenceEquivalent {
                    actual,
                    expected,
                    proof: request,
                } => {
                    let expected_digest = normalized_functional_refinement_formula_hash_for_kernel_v2(
                    kernel,
                    block_index,
                    operation_index,
                    *actual,
                    *expected,
                    request.binding().subjects(),
                )
                .map_err(|_| {
                    ProductionFunctionalRefinementAdmissionErrorV2::ObligationEffectDigestMismatch(
                        request.receipt_identity(),
                    )
                })?;
                    admit(request, expected_digest)?;
                }
                ProductionRankedOperationV1::RequireEffectRefinement { contract, proof } => {
                    let expected_digest = normalized_effect_refinement_hash_for_kernel_v2(
                    kernel,
                    block_index,
                    operation_index,
                    contract,
                    proof.binding().subjects(),
                )
                .map_err(|_| {
                    ProductionFunctionalRefinementAdmissionErrorV2::ObligationEffectDigestMismatch(
                        proof.receipt_identity(),
                    )
                })?;
                    admit(proof, expected_digest)?;
                }
                ProductionRankedOperationV1::RequireNumericalRefinement { contract, proof } => {
                    let expected_digest = normalized_numerical_refinement_hash_for_kernel_v2(
                        kernel,
                        block_index,
                        operation_index,
                        *contract,
                        proof.binding().subjects(),
                    )
                    .map_err(|_| {
                        ProductionFunctionalRefinementAdmissionErrorV2::ObligationEffectDigestMismatch(
                            proof.receipt_identity(),
                        )
                    })?;
                    admit(proof, expected_digest)?;
                }
                ProductionRankedOperationV1::RequireTensorRefinement { contract, proof } => {
                    let expected_digest = normalized_tensor_refinement_hash_for_kernel_v1(
                        kernel,
                        block_index,
                        operation_index,
                        contract,
                        proof.binding().subjects(),
                    )
                    .map_err(|_| {
                        ProductionFunctionalRefinementAdmissionErrorV2::ObligationEffectDigestMismatch(
                            proof.receipt_identity(),
                        )
                    })?;
                    admit(proof, expected_digest)?;
                }
                _ => {}
            }
        }
    }
    if let Some(identity) = available.keys().next().copied() {
        return Err(
            ProductionFunctionalRefinementAdmissionErrorV2::UnusedImportedReceipt(identity),
        );
    }
    *policy_checked_refinement_staging = retained;
    Ok(())
}

/// Executes the sole closed ranked-kernel production path through construction,
/// recursive structural verification, the fixed generic verifier pipeline, and
/// one checked lowering transition.
pub fn compile_ranked_kernel_for_lowering_v1(
    construction: ProductionConstructionV1,
    limits: ProductionSessionLimitsV1,
) -> Result<ProductionRankedKernelLoweringInputV1, ProductionRankedCompileErrorV1> {
    compile_ranked_kernel_for_lowering_with_target_v1(construction, limits, None)
}

pub fn compile_ranked_kernel_for_gfx942_lowering_v1(
    construction: ProductionConstructionV1,
    limits: ProductionSessionLimitsV1,
    system_coherent_allocations: impl IntoIterator<Item = u64>,
) -> Result<ProductionRankedKernelLoweringInputV1, ProductionRankedCompileErrorV1> {
    let target = PlironAtomicTargetContextV1::new([PlironAtomicTargetCapabilityV1::new(
        32,
        MemorySpaceAttr::Global,
        AtomicScopeAttr::System,
    )
    .map_err(ProductionRankedCompileErrorV1::AtomicTarget)?])
    .and_then(|target| target.with_system_coherent_allocations(system_coherent_allocations))
    .map_err(ProductionRankedCompileErrorV1::AtomicTarget)?;
    compile_ranked_kernel_for_lowering_with_target_v1(construction, limits, Some(target))
}

fn compile_ranked_kernel_for_lowering_with_target_v1(
    construction: ProductionConstructionV1,
    limits: ProductionSessionLimitsV1,
    atomic_target: Option<PlironAtomicTargetContextV1>,
) -> Result<ProductionRankedKernelLoweringInputV1, ProductionRankedCompileErrorV1> {
    let kernel_registration = dialect_kernel::dialect_registration()
        .map_err(ProductionRankedCompileErrorV1::Registration)?;
    let gpu_registration = dialect_gpu::dialect_registration()
        .map_err(ProductionRankedCompileErrorV1::Registration)?;
    let proof_registration = dialect_proof::dialect_registration()
        .map_err(ProductionRankedCompileErrorV1::Registration)?;
    let mut session = ProductionPlironSessionV1::new(
        limits,
        [kernel_registration, gpu_registration, proof_registration],
    )
    .map_err(ProductionRankedCompileErrorV1::Context)?;
    if let Some(target) = atomic_target {
        session.bind_atomic_target(target);
    }
    let registered = session
        .register_construction(construction)
        .map_err(ProductionRankedCompileErrorV1::Session)?;
    let (constructed, root) = session
        .construct_registered(registered)
        .map_err(ProductionRankedCompileErrorV1::Session)?;
    let (verified, root) = session
        .verify_production_ranked_kernel_pipeline(constructed, root)
        .map_err(ProductionRankedCompileErrorV1::Session)?;
    session
        .prepare_ranked_lowering(verified, root)
        .map_err(ProductionRankedCompileErrorV1::Session)
}

/// Stages caller-policy-checked V2 receipts against exact ranked obligations.
///
/// This workspace-internal transition is deliberately non-authoritative: signatures
/// under a caller-selected policy do not prove verifier execution. Only the private
/// aggregate exact-formula Verus replay may grant MIR-to-live-PLIRON refinement.
/// It grants no compiler, lowering, ISA, artifact, load, launch, or hardware authority.
#[cfg(feature = "internal-proof-staging")]
pub fn compile_ranked_kernel_with_policy_checked_refinement_staging_v2(
    mut construction: ProductionConstructionV1,
    limits: ProductionSessionLimitsV1,
    imported: Vec<ImportedFunctionalRefinementProofV2>,
    policy: ProductionRefinementStagingPolicyV2,
) -> Result<ProductionRankedKernelLoweringInputV1, ProductionRankedCompileErrorV2> {
    admit_functional_refinement_v2(&mut construction, imported, &policy)
        .map_err(ProductionRankedCompileErrorV2::Proof)?;
    compile_ranked_kernel_for_lowering_v1(construction, limits)
        .map_err(ProductionRankedCompileErrorV2::Pipeline)
}
